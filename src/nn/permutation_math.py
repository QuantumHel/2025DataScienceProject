import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_add_pool
from einops import rearrange
from src.utils import tableau_from_circuit, random_hscx_circuit
from pauliopt.clifford.tableau import CliffordTableau
from src.nn.brute_force_data import get_best_cnots
from pauliopt.topologies import Topology
from torch_geometric.data import Data
import numpy as np
import warnings

from torch_geometric.nn import GINEConv
from scipy.optimize import linear_sum_assignment

import pickle

# Import PyTorch Geometric's specialized DataLoader
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch.utils.data import Dataset

from pauliopt.clifford.tableau_synthesis import synthesize_tableau_perm_row_col
from pauliopt.circuits import Circuit

import matplotlib.pyplot as plt

import copy

# Suppress all overflow warnings globally
np.seterr(over="ignore")

# Suppress FutureWarning
warnings.simplefilter(action="ignore", category=FutureWarning)


def get_default_device():
    """Return the best available device (CUDA → MPS → CPU)"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


# Try a Block-Aware Residual Encoder (Expect to be better for Tableau Structure, but in fact not)
class ResidualBlockAwareEncoder(nn.Module):
    def __init__(self, n_qubits):
        super().__init__()
        self.n_qubits = n_qubits

        # Extract features from X/Z blocks separately
        self.x_block_encoder = nn.Sequential(
            nn.Conv2d(2, 32, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 48, 3, padding=1),
        )

        self.z_block_encoder = nn.Sequential(
            nn.Conv2d(2, 32, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 48, 3, padding=1),
        )

        # Combine X/Z information
        self.combiner = nn.Conv2d(96, 64, 1)

        # Final processing
        self.final = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )

    def forward(self, x):
        # Split X and Z blocks
        n_qubits = self.n_qubits
        x_part = x[:, :, :n_qubits, :]
        z_part = x[:, :, n_qubits:, :]

        # Process blocks separately
        x_features = self.x_block_encoder(x_part)
        z_features = self.z_block_encoder(z_part)

        # Combine features
        combined = torch.cat([x_features, z_features], dim=1)
        combined = self.combiner(combined)

        # Final processing
        output = self.final(combined)
        return output.flatten(start_dim=1)


# Try a tableau-aware encoder, also not as good as expected
class TableauStructureEncoder(nn.Module):
    def __init__(self, n_qubits):
        super().__init__()
        self.n_qubits = n_qubits

        # Single pathway with structural awareness
        self.encoder = nn.Sequential(
            # Initial feature extraction
            nn.Conv2d(2, 32, 3, padding=1),
            nn.GELU(),
            nn.BatchNorm2d(32),
            # Capture larger patterns
            nn.Conv2d(32, 64, 5, padding=2),
            nn.GELU(),
            nn.BatchNorm2d(64),
            # Global context
            nn.Conv2d(64, 128, 3, padding=1),
            nn.GELU(),
            nn.BatchNorm2d(128),
        )

        # Position-aware readout
        self.pool = nn.AdaptiveAvgPool2d((4, 4))

        # Final projection
        self.project = nn.Linear(128 * 4 * 4, 256)

    def forward(self, x):
        # Process the whole tableau together
        features = self.encoder(x)
        pooled = self.pool(features)
        return self.project(pooled.flatten(1))


# Experiments show that time-consuming and not as good as expected
class TransformerTableauEncoder(nn.Module):
    def __init__(self, n_qubits, dim=256, num_layers=4, num_heads=8):
        super().__init__()
        self.n_qubits = n_qubits

        # Embedding for tableau entries
        self.embedding = nn.Linear(2, dim)  # 2 channels to dimension

        # 2D positional encoding
        self.row_pos = nn.Parameter(torch.randn(2 * n_qubits, dim // 2))
        self.col_pos = nn.Parameter(torch.randn(2 * n_qubits, dim // 2))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=dim * 4,
            activation=F.gelu,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output projection
        self.output_proj = nn.Linear(dim * (2 * n_qubits) ** 2, dim)

    def forward(self, x):
        B = x.size(0)
        n = 2 * self.n_qubits

        # Reshape input to [batch, n*n, 2]
        x = x.permute(0, 2, 3, 1).reshape(B, n * n, 2)

        # Embed features
        x = self.embedding(x)  # [batch, n*n, dim]

        # Add 2D positional encoding
        pos_indices = torch.arange(n, device=x.device)
        row_idx = pos_indices.repeat_interleave(n).view(n, n)
        col_idx = pos_indices.repeat(n, 1)

        row_emb = self.row_pos[row_idx.flatten()]  # [n*n, dim//2]
        col_emb = self.col_pos[col_idx.flatten()]  # [n*n, dim//2]
        pos_emb = torch.cat([row_emb, col_emb], dim=-1)  # [n*n, dim]

        # Add positional embeddings
        x = x + pos_emb.unsqueeze(0)  # [batch, n*n, dim]

        # Pass through transformer
        x = self.transformer(x)  # [batch, n*n, dim]

        # Global pooling with attention
        x = x.flatten(1)  # [batch, n*n*dim]
        x = self.output_proj(x)  # [batch, dim]

        return x


# Try a self-attention block
class SelfAttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.query = nn.Conv2d(channels, channels // 8, 1)
        self.key = nn.Conv2d(channels, channels // 8, 1)
        self.value = nn.Conv2d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch, c, h, w = x.size()
        query = self.query(x).view(batch, -1, h * w).permute(0, 2, 1)
        key = self.key(x).view(batch, -1, h * w)
        energy = torch.bmm(query, key)
        attention = F.softmax(energy, dim=-1)
        value = self.value(x).view(batch, -1, h * w)
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch, c, h, w)
        return x + self.gamma * out


# Try a CX-aware decoder
class PermutationCXAwareDecoder(nn.Module):
    def __init__(self, dim, n_qubits, num_layers=4, nhead=4):
        super().__init__()
        # Base transformer decoder
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=dim, nhead=nhead, dim_feedforward=4 * dim, activation=F.gelu
            ),
            num_layers=num_layers,
        )

        # CX-aware attention mechanism
        self.cx_attention = nn.Linear(dim, n_qubits**2)

        # # Direct CX estimation per permutation element (unused)
        # self.element_cx_impact = nn.Parameter(torch.zeros(n_qubits, n_qubits))

        # Final CX integration layer
        self.cx_integration = nn.Sequential(
            nn.Linear(dim + n_qubits**2, dim), nn.GELU(), nn.LayerNorm(dim)
        )

    def forward(self, tgt, memory):
        # Regular transformer decoding
        output = self.decoder(tgt, memory)

        # Generate CX-aware attention weights
        cx_weights = torch.sigmoid(self.cx_attention(output))

        # Reshape for visualization and further processing
        batch_size = output.size(1)
        seq_len = output.size(0)
        cx_weights = cx_weights.view(seq_len, batch_size, -1)

        # Mix output with CX awareness
        enhanced_output = self.cx_integration(torch.cat([output, cx_weights], dim=-1))

        return enhanced_output, cx_weights


# Residual block for CX predictions
class ResidualCXBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.LayerNorm(dim))

    def forward(self, x):
        return x + self.net(x)  # Residual connection


class OrderedPermutationTransformer(nn.Module):
    # def __init__(self, n_qubits, dim=128, num_layers=4):
    def __init__(self, n_qubits, dim=256, num_layers=4, num_heads=8, dropout=0.1):
        super().__init__()
        self.n_qubits = n_qubits
        self.dim = dim

        # # Tableau Encoder
        # self.tableau_encoder = nn.Sequential(
        #     nn.Conv2d(2, 16, 3, padding=1),
        #     nn.GELU(),
        #     nn.Conv2d(16, 32, 3, padding=1),
        #     nn.AdaptiveAvgPool2d((4, 4)),
        # )

        # # Try enhanced CNN encoder, works better!
        # self.tableau_encoder = nn.Sequential(
        #     nn.Conv2d(2, 32, 3, padding=1),
        #     nn.GELU(),
        #     nn.BatchNorm2d(32),  # Added normalization
        #     nn.Conv2d(32, 64, 3, padding=1),
        #     nn.GELU(),
        #     nn.BatchNorm2d(64),  # Added normalization
        #     nn.Conv2d(64, 64, 3, padding=1),  # Added third layer
        #     nn.GELU(),
        #     nn.AdaptiveAvgPool2d((4, 4)),
        # )

        # Try even further with self-attention, results?
        self.tableau_encoder = nn.Sequential(
            nn.Conv2d(2, 32, 3, padding=1),
            nn.GELU(),
            nn.BatchNorm2d(32),
            nn.Dropout2d(dropout / 2),  # New dropout, results?
            SelfAttentionBlock(32),  # Add self-attention between conv layers
            nn.Conv2d(32, 64, 3, padding=1),
            nn.GELU(),
            nn.BatchNorm2d(64),
            nn.Dropout2d(dropout / 2),  # New dropout, results?
            SelfAttentionBlock(64),  # Add self-attention between conv layers
            nn.Conv2d(64, 64, 3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )

        # Try a Transformer-based encoder
        # self.tableau_encoder = TransformerTableauEncoder(
        #     n_qubits=n_qubits, dim=dim, num_layers=num_layers, num_heads=num_heads
        # )

        # try a Residual Block-Aware Encoder
        # Not as good as expected
        # self.tableau_encoder = ResidualBlockAwareEncoder(n_qubits)
        # self.tableau_encoder = TableauStructureEncoder(n_qubits)

        # Linear layer to match the dimension
        # self.linear = nn.Linear(32 * 4 * 4, dim)
        # No need for linear projection layer (the line above) - encoder already outputs dim-dimensional vectors
        # As the TransformerTableauEncoder handles this internally

        # Linear layer to match the enhanced CNN encoder output to dim
        self.linear = nn.Linear(64 * 4 * 4, dim)

        # try a Residual Block-Aware Encoder or Tableau Structure Encoder
        # self.linear = nn.Linear(64 * 4 * 4, dim)
        # self.linear = nn.Linear(256, dim)

        # # Sequence Decoder, works well
        # self.decoder = nn.TransformerDecoder(
        #     nn.TransformerDecoderLayer(
        #         d_model=dim, nhead=4, dim_feedforward=4 * dim, activation=F.gelu
        #     ),
        #     num_layers=num_layers,
        # )

        # Try a CX-aware decoder, not improving
        self.decoder = PermutationCXAwareDecoder(
            dim=dim, n_qubits=n_qubits, num_layers=num_layers, nhead=num_heads
        )

        # Position-Aware Prediction Heads
        self.step_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(dim, 2 * n_qubits**2),
                    nn.GELU(),
                    nn.Linear(2 * n_qubits**2, n_qubits**2),
                )
                for _ in range(10)  # Max sequence length
            ]
        )

        # Adaptive Sequence Length Prediction
        self.stop_head = nn.Linear(dim, 1)

        # # Add CX Prediction Head, works well
        # self.cx_head = nn.Sequential(
        #     nn.Linear(dim, dim // 2),
        #     nn.GELU(),
        #     nn.Dropout(0.1),
        #     nn.Linear(dim // 2, 1),
        #     nn.ReLU(),  # Ensure non-negative predictions
        # )

        # # Try making CX head more sophisticated and directly connected to permutation prediction
        # self.cx_head = nn.Sequential(
        #     nn.Linear(dim + n_qubits**2, 256),  # Add permutation matrix features
        #     nn.GELU(),
        #     nn.Dropout(0.1),
        #     nn.LayerNorm(256),
        #     nn.Linear(256, 128),
        #     nn.GELU(),
        #     nn.Linear(128, 1),
        #     nn.Softplus(),  # Ensure non-negative predictions
        # )

        # # Try a more sophisticated CX head to accept the additional cx_aware features
        # self.cx_head = nn.Sequential(
        #     nn.Linear(
        #         dim + n_qubits**2 + n_qubits**2, 256
        #     ),  # Added cx_awareness features
        #     nn.GELU(),
        #     nn.Dropout(0.1),
        #     nn.LayerNorm(256),
        #     nn.Linear(256, 128),
        #     nn.GELU(),
        #     nn.Linear(128, 1),
        #     nn.Softplus(),  # Ensure non-negative predictions
        # )

        # Enhanced CX-head with residual connections
        self.cx_head = nn.Sequential(
            nn.Linear(dim + n_qubits**2 + n_qubits**2, 384),  # Wider network
            nn.GELU(),
            nn.Dropout(0.15),  # Slightly higher dropout
            nn.LayerNorm(384),
            ResidualCXBlock(384),  # Add residual connections
            nn.Linear(384, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.ReLU(),  # Use ReLU instead of Softplus for sharper predictions
        )

        # Try Add initialization here, after defining all components, results?
        # Enhanced initialization
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.kaiming_normal_(p, mode="fan_out", nonlinearity="relu")

    def forward(self, tableau):
        # Encode Tableau
        B = tableau.size(0)
        x = self.tableau_encoder(tableau)
        # The two lines below are not needed for the Transformer-based encoder
        x = x.view(B, -1)
        x = self.linear(x)  # Ensure the dimension matches

        # Generate Sequence
        memory = x.unsqueeze(0)  # [1, batch_size, dim]
        output = torch.zeros(B, self.dim, device=tableau.device)  # [batch_size, dim]

        predictions = []
        stop_logits = []
        cx_predicitons = []  # For CX-prediction

        for t in range(10):  # Max steps
            # # Transformer Decoder
            # output = self.decoder(tgt=output.unsqueeze(0), memory=memory).squeeze(0)

            # Try CX-aware decoder
            output_enhanced, cx_awareness = self.decoder(
                tgt=output.unsqueeze(0), memory=memory
            )
            output = output_enhanced.squeeze(0)

            # Step-Specific Prediction
            pred = self.step_heads[t](output)  # [batch_size, n_qubits*n_qubits]

            # Reshape for predictions
            predictions.append(pred.view(B, self.n_qubits, self.n_qubits))

            # Stop Prediction
            stop_logits.append(self.stop_head(output))

            # For CX-prediction
            # cx_pred = self.cx_head(output) # This line is replaced by the two lines below for sophisticated CX prediction
            perm_features = pred.view(B, self.n_qubits, self.n_qubits).flatten(1)
            # cx_pred = self.cx_head(torch.cat([output, perm_features], dim=1))
            # cx_predicitons.append(cx_pred)

            # Try mixing with cx_awareness information from decoder
            cx_aware_features = torch.cat(
                [output, perm_features, cx_awareness.squeeze(0)], dim=1
            )

            # Updated CX head to use awareness features
            cx_pred = self.cx_head(cx_aware_features)
            cx_predicitons.append(cx_pred)

        # Stack along sequence dimension -> [batch_size, seq_len, n_qubits, n_qubits]
        predictions = torch.stack(predictions, dim=1)

        # Stack stop logits -> [batch_size, seq_len]
        stop_logits = torch.cat(stop_logits, dim=1)

        # # For CX-prediction
        # cx_predicitons = torch.cat(cx_predicitons, dim=1)  # [batch_size, seq_len]

        # Try sofisticated CX prediction
        # For CX prediction, use both tableau features and permutation info
        # perm_features = predictions[0].flatten(1)  # Use permutation matrix features
        # cx_predicitons = self.cx_head(torch.cat([output, perm_features], dim=1))
        cx_predicitons = torch.cat(cx_predicitons, dim=1)

        return predictions, stop_logits, cx_predicitons


class SequentialSinkhorn(nn.Module):
    def __init__(self, temp=0.1, n_iters=20):
        super().__init__()
        self.temp = temp
        self.n_iters = n_iters

    def forward(self, logits_seq):
        normalized = []
        for logits in logits_seq:
            for _ in range(self.n_iters):
                logits = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
                logits = logits - torch.logsumexp(logits, dim=-2, keepdim=True)
            normalized.append(torch.exp(logits / self.temp))
        return torch.stack(normalized)


class OrderedPermutationLoss(nn.Module):
    # def __init__(self, alpha=0.5, beta=0.1):
    # def __init__(self, alpha=0.5, beta=0.1, gamma=2.0): # original
    def __init__(
        self, alpha=0.5, beta=0.1, gamma=2.0, entropy_weight=0.1, max_entropy=0.5
    ):  # gamma=8.0 will sometimes result in stuck at loss 270.0
        super().__init__()
        self.alpha = alpha  # Stop signal loss weight
        self.beta = beta  # Length regularization weight
        self.gamma = gamma  # CX prediction loss weight
        # self.entropy_weight = entropy_weight  # Entropy regularization weight
        # self.max_entropy = max_entropy  # Max entropy for scaling
        # self.plateau_counter = 0

    # # Try adjusting entropy weight dynamically (the model is stuck at loss 170.0)
    # def adjust_entropy_weight(self, current_loss, prev_loss, patience=3):
    #     """Dynamically adjust entropy weight based on training progress"""
    #     if current_loss > prev_loss * 0.995:  # No significant improvement
    #         self.plateau_counter += 1
    #         if self.plateau_counter >= patience:
    #             # Increase entropy to escape local minimum
    #             self.entropy_weight = min(self.entropy_weight * 1.5, self.max_entropy)
    #             self.plateau_counter = 0
    #             return True
    #     else:
    #         # Improving - gradually reduce entropy
    #         self.entropy_weight = max(self.entropy_weight * 0.9, 0.05)
    #         self.plateau_counter = 0
    #     return False

    # def forward(self, preds, stop_logits, targets, masks):
    def forward(
        self, preds, stop_logits, targets, masks, cx_preds=None, cx_targets=None
    ):
        """
        preds: [bs, seq_len, n, n]
        stop_logits: [bs, seq_len] - This has shape [32, 10]
        targets: [bs, seq_len, n, n]
        masks: [bs, seq_len] - But this might have shape [32, 1]
        cx_preds: [bs, seq_len]
        cx_targets: [bs, seq_len]
        """
        bs, seq_len = stop_logits.shape  # Use stop_logits shape instead of masks

        # # 1. Permutation matrix loss (masked)
        # # Make sure targets has same sequence length as preds
        # if targets.size(1) < preds.size(1):
        #     padding = torch.zeros(
        #         bs,
        #         preds.size(1) - targets.size(1),
        #         *targets.shape[2:],
        #         device=targets.device,
        #     )
        #     targets = torch.cat([targets, padding], dim=1)

        # perm_loss = F.mse_loss(preds, targets, reduction="none")

        # # If masks is too short, pad it
        # if masks.size(1) < seq_len:
        #     mask_padding = torch.zeros(bs, seq_len - masks.size(1), device=masks.device)
        #     masks = torch.cat([masks, mask_padding], dim=1)

        # perm_loss = perm_loss.mean(dim=(-1, -2)) * masks  # [bs, seq_len]
        # perm_loss = perm_loss.sum() / masks.sum().clamp(min=1.0)  # Avoid div by zero

        # 1. Try computing permutation loss for each valid target and use the minimum
        perm_losses = []
        for i in range(len(preds)):
            losses = []
            for valid_seq in targets[i]:  # valid_seq: [seq_len, n, n]
                # Pad valid_seq if needed
                if valid_seq.size(0) < preds[i].size(
                    0
                ):  # Compare sequence lengths (dim 0)
                    n = valid_seq.size(-1)  # Get n_qubits dimension size
                    padding = torch.zeros(
                        preds[i].size(0) - valid_seq.size(0),
                        n,
                        n,
                        device=valid_seq.device,
                    )
                    valid_seq_padded = torch.cat(
                        [valid_seq, padding], dim=0
                    )  # Along seq dim
                else:
                    valid_seq_padded = valid_seq[
                        : preds[i].size(0)
                    ]  # Truncate if too long

                # Ensure device matching
                valid_seq_padded = valid_seq_padded.to(preds[i].device)

                # # Try adding label smoothing (the model is stuck at loss 170.0)
                # smoothing = 0.1
                # valid_seq_padded = valid_seq_padded * (1 - smoothing) + smoothing / n

                loss = (
                    F.mse_loss(preds[i], valid_seq_padded, reduction="none").mean(
                        dim=(-1, -2)
                    )
                    * masks[i]
                )
                # # Further try: adding KL divergence loss (the model is stuck at loss 170.0)
                # mse_loss = (
                #     F.mse_loss(preds[i], valid_seq_padded, reduction="none").mean(
                #         dim=(-1, -2)
                #     )
                #     * masks[i]
                # )
                # # Add KL divergence loss (sharper probabilities)
                # preds_softmax = F.softmax(
                #     preds[i] * 10, dim=-1
                # )  # Temperature for sharpness
                # target_softmax = F.softmax(valid_seq_padded * 10, dim=-1)
                # kl_loss = (
                #     F.kl_div(preds_softmax.log(), target_softmax, reduction="none").sum(
                #         dim=(-1, -2)
                #     )
                #     * masks[i]
                # )
                # # Combine losses
                # loss = 0.7 * mse_loss + 0.3 * kl_loss
                loss = loss.sum() / masks[i].sum().clamp(min=1.0)
                losses.append(loss)
            perm_losses.append(torch.stack(losses).min())
        perm_loss = torch.stack(perm_losses).mean()

        # 2. Stop signal loss with properly sized tensors
        # Create stop_labels with the right size [bs, seq_len]
        stop_labels = torch.zeros_like(stop_logits)  # Match size exactly

        # Fill first positions with 1s (assuming first step should always stop)
        if masks.size(1) > 0:
            stop_labels[:, 0] = 1.0

            # If masks has more than 1 column, shift it
            if masks.size(1) > 1:
                stop_labels[:, 1:] = masks[:, :-1]  # Shift right

        stop_loss = F.binary_cross_entropy_with_logits(
            stop_logits, stop_labels, reduction="mean"
        )

        # 3. Length regularization
        pred_lengths = torch.sigmoid(stop_logits).sum(dim=1)  # [bs]
        true_lengths = masks.sum(dim=1).clamp(min=1.0)  # [bs], avoid zeros
        length_loss = F.l1_loss(pred_lengths, true_lengths)

        # 4. CX prediction loss (if cx_targets is provided)
        cx_loss = 0
        if cx_preds is not None and cx_targets is not None:
            # Make sure cx_targets is a tensor
            if not isinstance(cx_targets, torch.Tensor):
                cx_targets = torch.tensor(cx_targets, device=cx_preds.device).float()

            # Ensure shapes match
            if cx_targets.dim() == 1:
                cx_targets = cx_targets.unsqueeze(1).expand(-1, cx_preds.size(1))

            # Use MSE loss for regression (no sigmoid needed with ReLU output), grows quadratically
            cx_loss = F.mse_loss(cx_preds, cx_targets, reduction="mean")
            # Try Replace MSE with Huber loss for CX prediction, results not that good? less sensitive to outliers, grows linearly
            # cx_loss = F.huber_loss(cx_preds, cx_targets, reduction="mean")

        # # 5. Try Adding Entropy regularization (encourage exploration, not improving)
        # # preds: [bs, seq_len, n, n]
        # entropy = 0
        # bs, seq_len, n, _ = preds.shape
        # for b in range(bs):
        #     for t in range(seq_len):
        #         p = preds[b, t]
        #         p = p / (p.sum() + 1e-8)  # Normalize
        #         entropy -= (p * (p + 1e-8).log()).sum()  # Negative entropy

        # entropy = entropy / (bs * seq_len)

        return (
            perm_loss
            + self.alpha * stop_loss
            + self.beta * length_loss
            + self.gamma * cx_loss
            # + self.entropy_weight * entropy  # Add entropy regularization (not improving)
        )


def train(model, dataloader, epochs, device):
    # Initialization
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=5e-4, total_steps=epochs * len(dataloader), pct_start=0.3
    )  # increase max_lr from 2e-4 to 5e-4, results, not improving!
    # # Try another scheduler, results, not good
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer, T_max=epochs, eta_min=1e-6, last_epoch=-1
    # )
    # # Try aggressive restart???
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    #     optimizer, T_0=10, T_mult=2, eta_min=1e-6
    # )
    # # # Gradient clipping, introduced with CosineAnnealingLR and CosineAnnealingWarmRestarts
    # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)

    # Loss function
    criterion = OrderedPermutationLoss(gamma=5.0)  # Adjust gamma as needed
    # sinkhorn = SequentialSinkhorn()

    # Gradient accumulation (for larger batches)
    accum_steps = 4

    loss_history = []  # To track loss per epoch
    # best_validation_score = float("inf")  # for tracking the best model
    # best_model_state = None
    # validation_scores = []

    # prev_loss = float("inf")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        batches = 0

        optimizer.zero_grad()

        for batch_idx, (tableaus, raw_targets) in enumerate(dataloader):
            # print(f"raw_targets: {raw_targets}")
            # 1. Prepare batch -------------------------------------------------
            # Convert raw targets to padded tensor
            # targets, masks = pad_targets(raw_targets, device)
            targets, masks = pad_targets_all_seq(
                raw_targets, device
            )  # try returning all sequences

            # Move data to device
            tableaus = tableaus.to(device)

            # Extract cx targets from `raw_targets`
            cx_targets = [item[1] for item in raw_targets]

            # print(f"cx_targets: {cx_targets}")

            # 2. Forward pass -------------------------------------------------
            # raw_preds, stop_logits = model(tableaus)  # [batch_size, seq_len, n, n]
            raw_preds, stop_logits, cx_preds = model(
                tableaus
            )  # [batch_size, seq_len, n, n]

            # 3. Apply Sinkhorn normalization to batch-first format
            sinkhorn_preds = []
            for i in range(raw_preds.size(0)):  # For each batch item
                seq_preds = []
                for j in range(raw_preds.size(1)):  # For each sequence step
                    logits = raw_preds[i, j]
                    # Apply Sinkhorn
                    for _ in range(20):  # n_iters
                        logits = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
                        logits = logits - torch.logsumexp(logits, dim=-2, keepdim=True)
                    seq_preds.append(torch.exp(logits / 0.1))
                    # # Try Applying another Sinkhorn normalization, worse!
                    # # Temperature annealing (start high, decrease during training)
                    # current_temp = max(0.5, 1.0 - epoch / epochs * 0.8)
                    # seq_preds.append(tiny_gumbel_sinkhorn(logits, temp=current_temp))
                sinkhorn_preds.append(torch.stack(seq_preds))
            preds = torch.stack(sinkhorn_preds)  # [batch_size, seq_len, n, n]

            # 4. Loss calculation ----------------------------------------------
            # Convert cx_targets from list to properly shaped tensor
            cx_targets_tensor = torch.tensor(cx_targets, device=device).float()

            # Reshape to match cx_preds dimensions [batch_size, seq_len]
            cx_targets_tensor = cx_targets_tensor.unsqueeze(1).expand(
                -1, cx_preds.size(1)
            )

            # print(cx_targets_tensor)
            # print(f"cx_preds: {cx_preds}")

            loss = criterion(
                preds,  # [batch_size, seq_len, n, n]
                stop_logits,  # [batch_size, seq_len]
                targets,  # [batch_size, seq_len, n, n]
                masks,  # [batch_size, seq_len]
                cx_preds=cx_preds,  # [batch_size, seq_len]
                cx_targets=cx_targets_tensor,  # [batch_size, seq_len]
            )

            # 5. Backpropagation ----------------------------------------------
            loss.backward()

            # 6. Gradient accumulation -----------------------------------------
            if (batch_idx + 1) % accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()  # For OneCycleLR
                # scheduler.step(epoch + batch_idx / len(dataloader))  # For WarmRestarts

            # 7. Logging ------------------------------------------------------
            total_loss += loss.item()
            batches += 1

            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1} | Batch {batch_idx} | Loss: {loss.item():.4f}")

        # 8. Epoch summary -----------------------------------------------------
        avg_loss = total_loss / batches
        # # Try adjusting entropy weight dynamically
        # entropy_increased = criterion.adjust_entropy_weight(avg_loss, prev_loss)
        # if entropy_increased:
        #     print(
        #         f"Increased entropy weight to {criterion.entropy_weight:.4f} to escape local minimum"
        #     )
        # prev_loss = avg_loss  # Try adjusting entropy weight dynamically
        loss_history.append(avg_loss)
        print(f"Epoch {epoch+1} completed | Average Loss: {avg_loss:.4f}")

    # Plotting loss history
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(loss_history) + 1), loss_history, marker="o")
    plt.title("Training Loss by Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig("loss_history.png")
    plt.close()
    # Save the model
    torch.save(model.state_dict(), "ordered_permutation_model.pth")
    print("Model saved as ordered_permutation_model.pth")
    # Save loss history
    with open("loss_history.pkl", "wb") as f:
        pickle.dump(loss_history, f)
    print("Loss history saved as loss_history.pkl")

    return model  # Return the trained model


def pad_targets(raw_targets, device):
    # # Extract first sequence from each batch item
    # first_sequences = [item[0] for item in raw_targets]

    """Handle the correct nested structure of permutation sequences"""
    # Extract first sequence from each batch item
    first_sequences = []

    for item in raw_targets:
        perm_sequences = item[0]  # This is the all_perm_sequences list
        # Take the first permutation sequence
        first_seq = perm_sequences[0]
        first_sequences.append(first_seq)

    # Determine max length and dimensions
    max_len = max(len(seq) for seq in first_sequences)
    n_qubits = first_sequences[0][0].shape[0]

    # Ensure max_len is at least 1
    max_len = max(max_len, 1)

    # Pad sequences to max_len
    padded = torch.stack(
        [
            torch.cat(
                [
                    (
                        torch.stack(seq)
                        if len(seq) > 0
                        else torch.eye(n_qubits).unsqueeze(0)
                    ),
                    torch.zeros(max_len - min(len(seq), max_len), n_qubits, n_qubits),
                ]
            )
            for seq in first_sequences
        ]
    )

    # Create masks with the same length
    mask = torch.stack(
        [
            torch.cat(
                [
                    torch.ones(min(len(seq), max_len)),
                    torch.zeros(max_len - min(len(seq), max_len)),
                ]
            )
            for seq in first_sequences
        ]
    )

    return padded.to(device), mask.to(device)


# Try returning all permutation sequences
def pad_targets_all_seq(raw_targets, device):
    """
    Returns:
        all_padded_sequences: list of [num_valid_seqs, max_len, n, n] tensors (per batch item)
        masks: [batch_size, max_len] tensor (mask for the longest sequence in the batch)
    """
    all_sequences = []
    max_len = 0
    n_qubits = None

    # Gather all valid sequences and find max length
    for item in raw_targets:
        perm_sequences = item[0]  # list of valid sequences
        all_sequences.append(perm_sequences)
        for seq in perm_sequences:
            if n_qubits is None and len(seq) > 0:
                n_qubits = seq[0].shape[0]
            max_len = max(max_len, len(seq))

    # Pad all sequences to max_len
    all_padded_sequences = []
    for perm_sequences in all_sequences:
        padded_seqs = []
        for seq in perm_sequences:
            seq_len = len(seq)
            if seq_len < max_len:
                pad = [torch.eye(n_qubits, device=device)] * (max_len - seq_len)
                padded_seq = torch.stack(seq + pad)
            else:
                padded_seq = torch.stack(seq[:max_len])
            padded_seqs.append(padded_seq)
        all_padded_sequences.append(
            torch.stack(padded_seqs)
        )  # [num_valid_seqs, max_len, n, n]

    # Create masks for the longest sequence in the batch (used for all)
    mask = torch.zeros(len(all_sequences), max_len, device=device)
    for i, perm_sequences in enumerate(all_sequences):
        # Use the length of the longest sequence for this sample
        seq_len = max(len(seq) for seq in perm_sequences)
        mask[i, :seq_len] = 1

    return all_padded_sequences, mask


class TableauPermutationDataset(Dataset):
    def __init__(self, data_file, n_qubits=None, max_samples=None, shuffle=True):
        print(f"Loading data from {data_file}...")
        with open(data_file, "rb") as f:
            all_data = pickle.load(f)

        # Detect n_qubits from the first tableau in the dataset
        if n_qubits is None and all_data:
            self.n_qubits = all_data[0][0].n_qubits
            print(f"Auto-detected {self.n_qubits} qubits from data")
        else:
            self.n_qubits = n_qubits

        # Option to load only a subset
        if max_samples is not None and max_samples < len(all_data):
            if shuffle:
                # Randomly select subset
                indices = torch.randperm(len(all_data))[:max_samples].tolist()
                self.data = [all_data[i] for i in indices]
            else:
                # Take first max_samples
                self.data = all_data[:max_samples]
            print(f"Using subset of {max_samples} examples from {len(all_data)} total")
        else:
            self.data = all_data
            print(f"Loaded all {len(self.data)} training examples")

    def __len__(self):
        return len(self.data)

    def tableau_to_tensor(self, tableau):
        # Get number of qubits
        n_qubits = tableau.n_qubits

        # Create tensor with 2 channels - first for tableau, second for signs
        combined = torch.zeros(2, 2 * n_qubits, 2 * n_qubits)

        # Get tableau data and signs
        tableau_data = tableau.tableau
        signs_data = tableau.signs

        # Convert numpy arrays to torch tensors if needed
        if isinstance(tableau_data, np.ndarray):
            tableau_data = torch.from_numpy(tableau_data).float()
        if isinstance(signs_data, np.ndarray):
            signs_data = torch.from_numpy(signs_data).float()

        # Fill first channel with tableau data
        combined[0, :, :] = tableau_data

        # Add signs to second channel (broadcast along columns)
        combined[1, :, 0] = signs_data

        return combined

    def __getitem__(self, idx):
        tableau, best_perms = self.data[idx]
        n_qubits = tableau.n_qubits

        # Get CX count from first permutation (all should be same)
        cx_count = best_perms[0][1] if best_perms else float("inf")
        # print(f"cx_count: {cx_count}")

        # print(best_perms)

        # For debugging
        # print(f"Best perms structure: {type(best_perms)}, length: {len(best_perms)}")

        # Create a list to hold all permutation sequences
        all_perm_sequences = []

        # Process each permutation sequence in best_perms
        for sequence in best_perms:
            # print(f"Processing sequence: {sequence}")
            # each `sequence` is a tuple [a list of permutation tuples, cx_count]

            # For each sequence, create a list of matrices
            perm_matrices = []
            current_perm = torch.eye(n_qubits)  # Initialize with identity

            for step in sequence:
                # print(f"Step: {step}")
                # Each step is a tuple (i,j) representing a swap
                if isinstance(step, tuple) and len(step) == 2:
                    i, j = step

                    # Convert to integers
                    if isinstance(i, (list, tuple)):
                        i = i[0] if i else 0
                    if isinstance(j, (list, tuple)):
                        j = j[0] if i else 0

                    i = int(i) if hasattr(i, "__int__") else 0
                    j = int(j) if hasattr(j, "__int__") else 0

                    # Create permutation matrix for this step
                    step_matrix = torch.eye(n_qubits)

                    # Safe row swapping
                    temp = step_matrix[i].clone()
                    step_matrix[i] = step_matrix[j]
                    step_matrix[j] = temp

                    # Compose with previous permutations
                    current_perm = step_matrix @ current_perm
                    perm_matrices.append(current_perm.clone())

            # If this sequence produced valid matrices, add it
            if perm_matrices:
                all_perm_sequences.append(perm_matrices)

        # If not create any valid sequences, add a default one
        if not all_perm_sequences:
            all_perm_sequences.append([torch.eye(n_qubits)])

        return self.tableau_to_tensor(tableau), (all_perm_sequences, cx_count)


# Example of usage:
# # For testing with a small random subset (320 samples)
# dataset = TableauPermutationDataset(
#     data_file="training_data_perm.pkl",
#     max_samples=320,
#     shuffle=True  # Random subset
# )

# # For using the first 320 samples (deterministic)
# dataset = TableauPermutationDataset(
#     data_file="training_data_perm.pkl",
#     max_samples=320,
#     shuffle=False  # First N samples
# )

# # For final training with all data
# dataset = TableauPermutationDataset(
#     data_file="training_data_perm.pkl"  # No max_samples means use all data
# )


def custom_collate_fn(batch):
    """
    Custom collate function to handle variable-length permutation lists
    """
    # Extract tableaus and targets
    tableaus = [item[0] for item in batch]
    targets = [item[1] for item in batch]

    # Stack tableaus (they should all have the same shape)
    tableaus = torch.stack(tableaus)

    # Don't try to stack targets, just return them as a list
    return tableaus, targets


def pretrain_a_model_from_file(data_file, max_samples, epochs, device):
    dataset = TableauPermutationDataset(
        data_file, n_qubits=None, max_samples=max_samples
    )
    # model = OrderedPermutationTransformer(n_qubits=dataset.n_qubits)
    model = OrderedPermutationTransformer(
        n_qubits=dataset.n_qubits, dim=256, num_layers=12
    )  # Try wider model 6 -> 12

    # Use standard PyTorch DataLoader with custom collate function
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        collate_fn=custom_collate_fn,
    )

    # # Auto-detect device
    # # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # train(model, dataloader, epochs=10, device=device)
    train(
        model, dataloader, epochs, device
    )  # 100 epochs too much, plateau at 50; well, 50 epochs seem too much too
    return model


def convert_tensor_to_tableau(tableau_tensor):
    """Convert tensor representation back to CliffordTableau object"""
    # Extract tableau data from tensor
    n_qubits = tableau_tensor.shape[-1] // 2

    # Convert to int8 for proper bitwise operations (important fix!)
    tableau_data = tableau_tensor[0, :, :].cpu().numpy().astype(np.int8)
    signs_data = tableau_tensor[1, :, 0].cpu().numpy().astype(np.int8)

    # Create a new tableau
    tableau = CliffordTableau(n_qubits)
    tableau.tableau = tableau_data
    tableau.signs = signs_data

    return tableau


# # This function has some bugs, so not ready to use
# def rl_fine_tune(model, dataset, epochs=5, device="cpu"):
#     """Fine-tune permutation model with RL targeting CX reduction"""
#     model.train()
#     optimizer = torch.optim.Adam(model.parameters(), lr=5e-6)
#     reward_history = []  # For reward normalization

#     for epoch in range(epochs):
#         total_reward = 0
#         dataloader = torch.utils.data.DataLoader(
#             dataset, batch_size=1, shuffle=True, collate_fn=custom_collate_fn
#         )

#         for batch_idx, (tableau_tensors, _) in enumerate(dataloader):
#             # Get tableau and convert to proper format
#             tableau_tensor = tableau_tensors[0]
#             clifford_tableau = convert_tensor_to_tableau(tableau_tensor)
#             tableau_tensor = tableau_tensor.unsqueeze(0).to(device)

#             # BASELINE: Get CX count with normal heuristic
#             topology = Topology.complete(clifford_tableau.n_qubits)
#             baseline_circuit = synthesize_tableau_perm_row_col(
#                 clifford_tableau, topology
#             )
#             baseline_cx = collect_circuit_data(baseline_circuit)["cx"]

#             # Get model prediction with gradient tracking
#             optimizer.zero_grad()
#             raw_preds, _ = model(tableau_tensor)
#             logits = raw_preds[0, 0]

#             # Apply Sinkhorn with gradient tracking
#             for _ in range(20):
#                 logits = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
#                 logits = logits - torch.logsumexp(logits, dim=-2, keepdim=True)

#             # Sample permutation using Gumbel-Softmax
#             n_qubits = logits.shape[0]
#             gumbels = -torch.log(-torch.log(torch.rand_like(logits)))
#             gumbel_logits = (logits + gumbels) / 0.5  # Temperature
#             probs = F.softmax(gumbel_logits, dim=-1)

#             # Extract permutation using Hungarian algorithm
#             perm_matrix = probs.detach().cpu().numpy()
#             row_ind, col_ind = linear_sum_assignment(-perm_matrix)
#             perm = [(int(i), int(j)) for i, j in zip(row_ind, col_ind)]

#             # Track log probabilities for gradient flow
#             log_probs = []
#             for i, j in zip(row_ind, col_ind):
#                 log_probs.append(torch.log(probs[i, j] + 1e-10))

#             # Evaluate CX count (detached from graph)
#             with torch.no_grad():
#                 test_tableau = copy.deepcopy(clifford_tableau)
#                 pred_iter = iter(perm)

#                 def pred_callback(G, remaining, remaining_rows, choice_fn=min):
#                     try:
#                         row, col = next(pred_iter)
#                         if isinstance(row, torch.Tensor):
#                             row = row.item()
#                         if isinstance(col, torch.Tensor):
#                             col = col.item()

#                         # CRITICAL: Validate pivot exists in graph
#                         graph_nodes = set(G.nodes())
#                         if (
#                             row in remaining_rows
#                             and row in graph_nodes
#                             and col in graph_nodes
#                             and G.has_edge(row, col)
#                         ):
#                             return int(row), int(col)
#                         else:
#                             raise StopIteration

#                     except StopIteration:
#                         # Safe fallback using G
#                         valid_rows = [r for r in remaining_rows if r in G]
#                         if not valid_rows:
#                             row = min(remaining_rows)
#                             return row, row

#                         row = choice_fn(valid_rows)
#                         if G[row]:
#                             col = next(iter(G[row]))  # Guaranteed safe
#                         else:
#                             col = row
#                         return int(row), int(col)

#                 circuit = synthesize_tableau_perm_row_col(
#                     test_tableau, topology, pick_pivot_callback=pred_callback
#                 )
#                 cx_count = collect_circuit_data(circuit)["cx"]

#             # Calculate reward with normalization
#             reward = baseline_cx - cx_count
#             reward_history.append(reward)

#             if len(reward_history) > 10:
#                 mean_reward = sum(reward_history[-10:]) / 10
#                 std_reward = max(1.0, np.std(reward_history[-10:]))
#                 normalized_reward = (reward - mean_reward) / std_reward
#             else:
#                 normalized_reward = reward

#             # REINFORCE loss
#             policy_loss = -sum(log_probs) * normalized_reward

#             # Backprop and update
#             policy_loss.backward()
#             optimizer.step()

#             total_reward += reward

#             if batch_idx % 10 == 0:
#                 print(
#                     f"Epoch {epoch+1} | Batch {batch_idx} | Reward: {reward:.2f} | CX: {cx_count}"
#                 )

#         print(f"Epoch {epoch+1} | Avg Reward: {total_reward/len(dataloader):.2f}")

#     return model


# Ready to use, but no big improvement as expected! So, did not used mostly.
def supervised_cx_fine_tune(model, dataset, epochs, device):
    """Fine-tune with supervised learning focusing on CX reduction"""
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    criterion = OrderedPermutationLoss()

    for epoch in range(epochs):
        total_improvement = 0
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=8, shuffle=True, collate_fn=custom_collate_fn
        )

        for batch_idx, (tableau_tensors, _) in enumerate(dataloader):
            # For each tableau in batch
            for tableau_tensor in tableau_tensors:
                optimizer.zero_grad()
                clifford_tableau = convert_tensor_to_tableau(tableau_tensor)
                tableau_tensor = tableau_tensor.unsqueeze(0).to(device)

                # Get baseline CX count
                topology = Topology.complete(clifford_tableau.n_qubits)
                baseline_circuit = synthesize_tableau_perm_row_col(
                    clifford_tableau, topology
                )
                baseline_cx = collect_circuit_data(baseline_circuit)["cx"]

                # Generate permutation candidates and find best one
                with torch.no_grad():
                    raw_preds, _, cx_preds = model(tableau_tensor)
                    target_perm = None
                    best_cx = baseline_cx

                    # Try using CX predictions to guide step selection
                    cx_weights = torch.exp(
                        -cx_preds[0] * 2
                    )  # Higher weight for lower CX predictions
                    step_probs = F.softmax(cx_weights, dim=0)
                    step_indices = torch.multinomial(
                        step_probs, min(8, raw_preds.shape[1]), replacement=False
                    )

                    # Try steps probabilistically selected based on predicted CX efficiency
                    for step_idx in step_indices:
                        # # Try different permutations
                        # for step_idx in range(min(3, raw_preds.shape[1])):
                        logits = raw_preds[0, step_idx]
                        for _ in range(20):
                            logits = logits - torch.logsumexp(
                                logits, dim=-1, keepdim=True
                            )
                            logits = logits - torch.logsumexp(
                                logits, dim=-2, keepdim=True
                            )

                        perm_matrix = torch.exp(logits / 0.1).cpu().numpy()
                        row_ind, col_ind = linear_sum_assignment(-perm_matrix)
                        perm = [(int(i), int(j)) for i, j in zip(row_ind, col_ind)]
                        pred_iter = iter(perm)

                        def pred_callback(G, remaining, remaining_rows, choice_fn=min):
                            try:
                                row, col = next(pred_iter)
                                # Convert to int if they're tensors
                                if isinstance(row, torch.Tensor):
                                    row = row.item()
                                if isinstance(col, torch.Tensor):
                                    col = col.item()
                                return int(row), int(col)  # Ensure integers
                            # except StopIteration:
                            #     row = choice_fn(remaining_rows)
                            #     return row, row

                            # Try smarter fallback, also no big improvement as expected
                            except StopIteration:
                                # Use graph analysis for better pivot selection
                                row = choice_fn(remaining_rows)

                                if G[row]:
                                    # Choose column with highest impact
                                    impact_scores = {}
                                    for col in G[row]:
                                        # Count affected rows
                                        impact = sum(
                                            1 for r in remaining_rows if col in G[r]
                                        )
                                        impact_scores[col] = impact

                                    col = max(
                                        impact_scores.items(), key=lambda x: x[1]
                                    )[0]
                                else:
                                    col = row

                                return int(row), int(col)

                        circuit = synthesize_tableau_perm_row_col(
                            clifford_tableau,
                            topology,
                            pick_pivot_callback=pred_callback,
                        )
                        cx_count = collect_circuit_data(circuit)["cx"]
                        if cx_count < best_cx:
                            best_cx = cx_count
                            target_perm = perm

                # # If we found a better permutation, train toward it
                # if target_perm is not None and best_cx < baseline_cx:
                #     # Create target matrix
                #     target = torch.zeros_like(raw_preds[0, 0])
                #     for i, j in target_perm:
                #         target[i, j] = 1.0

                #     # Train model to predict this permutation
                #     new_preds, _, _ = model(tableau_tensor)
                #     logits = new_preds[0, 0]
                #     loss = F.mse_loss(torch.softmax(logits, dim=-1), target)
                #     loss.backward()
                #     optimizer.step()

                #     total_improvement += baseline_cx - best_cx

                # Try keeping consistent with OrderedPermutationLoss
                # If we found a better permutation, train toward it
                if target_perm is not None and best_cx < baseline_cx:
                    # Create target matrix
                    target = torch.zeros_like(raw_preds[0, 0])
                    for i, j in target_perm:
                        target[i, j] = 1.0

                    # Prepare targets and masks in the format expected by OrderedPermutationLoss
                    target_tensor = target.unsqueeze(0).unsqueeze(0)  # [1, 1, n, n]
                    mask_tensor = torch.ones(1, 1, device=device)  # [1, 1]

                    # Train model to predict this permutation
                    new_preds, stop_logits, new_cx_preds = model(tableau_tensor)

                    # Apply Sinkhorn normalization (same as in train function)
                    logits = new_preds[0, 0]
                    for _ in range(20):
                        logits = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
                        logits = logits - torch.logsumexp(logits, dim=-2, keepdim=True)
                    pred_tensor = (
                        torch.exp(logits / 0.1).unsqueeze(0).unsqueeze(0)
                    )  # [1, 1, n, n]

                    # Try adding strong explicit CX training (add this block)
                    cx_target = torch.tensor([best_cx], device=device).float()
                    cx_weight = 5.0  # Stronger weight for explicit CX training
                    cx_loss = cx_weight * F.mse_loss(
                        new_cx_preds[0, 0].unsqueeze(0), cx_target
                    )
                    cx_loss.backward(
                        retain_graph=True
                    )  # First backward pass for CX only

                    # Use OrderedPermutationLoss for consistent training objectives
                    cx_target_tensor = torch.tensor([[best_cx]], device=device).float()

                    loss = criterion(
                        pred_tensor,  # [1, 1, n, n]
                        stop_logits[:, 0:1],  # [1, 1]
                        target_tensor,  # [1, 1, n, n]
                        mask_tensor,  # [1, 1]
                        cx_preds=new_cx_preds[:, 0:1],  # [1, 1]
                        cx_targets=cx_target_tensor,  # [1, 1]
                    )

                    loss.backward()
                    optimizer.step()

                    total_improvement += baseline_cx - best_cx

            if batch_idx % 10 == 0:
                print(
                    f"Epoch {epoch+1} | Batch {batch_idx} | Improvement: {total_improvement/(batch_idx+1):.2f}"
                )

        print(
            f"Epoch {epoch+1} | Avg CX Improvement: {total_improvement/len(dataloader):.2f}"
        )

    return model


# def predict_permutation(model, clifford_tableau, device="cpu"):
#     """
#     Predicts valid permutation tuples for a Clifford tableau using the trained model.
#     Returns a list of permutation tuples, potentially as a sequence.
#     """
#     # 1. Convert tableau to tensor format
#     n_qubits = clifford_tableau.n_qubits
#     tableau_tensor = TableauPermutationDataset.tableau_to_tensor(None, clifford_tableau)
#     tableau_tensor = tableau_tensor.unsqueeze(0).to(device)  # Add batch dimension

#     # 2. Model forward pass
#     model.eval()
#     with torch.no_grad():
#         # For transformer model
#         raw_preds, stop_logits = model(tableau_tensor)

#         # Apply Sinkhorn normalization if available
#         try:
#             sinkhorn = SequentialSinkhorn()
#             preds = sinkhorn(raw_preds)
#         except:
#             preds = F.softmax(raw_preds, dim=-1)

#         # Return sequence of permutations
#         permutations = []
#         for t in range(preds.size(0)):
#             # Stop when stop token is activated
#             if t > 0 and torch.sigmoid(stop_logits[t - 1, 0]) > 0.5:
#                 break

#             # Use Hungarian algorithm to get permutation
#             perm_matrix = preds[t, 0].cpu().numpy()
#             row_ind, col_ind = linear_sum_assignment(-perm_matrix)  # Maximize
#             perm = [(int(r), int(c)) for r, c in zip(row_ind, col_ind)]
#             permutations.append(perm)

#         return permutations


def tableau_to_tensor(tableau):
    # Get number of qubits
    n_qubits = tableau.n_qubits

    # Create tensor with 2 channels - first for tableau, second for signs
    combined = torch.zeros(2, 2 * n_qubits, 2 * n_qubits)

    # Get tableau data and signs
    tableau_data = tableau.tableau
    signs_data = tableau.signs

    # Convert numpy arrays to torch tensors if needed
    if isinstance(tableau_data, np.ndarray):
        tableau_data = torch.from_numpy(tableau_data).float()
    if isinstance(signs_data, np.ndarray):
        signs_data = torch.from_numpy(signs_data).float()

    # Fill first channel with tableau data
    combined[0, :, :] = tableau_data

    # Add signs to second channel (broadcast along columns)
    combined[1, :, 0] = signs_data

    return combined


# def predict_permutation(model, clifford_tableau, device="cpu"):
#     """
#     Predicts optimal permutation steps for a Clifford tableau.
#     Returns a list of permutation tuples as (i,j) swap operations.
#     """
#     # Set device
#     device = torch.device(device)
#     model = model.to(device)

#     # 1. Convert tableau to tensor format
#     # dataset = TableauPermutationDataset(None)
#     tableau_tensor = tableau_to_tensor(clifford_tableau)
#     tableau_tensor = tableau_tensor.unsqueeze(0).to(device)  # Add batch dimension

#     # 2. Model forward pass
#     model.eval()
#     with torch.no_grad():
#         raw_preds, stop_logits = model(tableau_tensor)

#         # Apply Sinkhorn normalization to the FIRST step only
#         logits = raw_preds[0, 0]  # First batch item, first step
#         # Apply Sinkhorn
#         for _ in range(20):
#             logits = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
#             logits = logits - torch.logsumexp(logits, dim=-2, keepdim=True)
#         perm_matrix = torch.exp(logits / 0.1).cpu().numpy()

#         # Use Hungarian algorithm to find optimal assignment
#         row_ind, col_ind = linear_sum_assignment(-perm_matrix)  # Maximize

#         # Return a single permutation
#         permutation = [(int(i), int(j)) for i, j in zip(row_ind, col_ind)]
#         return [permutation]  # Wrap in list for compatibility


def collect_circuit_data(circuit: Circuit) -> dict:
    circuit.final_permutation = None
    ops = circuit.to_qiskit().count_ops()
    return {
        "num_qubits": circuit.n_qubits,
        "h": ops.get("h", 0),
        "s": ops.get("s", 0),
        "cx": ops.get("cx", 0),
        "depth": circuit.to_qiskit().depth(),
    }


def compute_weighted_score(pred_metrics):
    """
    Computes a weighted score based on the number of CX gates and circuit depth.
    The weights can be adjusted based on the importance of each metric.
    """
    # Define weights for each metric
    cx_weight = 10.0  # Try adjusting from 10 to 50; 50 not good either
    depth_weight = 1.0  # 1.0 seems better than 0.1???
    # Compute weighted score, only keep cx seems worse
    score = cx_weight * pred_metrics["cx"] + depth_weight * pred_metrics["depth"]
    # score = pred_metrics["cx"]
    return score


# def compute_adaptive_score(pred_metrics, baseline_metrics):
#     """Adapt weights based on how close we are to optimum"""
#     cx_ratio = pred_metrics["cx"] / baseline_metrics["cx"]
#     depth_ratio = pred_metrics["depth"] / baseline_metrics["depth"]

#     # If CX is much worse than depth, focus more on CX
#     if cx_ratio > depth_ratio * 1.2:
#         cx_weight = 20.0
#         depth_weight = 0.5
#     else:
#         cx_weight = 10.0
#         depth_weight = 1.0

#     return cx_weight * pred_metrics["cx"] + depth_weight * pred_metrics["depth"]


def predict_permutation(model, clifford_tableau, device):
    """Returns the best permutation after evaluating multiple candidates"""
    device = torch.device(device)
    model = model.to(device)

    # Get tableau tensor
    tableau_tensor = tableau_to_tensor(clifford_tableau)
    tableau_tensor = tableau_tensor.unsqueeze(0).to(device)

    # Store best permutation and its score
    best_perm = None
    best_score = float("inf")
    topology = Topology.complete(clifford_tableau.n_qubits)

    with torch.no_grad():
        # Get predictions for all steps
        raw_preds, _, cx_preds = model(tableau_tensor)

        # # Sort steps by cx count
        # step_indices = torch.argsort(
        #     cx_preds[0], descending=True
        # ).tolist()  # Deterministic order

        # Try a probabilistic approach
        # Weight step selection by predicted CX reduction potential
        cx_weights = torch.exp(
            -cx_preds[0] * 2
        )  # Higher weight for lower CX predictions
        step_probs = F.softmax(cx_weights, dim=0)
        step_indices = torch.multinomial(
            step_probs, min(8, raw_preds.shape[1]), replacement=False
        )

        # Try the top 3 steps and keep the best one
        # for step_idx in range(min(8, raw_preds.shape[1])):  # try 5 instead of 3
        for step_idx in step_indices:  # Integrate cx_preds
            # Apply Sinkhorn normalization
            logits = raw_preds[0, step_idx]
            for _ in range(20):
                logits = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
                logits = logits - torch.logsumexp(logits, dim=-2, keepdim=True)

            # Try different temperatures, no big improvement as expected
            for temp in [0.01, 0.02, 0.05, 0.1, 0.2, 0.5]:
                # perm_matrix = torch.exp(logits / 0.1).cpu().numpy()
                perm_matrix = torch.exp(logits / temp).cpu().numpy()  # Temperatures

                # Extract permutation using Hungarian algorithm
                row_ind, col_ind = linear_sum_assignment(-perm_matrix)
                perm = [(int(i), int(j)) for i, j in zip(row_ind, col_ind)]

                pred_iter = iter(perm)

                def pred_callback(G, remaining, remaining_rows, choice_fn=min):
                    try:
                        row, col = next(pred_iter)
                        # Convert to int if they're tensors
                        if isinstance(row, torch.Tensor):
                            row = row.item()
                        if isinstance(col, torch.Tensor):
                            col = col.item()
                        return int(row), int(col)  # Ensure integers
                    # except StopIteration:
                    #     row = choice_fn(remaining_rows)
                    #     return row, row

                    # Try smarter fallback, also no big improvement as expected
                    except StopIteration:
                        # Use graph analysis for better pivot selection
                        row = choice_fn(remaining_rows)

                        if G[row]:
                            # Choose column with highest impact
                            impact_scores = {}
                            for col in G[row]:
                                # Count affected rows
                                impact = sum(1 for r in remaining_rows if col in G[r])
                                impact_scores[col] = impact

                            col = max(impact_scores.items(), key=lambda x: x[1])[0]
                        else:
                            col = row

                        return int(row), int(col)

                pred_circuit = synthesize_tableau_perm_row_col(
                    clifford_tableau, topology, pick_pivot_callback=pred_callback
                )
                pred_metrics = collect_circuit_data(pred_circuit)

                # Synthesize circuit and count gates
                # score = pred_metrics["cx"] + pred_metrics["depth"] # Works fine already
                score = compute_weighted_score(pred_metrics)
                # score = compute_adaptive_score(pred_metrics, collect_circuit_data(circuit))

                # Keep track of best permutation
                if score < best_score:
                    best_score = score
                    best_perm = perm

    # # Fall back to first permutation if nothing improved the score
    # if best_perm is None:
    #     row_ind, col_ind = linear_sum_assignment(
    #         -torch.exp(raw_preds[0, 0] / 0.1).cpu().numpy()
    #     )
    #     best_perm = [(int(i), int(j)) for i, j in zip(row_ind, col_ind)]

    # # Filter out identity mappings
    # best_perm = [(i, j) for i, j in best_perm if i != j]

    # print(f"Best score: {best_score}")
    return [best_perm]  # Keep list format for compatibility


def gumbel_sinkhorn(logits, temp=0.1, n_samples=5):
    samples = []
    for _ in range(n_samples):
        # Generate fresh Gumbel noise for each sample
        gumbels = -torch.log(-torch.log(torch.rand_like(logits) + 1e-10) + 1e-10)
        noisy_logits = (logits + gumbels) / temp

        # Apply Sinkhorn normalization
        s = noisy_logits.clone()  # Important: clone to avoid in-place modification
        for _ in range(20):
            s = s - torch.logsumexp(s, dim=-1, keepdim=True)
            s = s - torch.logsumexp(s, dim=-2, keepdim=True)
        # samples.append(torch.exp(s))
        # Try Add final softmax instead of exp() for better numerical stability (results?)
        samples.append(F.softmax(s, dim=-1))

    # Return all samples (don't average - keep the diversity)
    return samples


def tiny_gumbel_sinkhorn(logits, temp=1.0, n_iters=20):
    gumbels = -torch.log(-torch.log(torch.rand_like(logits) + 1e-10))
    noisy_logits = (logits + gumbels) / temp
    for _ in range(n_iters):
        noisy_logits = noisy_logits - torch.logsumexp(
            noisy_logits, dim=-1, keepdim=True
        )
        noisy_logits = noisy_logits - torch.logsumexp(
            noisy_logits, dim=-2, keepdim=True
        )
    return torch.exp(noisy_logits)


def predict_permutation_gumbel(model, clifford_tableau, device):
    """Returns the best permutation after evaluating multiple candidates"""
    device = torch.device(device)
    model = model.to(device)

    # Get tableau tensor
    tableau_tensor = tableau_to_tensor(clifford_tableau)
    tableau_tensor = tableau_tensor.unsqueeze(0).to(device)

    # Store best permutation and its score
    best_perm = None
    best_score = float("inf")
    topology = Topology.complete(clifford_tableau.n_qubits)

    with torch.no_grad():
        # Get predictions for all steps
        raw_preds, _, cx_preds = model(tableau_tensor)

        # # Sort steps by cx count
        # step_indices = torch.argsort(
        #     cx_preds[0], descending=True
        # ).tolist()  # Deterministic order

        # Try a probabilistic approach
        # Weight step selection by predicted CX reduction potential
        cx_weights = torch.exp(
            -cx_preds[0] * 2
        )  # Higher weight for lower CX predictions
        step_probs = F.softmax(cx_weights, dim=0)
        step_indices = torch.multinomial(
            step_probs, min(8, raw_preds.shape[1]), replacement=False
        )

        # Try the top 3 steps and keep the best one
        # for step_idx in range(min(8, raw_preds.shape[1])):  # try 5 instead of 3
        for step_idx in step_indices:  # Integrate cx_preds
            # Apply Sinkhorn normalization
            logits = raw_preds[0, step_idx]

            # Generate multiple permutation samples with Gumbel-Sinkhorn
            perm_samples = gumbel_sinkhorn(logits, temp=0.1, n_samples=10)
            for sample_matrix in perm_samples:
                perm_matrix = sample_matrix.cpu().numpy()

                # Extract permutation using Hungarian algorithm
                row_ind, col_ind = linear_sum_assignment(-perm_matrix)
                perm = [(int(i), int(j)) for i, j in zip(row_ind, col_ind)]

                pred_iter = iter(perm)

                def pred_callback(G, remaining, remaining_rows, choice_fn=min):
                    try:
                        row, col = next(pred_iter)
                        # Convert to int if they're tensors
                        if isinstance(row, torch.Tensor):
                            row = row.item()
                        if isinstance(col, torch.Tensor):
                            col = col.item()
                        return int(row), int(col)  # Ensure integers
                    # except StopIteration:
                    #     row = choice_fn(remaining_rows)
                    #     return row, row

                    # Try smarter fallback, also no big improvement as expected
                    except StopIteration:
                        # Use graph analysis for better pivot selection
                        row = choice_fn(remaining_rows)

                        if G[row]:
                            # Choose column with highest impact
                            impact_scores = {}
                            for col in G[row]:
                                # Count affected rows
                                impact = sum(1 for r in remaining_rows if col in G[r])
                                impact_scores[col] = impact

                            col = max(impact_scores.items(), key=lambda x: x[1])[0]
                        else:
                            col = row

                        return int(row), int(col)

                pred_circuit = synthesize_tableau_perm_row_col(
                    clifford_tableau, topology, pick_pivot_callback=pred_callback
                )
                pred_metrics = collect_circuit_data(pred_circuit)

                # Synthesize circuit and count gates
                # score = pred_metrics["cx"] + pred_metrics["depth"] # Works fine already
                score = compute_weighted_score(pred_metrics)
                # score = compute_adaptive_score(pred_metrics, collect_circuit_data(circuit))

                # Keep track of best permutation
                if score < best_score:
                    best_score = score
                    best_perm = perm

    # # Fall back to first permutation if nothing improved the score
    # if best_perm is None:
    #     row_ind, col_ind = linear_sum_assignment(
    #         -torch.exp(raw_preds[0, 0] / 0.1).cpu().numpy()
    #     )
    #     best_perm = [(int(i), int(j)) for i, j in zip(row_ind, col_ind)]

    # # Filter out identity mappings
    # best_perm = [(i, j) for i, j in best_perm if i != j]

    # print(f"Best score: {best_score}")
    return [best_perm]  # Keep list format for compatibility


# Try beam search based on the predict_permutation_gumbel function
# Still in development, not ready to use, contains some bugs
def predict_permutation_beam(model, clifford_tableau, device="cpu", beam_width=5):
    """Uses beam search to find the best permutation"""
    device = torch.device(device)
    model = model.to(device)

    tableau_tensor = tableau_to_tensor(clifford_tableau)
    tableau_tensor = tableau_tensor.unsqueeze(0).to(device)
    n_qubits = clifford_tableau.n_qubits
    topology = Topology.complete(n_qubits)

    with torch.no_grad():
        raw_preds, _, cx_preds = model(tableau_tensor)

        # Initialize beam with empty permutation
        beam = [([], 0)]  # (partial_perm, score)

        # For each position in permutation
        for step in range(n_qubits):
            candidates = []

            # For each partial permutation in beam
            for partial_perm, score in beam:
                # Weight step selection by predicted CX
                cx_weights = torch.exp(-cx_preds[0] * 2)
                step_probs = F.softmax(cx_weights, dim=0)
                step_indices = torch.multinomial(
                    step_probs, min(3, raw_preds.shape[1]), replacement=False
                )

                # Try different model-predicted steps
                for step_idx in step_indices:
                    logits = raw_preds[0, step_idx]

                    # Sample with Gumbel-Sinkhorn at different temperatures
                    for temp in [0.05, 0.1, 0.2]:
                        perm_samples = gumbel_sinkhorn(logits, temp=temp, n_samples=3)

                        for sample_matrix in perm_samples:
                            perm_matrix = sample_matrix.cpu().numpy()
                            row_ind, col_ind = linear_sum_assignment(-perm_matrix)
                            full_perm = [
                                (int(i), int(j)) for i, j in zip(row_ind, col_ind)
                            ]

                            # Get next mapping for this step
                            if step < len(full_perm):
                                next_elem = full_perm[step]

                                # Create new candidate permutation
                                new_perm = partial_perm + [next_elem]

                                # Skip if invalid or incomplete
                                if len(new_perm) <= step:
                                    continue

                                # Evaluate with synthesis
                                pred_iter = iter(new_perm)

                                def pred_callback(
                                    G, remaining, remaining_rows, choice_fn=min
                                ):
                                    try:
                                        row, col = next(pred_iter)
                                        if isinstance(row, torch.Tensor):
                                            row = row.item()
                                        if isinstance(col, torch.Tensor):
                                            col = col.item()

                                        # Check if this pivot is valid in the tableau
                                        if (
                                            hasattr(remaining, "_x_out")
                                            and remaining._x_out(int(row), int(col))
                                            == 1
                                        ):
                                            return int(row), int(col)
                                        else:
                                            # If not valid, fall back to the default approach
                                            raise StopIteration

                                    except StopIteration:
                                        # SAFE FALLBACK: Scan the entire tableau for ANY valid pivot
                                        # This is guaranteed to find a pivot if the tableau is valid
                                        for r in remaining_rows:
                                            for c in range(remaining.n_qubits * 2):
                                                if remaining._x_out(r, c) == 1:
                                                    return int(r), int(c)

                                        # Last resort: If somehow we got here (shouldn't happen with valid tableaus)
                                        # Just return the first remaining row with itself as column
                                        if remaining_rows:
                                            row = min(remaining_rows)
                                            return int(row), int(row)
                                        else:
                                            # Absolute last resort - should never reach here with valid inputs
                                            return 0, 0

                                circuit = synthesize_tableau_perm_row_col(
                                    clifford_tableau,
                                    topology,
                                    pick_pivot_callback=pred_callback,
                                )
                                metrics = collect_circuit_data(circuit)
                                new_score = compute_weighted_score(metrics)
                                candidates.append((new_perm, new_score))

            # Keep top-k candidates for next step
            beam = sorted(candidates, key=lambda x: x[1])[:beam_width]

        # Get best permutation from final beam
        if beam:
            best_perm = beam[0][0]
            # # Apply local refinement to the best permutation (too computationally expensive)
            # best_perm = local_search_refine(clifford_tableau, best_perm, topology)
            return [best_perm]
        else:
            # Fallback to identity permutation
            return [[(i, i) for i in range(n_qubits)]]


def entropy_guided_search(model, clifford_tableau, device, n_samples=5):
    """Focus search on areas where model is uncertain to find better permutations"""
    device = torch.device(device)
    model = model.to(device)

    tableau_tensor = tableau_to_tensor(clifford_tableau)
    tableau_tensor = tableau_tensor.unsqueeze(0).to(device)
    topology = Topology.complete(clifford_tableau.n_qubits)

    with torch.no_grad():
        # Forward pass to get raw predictions
        raw_preds, _, cx_preds = model(tableau_tensor)

        # Calculate entropy for each position (high entropy = uncertainty)
        entropies = []
        for step in range(raw_preds.shape[1]):
            logits = raw_preds[0, step]

            # Apply Sinkhorn normalization
            for _ in range(20):
                logits = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
                logits = logits - torch.logsumexp(logits, dim=-2, keepdim=True)

            # Convert to probabilities
            probs = torch.exp(logits)

            # Calculate entropy: -sum(p * log(p))
            entropy = -torch.sum(probs * torch.log(probs + 1e-10))
            entropies.append((step, entropy.item()))

        # Sort steps by uncertainty (highest entropy first)
        uncertain_steps = sorted(entropies, key=lambda x: -x[1])
        # print(f"Step entropies: {[f'{i}:{e:.2f}' for i, e in uncertain_steps[:3]]}")

        # Store best permutation and score
        best_perm = None
        best_score = float("inf")
        best_metrics = None

        # Explore uncertain steps more thoroughly
        for step_idx, entropy in uncertain_steps[:3]:  # Focus on top-3 most uncertain
            # Use adaptive temperature range - more temps for higher entropy
            temps = [0.01, 0.03, 0.05, 0.1, 0.2, 0.5]
            if entropy > 3.0:  # Very uncertain
                temps.extend([0.7, 1.0])  # Add higher temps for exploration

            # Sample more permutations for high-entropy areas
            samples_for_step = n_samples + int(3 * entropy)

            # print(f"Step {step_idx}: entropy={entropy:.2f}, samples={samples_for_step}")

            for temp in temps:
                # Generate multiple samples with Gumbel-Sinkhorn
                perm_samples = gumbel_sinkhorn(
                    raw_preds[0, step_idx], temp=temp, n_samples=samples_for_step
                )

                for sample_matrix in perm_samples:
                    perm_matrix = sample_matrix.cpu().numpy()
                    row_ind, col_ind = linear_sum_assignment(-perm_matrix)
                    perm = [(int(i), int(j)) for i, j in zip(row_ind, col_ind)]

                    # Evaluate permutation
                    pred_iter = iter(perm)

                    def pred_callback(G, remaining, remaining_rows, choice_fn=min):
                        try:
                            row, col = next(pred_iter)
                            # Convert to int if they're tensors
                            if isinstance(row, torch.Tensor):
                                row = row.item()
                            if isinstance(col, torch.Tensor):
                                col = col.item()
                            return int(row), int(col)  # Ensure integers
                        # except StopIteration:
                        #     row = choice_fn(remaining_rows)
                        #     return row, row

                        # Try smarter fallback, also no big improvement as expected
                        except StopIteration:
                            # Use graph analysis for better pivot selection
                            row = choice_fn(remaining_rows)

                            if G[row]:
                                # Choose column with highest impact
                                impact_scores = {}
                                for col in G[row]:
                                    # Count affected rows
                                    impact = sum(
                                        1 for r in remaining_rows if col in G[r]
                                    )
                                    impact_scores[col] = impact

                                col = max(impact_scores.items(), key=lambda x: x[1])[0]
                            else:
                                col = row

                            return int(row), int(col)

                    circuit = synthesize_tableau_perm_row_col(
                        clifford_tableau, topology, pick_pivot_callback=pred_callback
                    )
                    metrics = collect_circuit_data(circuit)
                    score = compute_weighted_score(metrics)

                    # Update best if improved
                    if score < best_score:
                        best_score = score
                        best_perm = perm
                        best_metrics = metrics
                        # print(
                        #     f"Found better: CX={metrics['cx']}, Depth={metrics['depth']}"
                        # )

        # In case nothing improved, fall back to basic prediction
        if best_perm is None:
            # Use basic Gumbel-Sinkhorn with default temp
            perm_samples = gumbel_sinkhorn(raw_preds[0, 0], temp=0.1, n_samples=1)
            perm_matrix = perm_samples[0].cpu().numpy()
            row_ind, col_ind = linear_sum_assignment(-perm_matrix)
            best_perm = [(int(i), int(j)) for i, j in zip(row_ind, col_ind)]

        # if best_metrics:
        #     print(
        #         f"Best permutation: CX={best_metrics['cx']}, Depth={best_metrics['depth']}"
        #     )

        return [best_perm]  # Keep list format for compatibility


def ensemble_predict_permutation(model, tableau, device):
    """Run multiple permutation prediction strategies and select best result"""
    candidates = []

    # Run all prediction strategies
    gumbel_perms = predict_permutation_gumbel(model, tableau, device)
    entropy_perms = entropy_guided_search(model, tableau, device, n_samples=8)

    # Add more aggressive temperature sampling
    tableau_tensor = tableau_to_tensor(tableau).unsqueeze(0).to(device)
    with torch.no_grad():
        raw_preds, _, cx_preds = model(tableau_tensor)
        # Try extreme temperatures for more diversity
        for temp in [0.005, 1.0]:  # Very low and very high temps
            perm_samples = gumbel_sinkhorn(raw_preds[0, 0], temp=temp, n_samples=5)
            for sample in perm_samples:
                perm_matrix = sample.cpu().numpy()
                row_ind, col_ind = linear_sum_assignment(-perm_matrix)
                candidates.append([(int(i), int(j)) for i, j in zip(row_ind, col_ind)])

    # Add the candidates from standard methods
    candidates.extend(gumbel_perms)
    candidates.extend(entropy_perms)

    # Evaluate all candidates
    topology = Topology.complete(tableau.n_qubits)
    best_perm = None
    best_score = float("inf")
    best_metrics = None

    for perm in candidates:
        # Evaluate permutation
        pred_iter = iter(perm)

        def pred_callback(G, remaining, remaining_rows, choice_fn=min):
            try:
                row, col = next(pred_iter)
                # Convert to int if they're tensors
                if isinstance(row, torch.Tensor):
                    row = row.item()
                if isinstance(col, torch.Tensor):
                    col = col.item()
                return int(row), int(col)  # Ensure integers
            except StopIteration:
                row = choice_fn(remaining_rows)
                return row, row
                # Try smarter fallback, also no big improvement as expected
            except StopIteration:
                # Use graph analysis for better pivot selection
                row = choice_fn(remaining_rows)

                if G[row]:
                    # Choose column with highest impact
                    impact_scores = {}
                    for col in G[row]:
                        # Count affected rows
                        impact = sum(1 for r in remaining_rows if col in G[r])
                        impact_scores[col] = impact

                    col = max(impact_scores.items(), key=lambda x: x[1])[0]
                else:
                    col = row

                return int(row), int(col)

        circuit = synthesize_tableau_perm_row_col(
            tableau, topology, pick_pivot_callback=pred_callback
        )
        metrics = collect_circuit_data(circuit)
        score = compute_weighted_score(metrics)

        if score < best_score:
            best_score = score
            best_perm = perm
            best_metrics = metrics

    # print(f"Ensemble best: CX={best_metrics['cx']}, Depth={best_metrics['depth']}")
    return [best_perm]


# import math
# import random
# import copy


# class MCTSNode:
#     def __init__(self, tableau, parent=None, pivot=None, remaining_rows=None):
#         """A node in the MCTS search tree representing a partial pivot sequence"""
#         self.tableau = tableau  # Current tableau state
#         self.parent = parent  # Parent node
#         self.pivot = pivot  # (row, col) that led to this node
#         self.children = {}  # Map from pivot to child nodes
#         self.visits = 0  # Number of visits
#         self.reward = 0  # Cumulative reward
#         self.remaining_rows = remaining_rows  # Rows still needing pivots
#         self.valid_pivots = None  # Cache of valid pivots

#     def is_fully_expanded(self):
#         """Check if all possible pivot choices have been tried"""
#         return len(self.get_valid_pivots()) == 0 or len(self.children) == len(
#             self.get_valid_pivots()
#         )

#     def is_terminal(self):
#         """Check if this is a terminal node (no more rows to eliminate)"""
#         if self.remaining_rows is None:
#             self.remaining_rows = self._get_remaining_rows()
#         return len(self.remaining_rows) == 0

#     def _get_remaining_rows(self):
#         """Get rows that still need pivots"""
#         # Simple implementation: if a node doesn't have remaining rows explicitly set,
#         # use parent's remaining rows minus the current pivot row
#         if self.parent is None:
#             # Root node - all rows need pivots
#             return list(range(self.tableau.n_qubits))
#         elif self.pivot is None:
#             # No pivot applied - same rows as parent
#             return (
#                 self.parent.remaining_rows.copy() if self.parent.remaining_rows else []
#             )
#         else:
#             # Remove current pivot row from parent's remaining rows
#             return [r for r in self.parent.remaining_rows if r != self.pivot[0]]

#     def get_valid_pivots(self):
#         """Get all valid (row,col) pivots from this tableau state"""
#         if self.valid_pivots is None:
#             self.valid_pivots = []

#             if self.remaining_rows is None:
#                 self.remaining_rows = self._get_remaining_rows()

#             # For each remaining row, find all columns with 1 in tableau
#             for row in self.remaining_rows:
#                 for col in range(
#                     self.tableau.n_qubits
#                 ):  # Only check first n_qubits columns
#                     if self.tableau._x_out(row, col) == 1:
#                         self.valid_pivots.append((row, col))

#         # Filter out pivots that already have children
#         return [p for p in self.valid_pivots if p not in self.children]

#     def select_child(self, exploration_weight=1.0):
#         """Select child using UCB1 formula"""
#         if not self.children:
#             return None

#         # UCB1 formula: exploitation + exploration
#         log_visits = math.log(self.visits + 1e-10)

#         def ucb_score(child):
#             exploitation = child.reward / (child.visits + 1e-10)
#             exploration = exploration_weight * math.sqrt(
#                 log_visits / (child.visits + 1e-10)
#             )
#             return exploitation + exploration

#         return max(self.children.values(), key=ucb_score)

#     def apply_pivot_to_tableau(tableau, pivot):
#         """Apply the pivot operation to transform the tableau

#         This simulates the row operation in Gaussian elimination
#         that happens when a pivot (row, col) is selected
#         """
#         row, col = pivot
#         n_qubits = tableau.n_qubits

#         # Get the set of rows that need to be modified
#         # (rows that have a 1 in the pivot column)
#         rows_to_modify = []
#         for r in range(2 * n_qubits):
#             if r != row and tableau.tableau[r, col] == 1:
#                 rows_to_modify.append(r)

#         # Apply the row operations (XOR the pivot row with each affected row)
#         for r in rows_to_modify:
#             # Update tableau bits
#             for c in range(2 * n_qubits):
#                 if tableau.tableau[row, c] == 1:
#                     tableau.tableau[r, c] = tableau.tableau[r, c] ^ 1

#             # Update signs if necessary
#             if tableau.tableau[row, col] == 1 and tableau.tableau[r, col] == 1:
#                 tableau.signs[r] = tableau.signs[r] ^ tableau.signs[row]

#     def add_child(self, pivot, tableau, remaining_rows):
#         """Add a child node with the given pivot and transformed tableau"""
#         # Create deep copy of the tableau
#         import copy as py_copy

#         new_tableau = py_copy.deepcopy(tableau)

#         # Apply the pivot transformation to the new tableau
#         MCTSNode.apply_pivot_to_tableau(new_tableau, pivot)

#         # Create child with the transformed tableau
#         child = MCTSNode(new_tableau, self, pivot, remaining_rows)
#         self.children[pivot] = child
#         return child

#     def get_path_from_root(self):
#         """Get sequence of pivots from root to this node"""
#         path = []
#         node = self
#         while node.parent is not None:
#             if node.pivot is not None:
#                 path.append(node.pivot)
#             node = node.parent

#         path.reverse()  # Root to leaf order
#         return path


# def predict_permutation_mcts(model, clifford_tableau, device="cpu", n_simulations=100):
#     """MCTS-based permutation search with neural guidance"""
#     device = torch.device(device)
#     model = model.to(device)

#     # Create root node
#     root = MCTSNode(clifford_tableau)
#     topology = Topology.complete(clifford_tableau.n_qubits)

#     # Create a cache for tableau evaluations to avoid repeated synthesis
#     evaluation_cache = {}

#     # Run MCTS simulations
#     for i in range(n_simulations):
#         # Phase 1: Selection - traverse tree using UCB until we reach a non-fully expanded node
#         node = root
#         while node.is_fully_expanded() and not node.is_terminal():
#             next_node = node.select_child(exploration_weight=1.5)
#             if next_node is None:
#                 # No children to select, break the loop
#                 break
#             node = next_node  # Only update if non-None

#         # Phase 2: Expansion - add a new child node if not terminal
#         if not node.is_terminal():
#             # Get neural predictions to guide expansion
#             tableau_tensor = tableau_to_tensor(node.tableau)
#             tableau_tensor = tableau_tensor.unsqueeze(0).to(device)

#             with torch.no_grad():
#                 # Get model predictions for current state
#                 raw_preds, _, cx_preds = model(tableau_tensor)

#                 # Use these predictions to bias pivot selection
#                 valid_pivots = node.get_valid_pivots()
#                 if valid_pivots:
#                     # Extract pivot probabilities from model output
#                     logits = raw_preds[0, 0]  # Use first step prediction

#                     # Apply Sinkhorn normalization
#                     for _ in range(20):
#                         logits = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
#                         logits = logits - torch.logsumexp(logits, dim=-2, keepdim=True)

#                     probs = torch.exp(logits / 0.1)

#                     # Calculate probability scores for valid pivots
#                     pivot_scores = {}
#                     for row, col in valid_pivots:
#                         pivot_scores[(row, col)] = probs[row, col].item()

#                     # Weighted selection based on model probabilities
#                     total_score = sum(pivot_scores.values()) + 1e-10  # Avoid div by 0
#                     rand_val = random.random() * total_score
#                     cumulative = 0
#                     selected_pivot = valid_pivots[0]  # Default in case of issues

#                     for pivot, score in pivot_scores.items():
#                         cumulative += score
#                         if cumulative >= rand_val:
#                             selected_pivot = pivot
#                             break

#                     # Apply the selected pivot to create new tableau state
#                     # new_tableau = node.tableau

#                     # When pivot is applied, the row is eliminated
#                     new_remaining = [
#                         r for r in node.remaining_rows if r != selected_pivot[0]
#                     ]

#                     # Create and add the new child
#                     node = node.add_child(selected_pivot, node.tableau, new_remaining)

#         # Phase 3: Simulation phase
#         # Force complete sequence with exactly n_qubits pivots
#         sequence = node.get_path_from_root()
#         completed_sequence = force_complete_sequence(
#             sequence, node.tableau, clifford_tableau.n_qubits
#         )
#         reward = evaluate_sequence(
#             completed_sequence, clifford_tableau, topology, evaluation_cache
#         )

#         # Phase 4: Backpropagation
#         while node is not None:
#             node.visits += 1
#             node.reward += reward
#             node = node.parent

#     # Get best sequence and ensure it's complete
#     best_sequence = get_complete_sequence(root, clifford_tableau.n_qubits)
#     return [best_sequence]


# def force_complete_sequence(partial_sequence, tableau, n_qubits):
#     """Complete a partial sequence with valid pivots at each step"""
#     # Make a copy and apply partial sequence to track tableau state
#     import copy as py_copy

#     current_tableau = py_copy.deepcopy(tableau)

#     # Apply all pivots in partial sequence to track tableau state
#     for row, col in partial_sequence:
#         MCTSNode.apply_pivot_to_tableau(current_tableau, (row, col))

#     # Find remaining rows
#     handled_rows = set(pivot[0] for pivot in partial_sequence)
#     missing_rows = set(range(n_qubits)) - handled_rows

#     # Build completion with valid pivots
#     completed_sequence = partial_sequence.copy()

#     for row in missing_rows:
#         # Try to find any valid pivot for this row
#         found = False
#         for col in range(n_qubits):  # Try all columns
#             if current_tableau._x_out(row, col) == 1:
#                 completed_sequence.append((row, col))
#                 found = True
#                 # Update tableau state
#                 MCTSNode.apply_pivot_to_tableau(current_tableau, (row, col))
#                 break

#         if not found:  # If no valid pivot, this is an invalid sequence
#             return partial_sequence  # Return the original sequence as a signal

#     return completed_sequence


# def get_complete_sequence(root, n_qubits):
#     """Get the best complete sequence from the MCTS tree"""
#     # First traverse the tree to get the best path (might be incomplete)
#     node = root
#     sequence = []

#     # We want EXACTLY one pivot per qubit (i.e., n_qubits total pivots)
#     handled_rows = set()

#     # First try to extract sequence from tree
#     while node.children and len(handled_rows) < n_qubits:
#         if not node.children:
#             break

#         # Select best child
#         best_child = max(
#             node.children.values(), key=lambda n: n.reward / (n.visits + 1e-10)
#         )
#         if best_child.pivot:
#             row, col = best_child.pivot
#             sequence.append((row, col))
#             handled_rows.add(row)
#         node = best_child

#     # Now ensure we have exactly one pivot for each row
#     missing_rows = set(range(n_qubits)) - handled_rows
#     duplicate_check = {}

#     # Clean up any duplicates from the sequence
#     clean_sequence = []
#     used_rows = set()

#     for row, col in sequence:
#         if row not in used_rows:
#             clean_sequence.append((row, col))
#             used_rows.add(row)

#     # Add missing rows with valid pivots - important to use the ORIGINAL tableau
#     # since we need pivots that are valid in the initial state
#     for row in missing_rows:
#         # Try to find a non-diagonal pivot first
#         found = False
#         for col in range(n_qubits):
#             if col != row and root.tableau._x_out(row, col) == 1:
#                 clean_sequence.append((row, col))
#                 found = True
#                 break

#         if not found:  # Fall back to diagonal
#             clean_sequence.append((row, row))

#     # Ensure we have exactly n_qubits pivots
#     assert (
#         len(clean_sequence) == n_qubits
#     ), f"Invalid sequence length: {len(clean_sequence)} vs {n_qubits}"

#     # Verify each row appears exactly once
#     rows = [r for r, _ in clean_sequence]
#     assert len(set(rows)) == n_qubits, f"Rows aren't unique: {rows}"

#     return clean_sequence


# def get_best_sequence(root):
#     """Extract best pivot sequence from the MCTS tree"""
#     if not root.children:
#         return []

#     # Find child with highest average reward
#     best_child = max(
#         root.children.values(), key=lambda n: n.reward / (n.visits + 1e-10)
#     )

#     # Build sequence
#     sequence = [best_child.pivot]

#     # Recursively add best children
#     current = best_child
#     while current.children:
#         if not current.children:
#             break

#         best_child = max(
#             current.children.values(), key=lambda n: n.reward / (n.visits + 1e-10)
#         )
#         if best_child.pivot:
#             sequence.append(best_child.pivot)
#         current = best_child

#     return sequence


# def simulate_to_completion(
#     partial_sequence, tableau, model, device, topology, cache=None
# ):
#     """Complete a partial pivot sequence with neural guidance"""
#     remaining_rows = [
#         r
#         for r in range(tableau.n_qubits)
#         if r not in [pivot[0] for pivot in partial_sequence]
#     ]

#     # Starting with the partial sequence
#     sequence = partial_sequence.copy()

#     # Complete sequence using model guidance
#     while remaining_rows:
#         # Get neural predictions
#         tableau_tensor = tableau_to_tensor(tableau)
#         tableau_tensor = tableau_tensor.unsqueeze(0).to(device)

#         with torch.no_grad():
#             raw_preds, _, cx_preds = model(tableau_tensor)

#             # Get valid pivots
#             valid_pivots = []
#             for row in remaining_rows:
#                 for col in range(tableau.n_qubits):
#                     if tableau._x_out(row, col) == 1:
#                         valid_pivots.append((row, col))

#             if not valid_pivots:  # Should not happen with valid tableaus
#                 break

#             # Sample next pivot using model probabilities
#             logits = raw_preds[0, 0]

#             # Apply Sinkhorn normalization
#             for _ in range(20):
#                 logits = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
#                 logits = logits - torch.logsumexp(logits, dim=-2, keepdim=True)

#             probs = torch.exp(logits / 0.1)

#             # Weighted selection among valid pivots
#             pivot_scores = {}
#             for row, col in valid_pivots:
#                 pivot_scores[(row, col)] = probs[row, col].item()

#             # Select pivot with temperature-based sampling
#             temperature = 0.5  # Adjust for exploration/exploitation balance
#             pivot_weights = {
#                 p: math.exp(s / temperature) for p, s in pivot_scores.items()
#             }
#             total_weight = sum(pivot_weights.values()) + 1e-10

#             rand_val = random.random() * total_weight
#             cumulative = 0
#             next_pivot = valid_pivots[0]  # Default fallback

#             for pivot, weight in pivot_weights.items():
#                 cumulative += weight
#                 if cumulative >= rand_val:
#                     next_pivot = pivot
#                     break

#             # Update sequence and state
#             sequence.append(next_pivot)
#             remaining_rows = [r for r in remaining_rows if r != next_pivot[0]]

#     # Evaluate completed sequence
#     return evaluate_sequence(sequence, tableau, topology, cache)


# def evaluate_sequence(sequence, tableau, topology, cache=None):
#     """Evaluate a pivot sequence by synthesizing the circuit"""
#     if cache is not None:
#         # Check cache first
#         key = tuple(sequence)
#         if key in cache:
#             return cache[key]

#     pred_iter = iter(sequence)

#     def pred_callback(G, remaining, remaining_rows, choice_fn=min):
#         try:
#             row, col = next(pred_iter)
#             if isinstance(row, torch.Tensor):
#                 row = row.item()
#             if isinstance(col, torch.Tensor):
#                 col = col.item()

#             # Verify pivot is valid
#             if (
#                 hasattr(remaining, "_x_out")
#                 and remaining._x_out(int(row), int(col)) == 1
#             ):
#                 return int(row), int(col)
#             else:
#                 raise StopIteration

#         except StopIteration:
#             # Safe fallback for valid pivot
#             for r in remaining_rows:
#                 for c in range(
#                     remaining.n_qubits
#                 ):  # Only first n_qubits to avoid index error
#                     if remaining._x_out(r, c) == 1:
#                         return int(r), int(c)

#             # Last resort
#             if remaining_rows:
#                 row = min(remaining_rows)
#                 return int(row), int(row)
#             else:
#                 return 0, 0

#     try:
#         circuit = synthesize_tableau_perm_row_col(
#             tableau, topology, pick_pivot_callback=pred_callback
#         )
#         metrics = collect_circuit_data(circuit)

#         # Calculate reward (focus heavily on CX count)
#         # Using negative score as reward (since MCTS maximizes reward)
#         score = compute_weighted_score(metrics)
#         reward = 100.0 / (
#             score + 1.0
#         )  # Transform to positive reward that increases with quality

#         # Cache the result
#         if cache is not None:
#             cache[tuple(sequence)] = reward

#         return reward

#     except Exception as e:
#         # Handle synthesis failures gracefully
#         return 0.0  # Worst possible reward


# def evaluate_permutation(perm, tableau, topology):
#     """Evaluate a permutation, returning both score and metrics"""
#     pred_iter = iter(perm)

#     def pred_callback(G, remaining, remaining_rows, choice_fn=min):
#         try:
#             row, col = next(pred_iter)
#             if isinstance(row, torch.Tensor):
#                 row = row.item()
#             if isinstance(col, torch.Tensor):
#                 col = col.item()

#             # Check if valid pivot
#             if (
#                 hasattr(remaining, "_x_out")
#                 and remaining._x_out(int(row), int(col)) == 1
#             ):
#                 return int(row), int(col)
#             else:
#                 raise StopIteration
#         except StopIteration:
#             # Safe fallback
#             for r in remaining_rows:
#                 for c in range(remaining.n_qubits):
#                     if remaining._x_out(r, c) == 1:
#                         return int(r), int(c)

#             if remaining_rows:
#                 row = min(remaining_rows)
#                 return int(row), int(row)
#             else:
#                 return 0, 0

#     try:
#         circuit = synthesize_tableau_perm_row_col(
#             tableau, topology, pick_pivot_callback=pred_callback
#         )
#         metrics = collect_circuit_data(circuit)
#         score = compute_weighted_score(metrics)
#         return score, metrics
#     except Exception as e:
#         # Handle synthesis failures
#         return float("inf"), {"cx": float("inf"), "depth": float("inf")}


import pickle
import tempfile
import os


def curriculum_train(data_file, max_samples, epochs_per_stage, device):
    """Curriculum training with staged data"""
    # Load and preprocess your data
    all_data = TableauPermutationDataset(
        data_file=data_file, n_qubits=None, max_samples=max_samples
    )
    # all_data should be a list of (tableau, best_perms, cx_count)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Check for CUDA first, then MPS, then fall back to CPU
    # if torch.cuda.is_available():
    #     device = torch.device("cuda")
    # elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    #     device = torch.device("mps")
    # else:
    #     device = torch.device("cpu")
    model = OrderedPermutationTransformer(
        n_qubits=all_data.n_qubits, dim=256, num_layers=6
    )
    all_data = all_data.data
    # all_data: list of (tableau, best_perms, cx_count)
    all_data = sorted(all_data, key=lambda x: x[1])  # x[2] = cx_count

    n = len(all_data)
    stages = [
        all_data[: n // 4],
        all_data[n // 4 : n // 2],
        all_data[n // 2 : 3 * n // 4],
        all_data[3 * n // 4 :],
    ]

    for stage_idx, stage_data in enumerate(stages):
        print(
            f"Training on curriculum stage {stage_idx+1} with {len(stage_data)} samples"
        )
        # Write stage_data to a temporary pickle file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as tmp:
            pickle.dump(stage_data, tmp)
            tmp_filename = tmp.name

        dataset = TableauPermutationDataset(tmp_filename)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=32, shuffle=True, collate_fn=custom_collate_fn
        )
        model = train(model, dataloader, epochs_per_stage, device)

        # Clean up the temporary file
        os.remove(tmp_filename)
    return model


# # Example 1 of usage (without fine-tuning):
# device = get_default_device()
# print(f"Using device: {device}")
# device = "cpu"
# n_qubit = 5
# model = pretrain_a_model_from_file("training_data_perm_5_qubit.pkl", 320, 10, device)
# # model = curriculum_train(
# #     data_file="training_data_perm.pkl", max_samples=320, epochs_per_stage=5
# # )
# circuit = random_hscx_circuit(nr_qubits=n_qubit, nr_gates=1000)
# tableau = tableau_from_circuit(CliffordTableau(n_qubit), circuit)
# permutations = predict_permutation_gumbel(model, tableau, device)
# # permutations = predict_permutation_beam(model, tableau)
# # permutations = entropy_guided_search(model, tableau, device)
# # permutations = ensemble_predict_permutation(model, tableau, device)
# # permutations = predict_permutation_mcts(model, tableau)
# print(permutations)

# # Example 2 of usage (with SL fine-tuning):
# model = pretrain_a_model_from_file("training_data_perm.pkl", max_samples=320, epochs=20)
# # Try SL fine-tuning
# sl_dataset = TableauPermutationDataset(
#     data_file="training_data_perm_4_qubit.pkl", n_qubits=4, max_samples=320
# )
# sl_model = supervised_cx_fine_tune(
#     model, sl_dataset, epochs=5, device="cuda" if torch.cuda.is_available() else "cpu"
# )
# circuit = random_hscx_circuit(nr_qubits=4, nr_gates=1000)
# tableau = tableau_from_circuit(CliffordTableau(4), circuit)
# permutations = predict_permutation_gumbel(sl_model, tableau)
# print(permutations)
