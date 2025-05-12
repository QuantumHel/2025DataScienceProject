import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from src.utils import tableau_from_circuit, random_hscx_circuit
from pauliopt.clifford.tableau import CliffordTableau
from pauliopt.topologies import Topology
from pauliopt.clifford.tableau_synthesis import synthesize_tableau_perm_row_col
from pauliopt.circuits import Circuit
from scipy.optimize import linear_sum_assignment
import numpy as np
import warnings
import pickle
import tempfile
import os
import matplotlib.pyplot as plt


# Suppress all overflow warnings globally
np.seterr(over="ignore")

# Suppress FutureWarning
warnings.simplefilter(action="ignore", category=FutureWarning)


# This current code file is slower running on GPU, as it needs to move data between CPU and GPU frequently.
def get_default_device():
    """Return the best available device (CUDA → MPS → CPU)"""

    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


# # Try a Block-Aware Residual Encoder (Expected to be better for Tableau Structure, but empirically not)
# class ResidualBlockAwareEncoder(nn.Module):
#     def __init__(self, n_qubits):
#         super().__init__()
#         self.n_qubits = n_qubits

#         # Extract features from X/Z blocks separately
#         self.x_block_encoder = nn.Sequential(
#             nn.Conv2d(2, 32, 3, padding=1),
#             nn.GELU(),
#             nn.Conv2d(32, 48, 3, padding=1),
#         )

#         self.z_block_encoder = nn.Sequential(
#             nn.Conv2d(2, 32, 3, padding=1),
#             nn.GELU(),
#             nn.Conv2d(32, 48, 3, padding=1),
#         )

#         # Combine X/Z information
#         self.combiner = nn.Conv2d(96, 64, 1)

#         # Final processing
#         self.final = nn.Sequential(
#             nn.GELU(),
#             nn.Conv2d(64, 64, 3, padding=1),
#             nn.GELU(),
#             nn.AdaptiveAvgPool2d((4, 4)),
#         )

#     def forward(self, x):
#         # Split X and Z blocks
#         n_qubits = self.n_qubits
#         x_part = x[:, :, :n_qubits, :]
#         z_part = x[:, :, n_qubits:, :]

#         # Process blocks separately
#         x_features = self.x_block_encoder(x_part)
#         z_features = self.z_block_encoder(z_part)

#         # Combine features
#         combined = torch.cat([x_features, z_features], dim=1)
#         combined = self.combiner(combined)

#         # Final processing
#         output = self.final(combined)
#         return output.flatten(start_dim=1)


# # Try a tableau-aware encoder, also empirically not as good as expected
# class TableauStructureEncoder(nn.Module):
#     def __init__(self, n_qubits):
#         super().__init__()
#         self.n_qubits = n_qubits

#         # Single pathway with structural awareness
#         self.encoder = nn.Sequential(
#             # Initial feature extraction
#             nn.Conv2d(2, 32, 3, padding=1),
#             nn.GELU(),
#             nn.BatchNorm2d(32),
#             # Capture larger patterns
#             nn.Conv2d(32, 64, 5, padding=2),
#             nn.GELU(),
#             nn.BatchNorm2d(64),
#             # Global context
#             nn.Conv2d(64, 128, 3, padding=1),
#             nn.GELU(),
#             nn.BatchNorm2d(128),
#         )

#         # Position-aware readout
#         self.pool = nn.AdaptiveAvgPool2d((4, 4))

#         # Final projection
#         self.project = nn.Linear(128 * 4 * 4, 256)

#     def forward(self, x):
#         # Process the whole tableau together
#         features = self.encoder(x)
#         pooled = self.pool(features)
#         return self.project(pooled.flatten(1))


# # Empirically more time-consuming and not as good as expected
# class TransformerTableauEncoder(nn.Module):
#     def __init__(self, n_qubits, dim=256, num_layers=4, num_heads=8):
#         super().__init__()
#         self.n_qubits = n_qubits

#         # Embedding for tableau entries
#         self.embedding = nn.Linear(2, dim)  # 2 channels to dimension

#         # 2D positional encoding
#         self.row_pos = nn.Parameter(torch.randn(2 * n_qubits, dim // 2))
#         self.col_pos = nn.Parameter(torch.randn(2 * n_qubits, dim // 2))

#         # Transformer encoder
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=dim,
#             nhead=num_heads,
#             dim_feedforward=dim * 4,
#             activation=F.gelu,
#             batch_first=True,
#         )
#         self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

#         # Output projection
#         self.output_proj = nn.Linear(dim * (2 * n_qubits) ** 2, dim)

#     def forward(self, x):
#         B = x.size(0)
#         n = 2 * self.n_qubits

#         # Reshape input to [batch, n*n, 2]
#         x = x.permute(0, 2, 3, 1).reshape(B, n * n, 2)

#         # Embed features
#         x = self.embedding(x)  # [batch, n*n, dim]

#         # Add 2D positional encoding
#         pos_indices = torch.arange(n, device=x.device)
#         row_idx = pos_indices.repeat_interleave(n).view(n, n)
#         col_idx = pos_indices.repeat(n, 1)

#         row_emb = self.row_pos[row_idx.flatten()]  # [n*n, dim//2]
#         col_emb = self.col_pos[col_idx.flatten()]  # [n*n, dim//2]
#         pos_emb = torch.cat([row_emb, col_emb], dim=-1)  # [n*n, dim]

#         # Add positional embeddings
#         x = x + pos_emb.unsqueeze(0)  # [batch, n*n, dim]

#         # Pass through transformer
#         x = self.transformer(x)  # [batch, n*n, dim]

#         # Global pooling with attention
#         x = x.flatten(1)  # [batch, n*n*dim]
#         x = self.output_proj(x)  # [batch, dim]

#         return x


class SelfAttentionBlock(nn.Module):
    """Self-attention block for feature extraction."""

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
    """
    Permutation CX-aware decoder for sequence generation.
    This decoder uses a transformer architecture with an additional CX-aware attention mechanism.
    """

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
    """Residual block for CX predictions."""

    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.LayerNorm(dim))

    def forward(self, x):
        return x + self.net(x)  # Residual connection


class OrderedPermutationTransformer(nn.Module):
    """The model for ordered permutation prediction. Experiments with different architectures were in the comments."""

    def __init__(self, n_qubits, dim=256, num_layers=4, num_heads=8, dropout=0.1):
        super().__init__()
        self.n_qubits = n_qubits
        self.dim = dim

        # # Simple CNN encoder, already worked very well
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

        # Enhanced CNN encoder with self-attention
        self.tableau_encoder = nn.Sequential(
            nn.Conv2d(2, 32, 3, padding=1),
            nn.GELU(),
            nn.BatchNorm2d(32),
            nn.Dropout2d(dropout / 2),
            SelfAttentionBlock(32),  # Add self-attention between conv layers
            nn.Conv2d(32, 64, 3, padding=1),
            nn.GELU(),
            nn.BatchNorm2d(64),
            nn.Dropout2d(dropout / 2),
            SelfAttentionBlock(64),  # Add self-attention between conv layers
            nn.Conv2d(64, 64, 3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )

        # Linear layer to match the enhanced CNN encoder output to dim
        self.linear = nn.Linear(64 * 4 * 4, dim)

        # # Simple sequence Decoder, already worked very well
        # self.decoder = nn.TransformerDecoder(
        #     nn.TransformerDecoderLayer(
        #         d_model=dim, nhead=4, dim_feedforward=4 * dim, activation=F.gelu
        #     ),
        #     num_layers=num_layers,
        # )

        # Enhanced sequence Decoder with CX-awareness, empirically not as good as expected
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

        # # Simple CX head, already worked very well
        # self.cx_head = nn.Sequential(
        #     nn.Linear(dim, dim // 2),
        #     nn.GELU(),
        #     nn.Dropout(0.1),
        #     nn.Linear(dim // 2, 1),
        #     nn.ReLU(),  # Ensure non-negative predictions
        # )

        # # Try sophisticated CX head, directly connected to permutation prediction
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

        # # Try more sophisticated CX head to accept the additional cx_aware features
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

        # Further enhanced more sophisticated CX-head with residual connections
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

        # Enhanced initialization, empirically no improvement
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.kaiming_normal_(p, mode="fan_out", nonlinearity="relu")

    def forward(self, tableau):
        """
        Forward pass through the model.
        Args:
            tableau: Input tableau tensor of shape [batch_size, 2, n_qubits, n_qubits]
        Returns:
            predictions: Predicted permutations of shape [batch_size, seq_len, n_qubits, n_qubits]
            stop_logits: Logits for stop signal of shape [batch_size, seq_len]
            cx_predictions: Predicted CX values of shape [batch_size, seq_len]
        """
        # Encode Tableau
        B = tableau.size(0)
        x = self.tableau_encoder(tableau)
        x = x.view(B, -1)
        x = self.linear(x)  # Ensure the dimension matches

        # Generate Sequence
        memory = x.unsqueeze(0)  # [1, batch_size, dim]
        output = torch.zeros(B, self.dim, device=tableau.device)  # [batch_size, dim]

        predictions = []
        stop_logits = []
        cx_predicitons = []  # For CX-prediction

        for t in range(10):  # Max steps. When set to n_qubits, stop head is not needed
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
            perm_features = pred.view(B, self.n_qubits, self.n_qubits).flatten(1)

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

        # Stack cx predictions -> [batch_size, seq_len]
        cx_predicitons = torch.cat(cx_predicitons, dim=1)

        return predictions, stop_logits, cx_predicitons


class OrderedPermutationLoss(nn.Module):
    """Loss function for ordered permutation prediction."""

    def __init__(self, alpha=0.5, beta=0.1, gamma=2.0):
        super().__init__()
        self.alpha = alpha  # Stop signal loss weight
        self.beta = beta  # Length regularization weight
        self.gamma = gamma  # CX prediction loss weight, gamma=8.0 sometimes result in stuck at loss 270.0

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
        # 1. Computing permutation loss for each valid target and use the minimum
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

                loss = (
                    F.mse_loss(preds[i], valid_seq_padded, reduction="none").mean(
                        dim=(-1, -2)
                    )
                    * masks[i]
                )
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

        return (
            perm_loss
            + self.alpha * stop_loss
            + self.beta * length_loss
            + self.gamma * cx_loss
        )


def train(model, dataloader, epochs, device):
    """
    Train the model with the given dataloader and number of epochs.
    Args:
        model: The model to train.
        dataloader: The DataLoader for the training data.
        epochs: Number of epochs to train.
        device: Device to use for training (CPU or GPU).
    """
    # Initialization
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=5e-4, total_steps=epochs * len(dataloader), pct_start=0.3
    )

    # Loss function
    criterion = OrderedPermutationLoss(gamma=5.0)  # Adjust gamma as needed

    # Gradient accumulation (for larger batches)
    accum_steps = 4

    loss_history = []  # To track loss per epoch

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        batches = 0

        optimizer.zero_grad()

        for batch_idx, (tableaus, raw_targets) in enumerate(dataloader):
            # 1. Prepare batch -------------------------------------------------
            # Convert raw targets to padded tensor
            targets, masks = pad_targets_all_seq(
                raw_targets, device
            )  # returning all sequences

            # Move data to device
            tableaus = tableaus.to(device)

            # Extract cx targets from `raw_targets`
            cx_targets = [item[1] for item in raw_targets]

            # 2. Forward pass -------------------------------------------------
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
                sinkhorn_preds.append(torch.stack(seq_preds))
            preds = torch.stack(sinkhorn_preds)  # [batch_size, seq_len, n, n]

            # 4. Loss calculation ----------------------------------------------
            # Convert cx_targets from list to properly shaped tensor
            cx_targets_tensor = torch.tensor(cx_targets, device=device).float()

            # Reshape to match cx_preds dimensions [batch_size, seq_len]
            cx_targets_tensor = cx_targets_tensor.unsqueeze(1).expand(
                -1, cx_preds.size(1)
            )

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

            # 7. Logging ------------------------------------------------------
            total_loss += loss.item()
            batches += 1

            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1} | Batch {batch_idx} | Loss: {loss.item():.4f}")

        # 8. Epoch summary -----------------------------------------------------
        avg_loss = total_loss / batches
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
    """
    Handle the correct nested structure of permutation sequences.
    Args:
        raw_targets: list of tuples (tableau, perm_sequences)
        device: device to move tensors to (CPU or GPU)
    Returns:
        padded: [batch_size, max_len, n_qubits, n_qubits] tensor
        mask: [batch_size, max_len] tensor (mask for the longest sequence in the batch)
    """
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
    Handle the correct nested structure of permutation sequences.
    Args:
        raw_targets: list of tuples (tableau, perm_sequences)
        device: device to move tensors to (CPU or GPU)
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
    """
    Custom dataset for loading tableau permutation data.
    Usage:
        # For testing with a small random subset (320 samples)
        dataset = TableauPermutationDataset(
            data_file="training_data_perm.pkl",
            max_samples=320,
            shuffle=True  # Random subset
        )
        # For using the first 320 samples (deterministic)
        dataset = TableauPermutationDataset(
            data_file="training_data_perm.pkl",
            max_samples=320,
            shuffle=False  # First N samples
        )
        # For final training with all data
        dataset = TableauPermutationDataset(
            data_file="training_data_perm.pkl"  # No max_samples means use all data
        )
    """

    def __init__(self, data_file, max_samples=None, shuffle=True):
        print(f"Loading data from {data_file}...")
        with open(data_file, "rb") as f:
            all_data = pickle.load(f)

        # Detect n_qubits from the first tableau in the dataset
        self.n_qubits = all_data[0][0].n_qubits
        print(f"Auto-detected {self.n_qubits} qubits from data")

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
        """
        Convert a Clifford tableau to a tensor representation.
        Args:
            tableau: CliffordTableau object.
        Returns:
            Tensor representation of the tableau.
        """
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
        """
        Get item from dataset.
        Args:
            idx: Index of the item.
        Returns:
            tableau: Tensor representation of the tableau.
            best_perms: List of best permutation sequences and their CX counts.
        """
        tableau, best_perms = self.data[idx]
        n_qubits = tableau.n_qubits

        # Get CX count from first permutation (all should be same)
        cx_count = best_perms[0][1] if best_perms else float("inf")

        # Create a list to hold all permutation sequences
        all_perm_sequences = []

        # Process each permutation sequence in best_perms
        for sequence in best_perms:
            # Each `sequence` is a tuple [a list of permutation tuples, cx_count]

            # For each sequence, create a list of matrices
            perm_matrices = []
            current_perm = torch.eye(n_qubits)  # Initialize with identity

            for step in sequence:
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


def custom_collate_fn(batch):
    """
    Custom collate function to handle variable-length permutation lists.
    Args:
        batch: List of tuples (tableau, best_perms).
    Returns:
        tableaus: Stacked tensor of tableaus.
        targets: List of best permutation sequences and their CX counts.
    """
    # Extract tableaus and targets
    tableaus = [item[0] for item in batch]
    targets = [item[1] for item in batch]

    # Stack tableaus (they should all have the same shape)
    tableaus = torch.stack(tableaus)

    # Don't try to stack targets, just return them as a list
    return tableaus, targets


def pretrain_a_model_from_file(data_file, max_samples, epochs, device):
    """
    A wrapper function to pretrain a model from a file with the given parameters.
    Args:
        data_file: Path to the data file.
        max_samples: Maximum number of samples to use.
        epochs: Number of epochs to train.
        device: Device to use for training (CPU or GPU).
    Returns:
        model: The trained model.
    """
    dataset = TableauPermutationDataset(data_file, max_samples=max_samples)
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

    train(model, dataloader, epochs, device)  # 100 epochs too much, plateaus before 50;
    return model


def convert_tensor_to_tableau(tableau_tensor):
    """
    Convert tensor representation back to CliffordTableau object.
    Args:
        tableau_tensor: Tensor representation of the tableau.
    Returns:
        tableau: CliffordTableau object.
    """
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


# Ready to use, but no big improvement as expected! So, did not used mostly.
def supervised_cx_fine_tune(model, dataset, epochs, device):
    """
    Fine-tune with supervised learning focusing on CX reduction.
    Args:
        model: The model to fine-tune.
        dataset: The dataset for fine-tuning.
        epochs: Number of epochs to fine-tune.
        device: Device to use for training (CPU or GPU).
    """
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

                            # Try smarter fallback, empirically no big improvement as expected
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


def tableau_to_tensor(tableau):
    """
    Convert a Clifford tableau to a tensor representation.
    Args:
        tableau: CliffordTableau object.
    Returns:
        Tensor representation of the tableau.
    """
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


def collect_circuit_data(circuit: Circuit) -> dict:
    """
    Collects data from a circuit object.
    Args:
        circuit: A Circuit object.
    Returns:
        A dictionary with the number of qubits, H gates, S gates, CX gates, and circuit depth.
    """
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
    Args:
        pred_metrics: A dictionary containing the predicted metrics.
    Returns:
        A weighted score based on the CX count and circuit depth.
    """
    # Define weights for each metric
    cx_weight = 10.0  # Try adjusting from 10 to 50; 50 not better
    depth_weight = 1.0  # 1.0 seems better than 0.1
    # Compute weighted score, only keep cx seems worse
    score = cx_weight * pred_metrics["cx"] + depth_weight * pred_metrics["depth"]
    return score


# Try to adapt weights based on how close we are to optimum, but not much better
def compute_adaptive_score(pred_metrics, baseline_metrics):
    """
    Adapt weights based on how close we are to optimum.
    Args:
        pred_metrics: A dictionary containing the predicted metrics.
        baseline_metrics: A dictionary containing the baseline metrics.
    Returns:
        A weighted score based on the CX count and circuit depth.
    """
    cx_ratio = pred_metrics["cx"] / baseline_metrics["cx"]
    depth_ratio = pred_metrics["depth"] / baseline_metrics["depth"]

    # If CX is much worse than depth, focus more on CX
    if cx_ratio > depth_ratio * 1.2:
        cx_weight = 20.0
        depth_weight = 0.5
    else:
        cx_weight = 10.0
        depth_weight = 1.0

    return cx_weight * pred_metrics["cx"] + depth_weight * pred_metrics["depth"]


def gumbel_sinkhorn(logits, temp=0.1, n_samples=5, n_iters=20):
    """
    Generate multiple permutation samples using Gumbel-Sinkhorn normalization.
    Args:
        logits: Tensor of logits to sample from.
        temp: Temperature for Gumbel noise.
        n_samples: Number of samples to generate.
    Returns:
        samples: List of sampled permutation matrices.
    """
    samples = []
    for _ in range(n_samples):
        # Generate fresh Gumbel noise for each sample
        gumbels = -torch.log(-torch.log(torch.rand_like(logits) + 1e-10) + 1e-10)
        noisy_logits = (logits + gumbels) / temp

        # Apply Sinkhorn normalization
        s = noisy_logits.clone()  # Important: clone to avoid in-place modification
        for _ in range(n_iters):
            s = s - torch.logsumexp(s, dim=-1, keepdim=True)
            s = s - torch.logsumexp(s, dim=-2, keepdim=True)
        # samples.append(torch.exp(s))
        # Try adding final softmax instead of exp() for better numerical stability
        samples.append(F.softmax(s, dim=-1))

    # Return all samples (not average - keep the diversity)
    return samples


def predict_permutation_gumbel(model, clifford_tableau, device):
    """Returns the best permutation after evaluating multiple candidates.
    Args:
        model: The trained model.
        clifford_tableau: The input tableau to predict the permutation for.
        device: Device to use for prediction (CPU or GPU).
    Returns:
        best_perm: The best permutation found.
    """
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

        # # Sort steps by cx count. Deterministic order
        # step_indices = torch.argsort(cx_preds[0], descending=True).tolist()

        # Try a probabilistic approach. Weight step selection by predicted CX reduction potential
        # Higher weight for lower CX predictions
        cx_weights = torch.exp(-cx_preds[0] * 2)
        step_probs = F.softmax(cx_weights, dim=0)
        step_indices = torch.multinomial(
            step_probs, min(8, raw_preds.shape[1]), replacement=False
        )

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

                    # Try smarter fallback, empirically no big improvement as expected
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
                # score = pred_metrics["cx"] + pred_metrics["depth"] # Works fine already, weights = [1,1]
                score = compute_weighted_score(pred_metrics)
                # score = compute_adaptive_score(pred_metrics, collect_circuit_data(circuit))

                # Keep track of best permutation
                if score < best_score:
                    best_score = score
                    best_perm = perm

    return [best_perm]  # Keep list format for compatibility


def entropy_guided_search(model, clifford_tableau, device, n_samples=5):
    """Focus search on areas where model is uncertain to find better permutations.
    Args:
        model: The trained model.
        clifford_tableau: The input tableau to predict the permutation for.
        device: Device to use for prediction (CPU or GPU).
        n_samples: Number of samples to generate for each step.
    Returns:
        best_perm: The best permutation found.
    """
    device = torch.device(device)
    model = model.to(device)

    tableau_tensor = tableau_to_tensor(clifford_tableau)
    tableau_tensor = tableau_tensor.unsqueeze(0).to(device)
    topology = Topology.complete(clifford_tableau.n_qubits)

    with torch.no_grad():
        # Forward pass to get raw predictions
        raw_preds, _, _ = model(tableau_tensor)

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

        # Store best permutation and score
        best_perm = None
        best_score = float("inf")

        # Explore uncertain steps more thoroughly
        for step_idx, entropy in uncertain_steps[:3]:  # Focus on top-3 most uncertain
            # Use adaptive temperature range - more temps for higher entropy
            temps = [0.01, 0.03, 0.05, 0.1, 0.2, 0.5]
            if entropy > 3.0:  # Very uncertain areas
                temps.extend([0.7, 1.0])  # Add higher temps for exploration

            # Sample more permutations for high-entropy areas
            samples_for_step = n_samples + int(3 * entropy)

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

                        # Try smarter fallback, empirically no big improvement as expected
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

        # In case nothing improved, fall back to basic prediction
        if best_perm is None:
            # Use basic Gumbel-Sinkhorn with default temp
            perm_samples = gumbel_sinkhorn(raw_preds[0, 0], temp=0.1, n_samples=1)
            perm_matrix = perm_samples[0].cpu().numpy()
            row_ind, col_ind = linear_sum_assignment(-perm_matrix)
            best_perm = [(int(i), int(j)) for i, j in zip(row_ind, col_ind)]

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
            # except StopIteration:
            #     row = choice_fn(remaining_rows)
            #     return row, row

            # Try smarter fallback, empirically no big improvement as expected
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

    return [best_perm]


def curriculum_train(data_file, max_samples, epochs_per_stage, device):
    """Curriculum training with staged data"""
    # Load and preprocess your data
    # `all_data` should be a list of (tableau, best_perms, cx_count)
    all_data = TableauPermutationDataset(data_file=data_file, max_samples=max_samples)
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


def main():
    """
    Main function to run the example usage of the model.
    """
    # Example usage without fine-tuning
    device = "cpu"
    n_qubit = 4
    model = pretrain_a_model_from_file("training_data_perm.pkl", 320, 5, device)
    # # Try curriculum training
    # model = curriculum_train(
    #     data_file="training_data_perm.pkl", max_samples=320, epochs_per_stage=5
    # )
    circuit = random_hscx_circuit(nr_qubits=n_qubit, nr_gates=1000)
    tableau = tableau_from_circuit(CliffordTableau(n_qubit), circuit)
    # permutations = predict_permutation_gumbel(model, tableau, device) # Alternative for prediction
    permutations = entropy_guided_search(
        model, tableau, device
    )  # Alternative for prediction
    # permutations = ensemble_predict_permutation(model, tableau, device) # Ensemble method for prediction
    print(permutations)

    # # Example usage with SL fine-tuning
    # model = pretrain_a_model_from_file("training_data_perm.pkl", 320, 5, device)
    # # Try SL fine-tuning
    # sl_dataset = TableauPermutationDataset(
    #     data_file="training_data_perm_4_qubit.pkl", max_samples=320
    # )
    # sl_model = supervised_cx_fine_tune(model, sl_dataset, 5, device)
    # circuit = random_hscx_circuit(nr_qubits=4, nr_gates=1000)
    # tableau = tableau_from_circuit(CliffordTableau(4), circuit)
    # permutations = predict_permutation_gumbel(sl_model, tableau)
    # print(permutations)

    return 0


if __name__ == "__main__":
    main()
