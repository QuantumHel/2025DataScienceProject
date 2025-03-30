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


# Try a Block-Aware Residual Encoder (Expect to be better for Tableau Structure)
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


# Try a tableau-aware encoder
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


class OrderedPermutationTransformer(nn.Module):
    # def __init__(self, n_qubits, dim=128, num_layers=4):
    def __init__(self, n_qubits, dim=256, num_layers=4, num_heads=8):
        super().__init__()
        self.n_qubits = n_qubits
        self.dim = dim

        # Tableau Encoder
        # self.tableau_encoder = nn.Sequential(
        #     nn.Conv2d(2, 16, 3, padding=1),
        #     nn.GELU(),
        #     nn.Conv2d(16, 32, 3, padding=1),
        #     nn.AdaptiveAvgPool2d((4, 4)),
        # )

        # Try a Transformer-based encoder
        self.tableau_encoder = TransformerTableauEncoder(
            n_qubits=n_qubits, dim=dim, num_layers=num_layers, num_heads=num_heads
        )

        # try a Residual Block-Aware Encoder
        # Not as good as expected
        # self.tableau_encoder = ResidualBlockAwareEncoder(n_qubits)
        # self.tableau_encoder = TableauStructureEncoder(n_qubits)

        # Linear layer to match the dimension
        # self.linear = nn.Linear(32 * 4 * 4, dim)
        # No need for linear projection layer (the line above) - encoder already outputs dim-dimensional vectors
        # As the TransformerTableauEncoder handles this internally

        # try a Residual Block-Aware Encoder or Tableau Structure Encoder
        # self.linear = nn.Linear(64 * 4 * 4, dim)
        # self.linear = nn.Linear(256, dim)

        # Sequence Decoder
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=dim, nhead=4, dim_feedforward=4 * dim, activation=F.gelu
            ),
            num_layers=num_layers,
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

    def forward(self, tableau):
        # Encode Tableau
        B = tableau.size(0)
        x = self.tableau_encoder(tableau)
        # The two lines below are not needed for the Transformer-based encoder
        # x = x.view(B, -1)
        # x = self.linear(x)  # Ensure the dimension matches

        # Generate Sequence
        memory = x.unsqueeze(0)  # [1, batch_size, dim]
        output = torch.zeros(B, self.dim, device=tableau.device)  # [batch_size, dim]

        predictions = []
        stop_logits = []

        for t in range(10):  # Max steps
            # Transformer Decoder
            output = self.decoder(tgt=output.unsqueeze(0), memory=memory).squeeze(0)

            # Step-Specific Prediction
            pred = self.step_heads[t](output)  # [batch_size, n_qubits*n_qubits]

            # Reshape for predictions
            predictions.append(pred.view(B, self.n_qubits, self.n_qubits))

            # Stop Prediction
            stop_logits.append(self.stop_head(output))

        # Stack along sequence dimension -> [batch_size, seq_len, n_qubits, n_qubits]
        predictions = torch.stack(predictions, dim=1)

        # Stack stop logits -> [batch_size, seq_len]
        stop_logits = torch.cat(stop_logits, dim=1)

        return predictions, stop_logits


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
    def __init__(self, alpha=0.5, beta=0.1):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, preds, stop_logits, targets, masks):
        """
        preds: [bs, seq_len, n, n]
        stop_logits: [bs, seq_len] - This has shape [32, 10]
        targets: [bs, seq_len, n, n]
        masks: [bs, seq_len] - But this might have shape [32, 1]
        """
        bs, seq_len = stop_logits.shape  # Use stop_logits shape instead of masks

        # 1. Permutation matrix loss (masked)
        # Make sure targets has same sequence length as preds
        if targets.size(1) < preds.size(1):
            padding = torch.zeros(
                bs,
                preds.size(1) - targets.size(1),
                *targets.shape[2:],
                device=targets.device,
            )
            targets = torch.cat([targets, padding], dim=1)

        perm_loss = F.mse_loss(preds, targets, reduction="none")

        # If masks is too short, pad it
        if masks.size(1) < seq_len:
            mask_padding = torch.zeros(bs, seq_len - masks.size(1), device=masks.device)
            masks = torch.cat([masks, mask_padding], dim=1)

        perm_loss = perm_loss.mean(dim=(-1, -2)) * masks  # [bs, seq_len]
        perm_loss = perm_loss.sum() / masks.sum().clamp(min=1.0)  # Avoid div by zero

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

        return perm_loss + self.alpha * stop_loss + self.beta * length_loss


def train(model, dataloader, epochs=100, device=None):
    # Initialization
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=2e-4, total_steps=epochs * len(dataloader), pct_start=0.3
    )
    criterion = OrderedPermutationLoss()
    # sinkhorn = SequentialSinkhorn()

    # Gradient accumulation (for larger batches)
    accum_steps = 4

    loss_history = []  # To track loss per epoch
    # best_validation_score = float("inf")  # for tracking the best model
    # best_model_state = None
    # validation_scores = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        batches = 0

        optimizer.zero_grad()

        for batch_idx, (tableaus, raw_targets) in enumerate(dataloader):
            # 1. Prepare batch -------------------------------------------------
            # Convert raw targets to padded tensor
            targets, masks = pad_targets(raw_targets, device)

            # Move data to device
            tableaus = tableaus.to(device)

            # 2. Forward pass -------------------------------------------------
            raw_preds, stop_logits = model(tableaus)  # [batch_size, seq_len, n, n]

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
            loss = criterion(
                preds,  # [batch_size, seq_len, n, n]
                stop_logits,  # [batch_size, seq_len]
                targets,  # [batch_size, seq_len, n, n]
                masks,  # [batch_size, seq_len]
            )

            # 5. Backpropagation ----------------------------------------------
            loss.backward()

            # 6. Gradient accumulation -----------------------------------------
            if (batch_idx + 1) % accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

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
    # Extract first sequence from each batch item
    first_sequences = [item[0] for item in raw_targets]

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


class TableauPermutationDataset(Dataset):
    def __init__(self, data_file, n_qubits=4, max_samples=None, shuffle=True):
        print(f"Loading data from {data_file}...")
        with open(data_file, "rb") as f:
            all_data = pickle.load(f)

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

        self.n_qubits = n_qubits

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

        # For debugging
        # print(f"Best perms structure: {type(best_perms)}, length: {len(best_perms)}")

        # Create a list to hold all permutation sequences
        all_perm_sequences = []

        # Process each permutation sequence in best_perms
        for sequence in best_perms:
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

        # If we couldn't create any valid sequences, add a default one
        if not all_perm_sequences:
            all_perm_sequences.append([torch.eye(n_qubits)])

        return self.tableau_to_tensor(tableau), all_perm_sequences


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


def pretrain_a_model_from_file(data_file="training_data_perm.pkl", max_samples=None):
    n_qubits = 4
    # model = OrderedPermutationTransformer(n_qubits=n_qubits)
    model = OrderedPermutationTransformer(n_qubits=4, dim=256, num_layers=6)
    dataset = TableauPermutationDataset(
        data_file, n_qubits=n_qubits, max_samples=max_samples
    )

    # Use standard PyTorch DataLoader with custom collate function
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        collate_fn=custom_collate_fn,
    )

    # Auto-detect device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # train(model, dataloader, epochs=10, device=device)
    train(
        model, dataloader, epochs=30, device=device
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


def rl_fine_tune(model, dataset, epochs=5, device="cpu"):
    """Fine-tune permutation model with RL targeting CX reduction"""
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-6)
    reward_history = []  # For reward normalization

    for epoch in range(epochs):
        total_reward = 0
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=1, shuffle=True, collate_fn=custom_collate_fn
        )

        for batch_idx, (tableau_tensors, _) in enumerate(dataloader):
            # Get tableau and convert to proper format
            tableau_tensor = tableau_tensors[0]
            clifford_tableau = convert_tensor_to_tableau(tableau_tensor)
            tableau_tensor = tableau_tensor.unsqueeze(0).to(device)

            # BASELINE: Get CX count with normal heuristic
            topology = Topology.complete(clifford_tableau.n_qubits)
            baseline_circuit = synthesize_tableau_perm_row_col(
                clifford_tableau, topology
            )
            baseline_cx = collect_circuit_data(baseline_circuit)["cx"]

            # Get model prediction with gradient tracking
            optimizer.zero_grad()
            raw_preds, _ = model(tableau_tensor)
            logits = raw_preds[0, 0]

            # Apply Sinkhorn with gradient tracking
            for _ in range(20):
                logits = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
                logits = logits - torch.logsumexp(logits, dim=-2, keepdim=True)

            # Sample permutation using Gumbel-Softmax
            n_qubits = logits.shape[0]
            gumbels = -torch.log(-torch.log(torch.rand_like(logits)))
            gumbel_logits = (logits + gumbels) / 0.5  # Temperature
            probs = F.softmax(gumbel_logits, dim=-1)

            # Extract permutation using Hungarian algorithm
            perm_matrix = probs.detach().cpu().numpy()
            row_ind, col_ind = linear_sum_assignment(-perm_matrix)
            perm = [(int(i), int(j)) for i, j in zip(row_ind, col_ind)]

            # Track log probabilities for gradient flow
            log_probs = []
            for i, j in zip(row_ind, col_ind):
                log_probs.append(torch.log(probs[i, j] + 1e-10))

            # Evaluate CX count (detached from graph)
            with torch.no_grad():
                test_tableau = copy.deepcopy(clifford_tableau)
                pred_iter = iter(perm)

                def pred_callback(G, remaining, remaining_rows, choice_fn=min):
                    try:
                        row, col = next(pred_iter)
                        if isinstance(row, torch.Tensor):
                            row = row.item()
                        if isinstance(col, torch.Tensor):
                            col = col.item()

                        # CRITICAL: Validate pivot exists in graph
                        graph_nodes = set(G.nodes())
                        if (
                            row in remaining_rows
                            and row in graph_nodes
                            and col in graph_nodes
                            and G.has_edge(row, col)
                        ):
                            return int(row), int(col)
                        else:
                            raise StopIteration

                    except StopIteration:
                        # Safe fallback using G
                        valid_rows = [r for r in remaining_rows if r in G]
                        if not valid_rows:
                            row = min(remaining_rows)
                            return row, row

                        row = choice_fn(valid_rows)
                        if G[row]:
                            col = next(iter(G[row]))  # Guaranteed safe
                        else:
                            col = row
                        return int(row), int(col)

                circuit = synthesize_tableau_perm_row_col(
                    test_tableau, topology, pick_pivot_callback=pred_callback
                )
                cx_count = collect_circuit_data(circuit)["cx"]

            # Calculate reward with normalization
            reward = baseline_cx - cx_count
            reward_history.append(reward)

            if len(reward_history) > 10:
                mean_reward = sum(reward_history[-10:]) / 10
                std_reward = max(1.0, np.std(reward_history[-10:]))
                normalized_reward = (reward - mean_reward) / std_reward
            else:
                normalized_reward = reward

            # REINFORCE loss
            policy_loss = -sum(log_probs) * normalized_reward

            # Backprop and update
            policy_loss.backward()
            optimizer.step()

            total_reward += reward

            if batch_idx % 10 == 0:
                print(
                    f"Epoch {epoch+1} | Batch {batch_idx} | Reward: {reward:.2f} | CX: {cx_count}"
                )

        print(f"Epoch {epoch+1} | Avg Reward: {total_reward/len(dataloader):.2f}")

    return model


def supervised_cx_fine_tune(model, dataset, epochs=5, device="cpu"):
    """Fine-tune with supervised learning focusing on CX reduction"""
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

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
                    raw_preds, _ = model(tableau_tensor)
                    target_perm = None
                    best_cx = baseline_cx

                    # Try different permutations
                    for step_idx in range(min(3, raw_preds.shape[1])):
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

                # If we found a better permutation, train toward it
                if target_perm is not None and best_cx < baseline_cx:
                    # Create target matrix
                    target = torch.zeros_like(raw_preds[0, 0])
                    for i, j in target_perm:
                        target[i, j] = 1.0

                    # Train model to predict this permutation
                    new_preds, _ = model(tableau_tensor)
                    logits = new_preds[0, 0]
                    loss = F.mse_loss(torch.softmax(logits, dim=-1), target)
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
    depth_weight = 1.0
    # Compute weighted score, only keep cx seems worse
    score = cx_weight * pred_metrics["cx"] + depth_weight * pred_metrics["depth"]
    # score = pred_metrics["cx"]
    return score


def compute_adaptive_score(pred_metrics, baseline_metrics):
    """Adapt weights based on how close we are to optimum"""
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


def predict_permutation(model, clifford_tableau, device="cpu"):
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
        raw_preds, _ = model(tableau_tensor)

        # Try the top 3 steps and keep the best one
        for step_idx in range(min(8, raw_preds.shape[1])):  # try 5 instead of 3
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
    print(f"Best score: {best_score}")
    return [best_perm]  # Keep list format for compatibility


# # Example of usage:
# model = pretrain_a_model_from_file("training_data_perm.pkl", max_samples=320)
# # Try RL fine-tuning
# # rl_dataset = TableauPermutationDataset(
# #     data_file="training_data_perm_4_qubit.pkl", n_qubits=4, max_samples=320
# # )
# # rl_model = rl_fine_tune(
# #     model, rl_dataset, epochs=5, device="cuda" if torch.cuda.is_available() else "cpu"
# # )
# sl_dataset = TableauPermutationDataset(
#     data_file="training_data_perm_4_qubit.pkl", n_qubits=4, max_samples=320
# )
# sl_model = supervised_cx_fine_tune(
#     model, sl_dataset, epochs=5, device="cuda" if torch.cuda.is_available() else "cpu"
# )
# circuit = random_hscx_circuit(nr_qubits=4, nr_gates=1000)
# tableau = tableau_from_circuit(CliffordTableau(4), circuit)
# # permutations = predict_permutation(model, tableau)
# permutations = predict_permutation(sl_model, tableau)
# print(permutations)
