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

# Suppress all overflow warnings globally
np.seterr(over="ignore")

# Suppress FutureWarning
warnings.simplefilter(action="ignore", category=FutureWarning)


class OrderedPermutationTransformer(nn.Module):
    def __init__(self, n_qubits, dim=128, num_layers=4):
        super().__init__()
        self.n_qubits = n_qubits
        self.dim = dim

        # Tableau Encoder
        self.tableau_encoder = nn.Sequential(
            nn.Conv2d(2, 16, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.AdaptiveAvgPool2d((4, 4)),
        )

        # Linear layer to match the dimension
        self.linear = nn.Linear(32 * 4 * 4, dim)

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
        x = x.view(B, -1)
        x = self.linear(x)  # Ensure the dimension matches

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
    sinkhorn = SequentialSinkhorn()

    # Gradient accumulation (for larger batches)
    accum_steps = 4

    loss_history = []  # To track loss per epoch
    best_validation_score = float("inf")  # for tracking the best model
    best_model_state = None
    validation_scores = []

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

        # Optional - plot every few epochs
        if epoch % 5 == 0 or epoch == epochs - 1:
            plt.figure(figsize=(10, 6))
            plt.plot(range(1, len(loss_history) + 1), loss_history, marker="o")
            plt.title("Training Loss by Epoch")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.grid(True)
            plt.savefig(f"loss_epoch_{epoch+1}.png")
            plt.close()

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
        collate_fn=custom_collate_fn,  # Add this line
    )

    # Auto-detect device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # train(model, dataloader, epochs=10, device=device)
    train(model, dataloader, epochs=100, device=device)
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

    with torch.no_grad():
        # Get predictions for all steps
        raw_preds, stop_logits = model(tableau_tensor)

        # Try the top 3 steps and keep the best one
        for step_idx in range(min(3, raw_preds.shape[1])):
            # Apply Sinkhorn normalization
            logits = raw_preds[0, step_idx]
            for _ in range(20):
                logits = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
                logits = logits - torch.logsumexp(logits, dim=-2, keepdim=True)
            perm_matrix = torch.exp(logits / 0.1).cpu().numpy()

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
                except StopIteration:
                    row = choice_fn(remaining_rows)
                    return row, row

            topology = Topology.complete(clifford_tableau.n_qubits)
            pred_circuit = synthesize_tableau_perm_row_col(
                clifford_tableau, topology, pick_pivot_callback=pred_callback
            )
            pred_metrics = collect_circuit_data(pred_circuit)

            # Synthesize circuit and count gates
            score = pred_metrics["cx"] + pred_metrics["depth"]

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
# circuit = random_hscx_circuit(nr_qubits=4, nr_gates=1000)
# tableau = tableau_from_circuit(CliffordTableau(4), circuit)
# permutations = predict_permutation(model, tableau)
# print(permutations)
