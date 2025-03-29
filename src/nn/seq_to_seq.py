import torch
import torch.nn as nn
import torch.optim as optim
import random

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

# Suppress all overflow warnings globally
np.seterr(over="ignore")

# Suppress FutureWarning
warnings.simplefilter(action="ignore", category=FutureWarning)


class TableauEncoder(nn.Module):
    def __init__(self, n_qubits, hidden_dim=128):
        super().__init__()
        # Process 2-channel tableau input
        self.conv = nn.Sequential(
            nn.Conv2d(2, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(32 * 4 * 4, hidden_dim)
        self.rnn = nn.LSTM(hidden_dim, hidden_dim)

    def forward(self, x):
        # x shape: [batch_size, 2, 2*n_qubits, 2*n_qubits]
        batch_size = x.size(0)
        features = self.conv(x)
        features = self.flatten(features)
        features = self.linear(features)
        features = features.unsqueeze(0)  # Add sequence dim [1, batch, hidden]
        _, (hidden, cell) = self.rnn(features)
        return hidden, cell


class PermutationDecoder(nn.Module):
    def __init__(self, n_qubits, hidden_dim=128):
        super().__init__()
        self.n_qubits = n_qubits

        # Embedding for representing permutation operations
        self.embed = nn.Embedding(
            n_qubits * n_qubits + 1, hidden_dim
        )  # +1 for start token

        # RNN for sequence generation
        self.rnn = nn.LSTM(hidden_dim, hidden_dim)

        # Output layers for permutation matrix
        self.perm_out = nn.Linear(hidden_dim, n_qubits * n_qubits)

        # Output for stop prediction
        self.stop_out = nn.Linear(hidden_dim, 1)

    def forward(self, prev_perm, hidden, cell):
        # prev_perm: [batch_size] indices of previous permutation
        embedded = self.embed(prev_perm).unsqueeze(0)  # [1, batch, hidden]

        # Run RNN step
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))

        # Generate permutation matrix logits
        perm_logits = self.perm_out(output.squeeze(0))
        perm_logits = perm_logits.view(-1, self.n_qubits, self.n_qubits)

        # Generate stop token
        stop_logit = self.stop_out(output.squeeze(0))

        return perm_logits, stop_logit, hidden, cell


class PermutationSeq2Seq(nn.Module):
    def __init__(self, n_qubits, hidden_dim=128):
        super().__init__()
        self.n_qubits = n_qubits
        self.encoder = TableauEncoder(n_qubits, hidden_dim)
        self.decoder = PermutationDecoder(n_qubits, hidden_dim)
        self.max_len = 10  # Maximum sequence length

    def forward(self, tableau, target_seq=None, teacher_forcing_ratio=0.5):
        """
        tableau: [batch, 2, 2*n_qubits, 2*n_qubits] - Tableau tensor
        target_seq: List of permutation matrices (optional, for training)
        """
        batch_size = tableau.size(0)
        device = tableau.device

        # Encode the tableau
        hidden, cell = self.encoder(tableau)

        # Initialize outputs
        perm_logits_seq = []
        stop_logits_seq = []

        # Start with a "start" token
        prev_perm = torch.full(
            (batch_size,), self.n_qubits * self.n_qubits, device=device
        )  # Start token

        # Generate sequence
        for t in range(self.max_len):
            # Decode step
            perm_logits, stop_logit, hidden, cell = self.decoder(
                prev_perm, hidden, cell
            )

            # Save outputs
            perm_logits_seq.append(perm_logits)
            stop_logits_seq.append(stop_logit)

            # Teacher forcing (if training)
            use_teacher_forcing = (
                random.random() < teacher_forcing_ratio and target_seq is not None
            )

            if use_teacher_forcing and t < len(target_seq):
                # Convert target permutation matrix to index
                indices = target_seq[t].flatten(1).argmax(dim=1)
                prev_perm = indices
            else:
                # Use model's prediction
                perm_matrix = perm_logits.view(batch_size, -1)
                prev_perm = perm_matrix.argmax(dim=1)

            # Stop if all batches predict stop token
            if t > 0 and torch.sigmoid(stop_logits_seq[-2]).mean() > 0.5:
                break

        return (
            torch.stack(perm_logits_seq, dim=1),  # [batch, seq, n, n]
            torch.cat(stop_logits_seq, dim=1),
        )  # [batch, seq]


def pretrain_a_model_from_file(data_file="training_data_perm.pkl", max_samples=None):
    n_qubits = 4
    model = PermutationSeq2Seq(n_qubits=n_qubits)
    dataset = TableauPermutationDataset(
        data_file, n_qubits=n_qubits, max_samples=max_samples
    )

    # Use standard PyTorch DataLoader with custom collate function
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=32, shuffle=True, collate_fn=custom_collate_fn
    )

    # Auto-detect device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setup optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = OrderedPermutationLoss()  # Your custom loss

    # Training loop
    model.to(device)
    num_epochs = 10

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        total_batches = 0

        for tableaus, raw_targets in dataloader:
            # Move data to device
            tableaus = tableaus.to(device)

            # Convert raw_targets to tensor format
            targets, masks = pad_targets(raw_targets, device)

            # Forward pass
            perm_logits, stop_logits = model(tableaus, raw_targets)

            # Calculate loss
            loss = criterion(perm_logits, stop_logits, targets, masks)

            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_batches += 1

        # Print epoch summary
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/total_batches:.4f}")

    return model


def predict_permutation(model, clifford_tableau, device=None):
    """
    Predicts valid permutation tuples for a Clifford tableau using the trained model.
    Returns a list of permutation tuples.
    """
    # Auto-detect device if not specified
    if device is None:
        device = next(model.parameters()).device

    # 1. Convert tableau to tensor format
    n_qubits = clifford_tableau.n_qubits
    tensor_converter = TableauPermutationDataset(None, n_qubits=n_qubits)
    tableau_tensor = tensor_converter.tableau_to_tensor(clifford_tableau)
    tableau_tensor = tableau_tensor.unsqueeze(0).to(device)  # Add batch dimension

    # 2. Model forward pass in evaluation mode
    model.eval()
    with torch.no_grad():
        perm_logits, stop_logits = model(tableau_tensor)

    # 3. Process the predictions
    permutation_sequence = []
    for t in range(perm_logits.size(1)):
        # Stop when stop token is activated
        if t > 0 and torch.sigmoid(stop_logits[0, t - 1]) > 0.5:
            break

        # Extract permutation matrix
        perm_matrix = perm_logits[0, t].cpu()

        # Convert to tuples using Hungarian algorithm
        row_ind, col_ind = linear_sum_assignment(-perm_matrix.detach().numpy())
        perm_step = [(int(r), int(c)) for r, c in zip(row_ind, col_ind)]

        # Add to sequence
        permutation_sequence.append(perm_step)

    return permutation_sequence


# Example usage
model = pretrain_a_model_from_file("training_data_perm.pkl")
clifford_tableau = CliffordTableau(4)
clifford_tableau = tableau_from_circuit(clifford_tableau, random_hscx_circuit(4, 100))
permutation_sequence = predict_permutation(model, clifford_tableau)
print(permutation_sequence)
