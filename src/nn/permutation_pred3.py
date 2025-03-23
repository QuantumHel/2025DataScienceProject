import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool
from torch_geometric.nn import GraphConv
from einops import rearrange
from src.utils import tableau_from_circuit, random_hscx_circuit
from pauliopt.clifford.tableau import CliffordTableau
from src.nn.brute_force_data import get_best_cnots
from pauliopt.topologies import Topology
from torch_geometric.data import Data
import numpy as np
import warnings
from scipy.optimize import linear_sum_assignment


# Suppress all overflow warnings globally
np.seterr(over="ignore")

# Suppress FutureWarning
warnings.simplefilter(action="ignore", category=FutureWarning)


def clifford_tableau_to_graph(clifford_tableau, max_qubits=5):
    n_qubits = clifford_tableau.n_qubits
    tableau = clifford_tableau.tableau
    signs = clifford_tableau.signs

    # Calculate target feature dimension
    target_dim = 1 + 4 * max_qubits

    # Node features with stabilizer/destabilizer info
    node_features = []
    for q in range(n_qubits):
        # Extract the actual features for this qubit
        x_stab = tableau[:n_qubits, q]
        z_stab = tableau[:n_qubits, q + n_qubits]
        x_destab = tableau[n_qubits:, q]
        z_destab = tableau[n_qubits:, q + n_qubits]

        # Combine features and pad to target dimension
        features = [q] + list(x_stab) + list(z_stab) + list(x_destab) + list(z_destab)

        # Pad with zeros to reach the target dimension
        features = features + [0] * (target_dim - len(features))

        node_features.append(features)

    node_features = torch.tensor(node_features, dtype=torch.float)

    # Edge construction with fallback
    edge_indices = []
    edge_features = []

    # Create edges from tableau
    for row_idx in range(2 * n_qubits):
        x_part = tableau[row_idx, :n_qubits]
        z_part = tableau[row_idx, n_qubits:]
        involved_qubits = [q for q in range(n_qubits) if x_part[q] or z_part[q]]

        # Add edges for multi-qubit operators
        for i in range(len(involved_qubits)):
            for j in range(i + 1, len(involved_qubits)):
                edge_indices.append([involved_qubits[i], involved_qubits[j]])
                edge_feature = [
                    x_part[involved_qubits[i]],
                    z_part[involved_qubits[i]],
                    x_part[involved_qubits[j]],
                    z_part[involved_qubits[j]],
                    row_idx < n_qubits,  # Stabilizer flag
                    signs[row_idx],
                ]
                edge_features.append(edge_feature)

    # Add self-loops if no edges found
    if not edge_indices:
        for q in range(n_qubits):
            edge_indices.append([q, q])
            edge_features.append([0, 0, 0, 0, 1, 0])  # Dummy features

    edge_indices = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    edge_features = torch.tensor(edge_features, dtype=torch.float)

    return Data(x=node_features, edge_index=edge_indices, edge_attr=edge_features)


class FlexibleGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, max_qubits=5):
        super().__init__()
        self.input_dim = input_dim
        self.max_qubits = max_qubits
        # Use GraphConv which handles edges differently than GCNConv
        self.conv1 = GraphConv(input_dim, hidden_dim)
        self.conv2 = GraphConv(hidden_dim, hidden_dim)
        self.conv3 = GraphConv(hidden_dim, hidden_dim)
        self.fc = nn.Linear(
            hidden_dim, max_qubits * max_qubits
        )  # Change output size to max_qubits^2

    def forward(self, data):
        # Handle empty edge cases
        if data.edge_index.size(1) == 0:
            # Create temporary self-loops
            edge_index = (
                torch.tensor([[i, i] for i in range(data.x.size(0))], dtype=torch.long)
                .t()
                .contiguous()
            )
        else:
            edge_index = data.edge_index

        # GraphConv doesn't require edge_attr
        x = F.relu(self.conv1(data.x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        x = global_add_pool(x, batch=torch.zeros(x.size(0), dtype=torch.long))

        # Get output logits (now with correct dimensions)
        logits = self.fc(x)  # Shape: [batch_size, max_qubits^2]

        # Extract only the needed values and reshape to match current qubit count
        n_qubits = data.x.size(0)
        return logits[:, : n_qubits * n_qubits].view(-1, n_qubits, n_qubits)


def robust_decode(logits):
    """Guaranteed non-empty permutation decoder"""
    n_qubits = logits.shape[0]

    # Add small noise to prevent all-zero selection
    logits = logits + torch.randn_like(logits) * 0.01

    # Hungarian algorithm with fallback
    try:
        row_ind, col_ind = linear_sum_assignment(logits.detach().numpy())
    except ValueError:
        # Fallback to diagonal pairs if Hungarian fails
        return [(i, (i + 1) % n_qubits) for i in range(n_qubits)]

    # Ensure non-empty output
    pairs = [(int(r), int(c)) for r, c in zip(row_ind, col_ind) if r != c]
    return pairs if pairs else [(i, (i + 1) % n_qubits) for i in range(n_qubits)]


class AdaptiveLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, outputs, targets):
        losses = []
        for logits, target in zip(outputs, targets):
            n = logits.shape[0]
            target_mat = torch.zeros(n, n)
            for r, c in target:
                if r < n and c < n:  # Handle variable sizes
                    target_mat[r, c] = 1
            losses.append(F.binary_cross_entropy_with_logits(logits, target_mat))
        return torch.mean(torch.stack(losses))


def validate_permutation(perm, n_qubits):
    """Ensures valid dual permutation constraints"""
    # Check bijective mapping
    rows = set()
    cols = set()
    for r, c in perm:
        if r in rows or c in cols or r == c:
            return False
        rows.add(r)
        cols.add(c)
    return len(rows) == n_qubits and len(cols) == n_qubits


def train_flexible(
    model, optimizer, n_qubits_list=[2, 3, 4, 5], batch_size=32, num_epochs=100
):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for _ in range(batch_size):
            # Random qubit count
            n_qubits = int(np.random.choice(n_qubits_list))
            # print(f"Training for {n_qubits} qubits")

            # Generate circuit with minimum gates
            circuit = random_hscx_circuit(nr_qubits=n_qubits, nr_gates=1000)
            tableau = tableau_from_circuit(CliffordTableau(n_qubits), circuit)
            graph = clifford_tableau_to_graph(tableau)

            # Get best permutation
            best_perms = get_best_cnots(tableau, Topology.complete(n_qubits))
            target = best_perms[0][0] if best_perms else []

            # Forward pass
            optimizer.zero_grad()
            output = model(graph)

            # Create target matrix
            target_mat = create_target_matrix(n_qubits, target)

            # Calculate loss
            loss = F.binary_cross_entropy_with_logits(output.squeeze(0), target_mat)

            # Backprop
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss/batch_size:.4f}")


def create_target_matrix(n_qubits, permutation):
    """
    Converts permutation tuples into a binary target matrix
    Example: [(0,1), (1,0)] -> [[0,1],
                                [1,0]]
    """
    mat = torch.zeros(n_qubits, n_qubits)
    for r, c in permutation:
        if 0 <= r < n_qubits and 0 <= c < n_qubits:
            mat[r, c] = 1
    return mat


def predict_permutation(model, clifford_tableau, exclude_self_loops=False):
    """Predicts permutation tuples for a given tableau"""
    model.eval()
    with torch.no_grad():
        graph = clifford_tableau_to_graph(clifford_tableau)
        logits = model(graph).squeeze(0)

        # Optionally penalize self-loops to avoid them in the solution
        if exclude_self_loops:
            # Add large penalty for diagonal elements (self-loops)
            penalty_matrix = torch.eye(logits.shape[0]) * 1000
            logits = logits - penalty_matrix

        # Hungarian algorithm decoding
        row_ind, col_ind = linear_sum_assignment(-logits.numpy())

        # Return full permutation or filtered version
        if exclude_self_loops:
            return [(int(r), int(c)) for r, c in zip(row_ind, col_ind) if r != c]
        else:
            return [(int(r), int(c)) for r, c in zip(row_ind, col_ind)]


def pretrain_a_model() -> FlexibleGNN:
    model = FlexibleGNN(input_dim=max(1 + 4 * q for q in [2, 3, 4, 5]))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    train_flexible(model, optimizer)
    print("Model pre-trained")
    return model


# # Usage example
# # Initialize model and optimizer
# # For a model that can handle different qubit counts:
# model = FlexibleGNN(input_dim=max(1 + 4 * q for q in [2, 3, 4, 5]))
# # # For a model trained on specific qubit count (e.g., 5):
# # model = FlexibleGNN(input_dim=1 + 4*5)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# # Train the model
# train_flexible(model, optimizer)

# # Test prediction
# tableau = CliffordTableau(5)
# tableau = tableau_from_circuit(tableau, random_hscx_circuit(nr_qubits=5, nr_gates=100))
# pred = predict_permutation(model, tableau)
# print(f"Predicted permutation: {pred}")

# # How to measure by depth and cnot counts?
