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

from scipy.optimize import linear_sum_assignment

# Suppress all overflow warnings globally
np.seterr(over="ignore")

# Suppress FutureWarning
warnings.simplefilter(action="ignore", category=FutureWarning)


def clifford_tableau_to_graph(clifford_tableau):
    """
    Convert a CliffordTableau to a graph with nodes as qubits.
    Integrates stabilizer and destabilizer interactions into edges.
    """
    n_qubits = clifford_tableau.n_qubits
    tableau = clifford_tableau.tableau
    signs = clifford_tableau.signs

    # Node features: qubit indices (optional, can be expanded)
    node_features = torch.arange(n_qubits, dtype=torch.float).view(-1, 1)

    # Edge indices and features
    edge_indices = []
    edge_features = []

    # Iterate over all rows (stabilizers + destabilizers)
    for row_idx in range(2 * n_qubits):
        row_type = 0 if row_idx < n_qubits else 1  # 0=stabilizer, 1=destabilizer
        sign = signs[row_idx]
        x_part = tableau[row_idx, :n_qubits]  # X components
        z_part = tableau[row_idx, n_qubits:]  # Z components

        # Find qubits involved in this row
        involved_qubits = [q for q in range(n_qubits) if x_part[q] or z_part[q]]

        # Create edges between all pairs of involved qubits
        for i in range(len(involved_qubits)):
            for j in range(i + 1, len(involved_qubits)):
                q_i = involved_qubits[i]
                q_j = involved_qubits[j]

                # Edge features: X/Z terms for both qubits, row type, sign
                edge_feature = [
                    x_part[q_i],
                    z_part[q_i],  # Pauli terms for q_i
                    x_part[q_j],
                    z_part[q_j],  # Pauli terms for q_j
                    row_type,  # Stabilizer or destabilizer
                    sign,  # Sign of the operator
                ]
                edge_indices.append([q_i, q_j])
                edge_features.append(edge_feature)

    edge_indices = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    edge_features = torch.tensor(edge_features, dtype=torch.float)

    # Create PyTorch Geometric Data object
    graph = Data(
        x=node_features,  # Node features (qubit indices)
        edge_index=edge_indices,  # Edge indices
        edge_attr=edge_features,  # Edge features
    )
    return graph


class PermutationConstrainedGNN(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim, n_qubits):
        super().__init__()
        self.n_qubits = n_qubits
        self.embed = nn.Linear(node_dim, hidden_dim)

        self.gnn_layers = nn.ModuleList(
            [GCNConv(hidden_dim, hidden_dim) for _ in range(3)]
        )

        # Permutation prediction head
        self.perm_head = nn.Sequential(
            nn.Linear(hidden_dim, 4 * n_qubits**2),
            nn.ReLU(),
            nn.Linear(4 * n_qubits**2, n_qubits**2),
        )

    def forward(self, data):
        x = self.embed(data.x)
        for conv in self.gnn_layers:
            x = conv(x, data.edge_index)
            x = F.relu(x)

        # Global pooling
        x = global_add_pool(x, batch=torch.zeros(x.size(0), dtype=torch.long))

        # Predict permutation matrix logits
        logits = self.perm_head(x)
        logits = rearrange(logits, "b (n m) -> b n m", n=self.n_qubits, m=self.n_qubits)
        return logits


# class PermutationLoss(nn.Module):
#     def __init__(self, alpha=0.1):
#         super().__init__()
#         self.alpha = alpha

#     def forward(self, logits, targets):
#         # Convert targets to permutation matrices
#         target_mats = torch.zeros_like(logits)
#         for i, perm in enumerate(targets):
#             for idx, (row, col) in enumerate(perm):
#                 target_mats[i, row, col] = 1

#         # Main cross entropy loss
#         ce_loss = F.binary_cross_entropy_with_logits(logits, target_mats)

#         # Birkhoff polytope regularization
#         row_sum = torch.sigmoid(logits).sum(dim=2)
#         col_sum = torch.sigmoid(logits).sum(dim=1)
#         birk_loss = F.mse_loss(row_sum, torch.ones_like(row_sum)) + F.mse_loss(
#             col_sum, torch.ones_like(col_sum)
#         )

#         return ce_loss + self.alpha * birk_loss


# class ImprovedPermutationLoss(nn.Module):
class PermutationLoss(nn.Module):
    def __init__(self, temp=0.1, alpha=0.1, sinkhorn_iters=20):
        super().__init__()
        self.temp = temp
        self.alpha = alpha
        self.sinkhorn_iters = sinkhorn_iters

    def sinkhorn(self, logits):
        """Differentiable Sinkhorn normalization"""
        for _ in range(self.sinkhorn_iters):
            logits = logits - torch.logsumexp(logits, dim=2, keepdim=True)
            logits = logits - torch.logsumexp(logits, dim=1, keepdim=True)
        return torch.exp(logits)

    def forward(self, logits, target_perms):
        # Apply Sinkhorn to raw logits
        sinkhorn_probs = self.sinkhorn(logits / self.temp)

        # Convert targets to matrices
        target_mats = torch.zeros_like(logits)
        for i, perm in enumerate(target_perms):
            for r, c in perm:
                target_mats[i, r, c] = 1

        # KL divergence (prediction vs target)
        kl_loss = F.kl_div(
            sinkhorn_probs.log(),
            target_mats,  # No temperature on targets
            reduction="batchmean",
        )

        # Validity loss (encourage sharpness)
        validity_loss = (1 - sinkhorn_probs.max(dim=2).values.mean()) + (
            1 - sinkhorn_probs.max(dim=1).values.mean()
        )

        return kl_loss + self.alpha * validity_loss


def constrained_decode(logits, n_qubits):
    """Guaranteed valid dual permutation decoder"""
    # Convert logits to probability matrix
    probs = F.softmax(logits, dim=-1)
    probs = probs.view(n_qubits, n_qubits).cpu().numpy()

    # Use Hungarian algorithm for optimal assignment
    from scipy.optimize import linear_sum_assignment

    row_ind, col_ind = linear_sum_assignment(-probs)  # Negative for maximization

    # Convert to permutation pairs
    permutation = list(zip(row_ind.tolist(), col_ind.tolist()))

    # Post-process to CNOT sequence format
    return [
        (int(r), int(c)) for r, c in permutation if r != c
    ]  # Exclude identity pairs


def constrained_decode_complete(logits, n_qubits):
    """Return complete permutation including identity mappings"""
    probs = F.softmax(logits, dim=-1)
    probs = probs.view(n_qubits, n_qubits).cpu().numpy()

    # Use Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(-probs)

    # Return ALL mappings, including self-loops
    return [(int(r), int(c)) for r, c in zip(row_ind, col_ind)]


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


def train_model(n_qubits, model, optimizer, criterion, batch_size=32, num_gates=100):
    model.train()
    total_loss = 0

    for _ in range(batch_size):
        # Generate training data
        circuit = random_hscx_circuit(nr_qubits=n_qubits, nr_gates=num_gates)
        tableau = CliffordTableau(n_qubits)
        tableau = tableau_from_circuit(tableau, circuit)
        graph = clifford_tableau_to_graph(tableau)

        # Get valid permutations
        best_perms = get_best_cnots(tableau, Topology.complete(n_qubits))
        # target = best_perms[0][0] if best_perms else []

        # Forward pass
        optimizer.zero_grad()
        logits = model(graph)

        # Calculate loss
        # loss = criterion(logits, [target])
        # Calculate losses for each permutation
        min_loss = float("inf")  # Track minimum loss
        for perm in best_perms:
            curr_loss = criterion(logits, [perm[0]])
            min_loss = min(min_loss, curr_loss)

        # Backpropagate
        # loss.backward()
        min_loss.backward()

        # After loss.backward() but before optimizer.step()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Print gradients here
        print("Gradients for first GNN layer:", model.gnn_layers[0].lin.weight.grad)
        print("Gradients for embedding layer:", model.embed.weight.grad)

        optimizer.step()

        # total_loss += loss.item()
        total_loss += min_loss.item()

    return total_loss / batch_size


# # Example usage
# n_qubits = 4
# model = PermutationConstrainedGNN(
#     node_dim=1, edge_dim=6, hidden_dim=128, n_qubits=n_qubits
# )
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
# criterion = PermutationLoss(alpha=0.5)

# # Training loop
# for epoch in range(3):
#     loss = train_model(n_qubits, model, optimizer, criterion)
#     print(f"Epoch {epoch+1}, Loss: {loss:.4f}")


def pretrain_a_model() -> PermutationConstrainedGNN:
    n_qubits = 4
    model = PermutationConstrainedGNN(
        node_dim=1, edge_dim=6, hidden_dim=128, n_qubits=n_qubits
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
    criterion = PermutationLoss(alpha=0.5)

    # Start with simpler circuits and gradually increase complexity
    gate_schedule = [50, 100, 200, 500, 1000]
    for epoch in range(20):
        # Select gate count based on training progress
        gates = gate_schedule[min(epoch // 4, len(gate_schedule) - 1)]
        loss = train_model(n_qubits, model, optimizer, criterion, num_gates=gates)
        print(f"Epoch {epoch+1}, Gates: {gates}, Loss: {loss:.4f}")

    return model


def predict_permutation(model, clifford_tableau, device="cpu"):
    """
    Predicts valid permutation tuples for a Clifford tableau using the trained model
    Returns: List of (control, target) tuples
    """
    # 1. Convert tableau to graph
    graph = clifford_tableau_to_graph(clifford_tableau)

    # 2. Prepare input for model
    data = graph.to(device)

    # 3. Model forward pass
    model.eval()
    with torch.no_grad():
        raw_output = model(data)

    # 4. Constrained decoding with validity checks
    print(raw_output)
    permutation = constrained_decode_complete(
        raw_output, n_qubits=clifford_tableau.n_qubits
    )

    # assert validate_permutation(permutation, clifford_tableau.n_qubits), "Invalid permutation generated"
    # print(f"Valid permutation: {permutation}")
    # # Example output: [(0, 1), (1, 0), (2, 3), (3, 2)]

    # 5. Post-processing to ensure physical validity
    # return enforce_permutation_constraints(permutation)
    return permutation


def enforce_permutation_constraints(permutation):
    """Ensures final permutation validity"""
    # 1. Remove duplicates
    seen = set()
    return [
        tuple(pair)
        for pair in permutation
        if not (tuple(pair) in seen or seen.add(tuple(pair)))
    ]


# tableau = CliffordTableau(4)
# tableau = tableau_from_circuit(tableau, random_hscx_circuit(nr_qubits=4, nr_gates=100))
# pred = predict_permutation(model, tableau)
# print(pred)
