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
from torch_scatter import scatter

from scipy.optimize import linear_sum_assignment

# Suppress all overflow warnings globally
np.seterr(over="ignore")

# Suppress FutureWarning
warnings.simplefilter(action="ignore", category=FutureWarning)

from torch_geometric.nn import GlobalAttention
from torch_geometric.nn import aggr


# def clifford_tableau_to_graph(clifford_tableau):
#     """Enhanced node features with stabilizer/destabilizer data"""
#     n = clifford_tableau.n_qubits
#     tableau = clifford_tableau.tableau

#     # Node features: [qubit_idx, destab_x, destab_z, stab_x, stab_z]
#     node_feats = []
#     for q in range(n):
#         stab_x = tableau[:n, q]
#         stab_z = tableau[:n, q + n]
#         destab_x = tableau[n:, q]
#         destab_z = tableau[n:, q + n]
#         node_feats.append(
#             torch.cat(
#                 [
#                     torch.tensor([q]),
#                     torch.tensor(stab_x, dtype=torch.float),  # Convert from numpy
#                     torch.tensor(stab_z, dtype=torch.float),  # Convert from numpy
#                     torch.tensor(destab_x, dtype=torch.float),  # Convert from numpy
#                     torch.tensor(destab_z, dtype=torch.float),  # Convert from numpy
#                 ]
#             )
#         )

#     # Edge construction (optimized)
#     edges = []
#     for row in range(2 * n):
#         x = torch.tensor(
#             tableau[row, :n], dtype=torch.bool
#         )  # Convert to PyTorch tensor
#         z = torch.tensor(
#             tableau[row, n:], dtype=torch.bool
#         )  # Convert to PyTorch tensor
#         involved = torch.where(x | z)[0]
#         if len(involved) >= 2:
#             pairs = torch.combinations(involved, 2)
#             edge_feats = torch.stack(
#                 [
#                     x[pairs[:, 0]],  # Already a tensor, no need for torch.tensor()
#                     z[pairs[:, 0]],
#                     x[pairs[:, 1]],
#                     z[pairs[:, 1]],
#                     torch.full((len(pairs),), float(row < n)),
#                 ],
#                 dim=1,
#             )
#             edges.append((pairs, edge_feats))

#     # Handle empty edges case
#     if not edges:
#         # Create self-loops with dummy features as fallback
#         dummy_edges = torch.stack([torch.arange(n), torch.arange(n)], dim=0)
#         dummy_attr = torch.zeros((n, 5))  # 5 edge features with zeros
#         return Data(
#             x=torch.stack(node_feats), edge_index=dummy_edges, edge_attr=dummy_attr
#         )

#     edge_index = torch.cat([p.t() for p, _ in edges], dim=1)
#     edge_attr = torch.cat([e for _, e in edges], dim=0)

#     return Data(x=torch.stack(node_feats), edge_index=edge_index, edge_attr=edge_attr)


def clifford_tableau_to_graph(clifford_tableau):
    """Enhanced graph construction with simplified edge features and global context"""
    n = clifford_tableau.n_qubits
    tableau = clifford_tableau.tableau

    # Global features
    global_feats = torch.tensor(
        [
            tableau.sum(),  # Total non-zero entries
            (tableau[:n, :n] != 0).sum(),  # X-part density
            (tableau[:n, n:] != 0).sum(),  # Z-part density
        ],
        dtype=torch.float,
    )

    # Node features: [qubit_idx, destab_x, destab_z, stab_x, stab_z, global_feats]
    node_feats = []
    for q in range(n):
        stab_x = tableau[:n, q]
        stab_z = tableau[:n, q + n]
        destab_x = tableau[n:, q]
        destab_z = tableau[n:, q + n]
        node_feats.append(
            torch.cat(
                [
                    torch.tensor([q]),
                    torch.tensor(stab_x, dtype=torch.float),
                    torch.tensor(stab_z, dtype=torch.float),
                    torch.tensor(destab_x, dtype=torch.float),
                    torch.tensor(destab_z, dtype=torch.float),
                    global_feats,
                ]
            )
        )

    # Edge construction (pruned and simplified)
    edges = []
    for row in range(2 * n):
        x = torch.tensor(tableau[row, :n], dtype=torch.bool)
        z = torch.tensor(tableau[row, n:], dtype=torch.bool)
        involved = torch.where(x | z)[0]
        if len(involved) >= 2:
            pairs = torch.combinations(involved, 2)
            interaction_strength = (x[pairs[:, 0]] * x[pairs[:, 1]]) + (
                z[pairs[:, 0]] * z[pairs[:, 1]]
            )
            mask = interaction_strength > 0  # Prune weak interactions
            pairs = pairs[mask]
            edge_feats = torch.stack(
                [
                    (x[pairs[:, 0]] & z[pairs[:, 1]])
                    | (z[pairs[:, 0]] & x[pairs[:, 1]]),  # XZ or ZX
                    (x[pairs[:, 0]] & x[pairs[:, 1]])
                    | (z[pairs[:, 0]] & z[pairs[:, 1]]),  # XX or ZZ
                    torch.full(
                        (len(pairs),), float(row < n)
                    ),  # Stabilizer or destabilizer
                ],
                dim=1,
            )
            edges.append((pairs, edge_feats))

    # Handle empty edges case
    if not edges:
        # Create self-loops with meaningful features
        dummy_edges = torch.stack([torch.arange(n), torch.arange(n)], dim=0)
        dummy_attr = torch.stack(
            [
                torch.tensor([1.0, 0.0, 0.0])  # Self-loop with identity interaction
                for _ in range(n)
            ]
        )
        return Data(
            x=torch.stack(node_feats), edge_index=dummy_edges, edge_attr=dummy_attr
        )

    edge_index = torch.cat([p.t() for p, _ in edges], dim=1)
    edge_attr = torch.cat([e for _, e in edges], dim=0)

    return Data(x=torch.stack(node_feats), edge_index=edge_index, edge_attr=edge_attr)


class EdgeConvWithAttr(torch.nn.Module):
    def __init__(self, nn, aggr="max"):
        super().__init__()
        self.nn = nn
        self.aggr = aggr

    def forward(self, x, edge_index, edge_attr=None):
        # Collect node features from both ends of edges
        row, col = edge_index
        x_i, x_j = x[row], x[col]

        # If edge attributes are provided, concatenate them
        if edge_attr is not None:
            edge_features = torch.cat([x_i, x_j, edge_attr], dim=1)
        else:
            edge_features = torch.cat([x_i, x_j], dim=1)

        # Apply neural network
        edge_features = self.nn(edge_features)

        # Aggregate by destination node
        out = scatter(edge_features, col, dim=0, dim_size=x.size(0), reduce=self.aggr)

        return out


class PermutationGNN(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim, n_qubits, sinkhorn_iters=20):
        super().__init__()
        self.n_qubits = n_qubits
        self.sinkhorn_iters = sinkhorn_iters

        # Edge-enhanced node processing
        self.edge_conv1 = EdgeConvWithAttr(
            nn.Sequential(
                nn.Linear(2 * node_dim + edge_dim, hidden_dim),
                nn.LeakyReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            ),
            aggr="max",
        )

        self.edge_conv2 = EdgeConvWithAttr(
            nn.Sequential(
                nn.Linear(2 * hidden_dim + edge_dim, hidden_dim),
                nn.LeakyReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            ),
            aggr="mean",
        )

        # Attention-based pooling
        # Use attention aggregator from nn.aggr
        self.attention_aggr = aggr.AttentionalAggregation(
            nn.Sequential(nn.Linear(hidden_dim, 1), nn.Tanh())
        )

        # Permutation prediction head
        self.perm_head = nn.Sequential(
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(2 * hidden_dim, n_qubits**2),
        )

    def sinkhorn(self, logits):
        """Differentiable Sinkhorn normalization"""
        for _ in range(self.sinkhorn_iters):
            logits = logits - torch.logsumexp(logits, dim=2, keepdim=True)
            logits = logits - torch.logsumexp(logits, dim=1, keepdim=True)
        return torch.exp(logits)

    def forward(self, data):
        # Node embeddings
        x = F.leaky_relu(self.edge_conv1(data.x, data.edge_index, data.edge_attr))
        x = F.leaky_relu(self.edge_conv2(x, data.edge_index, data.edge_attr))

        # Attention-weighted node features
        # attn_weights = self.attn_pool(x)
        index = torch.zeros(x.size(0), dtype=torch.long)
        x_pool = self.attention_aggr(
            x, index=index
        )  # Use index parameter instead of batch

        # Permutation matrix logits
        logits = self.perm_head(x_pool).view(-1, self.n_qubits, self.n_qubits)

        # Sinkhorn normalization during training
        if self.training:
            return self.sinkhorn(logits)
        return logits


# class PermutationLoss(nn.Module):
#     def __init__(self, alpha=0.1, temp=0.1):
#         super().__init__()
#         self.alpha = alpha
#         self.temp = temp  # For Sinkhorn sharpness

#     def forward(self, pred, target_perms):
#         # Convert targets to permutation matrices
#         # batch_size = pred.size(0)
#         target_mats = torch.zeros_like(pred)
#         for i, perm in enumerate(target_perms):
#             for r, c in perm:
#                 target_mats[i, r, c] = 1

#         # Temperature-scaled KL divergence
#         loss = F.kl_div(
#             F.log_softmax(pred / self.temp, dim=-1),
#             F.softmax(target_mats / self.temp, dim=-1),
#             reduction="batchmean",
#         )

#         # Doubly stochastic regularization
#         row_sum = pred.sum(dim=2)
#         col_sum = pred.sum(dim=1)
#         reg_loss = F.mse_loss(row_sum, torch.ones_like(row_sum)) + F.mse_loss(
#             col_sum, torch.ones_like(col_sum)
#         )

#         return loss + self.alpha * reg_loss


# class StraightThroughLoss(nn.Module):
class PermutationLoss(nn.Module):
    def __init__(self, temp=0.1, alpha=0.1, sinkhorn_iters=20):
        super().__init__()
        self.temp = temp
        self.alpha = alpha
        self.sinkhorn_iters = sinkhorn_iters  # Initialize the attribute

    def sinkhorn(self, logits):
        """Differentiable Sinkhorn normalization"""
        for _ in range(self.sinkhorn_iters):
            logits = logits - torch.logsumexp(logits, dim=2, keepdim=True)
            logits = logits - torch.logsumexp(logits, dim=1, keepdim=True)
        return torch.exp(logits)

    def forward(self, logits, target_perms):
        # Apply Sinkhorn during forward pass
        sinkhorn_probs = self.sinkhorn(logits / self.temp)

        # Convert targets to matrices
        target_mats = torch.zeros_like(logits)
        for i, perm in enumerate(target_perms):
            for r, c in perm:
                target_mats[i, r, c] = 1

        # KL divergence (prediction vs target)
        kl_loss = F.kl_div(sinkhorn_probs.log(), target_mats, reduction="batchmean")

        # Straight-through gradient: Use Sinkhorn in forward, raw logits in backward
        straight_through_probs = (sinkhorn_probs - logits).detach() + logits
        return kl_loss + self.alpha * F.mse_loss(straight_through_probs, target_mats)


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
        # min_loss = float("inf")  # Track minimum loss
        # for perm in best_perms:
        #     curr_loss = criterion(logits, [perm[0]])
        #     min_loss = min(min_loss, curr_loss)

        # Calculate loss for all permutations
        losses = [criterion(logits, [perm[0]]) for perm in best_perms]
        loss = torch.mean(torch.stack(losses))  # Average loss

        # Backpropagate
        loss.backward()
        # min_loss.backward()

        # Print gradients here
        print("Gradients for edge_conv1:", model.edge_conv1.nn[0].weight.grad)
        print("Gradients for edge_conv2:", model.edge_conv2.nn[0].weight.grad)
        print("Gradients for attention:", model.attention_aggr.gate_nn[0].weight.grad)
        print("Gradients for permutation head:", model.perm_head[0].weight.grad)

        optimizer.step()

        total_loss += loss.item()
        # total_loss += min_loss.item()

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


def pretrain_a_model() -> PermutationGNN:
    n_qubits = 4
    # Include global features (3)
    node_dim = 1 + 4 * n_qubits + 3  # = 20

    # Fix edge_dim - now using 3 features per edge, not 5
    edge_dim = 3

    model = PermutationGNN(
        node_dim=node_dim, edge_dim=edge_dim, hidden_dim=128, n_qubits=n_qubits
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = PermutationLoss(alpha=0.5)

    for epoch in range(20):
        loss = train_model(n_qubits, model, optimizer, criterion)
        print(f"Epoch {epoch+1}, Loss: {loss:.4f}")

    return model


def predict_permutation(model, tableau):
    model.eval()
    with torch.no_grad():
        graph = clifford_tableau_to_graph(tableau)
        logits = model(graph)
        _, col_ind = linear_sum_assignment(-logits.squeeze().cpu().numpy())
        return list(enumerate(col_ind))


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
