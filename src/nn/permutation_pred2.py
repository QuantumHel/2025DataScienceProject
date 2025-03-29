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


class SinkhornKnopp(nn.Module):
    def __init__(self, num_iters=20, epsilon=1e-3):
        super().__init__()
        self.num_iters = num_iters
        self.epsilon = epsilon

    def forward(self, logits):
        # Numerical stability: Ensure logits are in a reasonable range
        logits = torch.clamp(logits, min=-100, max=100)

        # Apply log-space computations for stability
        logits = logits / self.epsilon
        logits = logits - logits.max(dim=-1, keepdim=True)[0]
        M = torch.exp(logits)

        # Prevent zeros in M
        M = M + 1e-8

        for _ in range(self.num_iters):
            # Row normalization
            row_sums = M.sum(dim=-1, keepdim=True)
            M = M / row_sums

            # Column normalization
            col_sums = M.sum(dim=-2, keepdim=True)
            M = M / col_sums

            # Check for NaNs and fix
            if torch.isnan(M).any():
                M = torch.nan_to_num(M, nan=1.0 / (M.size(-1) * M.size(-2)))

        return M


class SequentialPermutationGNN(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim, n_qubits, max_perms=10):
        super().__init__()
        self.n_qubits = n_qubits
        self.max_perms = max_perms

        # Keep existing GNN encoder
        self.embed = nn.Linear(node_dim, hidden_dim)
        self.edge_embed = nn.Linear(edge_dim, hidden_dim)  # Edge embedding
        self.gnn_layers = nn.ModuleList(
            [
                GINEConv(
                    nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, hidden_dim),
                    )
                )
                for _ in range(3)
            ]
        )

        # Add RNN decoder components
        self.rnn = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
        )

        # Keep existing permutation head, but now used at each step
        self.perm_head = nn.Sequential(
            nn.Linear(hidden_dim, 4 * n_qubits**2),
            nn.ReLU(),
            nn.Linear(4 * n_qubits**2, n_qubits**2),
        )

        # Add stop prediction for sequence termination
        self.stop_head = nn.Linear(hidden_dim, 1)

        # Add Sinkhorn-Knopp layer
        self.sinkhorn = SinkhornKnopp()

    def forward(self, data):
        # 1. Encode graph (keep existing code)
        x = self.embed(data.x)
        edge_attr = self.edge_embed(data.edge_attr)

        for conv in self.gnn_layers:
            x = conv(x, data.edge_index, edge_attr)
            x = F.relu(x)

        graph_embed = global_add_pool(x, batch=torch.zeros(x.size(0), dtype=torch.long))

        # 2. Initialize sequence generation
        batch_size = 1  # Assuming single graph input for now
        hidden = torch.zeros(
            2, batch_size, graph_embed.size(1), device=graph_embed.device
        )

        # 3. Generate permutation sequence
        perm_logits_seq = []
        stop_logits_seq = []

        # First input is the graph embedding
        decoder_input = graph_embed.unsqueeze(1)  # [batch, 1, hidden_dim]

        for step in range(self.max_perms):
            # RNN step
            output, hidden = self.rnn(decoder_input, hidden)

            # Predict permutation for this step
            step_perm_logits = self.perm_head(output.squeeze(1))
            step_perm_logits = rearrange(
                step_perm_logits, "b (n m) -> b n m", n=self.n_qubits, m=self.n_qubits
            )

            # Apply Sinkhorn-Knopp to get doubly stochastic matrix
            step_perm_probs = self.sinkhorn(step_perm_logits)

            # Predict stop token
            stop_logit = self.stop_head(output.squeeze(1))

            # Save predictions
            perm_logits_seq.append(step_perm_probs)
            stop_logits_seq.append(stop_logit)

            # Next input is current output (autoregressive)
            decoder_input = output

        # Stack along sequence dimension
        perm_logits = torch.stack(perm_logits_seq, dim=1)  # [batch, seq, n, n]
        stop_logits = torch.cat(stop_logits_seq, dim=1)  # [batch, seq]

        return perm_logits, stop_logits


class PermutationLoss(nn.Module):
    def __init__(self, alpha=0.1):
        super().__init__()
        self.alpha = alpha

    def forward(self, logits, targets):
        # Convert targets to permutation matrices
        target_mats = torch.zeros_like(logits)
        for i, perm in enumerate(targets):
            for idx, (row, col) in enumerate(perm):
                target_mats[i, row, col] = 1

        # Main cross entropy loss
        ce_loss = F.binary_cross_entropy_with_logits(logits, target_mats)

        # Birkhoff polytope regularization
        row_sum = torch.sigmoid(logits).sum(dim=2)
        col_sum = torch.sigmoid(logits).sum(dim=1)
        birk_loss = F.mse_loss(row_sum, torch.ones_like(row_sum)) + F.mse_loss(
            col_sum, torch.ones_like(col_sum)
        )

        return ce_loss + self.alpha * birk_loss


# class ImprovedPermutationLoss(nn.Module):
# class PermutationLoss(nn.Module):
#     def __init__(self, temp=0.1, alpha=0.1, sinkhorn_iters=20):
#         super().__init__()
#         self.temp = temp
#         self.alpha = alpha
#         self.sinkhorn_iters = sinkhorn_iters

#     def sinkhorn(self, logits):
#         """Differentiable Sinkhorn normalization"""
#         for _ in range(self.sinkhorn_iters):
#             logits = logits - torch.logsumexp(logits, dim=2, keepdim=True)
#             logits = logits - torch.logsumexp(logits, dim=1, keepdim=True)
#         return torch.exp(logits)

#     def forward(self, logits, target_perms):
#         # Apply Sinkhorn to raw logits
#         sinkhorn_probs = self.sinkhorn(logits / self.temp)

#         # Convert targets to matrices
#         target_mats = torch.zeros_like(logits)
#         for i, perm in enumerate(target_perms):
#             for r, c in perm:
#                 target_mats[i, r, c] = 1

#         # KL divergence (prediction vs target)
#         kl_loss = F.kl_div(
#             sinkhorn_probs.log(),
#             target_mats,  # No temperature on targets
#             reduction="batchmean",
#         )

#         # Validity loss (encourage sharpness)
#         validity_loss = (1 - sinkhorn_probs.max(dim=2).values.mean()) + (
#             1 - sinkhorn_probs.max(dim=1).values.mean()
#         )

#         return kl_loss + self.alpha * validity_loss


class SequentialPermutationLoss(nn.Module):
    def __init__(self, alpha=0.1, beta=0.5):
        super().__init__()
        self.alpha = alpha  # Birkhoff regularization weight
        self.beta = beta  # Stop token weight

    def forward(self, perm_logits, stop_logits, targets):
        """
        Args:
            perm_logits: [batch, seq_len, n_qubits, n_qubits]
            stop_logits: [batch, seq_len]
            targets: List of lists of permutation tuples
        """
        batch_size = perm_logits.size(0)
        seq_len = perm_logits.size(1)
        device = perm_logits.device

        total_ce_loss = 0
        total_birk_loss = 0
        total_stop_loss = 0

        for b in range(batch_size):
            target_seq = targets[b]
            seq_target_len = len(target_seq)

            # Targets for stop prediction (1 at the end, 0 elsewhere)
            target_stops = torch.zeros(seq_len, device=device)
            if seq_target_len > 0:
                target_stops[seq_target_len - 1] = 1

            # Process each step
            for t in range(seq_len):
                if t < seq_target_len:
                    # Create target matrix for this step
                    target_mat = torch.zeros_like(perm_logits[b, t])
                    for row, col in target_seq[t]:
                        target_mat[row, col] = 1

                    # CE loss
                    step_ce = F.binary_cross_entropy_with_logits(
                        perm_logits[b, t], target_mat
                    )

                    # Birkhoff loss
                    probs = torch.sigmoid(perm_logits[b, t])
                    row_sum = probs.sum(dim=1)
                    col_sum = probs.sum(dim=0)
                    step_birk = F.mse_loss(
                        row_sum, torch.ones_like(row_sum)
                    ) + F.mse_loss(col_sum, torch.ones_like(col_sum))

                    total_ce_loss += step_ce
                    total_birk_loss += step_birk

                # Stop token loss for all positions
                step_stop_loss = F.binary_cross_entropy_with_logits(
                    stop_logits[b, t], target_stops[t]
                )
                total_stop_loss += step_stop_loss

        # Normalize losses
        total_ce_loss /= batch_size
        total_birk_loss /= batch_size
        total_stop_loss /= batch_size

        return (
            total_ce_loss + self.alpha * total_birk_loss + self.beta * total_stop_loss
        )


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


class CircuitSynthesisLoss(nn.Module):
    def __init__(self, depth_weight=0.1):
        super().__init__()
        self.depth_weight = depth_weight  # Weight for depth vs CNOT count

    def forward(self, perm_logits, stop_logits, targets, tableau, topology):
        """
        Direct circuit synthesis loss using actual metrics.
        """
        batch_size = perm_logits.size(0)
        total_loss = 0

        for b in range(batch_size):
            # 1. Decode model predictions into permutation sequence
            pred_sequence = []
            for t in range(perm_logits.size(1)):
                # Stop if stop token is activated
                if t > 0 and torch.sigmoid(stop_logits[b, t - 1]) > 0.5:
                    break

                # Get permutation for this step
                step_perm = constrained_decode_complete(
                    perm_logits[b, t], n_qubits=tableau.n_qubits
                )
                pred_sequence.append(step_perm)

            # 2. Synthesize circuit using predicted permutations
            pred_iter = iter([item for sublist in pred_sequence for item in sublist])

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

            pred_circuit = synthesize_tableau_perm_row_col(
                tableau, topology, pick_pivot_callback=pred_callback
            )
            pred_metrics = collect_circuit_data(pred_circuit)

            # 3. Do the same for target permutations
            target_seq = targets[b]
            target_iter = iter([item for sublist in target_seq for item in sublist])

            # Same fix for target_callback
            def target_callback(G, remaining, remaining_rows, choice_fn=min):
                try:
                    row, col = next(target_iter)
                    # Convert to int if they're tensors
                    if isinstance(row, torch.Tensor):
                        row = row.item()
                    if isinstance(col, torch.Tensor):
                        col = col.item()
                    return int(row), int(col)  # Ensure integers
                except StopIteration:
                    row = choice_fn(remaining_rows)
                    return row, row

            target_circuit = synthesize_tableau_perm_row_col(
                tableau, topology, pick_pivot_callback=target_callback
            )
            target_metrics = collect_circuit_data(target_circuit)

            # 4. Calculate metrics-based scaling factor
            # Modify the metric factor calculation to be more bounded
            cx_ratio = max(0.5, min(2.0, pred_metrics["cx"] / target_metrics["cx"]))
            depth_ratio = max(
                0.5, min(2.0, pred_metrics["depth"] / target_metrics["depth"])
            )

            # Create a metric scaling factor (1.0 is same as target, <1.0 is better, >1.0 is worse)
            metric_factor = cx_ratio + self.depth_weight * depth_ratio

            # 5. Base loss from permutation structure (to maintain gradient flow)
            base_perm_loss = 0
            for t in range(len(pred_sequence)):
                target_mat = torch.ones_like(perm_logits[b, t]) * 0.5
                base_perm_loss += F.binary_cross_entropy_with_logits(
                    perm_logits[b, t], target_mat
                )

            # Add stop token loss
            target_stop = torch.zeros_like(stop_logits[b])
            if len(pred_sequence) > 0:
                target_stop[len(pred_sequence) - 1] = 1
            stop_loss = F.binary_cross_entropy_with_logits(stop_logits[b], target_stop)

            # Scale base loss by metric factor - better circuits get lower loss
            batch_loss = (base_perm_loss + 0.3 * stop_loss) * metric_factor
            total_loss += batch_loss

        return total_loss / batch_size


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
    probs = probs.view(n_qubits, n_qubits).detach().cpu().numpy()

    # Use Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(-probs)

    # Return ALL mappings, including self-loops
    return [(int(r), int(c)) for r, c in zip(row_ind, col_ind)]


def decode_sequential_permutation(model, clifford_tableau, device="cpu"):
    """Decode sequence of permutations from model output"""
    # Convert tableau to graph
    graph = clifford_tableau_to_graph(clifford_tableau)
    data = graph.to(device)

    # Get model predictions
    model.eval()
    with torch.no_grad():
        perm_logits_seq, stop_logits_seq = model(data)

    # Extract permutation sequence
    permutation_sequence = []

    for t in range(perm_logits_seq.size(1)):
        # Check stop condition
        if t > 0 and torch.sigmoid(stop_logits_seq[0, t - 1]) > 0.5:
            break

        # Extract permutation at this step
        step_perm = constrained_decode_complete(
            perm_logits_seq[0, t], n_qubits=clifford_tableau.n_qubits
        )

        if step_perm:  # Only add non-empty permutations
            permutation_sequence.append(step_perm)

    return permutation_sequence


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
        self._graph_cache = {}  # Optional: Cache for graph conversions

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get tableau and best permutations
        tableau, best_perms = self.data[idx]

        # Convert tableau to graph (with optional caching)
        if idx in self._graph_cache:
            graph = self._graph_cache[idx]
        else:
            graph = clifford_tableau_to_graph(tableau)
            self._graph_cache[idx] = graph

        return graph, best_perms


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

        # Check if sequential model
        if isinstance(model, SequentialPermutationGNN):
            # Sequential model returns tuple (perm_logits, stop_logits)
            perm_logits, stop_logits = model(graph)

            # Format target for sequential model - needs sequence structure
            target_seq = []
            # Try all permutations and use the one with lowest loss
            min_loss = float("inf")
            best_target = None
            for perm in best_perms:
                target_seq = [[perm[0]]]
                curr_loss = criterion(perm_logits, stop_logits, [target_seq])
                if curr_loss < min_loss:
                    min_loss = curr_loss
                    best_target = target_seq
            loss = criterion(perm_logits, stop_logits, [best_target])
        else:
            # Handle non-sequential model
            logits = model(graph)
            min_loss = float("inf")
            for perm in best_perms:
                curr_loss = criterion(logits, [perm[0]])
                min_loss = min(min_loss, curr_loss)
            loss = min_loss

        # Backpropagate
        loss.backward()

        # After loss.backward() but before optimizer.step()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Print gradients here
        print("Gradients for first GNN layer:", model.gnn_layers[0].lin.weight.grad)
        print("Gradients for embedding layer:", model.embed.weight.grad)

        optimizer.step()

        total_loss += loss.item()

    return total_loss / batch_size


def train_model_from_dataset(
    model, optimizer, criterion, dataset, batch_size=64, accumulation_steps=16
):
    model.train()
    total_loss = 0
    total_batches = 0

    topology = Topology.complete(dataset.n_qubits)
    dataloader = PyGDataLoader(dataset, batch_size=1, shuffle=True)

    # Process all graphs with gradient accumulation
    optimizer.zero_grad()
    accumulated_samples = 0

    for i, (graph, best_perms) in enumerate(dataloader):
        if accumulated_samples >= batch_size:
            break

        # Get tableau from dataset
        tableau = dataset.data[i][0]

        # Forward pass
        perm_logits, stop_logits = model(graph)

        # Format target and calculate loss
        target_seq = []
        for perm_tuple in best_perms:
            target_seq.append([perm_tuple[0]])

        loss = criterion(perm_logits, stop_logits, target_seq, tableau, topology)
        loss = loss / accumulation_steps  # Scale for accumulation

        # Backward pass
        loss.backward()
        accumulated_samples += 1

        # Update parameters only after accumulating gradients
        if accumulated_samples % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * accumulation_steps
        total_batches += 1

    # Handle any remaining gradients
    if accumulated_samples % accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad()

    return total_loss / total_batches


def pretrain_a_model() -> SequentialPermutationGNN:
    n_qubits = 4
    # model = PermutationConstrainedGNN(
    #     node_dim=1, edge_dim=6, hidden_dim=128, n_qubits=n_qubits
    # )
    model = SequentialPermutationGNN(
        node_dim=1, edge_dim=6, hidden_dim=128, n_qubits=n_qubits
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # criterion = PermutationLoss(alpha=0.5)
    criterion = SequentialPermutationLoss(alpha=0.5, beta=0.1)

    # Start with simpler circuits and gradually increase complexity
    gate_schedule = [50, 100, 200, 500, 1000]
    for epoch in range(10):
        # Select gate count based on training progress
        gates = gate_schedule[min(epoch // 4, len(gate_schedule) - 1)]
        loss = train_model(n_qubits, model, optimizer, criterion, num_gates=gates)
        print(f"Epoch {epoch+1}, Gates: {gates}, Loss: {loss:.4f}")

    return model


def pretrain_a_model_from_file(data_file="training_data_perm.pkl", max_samples=None):
    n_qubits = 4
    model = SequentialPermutationGNN(
        node_dim=1, edge_dim=6, hidden_dim=256, n_qubits=n_qubits
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3, weight_decay=1e-5)

    # Add scheduler here, after optimizer creation
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=2, factor=0.5, min_lr=1e-5
    )

    # criterion = SequentialPermutationLoss(alpha=0.3, beta=0.3)
    criterion = CircuitSynthesisLoss(depth_weight=0.1)

    # Load dataset with subset option
    dataset = TableauPermutationDataset(
        data_file, n_qubits=n_qubits, max_samples=max_samples
    )

    # Training loop
    num_epochs = 20
    for epoch in range(num_epochs):
        loss = train_model_from_dataset(model, optimizer, criterion, dataset)
        # Print learning rate
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch+1}, Loss: {loss:.4f}, LR: {current_lr}")
        # Add scheduler step here, after epoch loss is calculated
        scheduler.step(loss)

    return model


# def predict_permutation(model, clifford_tableau, device="cpu"):
#     """
#     Predicts valid permutation tuples for a Clifford tableau using the trained model
#     Returns: List of (control, target) tuples
#     """
#     # 1. Convert tableau to graph
#     graph = clifford_tableau_to_graph(clifford_tableau)

#     # 2. Prepare input for model
#     data = graph.to(device)

#     # 3. Model forward pass
#     model.eval()
#     with torch.no_grad():
#         raw_output = model(data)

#     # 4. Constrained decoding with validity checks
#     print(raw_output)
#     permutation = constrained_decode_complete(
#         raw_output, n_qubits=clifford_tableau.n_qubits
#     )

#     # assert validate_permutation(permutation, clifford_tableau.n_qubits), "Invalid permutation generated"
#     # print(f"Valid permutation: {permutation}")
#     # # Example output: [(0, 1), (1, 0), (2, 3), (3, 2)]

#     # 5. Post-processing to ensure physical validity
#     # return enforce_permutation_constraints(permutation)
#     return permutation


def predict_permutation(model, clifford_tableau, device="cpu"):
    """
    Predicts valid permutation tuples for a Clifford tableau using the trained model.
    Returns just a single permutation rather than a sequence.
    """
    # 1. Convert tableau to graph
    graph = clifford_tableau_to_graph(clifford_tableau)
    data = graph.to(device)

    # 2. Model forward pass
    model.eval()
    with torch.no_grad():
        output = model(data)

    # 3. Handle different model types
    if isinstance(model, SequentialPermutationGNN):
        # Get just the first permutation from sequence
        perm_logits_seq, _ = output
        first_perm = constrained_decode_complete(
            perm_logits_seq[0, 0], n_qubits=clifford_tableau.n_qubits
        )
        return [first_perm]
    else:
        # For single-permutation models
        permutation = constrained_decode_complete(
            output, n_qubits=clifford_tableau.n_qubits
        )
        return [permutation]


def enforce_permutation_constraints(permutation):
    """Ensures final permutation validity"""
    # 1. Remove duplicates
    seen = set()
    return [
        tuple(pair)
        for pair in permutation
        if not (tuple(pair) in seen or seen.add(tuple(pair)))
    ]


# # Example usage
# n_qubits = 4
# model = pretrain_a_model_from_file("training_data_perm.pkl", max_samples=None)
# tableau = CliffordTableau(4)
# tableau = tableau_from_circuit(tableau, random_hscx_circuit(nr_qubits=4, nr_gates=100))
# pred = predict_permutation(model, tableau)
# print(pred)
