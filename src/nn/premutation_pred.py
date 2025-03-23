import numpy as np
import torch
from torch_geometric.data import Data
from src.utils import tableau_from_circuit, random_hscx_circuit
from pauliopt.clifford.tableau import CliffordTableau
from src.nn.brute_force_data import get_best_cnots

from pauliopt.topologies import Topology
from torch import nn
from torch_geometric.nn import GCNConv
from torch_geometric.utils import from_networkx
import warnings
import torch.nn.functional as F


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


# Example usage
n_qubits = 2
clifford_tableau = CliffordTableau(n_qubits)
clifford_tableau.append_h(0)
clifford_tableau.append_cnot(0, 1)
graph = clifford_tableau_to_graph(clifford_tableau)
print(graph)


# # Example output
n_qubits = 4
nr_gates = 100
circuit = random_hscx_circuit(nr_qubits=n_qubits, nr_gates=nr_gates)
clifford_tableau = CliffordTableau(n_qubits)
clifford_tableau = tableau_from_circuit(clifford_tableau, circuit)
print(clifford_tableau.signs)
print(clifford_tableau.tableau)
# there is a typo in the Pauli lib, where .clifford -> .tableau
# graph = clifford_tableau_to_graph(clifford_tableau.tableau, clifford_tableau.signs)
graph = clifford_tableau_to_graph(clifford_tableau)
print(graph)


# Assuming the provided clifford_tableau_to_graph function is already defined
# as:
# def clifford_tableau_to_graph(clifford_tableau):
#    ... (provided implementation)


# Define the Graph Neural Network (GNN) layer
class GNNLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GNNLayer, self).__init__()
        self.conv1 = GCNConv(in_channels, out_channels)
        self.conv2 = GCNConv(out_channels, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return x


# Define the Transformer-based model for Sequence Prediction
class GraphToSequenceTransformer(nn.Module):
    def __init__(
        self, gnn_out_channels, transformer_dim, num_heads, num_layers, n_qubits
    ):
        super(GraphToSequenceTransformer, self).__init__()

        # Initialize the GNN layers
        self.gnn = GNNLayer(
            1, gnn_out_channels
        )  # 2 input channels (qubit index and sign)

        # Initialize the Transformer model
        self.transformer = nn.Transformer(
            d_model=transformer_dim,  # Model dimension
            nhead=num_heads,  # Number of attention heads
            num_encoder_layers=num_layers,  # Number of transformer encoder layers
            num_decoder_layers=num_layers,  # Number of transformer decoder layers
            batch_first=True,  # Batch dimension first
        )

        self.fc_out = nn.Linear(
            gnn_out_channels, 2
        )  # Output the 2D permutation tuple (row, col)

    def forward(self, data):
        # Step 1: Process the graph using GNN
        gnn_out = self.gnn(data)

        # Step 2: Prepare the graph output for Transformer (flatten the node features)
        # Convert the GNN output to a suitable sequence format (node-level feature to sequence)
        gnn_out = gnn_out.view(-1, gnn_out.shape[1]).unsqueeze(0)  # Batch size of 1

        # Step 3: Generate the ordered permutation using Transformer
        transformer_out = self.transformer(
            gnn_out, gnn_out
        )  # For simplicity, using same input for both encoder and decoder

        # Step 4: Project the output of Transformer to the permutation pair (row, col) predictions
        output = self.fc_out(transformer_out)

        return output


def permutation_penalty(predictions, num_tuples):
    """
    Penalizes invalid permutations by adding a term to the loss function.
    - Penalizes duplicate entries.
    - Penalizes missing entries.
    - Penalizes out-of-range indices.
    """
    batch_size, num_permutations, _ = predictions.shape

    # Convert predictions to discrete indices (round to nearest integer)
    rounded_preds = predictions.round().long()

    # Print rounded_preds for debugging
    print(f"rounded_preds: {rounded_preds}")

    # Ensure indices are within bounds
    penalty_out_of_range = (rounded_preds < 0).any(dim=-1) | (
        rounded_preds >= num_tuples
    ).any(dim=-1)
    penalty_unique = torch.zeros(
        batch_size, num_permutations, device=predictions.device
    )

    # Penalizing out-of-range indices
    if penalty_out_of_range.any():
        penalty_unique += (
            penalty_out_of_range.long() * 10
        )  # large penalty for out-of-range

    # Check for uniqueness within each permutation tuple (row, col)
    unique_counts = torch.zeros((batch_size, num_tuples), device=predictions.device)
    for i in range(num_permutations):
        # Check for invalid indices before scatter_add_
        valid_indices = (rounded_preds[:, i, :] >= 0) & (
            rounded_preds[:, i, :] < num_tuples
        )
        if not valid_indices.all():
            print(
                f"Invalid indices found in rounded_preds[:, {i}, :]: {rounded_preds[:, i, :]}"
            )
            continue

        unique_counts.scatter_add_(
            1,
            rounded_preds[:, i, :],
            torch.ones_like(rounded_preds[:, i, :], dtype=torch.float),
        )

    # Penalize deviations from a unique valid permutation
    penalty_unique += ((unique_counts - 1) ** 2).sum(dim=1)

    # Combine all penalties (unique + out-of-range)
    penalty = penalty_unique.mean()
    return penalty


def train_model(
    n_qubits, nr_gates, batch_size, model, optimizer, criterion, penalty_weight=0.1
):
    model.train()  # Ensure the model is in training mode
    total_loss = 0.0  # To accumulate the total loss over all batches

    topo = Topology.complete(n_qubits)

    for _ in range(batch_size):  # Loop over the batch size
        # Generate a random circuit and convert it to a Clifford tableau
        circuit = random_hscx_circuit(nr_qubits=n_qubits, nr_gates=nr_gates)
        clifford_tableau = CliffordTableau(n_qubits)
        clifford_tableau = tableau_from_circuit(clifford_tableau, circuit)

        # Prepare labels using get_best_cnots (for supervised training)
        possible_labels = get_best_cnots(
            clifford_tableau, topo
        )  # List of possible label sets
        # possible_labels = [
        #     ([(2, 3), (0, 2), (1, 0), (3, 1)], 3),
        #     ([(0, 2), (2, 0), (1, 1), (3, 3)], 3),
        # ]  # Example labels

        # Extract the first element of each tuple in possible_labels
        possible_labels = [label for label, _ in possible_labels]

        # Convert each label to a tensor
        possible_labels = [
            torch.tensor(label, dtype=torch.float) for label in possible_labels
        ]

        # Forward pass: Feed the graph into the model
        optimizer.zero_grad()
        data = clifford_tableau_to_graph(
            clifford_tableau
        )  # Convert the tableau to graph
        predictions = model(data)  # The model predicts one output

        # Compute the main loss (e.g., MSE, cosine similarity, etc.)
        losses = torch.stack(
            [criterion(predictions, label.unsqueeze(0)) for label in possible_labels]
        )
        best_loss, best_idx = torch.min(losses, dim=0)

        # Compute permutation penalty
        penalty = permutation_penalty(predictions, n_qubits)

        # Total loss: Main loss + penalty
        total_loss_batch = best_loss + penalty_weight * penalty

        # Backpropagation
        total_loss_batch.backward()
        optimizer.step()

        # Accumulate loss
        total_loss += total_loss_batch.item()

    # Return the average loss
    return total_loss / batch_size


# Catch and handle overflow warnings
warnings.filterwarnings("error", category=RuntimeWarning)

try:
    # Your code that may produce overflow warnings
    n_qubits = 2
    clifford_tableau = CliffordTableau(n_qubits)
    clifford_tableau.append_h(0)
    clifford_tableau.append_cnot(0, 1)
    print(clifford_tableau.signs)
    print(clifford_tableau.tableau)
except RuntimeWarning as e:
    print(f"RuntimeWarning caught: {e}")
    # Handle the overflow warning (e.g., by using a different data type or adjusting the values)


# Initialize the model, optimizer, and loss function
model = GraphToSequenceTransformer(
    gnn_out_channels=128, transformer_dim=128, num_heads=4, num_layers=3, n_qubits=4
)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.MSELoss()  # You can also try using cosine similarity here


def cosine_similarity_loss(prediction, label):
    return 1 - F.cosine_similarity(prediction, label, dim=-1).mean()


# Parameters for training
n_qubits = 4
n_gates = 100
batch_size = 10  # Number of examples per batch

# Train the model with the training function
loss = train_model(
    n_qubits, n_gates, batch_size, model, optimizer, cosine_similarity_loss
)
print(f"Training loss: {loss:.4f}")

# Test the model with the test function
