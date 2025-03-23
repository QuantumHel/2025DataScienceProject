import pickle
from src.utils import tableau_from_circuit, random_hscx_circuit
from pauliopt.clifford.tableau import CliffordTableau
from src.nn.brute_force_data import get_best_cnots
from pauliopt.topologies import Topology
from src.nn.permutation_pred3 import clifford_tableau_to_graph

# Configuration
n_qubits = 4  # Specify the number of qubits (e.g., 5)
batch_size = 32
num_epochs = 100
total_data_points = batch_size * num_epochs  # 3200
filename = "training_data_perm.pkl"

# Generate and save data
data = []

for _ in range(total_data_points):
    # Generate circuit with minimum gates
    circuit = random_hscx_circuit(nr_qubits=n_qubits, nr_gates=1000)
    tableau = tableau_from_circuit(CliffordTableau(n_qubits), circuit)
    graph = clifford_tableau_to_graph(tableau)
    best_perms = get_best_cnots(tableau, Topology.complete(n_qubits))

    # Store as (input_graph, target_perms)
    data.append((graph, best_perms))
    if len(data) % 32 == 0:
        print(f"Generated {len(data) / 32} epochs data points")

# Save to file
with open(filename, "wb") as f:
    pickle.dump(data, f)

print(f"Generated {len(data)} data points saved to {filename}")
