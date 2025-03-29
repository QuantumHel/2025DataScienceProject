import pickle
from src.utils import tableau_from_circuit, random_hscx_circuit
from pauliopt.clifford.tableau import CliffordTableau
from src.nn.brute_force_data import get_best_cnots
from pauliopt.topologies import Topology
import numpy as np
import warnings
from tqdm import tqdm  # Import tqdm for progress bar

# Suppress all overflow warnings globally
np.seterr(over="ignore")

# Suppress FutureWarning
warnings.simplefilter(action="ignore", category=FutureWarning)


def generate_data():
    # Configuration
    n_qubits = 4  # Specify the number of qubits (e.g., 4)
    batch_size = 32
    num_epochs = 100
    total_data_points = batch_size * num_epochs  # 3200
    filename = "training_data_perm.pkl"

    # Generate and save data
    data = []

    # Create a progress bar
    with tqdm(total=total_data_points, desc="Generating data", unit="circuits") as pbar:
        for i in range(total_data_points):
            # Generate circuit with minimum gates
            circuit = random_hscx_circuit(nr_qubits=n_qubits, nr_gates=1000)
            tableau = tableau_from_circuit(CliffordTableau(n_qubits), circuit)
            best_perms = get_best_cnots(tableau, Topology.complete(n_qubits))

            # Store as (input_tableau, target_perms) for various tableau-to-graph implementations
            data.append((tableau, best_perms))

            # Update progress bar
            pbar.update(1)

            # Still print epoch completion
            if (i + 1) % batch_size == 0:
                epoch = (i + 1) // batch_size
                pbar.set_postfix({"Epochs": f"{epoch}/{num_epochs}"})

    # Save to file
    print(f"Saving {len(data)} data points to {filename}...")
    with open(filename, "wb") as f:
        pickle.dump(data, f)

    print(f"Generated {len(data)} data points saved to {filename}")


def main():
    generate_data()
    return 0


if __name__ == "__main__":
    main()
