"""Code to possibly evaluate the NN training approach. Currently, this only compares our and the CNN compilation. """
import warnings
from typing import List

import torch
import numpy as np
import pandas as pd
from pauliopt.circuits import Circuit
from pauliopt.clifford.tableau import CliffordTableau
from pauliopt.clifford.tableau_synthesis import synthesize_tableau_perm_row_col
from pauliopt.topologies import Topology

from src.nn.brute_force_data import get_best_cnots
from src.utils import random_hscx_circuit, tableau_from_circuit

from src.nn.best_qubit_model import BestQubitModel

# Suppress all overflow warnings globally
np.seterr(over='ignore')

# Suppress FutureWarning
warnings.simplefilter(action='ignore', category=FutureWarning)


def collect_circuit_data(circuit: Circuit) -> dict:
    circuit.final_permutation = None
    ops = circuit.to_qiskit().count_ops()
    return {
        "num_qubits": circuit.n_qubits,
        "h": ops.get("h", 0),
        "s": ops.get("s", 0),
        "cx": ops.get("cx", 0),
        "depth": circuit.to_qiskit().depth()
    }


def our_compilation(circuit: Circuit, topology: Topology, n_rep: int):
    """
    Compilation from previous paper as a baseline.

    :param circuit:
    :param topology:
    :param n_rep:
    :return:
    """
    clifford_tableau = CliffordTableau(circuit.n_qubits)
    clifford_tableau = tableau_from_circuit(clifford_tableau, circuit)

    circ_out = synthesize_tableau_perm_row_col(clifford_tableau, topology)
    return {"n_rep": n_rep} | collect_circuit_data(circ_out) | {"method": "normal_heuristic"}


def random_compilation(circuit: Circuit, topology: Topology, n_rep: int):
    """
    Brute force compilation of the circuit (may be slow for >=4 qubits!)
    :param circuit:
    :param topology:
    :param n_rep:
    :return:
    """
    clifford_tableau = CliffordTableau(circuit.n_qubits)
    clifford_tableau = tableau_from_circuit(clifford_tableau, circuit)

    def pick_pivot_callback(G, remaining: "CliffordTableau", remaining_rows: List[int], choice_fn=min):
        row = np.random.choice(remaining_rows)
        col = row
        return row, col

    circ_out = synthesize_tableau_perm_row_col(clifford_tableau, topology, pick_pivot_callback=pick_pivot_callback)
    return {"n_rep": n_rep} | collect_circuit_data(circ_out) | {"method": "random"}

def optimal_compilation(circuit: Circuit, topology: Topology, n_rep: int):
    """
    Brute force compilation of the circuit (may be slow for >=4 qubits!)
    :param circuit:
    :param topology:
    :param n_rep:
    :return:
    """
    clifford_tableau = CliffordTableau(circuit.n_qubits)
    clifford_tableau = tableau_from_circuit(clifford_tableau, circuit)

    best_permutation, score = get_best_cnots(clifford_tableau.inverse().inverse(), topology)[0]
    best_permutation = iter(best_permutation)

    def pick_pivot_callback(G, remaining: "CliffordTableau", remaining_rows: List[int], choice_fn=min):
        row, col = next(best_permutation)
        return row, col

    circ_out = synthesize_tableau_perm_row_col(clifford_tableau, topology, pick_pivot_callback=pick_pivot_callback)
    return {"n_rep": n_rep} | collect_circuit_data(circ_out) | {"method": "optimum"}


def nn_compilation(circuit: Circuit, topology: Topology, n_rep: int):
    """
    Compilation using the trained neural network to infer the best pivot qubit.
    """
    # Load the trained model weights
    model = BestQubitModel(n_size=circuit.n_qubits, hidden_layers=4, hidden_size=128, dropout_rate=0.3)
    model.load_state_dict(torch.load("best_qubit_model_weights.pt", map_location=torch.device('cpu')))
    model.eval()

    # Prepare the Clifford tableau from the circuit
    clifford_tableau = CliffordTableau(circuit.n_qubits)
    clifford_tableau = tableau_from_circuit(clifford_tableau, circuit)

    # Ensure matrices are numpy arrays with the expected shape (n_qubits x n_qubits)
    n = circuit.n_qubits  # 4 for example
    # Reshape x_mat and z_mat to (n, n)
    x_mat = np.array(clifford_tableau.x_matrix).reshape(n, n)
    print(clifford_tableau.x_matrix)
    z_mat = np.array(clifford_tableau.z_matrix).reshape(n, n)

    # Create an input tensor of shape [1, 3, n, n]
    input_tensor = torch.zeros(1, 3, n, n, dtype=torch.float32)
    input_tensor[0, 0] = torch.tensor(x_mat, dtype=torch.float32)
    input_tensor[0, 1] = torch.tensor(z_mat, dtype=torch.float32)
    # The third channel remains zero (or filled as needed)

    with torch.no_grad():
        output = model(input_tensor)
        # Flatten output scores to pick the best pivot index:
        pivot_index = int(output.view(output.size(0), -1).argmax(dim=1).item())

    def pick_pivot_callback(G, remaining: "CliffordTableau", remaining_rows: List[int], choice_fn=min):
        if pivot_index in remaining_rows:
            return pivot_index, pivot_index
        else:
            row = np.random.choice(remaining_rows)
            return row, row

    circ_out = synthesize_tableau_perm_row_col(clifford_tableau, topology, pick_pivot_callback=pick_pivot_callback)
    return {"n_rep": n_rep} | collect_circuit_data(circ_out) | {"method": "nn"}



def main(n_qubits: int = 4, nr_gates: int = 1000):
    """
    Execute a single experiment with random clifford circuits and store the respective gate count into a dataframe
    :param n_qubits:
    :param nr_gates:
    :return:
    """

    df = pd.DataFrame(columns=["n_rep", "num_qubits", "method", "h", "s", "cx", "depth"])
    topo = Topology.complete(n_qubits)
    for i in range(20):
        print(i)
        circuit = random_hscx_circuit(nr_qubits=n_qubits, nr_gates=nr_gates)

        # Our compilation e.g. the baseline from the paper
        df_dictionary = pd.DataFrame([our_compilation(circuit.copy(), topo, i)])
        df = pd.concat([df, df_dictionary], ignore_index=True)
        print("Min", df_dictionary["cx"])

        # Optimal compilation
        df_dictionary = pd.DataFrame([optimal_compilation(circuit.copy(), topo, i)])
        df = pd.concat([df, df_dictionary], ignore_index=True)
        print("OPTIMUM", df_dictionary["cx"])

        # Random compilation
        df_dictionary = pd.DataFrame([random_compilation(circuit.copy(), topo, i)])
        df = pd.concat([df, df_dictionary], ignore_index=True)
        print("Random", df_dictionary["cx"])

        # Group's first ANN compilation
        df_dictionary = pd.DataFrame([nn_compilation(circuit.copy(), topo, i)])
        df = pd.concat([df, df_dictionary], ignore_index=True)
        print("NN", df_dictionary["cx"])

    df.to_csv("test_clifford_synthesis.csv", index=False)
    print(df.groupby("method").mean())


if __name__ == "__main__":
    main()