"""Code to possibly evaluate the NN training approach. Currently, this only compares our and the CNN compilation. """
import warnings
from typing import List

import numpy as np
import pandas as pd
from pauliopt.circuits import Circuit
from pauliopt.clifford.tableau import CliffordTableau
from pauliopt.clifford.tableau_synthesis import synthesize_tableau_perm_row_col
from pauliopt.topologies import Topology

from src.nn.brute_force_data import get_best_cnots
from src.utils import random_hscx_circuit, tableau_from_circuit

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

        df_dictionary = pd.DataFrame([our_compilation(circuit.copy(), topo, i)])
        df = pd.concat([df, df_dictionary], ignore_index=True)
        print("Min", df_dictionary["cx"])
        df_dictionary = pd.DataFrame([optimal_compilation(circuit.copy(), topo, i)])
        df = pd.concat([df, df_dictionary], ignore_index=True)
        print("OPTIMUM", df_dictionary["cx"])

        df_dictionary = pd.DataFrame([random_compilation(circuit.copy(), topo, i)])
        df = pd.concat([df, df_dictionary], ignore_index=True)
        print("Random", df_dictionary["cx"])

    df.to_csv("test_clifford_synthesis.csv", index=False)
    print(df.groupby("method").mean())


if __name__ == "__main__":
    main()
