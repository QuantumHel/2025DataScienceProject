"""Code to possibly evaluate the NN training approach. Currently, this only compares our and the CNN compilation."""

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

# from src.nn.permutation_pred3 import predict_permutation, pretrain_a_model, FlexibleGNN

# from src.nn.permutation_pred2 import (
#     pretrain_a_model,
#     predict_permutation,
#     PermutationConstrainedGNN,
# )

from src.nn.permutation_pred2_optm import predict_permutation, pretrain_a_model, PermutationGNN

# Suppress all overflow warnings globally
np.seterr(over="ignore")

# Suppress FutureWarning
warnings.simplefilter(action="ignore", category=FutureWarning)


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


# "number of repetitions", "repetition index"???
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
    return (
        {"n_rep": n_rep}
        | collect_circuit_data(circ_out)
        | {"method": "normal_heuristic"}
    )


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

    def pick_pivot_callback(
        G, remaining: "CliffordTableau", remaining_rows: List[int], choice_fn=min
    ):
        row = np.random.choice(remaining_rows)
        col = row
        return row, col

    circ_out = synthesize_tableau_perm_row_col(
        clifford_tableau, topology, pick_pivot_callback=pick_pivot_callback
    )
    return {"n_rep": n_rep} | collect_circuit_data(circ_out) | {"method": "random"}


# Bruteforce compilation, of course it is the optimal...
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

    best_permutation, score = get_best_cnots(
        clifford_tableau.inverse().inverse(), topology
    )[0]

    # print_perm = get_best_cnots(clifford_tableau.inverse().inverse(), topology)
    # print(f"best_permutation: {print_perm}")

    best_permutation = iter(best_permutation)

    def pick_pivot_callback(
        G, remaining: "CliffordTableau", remaining_rows: List[int], choice_fn=min
    ):
        row, col = next(best_permutation)
        return row, col

    circ_out = synthesize_tableau_perm_row_col(
        clifford_tableau, topology, pick_pivot_callback=pick_pivot_callback
    )
    return {"n_rep": n_rep} | collect_circuit_data(circ_out) | {"method": "optimum"}


def dummy_perm_compilation(
    circuit: Circuit, topology: Topology, n_rep: int, model: PermutationGNN
):
    clifford_tableau = CliffordTableau(circuit.n_qubits)
    clifford_tableau = tableau_from_circuit(clifford_tableau, circuit)
    best_permutation = predict_permutation(model, clifford_tableau)
    best_permutation = iter(best_permutation)

    def pick_pivot_callback(
        G, remaining: "CliffordTableau", remaining_rows: List[int], choice_fn=min
    ):
        row, col = next(best_permutation)
        return row, col

    circ_out = synthesize_tableau_perm_row_col(
        clifford_tableau, topology, pick_pivot_callback=pick_pivot_callback
    )
    return {"n_rep": n_rep} | collect_circuit_data(circ_out) | {"method": "dummy-perm"}


def main(n_qubits: int = 4, nr_gates: int = 1000):
    """
    Execute a single experiment with random clifford circuits and store the respective gate count into a dataframe
    :param n_qubits:
    :param nr_gates:
    :return:
    """

    # Pre-train a model
    model = pretrain_a_model()

    df = pd.DataFrame(
        columns=["n_rep", "num_qubits", "method", "h", "s", "cx", "depth"]
    )
    topo = Topology.complete(n_qubits)
    for i in range(50):
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

        # # Group's first ANN compilation
        # df_dictionary = pd.DataFrame([nn_compilation(circuit.copy(), topo, i)])
        # df = pd.concat([df, df_dictionary], ignore_index=True)
        # print("NN", df_dictionary["cx"])

        # Dummy_perm compilation
        df_dictionary = pd.DataFrame(
            [dummy_perm_compilation(circuit.copy(), topo, i, model)]
        )
        df = pd.concat([df, df_dictionary], ignore_index=True)
        print("Dummy-perm", df_dictionary["cx"])

    # Convert the cx column to a numerical type
    df["cx"] = pd.to_numeric(df["cx"])

    df.to_csv("test_clifford_synthesis.csv", index=False)
    # Question: what should be the comparision metric? Mean, median, std, mse, etc.?
    print(df.groupby("method").mean())

    # Is the difference just luck?
    from scipy.stats import ttest_ind

    nn_cx_values = df[df["method"] == "nn"]["cx"]
    random_cx_values = df[df["method"] == "random"]["cx"]
    t_stat, p_value = ttest_ind(nn_cx_values, random_cx_values)

    print(f"T-test results: t-statistic = {t_stat}, p-value = {p_value}")
    if p_value < 0.05:
        print(
            "The difference in cx values between nn and random is statistically significant (p < 0.05)."
        )
    else:
        print(
            "The difference in cx values between nn and random is not statistically significant (p >= 0.05)."
        )


if __name__ == "__main__":
    main()
