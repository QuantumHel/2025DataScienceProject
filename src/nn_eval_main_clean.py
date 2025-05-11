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


from src.nn.permutation_clean import (
    pretrain_a_model_from_file,
    OrderedPermutationTransformer,
    TableauPermutationDataset,
    supervised_cx_fine_tune,
    predict_permutation_gumbel,
    entropy_guided_search,
    ensemble_predict_permutation,
    curriculum_train,
)

import torch
import matplotlib.pyplot as plt
import seaborn as sns


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
    circuit: Circuit,
    topology: Topology,
    n_rep: int,
    model: OrderedPermutationTransformer,
    device,
):
    """
    Compilation using the neural network approach.
    """
    clifford_tableau = CliffordTableau(circuit.n_qubits)
    clifford_tableau = tableau_from_circuit(clifford_tableau, circuit)
    # best_permutation = predict_permutation_gumbel(model, clifford_tableau, device)
    best_permutation = entropy_guided_search(model, clifford_tableau, device)
    # best_permutation = ensemble_predict_permutation(model, clifford_tableau, device)
    best_permutation = iter(best_permutation[0])

    def pick_pivot_callback(
        G, remaining: "CliffordTableau", remaining_rows: List[int], choice_fn=min
    ):
        row, col = next(best_permutation)
        return row, col

    circ_out = synthesize_tableau_perm_row_col(
        clifford_tableau, topology, pick_pivot_callback=pick_pivot_callback
    )
    return {"n_rep": n_rep} | collect_circuit_data(circ_out) | {"method": "dummy-perm"}


def visualize_optimality_gaps(df):
    """
    Plot the gap between each method and the optimum solution.
    Args:
        df (pd.DataFrame): DataFrame containing the results of the experiments.
    """
    plt.figure(figsize=(14, 8))
    sns.set_style("whitegrid")

    # Extract the methods we care about
    methods = ["normal_heuristic", "dummy-perm", "combined_min", "optimum"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    labels = ["Standard Heuristic", "Neural Network", "Combined", "Optimum"]

    # Plot main trends
    for method, color, label in zip(methods, colors, labels):
        method_df = df[df["method"] == method].sort_values("n_rep")
        rolling_avg = method_df["cx"].rolling(window=50, min_periods=1).mean()

        # Plot rolling average
        plt.plot(method_df["n_rep"], rolling_avg, color=color, linewidth=3, label=label)

    # Add title and labels
    plt.title("CX Count Comparison with Optimum", fontsize=16)
    plt.xlabel("Circuit Evaluation Index", fontsize=14)
    plt.ylabel("Number of CX Gates", fontsize=14)
    plt.legend(fontsize=12)

    # Add average gap statistics in text box
    gaps_text = "Average Gap to Optimum:\n"
    opt_mean = df[df["method"] == "optimum"]["cx"].mean()

    for method, label in zip(methods, labels):
        if method != "optimum":
            method_mean = df[df["method"] == method]["cx"].mean()
            gap = method_mean - opt_mean
            gap_percent = (gap / opt_mean) * 100
            gaps_text += f"{label}: +{gap:.2f} gates (+{gap_percent:.1f}%)\n"

    plt.figtext(
        0.02, 0.02, gaps_text, fontsize=12, bbox=dict(facecolor="white", alpha=0.9)
    )

    plt.tight_layout()
    plt.savefig("optimality_gap_comparison.png", dpi=300)
    plt.show()


def main(n_qubits: int = 4, nr_gates: int = 1000):
    """
    Execute a single experiment with random clifford circuits and store the respective gate count into a dataframe
    :param n_qubits:
    :param nr_gates:
    :return:
    """

    device = "cpu"
    cnt_eval = 10

    # # If want to train a new model, uncomment the following line.
    # model = pretrain_a_model_from_file("nn/training_data_perm.pkl", None, 50, device)

    # # If want to use curriculum training, uncomment the following line.
    # model = curriculum_train(
    #     "nn/training_data_perm.pkl", max_samples=None, epochs_per_stage=25
    # )

    # # If want to use supervised learning fine-tuning, uncomment the following lines.
    # sl_dataset = TableauPermutationDataset(
    #     "nn/training_data_perm_4_qubit.pkl", max_samples=320
    # )
    # sl_model = supervised_cx_fine_tune(model, sl_dataset, epochs=50, device="cpu")

    # If want to use pre-trained model, uncomment the following lines.
    # The 4-qubit model `ordered_permutation_model.pth` is ready to use.
    checkpoint = torch.load("ordered_permutation_model.pth", map_location=device)
    print(type(checkpoint))
    if isinstance(checkpoint, dict):
        print(checkpoint.keys())
    model = OrderedPermutationTransformer(n_qubits=n_qubits, dim=256, num_layers=12)
    model.load_state_dict(checkpoint)
    model.eval()

    df = pd.DataFrame(
        columns=["n_rep", "num_qubits", "method", "h", "s", "cx", "depth"]
    )
    topo = Topology.complete(n_qubits)
    for i in range(cnt_eval):
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
            [dummy_perm_compilation(circuit.copy(), topo, i, model, device)]
        )
        # df_dictionary = pd.DataFrame(
        #     [dummy_perm_compilation(circuit.copy(), topo, i, sl_model)]
        # ) # Replace the above line with this line if using SL fine-tuning
        df = pd.concat([df, df_dictionary], ignore_index=True)
        print("Dummy-perm", df_dictionary["cx"])

    # Convert the cx column to a numerical type
    df["cx"] = pd.to_numeric(df["cx"])

    # Create a combined method from existing results
    combined_results = []
    for rep in df["n_rep"].unique():
        # Get results for this circuit
        circuit_df = df[df["n_rep"] == rep]
        # Get rows for both methods
        heuristic_row = circuit_df[circuit_df["method"] == "normal_heuristic"].iloc[0]
        dummy_row = circuit_df[circuit_df["method"] == "dummy-perm"].iloc[0]
        # Choose the better one
        if heuristic_row["cx"] <= dummy_row["cx"]:
            best_row = heuristic_row.copy()
        else:
            best_row = dummy_row.copy()
        # Update the method name
        best_row["method"] = "combined_min"
        # Add to results
        combined_results.append(best_row)
    # Add combined results to the DataFrame
    combined_df = pd.DataFrame(combined_results)
    df = pd.concat([df, combined_df], ignore_index=True)

    df.to_csv("test_clifford_synthesis.csv", index=False)
    # Question: what should be the comparision metric? Mean, median, std, mse, etc.?
    print(df.groupby("method").mean())

    visualize_optimality_gaps(df)  # Plot the results

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
