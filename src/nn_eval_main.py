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

# from src.nn.permutation_pred2_optm import predict_permutation, pretrain_a_model, PermutationGNN

# from src.nn.permutation_pred2 import (
#     predict_permutation,
#     pretrain_a_model_from_file,
#     SequentialPermutationGNN,
# )

from src.nn.permutation_math import (
    predict_permutation,
    pretrain_a_model_from_file,
    OrderedPermutationTransformer,
    TableauPermutationDataset,
    supervised_cx_fine_tune,
    predict_permutation_gumbel,
    predict_permutation_beam,
    entropy_guided_search,
    ensemble_predict_permutation,
    curriculum_train,
    get_default_device,
    predict_permutation_fixed_dismatch,
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
    clifford_tableau = CliffordTableau(circuit.n_qubits)
    clifford_tableau = tableau_from_circuit(clifford_tableau, circuit)
    # best_permutation = predict_permutation_gumbel(model, clifford_tableau, device)
    best_permutation = predict_permutation_fixed_dismatch(
        model, clifford_tableau, device
    )
    # best_permutation = predict_permutation_beam(model, clifford_tableau, device)
    # best_permutation = entropy_guided_search(model, clifford_tableau, device)
    # best_permutation = ensemble_predict_permutation(model, clifford_tableau, device)
    # print(best_permutation)

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


def add_cx_trend_plot(df):
    # Set up the plot style
    plt.figure(figsize=(12, 7))
    sns.set_style("whitegrid")

    # Get data for each method
    methods = ["normal_heuristic", "dummy-perm", "combined_min", "optimum"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    labels = ["Standard Heuristic", "Neural Network", "Combined Method", "Optimum"]

    # for method, color, label in zip(methods, colors, labels):
    #     # Extract data for this method
    #     method_df = df[df["method"] == method].sort_values("n_rep")

    #     # Plot CX count trend
    #     plt.plot(
    #         method_df["n_rep"], method_df["cx"], color=color, alpha=0.7, label=label
    #     )

    #     # Add rolling average for clarity
    #     rolling_avg = method_df["cx"].rolling(window=50, min_periods=1).mean()
    #     plt.plot(method_df["n_rep"], rolling_avg, color=color, linewidth=2.5)

    for method, color, label in zip(methods, colors, labels):
        # Extract data for this method
        method_df = df[df["method"] == method].sort_values("n_rep")

        # Calculate rolling statistics
        rolling_avg = method_df["cx"].rolling(window=50, min_periods=1).mean()
        rolling_std = method_df["cx"].rolling(window=50, min_periods=1).std()

        # Plot the mean line
        plt.plot(
            method_df["n_rep"], rolling_avg, color=color, linewidth=2.5, label=label
        )

        # Add shaded area for standard deviation
        plt.fill_between(
            method_df["n_rep"],
            rolling_avg - rolling_std,
            rolling_avg + rolling_std,
            color=color,
            alpha=0.2,
        )

    # Add titles and labels
    plt.title("CX Gate Count Comparison Across Compilation Methods", fontsize=16)
    plt.xlabel("Circuit Evaluation Index", fontsize=14)
    plt.ylabel("Number of CX Gates", fontsize=14)
    plt.legend(fontsize=12)

    # Add statistics in text box
    stats_text = "Mean CX Count:\n"
    for method, label in zip(methods, labels):
        mean_cx = df[df["method"] == method]["cx"].mean()
        stats_text += f"{label}: {mean_cx:.2f}\n"

    plt.figtext(
        0.02, 0.02, stats_text, fontsize=12, bbox=dict(facecolor="white", alpha=0.8)
    )

    # Save the figure
    plt.tight_layout()
    plt.savefig("cx_count_comparison.png", dpi=300)
    plt.show()


def main(n_qubits: int = 4, nr_gates: int = 1000):
    """
    Execute a single experiment with random clifford circuits and store the respective gate count into a dataframe
    :param n_qubits:
    :param nr_gates:
    :return:
    """

    device = get_default_device()  # mps seems not working well???
    device = "cpu"
    # Pre-train a model
    # model = pretrain_a_model_from_file("nn/training_data_perm.pkl", None, 100, device)
    # model = curriculum_train(
    #     "nn/training_data_perm.pkl", max_samples=None, epochs_per_stage=25
    # )
    # sl_dataset = TableauPermutationDataset(
    #     "nn/training_data_perm_4_qubit.pkl", n_qubits=4, max_samples=320
    # )
    # sl_model = supervised_cx_fine_tune(model, sl_dataset, epochs=50, device="cpu")

    checkpoint = torch.load(
        "ordered_permutation_model_100_earlystop_butno.pth", map_location="cpu"
    )
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
    for i in range(1):
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
        # ) # Uncomment this line to use the model with SL fine-tuning
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

    add_cx_trend_plot(df)  # Plot the results

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
