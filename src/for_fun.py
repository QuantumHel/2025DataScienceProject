# from nn_eval_main import add_cx_trend_plot
import pandas as pd

df = pd.read_csv("test_clifford_synthesis.csv")
# add_cx_trend_plot(df)


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def visualize_optimality_gaps(df):
    """Plot the gap between each method and the optimum solution"""
    plt.figure(figsize=(14, 8))
    sns.set_style("whitegrid")

    # Extract the methods we care about
    methods = ["normal_heuristic", "dummy-perm", "combined_min", "optimum"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    labels = ["Standard Heuristic", "Neural Network", "Combined", "Optimum"]

    # Get data for the optimum method first
    opt_df = df[df["method"] == "optimum"].sort_values("n_rep")
    opt_rolling = opt_df["cx"].rolling(window=50, min_periods=1).mean()

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


visualize_optimality_gaps(df)
