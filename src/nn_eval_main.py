import os
import warnings
from typing import List
import networkx as nx
import numpy as np
import pandas as pd
import torch

from pauliopt.circuits import Circuit
from pauliopt.clifford.tableau import CliffordTableau
from pauliopt.clifford.tableau_synthesis import synthesize_tableau_perm_row_col
from pauliopt.topologies import Topology

from src.rl.env import CliffordTableauEnv
from src.rl.agent import DQNAgent
from src.nn.brute_force_data import get_best_cnots
from src.utils import random_hscx_circuit, tableau_from_circuit

# Suppress warnings
np.seterr(over='ignore')
warnings.simplefilter(action='ignore', category=FutureWarning)

model_path = "models/best_model.pt"
checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
CONFIG = checkpoint["config"]

n_qubits = 4
agent = DQNAgent(n_qubits=n_qubits, config=CONFIG)
agent.model.load_state_dict(checkpoint["model_state_dict"])
agent.model.eval()
agent.epsilon = 0.0

def pretty_print_circuit(circuit):
    """Pretty print the circuit."""
    print("Circuit:")
    for gate in circuit.gates:
        gate_type = gate.__class__.__name__
        qubits = ", ".join(str(q) for q in gate.qubits)
        print(f"  {gate_type} on qubit(s): {qubits}")

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

def our_compilation(circuit: Circuit, topology: Topology, n_rep: int):
    tableau = tableau_from_circuit(CliffordTableau(circuit.n_qubits), circuit)
    circ_out = synthesize_tableau_perm_row_col(tableau, topology)
    return {"n_rep": n_rep, "method": "normal_heuristic", **collect_circuit_data(circ_out)}

def random_compilation(circuit: Circuit, topology: Topology, n_rep: int):
    tableau = tableau_from_circuit(CliffordTableau(circuit.n_qubits), circuit)
    def random_pick(G, remaining, rows, choice_fn=min):
        row = np.random.choice(rows)
        return row, row
    circ_out = synthesize_tableau_perm_row_col(tableau, topology, pick_pivot_callback=random_pick)
    return {"n_rep": n_rep, "method": "random", **collect_circuit_data(circ_out)}

def optimal_compilation(circuit: Circuit, topology: Topology, n_rep: int):
    tableau = tableau_from_circuit(CliffordTableau(circuit.n_qubits), circuit)
    best_perm, _ = get_best_cnots(tableau.inverse().inverse(), topology)[0]
    best_perm = iter(best_perm)
    def best_pick(G, remaining, rows, choice_fn=min):
        return next(best_perm)
    circ_out = synthesize_tableau_perm_row_col(tableau, topology, pick_pivot_callback=best_pick)
    return {"n_rep": n_rep, "method": "optimum", **collect_circuit_data(circ_out)}

def rl_compilation(circuit: Circuit, topology: Topology, n_rep: int):
    """Use the RL agent to compile the circuit."""
    tableau = tableau_from_circuit(CliffordTableau(circuit.n_qubits), circuit)
    env = CliffordTableauEnv(
        n_qubits=circuit.n_qubits,
        nr_gates=0,
        topology=topology,
        cx_penalty=0.0,
        h_penalty=0.0,
        s_penalty=0.0,
        final_reward=0.0
    )
    env.clifford_tableau_to_reduce = tableau.inverse()
    env.final_circuit = Circuit(circuit.n_qubits)
    env.final_cx = None
    env.allowed_rows = list(range(circuit.n_qubits))
    env.allowed_cols = list(range(circuit.n_qubits))
    env.qubits_reduced = 0
    env.graph = env.topology.to_nx
    env.adjacency_matrix = nx.adjacency_matrix(env.graph).toarray()

    def pick_pivot(G, remaining, rows, choice_fn=min):
        obs = env._get_obs()
        row, col = agent.act(obs, env.allowed_rows, env.allowed_cols, explore=False)
        env.allowed_rows.remove(row)
        env.allowed_cols.remove(col)
        env.graph.remove_node(col)
        return row, col

    circ_out = synthesize_tableau_perm_row_col(tableau, topology, pick_pivot_callback=pick_pivot)
    return {"n_rep": n_rep, "method": "rl_model", **collect_circuit_data(circ_out)}

def main(n_qubits: int = 4, nr_gates: int = 10):
    df = pd.DataFrame(columns=["n_rep", "num_qubits", "method", "h", "s", "cx", "depth"])
    topology = Topology.complete(n_qubits)

    # Initialize confusion matrix-like structure for RL vs. Optimum scores
    confusion_matrix = pd.DataFrame() 
    if nr_gates > 20: print("Warning: nr_gates > 20, RL agent only trained up to 20 gate compexity.")  

    for i in range(1000):
        print(f"Iteration {i}")
        circuit = random_hscx_circuit(nr_qubits=n_qubits, nr_gates=nr_gates)

        # Store scores for each method
        method_scores = {}
        for method_fn in [our_compilation, optimal_compilation, random_compilation, rl_compilation]:
            row = method_fn(circuit.copy(), topology, i)
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
            method_scores[row["method"]] = row["cx"]
            print(f"{row['method']}: {row['cx']}", end=" | ")
        print("\n")

        # Compare RL score to the optimum score
        rl_score = method_scores["rl_model"]
        optimum_score = method_scores["optimum"]

        # Update confusion matrix-like structure
        if optimum_score not in confusion_matrix.index or rl_score not in confusion_matrix.columns:
            confusion_matrix.loc[optimum_score, rl_score] = 0

        confusion_matrix.loc[optimum_score, rl_score] += 1

        # Print circuit if RL score is +5 worse than optimum
        score_diff = rl_score - optimum_score
        #if score_diff >= 5:
            #print(f"RL score is significantly worse (+{score_diff}) than optimum. Circuit:")
            #pretty_print_circuit(circuit)

    # Save results to CSV
    df.to_csv("test_clifford_synthesis.csv", index=False)

    # Print confusion matrix
    print("\nConfusion Matrix (Optimum vs RL Scores):")
    confusion_matrix = confusion_matrix.sort_index(axis=0).sort_index(axis=1)
    confusion_matrix = confusion_matrix.fillna(0)
    print(confusion_matrix)

    # Print mean scores grouped by method
    print("\nMean Scores by Method:")
    print(df.groupby("method").mean())

if __name__ == "__main__":
    main()
