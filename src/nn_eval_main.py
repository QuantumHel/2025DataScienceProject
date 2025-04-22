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

def main(n_qubits: int = 4, nr_gates: int = 5):
    df = pd.DataFrame(columns=["n_rep", "num_qubits", "method", "h", "s", "cx", "depth"])
    topology = Topology.complete(n_qubits)

    for i in range(1000):
        print(f"Iteration {i}")
        circuit = random_hscx_circuit(nr_qubits=n_qubits, nr_gates=nr_gates)

        for method_fn in [our_compilation, optimal_compilation, random_compilation, rl_compilation]:
            row = method_fn(circuit.copy(), topology, i)
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
            print(f"{row['method']} CX: {row['cx']}")
        print("\n")
    df.to_csv("test_clifford_synthesis.csv", index=False)
    print(df.groupby("method").mean())

if __name__ == "__main__":
    main()
