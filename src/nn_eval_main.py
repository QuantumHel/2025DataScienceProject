"""Code to possibly evaluate the NN training approach. Currently, this only compares our and the CNN compilation. """
import warnings
from typing import List
import networkx as nx
import numpy as np
import pandas as pd
from pauliopt.circuits import Circuit
from pauliopt.clifford.tableau import CliffordTableau
from pauliopt.clifford.tableau_synthesis import synthesize_tableau_perm_row_col
from pauliopt.topologies import Topology
from src.rl.env import CliffordTableauEnv

from src.nn.brute_force_data import get_best_cnots
from src.utils import random_hscx_circuit, tableau_from_circuit

import torch

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

from src.rl.agent import DQNAgent

n_qubits = 4
model_path = "models/best_model.pt"
CONFIG = {
    "learning_rate": 3e-4,
    "batch_size": 32,
    "epsilon_decay": 0.999,
    "epsilon_min": 0.005,
    "gamma": 0.995,
    "gradient_clip_norm": 10.0,
    "aux_loss_weight": 0.4
}

agent = DQNAgent(n_qubits=n_qubits, config=CONFIG)
checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
agent.model.load_state_dict(checkpoint["model_state_dict"])
agent.model.eval()
agent.epsilon = 0.0

def rl_compilation(circuit: Circuit, topology: Topology, n_rep: int, agent):
    n_qubits = circuit.n_qubits

    clifford_tableau = CliffordTableau(n_qubits)
    clifford_tableau = tableau_from_circuit(clifford_tableau, circuit)

  
    env = CliffordTableauEnv(
        n_qubits=n_qubits,
        nr_gates=0, 
        topology=topology,
        step_penalty=0.0,
        final_reward_max=0.0
    )

    env.clifford_tableau_to_reduce = clifford_tableau.inverse()
    env.final_circuit = Circuit(n_qubits)
    env.allowed_rows = list(range(n_qubits))
    env.allowed_cols = list(range(n_qubits))
    env.qubits_reduced = 0
    env.graph = env.topology.to_nx
    env.adjacency_matrix = nx.adjacency_matrix(env.graph).toarray()

    def pick_pivot_callback(G, remaining: "CliffordTableau", remaining_rows: List[int], choice_fn=min):
        obs = env._get_obs()
        row, col = agent.act(obs, env.allowed_rows, env.allowed_cols)

        env.allowed_rows.remove(row)
        env.allowed_cols.remove(col)
        env.graph.remove_node(col)

        return row, col

    circ_out = synthesize_tableau_perm_row_col(
        clifford_tableau, topology, pick_pivot_callback=pick_pivot_callback
    )

    return {"n_rep": n_rep} | collect_circuit_data(circ_out) | {"method": "rl_model"}


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

        df_dictionary = pd.DataFrame([rl_compilation(circuit.copy(), topo, i, agent)])
        df = pd.concat([df, df_dictionary], ignore_index=True)
        print("RL", df_dictionary["cx"])

    df.to_csv("test_clifford_synthesis.csv", index=False)
    print(df.groupby("method").mean())


if __name__ == "__main__":
    main()
