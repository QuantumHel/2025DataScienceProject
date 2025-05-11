from typing import Tuple, Optional, Union, List
import gym
import networkx as nx
import numpy as np
from gym.core import RenderFrame
from pauliopt.circuits import Circuit
from pauliopt.clifford.tableau import CliffordTableau
from pauliopt.clifford.tableau_synthesis import steiner_reduce_column
from pauliopt.gates import CX, H, S
from pauliopt.topologies import Topology
from pauliopt.utils import is_cutting
from src.utils import random_hscx_circuit, tableau_from_circuit
from src.nn.brute_force_data import get_best_cnots

Array3D = np.ndarray

class CliffordTableauEnv(gym.Env[Tuple[int, int], np.ndarray]):
    def __init__(self, n_qubits: int, nr_gates: int = 1000, topology: Topology = None,
                 cx_penalty: float = -0.5, h_penalty: float = -0.1, s_penalty: float = -0.1,
                 final_reward: float = 30.0, final_exp_decay: float = 0.3):
        super().__init__()
        self.n_qubits = n_qubits
        self.nr_gates = nr_gates
        self.cx_penalty = cx_penalty
        self.h_penalty = h_penalty
        self.s_penalty = s_penalty
        self.final_reward = final_reward * (1 + ((self.nr_gates - 5) // 5))
        self.final_exp_decay = final_exp_decay
        self.topology = topology or Topology.complete(n_qubits)

    def reset(self, **kwargs):
        self.graph = self.topology.to_nx
        self.adjacency_matrix = nx.adjacency_matrix(self.graph).toarray()
        self.allowed_rows = list(range(self.n_qubits))
        self.allowed_cols = list(range(self.n_qubits))
        self.qubits_reduced = 0
        self.final_circuit = Circuit(self.n_qubits)
        self.final_cx = None

        circuit = random_hscx_circuit(nr_qubits=self.n_qubits, nr_gates=self.nr_gates)
        tableau = CliffordTableau(self.n_qubits)
        self.initial_tableau = tableau_from_circuit(tableau, circuit)
        self.clifford_tableau_to_reduce = self.initial_tableau.inverse()

        self.true_optimal_cx = get_best_cnots(self.clifford_tableau_to_reduce.inverse(), self.topology)[0][1]
        
        return self._get_obs(), self.allowed_rows.copy(), self.allowed_cols.copy()

    def get_current_stats(self) -> int:
        if self.final_cx is not None:
            return self.final_cx
        return self.final_circuit.to_qiskit().count_ops().get("cx", 0)
    
    def _compute_supervised_reward(self, cx_found: int, cx_optimal: int) -> float:
        if cx_found <= cx_optimal:
            return self.final_reward
        elif cx_found <= cx_optimal + 1:
            return self.final_reward * 0.5
        elif cx_found <= cx_optimal + 2:
            return self.final_reward * 0.25
        else:
            overshoot = cx_found - cx_optimal
            penalty = self.final_reward * np.exp(-0.2 * overshoot)
            penalty = penalty - self.final_reward
            return max(penalty, self.cx_penalty)

    def step(self, action: Tuple[int, int]):
        pivot_row, pivot_col = action
        assert not is_cutting(pivot_col, self.graph)

        reward = 0.0
        current_circuit = Circuit(self.n_qubits)

        def apply(gate_name: str, args: tuple):
            gate_cls = {"CNOT": CX, "H": H, "S": S}[gate_name]
            getattr(self.clifford_tableau_to_reduce, f"append_{gate_name.lower()}")(*args)
            self.final_circuit.add_gate(gate_cls(*args))
            current_circuit.add_gate(gate_cls(*args))

        if pivot_row in self.allowed_rows:
            self.allowed_rows.remove(pivot_row)
        if pivot_col in self.allowed_cols:
            self.allowed_cols.remove(pivot_col)
        self.qubits_reduced += 1

        steiner_reduce_column(pivot_col, pivot_row, self.graph, self.clifford_tableau_to_reduce, apply)
        self.graph.remove_node(pivot_col)

        # Reward based on gate cost
        ops = current_circuit.to_qiskit().count_ops()
        reward += self.cx_penalty * ops.get("cx", 0)
        reward += self.h_penalty * ops.get("h", 0)
        reward += self.s_penalty * ops.get("s", 0)

        done = self.qubits_reduced >= self.n_qubits
        
        """Basic reward structure (commented out):
        if done:
            self.final_cx = self.final_circuit.to_qiskit().count_ops().get("cx", 0)
            bonus = self.final_reward * np.exp(-self.final_exp_decay * self.final_cx)
            curriculum_level = max((self.nr_gates - 5) // 10, 0)
            curriculum_bonus = curriculum_level * 6.0
            reward += bonus + curriculum_bonus
        """
        
        """
        Secondary reward structure (commented out):
        if done:
            self.final_cx = self.final_circuit.to_qiskit().count_ops().get("cx", 0)

            # Smoother final reward (larger values still get rewarded, just less)
            decay = 0.25  # You can tune this down to 0.2 or up to 0.4
            smooth_bonus = self.final_reward / (1 + decay * self.final_cx)

            # Curriculum bonus — give a small fixed increase every +2 gates
            curriculum_level = max((self.nr_gates - 5) // 2, 0)
            curriculum_bonus = curriculum_level * 7.5  # You can try 5.0 → 7.5 → 10.0
            """
        """Supervised finetuning reward structure with true optimal circuit (commented out):"""
        if done:
            self.final_cx = self.final_circuit.to_qiskit().count_ops().get("cx", 0)
            reward = self._compute_supervised_reward(self.final_cx, self.true_optimal_cx)

        return (self._get_obs(), self.allowed_rows.copy(), self.allowed_cols.copy()), reward, done, {}

    def _get_obs(self) -> Array3D:
        n = self.n_qubits

        bitmap = np.ones((n, n), dtype=np.float32)
        for i in range(n):
            if i not in self.allowed_rows:
                bitmap[i, :] = 0.0
            if i not in self.allowed_cols:
                bitmap[:, i] = 0.0

        x_block = self.clifford_tableau_to_reduce.x_matrix[:n, :n]
        z_block = self.clifford_tableau_to_reduce.z_matrix[:n, :n]
        adj_matrix = self.adjacency_matrix.astype(np.float32)
        signs = self.clifford_tableau_to_reduce.signs[:n].astype(np.float32)
        sign_map = np.tile(signs[:, np.newaxis], (1, n))

        row_coords = np.tile(np.linspace(0, 1, n).reshape(n, 1), (1, n)).astype(np.float32)
        col_coords = np.tile(np.linspace(0, 1, n).reshape(1, n), (n, 1)).astype(np.float32)
        cx_channel = np.full((n, n), np.log1p(self.get_current_stats()), dtype=np.float32)

        return np.stack([
            x_block, z_block, bitmap, adj_matrix,
            sign_map, row_coords, col_coords, cx_channel
        ], axis=0)

    def render(self) -> Optional[Union[RenderFrame, List[RenderFrame]]]:
        print(self.clifford_tableau_to_reduce)
        return None
