from typing import Tuple, Optional, Union, List, Any, Dict
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

Array3D = np.array


def get_optimal_cx_estimate(n_qubits: int) -> int:
    baseline = {2: 2, 3: 3, 4: 5, 5: 7, 6: 10}
    return baseline.get(n_qubits, int(round(0.5 * n_qubits * (n_qubits - 1) / 2)))


class CliffordTableauEnv(gym.Env[Tuple[int, int], np.ndarray]):
    def __init__(self, n_qubits: int, nr_gates: int = 1000, topology: Topology = None,
                 step_penalty=-0.3, final_reward_max=30.0, use_true_cx: bool = False):
        super().__init__()
        self.n_qubits = n_qubits
        self.nr_gates = nr_gates
        self.step_penalty = step_penalty
        self.final_reward_max = final_reward_max
        self.use_true_cx = use_true_cx

        self.topology = topology or Topology.complete(n_qubits)
        self.graph = self.topology.to_nx
        self.adjacency_matrix = nx.adjacency_matrix(self.graph).toarray()
        self.allowed_rows = list(range(n_qubits))
        self.allowed_cols = list(range(n_qubits))
        self.clifford_tableau_to_reduce = None
        self.final_circuit = None
        self.final_cx = None
        self.qubits_reduced = 0

    def reset(self, **kwargs):
        circuit = random_hscx_circuit(nr_qubits=self.n_qubits, nr_gates=self.nr_gates)
        tableau = CliffordTableau(self.n_qubits)
        tableau = tableau_from_circuit(tableau, circuit)
        self.clifford_tableau_to_reduce = tableau.inverse()
        self.final_circuit = Circuit(self.n_qubits)
        self.graph = self.topology.to_nx
        self.adjacency_matrix = nx.adjacency_matrix(self.graph).toarray()
        self.allowed_rows = list(range(self.n_qubits))
        self.allowed_cols = list(range(self.n_qubits))
        self.qubits_reduced = 0
        self.final_cx = None

        return self._get_obs(), self.allowed_rows.copy(), self.allowed_cols.copy()

    def get_true_optimal_cx(self) -> int:
        best_perm, score = get_best_cnots(self.clifford_tableau_to_reduce.inverse(), self.topology)[0]
        return score

    def get_current_stats(self) -> float:
        return self.final_circuit.to_qiskit().count_ops().get("cx", 0)

    def get_auxiliary_label(self) -> float:
        if self.final_cx is None:
            return 0.0
        optimal = self.get_true_optimal_cx() if self.use_true_cx else get_optimal_cx_estimate(self.n_qubits)
        delta = self.final_cx - optimal
        margin = 8

        if delta <= 0:
            label = 1.0 + (abs(delta) / margin)
        else:
            label = max(0.0, 1.0 - (delta / margin))

        return float(np.clip(label, 0.0, 1.2))  # clamp just in case

    def step(self, action: Tuple[int, int]):
        pivot_row, pivot_col = action
        assert not is_cutting(pivot_col, self.graph)

        current_circuit = Circuit(self.n_qubits)

        def apply(gate_name: str, gate_data: tuple) -> None:
            if gate_name == "CNOT":
                self.clifford_tableau_to_reduce.append_cnot(gate_data[0], gate_data[1])
                self.final_circuit.add_gate(CX(*gate_data))
                current_circuit.add_gate(CX(*gate_data))
            elif gate_name == "H":
                self.clifford_tableau_to_reduce.append_h(gate_data[0])
                self.final_circuit.add_gate(H(gate_data[0]))
                current_circuit.add_gate(H(gate_data[0]))
            elif gate_name == "S":
                self.clifford_tableau_to_reduce.append_s(gate_data[0])
                self.final_circuit.add_gate(S(gate_data[0]))
                current_circuit.add_gate(S(gate_data[0]))

        self.allowed_rows.remove(pivot_row)
        self.allowed_cols.remove(pivot_col)
        self.qubits_reduced += 1
        steiner_reduce_column(pivot_col, pivot_row, self.graph, self.clifford_tableau_to_reduce, apply)
        self.graph.remove_node(pivot_col)

        done = self.qubits_reduced >= self.n_qubits
        if done:
            final_perm = np.argmax(self.clifford_tableau_to_reduce.x_matrix, axis=1)
            signs_z = self.clifford_tableau_to_reduce.signs[self.n_qubits:2 * self.n_qubits].copy()

            for col in range(self.n_qubits):
                if signs_z[col] != 0:
                    apply("H", (final_perm[col],))
                    apply("S", (final_perm[col],))
                    apply("S", (final_perm[col],))
                    apply("H", (final_perm[col],))

            for col in range(self.n_qubits):
                if self.clifford_tableau_to_reduce.signs[col] != 0:
                    apply("S", (final_perm[col],))
                    apply("S", (final_perm[col],))

        cx_count = current_circuit.to_qiskit().count_ops().get("cx", 0)
        max_possible_cx = self.n_qubits * (self.n_qubits - 1)

        reward = 3.0 * (1.0 - (cx_count / max_possible_cx) ** 1.2)
        reward += self.step_penalty

        if done:
            self.final_cx = self.final_circuit.to_qiskit().count_ops().get("cx", 0)
            optimal = self.get_true_optimal_cx() if self.use_true_cx else get_optimal_cx_estimate(self.n_qubits)
            delta = self.final_cx - optimal
            margin = 6

            if delta <= 0:
                # Overachievement bonus: sqrt-based for diminishing returns
                over_bonus = (abs(delta) + 1) ** 0.5
                bonus = self.final_reward_max + over_bonus
            else:
                # Normal smooth bonus
                bonus = self.final_reward_max * max(0.0, 1.0 - (delta / margin) ** 1.2)

            reward += bonus


        return (self._get_obs(), self.allowed_rows.copy(), self.allowed_cols.copy()), reward, done, {}

    def _get_obs(self) -> Array3D:
        bitmap = np.ones((self.n_qubits, self.n_qubits), dtype=np.float32)
        for i in range(self.n_qubits):
            if i not in self.allowed_rows:
                bitmap[i, :] = 0.0
            if i not in self.allowed_cols:
                bitmap[:, i] = 0.0

        return np.stack([
            self.clifford_tableau_to_reduce.x_matrix,
            self.clifford_tableau_to_reduce.z_matrix,
            bitmap,
            self.adjacency_matrix.astype(np.float32)
        ], axis=0)

    def render(self) -> Optional[Union[RenderFrame, List[RenderFrame]]]:
        print(self.clifford_tableau_to_reduce)
        return None