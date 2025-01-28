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

Array3D = np.array


class CliffordTableauEnv(gym.Env[Tuple[int, int], np.ndarray]):
    def __init__(self, n_qubits: int, nr_gates: int = 1000, topology: Topology = None):
        """
        Defines a RL environment, that describes the synthesis of a clifford tableau.

        :param n_qubits: Nr of qubits of the tableau
        :param nr_gates: Nr of gates of the tableau
        :param topology: Which topology to synth the tableau with
        """
        super(CliffordTableauEnv, self).__init__()
        self.n_qubits = n_qubits
        self.nr_gates = nr_gates
        self.clifford_tableau_to_reduce = None
        self.final_circuit = None
        self.qubits_reduced = 0
        if topology is None:
            self.topology = Topology.complete(self.n_qubits)
        else:
            self.topology = topology
        self.graph = self.topology.to_nx
        self.adjacency_matrix = nx.adjacency_matrix(self.graph).toarray()
        self.allowed_rows = list(range(self.n_qubits))
        self.allowed_cols = list(range(self.n_qubits))

    def reset(self, **kwargs):
        """

        :param kwargs:
        :return:
        """
        circuit = random_hscx_circuit(nr_qubits=self.n_qubits, nr_gates=self.nr_gates)
        clifford_tableau = CliffordTableau(self.n_qubits)
        clifford_tableau = tableau_from_circuit(clifford_tableau, circuit)
        self.clifford_tableau_to_reduce = clifford_tableau.inverse()
        self.final_circuit = Circuit(self.n_qubits)
        self.graph = self.topology.to_nx
        self.allowed_rows = list(range(self.n_qubits))
        self.allowed_cols = list(range(self.n_qubits))
        self.adjacency_matrix = nx.adjacency_matrix(self.graph).toarray()
        self.qubits_reduced = 0

        return self._get_obs(), self.allowed_rows, self.allowed_cols

    def get_current_stats(self) -> float:
        return self.final_circuit.to_qiskit().count_ops().get("cx", 0)

    def step(self, action: Tuple[int, int]) -> Tuple[Tuple[Array3D, list, list], float, bool, Dict[Any, Any]]:
        current_circuit = Circuit(self.n_qubits)

        def apply(gate_name: str, gate_data: tuple) -> None:
            if gate_name == "CNOT":
                self.clifford_tableau_to_reduce.append_cnot(gate_data[0], gate_data[1])
                self.final_circuit.add_gate(CX(gate_data[0], gate_data[1]))
                current_circuit.add_gate(CX(gate_data[0], gate_data[1]))
            elif gate_name == "H":
                self.clifford_tableau_to_reduce.append_h(gate_data[0])
                self.final_circuit.add_gate(H(gate_data[0]))
                current_circuit.add_gate(H(gate_data[0]))
            elif gate_name == "S":
                self.clifford_tableau_to_reduce.append_s(gate_data[0])
                self.final_circuit.add_gate(S(gate_data[0]))
                current_circuit.add_gate(S(gate_data[0]))
            else:
                raise Exception("Unknown Gate")

        pivot_row, pivot_col = action
        assert not is_cutting(pivot_col, self.graph)

        self.allowed_rows.remove(pivot_row)
        self.allowed_cols.remove(pivot_col)
        self.qubits_reduced += 1
        steiner_reduce_column(pivot_col, pivot_row, self.graph, self.clifford_tableau_to_reduce, apply)
        self.graph.remove_node(pivot_col)
        done = False
        if self.qubits_reduced >= self.n_qubits:
            final_permutation = np.argmax(self.clifford_tableau_to_reduce.x_matrix, axis=1)
            signs_copy_z = self.clifford_tableau_to_reduce.signs[
                           self.clifford_tableau_to_reduce.n_qubits: 2 * self.clifford_tableau_to_reduce.n_qubits].copy()

            for col in range(self.clifford_tableau_to_reduce.n_qubits):
                if signs_copy_z[col] != 0:
                    apply("H", (final_permutation[col],))
                    apply("S", (final_permutation[col],))
                    apply("S", (final_permutation[col],))
                    apply("H", (final_permutation[col],))

            for col in range(self.clifford_tableau_to_reduce.n_qubits):
                if self.clifford_tableau_to_reduce.signs[col] != 0:
                    apply("S", (final_permutation[col],))
                    apply("S", (final_permutation[col],))

            done = True

        reward = np.exp(-current_circuit.to_qiskit().count_ops().get("cx", float("inf")))
        return (self._get_obs(), self.allowed_rows, self.allowed_cols), reward, done, {}

    def render(self) -> Optional[Union[RenderFrame, List[RenderFrame]]]:
        print(self.clifford_tableau_to_reduce)
        return None

    def _get_obs(self) -> Array3D:
        disallowed_rows = list(set(list(range(self.n_qubits))) - set(self.allowed_rows))
        disallowed_columns = list(set(list(range(self.n_qubits))) - set(self.allowed_cols))
        bitmap = np.ones((self.n_qubits, self.n_qubits))
        bitmap[disallowed_rows, :] = 0.0
        bitmap[:, disallowed_columns] = 0.0
        return np.stack([
            self.clifford_tableau_to_reduce.x_matrix,
            self.clifford_tableau_to_reduce.z_matrix,
            bitmap
        ], axis=0)

    def _was_selected_previously(self, pivot_row: int, pivot_column: int) -> bool:
        return pivot_row not in self.allowed_rows or pivot_column not in self.allowed_cols
