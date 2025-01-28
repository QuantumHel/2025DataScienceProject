import itertools
from typing import Tuple, List, Callable

import numpy as np
import torch
from pauliopt.clifford.tableau import CliffordTableau
from pauliopt.clifford.tableau_synthesis import steiner_reduce_column, \
    synthesize_tableau_perm_row_col
from pauliopt.topologies import Topology
from tqdm import tqdm

from src.utils import tableau_from_circuit, random_hscx_circuit


def enumerate_combinations(n):
    """
    Enumerate all combinations of rows and cols.
    :param n:
    :return:
    """
    rows = list(range(n))
    cols = list(range(n))
    row_permutations = itertools.permutations(rows)
    combinations = []
    for row_perm in row_permutations:
        for col_perm in itertools.permutations(cols, n):
            pair_sequence = list(zip(row_perm, col_perm))
            combinations.append(pair_sequence)

    return combinations


best_cnots = []


def get_best_cnots(clifford_tableau: CliffordTableau, topo: Topology = None) -> List[Tuple[list, int]]:
    """
    Brute force all possible cnot permutations and return the best n with the lowest score.
    :param clifford_tableau:
    :param topo:
    :return:
    """
    if topo is None:
        topo = Topology.complete(clifford_tableau.n_qubits)

    permutations_with_cx_count = []

    for permutation in enumerate_combinations(clifford_tableau.n_qubits):
        perm_iter = iter(permutation)

        def pick_pivot_callback(G, remaining: "CliffordTableau", remaining_rows: List[int], choice_fn=min):
            return next(perm_iter)

        qc, _ = synthesize_tableau_perm_row_col(clifford_tableau, topo, pick_pivot_callback=pick_pivot_callback)
        qc.final_permutation = None

        cx_count = qc.to_qiskit().count_ops().get("cx", float("inf"))

        permutations_with_cx_count.append((permutation, cx_count))

    best_cx_count = min(cx_count for _, cx_count in permutations_with_cx_count)
    if best_cx_count == float("inf"):
        return []
    best_perms = [(list(perm), cx_count) for perm, cx_count in permutations_with_cx_count if cx_count == best_cx_count]
    return best_perms


def generate_data_series_ct(clifford_tableau: CliffordTableau, topo: Topology = None) -> Tuple[
    list[torch.Tensor], list[torch.Tensor]]:
    """
    Generates a optimal data series for a clifford tableau.
    The idea is to mark for every tableau all pivots that will lead to a optimal solution on a bitmap.

    Returned are all reached tableau configurations combined with the choice of optimal pivots.
    :param clifford_tableau:
    :param topo:
    :return:
    """
    if topo is None:
        topo = Topology.complete(clifford_tableau.n_qubits)

    best_permutations = get_best_cnots(clifford_tableau, topo)
    representations = dict()
    for best_permutation, score in best_permutations:
        disallowed_columns = []
        disallowed_rows = []
        graph = topo.to_nx
        ct_to_reduce = clifford_tableau.inverse()

        def apply(gate_name: str, gate_data: tuple) -> None:
            if gate_name == "CNOT":
                ct_to_reduce.append_cnot(gate_data[0], gate_data[1])
            elif gate_name == "H":
                ct_to_reduce.append_h(gate_data[0])
            elif gate_name == "S":
                ct_to_reduce.append_s(gate_data[0])
            else:
                raise Exception("Unknown Gate")

        for pivot_col, pivot_row in best_permutation:
            expected_map = torch.zeros(size=(1, ct_to_reduce.n_qubits, ct_to_reduce.n_qubits))
            expected_map[0, pivot_row, pivot_col] = 1.0
            steiner_reduce_column(pivot_col, pivot_row, graph, ct_to_reduce, apply)

            reduced_elements = np.ones((clifford_tableau.n_qubits, clifford_tableau.n_qubits))
            reduced_elements[disallowed_rows, :] = 0.0
            reduced_elements[:, disallowed_columns] = 0.0
            disallowed_columns.append(pivot_col)
            disallowed_rows.append(pivot_row)

            tableau_representation = np.stack([
                np.where(ct_to_reduce.x_matrix == 0, 0, 1),
                np.where(ct_to_reduce.z_matrix == 0, 0, 1),
                reduced_elements
            ], axis=0)
            graph.remove_node(pivot_col)
            tableau_representation = torch.from_numpy(tableau_representation)
            tensor_key = tuple(map(lambda x: tuple(map(tuple, x)), tableau_representation.tolist()))
            if tensor_key not in representations:
                representations[tensor_key] = expected_map
            else:
                representations[tensor_key][0, pivot_row, pivot_col] = 1.0

    return [torch.tensor(key) for key in representations.keys()], list(representations.values())


def generate_dataset_ct(nr_samples: int, qubits: List[int],
                        gates: List[int],
                        topo_factory: Callable[[int], Topology] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate a dataset:

    X: [
        x_matrix,             => A bitmap describing the stabilizers of the tableau
        z_matrix,             => A bitmap describing the destabilizers of the tableau
        reduced_elements      => A bitmap where all reduced rows and cols are marked as zero
       ]
    Y: A bitmap showing all possible choices of pivots that are sensible.
    :param nr_samples:
    :param qubits:
    :param gates:
    :param topo_factory:
    :return:
    """
    if topo_factory is None:
        topo_factory = lambda nr_qubits: Topology.complete(nr_qubits)
    all_tableaus = []
    all_expected_pr_map = []
    for n_qubits in qubits:
        for nr_gates in gates:
            topo = topo_factory(n_qubits)
            for _ in tqdm(range(nr_samples), desc="Generating Dataset"):
                # for _ in range(nr_samples):
                circuit = random_hscx_circuit(nr_qubits=n_qubits, nr_gates=nr_gates)
                clifford_tableau = CliffordTableau(n_qubits)
                clifford_tableau = tableau_from_circuit(clifford_tableau, circuit)
                tableaus, expected_pr_map = generate_data_series_ct(clifford_tableau, topo)
                all_tableaus += tableaus
                all_expected_pr_map += expected_pr_map

    return torch.stack(all_tableaus).float(), torch.stack(all_expected_pr_map).float()
