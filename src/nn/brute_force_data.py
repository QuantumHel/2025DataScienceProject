import itertools
from typing import Tuple, List, Callable

import numpy as np
import torch
from pauliopt.clifford.tableau import CliffordTableau
from pauliopt.clifford.tableau_synthesis import steiner_reduce_column, \
    synthesize_tableau_perm_row_col
from pauliopt.topologies import Topology
from tqdm import tqdm

from src.nn.preprocess_data import *
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

def get_best_cnots(clifford_tableau: CliffordTableau, topo: Topology) -> List[Tuple[list, int]]:
    """
    Brute force all possible cnot permutations and return the best n with the lowest score.
    :param clifford_tableau:
    :param topo:
    :return:
    """
    permutations_with_cx_count = []

    for permutation in enumerate_combinations(clifford_tableau.n_qubits):
        perm_iter = iter(permutation)

        def pick_pivot_callback(G, remaining: "CliffordTableau", remaining_rows: List[int], choice_fn=min):
            return next(perm_iter)

        qc = synthesize_tableau_perm_row_col(clifford_tableau, topo, pick_pivot_callback=pick_pivot_callback)
        qc.final_permutation = None

        cx_count = qc.to_qiskit().count_ops().get("cx", 0)

        permutations_with_cx_count.append((permutation, cx_count))

    best_cx_count = min(cx_count for _, cx_count in permutations_with_cx_count)
    best_perms = [(list(perm), cx_count) for perm, cx_count in permutations_with_cx_count if cx_count == best_cx_count]
    return best_perms


def generate_data_series_ct(clifford_tableau: CliffordTableau, topo: Topology, preprocessing_type: PreprocessingType = PreprocessingType.ORIGINAL) -> Tuple[
    list[torch.Tensor], list[torch.Tensor]]:
    """
    Generates a optimal data series for a clifford tableau.
    The idea is to mark for every tableau all pivots that will lead to a optimal solution on a bitmap.

    The training input data X is defined by the `preprocess_type` which indicates which function to use.
    By default, this is the same as it was before.

    The training label Y is defined as follows:
    Y: A bitmap showing all possible choices of pivots that go towards an optimal solution.

    As a result, X will always be unique and all possible optimal choices of pivots are in Y.

    Returned are all reached tableau configurations combined with the choice of optimal pivots.
    :param clifford_tableau:
    :param topo:
    :return:
    """
    # Find the optimal solutions - there may be more
    best_permutations = get_best_cnots(clifford_tableau, topo)
    dataset = dict()
    for best_permutation, score in best_permutations:
        # For each solution, simulate what happened.
        remaining_rows = list(range(clifford_tableau.n_qubits))
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
            # Each step choses a pivot row and pivot column
            input_data = PREPROCESSING_SCRIPTS[preprocessing_type](graph, ct_to_reduce, remaining_rows)

            # We may have encountered the same input data before, but with a different choice or row and column
            # Thus we make a unique key and update the label
            tensor_key = tuple(map(lambda x: tuple(map(tuple, x)), input_data.tolist()))
            dataset[tensor_key] = dataset.get(tensor_key, torch.zeros(size=(1, ct_to_reduce.n_qubits, ct_to_reduce.n_qubits)))
            dataset[tensor_key][0, pivot_row, pivot_col] = 1.0 # label

            # Run the simulation to get the next state
            steiner_reduce_column(pivot_col, pivot_row, graph, ct_to_reduce, apply)
            graph.remove_node(pivot_col)
            remaining_rows.remove(pivot_row)

    return [torch.tensor(key) for key in dataset.keys()], list(dataset.values())

cnot_count = 0
def generate_data_as_project_description(clifford_tableau: CliffordTableau, topo: Topology, preprocess_type: PreprocessingType = PreprocessingType.FROM_PROJECT_DESCRIPTION) -> Tuple[
    list[torch.Tensor], list[torch.Tensor]]:
    """
    Generates a optimal data series for a clifford tableau.

    The project description stated: 
    - a 2nx2n binary matrix representing the quantum circuit
    - a nxn binary adjacency matrix of a graph that represents how the qubits are connected in the quantum computer
    - n integer vector representing the minimal cost of choosing that qubit. The classification task is to find the index of the smallest value.
    Note that n is technically variable. We hope to have a single classifier for all n, rather than n different classifiers. So the matrices are padded.

    This is not entirely what is created here.

    The training input data X is defined by the `preprocess_type` which indicates which function to use.
    By default, this closely resembles the description above.

    The training label Y is defined as follows:
    Y: N integer vector representing the minimal cost of choosing that qubit. The classification task is to find the index of the smallest value.

    As a result, X will always be unique and all possible choices of pivots correspond with the lowest value in Y.
    Note that N is technically variable, so if a pivot cannot be chosen at all, the value corresponds to np.float("inf"). 

    Returned are all reached tableau configurations combined with the choice of optimal pivots.
    :param clifford_tableau:
    :param topo:
    :return:
    """
    global cnot_count
    queue = [(clifford_tableau.inverse(), list(range(clifford_tableau.n_qubits)), list(range(clifford_tableau.n_qubits)), topo)]
    dataset = {}
    round = 0
    while len(queue) > 0: 
        # Get a step from the queue
        tableau, remaining_rows, remaining_cols, topology = queue.pop()
        print(tableau)
        print(remaining_rows, remaining_cols)
        # Make the data into a tensor
        input_data = PREPROCESSING_SCRIPTS[preprocess_type](topology.to_nx, tableau, remaining_rows)
        # We may have encountered the same input data before, but with a different choice or row and column
        # Thus we make a unique key and update the label
        tensor_key = tuple(map(lambda x: tuple(map(tuple, x)), input_data.tolist()))
        
        if tensor_key not in dataset:
            label = torch.zeros(size=(1, clifford_tableau.n_qubits, clifford_tableau.n_qubits))
            # What are the options here?
            for row, col in itertools.product(remaining_rows, remaining_cols):
                round += 1
                print(round)
                # Run one step in the simulation to find the next step
                tableau_copy = tableau.inverse().inverse() # TODO make faster using deepcopy
                cnot_count = 0
                def apply(gate_name: str, gate_data: tuple) -> None:
                    global cnot_count
                    if gate_name == "CNOT":
                        cnot_count += 1
                        tableau_copy.append_cnot(gate_data[0], gate_data[1])
                    elif gate_name == "H":
                        tableau_copy.append_h(gate_data[0])
                    elif gate_name == "S":
                        tableau_copy.append_s(gate_data[0])
                    else:
                        raise Exception("Unknown Gate")
                steiner_reduce_column(col, row,  topology.to_nx, tableau_copy, apply)
                # Calculate the score of the next state
                best_perms = get_best_cnots(tableau_copy, topology)
                score = best_perms[0][1]
                # Total score is the cost of taking the step + cost of finishing
                print("Cost",(row, col), score+cnot_count)
                label[0, row, col] = score+cnot_count
                # Push the new tableau into the queue
                queue.append((tableau_copy, [r for r in remaining_rows if r != row], [c for c in remaining_cols if c!= col], topology))
            dataset[tensor_key] = label

    return [torch.tensor(key) for key in dataset.keys()], list(dataset.values())


def generate_dataset_ct(nr_samples: int, qubits: List[int],
                        gates: List[int],
                        topo_factory: Callable[[int], Topology] = None, labels_as_described:bool = False, preprocessing_type: PreprocessingType = PreprocessingType.ORIGINAL) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate a dataset defined by labels_as_described and preprocessing_type.
    
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
                if labels_as_described:
                    tableaus, expected_pr_map = generate_data_as_project_description(clifford_tableau, topo, preprocess_type=preprocessing_type)
                else:
                    tableaus, expected_pr_map = generate_data_series_ct(clifford_tableau, topo, preprocessing_type)
                all_tableaus += tableaus
                all_expected_pr_map += expected_pr_map

    return torch.stack(all_tableaus).float(), torch.stack(all_expected_pr_map).float()
