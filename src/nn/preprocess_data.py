import torch
import numpy as np
from enum import Enum
from networkx import Graph

from pauliopt.clifford.tableau import CliffordTableau

class PreprocessingType(Enum):
    ORIGINAL = "original"
    FROM_PROJECT_DESCRIPTION = "from_project_description"

def remaining_rows_to_reduced_elements(tableau: CliffordTableau, remaining_rows: list[int]) -> np.typing.ArrayLike:
    # Transform remaining rows into disallowed rows
    disallowed_rows = [r for r in range(tableau.n_qubits) if r not in remaining_rows]
    # Calculate the disallowed columns from the disallowed rows
    disallowed_columns = [np.where(tableau.x_matrix[r] == 1)[0][0] for r in disallowed_rows]
    assert all(c < tableau.n_qubits for c in disallowed_columns)

    # Generate the bitmask that shows where the tableau is already done
    reduced_elements = np.ones((tableau.n_qubits, tableau.n_qubits))
    reduced_elements[disallowed_rows, :] = 0.0
    reduced_elements[:, disallowed_columns] = 0.0
    return reduced_elements


def preprocess_data_original(G: Graph, remaining: CliffordTableau, remaining_rows: list[int], choice_fn=min) -> torch.Tensor:
    """
    Preprocesses the data into the following form:

    X: [
        x_matrix,             => A bitmap describing the stabilizers of the tableau
        z_matrix,             => A bitmap describing the destabilizers of the tableau
        reduced_elements      => A bitmap where all reduced rows and cols are marked as zero
       ]

    Args:
        G (Graph): The graph representing the qubit connectivity - currently ignored
        remaining (CliffordTableau): The remaining CliffordTableau to be reduced
        remaining_rows (List[int]): The rows that have not yet been reduced
        choice_fn (fn, optional): A function for the choice heuristic - completely irrelevant. Defaults to min.

    Returns:
        torch.Tensor: The data in the shape described above.
    """
    clifford_tableau = remaining
    reduced_elements = remaining_rows_to_reduced_elements(clifford_tableau, remaining_rows)

    # Glue it all together into a tensor
    tableau_representation = np.stack([
        np.where(clifford_tableau.x_matrix == 0, 0, 1),
        np.where(clifford_tableau.z_matrix == 0, 0, 1),
        reduced_elements
    ], axis=0)
    input_data = torch.from_numpy(tableau_representation)
    return input_data

def preprocess_data_project_description(G: Graph, remaining: CliffordTableau, remaining_rows: list[int], choice_fn=min) -> torch.Tensor:
    """
    The project description stated: 
    - a 2nx2n binary matrix representing the quantum circuit
    - a nxn binary adjacency matrix of a graph that represents how the qubits are connected in the quantum computer
    - n integer vector representing the minimal cost of choosing that qubit. The classification task is to find the index of the smallest value.
    Note that n is technically variable. We hope to have a single classifier for all n, rather than n different classifiers. So the matrices are padded.

    This is not entirely what is created here.
    X: [
        tableau: an nxn matrix with values from range(4) so that it has the same shape as the adjacency matrix
        graph: an nxn binary adjacency matrix of the graph. Currently all-to-all connected,
        reduced_elements: Not in the project description. An nxn binary matrix that can be used to mask the other matrices to zero-pad the matrices. Optional.
    ]

    Args:
        G (Graph): The graph representing the qubit connectivity - currently ignored
        remaining (CliffordTableau): The remaining CliffordTableau to be reduced
        remaining_rows (List[int]): The rows that have not yet been reduced
        choice_fn (fn, optional): A function for the choice heuristic - completely irrelevant. Defaults to min.

    Returns:
        torch.Tensor: The data in the shape described above.
    """
    tableau = remaining.x_matrix + 2*remaining.z_matrix
    reduced_elements = remaining_rows_to_reduced_elements(remaining, remaining_rows)
    graph = np.zeros((remaining.n_qubits, remaining.n_qubits))
    for node1, neighbors in G.adjacency():
        for node2 in neighbors:
            graph[node1, node2] = 1
    # Glue it all together into a tensor
    tableau_representation = np.stack([
        tableau, 
        graph,
        reduced_elements
    ], axis=0)
    input_data = torch.from_numpy(tableau_representation)
    return input_data

PREPROCESSING_SCRIPTS = {
    PreprocessingType.ORIGINAL: preprocess_data_original,
    PreprocessingType.FROM_PROJECT_DESCRIPTION: preprocess_data_project_description,
}