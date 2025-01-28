# 2025 Data Science Project

## General Description

The core idea of the project is to use [this](https://arxiv.org/pdf/2309.08972) paper as a baseline.  
The paper describes how Gaussian elimination can be used to synthesize a Clifford tableau into a quantum circuit.  
In each step of the Gaussian elimination, a pivot row and column need to be chosen, which affects the performance of the algorithm.  
The overall goal of the project is to select a pivot such that the number of CNOT gates, which are considered more costly on quantum devices, is minimized.

The paper is currently implemented in the [PauliOpt](https://github.com/hashberg-io/pauliopt) library.  
The algorithm described in the paper can be executed as follows:

```python
def our_compilation(circuit: Circuit, topology: Topology, n_rep: int):
    """
    :param circuit: Quantum circuit to process
    :param topology: Device topology information
    :param n_rep: Number of repetitions for the algorithm
    :return: Processed circuit data and metadata
    """
    clifford_tableau = CliffordTableau(circuit.n_qubits)
    clifford_tableau = tableau_from_circuit(clifford_tableau, circuit)

    circ_out, _ = synthesize_tableau_perm_row_col(clifford_tableau, topology)
    return {"n_rep": n_rep} | collect_circuit_data(circ_out) | {"method": "normal_heuristic"}
```

## Core Idea

We have adapted the library to allow for a callback that selects the pivot row and column:

```python
perm_iter = iter(permutation) # <- some permutation that was user choosen

def pick_pivot_callback(G, remaining: "CliffordTableau", remaining_rows: list[int], choice_fn=min):
    return next(perm_iter)

qc, _ = synthesize_tableau_perm_row_col(clifford_tableau, topo, pick_pivot_callback=pick_pivot_callback)
```

Your task is to find good heuristics using machine learning techniques.

## Setup

The project uses [pyenv](https://github.com/pyenv/pyenv) combined with pip-tools.  
Follow the instructions in their repository to install it.

To create a new environment, run:

```bash
make clean_setup
```

To install dependencies, run:

```bash
make setup
```

To update dependencies (run [pip-compile](https://github.com/jazzband/pip-tools)), run:

```bash
make update_dependencies
```

There is currently no linter setup. If you want to use linting, feel free to implement it.

## Project Structure

- `nn/`: Contains code to generate a dataset for the neural network-based approach.  
- `rl/`: Includes the reinforcement learning (RL)-based agent and its environment.  
- Top-level Python modules:
  - `nn_train_main.py`: A script to generate datasets.  
  - `nn_eval_main.py`: A script for evaluating the neural network approach.  
  - `rl_main.py`: A script to run the RL algorithm.  
