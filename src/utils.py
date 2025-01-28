import numpy as np
from pauliopt.circuits import Circuit
from pauliopt.clifford.tableau import CliffordTableau


def tableau_from_circuit(tableau: CliffordTableau, circ: Circuit) -> CliffordTableau:
    """
    Convert a Clifford circuit into a CliffordTableau
    :param tableau:
    :param circ:
    :return:
    """
    for gate in circ.gates:
        tableau.append_gate(gate)
    return tableau


def random_clifford_circuit(nr_gates=20, nr_qubits=4, gate_choice=None) -> Circuit:
    """
    Generate a Clifford circuit.

    :param nr_gates:
    :param nr_qubits:
    :param gate_choice: Subset of ["CX", "H", "S", "V", "CY", "CZ", "Sdg", "Vdg", "X", "Y", "Z"]
    :return:
    """
    qc = Circuit(nr_qubits)
    if gate_choice is None:
        gate_choice = ["CX", "H", "S", "V", "CY", "CZ", "Sdg", "Vdg", "X", "Y", "Z"]
    for _ in range(nr_gates):
        gate_t = np.random.choice(gate_choice)
        qubit = np.random.choice([i for i in range(nr_qubits)])
        if gate_t == "CX":
            target = np.random.choice([i for i in range(nr_qubits) if i != qubit])
            qc.cx(qubit, target)
        elif gate_t == "CY":
            target = np.random.choice([i for i in range(nr_qubits) if i != qubit])
            qc.cy(qubit, target)
        elif gate_t == "CZ":
            target = np.random.choice([i for i in range(nr_qubits) if i != qubit])
            qc.cz(qubit, target)
        elif gate_t == "H":
            qc.h(qubit)
        elif gate_t == "S":
            qc.s(qubit)
        elif gate_t == "V":
            qc.v(qubit)
        elif gate_t == "Vdg":
            qc.vdg(qubit)
        elif gate_t == "Sdg":
            qc.sdg(qubit)
        elif gate_t == "X":
            qc.x(qubit)
        elif gate_t == "Y":
            qc.y(qubit)
        elif gate_t == "Z":
            qc.z(qubit)
        else:
            raise Exception(f"Unknown Gate: {gate_t}")

    return qc


def random_hscx_circuit(nr_gates=20, nr_qubits=4):
    """
    Generate a Clifford circuit with H, S and CX gates.

    *Note:* CX = CNOT
    :param nr_gates:
    :param nr_qubits:
    :return:
    """
    gate_choice = ["CX", "H", "S"]
    return random_clifford_circuit(
        nr_gates=nr_gates, nr_qubits=nr_qubits, gate_choice=gate_choice
    )
