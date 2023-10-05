from qiskit.circuit import Gate
from problem import RawProblem, Solution
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit.quantum_info.operators import Operator, Pauli
from qiskit.visualization import circuit_drawer


def to_qiskit_circuit(problem: RawProblem, solution: Solution, include_bit_assignment: bool = False):


    order = sorted(list(set(x for a, b in problem.topology for x in [a, b])))

    register = QuantumRegister(len(order), "p")
    register_map = {n: q for n, q in zip(order, register)}

    circuit = QuantumCircuit(register)

    if include_bit_assignment:
        for n, q in register_map.items():
            circuit.x(q, solution.bit_assignment[n])

    for op in solution.operations:
        if op.gate is not None:
            #
            # For this to be a specific quantum circuit, these gates need to be mapped to
            # specific quantum gates.
            #
            circuit.append(Gate(name=f"g{op.gate+1}", num_qubits=2, params=[]), [register_map[n] for n in op.edge])
        else:
            a, b = op.edge
            circuit.swap(register_map[a], register_map[b])

    return circuit


def plot_solution(problem: RawProblem, solution: Solution):
    circuit = to_qiskit_circuit(problem, solution, include_bit_assignment=True)
    txt = str(circuit_drawer(circuit, vertical_compression="low", idle_wires=True))
    return txt
