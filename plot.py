from qiskit.circuit import Gate
from qswaps import RawProblem, Solution
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit.quantum_info.operators import Operator, Pauli
from qiskit.visualization import circuit_drawer

def plot_solution(problem :RawProblem, solution :Solution):
    #
    # Assuming linear topology in the topology nodes' sorted order!
    #

    order = sorted(list(set(x for a,b in problem.topology for x in [a,b])))
    q = QuantumRegister(len(order),'p')
    print(q[0].register)
    node_to_q = { n:q for n,q in zip(order,q) }
    circuit = QuantumCircuit(q)

    for n,q in node_to_q.items():
        circuit.x(q, solution.bit_assignment[n])

    for op in solution.operations:
        if op.gate is not None:
            circuit.append(Gate(name=f"g{op.gate+1}", num_qubits=2, params=[]), [node_to_q[n] for n in op.edge])
        else:
            circuit.swap(*[node_to_q[n] for n in op.edge])

    txt = str(circuit_drawer(circuit, vertical_compression="low", idle_wires=True))
    return txt

