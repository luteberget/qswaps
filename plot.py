from qswaps import RawProblem, Solution
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister

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
            circuit.cx(*[node_to_q[n] for n in op.edge], label=f"g{op.gate+1}")
        else:
            circuit.swap(*[node_to_q[n] for n in op.edge])
    print(circuit)

