import glob
import json
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Gate
from qswaps import RawProblem, Solution
from qiskit.visualization import circuit_drawer

def to_qiskit_circuit(problem: RawProblem, solution: Solution):
    order = sorted(list(set(x for a, b in problem.topology for x in [a, b])))

    q = QuantumRegister(len(order), "p")
    node_to_q = {n: q for n, q in zip(order, q)}

    circuit = QuantumCircuit(q)

    for op in solution.operations:
        if op.gate is not None:

            #
            # For this to be a specific quantum circuit, these gates need to be mapped to 
            # specific quantum gates.
            #
            circuit.append(
                Gate(name=f"g{op.gate+1}", num_qubits=2, params=[]), [node_to_q[n] for n in op.edge]
            )
        else:
            circuit.swap(*[node_to_q[n] for n in op.edge])

    return circuit


for filename in glob.glob("experiments/layers_nonoptimal/*json"):
    print(f"Reading file: {filename}")
    with open(filename, "r") as f:
        instance = json.load(f)

    problem = RawProblem.from_dict(instance["problem"])
    nonlayered_solution = Solution.from_dict(instance["nonlayered_solution"]["solution"])
    layered_solution = Solution.from_dict(instance["layered_solution"]["solution"])

    nonlayered_circuit = to_qiskit_circuit(problem, nonlayered_solution)
    layered_circuit = to_qiskit_circuit(problem, layered_solution)

    print("## nonlayered")
    print(circuit_drawer(nonlayered_circuit, vertical_compression="low", idle_wires=True))
    print("## layered")
    print(circuit_drawer(layered_circuit, vertical_compression="low", idle_wires=True))
