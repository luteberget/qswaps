import networkx as nx
from qiskit.converters import circuit_to_dag
from qiskit_experiments.library import QuantumVolume
from qiskit.compiler import transpile
from qiskit.transpiler import CouplingMap

class Job:

    def __init__(self, id : int, qubit_pair : tuple[int,int], duration : int) -> None:
        if len(qubit_pair) != 2:
            raise Exception("The qubit pair is not a pair.")
        if qubit_pair[0] == qubit_pair[1]:
            raise Exception("Qubits of a gate must be different.")
        self.id = id
        self.pair = qubit_pair
        self.duration = duration

    def __repr__(self):
        return f'{self.id}-({self.pair[0]},{self.pair[1]})-{self.duration}'
    

class Problem:

    def __init__(self, hardware_graph : nx.Graph, jobs : list[Job]) -> None:
        self.swap_duration = 3
        self.hardware_graph = hardware_graph
        self.jobs = jobs
        if len({job.id for job in jobs}) < len(jobs):
            raise Exception("Jobs don't have unique ids.")
        self.logical_qubits = {q for j in self.jobs for q in j.pair}
        if len(self.logical_qubits) > hardware_graph.number_of_nodes():
            raise Exception(f"Number of logical qubits ({len(self.logical_qubits)}) greater " + 
                            f"than the number of physical qubits ({hardware_graph.number_of_nodes()}).")
        self.interdicted_edges = dict()
        for edge in self.hardware_graph.edges:
            self.interdicted_edges[edge] = set()
            for sub_edge in self.hardware_graph.edges(edge):
                self.interdicted_edges[edge].add(sub_edge)


def qc_to_jobs(qc):
    dag = circuit_to_dag(qc)
    two_q_list=dag.collect_2q_runs()
    list_of_jobs = []
    for i, gates in enumerate(two_q_list):
        num_cnots=0
        for j,gate in enumerate(gates):
            if gate.op.name=="cx":
                qubits=[]
                for qubit in gate.qargs:
                    qubits.append(qubit.index)
                num_cnots+=1
        job = Job(i+1, (qubits[0]+1, qubits[1]+1), num_cnots)
        list_of_jobs.append(job)

    return list_of_jobs

def linear_coupling(n):
    hardware_graph = nx.Graph()
    for i in range (1,n):
        hardware_graph.add_edge(i,i+1)
    return CouplingMap.from_line(n),hardware_graph


def gen_instances(num_qubits):
    qubits = tuple(range(num_qubits)) 
    c_map,hardware_graph=linear_coupling(num_qubits)

    qv_exp = QuantumVolume(qubits, seed=420,trials=10)
    qv_circuits=qv_exp.circuits()
    qiskit_circuits_3=[]
    qiskit_circuits_0=[]
    bip_circuits=[]
    qv_graphs=[]
    n_cnots_id=[]
    n_cnots=[]
    for circuit in qv_circuits:
        qc=transpile(circuit,optimization_level=0,basis_gates=["cx","u3"])
        n_cnots_id.append(qc.count_ops()["cx"])
        n_cnots.append(qc.count_ops()["cx"])
        qv_graphs.append(qc_to_jobs(qc))
        qiskit_circuits_0.append(transpile(circuit,optimization_level=0,basis_gates=["cx","u3"],coupling_map=c_map))
        qiskit_circuits_3.append(transpile(circuit,optimization_level=3,basis_gates=["cx","u3"],coupling_map=c_map))

    n_cnots_qiskit_0=[]
    n_cnots_qiskit_3=[]
    solutions=[]
    for i,graph in enumerate(qv_graphs):
        prob=Problem(hardware_graph, graph)
        # print(graph)
        # print(prob.logical_qubits)
        yield prob


if __name__ == '__main__':
    print(gen_instances(4))
