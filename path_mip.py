from __future__ import annotations
from dataclasses import dataclass
import time
from typing import List, Tuple
import gurobipy as grb
from gen_problems import gen_instances
from plot import plot_solution
from postprocess import remove_unnecessary_swaps
from problem import Operation, RawGate, RawProblem, Solution
from qswaps import solve as solve_sat


def solve_full_pathmip(
    raw_problem: RawProblem,
    gate_execution_time: int,
    use_fixed_layers: bool = False,
    max_depth: int = 20,
) -> Tuple[int, Solution]:
    """
    Full (exponential-size) generation of the bit-path MIP formulation.
    """

    print("Path-MIP solving", raw_problem)

    assert gate_execution_time == 1
    assert not use_fixed_layers

    logical_bits = set(bit for g in raw_problem.gates for bit in [g.line1, g.line2])
    physical_bits = set(bit for a, b in raw_problem.topology for bit in [a, b])

    while len(logical_bits) < len(physical_bits):
        logical_bits.add(f"dummy_{len(logical_bits)+1}")

    print(f"Logical bits {logical_bits}")
    print(f"Physical bits {physical_bits}")

    for n_steps in range(1, max_depth + 1):
        model = grb.Model()

        constraints = {}

        # CONSTRAINT 1
        # Select one path per logical bit
        for lb in logical_bits:
            name = f"lb{lb}"
            constraints[name] = model.addConstr(grb.LinExpr(0) == 1, name)

        # CONSTRAINT 2
        # Non-overlapping bit paths
        for pb in physical_bits:
            for t in range(n_steps + 1):
                # Constraint that models that exactly one logical bit resides
                # at this physical bit at this time
                name = f"{pb}_t{t}"
                constraints[name] = model.addConstr(grb.LinExpr(0) == 1, name)

        # CONSTRAINT 3
        # Gate constraints
        for gi, _gate in enumerate(raw_problem.gates):
            for a, b in raw_problem.topology:
                for t in range(n_steps):
                    # Constraint that balances use of this gate at this edge at this time
                    name = f"g{gi}_({a},{b})_t{t}"
                    constraints[name] = model.addConstr(grb.LinExpr(0) == 0, name)

        # CONSTRAINT 4
        # Swap constraints
        for a, b in raw_problem.topology:
            for t in range(n_steps):
                # Constraint that balances use of swapping this edge at this time
                name = f"sw_({a},{b})_t{t}"
                constraints[name] = model.addConstr(grb.LinExpr(0) == 0, name)

        #
        # Now, generate variables that represents each bit's path through the circuit.
        #

        @dataclass
        class GateRef:
            gate_idx: int | None
            edge: Tuple[str, str]

        @dataclass
        class PathNode:
            pb: str
            gate_seq_idx: int
            time: int
            prev_gate: GateRef | None
            prev_node: PathNode | None

        def gen_paths():
            for lb in logical_bits:
                # For a single logical bit, there is a fixed sequence of
                # gates it needs to go through. The PathNode class holds the
                # index of the next gate to place.
                gate_seq = [
                    gate_idx
                    for gate_idx, gate in enumerate(problem.gates)
                    if gate.line1 == lb or gate.line2 == lb
                ]

                # Start with placing the LB in each physical bit.
                queue = [PathNode(pb, 0, 0, None, None) for pb in physical_bits]

                while len(queue) > 0:
                    node = queue.pop()

                    # Check for termination
                    if node.gate_seq_idx >= len(gate_seq) and node.time == n_steps:
                        # We have placed all the gates. Report the solution.
                        yield (lb, node)
                        continue

                    # Extend this path.
                    # 1st choice: do nothing
                    if node.time + 1 <= n_steps:
                        queue.append(
                            PathNode(
                                node.pb, node.gate_seq_idx, node.time + 1, None, node
                            )
                        )

                    # 2nd choice: place next gate
                    if (
                        node.gate_seq_idx < len(gate_seq)
                        and node.time + gate_execution_time <= n_steps
                    ):
                        for a, b in problem.topology:
                            if a == node.pb or b == node.pb:
                                queue.append(
                                    PathNode(
                                        node.pb,
                                        node.gate_seq_idx + 1,
                                        node.time + gate_execution_time,
                                        GateRef(gate_seq[node.gate_seq_idx], (a, b)),
                                        node,
                                    )
                                )
                    # 3rd chocie: place a swap
                    if node.time + problem.swap_time <= n_steps:
                        for a, b in problem.topology:
                            if a == node.pb or b == node.pb:
                                other_pb = b if a == node.pb else a
                                queue.append(
                                    PathNode(
                                        other_pb,
                                        node.gate_seq_idx,
                                        node.time + problem.swap_time,
                                        GateRef(None, (a, b)),
                                        node,
                                    )
                                )

        @dataclass
        class PathData:
            lb: str
            initial_pb: str
            gates: List[(int, GateRef)]

        paths: Tuple[grb.Var, PathData] = []
        for path_lb, path_end_node in gen_paths():
            #
            # Convert a bit path into a MIP column
            #

            node = path_end_node
            next_time = node.time + 1

            column = []

            # CONSTRAINT 1 (select one path per bit)
            column.append((1, f"lb{path_lb}"))

            path_initial_pb = None
            path_gates = []

            while node is not None:
                assert node.time == 0 or node.prev_node is not None

                for t in range(node.time, next_time):
                    # CONSTRAINT 2 (non-overlapping bits), at every time step
                    column.append((1, f"{node.pb}_t{t}"))

                    if t == 0:
                        # Save the initial assignment for use when reconstructing a solution from paths.
                        path_initial_pb = node.pb

                # Work backwards through the bit path, adding swaps and gates constraints.
                if node.prev_gate is not None:
                    gate_t = node.prev_node.time
                    gref = node.prev_gate
                    a, b = gref.edge

                    path_gates.append((gate_t, gref))

                    dir = 1 if node.pb == a else -1
                    if gref.gate_idx is not None:
                        # CONSTRAINT 3:
                        # We are using the gate at this physical edge (in this direction)
                        column.append((dir, f"g{gref.gate_idx}_({a},{b})_t{gate_t}"))
                    else:
                        # CONSTRAINT 4:
                        # We are using the swap at this physical edge (in this direction)
                        column.append((dir, f"sw_({a},{b})_t{gate_t}"))

                next_time = node.time
                node = node.prev_node

            column.reverse()

            print(f"Path for {path_lb} uses {column}")

            var = model.addVar(
                obj=0.0,
                column=grb.Column(
                    [coeff for coeff, _ in column],
                    [constraints[ref] for _, ref in column],
                ),
                name=f"path_{path_lb}_{len(paths)}",
                vtype=grb.GRB.BINARY,
            )

            paths.append((var, PathData(path_lb, path_initial_pb, path_gates)))

        print(f"Generated MIP with depth {n_steps} and {len(paths)} paths.")
        model.write("test.lp")
        model.optimize()

        if model.status == grb.GRB.INFEASIBLE:
            # Might need more time steps.
            pass

        elif model.status == grb.GRB.OPTIMAL:
            #
            # Reconstruct solution from the selected path variables.
            #

            initial_assignment = {}
            gate_map = {}
            swap_set = set()
            swap_blocked_set = set()
            operations_set = set()
            for var, pathdata in paths:
                if var.X >= 0.5:
                    print(f"Selected path {pathdata} ")
                    for gate in pathdata.gates:
                        print("  gate", gate)

                    assert pathdata.initial_pb not in initial_assignment
                    initial_assignment[pathdata.initial_pb] = pathdata.lb
                    for t, gref in pathdata.gates:
                        print("     adding ", t, gref)
                        gref: GateRef = gref

                        operations_set.add((t, gref.gate_idx, gref.edge))

                        # Check gate consistency
                        if gref.gate_idx is not None:
                            if gref.gate_idx not in gate_map:
                                gate_map[gref.gate_idx] = (t, gref.edge)
                                assert (t, gref.edge) not in swap_blocked_set
                                swap_blocked_set.add((t, gref.edge))

                            assert gate_map[gref.gate_idx] == (t, gref.edge)

                        # Check swap consistency
                        if gref.gate_idx is None:
                            if (t, gref.edge) not in swap_set:
                                swap_set.add((t, gref.edge))
                                for dt in range(problem.swap_time):
                                    assert ((t + dt), gref.edge) not in swap_blocked_set
                                    swap_blocked_set.add(((t + dt), gref.edge))

                    print(f"     decoded path to {initial_assignment}   {operations_set}")

            operations = [Operation(*x) for x in operations_set]
            operations.sort(key=lambda op: op.time)

            solution = Solution(initial_assignment, operations)
            solution = remove_unnecessary_swaps(problem, solution)

            return (n_steps, solution)

        else:
            raise Exception(f"Unknown grb status {model.status}")

    raise Exception(f"Could not solve instance")


if __name__ == "__main__":
    swaptime = 3
    for size in [4]:
        for i, instance in enumerate(gen_instances(size, seed=420, num_instances=50)):
            instance_name = f"sz{size}_swt{swaptime}_{i}"
            print(f"# INSTANCE {instance_name}")

            problem = RawProblem(
                [RawGate(f"l{j.pair[0]}", f"l{j.pair[1]}", 1) for j in instance.jobs],
                [(f"p{a}", f"p{b}") for a, b in instance.hardware_graph.edges],
                swaptime,
            )

            depth, sat_solution = solve_sat(problem, 0 if swaptime == 1 else 1)
            print(plot_solution(problem, sat_solution))

            t0 = time.time()
            print(f"Solving path-mip with max_depth={depth}")
            depth2, sol2 = solve_full_pathmip(
                problem, 0 if swaptime == 1 else 1, max_depth=depth
            )
            t1 = time.time()
            print(plot_solution(problem, sol2))

            print(f"Solved {instance_name} in {t1-t0:.2f}s.")
