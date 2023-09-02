#
# Cfr.
#  https://arxiv.org/pdf/2306.08629.pdf
#  https://arxiv.org/pdf/2208.13679.pdf
#

from dataclasses import dataclass
from typing import List, Tuple
from pysat.formula import IDPool
from pysat.solvers import Cadical153


@dataclass
class RawGate:
    line1: str
    line2: str
    duration: int


@dataclass
class RawProblem:
    gates: List[RawGate]
    topology: List[Tuple[str, str]]
    swap_time: int


def solve(raw_problem: RawProblem):
    vpool = IDPool()
    solver = Cadical153()

    def pairwise_atmostone(solver, vars):
        for v_i in range(len(vars)):
            for v_j in range(v_i + 1, len(vars)):
                solver.add_clause([-vars[v_i], -vars[v_j]])

    logical_bits = set(bit for g in raw_problem.gates for bit in [g.line1, g.line2])
    physical_bits = set(bit for a, b in raw_problem.topology for bit in [a, b])

    assert len(logical_bits) == len(physical_bits)

    n_states = 0

    def add_state():
        nonlocal n_states
        t = n_states

        #
        # STATE
        #
        # Each state assigns logical bits to physical bits

        for lb in logical_bits:
            vars = [vpool.id(f"t{t}_{lb}@{pb}") for pb in physical_bits]

            # Constraint: each lb is mapped to a pb
            solver.add_clause(vars)

            # Constraints: each lb is mapped to at most one pb
            pairwise_atmostone(solver, vars)

        for pb in physical_bits:
            vars = [vpool.id(f"t{t}_{lb}@{pb}") for lb in logical_bits]

            # Constraint: each pb hosts at most one lb
            pairwise_atmostone(solver, vars)

        #
        # ACTIONS
        #
        # GATES:
        # Execute gates at this time step (on a physical edge)
        for gate_idx, gate in enumerate(raw_problem.gates):
            execute = vpool.id(f"t{t}_g{gate_idx}!")
            executed_after = vpool.id(f"t{t}_g{gate_idx}")

            # the gate has not been executed before (ladder encoding)
            if t == 0:
                solver.add_clause([-execute, executed_after])
                solver.add_clause([-executed_after, execute])
            if t > 0:
                executed_before = vpool.id(f"t{t-1}_g{gate_idx}")
                solver.add_clause([-executed_before, executed_after])
                solver.add_clause([-execute, -executed_before])
                solver.add_clause([-execute, executed_after])
                solver.add_clause([executed_before, -executed_after, execute])

            # Find predecessors
            pred_a = None
            pred_b = None
            for gate_j in reversed(range(0, gate_idx)):
                other_gate = raw_problem.gates[gate_j]
                if (
                    other_gate.line1 == gate.line1 or other_gate.line2 == gate.line1
                ) and pred_a == None:
                    pred_a = gate_j
                if (
                    other_gate.line1 == gate.line2 or other_gate.line2 == gate.line2
                ) and pred_b == None:
                    pred_b = gate_j

            # The gate's predecessors have already been executed
            preds = set(x for x in [pred_a, pred_b] if x is not None)
            # print("gate ", gate_idx, "preds", preds)
            for pred in preds:
                if t == 0:
                    # If we have predecessors, and t=0, then we cannot execute.
                    # print("cannot execute", f"t{t}_g{gate_idx}!")
                    solver.add_clause([-execute])
                else:
                    solver.add_clause([-execute, vpool.id(f"t{t-1}_g{pred}")])

            locations = []

            for edge in raw_problem.topology:
                # either:
                #  - gate.line1 maps to edge[0] and gate.line2 maps to edge[1] 
                #  - or the other way around
                # it doesn't matter, because we can just flip it around to match.
                # The names of the bits are anyway tracking the correct information flow,
                # so we don't need to explicitly model the direction of the edge

                execute_at = vpool.id(f"t{t}_g{gate_idx}@({edge[0]},{edge[1]})!")
                locations.append(execute_at)

                # At the current time, the logical inputs to the gate must be mapped
                # to the physical bits.
                for lb in [gate.line1, gate.line2]:
                    cl = [-execute_at, vpool.id(f"t{t}_{lb}@{edge[0]}"), vpool.id(f"t{t}_{lb}@{edge[1]}")]
                    solver.add_clause(cl)

            # If executing the gate, it has to be executed at one of the possible physical locations
            # print("gate ", gate_idx, "locations", locations)
            solver.add_clause([-execute] + locations)

        #
        # SWAPS:
        # Bits can move physically, if a swap gate is present.
        #

        swap_start = t - raw_problem.swap_time
        if t >= 1:
            for pb in physical_bits:
                # possible swaps involving this bit
                adjacent_edges = {
                    pb1 if pb == pb2 else pb2: (pb1, pb2)
                    for pb1, pb2 in raw_problem.topology
                    if pb1 == pb or pb2 == pb
                }

                for other_pb in (b for b in physical_bits if b != pb):
                    swap_var = None
                    if swap_start >= 0 and other_pb in adjacent_edges:
                        other_pb1, other_pb2 = adjacent_edges[other_pb]
                        swap_var = f"t{swap_start}_sw({other_pb1},{other_pb2})"

                    for lb in logical_bits:
                        # Explanatory frame axiom: if the position changes, we have done the swap
                        # (there is only one possible swap for a pair of consecutive positions)

                        solver.add_clause(
                            [
                                -vpool.id(f"t{t-1}_{lb}@{other_pb}"),
                                -vpool.id(f"t{t}_{lb}@{pb}"),
                            ]
                            + ([vpool.id(swap_var)] if swap_var is not None else [])
                        )

                        # Action post-condition (effect): if the swap was selected,
                        # the bits will have to move.

                        if swap_var is not None:
                            solver.add_clause(
                                [
                                    -vpool.id(swap_var),
                                    -vpool.id(f"t{t-1}_{lb}@{other_pb}"),
                                    vpool.id(f"t{t}_{lb}@{pb}"),
                                ]
                            )

        # If we did swap, then we cannot also execute gates or other swaps
        # from time `t-swap_time` up to time `t`.
        if swap_start >= 0:
            for this_edge in raw_problem.topology:
                adjacent_edges = [
                    other_edge
                    for other_edge in raw_problem.topology
                    if other_edge != this_edge
                    and len(set(other_edge).intersection(set(this_edge))) > 0
                ]

                this_swap = vpool.id(f"t{swap_start}_sw({this_edge[0]},{this_edge[1]})")

                for other_time in range(swap_start, t):
                    for other_edge in adjacent_edges + [this_edge]:
                        #
                        # Gate incompatibility

                        for gate_idx, _gate in enumerate(raw_problem.gates):
                            other_gate = vpool.id(
                                f"t{other_time}_g{gate_idx}@({other_edge[0]},{other_edge[1]})!"
                            )

                            solver.add_clause([-this_swap, -other_gate])

                        #
                        # Swap incompatibility
                        other_swap = vpool.id(
                            f"t{other_time}_sw({other_edge[0]},{other_edge[1]})"
                        )

                        if this_swap != other_swap:
                            solver.add_clause([-this_swap, -other_swap])

        n_states += 1

    add_state()  # Need at least one state for the assumptions to be non-trivially satisfied

    while True:
        print(f"solving with {n_states} states...")

        t = n_states - 1
        all_gates_executed = [
            f"t{t}_g{gate_idx}" for gate_idx, _ in enumerate(raw_problem.gates)
        ]

        # for i, v in vpool.id2obj.items():
        #     print("var ", i, ". ", v)

        print("clauses", solver.nof_clauses(), "vars", solver.nof_vars())
        # print(solver.accum_stats())
        # print("assumptions", all_gates_executed)


        status = solver.solve([vpool.id(v) for v in all_gates_executed])
        # print(f"done status={status}")

        if status:
            model = set(solver.get_model())

            # for i, v in vpool.id2obj.items():
            #     print("var ", i, ". ", v,"=", i in model)
            #     assert i in model or -i in model
            # print(model)


            for t in range(0,n_states):
                print(f"@t={t}")
                for gate_idx,gate in enumerate(raw_problem.gates):
                    if vpool.id(f"t{t}_g{gate_idx}!") in model:
                        for e in raw_problem.topology:
                            if vpool.id(f"t{t}_g{gate_idx}@({e[0]},{e[1]})!") in model:
                                print(f"  g{gate_idx+1} ({gate.line1},{gate.line2}) at ({e[0]},{e[1]})")

                for e in raw_problem.topology:
                    if vpool.id(f"t{t}_sw({e[0]},{e[1]})") in model:
                        print(f"  swap ({e[0]},{e[1]})")
            print("SAT")
            break
        else:
            if n_states >= 100:
                raise Exception("UNSAT")
            else:
                add_state()


if __name__ == "__main__":

    

    example1 = RawProblem(
        gates=[RawGate("l1", "l2", 1), RawGate("l3", "l4", 1), RawGate("l1", "l3", 1), RawGate("l2", "l4", 1)],
        topology=[("p1", "p2"), ("p2", "p3"), ("p3","p4")],
        swap_time=3,
    )

    example2 = "1-(2,4)-3, 2-(3,1)-3, 3-(3,4)-3, 4-(2,1)-3, 5-(2,4)-3, 6-(1,3)-3, 7-(3,2)-3, 8-(4,1)-3"
    print([parts for g in example2.split(", ") for parts in g.split("-")])
    
    example2 = RawProblem(
        gates=[
            RawGate(
                line1="l" + g.split("-")[1][1:].split(",")[0],
                line2="l" + g.split("-")[1][:-1].split(",")[1],
                duration=1,
            )
            for g in example2.split(", ")
        ],
        topology=[("p1", "p2"), ("p2", "p3"), ("p3","p4")],
        swap_time=3
    )

    print(example2)
    solve(example2)
