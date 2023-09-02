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
                solver.add_clause([-execute, executed_before])
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
            preds  =[x for x in [pred_a, pred_b] if x is not None]
            print("gate ", gate_idx, "preds", preds)
            for pred in preds:
                if t == 0:
                    # If we have predecessors, and t=0, then we cannot execute.
                    solver.add_clause([-execute])
                else:
                    solver.add_clause([-execute, vpool.id(f"t{t-1}_g{pred}")])

            locations = []
            for pb1, pb2 in (
                e for a, b in raw_problem.topology for e in [(a, b), (b, a)]
            ):
                execute_at = vpool.id(f"t{t}_g{gate_idx}@({pb1},{pb2})!")
                locations.append(execute_at)

                # At the current time, the logical inputs to the gate must be mapped
                # to the physical bits.

                for lb,pb in [(gate.line1, pb1), (gate.line2, pb2)]:
                    cl = [-execute_at, vpool.id(f"t{t}_{lb}@{pb}")]
                    print("execute_at clause:",cl)
                    solver.add_clause(cl)

            # if executing the gate, it has to be executed at one of the possible physical locations
            print("gate ", gate_idx, "locations", locations)
            solver.add_clause([-execute] + locations)

        #
        # SWAPS:
        # Bits can be moved physically, but it takes time.

        for pb in physical_bits:
            # possible swaps involving this bit
            adjacent_edges = [
                (pb1 if pb == pb2 else pb2, (pb1, pb2))
                for pb1, pb2 in raw_problem.topology
                if pb1 == pb or pb2 == pb
            ]

            # non-adjacent bits are unreachable
            if t >= 1:
                for unreachable_pb in physical_bits:
                    if unreachable_pb == pb or unreachable_pb in (
                        a for a, _ in adjacent_edges
                    ):
                        continue

                    for lb in logical_bits:
                        solver.add_clause(
                            [
                                -vpool.id(f"t{t-1}_{lb}_{unreachable_pb}"),
                                -vpool.id(f"t{t}_{lb}_{pb}"),
                            ]
                        )

            # adjacent bits are reachable if we did a swap
            for other_pb, (pb1, pb2) in adjacent_edges:
                if t >= raw_problem.swap_time:
                    did_swap = vpool.id(f"t{t-raw_problem.swap_time}_sw({pb1},{pb2})")
                    for lb in logical_bits:
                        solver.add_clause(
                            [
                                -vpool.id(f"t{t-1}_{lb}_{other_pb}"),
                                -vpool.id(f"t{t}_{lb}_{pb}"),
                                did_swap,
                            ]
                        )

                elif t >= 1:
                    # have not have time to swap yet
                    for lb in logical_bits:
                        solver.add_clause(
                            [
                                -vpool.id(f"t{t-1}_{lb}_{other_pb}"),
                                -vpool.id(f"t{t}_{lb}_{pb}"),
                            ]
                        )

            # If we did swap, the bits are moved after `swap_time`.
            if t >= raw_problem.swap_time:
                for other_pb, (pb1, pb2) in adjacent_edges:
                    for lb in logical_bits:
                        solver.add_clause(
                            [
                                -vpool.id(
                                    f"t{t-raw_problem.swap_time}_sw({pb1},{pb2})"
                                ),
                                -vpool.id(f"t{t-1}_{lb}@{pb}"),
                                vpool.id(f"t{t}_{lb}@{other_pb}"),
                            ]
                        )

        # If we did swap, then we cannot also execute gates or other swaps 
        # from time `t-swap_time` up to time `t`.
        if t >= raw_problem.swap_time:
            for pb1, pb2 in raw_problem.topology:
                other_edges = [
                    (a, b)
                    for a, b in raw_problem.topology
                    if (a, b) != (pb1, pb2)
                    and len(set([a, b]).intersection(set([pb1, pb2]))) > 0
                ]

                running_swap = vpool.id(f"t{t-raw_problem.swap_time}_sw({pb1},{pb2})")
                for conflicting_time in range(t-raw_problem.swap_time+1, t+1):

                    # Don't run conflicting swaps
                    for other_pb1,other_pb2 in other_edges:
                        solver.add_clause([
                            -running_swap,
                            -vpool.id(f"t{conflicting_time}_sw({other_pb1},{other_pb2})")
                        ])

                    # Don't run any gates on these physical bits
                    for gate_idx,gate in enumerate(raw_problem.gates):
                        solver.add_clause([
                            -running_swap,
                            -vpool.id(f"t{conflicting_time}_g{gate_idx}@({pb1},{pb2})!")
                        ])

        n_states += 1

    add_state() # Need at least one state for the assumptions to be non-trivially satisfied

    while True:
        print(f"solving with {n_states} states...")
        
        t = n_states-1
        all_gates_executed = [f"t{t}_g{gate_idx}" for gate_idx,_ in enumerate(raw_problem.gates)]

        for i,v in vpool.id2obj.items():
            print("var ", i, ". ", v)

        print("clauses", solver.nof_clauses(), "vars", solver.nof_vars())
        print(solver.accum_stats())
        print("assumptions", all_gates_executed)
        status = solver.solve([vpool.id(v) for v in all_gates_executed])
        print(f"done status={status}")

        if status:
            print("SAT")
            break
        else:
            if n_states >= 2:
                raise Exception("UNSAT")
            else:
                add_state()


example1 = RawProblem(
    gates=[RawGate("l1", "l2", 1), RawGate("l2","l3",1)], topology=[("p1", "p2"), ("p2","p3"), ("p3","p4")], swap_time=3
)

solve(example1)
