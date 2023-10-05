from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class RawGate:
    line1: str
    line2: str
    duration: int


@dataclass_json
@dataclass
class RawProblem:
    gates: List[RawGate]
    topology: List[Tuple[str, str]]
    swap_time: int


@dataclass_json
@dataclass
class Operation:
    time: int
    gate: Optional[int]
    edge: Tuple[str, str]


@dataclass_json
@dataclass
class Solution:
    bit_assignment: Dict[str, str]
    operations: List[Operation]


def verify_solution_bit_dependencies(problem: RawProblem, solution: Solution) -> Optional[str]:
    current_assignment = dict(solution.bit_assignment)

    for op in solution.operations:
        if op.gate is not None:
            current_assignment_input_bits = set((current_assignment[n] for n in op.edge))
            correct_input_bits = set([problem.gates[op.gate].line1, problem.gates[op.gate].line2])
            assert len(correct_input_bits) == 2
            if current_assignment_input_bits != correct_input_bits:
                return f"Input bits for gate {op.gate}:{problem.gates[op.gate]} do not match {current_assignment_input_bits}"
        else:
            a, b = op.edge
            current_assignment[a], current_assignment[b] = current_assignment[b], current_assignment[a]

    return None


def remove_unnecessary_swaps(problem: RawProblem, solution: Solution) -> Solution:
    assert verify_solution_bit_dependencies(problem, solution) is None

    # Remove swaps in the beginning of circuit
    while True:
        removed = False
        untouched_bits = set(bit for a, b in problem.topology for bit in [a, b])
        for op_idx, op in enumerate(solution.operations):
            if op.gate is None:
                if op.edge[0] in untouched_bits and op.edge[1] in untouched_bits:
                    # This swap is unnecessary, as noone used this hardware edge before,
                    # and we can just swap the logical input bits.

                    new_initial_assignment = dict(solution.bit_assignment)

                    # Swap the initial assignemtn
                    new_initial_assignment[op.egde[0]], new_initial_assignment[op.egde[1]] = (
                        new_initial_assignment[op.egde[1]],
                        new_initial_assignment[op.egde[0]],
                    )

                    # Remove the swap gate
                    new_ops = list(solution.operations)
                    new_ops.pop(op_idx)
                    solution = Solution(new_initial_assignment, new_ops)
                    print("Removed unnecessary swap (by initial assignment heuristic).")
                    assert verify_solution_bit_dependencies(problem, solution) is None
                    removed = True
                    break
            untouched_bits.remove(op.edge[0])
            untouched_bits.remove(op.edge[1])

        if not removed:
            break

    # Remove swaps that have no effect
    while True:
        removed = False
        for op_idx, op in enumerate(solution.operations):
            if op.gate is None:
                new_ops = list(solution.operations)
                new_ops.pop(op_idx)
                removed_swap = Solution(solution.bit_assignment, new_ops)
                if verify_solution_bit_dependencies(problem, removed_swap) is None:
                    print("Removed unnecessary swap (by end swap heuristic).")
                    solution = removed_swap
                    assert verify_solution_bit_dependencies(problem, solution) is None
                    removed = True
                    break

        if not removed:
            break

    return solution
