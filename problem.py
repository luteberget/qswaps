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
                msg = f"Input bits for gate {op.gate}:{problem.gates[op.gate]} do not match {current_assignment_input_bits}"
                return msg
        else:
            a, b = op.edge
            current_assignment[a], current_assignment[b] = current_assignment[b], current_assignment[a]

    return None

