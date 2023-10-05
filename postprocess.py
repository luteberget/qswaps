from plot import plot_solution
from problem import *


def remove_unnecessary_swaps(problem: RawProblem, solution: Solution) -> Solution:
    assert verify_solution_bit_dependencies(problem, solution) is None
    physical_bits = set(bit for a, b in problem.topology for bit in [a, b])

    # Remove double swaps
    while True:
        removed = False
        pb_last_op = {pb: None for pb in physical_bits}
        for op_idx, op in enumerate(solution.operations):
            a, b = op.edge

            if (
                op.gate is None
                and pb_last_op[a] == pb_last_op[b]
                and pb_last_op[a] is not None
                and solution.operations[pb_last_op[a]].gate is None
            ):
                # We are swapping when the last operation on these bits was the same swap operation.
                # This is unnecessary.

                new_ops = list(solution.operations)
                new_ops.pop(op_idx)
                new_ops.pop(pb_last_op[a])

                print(plot_solution(problem, solution))
                solution = Solution(solution.bit_assignment, new_ops)
                print(plot_solution(problem, solution))
                print("Removed unnecessary swaps (by double-swap heuristic).")
                assert verify_solution_bit_dependencies(problem, solution) is None

                removed = True
                break

            pb_last_op[a] = op_idx
            pb_last_op[b] = op_idx
        if not removed:
            break

    # Remove swaps in the beginning of circuit
    while True:
        removed = False
        untouched_bits = set(physical_bits)
        for op_idx, op in enumerate(solution.operations):
            a, b = op.edge
            if op.gate is None:
                if a in untouched_bits and b in untouched_bits:
                    # This swap is unnecessary, as noone used this hardware edge before,
                    # and we can just swap the logical input bits.

                    new_assn = dict(solution.bit_assignment)

                    # Swap the initial assignemtn
                    new_assn[a], new_assn[b] = new_assn[b], new_assn[a]

                    # Remove the swap gate
                    new_ops = list(solution.operations)
                    new_ops.pop(op_idx)
                    print(plot_solution(problem, solution))
                    solution = Solution(new_assn, new_ops)
                    print(plot_solution(problem, solution))
                    print("Removed unnecessary swap (by initial assignment heuristic).")
                    assert verify_solution_bit_dependencies(problem, solution) is None
                    removed = True
                    break

            if a in untouched_bits:
                untouched_bits.remove(a)
            if b in untouched_bits:
                untouched_bits.remove(b)
            if len(untouched_bits) < 2:
                break

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
                    print(plot_solution(problem, solution))
                    solution = removed_swap
                    print(plot_solution(problem, solution))

                    print("Removed unnecessary swap (by end swap heuristic).")
                    assert verify_solution_bit_dependencies(problem, solution) is None

                    removed = True
                    break

        if not removed:
            break

    return solution
