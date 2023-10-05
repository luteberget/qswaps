from collections import defaultdict
import dataclasses
from problem import verify_solution_bit_dependencies
import qswaps
from qswaps import RawProblem, RawGate, solve
from plot import plot_solution
from gen_problems import gen_instances
import qswap_beam
import time
import json


depths = defaultdict(list)

for size in [4, 5, 6, 7]:
    for swaptime in [3]:
        for i, instance in enumerate(gen_instances(size, seed=420, num_instances=50)):
            instance_name = f"sz{size}_swt{swaptime}_{i}"
            print(f"# INSTANCE {instance_name}")

            problem = RawProblem(
                [RawGate(f"l{j.pair[0]}", f"l{j.pair[1]}", 1) for j in instance.jobs],
                [(f"p{a}", f"p{b}") for a, b in instance.hardware_graph.edges],
                swaptime,
            )

            depth1, sat_solution = solve(problem, 0 if swaptime == 1 else 1)
            depth2, sat_layers_solution = solve(problem, 0 if swaptime == 1 else 1, use_fixed_layers=True)


            assert verify_solution_bit_dependencies(problem, sat_solution) is None
            assert verify_solution_bit_dependencies(problem, sat_layers_solution) is None

            def as_dict(sol):
                return {"solution": sol.to_dict(), "plot": plot_solution(problem, sol)}

            depths[size].append((depth1, depth2))

            if depth1 != depth2:
                print(f"MISMATCH {instance_name} {depth1} {depth2}")
                print(plot_solution(problem, sat_solution))
                print(plot_solution(problem, sat_layers_solution))

                with open(f"experiments/layers_nonoptimal/{instance_name}.json", "w") as f:
                    json.dump(
                        {
                            "problem": problem.to_dict(),
                            "nonlayered_solution": as_dict(sat_solution),
                            "layered_solution": as_dict(sat_layers_solution),
                        },
                        f,
                    )

# Print statistics
for size in sorted(depths.keys()):
    avg_d1 = sum(d1 for d1,_ in depths[size]) / len(depths[size])
    avg_d2 = sum(d2 for _,d2 in depths[size]) / len(depths[size])
    print(f"Size {size} avg. depth {avg_d1} avg. layered depth {avg_d2} relative {avg_d2 / avg_d1}")


