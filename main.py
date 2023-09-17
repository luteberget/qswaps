from qswaps import RawProblem, RawGate, solve
from plot import plot_solution
from gen_problems import gen_instances
import time

example1 = RawProblem(
    gates=[
        RawGate("l1", "l2", 1),
        RawGate("l3", "l4", 1),
        RawGate("l1", "l3", 1),
        RawGate("l2", "l4", 1),
    ],
    topology=[("p1", "p2"), ("p2", "p3"), ("p3", "p4")],
    swap_time=3,
)

example2 = "1-(2,4)-3, 2-(3,1)-3, 3-(3,4)-3, 4-(2,1)-3, 5-(2,4)-3, 6-(1,3)-3, 7-(3,2)-3, 8-(4,1)-3"
# example2 = "1-(2,4)-3, 2-(3,1)-3, 3-(3,4)-3, 4-(2,1)-3"
# print([parts for g in example2.split(", ") for parts in g.split("-")])

example2 = RawProblem(
    gates=[
        RawGate(
            line1="l" + g.split("-")[1][1:].split(",")[0],
            line2="l" + g.split("-")[1][:-1].split(",")[1],
            duration=1,
        )
        for g in example2.split(", ")
    ],
    topology=[("p1", "p2"), ("p2", "p3"), ("p3", "p4")],
    swap_time=3,
)

results = {}

for size in [4]:
    for swaptime in [3]:
        results[(size,swaptime)] = []
        for i,instance in enumerate(gen_instances(size, seed=420)):

            problem = RawProblem(
                [RawGate(f"l{j.pair[0]}", f"l{j.pair[1]}", 1) for j in instance.jobs],
                [(f"p{a}", f"p{b}") for a, b in instance.hardware_graph.edges],
                swaptime,
            )

            print("PROBLEM", problem)

            print("solving...")
            t0 = time.time()
            solution = solve(problem, 0 if swaptime == 1 else 1)
            t1 = time.time()
            plot_solution(problem, solution)
            print(f"solved size{size} swaptime{swaptime} in {(t1-t0):0.2f}")
            results[(size,swaptime)].append(t1-t0)
            break

print(results)

