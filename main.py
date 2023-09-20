import qswaps
from qswaps import RawProblem, RawGate, solve
from plot import plot_solution
from gen_problems import gen_instances
import qswap_beam
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

exact_results = {}
heur_results = {}

# p1 = RawProblem(
#     gates=[
#         RawGate(line1="l5", line2="l3", duration=1),
#         RawGate(line1="l2", line2="l1", duration=1),
#         RawGate(line1="l3", line2="l2", duration=1),
#         RawGate(line1="l4", line2="l1", duration=1),
#         RawGate(line1="l4", line2="l2", duration=1),
#         RawGate(line1="l1", line2="l5", duration=1),
#     ],
#     topology=[("p1", "p2"), ("p2", "p3"), ("p3", "p4"), ("p4", "p5")],
#     swap_time=3,
# )
# solve(p1, 1)
# raise Exception()


for size in [4,5,6,7]:
    for swaptime in [3]:
        exact_results[(size, swaptime)] = []
        heur_results[(size, swaptime)] = []
        for i, instance in enumerate(gen_instances(size, seed=420, num_instances=50)):
            instance_name = f"sz{size}_swt{swaptime}_{i}"
            print(f"# INSTANCE {instance_name}")
            problem = RawProblem(
                [RawGate(f"l{j.pair[0]}", f"l{j.pair[1]}", 1) for j in instance.jobs],
                [(f"p{a}", f"p{b}") for a, b in instance.hardware_graph.edges],
                swaptime,
            )

            print("PROBLEM", problem)

            print("solving...")
            t0 = time.time()
            depth1, solution = solve(problem, 0 if swaptime == 1 else 1)
            t1 = time.time()
            plot_solution(problem, solution)
            print(
                f"SAT solved size{size} swaptime{swaptime}  d={depth1} in {(t1-t0):0.2f}"
            )
            print(f"SOLVED {instance_name} sat {depth1} {t1-t0:.2f}")

            exact_results[(size, swaptime)].append((depth1, t1 - t0))
            # h_problem = qswap_beam.

            lb_list = []
            lb_map = {}

            for gate in problem.gates:
                for n in [gate.line1, gate.line2]:
                    if not n in lb_map:
                        lb_map[n] = len(lb_list)
                        lb_list.append(n)

            pb_list = []
            pb_map = {}

            for a, b in problem.topology:
                for bit in [a, b]:
                    if not bit in pb_map:
                        pb_map[bit] = len(pb_list)
                        pb_list.append(bit)

            assert len(lb_list) <= len(pb_list)
            while len(lb_list) < len(pb_list):
                n = f"dummy_{len(lb_list)}"
                lb_map[n] = len(lb_list)
                lb_list.append(n)


            h_problem = qswap_beam.Problem(
                n_logical_bits=len(lb_list),
                n_physical_bits=len(pb_list),
                gate_dt=0 if swaptime == 1 else 1,
                swap_dt=swaptime,
                gates=[
                    (lb_map[gate.line1], lb_map[gate.line2]) for gate in problem.gates
                ],
                topology=[(pb_map[a], pb_map[b]) for a, b in problem.topology],
            )

            h_params = qswap_beam.BeamSearchParams(
                width=1000 * len(lb_list),
                layer_discount=0.5,
                depth_cost=1.0,
                swap_cost=0.1,
                heuristic_cost_factor=1.0,
            )

            print("SOLVING heur")
            t0 = time.time()
            sol = qswap_beam.qswap_beam_solve(h_problem, h_params)
            t1 = time.time()
            depth2 = sol.depth

            sol = qswaps.Solution(
                bit_assignment={
                    pb_list[pb]: lb_list[lb] for pb, lb in enumerate(sol.input_bits)
                },
                operations=[
                    qswaps.Operation(
                        0,
                        None if g.gate == -1 else g.gate,
                        (pb_list[g.edge[0]], pb_list[g.edge[1]]),
                    )
                    for g in sol.gates
                ],
            )

            plot_solution(problem, sol)
            print(
                f"HEUR solved size{size} swaptime{swaptime} d={depth2} in {(t1-t0):0.2f}"
            )
            print(f"SOLVED {instance_name} beam {depth2} {t1-t0:.2f}")

            heur_results[(size, swaptime)].append((depth2, t1 - t0))

            if depth1 != depth2:
                print(f"MISMATCH {instance_name} {depth1} {depth2}")

            # break

print("exact:", exact_results)
print("heur: ", heur_results)
