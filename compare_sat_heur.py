import dataclasses
import qswaps
from qswaps import RawProblem, RawGate, solve
from plot import plot_solution
from gen_problems import gen_instances
import qswap_beam
import time
import json

exact_results = {}
heur_results = {}

#
# Solve bit routing on quantum volume circuits with these approaches:
#  1. A SAT-based solver, solving min-depth exactly for a model where all gates take the 
#     same (integer) amount of time, and all swaps take another integer amount of time.
#  2. A beam-search solver, solving the same problem heuristically.
#

for size in [4, 5, 6, 7]:
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
            depth1, sat_solution = solve(problem, 0 if swaptime == 1 else 1)
            t1 = time.time()
            print(plot_solution(problem, sat_solution))
            print(f"SAT solved size{size} swaptime{swaptime}  d={depth1} in {(t1-t0):0.2f}")
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
                gates=[(lb_map[gate.line1], lb_map[gate.line2]) for gate in problem.gates],
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
            heur_solution = qswap_beam.qswap_beam_solve(h_problem, h_params)
            t1 = time.time()
            depth2 = heur_solution.depth

            heur_solution = qswaps.Solution(
                bit_assignment={pb_list[pb]: lb_list[lb] for pb, lb in enumerate(heur_solution.input_bits)},
                operations=[
                    qswaps.Operation(
                        0,
                        None if g.gate == -1 else g.gate,
                        (pb_list[g.edge[0]], pb_list[g.edge[1]]),
                    )
                    for g in heur_solution.gates
                ],
            )

            print(plot_solution(problem, heur_solution))
            print(f"HEUR solved size{size} swaptime{swaptime} d={depth2} in {(t1-t0):0.2f}")
            print(f"SOLVED {instance_name} beam {depth2} {t1-t0:.2f}")

            heur_results[(size, swaptime)].append((depth2, t1 - t0))

            if depth1 != depth2:
                print(f"MISMATCH {instance_name} {depth1} {depth2}")

print("exact:", exact_results)
print("heur: ", heur_results)
