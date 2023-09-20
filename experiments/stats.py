from tabulate import tabulate
from collections import defaultdict

size_depth = defaultdict(list)
size_heur_depth = defaultdict(list)
size_heur_time = defaultdict(list)
size_sat_time = defaultdict(list)
size_instance_depth = {}

for filename in ["test-2023-09-19.txt", "test-2023-09-19-part2.txt"]:
    with open(filename,"r") as f:
        for l in f:
            if l.startswith("SOLVED"):
                _,instance,solver,depth,time = l.split()
                depth = int(depth)
                time = float(time)
                size = int(instance[2])

                if not size in size_instance_depth:
                    size_instance_depth[size] = {}

                if not instance in size_instance_depth[size]:
                    size_instance_depth[size][instance] = [None,None]


                if solver == "sat":
                    size_sat_time[size].append(time)
                    size_depth[size].append(depth)
                    size_instance_depth[size][instance][1] = depth
                if solver == "beam":
                    size_heur_depth[size].append(depth)
                    size_heur_time[size].append(time)
                    size_instance_depth[size][instance][0] = depth




# size  number  average optmial depth average heuristic run time averagee

rows = []
for size in sorted(size_depth.keys()):
    number = len(size_heur_time[size])
    assert len(size_sat_time[size]) == number
    assert len(size_instance_depth[size]) == number
    assert len(size_heur_depth[size]) == number

    avg_opt_depth = sum(size_depth[size]) / len(size_depth[size])
    avg_heur_depth = sum(size_heur_depth[size]) / len(size_heur_depth[size])
    avg_sat_time = sum(size_sat_time[size]) / len(size_sat_time[size])
    avg_heur_time = sum(size_heur_time[size]) / len(size_heur_time[size])
    n_optimal = len([ True for h,s in size_instance_depth[size].values() if h == s ])
    n_optimal_str = f"{n_optimal}/{number}"
    avg_nonoptimal_abs_gap = sum(( h-s for h,s in size_instance_depth[size].values() if h != s )) / (number - n_optimal) if number-n_optimal > 0 else 0
    avg_abs_gap = sum(( h-s for h,s in size_instance_depth[size].values() )) / number
    avg_rel_gap = sum(( (h-s)/s for h,s in size_instance_depth[size].values() )) / number

    rows.append((number,n_optimal,avg_opt_depth,avg_heur_depth,avg_sat_time,avg_heur_time,avg_nonoptimal_abs_gap,avg_abs_gap,avg_rel_gap))

cols = [list(i) for i in zip(*rows)]
cols[0].insert(0, "number of instances")
cols[1].insert(0, "number of instances where heuristic gives optimal solution")
cols[2].insert(0, "average optimal depth")
cols[3].insert(0, "average heuristic depth")
cols[4].insert(0, "average optimal solve time (s)")
cols[5].insert(0, "average heuristic solve time (s)")
cols[6].insert(0, "average absolute gap for non-optimal instances")
cols[7].insert(0, "average absolute gap")
cols[8].insert(0, "average relative gap")
print(tabulate(cols, headers=[""] + [f"n_bits={s}" for s in sorted(size_depth.keys())]))


