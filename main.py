from qswaps import RawProblem, RawGate, solve
from plot import plot_solution

example1 = RawProblem(
    gates=[RawGate("l1", "l2", 1), RawGate("l3", "l4", 1), RawGate("l1", "l3", 1), RawGate("l2", "l4", 1)],
    topology=[("p1", "p2"), ("p2", "p3"), ("p3","p4")],
    swap_time=3,
)

example2 = "1-(2,4)-3, 2-(3,1)-3, 3-(3,4)-3, 4-(2,1)-3, 5-(2,4)-3, 6-(1,3)-3, 7-(3,2)-3, 8-(4,1)-3"
#example2 = "1-(2,4)-3, 2-(3,1)-3, 3-(3,4)-3, 4-(2,1)-3"
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
sol = solve(example2)
plot_solution(example2, sol)
