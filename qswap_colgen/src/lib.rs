use ordered_float::{NotNan, OrderedFloat};
use pyo3::{exceptions::PyAssertionError, prelude::*};
use qswap_beam::{Problem, Solution};
use std::{
    cmp::Reverse,
    collections::{BinaryHeap, HashMap},
    rc::Rc,
};

#[pyfunction]
fn qswap_colgen_solve(problem: &Problem) -> PyResult<Option<Solution>> {
    solve_pathmip_colgen(problem)
        .map_err(|e| PyAssertionError::new_err(format!("Gurobi error: {}", e)))
}

#[pymodule]
fn qswap_colgen(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(qswap_colgen_solve, m)?)?;
    m.add_class::<Problem>()?;
    m.add_class::<Solution>()?;
    Ok(())
}

#[derive(Copy, Clone)]
struct GateRef {
    gate_idx: i16,
    edge: (u8, u8),
}

#[derive(Clone)]
struct PathNode {
    cost: Reverse<NotNan<f64>>,
    lb: u8,
    pb: u8,
    gate_seq_idx: u16,
    time: u16,
    link: Option<(Option<GateRef>, Rc<PathNode>)>,
}

impl PartialEq for PathNode {
    fn eq(&self, other: &Self) -> bool {
        false
    }
}

impl Eq for PathNode {}

impl PartialOrd for PathNode {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.cost.partial_cmp(&other.cost)
    }
}

impl Ord for PathNode {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.cost.cmp(&other.cost)
    }
}

fn initial_paths(problem: &Problem, mut yield_path: impl FnMut(&PathNode)) {
    todo!()
}

enum ConstraintRef {
    Bit(u8),
    HwNode(u16, u8),
    Gate(u16, (u8, u8)),
    Swap(u16, (u8, u8)),
}

fn price_paths(
    problem: &Problem,
    gate_seqs: &[Vec<usize>],
    price: impl Fn(ConstraintRef) -> grb::Result<f64>,
    mut yield_path: impl FnMut(&PathNode),
) -> grb::Result<()> {
    let mut queue: BinaryHeap<Rc<PathNode>> = Default::default();

    // Start from all bits in all positions
    for lb in 0..problem.n_logical_bits as u8 {
        let bit_cost = price(ConstraintRef::Bit(lb))?;

        for pb in 0..problem.n_physical_bits as u8 {
            let pb_cost = price(ConstraintRef::HwNode(0, pb))?;

            queue.push(Rc::new(PathNode {
                cost: Reverse(NotNan::new(bit_cost + pb_cost).unwrap()),
                lb,
                pb,
                gate_seq_idx: 0,
                time: 0,
                link: None,
            }));
        }
    }

    while let Some(node) = queue.pop() {
        let mut new_node = |p: Rc<PathNode>| {
            let is_terminal = false;

            if is_terminal {
                yield_path(&*p);
            }

            queue.push(p);
        };

        //
        // SUCCESSOR TYPE #1: do nothing
        //
        new_node(Rc::new({
            let added_cost = price(ConstraintRef::HwNode(node.time + 1, node.pb))?;
            let cost = node.cost.0 + added_cost;
            PathNode {
                cost: Reverse(cost),
                time: node.time + 1,
                link: Some((None, node.clone())),
                ..*node
            }
        }));

        //
        // SUCCESSOR TYPE #2: place next gate
        //
        assert!(problem.gate_dt == 1);
        if (node.gate_seq_idx as usize) < gate_seqs[node.lb as usize].len() {
            for (a, b) in problem.topology.iter().copied() {
                if a == node.pb || b == node.pb {
                    let gate_idx = gate_seqs[node.lb as usize][node.gate_seq_idx as usize] as u16;
                    let occ_cost = price(ConstraintRef::HwNode(node.time + 1, node.pb))?;

                    // TODO  GATE DIRECTION
                    let gate_cost = price(ConstraintRef::Gate(gate_idx, (a, b)))?;

                    let cost = node.cost.0 + occ_cost + gate_cost;

                    new_node(Rc::new({
                        PathNode {
                            cost: Reverse(cost),
                            gate_seq_idx: node.gate_seq_idx + 1,
                            time: node.time + problem.gate_dt,
                            link: Some((
                                Some(GateRef {
                                    edge: (a, b),
                                    gate_idx: gate_idx as i16,
                                }),
                                node.clone(),
                            )),
                            ..*node
                        }
                    }));
                }
            }
        }

        // SUCCESSOR TYPE #3: place a swap
        for (a, b) in problem.topology.iter().copied() {
            if a == node.pb || b == node.pb {
                let other_pb = if a == node.pb { b } else { a };

                let mut occ_cost = 0.0;
                for dt in 1..problem.swap_dt {
                    occ_cost += price(ConstraintRef::HwNode(node.time + dt, node.pb))?;
                }
                occ_cost += price(ConstraintRef::HwNode(node.time + problem.swap_dt, other_pb))?;

                // TODO SWAP DIRECTION PRICE
                // TODO SWAP TIME DEFINITION
                let swap_shadow_price = price(ConstraintRef::Swap(node.time, (a, b)))?;
                // TODO swap price as function parameter
                let swap_price = 1.0;

                let cost = node.cost.0 + occ_cost + swap_shadow_price + 0.5 * swap_price;
                new_node(Rc::new({
                    PathNode {
                        cost: Reverse(cost),
                        time: node.time + problem.gate_dt,
                        pb: other_pb,
                        link: Some((
                            Some(GateRef {
                                gate_idx: -1,
                                edge: (a, b),
                            }),
                            node.clone(),
                        )),
                        ..*node
                    }
                }))
            }
        }
    }
    Ok(())
}

fn solve_pathmip_colgen(problem: &Problem) -> grb::Result<Option<Solution>> {
    use grb::prelude::*;

    assert!(problem.gate_dt == 1);
    assert!(problem.swap_dt == 3);
    assert!(problem.n_logical_bits == problem.n_physical_bits);

    println!("SOLVE PATHMIP COLGEN problem= {:?}", problem);

    let gate_seqs = (0..problem.n_logical_bits as u8)
        .map(|lb| {
            problem
                .gates
                .iter()
                .enumerate()
                .filter_map(|(i, (b1, b2))| (lb == *b1 || lb == *b2).then(|| i))
                .collect()
        })
        .collect::<Vec<Vec<_>>>();

    let mut model = grb::Model::new("m")?;
    model.set_attr(grb::attr::ModelSense, ModelSense::Minimize)?;

    let delay_obj = add_ctsvar!(model, name:"delay", bounds:0.0.., obj: 1.0)?;

    let makespan_constraint = model.add_constr("makespan", c!(0 <= delay_obj))?;

    let c1_lbpath = (0..problem.n_logical_bits)
        .map(|lb| model.add_constr(&format!("lb{}", lb), c!(0 <= 1)))
        .collect::<Result<Vec<_>, _>>()?;

    let c2_pbmap: Vec<Vec<Constr>> = Vec::new();
    let c3_gates: Vec<HashMap<u32, Constr>> = Vec::new();
    let c4_swaps: Vec<HashMap<u32, Constr>> = Vec::new();

    let mut lower_bound = f64::INFINITY;
    let mut upper_bound = f64::INFINITY;

    struct PathData {}
    let mut paths: Vec<PathData> = Default::default();

    loop {
        // Column generation loop

        let add_path = |p: &PathNode| {
            println!("path");
        };

        let n_paths_before = paths.len();
        if n_paths_before == 0 {
            initial_paths(problem, add_path);
        } else {
            assert!(model.status()? == Status::Optimal);
            price_paths(
                problem,
                &gate_seqs,
                |constraint| match constraint {
                    ConstraintRef::Bit(x) => {
                        model.get_obj_attr(grb::attr::Pi, &c1_lbpath[x as usize])
                    }
                    ConstraintRef::HwNode(_, _) => todo!(),
                    ConstraintRef::Gate(_, _) => todo!(),
                    ConstraintRef::Swap(_, _) => todo!(),
                },
                add_path,
            )?;
        }

        if n_paths_before == paths.len() {
            todo!("no more columns to add");
        }

        // We are solving a linear relaxation of the (restricted) master problem.
        assert_eq!(model.get_attr(attr::IsMIP)?, 0);

        model.optimize()?;
        match model.status()? {
            Status::Optimal => {
                let new_lower_bound = model.get_attr(attr::ObjVal)?;
                // When adding columns, the lower bound should decrease.
                assert!(new_lower_bound <= lower_bound);
                lower_bound = new_lower_bound;

                if upper_bound.is_finite() && (upper_bound - lower_bound) / (lower_bound) < 0.01 {
                    println!("Found optimal.");
                    todo!("terminate");
                }
            }
            Status::TimeLimit | Status::Interrupted => {
                break;
            }
            Status::Infeasible | Status::InfOrUnbd => {
                panic!("infeasible");
            }
            _ => {
                panic!("unexpected status")
            }
        };
    }

    todo!("return best solution")
}
