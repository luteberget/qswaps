use log::{debug, error, info, trace};
use ordered_float::NotNan;
use pyo3::{exceptions::PyAssertionError, prelude::*};
use qswap_beam::{Problem, Solution};
use std::{
    cell::RefCell,
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

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct GateRef {
    gate_idx: i16,
    edge: (u8, u8),
}

#[derive(Clone, Debug)]
struct PathNode {
    cost: Reverse<NotNan<f64>>,
    lb: u8,
    pb: u8,
    gate_seq_idx: u16,
    time: u16,
    link: Option<(Option<GateRef>, Rc<PathNode>)>,
}

impl PartialEq for PathNode {
    fn eq(&self, _other: &Self) -> bool {
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

fn initial_paths(problem: &Problem, mut yield_path: impl FnMut(&PathNode) -> bool) {
    let initial_solution = qswap_beam::solve_heur_beam_full(
        &problem,
        &qswap_beam::BeamSearchParams {
            width: 1000,
            layer_discount: 0.75,
            depth_cost: 1.0,
            swap_cost: 1.0,
            heuristic_cost_factor: 1.0,
        },
    )
    .unwrap();

    debug!("Starting from initial solution {:?}", initial_solution);

    // Convert solution to paths.
    let mut lbs: Vec<Option<Rc<PathNode>>> = vec![None; problem.n_logical_bits];
    for (pb, lb) in initial_solution.input_bits.iter().copied().enumerate() {
        assert!(lbs[lb as usize].is_none());
        lbs[lb as usize] = Some(Rc::new(PathNode {
            // NOTE costs are not used here, the column's cost coefficient
            // is recalculated when adding the path to the master problem.
            cost: Reverse(NotNan::new(0.0).unwrap()),
            lb,
            pb: pb as u8,
            gate_seq_idx: 0,
            time: 0,
            link: None,
        }));
    }
    assert!(lbs.iter().all(|l| l.is_some()));
    let mut lbs = lbs.into_iter().map(|x| x.unwrap()).collect::<Vec<_>>();

    let edge_set = problem
        .topology
        .iter()
        .collect::<std::collections::HashSet<_>>();
    for gate in initial_solution.gates.iter() {
        let (pb1, pb2) = gate.edge;
        assert!(edge_set.contains(&(pb1, pb2)) ^ edge_set.contains(&(pb2, pb1)));

        let (pb1, pb2) = if edge_set.contains(&(pb1, pb2)) {
            (pb1, pb2)
        } else {
            (pb2, pb1)
        };

        debug!("Placing gate {:?} at {:?}", gate, (pb1, pb2));

        let gate_idx = gate.gate;
        if gate_idx >= 0 {
            let (lb1, lb2) = problem.gates[gate_idx as usize];

            let mut lb1_node = lbs[lb1 as usize].clone();
            let mut lb2_node = lbs[lb2 as usize].clone();

            assert!(
                (lb1_node.pb == pb1 && lb2_node.pb == pb2)
                    || (lb1_node.pb == pb2 && lb2_node.pb == pb1)
            );

            debug!("T1= {}  T2= {}", lb1_node.time, lb2_node.time);
            let t = lb1_node.time.max(lb2_node.time);

            if lb1_node.time < t {
                lb1_node = Rc::new(PathNode {
                    time: t,
                    link: Some((None, lb1_node.clone())),
                    ..*lb1_node
                });
            }

            lbs[lb1 as usize] = Rc::new(PathNode {
                gate_seq_idx: lb1_node.gate_seq_idx + 1,
                time: t + problem.gate_dt,
                link: Some((
                    Some(GateRef {
                        edge: (pb1, pb2),
                        gate_idx: gate_idx as i16,
                    }),
                    lb1_node.clone(),
                )),
                ..*lb1_node
            });

            if lb2_node.time < t {
                lb2_node = Rc::new(PathNode {
                    time: t,
                    link: Some((None, lb2_node.clone())),
                    ..*lb2_node
                });
            }

            lbs[lb2 as usize] = Rc::new(PathNode {
                gate_seq_idx: lb2_node.gate_seq_idx + 1,
                time: t + problem.gate_dt,
                link: Some((
                    Some(GateRef {
                        edge: (pb1, pb2),
                        gate_idx: gate_idx as i16,
                    }),
                    lb2_node.clone(),
                )),
                ..*lb2_node
            });
        } else {
            let mut lb1_node = lbs.iter().find(|l| l.pb == pb1).unwrap().clone();
            let lb1 = lb1_node.lb;
            let mut lb2_node = lbs.iter().find(|l| l.pb == pb2).unwrap().clone();
            let lb2 = lb2_node.lb;
            assert!(lb1 != lb2);

            let t = lb1_node.time.max(lb2_node.time);

            if lb1_node.time < t {
                lb1_node = Rc::new(PathNode {
                    time: t,
                    link: Some((None, lb1_node.clone())),
                    ..*lb1_node
                });
            }

            lbs[lb1 as usize] = Rc::new(PathNode {
                pb: pb2,
                time: t + problem.swap_dt,
                link: Some((
                    Some(GateRef {
                        edge: (pb1, pb2),
                        gate_idx: -1,
                    }),
                    lb1_node.clone(),
                )),
                ..*lb1_node
            });

            if lb2_node.time < t {
                lb2_node = Rc::new(PathNode {
                    time: t,
                    link: Some((None, lb2_node.clone())),
                    ..*lb2_node
                });
            }

            lbs[lb2 as usize] = Rc::new(PathNode {
                pb: pb1,
                time: t + problem.swap_dt,
                link: Some((
                    Some(GateRef {
                        edge: (pb1, pb2),
                        gate_idx: -1,
                    }),
                    lb2_node.clone(),
                )),
                ..*lb2_node
            });
        }
    }

    // assert!(lbs.iter().all(|l| l.time == lbs[0].time));
    for lb in &lbs {
        yield_path(lb);
    }
}

#[derive(Debug)]
enum ShadowPriceQuery {
    Bit(u8),
    HwNode(u16, u8),
    Gate(u16, i16, (u8, u8), bool),
}

fn price_paths(
    problem: &Problem,
    gate_seqs: &[Vec<usize>],
    price: impl Fn(ShadowPriceQuery) -> grb::Result<f64>,
    max_t: u16,
    mut yield_path: impl FnMut(&PathNode) -> bool,
) -> grb::Result<()> {
    debug!("pricing...");
    let mut queue: BinaryHeap<Rc<PathNode>> = Default::default();
    let makespan_coeff = 1.0 / problem.n_logical_bits as f64;

    // Start from all bits in all positions
    for lb in 0..problem.n_logical_bits as u8 {
        let bit_cost = -price(ShadowPriceQuery::Bit(lb))?;

        for pb in 0..problem.n_physical_bits as u8 {
            let pb_cost = -price(ShadowPriceQuery::HwNode(0, pb))?;

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
        // trace!("Examining node {:?}", node);
        if *node.cost.0 > 0.0 && node.time >= max_t {
            // There is no hope to achieve a negative cost from here.
            continue;
        }

        let mut new_node = |p: Rc<PathNode>| -> bool {
            let is_terminal = p.gate_seq_idx as usize == gate_seqs[p.lb as usize].len();

            if is_terminal && *p.cost.0 < -1e-5 {
                let proceed = yield_path(&*p);
                if !proceed {
                    return false;
                }
            }

            queue.push(p);
            true
        };

        //
        // SUCCESSOR TYPE #1: do nothing
        //
        {
            let proceed = new_node(Rc::new({
                let hw_cost = -price(ShadowPriceQuery::HwNode(node.time + 1, node.pb))?;
                let dt = 1;
                let cost = node.cost.0 + (dt as f64) * makespan_coeff + hw_cost;
                PathNode {
                    cost: Reverse(cost),
                    time: node.time + dt,
                    link: Some((None, node.clone())),
                    ..*node
                }
            }));
            if !proceed {
                return Ok(());
            }
        }

        //
        // SUCCESSOR TYPE #2: place next gate
        //
        assert!(problem.gate_dt == 1);
        if (node.gate_seq_idx as usize) < gate_seqs[node.lb as usize].len() {
            for (a, b) in problem.topology.iter().copied() {
                if a == node.pb || b == node.pb {
                    let dt = problem.gate_dt;
                    let gate_idx = gate_seqs[node.lb as usize][node.gate_seq_idx as usize] as u16;
                    let occ_cost = -price(ShadowPriceQuery::HwNode(node.time + 1, node.pb))?;
                    let gate_cost = -price(ShadowPriceQuery::Gate(
                        node.time,
                        gate_idx as i16,
                        (a, b),
                        a == node.pb,
                    ))?;
                    let cost = node.cost.0 + (dt as f64) * makespan_coeff + occ_cost + gate_cost;

                    let proceed = new_node(Rc::new({
                        PathNode {
                            cost: Reverse(cost),
                            gate_seq_idx: node.gate_seq_idx + 1,
                            time: node.time + dt,
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
                    if !proceed {
                        return Ok(());
                    }
                }
            }
        }

        // SUCCESSOR TYPE #3: place a swap
        for (a, b) in problem.topology.iter().copied() {
            if a == node.pb || b == node.pb {
                let other_pb = if a == node.pb { b } else { a };
                let dt = problem.swap_dt;

                let mut occ_cost = 0.0;
                for step in 1..dt {
                    occ_cost -= price(ShadowPriceQuery::HwNode(node.time + step, node.pb))?;
                }
                occ_cost -= price(ShadowPriceQuery::HwNode(node.time + dt, other_pb))?;

                // TODO SWAP TIME DEFINITION
                let swap_shadow_price =
                    -price(ShadowPriceQuery::Gate(node.time, -1, (a, b), a == node.pb))?;
                // TODO swap price as function parameter
                let swap_price = 1.0;

                let cost = node.cost.0
                    + (dt as f64) * makespan_coeff
                    + occ_cost
                    + swap_shadow_price
                    + 0.5 * swap_price;

                let proceed = new_node(Rc::new({
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
                }));
                if !proceed {
                    return Ok(());
                }
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

    info!("SOLVE PATHMIP COLGEN problem= {:?}", problem);

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

    // TODO add makespan objective (instead of each path's individual makespan in the objective)
    // let delay_obj = add_ctsvar!(model, name:"delay", bounds:0.0.., obj: 1.0)?;
    // let c0_makespan = (0..problem.n_logical_bits)
    //     .map(|lb| model.add_constr(&format!("makespan_lb{}", lb), c!(0 <= delay_obj)))
    //     .collect::<Result<Vec<_>, _>>()?;

    let lbpath_constraints = (0..problem.n_logical_bits)
        .map(|lb| model.add_constr(&format!("lb{}", lb), c!(0 == 1)))
        .collect::<Result<Vec<_>, _>>()?;

    struct LazyConstraints {
        hwmap: Vec<Vec<Constr>>,
        gates: Vec<HashMap<(i16, (u8, u8)), Constr>>,
    }

    let lazy_constraints = RefCell::new(LazyConstraints {
        hwmap: Default::default(),
        gates: Default::default(),
    });

    let mut lower_bound = f64::INFINITY;
    let mut upper_bound = f64::INFINITY;

    struct PathData {
        lb: u8,
        pb: u8,
        var: grb::Var,
        cost: f64,
        gates: Vec<(u16, GateRef)>,
    }

    #[derive(Hash, PartialEq, Eq, PartialOrd, Ord)]
    #[derive(Debug  )]
    struct PathFingerprint {
        lb: u8,
        pb: u8,
        gates: Vec<(u16, GateRef)>,
    }

    let mut paths: Vec<PathData> = Default::default();
    let mut path_set: std::collections::HashSet<PathFingerprint> = Default::default();
    let model = RefCell::new(model);
    let mut n_iters = 0;

    loop {
        // Column generation loop
        let n_paths_before = paths.len();
        let add_path = |mut next_node: &PathNode| {
            let mut model = model.borrow_mut();
            let mut lazy_constraints = lazy_constraints.borrow_mut();
            let lb = next_node.lb;
            let reduced_cost = *next_node.cost.0;
            let mut cost = next_node.time as f64 / problem.n_logical_bits as f64;
            let var = add_ctsvar!(model, name :&format!("path{}", paths.len()), bounds: 0.0..1.0)
                .unwrap();
            let mut gates: Vec<(u16, GateRef)> = Default::default();
            let initial_pb: u8;

            let c = &lbpath_constraints[lb as usize];
            model.set_coeff(&var, c, 1.0).unwrap();

            loop {
                assert!((next_node.time == 0) == next_node.link.is_none());

                if let Some((gate, prev_node)) = next_node.link.as_ref() {
                    for t in prev_node.time..next_node.time {
                        while (t as usize) >= lazy_constraints.hwmap.len() {
                            let add_t = lazy_constraints.hwmap.len();
                            lazy_constraints.hwmap.push(
                                (0..problem.n_physical_bits)
                                    .map(|pb| {
                                        model
                                            .add_constr(
                                                &format!("hw_t{}_pb{}", add_t, pb),
                                                c!(0 <= 1),
                                            )
                                            .unwrap()
                                    })
                                    .collect(),
                            );
                        }

                        let c = &lazy_constraints.hwmap[t as usize][prev_node.pb as usize];
                        model.set_coeff(&var, c, 1.0).unwrap();
                    }

                    if let Some(gate) = gate {
                        let gate_time = prev_node.time;
                        let (a, b) = gate.edge;

                        let coeff = if prev_node.pb == a { 1.0 } else { -1.0 };

                        while (gate_time as usize) >= lazy_constraints.gates.len() {
                            lazy_constraints.gates.push(Default::default());
                        }

                        let constraint = lazy_constraints.gates[gate_time as usize]
                            .entry((gate.gate_idx, (a, b)));
                        let constraint = constraint.or_insert_with(|| {
                            model
                                .add_constr(
                                    &format!("g{}_({},{})_t{}", gate.gate_idx, a, b, gate_time),
                                    c!(0 == 0),
                                )
                                .unwrap()
                        });

                        model.set_coeff(&var, constraint, coeff).unwrap();

                        if gate.gate_idx < 0 {
                            debug!("Path {} goes through swap", paths.len());
                            cost += 0.5;
                        }

                        gates.push((gate_time, *gate));
                    }
                    next_node = prev_node;
                } else {
                    initial_pb = next_node.pb;
                    break;
                }
            }

            model.set_obj_attr(grb::attr::Obj, &var, cost).unwrap();
            gates.sort_by_key(|(t,_)| *t);

            let fingerprint = PathFingerprint {
                gates: gates.clone(),
                lb,
                pb: initial_pb,
            };
            
            debug!("Adding column c={} {:?}", reduced_cost, fingerprint);
            if path_set.contains(&fingerprint) {
                panic!("duplicate column");
            } else {
                path_set.insert(fingerprint);
            }

            paths.push(PathData {
                lb,
                pb: initial_pb,
                var,
                cost,
                gates,
            });

            false
        };

        if n_paths_before == 0 {
            initial_paths(problem, add_path);
        } else {
            assert!(model.borrow().status()? == Status::Optimal);
            let max_t = lazy_constraints.borrow().hwmap.len() as u16;

            let price_fn = |constraint| {
                let price = {
                    let model = model.borrow();
                    let lazy_constraints = lazy_constraints.borrow();

                    match constraint {
                        ShadowPriceQuery::Bit(x) => {
                            model.get_obj_attr(grb::attr::Pi, &lbpath_constraints[x as usize])
                        }
                        ShadowPriceQuery::HwNode(time, pb) => {
                            let constraint = lazy_constraints
                                .hwmap
                                .get(time as usize)
                                .and_then(|c| c.get(pb as usize));
                            constraint
                                .map(|c| model.get_obj_attr(grb::attr::Pi, c))
                                .unwrap_or(Ok(0.0))
                        }
                        ShadowPriceQuery::Gate(time, gate, (a, b), use_a) => {
                            let constraint = lazy_constraints
                                .gates
                                .get(time as usize)
                                .and_then(|m| m.get(&(gate, (a, b))));
                            constraint
                                .map(|c| {
                                    model.get_obj_attr(grb::attr::Pi, c).map(|v| {
                                        if use_a {
                                            v
                                        } else {
                                            -v
                                        }
                                    })
                                })
                                .unwrap_or(Ok(0.0))
                        }
                    }
                };
                // trace!("{:?} --> {:?}", constraint, price);
                price
            };

            price_paths(problem, &gate_seqs, price_fn, max_t, add_path)?;
        }
        debug!(
            "Added {}+{}={} paths",
            n_paths_before,
            paths.len() - n_paths_before,
            paths.len()
        );

        let mut model = model.borrow_mut();
        if n_paths_before == paths.len() {
            for path in paths.iter() {
                let x = model.get_obj_attr(grb::attr::X, &path.var)?;
                // info!("hey path {:?} = w{}", path.var, x);
                if x > 1e-3 {
                    info!(
                        "path x={} c={} lb={} pb={} g={:?}",
                        x, path.cost, path.lb, path.pb, path.gates
                    );
                }
            }
            todo!("root node converged");
        }

        // We are solving a linear relaxation of the (restricted) master problem.
        assert_eq!(model.get_attr(attr::IsMIP)?, 0);

        // if n_iters == 100 {
        //     todo!("doesn't converge?");
        // }

        model.write(&format!("model_{}.lp", n_iters))?;
        model.optimize()?;
        match model.status()? {
            Status::Optimal => {
                let new_lower_bound = model.get_attr(attr::ObjVal)?;
                // When adding columns, the lower bound should decrease.
                debug!("Lower bound {} (prev {})", new_lower_bound, lower_bound);
                assert!(new_lower_bound <= lower_bound + 1e-6);
                lower_bound = new_lower_bound;

                debug!("Shadow_prices bits: {:?}", {
                    lbpath_constraints
                        .iter()
                        .map(|c| model.get_obj_attr(grb::attr::Pi, c))
                        .enumerate()
                        .collect::<Vec<_>>()
                });

                debug!("Shadow_prices hw bits {}", {
                    let hwmap = &lazy_constraints.borrow().hwmap;
                    for (t, pbs) in hwmap.iter().enumerate() {
                        debug!(
                            "Shadow price hw t={} {:?}",
                            t,
                            pbs.iter()
                                .map(|c| { model.get_obj_attr(grb::attr::Pi, c) })
                                .collect::<Vec<_>>()
                        )
                    }
                    ""
                });

                debug!("Shadow price gates {}", {
                    let gates = &lazy_constraints.borrow().gates;
                    for (t, gs) in gates.iter().enumerate() {
                        debug!(
                            "Shadow prices Gates t={} {:?}",
                            t,
                            gs.iter()
                                .map(|((g, (a, b)), c)| (
                                    (g, (a, b)),
                                    model.get_obj_attr(grb::attr::Pi, c)
                                ))
                                .collect::<Vec<_>>()
                        );
                    }
                    ""
                });

                if upper_bound.is_finite() && (upper_bound - lower_bound) / (lower_bound) < 0.01 {
                    info!("Found optimal.");
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

        n_iters += 1;
    }

    todo!("return best solution")
}

#[cfg(test)]
mod tests {
    use super::*;
    use test_log::test;

    #[test]
    fn test_solve_small() {
        info!("hello");
        let problem = Problem {
            n_logical_bits: 4,
            gates: vec![(0, 1), (0, 1)],
            n_physical_bits: 4,
            topology: vec![(1, 2), (0, 1), (2, 3)],
            swap_dt: 3,
            gate_dt: 1,
        };

        let solution: Solution = solve_pathmip_colgen(&problem).unwrap().unwrap();
        assert!(solution.n_swaps == 0 && solution.depth == 1 * problem.gate_dt);
        println!("SOLUTION {:?}", solution);
    }

    #[test]
    fn test_solve_medium1() {
        let problem = Problem {
            n_logical_bits: 4,
            gates: vec![(1, 2), (3, 0), (3, 1), (0, 2)],
            n_physical_bits: 4,
            topology: vec![(0, 1), (1, 2), (2, 3)],
            swap_dt: 3,
            gate_dt: 1,
        };
        let solution: Solution = solve_pathmip_colgen(&problem).unwrap().unwrap();
        assert!(solution.n_swaps == 0 && solution.depth == 1 * problem.gate_dt);
        println!("SOLUTION {:?}", solution);
    }
}
