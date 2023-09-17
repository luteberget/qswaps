use min_max_heap::MinMaxHeap;
use ordered_float::OrderedFloat;
use rand::seq::SliceRandom;
use rand::thread_rng;
use smallvec::SmallVec;
use std::collections::HashMap;
use std::rc::Rc;

pub struct Problem {
    pub n_logical_bits: usize,
    pub n_physical_bits: usize,
    pub gate_dt: u16,
    pub swap_dt: u16,
    pub gates: Vec<(u8, u8)>,
    pub topology: Vec<(u8, u8)>,
}

#[derive(Debug)]
pub struct Solution {
    pub input_bits: Vec<u8>,
    pub gates: Vec<Gate>,
    pub depth: u16,
    pub n_swaps: u16,
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum Gate {
    InputGate(u8),
    SwapGate(u8, u8),
}

pub struct BeamSearchParams {
    pub width: usize,
    pub layer_discount: f32,

    pub depth_cost: f32,
    pub swap_cost: f32,
    pub heuristic_cost_factor: f32,
}

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Node {
    pub f_cost: OrderedFloat<f32>,
    pub n_swaps: u16,
    pub state: State,
    pub parent: Option<(Rc<Node>, Gate)>,
}

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct State {
    pub bits: Bits,
    pub placed_gates: u64,
    pub time: SmallVec<[u16; 16]>,
}

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct Bits {
    pub forward: SmallVec<[u8; 16]>,
    pub backward: SmallVec<[u8; 16]>,
}

impl Bits {
    pub fn new(n: usize) -> Self {
        let v: SmallVec<[u8; 16]> = (0..(n as u8)).collect();
        Self {
            backward: v.clone(),
            forward: v,
        }
    }

    pub fn swap_physical(&mut self, a: usize, b: usize) {
        // Swap logical bits that are located in physical bits

        // The backward array maps logical bits to physical bits,
        // so we need to find the logical bits that we are swapping.
        let log_a = self.forward[a];
        let log_b = self.forward[b];
        // ... and then swap the logical bits.
        self.backward.swap(log_a as usize, log_b as usize);

        // The forward array maps physical bits to logical bits,
        // and here we simply swap the bits.
        self.forward.swap(a, b);
    }
}

type GateLayers = SmallVec<[i8; 32]>;

fn cached_toposort_layers<'a>(
    cache: &'a mut HashMap<u64, GateLayers>,
    problem: &Problem,
    placed_gates: u64,
) -> &'a GateLayers {
    cache
        .entry(placed_gates)
        .or_insert_with(|| toposort_layers(problem, placed_gates))
}

fn toposort_layers(problem: &Problem, placed_gates: u64) -> GateLayers {
    let mut incoming_edges: Vec<Vec<u8>> = problem.gates.iter().map(|_| Vec::new()).collect();
    let mut outgoing_edges = incoming_edges.clone();
    let mut lb_output_gate: Vec<Option<u8>> = vec![None; problem.n_logical_bits];
    for (this_gate, (input1, input2)) in problem.gates.iter().enumerate() {
        if placed_gates & (1 << this_gate) > 0 {
            continue;
        }

        // Dependency arc for LB inputs to this gate
        for prev_input in [input1, input2] {
            if let Some(prev_gate) = lb_output_gate[*prev_input as usize] {
                incoming_edges[this_gate].push(prev_gate);
                outgoing_edges[prev_gate as usize].push(this_gate as u8);
            }

            lb_output_gate[*prev_input as usize] = Some(this_gate as u8);
        }
    }

    let mut added_nodes = 0;
    let mut result: GateLayers = Default::default();

    let mut vec1: SmallVec<[u8; 16]> = Default::default();
    let mut vec2: SmallVec<[u8; 16]> = Default::default();

    let mut current_layer = &mut vec1;
    let mut next_layer = &mut vec2;

    // First iteration, loop through all
    for (gate, incoming) in incoming_edges.iter().enumerate() {
        if placed_gates & (1 << gate) > 0 {
            continue;
        }
        if incoming.is_empty() {
            current_layer.push(gate as u8);
        }
    }

    let n_gates_to_place = problem.gates.len() - placed_gates.count_ones() as usize;
    while added_nodes < n_gates_to_place {
        // Add the nodes from the next layer
        // Since this is a DAG by construction (above),
        // there should be nodes without incoming edges.
        assert!(!current_layer.is_empty());
        assert!(next_layer.is_empty());

        for gate in current_layer.iter().copied() {
            added_nodes += 1;
            result.push(gate as i8);
            for other_gate in outgoing_edges[gate as usize].iter().copied() {
                let incoming = &mut incoming_edges[other_gate as usize];
                let incoming_idx = incoming.iter().position(|g| *g == gate).unwrap();
                incoming.remove(incoming_idx);
                if incoming.is_empty() {
                    next_layer.push(other_gate);
                }
            }
        }

        result.push(-1);
        std::mem::swap(&mut current_layer, &mut next_layer);
        next_layer.clear();
    }

    assert!(incoming_edges.iter().all(|e| e.is_empty()));
    assert!(added_nodes == 0 || result.pop() == Some(-1)); // Don't need the last -1.
    result
}

fn state_dist_cost(
    problem: &Problem,
    bits: &Bits,
    layers: &[i8],
    dist: &Vec<Vec<u8>>,
    layer_discount_factor: f32,
) -> f32 {
    let mut factor = 1.0;

    layers
        .iter()
        .copied()
        .map(|gate| {
            if gate == -1 {
                factor *= layer_discount_factor;
                return 0.0;
            }
            let gate_dist = gate_dist_cost(problem, gate as u8, bits, dist) as f32;

            // Look up the distance between the physical bits
            factor * gate_dist
        })
        .sum()
}

fn gate_dist_cost(problem: &Problem, gate: u8, bits: &Bits, dist: &Vec<Vec<u8>>) -> u8 {
    let (lb1, lb2) = problem.gates[gate as usize];
    let pb1 = bits.backward[lb1 as usize];
    let pb2 = bits.backward[lb2 as usize];
    dist[pb1 as usize][pb2 as usize]
}

fn mk_dist_map(problem: &Problem) -> Vec<Vec<u8>> {
    let mut dist = vec![vec![u8::MAX; problem.n_physical_bits]; problem.n_physical_bits];

    for i in 0..problem.n_physical_bits {
        dist[i][i] = 0;
    }

    for (i, j) in problem.topology.iter() {
        dist[*i as usize][*j as usize] = 1;
        dist[*j as usize][*i as usize] = 1;
    }

    for k in 0..problem.n_physical_bits {
        for i in 0..problem.n_physical_bits {
            for j in 0..problem.n_physical_bits {
                dist[i][j] = dist[i][j].min(dist[i][k].saturating_add(dist[k][j]));
            }
        }
    }

    for i in 0..problem.n_physical_bits {
        for j in 0..problem.n_physical_bits {
            if dist[i][j] > 0 {
                dist[i][j] -= 1;
            }
        }
    }

    dist
}

fn get_cost(
    problem: &Problem,
    params: &BeamSearchParams,
    state: &State,
    n_swaps: u16,
    toposort_cache: &mut HashMap<u64, GateLayers>,
    dist_map: &Vec<Vec<u8>>,
) -> f32 {
    let depth = *state.time.iter().max().unwrap();

    let heuristic_cost = state_dist_cost(
        problem,
        &state.bits,
        cached_toposort_layers(toposort_cache, problem, state.placed_gates),
        dist_map,
        params.layer_discount,
    );

    params.depth_cost * depth as f32
        + params.swap_cost * n_swaps as f32
        + params.heuristic_cost_factor * params.swap_cost * heuristic_cost
}

pub fn solve_heur_beam_full(problem: &Problem, params: &BeamSearchParams) -> Option<Solution> {
    let mut best: Option<Node> = None;
    solve_heur_beam(problem, params, |this_node| {
        if best
            .as_ref()
            .map(|best_node| best_node.f_cost.0)
            .unwrap_or(f32::INFINITY)
            > this_node.f_cost.0
        {
            best = Some(this_node);
        }
    });

    best.map(|node| {
        let depth = *node.state.time.iter().max().unwrap();
        let n_swaps = node.n_swaps;
        let mut node = &node;
        let mut gates = vec![];
        while let Some((prev_node, gate)) = node.parent.as_ref() {
            gates.push(*gate);
            node = &prev_node;
        }

        gates.reverse();
        let input_bits = node.state.bits.forward.iter().copied().collect();
        Solution {
            depth,
            n_swaps,
            gates,
            input_bits,
        }
    })
}

pub fn solve_heur_beam(problem: &Problem, params: &BeamSearchParams, mut f: impl FnMut(Node)) {
    assert!(problem.n_physical_bits == problem.n_logical_bits);

    let dist_map = mk_dist_map(problem);
    let mut this_layer = initial_permutations(params, problem, &dist_map);
    let mut next_macro_layer: MinMaxHeap<Rc<Node>> = Default::default();
    let mut toposort_cache: HashMap<u64, GateLayers> = Default::default();
    let mut best_state_scores: HashMap<u64, HashMap<SmallVec<[u8; 16]>, f32>> = Default::default();

    //
    // BEAM SEARCH: MACRO LAYERS
    //
    for n_gates_placed in 0..(problem.gates.len()) {
        println!("Placing gate {}", n_gates_placed);
        // We will never use any previous layers' placed_gates bit sets because
        // they cannot be the same, since the size of the set is strictly increasing
        // with macro iterations.
        best_state_scores.clear();
        toposort_cache.clear();

        let use_swap_cost = n_gates_placed > 0;

        assert!(!this_layer.is_empty());
        assert!(next_macro_layer.is_empty());

        //
        // BEAM SEARCH MICRO LAYERS
        //
        {
            let mut swap_depth = problem.n_logical_bits;
            let mut micro_iter = 0;
            let mut next_micro_layer: MinMaxHeap<Rc<Node>> = Default::default();

            while micro_iter < swap_depth.min(params.width) {
                println!("  micro beam {}", micro_iter);
                assert!(next_micro_layer.is_empty());

                for node in this_layer.iter() {
                    let gate_layers = cached_toposort_layers(
                        &mut toposort_cache,
                        problem,
                        node.state.placed_gates,
                    );
                    let is_micro_terminal = gate_layers.iter().take_while(|x| **x != -1).any(|g| {
                        gate_dist_cost(problem, *g as u8, &node.state.bits, &dist_map) == 0
                    });

                    if is_micro_terminal {
                        // add macro node
                        if node.f_cost
                            < next_macro_layer
                                .peek_max()
                                .map(|n| n.f_cost)
                                .unwrap_or(OrderedFloat(f32::INFINITY))
                        {
                            println!("  micro terminal cost {}", node.f_cost);
                            next_macro_layer.push(node.clone());

                            if next_macro_layer.len() > params.width {
                                next_macro_layer.pop_max();
                            }
                        }
                    }

                    println!(
                        "  swap term={} {:?}\n    topo {:?}",
                        is_micro_terminal, node.state, gate_layers
                    );

                    for (pb1, pb2) in problem.topology.iter().copied() {
                        let mut new_state = node.state.clone();

                        let start_time =
                            new_state.time[pb1 as usize].max(new_state.time[pb2 as usize]);
                        let finish_time =
                            start_time + if use_swap_cost { problem.swap_dt } else { 0 };

                        new_state.time[pb1 as usize] = finish_time;
                        new_state.time[pb2 as usize] = finish_time;

                        let n_swaps = node.n_swaps + if use_swap_cost { 1 } else { 0 };

                        let new_cost = OrderedFloat(get_cost(
                            problem,
                            params,
                            &new_state,
                            n_swaps,
                            &mut toposort_cache,
                            &dist_map,
                        ));

                        let add_node = next_micro_layer.len() < params.width
                            || next_micro_layer
                                .peek_max()
                                .map(|n| n.f_cost)
                                .unwrap_or(OrderedFloat(f32::INFINITY))
                                > new_cost;
                        if !add_node {
                            continue;
                        }

                        let existing_cost = best_state_scores
                            .entry(new_state.placed_gates)
                            .or_default()
                            .entry(new_state.bits.forward.clone())
                            .or_insert(f32::INFINITY);

                        if *existing_cost <= node.f_cost.0 {
                            continue;
                        }

                        println!("    New micrlayer {} best micro {} best macro {}", new_cost, next_micro_layer
                        .peek_min()
                        .map(|x| x.f_cost)
                        .unwrap_or(f32::INFINITY.into()),next_macro_layer
                        .peek_min()
                        .map(|x| x.f_cost)
                        .unwrap_or(f32::INFINITY.into()));

                        if new_cost
                            < next_macro_layer
                                .peek_min()
                                .map(|x| x.f_cost)
                                .unwrap_or(f32::INFINITY.into())
                            && 2 * micro_iter > swap_depth
                        {
                            println!(" * placing {}: Increasing swap_depth to {}", n_gates_placed, 2 * micro_iter);
                            swap_depth = swap_depth.max(2 * micro_iter);
                        }

                        let new_node = Rc::new(Node {
                            f_cost: new_cost,
                            n_swaps,
                            state: new_state,
                            parent: if use_swap_cost {
                                Some((node.clone(), Gate::SwapGate(pb1, pb2)))
                            } else {
                                None // No need to remember the parent, we're just modifying the initial bits.},
                            },
                        });

                        next_micro_layer.push(new_node);

                        if next_micro_layer.len() > params.width {
                            let prev_max = next_micro_layer.pop_max().unwrap();
                            let map = best_state_scores
                                .get_mut(&prev_max.state.placed_gates)
                                .unwrap();

                            if map[&prev_max.state.bits.forward] == prev_max.f_cost.0 {
                                map.remove(&prev_max.state.bits.forward);
                            }
                        }
                    }
                }

                this_layer = std::mem::take(&mut next_micro_layer).into_vec();
                micro_iter += 1;
            }
        }

        this_layer = std::mem::take(&mut next_macro_layer).into_vec();

        assert!(!this_layer.is_empty());
        assert!(next_macro_layer.is_empty());

        println!("Macro beam");

        for node in this_layer.iter() {
            // Get the layers definition for this node
            let gate_layers =
                cached_toposort_layers(&mut toposort_cache, problem, node.state.placed_gates);

            // Place each gate that has dist 0
            let placeable_gates = gate_layers
                .iter()
                .take_while(|x| **x != -1)
                .filter(|g| gate_dist_cost(problem, **g as u8, &node.state.bits, &dist_map) == 0)
                .map(|g| *g as u8)
                .collect::<SmallVec<[u8; 16]>>();

            for place_gate in placeable_gates {
                let mut new_state = node.state.clone();
                new_state.placed_gates |= 1 << place_gate;

                let (lb1, lb2) = problem.gates[place_gate as usize];
                let pb1 = new_state.bits.backward[lb1 as usize];
                let pb2 = new_state.bits.backward[lb2 as usize];
                let start_time = new_state.time[pb1 as usize].max(new_state.time[pb2 as usize]);
                let finish_time = start_time + problem.gate_dt;

                new_state.time[pb1 as usize] = finish_time;
                new_state.time[pb2 as usize] = finish_time;

                let new_cost = get_cost(
                    problem,
                    params,
                    &new_state,
                    node.n_swaps,
                    &mut toposort_cache,
                    &dist_map,
                );

                let new_node = Node {
                    f_cost: OrderedFloat(new_cost),
                    n_swaps: node.n_swaps,
                    parent: Some((node.clone(), Gate::InputGate(place_gate))),
                    state: new_state,
                };

                if n_gates_placed + 1 == problem.gates.len() {
                    f(new_node);
                } else {
                    // Put the node into the swap-phase
                    // We don't need to check best_cost_states here
                    // because they will all be unique.

                    if new_node.f_cost
                        < next_macro_layer
                            .peek_max()
                            .map(|n| n.f_cost)
                            .unwrap_or(OrderedFloat(f32::INFINITY))
                    {
                        next_macro_layer.push(Rc::new(new_node));
                        if next_macro_layer.len() > params.width {
                            next_macro_layer.pop_max();
                        }
                    }
                }
            }
        }

        this_layer = std::mem::take(&mut next_macro_layer).into_vec();
    }
}

fn initial_permutations(
    params: &BeamSearchParams,
    problem: &Problem,
    dist_map: &Vec<Vec<u8>>,
) -> Vec<Rc<Node>> {
    let mut rng = thread_rng();

    // Generate random starting permutations
    let mut initial_permutations: Vec<Rc<Node>> = Vec::with_capacity(params.width);
    let initial_gate_layers = toposort_layers(problem, 0);

    for _ in 0..params.width {
        let mut bits_forward: SmallVec<[u8; 16]> = (0..(problem.n_logical_bits as u8)).collect();
        bits_forward.shuffle(&mut rng);

        let mut bits_backward = bits_forward.clone();
        for i in 0..bits_forward.len() {
            bits_backward[bits_forward[i] as usize] = i as u8;
        }

        let bits = Bits {
            forward: bits_forward,
            backward: bits_backward,
        };

        let h_cost = OrderedFloat(state_dist_cost(
            problem,
            &bits,
            &initial_gate_layers,
            dist_map,
            params.layer_discount,
        ));

        initial_permutations.push(Rc::new(Node {
            f_cost: h_cost,
            n_swaps: 0,
            state: State {
                time: (0..problem.n_logical_bits).map(|_| 0).collect(),
                bits,
                placed_gates: 0,
            },
            parent: None,
        }));
    }
    initial_permutations
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    pub fn test_bits() {
        let mut bits = Bits::new(5);

        println!("{:?}", bits);
        assert!(bits.forward == [0, 1, 2, 3, 4].into());
        assert!(bits.backward == [0, 1, 2, 3, 4].into());

        bits.swap_physical(0, 1);

        println!("{:?}", bits);
        assert!(bits.forward == [1, 0, 2, 3, 4].into());
        assert!(bits.backward == [1, 0, 2, 3, 4].into());

        bits.swap_physical(1, 2);

        println!("{:?}", bits);
        assert!(bits.forward == [1, 2, 0, 3, 4].into());
        assert!(bits.backward == [2, 0, 1, 3, 4].into());

        bits.swap_physical(0, 4);

        println!("{:?}", bits);
        assert!(bits.forward == [4, 2, 0, 3, 1].into());
        assert!(bits.backward == [2, 4, 1, 3, 0].into());
    }

    #[test]
    pub fn test_toposort() {
        let problem = Problem {
            gate_dt: 0,
            n_logical_bits: 4,
            gates: vec![(0, 1), (2, 3), (1, 2), (0, 3), (1, 2)],
            n_physical_bits: 4,
            topology: vec![],
            swap_dt: 0,
        };

        let layers = toposort_layers(&problem, 0);
        println!("Layers {:?}", layers);
        assert!(layers == [0, 1, -1, 2, 3, -1, 4].into());

        let problem = Problem {
            gate_dt: 0,
            n_logical_bits: 4,
            gates: vec![(0, 1), (0, 1), (2, 3), (2, 3)],
            n_physical_bits: 4,
            topology: vec![],
            swap_dt: 0,
        };

        let layers = toposort_layers(&problem, 0);
        println!("Layers {:?}", layers);
        assert!(layers == [0, 2, -1, 1, 3].into());

        let layers = toposort_layers(&problem, 0 | (1 << 0));
        println!("Layers {:?}", layers);
        assert!(layers == [1, 2, -1, 3].into());
    }

    #[test]
    fn test_dist_map() {
        let problem = Problem {
            gate_dt: 0,
            n_logical_bits: 4,
            gates: vec![],
            n_physical_bits: 4,
            topology: vec![(1, 2), (0, 1), (2, 3)],
            swap_dt: 0,
        };

        let dist_map = mk_dist_map(&problem);

        assert!(
            dist_map
                == vec![
                    vec![0, 0, 1, 2],
                    vec![0, 0, 0, 1],
                    vec![1, 0, 0, 0],
                    vec![2, 1, 0, 0]
                ]
        );
    }

    #[test]
    fn test_solve_small() {
        let problem = Problem {
            n_logical_bits: 4,
            gates: vec![(0, 1)],
            n_physical_bits: 4,
            topology: vec![(1, 2), (0, 1), (2, 3)],
            swap_dt: 3,
            gate_dt: 1,
        };

        let params = BeamSearchParams {
            width: 25,
            layer_discount: 0.5,
            depth_cost: 10.0,
            swap_cost: 0.5,
            heuristic_cost_factor: 0.2,
        };

        let solution = solve_heur_beam_full(&problem, &params).unwrap();
        assert!(solution.n_swaps == 0 && solution.depth == 1*problem.gate_dt);
        println!("SOLUTION {:?}", solution);
    }

    #[test]
    fn test_solve_medium() {
        let problem = Problem {
            n_logical_bits: 4,
            gates: vec![(1,2),(3,0),(3,1),(0,2)],
            n_physical_bits: 4,
            topology: vec![(0,1),(1,2),(2,3)],
            swap_dt: 3,
            gate_dt: 1,
        };

        let params = BeamSearchParams {
            width: 25,
            layer_discount: 0.5,
            depth_cost: 1.0,
            swap_cost: 0.1,
            heuristic_cost_factor: 1.0,
        };

        let solution = solve_heur_beam_full(&problem, &params).unwrap();
        assert!(solution.n_swaps == 1 && solution.depth == 2*problem.gate_dt + problem.swap_dt);
        println!("SOLUTION {:?}", solution);
    }
}

fn main() {
    println!("Hello, world!");
}
