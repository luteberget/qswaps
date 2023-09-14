use std::rc::Rc;

use smallvec::SmallVec;

pub struct Problem {
    pub gates: Vec<(u8, u8)>,
    pub topology: Vec<(u8, u8)>,
}

#[derive(PartialEq, Eq, PartialOrd, Ord)]
pub struct State {
    pub score :usize,
    pub bits: SmallVec<[u8;16]>,
    pub next_gates :SmallVec<[u8;8]>,
    pub place_gates :SmallVec<[Result<u8,(u8,u8)>; 8]>,
    pub parent :Option<Rc<State>>,
}

pub fn succ(problem :&Problem, state :&State, mut f :impl FnMut(State)) {
    todo!()
}

pub fn solve_heuristic(problem: &Problem, initial_state_bits :Vec<u8>) {

    let initial_state = State {
        score: 0,
        bits: initial_state_bits.into(),
        next_gates: todo!(),
        parent: None,
        place_gates: std::iter::empty().collect(),
    };

    const WIDTH :usize = 500;

    let mut states = vec![initial_state];
    
    let mut best :Option<State> = None;
    loop {
        let mut next_states = min_max_heap::MinMaxHeap::new();

        for state in states.iter() {
            // Generate succesor
            succ(problem, state, |next| {
                next_states.push(next);
                if next_states.len() > WIDTH {
                    next_states.pop_max();
                }
            });
        }

        states = next_states.into_vec();
    }
}

fn main() {
    println!("Hello, world!");
}
