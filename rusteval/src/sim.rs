extern crate libc;
extern crate rand;

use rank::rank;
use self::libc::{c_long, c_float, c_int};
use self::rand::distributions::{IndependentSample, Range};

#[no_mangle]
pub extern fn winner(hand1: *const [c_long; 4], hand2: *const [c_long; 4], board: *const [c_long; 5]) -> c_int {
    let rank1 = unsafe { best_rank_w_board(*hand1, *board) };
    let rank2 = unsafe { best_rank_w_board(*hand2, *board) };
    if rank1 < rank2 {
        1
    } else if rank1 > rank2 {
        -1
    } else {
        0
    }
}

#[no_mangle]
pub extern fn hand_vs_hand(hand1: *const [c_long; 4], hand2: *const [c_long; 4], deck: *const [c_long; 44], iterations: c_int) -> c_float {
    let mut deck_copy = unsafe { *deck };
    unsafe { run_hand_vs_hand(*hand1, *hand2, &mut deck_copy, iterations) }
}

#[no_mangle]
pub extern fn hand_with_board_rank(hand: *const [c_long; 4], board: *const [c_long; 5]) -> c_int {
    unsafe { best_rank_w_board(*hand, *board) }
}

#[no_mangle]
pub extern fn holdem_hand_with_board_rank(hand: *const [c_long; 2], board: *const [c_long; 5]) -> c_int {
    unsafe { holdem_best_rank_w_board(*hand, *board) }
}

#[no_mangle]
pub extern fn holdem_winner(hand1: *const [c_long; 2], hand2: *const [c_long; 2], board: *const [c_long; 5]) -> c_int {
    let rank1 = unsafe { holdem_best_rank_w_board(*hand1, *board) };
    let rank2 = unsafe { holdem_best_rank_w_board(*hand2, *board) };
    if rank1 < rank2 {
        1
    } else if rank1 > rank2 {
        -1
    } else {
        0
    }
}

fn run_hand_vs_hand(hand1: [c_long; 4], hand2: [c_long; 4], deck: &mut [c_long; 44], iterations: c_int) -> c_float {
    let mut wins = 0;
    for _ in 0..iterations {
        shuffle_5(deck);
        let rank1 = best_rank(hand1, deck);
        let rank2 = best_rank(hand2, deck);
        if rank1 < rank2 {
            wins += 1;
        }
    }
    wins as c_float / iterations as c_float
}

fn holdem_best_rank_w_board(hand: [c_long; 2], board: [c_long; 5]) -> i32 {
    let mut cur_rank: i32 = 0xFFFF;
    for bi in 0..10 {
        let bc = [(0,1,2),(0,1,3),(0,1,4),(0,2,3),(0,2,4),(0,3,4),(1,2,3),(1,2,4),(1,3,4),(2,3,4)][bi];
        let new_rank = rank(hand[0], hand[1], board[bc.0], board[bc.1], board[bc.2]);
        if new_rank < cur_rank {
            cur_rank = new_rank;
        }
    }
    for hi in 0..2 {
        for bi in 0..5 {
            let bc = [(0,1,2,3),(0,1,2,4),(0,1,3,4),(0,2,3,4),(1,2,3,4)][bi];
            let new_rank = rank(hand[hi], board[bc.0], board[bc.1], board[bc.2], board[bc.3]);
            if new_rank < cur_rank {
                cur_rank = new_rank;
            }
        }
    }
    let new_rank = rank(board[0], board[1], board[2], board[3], board[4]);
    if new_rank < cur_rank {
        cur_rank = new_rank;
    }
    cur_rank
}

fn best_rank_w_board(hand: [c_long; 4], board: [c_long; 5]) -> i32 {
    let mut cur_rank: i32 = 0xFFFF;
    for hi in 0..6 {
        let hc = [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)][hi];
        for bi in 0..10 {
            let bc = [(0,1,2),(0,1,3),(0,1,4),(0,2,3),(0,2,4),(0,3,4),(1,2,3),(1,2,4),(1,3,4),(2,3,4)][bi];
            let new_rank = rank(hand[hc.0], hand[hc.1], board[bc.0], board[bc.1], board[bc.2]);
            if new_rank < cur_rank {
                cur_rank = new_rank;
            }
        }
    }
    cur_rank
}

fn best_rank(hand: [c_long; 4], deck: &mut [c_long; 44]) -> i32 {
    let mut cur_rank: i32 = 0xFFFF;
    for hi in 0..6 {
        let hc = [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)][hi];
        for bi in 0..10 {
            let bc = [(0,1,2),(0,1,3),(0,1,4),(0,2,3),(0,2,4),(0,3,4),(1,2,3),(1,2,4),(1,3,4),(2,3,4)][bi];
            let new_rank = rank(hand[hc.0], hand[hc.1], deck[bc.0], deck[bc.1], deck[bc.2]);
            if new_rank < cur_rank {
                cur_rank = new_rank;
            }
        }
    }
    cur_rank
}

fn shuffle_5(deck: &mut [c_long; 44]) {
    let mut rng = rand::thread_rng();
    let mut x: c_long;

    for start in 0..5 {
        let i = Range::new(start, 44).ind_sample(&mut rng);
        x = deck[start];
        deck[start] = deck[i];
        deck[i] = x;
    }
}
