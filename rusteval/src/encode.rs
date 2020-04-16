extern crate libc;

use self::libc::{c_long, c_char};

pub fn card_format(card: c_long) -> String {
    format!("{:08b} {:08b} {:08b} {:08b}", card >> 24, (card >> 16) & 0xFF, (card >> 8) & 0xFF, card & 0xFF)
}

#[no_mangle]
pub extern fn encode(rank: c_char, suit: c_char) -> c_long {
    let prime_rank: c_long = [2,3,5,7,11,13,17,19,23,29,31,37,41][rank as usize];
    (1 << (12 + suit)) as c_long | (1 << (16 + rank)) as c_long | ((rank as c_long) << 8) | prime_rank
}
