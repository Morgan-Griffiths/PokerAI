extern crate libc;

use self::libc::{c_long, c_char};
use std::ffi::CStr;

#[no_mangle]
pub extern fn encode(s: *const c_char) -> c_long {
    let st = unsafe { CStr::from_ptr(s) };
    let b = st.to_bytes();
    if b.len() != 2 {
        return -1;
    }
    let rank = match rank_from_char(b[0]) {
        None => return -1,
        Some(x) => x
    };
    let suit = match suit_from_char(b[1]) {
        None => return -1,
        Some(x) => x
    };
    return rank | suit;
}

#[no_mangle]
pub extern fn decode(card: c_long) -> i16 {
    let rank: i16 = [b'2',b'3',b'4',b'5',b'6',b'7',b'8',b'9',b'T',b'J',b'Q',b'K',b'A'][((card >> 8) & 0xF) as usize] as i16;
    let suit: i16 = [0,b'c',b'd',0,b'h',0,0,0,b's'][((card >> 12) & 0xF) as usize] as i16;
    return (rank << 8) | suit;
}

fn suit_from_char(c: u8) -> Option<i64> {
    match c {
        b's' => Some(0b1000 << 12),
        b'h' => Some(0b0100 << 12),
        b'd' => Some(0b0010 << 12),
        b'c' => Some(0b0001 << 12),
        _ => None
    }
}

fn rank_from_char(c: u8) -> Option<i64> {
    match c {
        b'2'      => Some((0b00000000_00000001 << 16) | (0 << 8) | 2),
        b'3'      => Some((0b00000000_00000010 << 16) | (1 << 8) | 3),
        b'4'      => Some((0b00000000_00000100 << 16) | (2 << 8) | 5),
        b'5'      => Some((0b00000000_00001000 << 16) | (3 << 8) | 7),
        b'6'      => Some((0b00000000_00010000 << 16) | (4 << 8) | 11),
        b'7'      => Some((0b00000000_00100000 << 16) | (5 << 8) | 13),
        b'8'      => Some((0b00000000_01000000 << 16) | (6 << 8) | 17),
        b'9'      => Some((0b00000000_10000000 << 16) | (7 << 8) | 19),
        b'T'|b't' => Some((0b00000001_00000000 << 16) | (8 << 8) | 23),
        b'J'|b'j' => Some((0b00000010_00000000 << 16) | (9 << 8) | 29),
        b'Q'|b'q' => Some((0b00000100_00000000 << 16) | (10 << 8) | 31),
        b'K'|b'k' => Some((0b00001000_00000000 << 16) | (11 << 8) | 37),
        b'A'|b'a' => Some((0b00010000_00000000 << 16) | (12 << 8) | 41),
        _ => None
    }
}
