extern crate libc;

use self::libc::{c_long, c_int};
use tables;

#[no_mangle]
pub extern fn rank(a: c_long, b: c_long, c: c_long, d: c_long, e: c_long) -> c_int {
    let distinct_index = ((a | b | c | d | e) >> 16) as usize;
    // check if we have a flush
    if a & b & c & d & e & 0xF000 != 0 {
        return tables::FLUSHES[distinct_index] as c_int;
    }
    // check if we have 5 different cards
    if tables::UNIQUE5[distinct_index] != 0 {
        return tables::UNIQUE5[distinct_index] as c_int;
    }
    // don't have 5 different cards, so search final table
    let product = ((a & 0xFF) * (b & 0xFF) * (c & 0xFF) * (d & 0xFF) * (e & 0xFF)) as i32;
    let mut start = 0;
    let mut end = tables::CARD_PRODUCTS.len() - 1;
    let mut guess = end / 2;
    while end >= start {
        let found = tables::CARD_PRODUCTS[guess];
        if found == product {
            return tables::PRODUCT_RANKS[guess] as c_int;
        } else if product > found {
            start = guess + 1;
        } else {
            end = guess - 1;
        }
        guess = (end + start) / 2;
    }
    panic!("couldn't find {} in CARD_PRODUCTS", product);
}

pub fn flush_lookup(a: c_long, b: c_long, c: c_long, d: c_long, e: c_long) -> c_long {
    // check flush
    match a & b & c & d & e & 0xF000 {
        0 => 0,
        _ => (a | b | c | d | e) >> 16
    }
}

pub fn prime_lookup(a: c_long, b: c_long, c: c_long, d: c_long, e: c_long) -> c_long {
    (a & 0xFF) * (b & 0xFF) * (c & 0xFF) * (d & 0xFF) * (e & 0xFF)
}
