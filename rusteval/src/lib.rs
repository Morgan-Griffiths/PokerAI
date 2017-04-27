extern crate libc;
use libc::{c_long, size_t, c_char};

use std::ffi::CStr;

#[no_mangle]
pub extern fn process(s: *const c_char) -> c_long {
    let st = unsafe { CStr::from_ptr(s) };
    let b = st.to_bytes();
    // let c: c_char = b'A' as c_char;
    println!("{:?}", b);
    12
}
