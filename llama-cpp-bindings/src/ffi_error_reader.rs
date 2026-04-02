use std::ffi::{CStr, c_char};

/// Reads a C error string, converts to Rust `String`, and frees the C memory.
///
/// # Safety
///
/// `error_ptr` must be either null or a valid pointer to a null-terminated
/// C string allocated by `llama_rs_dup_string`.
pub unsafe fn read_and_free_cpp_error(error_ptr: *mut c_char) -> String {
    if error_ptr.is_null() {
        return "unknown error".to_owned();
    }

    let message = unsafe { CStr::from_ptr(error_ptr) }
        .to_string_lossy()
        .into_owned();

    unsafe { llama_cpp_bindings_sys::llama_rs_string_free(error_ptr) };

    message
}

#[cfg(test)]
mod tests {
    use super::read_and_free_cpp_error;

    #[test]
    fn returns_unknown_for_null_pointer() {
        let result = unsafe { read_and_free_cpp_error(std::ptr::null_mut()) };

        assert_eq!(result, "unknown error");
    }
}
