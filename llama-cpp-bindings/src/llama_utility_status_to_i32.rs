/// Converts a status code to its underlying `i32` representation.
#[must_use]
pub fn status_to_i32(status: llama_cpp_bindings_sys::llama_rs_status) -> i32 {
    status
}

#[cfg(test)]
mod tests {
    use super::status_to_i32;

    #[test]
    fn preserves_value() {
        assert_eq!(
            status_to_i32(llama_cpp_bindings_sys::LLAMA_RS_STATUS_OK),
            llama_cpp_bindings_sys::LLAMA_RS_STATUS_OK
        );
        assert_eq!(status_to_i32(42), 42);
        assert_eq!(status_to_i32(-1), -1);
    }
}
