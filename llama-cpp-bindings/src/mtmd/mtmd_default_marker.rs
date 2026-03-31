use std::ffi::CStr;

/// Get the default media marker string.
///
/// Returns the default marker used to identify media positions in text
/// (typically `"<__media__>"`). This marker should be used in your input text
/// to indicate where media content should be inserted.
///
/// # Examples
///
/// ```
/// use llama_cpp_bindings::mtmd::mtmd_default_marker;
///
/// let marker = mtmd_default_marker();
/// assert!(!marker.is_empty());
///
/// let text = format!("Describe this image: {}", marker);
/// assert!(text.contains(marker));
/// ```
#[must_use]
pub fn mtmd_default_marker() -> &'static str {
    unsafe {
        let c_str = llama_cpp_bindings_sys::mtmd_default_marker();
        CStr::from_ptr(c_str).to_str().unwrap_or("<__media__>")
    }
}

#[cfg(test)]
mod tests {
    use super::mtmd_default_marker;

    #[test]
    fn returns_non_empty_string() {
        let marker = mtmd_default_marker();
        assert!(!marker.is_empty());
    }
}
