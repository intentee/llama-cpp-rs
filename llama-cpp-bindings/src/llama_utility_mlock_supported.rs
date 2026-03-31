/// Is memory locking supported according to llama.cpp.
///
/// ```
/// # use llama_cpp_bindings::mlock_supported;
/// let mlock_supported = mlock_supported();
/// if mlock_supported {
///    println!("mlock_supported!");
/// }
/// ```
#[must_use]
pub fn mlock_supported() -> bool {
    unsafe { llama_cpp_bindings_sys::llama_supports_mlock() }
}

#[cfg(test)]
mod tests {
    use super::mlock_supported;

    #[test]
    fn returns_bool_without_panic() {
        let _supported: bool = mlock_supported();
    }
}
