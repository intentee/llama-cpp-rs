/// Is memory mapping supported according to llama.cpp.
///
/// ```
/// # use llama_cpp_bindings::mmap_supported;
/// let mmap_supported = mmap_supported();
/// if mmap_supported {
///   println!("mmap_supported!");
/// }
/// ```
#[must_use]
pub fn mmap_supported() -> bool {
    unsafe { llama_cpp_bindings_sys::llama_supports_mmap() }
}
