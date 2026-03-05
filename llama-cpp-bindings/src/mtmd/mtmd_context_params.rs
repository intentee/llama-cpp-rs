use std::ffi::{CStr, CString};

/// Configuration parameters for MTMD context
///
/// # Examples
///
/// ```
/// use llama_cpp_bindings::mtmd::{MtmdContextParams, mtmd_default_marker};
/// use std::ffi::CString;
///
/// let params = MtmdContextParams {
///     use_gpu: false,
///     print_timings: true,
///     n_threads: 4,
///     media_marker: CString::new(mtmd_default_marker()).unwrap(),
/// };
/// ```
#[derive(Debug, Clone)]
pub struct MtmdContextParams {
    /// Whether to use GPU acceleration
    pub use_gpu: bool,
    /// Whether to print timing information
    pub print_timings: bool,
    /// Number of threads to use for processing
    pub n_threads: i32,
    /// Media marker string used to identify media positions in text
    pub media_marker: CString,
}

impl Default for MtmdContextParams {
    fn default() -> Self {
        unsafe { llama_cpp_bindings_sys::mtmd_context_params_default() }.into()
    }
}

impl From<&MtmdContextParams> for llama_cpp_bindings_sys::mtmd_context_params {
    fn from(params: &MtmdContextParams) -> Self {
        let mut context = unsafe { llama_cpp_bindings_sys::mtmd_context_params_default() };
        let MtmdContextParams {
            use_gpu,
            print_timings,
            n_threads,
            media_marker,
        } = params;

        context.use_gpu = *use_gpu;
        context.print_timings = *print_timings;
        context.n_threads = *n_threads;
        context.media_marker = media_marker.as_ptr();

        context
    }
}

impl From<llama_cpp_bindings_sys::mtmd_context_params> for MtmdContextParams {
    fn from(params: llama_cpp_bindings_sys::mtmd_context_params) -> Self {
        Self {
            use_gpu: params.use_gpu,
            print_timings: params.print_timings,
            n_threads: params.n_threads,
            media_marker: unsafe { CStr::from_ptr(params.media_marker) }.to_owned(),
        }
    }
}
