#![cfg(all(feature = "llm-tests", feature = "mtmd"))]

use anyhow::Result;
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::mtmd::MtmdContextParams;
use std::ffi::CString;

#[test]
fn multimodal_context_initializes_from_model() -> Result<()> {
    // This test only verifies that the mtmd module compiles and the types are usable.
    // Full multimodal testing requires a model with a multimodal projection file,
    // which is too large for automated testing.

    let backend = LlamaBackend::init()?;
    let _model_params = LlamaModelParams::default();

    let _ctx_params = MtmdContextParams {
        use_gpu: false,
        print_timings: false,
        n_threads: 1,
        media_marker: CString::new(llama_cpp_2::mtmd::mtmd_default_marker().to_string())?,
    };

    let _context_params = LlamaContextParams::default();

    drop(backend);

    Ok(())
}
