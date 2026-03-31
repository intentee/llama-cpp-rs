use std::env;
use std::path::PathBuf;

use anyhow::Result;

fn required_env(var_name: &str) -> Result<String> {
    env::var(var_name).map_err(|_| {
        anyhow::anyhow!(
            "Required env var {var_name} is not set. Source .env.test or set it manually."
        )
    })
}

fn hf_repo() -> Result<String> {
    required_env("LLAMA_TEST_HF_REPO")
}

fn hf_model() -> Result<String> {
    required_env("LLAMA_TEST_HF_MODEL")
}

fn hf_mmproj() -> String {
    env::var("LLAMA_TEST_HF_MMPROJ").unwrap_or_default()
}

fn hf_embed_repo() -> Result<String> {
    required_env("LLAMA_TEST_HF_EMBED_REPO")
}

fn hf_embed_model() -> Result<String> {
    required_env("LLAMA_TEST_HF_EMBED_MODEL")
}

fn hf_encoder_repo() -> Result<String> {
    required_env("LLAMA_TEST_HF_ENCODER_REPO")
}

fn hf_encoder_model() -> Result<String> {
    required_env("LLAMA_TEST_HF_ENCODER_MODEL")
}

fn download_file(repo: &str, filename: &str) -> Result<PathBuf> {
    let path = hf_hub::api::sync::ApiBuilder::new()
        .with_progress(true)
        .build()?
        .model(repo.to_string())
        .get(filename)?;

    Ok(path)
}

/// Downloads the configured test model from Hugging Face.
///
/// # Errors
/// Returns an error if the required environment variables are not set or the download fails.
pub fn download_model() -> Result<PathBuf> {
    download_file(&hf_repo()?, &hf_model()?)
}

/// Downloads the configured mmproj file from Hugging Face.
///
/// # Errors
/// Returns an error if the required environment variables are not set or the download fails.
pub fn download_mmproj() -> Result<PathBuf> {
    download_file(&hf_repo()?, &hf_mmproj())
}

/// Downloads the configured embedding model from Hugging Face.
///
/// # Errors
/// Returns an error if the required environment variables are not set or the download fails.
pub fn download_embedding_model() -> Result<PathBuf> {
    download_file(&hf_embed_repo()?, &hf_embed_model()?)
}

/// Downloads the configured encoder model from Hugging Face.
///
/// # Errors
/// Returns an error if the required environment variables are not set or the download fails.
pub fn download_encoder_model() -> Result<PathBuf> {
    download_file(&hf_encoder_repo()?, &hf_encoder_model()?)
}

/// Returns whether a multimodal projection model is configured.
#[must_use]
pub fn has_mmproj() -> bool {
    !hf_mmproj().is_empty()
}

/// Returns the path to the test fixtures directory.
#[must_use]
pub fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures")
}

/// Loads the default test model and backend.
///
/// # Errors
/// Returns an error if the backend cannot be initialized or the model cannot be loaded.
pub fn load_default_model() -> Result<(crate::llama_backend::LlamaBackend, crate::model::LlamaModel)>
{
    let backend = crate::llama_backend::LlamaBackend::init()?;
    let model_path = download_model()?;
    let model_params = crate::model::params::LlamaModelParams::default();
    let model = crate::model::LlamaModel::load_from_file(&backend, model_path, &model_params)?;
    Ok((backend, model))
}

/// Loads the default test model, backend, and multimodal context.
///
/// # Errors
/// Returns an error if the backend cannot be initialized, the model cannot be loaded,
/// or the multimodal projection file is not configured.
#[cfg(feature = "mtmd")]
pub fn load_default_mtmd() -> Result<(
    crate::llama_backend::LlamaBackend,
    crate::model::LlamaModel,
    crate::mtmd::MtmdContext,
)> {
    if !has_mmproj() {
        anyhow::bail!("MTMD tests require mmproj — set LLAMA_TEST_HF_MMPROJ");
    }

    let backend = crate::llama_backend::LlamaBackend::init()?;
    let model_path = download_model()?;
    let mmproj_path = download_mmproj()?;
    let model_params = crate::model::params::LlamaModelParams::default();
    let model = crate::model::LlamaModel::load_from_file(&backend, &model_path, &model_params)?;
    let mtmd_params = crate::mtmd::MtmdContextParams::default();
    let mmproj_str = mmproj_path
        .to_str()
        .ok_or_else(|| anyhow::anyhow!("mmproj path is not valid UTF-8"))?;
    let mtmd_ctx = crate::mtmd::MtmdContext::init_from_file(mmproj_str, &model, &mtmd_params)?;
    Ok((backend, model, mtmd_ctx))
}

#[cfg(test)]
mod tests {
    #[test]
    fn required_env_returns_error_for_missing_var() {
        let result = super::required_env("LLAMA_TEST_NONEXISTENT_VAR_THAT_SHOULD_NOT_EXIST");

        assert!(result.is_err());
    }

    #[cfg(feature = "mtmd")]
    #[test]
    fn load_default_mtmd_fails_without_mmproj() {
        unsafe { std::env::set_var("LLAMA_TEST_HF_MMPROJ", "") };
        let result = super::load_default_mtmd();
        unsafe { std::env::set_var("LLAMA_TEST_HF_MMPROJ", "mmproj-F16.gguf") };

        assert!(result.is_err());
    }

    #[test]
    fn download_file_with_nonexistent_file_returns_error() {
        let result =
            super::download_file("unsloth/Qwen3.5-0.8B-GGUF", "this-file-does-not-exist.gguf");

        assert!(result.is_err());
    }
}
