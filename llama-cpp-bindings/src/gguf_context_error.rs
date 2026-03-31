//! Error types for GGUF context operations.

use std::ffi::NulError;
use std::path::PathBuf;

/// Errors that can occur when working with GGUF contexts.
#[derive(Debug, thiserror::Error)]
pub enum GgufContextError {
    /// Failed to initialize GGUF context from file
    #[error("Failed to initialize GGUF context from file: {0}")]
    InitFailed(PathBuf),

    /// Key not found in GGUF metadata
    #[error("Key not found in GGUF context: {key}")]
    KeyNotFound {
        /// The key that was not found
        key: String,
    },

    /// Null byte in string
    #[error("null byte in string: {0}")]
    NulError(#[from] NulError),

    /// Path cannot be converted to UTF-8
    #[error("failed to convert path {0} to str")]
    PathToStrError(PathBuf),

    /// Value is not valid UTF-8
    #[error("GGUF value is not valid UTF-8: {0}")]
    Utf8Error(#[from] std::str::Utf8Error),
}
