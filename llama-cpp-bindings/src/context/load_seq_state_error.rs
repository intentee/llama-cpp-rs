//! Error type for sequence state file load operations.

use std::ffi::NulError;
use std::path::PathBuf;

/// Failed to load a sequence state file.
#[derive(Debug, Eq, PartialEq, thiserror::Error)]
pub enum LoadSeqStateError {
    /// llama.cpp failed to load the sequence state file
    #[error("Failed to load sequence state file")]
    FailedToLoad,

    /// null byte in string
    #[error("null byte in string {0}")]
    NullError(#[from] NulError),

    /// failed to convert path to str
    #[error("failed to convert path {0} to str")]
    PathToStrError(PathBuf),

    /// Insufficient max length
    #[error("max_length is not large enough to hold {n_out} (was {max_tokens})")]
    InsufficientMaxLength {
        /// The length of the loaded sequence
        n_out: usize,
        /// The maximum length
        max_tokens: usize,
    },
}
