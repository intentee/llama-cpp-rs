//! Error type for sequence state file save operations.

use std::ffi::NulError;
use std::path::PathBuf;

/// Failed to save a sequence state file.
#[derive(Debug, Eq, PartialEq, thiserror::Error)]
pub enum SaveSeqStateError {
    /// llama.cpp failed to save the sequence state file
    #[error("Failed to save sequence state file")]
    FailedToSave,

    /// null byte in string
    #[error("null byte in string {0}")]
    NullError(#[from] NulError),

    /// failed to convert path to str
    #[error("failed to convert path {0} to str")]
    PathToStrError(PathBuf),
}
