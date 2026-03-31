//! Error type for session file save operations.

use std::ffi::NulError;
use std::path::PathBuf;

/// Failed to save a session file.
#[derive(Debug, Eq, PartialEq, thiserror::Error)]
pub enum SaveSessionError {
    /// llama.cpp failed to save the session file
    #[error("Failed to save session file")]
    FailedToSave,

    /// null byte in string
    #[error("null byte in string {0}")]
    NullError(#[from] NulError),

    /// failed to convert path to str
    #[error("failed to convert path {0} to str")]
    PathToStrError(PathBuf),
}
