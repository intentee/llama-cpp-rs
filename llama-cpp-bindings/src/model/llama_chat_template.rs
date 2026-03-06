use std::ffi::{CStr, CString};
use std::str::Utf8Error;

/// A performance-friendly wrapper around [`super::LlamaModel::chat_template`] which is then
/// fed into [`super::LlamaModel::apply_chat_template`] to convert a list of messages into an LLM
/// prompt. Internally the template is stored as a `CString` to avoid round-trip conversions
/// within the FFI.
#[derive(Eq, PartialEq, Clone, PartialOrd, Ord, Hash)]
pub struct LlamaChatTemplate(pub CString);

impl LlamaChatTemplate {
    /// Create a new template from a string. This can either be the name of a llama.cpp [chat template](https://github.com/ggerganov/llama.cpp/blob/8a8c4ceb6050bd9392609114ca56ae6d26f5b8f5/src/llama-chat.cpp#L27-L61)
    /// like "chatml" or "llama3" or an actual Jinja template for llama.cpp to interpret.
    ///
    /// # Errors
    /// Returns an error if the template string contains null bytes.
    pub fn new(template: &str) -> Result<Self, std::ffi::NulError> {
        Ok(Self(CString::new(template)?))
    }

    /// Accesses the template as a c string reference.
    #[must_use]
    pub fn as_c_str(&self) -> &CStr {
        &self.0
    }

    /// Attempts to convert the `CString` into a Rust str reference.
    ///
    /// # Errors
    /// Returns an error if the template is not valid UTF-8.
    pub fn to_str(&self) -> Result<&str, Utf8Error> {
        self.0.to_str()
    }

    /// Convenience method to create an owned String.
    ///
    /// # Errors
    /// Returns an error if the template is not valid UTF-8.
    pub fn to_string(&self) -> Result<String, Utf8Error> {
        self.to_str().map(str::to_string)
    }
}

impl std::fmt::Debug for LlamaChatTemplate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}
