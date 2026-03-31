use std::ffi::{CStr, CString};
use std::str::Utf8Error;

/// A performance-friendly wrapper around [`super::LlamaModel::chat_template`].
///
/// This is fed into [`super::LlamaModel::apply_chat_template`] to convert a list of messages into
/// an LLM prompt. Internally the template is stored as a `CString` to avoid round-trip conversions
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

#[cfg(test)]
mod tests {
    use super::LlamaChatTemplate;

    #[test]
    fn valid_template_creation() {
        let template = LlamaChatTemplate::new("chatml").unwrap();
        let template_str = template.to_str().unwrap();

        assert_eq!(template_str, "chatml");
    }

    #[test]
    fn null_byte_returns_error() {
        let template = LlamaChatTemplate::new("null\0byte");

        assert!(template.is_err());
    }

    #[test]
    fn debug_formatting() {
        let template = LlamaChatTemplate::new("chatml").unwrap();
        let debug_output = format!("{template:?}");

        assert!(debug_output.contains("chatml"));
    }

    #[test]
    fn to_string_returns_owned_string() {
        let template = LlamaChatTemplate::new("llama3").unwrap();
        let owned = template.to_string().unwrap();

        assert_eq!(owned, "llama3");
    }

    #[test]
    fn as_c_str_returns_valid_cstr() {
        let template = LlamaChatTemplate::new("test").unwrap();
        let cstr = template.as_c_str();

        assert_eq!(cstr.to_str().unwrap(), "test");
    }
}
