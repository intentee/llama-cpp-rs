use std::ffi::{CStr, CString, c_char};

use crate::error::{LlamaCppError, Result};
use crate::ffi_status_is_ok::status_is_ok;

/// Convert a JSON schema string into a llama.cpp grammar string.
///
/// # Errors
/// Returns an error if the schema contains null bytes or the conversion fails.
pub fn json_schema_to_grammar(schema_json: &str) -> Result<String> {
    let schema_cstr = CString::new(schema_json)
        .map_err(|err| LlamaCppError::JsonSchemaToGrammarError(err.to_string()))?;
    let mut out: *mut c_char = std::ptr::null_mut();
    let mut error_ptr: *mut c_char = std::ptr::null_mut();

    let status = unsafe {
        llama_cpp_bindings_sys::llama_rs_json_schema_to_grammar(
            schema_cstr.as_ptr(),
            false,
            &raw mut out,
            &raw mut error_ptr,
        )
    };

    if !status_is_ok(status) || out.is_null() {
        let message = if error_ptr.is_null() {
            "unknown error".to_owned()
        } else {
            let message = unsafe { CStr::from_ptr(error_ptr) }
                .to_string_lossy()
                .into_owned();

            unsafe { llama_cpp_bindings_sys::llama_rs_string_free(error_ptr) };

            message
        };

        return Err(LlamaCppError::JsonSchemaToGrammarError(message));
    }

    let grammar_bytes = unsafe { CStr::from_ptr(out) }.to_bytes().to_vec();

    unsafe { llama_cpp_bindings_sys::llama_rs_string_free(out) };

    String::from_utf8(grammar_bytes)
        .map_err(|err| LlamaCppError::JsonSchemaToGrammarError(err.to_string()))
}

#[cfg(test)]
mod tests {
    use super::json_schema_to_grammar;

    #[test]
    fn simple_object() {
        let schema = r#"{"type": "object", "properties": {"name": {"type": "string"}}}"#;
        let grammar = json_schema_to_grammar(schema).unwrap();

        assert!(!grammar.is_empty());
    }

    #[test]
    fn null_byte_returns_error() {
        let schema = "{\x00}";
        let result = json_schema_to_grammar(schema);

        assert!(result.is_err());
    }

    #[test]
    fn simple_string() {
        let schema = r#"{"type": "string"}"#;
        let grammar = json_schema_to_grammar(schema).unwrap();

        assert!(!grammar.is_empty());
    }

    #[test]
    fn invalid_json_returns_ffi_error() {
        let schema = "not valid json at all";
        let result = json_schema_to_grammar(schema);

        assert!(result.is_err());
    }
}
