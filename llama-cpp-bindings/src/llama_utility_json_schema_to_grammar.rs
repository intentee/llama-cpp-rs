use std::ffi::{CStr, CString};

use crate::error::{LlamaCppError, Result};
use crate::llama_utility_status_is_ok::status_is_ok;
use crate::llama_utility_status_to_i32::status_to_i32;

/// Convert a JSON schema string into a llama.cpp grammar string.
///
/// # Errors
/// Returns an error if the schema contains null bytes or the conversion fails.
pub fn json_schema_to_grammar(schema_json: &str) -> Result<String> {
    let schema_cstr = CString::new(schema_json)
        .map_err(|err| LlamaCppError::JsonSchemaToGrammarError(err.to_string()))?;
    let mut out = std::ptr::null_mut();
    let rc = unsafe {
        llama_cpp_bindings_sys::llama_rs_json_schema_to_grammar(
            schema_cstr.as_ptr(),
            false,
            &raw mut out,
        )
    };

    let result = {
        if !status_is_ok(rc) || out.is_null() {
            return Err(LlamaCppError::JsonSchemaToGrammarError(format!(
                "ffi error {}",
                status_to_i32(rc)
            )));
        }
        let grammar_bytes = unsafe { CStr::from_ptr(out) }.to_bytes().to_vec();
        let grammar = String::from_utf8(grammar_bytes)
            .map_err(|err| LlamaCppError::JsonSchemaToGrammarError(err.to_string()))?;

        Ok(grammar)
    };

    unsafe { llama_cpp_bindings_sys::llama_rs_string_free(out) };

    result
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
