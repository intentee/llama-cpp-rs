use std::ffi::{CStr, CString, c_char};
use std::mem;
use std::ptr::{self, NonNull};
use std::slice;

use crate::{ChatParseError, status_is_ok, status_to_i32};

const fn check_ffi_status(
    status: llama_cpp_bindings_sys::llama_rs_status,
) -> Result<(), ChatParseError> {
    if status_is_ok(status) {
        Ok(())
    } else {
        Err(ChatParseError::FfiError(status_to_i32(status)))
    }
}

const fn check_not_null_with_count(
    pointer: *const llama_cpp_bindings_sys::llama_rs_chat_msg_diff_oaicompat,
    count: usize,
) -> Result<(), ChatParseError> {
    if count > 0 && pointer.is_null() {
        Err(ChatParseError::NullResult)
    } else {
        Ok(())
    }
}

/// # Safety
///
/// `diffs_ptr` must point to at least `count` valid `llama_rs_chat_msg_diff_oaicompat`
/// values that remain valid for the lifetime `'diffs`.
const unsafe fn diffs_as_slice<'diffs>(
    diffs_ptr: *const llama_cpp_bindings_sys::llama_rs_chat_msg_diff_oaicompat,
    count: usize,
) -> &'diffs [llama_cpp_bindings_sys::llama_rs_chat_msg_diff_oaicompat] {
    if count == 0 {
        &[]
    } else {
        unsafe { slice::from_raw_parts(diffs_ptr, count) }
    }
}

const fn check_json_not_null(json_ptr: *const c_char) -> Result<(), ChatParseError> {
    if json_ptr.is_null() {
        Err(ChatParseError::NullResult)
    } else {
        Ok(())
    }
}

fn handle_diff_json_error(
    status: llama_cpp_bindings_sys::llama_rs_status,
    json_ptr: *mut c_char,
) -> Result<(), ChatParseError> {
    if !status_is_ok(status) {
        if !json_ptr.is_null() {
            unsafe { llama_cpp_bindings_sys::llama_rs_string_free(json_ptr) };
        }

        return Err(ChatParseError::FfiError(status_to_i32(status)));
    }

    Ok(())
}

/// Streaming OpenAI-compatible parser state.
#[derive(Debug)]
pub struct ChatParseStateOaicompat {
    /// Raw pointer to the underlying FFI parser state.
    pub state: NonNull<llama_cpp_bindings_sys::llama_rs_chat_parse_state_oaicompat>,
}

impl ChatParseStateOaicompat {
    /// Update the parser with additional text and return OpenAI-compatible deltas as JSON strings.
    ///
    /// # Errors
    /// Returns an error if the FFI call fails or the result is null.
    pub fn update(
        &mut self,
        text_added: &str,
        is_partial: bool,
    ) -> Result<Vec<String>, ChatParseError> {
        let text_cstr = CString::new(text_added)?;
        let mut out_msg: llama_cpp_bindings_sys::llama_rs_chat_msg_oaicompat =
            unsafe { mem::zeroed() };
        let mut out_diffs: *mut llama_cpp_bindings_sys::llama_rs_chat_msg_diff_oaicompat =
            ptr::null_mut();
        let mut out_diffs_count: usize = 0;
        let rc = unsafe {
            llama_cpp_bindings_sys::llama_rs_chat_parse_state_update_oaicompat(
                self.state.as_ptr(),
                text_cstr.as_ptr(),
                is_partial,
                &raw mut out_msg,
                &raw mut out_diffs,
                &raw mut out_diffs_count,
            )
        };

        let result = {
            check_ffi_status(rc)?;
            check_not_null_with_count(out_diffs, out_diffs_count)?;

            let diffs = unsafe { diffs_as_slice(out_diffs, out_diffs_count) };
            let mut deltas = Vec::with_capacity(diffs.len());

            for diff in diffs {
                let mut out_json: *mut c_char = ptr::null_mut();
                let rc = unsafe {
                    llama_cpp_bindings_sys::llama_rs_chat_msg_diff_to_oaicompat_json(
                        diff,
                        &raw mut out_json,
                    )
                };
                handle_diff_json_error(rc, out_json)?;
                check_json_not_null(out_json)?;

                let bytes = unsafe { CStr::from_ptr(out_json) }.to_bytes().to_vec();
                unsafe { llama_cpp_bindings_sys::llama_rs_string_free(out_json) };
                deltas.push(String::from_utf8(bytes)?);
            }

            Ok(deltas)
        };

        unsafe { llama_cpp_bindings_sys::llama_rs_chat_msg_free_oaicompat(&raw mut out_msg) };
        unsafe {
            llama_cpp_bindings_sys::llama_rs_chat_msg_diff_free_oaicompat(
                out_diffs,
                out_diffs_count,
            );
        };

        result
    }
}

impl Drop for ChatParseStateOaicompat {
    fn drop(&mut self) {
        unsafe {
            llama_cpp_bindings_sys::llama_rs_chat_parse_state_free_oaicompat(self.state.as_ptr());
        };
    }
}

#[cfg(test)]
mod tests {
    use crate::model::chat_template_result::ChatTemplateResult;

    fn content_only_template() -> ChatTemplateResult {
        ChatTemplateResult::default()
    }

    #[test]
    fn update_with_simple_text() {
        let mut state = content_only_template().streaming_state_oaicompat().unwrap();
        let deltas = state.update("Hello", true);
        assert!(deltas.is_ok());
    }

    #[test]
    fn update_null_byte_returns_error() {
        let mut state = content_only_template().streaming_state_oaicompat().unwrap();
        let result = state.update("hello\0world", true);
        assert!(result.unwrap_err().to_string().contains("nul byte"));
    }

    #[test]
    fn update_finalized_produces_deltas() {
        let mut state = content_only_template().streaming_state_oaicompat().unwrap();
        let deltas = state.update("Hello world", false).unwrap();

        assert!(!deltas.is_empty());
    }

    #[test]
    fn check_ffi_status_returns_error_for_invalid() {
        let result =
            super::check_ffi_status(llama_cpp_bindings_sys::LLAMA_RS_STATUS_INVALID_ARGUMENT);

        assert!(result.unwrap_err().to_string().contains("ffi error"));
    }

    #[test]
    fn check_not_null_with_count_returns_error() {
        let result = super::check_not_null_with_count(std::ptr::null(), 1);

        assert!(result.unwrap_err().to_string().contains("null result"));
    }

    #[test]
    fn check_not_null_with_count_zero_is_ok() {
        let result = super::check_not_null_with_count(std::ptr::null(), 0);

        assert!(result.is_ok());
    }

    #[test]
    fn check_json_not_null_returns_error() {
        let result = super::check_json_not_null(std::ptr::null());

        assert!(result.unwrap_err().to_string().contains("null result"));
    }

    #[test]
    fn handle_diff_json_error_frees_and_returns_error() {
        let result = super::handle_diff_json_error(
            llama_cpp_bindings_sys::LLAMA_RS_STATUS_EXCEPTION,
            std::ptr::null_mut(),
        );

        assert!(result.unwrap_err().to_string().contains("ffi error"));
    }

    #[test]
    fn handle_diff_json_error_frees_non_null_pointer_on_error() {
        let leaked_string = std::ffi::CString::new("test").unwrap().into_raw();
        let result = super::handle_diff_json_error(
            llama_cpp_bindings_sys::LLAMA_RS_STATUS_EXCEPTION,
            leaked_string,
        );

        assert!(result.unwrap_err().to_string().contains("ffi error"));
    }

    #[test]
    fn diffs_as_slice_returns_empty_for_zero_count() {
        let result = unsafe { super::diffs_as_slice(std::ptr::null(), 0) };

        assert!(result.is_empty());
    }
}
