use std::ffi::{CStr, CString, c_char};
use std::ptr::{self, NonNull};
use std::slice;

use crate::model::grammar_trigger::{GrammarTrigger, GrammarTriggerType};
use crate::openai::ChatParseStateOaicompat;
use crate::token::LlamaToken;
use crate::{ApplyChatTemplateError, ChatParseError, status_is_ok, status_to_i32};

/// Result of applying a chat template with tool grammar support.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatTemplateResult {
    /// Rendered chat prompt.
    pub prompt: String,
    /// Optional grammar generated from tool definitions.
    pub grammar: Option<String>,
    /// Whether to use lazy grammar sampling.
    pub grammar_lazy: bool,
    /// Lazy grammar triggers derived from the template.
    pub grammar_triggers: Vec<GrammarTrigger>,
    /// Tokens that should be preserved for sampling.
    pub preserved_tokens: Vec<String>,
    /// Additional stop sequences added by the template.
    pub additional_stops: Vec<String>,
    /// Chat format used for parsing responses.
    pub chat_format: i32,
    /// Optional serialized PEG parser for tool-call parsing.
    pub parser: Option<String>,
    /// Whether the parser expects a forced-open thinking block.
    pub thinking_forced_open: bool,
    /// Whether tool calls should be parsed from the response.
    pub parse_tool_calls: bool,
}

pub fn new_empty_chat_template_raw_result() -> llama_cpp_bindings_sys::llama_rs_chat_template_result
{
    llama_cpp_bindings_sys::llama_rs_chat_template_result {
        prompt: ptr::null_mut(),
        grammar: ptr::null_mut(),
        parser: ptr::null_mut(),
        chat_format: 0,
        thinking_forced_open: false,
        grammar_lazy: false,
        grammar_triggers: ptr::null_mut(),
        grammar_triggers_count: 0,
        preserved_tokens: ptr::null_mut(),
        preserved_tokens_count: 0,
        additional_stops: ptr::null_mut(),
        additional_stops_count: 0,
    }
}

/// # Safety
///
/// `raw_cstr_array` must point to `count` valid, null-terminated C strings.
unsafe fn parse_raw_cstr_array(
    raw_cstr_array: *const *mut c_char,
    count: usize,
) -> Result<Vec<String>, ApplyChatTemplateError> {
    if count == 0 {
        return Ok(Vec::new());
    }

    if raw_cstr_array.is_null() {
        return Err(ApplyChatTemplateError::InvalidGrammarTriggerType);
    }

    let raw_entries = unsafe { slice::from_raw_parts(raw_cstr_array, count) };
    let mut parsed = Vec::with_capacity(raw_entries.len());

    for entry in raw_entries {
        if entry.is_null() {
            return Err(ApplyChatTemplateError::InvalidGrammarTriggerType);
        }
        let bytes = unsafe { CStr::from_ptr(*entry) }.to_bytes().to_vec();
        parsed.push(String::from_utf8(bytes)?);
    }

    Ok(parsed)
}

/// # Safety
///
/// `raw_triggers` must point to `count` valid `llama_rs_grammar_trigger` structs.
unsafe fn parse_raw_grammar_triggers(
    raw_triggers: *const llama_cpp_bindings_sys::llama_rs_grammar_trigger,
    count: usize,
) -> Result<Vec<GrammarTrigger>, ApplyChatTemplateError> {
    if count == 0 {
        return Ok(Vec::new());
    }

    if raw_triggers.is_null() {
        return Err(ApplyChatTemplateError::InvalidGrammarTriggerType);
    }

    let triggers = unsafe { slice::from_raw_parts(raw_triggers, count) };
    let mut parsed = Vec::with_capacity(triggers.len());

    for trigger in triggers {
        let trigger_type = match trigger.type_ {
            0 => GrammarTriggerType::Token,
            1 => GrammarTriggerType::Word,
            2 => GrammarTriggerType::Pattern,
            3 => GrammarTriggerType::PatternFull,
            _ => return Err(ApplyChatTemplateError::InvalidGrammarTriggerType),
        };
        let value = if trigger.value.is_null() {
            return Err(ApplyChatTemplateError::InvalidGrammarTriggerType);
        } else {
            let bytes = unsafe { CStr::from_ptr(trigger.value) }.to_bytes().to_vec();
            String::from_utf8(bytes)?
        };
        let token = if trigger_type == GrammarTriggerType::Token {
            Some(LlamaToken(trigger.token))
        } else {
            None
        };
        parsed.push(GrammarTrigger {
            trigger_type,
            value,
            token,
        });
    }

    Ok(parsed)
}

/// # Safety
///
/// `raw_result` must point to a valid, initialized `llama_rs_chat_template_result`.
pub unsafe fn parse_chat_template_raw_result(
    ffi_return_code: llama_cpp_bindings_sys::llama_rs_status,
    raw_result: *mut llama_cpp_bindings_sys::llama_rs_chat_template_result,
    parse_tool_calls: bool,
) -> Result<ChatTemplateResult, ApplyChatTemplateError> {
    let result = (|| {
        if !status_is_ok(ffi_return_code) {
            return Err(ApplyChatTemplateError::FfiError(status_to_i32(
                ffi_return_code,
            )));
        }

        let raw = unsafe { &*raw_result };

        if raw.prompt.is_null() {
            return Err(ApplyChatTemplateError::NullResult);
        }

        let prompt_bytes = unsafe { CStr::from_ptr(raw.prompt) }.to_bytes().to_vec();
        let prompt = String::from_utf8(prompt_bytes)?;

        let grammar = if raw.grammar.is_null() {
            None
        } else {
            let grammar_bytes = unsafe { CStr::from_ptr(raw.grammar) }.to_bytes().to_vec();
            Some(String::from_utf8(grammar_bytes)?)
        };

        let parser = if raw.parser.is_null() {
            None
        } else {
            let parser_bytes = unsafe { CStr::from_ptr(raw.parser) }.to_bytes().to_vec();
            Some(String::from_utf8(parser_bytes)?)
        };

        let grammar_triggers = unsafe {
            parse_raw_grammar_triggers(raw.grammar_triggers, raw.grammar_triggers_count)
        }?;

        let preserved_tokens =
            unsafe { parse_raw_cstr_array(raw.preserved_tokens, raw.preserved_tokens_count) }?;

        let additional_stops =
            unsafe { parse_raw_cstr_array(raw.additional_stops, raw.additional_stops_count) }?;

        Ok(ChatTemplateResult {
            prompt,
            grammar,
            grammar_lazy: raw.grammar_lazy,
            grammar_triggers,
            preserved_tokens,
            additional_stops,
            chat_format: raw.chat_format,
            parser,
            thinking_forced_open: raw.thinking_forced_open,
            parse_tool_calls,
        })
    })();

    unsafe { llama_cpp_bindings_sys::llama_rs_chat_template_result_free(raw_result) };

    result
}

impl ChatTemplateResult {
    /// Parse a generated response into an OpenAI-compatible message JSON string.
    ///
    /// # Errors
    /// Returns an error if the FFI call fails or the result is null.
    pub fn parse_response_oaicompat(
        &self,
        text: &str,
        is_partial: bool,
    ) -> Result<String, ChatParseError> {
        let text_cstr = CString::new(text)?;
        let parser_cstr = self.parser.as_deref().map(CString::new).transpose()?;
        let mut out_json: *mut c_char = ptr::null_mut();
        let rc = unsafe {
            llama_cpp_bindings_sys::llama_rs_chat_parse_to_oaicompat(
                text_cstr.as_ptr(),
                is_partial,
                self.chat_format,
                self.parse_tool_calls,
                parser_cstr
                    .as_ref()
                    .map_or(ptr::null(), |cstr| cstr.as_ptr()),
                self.thinking_forced_open,
                &raw mut out_json,
            )
        };

        let result = (|| {
            if !status_is_ok(rc) {
                return Err(ChatParseError::FfiError(status_to_i32(rc)));
            }
            if out_json.is_null() {
                return Err(ChatParseError::NullResult);
            }
            let bytes = unsafe { CStr::from_ptr(out_json) }.to_bytes().to_vec();
            Ok(String::from_utf8(bytes)?)
        })();

        unsafe { llama_cpp_bindings_sys::llama_rs_string_free(out_json) };

        result
    }

    /// Initialize a streaming parser for OpenAI-compatible chat deltas.
    ///
    /// # Errors
    /// Returns an error if the parser state cannot be initialized.
    pub fn streaming_state_oaicompat(&self) -> Result<ChatParseStateOaicompat, ChatParseError> {
        let parser_cstr = self.parser.as_deref().map(CString::new).transpose()?;
        let state = unsafe {
            llama_cpp_bindings_sys::llama_rs_chat_parse_state_init_oaicompat(
                self.chat_format,
                self.parse_tool_calls,
                parser_cstr
                    .as_ref()
                    .map_or(ptr::null(), |cstr| cstr.as_ptr()),
                self.thinking_forced_open,
            )
        };
        let state = NonNull::new(state).ok_or(ChatParseError::NullResult)?;

        Ok(ChatParseStateOaicompat { state })
    }
}
