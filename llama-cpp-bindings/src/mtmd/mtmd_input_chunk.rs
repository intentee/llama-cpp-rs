use std::ffi::CStr;
use std::ptr::NonNull;
use std::slice;

use crate::token::LlamaToken;

use super::mtmd_error::MtmdInputChunkError;
use super::mtmd_input_chunk_type::{MtmdInputChunkType, MtmdInputChunkTypeError};

/// # Safety
///
/// `tokens_ptr` must point to at least `n_tokens` valid `llama_token` values
/// that remain valid for the lifetime `'chunk`.
unsafe fn tokens_from_raw_ptr<'chunk>(
    tokens_ptr: *const llama_cpp_bindings_sys::llama_token,
    n_tokens: usize,
) -> Option<&'chunk [LlamaToken]> {
    if tokens_ptr.is_null() || n_tokens == 0 {
        None
    } else {
        unsafe {
            Some(slice::from_raw_parts(
                tokens_ptr.cast::<LlamaToken>(),
                n_tokens,
            ))
        }
    }
}

/// Safe wrapper around `mtmd_input_chunk`.
///
/// Represents a single chunk of input data, which can be either text tokens,
/// image tokens, or audio tokens. The chunk type determines what kind of
/// data and operations are available.
#[derive(Debug)]
pub struct MtmdInputChunk {
    /// Raw pointer to the underlying `mtmd_input_chunk`.
    pub chunk: NonNull<llama_cpp_bindings_sys::mtmd_input_chunk>,
    pub owned: bool,
}

impl MtmdInputChunk {
    /// Get the type of this chunk
    ///
    /// # Errors
    /// Returns an error if the chunk type is unknown.
    pub fn chunk_type(&self) -> Result<MtmdInputChunkType, MtmdInputChunkTypeError> {
        let chunk_type =
            unsafe { llama_cpp_bindings_sys::mtmd_input_chunk_get_type(self.chunk.as_ptr()) };
        MtmdInputChunkType::try_from(chunk_type)
    }

    /// Get text tokens from this chunk.
    ///
    /// Only valid for text chunks. Returns `None` for image or audio chunks.
    #[must_use]
    pub fn text_tokens(&self) -> Option<&[LlamaToken]> {
        if self.chunk_type() != Ok(MtmdInputChunkType::Text) {
            return None;
        }

        let mut n_tokens = 0usize;
        let tokens_ptr = unsafe {
            llama_cpp_bindings_sys::mtmd_input_chunk_get_tokens_text(
                self.chunk.as_ptr(),
                &raw mut n_tokens,
            )
        };

        unsafe { tokens_from_raw_ptr(tokens_ptr, n_tokens) }
    }

    /// Get the number of tokens in this chunk
    #[must_use]
    pub fn n_tokens(&self) -> usize {
        unsafe { llama_cpp_bindings_sys::mtmd_input_chunk_get_n_tokens(self.chunk.as_ptr()) }
    }

    /// Get the number of positions in this chunk.
    #[must_use]
    pub fn n_positions(&self) -> i32 {
        unsafe { llama_cpp_bindings_sys::mtmd_input_chunk_get_n_pos(self.chunk.as_ptr()) }
    }

    /// Get chunk ID if available.
    ///
    /// Returns `None` for text chunks, may return an ID for image/audio chunks.
    #[must_use]
    pub fn id(&self) -> Option<String> {
        let ptr = unsafe { llama_cpp_bindings_sys::mtmd_input_chunk_get_id(self.chunk.as_ptr()) };
        if ptr.is_null() {
            None
        } else {
            unsafe { CStr::from_ptr(ptr) }
                .to_string_lossy()
                .into_owned()
                .into()
        }
    }

    /// Create a copy of this chunk that you own.
    ///
    /// # Errors
    ///
    /// Returns `MtmdInputChunkError::NullResult` if copying fails.
    pub fn copy(&self) -> Result<Self, MtmdInputChunkError> {
        let chunk = unsafe { llama_cpp_bindings_sys::mtmd_input_chunk_copy(self.chunk.as_ptr()) };
        let chunk = NonNull::new(chunk).ok_or(MtmdInputChunkError::NullResult)?;

        Ok(Self { chunk, owned: true })
    }
}

impl Drop for MtmdInputChunk {
    fn drop(&mut self) {
        if self.owned {
            unsafe { llama_cpp_bindings_sys::mtmd_input_chunk_free(self.chunk.as_ptr()) }
        }
    }
}

#[cfg(test)]
mod unit_tests {
    use super::tokens_from_raw_ptr;

    #[test]
    fn tokens_from_raw_ptr_returns_none_for_null() {
        assert!(unsafe { tokens_from_raw_ptr(std::ptr::null(), 5) }.is_none());
    }

    #[test]
    fn tokens_from_raw_ptr_returns_none_for_zero_count() {
        let token: llama_cpp_bindings_sys::llama_token = 42;
        assert!(unsafe { tokens_from_raw_ptr(&token, 0) }.is_none());
    }

    #[test]
    fn tokens_from_raw_ptr_returns_some_for_valid() {
        let tokens: [llama_cpp_bindings_sys::llama_token; 2] = [1, 2];
        let result = unsafe { tokens_from_raw_ptr(tokens.as_ptr(), 2) };

        assert!(result.is_some());
        assert_eq!(result.unwrap().len(), 2);
    }
}

#[cfg(test)]
#[cfg(feature = "tests_that_use_llms")]
mod tests {
    use serial_test::serial;

    use crate::mtmd::mtmd_bitmap::MtmdBitmap;
    use crate::mtmd::mtmd_input_chunk_type::MtmdInputChunkType;
    use crate::mtmd::mtmd_input_text::MtmdInputText;
    use crate::test_model;

    #[test]
    #[serial]
    fn text_chunk_has_text_type() {
        let (_backend, _model, mtmd_ctx) = test_model::load_default_mtmd().unwrap();
        let image_data = vec![128u8; 64 * 64 * 3];
        let bitmap = MtmdBitmap::from_image_data(64, 64, &image_data).unwrap();
        let input_text = MtmdInputText {
            text: "Hello world <__media__>".to_string(),
            add_special: true,
            parse_special: true,
        };
        let chunks = mtmd_ctx.tokenize(input_text, &[&bitmap]).unwrap();
        let first_chunk = chunks.get(0).unwrap();
        assert_eq!(first_chunk.chunk_type().unwrap(), MtmdInputChunkType::Text);
    }

    #[test]
    #[serial]
    fn text_chunk_returns_text_tokens() {
        let (_backend, _model, mtmd_ctx) = test_model::load_default_mtmd().unwrap();
        let image_data = vec![128u8; 64 * 64 * 3];
        let bitmap = MtmdBitmap::from_image_data(64, 64, &image_data).unwrap();
        let input_text = MtmdInputText {
            text: "Hello world <__media__>".to_string(),
            add_special: true,
            parse_special: true,
        };
        let chunks = mtmd_ctx.tokenize(input_text, &[&bitmap]).unwrap();
        let first_chunk = chunks.get(0).unwrap();
        let tokens = first_chunk.text_tokens();
        assert!(tokens.is_some());
        assert!(!tokens.unwrap().is_empty());
    }

    #[test]
    #[serial]
    fn chunk_n_tokens_is_positive() {
        let (_backend, _model, mtmd_ctx) = test_model::load_default_mtmd().unwrap();
        let image_data = vec![128u8; 64 * 64 * 3];
        let bitmap = MtmdBitmap::from_image_data(64, 64, &image_data).unwrap();
        let input_text = MtmdInputText {
            text: "Hello world <__media__>".to_string(),
            add_special: true,
            parse_special: true,
        };
        let chunks = mtmd_ctx.tokenize(input_text, &[&bitmap]).unwrap();
        let first_chunk = chunks.get(0).unwrap();
        assert!(first_chunk.n_tokens() > 0);
    }

    #[test]
    #[serial]
    fn chunk_n_positions_is_positive() {
        let (_backend, _model, mtmd_ctx) = test_model::load_default_mtmd().unwrap();
        let image_data = vec![128u8; 64 * 64 * 3];
        let bitmap = MtmdBitmap::from_image_data(64, 64, &image_data).unwrap();
        let input_text = MtmdInputText {
            text: "Hello world <__media__>".to_string(),
            add_special: true,
            parse_special: true,
        };
        let chunks = mtmd_ctx.tokenize(input_text, &[&bitmap]).unwrap();
        let first_chunk = chunks.get(0).unwrap();
        assert!(first_chunk.n_positions() > 0);
    }

    #[test]
    #[serial]
    fn copy_creates_owned_duplicate() {
        let (_backend, _model, mtmd_ctx) = test_model::load_default_mtmd().unwrap();
        let image_data = vec![128u8; 64 * 64 * 3];
        let bitmap = MtmdBitmap::from_image_data(64, 64, &image_data).unwrap();
        let input_text = MtmdInputText {
            text: "Hello <__media__>".to_string(),
            add_special: true,
            parse_special: true,
        };
        let chunks = mtmd_ctx.tokenize(input_text, &[&bitmap]).unwrap();
        let first_chunk = chunks.get(0).unwrap();
        let copied = first_chunk.copy().unwrap();

        assert!(copied.owned);
        assert_eq!(copied.n_tokens(), first_chunk.n_tokens());
    }

    #[test]
    #[serial]
    fn text_chunk_id_returns_none() {
        let (_backend, _model, mtmd_ctx) = test_model::load_default_mtmd().unwrap();
        let image_data = vec![128u8; 64 * 64 * 3];
        let bitmap = MtmdBitmap::from_image_data(64, 64, &image_data).unwrap();
        let input_text = MtmdInputText {
            text: "Hello <__media__>".to_string(),
            add_special: true,
            parse_special: true,
        };
        let chunks = mtmd_ctx.tokenize(input_text, &[&bitmap]).unwrap();
        let first_chunk = chunks.get(0).unwrap();

        assert_eq!(first_chunk.chunk_type().unwrap(), MtmdInputChunkType::Text);
        assert!(first_chunk.id().is_none());
    }

    #[test]
    #[serial]
    fn image_chunk_returns_none_for_text_tokens() {
        let (_backend, _model, mtmd_ctx) = test_model::load_default_mtmd().unwrap();
        let image_data = vec![128u8; 64 * 64 * 3];
        let bitmap = MtmdBitmap::from_image_data(64, 64, &image_data).unwrap();
        let input_text = MtmdInputText {
            text: "Hello <__media__>".to_string(),
            add_special: true,
            parse_special: true,
        };
        let chunks = mtmd_ctx.tokenize(input_text, &[&bitmap]).unwrap();

        for chunk_index in 0..chunks.len() {
            let chunk = chunks.get(chunk_index).unwrap();
            if chunk.chunk_type() == Ok(MtmdInputChunkType::Image) {
                assert!(chunk.text_tokens().is_none());

                return;
            }
        }
    }

    #[test]
    #[serial]
    fn image_chunk_id_returns_some() {
        let (_backend, _model, mtmd_ctx) = test_model::load_default_mtmd().unwrap();
        let image_data = vec![128u8; 64 * 64 * 3];
        let bitmap = MtmdBitmap::from_image_data(64, 64, &image_data).unwrap();
        let input_text = MtmdInputText {
            text: "Hello <__media__>".to_string(),
            add_special: true,
            parse_special: true,
        };
        let chunks = mtmd_ctx.tokenize(input_text, &[&bitmap]).unwrap();

        for chunk_index in 0..chunks.len() {
            let chunk = chunks.get(chunk_index).unwrap();
            if chunk.chunk_type() == Ok(MtmdInputChunkType::Image) {
                assert!(chunk.id().is_some());

                return;
            }
        }
    }
}
