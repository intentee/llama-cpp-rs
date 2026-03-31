//! Safe wrapper around `llama_context`.

use std::fmt::{Debug, Formatter};
use std::num::NonZeroI32;
use std::ptr::NonNull;
use std::slice;

use crate::llama_batch::LlamaBatch;
use crate::model::{LlamaLoraAdapter, LlamaModel};
use crate::timing::LlamaTimings;
use crate::token::LlamaToken;
use crate::token::data::LlamaTokenData;
use crate::token::data_array::LlamaTokenDataArray;
use crate::{
    DecodeError, EmbeddingsError, EncodeError, LlamaLoraAdapterRemoveError,
    LlamaLoraAdapterSetError,
};

const fn check_lora_set_result(err_code: i32) -> Result<(), LlamaLoraAdapterSetError> {
    if err_code != 0 {
        return Err(LlamaLoraAdapterSetError::ErrorResult(err_code));
    }

    Ok(())
}

const fn check_lora_remove_result(err_code: i32) -> Result<(), LlamaLoraAdapterRemoveError> {
    if err_code != 0 {
        return Err(LlamaLoraAdapterRemoveError::ErrorResult(err_code));
    }

    Ok(())
}

pub mod kv_cache;
pub mod params;
pub mod session;

/// Safe wrapper around `llama_context`.
pub struct LlamaContext<'model> {
    /// Raw pointer to the underlying `llama_context`.
    pub context: NonNull<llama_cpp_bindings_sys::llama_context>,
    /// A reference to the context's model.
    pub model: &'model LlamaModel,
    initialized_logits: Vec<i32>,
    embeddings_enabled: bool,
}

impl Debug for LlamaContext<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LlamaContext")
            .field("context", &self.context)
            .finish()
    }
}

impl<'model> LlamaContext<'model> {
    /// Wraps existing raw pointers into a new `LlamaContext`.
    #[must_use]
    pub const fn new(
        llama_model: &'model LlamaModel,
        llama_context: NonNull<llama_cpp_bindings_sys::llama_context>,
        embeddings_enabled: bool,
    ) -> Self {
        Self {
            context: llama_context,
            model: llama_model,
            initialized_logits: Vec::new(),
            embeddings_enabled,
        }
    }

    /// Gets the max number of logical tokens that can be submitted to decode. Must be greater than or equal to [`Self::n_ubatch`].
    #[must_use]
    pub fn n_batch(&self) -> u32 {
        unsafe { llama_cpp_bindings_sys::llama_n_batch(self.context.as_ptr()) }
    }

    /// Gets the max number of physical tokens (hardware level) to decode in batch. Must be less than or equal to [`Self::n_batch`].
    #[must_use]
    pub fn n_ubatch(&self) -> u32 {
        unsafe { llama_cpp_bindings_sys::llama_n_ubatch(self.context.as_ptr()) }
    }

    /// Gets the size of the context.
    #[must_use]
    pub fn n_ctx(&self) -> u32 {
        unsafe { llama_cpp_bindings_sys::llama_n_ctx(self.context.as_ptr()) }
    }

    /// Decodes the batch.
    ///
    /// # Errors
    ///
    /// - `DecodeError` if the decoding failed.
    ///
    /// # Panics
    ///
    /// - the returned [`std::ffi::c_int`] from llama-cpp does not fit into a i32 (this should never happen on most systems)
    pub fn decode(&mut self, batch: &mut LlamaBatch) -> Result<(), DecodeError> {
        let result = unsafe {
            llama_cpp_bindings_sys::llama_decode(self.context.as_ptr(), batch.llama_batch)
        };

        match NonZeroI32::new(result) {
            None => {
                self.initialized_logits
                    .clone_from(&batch.initialized_logits);
                Ok(())
            }
            Some(error) => Err(DecodeError::from(error)),
        }
    }

    /// Encodes the batch.
    ///
    /// # Errors
    ///
    /// - `EncodeError` if the decoding failed.
    ///
    /// # Panics
    ///
    /// - the returned [`std::ffi::c_int`] from llama-cpp does not fit into a i32 (this should never happen on most systems)
    pub fn encode(&mut self, batch: &mut LlamaBatch) -> Result<(), EncodeError> {
        let status = unsafe {
            llama_cpp_bindings_sys::llama_rs_encode(self.context.as_ptr(), batch.llama_batch)
        };

        self.handle_encode_result(status, batch)
    }

    fn handle_encode_result(
        &mut self,
        status: llama_cpp_bindings_sys::llama_rs_status,
        batch: &mut LlamaBatch,
    ) -> Result<(), EncodeError> {
        if crate::status_is_ok(status) {
            self.initialized_logits
                .clone_from(&batch.initialized_logits);

            Ok(())
        } else {
            Err(EncodeError::from(
                NonZeroI32::new(crate::status_to_i32(status))
                    .unwrap_or(NonZeroI32::new(1).expect("1 is non-zero")),
            ))
        }
    }

    /// Get the embeddings for the given sequence in the current context.
    ///
    /// # Returns
    ///
    /// A slice containing the embeddings for the last decoded batch.
    /// The size corresponds to the `n_embd` parameter of the context's model.
    ///
    /// # Errors
    ///
    /// - When the current context was constructed without enabling embeddings.
    /// - If the current model had a pooling type of [`llama_cpp_bindings_sys::LLAMA_POOLING_TYPE_NONE`]
    /// - If the given sequence index exceeds the max sequence id.
    ///
    /// # Panics
    ///
    /// * `n_embd` does not fit into a usize
    pub fn embeddings_seq_ith(&self, sequence_index: i32) -> Result<&[f32], EmbeddingsError> {
        if !self.embeddings_enabled {
            return Err(EmbeddingsError::NotEnabled);
        }

        let n_embd =
            usize::try_from(self.model.n_embd()).expect("n_embd does not fit into a usize");

        unsafe {
            let embedding = llama_cpp_bindings_sys::llama_get_embeddings_seq(
                self.context.as_ptr(),
                sequence_index,
            );

            if embedding.is_null() {
                Err(EmbeddingsError::NonePoolType)
            } else {
                Ok(slice::from_raw_parts(embedding, n_embd))
            }
        }
    }

    /// Get the embeddings for the given token in the current context.
    ///
    /// # Returns
    ///
    /// A slice containing the embeddings for the last decoded batch of the given token.
    /// The size corresponds to the `n_embd` parameter of the context's model.
    ///
    /// # Errors
    ///
    /// - When the current context was constructed without enabling embeddings.
    /// - When the given token didn't have logits enabled when it was passed.
    /// - If the given token index exceeds the max token id.
    ///
    /// # Panics
    ///
    /// * `n_embd` does not fit into a usize
    pub fn embeddings_ith(&self, token_index: i32) -> Result<&[f32], EmbeddingsError> {
        if !self.embeddings_enabled {
            return Err(EmbeddingsError::NotEnabled);
        }

        let n_embd =
            usize::try_from(self.model.n_embd()).expect("n_embd does not fit into a usize");

        unsafe {
            let embedding = llama_cpp_bindings_sys::llama_get_embeddings_ith(
                self.context.as_ptr(),
                token_index,
            );

            if embedding.is_null() {
                Err(EmbeddingsError::LogitsNotEnabled)
            } else {
                Ok(slice::from_raw_parts(embedding, n_embd))
            }
        }
    }

    /// Get the logits for the last token in the context.
    ///
    /// # Returns
    /// An iterator over unsorted `LlamaTokenData` containing the
    /// logits for the last token in the context.
    ///
    pub fn candidates(&self) -> impl Iterator<Item = LlamaTokenData> + '_ {
        (0_i32..).zip(self.get_logits()).map(|(token_id, logit)| {
            let token = LlamaToken::new(token_id);
            LlamaTokenData::new(token, *logit, 0_f32)
        })
    }

    /// Get the token data array for the last token in the context.
    ///
    /// This is a convenience method that implements:
    /// ```ignore
    /// LlamaTokenDataArray::from_iter(ctx.candidates(), false)
    /// ```
    ///
    #[must_use]
    pub fn token_data_array(&self) -> LlamaTokenDataArray {
        LlamaTokenDataArray::from_iter(self.candidates(), false)
    }

    /// Token logits obtained from the last call to `decode()`.
    /// The logits for which `batch.logits[i] != 0` are stored contiguously
    /// in the order they have appeared in the batch.
    /// Rows: number of tokens for which `batch.logits[i] != 0`
    /// Cols: `n_vocab`
    ///
    /// # Returns
    ///
    /// A slice containing the logits for the last decoded token.
    /// The size corresponds to the `n_vocab` parameter of the context's model.
    ///
    /// # Panics
    ///
    /// Panics if the logits data pointer is null or `n_vocab` does not fit into a `usize`.
    #[must_use]
    pub fn get_logits(&self) -> &[f32] {
        let data = unsafe { llama_cpp_bindings_sys::llama_get_logits(self.context.as_ptr()) };
        assert!(!data.is_null(), "logits data for last token is null");
        let len = usize::try_from(self.model.n_vocab()).expect("n_vocab does not fit into a usize");

        unsafe { slice::from_raw_parts(data, len) }
    }

    /// Get the logits for the ith token in the context.
    ///
    /// # Panics
    ///
    /// - logit `i` is not initialized.
    pub fn candidates_ith(&self, token_index: i32) -> impl Iterator<Item = LlamaTokenData> + '_ {
        (0_i32..)
            .zip(self.get_logits_ith(token_index))
            .map(|(token_id, logit)| {
                let token = LlamaToken::new(token_id);
                LlamaTokenData::new(token, *logit, 0_f32)
            })
    }

    /// Get the token data array for the ith token in the context.
    ///
    /// This is a convenience method that implements:
    /// ```ignore
    /// LlamaTokenDataArray::from_iter(ctx.candidates_ith(token_index), false)
    /// ```
    ///
    /// # Panics
    ///
    /// - logit `i` is not initialized.
    #[must_use]
    pub fn token_data_array_ith(&self, token_index: i32) -> LlamaTokenDataArray {
        LlamaTokenDataArray::from_iter(self.candidates_ith(token_index), false)
    }

    /// Get the logits for the ith token in the context.
    ///
    /// # Panics
    ///
    /// - `token_index` is greater than `n_ctx`
    /// - `n_vocab` does not fit into a usize
    /// - logit `token_index` is not initialized.
    #[must_use]
    pub fn get_logits_ith(&self, token_index: i32) -> &[f32] {
        assert!(self.initialized_logits.contains(&token_index));
        assert!(
            self.n_ctx() > u32::try_from(token_index).expect("token_index does not fit into a u32")
        );

        let data = unsafe {
            llama_cpp_bindings_sys::llama_get_logits_ith(self.context.as_ptr(), token_index)
        };
        let len = usize::try_from(self.model.n_vocab()).expect("n_vocab does not fit into a usize");

        unsafe { slice::from_raw_parts(data, len) }
    }

    /// Reset the timings for the context.
    pub fn reset_timings(&mut self) {
        unsafe { llama_cpp_bindings_sys::llama_perf_context_reset(self.context.as_ptr()) }
    }

    /// Returns the timings for the context.
    pub fn timings(&mut self) -> LlamaTimings {
        let timings = unsafe { llama_cpp_bindings_sys::llama_perf_context(self.context.as_ptr()) };
        LlamaTimings { timings }
    }

    /// Sets a lora adapter.
    ///
    /// # Errors
    ///
    /// See [`LlamaLoraAdapterSetError`] for more information.
    pub fn lora_adapter_set(
        &self,
        adapter: &mut LlamaLoraAdapter,
        scale: f32,
    ) -> Result<(), LlamaLoraAdapterSetError> {
        let mut adapters = [adapter.lora_adapter.as_ptr()];
        let mut scales = [scale];
        let err_code = unsafe {
            llama_cpp_bindings_sys::llama_set_adapters_lora(
                self.context.as_ptr(),
                adapters.as_mut_ptr(),
                1,
                scales.as_mut_ptr(),
            )
        };
        check_lora_set_result(err_code)?;

        tracing::debug!("Set lora adapter");
        Ok(())
    }

    /// Remove all lora adapters.
    ///
    /// Note: The upstream API now replaces all adapters at once via
    /// `llama_set_adapters_lora`. This clears all adapters from the context.
    ///
    /// # Errors
    ///
    /// See [`LlamaLoraAdapterRemoveError`] for more information.
    pub fn lora_adapter_remove(
        &self,
        _adapter: &mut LlamaLoraAdapter,
    ) -> Result<(), LlamaLoraAdapterRemoveError> {
        let err_code = unsafe {
            llama_cpp_bindings_sys::llama_set_adapters_lora(
                self.context.as_ptr(),
                std::ptr::null_mut(),
                0,
                std::ptr::null_mut(),
            )
        };
        check_lora_remove_result(err_code)?;

        tracing::debug!("Remove lora adapter");
        Ok(())
    }
}

impl Drop for LlamaContext<'_> {
    fn drop(&mut self) {
        unsafe { llama_cpp_bindings_sys::llama_free(self.context.as_ptr()) }
    }
}

#[cfg(test)]
mod unit_tests {
    use crate::LlamaLoraAdapterRemoveError;
    use crate::LlamaLoraAdapterSetError;

    use super::{check_lora_remove_result, check_lora_set_result};

    #[test]
    fn check_lora_set_result_ok_for_zero() {
        assert!(check_lora_set_result(0).is_ok());
    }

    #[test]
    fn check_lora_set_result_error_for_nonzero() {
        let result = check_lora_set_result(-1);

        assert_eq!(result, Err(LlamaLoraAdapterSetError::ErrorResult(-1)));
    }

    #[test]
    fn check_lora_remove_result_ok_for_zero() {
        assert!(check_lora_remove_result(0).is_ok());
    }

    #[test]
    fn check_lora_remove_result_error_for_nonzero() {
        let result = check_lora_remove_result(-1);

        assert_eq!(result, Err(LlamaLoraAdapterRemoveError::ErrorResult(-1)));
    }
}

#[cfg(test)]
#[cfg(feature = "tests_that_use_llms")]
mod tests {
    use serial_test::serial;

    use crate::context::params::LlamaContextParams;
    use crate::llama_batch::LlamaBatch;
    use crate::model::AddBos;
    use crate::test_model;

    #[test]
    #[serial]
    fn context_creation_and_properties() {
        let (backend, model) = test_model::load_default_model().unwrap();
        let ctx_params = LlamaContextParams::default().with_n_ctx(std::num::NonZeroU32::new(512));
        let context = model.new_context(&backend, ctx_params).unwrap();
        assert!(context.n_ctx() > 0);
        assert!(context.n_batch() > 0);
        assert!(context.n_ubatch() > 0);
    }

    #[test]
    #[serial]
    fn decode_and_get_logits() {
        let (backend, model) = test_model::load_default_model().unwrap();
        let ctx_params = LlamaContextParams::default().with_n_ctx(std::num::NonZeroU32::new(512));
        let mut context = model.new_context(&backend, ctx_params).unwrap();
        let tokens = model.str_to_token("hello", AddBos::Always).unwrap();
        let mut batch = LlamaBatch::new(512, 1).unwrap();
        batch.add_sequence(&tokens, 0, false).unwrap();

        let decode_result = context.decode(&mut batch);
        assert!(decode_result.is_ok());

        let logits = context.get_logits();
        assert!(!logits.is_empty());
    }

    #[test]
    #[serial]
    fn timings_work() {
        let (backend, model) = test_model::load_default_model().unwrap();
        let ctx_params = LlamaContextParams::default().with_n_ctx(std::num::NonZeroU32::new(512));
        let mut context = model.new_context(&backend, ctx_params).unwrap();
        context.reset_timings();
        let timings = context.timings();
        assert!(timings.t_start_ms() >= 0.0);
    }

    #[test]
    #[serial]
    fn token_data_array_has_entries_after_decode() {
        let (backend, model) = test_model::load_default_model().unwrap();
        let ctx_params = LlamaContextParams::default().with_n_ctx(std::num::NonZeroU32::new(512));
        let mut context = model.new_context(&backend, ctx_params).unwrap();
        let tokens = model.str_to_token("hello", AddBos::Always).unwrap();
        let mut batch = LlamaBatch::new(512, 1).unwrap();
        batch.add_sequence(&tokens, 0, false).unwrap();
        context.decode(&mut batch).unwrap();

        let token_data_array = context.token_data_array();

        assert!(!token_data_array.data.is_empty());
    }

    #[test]
    #[serial]
    fn get_logits_ith_returns_valid_slice() {
        let (backend, model) = test_model::load_default_model().unwrap();
        let ctx_params = LlamaContextParams::default().with_n_ctx(std::num::NonZeroU32::new(512));
        let mut context = model.new_context(&backend, ctx_params).unwrap();
        let tokens = model.str_to_token("hello", AddBos::Always).unwrap();
        let last_index = i32::try_from(tokens.len() - 1).unwrap();
        let mut batch = LlamaBatch::new(512, 1).unwrap();
        batch.add_sequence(&tokens, 0, false).unwrap();
        context.decode(&mut batch).unwrap();

        let logits = context.get_logits_ith(last_index);

        assert_eq!(logits.len(), model.n_vocab() as usize);
    }

    #[test]
    #[serial]
    fn token_data_array_ith_returns_valid_data() {
        let (backend, model) = test_model::load_default_model().unwrap();
        let ctx_params = LlamaContextParams::default().with_n_ctx(std::num::NonZeroU32::new(512));
        let mut context = model.new_context(&backend, ctx_params).unwrap();
        let tokens = model.str_to_token("hello", AddBos::Always).unwrap();
        let last_index = i32::try_from(tokens.len() - 1).unwrap();
        let mut batch = LlamaBatch::new(512, 1).unwrap();
        batch.add_sequence(&tokens, 0, false).unwrap();
        context.decode(&mut batch).unwrap();

        let token_data_array = context.token_data_array_ith(last_index);

        assert_eq!(token_data_array.data.len(), model.n_vocab() as usize);
    }

    #[test]
    #[serial]
    fn embeddings_ith_returns_error_when_embeddings_disabled() {
        let (backend, model) = test_model::load_default_model().unwrap();
        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(std::num::NonZeroU32::new(512))
            .with_embeddings(false);
        let context = model.new_context(&backend, ctx_params).unwrap();

        let result = context.embeddings_ith(0);

        assert!(result.is_err());
    }

    #[test]
    #[serial]
    fn embeddings_seq_ith_returns_error_when_embeddings_disabled() {
        let (backend, model) = test_model::load_default_model().unwrap();
        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(std::num::NonZeroU32::new(512))
            .with_embeddings(false);
        let context = model.new_context(&backend, ctx_params).unwrap();

        let result = context.embeddings_seq_ith(0);

        assert!(result.is_err());
    }

    #[test]
    #[serial]
    fn candidates_returns_n_vocab_entries() {
        let (backend, model) = test_model::load_default_model().unwrap();
        let ctx_params = LlamaContextParams::default().with_n_ctx(std::num::NonZeroU32::new(512));
        let mut context = model.new_context(&backend, ctx_params).unwrap();
        let tokens = model.str_to_token("hello", AddBos::Always).unwrap();
        let mut batch = LlamaBatch::new(512, 1).unwrap();
        batch.add_sequence(&tokens, 0, false).unwrap();
        context.decode(&mut batch).unwrap();

        let count = context.candidates().count();

        assert_eq!(count, model.n_vocab() as usize);
    }

    #[test]
    #[serial]
    fn debug_format_contains_struct_name() {
        let (backend, model) = test_model::load_default_model().unwrap();
        let ctx_params = LlamaContextParams::default().with_n_ctx(std::num::NonZeroU32::new(512));
        let context = model.new_context(&backend, ctx_params).unwrap();
        let debug_output = format!("{context:?}");

        assert!(debug_output.contains("LlamaContext"));
    }

    #[test]
    #[serial]
    fn decode_with_embeddings_enabled() {
        let backend = crate::llama_backend::LlamaBackend::init().unwrap();
        let model_path = test_model::download_embedding_model().unwrap();
        let model_params = crate::model::params::LlamaModelParams::default();
        let model =
            crate::model::LlamaModel::load_from_file(&backend, &model_path, &model_params).unwrap();
        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(std::num::NonZeroU32::new(512))
            .with_embeddings(true);
        let mut context = model.new_context(&backend, ctx_params).unwrap();
        let tokens = model.str_to_token("hello", AddBos::Always).unwrap();
        let mut batch = LlamaBatch::new(512, 1).unwrap();
        batch.add_sequence(&tokens, 0, false).unwrap();

        let result = context.decode(&mut batch);

        assert!(result.is_ok());
    }

    #[test]
    #[serial]
    fn embeddings_seq_ith_returns_valid_embeddings() {
        let backend = crate::llama_backend::LlamaBackend::init().unwrap();
        let model_path = test_model::download_embedding_model().unwrap();
        let model_params = crate::model::params::LlamaModelParams::default();
        let model =
            crate::model::LlamaModel::load_from_file(&backend, &model_path, &model_params).unwrap();
        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(std::num::NonZeroU32::new(512))
            .with_embeddings(true);
        let mut context = model.new_context(&backend, ctx_params).unwrap();
        let tokens = model.str_to_token("hello", AddBos::Always).unwrap();
        let mut batch = LlamaBatch::new(512, 1).unwrap();
        batch.add_sequence(&tokens, 0, false).unwrap();
        context.decode(&mut batch).unwrap();

        let embeddings = context.embeddings_seq_ith(0).unwrap();

        assert_eq!(embeddings.len(), model.n_embd() as usize);
    }

    #[test]
    #[serial]
    fn embeddings_ith_returns_valid_embeddings() {
        let backend = crate::llama_backend::LlamaBackend::init().unwrap();
        let model_path = test_model::download_embedding_model().unwrap();
        let model_params = crate::model::params::LlamaModelParams::default();
        let model =
            crate::model::LlamaModel::load_from_file(&backend, &model_path, &model_params).unwrap();
        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(std::num::NonZeroU32::new(512))
            .with_embeddings(true);
        let mut context = model.new_context(&backend, ctx_params).unwrap();
        let tokens = model.str_to_token("hello", AddBos::Always).unwrap();
        let last_index = i32::try_from(tokens.len() - 1).unwrap();
        let mut batch = LlamaBatch::new(512, 1).unwrap();
        batch.add_sequence(&tokens, 0, false).unwrap();
        context.decode(&mut batch).unwrap();

        let embeddings = context.embeddings_ith(last_index).unwrap();

        assert_eq!(embeddings.len(), model.n_embd() as usize);
    }

    #[test]
    #[serial]
    fn candidates_ith_returns_n_vocab_entries() {
        let (backend, model) = test_model::load_default_model().unwrap();
        let ctx_params = LlamaContextParams::default().with_n_ctx(std::num::NonZeroU32::new(512));
        let mut context = model.new_context(&backend, ctx_params).unwrap();
        let tokens = model.str_to_token("hello", AddBos::Always).unwrap();
        let last_index = i32::try_from(tokens.len() - 1).unwrap();
        let mut batch = LlamaBatch::new(512, 1).unwrap();
        batch.add_sequence(&tokens, 0, false).unwrap();
        context.decode(&mut batch).unwrap();

        let count = context.candidates_ith(last_index).count();

        assert_eq!(count, model.n_vocab() as usize);
    }

    #[test]
    #[serial]
    fn lora_adapter_remove_succeeds_with_no_adapters() {
        let (backend, model) = test_model::load_default_model().unwrap();
        let ctx_params = LlamaContextParams::default().with_n_ctx(std::num::NonZeroU32::new(512));
        let context = model.new_context(&backend, ctx_params).unwrap();
        let mut adapter = crate::model::LlamaLoraAdapter {
            lora_adapter: std::ptr::NonNull::dangling(),
        };

        let result = context.lora_adapter_remove(&mut adapter);

        assert!(result.is_ok());
    }

    #[test]
    #[serial]
    fn encode_on_non_encoder_model_returns_error() {
        let (backend, model) = test_model::load_default_model().unwrap();
        let ctx_params = LlamaContextParams::default().with_n_ctx(std::num::NonZeroU32::new(512));
        let mut context = model.new_context(&backend, ctx_params).unwrap();
        let tokens = model.str_to_token("hello", AddBos::Always).unwrap();
        let mut batch = LlamaBatch::new(512, 1).unwrap();
        batch.add_sequence(&tokens, 0, false).unwrap();

        let result = context.encode(&mut batch);

        assert!(result.is_err());
    }

    #[test]
    #[serial]
    fn lora_adapter_set_with_dangling_pointer_succeeds_or_errors() {
        let (backend, model) = test_model::load_default_model().unwrap();
        let ctx_params = LlamaContextParams::default().with_n_ctx(std::num::NonZeroU32::new(512));
        let context = model.new_context(&backend, ctx_params).unwrap();
        let mut adapter = crate::model::LlamaLoraAdapter {
            lora_adapter: std::ptr::NonNull::dangling(),
        };

        let result = context.lora_adapter_set(&mut adapter, 1.0);

        assert!(result.is_ok());
    }

    #[test]
    #[serial]
    fn embeddings_ith_returns_null_embedding_error_for_non_embedding_token() {
        let backend = crate::llama_backend::LlamaBackend::init().unwrap();
        let model_path = test_model::download_embedding_model().unwrap();
        let model_params = crate::model::params::LlamaModelParams::default();
        let model =
            crate::model::LlamaModel::load_from_file(&backend, &model_path, &model_params).unwrap();
        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(std::num::NonZeroU32::new(512))
            .with_embeddings(true);
        let context = model.new_context(&backend, ctx_params).unwrap();

        let result = context.embeddings_ith(999);

        assert!(result.is_err());
    }

    #[test]
    #[serial]
    fn embeddings_seq_ith_returns_null_embedding_error_for_invalid_seq() {
        let (backend, model) = test_model::load_default_model().unwrap();
        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(std::num::NonZeroU32::new(512))
            .with_embeddings(true);
        let mut context = model.new_context(&backend, ctx_params).unwrap();
        let tokens = model.str_to_token("hello", AddBos::Always).unwrap();
        let mut batch = LlamaBatch::new(512, 1).unwrap();
        batch.add_sequence(&tokens, 0, false).unwrap();
        context.decode(&mut batch).unwrap();

        let result = context.embeddings_seq_ith(999);

        assert!(result.is_err());
    }

    #[test]
    #[serial]
    fn decode_empty_batch_returns_error() {
        let (backend, model) = test_model::load_default_model().unwrap();
        let ctx_params = LlamaContextParams::default().with_n_ctx(std::num::NonZeroU32::new(512));
        let mut context = model.new_context(&backend, ctx_params).unwrap();
        let mut batch = LlamaBatch::new(512, 1).unwrap();

        let result = context.decode(&mut batch);

        assert!(result.is_err());
    }

    #[test]
    #[serial]
    fn encode_succeeds_with_encoder_model() {
        let backend = crate::llama_backend::LlamaBackend::init().unwrap();
        let model_path = test_model::download_encoder_model().unwrap();
        let model_params = crate::model::params::LlamaModelParams::default();
        let model =
            crate::model::LlamaModel::load_from_file(&backend, &model_path, &model_params).unwrap();
        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(std::num::NonZeroU32::new(512))
            .with_embeddings(true);
        let mut context = model.new_context(&backend, ctx_params).unwrap();
        let tokens = model.str_to_token("hello", AddBos::Never).unwrap();
        let mut batch = LlamaBatch::new(512, 1).unwrap();
        batch.add_sequence(&tokens, 0, false).unwrap();

        let result = context.encode(&mut batch);

        assert!(result.is_ok());
    }

    #[test]
    #[serial]
    fn handle_encode_result_ok_updates_logits() {
        let (backend, model) = test_model::load_default_model().unwrap();
        let ctx_params = LlamaContextParams::default().with_n_ctx(std::num::NonZeroU32::new(512));
        let mut context = model.new_context(&backend, ctx_params).unwrap();
        let tokens = model.str_to_token("hello", AddBos::Always).unwrap();
        let mut batch = LlamaBatch::new(512, 1).unwrap();
        batch.add_sequence(&tokens, 0, true).unwrap();

        let result =
            context.handle_encode_result(llama_cpp_bindings_sys::LLAMA_RS_STATUS_OK, &mut batch);

        assert!(result.is_ok());
        assert!(!context.initialized_logits.is_empty());
    }
}
