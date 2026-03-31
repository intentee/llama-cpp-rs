//! Safe wrapper around `gguf_context` for reading GGUF file metadata.
//!
//! Provides metadata-only access to GGUF files without loading tensor data.

use std::ffi::{CStr, CString};
use std::path::Path;
use std::ptr::NonNull;

use crate::gguf_context_error::GgufContextError;
use crate::gguf_type::GgufType;

/// A safe wrapper around `gguf_context`.
///
/// Opens a GGUF file in metadata-only mode (`no_alloc = true`), allowing
/// inspection of key-value pairs and tensor metadata without loading tensor data.
pub struct GgufContext {
    context: NonNull<llama_cpp_bindings_sys::gguf_context>,
}

impl GgufContext {
    /// Open a GGUF file and parse its metadata header.
    ///
    /// # Errors
    ///
    /// Returns [`GgufContextError::InitFailed`] if the file cannot be opened or parsed.
    /// Returns [`GgufContextError::PathToStrError`] if the path is not valid UTF-8.
    /// Returns [`GgufContextError::NulError`] if the path contains a null byte.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, GgufContextError> {
        let path_ref = path.as_ref();
        let path_str = path_ref
            .to_str()
            .ok_or_else(|| GgufContextError::PathToStrError(path_ref.to_path_buf()))?;
        let c_path = CString::new(path_str)?;

        let init_params = llama_cpp_bindings_sys::gguf_init_params {
            no_alloc: true,
            ctx: std::ptr::null_mut(),
        };

        let raw =
            unsafe { llama_cpp_bindings_sys::gguf_init_from_file(c_path.as_ptr(), init_params) };
        let context = NonNull::new(raw)
            .ok_or_else(|| GgufContextError::InitFailed(path_ref.to_path_buf()))?;

        Ok(Self { context })
    }

    /// Returns the number of key-value pairs in the GGUF file.
    #[must_use]
    pub fn n_kv(&self) -> i64 {
        unsafe { llama_cpp_bindings_sys::gguf_get_n_kv(self.context.as_ptr()) }
    }

    /// Find the index of a key by name.
    ///
    /// # Errors
    ///
    /// Returns [`GgufContextError::KeyNotFound`] if the key does not exist.
    /// Returns [`GgufContextError::NulError`] if the key contains a null byte.
    pub fn find_key(&self, key: &str) -> Result<i64, GgufContextError> {
        let c_key = CString::new(key)?;
        let index =
            unsafe { llama_cpp_bindings_sys::gguf_find_key(self.context.as_ptr(), c_key.as_ptr()) };

        if index < 0 {
            return Err(GgufContextError::KeyNotFound {
                key: key.to_string(),
            });
        }

        Ok(index)
    }

    /// Returns the key name at the given index.
    ///
    /// # Safety considerations
    ///
    /// The caller must ensure `key_id` is in range `[0, n_kv())`.
    ///
    /// # Errors
    ///
    /// Returns [`GgufContextError::Utf8Error`] if the key name is not valid UTF-8.
    pub fn key_at(&self, key_id: i64) -> Result<&str, GgufContextError> {
        let c_str = unsafe {
            CStr::from_ptr(llama_cpp_bindings_sys::gguf_get_key(
                self.context.as_ptr(),
                key_id,
            ))
        };

        Ok(c_str.to_str()?)
    }

    /// Returns the value type of the key-value pair at the given index.
    ///
    /// # Safety considerations
    ///
    /// The caller must ensure `key_id` is in range `[0, n_kv())`.
    #[must_use]
    pub fn kv_type(&self, key_id: i64) -> Option<GgufType> {
        let raw =
            unsafe { llama_cpp_bindings_sys::gguf_get_kv_type(self.context.as_ptr(), key_id) };

        GgufType::from_raw(raw)
    }

    /// Returns the u32 value at the given key index.
    ///
    /// # Safety considerations
    ///
    /// The caller must ensure the key at `key_id` has type [`GgufType::Uint32`].
    #[must_use]
    pub fn val_u32(&self, key_id: i64) -> u32 {
        unsafe { llama_cpp_bindings_sys::gguf_get_val_u32(self.context.as_ptr(), key_id) }
    }

    /// Returns the i32 value at the given key index.
    ///
    /// # Safety considerations
    ///
    /// The caller must ensure the key at `key_id` has type [`GgufType::Int32`].
    #[must_use]
    pub fn val_i32(&self, key_id: i64) -> i32 {
        unsafe { llama_cpp_bindings_sys::gguf_get_val_i32(self.context.as_ptr(), key_id) }
    }

    /// Returns the u64 value at the given key index.
    ///
    /// # Safety considerations
    ///
    /// The caller must ensure the key at `key_id` has type [`GgufType::Uint64`].
    #[must_use]
    pub fn val_u64(&self, key_id: i64) -> u64 {
        unsafe { llama_cpp_bindings_sys::gguf_get_val_u64(self.context.as_ptr(), key_id) }
    }

    /// Returns the string value at the given key index.
    ///
    /// # Safety considerations
    ///
    /// The caller must ensure the key at `key_id` has type [`GgufType::String`].
    ///
    /// # Errors
    ///
    /// Returns [`GgufContextError::Utf8Error`] if the string value is not valid UTF-8.
    pub fn val_str(&self, key_id: i64) -> Result<&str, GgufContextError> {
        let c_str = unsafe {
            CStr::from_ptr(llama_cpp_bindings_sys::gguf_get_val_str(
                self.context.as_ptr(),
                key_id,
            ))
        };

        Ok(c_str.to_str()?)
    }

    /// Returns the number of tensors in the GGUF file.
    #[must_use]
    pub fn n_tensors(&self) -> i64 {
        unsafe { llama_cpp_bindings_sys::gguf_get_n_tensors(self.context.as_ptr()) }
    }
}

impl Drop for GgufContext {
    fn drop(&mut self) {
        unsafe { llama_cpp_bindings_sys::gguf_free(self.context.as_ptr()) }
    }
}

#[cfg(test)]
mod tests {
    use super::GgufContext;
    use crate::gguf_context_error::GgufContextError;
    use crate::gguf_type::GgufType;

    fn fixture_path() -> std::path::PathBuf {
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("fixtures")
            .join("ggml-vocab-bert-bge.gguf")
    }

    #[test]
    fn from_file_opens_valid_gguf() {
        let context = GgufContext::from_file(fixture_path());

        assert!(context.is_ok());
    }

    #[test]
    fn from_file_nonexistent_returns_init_failed() {
        let result = GgufContext::from_file("/nonexistent/file.gguf");

        assert!(matches!(result, Err(GgufContextError::InitFailed(_))));
    }

    #[test]
    fn n_kv_returns_positive_count() {
        let context = GgufContext::from_file(fixture_path()).unwrap();

        assert!(context.n_kv() > 0);
    }

    #[test]
    fn n_tensors_returns_count() {
        let context = GgufContext::from_file(fixture_path()).unwrap();

        assert!(context.n_tensors() >= 0);
    }

    #[test]
    fn find_key_returns_valid_index_for_known_key() {
        let context = GgufContext::from_file(fixture_path()).unwrap();
        let index = context.find_key("general.architecture");

        assert!(index.is_ok());
        assert!(index.unwrap() >= 0);
    }

    #[test]
    fn find_key_returns_error_for_missing_key() {
        let context = GgufContext::from_file(fixture_path()).unwrap();
        let result = context.find_key("nonexistent.key");

        assert!(matches!(result, Err(GgufContextError::KeyNotFound { .. })));
    }

    #[test]
    fn key_at_returns_expected_name() {
        let context = GgufContext::from_file(fixture_path()).unwrap();
        let index = context.find_key("general.architecture").unwrap();
        let key_name = context.key_at(index).unwrap();

        assert_eq!(key_name, "general.architecture");
    }

    #[test]
    fn kv_type_returns_expected_type_for_string_key() {
        let context = GgufContext::from_file(fixture_path()).unwrap();
        let index = context.find_key("general.architecture").unwrap();
        let value_type = context.kv_type(index);

        assert_eq!(value_type, Some(GgufType::String));
    }

    #[test]
    fn val_str_returns_architecture_value() {
        let context = GgufContext::from_file(fixture_path()).unwrap();
        let index = context.find_key("general.architecture").unwrap();
        let value = context.val_str(index).unwrap();

        assert!(!value.is_empty());
    }

    #[cfg(unix)]
    #[test]
    fn from_file_non_utf8_path_returns_error() {
        use std::ffi::OsStr;
        use std::os::unix::ffi::OsStrExt;

        let non_utf8_path = std::path::Path::new(OsStr::from_bytes(b"/tmp/\xff\xfe.gguf"));
        let result = GgufContext::from_file(non_utf8_path);

        assert!(matches!(result, Err(GgufContextError::PathToStrError(_))));
    }
}
