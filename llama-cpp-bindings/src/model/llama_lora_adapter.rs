use std::ptr::NonNull;

/// A safe wrapper around `llama_lora_adapter`.
#[derive(Debug)]
#[repr(transparent)]
pub struct LlamaLoraAdapter {
    /// Raw pointer to the underlying `llama_adapter_lora`.
    pub lora_adapter: NonNull<llama_cpp_bindings_sys::llama_adapter_lora>,
}
