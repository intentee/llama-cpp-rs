/// Input chunk types for multimodal data
///
/// # Examples
///
/// ```
/// use llama_cpp_bindings::mtmd::MtmdInputChunkType;
///
/// let text_chunk = MtmdInputChunkType::Text;
/// let image_chunk = MtmdInputChunkType::Image;
/// let audio_chunk = MtmdInputChunkType::Audio;
///
/// assert_eq!(text_chunk, MtmdInputChunkType::Text);
/// assert_eq!(text_chunk, llama_cpp_bindings_sys::MTMD_INPUT_CHUNK_TYPE_TEXT.into());
/// assert_ne!(text_chunk, image_chunk);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum MtmdInputChunkType {
    /// Text input chunk
    Text = llama_cpp_bindings_sys::MTMD_INPUT_CHUNK_TYPE_TEXT as _,
    /// Image input chunk
    Image = llama_cpp_bindings_sys::MTMD_INPUT_CHUNK_TYPE_IMAGE as _,
    /// Audio input chunk
    Audio = llama_cpp_bindings_sys::MTMD_INPUT_CHUNK_TYPE_AUDIO as _,
}

impl From<llama_cpp_bindings_sys::mtmd_input_chunk_type> for MtmdInputChunkType {
    fn from(chunk_type: llama_cpp_bindings_sys::mtmd_input_chunk_type) -> Self {
        match chunk_type {
            llama_cpp_bindings_sys::MTMD_INPUT_CHUNK_TYPE_TEXT => MtmdInputChunkType::Text,
            llama_cpp_bindings_sys::MTMD_INPUT_CHUNK_TYPE_IMAGE => MtmdInputChunkType::Image,
            llama_cpp_bindings_sys::MTMD_INPUT_CHUNK_TYPE_AUDIO => MtmdInputChunkType::Audio,
            _ => panic!("Unknown MTMD input chunk type: {chunk_type}"),
        }
    }
}
