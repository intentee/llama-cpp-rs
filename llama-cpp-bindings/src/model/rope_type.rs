/// The Rope type that's used within the model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RopeType {
    /// Standard rotary positional encoding.
    Norm,
    /// GPT-NeoX style rotary positional encoding.
    NeoX,
    /// Multi-dimensional rotary positional encoding.
    MRope,
    /// Vision model rotary positional encoding.
    Vision,
}

/// Converts a raw llama.cpp rope type constant to a `RopeType`.
/// Returns `None` for unknown or "none" rope types.
#[must_use]
pub const fn rope_type_from_raw(raw: i32) -> Option<RopeType> {
    match raw {
        llama_cpp_bindings_sys::LLAMA_ROPE_TYPE_NORM => Some(RopeType::Norm),
        llama_cpp_bindings_sys::LLAMA_ROPE_TYPE_NEOX => Some(RopeType::NeoX),
        llama_cpp_bindings_sys::LLAMA_ROPE_TYPE_MROPE => Some(RopeType::MRope),
        llama_cpp_bindings_sys::LLAMA_ROPE_TYPE_VISION => Some(RopeType::Vision),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::{RopeType, rope_type_from_raw};

    #[test]
    fn rope_type_none() {
        assert_eq!(
            rope_type_from_raw(llama_cpp_bindings_sys::LLAMA_ROPE_TYPE_NONE),
            None
        );
    }

    #[test]
    fn rope_type_norm() {
        assert_eq!(
            rope_type_from_raw(llama_cpp_bindings_sys::LLAMA_ROPE_TYPE_NORM),
            Some(RopeType::Norm)
        );
    }

    #[test]
    fn rope_type_neox() {
        assert_eq!(
            rope_type_from_raw(llama_cpp_bindings_sys::LLAMA_ROPE_TYPE_NEOX),
            Some(RopeType::NeoX)
        );
    }

    #[test]
    fn rope_type_mrope() {
        assert_eq!(
            rope_type_from_raw(llama_cpp_bindings_sys::LLAMA_ROPE_TYPE_MROPE),
            Some(RopeType::MRope)
        );
    }

    #[test]
    fn rope_type_vision() {
        assert_eq!(
            rope_type_from_raw(llama_cpp_bindings_sys::LLAMA_ROPE_TYPE_VISION),
            Some(RopeType::Vision)
        );
    }

    #[test]
    fn rope_type_unknown_returns_none() {
        assert_eq!(rope_type_from_raw(9999), None);
    }
}
