//! Flags for extended state sequence operations on hybrid/recurrent models.

/// Flags controlling which parts of state to save/restore for sequence operations.
///
/// Used with the `state_seq_*_ext` methods on [`super::LlamaContext`] to enable
/// partial state operations (e.g., saving only recurrent/SSM state for hybrid models).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LlamaStateSeqFlags {
    flags: u32,
}

impl LlamaStateSeqFlags {
    /// Save/restore only partial (recurrent/SSM) state, skipping attention KV cache.
    pub const PARTIAL_ONLY: Self = Self { flags: 1 };

    /// No flags set.
    #[must_use]
    pub const fn empty() -> Self {
        Self { flags: 0 }
    }

    /// Returns the raw bit representation.
    #[must_use]
    pub const fn bits(&self) -> u32 {
        self.flags
    }

    /// Returns true if `self` contains all bits in `other`.
    #[must_use]
    pub const fn contains(&self, other: Self) -> bool {
        (self.flags & other.flags) == other.flags
    }
}

impl std::ops::BitOr for LlamaStateSeqFlags {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self {
        Self {
            flags: self.flags | rhs.flags,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::LlamaStateSeqFlags;

    #[test]
    fn empty_has_no_bits_set() {
        assert_eq!(LlamaStateSeqFlags::empty().bits(), 0);
    }

    #[test]
    fn partial_only_has_bit_one() {
        assert_eq!(LlamaStateSeqFlags::PARTIAL_ONLY.bits(), 1);
    }

    #[test]
    fn bitor_combines_flags() {
        let combined = LlamaStateSeqFlags::empty() | LlamaStateSeqFlags::PARTIAL_ONLY;

        assert_eq!(combined.bits(), 1);
    }

    #[test]
    fn contains_detects_set_flag() {
        let flags = LlamaStateSeqFlags::PARTIAL_ONLY;

        assert!(flags.contains(LlamaStateSeqFlags::PARTIAL_ONLY));
    }

    #[test]
    fn empty_does_not_contain_partial_only() {
        let flags = LlamaStateSeqFlags::empty();

        assert!(!flags.contains(LlamaStateSeqFlags::PARTIAL_ONLY));
    }

    #[test]
    fn equality_works() {
        assert_eq!(LlamaStateSeqFlags::empty(), LlamaStateSeqFlags::empty());
        assert_ne!(
            LlamaStateSeqFlags::empty(),
            LlamaStateSeqFlags::PARTIAL_ONLY
        );
    }
}
