use crate::token::LlamaToken;

/// Grammar trigger kinds used for lazy grammar sampling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GrammarTriggerType {
    /// Trigger on a specific token.
    Token = 0,
    /// Trigger on a literal word.
    Word = 1,
    /// Trigger on a regex pattern.
    Pattern = 2,
    /// Trigger on a fully anchored regex pattern.
    PatternFull = 3,
}

/// Lazy grammar trigger from chat template generation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GrammarTrigger {
    /// Trigger kind.
    pub trigger_type: GrammarTriggerType,
    /// Trigger text or pattern.
    pub value: String,
    /// Token id for token triggers.
    pub token: Option<LlamaToken>,
}
