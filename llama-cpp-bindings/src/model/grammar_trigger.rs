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

#[cfg(test)]
mod tests {
    use super::GrammarTrigger;
    use super::GrammarTriggerType;
    use crate::token::LlamaToken;

    #[test]
    fn token_type_has_token() {
        let trigger = GrammarTrigger {
            trigger_type: GrammarTriggerType::Token,
            value: String::from("test_token"),
            token: Some(LlamaToken(42)),
        };

        assert!(trigger.token.is_some());
    }

    #[test]
    fn non_token_type_has_no_token() {
        let trigger = GrammarTrigger {
            trigger_type: GrammarTriggerType::Word,
            value: String::from("test_word"),
            token: None,
        };

        assert!(trigger.token.is_none());
    }
}
