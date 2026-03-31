//! Safe wrapper around `llama_logit_bias`.
use crate::token::LlamaToken;

/// A transparent wrapper around `llama_logit_bias`.
///
/// Represents a bias to be applied to a specific token during text generation.
/// The bias modifies the likelihood of the token being selected.
///
/// Do not rely on `repr(transparent)` for this type. It should be considered an implementation
/// detail and may change across minor versions.
#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(transparent)]
pub struct LlamaLogitBias {
    logit_bias: llama_cpp_bindings_sys::llama_logit_bias,
}

impl LlamaLogitBias {
    /// Creates a new logit bias for a specific token with the given bias value.
    ///
    /// # Examples
    /// ```
    /// # use llama_cpp_bindings::token::{LlamaToken, logit_bias::LlamaLogitBias};
    /// let token = LlamaToken::new(1);
    /// let bias = LlamaLogitBias::new(token, 1.5);
    /// ```
    #[must_use]
    pub const fn new(LlamaToken(token): LlamaToken, bias: f32) -> Self {
        Self {
            logit_bias: llama_cpp_bindings_sys::llama_logit_bias { token, bias },
        }
    }

    /// Gets the token this bias applies to.
    ///
    /// # Examples
    /// ```
    /// # use llama_cpp_bindings::token::{LlamaToken, logit_bias::LlamaLogitBias};
    /// let token = LlamaToken::new(1);
    /// let bias = LlamaLogitBias::new(token, 1.5);
    /// assert_eq!(bias.token(), token);
    /// ```
    #[must_use]
    pub const fn token(&self) -> LlamaToken {
        LlamaToken(self.logit_bias.token)
    }

    /// Gets the bias value.
    ///
    /// # Examples
    /// ```
    /// # use llama_cpp_bindings::token::{LlamaToken, logit_bias::LlamaLogitBias};
    /// let token = LlamaToken::new(1);
    /// let bias = LlamaLogitBias::new(token, 1.5);
    /// assert_eq!(bias.bias(), 1.5);
    /// ```
    #[must_use]
    pub const fn bias(&self) -> f32 {
        self.logit_bias.bias
    }

    /// Sets the token this bias applies to.
    ///
    /// # Examples
    /// ```
    /// # use llama_cpp_bindings::token::{LlamaToken, logit_bias::LlamaLogitBias};
    /// let token = LlamaToken::new(1);
    /// let mut bias = LlamaLogitBias::new(token, 1.5);
    /// let new_token = LlamaToken::new(2);
    /// bias.set_token(new_token);
    /// assert_eq!(bias.token(), new_token);
    /// ```
    pub const fn set_token(&mut self, token: LlamaToken) {
        self.logit_bias.token = token.0;
    }

    /// Sets the bias value.
    ///
    /// # Examples
    /// ```
    /// # use llama_cpp_bindings::token::{LlamaToken, logit_bias::LlamaLogitBias};
    /// let token = LlamaToken::new(1);
    /// let mut bias = LlamaLogitBias::new(token, 1.5);
    /// bias.set_bias(2.0);
    /// assert_eq!(bias.bias(), 2.0);
    /// ```
    pub const fn set_bias(&mut self, bias: f32) {
        self.logit_bias.bias = bias;
    }
}

#[cfg(test)]
mod tests {
    use super::LlamaLogitBias;
    use crate::token::LlamaToken;

    #[test]
    fn new_stores_token_and_bias() {
        let token = LlamaToken::new(42);
        let logit_bias = LlamaLogitBias::new(token, 1.5);
        assert_eq!(logit_bias.token(), token);
        assert!((logit_bias.bias() - 1.5_f32).abs() < f32::EPSILON);
    }

    #[test]
    fn set_token_updates_token() {
        let mut logit_bias = LlamaLogitBias::new(LlamaToken::new(1), 0.5);
        let new_token = LlamaToken::new(99);
        logit_bias.set_token(new_token);
        assert_eq!(logit_bias.token(), new_token);
    }

    #[test]
    fn set_bias_updates_bias() {
        let mut logit_bias = LlamaLogitBias::new(LlamaToken::new(1), 0.5);
        logit_bias.set_bias(-3.0);
        assert!((logit_bias.bias() - (-3.0_f32)).abs() < f32::EPSILON);
    }
}
