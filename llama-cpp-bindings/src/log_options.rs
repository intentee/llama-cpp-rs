/// Options to configure how llama.cpp logs are intercepted.
#[derive(Default, Debug, Clone)]
pub struct LogOptions {
    pub disabled: bool,
    pub demote_info_to_debug: bool,
}

impl LogOptions {
    /// If enabled, logs are sent to tracing. If disabled, all logs are suppressed. Default is for
    /// logs to be sent to tracing.
    #[must_use]
    pub const fn with_logs_enabled(mut self, enabled: bool) -> Self {
        self.disabled = !enabled;

        self
    }

    /// When enabled, llama.cpp and ggml INFO logs are demoted to DEBUG tracing level. WARN and
    /// ERROR logs retain their original severity. This suppresses verbose informational output
    /// under a typical INFO-level subscriber while keeping important diagnostics visible.
    /// All demoted logs remain available via `RUST_LOG=debug`.
    #[must_use]
    pub const fn with_demote_info_to_debug(mut self, demote: bool) -> Self {
        self.demote_info_to_debug = demote;

        self
    }
}

#[cfg(test)]
mod tests {
    use super::LogOptions;

    #[test]
    fn builder_chain_sets_both_flags() {
        let options = LogOptions::default()
            .with_logs_enabled(false)
            .with_demote_info_to_debug(true);

        assert!(options.disabled);
        assert!(options.demote_info_to_debug);
    }
}
