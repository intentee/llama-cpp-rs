/// Options to configure how llama.cpp logs are intercepted.
#[derive(Default, Debug, Clone)]
pub struct LogOptions {
    pub disabled: bool,
}

impl LogOptions {
    /// If enabled, logs are sent to tracing. If disabled, all logs are suppressed. Default is for
    /// logs to be sent to tracing.
    #[must_use]
    pub fn with_logs_enabled(mut self, enabled: bool) -> Self {
        self.disabled = !enabled;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::LogOptions;

    #[test]
    fn with_logs_enabled_true() {
        let options = LogOptions::default().with_logs_enabled(true);

        assert!(!options.disabled);
    }

    #[test]
    fn with_logs_enabled_false() {
        let options = LogOptions::default().with_logs_enabled(false);

        assert!(options.disabled);
    }
}
