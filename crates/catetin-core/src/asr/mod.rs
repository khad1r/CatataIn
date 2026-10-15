use anyhow::Result;
#[allow(unused_imports)]
use std::path::Path;

pub trait SpeechRecognizer: Send + Sync {
    /// Transcribe a slice of 16,000Hz mono f32 samples into text
    fn transcribe(&mut self, samples_16k_mono: &[f32]) -> Result<String>;
    /// Current language setting (e.g. "auto", "id", "en")
    fn language(&self) -> &str;
}

/// Lightweight testing / mock speech recognizer for verification without downloading weights
#[derive(Debug, Clone)]
pub struct MockRecognizer {
    language: String,
}

impl MockRecognizer {
    pub fn new(language: impl Into<String>) -> Self {
        Self {
            language: language.into(),
        }
    }
}

impl SpeechRecognizer for MockRecognizer {
    fn transcribe(&mut self, samples: &[f32]) -> Result<String> {
        let duration_s = samples.len() as f32 / 16_000.0;
        Ok(format!("[Speech detected: {:.2}s audio]", duration_s))
    }

    fn language(&self) -> &str {
        &self.language
    }
}

#[cfg(feature = "whisper")]
pub mod whisper_backend {
    use super::*;
    use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

    pub struct WhisperEngine {
        ctx: WhisperContext,
        language: String,
    }

    impl WhisperEngine {
        pub fn load_from_file<P: AsRef<Path>>(path: P, language: &str) -> Result<Self> {
            let path_str = path.as_ref().to_str().ok_or_else(|| {
                anyhow::anyhow!("Invalid unicode in model path: {:?}", path.as_ref())
            })?;

            let params = WhisperContextParameters::default();
            let ctx = WhisperContext::new_with_params(path_str, params)
                .map_err(|e| anyhow::anyhow!("Failed to initialize WhisperContext: {:?}", e))?;

            Ok(Self {
                ctx,
                language: language.to_string(),
            })
        }

        pub fn load_from_bytes(bytes: &[u8], language: &str) -> Result<Self> {
            let params = WhisperContextParameters::default();
            let ctx = WhisperContext::new_from_buffer_with_params(bytes, params)
                .map_err(|e| anyhow::anyhow!("Failed to initialize WhisperContext from buffer: {:?}", e))?;

            Ok(Self {
                ctx,
                language: language.to_string(),
            })
        }
    }

    impl SpeechRecognizer for WhisperEngine {
        fn transcribe(&mut self, samples: &[f32]) -> Result<String> {
            if samples.is_empty() {
                return Ok(String::new());
            }

            let mut state = self
                .ctx
                .create_state()
                .map_err(|e| anyhow::anyhow!("Failed to create Whisper state: {:?}", e))?;

            let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
            params.set_language(Some(&self.language));
            params.set_translate(false); // Verbatim transcription
            params.set_no_timestamps(true);
            params.set_print_special(false);
            params.set_print_progress(false);
            params.set_print_realtime(false);
            params.set_print_timestamps(false);

            state
                .full(params, samples)
                .map_err(|e| anyhow::anyhow!("Failed to run Whisper inference: {:?}", e))?;

            let num_segments = state
                .full_n_segments()
                .map_err(|e| anyhow::anyhow!("Failed to get segment count: {:?}", e))?;

            let mut text = String::new();
            for i in 0..num_segments {
                if let Ok(segment) = state.full_get_segment_text(i) {
                    text.push_str(&segment);
                    text.push(' ');
                }
            }

            Ok(text.trim().to_string())
        }

        fn language(&self) -> &str {
            &self.language
        }
    }
}
