use anyhow::Result;
#[cfg(feature = "whisper")]
use std::path::Path;

pub trait SpeechRecognizer: Send + Sync {
    /// Transcribe a slice of 16,000Hz mono f32 samples into text
    fn transcribe(&mut self, samples_16k_mono: &[f32]) -> Result<String>;
    /// Current language setting (e.g. "auto", "id", "en")
    fn language(&self) -> &str;
}




/// Lightweight testing recognizer strictly for automated unit tests
#[cfg(test)]
#[derive(Debug, Clone)]
pub struct MockRecognizer {
    language: String,
}

#[cfg(test)]
impl MockRecognizer {
    pub fn new(language: impl Into<String>) -> Self {
        Self {
            language: language.into(),
        }
    }
}

#[cfg(test)]
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
    use std::sync::Once;
    use whisper_rs::{
        FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters, WhisperState,
    };

    static WHISPER_LOG_INIT: Once = Once::new();

    fn ensure_logging_hooks() {
        WHISPER_LOG_INIT.call_once(|| {
            whisper_rs::install_logging_hooks();
        });
    }

    pub struct WhisperEngine {
        _ctx: WhisperContext,
        state: WhisperState,
        language: String,
    }

    // Safety: WhisperEngine is owned and accessed sequentially by the dedicated ASR worker thread
    unsafe impl Send for WhisperEngine {}
    unsafe impl Sync for WhisperEngine {}

    impl WhisperEngine {
        pub fn load_from_file<P: AsRef<Path>>(path: P, language: &str) -> Result<Self> {
            ensure_logging_hooks();
            let path_str = path.as_ref().to_str().ok_or_else(|| {
                anyhow::anyhow!("Invalid unicode in model path: {:?}", path.as_ref())
            })?;

            let params = WhisperContextParameters::default();
            let ctx = WhisperContext::new_with_params(path_str, params)
                .map_err(|e| anyhow::anyhow!("Failed to initialize WhisperContext: {:?}", e))?;
            let state = ctx
                .create_state()
                .map_err(|e| anyhow::anyhow!("Failed to create WhisperState: {:?}", e))?;

            Ok(Self {
                _ctx: ctx,
                state,
                language: language.to_string(),
            })
        }

        pub fn load_from_bytes(bytes: &[u8], language: &str) -> Result<Self> {
            ensure_logging_hooks();
            let params = WhisperContextParameters::default();
            let ctx = WhisperContext::new_from_buffer_with_params(bytes, params).map_err(|e| {
                anyhow::anyhow!("Failed to initialize WhisperContext from buffer: {:?}", e)
            })?;
            let state = ctx
                .create_state()
                .map_err(|e| anyhow::anyhow!("Failed to create WhisperState: {:?}", e))?;

            Ok(Self {
                _ctx: ctx,
                state,
                language: language.to_string(),
            })
        }
    }

    impl SpeechRecognizer for WhisperEngine {
        fn transcribe(&mut self, samples: &[f32]) -> Result<String> {
            // 1. Ignore micro-clicks shorter than 200ms (3,200 samples at 16kHz)
            if samples.len() < 3_200 {
                return Ok(String::new());
            }

            // 2. Cap maximum chunk size at 6.0s (96,000 samples)
            let capped = if samples.len() > 96_000 {
                &samples[..96_000]
            } else {
                samples
            };

            let peak = capped
                .iter()
                .fold(0.0f32, |acc, &x| acc.max(x.abs()));

            // Skip pure background hum/air if peak amplitude is too low to be speech
            if peak < 0.012 {
                return Ok(String::new());
            }

            let gain = if peak < 0.40 {
                (0.55 / peak).min(10.0)
            } else {
                1.0
            };

            // Dynamic audio_ctx scaling: 1500 encoder frames = 30.0s (320 samples per encoder frame at 16kHz).
            // Add 24 frames (~480ms) of clean trailing silence so the encoder clearly sees end-of-speech
            // and pad `prepared` to the exact `audio_ctx * 320` window so reused WhisperState mel buffers
            // never contain leftover audio from a previous utterance.
            let speech_frames = (capped.len() as i32 + 319) / 320;
            let needed_frames = speech_frames + 24;
            let audio_ctx = ((needed_frames + 63) / 64 * 64).clamp(128, 1500);
            let target_samples = (audio_ctx as usize * 320).max(20_480);

            // Add 160ms (2,560 samples) of leading silence + speech + trailing silence up to target_samples
            let lead_pad = 2_560usize;
            let mut prepared = Vec::with_capacity(target_samples + lead_pad);
            prepared.resize(lead_pad, 0.0);
            for &s in capped {
                prepared.push((s * gain).clamp(-1.0, 1.0));
            }
            if prepared.len() < target_samples {
                prepared.resize(target_samples, 0.0);
            }

            let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
            let n_threads = std::thread::available_parallelism()
                .map(|n| n.get().clamp(4, 8) as i32)
                .unwrap_or(4);
            params.set_n_threads(n_threads);
            params.set_audio_ctx(audio_ctx);
            // A 1-6s utterance has at most ~25 tokens; capping at 32 prevents runaway decoder loops
            params.set_max_tokens(32);

            if self.language.is_empty() || self.language.eq_ignore_ascii_case("auto") {
                params.set_language(Some("id"));
            } else {
                params.set_language(Some(&self.language));
            }

            params.set_translate(false); // Verbatim transcription
            params.set_no_context(true); // Prevent hallucination loops across chunks
            params.set_single_segment(true);
            // IMPORTANT: Keep timestamp tokens enabled internally (`no_timestamps = false`)!
            // Whisper uses `<|0.00|> ... <|1.80|>` timestamp tokens to anchor attention monotonically
            // in time; disabling them causes Whisper to loop back to 0.00s and repeat sentences 2-3x!
            params.set_no_timestamps(false);
            params.set_suppress_blank(true);
            params.set_suppress_nst(true);
            params.set_no_speech_thold(0.60);
            params.set_logprob_thold(-1.0);
            params.set_entropy_thold(2.40);
            params.set_temperature(0.0);
            params.set_temperature_inc(0.0); // Disable slow 6x temperature fallback retries
            params.set_print_special(false);
            params.set_print_progress(false);
            params.set_print_realtime(false);
            params.set_print_timestamps(false);

            self.state
                .full(params, &prepared)
                .map_err(|e| anyhow::anyhow!("Failed to run Whisper inference: {:?}", e))?;

            let mut text = String::new();
            for segment in self.state.as_iter() {
                let seg_str = segment.to_string();
                let clean = seg_str.trim();
                // Filter out Whisper non-speech placeholder tags
                if !clean.is_empty()
                    && !clean.starts_with('[')
                    && !clean.starts_with('(')
                    && !clean.starts_with('*')
                {
                    text.push_str(clean);
                    text.push(' ');
                }
            }

            Ok(deduplicate_repetitions(text.trim()))
        }

        fn language(&self) -> &str {
            &self.language
        }
    }

    /// Collapse immediate consecutive sentence/phrase repetitions that can occur on short audio buffers
    /// (e.g. "Halo, halo, 123. Halo, halo, 123." -> "Halo, halo, 123.").
    fn deduplicate_repetitions(input: &str) -> String {
        if input.is_empty() {
            return String::new();
        }
        let mut unique_parts: Vec<&str> = Vec::new();
        for part in input.split_inclusive(&['.', '!', '?'][..]) {
            let trimmed = part.trim();
            if trimmed.is_empty() {
                continue;
            }
            if let Some(&last) = unique_parts.last() {
                if last.trim().eq_ignore_ascii_case(trimmed) {
                    continue;
                }
            }
            unique_parts.push(trimmed);
        }
        if unique_parts.is_empty() {
            input.to_string()
        } else {
            unique_parts.join(" ")
        }
    }
}
