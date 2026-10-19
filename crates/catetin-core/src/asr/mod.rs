use anyhow::Result;
use std::path::Path;

pub mod discovery;
pub use discovery::{
    discover_all_models, discover_transducer_models, discover_whisper_models, find_transducer_model_files,
    find_whisper_model_files, DiscoveredModel, ModelKind,
};

pub mod transducer;
pub use transducer::{
    create_sherpa_whisper, create_transducer, create_transducer_for_lang, ensure_runtime_extracted,
    has_transducer_installed, has_transducer_installed_for_lang,
    has_whisper_onnx_installed, ipa_phonemes_to_indonesian, try_create_transducer,
    DualStreamingTransducer, StreamingTransducer,
};
#[cfg(feature = "transducer")]
pub use transducer::SherpaWhisperEngine;

/// Factory to instantiate the Whisper SpeechRecognizer for Pass 2 (100% ONNX CPU/DirectML):
pub fn create_speech_recognizer(
    models_dir: &Path,
    tier: &str,
    language: &str,
) -> Result<Box<dyn SpeechRecognizer>> {
    if has_whisper_onnx_installed(models_dir, tier) {
        log::info!("[ASR] Found Whisper ONNX model for tier '{}', initializing engine...", tier);
        return create_sherpa_whisper(models_dir, tier, language);
    }

    anyhow::bail!(
        "Whisper ONNX model not found for tier '{}' in {:?}.\n\
         Run 'catetin' to download the required Whisper ONNX model.",
        tier, models_dir
    )
}

pub trait SpeechRecognizer: Send + Sync {
    /// Transcribe a slice of 16,000Hz mono f32 samples into text
    fn transcribe(&mut self, samples_16k_mono: &[f32]) -> Result<String>;

    /// Transcribe with an explicit rolling context prompt and return `(text, confidence)`
    /// where `confidence` is the mean token probability in `[0.0, 1.0]`.
    fn transcribe_with_context(
        &mut self,
        samples_16k_mono: &[f32],
        _context_prompt: Option<&str>,
    ) -> Result<(String, f32)> {
        let text = self.transcribe(samples_16k_mono)?;
        Ok((text, 0.90))
    }

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


/// Detect common Whisper hallucination artifacts generated on silence or music
pub fn is_hallucination(text: &str) -> bool {
    let lower = text.to_lowercase();
    // Check for music symbols anywhere in text
    if lower.contains('♪') || lower.contains('♫') || lower.contains('♩') || lower.contains('♬') {
        return true;
    }

    let trimmed = lower.trim().trim_matches(|c: char| !c.is_alphanumeric());
    if trimmed.is_empty() {
        return true;
    }

    // Check for common YouTube / social media outro hallucinations
    if lower.contains("subscribe")
        || lower.contains("subkrek")
        || lower.contains("langganan")
        || lower.contains("lonceng")
        || lower.contains("notifikasi")
    {
        if lower.contains("like")
            || lower.contains("share")
            || lower.contains("video")
            || lower.contains("channel")
            || lower.contains("kanal")
            || lower.contains("jangan lupa")
            || lower.contains("tombol")
            || lower.contains("kalian yang")
            || lower.contains("teman-teman")
            || lower.contains("gratis")
            || lower.contains("update")
            || lower.contains("konten")
        {
            return true;
        }
    }

    if lower.contains("jangan lupa")
        && (lower.contains("like")
            || lower.contains("share")
            || lower.contains("komen")
            || lower.contains("tonton")
            || lower.contains("subscribe"))
    {
        return true;
    }

    if lower.contains("like, comment")
        || lower.contains("like dan share")
        || lower.contains("like & share")
        || lower.contains("like and share")
        || lower.contains("like and subscribe")
        || lower.contains("kolom komentar")
        || lower.contains("link di deskripsi")
        || lower.contains("baca deskripsi")
        || lower.contains("terima kasih telah menonton")
        || lower.contains("terima kasih sudah menonton")
        || lower.contains("sampai jumpa di video")
        || lower.contains("tonton video")
        || lower.contains("video ini untuk kalian")
        || lower.contains("video ini bagi kalian")
        || lower.contains("kalian yang suka")
        || lower.contains("untuk kalian yang suka")
        || lower.contains("menonton video ini")
        || lower.contains("untuk menonton video")
        || lower.contains("tonton video ini")
        || lower.contains("video ini untuk menonton")
        || lower.contains("ingin menonton video")
    {
        return true;
    }

    const HALLUCINATIONS: &[&str] = &[
        "thank you for watching",
        "thanks for watching",
        "thank you for watching this video",
        "see you next time",
        "see you in the next video",
        "thank you for watching this video and see you next time",
        "please subscribe",
        "subscribe to my channel",
        "like and subscribe",
        "like, comment, and subscribe",
        "subtitles by",
        "translated by",
        "captioned by",
        "terima kasih telah menonton",
        "terima kasih sudah menonton",
        "sampai jumpa di video selanjutnya",
        "sampai jumpa lagi",
        "sampai jumpa",
        "menonton video ini",
        "untuk menonton video ini",
        "music",
        "musik",
        "instrumental",
        "suara musik",
        "alunan musik",
        "backsound",
        "lagu",
        "nada",
        "hening",
        "silence",
        "bye",
        "bye bye",
        "dadah",
        "jangan lupa subscribe",
        "jangan lupa like",
        "jangan lupa share",
        "jangan lupa like dan share",
        "subscribe dan like",
        "subscribe dan share",
    ];

    for &h in HALLUCINATIONS {
        if trimmed == h || trimmed.contains(h) {
            return true;
        }
    }

    if has_repetition_hallucination(trimmed) {
        return true;
    }

    false
}

/// Detects runaway autoregressive loop hallucinations in Whisper output.
/// Returns true if any 1-to-8 word n-gram repeats consecutively 3 or more times,
/// or if word diversity (unique words / total words) is abnormally low on long sentences.
pub fn has_repetition_hallucination(text: &str) -> bool {
    let words: Vec<&str> = text
        .split_whitespace()
        .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()))
        .filter(|w| !w.is_empty())
        .collect();
    if words.len() < 3 {
        return false;
    }

    // Check for 1-word through 8-word phrases repeating 3 or more times consecutively
    for n in 1..=8 {
        if words.len() < n * 3 {
            continue;
        }
        for i in 0..=(words.len() - n * 3) {
            let chunk1 = &words[i..i + n];
            let chunk2 = &words[i + n..i + 2 * n];
            let chunk3 = &words[i + 2 * n..i + 3 * n];
            if chunk1 == chunk2 && chunk2 == chunk3 {
                return true;
            }
        }
    }

    // Check word diversity (compression ratio approximation):
    // If an utterance has >= 14 words, but fewer than 35% are unique words
    if words.len() >= 14 {
        let mut unique = std::collections::HashSet::new();
        for w in &words {
            unique.insert(w.to_ascii_lowercase());
        }
        if (unique.len() as f32 / words.len() as f32) < 0.35 {
            return true;
        }
    }

    false
}

/// Clean up whitespace, strip accidental prompt echo, and ensure sentence capitalization.
pub fn format_utterance(input: &str, prompt: &str) -> String {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        return String::new();
    }
    let prompt_trim = prompt.trim();
    let working = if !prompt_trim.is_empty() && trimmed.len() > prompt_trim.len() {
        if trimmed
            .get(..prompt_trim.len())
            .map(|p| p.eq_ignore_ascii_case(prompt_trim))
            .unwrap_or(false)
        {
            trimmed[prompt_trim.len()..].trim()
        } else {
            trimmed
        }
    } else {
        trimmed
    };

    let mut chars = working.chars();
    match chars.next() {
        None => String::new(),
        Some(first) => first.to_uppercase().collect::<String>() + chars.as_str(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_repetition_hallucination_detection() {
        let repeated = "Saya ingin menonton video ini untuk menonton video ini dan menonton video ini untuk menonton video ini dan menonton video ini";
        assert!(has_repetition_hallucination(repeated));
        assert!(is_hallucination(repeated));

        let normal = "Selamat pagi rekan-rekan, hari ini kita akan membahas tentang kemajuan proyek terkini.";
        assert!(!has_repetition_hallucination(normal));
        assert!(!is_hallucination(normal));
    }
}

