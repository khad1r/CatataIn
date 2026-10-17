use wasm_bindgen::prelude::*;
use catetin_core::audio::AudioResampler;
use catetin_core::diarization::vad::VoiceActivityDetector;
use catetin_core::diarization::clustering::{OnlineSpeakerClusterer, cosine_similarity};
use catetin_core::events::SpeakerId;
use catetin_core::i18n::UiLanguage;

#[wasm_bindgen]
pub fn init_panic_hook() {
    #[cfg(feature = "console_error_panic_hook")]
    console_error_panic_hook::set_once();
}

/// WebAudio resampler converting arbitrary browser audio (e.g. 44.1kHz, 48kHz stereo/mono)
/// into standardized 16kHz mono audio for transcription & diarization.
#[wasm_bindgen]
pub struct WasmAudioResampler {
    inner: AudioResampler,
}

#[wasm_bindgen]
impl WasmAudioResampler {
    #[wasm_bindgen(constructor)]
    pub fn new(source_sample_rate: u32, source_channels: u16, target_sample_rate: u32) -> Self {
        Self {
            inner: AudioResampler::new(source_sample_rate, source_channels, target_sample_rate),
        }
    }

    #[wasm_bindgen]
    pub fn process_interleaved(&mut self, samples: &[f32]) -> Vec<f32> {
        self.inner.process_interleaved(samples)
    }

    #[wasm_bindgen]
    pub fn process_mono(&mut self, samples: &[f32]) -> Vec<f32> {
        self.inner.process_mono(samples)
    }
}

/// One-shot audio resampler convenience function for web clients.
#[wasm_bindgen]
pub fn resample_audio(samples: &[f32], source_rate: u32, source_channels: u16, target_rate: u32) -> Vec<f32> {
    let mut resampler = AudioResampler::new(source_rate, source_channels, target_rate);
    resampler.process_interleaved(samples)
}

/// Browser Voice Activity Detector (VAD).
/// Detects speech activity in 30ms frames using adaptive energy thresholding.
#[wasm_bindgen]
pub struct WasmVoiceActivityDetector {
    inner: VoiceActivityDetector,
}

#[wasm_bindgen]
impl WasmVoiceActivityDetector {
    #[wasm_bindgen(constructor)]
    pub fn new(sample_rate: u32, frame_size: usize) -> Self {
        Self {
            inner: VoiceActivityDetector::new(sample_rate, frame_size),
        }
    }

    #[wasm_bindgen]
    pub fn process_frame(&mut self, frame: &[f32]) -> bool {
        self.inner.process_frame(frame)
    }

    #[wasm_bindgen]
    pub fn frame_rms(&self, frame: &[f32]) -> f32 {
        self.inner.frame_rms(frame)
    }
}

/// Browser Real-time Speaker Clustering & Diarization engine.
/// Uses cosine similarity against centroid embeddings to track and cluster speakers.
#[wasm_bindgen]
pub struct WasmSpeakerClusterer {
    inner: OnlineSpeakerClusterer,
}

#[wasm_bindgen]
impl WasmSpeakerClusterer {
    #[wasm_bindgen(constructor)]
    pub fn new(similarity_threshold: f32, momentum: f32) -> Self {
        Self {
            inner: OnlineSpeakerClusterer::new(similarity_threshold, momentum),
        }
    }

    #[wasm_bindgen]
    pub fn identify_or_register(&mut self, embedding: &[f32]) -> String {
        let spk_id = self.inner.identify_or_register(embedding);
        match spk_id {
            SpeakerId::LocalUser => "You".to_string(),
            SpeakerId::Remote(id) => format!("Speaker {}", id),
            SpeakerId::InPerson(id) => format!("Speaker {}", id),
            SpeakerId::Unknown => "Unknown".to_string(),
        }
    }

    #[wasm_bindgen]
    pub fn speaker_count(&self) -> usize {
        self.inner.speaker_count()
    }

    #[wasm_bindgen]
    pub fn reset(&mut self) {
        self.inner.reset();
    }
}

/// Compute cosine similarity between two voiceprint vectors in JavaScript.
#[wasm_bindgen]
pub fn compute_cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    cosine_similarity(a, b)
}

/// Internationalization & UI localization for Catetin Web UI.
#[wasm_bindgen]
pub struct WasmI18n {
    lang: UiLanguage,
}

#[wasm_bindgen]
impl WasmI18n {
    #[wasm_bindgen(constructor)]
    pub fn new(code: &str) -> Self {
        Self {
            lang: UiLanguage::from_code(code),
        }
    }

    #[wasm_bindgen]
    pub fn get_text(&self, key: &str) -> String {
        self.lang.text(key)
    }

    #[wasm_bindgen]
    pub fn language_code(&self) -> String {
        self.lang.code().to_string()
    }

    #[wasm_bindgen]
    pub fn display_name(&self) -> String {
        self.lang.display_name()
    }
}

/// Returns list of available UI language codes (e.g. ["id", "en"]).
#[wasm_bindgen]
pub fn get_supported_languages() -> js_sys::Array {
    let array = js_sys::Array::new();
    array.push(&JsValue::from_str("id"));
    array.push(&JsValue::from_str("en"));
    array
}
