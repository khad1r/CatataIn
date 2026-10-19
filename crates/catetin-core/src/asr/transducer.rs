//! Streaming Transducer module for ultra-low latency Pass 1 captioning (< 50ms)
//! Powered by Zipformer / RNN-T Streaming Transducer (Sherpa-ONNX) with zero cloud dependencies.
//!
//! Models are loaded dynamically from external files on disk (never embedded into the binary),
//! maintaining lightweight executables and complete user flexibility.

#[allow(unused_imports)]
use anyhow::{Context, Result};
use std::path::Path;

/// Unified interface for streaming speech transducers (Zipformer, RNN-T, Kaldi)
pub trait StreamingTransducer: Send + Sync {
    /// Feed a chunk of 16kHz mono audio samples normalized to [-1.0, 1.0]
    fn accept_waveform(&mut self, samples: &[f32]);

    /// Decode available audio frames and return the latest partial hypothesis
    fn decode_step(&mut self) -> Option<String>;

    /// Retrieve the current text hypothesis accumulated so far
    fn get_partial_result(&self) -> String;

    /// Reset stream state for a new speech turn
    fn reset(&mut self);

    /// Check if the transducer detected an utterance endpoint (silence)
    fn is_endpoint(&self) -> bool;
}

/// Convert raw IPA phoneme output from Bookbot Zipformer model into clean, readable Indonesian text.
/// Matches the 35 tokens from Bookbot's Zipformer vocabulary (tokens.txt).
pub fn ipa_phonemes_to_indonesian(raw: &str) -> String {
    // If the text does not contain any Bookbot IPA markers, it is already standard text (e.g. Zipformer2 BPE)
    let has_ipa = raw.contains('|')
        || raw.contains('ɡ')
        || raw.contains('ə')
        || raw.contains('ŋ')
        || raw.contains('ɲ')
        || raw.contains('ʃ')
        || raw.contains('ʔ')
        || raw.contains("tʃ")
        || raw.contains("dʒ");

    if !has_ipa {
        return raw.trim().to_string();
    }

    let mut out = String::with_capacity(raw.len() + 16);
    let chars: Vec<char> = raw.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        // Multi-character phonemes first
        if i + 1 < chars.len() {
            // Voiceless postalveolar affricate /tʃ/ -> Indonesian 'c' (e.g. tʃompu -> compu)
            if chars[i] == 't' && chars[i + 1] == 'ʃ' {
                out.push('c');
                i += 2;
                continue;
            }
            // Voiced postalveolar affricate /dʒ/ -> Indonesian 'j' (e.g. dʒalan -> jalan)
            if chars[i] == 'd' && chars[i + 1] == 'ʒ' {
                out.push('j');
                i += 2;
                continue;
            }
        }

        match chars[i] {
            // Word boundary separator in Bookbot Zipformer is '|'
            '|' => out.push(' '),
            // Latin small script g (U+0261) -> standard Latin 'g'
            'ɡ' => out.push('g'),
            // Schwa / pepet (U+0259) -> 'e' (e.g. kəmaka -> kemaka, məra -> mera)
            'ə' => out.push('e'),
            // Velar nasal (U+014B) -> 'ng' (e.g. pənuŋɡuŋ -> penunggung)
            'ŋ' => out.push_str("ng"),
            // Palatal nasal (U+0272) -> 'ny' (e.g. baɲak -> banyak)
            'ɲ' => out.push_str("ny"),
            // Voiceless postalveolar fricative (U+0283) -> 'sy' (e.g. ʃarat -> syarat)
            'ʃ' => out.push_str("sy"),
            // Voiceless velar fricative -> 'kh'
            'x' => out.push_str("kh"),
            // IPA semivowel /j/ represents Indonesian letter 'y' (e.g. saja -> saya, ya)
            'j' => out.push('y'),
            // Glottal stop (U+0294): drop at word beginning (ʔalo -> alo), use 'k' at end/middle
            'ʔ' => {
                if out.is_empty() || out.ends_with(' ') {
                    // Initial glottal stop is silent in Indonesian spelling
                } else {
                    out.push('k');
                }
            }
            c => out.push(c),
        }
        i += 1;
    }

    // Collapse multiple consecutive spaces and trim
    let mut cleaned = String::with_capacity(out.len());
    let mut last_was_space = true;
    for c in out.chars() {
        if c == ' ' {
            if !last_was_space {
                cleaned.push(' ');
                last_was_space = true;
            }
        } else {
            cleaned.push(c);
            last_was_space = false;
        }
    }
    cleaned.trim().to_string()
}

/// Bilingual dual streaming transducer running both Indonesian and English Zipformers concurrently
pub struct DualStreamingTransducer {
    id_engine: Box<dyn StreamingTransducer>,
    en_engine: Box<dyn StreamingTransducer>,
    last_text: String,
}

impl DualStreamingTransducer {
    pub fn new(id_engine: Box<dyn StreamingTransducer>, en_engine: Box<dyn StreamingTransducer>) -> Self {
        Self {
            id_engine,
            en_engine,
            last_text: String::new(),
        }
    }
}

impl StreamingTransducer for DualStreamingTransducer {
    fn accept_waveform(&mut self, samples: &[f32]) {
        self.id_engine.accept_waveform(samples);
        self.en_engine.accept_waveform(samples);
    }

    fn decode_step(&mut self) -> Option<String> {
        let _ = self.id_engine.decode_step();
        let _ = self.en_engine.decode_step();

        let id_res = self.id_engine.get_partial_result();
        let en_res = self.en_engine.get_partial_result();

        let best = select_best_hypothesis(&id_res, &en_res);
        if best != self.last_text && !best.is_empty() {
            self.last_text = best.clone();
            Some(best)
        } else {
            None
        }
    }

    fn get_partial_result(&self) -> String {
        self.last_text.clone()
    }

    fn reset(&mut self) {
        self.id_engine.reset();
        self.en_engine.reset();
        self.last_text.clear();
    }

    fn is_endpoint(&self) -> bool {
        self.id_engine.is_endpoint() || self.en_engine.is_endpoint()
    }
}

/// Heuristic language & lexical scoring to select between Indonesian and English streaming hypotheses
pub fn select_best_hypothesis(id_res: &str, en_res: &str) -> String {
    let id_t = id_res.trim();
    let en_t = en_res.trim();

    if id_t.is_empty() {
        return en_t.to_string();
    }
    if en_t.is_empty() {
        return id_t.to_string();
    }
    if id_t.eq_ignore_ascii_case(en_t) {
        return id_t.to_string();
    }

    const EN_WORDS: &[&str] = &[
        "the", "is", "are", "you", "i", "my", "we", "they", "this", "that", "have", "with",
        "for", "and", "good", "morning", "hello", "can", "hear", "voice", "meeting", "okay",
        "yes", "no", "what", "how", "when", "where", "why", "who", "all", "so", "to", "in", "on",
    ];

    const ID_WORDS: &[&str] = &[
        "yang", "dan", "di", "ini", "itu", "saya", "anda", "kita", "kami", "dengan", "untuk",
        "apakah", "selamat", "pagi", "suara", "terdengar", "rapat", "bisa", "ada", "tidak", "ya",
        "apa", "bagaimana", "kapan", "dimana", "mengapa", "siapa", "semua", "jadi", "ke",
    ];

    let id_lower = id_t.to_lowercase();
    let en_lower = en_t.to_lowercase();

    let id_words: Vec<&str> = id_lower.split_whitespace().collect();
    let en_words: Vec<&str> = en_lower.split_whitespace().collect();

    let id_score = id_words.iter().filter(|w| ID_WORDS.contains(w)).count();
    let en_score = en_words.iter().filter(|w| EN_WORDS.contains(w)).count();

    if en_score > id_score {
        en_t.to_string()
    } else if id_score > en_score {
        id_t.to_string()
    } else if en_words.len() > id_words.len() && en_score > 0 {
        en_t.to_string()
    } else {
        id_t.to_string()
    }
}

#[cfg(feature = "transducer")]
mod sherpa_backend {
    use super::*;
    use anyhow::{Context, Result};
    use libloading::{Library, Symbol};
    use std::ffi::{CStr, CString};
    use std::path::PathBuf;
    use std::sync::Arc;

    // --- C ABI Structure Definitions (matching sherpa-onnx c-api.h) ---

    #[repr(C)]
    #[derive(Default)]
    pub struct SherpaOnnxOnlineTransducerModelConfig {
        pub encoder: *const std::os::raw::c_char,
        pub decoder: *const std::os::raw::c_char,
        pub joiner: *const std::os::raw::c_char,
    }

    #[repr(C)]
    #[derive(Default)]
    pub struct SherpaOnnxOnlineParaformerModelConfig {
        pub encoder: *const std::os::raw::c_char,
        pub decoder: *const std::os::raw::c_char,
    }

    #[repr(C)]
    #[derive(Default)]
    pub struct SherpaOnnxOnlineZipformer2CtcModelConfig {
        pub model: *const std::os::raw::c_char,
    }

    #[repr(C)]
    #[derive(Default)]
    pub struct SherpaOnnxOnlineNemoCtcModelConfig {
        pub model: *const std::os::raw::c_char,
    }

    #[repr(C)]
    #[derive(Default)]
    pub struct SherpaOnnxOnlineToneCtcModelConfig {
        pub model: *const std::os::raw::c_char,
    }

    #[repr(C)]
    #[derive(Default)]
    pub struct SherpaOnnxOnlineModelConfig {
        pub transducer: SherpaOnnxOnlineTransducerModelConfig,
        pub paraformer: SherpaOnnxOnlineParaformerModelConfig,
        pub zipformer2_ctc: SherpaOnnxOnlineZipformer2CtcModelConfig,
        pub tokens: *const std::os::raw::c_char,
        pub num_threads: i32,
        pub provider: *const std::os::raw::c_char,
        pub debug: i32,
        pub model_type: *const std::os::raw::c_char,
        pub modeling_unit: *const std::os::raw::c_char,
        pub bpe_vocab: *const std::os::raw::c_char,
        pub tokens_buf: *const std::os::raw::c_char,
        pub tokens_buf_size: i32,
        pub nemo_ctc: SherpaOnnxOnlineNemoCtcModelConfig,
        pub t_one_ctc: SherpaOnnxOnlineToneCtcModelConfig,
    }

    #[repr(C)]
    #[derive(Default)]
    pub struct SherpaOnnxFeatureConfig {
        pub sample_rate: i32,
        pub feature_dim: i32,
    }

    #[repr(C)]
    #[derive(Default)]
    pub struct SherpaOnnxOnlineCtcFstDecoderConfig {
        pub graph: *const std::os::raw::c_char,
        pub max_active: i32,
    }

    #[repr(C)]
    #[derive(Default)]
    pub struct SherpaOnnxHomophoneReplacerConfig {
        pub dict_dir: *const std::os::raw::c_char,
        pub lexicon: *const std::os::raw::c_char,
        pub rule_fsts: *const std::os::raw::c_char,
    }

    #[repr(C)]
    #[derive(Default)]
    pub struct SherpaOnnxOnlineRecognizerConfig {
        pub feat_config: SherpaOnnxFeatureConfig,
        pub model_config: SherpaOnnxOnlineModelConfig,
        pub decoding_method: *const std::os::raw::c_char,
        pub max_active_paths: i32,
        pub enable_endpoint: i32,
        pub rule1_min_trailing_silence: f32,
        pub rule2_min_trailing_silence: f32,
        pub rule3_min_utterance_length: f32,
        pub hotwords_file: *const std::os::raw::c_char,
        pub hotwords_score: f32,
        pub ctc_fst_decoder_config: SherpaOnnxOnlineCtcFstDecoderConfig,
        pub rule_fsts: *const std::os::raw::c_char,
        pub rule_fars: *const std::os::raw::c_char,
        pub blank_penalty: f32,
        pub hotwords_buf: *const std::os::raw::c_char,
        pub hotwords_buf_size: i32,
        pub hr: SherpaOnnxHomophoneReplacerConfig,
    }

    #[repr(C)]
    pub struct SherpaOnnxOnlineRecognizerResult {
        pub text: *const std::os::raw::c_char,
        pub tokens: *const std::os::raw::c_char,
        pub tokens_arr: *const *const std::os::raw::c_char,
        pub timestamps: *mut f32,
        pub count: i32,
        pub json: *const std::os::raw::c_char,
    }

    // --- C ABI Structure Definitions for Offline Recognizer (Whisper ONNX) ---

    #[repr(C)]
    #[derive(Default)]
    pub struct SherpaOnnxOfflineTransducerModelConfig {
        pub encoder: *const std::os::raw::c_char,
        pub decoder: *const std::os::raw::c_char,
        pub joiner: *const std::os::raw::c_char,
    }

    #[repr(C)]
    #[derive(Default)]
    pub struct SherpaOnnxOfflineParaformerModelConfig {
        pub model: *const std::os::raw::c_char,
    }

    #[repr(C)]
    #[derive(Default)]
    pub struct SherpaOnnxOfflineNemoEncDecCtcModelConfig {
        pub model: *const std::os::raw::c_char,
    }

    #[repr(C)]
    #[derive(Default)]
    pub struct SherpaOnnxOfflineWhisperModelConfig {
        pub encoder: *const std::os::raw::c_char,
        pub decoder: *const std::os::raw::c_char,
        pub language: *const std::os::raw::c_char,
        pub task: *const std::os::raw::c_char,
        pub tail_paddings: i32,
        pub enable_token_timestamps: i32,
        pub enable_segment_timestamps: i32,
    }

    #[repr(C)]
    #[derive(Default)]
    pub struct SherpaOnnxOfflineTdnnModelConfig {
        pub model: *const std::os::raw::c_char,
    }

    #[repr(C)]
    #[derive(Default)]
    pub struct SherpaOnnxOfflineSenseVoiceModelConfig {
        pub model: *const std::os::raw::c_char,
        pub language: *const std::os::raw::c_char,
        pub use_itn: i32,
    }

    #[repr(C)]
    #[derive(Default)]
    pub struct SherpaOnnxOfflineMoonshineModelConfig {
        pub preprocessor: *const std::os::raw::c_char,
        pub encoder: *const std::os::raw::c_char,
        pub uncached_decoder: *const std::os::raw::c_char,
        pub cached_decoder: *const std::os::raw::c_char,
        pub merged_decoder: *const std::os::raw::c_char,
    }

    #[repr(C)]
    #[derive(Default)]
    pub struct SherpaOnnxOfflineFireRedAsrModelConfig {
        pub encoder: *const std::os::raw::c_char,
        pub decoder: *const std::os::raw::c_char,
    }

    #[repr(C)]
    #[derive(Default)]
    pub struct SherpaOnnxOfflineDolphinModelConfig {
        pub model: *const std::os::raw::c_char,
    }

    #[repr(C)]
    #[derive(Default)]
    pub struct SherpaOnnxOfflineZipformerCtcModelConfig {
        pub model: *const std::os::raw::c_char,
    }

    #[repr(C)]
    #[derive(Default)]
    pub struct SherpaOnnxOfflineCanaryModelConfig {
        pub encoder: *const std::os::raw::c_char,
        pub decoder: *const std::os::raw::c_char,
        pub src_lang: *const std::os::raw::c_char,
        pub tgt_lang: *const std::os::raw::c_char,
        pub use_pnc: i32,
    }

    #[repr(C)]
    #[derive(Default)]
    pub struct SherpaOnnxOfflineWenetCtcModelConfig {
        pub model: *const std::os::raw::c_char,
    }

    #[repr(C)]
    #[derive(Default)]
    pub struct SherpaOnnxOfflineOmnilingualAsrCtcModelConfig {
        pub model: *const std::os::raw::c_char,
    }

    #[repr(C)]
    #[derive(Default)]
    pub struct SherpaOnnxOfflineMedAsrCtcModelConfig {
        pub model: *const std::os::raw::c_char,
    }

    #[repr(C)]
    #[derive(Default)]
    pub struct SherpaOnnxOfflineFunASRNanoModelConfig {
        pub encoder_adaptor: *const std::os::raw::c_char,
        pub llm: *const std::os::raw::c_char,
        pub embedding: *const std::os::raw::c_char,
        pub tokenizer: *const std::os::raw::c_char,
        pub system_prompt: *const std::os::raw::c_char,
        pub user_prompt: *const std::os::raw::c_char,
        pub max_new_tokens: i32,
        pub temperature: f32,
        pub top_p: f32,
        pub seed: i32,
        pub language: *const std::os::raw::c_char,
        pub itn: i32,
        pub hotwords: *const std::os::raw::c_char,
    }

    #[repr(C)]
    #[derive(Default)]
    pub struct SherpaOnnxOfflineFireRedAsrCtcModelConfig {
        pub model: *const std::os::raw::c_char,
    }

    #[repr(C)]
    #[derive(Default)]
    pub struct SherpaOnnxOfflineQwen3ASRModelConfig {
        pub conv_frontend: *const std::os::raw::c_char,
        pub encoder: *const std::os::raw::c_char,
        pub decoder: *const std::os::raw::c_char,
        pub tokenizer: *const std::os::raw::c_char,
        pub max_total_len: i32,
        pub max_new_tokens: i32,
        pub temperature: f32,
        pub top_p: f32,
        pub seed: i32,
        pub hotwords: *const std::os::raw::c_char,
    }

    #[repr(C)]
    #[derive(Default)]
    pub struct SherpaOnnxOfflineCohereTranscribeModelConfig {
        pub encoder: *const std::os::raw::c_char,
        pub decoder: *const std::os::raw::c_char,
        pub language: *const std::os::raw::c_char,
        pub use_punct: i32,
        pub use_itn: i32,
    }

    #[repr(C)]
    #[derive(Default)]
    pub struct SherpaOnnxOfflineModelConfig {
        pub transducer: SherpaOnnxOfflineTransducerModelConfig,
        pub paraformer: SherpaOnnxOfflineParaformerModelConfig,
        pub nemo_ctc: SherpaOnnxOfflineNemoEncDecCtcModelConfig,
        pub whisper: SherpaOnnxOfflineWhisperModelConfig,
        pub tdnn: SherpaOnnxOfflineTdnnModelConfig,
        pub tokens: *const std::os::raw::c_char,
        pub num_threads: i32,
        pub debug: i32,
        pub provider: *const std::os::raw::c_char,
        pub model_type: *const std::os::raw::c_char,
        pub modeling_unit: *const std::os::raw::c_char,
        pub bpe_vocab: *const std::os::raw::c_char,
        pub telespeech_ctc: *const std::os::raw::c_char,
        pub sense_voice: SherpaOnnxOfflineSenseVoiceModelConfig,
        pub moonshine: SherpaOnnxOfflineMoonshineModelConfig,
        pub fire_red_asr: SherpaOnnxOfflineFireRedAsrModelConfig,
        pub dolphin: SherpaOnnxOfflineDolphinModelConfig,
        pub zipformer_ctc: SherpaOnnxOfflineZipformerCtcModelConfig,
        pub canary: SherpaOnnxOfflineCanaryModelConfig,
        pub wenet_ctc: SherpaOnnxOfflineWenetCtcModelConfig,
        pub omnilingual: SherpaOnnxOfflineOmnilingualAsrCtcModelConfig,
        pub medasr: SherpaOnnxOfflineMedAsrCtcModelConfig,
        pub funasr_nano: SherpaOnnxOfflineFunASRNanoModelConfig,
        pub fire_red_asr_ctc: SherpaOnnxOfflineFireRedAsrCtcModelConfig,
        pub qwen3_asr: SherpaOnnxOfflineQwen3ASRModelConfig,
        pub cohere_transcribe: SherpaOnnxOfflineCohereTranscribeModelConfig,
    }

    #[repr(C)]
    #[derive(Default)]
    pub struct SherpaOnnxOfflineLMConfig {
        pub model: *const std::os::raw::c_char,
        pub scale: f32,
    }

    #[repr(C)]
    #[derive(Default)]
    pub struct SherpaOnnxOfflineCtcFstDecoderConfig {
        pub graph: *const std::os::raw::c_char,
        pub max_active: i32,
    }

    #[repr(C)]
    #[derive(Default)]
    pub struct SherpaOnnxOfflineRecognizerConfig {
        pub feat_config: SherpaOnnxFeatureConfig,
        pub model_config: SherpaOnnxOfflineModelConfig,
        pub lm_config: SherpaOnnxOfflineLMConfig,
        pub decoding_method: *const std::os::raw::c_char,
        pub max_active_paths: i32,
        pub hotwords_file: *const std::os::raw::c_char,
        pub hotwords_score: f32,
        pub rule_fsts: *const std::os::raw::c_char,
        pub rule_fars: *const std::os::raw::c_char,
        pub blank_penalty: f32,
        pub hr: SherpaOnnxHomophoneReplacerConfig,
        pub ctc_fst_decoder_config: SherpaOnnxOfflineCtcFstDecoderConfig,
    }

    #[repr(C)]
    pub struct SherpaOnnxOfflineRecognizerResult {
        pub text: *const std::os::raw::c_char,
        pub timestamps: *mut f32,
        pub count: i32,
        pub tokens: *const std::os::raw::c_char,
        pub tokens_arr: *const *const std::os::raw::c_char,
        pub json: *const std::os::raw::c_char,
        pub lang: *const std::os::raw::c_char,
        pub emotion: *const std::os::raw::c_char,
        pub event: *const std::os::raw::c_char,
        pub durations: *mut f32,
        pub ys_log_probs: *mut f32,
        pub segment_timestamps: *const f32,
        pub segment_durations: *const f32,
        pub segment_texts: *const std::os::raw::c_char,
        pub segment_texts_arr: *const *const std::os::raw::c_char,
        pub segment_count: i32,
    }

    // --- Dynamic Function Pointer Table ---

    pub struct SherpaApi {
        _lib: Library,
        create_recognizer: unsafe extern "C" fn(*const SherpaOnnxOnlineRecognizerConfig) -> *const std::ffi::c_void,
        destroy_recognizer: unsafe extern "C" fn(*const std::ffi::c_void),
        create_stream: unsafe extern "C" fn(*const std::ffi::c_void) -> *const std::ffi::c_void,
        destroy_stream: unsafe extern "C" fn(*const std::ffi::c_void),
        accept_waveform: unsafe extern "C" fn(*const std::ffi::c_void, i32, *const f32, i32),
        is_ready: unsafe extern "C" fn(*const std::ffi::c_void, *const std::ffi::c_void) -> i32,
        decode_stream: unsafe extern "C" fn(*const std::ffi::c_void, *const std::ffi::c_void),
        get_result: unsafe extern "C" fn(*const std::ffi::c_void, *const std::ffi::c_void) -> *const SherpaOnnxOnlineRecognizerResult,
        destroy_result: unsafe extern "C" fn(*const SherpaOnnxOnlineRecognizerResult),
        reset_stream: unsafe extern "C" fn(*const std::ffi::c_void, *const std::ffi::c_void),
        is_endpoint: unsafe extern "C" fn(*const std::ffi::c_void, *const std::ffi::c_void) -> i32,

        // Offline Recognizer API (Whisper ONNX)
        create_offline_recognizer: unsafe extern "C" fn(*const SherpaOnnxOfflineRecognizerConfig) -> *const std::ffi::c_void,
        destroy_offline_recognizer: unsafe extern "C" fn(*const std::ffi::c_void),
        create_offline_stream: unsafe extern "C" fn(*const std::ffi::c_void) -> *const std::ffi::c_void,
        destroy_offline_stream: unsafe extern "C" fn(*const std::ffi::c_void),
        accept_waveform_offline: unsafe extern "C" fn(*const std::ffi::c_void, i32, *const f32, i32),
        decode_offline_stream: unsafe extern "C" fn(*const std::ffi::c_void, *const std::ffi::c_void),
        get_offline_stream_result: unsafe extern "C" fn(*const std::ffi::c_void) -> *const SherpaOnnxOfflineRecognizerResult,
        destroy_offline_recognizer_result: unsafe extern "C" fn(*const SherpaOnnxOfflineRecognizerResult),
    }

    impl SherpaApi {
        pub fn load<P: AsRef<Path>>(lib_path: P) -> Result<Arc<Self>> {
            unsafe {
                #[cfg(windows)]
                {
                    if let Some(parent) = lib_path.as_ref().parent() {
                        use std::os::windows::ffi::OsStrExt;
                        let wide: Vec<u16> = parent.as_os_str().encode_wide().chain(std::iter::once(0)).collect();
                        extern "system" {
                            fn SetDllDirectoryW(lpPathName: *const u16) -> i32;
                        }
                        let _ = SetDllDirectoryW(wide.as_ptr());
                    }
                }

                let lib = Library::new(lib_path.as_ref())
                    .with_context(|| format!("Failed to load dynamic library at {:?}", lib_path.as_ref()))?;

                let create_recognizer: Symbol<unsafe extern "C" fn(*const SherpaOnnxOnlineRecognizerConfig) -> *const std::ffi::c_void> =
                    lib.get(b"SherpaOnnxCreateOnlineRecognizer\0")?;
                let destroy_recognizer: Symbol<unsafe extern "C" fn(*const std::ffi::c_void)> =
                    lib.get(b"SherpaOnnxDestroyOnlineRecognizer\0")?;
                let create_stream: Symbol<unsafe extern "C" fn(*const std::ffi::c_void) -> *const std::ffi::c_void> =
                    lib.get(b"SherpaOnnxCreateOnlineStream\0")?;
                let destroy_stream: Symbol<unsafe extern "C" fn(*const std::ffi::c_void)> =
                    lib.get(b"SherpaOnnxDestroyOnlineStream\0")?;
                let accept_waveform: Symbol<unsafe extern "C" fn(*const std::ffi::c_void, i32, *const f32, i32)> =
                    lib.get(b"SherpaOnnxOnlineStreamAcceptWaveform\0")?;
                let is_ready: Symbol<unsafe extern "C" fn(*const std::ffi::c_void, *const std::ffi::c_void) -> i32> =
                    lib.get(b"SherpaOnnxIsOnlineStreamReady\0")?;
                let decode_stream: Symbol<unsafe extern "C" fn(*const std::ffi::c_void, *const std::ffi::c_void)> =
                    lib.get(b"SherpaOnnxDecodeOnlineStream\0")?;
                let get_result: Symbol<unsafe extern "C" fn(*const std::ffi::c_void, *const std::ffi::c_void) -> *const SherpaOnnxOnlineRecognizerResult> =
                    lib.get(b"SherpaOnnxGetOnlineStreamResult\0")?;
                let destroy_result: Symbol<unsafe extern "C" fn(*const SherpaOnnxOnlineRecognizerResult)> =
                    lib.get(b"SherpaOnnxDestroyOnlineRecognizerResult\0")?;
                let reset_stream: Symbol<unsafe extern "C" fn(*const std::ffi::c_void, *const std::ffi::c_void)> =
                    lib.get(b"SherpaOnnxOnlineStreamReset\0")?;
                let is_endpoint: Symbol<unsafe extern "C" fn(*const std::ffi::c_void, *const std::ffi::c_void) -> i32> =
                    lib.get(b"SherpaOnnxOnlineStreamIsEndpoint\0")?;

                let create_offline_recognizer: Symbol<unsafe extern "C" fn(*const SherpaOnnxOfflineRecognizerConfig) -> *const std::ffi::c_void> =
                    lib.get(b"SherpaOnnxCreateOfflineRecognizer\0")?;
                let destroy_offline_recognizer: Symbol<unsafe extern "C" fn(*const std::ffi::c_void)> =
                    lib.get(b"SherpaOnnxDestroyOfflineRecognizer\0")?;
                let create_offline_stream: Symbol<unsafe extern "C" fn(*const std::ffi::c_void) -> *const std::ffi::c_void> =
                    lib.get(b"SherpaOnnxCreateOfflineStream\0")?;
                let destroy_offline_stream: Symbol<unsafe extern "C" fn(*const std::ffi::c_void)> =
                    lib.get(b"SherpaOnnxDestroyOfflineStream\0")?;
                let accept_waveform_offline: Symbol<unsafe extern "C" fn(*const std::ffi::c_void, i32, *const f32, i32)> =
                    lib.get(b"SherpaOnnxAcceptWaveformOffline\0")?;
                let decode_offline_stream: Symbol<unsafe extern "C" fn(*const std::ffi::c_void, *const std::ffi::c_void)> =
                    lib.get(b"SherpaOnnxDecodeOfflineStream\0")?;
                let get_offline_stream_result: Symbol<unsafe extern "C" fn(*const std::ffi::c_void) -> *const SherpaOnnxOfflineRecognizerResult> =
                    lib.get(b"SherpaOnnxGetOfflineStreamResult\0")?;
                let destroy_offline_recognizer_result: Symbol<unsafe extern "C" fn(*const SherpaOnnxOfflineRecognizerResult)> =
                    lib.get(b"SherpaOnnxDestroyOfflineRecognizerResult\0")?;

                Ok(Arc::new(Self {
                    create_recognizer: *create_recognizer,
                    destroy_recognizer: *destroy_recognizer,
                    create_stream: *create_stream,
                    destroy_stream: *destroy_stream,
                    accept_waveform: *accept_waveform,
                    is_ready: *is_ready,
                    decode_stream: *decode_stream,
                    get_result: *get_result,
                    destroy_result: *destroy_result,
                    reset_stream: *reset_stream,
                    is_endpoint: *is_endpoint,
                    create_offline_recognizer: *create_offline_recognizer,
                    destroy_offline_recognizer: *destroy_offline_recognizer,
                    create_offline_stream: *create_offline_stream,
                    destroy_offline_stream: *destroy_offline_stream,
                    accept_waveform_offline: *accept_waveform_offline,
                    decode_offline_stream: *decode_offline_stream,
                    get_offline_stream_result: *get_offline_stream_result,
                    destroy_offline_recognizer_result: *destroy_offline_recognizer_result,
                    _lib: lib,
                }))
            }
        }
    }

    /// Safe RAII wrapper for Sherpa-ONNX streaming recognizer
    pub struct SherpaTransducerEngine {
        api: Arc<SherpaApi>,
        recognizer: *const std::ffi::c_void,
        stream: *const std::ffi::c_void,
        last_text: String,
    }

    unsafe impl Send for SherpaTransducerEngine {}
    unsafe impl Sync for SherpaTransducerEngine {}

    impl Drop for SherpaTransducerEngine {
        fn drop(&mut self) {
            unsafe {
                if !self.stream.is_null() {
                    (self.api.destroy_stream)(self.stream);
                }
                if !self.recognizer.is_null() {
                    (self.api.destroy_recognizer)(self.recognizer);
                }
            }
        }
    }

    impl SherpaTransducerEngine {
        pub fn new(
            api: Arc<SherpaApi>,
            encoder_path: &Path,
            decoder_path: &Path,
            joiner_path: &Path,
            tokens_path: &Path,
        ) -> Result<Self> {
            let enc_c = CString::new(encoder_path.to_str().context("encoder path")?)?;
            let dec_c = CString::new(decoder_path.to_str().context("decoder path")?)?;
            let join_c = CString::new(joiner_path.to_str().context("joiner path")?)?;
            let tok_c = CString::new(tokens_path.to_str().context("tokens path")?)?;
            let method_c = CString::new("greedy_search")?;

            let mut config = SherpaOnnxOnlineRecognizerConfig::default();
            config.feat_config.sample_rate = 16000;
            config.feat_config.feature_dim = 80;

            config.model_config.transducer.encoder = enc_c.as_ptr();
            config.model_config.transducer.decoder = dec_c.as_ptr();
            config.model_config.transducer.joiner = join_c.as_ptr();
            config.model_config.tokens = tok_c.as_ptr();
            config.model_config.num_threads = 2;
            config.model_config.debug = 0;

            config.decoding_method = method_c.as_ptr();
            config.max_active_paths = 4;
            config.enable_endpoint = 1;
            config.rule1_min_trailing_silence = 2.0;
            config.rule2_min_trailing_silence = 1.0;
            config.rule3_min_utterance_length = 20.0;

            let zipformer2_c = CString::new("zipformer2")?;
            let provider_c = CString::new("cpu")?;
            config.model_config.provider = provider_c.as_ptr();

            // 1. Auto-detect model type from ONNX metadata (std::ptr::null())
            config.model_config.model_type = std::ptr::null();
            let mut recognizer = unsafe { (api.create_recognizer)(&config) };

            // 2. If null model_type failed, retry with explicit "zipformer2"
            if recognizer.is_null() {
                config.model_config.model_type = zipformer2_c.as_ptr();
                recognizer = unsafe { (api.create_recognizer)(&config) };
            }

            if recognizer.is_null() {
                anyhow::bail!("SherpaOnnxCreateOnlineRecognizer returned null");
            }

            let stream = unsafe { (api.create_stream)(recognizer) };
            if stream.is_null() {
                unsafe { (api.destroy_recognizer)(recognizer) };
                anyhow::bail!("SherpaOnnxCreateOnlineStream returned null");
            }

            Ok(Self {
                api,
                recognizer,
                stream,
                last_text: String::new(),
            })
        }
    }

    impl StreamingTransducer for SherpaTransducerEngine {
        fn accept_waveform(&mut self, samples: &[f32]) {
            if samples.is_empty() {
                return;
            }
            unsafe {
                (self.api.accept_waveform)(
                    self.stream,
                    16000,
                    samples.as_ptr(),
                    samples.len() as i32,
                );
            }
        }

        fn decode_step(&mut self) -> Option<String> {
            unsafe {
                let mut decoded = false;
                while (self.api.is_ready)(self.recognizer, self.stream) != 0 {
                    (self.api.decode_stream)(self.recognizer, self.stream);
                    decoded = true;
                }

                if decoded {
                    let result_ptr = (self.api.get_result)(self.recognizer, self.stream);
                    if !result_ptr.is_null() {
                        let text_ptr = (*result_ptr).text;
                        let text = if !text_ptr.is_null() {
                            let raw = CStr::from_ptr(text_ptr).to_string_lossy();
                            let clean = raw.replace(' ', " ").replace('▁', " ");
                            ipa_phonemes_to_indonesian(&clean)
                        } else {
                            String::new()
                        };
                        (self.api.destroy_result)(result_ptr);

                        if text != self.last_text && !text.is_empty() {
                            self.last_text = text.clone();
                            return Some(text);
                        }
                    }
                }
            }
            None
        }

        fn get_partial_result(&self) -> String {
            self.last_text.clone()
        }

        fn reset(&mut self) {
            unsafe {
                (self.api.reset_stream)(self.recognizer, self.stream);
            }
            self.last_text.clear();
        }

        fn is_endpoint(&self) -> bool {
            unsafe { (self.api.is_endpoint)(self.recognizer, self.stream) != 0 }
        }
    }

    /// Safe RAII wrapper for Sherpa-ONNX offline Whisper recognizer with hardware acceleration (DirectML / CUDA)
    pub struct SherpaWhisperEngine {
        api: Arc<SherpaApi>,
        recognizer: *const std::ffi::c_void,
        language: String,
        active_provider: String,
    }

    unsafe impl Send for SherpaWhisperEngine {}
    unsafe impl Sync for SherpaWhisperEngine {}

    impl Drop for SherpaWhisperEngine {
        fn drop(&mut self) {
            unsafe {
                if !self.recognizer.is_null() {
                    (self.api.destroy_offline_recognizer)(self.recognizer);
                }
            }
        }
    }

    impl SherpaWhisperEngine {
        pub fn new(
            api: Arc<SherpaApi>,
            encoder_path: &Path,
            decoder_path: &Path,
            tokens_path: &Path,
            language: &str,
        ) -> Result<Self> {
            let enc_c = CString::new(encoder_path.to_str().context("encoder path")?)?;
            let dec_c = CString::new(decoder_path.to_str().context("decoder path")?)?;
            let tok_c = CString::new(tokens_path.to_str().context("tokens path")?)?;
            let lang_c = if language.is_empty() || language.eq_ignore_ascii_case("auto") {
                CString::new("")?
            } else {
                CString::new(language)?
            };
            let task_c = CString::new("transcribe")?;
            let method_c = CString::new("greedy_search")?;

            let mut config = SherpaOnnxOfflineRecognizerConfig::default();
            config.feat_config.sample_rate = 16000;
            config.feat_config.feature_dim = 80;

            config.model_config.whisper.encoder = enc_c.as_ptr();
            config.model_config.whisper.decoder = dec_c.as_ptr();
            config.model_config.whisper.language = lang_c.as_ptr();
            config.model_config.whisper.task = task_c.as_ptr();
            config.model_config.whisper.tail_paddings = -1;
            config.model_config.whisper.enable_segment_timestamps = 0;

            config.model_config.tokens = tok_c.as_ptr();
            config.model_config.num_threads = 4;
            config.model_config.debug = 0;
            config.decoding_method = method_c.as_ptr();

            let mut active_provider = "cpu".to_string();
            let mut recognizer: *const std::ffi::c_void = std::ptr::null();

            let directml_c = CString::new("directml")?;
            let cuda_c = CString::new("cuda")?;
            let cpu_c = CString::new("cpu")?;

            // 1. Try DirectML on Windows (DirectX 12 GPU acceleration for Intel, AMD, NVIDIA)
            #[cfg(target_os = "windows")]
            {
                config.model_config.provider = directml_c.as_ptr();
                let rec = unsafe { (api.create_offline_recognizer)(&config) };
                if !rec.is_null() {
                    recognizer = rec;
                    active_provider = "directml (GPU)".to_string();
                }
            }

            // 2. Try CUDA (NVIDIA GPU)
            if recognizer.is_null() {
                config.model_config.provider = cuda_c.as_ptr();
                let rec = unsafe { (api.create_offline_recognizer)(&config) };
                if !rec.is_null() {
                    recognizer = rec;
                    active_provider = "cuda (GPU)".to_string();
                }
            }

            // 3. Fallback to CPU
            if recognizer.is_null() {
                config.model_config.provider = cpu_c.as_ptr();
                recognizer = unsafe { (api.create_offline_recognizer)(&config) };
                active_provider = "cpu".to_string();
            }

            if recognizer.is_null() {
                anyhow::bail!("SherpaOnnxCreateOfflineRecognizer returned null for Whisper ONNX");
            }

            Ok(Self {
                api,
                recognizer,
                language: language.to_string(),
                active_provider,
            })
        }

        pub fn active_provider(&self) -> &str {
            &self.active_provider
        }
    }

    impl crate::asr::SpeechRecognizer for SherpaWhisperEngine {
        fn transcribe(&mut self, samples: &[f32]) -> Result<String> {
            let (text, _) = self.transcribe_with_context(samples, None)?;
            Ok(text)
        }

        fn transcribe_with_context(
            &mut self,
            samples: &[f32],
            context_prompt: Option<&str>,
        ) -> Result<(String, f32)> {
            if samples.len() < 4_000 {
                return Ok((String::new(), 0.0));
            }

            let capped = if samples.len() > 144_000 {
                &samples[..144_000]
            } else {
                samples
            };

            let peak = capped
                .iter()
                .fold(0.0f32, |acc, &x| acc.max(x.abs()));

            if peak < 0.015 {
                return Ok((String::new(), 0.0));
            }

            let gain = if peak < 0.40 {
                (0.55 / peak).min(8.0)
            } else {
                1.0
            };

            let lead_pad = 1_280usize;
            let trail_pad = 1_280usize;
            let mut prepared = Vec::with_capacity(capped.len() + lead_pad + trail_pad);
            prepared.resize(lead_pad, 0.0);
            for &s in capped {
                prepared.push((s * gain).clamp(-1.0, 1.0));
            }
            prepared.resize(prepared.len() + trail_pad, 0.0);

            unsafe {
                let stream = (self.api.create_offline_stream)(self.recognizer);
                if stream.is_null() {
                    anyhow::bail!("Failed to create offline stream for Whisper ONNX");
                }

                (self.api.accept_waveform_offline)(
                    stream,
                    16000,
                    prepared.as_ptr(),
                    prepared.len() as i32,
                );

                (self.api.decode_offline_stream)(self.recognizer, stream);

                let result_ptr = (self.api.get_offline_stream_result)(stream);
                let raw_text = if !result_ptr.is_null() {
                    let text_ptr = (*result_ptr).text;
                    let t = if !text_ptr.is_null() {
                        CStr::from_ptr(text_ptr).to_string_lossy().to_string()
                    } else {
                        String::new()
                    };
                    (self.api.destroy_offline_recognizer_result)(result_ptr);
                    t
                } else {
                    String::new()
                };

                (self.api.destroy_offline_stream)(stream);

                let clean = raw_text.trim();
                if clean.is_empty()
                    || clean.starts_with('[')
                    || clean.starts_with('(')
                    || clean.starts_with('*')
                    || clean.contains('♪')
                    || clean.contains('♫')
                    || clean.contains('♩')
                    || clean.contains('♬')
                    || crate::asr::is_hallucination(clean)
                {
                    return Ok((String::new(), 0.0));
                }

                let prompt_str = context_prompt.unwrap_or("");
                let formatted = crate::asr::format_utterance(clean, prompt_str);
                if crate::asr::is_hallucination(&formatted) || crate::asr::has_repetition_hallucination(&formatted) {
                    return Ok((String::new(), 0.0));
                }

                Ok((formatted, 0.95))
            }
        }

        fn language(&self) -> &str {
            &self.language
        }
    }

    #[cfg(target_os = "windows")]
    pub static EMBEDDED_SHERPA_DLL: &[u8] = include_bytes!("../../native/win-x64/sherpa-onnx-c-api.dll");
    #[cfg(target_os = "windows")]
    pub static EMBEDDED_ONNXRUNTIME_DLL: &[u8] = include_bytes!("../../native/win-x64/onnxruntime.dll");

    #[cfg(not(target_os = "windows"))]
    pub static EMBEDDED_SHERPA_SO: &[u8] = include_bytes!("../../native/linux-x64/libsherpa-onnx-c-api.so");
    #[cfg(not(target_os = "windows"))]
    pub static EMBEDDED_ONNXRUNTIME_SO: &[u8] = include_bytes!("../../native/linux-x64/libonnxruntime.so");

    /// Automatically extract embedded native dynamic libraries into `target_dir`
    pub fn ensure_runtime_extracted(target_dir: &Path) -> Option<PathBuf> {
        if std::fs::create_dir_all(target_dir).is_err() {
            return None;
        }

        #[cfg(target_os = "windows")]
        {
            let sherpa_path = target_dir.join("sherpa-onnx-c-api.dll");
            let onnx_path = target_dir.join("onnxruntime.dll");

            let need_sherpa = !sherpa_path.exists()
                || std::fs::metadata(&sherpa_path).map(|m| m.len()).unwrap_or(0) != EMBEDDED_SHERPA_DLL.len() as u64;
            if need_sherpa {
                if std::fs::write(&sherpa_path, EMBEDDED_SHERPA_DLL).is_err() {
                    return None;
                }
            }

            let need_onnx = !onnx_path.exists()
                || std::fs::metadata(&onnx_path).map(|m| m.len()).unwrap_or(0) != EMBEDDED_ONNXRUNTIME_DLL.len() as u64;
            if need_onnx {
                if std::fs::write(&onnx_path, EMBEDDED_ONNXRUNTIME_DLL).is_err() {
                    return None;
                }
            }

            if sherpa_path.exists() && onnx_path.exists() {
                Some(sherpa_path)
            } else {
                None
            }
        }

        #[cfg(not(target_os = "windows"))]
        {
            let sherpa_path = target_dir.join("libsherpa-onnx-c-api.so");
            let onnx_path = target_dir.join("libonnxruntime.so");

            let need_sherpa = !sherpa_path.exists()
                || std::fs::metadata(&sherpa_path).map(|m| m.len()).unwrap_or(0) != EMBEDDED_SHERPA_SO.len() as u64;
            if need_sherpa {
                if std::fs::write(&sherpa_path, EMBEDDED_SHERPA_SO).is_err() {
                    return None;
                }
            }

            let need_onnx = !onnx_path.exists()
                || std::fs::metadata(&onnx_path).map(|m| m.len()).unwrap_or(0) != EMBEDDED_ONNXRUNTIME_SO.len() as u64;
            if need_onnx {
                if std::fs::write(&onnx_path, EMBEDDED_ONNXRUNTIME_SO).is_err() {
                    return None;
                }
            }

            if sherpa_path.exists() && onnx_path.exists() {
                Some(sherpa_path)
            } else {
                None
            }
        }
    }

    /// Locate native dynamic libraries strictly relative to the executable or base directory.
    pub fn find_sherpa_dll(base_dir: &Path) -> Option<PathBuf> {
        // 1. Check if runtime DLLs already exist beside exe, base_dir, or current directory
        let mut check_dirs = Vec::new();
        if let Ok(exe) = std::env::current_exe() {
            if let Some(exe_dir) = exe.parent() {
                check_dirs.push(exe_dir.to_path_buf());
            }
        }
        check_dirs.push(base_dir.to_path_buf());
        check_dirs.push(PathBuf::from("."));

        for dir in &check_dirs {
            #[cfg(target_os = "windows")]
            let sherpa = dir.join("sherpa-onnx-c-api.dll");
            #[cfg(not(target_os = "windows"))]
            let sherpa = dir.join("libsherpa-onnx-c-api.so");

            if sherpa.exists() {
                return Some(sherpa);
            }
        }

        // 2. Extract embedded runtime DLLs strictly beside the executable or base directory
        if let Ok(exe) = std::env::current_exe() {
            if let Some(exe_dir) = exe.parent() {
                if let Some(extracted) = ensure_runtime_extracted(exe_dir) {
                    return Some(extracted);
                }
            }
        }
        if let Some(extracted) = ensure_runtime_extracted(base_dir) {
            return Some(extracted);
        }
        if let Some(extracted) = ensure_runtime_extracted(Path::new(".")) {
            return Some(extracted);
        }

        None
    }

    /// Dynamically locate streaming transducer model files on disk using structural fingerprinting
    pub fn find_transducer_model_dir_for_lang(base_dir: &Path, lang: &str) -> Option<(PathBuf, PathBuf, PathBuf, PathBuf)> {
        crate::asr::discovery::find_transducer_model_files(base_dir, "", lang)
    }

    /// Locate candidate streaming transducer model directories on disk (auto-detect)
    #[allow(dead_code)]
    pub fn find_transducer_model_dir(base_dir: &Path) -> Option<(PathBuf, PathBuf, PathBuf, PathBuf)> {
        find_transducer_model_dir_for_lang(base_dir, "auto")
    }

    /// Dynamically locate offline Whisper ONNX model files on disk using structural fingerprinting
    pub fn find_whisper_model_dir(base_dir: &Path, tier: &str) -> Option<(PathBuf, PathBuf, PathBuf)> {
        crate::asr::discovery::find_whisper_model_files(base_dir, tier)
    }
}

#[cfg(feature = "transducer")]
pub use sherpa_backend::SherpaWhisperEngine;

/// Check if Whisper ONNX model files and Sherpa dynamic runtime exist on disk
pub fn has_whisper_onnx_installed(models_dir: &Path, tier: &str) -> bool {
    #[cfg(feature = "transducer")]
    {
        let base_dir = models_dir.parent().unwrap_or(models_dir);
        (sherpa_backend::find_sherpa_dll(base_dir).is_some() || sherpa_backend::find_sherpa_dll(models_dir).is_some())
            && (sherpa_backend::find_whisper_model_dir(models_dir, tier).is_some()
                || sherpa_backend::find_whisper_model_dir(base_dir, tier).is_some())
    }
    #[cfg(not(feature = "transducer"))]
    {
        let _ = (models_dir, tier);
        false
    }
}

/// Create a GPU-accelerated Whisper recognizer powered by Sherpa-ONNX (DirectML/CUDA/CPU)
pub fn create_sherpa_whisper(
    models_dir: &Path,
    tier: &str,
    language: &str,
) -> Result<Box<dyn crate::asr::SpeechRecognizer>> {
    #[cfg(feature = "transducer")]
    {
        let base_dir = models_dir.parent().unwrap_or(models_dir);
        let dll_path = sherpa_backend::find_sherpa_dll(base_dir)
            .or_else(|| sherpa_backend::find_sherpa_dll(models_dir))
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "Sherpa-ONNX dynamic runtime library not found in {:?}.\n\
                     Run 'catetin' to automatically extract the embedded runtime.",
                    models_dir
                )
            })?;
        let (enc, dec, tok) = sherpa_backend::find_whisper_model_dir(models_dir, tier)
            .or_else(|| sherpa_backend::find_whisper_model_dir(base_dir, tier))
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "Whisper ONNX model files (encoder.onnx, decoder.onnx, tokens.txt) not found for tier '{}' in {:?}.\n\
                     Run 'catetin' to automatically prepare and download the required Whisper ONNX model.",
                    tier, models_dir
                )
            })?;

        log::info!(
            "[Whisper ONNX] Loading Sherpa-ONNX dynamic runtime from: {:?}",
            dll_path
        );
        let api = sherpa_backend::SherpaApi::load(&dll_path)
            .with_context(|| format!("Failed to load Sherpa-ONNX runtime from {:?}", dll_path))?;

        log::info!(
            "[Whisper ONNX] Initializing Whisper ONNX recognizer:\n  • Encoder: {:?}\n  • Decoder: {:?}\n  • Tokens:  {:?}",
            enc, dec, tok
        );

        let engine = sherpa_backend::SherpaWhisperEngine::new(api, &enc, &dec, &tok, language)
            .with_context(|| "Failed to instantiate SherpaWhisperEngine")?;

        log::info!(
            "[Whisper ONNX] Hardware acceleration active ({}) for Whisper Pass 2!",
            engine.active_provider()
        );
        Ok(Box::new(engine))
    }
    #[cfg(not(feature = "transducer"))]
    {
        let _ = (models_dir, tier, language);
        anyhow::bail!("Transducer/ONNX feature was not compiled into this binary build of Catetin.");
    }
}

/// Ensure embedded native dynamic libraries (sherpa-onnx + onnxruntime) are extracted to disk
pub fn ensure_runtime_extracted(target_dir: &Path) -> Option<std::path::PathBuf> {
    #[cfg(feature = "transducer")]
    {
        sherpa_backend::ensure_runtime_extracted(target_dir)
    }
    #[cfg(not(feature = "transducer"))]
    {
        let _ = target_dir;
        None
    }
}

/// Check if the Streaming Transducer runtime and model files exist on disk for a specific language
pub fn has_transducer_installed_for_lang(models_dir: &Path, lang: &str) -> bool {
    #[cfg(feature = "transducer")]
    {
        let base_dir = models_dir.parent().unwrap_or(models_dir);
        (sherpa_backend::find_sherpa_dll(base_dir).is_some() || sherpa_backend::find_sherpa_dll(models_dir).is_some())
            && (sherpa_backend::find_transducer_model_dir_for_lang(models_dir, lang).is_some()
                || sherpa_backend::find_transducer_model_dir_for_lang(base_dir, lang).is_some())
    }
    #[cfg(not(feature = "transducer"))]
    {
        let _ = (models_dir, lang);
        false
    }
}

/// Check if the Streaming Transducer runtime and model files exist on disk
pub fn has_transducer_installed(models_dir: &Path) -> bool {
    has_transducer_installed_for_lang(models_dir, "auto")
}

/// Instantiate the mandatory Streaming Transducer for Pass 1.
/// Bails with an actionable error if runtime DLLs or Zipformer model files are missing.
/// Instantiate the mandatory Streaming Transducer for Pass 1, with language awareness:
/// - If lang is "auto" and both ID and EN models exist: instantiates DualStreamingTransducer.
/// - If lang is "id" or "en": loads the respective Zipformer model.
/// - Gracefully falls back to whichever model is available.
pub fn create_transducer_for_lang(models_dir: &Path, lang: &str) -> Result<Box<dyn StreamingTransducer>> {
    #[cfg(feature = "transducer")]
    {
        let base_dir = models_dir.parent().unwrap_or(models_dir);
        let dll_path = sherpa_backend::find_sherpa_dll(base_dir)
            .or_else(|| sherpa_backend::find_sherpa_dll(models_dir))
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "Sherpa-ONNX dynamic runtime library (sherpa-onnx-c-api.dll / libsherpa-onnx-c-api.so) not found in {:?}.\n\
                     Run 'catetin' to automatically extract the required Streaming Transducer runtime.",
                    models_dir
                )
            })?;

        log::info!(
            "[Transducer] Loading Sherpa-ONNX dynamic runtime from: {:?}",
            dll_path
        );
        let api = sherpa_backend::SherpaApi::load(&dll_path)
            .with_context(|| format!("Failed to load Sherpa-ONNX dynamic runtime from {:?}", dll_path))?;

        let effective_lang = if lang.eq_ignore_ascii_case("auto") {
            "id"
        } else {
            lang
        };

        let (enc, dec, join, tok) = sherpa_backend::find_transducer_model_dir_for_lang(models_dir, effective_lang)
            .or_else(|| sherpa_backend::find_transducer_model_dir_for_lang(base_dir, effective_lang))
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "Streaming Transducer model files (encoder.onnx, decoder.onnx, joiner.onnx, tokens.txt) not found for language '{}' in {:?}.\n\
                     Run 'catetin' to automatically prepare and download the required Streaming Transducer model.",
                    effective_lang, models_dir
                )
            })?;

        log::info!(
            "[Transducer] Initializing Streaming Transducer (lang: {} [Pass-1 model: {}]):\n  • Encoder: {:?}\n  • Decoder: {:?}\n  • Joiner:  {:?}\n  • Tokens:  {:?}",
            lang, effective_lang, enc, dec, join, tok
        );

        let engine = sherpa_backend::SherpaTransducerEngine::new(api, &enc, &dec, &join, &tok)
            .with_context(|| "Failed to instantiate SherpaTransducerEngine")?;

        log::info!("[Transducer] Pass 1 Streaming Transducer successfully initialized (< 50ms live typing active)!");
        Ok(Box::new(engine))
    }
    #[cfg(not(feature = "transducer"))]
    {
        let _ = (models_dir, lang);
        anyhow::bail!("Transducer feature was not compiled into this binary build of Catetin.");
    }
}

pub fn create_transducer(models_dir: &Path) -> Result<Box<dyn StreamingTransducer>> {
    create_transducer_for_lang(models_dir, "auto")
}

/// Dynamically attempt to load and instantiate the Streaming Transducer for Pass 1.
pub fn try_create_transducer(models_dir: &Path) -> Option<Box<dyn StreamingTransducer>> {
    create_transducer(models_dir).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transducer_not_found_returns_none() {
        let fake_dir = Path::new("non_existent_models_dir_12345");
        assert!(try_create_transducer(fake_dir).is_none());
    }

    #[test]
    fn test_whisper_not_found_returns_false() {
        let fake_dir = Path::new("non_existent_models_dir_12345");
        assert!(!has_whisper_onnx_installed(fake_dir, "base"));
        assert!(!has_whisper_onnx_installed(fake_dir, "medium"));
    }

    #[test]
    fn test_ipa_phonemes_to_indonesian() {
        assert_eq!(ipa_phonemes_to_indonesian("ribumu|dunia"), "ribumu dunia");
        assert_eq!(ipa_phonemes_to_indonesian("|ɡuno|kəmaka|san|tif"), "guno kemaka san tif");
        assert_eq!(ipa_phonemes_to_indonesian("otal|məra"), "otal mera");
        assert_eq!(ipa_phonemes_to_indonesian("|almun|məmbolo"), "almun membolo");
        assert_eq!(ipa_phonemes_to_indonesian("tʃompu"), "compu");
        assert_eq!(ipa_phonemes_to_indonesian("|kəmukafi|pənuŋɡuŋ|mənumu"), "kemukafi penunggung menumu");
        assert_eq!(ipa_phonemes_to_indonesian("|alo"), "alo");
        assert_eq!(ipa_phonemes_to_indonesian("|saja|dʒalan|baɲaʔ|"), "saya jalan banyak");
        // Verify standard Latin / BPE text remains completely untouched (no j -> y change)
        assert_eq!(ipa_phonemes_to_indonesian("saya jalan banyak dengan kawan"), "saya jalan banyak dengan kawan");
        assert_eq!(ipa_phonemes_to_indonesian("halo selamat pagi semuanya"), "halo selamat pagi semuanya");
    }

    #[test]
    fn test_select_best_hypothesis() {
        let id_hyp = "selamat pagi apakah suara saya terdengar dengan baik";
        let en_hyp = "gud mornin evriwan";
        assert_eq!(select_best_hypothesis(id_hyp, en_hyp), id_hyp);

        let id_hyp2 = "helo ken yu hir mai vois";
        let en_hyp2 = "hello can you hear my voice";
        assert_eq!(select_best_hypothesis(id_hyp2, en_hyp2), en_hyp2);

        assert_eq!(select_best_hypothesis("", "testing one two"), "testing one two");
        assert_eq!(select_best_hypothesis("satu dua tiga", ""), "satu dua tiga");
    }
}
