use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

use crate::i18n::UiLanguage;
use crate::pipeline::DiarizationMode;



/// Centralized application configuration.
/// Can be loaded from and saved to a JSON configuration file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppConfig {
    /// UI display language (Indonesian, English, or custom)
    pub ui_language: UiLanguage,

    /// Diarization operating mode (online, in-person, hybrid)
    pub diarization_mode: DiarizationMode,

    /// Whisper model tier: "tiny", "base", "small", or "medium"
    pub model_tier: String,

    /// Mirror debug logs to stderr alongside the log file
    #[serde(default)]
    pub verbose: bool,

    /// ASR transcription language ("auto", "id", or "en")
    pub asr_language: String,

    /// Custom model file path or download URL override
    pub custom_model_url: Option<String>,

    /// Directory for model weights cache
    pub model_dir: PathBuf,

    /// Directory for log files
    pub log_dir: PathBuf,

    /// Minimum speaker clustering similarity threshold (default: 0.72)
    pub speaker_threshold: f32,

    /// Specific audio input device name (microphone), None for system default
    #[serde(default)]
    pub input_device_name: Option<String>,

    /// Specific audio output device name (system loopback), None for system default
    #[serde(default)]
    pub output_device_name: Option<String>,

    /// Enable split-screen AI meeting assistant pane (Phase 5)
    #[serde(default = "default_true")]
    pub show_assistant: bool,

    /// LLM API endpoint URL (default: "http://localhost:11434/v1" for Ollama)
    #[serde(default = "default_assistant_url")]
    pub assistant_url: String,

    /// LLM model name (default: "llama3.2:1b")
    #[serde(default = "default_assistant_model")]
    pub assistant_model: String,

    /// Optional API key for cloud LLMs (Groq, OpenAI, Gemini, DeepSeek)
    #[serde(default)]
    pub assistant_api_key: Option<String>,

    /// Streaming Transducer model identifier (default: "zipformer2-id")
    #[serde(default = "default_transducer_model")]
    pub transducer_model: String,

    /// Explicit directory path for Transducer model files (None for auto-detection in model_dir)
    #[serde(default)]
    pub transducer_model_dir: Option<PathBuf>,

    /// Noise suppression / cancellation level: "none", "default", or "aggressive"
    #[serde(default)]
    pub noise_suppression: NoiseSuppression,
}

/// Noise suppression mode for audio capture, VAD, and preprocessing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum NoiseSuppression {
    #[serde(rename = "none", alias = "no", alias = "off", alias = "false")]
    None,
    #[default]
    #[serde(rename = "default", alias = "some", alias = "moderate", alias = "normal")]
    Default,
    #[serde(rename = "aggressive", alias = "agresive", alias = "high")]
    Aggressive,
}

impl NoiseSuppression {
    pub fn display_label(&self, _lang: &UiLanguage) -> &'static str {
        match self {
            NoiseSuppression::None => "None",
            NoiseSuppression::Default => "Default",
            NoiseSuppression::Aggressive => "Aggressive",
        }
    }

    pub fn next(&self) -> Self {
        match self {
            NoiseSuppression::None => NoiseSuppression::Default,
            NoiseSuppression::Default => NoiseSuppression::Aggressive,
            NoiseSuppression::Aggressive => NoiseSuppression::None,
        }
    }

    pub fn prev(&self) -> Self {
        match self {
            NoiseSuppression::None => NoiseSuppression::Aggressive,
            NoiseSuppression::Default => NoiseSuppression::None,
            NoiseSuppression::Aggressive => NoiseSuppression::Default,
        }
    }
}

fn default_true() -> bool {
    true
}

fn default_assistant_url() -> String {
    "http://localhost:11434/v1".to_string()
}

fn default_assistant_model() -> String {
    "llama3.2:1b".to_string()
}

fn default_transducer_model() -> String {
    "zipformer2-id".to_string()
}

impl Default for AppConfig {
    fn default() -> Self {
        let base_dir = default_app_dir();
        Self {
            ui_language: UiLanguage::Indonesian, // Default Indonesian ("Tolong catetin rapat ini")
            diarization_mode: DiarizationMode::OnlineMeeting,
            model_tier: "base".to_string(),
            verbose: false,
            asr_language: "auto".to_string(),
            custom_model_url: None,
            model_dir: base_dir.join("models"),
            log_dir: base_dir.join("logs"),
            speaker_threshold: 0.72,
            input_device_name: None,
            output_device_name: None,
            show_assistant: true,
            assistant_url: default_assistant_url(),
            assistant_model: default_assistant_model(),
            assistant_api_key: None,
            transducer_model: default_transducer_model(),
            transducer_model_dir: None,
            noise_suppression: NoiseSuppression::Default,
        }
    }
}



impl AppConfig {
    /// Resolved directory for streaming transducer model
    pub fn transducer_dir(&self) -> PathBuf {
        if let Some(ref dir) = self.transducer_model_dir {
            dir.clone()
        } else {
            let direct = self.model_dir.join(format!("sherpa-onnx-streaming-{}", self.transducer_model));
            if direct.exists() {
                direct
            } else {
                self.model_dir.clone()
            }
        }
    }

    /// Standard location for config file: `%APPDATA%\catetin\config.json` or `~/.catetin/config.json`
    pub fn config_file_path() -> PathBuf {
        default_app_dir().join("config.json")
    }

    /// Load configuration from a specific file path
    pub fn load_from_path<P: AsRef<Path>>(path: P) -> Result<Self> {
        let p = path.as_ref();
        let content = std::fs::read_to_string(p)
            .with_context(|| format!("Failed to read config file at {:?}", p))?;
        serde_json::from_str::<Self>(&content)
            .with_context(|| format!("Failed to parse config file at {:?}", p))
    }

    /// Save configuration to a specific file path
    pub fn save_to_path<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let p = path.as_ref();
        if let Some(parent) = p.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("Failed to create directory {:?}", parent))?;
        }
        let json = serde_json::to_string_pretty(self)
            .context("Failed to serialize AppConfig to JSON")?;
        std::fs::write(p, json)
            .with_context(|| format!("Failed to write config file to {:?}", p))?;
        Ok(())
    }

    /// Load configuration from disk, falling back to defaults if not found.
    pub fn load_or_default() -> Self {
        let path = Self::config_file_path();
        if path.exists() {
            if let Ok(config) = Self::load_from_path(&path) {
                return config;
            }
        }
        Self::default()
    }

    /// Save configuration to the standard default path, with local fallback if restricted.
    pub fn save(&self) -> Result<()> {
        let path = Self::config_file_path();
        if self.save_to_path(&path).is_err() {
            return self.save_to_path(PathBuf::from("config.json"));
        }
        Ok(())
    }

    /// Check if the configured model tier corresponds to a Sherpa-ONNX Whisper model
    pub fn is_onnx_tier(&self) -> bool {
        true
    }

    /// User-friendly display label describing the active model architecture & hardware acceleration
    pub fn model_display_name(&self) -> String {
        format!("Whisper [{}] (Sherpa-ONNX CPU/GPU)", self.model_tier)
    }

    /// User-friendly display label describing the active transducer model
    pub fn transducer_display_name(&self) -> String {
        format!("Transducer [{}] (Sherpa-ONNX <50ms)", self.transducer_model)
    }

    /// User-friendly display label for ASR Audio transcription language
    pub fn asr_language_display(&self) -> &str {
        match self.asr_language.to_lowercase().as_str() {
            "id" | "indonesian" => "Bahasa Indonesia [id]",
            "en" | "english" => "English [en]",
            _ => "Auto-Detect [id/en/auto]",
        }
    }

    /// Cycle ASR audio transcription language forward or backward
    pub fn cycle_asr_language(&mut self, forward: bool) {
        let cur = self.asr_language.to_lowercase();
        self.asr_language = if forward {
            match cur.as_str() {
                "auto" => "id".to_string(),
                "id" => "en".to_string(),
                "en" => "auto".to_string(),
                _ => "auto".to_string(),
            }
        } else {
            match cur.as_str() {
                "auto" => "en".to_string(),
                "en" => "id".to_string(),
                "id" => "auto".to_string(),
                _ => "auto".to_string(),
            }
        };
    }
}

/// Default root application data directory:
/// Resolves directly beside the running binary itself (portable mode),
/// keeping config, logs, and models self-contained without cluttering the device.
pub fn default_app_dir() -> PathBuf {
    if let Ok(exe_path) = std::env::current_exe() {
        if let Some(parent) = exe_path.parent() {
            return parent.to_path_buf();
        }
    }
    PathBuf::from(".")
}
