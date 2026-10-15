use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

use crate::i18n::UiLanguage;
use crate::pipeline::DiarizationMode;

/// Centralized download endpoints and repository URLs for offline models.
/// All URLs can be customized or pointed to private mirrors/proxies.
pub mod urls {
    pub const WHISPER_TINY: &str =
        "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.bin";
    pub const WHISPER_BASE: &str =
        "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin";
    pub const WHISPER_SMALL: &str =
        "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.bin";
    pub const WHISPER_MEDIUM: &str =
        "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium.bin";
}

/// Standard filenames for cached models on local disk.
pub mod filenames {
    pub const WHISPER_TINY: &str = "ggml-tiny.bin";
    pub const WHISPER_BASE: &str = "ggml-base.bin";
    pub const WHISPER_SMALL: &str = "ggml-small.bin";
    pub const WHISPER_MEDIUM: &str = "ggml-medium.bin";
}


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
        }
    }
}



impl AppConfig {
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

    /// Resolve download URL for the configured model tier
    pub fn model_url(&self) -> &str {
        if let Some(ref custom) = self.custom_model_url {
            return custom.as_str();
        }
        match self.model_tier.to_lowercase().as_str() {
            "tiny" => urls::WHISPER_TINY,
            "small" => urls::WHISPER_SMALL,
            "medium" => urls::WHISPER_MEDIUM,
            _ => urls::WHISPER_BASE,
        }
    }

    /// Resolve filename on disk for the configured model tier
    pub fn model_filename(&self) -> &str {
        match self.model_tier.to_lowercase().as_str() {
            "tiny" => filenames::WHISPER_TINY,
            "small" => filenames::WHISPER_SMALL,
            "medium" => filenames::WHISPER_MEDIUM,
            _ => filenames::WHISPER_BASE,
        }
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
