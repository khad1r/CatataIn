use anyhow::{Context, Result};
use std::path::{Path, PathBuf};

pub const WHISPER_BASE_URL: &str =
    "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin";
pub const WHISPER_SMALL_URL: &str =
    "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.bin";

#[derive(Debug, Clone)]
pub struct ModelManager {
    model_dir: PathBuf,
}

impl Default for ModelManager {
    fn default() -> Self {
        Self::new()
    }
}

impl ModelManager {
    pub fn new() -> Self {
        let dir = Self::default_model_dir();
        Self { model_dir: dir }
    }

    pub fn with_dir<P: AsRef<Path>>(dir: P) -> Self {
        Self {
            model_dir: dir.as_ref().to_path_buf(),
        }
    }

    /// Default model storage directory:
    /// Windows: %APPDATA%\catetin\models
    /// Linux/Android/macOS: ~/.catetin/models
    pub fn default_model_dir() -> PathBuf {
        if let Ok(appdata) = std::env::var("APPDATA") {
            PathBuf::from(appdata).join("catetin").join("models")
        } else if let Ok(home) = std::env::var("HOME") {
            PathBuf::from(home).join(".catetin").join("models")
        } else {
            PathBuf::from("models")
        }
    }

    pub fn model_dir(&self) -> &Path {
        &self.model_dir
    }

    /// Get path to a model file, ensuring model directory exists
    pub fn get_model_path(&self, filename: &str) -> PathBuf {
        let local_path = PathBuf::from("models").join(filename);
        if local_path.exists() {
            return local_path;
        }
        self.model_dir.join(filename)
    }

    /// Check if a model file exists locally
    pub fn model_exists(&self, filename: &str) -> bool {
        let path = self.get_model_path(filename);
        path.exists() && path.metadata().map(|m| m.len() > 1024).unwrap_or(false)
    }

    /// Ensure the model directory exists
    pub fn ensure_dir(&self) -> Result<()> {
        if !self.model_dir.exists() {
            std::fs::create_dir_all(&self.model_dir)
                .with_context(|| format!("Failed to create model directory: {:?}", self.model_dir))?;
        }
        Ok(())
    }
}
