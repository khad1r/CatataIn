use anyhow::{Context, Result};
use std::path::{Path, PathBuf};

use crate::config::{default_app_dir, AppConfig};

/// Manages local offline model caches and paths.
/// All default URLs and filenames are centralized in `crate::config`.
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
        let dir = default_app_dir().join("models");
        Self { model_dir: dir }
    }

    pub fn from_config(config: &AppConfig) -> Self {
        Self {
            model_dir: config.model_dir.clone(),
        }
    }

    pub fn with_dir<P: AsRef<Path>>(dir: P) -> Self {
        Self {
            model_dir: dir.as_ref().to_path_buf(),
        }
    }

    pub fn model_dir(&self) -> &Path {
        &self.model_dir
    }

    /// Resolve absolute path to a model file, preferring a local `./models/` folder if present
    pub fn get_model_path(&self, filename: &str) -> PathBuf {
        let local_path = PathBuf::from("models").join(filename);
        if local_path.exists() {
            return local_path;
        }
        self.model_dir.join(filename)
    }

    /// Check if a model file exists locally and is non-empty
    pub fn model_exists(&self, filename: &str) -> bool {
        let path = self.get_model_path(filename);
        path.exists() && path.metadata().map(|m| m.len() > 1024).unwrap_or(false)
    }

    /// Ensure the target model directory exists on disk
    pub fn ensure_dir(&self) -> Result<()> {
        if !self.model_dir.exists() {
            std::fs::create_dir_all(&self.model_dir)
                .with_context(|| format!("Failed to create model directory: {:?}", self.model_dir))?;
        }
        Ok(())
    }
}
