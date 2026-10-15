use anyhow::{Context, Result};
use log::{Level, LevelFilter, Metadata, Record};
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::config::AppConfig;

/// Path of the currently active log file.
static LOG_FILE_PATH: OnceLock<PathBuf> = OnceLock::new();

/// Dedicated file-based logger to keep stdout/stderr clean for TUI & terminal screens.
struct CatetinLogger {
    file: Mutex<Option<File>>,
    enable_stderr: bool,
}

impl log::Log for CatetinLogger {
    fn enabled(&self, metadata: &Metadata) -> bool {
        metadata.level() <= Level::Debug
    }

    fn log(&self, record: &Record) {
        if !self.enabled(record.metadata()) {
            return;
        }

        let line = format!(
            "{} [{}] [{}] {}\n",
            format_utc_timestamp(),
            record.level(),
            record.target(),
            record.args()
        );

        if let Ok(mut file_lock) = self.file.lock() {
            if let Some(ref mut file) = *file_lock {
                let _ = file.write_all(line.as_bytes());
                let _ = file.flush();
            }
        }

        if self.enable_stderr {
            eprint!("{}", line);
        }
    }

    fn flush(&self) {
        if let Ok(mut file_lock) = self.file.lock() {
            if let Some(ref mut file) = *file_lock {
                let _ = file.flush();
            }
        }
    }
}

/// Initialize the central logging subsystem.
///
/// Log destination priority:
/// - Windows: `<exe_dir>\logs\catetin.log`
/// - Custom: Configured `AppConfig::log_dir`
///
/// If `enable_stderr` is `false`, logs are NEVER written to standard output or error,
/// completely eliminating screen flicker and cursor destruction in terminal / TUI mode.
pub fn init_logger(config: &AppConfig, enable_stderr: bool) -> Result<PathBuf> {
    let mut log_dir = config.log_dir.clone();
    if !log_dir.exists() {
        if let Err(_) = std::fs::create_dir_all(&log_dir) {
            let fallback_dir = PathBuf::from("logs");
            if std::fs::create_dir_all(&fallback_dir).is_ok() {
                log_dir = fallback_dir;
            }
        }
    }

    let mut log_file_path = log_dir.join("catetin.log");
    let file = match OpenOptions::new().create(true).append(true).open(&log_file_path) {
        Ok(f) => f,
        Err(_) => {
            log_file_path = PathBuf::from("catetin.log");
            OpenOptions::new()
                .create(true)
                .append(true)
                .open(&log_file_path)
                .with_context(|| format!("Failed to open log file at {:?}", log_file_path))?
        }
    };

    let _ = LOG_FILE_PATH.set(log_file_path.clone());

    let logger = CatetinLogger {
        file: Mutex::new(Some(file)),
        enable_stderr,
    };

    // If logger was already initialized (e.g. in test suites), ignore error
    let _ = log::set_logger(Box::leak(Box::new(logger)));
    log::set_max_level(LevelFilter::Debug);


    log::info!("=== Catetin Logging Initialized ===");
    log::info!("Log file path: {:?}", log_file_path);
    log::info!("UI Language: {}", config.ui_language);
    log::info!("Model Tier: {}", config.model_tier);

    Ok(log_file_path)
}

/// Retrieve the active log file path if logging has been initialized.
pub fn get_log_file_path() -> Option<&'static Path> {
    LOG_FILE_PATH.get().map(|p| p.as_path())
}

/// Format current UTC time using stdlib without external datetime dependencies.
pub fn format_utc_timestamp() -> String {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let s = secs % 60;
    let m = (secs / 60) % 60;
    let h = (secs / 3600) % 24;
    format!("{:02}:{:02}:{:02} UTC", h, m, s)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_utc_timestamp() {
        let ts = format_utc_timestamp();
        assert!(ts.contains("UTC"));
    }
}
