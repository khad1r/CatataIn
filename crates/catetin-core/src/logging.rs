use anyhow::{Context, Result};
use chrono::Utc;
use log::{Level, LevelFilter, Metadata, Record};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock, RwLock};

use crate::config::AppConfig;

/// A structured log entry cached in memory for TUI and UI diagnostic panels.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogEntry {
    pub timestamp: String,
    pub level: String,
    pub target: String,
    pub message: String,
}

impl LogEntry {
    pub fn new(level: Level, target: &str, message: &str) -> Self {
        let timestamp = Utc::now().format("%Y-%m-%d %H:%M:%S UTC").to_string();
        Self {
            timestamp,
            level: level.to_string(),
            target: target.to_string(),
            message: message.to_string(),
        }
    }

    pub fn formatted(&self) -> String {
        format!(
            "{} [{}] [{}] {}",
            self.timestamp, self.level, self.target, self.message
        )
    }
}

/// Thread-safe in-memory rolling buffer for recent log lines.
static RECENT_LOGS: RwLock<Option<VecDeque<LogEntry>>> = RwLock::new(None);
const MAX_IN_MEMORY_LOGS: usize = 250;

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

        let entry = LogEntry::new(
            record.level(),
            record.target(),
            &format!("{}", record.args()),
        );
        let line = format!("{}\n", entry.formatted());

        // 1. Append to in-memory circular buffer
        if let Ok(mut lock) = RECENT_LOGS.write() {
            if let Some(ref mut queue) = *lock {
                if queue.len() >= MAX_IN_MEMORY_LOGS {
                    queue.pop_front();
                }
                queue.push_back(entry);
            }
        }

        // 2. Append to log file
        if let Ok(mut file_lock) = self.file.lock() {
            if let Some(ref mut file) = *file_lock {
                let _ = file.write_all(line.as_bytes());
                let _ = file.flush();
            }
        }

        // 3. Stderr output only if explicitly enabled (non-TUI verbose mode)
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
/// - Windows: `%APPDATA%\catetin\logs\catetin.log`
/// - Linux/macOS: `~/.catetin/logs/catetin.log`
/// - Custom: Configured `AppConfig::log_dir`
///
/// If `enable_stderr` is `false`, logs are NEVER written to standard output or error,
/// completely eliminating screen flicker and cursor destruction in TUI mode.
pub fn init_logger(config: &AppConfig, enable_stderr: bool) -> Result<PathBuf> {
    let mut log_dir = config.log_dir.clone();
    if !log_dir.exists() {
        if let Err(_) = std::fs::create_dir_all(&log_dir) {
            // Graceful fallback for restricted, portable, or sandboxed environments
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
            // Secondary fallback: local directory directly
            log_file_path = PathBuf::from("catetin.log");
            OpenOptions::new()
                .create(true)
                .append(true)
                .open(&log_file_path)
                .with_context(|| format!("Failed to open log file at {:?}", log_file_path))?
        }
    };

    // Initialize in-memory buffer
    if let Ok(mut lock) = RECENT_LOGS.write() {
        *lock = Some(VecDeque::with_capacity(MAX_IN_MEMORY_LOGS));
    }

    let _ = LOG_FILE_PATH.set(log_file_path.clone());

    let logger = CatetinLogger {
        file: Mutex::new(Some(file)),
        enable_stderr,
    };

    // If logger was already initialized (e.g. in test suites), don't fail
    let _ = log::set_boxed_logger(Box::new(logger));
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

/// Fetch a snapshot of recent log entries for in-app UI display.
pub fn get_recent_logs(max_count: usize) -> Vec<LogEntry> {
    if let Ok(lock) = RECENT_LOGS.read() {
        if let Some(ref queue) = *lock {
            let start = queue.len().saturating_sub(max_count);
            return queue.iter().skip(start).cloned().collect();
        }
    }
    Vec::new()
}

/// Clear the in-memory log buffer.
pub fn clear_recent_logs() {
    if let Ok(mut lock) = RECENT_LOGS.write() {
        if let Some(ref mut queue) = *lock {
            queue.clear();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_log_entry_format() {
        let entry = LogEntry::new(Level::Info, "test_mod", "Hello world");
        let formatted = entry.formatted();
        assert!(formatted.contains("[INFO]"));
        assert!(formatted.contains("[test_mod]"));
        assert!(formatted.contains("Hello world"));
    }

    #[test]
    fn test_in_memory_ring_buffer() {
        let mut queue = VecDeque::with_capacity(3);
        for i in 0..5 {
            if queue.len() >= 3 {
                queue.pop_front();
            }
            queue.push_back(LogEntry::new(Level::Debug, "test", &format!("Msg {}", i)));
        }
        assert_eq!(queue.len(), 3);
        assert_eq!(queue[0].message, "Msg 2");
        assert_eq!(queue[1].message, "Msg 3");
        assert_eq!(queue[2].message, "Msg 4");
    }
}
