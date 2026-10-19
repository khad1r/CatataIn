use std::path::{Path, PathBuf};
use serde::{Deserialize, Serialize};

/// Structural classification of speech recognition models discovered on disk.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelKind {
    /// Pass 1 Streaming Transducer (Zipformer, RNN-T) requiring encoder, decoder, joiner, and tokens
    Transducer,
    /// Pass 2 Offline Whisper requiring encoder, decoder, and tokens (without joiner)
    Whisper,
}

/// Metadata and file paths for a dynamically discovered speech recognition model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscoveredModel {
    /// Folder name or identifier (e.g. "sherpa-onnx-streaming-zipformer2-id" or "custom-whisper")
    pub id: String,
    /// Human-friendly display label
    pub display_name: String,
    /// Base directory containing the model weights
    pub path: PathBuf,
    /// Model architecture classification
    pub kind: ModelKind,
    /// Detected language specialization ("id", "en", "auto", or "multilingual")
    pub language: String,
    /// Path to encoder ONNX model
    pub encoder: PathBuf,
    /// Path to decoder ONNX model
    pub decoder: PathBuf,
    /// Path to joiner ONNX model (Streaming Transducer only)
    pub joiner: Option<PathBuf>,
    /// Path to vocabulary tokens file (tokens.txt)
    pub tokens: PathBuf,
}

impl DiscoveredModel {
    pub fn is_transducer(&self) -> bool {
        self.kind == ModelKind::Transducer
    }

    pub fn is_whisper(&self) -> bool {
        self.kind == ModelKind::Whisper
    }
}

/// Inspect a single directory to check if it structurally matches a Transducer or Whisper model.
pub fn inspect_directory(dir: &Path) -> Option<DiscoveredModel> {
    if !dir.is_dir() {
        return None;
    }

    let dir_name = dir.file_name()?.to_str()?;
    // Skip system/hidden folders and non-model output directories
    if dir_name.starts_with('.') || dir_name == "logs" || dir_name == "target" || dir_name == "dist" {
        return None;
    }

    let mut tokens = None;
    let mut encoder = None;
    let mut decoder = None;
    let mut joiner = None;

    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_file() {
                let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("").to_lowercase();
                if name.ends_with(".txt") && (name.contains("tokens") || name == "vocab.txt") {
                    tokens = Some(path);
                } else if name.ends_with(".onnx") {
                    if name.contains("encoder") {
                        if name.contains("int8") || encoder.is_none() {
                            encoder = Some(path);
                        }
                    } else if name.contains("decoder") {
                        if name.contains("int8") || decoder.is_none() {
                            decoder = Some(path);
                        }
                    } else if name.contains("joiner") {
                        if name.contains("int8") || joiner.is_none() {
                            joiner = Some(path);
                        }
                    }
                }
            }
        }
    }

    let (enc, dec, tok) = (encoder?, decoder?, tokens?);

    let lower_name = dir_name.to_lowercase();
    let language = if lower_name.contains("-id") || lower_name.contains("_id") || lower_name.contains("indonesia") {
        "id".to_string()
    } else if lower_name.contains("-en") || lower_name.contains("_en") || lower_name.contains("english") {
        "en".to_string()
    } else {
        "auto".to_string()
    };

    if let Some(join) = joiner {
        let display_name = format!("Transducer [{}] (lang: {})", dir_name, language);
        Some(DiscoveredModel {
            id: dir_name.to_string(),
            display_name,
            path: dir.to_path_buf(),
            kind: ModelKind::Transducer,
            language,
            encoder: enc,
            decoder: dec,
            joiner: Some(join),
            tokens: tok,
        })
    } else {
        let display_name = format!("Whisper [{}]", dir_name);
        Some(DiscoveredModel {
            id: dir_name.to_string(),
            display_name,
            path: dir.to_path_buf(),
            kind: ModelKind::Whisper,
            language,
            encoder: enc,
            decoder: dec,
            joiner: None,
            tokens: tok,
        })
    }
}

/// Recursively discover all valid Transducer and Whisper models in `models_dir` and common locations.
pub fn discover_all_models(models_dir: &Path) -> Vec<DiscoveredModel> {
    let mut search_roots = Vec::new();
    search_roots.push(models_dir.to_path_buf());
    search_roots.push(models_dir.join("transducer"));
    search_roots.push(models_dir.join("whisper"));

    if let Ok(exe) = std::env::current_exe() {
        if let Some(exe_dir) = exe.parent() {
            search_roots.push(exe_dir.join("models"));
            search_roots.push(exe_dir.join("models").join("transducer"));
            search_roots.push(exe_dir.join("models").join("whisper"));
            search_roots.push(exe_dir.to_path_buf());
        }
    }

    search_roots.push(PathBuf::from("models"));
    search_roots.push(PathBuf::from("models").join("transducer"));
    search_roots.push(PathBuf::from("models").join("whisper"));
    search_roots.push(PathBuf::from("."));

    let mut discovered = Vec::new();
    let mut visited_paths = std::collections::HashSet::new();

    for root in search_roots {
        if !root.exists() {
            continue;
        }

        // 1. Direct folder check
        if let Some(model) = inspect_directory(&root) {
            if visited_paths.insert(model.path.clone()) {
                discovered.push(model);
            }
            continue;
        }

        // 2. Immediate subdirectories check
        if let Ok(entries) = std::fs::read_dir(&root) {
            for entry in entries.flatten() {
                let sub_path = entry.path();
                if sub_path.is_dir() {
                    if let Some(model) = inspect_directory(&sub_path) {
                        if visited_paths.insert(model.path.clone()) {
                            discovered.push(model);
                        }
                    } else if let Ok(sub_entries) = std::fs::read_dir(&sub_path) {
                        // 3. Nested 2nd-level subdirectories (e.g. models/transducer/* or models/whisper/*)
                        for sub_sub_entry in sub_entries.flatten() {
                            let sub_sub_path = sub_sub_entry.path();
                            if sub_sub_path.is_dir() {
                                if let Some(m) = inspect_directory(&sub_sub_path) {
                                    if visited_paths.insert(m.path.clone()) {
                                        discovered.push(m);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    discovered
}

/// Discover all valid streaming transducer models on disk
pub fn discover_transducer_models(models_dir: &Path) -> Vec<DiscoveredModel> {
    discover_all_models(models_dir)
        .into_iter()
        .filter(|m| m.is_transducer())
        .collect()
}

/// Discover all valid Whisper models on disk
pub fn discover_whisper_models(models_dir: &Path) -> Vec<DiscoveredModel> {
    discover_all_models(models_dir)
        .into_iter()
        .filter(|m| m.is_whisper())
        .collect()
}

/// Find a streaming transducer matching name/id, language, or path
pub fn find_transducer_model_files(
    models_dir: &Path,
    name_or_path: &str,
    lang: &str,
) -> Option<(PathBuf, PathBuf, PathBuf, PathBuf)> {
    let direct = Path::new(name_or_path);
    if direct.is_dir() {
        if let Some(m) = inspect_directory(direct) {
            if let Some(join) = m.joiner {
                return Some((m.encoder, m.decoder, join, m.tokens));
            }
        }
    }

    let candidates = discover_transducer_models(models_dir);
    if candidates.is_empty() {
        return None;
    }

    let norm_target = name_or_path.trim().to_lowercase();
    let norm_lang = lang.trim().to_lowercase();

    // Priority 1: Match ID or name directly
    if !norm_target.is_empty() && norm_target != "auto" {
        for m in &candidates {
            let m_id = m.id.to_lowercase();
            if m_id == norm_target || m_id.contains(&norm_target) || norm_target.contains(&m_id) {
                if let Some(join) = &m.joiner {
                    return Some((m.encoder.clone(), m.decoder.clone(), join.clone(), m.tokens.clone()));
                }
            }
        }
    }

    // Priority 2: Filter by requested language
    if norm_lang == "id" || norm_lang == "indonesian" {
        for m in &candidates {
            if m.language == "id" || m.id.to_lowercase().contains("id") {
                if let Some(join) = &m.joiner {
                    return Some((m.encoder.clone(), m.decoder.clone(), join.clone(), m.tokens.clone()));
                }
            }
        }
    } else if norm_lang == "en" || norm_lang == "english" {
        for m in &candidates {
            if m.language == "en" || m.id.to_lowercase().contains("en") {
                if let Some(join) = &m.joiner {
                    return Some((m.encoder.clone(), m.decoder.clone(), join.clone(), m.tokens.clone()));
                }
            }
        }
    }

    // Priority 3: Fallback to first available transducer
    let first = &candidates[0];
    if let Some(join) = &first.joiner {
        Some((first.encoder.clone(), first.decoder.clone(), join.clone(), first.tokens.clone()))
    } else {
        None
    }
}

/// Find a Whisper ONNX model matching tier name, model name, or path
pub fn find_whisper_model_files(
    models_dir: &Path,
    tier_or_name: &str,
) -> Option<(PathBuf, PathBuf, PathBuf)> {
    let direct = Path::new(tier_or_name);
    if direct.is_dir() {
        if let Some(m) = inspect_directory(direct) {
            if m.is_whisper() {
                return Some((m.encoder, m.decoder, m.tokens));
            }
        }
    }

    let candidates = discover_whisper_models(models_dir);
    if candidates.is_empty() {
        return None;
    }

    let norm_tier = tier_or_name.trim().to_lowercase();
    let base_tier = if norm_tier.starts_with("tiny") {
        "tiny"
    } else if norm_tier.starts_with("base") {
        "base"
    } else if norm_tier.starts_with("small") {
        "small"
    } else if norm_tier.starts_with("medium") {
        "medium"
    } else if norm_tier.starts_with("turbo") || norm_tier.contains("large-v3-turbo") {
        "turbo"
    } else {
        &norm_tier
    };

    // Priority 1: Match tier prefix or model name
    for m in &candidates {
        let m_id = m.id.to_lowercase();
        if m_id == norm_tier
            || m_id.contains(base_tier)
            || m_id.contains(&norm_tier)
            || norm_tier.contains(&m_id)
        {
            return Some((m.encoder.clone(), m.decoder.clone(), m.tokens.clone()));
        }
    }

    // Priority 2: First available whisper model
    Some((candidates[0].encoder.clone(), candidates[0].decoder.clone(), candidates[0].tokens.clone()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_inspect_directory_transducer() {
        let temp = std::env::temp_dir().join("catetin_test_model_transducer");
        let _ = std::fs::remove_dir_all(&temp);
        std::fs::create_dir_all(&temp).unwrap();

        std::fs::write(temp.join("tokens.txt"), "dummy").unwrap();
        std::fs::write(temp.join("encoder.onnx"), "dummy").unwrap();
        std::fs::write(temp.join("decoder.onnx"), "dummy").unwrap();
        std::fs::write(temp.join("joiner.onnx"), "dummy").unwrap();

        let inspected = inspect_directory(&temp).expect("Should discover transducer");
        assert_eq!(inspected.kind, ModelKind::Transducer);
        assert!(inspected.joiner.is_some());

        let _ = std::fs::remove_dir_all(&temp);
    }

    #[test]
    fn test_inspect_directory_whisper() {
        let temp = std::env::temp_dir().join("catetin_test_model_whisper");
        let _ = std::fs::remove_dir_all(&temp);
        std::fs::create_dir_all(&temp).unwrap();

        std::fs::write(temp.join("tokens.txt"), "dummy").unwrap();
        std::fs::write(temp.join("encoder.onnx"), "dummy").unwrap();
        std::fs::write(temp.join("decoder.onnx"), "dummy").unwrap();

        let inspected = inspect_directory(&temp).expect("Should discover whisper");
        assert_eq!(inspected.kind, ModelKind::Whisper);
        assert!(inspected.joiner.is_none());

        let _ = std::fs::remove_dir_all(&temp);
    }
}
