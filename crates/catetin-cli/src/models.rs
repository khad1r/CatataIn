use anyhow::Result;
use std::path::{Path, PathBuf};
use catetin_core::asr::{
    has_transducer_installed_for_lang, has_whisper_onnx_installed,
};
use catetin_core::config::AppConfig;
use catetin_core::i18n::UiLanguage;
use crate::kb::read_single_choice;

/// Declarative specification of a downloadable speech recognition model preset.
pub struct ModelPreset {
    pub id: &'static str,
    pub display_name: &'static str,
    pub description: &'static str,
    #[allow(dead_code)]
    pub is_transducer: bool,
    pub subfolder: &'static str,
    pub files: &'static [(&'static str, &'static str)],
}

pub static TRANSDUCER_PRESETS: &[ModelPreset] = &[
    ModelPreset {
        id: "zipformer2-id",
        display_name: "Zipformer2 Indonesian (spacewave-id)",
        description: "Streaming RNN-T INT8 (< 50ms, Bahasa Indonesia)",
        is_transducer: true,
        subfolder: "sherpa-onnx-streaming-zipformer2-id",
        files: &[
            ("tokens.txt", "https://huggingface.co/spacewave/sherpa-onnx-streaming-zipformer2-id/resolve/main/tokens.txt"),
            ("encoder.onnx", "https://huggingface.co/spacewave/sherpa-onnx-streaming-zipformer2-id/resolve/main/encoder-iter-100000-avg-15-chunk-32-left-256.int8.onnx"),
            ("decoder.onnx", "https://huggingface.co/spacewave/sherpa-onnx-streaming-zipformer2-id/resolve/main/decoder-iter-100000-avg-15-chunk-32-left-256.int8.onnx"),
            ("joiner.onnx", "https://huggingface.co/spacewave/sherpa-onnx-streaming-zipformer2-id/resolve/main/joiner-iter-100000-avg-15-chunk-32-left-256.int8.onnx"),
        ],
    },
    ModelPreset {
        id: "zipformer-en",
        display_name: "Zipformer English (csukuangfj/en)",
        description: "Streaming RNN-T INT8 (< 50ms, English)",
        is_transducer: true,
        subfolder: "sherpa-onnx-streaming-zipformer-en-2023-06-26",
        files: &[
            ("tokens.txt", "https://huggingface.co/csukuangfj/sherpa-onnx-streaming-zipformer-en-2023-06-26/resolve/main/tokens.txt"),
            ("encoder.onnx", "https://huggingface.co/csukuangfj/sherpa-onnx-streaming-zipformer-en-2023-06-26/resolve/main/encoder-epoch-99-avg-1-chunk-16-left-128.int8.onnx"),
            ("decoder.onnx", "https://huggingface.co/csukuangfj/sherpa-onnx-streaming-zipformer-en-2023-06-26/resolve/main/decoder-epoch-99-avg-1-chunk-16-left-128.onnx"),
            ("joiner.onnx", "https://huggingface.co/csukuangfj/sherpa-onnx-streaming-zipformer-en-2023-06-26/resolve/main/joiner-epoch-99-avg-1-chunk-16-left-128.int8.onnx"),
        ],
    },
];

pub static WHISPER_PRESETS: &[ModelPreset] = &[
    ModelPreset {
        id: "base",
        display_name: "Whisper Base",
        description: "~160 MB | DirectML GPU / CPU INT8 (Fast, recommended for meetings)",
        is_transducer: false,
        subfolder: "sherpa-onnx-whisper-base",
        files: &[
            ("base-tokens.txt", "https://huggingface.co/csukuangfj/sherpa-onnx-whisper-base/resolve/main/base-tokens.txt"),
            ("base-encoder.int8.onnx", "https://huggingface.co/csukuangfj/sherpa-onnx-whisper-base/resolve/main/base-encoder.int8.onnx"),
            ("base-decoder.int8.onnx", "https://huggingface.co/csukuangfj/sherpa-onnx-whisper-base/resolve/main/base-decoder.int8.onnx"),
        ],
    },
    ModelPreset {
        id: "medium",
        display_name: "Whisper Medium",
        description: "~980 MB | DirectML GPU / CPU INT8 (High accuracy multilingual & technical terms)",
        is_transducer: false,
        subfolder: "sherpa-onnx-whisper-medium",
        files: &[
            ("medium-tokens.txt", "https://huggingface.co/csukuangfj/sherpa-onnx-whisper-medium/resolve/main/medium-tokens.txt"),
            ("medium-encoder.int8.onnx", "https://huggingface.co/csukuangfj/sherpa-onnx-whisper-medium/resolve/main/medium-encoder.int8.onnx"),
            ("medium-decoder.int8.onnx", "https://huggingface.co/csukuangfj/sherpa-onnx-whisper-medium/resolve/main/medium-decoder.int8.onnx"),
        ],
    },
    ModelPreset {
        id: "small",
        display_name: "Whisper Small",
        description: "~480 MB | DirectML GPU / CPU INT8 (Balanced speed and precision)",
        is_transducer: false,
        subfolder: "sherpa-onnx-whisper-small",
        files: &[
            ("small-tokens.txt", "https://huggingface.co/csukuangfj/sherpa-onnx-whisper-small/resolve/main/small-tokens.txt"),
            ("small-encoder.int8.onnx", "https://huggingface.co/csukuangfj/sherpa-onnx-whisper-small/resolve/main/small-encoder.int8.onnx"),
            ("small-decoder.int8.onnx", "https://huggingface.co/csukuangfj/sherpa-onnx-whisper-small/resolve/main/small-decoder.int8.onnx"),
        ],
    },
    ModelPreset {
        id: "tiny",
        display_name: "Whisper Tiny",
        description: "~75 MB | DirectML GPU / CPU INT8 (Ultra lightweight)",
        is_transducer: false,
        subfolder: "sherpa-onnx-whisper-tiny",
        files: &[
            ("tiny-tokens.txt", "https://huggingface.co/csukuangfj/sherpa-onnx-whisper-tiny/resolve/main/tiny-tokens.txt"),
            ("tiny-encoder.int8.onnx", "https://huggingface.co/csukuangfj/sherpa-onnx-whisper-tiny/resolve/main/tiny-encoder.int8.onnx"),
            ("tiny-decoder.int8.onnx", "https://huggingface.co/csukuangfj/sherpa-onnx-whisper-tiny/resolve/main/tiny-decoder.int8.onnx"),
        ],
    },
    ModelPreset {
        id: "turbo",
        display_name: "Whisper Large-v3-Turbo",
        description: "~550 MB | DirectML GPU / CPU INT8 (Fast large model)",
        is_transducer: false,
        subfolder: "sherpa-onnx-whisper-turbo",
        files: &[
            ("turbo-tokens.txt", "https://huggingface.co/csukuangfj/sherpa-onnx-whisper-turbo/resolve/main/turbo-tokens.txt"),
            ("turbo-encoder.int8.onnx", "https://huggingface.co/csukuangfj/sherpa-onnx-whisper-turbo/resolve/main/turbo-encoder.int8.onnx"),
            ("turbo-decoder.int8.onnx", "https://huggingface.co/csukuangfj/sherpa-onnx-whisper-turbo/resolve/main/turbo-decoder.int8.onnx"),
        ],
    },
];

/// Download a model preset into `target_base_dir` via curl with progress indicator
pub fn download_preset(target_base_dir: &Path, preset: &ModelPreset) -> Result<PathBuf> {
    let dest_dir = if target_base_dir.file_name().and_then(|f| f.to_str()).unwrap_or("") == preset.subfolder {
        target_base_dir.to_path_buf()
    } else {
        target_base_dir.join(preset.subfolder)
    };
    std::fs::create_dir_all(&dest_dir)?;

    let curl_bin = if cfg!(windows) { "curl.exe" } else { "curl" };
    println!("============================================================");
    println!("  Catetin - Mengunduh Model ONNX ({})", preset.display_name);
    println!("  Deskripsi : {}", preset.description);
    println!("  Target    : {}", dest_dir.display());
    println!("============================================================");

    for (fname, url) in preset.files {
        let dest = dest_dir.join(fname);
        if dest.exists() && std::fs::metadata(&dest).map(|m| m.len()).unwrap_or(0) > 100 {
            println!("  [✓] {} sudah tersedia, melewati pengunduhan.", fname);
            continue;
        }
        println!("  -> Mengunduh {}...", fname);
        let status = std::process::Command::new(curl_bin)
            .args(["-C", "-", "-L", "--progress-bar", "-f", "-o", dest.to_str().unwrap_or(""), url])
            .status()?;
        if !status.success() {
            log::warn!("Peringatan: Gagal mengunduh {}: status {:?}", fname, status.code());
        }
    }

    println!("  [OK] Model {} siap digunakan di disk!\n", preset.display_name);
    Ok(dest_dir)
}

/// Download streaming transducer weights by name, language, or preset
pub fn download_transducer_by_name(target_dir: &Path, name_or_lang: &str) -> Result<PathBuf> {
    let lower = name_or_lang.to_lowercase();
    let preset = if lower.contains("en") || lower.contains("english") {
        &TRANSDUCER_PRESETS[1]
    } else {
        &TRANSDUCER_PRESETS[0]
    };
    download_preset(target_dir, preset)
}

/// Download Whisper ONNX weights by tier or name
pub fn download_whisper_by_tier(target_dir: &Path, tier: &str) -> Result<PathBuf> {
    let lower = tier.to_lowercase();
    let preset = WHISPER_PRESETS
        .iter()
        .find(|p| lower.contains(p.id) || p.id == lower)
        .unwrap_or(&WHISPER_PRESETS[0]);
    download_preset(target_dir, preset)
}

/// Interactive model selection prompt presenting choices with explicit trade-offs.
pub fn prompt_model_choice(current_tier: &str, lang: &UiLanguage) -> String {
    println!("============================================================");
    println!("  Catetin - {}", lang.text("setup.title"));
    println!("============================================================");
    println!("  {}", lang.text("setup.model_desc"));
    println!();
    println!("  --- Model Whisper ONNX (DirectML GPU / Multithreaded CPU) ---");
    println!("  [1] base      (~160 MB | DirectML GPU / CPU INT8 - Responsif < 1.0s) [Disarankan]");
    println!("      • Akselerasi: DirectML GPU bawaan Windows & CPU AVX2");
    println!("      • Kecepatan : ~0.8s - 1.2s per kalimat, ideal untuk rapat real-time");
    println!("  [2] medium    (~980 MB | DirectML GPU / CPU INT8 - Presisi Tinggi Multibahasa & ID)");
    println!("      • Akselerasi: DirectML GPU bawaan Windows & CPU AVX2");
    println!("      • Akurasi   : Sangat tinggi untuk kosa kata teknis & bilingual");
    println!("  [3] small     (~480 MB | DirectML GPU / CPU INT8 - Seimbang)");
    println!("  [4] tiny      (~75 MB  | DirectML GPU / CPU INT8 - Ultra-Ringan)");
    println!("  [5] turbo     (~550 MB | DirectML GPU / CPU INT8 - Large-v3-Turbo)");
    println!("============================================================");

    let default_choice = match current_tier.to_lowercase().as_str() {
        "base" => '1',
        "medium" => '2',
        "small" => '3',
        "tiny" => '4',
        "turbo" | "large-v3-turbo" => '5',
        _ => '1',
    };

    let prompt = format!("  {} [{}]: ", lang.text("wizard.prompt_choice"), default_choice);
    let chosen_char = read_single_choice(&prompt, &['1', '2', '3', '4', '5'], default_choice);
    let chosen = match chosen_char {
        '1' => "base",
        '2' => "medium",
        '3' => "small",
        '4' => "tiny",
        '5' => "turbo",
        _ => "base",
    };

    println!("  -> {}: {}\n", lang.text("label.model_tier"), chosen);
    chosen.to_string()
}

/// Pre-flight dependency check verifying and automatically preparing missing models
pub fn check_and_prepare_dependencies(config: &mut AppConfig, lang: &UiLanguage) -> Result<()> {
    // 1. Ensure Whisper ONNX model exists
    if !has_whisper_onnx_installed(&config.model_dir, &config.model_tier) {
        println!("============================================================");
        println!("  Catetin - {}", lang.text("app.subtitle"));
        println!("  Bobot model Whisper belum ditemukan di disk lokal.");
        println!("  Pilih bobot model Whisper sesuai spesifikasi komputer Anda:");
        println!("============================================================");
        let chosen_tier = prompt_model_choice(&config.model_tier, lang);
        config.model_tier = chosen_tier;

        if !has_whisper_onnx_installed(&config.model_dir, &config.model_tier) {
            download_whisper_by_tier(&config.model_dir, &config.model_tier)?;
        }
        let _ = config.save();
    }

    // 2. Ensure mandatory Streaming Transducer exists (Pass 1 live captioning < 50ms)
    let transducer_dir = config.transducer_dir();
    let asr_lang = if config.asr_language.is_empty() { "auto" } else { &config.asr_language };
    if !has_transducer_installed_for_lang(&transducer_dir, asr_lang) {
        println!("============================================================");
        println!("  Catetin - Persiapan Streaming Transducer (Pass 1 Live Typing)");
        println!("  Model Transducer: {} (lang: {})", config.transducer_model, asr_lang);
        println!("  Transducer Zipformer RNN-T wajib untuk pengetikan < 50ms.");
        println!("  Mengunduh model dan runtime secara otomatis...");
        println!("============================================================");
        download_transducer_by_name(&config.model_dir, &config.transducer_model)?;
    }

    Ok(())
}
