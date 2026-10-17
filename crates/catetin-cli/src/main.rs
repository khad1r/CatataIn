use anyhow::Result;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use catetin_core::asr::SpeechRecognizer;
use catetin_core::asr::whisper_backend::WhisperEngine;
use catetin_core::audio::DualAudioCapture;
use catetin_core::config::{AppConfig, NoiseSuppression};
use catetin_core::events::{SpeakerId, TranscriptEvent};
use catetin_core::i18n::UiLanguage;
use catetin_core::logging::init_logger;
use catetin_core::pipeline::{CatetinPipeline, DiarizationMode};

fn read_line(prompt: &str) -> String {
    print!("{}", prompt);
    let _ = io::stdout().flush();
    let mut line = String::new();
    let _ = io::stdin().read_line(&mut line);
    line.trim().to_string()
}

/// Resolve expected model path on disk, prioritizing local ./models/ directory
/// before falling back to system AppData models cache directory.
fn resolve_model_path(config: &AppConfig) -> PathBuf {
    let candidate_models = Path::new("models").join(config.model_filename());
    let candidate_appdata = config.model_dir.join(config.model_filename());
    if candidate_models.exists() {
        candidate_models
    } else if candidate_appdata.exists() {
        candidate_appdata
    } else if Path::new("models").exists() || std::fs::create_dir_all("models").is_ok() {
        candidate_models
    } else {
        candidate_appdata
    }
}

/// Ensure Whisper GGML weights exist locally.
/// Automatically downloads weights via native curl if missing with zero user friction.
fn ensure_model_downloaded(target_path: &Path, url: &str, tier: &str) -> Result<()> {
    if target_path.exists() {
        if let Ok(m) = std::fs::metadata(target_path) {
            if m.len() > 1_000_000 {
                return Ok(());
            }
        }
    }

    if let Some(parent) = target_path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }

    println!("============================================================");
    println!("  Catetin - Unduh Bobot Model Whisper / Model Download");
    println!("  Model  : '{}'", tier);
    println!("  URL    : {}", url);
    println!("  Target : {}", target_path.display());
    println!("============================================================");

    let curl_bin = if cfg!(windows) { "curl.exe" } else { "curl" };
    let status = std::process::Command::new(curl_bin)
        .args([
            "-C",
            "-",
            "-L",
            "--progress-bar",
            "-f",
            "-o",
            target_path.to_str().unwrap_or("model.bin"),
            url,
        ])
        .status()
        .map_err(|e| {
            anyhow::anyhow!(
                "Tidak dapat menjalankan {}: {}. Pastikan curl terinstal.",
                curl_bin,
                e
            )
        })?;

    if !status.success() {
        anyhow::bail!(
            "Gagal mengunduh model Whisper (exit code: {:?}). Periksa koneksi internet.",
            status.code()
        );
    }

    let meta = std::fs::metadata(target_path)
        .map_err(|e| anyhow::anyhow!("Gagal membaca berkas model setelah diunduh: {}", e))?;
    if meta.len() < 1_000_000 {
        let _ = std::fs::remove_file(target_path);
        anyhow::bail!("Berkas model tidak lengkap atau rusak (ukuran terlalu kecil).");
    }

    println!("  [OK] Model Whisper ('{}') berhasil diunduh dan siap digunakan!\n", tier);
    Ok(())
}

/// Interactive model selection prompt presenting choices with explicit trade-offs (Req 3).
fn prompt_model_choice(current_tier: &str, lang: &UiLanguage) -> String {
    println!("============================================================");
    println!("  Catetin - {}", lang.text("setup.title"));
    println!("============================================================");
    println!("  {}", lang.text("setup.model_desc"));
    println!();
    println!("  [1] tiny   (~75 MB bobot  | RAM ~500 MB)");
    println!("      • Kecepatan : Sangat Cepat / Ultra Fast");
    println!("      • Akurasi   : Standar");
    println!("      • Trade-off : Hemat memori & CPU; akurasi lebih rendah pada aksen/noise.");
    println!();
    println!("  [2] base   (~140 MB bobot | RAM ~1 GB)");
    println!("      • Kecepatan : Cepat (Real-time)");
    println!("      • Akurasi   : Baik untuk percakapan umum");
    println!("      • Trade-off : Keseimbangan ringan antara beban CPU dan akurasi rapat.");
    println!();
    println!("  [3] small  (~460 MB bobot | RAM ~2 GB)");
    println!("      • Kecepatan : Moderat");
    println!("      • Akurasi   : Tinggi");
    println!("      • Trade-off : Menangkap istilah teknis dan aksen beragam lebih presisi.");
    println!();
    println!("  [4] turbo  (~547 MB bobot | Large-v3-Turbo q5_0) [Disarankan / Recommended]");
    println!("      • Kecepatan : Cepat (4-layer distilled decoder + q5_0 SIMD)");
    println!("      • Akurasi   : Setara Large-v3 (Sangat Tinggi untuk Indonesia & Inggris)");
    println!("      • Trade-off : Akurasi kelas Large-v3 namun 3x lebih kecil & jauh lebih cepat dari medium.");
    println!();
    println!("  [5] medium (~1.5 GB bobot | RAM ~5 GB)");
    println!("      • Kecepatan : Lambat (butuh prosesor kuat)");
    println!("      • Akurasi   : Maksimal (Full FP16)");
    println!("      • Trade-off : Presisi tinggi namun membutuhkan komputasi CPU besar.");
    println!("============================================================");

    let default_choice = match current_tier.to_lowercase().as_str() {
        "tiny" => "1",
        "base" => "2",
        "small" => "3",
        "turbo" | "large-v3-turbo" => "4",
        "medium" => "5",
        _ => "4",
    };

    let prompt = format!("  {} [{}]: ", lang.text("wizard.prompt_choice"), default_choice);
    let input = read_line(&prompt);
    let trimmed = input.trim().to_lowercase();
    let chosen: &str = match trimmed.as_str() {
        "1" | "tiny" => "tiny",
        "2" | "base" => "base",
        "3" | "small" => "small",
        "4" | "turbo" | "large-v3-turbo" => "turbo",
        "5" | "medium" => "medium",
        "" => match default_choice {
            "1" => "tiny",
            "2" => "base",
            "3" => "small",
            "5" => "medium",
            _ => "turbo",
        },
        _ => "turbo",
    };

    println!("  -> {}: {}\n", lang.text("label.model_tier"), chosen);
    chosen.to_string()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConfigMenuAction {
    StartMeeting,
    Exit,
}

fn cycle_language(lang: &mut UiLanguage, _forward: bool) {
    *lang = match *lang {
        UiLanguage::Indonesian => UiLanguage::English,
        _ => UiLanguage::Indonesian,
    };
}

fn cycle_mode(mode: &mut DiarizationMode, forward: bool) {
    *mode = if forward {
        match *mode {
            DiarizationMode::OnlineMeeting => DiarizationMode::InPerson,
            DiarizationMode::InPerson => DiarizationMode::Hybrid,
            DiarizationMode::Hybrid => DiarizationMode::OnlineMeeting,
        }
    } else {
        match *mode {
            DiarizationMode::OnlineMeeting => DiarizationMode::Hybrid,
            DiarizationMode::InPerson => DiarizationMode::OnlineMeeting,
            DiarizationMode::Hybrid => DiarizationMode::InPerson,
        }
    };
}

const TIERS: [&str; 5] = ["tiny", "base", "small", "turbo", "medium"];
fn cycle_tier(tier: &mut String, forward: bool) {
    let cur = tier.to_lowercase();
    let idx = TIERS.iter().position(|&t| t == cur).unwrap_or(1);
    let next_idx = if forward {
        (idx + 1) % TIERS.len()
    } else if idx == 0 {
        TIERS.len() - 1
    } else {
        idx - 1
    };
    *tier = TIERS[next_idx].to_string();
}

fn cycle_device(current: &mut Option<String>, devices: &[String], forward: bool) {
    let total = devices.len() + 1;
    let cur_idx = match current {
        None => 0,
        Some(name) => devices.iter().position(|d| d == name).map(|i| i + 1).unwrap_or(0),
    };
    let next_idx = if forward {
        (cur_idx + 1) % total
    } else if cur_idx == 0 {
        total - 1
    } else {
        cur_idx - 1
    };
    *current = if next_idx == 0 {
        None
    } else {
        Some(devices[next_idx - 1].clone())
    };
}

const MODELS: [&str; 6] = [
    "llama3.2:1b",
    "llama3.2:3b",
    "qwen2.5:3b",
    "mistral:7b",
    "gpt-4o-mini",
    "gemini-1.5-flash",
];
fn cycle_assistant_model(model: &mut String, forward: bool) {
    let cur = model.to_lowercase();
    let idx = MODELS.iter().position(|&m| m == cur).unwrap_or(0);
    let next_idx = if forward {
        (idx + 1) % MODELS.len()
    } else if idx == 0 {
        MODELS.len() - 1
    } else {
        idx - 1
    };
    *model = MODELS[next_idx].to_string();
}

fn cycle_noise_suppression(current: &mut NoiseSuppression, forward: bool) {
    *current = if forward {
        current.next()
    } else {
        current.prev()
    };
}

fn apply_menu_change(
    config: &mut AppConfig,
    idx: usize,
    input_devices: &[String],
    output_devices: &[String],
    forward: bool,
) {
    match idx {
        0 => {
            cycle_language(&mut config.ui_language, forward);
            config.asr_language = config.ui_language.code().to_string();
        }
        1 => cycle_mode(&mut config.diarization_mode, forward),
        2 => cycle_tier(&mut config.model_tier, forward),
        3 => cycle_device(&mut config.input_device_name, input_devices, forward),
        4 => cycle_device(&mut config.output_device_name, output_devices, forward),
        5 => cycle_noise_suppression(&mut config.noise_suppression, forward),
        6 => config.show_assistant = !config.show_assistant,
        7 => cycle_assistant_model(&mut config.assistant_model, forward),
        8 => config.verbose = !config.verbose,
        _ => {}
    }
}

fn render_config_menu(
    config: &AppConfig,
    selected: usize,
    _input_devices: &[String],
    _output_devices: &[String],
    config_file_display: &str,
) {
    let (term_w, _) = tui::get_terminal_size();
    let width = term_w.max(76).min(100);
    let inner_w = width.saturating_sub(2);
    let lang = &config.ui_language;

    let mut out = String::with_capacity(4096);
    out.push_str("\x1b[H"); // Cursor home

    // Header border
    let title = format!("┌─ Catetin v{} - {} ", env!("CARGO_PKG_VERSION"), lang.text("wizard.title"));
    let pad = inner_w.saturating_sub(title.chars().count());
    out.push_str(&title);
    out.push_str(&"─".repeat(pad));
    out.push_str("┐\n");

    let nav_hint = " Gunakan [↑/↓] Pilih Menu │ [←/→/Enter] Ubah Nilai │ [Esc] Keluar";
    out.push_str(&format!("│{}│\n", tui::fit_cell(nav_hint, inner_w)));
    out.push_str(&format!("├{}┤\n", "─".repeat(inner_w)));

    let model_path = resolve_model_path(config);
    let model_cached = model_path.exists();
    let model_tag = if model_cached {
        "\x1b[1;32m[✓ Siap / Ready]\x1b[0m"
    } else {
        "\x1b[1;33m[↓ Unduh saat simpan]\x1b[0m"
    };

    let mode_desc = match config.diarization_mode {
        DiarizationMode::OnlineMeeting => "Online (Zoom/Meet/Teams)",
        DiarizationMode::InPerson => "Tatap Muka (Roundtable 1 Mic)",
        DiarizationMode::Hybrid => "Hybrid (Voiceprint)",
    };

    let mic_name = config.input_device_name.as_deref().unwrap_or("Default (Sistem)");
    let loopback_name = config.output_device_name.as_deref().unwrap_or("Default (Sistem)");

    let copilot_desc = if config.show_assistant {
        "Aktif / Enabled (Split Pane)"
    } else {
        "Nonaktif / Disabled (Full Width)"
    };

    let verbose_desc = if config.verbose {
        "Aktif / True (Mirror stderr)"
    } else {
        "Nonaktif / False"
    };

    let items = [
        (format!("1. {}", lang.text("label.language")), format!("{} ({})", config.ui_language.display_name(), config.ui_language.code())),
        (format!("2. {}", lang.text("label.mode")), mode_desc.to_string()),
        (format!("3. {}", lang.text("label.model_tier")), format!("{} ({}) {}", config.model_tier, config.model_filename(), model_tag)),
        (format!("4. {}", lang.text("label.microphone")), mic_name.to_string()),
        (format!("5. {}", lang.text("label.loopback")), loopback_name.to_string()),
        ("6. Peredam Bising (Noise Cancel)".to_string(), config.noise_suppression.display_label(lang).to_string()),
        ("7. Panel Asisten AI Copilot".to_string(), copilot_desc.to_string()),
        ("8. Model LLM Copilot".to_string(), config.assistant_model.clone()),
        ("9. Log Detail (Verbose)".to_string(), verbose_desc.to_string()),
    ];

    out.push_str(&format!("│{}│\n", tui::fit_cell("", inner_w)));

    for (i, (label, val)) in items.iter().enumerate() {
        let is_sel = selected == i;
        let prefix = if is_sel { "\x1b[1;36m ▶ " } else { "   " };
        let suffix = if is_sel { "\x1b[0m" } else { "" };
        let formatted = format!("{}{} : < {} >{}", prefix, tui::fit_cell(label, 32), val, suffix);
        out.push_str(&format!("│{}│\n", tui::fit_cell(&formatted, inner_w)));
    }

    out.push_str(&format!("│{}│\n", tui::fit_cell("", inner_w)));
    out.push_str(&format!("├{}┤\n", "─".repeat(inner_w)));

    let actions = [
        ("▶ Simpan & Mulai Rapat / Save & Start Meeting", 9),
        ("💾 Simpan Konfigurasi & Keluar / Save Config & Exit", 10),
        ("❌ Batal & Keluar / Cancel & Exit", 11),
    ];

    for (act_text, act_idx) in actions {
        let is_sel = selected == act_idx;
        let prefix = if is_sel { "\x1b[1;32m ▶ [ " } else { "   [ " };
        let suffix = if is_sel { " ]\x1b[0m" } else { " ]" };
        let formatted = format!("{}{}{}", prefix, act_text, suffix);
        out.push_str(&format!("│{}│\n", tui::fit_cell(&formatted, inner_w)));
    }

    out.push_str(&format!("├{}┤\n", "─".repeat(inner_w)));
    let footer = format!(" Berkas Konfigurasi / Config File: {}", config_file_display);
    out.push_str(&format!("│{}│\n", tui::fit_cell(&footer, inner_w)));
    out.push_str(&format!("└{}┘\n", "─".repeat(inner_w)));

    let mut stdout = io::stdout();
    let _ = stdout.write_all(out.as_bytes());
    let _ = stdout.flush();
}

/// Interactive terminal configuration menu with Up/Down arrow key navigation.
fn run_interactive_wizard(config: &mut AppConfig, config_file_display: &str) -> Result<ConfigMenuAction> {
    #[cfg(windows)]
    let _guard = console::RawConsoleGuard::enter();

    print!("\x1b[2J\x1b[H\x1b[?25l");
    let _ = io::stdout().flush();

    let input_devices = catetin_core::audio::list_input_devices();
    let output_devices = catetin_core::audio::list_output_devices();

    let mut selected: usize = 0;

    loop {
        render_config_menu(config, selected, &input_devices, &output_devices, config_file_display);

        let key = kb::read_key_blocking();
        match key {
            kb::Key::Up => {
                selected = if selected == 0 { 11 } else { selected - 1 };
            }
            kb::Key::Down => {
                selected = (selected + 1) % 12;
            }
            kb::Key::Left => {
                apply_menu_change(config, selected, &input_devices, &output_devices, false);
            }
            kb::Key::Right => {
                apply_menu_change(config, selected, &input_devices, &output_devices, true);
            }
            kb::Key::Enter | kb::Key::Char(' ') => {
                if selected < 9 {
                    apply_menu_change(config, selected, &input_devices, &output_devices, true);
                } else if selected == 9 {
                    config.save()?;
                    let target_path = resolve_model_path(config);
                    if !target_path.exists() {
                        print!("\x1b[?25h\n\n");
                        let _ = io::stdout().flush();
                        ensure_model_downloaded(&target_path, config.model_url(), &config.model_tier)?;
                    }
                    print!("\x1b[?25h\n");
                    let _ = io::stdout().flush();
                    return Ok(ConfigMenuAction::StartMeeting);
                } else if selected == 10 {
                    config.save()?;
                    print!("\x1b[?25h\n");
                    let _ = io::stdout().flush();
                    println!("============================================================");
                    println!("  [✓] Konfigurasi berhasil disimpan ke: {}", config_file_display);
                    println!("============================================================");
                    return Ok(ConfigMenuAction::Exit);
                } else if selected == 11 {
                    print!("\x1b[?25h\n");
                    let _ = io::stdout().flush();
                    return Ok(ConfigMenuAction::Exit);
                }
            }
            kb::Key::Escape | kb::Key::Char('q') | kb::Key::Char('Q') => {
                print!("\x1b[?25h\n");
                let _ = io::stdout().flush();
                return Ok(ConfigMenuAction::Exit);
            }
            kb::Key::Char('s') | kb::Key::Char('S') => {
                config.save()?;
                let target_path = resolve_model_path(config);
                if !target_path.exists() {
                    print!("\x1b[?25h\n\n");
                    let _ = io::stdout().flush();
                    ensure_model_downloaded(&target_path, config.model_url(), &config.model_tier)?;
                }
                print!("\x1b[?25h\n");
                let _ = io::stdout().flush();
                return Ok(ConfigMenuAction::StartMeeting);
            }
            kb::Key::Char('w') | kb::Key::Char('W') => {
                config.save()?;
                print!("\x1b[?25h\n");
                let _ = io::stdout().flush();
                println!("============================================================");
                println!("  [✓] Konfigurasi berhasil disimpan ke: {}", config_file_display);
                println!("============================================================");
                return Ok(ConfigMenuAction::Exit);
            }
            _ => {}
        }
    }
}

fn print_help(lang: &UiLanguage) {
    println!("Catetin CLI - {}", lang.text("app.subtitle"));
    println!();
    println!("{} catetin [OPTIONS]", lang.text("cli.usage"));
    println!();
    println!("{}", lang.text("cli.options"));
    println!("  --config, -c [path]      {}", lang.text("cli.help_config"));
    println!("  --setup, setup           Periksa & unduh dependensi / Check & download dependencies");
    println!("  --check, check           Verifikasi kesiapan sistem / Verify system readiness");
    println!("  --assistant              Enable split-screen AI meeting copilot pane");
    println!("  --no-assistant           Disable split pane; run in standard scrolling mode");
    println!("  --verbose, -v            {}", lang.text("cli.help_verbose"));
    println!("  --help, -h               Show this help message / Tampilkan bantuan");
    println!("  --version                Show version information / Tampilkan versi");
    println!();
    println!("Examples / Contoh:");
    println!("  catetin                       # Jalankan dengan konfigurasi tersimpan");
    println!("  catetin setup                 # Wizard interaktif persiapan model & dependensi");
    println!("  catetin check                 # Verifikasi kesiapan sistem (preflight check)");
    println!("  catetin --config              # Jalankan wizard konfigurasi interaktif");
    println!("  catetin --no-assistant        # Jalankan mode terminal scroll biasa");
}

fn format_ms(ms: u64) -> String {
    let total_secs = ms / 1000;
    let s = total_secs % 60;
    let m = (total_secs / 60) % 60;
    let h = (total_secs / 3600) % 24;
    format!("[{:02}:{:02}:{:02}]", h, m, s)
}

fn default_transcript_path() -> PathBuf {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let filename = format!("meeting_{}.md", now);
    AppConfig::config_file_path()
        .parent()
        .map(|p| p.join(&filename))
        .unwrap_or_else(|| PathBuf::from(&filename))
}

fn save_markdown_transcript(
    target_path: &Path,
    segments: &[(String, String, String)],
    topics: &[String],
    actions: &[String],
    lang: &UiLanguage,
) -> Result<()> {
    if let Some(parent) = target_path.parent() {
        if !parent.as_os_str().is_empty() {
            let _ = std::fs::create_dir_all(parent);
        }
    }

    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    let mut content = String::new();
    content.push_str(&format!("# Catetin - {}\n\n", lang.text("app.subtitle")));
    content.push_str(&format!("- Date/Timestamp: {}\n", now));
    content.push_str(&format!("- Total Utterances: {}\n\n", segments.len()));

    if !topics.is_empty() || !actions.is_empty() {
        content.push_str("## Executive Summary & Action Items\n\n");
        if !topics.is_empty() {
            content.push_str("### Key Topics\n");
            for t in topics {
                content.push_str(&format!("- {}\n", t));
            }
            content.push('\n');
        }
        if !actions.is_empty() {
            content.push_str("### Action Items\n");
            for a in actions {
                content.push_str(&format!("- {}\n", a));
            }
            content.push('\n');
        }
    }

    content.push_str("## Transcript\n\n");
    for (time, speaker, text) in segments {
        content.push_str(&format!("- **{} [{}]**: {}\n", time, speaker, text));
    }

    std::fs::write(target_path, content)?;
    Ok(())
}

// ponytail: Native OS Save Dialog via comdlg32 on Windows and zenity on Linux.
// Ceiling: Native GUI file picker on desktop systems.
// Upgrade path: fallback to interactive terminal prompt automatically when running headless or if dialog is cancelled.
mod file_dialog {
    use std::path::PathBuf;

    #[cfg(windows)]
    pub fn pick_save_file(default_filename: &str, title: &str) -> Option<PathBuf> {
        use std::os::windows::ffi::{OsStrExt, OsStringExt};

        #[repr(C)]
        #[allow(non_snake_case)]
        struct OPENFILENAMEW {
            lStructSize: u32,
            hwndOwner: usize,
            hInstance: usize,
            lpstrFilter: *const u16,
            lpstrCustomFilter: *mut u16,
            nMaxCustFilter: u32,
            nFilterIndex: u32,
            lpstrFile: *mut u16,
            nMaxFile: u32,
            lpstrFileTitle: *mut u16,
            nMaxFileTitle: u32,
            lpstrInitialDir: *const u16,
            lpstrTitle: *const u16,
            Flags: u32,
            nFileOffset: u16,
            nFileExtension: u16,
            lpstrDefExt: *const u16,
            lCustData: usize,
            lpfnHook: usize,
            lpTemplateName: *const u16,
            pvReserved: *mut std::ffi::c_void,
            dwReserved: u32,
            FlagsEx: u32,
        }

        #[link(name = "comdlg32")]
        extern "system" {
            fn GetSaveFileNameW(lpofn: *mut OPENFILENAMEW) -> i32;
        }

        const OFN_OVERWRITEPROMPT: u32 = 0x00000002;
        const OFN_PATHMUSTEXIST: u32 = 0x00000800;

        let mut buffer = [0u16; 1024];
        let default_wide: Vec<u16> = std::ffi::OsStr::new(default_filename)
            .encode_wide()
            .chain(std::iter::once(0))
            .collect();
        let len = default_wide.len().min(buffer.len() - 1);
        buffer[..len].copy_from_slice(&default_wide[..len]);

        let filter_wide: Vec<u16> = "Markdown (*.md)\0*.md\0All Files (*.*)\0*.*\0\0"
            .encode_utf16()
            .collect();
        let title_wide: Vec<u16> = std::ffi::OsStr::new(title)
            .encode_wide()
            .chain(std::iter::once(0))
            .collect();
        let def_ext_wide: Vec<u16> = "md\0".encode_utf16().collect();

        let mut ofn: OPENFILENAMEW = unsafe { std::mem::zeroed() };
        ofn.lStructSize = std::mem::size_of::<OPENFILENAMEW>() as u32;
        ofn.lpstrFilter = filter_wide.as_ptr();
        ofn.nFilterIndex = 1;
        ofn.lpstrFile = buffer.as_mut_ptr();
        ofn.nMaxFile = buffer.len() as u32;
        ofn.lpstrTitle = title_wide.as_ptr();
        ofn.Flags = OFN_OVERWRITEPROMPT | OFN_PATHMUSTEXIST;
        ofn.lpstrDefExt = def_ext_wide.as_ptr();

        let success = unsafe { GetSaveFileNameW(&mut ofn) };
        if success != 0 {
            let end = buffer.iter().position(|&c| c == 0).unwrap_or(buffer.len());
            let os_str = std::ffi::OsString::from_wide(&buffer[..end]);
            Some(PathBuf::from(os_str))
        } else {
            None
        }
    }

    #[cfg(all(not(windows), target_os = "linux"))]
    pub fn pick_save_file(default_filename: &str, title: &str) -> Option<PathBuf> {
        let output = std::process::Command::new("zenity")
            .args([
                "--file-selection",
                "--save",
                "--confirm-overwrite",
                "--title",
                title,
                "--filename",
                default_filename,
            ])
            .output()
            .ok()?;

        if output.status.success() {
            let path_str = String::from_utf8_lossy(&output.stdout).trim().to_string();
            if !path_str.is_empty() {
                return Some(PathBuf::from(path_str));
            }
        }
        None
    }

    #[cfg(not(any(windows, target_os = "linux")))]
    pub fn pick_save_file(_default_filename: &str, _title: &str) -> Option<PathBuf> {
        None
    }
}

// ponytail: native curl.exe for OpenAI/Ollama REST completions without reqwest/hyper/tokio bloat.
// Ceiling: subprocess invocation (~30ms overhead).
// Upgrade path: add ureq or streaming client if real-time word-by-word LLM token streaming is demanded.
mod assistant {
    use anyhow::Result;

    pub fn query_llm(
        endpoint: &str,
        model: &str,
        api_key: Option<&str>,
        system_prompt: &str,
        user_prompt: &str,
    ) -> Result<String> {
        let url = if endpoint.ends_with("/chat/completions") {
            endpoint.to_string()
        } else if endpoint.ends_with('/') {
            format!("{}chat/completions", endpoint)
        } else {
            format!("{}/chat/completions", endpoint)
        };

        let payload = serde_json::json!({
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.2
        });

        let payload_str = serde_json::to_string(&payload)?;

        let curl_bin = if cfg!(windows) { "curl.exe" } else { "curl" };
        let mut cmd = std::process::Command::new(curl_bin);
        cmd.args(["-s", "-X", "POST", &url, "-H", "Content-Type: application/json"]);
        if let Some(key) = api_key {
            if !key.is_empty() {
                cmd.args(["-H", &format!("Authorization: Bearer {}", key)]);
            }
        }
        cmd.args(["-d", &payload_str]);

        let output = cmd
            .output()
            .map_err(|e| anyhow::anyhow!("Failed to run curl: {}", e))?;

        if !output.status.success() {
            let err_text = String::from_utf8_lossy(&output.stderr);
            return Err(anyhow::anyhow!("LLM request failed: {}", err_text));
        }

        let response_str = String::from_utf8_lossy(&output.stdout);
        let val: serde_json::Value = serde_json::from_str(&response_str)
            .map_err(|_| anyhow::anyhow!("Invalid JSON response from LLM: {}", response_str))?;

        if let Some(content) = val["choices"][0]["message"]["content"].as_str() {
            Ok(content.trim().to_string())
        } else if let Some(err_msg) = val["error"]["message"].as_str() {
            Err(anyhow::anyhow!("LLM error: {}", err_msg))
        } else {
            Err(anyhow::anyhow!("Unknown LLM response shape"))
        }
    }

    pub fn generate_summary(
        endpoint: &str,
        model: &str,
        api_key: Option<&str>,
        transcript_text: &str,
    ) -> Result<(Vec<String>, Vec<String>)> {
        let sys = "You are an executive meeting copilot. Analyze the meeting transcript.\nReturn ONLY two sections formatted exactly like this:\nKEY TOPICS:\n- Topic 1\n- Topic 2\nACTION ITEMS:\n[ ] Person: Action 1\n[ ] Person: Action 2";
        let user = format!("Transcript:\n{}", transcript_text);

        let reply = query_llm(endpoint, model, api_key, sys, &user)?;

        let mut topics = Vec::new();
        let mut actions = Vec::new();
        let mut in_topics = false;
        let mut in_actions = false;

        for line in reply.lines() {
            let trimmed = line.trim();
            if trimmed.to_uppercase().contains("KEY TOPICS") {
                in_topics = true;
                in_actions = false;
            } else if trimmed.to_uppercase().contains("ACTION ITEMS") {
                in_actions = true;
                in_topics = false;
            } else if in_topics && !trimmed.is_empty() {
                let clean = trimmed.trim_start_matches(|c| c == '-' || c == '*' || c == '•').trim();
                if !clean.is_empty() {
                    topics.push(clean.to_string());
                }
            } else if in_actions && !trimmed.is_empty() {
                let clean = trimmed.trim_start_matches(|c| c == '-' || c == '*' || c == '•').trim();
                if !clean.is_empty() {
                    actions.push(clean.to_string());
                }
            }
        }

        Ok((topics, actions))
    }

    pub fn ask_assistant(
        endpoint: &str,
        model: &str,
        api_key: Option<&str>,
        transcript_text: &str,
        question: &str,
    ) -> Result<String> {
        let sys = "You are an AI meeting assistant. Answer the user's question directly based ONLY on the meeting transcript. Be concise, accurate, and direct.";
        let user = format!("Meeting Transcript:\n{}\n\nQuestion: {}", transcript_text, question);

        query_llm(endpoint, model, api_key, sys, &user)
    }
}

// Keyboard input handling for hotkeys and interactive navigation
mod kb {
    #[cfg(windows)]
    extern "C" {
        fn _kbhit() -> i32;
        fn _getch() -> i32;
    }

    #[cfg(windows)]
    extern "system" {
        fn GetStdHandle(nStdHandle: u32) -> isize;
        fn GetNumberOfConsoleInputEvents(hConsoleInput: isize, lpcNumberOfEvents: *mut u32) -> i32;
        fn ReadConsoleInputW(
            hConsoleInput: isize,
            lpBuffer: *mut InputRecord,
            nLength: u32,
            lpNumberOfEventsRead: *mut u32,
        ) -> i32;
    }

    #[cfg(windows)]
    const STD_INPUT_HANDLE: u32 = 0xFFFFFFF6;
    #[cfg(windows)]
    const KEY_EVENT: u16 = 0x0001;
    #[cfg(windows)]
    const VK_BACK: u16 = 0x08;
    #[cfg(windows)]
    const VK_RETURN: u16 = 0x0D;
    #[cfg(windows)]
    const VK_ESCAPE: u16 = 0x1B;
    #[cfg(windows)]
    const VK_SPACE: u16 = 0x20;
    #[cfg(windows)]
    const VK_LEFT: u16 = 0x25;
    #[cfg(windows)]
    const VK_UP: u16 = 0x26;
    #[cfg(windows)]
    const VK_RIGHT: u16 = 0x27;
    #[cfg(windows)]
    const VK_DOWN: u16 = 0x28;

    #[cfg(windows)]
    #[repr(C)]
    #[derive(Copy, Clone, Default)]
    struct KeyEventRecord {
        b_key_down: i32,
        w_repeat_count: u16,
        w_virtual_key_code: u16,
        w_virtual_scan_code: u16,
        u_char: u16,
        dw_control_key_state: u32,
    }

    #[cfg(windows)]
    #[repr(C)]
    #[derive(Copy, Clone, Default)]
    struct InputRecord {
        event_type: u16,
        _pad: u16,
        key_event: KeyEventRecord,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum Key {
        Up,
        Down,
        Left,
        Right,
        Enter,
        Escape,
        Char(char),
        Other,
    }

    pub fn poll_key() -> Option<i32> {
        #[cfg(windows)]
        unsafe {
            let in_handle = GetStdHandle(STD_INPUT_HANDLE);
            let mut num_events: u32 = 0;
            if GetNumberOfConsoleInputEvents(in_handle, &mut num_events) != 0 {
                while num_events > 0 {
                    let mut rec = InputRecord::default();
                    let mut read: u32 = 0;
                    if ReadConsoleInputW(in_handle, &mut rec, 1, &mut read) == 0 || read == 0 {
                        break;
                    }
                    num_events = num_events.saturating_sub(1);
                    if rec.event_type == KEY_EVENT && rec.key_event.b_key_down != 0 {
                        let ch = rec.key_event.u_char;
                        if ch == 27 {
                            // If Esc is immediately followed by '[' (ANSI CSI sequence from ConPTY),
                            // drain the remaining CSI sequence so arrow keys never trigger letter hotkeys.
                            let mut follow_events: u32 = 0;
                            if GetNumberOfConsoleInputEvents(in_handle, &mut follow_events) != 0
                                && follow_events > 0
                            {
                                while follow_events > 0 {
                                    let mut next_rec = InputRecord::default();
                                    let mut next_read: u32 = 0;
                                    if ReadConsoleInputW(in_handle, &mut next_rec, 1, &mut next_read) == 0
                                        || next_read == 0
                                    {
                                        break;
                                    }
                                    follow_events = follow_events.saturating_sub(1);
                                }
                                num_events = 0;
                                continue;
                            }
                            return Some(27);
                        } else if ch != 0 {
                            return Some(ch as i32);
                        } else {
                            match rec.key_event.w_virtual_key_code {
                                VK_ESCAPE => return Some(27),
                                VK_RETURN => return Some(13),
                                VK_SPACE => return Some(b' ' as i32),
                                VK_BACK => return Some(8),
                                _ => {}
                            }
                        }
                    }
                }
                None
            } else if _kbhit() != 0 {
                let ch = _getch();
                if ch == 0 || ch == 0xE0 {
                    if _kbhit() != 0 {
                        let _ = _getch();
                    }
                    None
                } else {
                    Some(ch)
                }
            } else {
                None
            }
        }
        #[cfg(not(windows))]
        {
            None
        }
    }

    pub fn read_key_blocking() -> Key {
        #[cfg(windows)]
        unsafe {
            let in_handle = GetStdHandle(STD_INPUT_HANDLE);
            let mut num_events: u32 = 0;
            if GetNumberOfConsoleInputEvents(in_handle, &mut num_events) != 0 {
                loop {
                    let mut rec = InputRecord::default();
                    let mut read: u32 = 0;
                    if ReadConsoleInputW(in_handle, &mut rec, 1, &mut read) == 0 || read == 0 {
                        return Key::Other;
                    }
                    if rec.event_type == KEY_EVENT && rec.key_event.b_key_down != 0 {
                        match rec.key_event.w_virtual_key_code {
                            VK_UP => return Key::Up,
                            VK_DOWN => return Key::Down,
                            VK_LEFT => return Key::Left,
                            VK_RIGHT => return Key::Right,
                            VK_RETURN => return Key::Enter,
                            VK_ESCAPE => return Key::Escape,
                            VK_SPACE => return Key::Char(' '),
                            _ => {}
                        }
                        let ch = rec.key_event.u_char;
                        if ch != 0 {
                            match ch as u8 {
                                13 | 10 => return Key::Enter,
                                27 => return Key::Escape,
                                b'k' | b'K' => return Key::Up,
                                b'j' | b'J' => return Key::Down,
                                b'h' | b'H' => return Key::Left,
                                b'l' | b'L' => return Key::Right,
                                b' ' => return Key::Char(' '),
                                b @ 32..=126 => return Key::Char(b as char),
                                _ => {}
                            }
                        }
                    }
                }
            }

            let ch = _getch();
            if ch < 0 {
                Key::Other
            } else if ch == 0 || ch == 0xE0 {
                let ext = _getch();
                match ext {
                    72 => Key::Up,
                    80 => Key::Down,
                    75 => Key::Left,
                    77 => Key::Right,
                    _ => Key::Other,
                }
            } else {
                let b = ch as u8;
                match b {
                    13 | 10 => Key::Enter,
                    27 => Key::Escape,
                    b'k' | b'K' => Key::Up,
                    b'j' | b'J' => Key::Down,
                    b'h' | b'H' => Key::Left,
                    b'l' | b'L' => Key::Right,
                    b' ' => Key::Char(' '),
                    32..=126 => Key::Char(b as char),
                    _ => Key::Other,
                }
            }
        }
        #[cfg(not(windows))]
        {
            use std::io::Read;
            let mut buf = [0u8; 4];
            let mut stdin = std::io::stdin();
            if let Ok(n) = stdin.read(&mut buf) {
                if n >= 3 && buf[0] == 27 && buf[1] == b'[' {
                    match buf[2] {
                        b'A' => Key::Up,
                        b'B' => Key::Down,
                        b'C' => Key::Right,
                        b'D' => Key::Left,
                        _ => Key::Other,
                    }
                } else if n >= 1 {
                    match buf[0] {
                        13 | 10 => Key::Enter,
                        27 => Key::Escape,
                        b'k' | b'K' => Key::Up,
                        b'j' | b'J' => Key::Down,
                        b'h' | b'H' => Key::Left,
                        b'l' | b'L' => Key::Right,
                        c if (32..=126).contains(&c) => Key::Char(c as char),
                        _ => Key::Other,
                    }
                } else {
                    Key::Other
                }
            } else {
                Key::Other
            }
        }
    }
}

#[cfg(windows)]
pub mod console {
    extern "system" {
        fn GetStdHandle(nStdHandle: u32) -> isize;
        fn GetConsoleMode(hConsoleHandle: isize, lpMode: *mut u32) -> i32;
        fn SetConsoleMode(hConsoleHandle: isize, dwMode: u32) -> i32;
        fn FlushConsoleInputBuffer(hConsoleInput: isize) -> i32;
    }

    const STD_INPUT_HANDLE: u32 = 0xFFFFFFF6;
    const STD_OUTPUT_HANDLE: u32 = 0xFFFFFFF5;
    const ENABLE_VIRTUAL_TERMINAL_PROCESSING: u32 = 0x0004;

    const ENABLE_PROCESSED_INPUT: u32 = 0x0001;
    const ENABLE_LINE_INPUT: u32 = 0x0002;
    const ENABLE_ECHO_INPUT: u32 = 0x0004;
    const ENABLE_MOUSE_INPUT: u32 = 0x0010;
    const ENABLE_QUICK_EDIT_MODE: u32 = 0x0040;
    const ENABLE_EXTENDED_FLAGS: u32 = 0x0080;
    const ENABLE_VIRTUAL_TERMINAL_INPUT: u32 = 0x0200;

    pub struct RawConsoleGuard {
        in_handle: isize,
        out_handle: isize,
        orig_in_mode: Option<u32>,
        orig_out_mode: Option<u32>,
    }

    impl RawConsoleGuard {
        pub fn enter() -> Self {
            unsafe {
                let in_handle = GetStdHandle(STD_INPUT_HANDLE);
                let out_handle = GetStdHandle(STD_OUTPUT_HANDLE);

                let mut orig_out_mode = None;
                let mut out_mode: u32 = 0;
                if GetConsoleMode(out_handle, &mut out_mode) != 0 {
                    orig_out_mode = Some(out_mode);
                    let _ = SetConsoleMode(out_handle, out_mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING);
                }

                let mut orig_in_mode = None;
                let mut in_mode: u32 = 0;
                if GetConsoleMode(in_handle, &mut in_mode) != 0 {
                    orig_in_mode = Some(in_mode);
                    // Disable line/echo/mouse/quick-edit/VT-input so keys arrive immediately as raw KEY_EVENTs
                    // and clicking inside the console window never freezes execution (QuickEdit).
                    let raw_in_mode = (in_mode
                        & !(ENABLE_PROCESSED_INPUT
                            | ENABLE_LINE_INPUT
                            | ENABLE_ECHO_INPUT
                            | ENABLE_MOUSE_INPUT
                            | ENABLE_QUICK_EDIT_MODE
                            | ENABLE_VIRTUAL_TERMINAL_INPUT))
                        | ENABLE_EXTENDED_FLAGS;
                    let _ = SetConsoleMode(in_handle, raw_in_mode);
                    let _ = FlushConsoleInputBuffer(in_handle);
                }

                Self {
                    in_handle,
                    out_handle,
                    orig_in_mode,
                    orig_out_mode,
                }
            }
        }
    }

    impl Drop for RawConsoleGuard {
        fn drop(&mut self) {
            unsafe {
                if let Some(in_mode) = self.orig_in_mode {
                    let _ = SetConsoleMode(self.in_handle, in_mode);
                }
                if let Some(out_mode) = self.orig_out_mode {
                    let _ = SetConsoleMode(self.out_handle, out_mode);
                }
            }
        }
    }
}

mod tui {
    use super::*;

    pub struct AssistantState {
        pub topics: Vec<String>,
        pub actions: Vec<String>,
        pub status_msg: Option<String>,
        pub qa_history: Vec<(String, String)>,
    }

    impl Default for AssistantState {
        fn default() -> Self {
            Self {
                topics: Vec::new(),
                actions: Vec::new(),
                status_msg: None,
                qa_history: Vec::new(),
            }
        }
    }

    pub fn get_terminal_size() -> (usize, usize) {
        #[cfg(windows)]
        {
            #[repr(C)]
            struct COORD {
                x: i16,
                y: i16,
            }
            #[repr(C)]
            #[allow(non_snake_case)]
            struct SMALL_RECT {
                left: i16,
                top: i16,
                right: i16,
                bottom: i16,
            }
            #[repr(C)]
            #[allow(non_snake_case)]
            struct CONSOLE_SCREEN_BUFFER_INFO {
                dwSize: COORD,
                dwCursorPosition: COORD,
                wAttributes: u16,
                srWindow: SMALL_RECT,
                dwMaximumWindowSize: COORD,
            }
            extern "system" {
                fn GetStdHandle(nStdHandle: u32) -> isize;
                fn GetConsoleScreenBufferInfo(
                    hConsoleOutput: isize,
                    lpConsoleScreenBufferInfo: *mut CONSOLE_SCREEN_BUFFER_INFO,
                ) -> i32;
            }

            const STD_OUTPUT_HANDLE: u32 = 0xFFFFFFF5;
            unsafe {
                let handle = GetStdHandle(STD_OUTPUT_HANDLE);
                let mut info = std::mem::zeroed::<CONSOLE_SCREEN_BUFFER_INFO>();
                if GetConsoleScreenBufferInfo(handle, &mut info) != 0 {
                    let w = (info.srWindow.right - info.srWindow.left + 1).max(75) as usize;
                    let h = (info.srWindow.bottom - info.srWindow.top + 1).max(20) as usize;
                    return (w, h);
                }
            }
        }

        let w = std::env::var("COLUMNS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(110);
        let h = std::env::var("LINES")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(30);
        (w.max(75), h.max(20))
    }

    pub fn visible_width(s: &str) -> usize {
        let mut in_escape = false;
        let mut width = 0;
        for c in s.chars() {
            if c == '\x1b' {
                in_escape = true;
            } else if in_escape {
                if c == 'm' {
                    in_escape = false;
                }
            } else {
                width += 1;
            }
        }
        width
    }

    pub fn fit_cell(s: &str, target_width: usize) -> String {
        let cur = visible_width(s);
        if cur > target_width {
            let mut out = String::new();
            let mut in_esc = false;
            let mut count = 0;
            for c in s.chars() {
                if c == '\x1b' {
                    in_esc = true;
                    out.push(c);
                } else if in_esc {
                    if c == 'm' {
                        in_esc = false;
                    }
                    out.push(c);
                } else if count < target_width.saturating_sub(1) {
                    out.push(c);
                    count += 1;
                } else {
                    out.push('…');
                    break;
                }
            }
            out.push_str("\x1b[0m");
            let final_cur = visible_width(&out);
            if final_cur < target_width {
                out.push_str(&" ".repeat(target_width - final_cur));
            }
            out
        } else {
            format!("{}{}", s, " ".repeat(target_width - cur))
        }
    }

    pub fn textwrap_simple(text: &str, max_len: usize) -> Vec<String> {
        let mut lines = Vec::new();
        let words = text.split_whitespace();
        let mut cur_line = String::new();

        for w in words {
            if cur_line.is_empty() {
                cur_line.push_str(w);
            } else if cur_line.len() + 1 + w.len() <= max_len {
                cur_line.push(' ');
                cur_line.push_str(w);
            } else {
                lines.push(cur_line);
                cur_line = w.to_string();
            }
        }
        if !cur_line.is_empty() {
            lines.push(cur_line);
        }
        lines
    }

    pub fn build_waveform(history: &[f32], max_bars: usize) -> (String, bool) {
        let bars = [' ', ' ', '▂', '▃', '▄', '▅', '▆', '▇', '█'];
        let mut out = String::new();
        let mut is_active = false;

        let slice = if history.len() > max_bars {
            &history[history.len() - max_bars..]
        } else {
            history
        };

        for &val in slice {
            if val > 0.0012 {
                is_active = true;
            }
            let idx = if val < 0.0005 {
                0
            } else if val < 0.0012 {
                1
            } else if val < 0.0030 {
                2
            } else if val < 0.0060 {
                3
            } else if val < 0.0120 {
                4
            } else if val < 0.0240 {
                5
            } else if val < 0.0450 {
                6
            } else if val < 0.0800 {
                7
            } else {
                8
            };
            out.push(bars[idx]);
        }

        if slice.len() < max_bars {
            out.push_str(&" ".repeat(max_bars - slice.len()));
        }

        (out, is_active)
    }

    pub fn render_copilot_tui(
        config: &AppConfig,
        lang: &UiLanguage,
        mode: DiarizationMode,
        mic_name: &str,
        loopback_name: &str,
        paused: bool,
        left_lines: &[String],
        assistant: &AssistantState,
        input_mode: bool,
        input_buffer: &str,
        mic_level: f32,
        loopback_level: f32,
        wave_history: &[f32],
        is_speaking: bool,
        transcribing_count: u32,
        spinner_idx: usize,
    ) {
        let (term_w, term_h) = get_terminal_size();
        // Leave 1 column margin on the right to prevent Windows Console auto-wrap + newline double-advancing
        let total_w = term_w.saturating_sub(1).max(75);
        let inner_w = total_w.saturating_sub(2);
        let show_right_pane = config.show_assistant;

        let (left_w, right_w) = if show_right_pane {
            let content_space = inner_w.saturating_sub(1);
            let l_w = (content_space * 52) / 100;
            let r_w = content_space.saturating_sub(l_w);
            (l_w, r_w)
        } else {
            (inner_w, 0)
        };

        let content_rows = term_h.saturating_sub(10).max(8);
        let upper_rows = (content_rows * 55) / 100;

        let spinner_frames = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'];
        let spin_char = spinner_frames[spinner_idx % spinner_frames.len()];

        let mut out = String::with_capacity(8192);
        out.push_str("\x1b[H"); // Cursor home

        // 1. Top border
        let status_str = if paused {
            "[PAUSED / DIJEDA]"
        } else if transcribing_count > 0 {
            "[⟳ TRANSCRIBING / MEMPROSES TEKS]"
        } else if is_speaking {
            "[● RECORDING SPEECH / MEREKAM SUARA]"
        } else {
            "[100% OFFLINE / LIVE]"
        };
        let app_title = if show_right_pane { "Catetin Copilot" } else { "Catetin" };
        let title_prefix = format!("┌─ {} v{} {} ", app_title, env!("CARGO_PKG_VERSION"), status_str);
        let title_pad = inner_w.saturating_sub(title_prefix.chars().count());
        out.push_str(&title_prefix);
        out.push_str(&"─".repeat(title_pad));
        out.push_str("┐\x1b[K\n");

        // 2. Status banner
        let mode_name = match mode {
            DiarizationMode::OnlineMeeting => "Online",
            DiarizationMode::InPerson => "In-Person",
            DiarizationMode::Hybrid => "Hybrid",
        };
        let asr_display = format!("[{}]", config.model_tier);
        let mic_dot = if paused {
            "\x1b[1;33m[⏸]\x1b[0m"
        } else if mic_level > 0.005 || is_speaking {
            "\x1b[1;32m[●]\x1b[0m" // Bright green dot when mic audio active
        } else {
            "\x1b[36m[●]\x1b[0m"   // Calm cyan dot when listening
        };
        let loopback_dot = if paused {
            "\x1b[1;33m[⏸]\x1b[0m"
        } else if loopback_level > 0.0015 {
            "\x1b[1;32m[●]\x1b[0m"
        } else {
            "\x1b[36m[●]\x1b[0m"
        };
        let status_line = format!(
            " Audio: {} {} | {} {} │ Lang: [{}] │ ASR: {} │ Mode: [{}]",
            mic_dot, mic_name, loopback_dot, loopback_name, lang.code(), asr_display, mode_name
        );
        out.push_str(&format!("│{}│\x1b[K\n", fit_cell(&status_line, inner_w)));

        // 3. Split Divider
        if show_right_pane {
            out.push_str(&format!("├{}┬{}┤\x1b[K\n", "─".repeat(left_w), "─".repeat(right_w)));
        } else {
            out.push_str(&format!("├{}┤\x1b[K\n", "─".repeat(inner_w)));
        }

        // 4. Pane Headers with dynamic live waveform badge & Whisper processing indicator
        let (bars_count, show_label) = if left_w >= 54 {
            (10, true)
        } else if left_w >= 40 {
            (6, true)
        } else {
            (4, false)
        };
        let (wave_chars, is_voice_active) = build_waveform(wave_history, bars_count);
        let wave_badge = if paused {
            "\x1b[1;33m[PAUSED]\x1b[0m".to_string()
        } else if is_voice_active || is_speaking {
            if show_label {
                format!("\x1b[1;32m[{} ● REC]\x1b[0m", wave_chars)
            } else {
                format!("\x1b[1;32m[{}●]\x1b[0m", wave_chars)
            }
        } else {
            if show_label {
                format!("\x1b[36m[{} ● LIVE]\x1b[0m", wave_chars)
            } else {
                format!("\x1b[36m[{}●]\x1b[0m", wave_chars)
            }
        };
        let asr_badge = if paused {
            String::new()
        } else if transcribing_count > 0 {
            format!(
                " \x1b[1;33m[{} Memproses suara... ({})]\x1b[0m",
                spin_char, transcribing_count
            )
        } else if is_speaking {
            " \x1b[1;32m[🎙 Mendengarkan ucapan...]\x1b[0m".to_string()
        } else {
            String::new()
        };
        let left_hdr = format!(" LIVE DIARIZED TRANSCRIPT  {}{}", wave_badge, asr_badge);
        let right_hdr = if let Some(ref st) = assistant.status_msg {
            format!(" LIVE SUMMARY & ACTION ITEMS [{}]", st)
        } else {
            " LIVE SUMMARY & ACTION ITEMS".to_string()
        };

        if show_right_pane {
            out.push_str(&format!(
                "│{}│{}│\x1b[K\n",
                fit_cell(&left_hdr, left_w),
                fit_cell(&right_hdr, right_w)
            ));
        } else {
            out.push_str(&format!("│{}│\x1b[K\n", fit_cell(&left_hdr, inner_w)));
        }

        // 5. Left pane content (shows guidance placeholder with live audio meter if empty, or live transcript + activity bar)
        let (meter_bars, meter_active) = build_waveform(wave_history, 16);
        let meter_line = if paused {
            "  \x1b[1;33m● Audio Dijeda / Capture Paused (Tekan 'p' untuk lanjut)\x1b[0m".to_string()
        } else if meter_active || is_speaking {
            format!("  \x1b[1;32m● Audio Terdeteksi / Speaking : [{}]\x1b[0m", meter_bars)
        } else {
            format!("  \x1b[36m● Mikrofon Aktif / Listening  : [{}]\x1b[0m", meter_bars)
        };

        let live_activity_line = if paused {
            "  \x1b[1;33m⏸ Perekaman dijeda / Recording paused (Tekan 'p' atau Space untuk lanjut)\x1b[0m".to_string()
        } else if transcribing_count > 0 {
            format!(
                "  \x1b[1;33m{} Whisper sedang mengubah suara ke teks ({} antrean)... / Transcribing...\x1b[0m",
                spin_char, transcribing_count
            )
        } else if is_speaking {
            format!(
                "  \x1b[1;32m🎙 Merekam kalimat Anda... (berhenti bicara sejenak untuk transkripsi) [{}]\x1b[0m",
                meter_bars
            )
        } else {
            "  \x1b[36m● Siap mendengarkan kalimat berikutnya... / Ready for speech\x1b[0m".to_string()
        };

        let mut composed_left: Vec<String> = Vec::new();
        if left_lines.is_empty() {
            composed_left.push("".to_string());
            composed_left.push(format!("  {} [100% OFFLINE / LIVE]", lang.text("status.ready")));
            composed_left.push(meter_line);
            composed_left.push(live_activity_line);
            composed_left.push("".to_string());
            composed_left.push("  • Suara mikrofon Anda akan terdeteksi sebagai [Anda] / [Pembicara]".to_string());
            composed_left.push("  • Suara dari Google Meet / Zoom sebagai [Pembicara 1, 2...]".to_string());
            composed_left.push("".to_string());
            composed_left.push("  (Silakan mulai berbicara atau putar audio rapat)".to_string());
        } else {
            composed_left.extend_from_slice(left_lines);
            composed_left.push(live_activity_line);
        }

        let left_start = composed_left.len().saturating_sub(content_rows);
        let visible_left = &composed_left[left_start..];

        // 6. Right upper content (Topics & Actions)
        let mut right_upper: Vec<String> = Vec::new();
        if show_right_pane {
            right_upper.push(" • Key Topics / Topik Utama:".to_string());
            if assistant.topics.is_empty() {
                right_upper.push("   - (Tekan 's' untuk generate ringkasan AI)".to_string());
            } else {
                for t in &assistant.topics {
                    for chunk in textwrap_simple(t, right_w.saturating_sub(6)) {
                        right_upper.push(format!("   - {}", chunk));
                    }
                }
            }
            right_upper.push(" • Action Items / Tindak Lanjut:".to_string());
            if assistant.actions.is_empty() {
                right_upper.push("   [ ] (Belum ada catatan tindak lanjut)".to_string());
            } else {
                for a in &assistant.actions {
                    for chunk in textwrap_simple(a, right_w.saturating_sub(6)) {
                        right_upper.push(format!("   {}", chunk));
                    }
                }
            }
        }

        // 7. Right lower content (Q&A history)
        let mut right_lower: Vec<String> = Vec::new();
        if show_right_pane {
            if let Some((q, a)) = assistant.qa_history.last() {
                right_lower.push(format!(" > {}", q));
                for chunk in textwrap_simple(a, right_w.saturating_sub(6)) {
                    right_lower.push(format!("   {}", chunk));
                }
            } else {
                right_lower.push("   Tekan '?' untuk bertanya kepada Asisten AI".to_string());
                right_lower.push("   Press '?' to ask AI assistant about meeting".to_string());
            }
        }

        // 8. Render content rows
        for i in 0..content_rows {
            let left_cell = visible_left.get(i).map(|s| s.as_str()).unwrap_or("");
            if show_right_pane {
                if i < upper_rows {
                    let right_cell = right_upper.get(i).map(|s| s.as_str()).unwrap_or("");
                    out.push_str(&format!(
                        "│{}│{}│\x1b[K\n",
                        fit_cell(left_cell, left_w),
                        fit_cell(right_cell, right_w)
                    ));
                } else if i == upper_rows {
                    let hdr = " ASK ASSISTANT (? untuk tanya) ";
                    let bar_len = right_w.saturating_sub(hdr.chars().count());
                    let right_div = format!("├─{}{}┤", hdr, "─".repeat(bar_len.saturating_sub(1)));
                    out.push_str(&format!(
                        "│{}│{}│\x1b[K\n",
                        fit_cell(left_cell, left_w),
                        fit_cell(&right_div, right_w)
                    ));
                } else {
                    let lower_idx = i - upper_rows - 1;
                    let right_cell = right_lower.get(lower_idx).map(|s| s.as_str()).unwrap_or("");
                    out.push_str(&format!(
                        "│{}│{}│\x1b[K\n",
                        fit_cell(left_cell, left_w),
                        fit_cell(right_cell, right_w)
                    ));
                }
            } else {
                out.push_str(&format!("│{}│\x1b[K\n", fit_cell(left_cell, inner_w)));
            }
        }

        // 9. Bottom divider
        if show_right_pane {
            out.push_str(&format!("├{}┴{}┤\x1b[K\n", "─".repeat(left_w), "─".repeat(right_w)));
        } else {
            out.push_str(&format!("├{}┤\x1b[K\n", "─".repeat(inner_w)));
        }

        // 10. Command hotkeys bar
        let cmd_bar = if show_right_pane {
            " [p] Pause/Resume │ [s] Summary AI │ [?] Tanya Asisten │ [m] Mode │ [l] Lang │ [c] Simpan │ [q] Keluar"
        } else {
            " [p] Pause/Resume │ [m] Mode │ [l] Lang │ [c] Simpan │ [q] Keluar"
        };
        out.push_str(&format!("│{}│\x1b[K\n", fit_cell(cmd_bar, inner_w)));

        // 11. Interactive prompt bar
        let prompt_line = if show_right_pane {
            if input_mode {
                format!(" > Tanya Asisten: {}_  [Enter: Kirim | Esc: Batal]", input_buffer)
            } else {
                " > Tekan '?' untuk bertanya kepada Asisten AI seputar isi rapat ini...".to_string()
            }
        } else {
            " > Mode Transkripsi Penuh / Full Transcript Mode (Tekan 'q' untuk selesai & simpan)".to_string()
        };
        out.push_str(&format!("│{}│\x1b[K\n", fit_cell(&prompt_line, inner_w)));

        // 12. Bottom border (no trailing newline so terminal buffer never scrolls down)
        out.push_str(&format!("└{}┘\x1b[K", "─".repeat(inner_w)));

        let mut stdout = io::stdout();
        let _ = stdout.write_all(out.as_bytes());
        let _ = stdout.flush();
    }
}

enum AiTaskResult {
    Summary(Vec<String>, Vec<String>),
    Answer(String, String),
    Error(String),
}

/// Comprehensive preflight check for all runtime dependencies (Req 4).
/// Verifies storage directories, audio capture devices, and user-chosen Whisper weights in-process.
fn run_startup_check(config: &AppConfig, lang: &UiLanguage) -> Result<bool> {
    println!("============================================================");
    println!("  Catetin - {}", lang.text("check.title"));
    println!("============================================================");

    let mut all_ok = true;

    // 1. Working directories check
    let models_dir = Path::new("models");
    let logs_dir = Path::new("logs");
    let models_ok = models_dir.exists() || std::fs::create_dir_all(models_dir).is_ok();
    let logs_ok = logs_dir.exists() || std::fs::create_dir_all(logs_dir).is_ok();
    if models_ok && logs_ok {
        println!("  [✓] Direktori Kerja / Working Dir: ./models, ./logs [Siap / Ready]");
    } else {
        println!("  [!] Direktori Kerja / Working Dir: Menggunakan fallback lokasi AppData.");
    }

    // 2. Audio Capture Devices
    let input_dev = config.input_device_name.as_deref().unwrap_or("Default (Sistem)");
    let output_dev = config.output_device_name.as_deref().unwrap_or("Default (Sistem)");
    println!("  [✓] Audio Input (Mikrofon)        : {}", input_dev);
    println!("  [✓] Audio Loopback (Suara Rapat) : {}", output_dev);

    // 3. User-chosen Whisper model weights check (In-process, zero external runner)
    let model_path = resolve_model_path(config);
    if model_path.exists() {
        let size = std::fs::metadata(&model_path).map(|m| m.len()).unwrap_or(0);
        let size_mb = size as f64 / (1024.0 * 1024.0);
        if size > 1_000_000 {
            println!(
                "  [✓] Bobot Model Whisper ('{}')  : {} ({:.1} MB) [Siap / Ready]",
                config.model_tier,
                model_path.display(),
                size_mb
            );
        } else {
            println!(
                "  [!] Bobot Model Whisper ('{}')  : {} ({:.1} MB) [Corrupt / File terlalu kecil]",
                config.model_tier,
                model_path.display(),
                size_mb
            );
            println!("      Jalankan 'catetin setup' untuk mengunduh ulang bobot model.");
            all_ok = false;
        }
    } else {
        println!(
            "  [!] Bobot Model Whisper ('{}')  : Belum ada di disk lokal.",
            config.model_tier
        );
        println!("      Lokasi yang dicari: {}", model_path.display());
        println!("      Jalankan 'catetin setup' untuk memilih dan mengunduh model.");
        all_ok = false;
    }

    // 4. In-process Diarization & VAD engine
    println!(
        "  [✓] Mesin Diarization & VAD       : Mode [{:?}] [Siap / In-Process]",
        config.diarization_mode
    );

    println!("============================================================");
    if all_ok {
        println!("  [✓] Semua dependensi siap digunakan! / All dependencies ready.");
    } else {
        println!("  [!] Ada dependensi yang memerlukan perhatian sebelum memulai rapat.");
    }
    println!("============================================================\n");

    Ok(all_ok)
}

/// Interactive setup workflow for model tier selection and dependency verification (Req 3 & 4).
fn run_setup_workflow(config: &mut AppConfig, lang: &UiLanguage) -> Result<()> {
    println!("============================================================");
    println!("  Catetin - {}", lang.text("setup.title"));
    println!("============================================================");

    // 1. Interactive Model Selection with trade-offs (Req 3)
    let selected_tier = prompt_model_choice(&config.model_tier, lang);
    config.model_tier = selected_tier;

    // 2. Storage & Model weight availability check
    let target_path = resolve_model_path(config);
    if target_path.exists() {
        let size = std::fs::metadata(&target_path).map(|m| m.len()).unwrap_or(0);
        let size_mb = size as f64 / (1024.0 * 1024.0);
        println!(
            "  [✓] Berkas model '{}' sudah tersedia: {} ({:.1} MB)",
            config.model_tier,
            target_path.display(),
            size_mb
        );
        let redownload = read_line("  Unduh ulang model ini? / Re-download? (y/N) [N]: ");
        if matches!(redownload.to_lowercase().as_str(), "y" | "yes" | "ya") {
            let _ = std::fs::remove_file(&target_path);
            ensure_model_downloaded(&target_path, config.model_url(), &config.model_tier)?;
        }
    } else {
        println!("  Berkas model belum ada di disk: {}", target_path.display());
        let confirm = read_line(&format!(
            "  Unduh bobot model '{}' sekarang? / Download now? (Y/n) [Y]: ",
            config.model_tier
        ));
        if !matches!(confirm.to_lowercase().as_str(), "n" | "no" | "tidak") {
            ensure_model_downloaded(&target_path, config.model_url(), &config.model_tier)?;
        } else {
            println!("  [!] Pengunduhan dilewati. Anda dapat mengunduhnya nanti dengan 'catetin setup'.");
        }
    }

    // 3. Save config
    config.save()?;
    println!("  [✓] Konfigurasi model tersimpan di: {}", AppConfig::config_file_path().display());
    println!();

    // 4. Preflight startup check (Req 4)
    run_startup_check(config, lang)?;

    println!("============================================================");
    println!("  [OK] Setup selesai! / Setup completed successfully!");
    println!("  Jalankan 'catetin' untuk memulai transkripsi rapat.");
    println!("============================================================");
    Ok(())
}

/// Comprehensive preflight check for all runtime dependencies on startup.
/// If user-chosen weights are missing, interactively prompts the user for model tier selection (Req 3)
/// and prepares the model before verifying all dependencies in-process (Req 4).
fn check_and_prepare_dependencies(config: &mut AppConfig, lang: &UiLanguage) -> Result<PathBuf> {
    let mut model_path = resolve_model_path(config);

    if !model_path.exists() {
        println!("============================================================");
        println!("  Catetin - {}", lang.text("app.subtitle"));
        println!("  Bobot model Whisper belum ditemukan di disk lokal.");
        println!("  Model weights not found. Choose your preferred model tier:");
        println!("============================================================");
        let chosen_tier = prompt_model_choice(&config.model_tier, lang);
        config.model_tier = chosen_tier;
        model_path = resolve_model_path(config);

        if !model_path.exists() {
            ensure_model_downloaded(&model_path, config.model_url(), &config.model_tier)?;
        }
        let _ = config.save();
    }

    run_startup_check(config, lang)?;

    Ok(model_path)
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();

    let mut custom_config_path = None;
    let mut trigger_wizard = false;
    let mut trigger_setup = false;
    let mut trigger_check = false;
    let mut show_assistant_override = None;
    let mut verbose_override = false;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--config" | "-c" | "config" => {
                if i + 1 < args.len() && !args[i + 1].starts_with('-') {
                    i += 1;
                    custom_config_path = Some(PathBuf::from(&args[i]));
                } else {
                    trigger_wizard = true;
                }
            }
            "--setup" | "setup" => {
                trigger_setup = true;
            }
            "--check" | "check" => {
                trigger_check = true;
            }
            "--assistant" => show_assistant_override = Some(true),
            "--no-assistant" => show_assistant_override = Some(false),
            "--verbose" | "-v" => verbose_override = true,
            "--help" | "-h" => {
                print_help(&UiLanguage::default());
                return Ok(());
            }
            "--version" => {
                println!("Catetin v{}", env!("CARGO_PKG_VERSION"));
                return Ok(());
            }
            _ => {}
        }
        i += 1;
    }

    let mut config = if let Some(ref path) = custom_config_path {
        AppConfig::load_from_path(path).unwrap_or_else(|_| AppConfig::default())
    } else {
        AppConfig::load_or_default()
    };

    if let Some(show) = show_assistant_override {
        config.show_assistant = show;
    }
    if verbose_override {
        config.verbose = true;
    }

    if trigger_wizard {
        let config_display = custom_config_path
            .as_ref()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|| AppConfig::config_file_path().display().to_string());

        match run_interactive_wizard(&mut config, &config_display)? {
            ConfigMenuAction::StartMeeting => {
                // User chose to start meeting directly! Proceed with initialized pipeline.
            }
            ConfigMenuAction::Exit => {
                return Ok(());
            }
        }
    }

    let current_lang = config.ui_language.clone();

    if trigger_setup {
        run_setup_workflow(&mut config, &current_lang)?;
        return Ok(());
    }

    if trigger_check {
        run_startup_check(&config, &current_lang)?;
        return Ok(());
    }

    // 5. Initialize dedicated file logger (with automatic local fallback)
    let log_path = init_logger(&config, config.verbose)?;

    let mut current_lang = config.ui_language.clone();
    let mode_desc = match config.diarization_mode {
        DiarizationMode::OnlineMeeting => current_lang.text("label.mode_online"),
        DiarizationMode::InPerson => current_lang.text("label.mode_in_person"),
        DiarizationMode::Hybrid => current_lang.text("label.mode_hybrid"),
    };

    println!("============================================================");
    println!(
        "  Catetin v{} - \"{}\"",
        env!("CARGO_PKG_VERSION"),
        current_lang.text("app.tagline")
    );
    println!("  {}", current_lang.text("app.subtitle"));
    println!("============================================================");
    println!(
        "  {}:     {} ({})",
        current_lang.text("label.language"),
        current_lang.display_name(),
        current_lang.code()
    );
    println!("  {}:         {}", current_lang.text("label.mode"), mode_desc);
    println!(
        "  {}:   {} ({})",
        current_lang.text("label.model_tier"),
        config.model_tier,
        config.model_filename()
    );
    println!("  {}:     {}", current_lang.text("label.log_file"), log_path.display());
    println!(
        "  {}:     {}",
        current_lang.text("label.microphone"),
        config.input_device_name.as_deref().unwrap_or("Default (Sistem)")
    );
    println!(
        "  {}:     {}",
        current_lang.text("label.loopback"),
        config.output_device_name.as_deref().unwrap_or("Default (Sistem)")
    );
    println!("============================================================");

    // 6. Preflight Dependency Readiness Check (Models, weights, audio)
    let model_path = check_and_prepare_dependencies(&mut config, &current_lang)?;

    println!("  -> Mengaktifkan mesin Whisper (100% offline)...");
    let asr_lang = if config.asr_language.is_empty() || config.asr_language.eq_ignore_ascii_case("auto") {
        config.ui_language.code()
    } else {
        config.asr_language.as_str()
    };
    let engine = WhisperEngine::load_from_file(&model_path, asr_lang)?;
    println!("  [OK] Model Whisper aktif (C++ in-process): {}\n", model_path.display());
    let recognizer: Box<dyn SpeechRecognizer> = Box::new(engine);

    // 7. Initialize dual-channel audio capture (Microphone + Loopback)
    println!("  -> Starting audio capture streams (Mic + System Loopback)...");
    let capture = match DualAudioCapture::from_config(&config) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("\n[ERROR] Audio capture failed: {:#}", e);
            eprintln!("Ensure a microphone and default playback device are available.");
            return Err(e);
        }
    };

    let mic_display = config.input_device_name.as_deref().unwrap_or("Default Mic");
    let loopback_display = config.output_device_name.as_deref().unwrap_or("Default Loopback");

    // 8. Launch headless transcription & diarization pipeline
    let (event_tx, event_rx) = std::sync::mpsc::channel::<TranscriptEvent>();
    let pipeline = CatetinPipeline::new(recognizer)
        .with_mode(config.diarization_mode)
        .with_noise_suppression(config.noise_suppression);
    let mut handle = pipeline.start(capture, event_tx)?;

    // Non-blocking stdin listener only for non-Windows / headless environments where kb::poll_key() returns None
    let (_cmd_tx, cmd_rx) = std::sync::mpsc::channel::<String>();
    #[cfg(not(windows))]
    if !config.show_assistant {
        let cmd_tx = _cmd_tx.clone();
        std::thread::spawn(move || {
            let stdin = io::stdin();
            let mut line = String::new();
            while stdin.read_line(&mut line).is_ok() {
                let trimmed = line.trim().to_string();
                if cmd_tx.send(trimmed).is_err() {
                    break;
                }
                line.clear();
            }
        });
    }

    // Channel for background AI worker tasks (Summary & Q&A)
    let (ai_tx, ai_rx) = std::sync::mpsc::channel::<AiTaskResult>();

    let mut recorded_segments: Vec<(String, String, String)> = Vec::new();
    let mut left_lines: Vec<String> = Vec::new();
    let mut assistant_state = tui::AssistantState::default();
    let mut paused = false;
    let mut input_mode = false;
    let mut input_buffer = String::new();
    let mut last_term_size = tui::get_terminal_size();
    let mut wave_history: Vec<f32> = vec![0.0; 16];
    let mut last_wave_tick = std::time::Instant::now();
    let mut was_audio_active = false;
    let mut spinner_idx: usize = 0;
    let mut last_speaking = false;
    let mut last_transcribing: u32 = 0;

    // Enter raw console mode on Windows to enable immediate non-blocking hotkey detection
    // without requiring Enter and without messy character echoing
    #[cfg(windows)]
    let _raw_guard = Some(console::RawConsoleGuard::enter());

    // Suppress stderr logs while the full-screen TUI is active so log lines never corrupt box borders
    catetin_core::logging::set_stderr_enabled(false);

    // Clear terminal screen and hide cursor for smooth TUI presentation
    print!("\x1b[2J\x1b[H\x1b[?25l");
    let _ = io::stdout().flush();
    tui::render_copilot_tui(
        &config,
        &current_lang,
        handle.current_mode(),
        mic_display,
        loopback_display,
        paused,
        &left_lines,
        &assistant_state,
        input_mode,
        &input_buffer,
        handle.mic_level(),
        handle.loopback_level(),
        &wave_history,
        handle.is_speaking(),
        handle.transcribing_count(),
        spinner_idx,
    );

    loop {
        let mut needs_redraw = false;

        // Waveform real-time audio tick (~40ms = 25 FPS)
        let now = std::time::Instant::now();
        if now.duration_since(last_wave_tick) >= Duration::from_millis(40) {
            last_wave_tick = now;
            if !paused {
                let current_mic = handle.mic_level();
                let current_lb = handle.loopback_level();
                let combined_level = current_mic.max(current_lb * 3.5);
                wave_history.push(combined_level);
                if wave_history.len() > 16 {
                    wave_history.remove(0);
                }

                let cur_speaking = handle.is_speaking();
                let cur_transcribing = handle.transcribing_count();
                if cur_transcribing > 0 {
                    spinner_idx = spinner_idx.wrapping_add(1);
                    needs_redraw = true;
                }
                if cur_speaking != last_speaking || cur_transcribing != last_transcribing {
                    last_speaking = cur_speaking;
                    last_transcribing = cur_transcribing;
                    needs_redraw = true;
                }

                let is_audio_active = combined_level > 0.003 || wave_history.iter().any(|&v| v > 0.003);
                if is_audio_active {
                    needs_redraw = true;
                    was_audio_active = true;
                } else if was_audio_active {
                    needs_redraw = true; // Settle waveform back into resting state
                    was_audio_active = false;
                }
            }
        }

        // 0. Check terminal resize dynamically
        let current_size = tui::get_terminal_size();
        if current_size != last_term_size {
            last_term_size = current_size;
            print!("\x1b[2J\x1b[H"); // Clear old borders when resized
            let _ = io::stdout().flush();
            needs_redraw = true;
        }

        // A. Handle background AI worker responses
        while let Ok(res) = ai_rx.try_recv() {
            match res {
                AiTaskResult::Summary(topics, actions) => {
                    assistant_state.topics = topics;
                    assistant_state.actions = actions;
                    assistant_state.status_msg = None;
                }
                AiTaskResult::Answer(q, a) => {
                    assistant_state.qa_history.push((q, a));
                    assistant_state.status_msg = None;
                }
                AiTaskResult::Error(err) => {
                    assistant_state.status_msg = Some(format!("Error: {}", err));
                }
            }
            needs_redraw = true;
        }

        // B. Instant Non-blocking Keyboard Polling
        let mut quit_requested = false;
        while let Some(key) = kb::poll_key() {
            if !input_mode {
                match key as u8 {
                    b'q' | b'Q' | 3 => {
                        quit_requested = true;
                        break;
                    }
                    b'p' | b'P' | b' ' => {
                        paused = !paused;
                        handle.set_paused(paused);
                        if paused {
                            wave_history.fill(0.0);
                            was_audio_active = false;
                        }
                        needs_redraw = true;
                    }
                    b'm' | b'M' => {
                        let _ = handle.toggle_mode();
                        needs_redraw = true;
                    }
                    b'l' | b'L' => {
                        current_lang = match current_lang {
                            UiLanguage::Indonesian => UiLanguage::English,
                            _ => UiLanguage::Indonesian,
                        };
                        needs_redraw = true;
                    }
                    b'c' | b'C' => {
                        let default_path = default_transcript_path();
                        let _ = save_markdown_transcript(
                            &default_path,
                            &recorded_segments,
                            &assistant_state.topics,
                            &assistant_state.actions,
                            &current_lang,
                        );
                        assistant_state.status_msg = Some("Snapshot disimpan!".into());
                        needs_redraw = true;
                    }
                    b's' | b'S' if config.show_assistant => {
                        assistant_state.status_msg = Some("Generating Summary...".into());
                        let url = config.assistant_url.clone();
                        let model = config.assistant_model.clone();
                        let api_key = config.assistant_api_key.clone();
                        let transcript_text = recorded_segments
                            .iter()
                            .map(|(t, s, txt)| format!("{} [{}]: {}", t, s, txt))
                            .collect::<Vec<_>>()
                            .join("\n");
                        let tx = ai_tx.clone();

                        std::thread::spawn(move || {
                            match assistant::generate_summary(&url, &model, api_key.as_deref(), &transcript_text) {
                                Ok((topics, actions)) => {
                                    let _ = tx.send(AiTaskResult::Summary(topics, actions));
                                }
                                Err(e) => {
                                    let _ = tx.send(AiTaskResult::Error(e.to_string()));
                                }
                            }
                        });
                        needs_redraw = true;
                    }
                    b'?' | b'/' if config.show_assistant => {
                        input_mode = true;
                        input_buffer.clear();
                        needs_redraw = true;
                    }
                    _ => {}
                }
            } else {
                // Interactive Question Typing Mode
                match key {
                    27 => { // Esc: cancel question
                        input_mode = false;
                        input_buffer.clear();
                        needs_redraw = true;
                    }
                    13 | 10 => { // Enter: submit question
                        let question = input_buffer.trim().to_string();
                        if !question.is_empty() {
                            assistant_state.status_msg = Some("Thinking...".into());
                            let url = config.assistant_url.clone();
                            let model = config.assistant_model.clone();
                            let api_key = config.assistant_api_key.clone();
                            let transcript_text = recorded_segments
                                .iter()
                                .map(|(t, s, txt)| format!("{} [{}]: {}", t, s, txt))
                                .collect::<Vec<_>>()
                                .join("\n");
                            let tx = ai_tx.clone();
                            let q_clone = question.clone();

                            std::thread::spawn(move || {
                                match assistant::ask_assistant(&url, &model, api_key.as_deref(), &transcript_text, &q_clone) {
                                    Ok(ans) => {
                                        let _ = tx.send(AiTaskResult::Answer(q_clone, ans));
                                    }
                                    Err(e) => {
                                        let _ = tx.send(AiTaskResult::Error(e.to_string()));
                                    }
                                }
                            });
                        }
                        input_mode = false;
                        input_buffer.clear();
                        needs_redraw = true;
                    }
                    8 => { // Backspace
                        input_buffer.pop();
                        needs_redraw = true;
                    }
                    32..=255 if key != 127 => { // Printable chars
                        if let Some(c) = char::from_u32(key as u32) {
                            input_buffer.push(c);
                            needs_redraw = true;
                        }
                    }
                    _ => {}
                }
            }
        }

        if quit_requested {
            break;
        }

        // C. Fallback for Unix/headless stdin lines
        if let Ok(cmd) = cmd_rx.try_recv() {
            let cmd_lower = cmd.to_lowercase();
            match cmd_lower.as_str() {
                "q" | "quit" | "keluar" | "exit" => {
                    break;
                }
                "p" | "pause" | " " => {
                    paused = !paused;
                    handle.set_paused(paused);
                    if paused {
                        wave_history.fill(0.0);
                        was_audio_active = false;
                    }
                    needs_redraw = true;
                }
                "m" | "mode" => {
                    let _ = handle.toggle_mode();
                    needs_redraw = true;
                }
                "l" | "lang" => {
                    current_lang = match current_lang {
                        UiLanguage::Indonesian => UiLanguage::English,
                        _ => UiLanguage::Indonesian,
                    };
                    needs_redraw = true;
                }
                "s" | "summary" => {
                    assistant_state.status_msg = Some("Generating Summary...".into());
                    let url = config.assistant_url.clone();
                    let model = config.assistant_model.clone();
                    let api_key = config.assistant_api_key.clone();
                    let transcript_text = recorded_segments
                        .iter()
                        .map(|(t, s, txt)| format!("{} [{}]: {}", t, s, txt))
                        .collect::<Vec<_>>()
                        .join("\n");
                    let tx = ai_tx.clone();

                    std::thread::spawn(move || {
                        match assistant::generate_summary(&url, &model, api_key.as_deref(), &transcript_text) {
                            Ok((topics, actions)) => {
                                let _ = tx.send(AiTaskResult::Summary(topics, actions));
                            }
                            Err(e) => {
                                let _ = tx.send(AiTaskResult::Error(e.to_string()));
                            }
                        }
                    });
                    needs_redraw = true;
                }
                "c" | "save" => {
                    let default_path = default_transcript_path();
                    let _ = save_markdown_transcript(
                        &default_path,
                        &recorded_segments,
                        &assistant_state.topics,
                        &assistant_state.actions,
                        &current_lang,
                    );
                    needs_redraw = true;
                }
                custom_query if custom_query.starts_with('?') || custom_query.starts_with("ask ") => {
                    let question = custom_query.trim_start_matches(|c| c == '?' || c == ' ').trim().to_string();
                    if !question.is_empty() {
                        assistant_state.status_msg = Some("Thinking...".into());
                        let url = config.assistant_url.clone();
                        let model = config.assistant_model.clone();
                        let api_key = config.assistant_api_key.clone();
                        let transcript_text = recorded_segments
                            .iter()
                            .map(|(t, s, txt)| format!("{} [{}]: {}", t, s, txt))
                            .collect::<Vec<_>>()
                            .join("\n");
                        let tx = ai_tx.clone();
                        let q_clone = question.clone();

                        std::thread::spawn(move || {
                            match assistant::ask_assistant(&url, &model, api_key.as_deref(), &transcript_text, &q_clone) {
                                Ok(ans) => {
                                    let _ = tx.send(AiTaskResult::Answer(q_clone, ans));
                                }
                                Err(e) => {
                                    let _ = tx.send(AiTaskResult::Error(e.to_string()));
                                }
                            }
                        });
                        needs_redraw = true;
                    }
                }
                _ => {}
            }
        }

        // D. Process real-time transcription events
        while let Ok(event) = event_rx.try_recv() {
            if paused {
                continue;
            }
            let time_str = format_ms(event.timestamp_ms);
            let speaker_name = event.speaker.to_localized_string(&current_lang);

            let color = match event.speaker {
                SpeakerId::LocalUser => "\x1b[1;36m",    // Bright Cyan (You)
                SpeakerId::Remote(1) | SpeakerId::InPerson(1) => "\x1b[1;33m",    // Bright Yellow
                SpeakerId::Remote(2) | SpeakerId::InPerson(2) => "\x1b[1;32m",    // Bright Green
                SpeakerId::Remote(3) | SpeakerId::InPerson(3) => "\x1b[1;35m",    // Bright Magenta
                _ => "\x1b[1;34m",                       // Bright Blue
            };
            let reset = "\x1b[0m";

            let (term_w, _) = last_term_size;
            let inner_w = term_w.saturating_sub(1).max(75).saturating_sub(2);
            let wrap_w = if config.show_assistant {
                ((inner_w.saturating_sub(1) * 52) / 100).saturating_sub(4).max(30)
            } else {
                inner_w.saturating_sub(6).max(40)
            };

            let header = format!("{} {}[{}]:{}", time_str, color, speaker_name, reset);
            left_lines.push(header);
            for chunk in tui::textwrap_simple(&event.text, wrap_w) {
                left_lines.push(format!("  \"{}\"", chunk));
            }
            left_lines.push(String::new());

            recorded_segments.push((time_str.clone(), speaker_name.clone(), event.text.clone()));
            needs_redraw = true;
        }

        if needs_redraw {
            tui::render_copilot_tui(
                &config,
                &current_lang,
                handle.current_mode(),
                mic_display,
                loopback_display,
                paused,
                &left_lines,
                &assistant_state,
                input_mode,
                &input_buffer,
                handle.mic_level(),
                handle.loopback_level(),
                &wave_history,
                handle.is_speaking(),
                handle.transcribing_count(),
                spinner_idx,
            );
        }

        std::thread::sleep(Duration::from_millis(25));
    }

    // Clear TUI and show cursor before stopping so user sees immediate feedback on 'q'
    print!("\x1b[2J\x1b[H\x1b[?25h");
    let _ = io::stdout().flush();

    if handle.is_speaking() || handle.transcribing_count() > 0 {
        println!("============================================================");
        println!("  ⏳ Menyelesaikan transkripsi kalimat terakhir... / Finishing final segment...");
        println!("============================================================");
        let _ = io::stdout().flush();
    }

    handle.stop();

    // Drain any final transcription events produced during flush/stop
    while let Ok(event) = event_rx.try_recv() {
        let time_str = format_ms(event.timestamp_ms);
        let speaker_name = event.speaker.to_localized_string(&current_lang);
        recorded_segments.push((time_str, speaker_name, event.text));
    }

    catetin_core::logging::set_stderr_enabled(config.verbose);

    // 9. Ask user whether to Save or Discard the Markdown transcript on exit
    if !recorded_segments.is_empty() {
        let default_path = default_transcript_path();
        println!("============================================================");
        println!(
            "  Catatan Rapat / Meeting Transcript: {} rekaman",
            recorded_segments.len()
        );
        println!("------------------------------------------------------------");
        println!("  [S / Enter] Simpan Transkrip (.md) / Save Transcript");
        println!("  [D / Esc]   Buang & Keluar / Discard & Exit");
        println!("============================================================");
        print!("  Pilihan Anda / Your choice [S/D]: ");
        let _ = io::stdout().flush();

        // Drain any leftover keys in buffer before waiting for S/D choice
        while kb::poll_key().is_some() {}

        let want_save = loop {
            if let Some(key) = kb::poll_key() {
                match key as u8 {
                    b's' | b'S' | b'y' | b'Y' | 13 | 10 => {
                        println!("Simpan / Save");
                        break true;
                    }
                    b'd' | b'D' | b'n' | b'N' | b'q' | b'Q' | 27 | 3 => {
                        println!("Buang / Discard");
                        break false;
                    }
                    _ => {}
                }
            }
            #[cfg(not(windows))]
            {
                let answer = read_line("");
                let trimmed = answer.trim().to_lowercase();
                break !matches!(trimmed.as_str(), "d" | "n" | "no" | "discard" | "buang" | "batal");
            }
            #[cfg(windows)]
            std::thread::sleep(Duration::from_millis(20));
        };

        #[cfg(windows)]
        drop(_raw_guard);

        if want_save {
            let default_filename = default_path
                .file_name()
                .and_then(|f| f.to_str())
                .unwrap_or("meeting.md");

            println!("  Membuka dialog penyimpanan berkas... / Opening Save As file dialog...");
            let picked = file_dialog::pick_save_file(
                default_filename,
                "Simpan Catatan Rapat / Save Meeting Transcript",
            );

            let target_path = match picked {
                Some(path) => Some(path),
                None => {
                    println!("  Dialog ditutup. Menyimpan ke lokasi bawaan / Saving to default path:");
                    println!("  -> {}", default_path.display());
                    Some(default_path)
                }
            };

            if let Some(path) = target_path {
                match save_markdown_transcript(
                    &path,
                    &recorded_segments,
                    &assistant_state.topics,
                    &assistant_state.actions,
                    &current_lang,
                ) {
                    Ok(_) => {
                        println!("============================================================");
                        println!(
                            "  [OK] {} {}",
                            current_lang.text("prompt.saved_success"),
                            path.display()
                        );
                        println!("============================================================");
                    }
                    Err(e) => {
                        eprintln!("  [ERROR] Gagal menyimpan transkrip ke {:?}: {:#}", path, e);
                    }
                }
            }
        } else {
            println!("  [i] Transkrip dibuang. / Transcript discarded.\n");
        }
    } else {
        #[cfg(windows)]
        drop(_raw_guard);

        println!("============================================================");
        println!("  Session ended. / Sesi selesai (tidak ada rekaman suara).");
        println!("============================================================");
    }

    Ok(())
}
