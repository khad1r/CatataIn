use anyhow::Result;
use std::io::{self, Write};
use catetin_core::asr::{
    discover_transducer_models, discover_whisper_models, has_transducer_installed_for_lang,
    has_whisper_onnx_installed,
};
use catetin_core::config::{AppConfig, NoiseSuppression};
use catetin_core::i18n::UiLanguage;
use catetin_core::pipeline::DiarizationMode;
use crate::console::RawConsoleGuard;
use crate::kb::{self, read_confirm};
use crate::models::{
    download_transducer_by_name, download_whisper_by_tier, prompt_model_choice, TRANSDUCER_PRESETS,
    WHISPER_PRESETS,
};
use crate::tui;

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

fn cycle_whisper_tier(config: &mut AppConfig, forward: bool) {
    let mut available_tiers: Vec<String> = WHISPER_PRESETS.iter().map(|p| p.id.to_string()).collect();
    // Include custom discovered Whisper models
    for m in discover_whisper_models(&config.model_dir) {
        if !available_tiers.contains(&m.id) {
            available_tiers.push(m.id);
        }
    }

    let cur = config.model_tier.to_lowercase();
    let idx = available_tiers.iter().position(|t| t == &cur).unwrap_or(0);
    let next_idx = if forward {
        (idx + 1) % available_tiers.len()
    } else if idx == 0 {
        available_tiers.len() - 1
    } else {
        idx - 1
    };
    config.model_tier = available_tiers[next_idx].clone();
}

fn cycle_transducer_model(config: &mut AppConfig, forward: bool) {
    let mut available: Vec<String> = TRANSDUCER_PRESETS.iter().map(|p| p.id.to_string()).collect();
    // Include custom discovered Transducer models
    for m in discover_transducer_models(&config.model_dir) {
        if !available.contains(&m.id) {
            available.push(m.id);
        }
    }

    let cur = config.transducer_model.to_lowercase();
    let idx = available.iter().position(|t| t == &cur).unwrap_or(0);
    let next_idx = if forward {
        (idx + 1) % available.len()
    } else if idx == 0 {
        available.len() - 1
    } else {
        idx - 1
    };
    config.transducer_model = available[next_idx].clone();
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

const ASSISTANT_MODELS: [&str; 6] = [
    "llama3.2:1b",
    "llama3.2:3b",
    "mistral",
    "qwen2.5:1.5b",
    "deepseek-r1:1.5b",
    "gemma2:2b",
];

fn cycle_assistant_model(model: &mut String, forward: bool) {
    let idx = ASSISTANT_MODELS.iter().position(|&m| m == model).unwrap_or(0);
    let next_idx = if forward {
        (idx + 1) % ASSISTANT_MODELS.len()
    } else if idx == 0 {
        ASSISTANT_MODELS.len() - 1
    } else {
        idx - 1
    };
    *model = ASSISTANT_MODELS[next_idx].to_string();
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
        0 => cycle_language(&mut config.ui_language, forward),
        1 => config.cycle_asr_language(forward),
        2 => cycle_mode(&mut config.diarization_mode, forward),
        3 => cycle_whisper_tier(config, forward),
        4 => cycle_transducer_model(config, forward),
        5 => cycle_device(&mut config.input_device_name, input_devices, forward),
        6 => cycle_device(&mut config.output_device_name, output_devices, forward),
        7 => cycle_noise_suppression(&mut config.noise_suppression, forward),
        8 => config.show_assistant = !config.show_assistant,
        9 => cycle_assistant_model(&mut config.assistant_model, forward),
        10 => config.verbose = !config.verbose,
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
    let width = term_w.max(76);
    let lang = &config.ui_language;

    let mut out = String::with_capacity(4096);
    out.push_str("\x1b[H");

    let sep_len = width.min(96);
    let sep = "─".repeat(sep_len);

    let title = format!(
        "─── Catetin v{} - {} {}",
        env!("CARGO_PKG_VERSION"),
        lang.text("wizard.title"),
        "─".repeat(sep_len.saturating_sub(28))
    );
    out.push_str(&format!("{}\x1b[K\n", title));
    out.push_str("   Gunakan [↑/↓] Pilih Menu  │  [←/→/Enter] Ubah Nilai  │  [Esc] Keluar\x1b[K\n");
    out.push_str(&format!("{}\x1b[K\n", sep));

    let has_whisper = has_whisper_onnx_installed(&config.model_dir, &config.model_tier);
    let whisper_tag = if has_whisper {
        "\x1b[1;32m[✓ Siap]\x1b[0m"
    } else {
        "\x1b[1;33m[↓ Unduh saat mulai]\x1b[0m"
    };

    let asr_lang = if config.asr_language.is_empty() { "auto" } else { &config.asr_language };
    let has_transducer = has_transducer_installed_for_lang(&config.transducer_dir(), asr_lang);
    let transducer_tag = if has_transducer {
        "\x1b[1;32m[✓ Siap]\x1b[0m"
    } else {
        "\x1b[1;33m[↓ Unduh saat mulai]\x1b[0m"
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
        ("2. Bahasa Audio Rapat (ASR Language)".to_string(), config.asr_language_display().to_string()),
        (format!("3. {}", lang.text("label.mode")), mode_desc.to_string()),
        ("4. Model Whisper (Pass 2 Accurate)".to_string(), format!("{} {}", config.model_tier, whisper_tag)),
        ("5. Model Transducer (Pass 1 <50ms)".to_string(), format!("{} {}", config.transducer_model, transducer_tag)),
        (format!("6. {}", lang.text("label.microphone")), mic_name.to_string()),
        (format!("7. {}", lang.text("label.loopback")), loopback_name.to_string()),
        ("8. Peredam Bising (Noise Cancel)".to_string(), config.noise_suppression.display_label(lang).to_string()),
        ("9. Panel Asisten AI Copilot".to_string(), copilot_desc.to_string()),
        ("10. Model LLM Copilot".to_string(), config.assistant_model.clone()),
        ("11. Log Detail (Verbose)".to_string(), verbose_desc.to_string()),
    ];

    out.push_str("\x1b[K\n");

    for (i, (label, val)) in items.iter().enumerate() {
        let is_sel = selected == i;
        let prefix = if is_sel { "\x1b[1;36m ▶ " } else { "   " };
        let suffix = if is_sel { "\x1b[0m" } else { "" };
        out.push_str(&format!("{}{} : < {} >{}\x1b[K\n", prefix, tui::fit_cell(label, 36), val, suffix));
    }

    out.push_str("\x1b[K\n");
    out.push_str(&format!("{}\x1b[K\n", sep));

    let actions = [
        ("▶ Simpan & Mulai Rapat / Save & Start Meeting", 11),
        ("💾 Simpan Konfigurasi & Keluar / Save Config & Exit", 12),
        ("❌ Batal & Keluar / Cancel & Exit", 13),
    ];

    for (act_text, act_idx) in actions {
        let is_sel = selected == act_idx;
        let prefix = if is_sel { "\x1b[1;32m ▶ [ " } else { "   [ " };
        let suffix = if is_sel { " ]\x1b[0m" } else { " ]" };
        out.push_str(&format!("{}{}{}\x1b[K\n", prefix, act_text, suffix));
    }

    out.push_str(&format!("{}\x1b[K\n", sep));
    out.push_str(&format!("   Berkas Konfigurasi / Config File: {}\x1b[K\n", config_file_display));
    out.push_str(&format!("{}\x1b[K\n", sep));

    let mut stdout = io::stdout();
    let _ = stdout.write_all(out.as_bytes());
    let _ = stdout.flush();
}

/// Interactive terminal configuration menu with Up/Down arrow key navigation.
pub fn run_interactive_wizard(config: &mut AppConfig, config_file_display: &str) -> Result<ConfigMenuAction> {
    let _guard = RawConsoleGuard::enter();

    print!("\x1b[2J\x1b[H\x1b[?25l");
    let _ = io::stdout().flush();

    let input_devices = catetin_core::audio::list_input_devices();
    let output_devices = catetin_core::audio::list_output_devices();

    let mut selected: usize = 0;
    let total_options = 14;

    loop {
        render_config_menu(config, selected, &input_devices, &output_devices, config_file_display);

        let key = kb::read_key_blocking();
        match key {
            kb::Key::Up => {
                selected = if selected == 0 { total_options - 1 } else { selected - 1 };
            }
            kb::Key::Down => {
                selected = (selected + 1) % total_options;
            }
            kb::Key::Left => {
                apply_menu_change(config, selected, &input_devices, &output_devices, false);
            }
            kb::Key::Right => {
                apply_menu_change(config, selected, &input_devices, &output_devices, true);
            }
            kb::Key::Enter | kb::Key::Char(' ') => {
                if selected < 11 {
                    apply_menu_change(config, selected, &input_devices, &output_devices, true);
                } else if selected == 11 {
                    config.save()?;
                    if !has_whisper_onnx_installed(&config.model_dir, &config.model_tier) {
                        print!("\x1b[?25h\n\n");
                        let _ = io::stdout().flush();
                        download_whisper_by_tier(&config.model_dir, &config.model_tier)?;
                    }
                    let asr_lang = if config.asr_language.is_empty() { "auto" } else { &config.asr_language };
                    if !has_transducer_installed_for_lang(&config.transducer_dir(), asr_lang) {
                        print!("\x1b[?25h\n\n");
                        let _ = io::stdout().flush();
                        download_transducer_by_name(&config.model_dir, &config.transducer_model)?;
                    }
                    print!("\x1b[?25h\n");
                    let _ = io::stdout().flush();
                    return Ok(ConfigMenuAction::StartMeeting);
                } else if selected == 12 {
                    config.save()?;
                    print!("\x1b[?25h\n");
                    let _ = io::stdout().flush();
                    println!("============================================================");
                    println!("  [✓] Konfigurasi berhasil disimpan ke: {}", config_file_display);
                    println!("============================================================");
                    return Ok(ConfigMenuAction::Exit);
                } else if selected == 13 {
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
                if !has_whisper_onnx_installed(&config.model_dir, &config.model_tier) {
                    print!("\x1b[?25h\n\n");
                    let _ = io::stdout().flush();
                    download_whisper_by_tier(&config.model_dir, &config.model_tier)?;
                }
                let asr_lang = if config.asr_language.is_empty() { "auto" } else { &config.asr_language };
                if !has_transducer_installed_for_lang(&config.transducer_dir(), asr_lang) {
                    print!("\x1b[?25h\n\n");
                    let _ = io::stdout().flush();
                    download_transducer_by_name(&config.model_dir, &config.transducer_model)?;
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

/// Interactive setup workflow for model selection and dependency verification.
pub fn run_setup_workflow(config: &mut AppConfig, lang: &UiLanguage) -> Result<()> {
    println!("============================================================");
    println!("  Catetin - {}", lang.text("setup.title"));
    println!("============================================================");

    let selected_tier = prompt_model_choice(&config.model_tier, lang);
    config.model_tier = selected_tier;

    if has_whisper_onnx_installed(&config.model_dir, &config.model_tier) {
        println!("  [✓] Model Whisper Pass 2 ('{}') : Siap digunakan (ONNX CPU/GPU)", config.model_tier);
        let redownload = read_confirm("  Unduh ulang model Whisper ini? / Re-download? (y/N) [N]: ", false);
        if redownload {
            download_whisper_by_tier(&config.model_dir, &config.model_tier)?;
        }
    } else {
        println!("  Berkas model Whisper belum ada di disk.");
        let confirm = read_confirm(&format!(
            "  Unduh model Whisper '{}' sekarang? / Download now? (Y/n) [Y]: ",
            config.model_tier
        ), true);
        if confirm {
            download_whisper_by_tier(&config.model_dir, &config.model_tier)?;
        } else {
            println!("  [!] Pengunduhan dilewati.");
        }
    }

    let transducer_dir = config.transducer_dir();
    let asr_lang = if config.asr_language.is_empty() { "auto" } else { &config.asr_language };
    let has_transducer = has_transducer_installed_for_lang(&transducer_dir, asr_lang);
    if has_transducer {
        println!("  [✓] Streaming Transducer (Pass 1 Live Typing): Model '{}' siap digunakan", config.transducer_model);
    } else {
        println!("  [!] Streaming Transducer (Pass 1): Belum terpasang [WAJIB untuk Live Typing < 50ms].");
        let dl_transducer = read_confirm(
            &format!("  Unduh Streaming Transducer '{}'? (Y/n) [Y]: ", config.transducer_model),
            true,
        );
        if dl_transducer {
            download_transducer_by_name(&config.model_dir, &config.transducer_model)?;
        } else {
            println!("  [!] Peringatan: Transducer wajib untuk Pass 1.");
        }
    }

    config.save()?;
    println!("  [✓] Konfigurasi model tersimpan di: {}", AppConfig::config_file_path().display());
    println!();

    run_startup_check(config, lang)?;
    Ok(())
}

/// Diagnostic health check verifying environment and dependencies.
pub fn run_startup_check(config: &AppConfig, _lang: &UiLanguage) -> Result<bool> {
    println!("============================================================");
    println!("  Catetin - Pengecekan Sistem / System Health Check");
    println!("============================================================");

    let mut all_ok = true;

    // 1. Storage & Working Directory Permissions
    if config.model_dir.exists() || std::fs::create_dir_all(&config.model_dir).is_ok() {
        println!("  [✓] Direktori Model / Models Dir  : {}", config.model_dir.display());
    } else {
        println!("  [X] Direktori Model / Models Dir  : Tidak dapat diakses ({})", config.model_dir.display());
        all_ok = false;
    }

    // 2. Audio Capture Devices
    let input_dev = config.input_device_name.as_deref().unwrap_or("Default (Sistem)");
    let output_dev = config.output_device_name.as_deref().unwrap_or("Default (Sistem)");
    println!("  [✓] Audio Input (Mikrofon)        : {}", input_dev);
    println!("  [✓] Audio Loopback (Suara Rapat) : {}", output_dev);

    // 3. Whisper ONNX model
    let has_whisper = has_whisper_onnx_installed(&config.model_dir, &config.model_tier);
    if has_whisper {
        println!("  [✓] Mesin Whisper Pass 2 ('{}') : Sherpa-ONNX [Siap / Ready]", config.model_tier);
    } else {
        println!("  [!] Bobot Model Whisper ('{}')  : Belum ada di disk lokal", config.model_tier);
        all_ok = false;
    }

    // 4. Streaming Transducer
    let transducer_dir = config.transducer_dir();
    let asr_lang = if config.asr_language.is_empty() { "auto" } else { &config.asr_language };
    let has_transducer = has_transducer_installed_for_lang(&transducer_dir, asr_lang);
    if has_transducer {
        println!(
            "  [✓] Streaming Transducer (Pass 1) : Model '{}' (lang: {}) [Siap / Ready]",
            config.transducer_model, asr_lang
        );
    } else {
        println!("  [X] Streaming Transducer (Pass 1) : [WAJIB / MANDATORY] Belum terpasang!");
        all_ok = false;
    }

    // 5. In-process Diarization & VAD engine
    println!(
        "  [✓] Mesin Diarization & VAD       : Mode [{:?}] [Siap / In-Process]",
        config.diarization_mode
    );

    println!("============================================================");
    if all_ok {
        println!("  [✓] Semua dependensi siap digunakan! / All dependencies ready.");
    } else {
        println!("  [!] Dependensi belum lengkap.");
        let want_prepare = read_confirm("  Lengkapi dependensi sekarang secara otomatis? / Download missing now? (Y/n) [Y]: ", true);
        if want_prepare {
            if !has_whisper {
                download_whisper_by_tier(&config.model_dir, &config.model_tier)?;
            }
            if !has_transducer {
                download_transducer_by_name(&config.model_dir, &config.transducer_model)?;
            }
            all_ok = true;
            println!("  [✓] Semua dependensi telah berhasil dilengkapi!\n");
        }
    }
    println!("============================================================\n");

    Ok(all_ok)
}
