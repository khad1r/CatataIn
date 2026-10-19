use anyhow::{Context, Result};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use catetin_core::asr::create_speech_recognizer;
use catetin_core::audio::DualAudioCapture;
use catetin_core::config::AppConfig;
use catetin_core::events::{SpeakerId, TranscriptEvent};
use catetin_core::i18n::UiLanguage;
use catetin_core::logging::init_logger;
use catetin_core::pipeline::{CatetinPipeline, DiarizationMode};

mod assistant;
mod console;
mod file_dialog;
mod kb;
mod models;
mod tui;
mod wizard;

enum AiTaskResult {
    Summary(Vec<String>, Vec<String>),
    Answer(String, String),
    Error(String),
}

fn print_help(lang: &UiLanguage) {
    println!("Catetin CLI - {}", lang.text("app.subtitle"));
    println!();
    println!("{} catetin [PERINTAH / COMMAND]", lang.text("cli.usage"));
    println!();
    println!("Perintah / Commands:");
    println!("  catetin                  Jalankan transkripsi rapat langsung / Start meeting transcription");
    println!("  catetin -c, --config     Buka menu konfigurasi interaktif / Interactive config wizard");
    println!("  catetin check, --check   Verifikasi kesiapan sistem & model / Verify system readiness");
    println!("  catetin setup, --setup   Periksa & unduh model / Check & download missing models");
    println!("  catetin -h, --help       Tampilkan bantuan ini / Show this help message");
    println!("  catetin --version        Tampilkan versi aplikasi / Show version information");
    println!();
    println!("Petunjuk:");
    println!("  Seluruh pengaturan (Bahasa Audio, Mode Rapat, Model Whisper/Transducer, Mikrofon,");
    println!("  Loopback, Peredam Bising, AI Copilot, dan Verbose) diatur interaktif via: 'catetin -c'");
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

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();

    let mut trigger_wizard = false;
    let mut trigger_setup = false;
    let mut trigger_check = false;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--config" | "-c" | "config" => {
                trigger_wizard = true;
            }
            "--setup" | "setup" => {
                trigger_setup = true;
            }
            "--check" | "check" => {
                trigger_check = true;
            }
            "--help" | "-h" | "help" => {
                print_help(&UiLanguage::default());
                return Ok(());
            }
            "--version" => {
                println!("Catetin v{}", env!("CARGO_PKG_VERSION"));
                return Ok(());
            }
            unknown => {
                eprintln!("Opsi tidak dikenal: '{}'.", unknown);
                eprintln!("Seluruh pengaturan rapat diatur lewat menu konfigurasi: 'catetin -c'");
                eprintln!("Jalankan 'catetin --help' untuk bantuan.\n");
                return Ok(());
            }
        }
        i += 1;
    }

    let mut config = AppConfig::load_or_default();

    if trigger_wizard {
        let config_display = AppConfig::config_file_path().display().to_string();

        match wizard::run_interactive_wizard(&mut config, &config_display)? {
            wizard::ConfigMenuAction::StartMeeting => {
                // User chose to start meeting directly! Proceed with initialized pipeline.
            }
            wizard::ConfigMenuAction::Exit => {
                return Ok(());
            }
        }
    }

    let current_lang = config.ui_language.clone();

    if trigger_setup {
        wizard::run_setup_workflow(&mut config, &current_lang)?;
        return Ok(());
    }

    if trigger_check {
        wizard::run_startup_check(&config, &current_lang)?;
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
    println!("  Bahasa Audio Rapat: {}", config.asr_language_display());
    println!("  {}:         {}", current_lang.text("label.mode"), mode_desc);
    println!(
        "  {}:   {} ({})",
        current_lang.text("label.model_tier"),
        config.model_tier,
        config.model_display_name()
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
    models::check_and_prepare_dependencies(&mut config, &current_lang)?;

    println!("  -> Mengaktifkan mesin Whisper (100% offline)...");
    let asr_lang = if config.asr_language.is_empty() {
        "auto"
    } else {
        config.asr_language.as_str()
    };
    let recognizer = create_speech_recognizer(
        &config.model_dir,
        &config.model_tier,
        asr_lang,
    )?;
    println!("  [OK] Mesin Transkripsi Pass 2 (Whisper) aktif & siap digunakan.\n");

    println!("  -> Mengaktifkan mesin Streaming Transducer (Pass 1 live typing)...");
    let mic_transducer = catetin_core::asr::create_transducer_for_lang(&config.transducer_dir(), asr_lang)
        .context("Failed to initialize microphone Streaming Transducer")?;
    let lb_transducer = catetin_core::asr::create_transducer_for_lang(&config.transducer_dir(), asr_lang)
        .context("Failed to initialize loopback Streaming Transducer")?;
    println!("  [OK] Mesin Transducer Pass 1 aktif & siap digunakan.\n");

    // 7. Initialize dual-channel audio capture (Microphone + Loopback)
    println!("  -> Mengaktifkan penangkap audio (Mikrofon & Audio Sistem)...");
    let capture = match DualAudioCapture::from_config(&config) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("\n[ERROR] Audio capture failed: {:#}", e);
            eprintln!("Ensure a microphone and default playback device are available.");
            return Err(e);
        }
    };
    println!("  [OK] Audio capture aktif. Membuka layar rapat...\n");

    let mic_display = config.input_device_name.as_deref().unwrap_or("Default Mic");
    let loopback_display = config.output_device_name.as_deref().unwrap_or("Default Loopback");

    // 8. Launch headless transcription & diarization pipeline
    let (event_tx, event_rx) = std::sync::mpsc::channel::<TranscriptEvent>();
    let pipeline = CatetinPipeline::new(recognizer)
        .with_mode(config.diarization_mode)
        .with_asr_language(config.asr_language.clone())
        .with_noise_suppression(config.noise_suppression)
        .with_models_dir(config.model_dir.clone())
        .with_transducer_dir(config.transducer_dir())
        .with_transducers(mic_transducer, lb_transducer);
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
    let mut segment_meta: Vec<(u64, SpeakerId, bool, bool)> = Vec::new();
    let mut left_lines: Vec<String> = Vec::new();
    let mut live_raw_turns: Vec<(u64, String)> = Vec::new();
    let live_raw_speaker = SpeakerId::LocalUser;
    let mut live_raw_text = String::new();
    let mut scroll_offset: usize = 0;
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

    // Enter raw console mode to enable immediate non-blocking hotkey detection
    let _raw_guard = console::RawConsoleGuard::enter();

    #[cfg(windows)]
    unsafe {
        extern "system" {
            fn GetCurrentThread() -> isize;
            fn SetThreadPriority(h_thread: isize, n_priority: i32) -> i32;
        }
        // Boost interactive UI thread priority so terminal rendering stays smooth
        let _ = SetThreadPriority(GetCurrentThread(), 1);
    }

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
        &live_raw_speaker,
        &live_raw_text,
        scroll_offset,
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

            if !paused {
                let current_mic = handle.mic_level();
                let current_lb = handle.loopback_level();
                let combined_level = current_mic.max(current_lb * 3.5);
                wave_history.push(combined_level);
                if wave_history.len() > 16 {
                    wave_history.remove(0);
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
                match key {
                    k if k == b'q' as i32 || k == b'Q' as i32 || k == 3 => {
                        quit_requested = true;
                        break;
                    }
                    k if k == b'p' as i32 || k == b'P' as i32 || k == b' ' as i32 => {
                        paused = !paused;
                        handle.set_paused(paused);
                        if paused {
                            wave_history.fill(0.0);
                            was_audio_active = false;
                        }
                        needs_redraw = true;
                    }
                    // Scrolling Hotkeys: Up / Down / PgUp / PgDn / Home / End / k / j / g
                    k if k == 1001 || k == b'k' as i32 || k == b'K' as i32 || k == b'w' as i32 || k == b'W' as i32 => {
                        scroll_offset = scroll_offset.saturating_add(3);
                        needs_redraw = true;
                    }
                    k if k == 1002 || k == b'j' as i32 || k == b'J' as i32 => {
                        scroll_offset = scroll_offset.saturating_sub(3);
                        needs_redraw = true;
                    }
                    1003 => {
                        scroll_offset = scroll_offset.saturating_add(10);
                        needs_redraw = true;
                    }
                    1004 => {
                        scroll_offset = scroll_offset.saturating_sub(10);
                        needs_redraw = true;
                    }
                    1005 => {
                        scroll_offset = 99999;
                        needs_redraw = true;
                    }
                    k if k == 1006 || k == b'g' as i32 || k == b'G' as i32 => {
                        scroll_offset = 0;
                        needs_redraw = true;
                    }
                    k if k == b'm' as i32 || k == b'M' as i32 => {
                        let _ = handle.toggle_mode();
                        needs_redraw = true;
                    }
                    k if k == b'l' as i32 || k == b'L' as i32 => {
                        current_lang = match current_lang {
                            UiLanguage::Indonesian => UiLanguage::English,
                            _ => UiLanguage::Indonesian,
                        };
                        needs_redraw = true;
                    }
                    k if k == b'c' as i32 || k == b'C' as i32 => {
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
                    k if (k == b's' as i32 || k == b'S' as i32) && config.show_assistant => {
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
                    k if (k == b'?' as i32 || k == b'/' as i32) && config.show_assistant => {
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

        // D. Process real-time transcription events:
        // - Pass 1 Transducer (!event.is_final): Live streaming raw text appended directly into the transcript with timestamp
        // - Pass 2 / Whisper (event.is_final): Preserves Transducer and inserts Whisper right next to it for comparison
        let mut rebuild_left = false;
        while let Ok(event) = event_rx.try_recv() {
            if !event.is_final {
                // Pass 1: Live raw streaming (Transducer <50ms)
                let trimmed = event.text.trim();
                if trimmed.is_empty() {
                    continue;
                }

                let turn_id = event.replaces_since_ms.unwrap_or(event.timestamp_ms);

                // Update existing turn or append new turn in live_raw_turns
                if let Some(pos) = live_raw_turns.iter().position(|(tid, _)| *tid == turn_id) {
                    live_raw_turns[pos].1 = trimmed.to_string();
                } else {
                    live_raw_turns.push((turn_id, trimmed.to_string()));
                }

                // Update live_raw_text by joining all active live raw turns (pure raw text without tags)
                live_raw_text = live_raw_turns
                    .iter()
                    .map(|(_, txt)| txt.as_str())
                    .collect::<Vec<_>>()
                    .join(" ");

                needs_redraw = true;
                continue;
            }

            // Pass 2: Model finished processing (Whisper):
            let final_ts = event.replaces_since_ms.unwrap_or(event.timestamp_ms);
            let trimmed_text = event.text.trim().to_string();

            // 1. Atomically remove the finished turn(s) from live_raw_turns:
            // Any turn with turn_id <= final_ts has been processed by Whisper.
            live_raw_turns.retain(|(tid, _)| *tid > final_ts);

            live_raw_text = live_raw_turns
                .iter()
                .map(|(_, txt)| txt.as_str())
                .collect::<Vec<_>>()
                .join(" ");

            // 2. If Whisper produced valid text, add to recorded_segments:
            if !trimmed_text.is_empty() {
                let time_str = format_ms(final_ts);
                let speaker_name = event.speaker.to_localized_string(&current_lang);
                recorded_segments.push((time_str, speaker_name, trimmed_text));
                segment_meta.push((final_ts, event.speaker, true, true));
                rebuild_left = true;
            }

            needs_redraw = true;
        }

        // Safety clear stale raw turns when silence resumes and no transcription is in flight for > 3s
        if !paused && !handle.is_speaking() && handle.transcribing_count() == 0 {
            if !live_raw_turns.is_empty() {
                let now_ms = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_millis() as u64;
                if live_raw_turns.iter().all(|(ts, _)| now_ms.saturating_sub(*ts) > 3_000) {
                    live_raw_turns.clear();
                    live_raw_text.clear();
                    needs_redraw = true;
                }
            }
        }

        if rebuild_left {
            let (term_w, _) = last_term_size;
            let inner_w = term_w.saturating_sub(1).max(75).saturating_sub(2);
            let wrap_w = if config.show_assistant {
                ((inner_w.saturating_sub(1) * 52) / 100).saturating_sub(4).max(30)
            } else {
                inner_w.saturating_sub(6).max(40)
            };

            left_lines.clear();

            // Group consecutive segments by speaker
            struct GroupedParagraph {
                time_str: String,
                speaker_id: SpeakerId,
                speaker_name: String,
                text_parts: Vec<String>,
                last_ts: u64,
            }

            let mut paragraphs: Vec<GroupedParagraph> = Vec::new();
            for ((t_str, spk_name, txt), &(ts, ref spk_id, _, _)) in
                recorded_segments.iter().zip(segment_meta.iter())
            {
                let clean_txt = txt.trim();
                if clean_txt.is_empty() {
                    continue;
                }

                let can_merge = if let Some(last) = paragraphs.last() {
                    last.speaker_id == *spk_id
                        && (ts == 0 || last.last_ts == 0 || ts.saturating_sub(last.last_ts) <= 30_000)
                } else {
                    false
                };

                if can_merge {
                    let last = paragraphs.last_mut().unwrap();
                    if last.time_str.is_empty() && !t_str.is_empty() {
                        last.time_str = t_str.clone();
                    }
                    last.text_parts.push(clean_txt.to_string());
                    if ts > 0 {
                        last.last_ts = ts;
                    }
                } else {
                    paragraphs.push(GroupedParagraph {
                        time_str: t_str.clone(),
                        speaker_id: spk_id.clone(),
                        speaker_name: spk_name.clone(),
                        text_parts: vec![clean_txt.to_string()],
                        last_ts: ts,
                    });
                }
            }

            for p in &paragraphs {
                let color = match p.speaker_id {
                    SpeakerId::LocalUser => "\x1b[1;36m", // Bright Cyan (You)
                    SpeakerId::Remote(1) | SpeakerId::InPerson(1) => "\x1b[1;33m", // Bright Yellow
                    SpeakerId::Remote(2) | SpeakerId::InPerson(2) => "\x1b[1;32m", // Bright Green
                    SpeakerId::Remote(3) | SpeakerId::InPerson(3) => "\x1b[1;35m", // Bright Magenta
                    _ => "\x1b[1;34m",                    // Bright Blue
                };
                let reset = "\x1b[0m";

                let time_prefix = if !p.time_str.is_empty() {
                    format!("{} ", p.time_str)
                } else {
                    String::new()
                };

                let header = format!(
                    "{}{}[{}]:{}",
                    time_prefix, color, p.speaker_name, reset
                );
                left_lines.push(header);

                let full_body = p.text_parts.join(" ");
                for chunk in tui::textwrap_simple(&full_body, wrap_w) {
                    left_lines.push(format!("  \"{}\"", chunk));
                }
                left_lines.push(String::new());
            }

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
                &live_raw_speaker,
                &live_raw_text,
                scroll_offset,
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
        if event.is_final && !event.text.trim().is_empty() {
            let time_str = format_ms(event.timestamp_ms);
            let speaker_name = event.speaker.to_localized_string(&current_lang);
            recorded_segments.push((time_str, speaker_name, event.text.trim().to_string()));
            segment_meta.push((event.timestamp_ms, event.speaker, true, true));
        }
    }

    // Clean up any remaining non-final raw entries before exit dialog
    let mut i = 0;
    while i < segment_meta.len() {
        if !segment_meta[i].2 {
            segment_meta.remove(i);
            recorded_segments.remove(i);
        } else {
            i += 1;
        }
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
                let answer = kb::read_line("");
                let trimmed = answer.trim().to_lowercase();
                break !matches!(trimmed.as_str(), "d" | "n" | "no" | "discard" | "buang" | "batal");
            }
            #[cfg(windows)]
            std::thread::sleep(Duration::from_millis(20));
        };

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
        drop(_raw_guard);

        println!("============================================================");
        println!("  Session ended. / Sesi selesai (tidak ada rekaman suara).");
        println!("============================================================");
    }

    Ok(())
}
