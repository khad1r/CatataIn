use anyhow::Result;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use catetin_core::asr::{MockRecognizer, SpeechRecognizer};
#[cfg(feature = "whisper")]
use catetin_core::asr::whisper_backend::WhisperEngine;
use catetin_core::audio::DualAudioCapture;
use catetin_core::config::AppConfig;
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

/// Interactive terminal configuration wizard.
fn run_interactive_wizard(config: &mut AppConfig) -> Result<()> {
    let mut lang = config.ui_language.clone();

    println!("============================================================");
    println!("  Catetin - {}", lang.text("wizard.title"));
    println!("============================================================");

    // Step 1: Language
    println!("{}", lang.text("wizard.step_lang"));
    println!("  [1] Bahasa Indonesia (id) [Bawaan/Default]");
    println!("  [2] English (en)");
    let lang_input = read_line(&format!("  {} [1]: ", lang.text("wizard.prompt_choice")));
    match lang_input.as_str() {
        "2" | "en" | "english" => {
            lang = UiLanguage::English;
        }
        "1" | "id" | "indonesian" | "" => {
            lang = UiLanguage::Indonesian;
        }
        custom => {
            lang = UiLanguage::from_code(custom);
        }
    }
    config.ui_language = lang.clone();
    println!(
        "  -> {}: {} ({})\n",
        lang.text("label.language"),
        lang.display_name(),
        lang.code()
    );

    // Step 2: Diarization Mode
    println!("{}", lang.text("wizard.step_mode"));
    println!("  [1] {} [Bawaan/Default]", lang.text("label.mode_online"));
    println!("  [2] {}", lang.text("label.mode_in_person"));
    println!("  [3] {}", lang.text("label.mode_hybrid"));
    let mode_input = read_line(&format!("  {} [1]: ", lang.text("wizard.prompt_choice")));
    match mode_input.as_str() {
        "2" | "in-person" | "roundtable" => {
            config.diarization_mode = DiarizationMode::InPerson;
        }
        "3" | "hybrid" => {
            config.diarization_mode = DiarizationMode::Hybrid;
        }
        _ => {
            config.diarization_mode = DiarizationMode::OnlineMeeting;
        }
    }
    let mode_desc = match config.diarization_mode {
        DiarizationMode::OnlineMeeting => lang.text("label.mode_online"),
        DiarizationMode::InPerson => lang.text("label.mode_in_person"),
        DiarizationMode::Hybrid => lang.text("label.mode_hybrid"),
    };
    println!("  -> {}: {}\n", lang.text("label.mode"), mode_desc);

    // Step 3: Whisper Model Tier
    println!("{}", lang.text("wizard.step_model"));
    println!("  [1] tiny   (~75 MB  - Sangat cepat / Ultra fast)");
    println!("  [2] base   (~140 MB - Seimbang / Balanced) [Bawaan/Default]");
    println!("  [3] small  (~460 MB - Akurasi tinggi / High accuracy)");
    println!("  [4] medium (~1.5 GB - Akurasi maksimal / Best accuracy)");
    let model_input = read_line(&format!("  {} [2]: ", lang.text("wizard.prompt_choice")));
    match model_input.as_str() {
        "1" | "tiny" => config.model_tier = "tiny".to_string(),
        "3" | "small" => config.model_tier = "small".to_string(),
        "4" | "medium" => config.model_tier = "medium".to_string(),
        _ => config.model_tier = "base".to_string(),
    }
    println!(
        "  -> {}: {} ({})\n",
        lang.text("label.model_tier"),
        config.model_tier,
        config.model_filename()
    );

    // Step 4: Microphone Input Device
    println!("4. Pilih Perangkat Mikrofon / Select Microphone Device:");
    println!("  [1] Bawaan Sistem / System Default");
    let input_devices = catetin_core::audio::list_input_devices();
    for (idx, dev_name) in input_devices.iter().enumerate() {
        println!("  [{}] {}", idx + 2, dev_name);
    }
    let dev_input = read_line(&format!("  {} [1]: ", lang.text("wizard.prompt_choice")));
    if let Ok(choice_num) = dev_input.parse::<usize>() {
        if choice_num >= 2 && choice_num <= input_devices.len() + 1 {
            config.input_device_name = Some(input_devices[choice_num - 2].clone());
        } else {
            config.input_device_name = None;
        }
    } else if !dev_input.is_empty() {
        config.input_device_name = Some(dev_input);
    } else {
        config.input_device_name = None;
    }
    println!(
        "  -> {}: {}\n",
        lang.text("label.microphone"),
        config.input_device_name.as_deref().unwrap_or("Default")
    );

    // Step 5: System Audio (Loopback) Output Device
    println!("5. Pilih Audio Sistem (Speaker/Headphones) / System Audio Device:");
    println!("  [1] Bawaan Sistem / System Default");
    let output_devices = catetin_core::audio::list_output_devices();
    for (idx, dev_name) in output_devices.iter().enumerate() {
        println!("  [{}] {}", idx + 2, dev_name);
    }
    let loopback_input = read_line(&format!("  {} [1]: ", lang.text("wizard.prompt_choice")));
    if let Ok(choice_num) = loopback_input.parse::<usize>() {
        if choice_num >= 2 && choice_num <= output_devices.len() + 1 {
            config.output_device_name = Some(output_devices[choice_num - 2].clone());
        } else {
            config.output_device_name = None;
        }
    } else if !loopback_input.is_empty() {
        config.output_device_name = Some(loopback_input);
    } else {
        config.output_device_name = None;
    }
    println!(
        "  -> {}: {}\n",
        lang.text("label.loopback"),
        config.output_device_name.as_deref().unwrap_or("Default")
    );

    // Step 6: AI Meeting Assistant (Phase 5)
    println!("6. Aktifkan Split-Screen AI Assistant Copilot? / Enable AI Copilot Pane?");
    println!("  [1] Ya / Yes (Split-Screen Copilot) [Bawaan/Default]");
    println!("  [2] Tidak / No (Scrolling Single Pane)");
    let copilot_input = read_line(&format!("  {} [1]: ", lang.text("wizard.prompt_choice")));
    config.show_assistant = !matches!(copilot_input.as_str(), "2" | "n" | "no");
    println!("  -> AI Assistant Pane: {}\n", config.show_assistant);

    // Step 7: Verbose logging
    println!("{}", lang.text("wizard.step_verbose"));
    let verbose_input = read_line(&format!("  {} [N]: ", lang.text("wizard.prompt_choice")));
    config.verbose = matches!(verbose_input.to_lowercase().as_str(), "y" | "yes" | "ya");
    println!("  -> Verbose stderr: {}\n", config.verbose);

    // Step 8: Save & Exit
    config.save()?;
    println!("============================================================");
    println!("  [OK] {}", lang.text("wizard.saved_success"));
    println!("  {}", AppConfig::config_file_path().display());
    println!("============================================================");

    Ok(())
}

fn print_help(lang: &UiLanguage) {
    println!("Catetin CLI - {}", lang.text("app.subtitle"));
    println!();
    println!("{} catetin [OPTIONS]", lang.text("cli.usage"));
    println!();
    println!("{}", lang.text("cli.options"));
    println!("  --config, -c [path]      {}", lang.text("cli.help_config"));
    println!("  --assistant              Enable split-screen AI meeting copilot pane");
    println!("  --no-assistant           Disable split pane; run in standard scrolling mode");
    println!("  --verbose, -v            {}", lang.text("cli.help_verbose"));
    println!("  --help, -h               Show this help message / Tampilkan bantuan");
    println!("  --version                Show version information / Tampilkan versi");
    println!();
    println!("Examples / Contoh:");
    println!("  catetin                       # Jalankan dengan konfigurasi tersimpan");
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

// ponytail: ANSI in-place redraw instead of ratatui layout engine.
// Ceiling: Fixed 101-column split-pane layout.
// Upgrade path: add ratatui if dynamic multi-split responsive layouts are requested.
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

    pub fn render_copilot_tui(
        config: &AppConfig,
        lang: &UiLanguage,
        mode: DiarizationMode,
        mic_name: &str,
        loopback_name: &str,
        paused: bool,
        left_lines: &[String],
        assistant: &AssistantState,
    ) {
        let mut out = String::with_capacity(4096);
        // Move cursor to top-left without clearing screen (flicker-free)
        out.push_str("\x1b[H");

        // Top border
        let status_str = if paused { "[PAUSED / DIJEDA]" } else { "[100% OFFLINE / LIVE]" };
        let title = format!("┌─ Catetin Copilot v{} {} ", env!("CARGO_PKG_VERSION"), status_str);
        let title_pad = 99usize.saturating_sub(title.chars().count());
        out.push_str(&title);
        out.push_str(&"─".repeat(title_pad));
        out.push_str("┐\n");

        // Status banner
        let mode_name = match mode {
            DiarizationMode::OnlineMeeting => "Online",
            DiarizationMode::InPerson => "In-Person",
            DiarizationMode::Hybrid => "Hybrid",
        };
        let line2 = format!(
            " Audio: [●] {} | [●] {} │ Lang: [{}] │ ASR: [{}] │ Mode: [{}]",
            mic_name, loopback_name, lang.code(), config.model_tier, mode_name
        );
        out.push_str(&format!("│{}│\n", fit_cell(&line2, 99)));

        // Split Divider
        out.push_str("├────────────────────────────────────────────────┬──────────────────────────────────────────────────┤\n");

        // Pane Headers
        let right_hdr = if let Some(ref st) = assistant.status_msg {
            format!(" LIVE SUMMARY & ACTION ITEMS [{}]", st)
        } else {
            " LIVE SUMMARY & ACTION ITEMS".to_string()
        };
        out.push_str(&format!(
            "│{}│{}│\n",
            fit_cell(" LIVE DIARIZED TRANSCRIPT", 48),
            fit_cell(&right_hdr, 50)
        ));

        // Prepare left visible lines (last 16 lines)
        let left_start = left_lines.len().saturating_sub(16);
        let visible_left = &left_lines[left_start..];

        // Prepare right upper lines (Topics & Action items, 10 lines)
        let mut right_upper: Vec<String> = Vec::new();
        right_upper.push(" • Key Topics / Topik Utama:".to_string());
        if assistant.topics.is_empty() {
            right_upper.push("   - (Tekan 's' untuk generate ringkasan AI)".to_string());
        } else {
            for t in &assistant.topics {
                for chunk in textwrap_simple(t, 46) {
                    right_upper.push(format!("   - {}", chunk));
                }
            }
        }
        right_upper.push(" • Action Items / Tindak Lanjut:".to_string());
        if assistant.actions.is_empty() {
            right_upper.push("   [ ] (Belum ada catatan tindak lanjut)".to_string());
        } else {
            for a in &assistant.actions {
                for chunk in textwrap_simple(a, 46) {
                    right_upper.push(format!("   {}", chunk));
                }
            }
        }

        // Rows 0..9 (Upper Right + Left)
        for i in 0..10 {
            let left_cell = visible_left.get(i).map(|s| s.as_str()).unwrap_or("");
            let right_cell = right_upper.get(i).map(|s| s.as_str()).unwrap_or("");
            out.push_str(&format!(
                "│{}│{}│\n",
                fit_cell(left_cell, 48),
                fit_cell(right_cell, 50)
            ));
        }

        // Horizontal divider in right pane
        let left_cell_10 = visible_left.get(10).map(|s| s.as_str()).unwrap_or("");
        out.push_str(&format!(
            "│{}├──────────────────────────────────────────────────┤\n",
            fit_cell(left_cell_10, 48)
        ));

        // Ask Assistant header
        let left_cell_11 = visible_left.get(11).map(|s| s.as_str()).unwrap_or("");
        out.push_str(&format!(
            "│{}│{}│\n",
            fit_cell(left_cell_11, 48),
            fit_cell(" ASK ASSISTANT (? <pertanyaan> untuk tanya)", 50)
        ));

        // Prepare right lower Q&A lines (4 lines)
        let mut right_lower: Vec<String> = Vec::new();
        if let Some((q, a)) = assistant.qa_history.last() {
            right_lower.push(format!(" > {}", q));
            for chunk in textwrap_simple(a, 46) {
                right_lower.push(format!("   {}", chunk));
            }
        } else {
            right_lower.push("   Ketik '? <tanya>' untuk bertanya tentang rapat".to_string());
            right_lower.push("   Type '? <question>' to ask about meeting".to_string());
        }

        // Rows 12..15 (Lower Right + Left)
        for i in 0..4 {
            let left_cell = visible_left.get(12 + i).map(|s| s.as_str()).unwrap_or("");
            let right_cell = right_lower.get(i).map(|s| s.as_str()).unwrap_or("");
            out.push_str(&format!(
                "│{}│{}│\n",
                fit_cell(left_cell, 48),
                fit_cell(right_cell, 50)
            ));
        }

        // Bottom border & command bar
        out.push_str("├────────────────────────────────────────────────┴──────────────────────────────────────────────────┤\n");
        out.push_str("│ [p] Pause/Resume │ [s] Summary AI │ [? <tanya>] Ask │ [m] Mode │ [c] Simpan │ [q] Keluar            │\n");
        out.push_str("└───────────────────────────────────────────────────────────────────────────────────────────────────┘\n");

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

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();

    let mut custom_config_path = None;
    let mut trigger_wizard = false;
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
        return run_interactive_wizard(&mut config);
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

    // 6. Resolve speech recognizer (Whisper engine or lightweight mock)
    let model_path = config.model_dir.join(config.model_filename());
    let recognizer: Box<dyn SpeechRecognizer> = {
        #[cfg(feature = "whisper")]
        {
            if model_path.exists() {
                println!("  [OK] Whisper model: {}", model_path.display());
                Box::new(WhisperEngine::load_from_file(&model_path, &config.asr_language)?)
            } else {
                println!("  [!] Offline model not found at: {}", model_path.display());
                println!("      Place {} in that folder for full Whisper ASR.", config.model_filename());
                println!("      Running in live diagnostic mode (MockRecognizer).\n");
                Box::new(MockRecognizer::new(&config.asr_language))
            }
        }
        #[cfg(not(feature = "whisper"))]
        {
            Box::new(MockRecognizer::new(&config.asr_language))
        }
    };

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
    let pipeline = CatetinPipeline::new(recognizer).with_mode(config.diarization_mode);
    let mut handle = pipeline.start(capture, event_tx)?;

    // Spawn non-blocking stdin command listener
    // ponytail: line-buffered stdin commands instead of raw crossterm. Ceiling: requires pressing Enter after commands ('m<Enter>', 'q<Enter>'). Upgrade path: add crossterm raw mode if single-keystroke hotkeys are explicitly demanded.
    let (cmd_tx, cmd_rx) = std::sync::mpsc::channel::<String>();
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

    // Channel for background AI worker tasks (Summary & Q&A)
    let (ai_tx, ai_rx) = std::sync::mpsc::channel::<AiTaskResult>();

    let mut recorded_segments: Vec<(String, String, String)> = Vec::new();
    let mut left_lines: Vec<String> = Vec::new();
    let mut assistant_state = tui::AssistantState::default();
    let mut paused = false;

    // Clear terminal screen and hide cursor for smooth TUI presentation
    if config.show_assistant {
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
        );
    } else {
        println!("============================================================");
        println!("  [STATUS] {}", current_lang.text("status.listening"));
        println!("  Commands: [m] Mode | [l] Lang | [s] Summary | [q] Quit");
        println!("============================================================\n");
    }

    loop {
        // A. Handle background AI worker responses
        if let Ok(res) = ai_rx.try_recv() {
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
            if config.show_assistant {
                tui::render_copilot_tui(
                    &config,
                    &current_lang,
                    handle.current_mode(),
                    mic_display,
                    loopback_display,
                    paused,
                    &left_lines,
                    &assistant_state,
                );
            }
        }

        // B. Process interactive keyboard commands
        if let Ok(cmd) = cmd_rx.try_recv() {
            let cmd_lower = cmd.to_lowercase();
            match cmd_lower.as_str() {
                "q" | "quit" | "keluar" | "exit" => {
                    break;
                }
                "p" | "pause" | " " => {
                    paused = !paused;
                    if config.show_assistant {
                        tui::render_copilot_tui(
                            &config,
                            &current_lang,
                            handle.current_mode(),
                            mic_display,
                            loopback_display,
                            paused,
                            &left_lines,
                            &assistant_state,
                        );
                    } else {
                        println!("\n  >> [CAPTURE {}]\n", if paused { "PAUSED" } else { "RESUMED" });
                    }
                }
                "m" | "mode" => {
                    let next = handle.toggle_mode();
                    if config.show_assistant {
                        tui::render_copilot_tui(
                            &config,
                            &current_lang,
                            next,
                            mic_display,
                            loopback_display,
                            paused,
                            &left_lines,
                            &assistant_state,
                        );
                    } else {
                        let desc = match next {
                            DiarizationMode::OnlineMeeting => current_lang.text("label.mode_online"),
                            DiarizationMode::InPerson => current_lang.text("label.mode_in_person"),
                            DiarizationMode::Hybrid => current_lang.text("label.mode_hybrid"),
                        };
                        println!("\n  >> [MODE] {}: {}\n", current_lang.text("label.mode"), desc);
                    }
                }
                "l" | "lang" => {
                    current_lang = match current_lang {
                        UiLanguage::Indonesian => UiLanguage::English,
                        _ => UiLanguage::Indonesian,
                    };
                    if config.show_assistant {
                        tui::render_copilot_tui(
                            &config,
                            &current_lang,
                            handle.current_mode(),
                            mic_display,
                            loopback_display,
                            paused,
                            &left_lines,
                            &assistant_state,
                        );
                    } else {
                        println!(
                            "\n  >> [LANG] {} ({})\n",
                            current_lang.display_name(),
                            current_lang.code()
                        );
                    }
                }
                "s" | "summary" => {
                    // Trigger AI summary in background
                    assistant_state.status_msg = Some("Generating Summary...".into());
                    if config.show_assistant {
                        tui::render_copilot_tui(
                            &config,
                            &current_lang,
                            handle.current_mode(),
                            mic_display,
                            loopback_display,
                            paused,
                            &left_lines,
                            &assistant_state,
                        );
                    } else {
                        println!("  [AI] Generating summary with LLM...");
                    }

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
                    if !config.show_assistant {
                        println!("\n  >> [SNAPSHOT] {} {}\n", current_lang.text("prompt.saved_success"), default_path.display());
                    }
                }
                custom_query if custom_query.starts_with('?') || custom_query.starts_with("ask ") => {
                    // User asking a question to the assistant
                    let question = custom_query.trim_start_matches(|c| c == '?' || c == ' ').trim().to_string();
                    if !question.is_empty() {
                        assistant_state.status_msg = Some("Thinking...".into());
                        if config.show_assistant {
                            tui::render_copilot_tui(
                                &config,
                                &current_lang,
                                handle.current_mode(),
                                mic_display,
                                loopback_display,
                                paused,
                                &left_lines,
                                &assistant_state,
                            );
                        }

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
                }
                _ => {}
            }
        }

        // C. Process real-time transcription events
        let mut had_event = false;
        while let Ok(event) = event_rx.try_recv() {
            if paused {
                continue;
            }
            had_event = true;
            let time_str = format_ms(event.timestamp_ms);
            let speaker_name = event.speaker.to_localized_string(&current_lang);

            let color = match event.speaker {
                SpeakerId::LocalUser => "\x1b[1;36m",    // Bright Cyan (You)
                SpeakerId::Remote(1) => "\x1b[1;33m",    // Bright Yellow
                SpeakerId::Remote(2) => "\x1b[1;32m",    // Bright Green
                SpeakerId::Remote(3) => "\x1b[1;35m",    // Bright Magenta
                _ => "\x1b[1;34m",                       // Bright Blue
            };
            let reset = "\x1b[0m";

            // Format for left transcript buffer
            let header = format!("{} {}[{}]:{}", time_str, color, speaker_name, reset);
            left_lines.push(header);
            for chunk in tui::textwrap_simple(&event.text, 44) {
                left_lines.push(format!("  \"{}\"", chunk));
            }
            left_lines.push(String::new()); // blank line separator

            recorded_segments.push((time_str.clone(), speaker_name.clone(), event.text.clone()));

            if !config.show_assistant {
                println!("{} {}[{}]{}: {}", time_str, color, speaker_name, reset, event.text);
            }
        }

        if had_event && config.show_assistant {
            tui::render_copilot_tui(
                &config,
                &current_lang,
                handle.current_mode(),
                mic_display,
                loopback_display,
                paused,
                &left_lines,
                &assistant_state,
            );
        }

        if !had_event {
            std::thread::sleep(Duration::from_millis(40));
        }
    }

    handle.stop();

    // Restore terminal cursor
    if config.show_assistant {
        print!("\x1b[?25h\n\n");
        let _ = io::stdout().flush();
    }

    // 9. Ask user where to save the Markdown transcript on exit (Native file picker + terminal fallback)
    if !recorded_segments.is_empty() {
        let default_path = default_transcript_path();
        println!("============================================================");
        println!(
            "  Catatan Rapat / Meeting Transcript: {} rekaman",
            recorded_segments.len()
        );

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
                // If GUI dialog was cancelled by user or if running headless without display:
                println!("  Simpan ke lokasi mana? / Where would you like to save the file?");
                println!("  [Enter] = Bawaan / Default: {}", default_path.display());
                println!("  [n]     = Jangan simpan / Skip saving");
                print!("  Lokasi / Destination: ");
                let _ = io::stdout().flush();

                let answer = cmd_rx.recv().unwrap_or_default();
                let trimmed = answer.trim();

                if matches!(
                    trimmed.to_lowercase().as_str(),
                    "n" | "no" | "skip" | "batal" | "tidak"
                ) {
                    None
                } else if trimmed.is_empty() {
                    Some(default_path)
                } else {
                    Some(PathBuf::from(trimmed))
                }
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
        } else {
            println!("  [!] Penyimpanan dibatalkan. / Save skipped.\n");
        }
    } else {
        println!("============================================================");
        println!("  Session ended. / Sesi selesai (tidak ada rekaman suara).");
        println!("============================================================");
    }

    Ok(())
}
