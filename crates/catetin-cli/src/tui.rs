use std::io::{self, Write};
use catetin_core::config::AppConfig;
use catetin_core::events::SpeakerId;
use catetin_core::i18n::UiLanguage;
use catetin_core::pipeline::DiarizationMode;

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

pub fn format_live_caption(
    paused: bool,
    is_speaking: bool,
    live_raw_text: &str,
    target_w: usize,
) -> Vec<String> {
    if paused {
        vec![" ⚡ LIVE: \x1b[1;33m[PAUSED / Perekaman Dijeda]\x1b[0m".to_string()]
    } else if live_raw_text.trim().is_empty() {
        if is_speaking {
            vec![" ⚡ LIVE: \x1b[36m(mendengarkan suara / listening...)\x1b[0m".to_string()]
        } else {
            vec![" ⚡ LIVE: \x1b[2;37m(siap mendengarkan / ready)\x1b[0m".to_string()]
        }
    } else {
        let avail_w = target_w.saturating_sub(12).max(10);
        let trimmed = live_raw_text.trim();
        let chunks = textwrap_simple(trimmed, avail_w);
        let (visible_chunks, is_truncated) = if chunks.len() > 4 {
            (&chunks[chunks.len() - 4..], true)
        } else {
            (&chunks[..], false)
        };

        let mut lines = Vec::new();
        let n = visible_chunks.len();
        for (idx, chunk) in visible_chunks.iter().enumerate() {
            if n == 1 {
                lines.push(format!(" ⚡ LIVE: \x1b[1;36m\"{}\"\x1b[0m", chunk));
            } else if idx == 0 {
                let prefix = if is_truncated {
                    " ⚡ LIVE: \x1b[1;36m…\""
                } else {
                    " ⚡ LIVE: \x1b[1;36m\""
                };
                lines.push(format!("{}{}\x1b[0m", prefix, chunk));
            } else if idx == n - 1 {
                lines.push(format!("          \x1b[1;36m{}\"\x1b[0m", chunk));
            } else {
                lines.push(format!("          \x1b[1;36m{}\x1b[0m", chunk));
            }
        }
        if lines.is_empty() {
            vec![" ⚡ LIVE: \x1b[2;37m(siap mendengarkan / ready)\x1b[0m".to_string()]
        } else {
            lines
        }
    }
}

pub fn render_copilot_tui(
    config: &AppConfig,
    lang: &UiLanguage,
    mode: DiarizationMode,
    mic_name: &str,
    loopback_name: &str,
    paused: bool,
    left_lines: &[String],
    _live_raw_speaker: &SpeakerId,
    live_raw_text: &str,
    scroll_offset: usize,
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

    let target_live_w = if show_right_pane { left_w } else { inner_w };
    let caption_lines = format_live_caption(paused, is_speaking, live_raw_text, target_live_w);
    let live_line_count = caption_lines.len().clamp(1, 4);

    let content_rows = term_h.saturating_sub(12 + live_line_count).max(4);
    let upper_rows = (content_rows * 55) / 100;

    let spinner_frames = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'];
    let spin_char = spinner_frames[spinner_idx % spinner_frames.len()];

    let mut out = String::with_capacity(8192);
    out.push_str("\x1b[H");

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
        "\x1b[1;32m[●]\x1b[0m"
    } else {
        "\x1b[36m[●]\x1b[0m"
    };
    let loopback_dot = if paused {
        "\x1b[1;33m[⏸]\x1b[0m"
    } else if loopback_level > 0.0015 {
        "\x1b[1;32m[●]\x1b[0m"
    } else {
        "\x1b[36m[●]\x1b[0m"
    };
    let asr_lang_tag = if config.asr_language.is_empty() {
        "auto"
    } else {
        &config.asr_language
    };
    let status_line = format!(
        " Audio: {} {} | {} {} │ Lang: [{}] │ ASR: {} │ Mode: [{}]",
        mic_dot, mic_name, loopback_dot, loopback_name, asr_lang_tag, asr_display, mode_name
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
    let asr_badge = if transcribing_count > 0 {
        format!(
            " \x1b[1;33m[{} Memproses suara... ({})]\x1b[0m",
            spin_char, transcribing_count
        )
    } else if !paused && is_speaking {
        " \x1b[1;32m[🎙 Mendengarkan ucapan...]\x1b[0m".to_string()
    } else {
        String::new()
    };
    let scroll_badge = if scroll_offset > 0 {
        format!(" \x1b[1;33m[↑ Gulir: +{} | 'G' ke akhir]\x1b[0m", scroll_offset)
    } else {
        String::new()
    };
    let left_hdr = format!(" LIVE DIARIZED TRANSCRIPT  {}{}{}", wave_badge, asr_badge, scroll_badge);
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

    // 4b. Live Auto Caption Bar on top of the transcript (max 4 lines)
    if show_right_pane {
        out.push_str(&format!("├{}┼{}┤\x1b[K\n", "─".repeat(left_w), "─".repeat(right_w)));
        for (i, line) in caption_lines.iter().enumerate() {
            let right_cell = if i == 0 {
                " • Topik & Tindak Lanjut Otomatis"
            } else {
                ""
            };
            out.push_str(&format!(
                "│{}│{}│\x1b[K\n",
                fit_cell(line, left_w),
                fit_cell(right_cell, right_w)
            ));
        }
        out.push_str(&format!("├{}┼{}┤\x1b[K\n", "─".repeat(left_w), "─".repeat(right_w)));
    } else {
        out.push_str(&format!("├{}┤\x1b[K\n", "─".repeat(inner_w)));
        for line in &caption_lines {
            out.push_str(&format!("│{}│\x1b[K\n", fit_cell(line, inner_w)));
        }
        out.push_str(&format!("├{}┤\x1b[K\n", "─".repeat(inner_w)));
    }

    // 5. Left pane content
    let (meter_bars, meter_active) = build_waveform(wave_history, 16);
    let meter_line = if paused {
        "  \x1b[1;33m● Audio Dijeda / Capture Paused (Tekan 'p' untuk lanjut)\x1b[0m".to_string()
    } else if meter_active || is_speaking {
        format!("  \x1b[1;32m● Audio Terdeteksi / Speaking : [{}]\x1b[0m", meter_bars)
    } else {
        format!("  \x1b[36m● Mikrofon Aktif / Listening  : [{}]\x1b[0m", meter_bars)
    };

    let live_activity_line = if paused && transcribing_count > 0 {
        format!(
            "  \x1b[1;33m⏸ Dijeda — {} Menyelesaikan antrean transkripsi ({} antrean tersisa)...\x1b[0m",
            spin_char, transcribing_count
        )
    } else if paused {
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

    let total_lines = composed_left.len();
    let max_scroll = total_lines.saturating_sub(content_rows);
    let active_scroll = scroll_offset.min(max_scroll);
    let left_start = total_lines.saturating_sub(content_rows + active_scroll);
    let left_end = (left_start + content_rows).min(total_lines);
    let visible_left = &composed_left[left_start..left_end];

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
        " [↑/↓] Gulir │ [p] Pause │ [s] Summary AI │ [?] Tanya │ [m] Mode │ [l] Lang │ [c] Simpan │ [q] Keluar"
    } else {
        " [↑/↓] Gulir │ [p] Pause │ [m] Mode │ [l] Lang │ [c] Simpan │ [q] Keluar"
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

    // 12. Bottom border
    out.push_str(&format!("└{}┘\x1b[K", "─".repeat(inner_w)));

    let mut stdout = io::stdout();
    let _ = stdout.write_all(out.as_bytes());
    let _ = stdout.flush();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_live_caption_paused() {
        let lines = format_live_caption(true, false, "some text", 80);
        assert_eq!(lines.len(), 1);
        assert!(lines[0].contains("PAUSED"));
    }

    #[test]
    fn test_format_live_caption_empty_ready() {
        let lines = format_live_caption(false, false, "", 80);
        assert_eq!(lines.len(), 1);
        assert!(lines[0].contains("ready"));
    }

    #[test]
    fn test_format_live_caption_empty_listening() {
        let lines = format_live_caption(false, true, "   ", 80);
        assert_eq!(lines.len(), 1);
        assert!(lines[0].contains("listening"));
    }

    #[test]
    fn test_format_live_caption_single_line() {
        let lines = format_live_caption(false, true, "KUIN YANG NAMANYA BONTI HUNTING BUAT", 80);
        assert_eq!(lines.len(), 1);
        assert!(lines[0].contains("KUIN YANG NAMANYA BONTI HUNTING BUAT"));
        assert!(lines[0].starts_with(" ⚡ LIVE:"));
    }

    #[test]
    fn test_format_live_caption_max_4_lines() {
        let text = "satu dua tiga empat lima enam tujuh delapan sembilan sepuluh sebelas dua belas tiga belas empat belas lima belas enam belas tujuh belas delapan belas sembilan belas dua puluh";
        let lines = format_live_caption(false, true, text, 30);
        assert!(lines.len() <= 4, "Must not exceed 4 lines, got {}", lines.len());
        assert_eq!(lines.len(), 4, "Should fill up to 4 lines when text is long");
        assert!(lines[0].contains("…\""));
        assert!(lines[3].ends_with("\"\x1b[0m"));
    }
}

