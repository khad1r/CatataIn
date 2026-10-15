use serde::{Deserialize, Serialize};
use std::fmt;

// ponytail: compile-time static string mapping instead of dynamic JSON file discovery & runtime HashMap.
// Ceiling: Indonesian and English static strings. Upgrade path: reintroduce dynamic locale catalog if external plugin translation files are requested.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum UiLanguage {
    #[serde(alias = "id", alias = "indonesian", alias = "in")]
    Indonesian,
    #[serde(alias = "en", alias = "english")]
    English,
    Custom(String),
}

impl Default for UiLanguage {
    fn default() -> Self {
        UiLanguage::Indonesian
    }
}

impl UiLanguage {
    pub fn from_code(code: &str) -> Self {
        match code.trim().to_lowercase().as_str() {
            "id" | "indonesian" | "in" => UiLanguage::Indonesian,
            "en" | "english" => UiLanguage::English,
            custom => UiLanguage::Custom(custom.to_string()),
        }
    }

    pub fn code(&self) -> &str {
        match self {
            UiLanguage::Indonesian => "id",
            UiLanguage::English => "en",
            UiLanguage::Custom(ref s) => s.as_str(),
        }
    }

    pub fn display_name(&self) -> &str {
        match self {
            UiLanguage::Indonesian => "Bahasa Indonesia",
            UiLanguage::English => "English",
            UiLanguage::Custom(code) => code.as_str(),
        }
    }

    pub fn text(&self, key: &str) -> &'static str {
        match self {
            UiLanguage::Indonesian => match_id(key),
            _ => match_en(key),
        }
    }

    pub fn text_fmt(&self, key: &str, args: &[&str]) -> String {
        let mut result = self.text(key).to_string();
        for (i, arg) in args.iter().enumerate() {
            result = result.replace(&format!("{{{}}}", i), arg);
        }
        result
    }
}

impl fmt::Display for UiLanguage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.code())
    }
}

pub fn format_speaker_label(speaker_num: u32, lang: &UiLanguage) -> String {
    format!("{} {}", lang.text("speaker.prefix"), speaker_num)
}

fn match_id(key: &str) -> &'static str {
    match key {
        "app.name" => "Catetin",
        "app.tagline" => "Tolong catetin rapat ini ya!",
        "app.subtitle" => "Mesin Transkripsi & Diarisasi Rapat Offline",
        "speaker.you" => "Anda",
        "speaker.prefix" => "Pembicara",
        "speaker.unknown" => "Tidak Diketahui",
        "status.ready" => "Siap merekam rapat",
        "status.listening" => "Mendengarkan...",
        "status.processing" => "Mentranskripsi...",
        "status.paused" => "Dijeda",
        "status.stopped" => "Selesai",
        "prompt.press_quit" => "Tekan 'm' ganti mode langsung, 'l' ganti bahasa, Ctrl+C selesai",
        "prompt.downloading_model" => "Mengunduh model suara offline...",
        "prompt.model_ready" => "Model suara siap digunakan",
        "prompt.saving_transcript" => "Menyimpan catatan rapat...",
        "prompt.saved_success" => "Catatan rapat berhasil disimpan ke: ",
        "label.microphone" => "Mikrofon (Dekat)",
        "label.loopback" => "Audio Sistem (Jauh/Speaker)",
        "label.mode_online" => "Mode Rapat Online (Zoom/Google Meet/Teams)",
        "label.mode_in_person" => "Mode Tatap Muka (Satu Mikrofon Meja)",
        "label.mode_hybrid" => "Mode Hibrida (Dengan Profil Suara)",
        "label.language" => "Bahasa",
        "label.mode" => "Mode",
        "label.model_tier" => "Tingkat Model",
        "label.log_file" => "Berkas Log",
        "label.status" => "Status",
        "label.tip" => "Petunjuk",
        "cli.usage" => "Penggunaan:",
        "cli.options" => "Opsi:",
        "cli.help_config" => "Buka wizard konfigurasi interaktif atau tentukan berkas JSON",
        "cli.help_verbose" => "Tampilkan log debug ke stderr selain ke berkas log",
        "wizard.title" => "Pengaturan Konfigurasi Interaktif",
        "wizard.step_lang" => "1. Pilih Bahasa Antarmuka (UI Language):",
        "wizard.step_mode" => "2. Pilih Mode Diarisasi (Diarization Mode):",
        "wizard.step_model" => "3. Pilih Tingkat Model Suara (Whisper Model Tier):",
        "wizard.step_verbose" => "4. Tampilkan log diagnostik ke terminal saat merekam? (y/N):",
        "wizard.prompt_choice" => "Masukkan pilihan",
        "wizard.saved_success" => "Konfigurasi berhasil disimpan ke:",
        _ => match_en(key),
    }
}

fn match_en(key: &str) -> &'static str {
    match key {
        "app.name" => "Catetin",
        "app.tagline" => "Please take notes for this meeting!",
        "app.subtitle" => "Offline Live Meeting Transcriber & Diarizer",
        "speaker.you" => "You",
        "speaker.prefix" => "Speaker",
        "speaker.unknown" => "Unknown",
        "status.ready" => "Ready to record meeting",
        "status.listening" => "Listening...",
        "status.processing" => "Transcribing...",
        "status.paused" => "Paused",
        "status.stopped" => "Stopped",
        "prompt.press_quit" => "Press 'm' to toggle mode on the fly, 'l' for lang, Ctrl+C to stop",
        "prompt.downloading_model" => "Downloading offline voice model...",
        "prompt.model_ready" => "Voice model ready",
        "prompt.saving_transcript" => "Saving meeting transcript...",
        "prompt.saved_success" => "Meeting transcript saved to: ",
        "label.microphone" => "Microphone (Near-end)",
        "label.loopback" => "System Audio (Far-end/Loopback)",
        "label.mode_online" => "Online Meeting Mode (Zoom/Google Meet/Teams)",
        "label.mode_in_person" => "In-Person Mode (Round-Table Single Mic)",
        "label.mode_hybrid" => "Hybrid Mode (With Enrolled Voiceprint)",
        "label.language" => "Language",
        "label.mode" => "Mode",
        "label.model_tier" => "Model Tier",
        "label.log_file" => "Log File",
        "label.status" => "Status",
        "label.tip" => "Tip",
        "cli.usage" => "Usage:",
        "cli.options" => "Options:",
        "cli.help_config" => "Launch interactive configuration wizard or specify JSON file",
        "cli.help_verbose" => "Print debug logs to stderr alongside log file",
        "wizard.title" => "Interactive Configuration Setup Wizard",
        "wizard.step_lang" => "1. Select UI Language:",
        "wizard.step_mode" => "2. Select Diarization Mode:",
        "wizard.step_model" => "3. Select Whisper Voice Model Tier:",
        "wizard.step_verbose" => "4. Print diagnostic logs to terminal during recording? (y/N):",
        "wizard.prompt_choice" => "Enter choice",
        "wizard.saved_success" => "Configuration successfully saved to:",
        _ => "Unknown",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_builtin_localization() {
        let id = UiLanguage::Indonesian;
        let en = UiLanguage::English;

        assert_eq!(id.text("app.tagline"), "Tolong catetin rapat ini ya!");
        assert_eq!(en.text("app.tagline"), "Please take notes for this meeting!");

        assert_eq!(id.text("speaker.you"), "Anda");
        assert_eq!(en.text("speaker.you"), "You");

        assert_eq!(format_speaker_label(1, &id), "Pembicara 1");
        assert_eq!(format_speaker_label(2, &en), "Speaker 2");
    }
}
