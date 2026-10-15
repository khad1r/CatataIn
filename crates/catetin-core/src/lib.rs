pub mod events {
    use serde::{Deserialize, Serialize};
    use crate::i18n::UiLanguage;

    #[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
    pub enum SpeakerId {
        LocalUser,
        Remote(u32),
        InPerson(u32),
        Unknown,
    }

    impl SpeakerId {
        /// Format speaker identifier according to the selected UI language.
        /// Dynamically localized into Indonesian, English, or any contributor-added language.
        pub fn to_localized_string(&self, lang: &UiLanguage) -> String {
            match self {
                SpeakerId::LocalUser => lang.text("speaker.you").to_string(),
                SpeakerId::Remote(id) => crate::i18n::format_speaker_label(*id, lang),
                SpeakerId::InPerson(id) => crate::i18n::format_speaker_label(*id, lang),
                SpeakerId::Unknown => lang.text("speaker.unknown").to_string(),
            }
        }
    }

    impl std::fmt::Display for SpeakerId {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            match self {
                SpeakerId::LocalUser => write!(f, "You"),
                SpeakerId::Remote(id) => write!(f, "Speaker {}", id),
                SpeakerId::InPerson(id) => write!(f, "Speaker {}", id),
                SpeakerId::Unknown => write!(f, "Unknown"),
            }
        }
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct TranscriptEvent {
        pub timestamp_ms: u64,
        pub speaker: SpeakerId,
        pub text: String,
        pub is_final: bool,
        pub confidence: f32,
    }
}

pub mod asr;
pub mod audio;
pub mod config;
pub mod diarization;
pub mod i18n;
pub mod logging;
pub mod pipeline;


pub use config::AppConfig;
pub use i18n::UiLanguage;
pub use pipeline::{CatetinPipeline, DiarizationMode, PipelineHandle};
