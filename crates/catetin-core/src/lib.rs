pub mod events {
    use serde::{Deserialize, Serialize};

    #[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
    pub enum SpeakerId {
        LocalUser,
        Remote(u32),
        Unknown,
    }

    impl std::fmt::Display for SpeakerId {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            match self {
                SpeakerId::LocalUser => write!(f, "You"),
                SpeakerId::Remote(id) => write!(f, "Speaker {}", id),
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
pub mod diarization;
pub mod model;
pub mod pipeline;
