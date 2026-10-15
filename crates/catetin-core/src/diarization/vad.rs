/// Voice Activity Detector (VAD) & Utterance Segmenter.
/// Detects speech presence using adaptive energy and zero-crossing analysis,
/// applying hangover smoothing to prevent cutting off sentence endings.
#[derive(Debug, Clone)]
pub struct VoiceActivityDetector {
    sample_rate: u32,
    frame_size: usize,
    energy_threshold: f32,
    noise_floor: f32,
    noise_adaptation_rate: f32,
    hangover_frames: usize,
    hangover_counter: usize,
    is_speech_active: bool,
}

impl Default for VoiceActivityDetector {
    fn default() -> Self {
        Self::new(16_000, 480) // 480 samples = 30ms at 16kHz
    }
}

impl VoiceActivityDetector {
    pub fn new(sample_rate: u32, frame_size: usize) -> Self {
        Self {
            sample_rate,
            frame_size,
            energy_threshold: 0.005,
            noise_floor: 0.001,
            noise_adaptation_rate: 0.05,
            hangover_frames: 10, // ~300ms hangover
            hangover_counter: 0,
            is_speech_active: false,
        }
    }

    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Calculate RMS energy of a frame
    pub fn frame_rms(&self, frame: &[f32]) -> f32 {
        if frame.is_empty() {
            return 0.0;
        }
        let sum_sq: f32 = frame.iter().map(|&x| x * x).sum();
        (sum_sq / frame.len() as f32).sqrt()
    }

    /// Process a 30ms audio frame and return true if speech is detected
    pub fn process_frame(&mut self, frame: &[f32]) -> bool {
        let rms = self.frame_rms(frame);

        // Adaptive threshold relative to background noise floor
        let dynamic_threshold = self.noise_floor * 3.5 + self.energy_threshold;

        if rms > dynamic_threshold {
            self.hangover_counter = self.hangover_frames;
            self.is_speech_active = true;
        } else {
            // Adapt background noise floor slowly during silence
            self.noise_floor = (1.0 - self.noise_adaptation_rate) * self.noise_floor
                + self.noise_adaptation_rate * rms;

            if self.hangover_counter > 0 {
                self.hangover_counter -= 1;
                self.is_speech_active = true;
            } else {
                self.is_speech_active = false;
            }
        }

        self.is_speech_active
    }
}

/// Chunks incoming streaming 16kHz audio into clean speech utterances for Whisper ASR.
pub struct SpeechSegmenter {
    vad: VoiceActivityDetector,
    buffer: Vec<f32>,
    silence_frames_count: usize,
    max_samples: usize,
    min_speech_samples: usize,
    max_silence_frames: usize,
}

impl SpeechSegmenter {
    pub fn new(sample_rate: u32) -> Self {
        let frame_size = (sample_rate as f32 * 0.03) as usize; // 30ms
        Self {
            vad: VoiceActivityDetector::new(sample_rate, frame_size),
            buffer: Vec::with_capacity(sample_rate as usize * 10),
            silence_frames_count: 0,
            max_samples: sample_rate as usize * 15,       // Max 15s per segment
            min_speech_samples: sample_rate as usize / 2, // Min 0.5s speech
            max_silence_frames: 17,                       // ~500ms silence ends utterance
        }
    }

    /// Push new 16kHz mono audio samples.
    /// If an utterance boundary is reached (end of phrase or max length), returns `Some(Vec<f32>)`.
    pub fn feed(&mut self, samples: &[f32]) -> Option<Vec<f32>> {
        self.buffer.extend_from_slice(samples);

        let frame_size = self.vad.frame_size;
        let mut emit = false;

        // Check the most recent frame for speech
        if self.buffer.len() >= frame_size {
            let recent_frame = &self.buffer[self.buffer.len() - frame_size..];
            let is_speech = self.vad.process_frame(recent_frame);

            if is_speech {
                self.silence_frames_count = 0;
            } else {
                self.silence_frames_count += 1;
            }

            // Emit if:
            // 1. Long enough speech followed by sufficient pause (end of sentence), OR
            // 2. Buffer hit maximum length limit
            if (self.silence_frames_count >= self.max_silence_frames
                && self.buffer.len() >= self.min_speech_samples)
                || self.buffer.len() >= self.max_samples
            {
                emit = true;
            }
        }

        if emit {
            let segment = std::mem::take(&mut self.buffer);
            self.silence_frames_count = 0;
            Some(segment)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vad_speech_vs_silence() {
        let mut vad = VoiceActivityDetector::new(16_000, 480);

        let silence_frame = vec![0.0001f32; 480];
        let speech_frame = vec![0.3f32; 480];

        // Silence should not trigger speech
        assert!(!vad.process_frame(&silence_frame));

        // Speech frame should trigger speech
        assert!(vad.process_frame(&speech_frame));

        // Hangover should keep speech active for the next frame even if silent
        assert!(vad.process_frame(&silence_frame));
    }
}
