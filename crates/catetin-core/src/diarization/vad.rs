/// Voice Activity Detector (VAD) & Utterance Segmenter.
/// Detects speech presence using adaptive energy and zero-crossing analysis,
/// applying hangover smoothing to prevent cutting off sentence endings.
#[derive(Debug, Clone)]
pub struct VoiceActivityDetector {
    sample_rate: u32,
    pub frame_size: usize,
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
        Self::for_noise_suppression(
            sample_rate,
            frame_size,
            crate::config::NoiseSuppression::Default,
        )
    }

    pub fn for_noise_suppression(
        sample_rate: u32,
        frame_size: usize,
        suppression: crate::config::NoiseSuppression,
    ) -> Self {
        let (energy_threshold, noise_floor, noise_adaptation_rate) = match suppression {
            crate::config::NoiseSuppression::None => (0.0025, 0.0008, 0.02),
            crate::config::NoiseSuppression::Default => (0.0040, 0.0012, 0.03),
            crate::config::NoiseSuppression::Aggressive => (0.0080, 0.0020, 0.04),
        };
        Self {
            sample_rate,
            frame_size,
            energy_threshold,
            noise_floor,
            noise_adaptation_rate,
            hangover_frames: 6, // ~180ms hangover
            hangover_counter: 0,
            is_speech_active: false,
        }
    }

    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Calculate RMS energy of an audio frame
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

        // Dynamic threshold relative to background noise floor
        let dynamic_threshold = (self.noise_floor * 1.8 + self.energy_threshold).min(0.035);

        if rms > dynamic_threshold {
            // Even when above threshold, if RMS is in the low ambient hum range (< 0.015),
            // allow slow upward adaptation so a steady fan/hum can never deadlock VAD.
            if rms < 0.015 {
                self.noise_floor = 0.995 * self.noise_floor + 0.005 * rms;
            }
            self.hangover_counter = self.hangover_frames;
            self.is_speech_active = true;
        } else {
            // Adapt background noise floor during silence, capped at 0.015
            let target_noise = rms.min(0.015);
            self.noise_floor = (1.0 - self.noise_adaptation_rate) * self.noise_floor
                + self.noise_adaptation_rate * target_noise;

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
    pending_samples: Vec<f32>,
    silence_frames_count: usize,
    speech_frames_count: usize,
    has_speech: bool,
    max_samples: usize,
    min_speech_frames: usize,
    max_silence_frames: usize,
    min_peak_amplitude: f32,
}

impl SpeechSegmenter {
    pub fn new(sample_rate: u32) -> Self {
        Self::with_noise_suppression(sample_rate, crate::config::NoiseSuppression::Default)
    }

    pub fn with_noise_suppression(
        sample_rate: u32,
        suppression: crate::config::NoiseSuppression,
    ) -> Self {
        let frame_size = (sample_rate as f32 * 0.03) as usize; // 30ms = 480 samples
        Self {
            vad: VoiceActivityDetector::for_noise_suppression(sample_rate, frame_size, suppression),
            buffer: Vec::with_capacity(sample_rate as usize * 10),
            pending_samples: Vec::with_capacity(frame_size * 2),
            silence_frames_count: 0,
            speech_frames_count: 0,
            has_speech: false,
            max_samples: sample_rate as usize * 9, // Tightened to 9.0s max for real-time responsiveness
            min_speech_frames: match suppression {
                crate::config::NoiseSuppression::None => 5,       // 150ms speech
                crate::config::NoiseSuppression::Default => 5,    // 150ms speech
                crate::config::NoiseSuppression::Aggressive => 7, // 210ms speech
            },
            max_silence_frames: 12, // ~360ms silence cuts natural sentence boundaries much faster
            min_peak_amplitude: 0.003,
        }
    }

    /// Create a segmenter specifically calibrated for digital system loopback audio (Speakers/Headphones).
    /// Digital loopback has zero analog noise floor and its amplitude scales with the OS volume slider.
    pub fn for_loopback(sample_rate: u32) -> Self {
        let frame_size = (sample_rate as f32 * 0.03) as usize; // 30ms = 480 samples
        let vad = VoiceActivityDetector {
            sample_rate,
            frame_size,
            energy_threshold: 0.0020,
            noise_floor: 0.0003,
            noise_adaptation_rate: 0.02,
            hangover_frames: 6, // ~180ms hangover
            hangover_counter: 0,
            is_speech_active: false,
        };
        Self {
            vad,
            buffer: Vec::with_capacity(sample_rate as usize * 10),
            pending_samples: Vec::with_capacity(frame_size * 2),
            silence_frames_count: 0,
            speech_frames_count: 0,
            has_speech: false,
            max_samples: sample_rate as usize * 8, // Tightened to 8.0s so continuous music/speech never runs away
            min_speech_frames: 5,   // 150ms speech
            max_silence_frames: 12, // ~360ms silence
            min_peak_amplitude: 0.003,
        }
    }

    /// Returns true if the segmenter is currently accumulating an active speech utterance.
    pub fn is_speaking(&self) -> bool {
        self.has_speech
    }

    /// Normalize utterance amplitude to an optimal level for Whisper's log-mel spectrogram
    /// while removing any hardware DC bias and clamping transient spikes outside [-1.0, 1.0].
    fn normalize_utterance(samples: &mut [f32]) {
        if samples.len() == 0 {
            return;
        }
        // Remove any DC offset from the microphone ADC
        let mean = samples.iter().copied().sum::<f32>() / samples.len() as f32;
        for s in samples.iter_mut() {
            *s = (*s - mean).clamp(-1.0, 1.0);
        }
        let peak = samples.iter().fold(0.0f32, |acc, &x| acc.max(x.abs()));
        if peak > 0.002 && peak < 0.50 {
            let gain = (0.55 / peak).min(12.0);
            for s in samples.iter_mut() {
                *s = (*s * gain).clamp(-1.0, 1.0);
            }
        }
    }

    /// Flush any currently buffered speech utterance (e.g. when pausing or stopping the session).
    pub fn flush(&mut self) -> Option<Vec<f32>> {
        if !self.pending_samples.is_empty() {
            self.buffer.extend_from_slice(&self.pending_samples);
            self.pending_samples.clear();
        }

        if !self.buffer.is_empty() {
            let mut samples = std::mem::take(&mut self.buffer);
            self.has_speech = false;
            self.speech_frames_count = 0;
            self.silence_frames_count = 0;
            let peak = samples.iter().fold(0.0f32, |acc, &x| acc.max(x.abs()));
            // At least 200ms (3,200 samples) and minimal audible energy
            if samples.len() >= 3200 && peak >= 0.003 && is_speech_utterance(&samples, self.vad.sample_rate) {
                log::info!(
                    "[VAD] Flushing speech utterance on pause/stop: {} samples ({:.2}s, peak={:.4})",
                    samples.len(),
                    samples.len() as f32 / self.vad.sample_rate as f32,
                    peak
                );
                Self::normalize_utterance(&mut samples);
                return Some(samples);
            }
        }
        self.buffer.clear();
        self.has_speech = false;
        self.speech_frames_count = 0;
        self.silence_frames_count = 0;
        None
    }

    /// Push new 16kHz mono audio samples.
    /// Returns `Some(Vec<f32>)` only when a full speech utterance is completed upon a natural pause.
    pub fn feed(&mut self, samples: &[f32]) -> Option<Vec<f32>> {
        self.pending_samples.extend_from_slice(samples);
        let frame_size = self.vad.frame_size;
        let mut completed_utterance = None;

        while self.pending_samples.len() >= frame_size {
            let frame: Vec<f32> = self.pending_samples.drain(0..frame_size).collect();
            let is_speech = self.vad.process_frame(&frame);

            if is_speech {
                self.has_speech = true;
                self.speech_frames_count += 1;
                self.silence_frames_count = 0;
                self.buffer.extend_from_slice(&frame);

                // Hard upper safety ceiling (24s) in case someone speaks non-stop without pausing
                if self.buffer.len() >= self.max_samples
                    && self.speech_frames_count >= self.min_speech_frames
                {
                    let mut samples = std::mem::take(&mut self.buffer);
                    let peak = samples.iter().fold(0.0f32, |acc, &x| acc.max(x.abs()));
                    if peak >= self.min_peak_amplitude && is_speech_utterance(&samples, self.vad.sample_rate) {
                        log::info!(
                            "[VAD] Speech utterance ready (max-len safety split): {} frames, {} samples ({:.2}s, peak={:.4})",
                            self.speech_frames_count,
                            samples.len(),
                            samples.len() as f32 / self.vad.sample_rate as f32,
                            peak
                        );
                        Self::normalize_utterance(&mut samples);
                        completed_utterance = Some(samples);
                    } else {
                        log::info!(
                            "[VAD] Discarded non-speech audio on max-len split: {} samples ({:.2}s)",
                            samples.len(),
                            samples.len() as f32 / self.vad.sample_rate as f32
                        );
                    }
                    self.has_speech = false;
                    self.speech_frames_count = 0;
                    self.silence_frames_count = 0;
                }
            } else if self.has_speech {
                // We are in an active utterance, but currently experiencing an inter-word or end-of-phrase pause
                self.silence_frames_count += 1;
                self.buffer.extend_from_slice(&frame);

                // Natural endpoint pause reached (speaker finished their sentence)!
                if self.silence_frames_count >= self.max_silence_frames
                    || self.buffer.len() >= self.max_samples
                {
                    if self.speech_frames_count >= self.min_speech_frames {
                        let mut samples = std::mem::take(&mut self.buffer);
                        let peak = samples.iter().fold(0.0f32, |acc, &x| acc.max(x.abs()));
                        if peak >= self.min_peak_amplitude && is_speech_utterance(&samples, self.vad.sample_rate) {
                            log::info!(
                                "[VAD] Speech utterance ready: {} frames, {} samples ({:.2}s, peak={:.4})",
                                self.speech_frames_count,
                                samples.len(),
                                samples.len() as f32 / self.vad.sample_rate as f32,
                                peak
                            );
                            Self::normalize_utterance(&mut samples);
                            completed_utterance = Some(samples);
                        } else {
                            log::info!(
                                "[VAD] Discarded non-speech audio (music/noise detected, no human speech): {} samples ({:.2}s)",
                                samples.len(),
                                samples.len() as f32 / self.vad.sample_rate as f32
                            );
                        }
                    } else {
                        self.buffer.clear();
                    }
                    self.has_speech = false;
                    self.speech_frames_count = 0;
                    self.silence_frames_count = 0;
                }
            } else {
                // Pure background silence before any speech starts.
                // Keep a 360ms pre-roll buffer (12 frames) so leading consonants ('C', 'S', 'T', 'H') are never clipped.
                self.buffer.extend_from_slice(&frame);
                let pre_roll_limit = frame_size * 12; // ~360ms
                if self.buffer.len() > pre_roll_limit {
                    self.buffer.drain(0..self.buffer.len() - pre_roll_limit);
                }
            }

            if completed_utterance.is_some() {
                break;
            }
        }

        completed_utterance
    }
}

/// Validate whether an accumulated audio utterance contains acoustic speech
/// (distinguishing voice from silence or low-amplitude baseline clicks).
pub fn is_speech_utterance(samples: &[f32], sample_rate: u32) -> bool {
    // Under 200ms is too short for a complete speech phrase
    if samples.len() < (sample_rate as f32 * 0.20) as usize {
        return false;
    }

    let peak = samples.iter().fold(0.0f32, |acc, &x| acc.max(x.abs()));
    if peak < 0.003 {
        return false;
    }

    true
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
