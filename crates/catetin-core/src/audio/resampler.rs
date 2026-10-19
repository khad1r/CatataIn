/// High-quality linear & band-limited resampler for streaming audio buffers.
/// Converts arbitrary input sample rates (e.g., 44100Hz, 48000Hz) and channel counts
/// into standard 16,000Hz mono f32 samples for VAD, Diarization, and Whisper ASR.
#[derive(Debug, Clone)]
pub struct AudioResampler {
    source_sample_rate: u32,
    target_sample_rate: u32,
    source_channels: u16,
    resample_ratio: f64,
    fractional_position: f64,
    last_sample: f32,
}

impl AudioResampler {
    pub fn new(source_sample_rate: u32, source_channels: u16, target_sample_rate: u32) -> Self {
        let resample_ratio = target_sample_rate as f64 / source_sample_rate as f64;
        Self {
            source_sample_rate,
            target_sample_rate,
            source_channels,
            resample_ratio,
            fractional_position: 0.0,
            last_sample: 0.0,
        }
    }

    pub fn source_sample_rate(&self) -> u32 {
        self.source_sample_rate
    }

    pub fn target_sample_rate(&self) -> u32 {
        self.target_sample_rate
    }

    pub fn source_channels(&self) -> u16 {
        self.source_channels
    }

    /// Convert interleaved multi-channel samples into mono f32 samples.
    pub fn downmix_to_mono(&self, interleaved: &[f32]) -> Vec<f32> {
        let channels = self.source_channels as usize;
        if channels <= 1 {
            return interleaved.to_vec();
        }

        let num_frames = interleaved.len() / channels;
        let mut mono = Vec::with_capacity(num_frames);

        // On laptops with 4-channel microphone arrays (e.g. Intel Smart Sound Technology / Realtek Quad),
        // channels 0 and 1 are the primary stereo front microphones.
        // Channels 2 and 3 are inverted phase / acoustic echo cancellation reference channels.
        // Summing all 4 channels causes destructive phase cancellation that destroys voice frequencies!
        // We therefore downmix ONLY the primary front channels (0 and 1) for all multi-channel devices.
        if channels >= 2 {
            // First check if one channel is dead (energy ~ 0)
            if num_frames >= 32 {
                let mut energy_l = 0.0f32;
                let mut energy_r = 0.0f32;
                for frame_idx in 0..num_frames {
                    let l = interleaved[frame_idx * channels];
                    let r = interleaved[frame_idx * channels + 1];
                    energy_l += l * l;
                    energy_r += r * r;
                }
                if energy_r < 1e-12 && energy_l > 1e-8 {
                    for frame_idx in 0..num_frames {
                        mono.push(interleaved[frame_idx * channels]);
                    }
                    return mono;
                } else if energy_l < 1e-12 && energy_r > 1e-8 {
                    for frame_idx in 0..num_frames {
                        mono.push(interleaved[frame_idx * channels + 1]);
                    }
                    return mono;
                }
            }

            for frame_idx in 0..num_frames {
                let offset = frame_idx * channels;
                let l = interleaved[offset];
                let r = interleaved[offset + 1];
                mono.push((l + r) * 0.5);
            }
            return mono;
        }

        interleaved.to_vec()
    }

    /// Resample mono input samples from source_sample_rate to target_sample_rate.
    /// Applies a 5-tap Hann-windowed low-pass anti-aliasing filter when downsampling (e.g. 48kHz -> 16kHz)
    /// so high-frequency consonants ('s', 'c', 't', 'k') are preserved cleanly without aliasing distortion.
    pub fn process_mono(&mut self, mono_input: &[f32]) -> Vec<f32> {
        if mono_input.is_empty() {
            return Vec::new();
        }

        // If sample rates are already equal, return clone directly
        if self.source_sample_rate == self.target_sample_rate {
            return mono_input.to_vec();
        }

        let input_len = mono_input.len();
        let step = 1.0 / self.resample_ratio;
        let mut output = Vec::with_capacity((input_len as f64 * self.resample_ratio) as usize + 2);

        // Pre-filter with 5-tap anti-aliasing low-pass kernel [0.09, 0.24, 0.34, 0.24, 0.09]
        // when downsampling (step >= 1.5, such as 44.1kHz/48kHz -> 16kHz)
        let filtered: Vec<f32> = if step >= 1.5 && input_len >= 5 {
            let mut buf = Vec::with_capacity(input_len);
            for i in 0..input_len {
                let s_m2 = if i >= 2 { mono_input[i - 2] } else { self.last_sample };
                let s_m1 = if i >= 1 { mono_input[i - 1] } else { self.last_sample };
                let s_0 = mono_input[i];
                let s_p1 = if i + 1 < input_len { mono_input[i + 1] } else { s_0 };
                let s_p2 = if i + 2 < input_len { mono_input[i + 2] } else { s_p1 };
                buf.push(0.09 * s_m2 + 0.24 * s_m1 + 0.34 * s_0 + 0.24 * s_p1 + 0.09 * s_p2);
            }
            buf
        } else {
            mono_input.to_vec()
        };

        let mut pos = self.fractional_position;
        while pos < input_len as f64 {
            let idx = pos.floor() as usize;
            let frac = (pos - idx as f64) as f32;

            let s0 = if idx == 0 {
                self.last_sample
            } else {
                filtered[idx - 1]
            };
            let s1 = filtered[idx];

            // Linear interpolation on band-limited signal
            let sample = s0 + frac * (s1 - s0);
            output.push(sample);

            pos += step;
        }

        self.fractional_position = pos - input_len as f64;
        self.last_sample = *filtered.last().unwrap_or(&0.0);

        output
    }

    /// Downmix multi-channel interleaved audio and resample to target rate (16kHz mono).
    pub fn process_interleaved(&mut self, interleaved: &[f32]) -> Vec<f32> {
        let mono = self.downmix_to_mono(interleaved);
        self.process_mono(&mono)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_downmix_stereo() {
        let resampler = AudioResampler::new(48000, 2, 16000);
        let stereo = vec![1.0, 0.0, 0.5, 0.5, -1.0, 1.0];
        let mono = resampler.downmix_to_mono(&stereo);
        assert_eq!(mono.len(), 3);
        assert!((mono[0] - 0.5).abs() < 1e-5);
        assert!((mono[1] - 0.5).abs() < 1e-5);
        assert!((mono[2] - 0.0).abs() < 1e-5);
    }

    #[test]
    fn test_resampling_ratio() {
        let mut resampler = AudioResampler::new(48000, 1, 16000);
        // 48000 samples at 48kHz is 1.0 second. At 16kHz, expect ~16000 samples.
        let input = vec![0.5f32; 48000];
        let output = resampler.process_mono(&input);
        assert!((output.len() as i32 - 16000).abs() <= 2);
    }
}
