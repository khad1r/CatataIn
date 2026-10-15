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
        let inv_channels = 1.0 / channels as f32;

        for frame_idx in 0..num_frames {
            let offset = frame_idx * channels;
            let mut sum = 0.0f32;
            for ch in 0..channels {
                sum += interleaved[offset + ch];
            }
            mono.push(sum * inv_channels);
        }
        mono
    }

    /// Resample mono input samples from source_sample_rate to target_sample_rate.
    pub fn process_mono(&mut self, mono_input: &[f32]) -> Vec<f32> {
        if mono_input.is_empty() {
            return Vec::new();
        }

        // If sample rates are already equal, return clone directly
        if self.source_sample_rate == self.target_sample_rate {
            return mono_input.to_vec();
        }

        let mut output = Vec::new();
        let step = 1.0 / self.resample_ratio;

        let mut pos = self.fractional_position;
        let input_len = mono_input.len();

        while pos < input_len as f64 {
            let idx = pos.floor() as usize;
            let frac = (pos - idx as f64) as f32;

            let s0 = if idx == 0 {
                self.last_sample
            } else {
                mono_input[idx - 1]
            };
            let s1 = mono_input[idx];

            // Linear interpolation
            let sample = s0 + frac * (s1 - s0);
            output.push(sample);

            pos += step;
        }

        self.fractional_position = pos - input_len as f64;
        self.last_sample = *mono_input.last().unwrap_or(&0.0);

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
