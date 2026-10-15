/// Normalized Least Mean Squares (NLMS) Acoustic Echo Canceller (AEC).
/// Dynamically estimates the room/speaker-to-mic acoustic impulse response
/// and subtracts meeting speaker audio (loopback) from the microphone stream.
pub struct EchoCanceller {
    filter_length: usize,
    step_size: f32,      // Learning rate mu
    regularization: f32, // Small epsilon to avoid division by zero
    weights: Vec<f32>,   // Adaptive filter tap weights
    history: Vec<f32>,   // Loopback reference signal ring buffer
    history_idx: usize,
}

impl EchoCanceller {
    /// Create a new NLMS echo canceller.
    /// `filter_length`: number of filter taps. At 16kHz, 512 taps = 32ms impulse response,
    /// 1024 taps = 64ms impulse response.
    pub fn new(filter_length: usize, step_size: f32) -> Self {
        Self {
            filter_length,
            step_size,
            regularization: 1e-4,
            weights: vec![0.0; filter_length],
            history: vec![0.0; filter_length],
            history_idx: 0,
        }
    }

    /// Process a single pair of synchronous 16kHz mono samples:
    /// `mic_sample`: raw sample from microphone (may contain echo + user speech)
    /// `ref_sample`: reference sample from system speaker loopback (meeting participants)
    /// Returns the cleaned microphone sample (echo removed).
    #[inline]
    pub fn process_sample(&mut self, mic_sample: f32, ref_sample: f32) -> f32 {
        // Update circular history buffer with new reference sample
        self.history[self.history_idx] = ref_sample;
        self.history_idx = (self.history_idx + 1) % self.filter_length;

        // Predict echo: y_hat = sum(w[k] * x[n - k])
        let mut estimated_echo = 0.0f32;
        let mut ref_energy = self.regularization;

        for k in 0..self.filter_length {
            let hist_idx = (self.history_idx + self.filter_length - 1 - k) % self.filter_length;
            let x_k = self.history[hist_idx];
            estimated_echo += self.weights[k] * x_k;
            ref_energy += x_k * x_k;
        }

        // Error signal (cleaned microphone output)
        let error = mic_sample - estimated_echo;

        // NLMS weight update: w[k] = w[k] + (step_size / (epsilon + ||x||^2)) * error * x[n - k]
        // Only adapt weights if there is active reference audio to avoid noise divergence
        if ref_energy > 1e-3 {
            let normalized_step = (self.step_size / ref_energy) * error;
            for k in 0..self.filter_length {
                let hist_idx = (self.history_idx + self.filter_length - 1 - k) % self.filter_length;
                let x_k = self.history[hist_idx];
                self.weights[k] += normalized_step * x_k;
            }
        }

        error
    }

    /// Process synchronized slices of mic audio and loopback audio at 16kHz mono.
    /// In cases where slice lengths differ slightly due to buffer timing,
    /// processes up to min(mic.len(), loopback.len()) and passes remaining mic samples through.
    pub fn process_buffer(&mut self, mic_samples: &[f32], ref_samples: &[f32]) -> Vec<f32> {
        let process_len = mic_samples.len().min(ref_samples.len());
        let mut output = Vec::with_capacity(mic_samples.len());

        for i in 0..process_len {
            let cleaned = self.process_sample(mic_samples[i], ref_samples[i]);
            output.push(cleaned);
        }

        // If mic has extra samples, pass them through
        if mic_samples.len() > process_len {
            output.extend_from_slice(&mic_samples[process_len..]);
        }

        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_echo_cancellation_attenuation() {
        let mut aec = EchoCanceller::new(256, 0.2);

        // Simulate a speaker sound: 440 Hz sine wave
        let sample_rate = 16000.0f32;
        let total_samples = 3200; // 0.2 seconds

        let mut ref_signal = Vec::new();
        let mut mic_signal = Vec::new();

        for i in 0..total_samples {
            let t = i as f32 / sample_rate;
            let ref_val = (2.0 * std::f32::consts::PI * 440.0 * t).sin() * 0.5;
            ref_signal.push(ref_val);

            // Echo reaches mic with 0.6 attenuation and small delay
            let echo = ref_val * 0.6;
            mic_signal.push(echo);
        }

        let cleaned = aec.process_buffer(&mic_signal, &ref_signal);

        // After filter converges (last 500 samples), echo energy should be significantly reduced
        let initial_echo_energy: f32 = mic_signal[2500..].iter().map(|&x| x * x).sum();
        let cleaned_echo_energy: f32 = cleaned[2500..].iter().map(|&x| x * x).sum();

        assert!(
            cleaned_echo_energy < initial_echo_energy * 0.1,
            "Echo should be attenuated by at least 10x after convergence (initial: {}, cleaned: {})",
            initial_echo_energy,
            cleaned_echo_energy
        );
    }
}
