use anyhow::Result;
use catetin_core::audio::{DualAudioCapture, STANDARD_SAMPLE_RATE};
use hound::{WavSpec, WavWriter};
use std::time::{Duration, Instant};

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    println!("============================================================");
    println!(" Catetin - Phase 1 Verification: Dual Audio Capture Test");
    println!(" Capturing Mic (Near-End) & Speaker Loopback (Far-End)");
    println!(" Target Sample Rate: {} Hz (Mono)", STANDARD_SAMPLE_RATE);
    println!("============================================================");

    println!("Initializing WASAPI audio streams...");
    let capture = DualAudioCapture::new()?;
    println!("Streams active! Recording 5 seconds of audio...");
    println!("Play some audio or speak to test both channels.\n");

    let wav_spec = WavSpec {
        channels: 1,
        sample_rate: STANDARD_SAMPLE_RATE,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };

    let mut mic_writer = WavWriter::create("mic_clean.wav", wav_spec)?;
    let mut loopback_writer = WavWriter::create("loopback.wav", wav_spec)?;

    let mut total_mic_samples = 0;
    let mut total_loopback_samples = 0;

    let start = Instant::now();
    let duration = Duration::from_secs(5);

    while start.elapsed() < duration {
        // Collect mic packets
        while let Ok(packet) = capture.mic_receiver.try_recv() {
            total_mic_samples += packet.samples.len();
            for sample in packet.samples {
                let sample_i16 = (sample.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
                mic_writer.write_sample(sample_i16)?;
            }
        }

        // Collect loopback packets
        while let Ok(packet) = capture.loopback_receiver.try_recv() {
            total_loopback_samples += packet.samples.len();
            for sample in packet.samples {
                let sample_i16 = (sample.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
                loopback_writer.write_sample(sample_i16)?;
            }
        }

        std::thread::sleep(Duration::from_millis(50));
        let remaining = duration.saturating_sub(start.elapsed()).as_secs_f32();
        print!("\r[Recording] Time remaining: {:.1}s | Mic samples: {} | Loopback samples: {}    ", 
            remaining, total_mic_samples, total_loopback_samples);
    }

    println!("\n\nFlushing WAV files...");
    mic_writer.finalize()?;
    loopback_writer.finalize()?;

    println!("Success! Saved files:");
    println!(" - mic_clean.wav ({} samples)", total_mic_samples);
    println!(" - loopback.wav ({} samples)", total_loopback_samples);
    println!("Phase 1 Audio Capture successfully verified.");

    Ok(())
}
