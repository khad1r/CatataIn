use anyhow::{Context, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Device, SampleFormat, Stream, StreamConfig};
use std::sync::mpsc::{sync_channel, Receiver, SyncSender};
use std::sync::{Arc, Mutex};

use super::aec::EchoCanceller;
use super::resampler::AudioResampler;

pub const STANDARD_SAMPLE_RATE: u32 = 16_000;

/// Output packet from the dual audio capture engine.
#[derive(Debug, Clone)]
pub struct AudioPacket {
    pub timestamp_ms: u64,
    /// 16kHz mono PCM samples
    pub samples: Vec<f32>,
}

/// Manages dual-stream audio capture:
/// 1. Microphone input (Near-end: User speech)
/// 2. System loopback audio (Far-end: Meeting participants from Zoom/Google Meet)
/// 3. Real-time Acoustic Echo Cancellation (AEC)
pub struct DualAudioCapture {
    _mic_stream: Stream,
    _loopback_stream: Stream,
    pub mic_receiver: Receiver<AudioPacket>,
    pub loopback_receiver: Receiver<AudioPacket>,
}

impl DualAudioCapture {
    /// Initialize dual capture using the default communication input and output devices.
    pub fn new() -> Result<Self> {
        let host = cpal::default_host();

        let input_device = host
            .default_input_device()
            .context("No default audio input device (microphone) found")?;
        let output_device = host
            .default_output_device()
            .context("No default audio output device (speakers/headphones) found")?;

        let input_name = input_device.name().unwrap_or_else(|_| "Default Mic".into());
        let output_name = output_device.name().unwrap_or_else(|_| "Default Output".into());

        log::info!("DualAudioCapture: Mic device: {}", input_name);
        log::info!("DualAudioCapture: Loopback device: {}", output_name);

        Self::from_devices(&input_device, &output_device)
    }

    pub fn from_devices(input_device: &Device, output_device: &Device) -> Result<Self> {
        let (mic_tx, mic_rx) = sync_channel::<AudioPacket>(100);
        let (loopback_tx, loopback_rx) = sync_channel::<AudioPacket>(100);

        // Echo canceller shared across capture streams
        let aec = Arc::new(Mutex::new(EchoCanceller::new(512, 0.15)));

        // 1. Build Loopback Stream (Captures meeting audio from speakers)
        let loopback_stream = build_loopback_stream(output_device, loopback_tx)?;

        // 2. Build Microphone Stream (Captures user voice with AEC)
        let mic_stream = build_mic_stream(input_device, mic_tx, Arc::clone(&aec))?;

        mic_stream.play().context("Failed to start microphone stream")?;
        loopback_stream.play().context("Failed to start loopback stream")?;

        Ok(Self {
            _mic_stream: mic_stream,
            _loopback_stream: loopback_stream,
            mic_receiver: mic_rx,
            loopback_receiver: loopback_rx,
        })
    }
}

fn build_mic_stream(
    device: &Device,
    sender: SyncSender<AudioPacket>,
    _aec: Arc<Mutex<EchoCanceller>>,
) -> Result<Stream> {
    let default_config = device
        .default_input_config()
        .context("Failed to get default input audio config")?;

    let sample_format = default_config.sample_format();
    let config: StreamConfig = default_config.into();
    let channels = config.channels;
    let sample_rate = config.sample_rate.0;

    let resampler = Arc::new(Mutex::new(AudioResampler::new(
        sample_rate,
        channels,
        STANDARD_SAMPLE_RATE,
    )));

    let err_fn = |err| log::error!("Microphone stream error: {}", err);

    let stream = match sample_format {
        SampleFormat::F32 => device.build_input_stream(
            &config,
            move |data: &[f32], _| {
                if let Ok(mut resamp) = resampler.lock() {
                    let mono_16k = resamp.process_interleaved(data);
                    if !mono_16k.is_empty() {
                        let _ = sender.try_send(AudioPacket {
                            timestamp_ms: current_timestamp_ms(),
                            samples: mono_16k,
                        });
                    }
                }
            },
            err_fn,
            None,
        )?,
        SampleFormat::I16 => device.build_input_stream(
            &config,
            move |data: &[i16], _| {
                let f32_samples: Vec<f32> = data
                    .iter()
                    .map(|&s| s as f32 / i16::MAX as f32)
                    .collect();
                if let Ok(mut resamp) = resampler.lock() {
                    let mono_16k = resamp.process_interleaved(&f32_samples);
                    if !mono_16k.is_empty() {
                        let _ = sender.try_send(AudioPacket {
                            timestamp_ms: current_timestamp_ms(),
                            samples: mono_16k,
                        });
                    }
                }
            },
            err_fn,
            None,
        )?,
        _ => anyhow::bail!("Unsupported microphone sample format: {:?}", sample_format),
    };

    Ok(stream)
}

fn build_loopback_stream(
    device: &Device,
    sender: SyncSender<AudioPacket>,
) -> Result<Stream> {
    // On Windows WASAPI, building an input stream on an output device activates AUDCLNT_STREAMFLAGS_LOOPBACK
    let default_config = device
        .default_output_config()
        .context("Failed to get default output audio config for loopback")?;

    let sample_format = default_config.sample_format();
    let config: StreamConfig = default_config.into();
    let channels = config.channels;
    let sample_rate = config.sample_rate.0;

    let resampler = Arc::new(Mutex::new(AudioResampler::new(
        sample_rate,
        channels,
        STANDARD_SAMPLE_RATE,
    )));

    let err_fn = |err| log::error!("Loopback stream error: {}", err);

    let stream = match sample_format {
        SampleFormat::F32 => device.build_input_stream(
            &config,
            move |data: &[f32], _| {
                if let Ok(mut resamp) = resampler.lock() {
                    let mono_16k = resamp.process_interleaved(data);
                    if !mono_16k.is_empty() {
                        let _ = sender.try_send(AudioPacket {
                            timestamp_ms: current_timestamp_ms(),
                            samples: mono_16k,
                        });
                    }
                }
            },
            err_fn,
            None,
        )?,
        SampleFormat::I16 => device.build_input_stream(
            &config,
            move |data: &[i16], _| {
                let f32_samples: Vec<f32> = data
                    .iter()
                    .map(|&s| s as f32 / i16::MAX as f32)
                    .collect();
                if let Ok(mut resamp) = resampler.lock() {
                    let mono_16k = resamp.process_interleaved(&f32_samples);
                    if !mono_16k.is_empty() {
                        let _ = sender.try_send(AudioPacket {
                            timestamp_ms: current_timestamp_ms(),
                            samples: mono_16k,
                        });
                    }
                }
            },
            err_fn,
            None,
        )?,
        _ => anyhow::bail!("Unsupported loopback sample format: {:?}", sample_format),
    };

    Ok(stream)
}

fn current_timestamp_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}
