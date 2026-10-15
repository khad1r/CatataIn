use anyhow::{Context, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Device, SampleFormat, Stream, StreamConfig};
use std::sync::mpsc::{sync_channel, Receiver, SyncSender};
use std::sync::{Arc, Mutex};

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
pub struct DualAudioCapture {
    _mic_stream: Stream,
    _loopback_stream: Stream,
    pub mic_receiver: Receiver<AudioPacket>,
    pub loopback_receiver: Receiver<AudioPacket>,
}

/// List all available audio input devices (microphones) on the system.
pub fn list_input_devices() -> Vec<String> {
    let host = cpal::default_host();
    let mut names = Vec::new();
    if let Ok(devices) = host.input_devices() {
        for device in devices {
            if let Ok(name) = device.name() {
                if !names.contains(&name) {
                    names.push(name);
                }
            }
        }
    }
    names
}

/// List all available audio output devices (speakers/headphones) on the system.
pub fn list_output_devices() -> Vec<String> {
    let host = cpal::default_host();
    let mut names = Vec::new();
    if let Ok(devices) = host.output_devices() {
        for device in devices {
            if let Ok(name) = device.name() {
                if !names.contains(&name) {
                    names.push(name);
                }
            }
        }
    }
    names
}

fn find_device(devices: impl Iterator<Item = Device>, target: &str) -> Option<Device> {
    let target_clean = target.trim().to_lowercase();
    let mut exact_match = None;
    let mut partial_match = None;

    for device in devices {
        if let Ok(name) = device.name() {
            let name_clean = name.trim().to_lowercase();
            if name_clean == target_clean {
                exact_match = Some(device);
                break;
            } else if name_clean.contains(&target_clean) && partial_match.is_none() {
                partial_match = Some(device);
            }
        }
    }

    exact_match.or(partial_match)
}

impl DualAudioCapture {
    /// Initialize dual capture using default communication input and output devices.
    pub fn new() -> Result<Self> {
        Self::with_device_names(None, None)
    }

    /// Initialize dual capture from user configuration.
    pub fn from_config(config: &crate::config::AppConfig) -> Result<Self> {
        Self::with_device_names(
            config.input_device_name.as_deref(),
            config.output_device_name.as_deref(),
        )
    }

    /// Initialize dual capture with optional device name filters.
    pub fn with_device_names(
        input_name_filter: Option<&str>,
        output_name_filter: Option<&str>,
    ) -> Result<Self> {
        let host = cpal::default_host();

        let input_device = if let Some(target) = input_name_filter {
            find_device(host.input_devices()?, target)
                .or_else(|| host.default_input_device())
                .context("No matching or default audio input device (microphone) found")?
        } else {
            host.default_input_device()
                .context("No default audio input device (microphone) found")?
        };

        let output_device = if let Some(target) = output_name_filter {
            find_device(host.output_devices()?, target)
                .or_else(|| host.default_output_device())
                .context("No matching or default audio output device (speakers/headphones) found")?
        } else {
            host.default_output_device()
                .context("No default audio output device (speakers/headphones) found")?
        };

        let input_name = input_device.name().unwrap_or_else(|_| "Default Mic".into());
        let output_name = output_device.name().unwrap_or_else(|_| "Default Output".into());

        log::info!("DualAudioCapture: Mic device: {}", input_name);
        log::info!("DualAudioCapture: Loopback device: {}", output_name);

        Self::from_devices(&input_device, &output_device)
    }


    pub fn from_devices(input_device: &Device, output_device: &Device) -> Result<Self> {
        let (mic_tx, mic_rx) = sync_channel::<AudioPacket>(100);
        let (loopback_tx, loopback_rx) = sync_channel::<AudioPacket>(100);

        let mic_stream = build_capture_stream(input_device, true, "Microphone", mic_tx)?;
        let loopback_stream = build_capture_stream(output_device, false, "Loopback", loopback_tx)?;

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

fn build_capture_stream(
    device: &Device,
    is_input: bool,
    stream_name: &'static str,
    sender: SyncSender<AudioPacket>,
) -> Result<Stream> {
    let default_config = if is_input {
        device
            .default_input_config()
            .with_context(|| format!("Failed to get default input config for {}", stream_name))?
    } else {
        // On Windows WASAPI, building an input stream on an output device activates AUDCLNT_STREAMFLAGS_LOOPBACK
        device
            .default_output_config()
            .with_context(|| format!("Failed to get default output config for {}", stream_name))?
    };

    let sample_format = default_config.sample_format();
    let config: StreamConfig = default_config.into();
    let channels = config.channels;
    let sample_rate = config.sample_rate.0;

    let resampler = Arc::new(Mutex::new(AudioResampler::new(
        sample_rate,
        channels,
        STANDARD_SAMPLE_RATE,
    )));

    let err_fn = move |err| log::error!("{} stream error: {}", stream_name, err);

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
        _ => anyhow::bail!("Unsupported {} sample format: {:?}", stream_name, sample_format),
    };

    Ok(stream)
}

fn current_timestamp_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}
