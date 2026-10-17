use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, RwLock};
use std::sync::mpsc::Sender;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};


use crate::asr::SpeechRecognizer;
use crate::audio::DualAudioCapture;
use crate::diarization::{OnlineSpeakerClusterer, SpeechSegmenter};
use crate::events::{SpeakerId, TranscriptEvent};

/// Operating mode for speaker attribution
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum DiarizationMode {
    /// Standard Online Meeting (Zoom, Google Meet, Teams):
    /// Audio from the near-end microphone is deterministically [You].
    /// Audio from far-end loopback speakers is clustered into [Speaker 1], [Speaker 2], etc.
    #[default]
    OnlineMeeting,

    /// In-Person / Full Mic (Round-table conference room):
    /// Multiple people sit in the same room talking into the same laptop/table microphone.
    /// The microphone stream is acoustically clustered into [Speaker 1], [Speaker 2], etc.
    InPerson,

    /// Hybrid Mode:
    /// Uses the local user's enrolled voiceprint. When your voice is detected on the mic,
    /// it is labeled [You]; other voices near the microphone are clustered as [Speaker 1, 2...].
    Hybrid,
}

impl DiarizationMode {
    /// Cycle to the next mode on the fly without restarting the engine
    pub fn next(&self) -> Self {
        match self {
            DiarizationMode::OnlineMeeting => DiarizationMode::InPerson,
            DiarizationMode::InPerson => DiarizationMode::Hybrid,
            DiarizationMode::Hybrid => DiarizationMode::OnlineMeeting,
        }
    }
}

pub struct PipelineHandle {
    _mic_stream: cpal::Stream,
    _loopback_stream: Vec<cpal::Stream>,
    is_running: Arc<AtomicBool>,
    is_paused: Arc<AtomicBool>,
    mode: Arc<RwLock<DiarizationMode>>,
    mic_level: Arc<AtomicU32>,
    loopback_level: Arc<AtomicU32>,
    is_speaking: Arc<AtomicBool>,
    transcribing_count: Arc<AtomicU32>,
    thread_handle: Option<JoinHandle<()>>,
    asr_thread_handle: Option<JoinHandle<()>>,
}

impl PipelineHandle {
    /// Read the latest microphone audio RMS level (0.0 to 1.0)
    pub fn mic_level(&self) -> f32 {
        if self.is_paused.load(Ordering::Relaxed) {
            return 0.0;
        }
        f32::from_bits(self.mic_level.load(Ordering::Relaxed))
    }

    /// Read the latest loopback audio RMS level (0.0 to 1.0)
    pub fn loopback_level(&self) -> f32 {
        if self.is_paused.load(Ordering::Relaxed) {
            return 0.0;
        }
        f32::from_bits(self.loopback_level.load(Ordering::Relaxed))
    }

    /// Returns true if VAD is currently accumulating an active speech segment
    pub fn is_speaking(&self) -> bool {
        if self.is_paused.load(Ordering::Relaxed) {
            return false;
        }
        self.is_speaking.load(Ordering::Relaxed)
    }

    /// Pause or resume audio processing in the background pipeline
    pub fn set_paused(&self, paused: bool) {
        self.is_paused.store(paused, Ordering::SeqCst);
        if paused {
            self.mic_level.store(0, Ordering::Relaxed);
            self.loopback_level.store(0, Ordering::Relaxed);
            self.is_speaking.store(false, Ordering::Relaxed);
        }
        log::info!("Pipeline pause state changed: paused={}", paused);
    }

    /// Returns true if the pipeline is currently paused
    pub fn is_paused(&self) -> bool {
        self.is_paused.load(Ordering::Relaxed)
    }

    /// Returns the number of speech utterances currently being transcribed or queued for Whisper
    pub fn transcribing_count(&self) -> u32 {
        self.transcribing_count.load(Ordering::Relaxed)
    }

    /// Switch diarization mode on the fly during an active meeting session.
    /// Takes effect instantly on the next audio packet without restarting audio or models.
    pub fn set_mode(&self, new_mode: DiarizationMode) {
        if let Ok(mut lock) = self.mode.write() {
            log::info!("Diarization mode switched on the fly: {:?} -> {:?}", *lock, new_mode);
            *lock = new_mode;
        }
    }

    /// Read the currently active diarization mode.
    pub fn current_mode(&self) -> DiarizationMode {
        self.mode.read().map(|m| *m).unwrap_or_default()
    }

    /// Cycle to the next diarization mode on the fly.
    pub fn toggle_mode(&self) -> DiarizationMode {
        let next = self.current_mode().next();
        self.set_mode(next);
        next
    }

    pub fn stop(&mut self) {
        self.is_running.store(false, Ordering::SeqCst);
        if let Some(handle) = self.thread_handle.take() {
            let _ = handle.join();
        }
        // Wait briefly (up to 300ms) for any fast in-flight utterance to complete,
        // then detach ASR worker so quitting ('q') never freezes on slow models.
        for _ in 0..15 {
            if self.transcribing_count.load(Ordering::Relaxed) == 0 {
                break;
            }
            std::thread::sleep(Duration::from_millis(20));
        }
        let _ = self.asr_thread_handle.take();
    }
}

impl Drop for PipelineHandle {
    fn drop(&mut self) {
        self.stop();
    }
}

struct AsrTask {
    timestamp_ms: u64,
    speaker: SpeakerId,
    utterance: Vec<f32>,
    confidence: f32,
    is_loopback: bool,
}

/// Headless real-time pipeline connecting:
/// WASAPI Capture -> NLMS AEC -> VAD Speech Segmentation -> Speaker Diarization -> Speech Recognition
pub struct CatetinPipeline {
    recognizer: Box<dyn SpeechRecognizer>,
    clusterer: OnlineSpeakerClusterer,
    mode: DiarizationMode,
    enrolled_voiceprint: Option<Vec<f32>>,
    noise_suppression: crate::config::NoiseSuppression,
}

impl CatetinPipeline {
    pub fn new(recognizer: Box<dyn SpeechRecognizer>) -> Self {
        Self {
            recognizer,
            clusterer: OnlineSpeakerClusterer::default(),
            mode: DiarizationMode::default(),
            enrolled_voiceprint: None,
            noise_suppression: crate::config::NoiseSuppression::default(),
        }
    }

    /// Set the initial diarization mode (OnlineMeeting, InPerson / FullMic, or Hybrid)
    pub fn with_mode(mut self, mode: DiarizationMode) -> Self {
        self.mode = mode;
        self
    }

    /// Set the noise suppression mode (None, Some, Aggressive)
    pub fn with_noise_suppression(mut self, suppression: crate::config::NoiseSuppression) -> Self {
        self.noise_suppression = suppression;
        self
    }

    /// Enroll the local user's voiceprint embedding for Hybrid mode
    pub fn with_enrolled_voice(mut self, voiceprint: Vec<f32>) -> Self {
        self.enrolled_voiceprint = Some(voiceprint);
        self
    }

    /// Start processing audio capture in dedicated background worker threads:
    /// 1. Fast Audio + VAD thread (never blocks on ASR; keeps waveform & capture 100% live)
    /// 2. Dedicated Whisper ASR thread with 32MB stack and bounded queue
    // ponytail: std::sync::mpsc instead of async tokio channel. Ceiling: synchronous thread-based event dispatch. Upgrade path: add tokio wrapper only if embedding in an async GUI framework like Tauri.
    pub fn start(
        self,
        capture: DualAudioCapture,
        event_sender: Sender<TranscriptEvent>,
    ) -> Result<PipelineHandle> {
        let DualAudioCapture {
            _mic_stream,
            _loopback_stream,
            mic_receiver,
            loopback_receiver,
        } = capture;

        let is_running = Arc::new(AtomicBool::new(true));
        let running_audio = Arc::clone(&is_running);

        let is_paused = Arc::new(AtomicBool::new(false));
        let paused_audio = Arc::clone(&is_paused);
        let paused_asr = Arc::clone(&is_paused);

        let mode_arc = Arc::new(RwLock::new(self.mode));
        let mode_clone = Arc::clone(&mode_arc);

        let mic_level = Arc::new(AtomicU32::new(0));
        let mic_level_clone = Arc::clone(&mic_level);

        let loopback_level = Arc::new(AtomicU32::new(0));
        let loopback_level_clone = Arc::clone(&loopback_level);

        let is_speaking = Arc::new(AtomicBool::new(false));
        let is_speaking_clone = Arc::clone(&is_speaking);

        let transcribing_count = Arc::new(AtomicU32::new(0));
        let transcribing_asr = Arc::clone(&transcribing_count);
        let transcribing_audio = Arc::clone(&transcribing_count);

        let (asr_tx, asr_rx) = std::sync::mpsc::sync_channel::<AsrTask>(4);

        let mut recognizer = self.recognizer;
        let asr_thread_handle = std::thread::Builder::new()
            .name("catetin-whisper-asr".into())
            .stack_size(32 * 1024 * 1024)
            .spawn(move || {
                while let Ok(task) = asr_rx.recv() {
                    if paused_asr.load(Ordering::Relaxed) {
                        transcribing_asr.fetch_sub(1, Ordering::Relaxed);
                        continue;
                    }
                    let tag = if task.is_loopback {
                        "Whisper Loopback"
                    } else {
                        "Whisper"
                    };
                    let start_asr = Instant::now();
                    let res = recognizer.transcribe(&task.utterance);
                    transcribing_asr.fetch_sub(1, Ordering::Relaxed);
                    match res {
                        Ok(text) => {
                            let elapsed = start_asr.elapsed();
                            let trimmed = text.trim();
                            if trimmed.is_empty() {
                                log::info!(
                                    "[{}] Inference took {:?}, returned blank/non-speech tokens",
                                    tag,
                                    elapsed
                                );
                            } else {
                                log::info!(
                                    "[{}] Transcribed in {:?}: \"{}\" -> Speaker: {:?}",
                                    tag,
                                    elapsed,
                                    trimmed,
                                    task.speaker
                                );
                                let event = TranscriptEvent {
                                    timestamp_ms: task.timestamp_ms,
                                    speaker: task.speaker,
                                    text: trimmed.to_string(),
                                    is_final: true,
                                    confidence: task.confidence,
                                };
                                let _ = event_sender.send(event);
                            }
                        }
                        Err(e) => {
                            log::error!("[{}] Transcription FAILED: {:#}", tag, e);
                        }
                    }
                }
            })?;

        let suppression = self.noise_suppression;
        let mut clusterer = self.clusterer;
        let enrolled_voiceprint = self.enrolled_voiceprint;

        let thread_handle = std::thread::Builder::new()
            .name("catetin-audio-vad".into())
            .spawn(move || {
                let mut mic_segmenter = SpeechSegmenter::with_noise_suppression(16_000, suppression);
                let mut loopback_segmenter = SpeechSegmenter::for_loopback(16_000);
                let mut last_stats_log = Instant::now();
                let mut mic_packets_count = 0u64;
                let mut loopback_packets_count = 0u64;
                let mut peak_mic_rms = 0.0f32;
                let mut peak_loopback_rms = 0.0f32;

                while running_audio.load(Ordering::SeqCst) {
                    if paused_audio.load(Ordering::Relaxed) {
                        // Drain hardware capture buffers while paused so stale audio never queues up
                        while mic_receiver.try_recv().is_ok() {}
                        while loopback_receiver.try_recv().is_ok() {}
                        let _ = mic_segmenter.flush();
                        let _ = loopback_segmenter.flush();
                        mic_level_clone.store(0, Ordering::Relaxed);
                        loopback_level_clone.store(0, Ordering::Relaxed);
                        is_speaking_clone.store(false, Ordering::Relaxed);
                        std::thread::sleep(Duration::from_millis(20));
                        continue;
                    }

                    let mut had_audio = false;
                    let active_mode = mode_clone
                        .read()
                        .map(|m| *m)
                        .unwrap_or(DiarizationMode::OnlineMeeting);

                    // Periodic audio diagnostic heartbeat to help pinpoint hardware / signal problems
                    if last_stats_log.elapsed() >= Duration::from_secs(3) {
                        last_stats_log = Instant::now();
                        let cur_mic = f32::from_bits(mic_level_clone.load(Ordering::Relaxed));
                        let cur_lb = f32::from_bits(loopback_level_clone.load(Ordering::Relaxed));

                        if mic_packets_count == 0 {
                            log::warn!(
                                "[Audio Pipeline] Warning: ZERO microphone audio packets received in last 3s! (Selected Mic may be unplugged, muted, or restricted by Windows permissions)"
                            );
                        } else {
                            log::info!(
                                "[Audio Pipeline] 3s stats -> Mic: {} pkts, cur_rms={:.5}, peak_rms={:.5} | Loopback: {} pkts, cur_rms={:.5}, peak_rms={:.5} | Mode: {:?}",
                                mic_packets_count, cur_mic, peak_mic_rms, loopback_packets_count, cur_lb, peak_loopback_rms, active_mode
                            );
                        }
                        mic_packets_count = 0;
                        loopback_packets_count = 0;
                        peak_mic_rms = 0.0;
                        peak_loopback_rms = 0.0;
                    }

                    // 1. Process Near-End User Microphone
                    while let Ok(packet) = mic_receiver.try_recv() {
                        had_audio = true;
                        mic_packets_count += 1;
                        if !packet.samples.is_empty() {
                            let sum_sq: f32 = packet.samples.iter().map(|&x| x * x).sum();
                            let rms = (sum_sq / packet.samples.len() as f32).sqrt();
                            if rms > peak_mic_rms {
                                peak_mic_rms = rms;
                            }
                            mic_level_clone.store(rms.to_bits(), Ordering::Relaxed);
                        }
                        if let Some(utterance) = mic_segmenter.feed(&packet.samples) {
                            let utterance_len_s = utterance.len() as f32 / 16000.0;
                            log::info!(
                                "[Pipeline] Feeding utterance to Whisper: {} samples ({:.2}s), Mode: {:?}",
                                utterance.len(),
                                utterance_len_s,
                                active_mode
                            );
                            let speaker = match active_mode {
                                DiarizationMode::OnlineMeeting => SpeakerId::LocalUser,
                                DiarizationMode::InPerson => {
                                    let voiceprint = compute_voiceprint_features(&utterance);
                                    match clusterer.identify_or_register(&voiceprint) {
                                        SpeakerId::Remote(id) => SpeakerId::InPerson(id),
                                        other => other,
                                    }
                                }
                                DiarizationMode::Hybrid => {
                                    let voiceprint = compute_voiceprint_features(&utterance);
                                    if let Some(ref enrolled) = enrolled_voiceprint {
                                        let sim = crate::diarization::clustering::cosine_similarity(
                                            &voiceprint,
                                            enrolled,
                                        );
                                        if sim >= 0.72 {
                                            SpeakerId::LocalUser
                                        } else {
                                            clusterer.identify_or_register(&voiceprint)
                                        }
                                    } else {
                                        let spk = clusterer.identify_or_register(&voiceprint);
                                        if spk == SpeakerId::Remote(1) {
                                            SpeakerId::LocalUser
                                        } else {
                                            spk
                                        }
                                    }
                                }
                            };

                            if asr_tx
                                .try_send(AsrTask {
                                    timestamp_ms: packet.timestamp_ms,
                                    speaker,
                                    utterance,
                                    confidence: 0.95,
                                    is_loopback: false,
                                })
                                .is_ok()
                            {
                                transcribing_audio.fetch_add(1, Ordering::Relaxed);
                            } else {
                                log::warn!(
                                    "[Pipeline] Whisper queue full (3 pending segments), skipping segment to keep live latency low"
                                );
                            }
                        }
                    }

                    // 2. Process Far-End Loopback Audio (Remote meeting participants)
                    while let Ok(packet) = loopback_receiver.try_recv() {
                        had_audio = true;
                        loopback_packets_count += 1;
                        if !packet.samples.is_empty() {
                            let sum_sq: f32 = packet.samples.iter().map(|&x| x * x).sum();
                            let rms = (sum_sq / packet.samples.len() as f32).sqrt();
                            if rms > peak_loopback_rms {
                                peak_loopback_rms = rms;
                            }
                            loopback_level_clone.store(rms.to_bits(), Ordering::Relaxed);
                        }
                        if let Some(utterance) = loopback_segmenter.feed(&packet.samples) {
                            let utterance_len_s = utterance.len() as f32 / 16000.0;
                            log::info!(
                                "[Pipeline] Feeding Loopback utterance to Whisper: {} samples ({:.2}s)",
                                utterance.len(),
                                utterance_len_s
                            );
                            let voiceprint = compute_voiceprint_features(&utterance);
                            let speaker = clusterer.identify_or_register(&voiceprint);

                            if asr_tx
                                .try_send(AsrTask {
                                    timestamp_ms: packet.timestamp_ms,
                                    speaker,
                                    utterance,
                                    confidence: 0.92,
                                    is_loopback: true,
                                })
                                .is_ok()
                            {
                                transcribing_audio.fetch_add(1, Ordering::Relaxed);
                            } else {
                                log::warn!(
                                    "[Pipeline] Whisper queue full (3 pending segments), skipping loopback segment"
                                );
                            }
                        }
                    }

                    is_speaking_clone.store(
                        mic_segmenter.is_speaking() || loopback_segmenter.is_speaking(),
                        Ordering::Relaxed,
                    );

                    if !had_audio {
                        // Decay audio energy levels smoothly during silence frames
                        let cur_mic = f32::from_bits(mic_level_clone.load(Ordering::Relaxed));
                        if cur_mic > 0.001 {
                            mic_level_clone.store((cur_mic * 0.70).to_bits(), Ordering::Relaxed);
                        } else {
                            mic_level_clone.store(0, Ordering::Relaxed);
                        }
                        let cur_lb = f32::from_bits(loopback_level_clone.load(Ordering::Relaxed));
                        if cur_lb > 0.001 {
                            loopback_level_clone.store((cur_lb * 0.70).to_bits(), Ordering::Relaxed);
                        } else {
                            loopback_level_clone.store(0, Ordering::Relaxed);
                        }
                        std::thread::sleep(Duration::from_millis(12));
                    }
                }

                // Flush any partial speech segments currently buffered when the user exits ('q')
                let now_ms = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_millis() as u64;
                if let Some(utterance) = mic_segmenter.flush() {
                    transcribing_audio.fetch_add(1, Ordering::Relaxed);
                    let _ = asr_tx.send(AsrTask {
                        timestamp_ms: now_ms,
                        speaker: SpeakerId::LocalUser,
                        utterance,
                        confidence: 0.95,
                        is_loopback: false,
                    });
                }
                if let Some(utterance) = loopback_segmenter.flush() {
                    let voiceprint = compute_voiceprint_features(&utterance);
                    let speaker = clusterer.identify_or_register(&voiceprint);
                    transcribing_audio.fetch_add(1, Ordering::Relaxed);
                    let _ = asr_tx.send(AsrTask {
                        timestamp_ms: now_ms,
                        speaker,
                        utterance,
                        confidence: 0.92,
                        is_loopback: true,
                    });
                }
            })?;

        Ok(PipelineHandle {
            _mic_stream,
            _loopback_stream,
            is_running,
            is_paused,
            mode: mode_arc,
            mic_level,
            loopback_level,
            is_speaking,
            transcribing_count,
            thread_handle: Some(thread_handle),
            asr_thread_handle: Some(asr_thread_handle),
        })
    }
}

/// Compute a 64-band spectral voiceprint feature vector for speaker identification
fn compute_voiceprint_features(samples: &[f32]) -> Vec<f32> {
    const NUM_BINS: usize = 64;
    let mut bins = vec![0.0f32; NUM_BINS];
    if samples.is_empty() {
        return bins;
    }

    let chunk_size = (samples.len() / NUM_BINS).max(1);
    for (i, bin) in bins.iter_mut().enumerate() {
        let start = i * chunk_size;
        let end = (start + chunk_size).min(samples.len());
        if start < samples.len() {
            let energy: f32 = samples[start..end].iter().map(|&x| x * x).sum();
            *bin = (energy / (end - start) as f32).sqrt();
        }
    }

    crate::diarization::normalize_vector(&bins)
}
