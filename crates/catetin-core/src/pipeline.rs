use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, RwLock};
use std::sync::mpsc::Sender;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};


use crate::asr::{SpeechRecognizer, StreamingTransducer};
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
    #[allow(dead_code)]
    turn_start_ms: u64,
    timestamp_ms: u64,
    speaker: SpeakerId,
    voiceprint: Vec<f32>,
    utterance: Vec<f32>,
    is_loopback: bool,
    has_live_transducer: bool,
    transducer_had_text: bool,
    transducer_text: String,
}

struct WindowEntry {
    timestamp_ms: u64,
    end_ms: u64,
    speaker: SpeakerId,
    voiceprint: Vec<f32>,
    utterance: Vec<f32>,
    text: String,
    confidence: f32,
    is_loopback: bool,
    context_before: String,
    #[allow(dead_code)]
    refined: bool,
}

/// Duration-weighted retroactive speaker smoothing over the recent 45s window (<0.2ms CPU).
/// Prevents short (<2.2s) utterances like "Oke" or "Setuju" from flipping speaker IDs.
fn smooth_speaker_with_window(
    initial_speaker: &SpeakerId,
    voiceprint: &[f32],
    duration_s: f32,
    is_loopback: bool,
    window: &[WindowEntry],
) -> SpeakerId {
    if *initial_speaker == SpeakerId::LocalUser || voiceprint.is_empty() || window.is_empty() {
        return initial_speaker.clone();
    }

    // Aggregate duration-weighted similarity per speaker in the recent window on the same stream
    let mut best_speaker: Option<SpeakerId> = None;
    let mut best_score = 0.0f32;

    for candidate_entry in window.iter().filter(|e| e.is_loopback == is_loopback) {
        if candidate_entry.speaker == SpeakerId::LocalUser || candidate_entry.voiceprint.is_empty() {
            continue;
        }
        let mut weighted_sim_sum = 0.0f32;
        let mut weight_sum = 0.0f32;
        for e in window
            .iter()
            .filter(|e| e.is_loopback == is_loopback && e.speaker == candidate_entry.speaker)
        {
            let w = (e.utterance.len() as f32 / 16_000.0).clamp(0.5, 8.0);
            let sim = crate::diarization::clustering::cosine_similarity(voiceprint, &e.voiceprint);
            weighted_sim_sum += sim * w;
            weight_sum += w;
        }
        if weight_sum > 0.0 {
            let avg_sim = weighted_sim_sum / weight_sum;
            // Short utterances (<2.2s) have noisier acoustic estimates; bias slightly toward recently active speaker
            let threshold = if duration_s < 2.2 { 0.68 } else { 0.73 };
            if avg_sim >= threshold && avg_sim > best_score {
                best_score = avg_sim;
                best_speaker = Some(candidate_entry.speaker.clone());
            }
        }
    }

    best_speaker.unwrap_or_else(|| initial_speaker.clone())
}

/// Build rolling 30-second text context from recent confirmed window entries
fn build_window_context(window: &[WindowEntry], now_ms: u64) -> String {
    let mut combined = String::new();
    for entry in window {
        if now_ms.saturating_sub(entry.timestamp_ms) <= 30_000 && entry.confidence >= 0.55 {
            if !combined.is_empty() {
                combined.push(' ');
            }
            combined.push_str(&entry.text);
        }
    }
    if combined.chars().count() > 160 {
        let skip = combined.chars().count() - 160;
        combined = combined.chars().skip(skip).collect();
        if let Some(idx) = combined.find(' ') {
            combined = combined[idx + 1..].to_string();
        }
    }
    combined
}

/// Headless real-time pipeline connecting:
/// WASAPI Capture -> NLMS AEC -> VAD Speech Segmentation -> Speaker Diarization -> Speech Recognition
pub struct CatetinPipeline {
    recognizer: Box<dyn SpeechRecognizer>,
    clusterer: OnlineSpeakerClusterer,
    mode: DiarizationMode,
    enrolled_voiceprint: Option<Vec<f32>>,
    noise_suppression: crate::config::NoiseSuppression,
    models_dir: Option<std::path::PathBuf>,
    transducer_dir: Option<std::path::PathBuf>,
    asr_language: String,
    mic_transducer: Option<Box<dyn StreamingTransducer>>,
    lb_transducer: Option<Box<dyn StreamingTransducer>>,
}

impl CatetinPipeline {
    pub fn new(recognizer: Box<dyn SpeechRecognizer>) -> Self {
        Self {
            recognizer,
            clusterer: OnlineSpeakerClusterer::default(),
            mode: DiarizationMode::default(),
            enrolled_voiceprint: None,
            noise_suppression: crate::config::NoiseSuppression::default(),
            models_dir: None,
            transducer_dir: None,
            asr_language: "auto".to_string(),
            mic_transducer: None,
            lb_transducer: None,
        }
    }

    /// Provide preloaded Streaming Transducers for Pass 1 live typing
    pub fn with_transducers(
        mut self,
        mic: Box<dyn StreamingTransducer>,
        lb: Box<dyn StreamingTransducer>,
    ) -> Self {
        self.mic_transducer = Some(mic);
        self.lb_transducer = Some(lb);
        self
    }

    /// Set the ASR speech recognition language ("auto", "id", or "en")
    pub fn with_asr_language(mut self, lang: impl Into<String>) -> Self {
        self.asr_language = lang.into();
        self
    }

    /// Set the directory where models and dynamic runtimes reside
    pub fn with_models_dir(mut self, dir: std::path::PathBuf) -> Self {
        self.models_dir = Some(dir);
        self
    }

    /// Set the directory where Streaming Transducer models and dynamic runtimes reside
    pub fn with_transducer_dir(mut self, dir: std::path::PathBuf) -> Self {
        self.transducer_dir = Some(dir);
        self
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
    /// 2. Dedicated Whisper ASR thread with 32MB stack, rolling 30s prompt context,
    ///    retroactive 45s diarization smoothing, and selective Pass-2 boundary stitching.
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

        let models_dir = self.models_dir.unwrap_or_else(|| {
            crate::config::default_app_dir().join("models")
        });
        let transducer_dir = self.transducer_dir.unwrap_or_else(|| models_dir.clone());

        let (mic_transducer, lb_transducer) = match (self.mic_transducer, self.lb_transducer) {
            (Some(m), Some(l)) => (m, l),
            _ => {
                let asr_lang = if self.asr_language.is_empty() { "auto" } else { self.asr_language.as_str() };
                if !crate::asr::has_transducer_installed_for_lang(&transducer_dir, asr_lang) {
                    anyhow::bail!(
                        "Streaming Transducer is mandatory for Catetin Pass 1 (< 50ms live captioning)!\n\
                         Missing runtime library (sherpa-onnx-c-api.dll / libsherpa-onnx-c-api.so)\n\
                         or Zipformer model files (encoder.onnx, decoder.onnx, joiner.onnx, tokens.txt) for language '{}' in {:?}.\n\
                         Run 'catetin' to automatically prepare and download the required Streaming Transducer files.",
                        asr_lang, transducer_dir
                    );
                }

                let m = crate::asr::create_transducer_for_lang(&transducer_dir, asr_lang)
                    .context("Failed to initialize microphone Streaming Transducer")?;
                let l = crate::asr::create_transducer_for_lang(&transducer_dir, asr_lang)
                    .context("Failed to initialize loopback Streaming Transducer")?;
                (m, l)
            }
        };

        let live_event_tx = event_sender.clone();

        let (asr_tx, asr_rx) = std::sync::mpsc::sync_channel::<AsrTask>(32);

        let mut recognizer = self.recognizer;
        let asr_thread_handle = std::thread::Builder::new()
            .name("catetin-whisper-asr".into())
            .stack_size(32 * 1024 * 1024)
            .spawn(move || {
                #[cfg(windows)]
                unsafe {
                    extern "system" {
                        fn GetCurrentThread() -> isize;
                        fn SetThreadPriority(h_thread: isize, n_priority: i32) -> i32;
                    }
                    // THREAD_PRIORITY_BELOW_NORMAL (-1): ensures Whisper inference yields to the UI (+1)
                    // and audio capture without Windows throttling the CPU clock down as happens on -2!
                    let _ = SetThreadPriority(GetCurrentThread(), -1);
                }

                let mut window: Vec<WindowEntry> = Vec::new();

                while let Ok(task) = asr_rx.recv() {
                    let tag = if task.is_loopback {
                        "Whisper Loopback"
                    } else {
                        "Whisper"
                    };

                    // Prune entries older than 45 seconds from the retroactive window
                    window.retain(|e| task.timestamp_ms.saturating_sub(e.timestamp_ms) <= 45_000);

                    let duration_ms = (task.utterance.len() as u64 * 1000) / 16_000;
                    let duration_s = task.utterance.len() as f32 / 16_000.0;
                    let smoothed_speaker = smooth_speaker_with_window(
                        &task.speaker,
                        &task.voiceprint,
                        duration_s,
                        task.is_loopback,
                        &window,
                    );
                    let context_before = build_window_context(&window, task.timestamp_ms);

                    // PASS 1: Immediate Live Caption with rolling 30s prompt context
                    let start_asr = Instant::now();
                    let res = recognizer.transcribe_with_context(
                        &task.utterance,
                        if context_before.is_empty() {
                            None
                        } else {
                            Some(context_before.as_str())
                        },
                    );
                    match res {
                        Ok((text, confidence)) => {
                            let elapsed = start_asr.elapsed();
                            let trimmed = text.trim();

                            // Detect non-Latin script hallucinations (Thai, CJK, Hangul, Japanese) on Indonesian/English audio
                            let is_non_latin_hallucination = trimmed.chars().any(|c| {
                                let u = c as u32;
                                (0x0E00..=0x0E7F).contains(&u) // Thai
                                    || (0x4E00..=0x9FFF).contains(&u) // CJK Ideographs
                                    || (0xAC00..=0xD7AF).contains(&u) // Hangul
                                    || (0x3040..=0x30FF).contains(&u) // Hiragana/Katakana
                            });

                            let is_hallucination = trimmed.is_empty()
                                || is_non_latin_hallucination
                                || crate::asr::is_hallucination(trimmed)
                                || crate::asr::has_repetition_hallucination(trimmed);

                            if is_hallucination {
                                transcribing_asr.fetch_sub(1, Ordering::Relaxed);
                                log::info!(
                                    "[{}] Inference took {:?}, returned blank/hallucination tokens: \"{}\"",
                                    tag,
                                    elapsed,
                                    trimmed
                                );
                                let fallback = task.transducer_text.trim();
                                if !fallback.is_empty() {
                                    log::info!(
                                        "[{}] -> Preserving valid Pass-1 Transducer text as final: \"{}\"",
                                        tag,
                                        fallback
                                    );
                                    let event = TranscriptEvent {
                                        timestamp_ms: task.turn_start_ms,
                                        speaker: smoothed_speaker.clone(),
                                        text: fallback.to_string(),
                                        is_final: true,
                                        confidence: 0.85,
                                        replaces_since_ms: None,
                                    };
                                    let _ = event_sender.send(event);
                                    window.push(WindowEntry {
                                        timestamp_ms: task.timestamp_ms,
                                        end_ms: task.timestamp_ms + duration_ms,
                                        speaker: smoothed_speaker,
                                        voiceprint: task.voiceprint,
                                        utterance: task.utterance,
                                        text: fallback.to_string(),
                                        confidence: 0.85,
                                        is_loopback: task.is_loopback,
                                        context_before,
                                        refined: false,
                                    });
                                } else if task.has_live_transducer {
                                    let _ = event_sender.send(TranscriptEvent {
                                        timestamp_ms: task.turn_start_ms,
                                        speaker: smoothed_speaker,
                                        text: String::new(),
                                        is_final: true,
                                        confidence: 0.85,
                                        replaces_since_ms: None,
                                    });
                                }
                                continue;
                            }

                            // If Pass 1 Transducer detected ZERO words, reject low-confidence uncorroborated Whisper text on instrumental music / noise
                            if task.has_live_transducer && !task.transducer_had_text && confidence < 0.28 {
                                transcribing_asr.fetch_sub(1, Ordering::Relaxed);
                                log::info!(
                                    "[{}] Discarding uncorroborated text on music/noise (Transducer heard 0 words, Whisper conf={:.2}): \"{}\"",
                                    tag,
                                    confidence,
                                    trimmed
                                );
                                let _ = event_sender.send(TranscriptEvent {
                                    timestamp_ms: task.turn_start_ms,
                                    speaker: smoothed_speaker,
                                    text: String::new(),
                                    is_final: true,
                                    confidence: 0.85,
                                    replaces_since_ms: None,
                                });
                                continue;
                            }

                            // Check if this segment is a candidate for Pass-2 turn stitching with the previous segment
                            let can_stitch_with_prev = (trimmed.len() >= 6 && duration_s >= 1.5 && confidence >= 0.50)
                                && window
                                    .iter()
                                    .rev()
                                    .find(|e| e.is_loopback == task.is_loopback)
                                    .map(|prev| {
                                        if prev.text.trim().len() < 6 || prev.confidence < 0.50 {
                                            return false;
                                        }
                                        let gap_ms = task.timestamp_ms.saturating_sub(prev.end_ms);
                                        let combined_samples = prev.utterance.len() + task.utterance.len();
                                        let same_or_close_spk = prev.speaker == smoothed_speaker
                                            || crate::diarization::clustering::cosine_similarity(
                                                &prev.voiceprint,
                                                &task.voiceprint,
                                            ) >= 0.72;
                                        same_or_close_spk
                                            && combined_samples <= 240_000 // <= 15.0s combined
                                            && (gap_ms <= 2_200
                                                || confidence < 0.78
                                                || prev.confidence < 0.78
                                                || duration_s < 2.0)
                                    })
                                    .unwrap_or(false);

                            log::info!(
                                "[{}] Model (Whisper Pass-1) in {:?} (conf={:.2}): \"{}\" -> Speaker: {:?}",
                                tag,
                                elapsed,
                                confidence,
                                trimmed,
                                smoothed_speaker
                            );

                            let event = TranscriptEvent {
                                timestamp_ms: task.turn_start_ms,
                                speaker: smoothed_speaker.clone(),
                                text: trimmed.to_string(),
                                is_final: true,
                                confidence,
                                replaces_since_ms: None,
                            };
                            let _ = event_sender.send(event);
                            transcribing_asr.fetch_sub(1, Ordering::Relaxed);

                            window.push(WindowEntry {
                                timestamp_ms: task.timestamp_ms,
                                end_ms: task.timestamp_ms + duration_ms,
                                speaker: smoothed_speaker.clone(),
                                voiceprint: task.voiceprint,
                                utterance: task.utterance,
                                text: trimmed.to_string(),
                                confidence,
                                is_loopback: task.is_loopback,
                                context_before,
                                refined: false,
                            });

                            // PASS 2: Selective Window Refinement & Boundary Stitching
                            // Disabled for live real-time performance: Pass 1 + Streaming Transducer
                            // already provides live verbatim text without doubling Whisper CPU workload.
                            const ENABLE_PASS2: bool = false;
                            if ENABLE_PASS2 && can_stitch_with_prev && transcribing_asr.load(Ordering::Relaxed) == 0 {
                                // Collect the trailing 2..=3 matching segments on the same channel
                                let mut indices: Vec<usize> = Vec::new();
                                let mut total_samples = 0usize;
                                for (idx, entry) in window.iter().enumerate().rev() {
                                    if entry.is_loopback != task.is_loopback {
                                        continue;
                                    }
                                    if indices.is_empty() {
                                        total_samples += entry.utterance.len();
                                        indices.push(idx);
                                    } else {
                                        let next_idx = *indices.last().unwrap();
                                        let next_entry = &window[next_idx];
                                        let gap_ms = next_entry.timestamp_ms.saturating_sub(entry.end_ms);
                                        let spk_match = entry.speaker == smoothed_speaker
                                            || crate::diarization::clustering::cosine_similarity(
                                                &entry.voiceprint,
                                                &next_entry.voiceprint,
                                            ) >= 0.72;
                                        if spk_match
                                            && gap_ms <= 2_500
                                            && total_samples + entry.utterance.len() <= 240_000
                                            && indices.len() < 3
                                        {
                                            total_samples += entry.utterance.len();
                                            indices.push(idx);
                                        } else {
                                            break;
                                        }
                                    }
                                }
                                indices.reverse();

                                if indices.len() >= 2 {
                                    let first_idx = indices[0];
                                    let last_idx = *indices.last().unwrap();
                                    let first_ts_ms = window[first_idx].timestamp_ms;
                                    let last_ts_ms = window[last_idx].timestamp_ms;
                                    let last_end_ms = window[last_idx].end_ms;
                                    let prompt_before = window[first_idx].context_before.clone();

                                    // Stitch audio with a 120ms (1,920 samples) natural pause bridge between phrases
                                    let bridge = 1_920usize;
                                    let mut stitched_audio =
                                        Vec::with_capacity(total_samples + bridge * indices.len());
                                    for (pos, &i) in indices.iter().enumerate() {
                                        if pos > 0 {
                                            stitched_audio.resize(stitched_audio.len() + bridge, 0.0);
                                        }
                                        stitched_audio.extend_from_slice(&window[i].utterance);
                                    }

                                    // Re-evaluate full-window voiceprint across the entire stitched turn
                                    let stitched_vp = compute_voiceprint_features(&stitched_audio);
                                    let stitched_dur_s = stitched_audio.len() as f32 / 16_000.0;
                                    let refined_speaker = smooth_speaker_with_window(
                                        &smoothed_speaker,
                                        &stitched_vp,
                                        stitched_dur_s,
                                        task.is_loopback,
                                        &window,
                                    );

                                    transcribing_asr.fetch_add(1, Ordering::Relaxed);
                                    let start_refine = Instant::now();
                                    let refine_res = recognizer.transcribe_with_context(
                                        &stitched_audio,
                                        if prompt_before.is_empty() {
                                            None
                                        } else {
                                            Some(prompt_before.as_str())
                                        },
                                    );
                                    transcribing_asr.fetch_sub(1, Ordering::Relaxed);

                                    if let Ok((refined_text, refined_conf)) = refine_res {
                                        let clean_refined = refined_text.trim();
                                        if !clean_refined.is_empty()
                                            && !crate::asr::is_hallucination(clean_refined)
                                            && !crate::asr::has_repetition_hallucination(clean_refined)
                                        {
                                            log::info!(
                                                "[{}] Pass-2 Refined {} segments ({:.1}s audio) in {:?} (conf={:.2}): \"{}\" -> Speaker: {:?}",
                                                tag,
                                                indices.len(),
                                                stitched_dur_s,
                                                start_refine.elapsed(),
                                                refined_conf,
                                                clean_refined,
                                                refined_speaker
                                            );

                                            let replace_event = TranscriptEvent {
                                                timestamp_ms: last_ts_ms,
                                                speaker: refined_speaker.clone(),
                                                text: clean_refined.to_string(),
                                                is_final: true,
                                                confidence: refined_conf,
                                                replaces_since_ms: Some(first_ts_ms),
                                            };
                                            let _ = event_sender.send(replace_event);

                                            // Replace the stitched sub-entries in window with the consolidated entry
                                            for &rm_idx in indices.iter().rev() {
                                                window.remove(rm_idx);
                                            }
                                            window.push(WindowEntry {
                                                timestamp_ms: first_ts_ms,
                                                end_ms: last_end_ms,
                                                speaker: refined_speaker,
                                                voiceprint: stitched_vp,
                                                utterance: stitched_audio,
                                                text: clean_refined.to_string(),
                                                confidence: refined_conf,
                                                is_loopback: task.is_loopback,
                                                context_before: prompt_before,
                                                refined: true,
                                            });
                                        }
                                    }
                                }
                            }
                        }
                        Err(e) => {
                            transcribing_asr.fetch_sub(1, Ordering::Relaxed);
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
                let mut mic_transducer = mic_transducer;
                let mut lb_transducer = lb_transducer;
                let mut mic_turn_start_ms = 0u64;
                let mut lb_turn_start_ms = 0u64;
                let mut mic_in_turn = false;
                let mut lb_in_turn = false;
                let mut mic_had_text = false;
                let mut lb_had_text = false;
                let mut last_stats_log = Instant::now();
                let mut mic_packets_count = 0u64;
                let mut loopback_packets_count = 0u64;
                let mut peak_mic_rms = 0.0f32;
                let mut peak_loopback_rms = 0.0f32;
                let mut last_mic_ts_ms = 0u64;
                let mut last_lb_ts_ms = 0u64;

                // Drain any startup audio packets that accumulated while models/transducers were loading
                let mut drained_mic = 0;
                while mic_receiver.try_recv().is_ok() {
                    drained_mic += 1;
                }
                let mut drained_lb = 0;
                while loopback_receiver.try_recv().is_ok() {
                    drained_lb += 1;
                }
                if drained_mic > 0 || drained_lb > 0 {
                    log::info!(
                        "DualAudioCapture: Drained startup audio buffers ({} mic, {} loopback packets). Live capture is starting clean.",
                        drained_mic,
                        drained_lb
                    );
                }

                let mut was_paused = false;
                while running_audio.load(Ordering::SeqCst) {
                    if paused_audio.load(Ordering::Relaxed) {
                        if !was_paused {
                            was_paused = true;
                            // Flush any in-progress speech spoken right before pressing pause into the Whisper queue
                            if let Some(utterance) = mic_segmenter.flush() {
                                let live_text = mic_transducer.get_partial_result();
                                mic_transducer.reset();
                                let turn_start = if mic_in_turn { mic_turn_start_ms } else { last_mic_ts_ms };
                                mic_in_turn = false;
                                let had_text = mic_had_text || !live_text.trim().is_empty();
                                mic_had_text = false;

                                let voiceprint = compute_voiceprint_features(&utterance);
                                if asr_tx
                                    .try_send(AsrTask {
                                        turn_start_ms: turn_start,
                                        timestamp_ms: last_mic_ts_ms,
                                        speaker: SpeakerId::LocalUser,
                                        voiceprint,
                                        utterance,
                                        is_loopback: false,
                                        has_live_transducer: true,
                                        transducer_had_text: had_text,
                                        transducer_text: live_text,
                                    })
                                    .is_ok()
                                {
                                    transcribing_audio.fetch_add(1, Ordering::Relaxed);
                                }
                            }
                            if let Some(utterance) = loopback_segmenter.flush() {
                                let live_text = lb_transducer.get_partial_result();
                                lb_transducer.reset();
                                let turn_start = if lb_in_turn { lb_turn_start_ms } else { last_lb_ts_ms };
                                lb_in_turn = false;
                                let had_text = lb_had_text || !live_text.trim().is_empty();
                                lb_had_text = false;

                                let voiceprint = compute_voiceprint_features(&utterance);
                                let speaker = clusterer.identify_or_register(&voiceprint);
                                if asr_tx
                                    .try_send(AsrTask {
                                        turn_start_ms: turn_start,
                                        timestamp_ms: last_lb_ts_ms,
                                        speaker,
                                        voiceprint,
                                        utterance,
                                        is_loopback: true,
                                        has_live_transducer: true,
                                        transducer_had_text: had_text,
                                        transducer_text: live_text,
                                    })
                                    .is_ok()
                                {
                                    transcribing_audio.fetch_add(1, Ordering::Relaxed);
                                }
                            }
                        }

                        // Discard all incoming audio while paused so pause is truly paused!
                        while mic_receiver.try_recv().is_ok() {}
                        while loopback_receiver.try_recv().is_ok() {}

                        mic_level_clone.store(0, Ordering::Relaxed);
                        loopback_level_clone.store(0, Ordering::Relaxed);
                        is_speaking_clone.store(false, Ordering::Relaxed);
                        std::thread::sleep(Duration::from_millis(20));
                        continue;
                    }
                    was_paused = false;

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
                        last_mic_ts_ms = packet.timestamp_ms;
                        if !packet.samples.is_empty() {
                            let sum_sq: f32 = packet.samples.iter().map(|&x| x * x).sum();
                            let rms = (sum_sq / packet.samples.len() as f32).sqrt();
                            if rms > peak_mic_rms {
                                peak_mic_rms = rms;
                            }
                            mic_level_clone.store(rms.to_bits(), Ordering::Relaxed);
                        }

                        let utterance_opt = mic_segmenter.feed(&packet.samples);
                        let is_speaking = mic_segmenter.is_speaking();

                        // PASS 1: Streaming Transducer for near-end user microphone (< 50ms live caption)
                        // Feed continuously while in an active speech turn to avoid dropping audio on unvoiced consonants
                        if is_speaking || mic_in_turn {
                            if !mic_in_turn {
                                mic_in_turn = true;
                                mic_turn_start_ms = packet.timestamp_ms;
                            }
                            mic_transducer.accept_waveform(&packet.samples);
                            if let Some(partial) = mic_transducer.decode_step() {
                                if !partial.trim().is_empty() {
                                    mic_had_text = true;
                                }
                                let _ = live_event_tx.send(TranscriptEvent {
                                    timestamp_ms: packet.timestamp_ms,
                                    speaker: SpeakerId::LocalUser,
                                    text: partial,
                                    is_final: false,
                                    confidence: 0.90,
                                    replaces_since_ms: Some(mic_turn_start_ms),
                                });
                            }
                        }

                        if let Some(utterance) = utterance_opt {
                            let live_text = mic_transducer.get_partial_result();
                            mic_transducer.reset();
                            let turn_start = if mic_in_turn { mic_turn_start_ms } else { packet.timestamp_ms };
                            mic_in_turn = false;
                            let had_text = mic_had_text || !live_text.trim().is_empty();
                            mic_had_text = false;

                            let utterance_len_s = utterance.len() as f32 / 16000.0;
                            log::info!(
                                "[Pipeline] Feeding utterance to Whisper Pass 2: {} samples ({:.2}s), Mode: {:?}",
                                utterance.len(),
                                utterance_len_s,
                                active_mode
                            );
                            let voiceprint = compute_voiceprint_features(&utterance);
                            let speaker = match active_mode {
                                DiarizationMode::OnlineMeeting => SpeakerId::LocalUser,
                                DiarizationMode::InPerson => {
                                    match clusterer.identify_or_register(&voiceprint) {
                                        SpeakerId::Remote(id) => SpeakerId::InPerson(id),
                                        other => other,
                                    }
                                }
                                DiarizationMode::Hybrid => {
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

                            let task_to_send = AsrTask {
                                turn_start_ms: turn_start,
                                timestamp_ms: packet.timestamp_ms,
                                speaker: speaker.clone(),
                                voiceprint,
                                utterance,
                                is_loopback: false,
                                has_live_transducer: true,
                                transducer_had_text: had_text,
                                transducer_text: live_text.clone(),
                            };

                            if asr_tx.try_send(task_to_send).is_ok() {
                                transcribing_audio.fetch_add(1, Ordering::Relaxed);
                            } else {
                                log::warn!(
                                    "[Pipeline] Whisper queue full (32 pending segments), committing Transducer text as final fallback"
                                );
                                if !live_text.trim().is_empty() {
                                    let _ = live_event_tx.send(TranscriptEvent {
                                        timestamp_ms: turn_start,
                                        speaker,
                                        text: live_text,
                                        is_final: true,
                                        confidence: 0.85,
                                        replaces_since_ms: None,
                                    });
                                }
                            }
                        }
                    }

                    // 2. Process Far-End Loopback Audio (Remote meeting participants)
                    while let Ok(packet) = loopback_receiver.try_recv() {
                        had_audio = true;
                        loopback_packets_count += 1;
                        last_lb_ts_ms = packet.timestamp_ms;
                        if !packet.samples.is_empty() {
                            let sum_sq: f32 = packet.samples.iter().map(|&x| x * x).sum();
                            let rms = (sum_sq / packet.samples.len() as f32).sqrt();
                            if rms > peak_loopback_rms {
                                peak_loopback_rms = rms;
                            }
                            loopback_level_clone.store(rms.to_bits(), Ordering::Relaxed);
                        }

                        // Digital loopback audio is often attenuated by Windows volume slider (often peak < 0.05).
                        // Scale quiet loopback speech dynamically to standard nominal range (~0.35) for Zipformer
                        let lb_peak = packet.samples.iter().fold(0.0f32, |m, &s| m.max(s.abs()));
                        let lb_samples: Vec<f32> = if lb_peak > 0.001 && lb_peak < 0.20 {
                            let gain = (0.35 / lb_peak).min(10.0);
                            packet.samples.iter().map(|&s| (s * gain).clamp(-1.0, 1.0)).collect()
                        } else {
                            packet.samples.clone()
                        };

                        let lb_utterance_opt = loopback_segmenter.feed(&lb_samples);
                        let lb_speaking = loopback_segmenter.is_speaking();

                        // PASS 1: Streaming Transducer for far-end loopback audio (< 50ms live caption)
                        // Feed continuously while in an active speech turn to avoid dropping audio on unvoiced consonants
                        if lb_speaking || lb_in_turn {
                            if !lb_in_turn {
                                lb_in_turn = true;
                                lb_turn_start_ms = packet.timestamp_ms;
                            }
                            lb_transducer.accept_waveform(&lb_samples);
                            if let Some(partial) = lb_transducer.decode_step() {
                                if !partial.trim().is_empty() {
                                    lb_had_text = true;
                                }
                                let _ = live_event_tx.send(TranscriptEvent {
                                    timestamp_ms: packet.timestamp_ms,
                                    speaker: SpeakerId::Remote(1),
                                    text: partial,
                                    is_final: false,
                                    confidence: 0.90,
                                    replaces_since_ms: Some(lb_turn_start_ms),
                                });
                            }
                        }

                        if let Some(utterance) = lb_utterance_opt {
                            let live_text = lb_transducer.get_partial_result();
                            lb_transducer.reset();
                            let turn_start = if lb_in_turn { lb_turn_start_ms } else { packet.timestamp_ms };
                            lb_in_turn = false;
                            let had_text = lb_had_text || !live_text.trim().is_empty();
                            lb_had_text = false;

                            let utterance_len_s = utterance.len() as f32 / 16000.0;
                            log::info!(
                                "[Pipeline] Feeding Loopback utterance to Whisper Pass 2: {} samples ({:.2}s)",
                                utterance.len(),
                                utterance_len_s
                            );
                            let voiceprint = compute_voiceprint_features(&utterance);
                            let speaker = clusterer.identify_or_register(&voiceprint);

                            let task_to_send = AsrTask {
                                turn_start_ms: turn_start,
                                timestamp_ms: packet.timestamp_ms,
                                speaker: speaker.clone(),
                                voiceprint,
                                utterance,
                                is_loopback: true,
                                has_live_transducer: true,
                                transducer_had_text: had_text,
                                transducer_text: live_text.clone(),
                            };

                            if asr_tx.try_send(task_to_send).is_ok() {
                                transcribing_audio.fetch_add(1, Ordering::Relaxed);
                            } else {
                                log::warn!(
                                    "[Pipeline] Whisper queue full (32 pending segments), committing Loopback Transducer text as final fallback"
                                );
                                if !live_text.trim().is_empty() {
                                    let _ = live_event_tx.send(TranscriptEvent {
                                        timestamp_ms: turn_start,
                                        speaker,
                                        text: live_text,
                                        is_final: true,
                                        confidence: 0.85,
                                        replaces_since_ms: None,
                                    });
                                }
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
                    let live_text = mic_transducer.get_partial_result();
                    mic_transducer.reset();
                    let turn_start = if mic_in_turn { mic_turn_start_ms } else { now_ms };

                    let voiceprint = compute_voiceprint_features(&utterance);
                    transcribing_audio.fetch_add(1, Ordering::Relaxed);
                    let _ = asr_tx.send(AsrTask {
                        turn_start_ms: turn_start,
                        timestamp_ms: now_ms,
                        speaker: SpeakerId::LocalUser,
                        voiceprint,
                        utterance,
                        is_loopback: false,
                        has_live_transducer: true,
                        transducer_had_text: mic_had_text || !live_text.trim().is_empty(),
                        transducer_text: live_text,
                    });
                }
                if let Some(utterance) = loopback_segmenter.flush() {
                    let live_text = lb_transducer.get_partial_result();
                    lb_transducer.reset();
                    let turn_start = if lb_in_turn { lb_turn_start_ms } else { now_ms };

                    let voiceprint = compute_voiceprint_features(&utterance);
                    let speaker = clusterer.identify_or_register(&voiceprint);
                    transcribing_audio.fetch_add(1, Ordering::Relaxed);
                    let _ = asr_tx.send(AsrTask {
                        turn_start_ms: turn_start,
                        timestamp_ms: now_ms,
                        speaker,
                        voiceprint,
                        utterance,
                        is_loopback: true,
                        has_live_transducer: true,
                        transducer_had_text: lb_had_text || !live_text.trim().is_empty(),
                        transducer_text: live_text,
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

/// Compute a duration-invariant 64-dimensional spectral & pitch (F0 + Formant) voiceprint vector
/// for speaker identification at 16,000 Hz (< 0.3ms CPU overhead).
/// - Bins 0..24: Pitch autocorrelation profile across 24 fundamental frequency lags (80 Hz .. 320 Hz)
/// - Bins 24..56: 32-band Goertzel spectral resonator profile across vocal tract formants (100 Hz .. 3800 Hz)
/// - Bins 56..64: Multi-order differential spectral slope & zero-crossing statistics across voiced frames
fn compute_voiceprint_features(samples: &[f32]) -> Vec<f32> {
    const NUM_BINS: usize = 64;
    let mut bins = vec![0.0f32; NUM_BINS];
    if samples.len() < 320 {
        return bins;
    }

    // Collect only active voiced 20ms frames (320 samples @ 16kHz) so silence gaps never dilute the voiceprint
    let frame_len = 320usize;
    let peak_rms = samples
        .chunks(frame_len)
        .map(|f| (f.iter().map(|&x| x * x).sum::<f32>() / f.len() as f32).sqrt())
        .fold(0.0f32, f32::max);
    let voice_gate = (peak_rms * 0.25).max(0.005);

    let mut voiced_frames: Vec<&[f32]> = Vec::new();
    for frame in samples.chunks_exact(frame_len) {
        let rms = (frame.iter().map(|&x| x * x).sum::<f32>() / frame_len as f32).sqrt();
        if rms >= voice_gate {
            voiced_frames.push(frame);
            if voiced_frames.len() >= 48 {
                break;
            }
        }
    }
    if voiced_frames.is_empty() {
        voiced_frames.push(&samples[..frame_len]);
    }

    let inv_frames = 1.0 / voiced_frames.len() as f32;

    // 1. Bins 0..24: Fundamental Pitch (F0) Autocorrelation lags from 50 (320 Hz) to 188 (85 Hz)
    for (bin_idx, lag_idx) in (0..24).enumerate() {
        let lag = 50 + lag_idx * 6; // 50..=188 samples at 16kHz
        let mut r_acc = 0.0f32;
        for frame in &voiced_frames {
            let mut num = 0.0f32;
            let mut den = 1e-9f32;
            for i in 0..(frame_len - lag) {
                num += frame[i] * frame[i + lag];
                den += frame[i] * frame[i];
            }
            r_acc += (num / den).max(0.0);
        }
        bins[bin_idx] = r_acc * inv_frames;
    }

    // 2. Bins 24..56: 32-band Goertzel Formant Filterbank (log-spaced 110 Hz to 3,700 Hz)
    for band in 0..32 {
        let frac = band as f32 / 31.0;
        let freq_hz = 110.0 * (3700.0f32 / 110.0).powf(frac);
        let omega = 2.0 * std::f32::consts::PI * freq_hz / 16_000.0;
        let coeff = 2.0 * omega.cos();
        let mut band_energy = 0.0f32;

        for frame in &voiced_frames {
            let mut s1 = 0.0f32;
            let mut s2 = 0.0f32;
            for &x in *frame {
                let s0 = x + coeff * s1 - s2;
                s2 = s1;
                s1 = s0;
            }
            let power = (s1 * s1 + s2 * s2 - coeff * s1 * s2).max(0.0);
            band_energy += (power / frame_len as f32).sqrt();
        }
        bins[24 + band] = (band_energy * inv_frames).ln_1p();
    }

    // 3. Bins 56..64: Zero-crossing rate & multi-scale high-frequency consonant/glottal slope
    for frame in &voiced_frames {
        let mut zcr = 0.0f32;
        let mut d1 = 0.0f32;
        let mut d2 = 0.0f32;
        let mut d4 = 0.0f32;
        let mut e0 = 1e-9f32;
        for i in 4..frame_len {
            if (frame[i] >= 0.0) != (frame[i - 1] >= 0.0) {
                zcr += 1.0;
            }
            e0 += frame[i] * frame[i];
            let diff1 = frame[i] - frame[i - 1];
            let diff2 = frame[i] - frame[i - 2];
            let diff4 = frame[i] - frame[i - 4];
            d1 += diff1 * diff1;
            d2 += diff2 * diff2;
            d4 += diff4 * diff4;
        }
        bins[56] += (zcr / frame_len as f32) * inv_frames;
        bins[57] += (d1 / e0).sqrt() * inv_frames;
        bins[58] += (d2 / e0).sqrt() * inv_frames;
        bins[59] += (d4 / e0).sqrt() * inv_frames;
    }
    // Spectral formant ratios (F1/F0, F2/F1, F3/F2, High/Low tilt)
    let low_sum: f32 = bins[24..34].iter().sum::<f32>() + 1e-6;
    let mid_low: f32 = bins[34..42].iter().sum::<f32>() + 1e-6;
    let mid_high: f32 = bins[42..50].iter().sum::<f32>() + 1e-6;
    let high_sum: f32 = bins[50..56].iter().sum::<f32>() + 1e-6;
    bins[60] = mid_low / low_sum;
    bins[61] = mid_high / mid_low;
    bins[62] = high_sum / mid_high;
    bins[63] = (mid_low + mid_high) / (low_sum + high_sum);

    crate::diarization::normalize_vector(&bins)
}
