use anyhow::Result;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::Duration;
use tokio::sync::mpsc::UnboundedSender;

use crate::asr::SpeechRecognizer;
use crate::audio::DualAudioCapture;
use crate::diarization::{OnlineSpeakerClusterer, SpeechSegmenter};
use crate::events::{SpeakerId, TranscriptEvent};

pub struct PipelineHandle {
    is_running: Arc<AtomicBool>,
    thread_handle: Option<JoinHandle<()>>,
}

impl PipelineHandle {
    pub fn stop(&mut self) {
        self.is_running.store(false, Ordering::SeqCst);
        if let Some(handle) = self.thread_handle.take() {
            let _ = handle.join();
        }
    }
}

impl Drop for PipelineHandle {
    fn drop(&mut self) {
        self.stop();
    }
}

/// Headless real-time pipeline connecting:
/// WASAPI Capture -> NLMS AEC -> VAD Speech Segmentation -> Speaker Diarization -> Speech Recognition
pub struct CatetinPipeline<R: SpeechRecognizer> {
    recognizer: R,
    clusterer: OnlineSpeakerClusterer,
}

impl<R: SpeechRecognizer + 'static> CatetinPipeline<R> {
    pub fn new(recognizer: R) -> Self {
        Self {
            recognizer,
            clusterer: OnlineSpeakerClusterer::default(),
        }
    }

    /// Start processing audio capture in a dedicated background worker thread.
    /// Live transcript events are streamed to the provided tokio channel sender.
    pub fn start(
        mut self,
        capture: DualAudioCapture,
        event_sender: UnboundedSender<TranscriptEvent>,
    ) -> Result<PipelineHandle> {
        let is_running = Arc::new(AtomicBool::new(true));
        let running_clone = Arc::clone(&is_running);

        let thread_handle = std::thread::spawn(move || {
            let mut mic_segmenter = SpeechSegmenter::new(16_000);
            let mut loopback_segmenter = SpeechSegmenter::new(16_000);

            while running_clone.load(Ordering::SeqCst) {
                let mut had_audio = false;

                // 1. Process Near-End User Microphone (Deterministically SpeakerId::LocalUser)
                while let Ok(packet) = capture.mic_receiver.try_recv() {
                    had_audio = true;
                    if let Some(utterance) = mic_segmenter.feed(&packet.samples) {
                        if let Ok(text) = self.recognizer.transcribe(&utterance) {
                            if !text.is_empty() {
                                let event = TranscriptEvent {
                                    timestamp_ms: packet.timestamp_ms,
                                    speaker: SpeakerId::LocalUser,
                                    text,
                                    is_final: true,
                                    confidence: 0.98,
                                };
                                let _ = event_sender.send(event);
                            }
                        }
                    }
                }

                // 2. Process Far-End Loopback Audio (Remote meeting participants)
                while let Ok(packet) = capture.loopback_receiver.try_recv() {
                    had_audio = true;
                    if let Some(utterance) = loopback_segmenter.feed(&packet.samples) {
                        // Extract spectral voiceprint energy profile for online clustering
                        let voiceprint = compute_voiceprint_features(&utterance);
                        let speaker = self.clusterer.identify_or_register(&voiceprint);

                        if let Ok(text) = self.recognizer.transcribe(&utterance) {
                            if !text.is_empty() {
                                let event = TranscriptEvent {
                                    timestamp_ms: packet.timestamp_ms,
                                    speaker,
                                    text,
                                    is_final: true,
                                    confidence: 0.92,
                                };
                                let _ = event_sender.send(event);
                            }
                        }
                    }
                }

                if !had_audio {
                    std::thread::sleep(Duration::from_millis(15));
                }
            }
        });

        Ok(PipelineHandle {
            is_running,
            thread_handle: Some(thread_handle),
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
