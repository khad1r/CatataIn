use catetin_core::audio::AudioResampler;
use catetin_core::diarization::vad::{VoiceActivityDetector, SpeechSegmenter};
use catetin_core::diarization::clustering::{OnlineSpeakerClusterer, cosine_similarity};
use catetin_core::events::SpeakerId;
use catetin_core::asr::{is_hallucination, format_utterance};
use catetin_core::asr::transducer::ipa_phonemes_to_indonesian;

// ============================================================================
// 1. Audio Resampler (Pure C-ABI, Zero External Imports)
// ============================================================================

#[no_mangle]
pub extern "C" fn catetin_resampler_new(source_rate: u32, source_channels: u16, target_rate: u32) -> *mut AudioResampler {
    Box::into_raw(Box::new(AudioResampler::new(source_rate, source_channels, target_rate)))
}

#[no_mangle]
pub unsafe extern "C" fn catetin_resampler_free(ptr: *mut AudioResampler) {
    if !ptr.is_null() {
        drop(Box::from_raw(ptr));
    }
}

#[no_mangle]
pub unsafe extern "C" fn catetin_resampler_process(
    ptr: *mut AudioResampler,
    in_ptr: *const f32,
    in_len: usize,
    out_ptr: *mut f32,
    out_cap: usize,
) -> usize {
    if ptr.is_null() || in_ptr.is_null() || out_ptr.is_null() || in_len == 0 {
        return 0;
    }
    let resampler = &mut *ptr;
    let input = std::slice::from_raw_parts(in_ptr, in_len);
    let output = resampler.process_interleaved(input);
    let to_copy = output.len().min(out_cap);
    std::ptr::copy_nonoverlapping(output.as_ptr(), out_ptr, to_copy);
    to_copy
}

// ============================================================================
// 2. Voice Activity Detector (VAD) (Pure C-ABI)
// ============================================================================

#[no_mangle]
pub extern "C" fn catetin_vad_new(sample_rate: u32, frame_size: usize) -> *mut VoiceActivityDetector {
    Box::into_raw(Box::new(VoiceActivityDetector::new(sample_rate, frame_size)))
}

#[no_mangle]
pub unsafe extern "C" fn catetin_vad_free(ptr: *mut VoiceActivityDetector) {
    if !ptr.is_null() {
        drop(Box::from_raw(ptr));
    }
}

#[no_mangle]
pub unsafe extern "C" fn catetin_vad_process_frame(
    ptr: *mut VoiceActivityDetector,
    frame_ptr: *const f32,
    frame_len: usize,
) -> i32 {
    if ptr.is_null() || frame_ptr.is_null() || frame_len == 0 {
        return 0;
    }
    let vad = &mut *ptr;
    let frame = std::slice::from_raw_parts(frame_ptr, frame_len);
    if vad.process_frame(frame) { 1 } else { 0 }
}

#[no_mangle]
pub unsafe extern "C" fn catetin_vad_frame_rms(
    ptr: *const VoiceActivityDetector,
    frame_ptr: *const f32,
    frame_len: usize,
) -> f32 {
    if frame_ptr.is_null() || frame_len == 0 {
        return 0.0;
    }
    let frame = std::slice::from_raw_parts(frame_ptr, frame_len);
    if !ptr.is_null() {
        (&*ptr).frame_rms(frame)
    } else {
        let sum_sq: f32 = frame.iter().map(|&x| x * x).sum();
        (sum_sq / frame.len() as f32).sqrt()
    }
}

// ============================================================================
// 3. Speech Segmenter (Utterance Endpointer) (Pure C-ABI)
// ============================================================================

#[no_mangle]
pub extern "C" fn catetin_segmenter_new(sample_rate: u32) -> *mut SpeechSegmenter {
    Box::into_raw(Box::new(SpeechSegmenter::new(sample_rate)))
}

#[no_mangle]
pub unsafe extern "C" fn catetin_segmenter_free(ptr: *mut SpeechSegmenter) {
    if !ptr.is_null() {
        drop(Box::from_raw(ptr));
    }
}

#[no_mangle]
pub unsafe extern "C" fn catetin_segmenter_feed(
    ptr: *mut SpeechSegmenter,
    samples_ptr: *const f32,
    samples_len: usize,
    out_ptr: *mut f32,
    out_cap: usize,
) -> usize {
    if ptr.is_null() || samples_ptr.is_null() || out_ptr.is_null() || samples_len == 0 {
        return 0;
    }
    let segmenter = &mut *ptr;
    let samples = std::slice::from_raw_parts(samples_ptr, samples_len);
    if let Some(utterance) = segmenter.feed(samples) {
        let to_copy = utterance.len().min(out_cap);
        std::ptr::copy_nonoverlapping(utterance.as_ptr(), out_ptr, to_copy);
        to_copy
    } else {
        0
    }
}

#[no_mangle]
pub unsafe extern "C" fn catetin_segmenter_flush(
    ptr: *mut SpeechSegmenter,
    out_ptr: *mut f32,
    out_cap: usize,
) -> usize {
    if ptr.is_null() || out_ptr.is_null() {
        return 0;
    }
    let segmenter = &mut *ptr;
    if let Some(utterance) = segmenter.flush() {
        let to_copy = utterance.len().min(out_cap);
        std::ptr::copy_nonoverlapping(utterance.as_ptr(), out_ptr, to_copy);
        to_copy
    } else {
        0
    }
}

#[no_mangle]
pub unsafe extern "C" fn catetin_segmenter_is_speaking(ptr: *const SpeechSegmenter) -> i32 {
    if ptr.is_null() {
        return 0;
    }
    if (&*ptr).is_speaking() { 1 } else { 0 }
}

// ============================================================================
// 4. Speaker Diarization & Clustering (Pure C-ABI)
// ============================================================================

#[no_mangle]
pub extern "C" fn catetin_clusterer_new(threshold: f32, momentum: f32) -> *mut OnlineSpeakerClusterer {
    Box::into_raw(Box::new(OnlineSpeakerClusterer::new(threshold, momentum)))
}

#[no_mangle]
pub unsafe extern "C" fn catetin_clusterer_free(ptr: *mut OnlineSpeakerClusterer) {
    if !ptr.is_null() {
        drop(Box::from_raw(ptr));
    }
}

#[no_mangle]
pub unsafe extern "C" fn catetin_clusterer_identify(
    ptr: *mut OnlineSpeakerClusterer,
    emb_ptr: *const f32,
    emb_len: usize,
) -> i32 {
    if ptr.is_null() || emb_ptr.is_null() || emb_len == 0 {
        return -1;
    }
    let clusterer = &mut *ptr;
    let emb = std::slice::from_raw_parts(emb_ptr, emb_len);
    match clusterer.identify_or_register(emb) {
        SpeakerId::LocalUser => 0,
        SpeakerId::Remote(id) => (id + 1) as i32,
        SpeakerId::InPerson(id) => (id + 1) as i32,
        SpeakerId::Unknown => -1,
    }
}

#[no_mangle]
pub unsafe extern "C" fn catetin_clusterer_count(ptr: *const OnlineSpeakerClusterer) -> usize {
    if ptr.is_null() { return 0; }
    (&*ptr).speaker_count()
}

#[no_mangle]
pub unsafe extern "C" fn catetin_clusterer_reset(ptr: *mut OnlineSpeakerClusterer) {
    if !ptr.is_null() {
        (&mut *ptr).reset();
    }
}

#[no_mangle]
pub unsafe extern "C" fn catetin_cosine_similarity(
    a_ptr: *const f32,
    b_ptr: *const f32,
    len: usize,
) -> f32 {
    if a_ptr.is_null() || b_ptr.is_null() || len == 0 {
        return 0.0;
    }
    let a = std::slice::from_raw_parts(a_ptr, len);
    let b = std::slice::from_raw_parts(b_ptr, len);
    cosine_similarity(a, b)
}

// ============================================================================
// 5. Post-Processing: Hallucination Filter & Phoneme Normalizer (Pure C-ABI)
// ============================================================================

#[no_mangle]
pub unsafe extern "C" fn catetin_is_hallucination(in_ptr: *const u8, in_len: usize) -> i32 {
    if in_ptr.is_null() || in_len == 0 {
        return 1;
    }
    let slice = std::slice::from_raw_parts(in_ptr, in_len);
    if let Ok(text) = std::str::from_utf8(slice) {
        if is_hallucination(text) { 1 } else { 0 }
    } else {
        1
    }
}

#[no_mangle]
pub unsafe extern "C" fn catetin_format_utterance(
    in_ptr: *const u8,
    in_len: usize,
    prompt_ptr: *const u8,
    prompt_len: usize,
    out_ptr: *mut u8,
    out_cap: usize,
) -> usize {
    if in_ptr.is_null() || out_ptr.is_null() || in_len == 0 || out_cap == 0 {
        return 0;
    }
    let in_slice = std::slice::from_raw_parts(in_ptr, in_len);
    let text = match std::str::from_utf8(in_slice) {
        Ok(t) => t,
        Err(_) => return 0,
    };
    let prompt = if !prompt_ptr.is_null() && prompt_len > 0 {
        let p_slice = std::slice::from_raw_parts(prompt_ptr, prompt_len);
        std::str::from_utf8(p_slice).unwrap_or("")
    } else {
        ""
    };

    let formatted = format_utterance(text, prompt);
    let bytes = formatted.as_bytes();
    let to_copy = bytes.len().min(out_cap);
    std::ptr::copy_nonoverlapping(bytes.as_ptr(), out_ptr, to_copy);
    to_copy
}

#[no_mangle]
pub unsafe extern "C" fn catetin_ipa_to_indonesian(
    in_ptr: *const u8,
    in_len: usize,
    out_ptr: *mut u8,
    out_cap: usize,
) -> usize {
    if in_ptr.is_null() || out_ptr.is_null() || in_len == 0 || out_cap == 0 {
        return 0;
    }
    let slice = std::slice::from_raw_parts(in_ptr, in_len);
    let text = match std::str::from_utf8(slice) {
        Ok(t) => t,
        Err(_) => return 0,
    };
    let converted = ipa_phonemes_to_indonesian(text);
    let bytes = converted.as_bytes();
    let to_copy = bytes.len().min(out_cap);
    std::ptr::copy_nonoverlapping(bytes.as_ptr(), out_ptr, to_copy);
    to_copy
}

// ============================================================================
// 6. Memory Allocation Helpers for JS-WASM Zero-Copy Interop
// ============================================================================

#[no_mangle]
pub extern "C" fn catetin_alloc_f32(len: usize) -> *mut f32 {
    let mut vec = Vec::<f32>::with_capacity(len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[no_mangle]
pub unsafe extern "C" fn catetin_free_f32(ptr: *mut f32, cap: usize) {
    if !ptr.is_null() && cap > 0 {
        drop(Vec::from_raw_parts(ptr, 0, cap));
    }
}

#[no_mangle]
pub extern "C" fn catetin_alloc_u8(len: usize) -> *mut u8 {
    let mut vec = Vec::<u8>::with_capacity(len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[no_mangle]
pub unsafe extern "C" fn catetin_free_u8(ptr: *mut u8, cap: usize) {
    if !ptr.is_null() && cap > 0 {
        drop(Vec::from_raw_parts(ptr, 0, cap));
    }
}
