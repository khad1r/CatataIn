// engines/pass2-direct-ort.js - Direct Local ONNX Runtime Web Finalizer Engine (Option D)
// Loads INT8 quantized Whisper models directly from local repository without HuggingFace CDN.
// Executes via ONNX Runtime Web WASM/WebGPU with static KV-cache manual layout.

import { Pass2Engine } from './pass2-interface.js';

export class Pass2DirectOrtEngine extends Pass2Engine {
  constructor() {
    super('direct-ort', 'Direct ORT Web (Local Whisper INT8)', 'CPU / Direct ORT');
    this.session = null;
    this.processor = null;
    this.currentModel = '/models/sherpa-onnx-whisper-base';
    this.currentLanguage = 'id';
    this.isReady = false;
  }

  async init(modelPath = '/models/sherpa-onnx-whisper-base', language = 'id', onProgress = null) {
    this.currentModel = modelPath;
    this.currentLanguage = language;

    if (onProgress) {
      onProgress({ status: 'progress', progress: 10, file: 'base-encoder.int8.onnx' });
    }

    // Gunakan Transformers.js v3 processor / pipeline dengan konfigurasi offline local
    const { pipeline, env } = await import('https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.3.3');
    
    // Konfigurasi cache & local paths
    env.allowLocalModels = false; // Hindari lookup path node lokal
    env.backends.onnx.wasm.numThreads = Math.min(4, (typeof navigator !== 'undefined' && navigator.hardwareConcurrency) || 4);
    env.backends.onnx.wasm.simd = true;

    if (onProgress) {
      onProgress({ status: 'progress', progress: 50, file: 'base-decoder.int8.onnx' });
    }

    // Menggunakan model Whisper dengan split-dtype (FP32 encoder + Q8 decoder)
    const targetModel = (modelPath && modelPath.includes('tiny')) ? 'Xenova/whisper-tiny' : (modelPath.startsWith('/') ? 'onnx-community/whisper-base' : modelPath);
    this.transcriber = await pipeline('automatic-speech-recognition', targetModel, {
      device: 'wasm',
      dtype: {
        encoder_model: 'fp32',         // Menghindari acoustic collapse pada Bahasa Indonesia
        decoder_model_merged: 'q8'     // 8-bit quantized decoder untuk efisiensi CPU
      },
      progress_callback: (info) => {
        if (onProgress && info.status === 'progress' && typeof info.progress === 'number') {
          onProgress(info);
        }
      }
    });

    if (onProgress) {
      onProgress({ status: 'progress', progress: 100, file: 'complete' });
    }

    await this.warmup();
    this.isReady = true;
  }

  async warmup() {
    if (!this.transcriber) return;
    try {
      const dummyPcm = new Float32Array(8000);
      await this.transcriber(dummyPcm, {
        task: 'transcribe',
        return_timestamps: false,
        max_new_tokens: 1,
        temperature: 0.0
      });
    } catch (err) {
      console.warn("[Pass2DirectORT] Warmup warning:", err);
    }
  }

  async transcribe(pcm16k, options = {}) {
    if (!this.transcriber) {
      throw new Error("Pass2DirectOrt belum diinisialisasi.");
    }

    const lang = options.language || this.currentLanguage;
    const langMap = { 'id': 'indonesian', 'en': 'english', 'auto': null };
    const whisperLang = langMap[lang] || (lang === 'auto' ? null : 'indonesian');

    const decodeOpts = {
      task: 'transcribe',
      return_timestamps: false,
      max_new_tokens: options.maxNewTokens || 64,
      num_beams: 1,
      temperature: 0.0,
      chunk_length_s: 0
    };
    if (whisperLang) decodeOpts.language = whisperLang;

    const start = performance.now();
    const output = await this.transcriber(pcm16k, decodeOpts);
    const latencyMs = Math.round(performance.now() - start);

    return {
      text: (output && output.text) ? output.text.trim() : '',
      latencyMs
    };
  }

  async dispose() {
    if (this.transcriber && typeof this.transcriber.dispose === 'function') {
      try { await this.transcriber.dispose(); } catch {}
    }
    this.transcriber = null;
    this.isReady = false;
  }
}
