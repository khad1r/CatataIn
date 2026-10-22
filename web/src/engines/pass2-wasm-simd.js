// engines/pass2-wasm-simd.js - Multi-Threaded WASM SIMD Finalizer Engine (Option C)
// Executes directly on CPU logical cores with zero driver IPC overhead and zero shader compilation.

import { Pass2Engine } from './pass2-interface.js';

export class Pass2WasmSimdEngine extends Pass2Engine {
  constructor() {
    super('wasm-simd', 'Multi-Thread WASM SIMD (CPU)', 'CPU-SIMD');
    this.transcriber = null;
    this.currentModel = null;
    this.currentLanguage = 'id';
    this.threads = 4;
  }

  async init(modelName = 'Xenova/whisper-tiny', language = 'id', onProgress = null) {
    this.currentModel = modelName;
    this.currentLanguage = language;

    const { pipeline, env } = await import('https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.3.3');
    env.allowLocalModels = false;

    this.threads = Math.min(8, (typeof navigator !== 'undefined' && navigator.hardwareConcurrency) || 4);
    env.backends.onnx.wasm.numThreads = this.threads;
    env.backends.onnx.wasm.simd = true;

    this.transcriber = await pipeline('automatic-speech-recognition', modelName, {
      device: 'wasm',
      dtype: {
        encoder_model: 'fp32',         // Presisi FP32 encoder mempertahankan resolusi Mel akustik Bahasa Indonesia
        decoder_model_merged: 'q8'     // Kuantisasi Q8 decoder untuk kecepatan inferensi CPU SIMD
      },
      progress_callback: (info) => {
        if (onProgress && info.status === 'progress' && typeof info.progress === 'number') {
          onProgress(info);
        }
      }
    });

    // Jalankan pre-warming untuk memanaskan memory pool dan worker thread
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
      console.warn("[Pass2WasmSimd] Pre-warming warning:", err);
    }
  }

  async transcribe(pcm16k, options = {}) {
    if (!this.transcriber) {
      throw new Error("Pass2WasmSimd belum diinisialisasi.");
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
