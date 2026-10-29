// engines/pass2-wasm-simd.js - Multi-Threaded WASM SIMD Finalizer Engine (Option C)
// Executes directly on CPU logical cores with zero driver IPC overhead and zero shader compilation.

import { Pass2Engine, resolveWhisperLanguage, loadLocalTransformers } from './pass2-interface.js';

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

    const { pipeline, env } = await loadLocalTransformers();

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
    if (!this.transcriber) {
      throw new Error('Pass2WasmSimd belum diinisialisasi.');
    }
    const dummyPcm = new Float32Array(8000);
    await this.transcriber(dummyPcm, {
      task: 'transcribe',
      return_timestamps: false,
      max_new_tokens: 1,
      temperature: 0.0
    });
  }

  async transcribe(pcm16k, options = {}) {
    if (!this.transcriber) {
      throw new Error("Pass2WasmSimd belum diinisialisasi.");
    }

    const whisperLang = resolveWhisperLanguage(options.language || this.currentLanguage);

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
    if (this.transcriber) {
      try {
        if (typeof this.transcriber.dispose === 'function') {
          await this.transcriber.dispose();
        } else if (this.transcriber.model && typeof this.transcriber.model.dispose === 'function') {
          await this.transcriber.model.dispose();
        }
      } catch (err) {
        console.warn('[WASM-SIMD] Gagal melepaskan pipeline:', err);
      }
    }
    this.transcriber = null;
    this.isReady = false;
  }
}
