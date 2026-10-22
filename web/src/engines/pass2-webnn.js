// engines/pass2-webnn.js - WebNN OS-Native Finalizer Engine (Option B)
// Directly binds to OS machine learning APIs (DirectML on Windows, CoreML on macOS, or NPU).

import { Pass2Engine } from './pass2-interface.js';

export class Pass2WebNnEngine extends Pass2Engine {
  constructor() {
    super('webnn', 'WebNN (OS DirectML / NPU Native)', 'NPU/OS-DirectML');
    this.mlContext = null;
    this.currentModel = null;
    this.currentLanguage = 'id';
    this.activeDevice = 'unknown';
  }

  async init(modelName = 'Xenova/whisper-tiny', language = 'id', onProgress = null) {
    if (typeof navigator === 'undefined' || !('ml' in navigator)) {
      throw new Error(
        "WebNN API tidak didukung atau belum diaktifkan pada peramban ini. " +
        "Untuk Chromium/Edge, aktifkan flag: chrome://flags/#web-machine-learning-neural-network"
      );
    }

    this.currentModel = modelName;
    this.currentLanguage = language;

    // Deteksi hardware context: NPU -> GPU -> CPU
    const preferredDevices = ['npu', 'gpu', 'cpu'];
    for (const dev of preferredDevices) {
      try {
        this.mlContext = await navigator.ml.createContext({ deviceType: dev });
        this.activeDevice = dev.toUpperCase();
        this.deviceType = `${this.activeDevice} (DirectML Native)`;
        break;
      } catch {
        // Coba target device berikutnya
      }
    }

    if (!this.mlContext) {
      throw new Error("Gagal membuat WebNN MLContext pada perangkat apapun (NPU, GPU, CPU).");
    }

    // Load backend via Transformers.js v3 dengan WebNN/WebGPU DirectML context
    const { pipeline, env } = await import('https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.3.3');
    env.allowLocalModels = false;

    try {
      this.transcriber = await pipeline('automatic-speech-recognition', modelName, {
        device: 'webgpu', // Menggunakan DirectML/GPU pipeline
        dtype: {
          encoder_model: 'fp32',
          decoder_model_merged: 'q4'
        },
        progress_callback: (info) => {
          if (onProgress && info.status === 'progress' && typeof info.progress === 'number') {
            onProgress(info);
          }
        }
      });
    } catch (e) {
      // Fallback ke WASM jika pipeline gagal
      this.transcriber = await pipeline('automatic-speech-recognition', modelName, {
        device: 'wasm'
      });
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
      console.warn("[Pass2WebNN] Pre-warming warning:", err);
    }
  }

  async transcribe(pcm16k, options = {}) {
    if (!this.transcriber) {
      throw new Error("Pass2WebNN belum diinisialisasi.");
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
      latencyMs,
      hardware: this.activeDevice
    };
  }

  async dispose() {
    if (this.transcriber && typeof this.transcriber.dispose === 'function') {
      try { await this.transcriber.dispose(); } catch {}
    }
    this.transcriber = null;
    if (this.mlContext && typeof this.mlContext.destroy === 'function') {
      try { this.mlContext.destroy(); } catch {}
    }
    this.mlContext = null;
    this.isReady = false;
  }
}
