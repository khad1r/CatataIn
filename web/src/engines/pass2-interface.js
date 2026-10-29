// engines/pass2-interface.js - Unified Contract for Pass 2 ASR Finalizer Engines
// Defines the strict interface implemented by WebGPU, WASM SIMD, and WebNN backends.

export const READINESS_SCHEMA_VERSION = 'catetin-hw-v1';

export function resolveWhisperLanguage(lang) {
  if (lang === 'id') return 'indonesian';
  if (lang === 'en') return 'english';
  if (lang === 'auto') return null;
  throw new Error(`Bahasa Whisper tidak dikenal: ${lang}`);
}

export async function loadLocalTransformers() {
  const vendorDirUrl = new URL('../../vendor/transformers/', import.meta.url).href;
  const moduleUrl = new URL('transformers.min.js', vendorDirUrl).href;
  const mod = await import(moduleUrl);
  mod.env.allowLocalModels = false;
  if (!mod.env.backends) {
    mod.env.backends = {};
  }
  if (!mod.env.backends.onnx) {
    mod.env.backends.onnx = {};
  }
  if (!mod.env.backends.onnx.wasm) {
    mod.env.backends.onnx.wasm = {};
  }
  mod.env.backends.onnx.wasm.wasmPaths = new URL('../../vendor/transformers/', import.meta.url).href;
  return mod;
}

export class Pass2Engine {
  constructor(id, name, deviceType) {
    this.id = id;                     // 'webgpu' | 'wasm-simd' | 'webnn'
    this.name = name;
    this.deviceType = deviceType;     // 'GPU' | 'CPU-SIMD' | 'NPU/OS-DirectML'
    this.available = true;
    this.isReady = false;
  }

  // Inisialisasi model dan alokasi resource pipeline
  async init(modelName, language = 'id', onProgress = null) {
    throw new Error(`${this.name}: init() not implemented`);
  }

  // Pre-warming wajib: Memaksa kompilasi kernel/shader sebelum pengguna berbicara (0ms first-utterance lag)
  async warmup() {
    throw new Error(`${this.name}: warmup() not implemented`);
  }

  // Transkripsi segmen utuh yang telah di-endpoint oleh VAD
  async transcribe(pcm16k, options = {}) {
    throw new Error(`${this.name}: transcribe() not implemented`);
  }

  // Pembersihan resource memori saat shutdown atau pergantian model
  async dispose() {
    this.isReady = false;
  }

  // Uji kesiapan hardware dengan micro-inference 0.5 detik audio (8.000 sampel @ 16kHz)
  async testHardware(modelName = 'Xenova/whisper-tiny', language = 'id', dummyPcm = null) {
    if (!this.available) {
      return {
        engineId: this.id,
        engineName: this.name,
        deviceType: this.deviceType,
        model: modelName,
        supported: false,
        initMs: 0,
        decodeMs: 0,
        rtf: Infinity,
        error: 'Engine belum tersedia'
      };
    }

    let actualLang = 'id';
    let onProgress = null;
    let pcmInput = dummyPcm;

    if (typeof language === 'function') {
      onProgress = language;
      actualLang = 'id';
    } else if (typeof language === 'string') {
      actualLang = language;
    }

    if (typeof dummyPcm === 'function') {
      onProgress = dummyPcm;
      pcmInput = null;
    }

    const pcm = pcmInput instanceof Float32Array && pcmInput.length > 0
      ? pcmInput
      : new Float32Array(8000);
    const audioDurationS = pcm.length / 16000;

    const tInit0 = performance.now();
    try {
      await this.init(modelName, actualLang, onProgress);
      const initMs = Math.round(performance.now() - tInit0);

      const tDec0 = performance.now();
      await this.transcribe(pcm, { language: actualLang, maxNewTokens: 16 });
      const decodeMs = Math.round(performance.now() - tDec0);
      const rtf = Number((decodeMs / (audioDurationS * 1000)).toFixed(3));

      await this.dispose();
      return {
        engineId: this.id,
        engineName: this.name,
        deviceType: this.deviceType,
        model: modelName,
        supported: true,
        initMs,
        decodeMs,
        rtf,
        error: null
      };
    } catch (err) {
      try { await this.dispose(); } catch (_) {}
      return {
        engineId: this.id,
        engineName: this.name,
        deviceType: this.deviceType,
        model: modelName,
        supported: false,
        initMs: Math.round(performance.now() - tInit0),
        decodeMs: 0,
        rtf: Infinity,
        error: err.message || 'Gagal menguji hardware'
      };
    }
  }
}
