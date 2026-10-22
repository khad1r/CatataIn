// engines/pass2-interface.js - Unified Contract for Pass 2 ASR Finalizer Engines
// Defines the strict interface implemented by WebGPU, WASM SIMD, and WebNN backends.

export class Pass2Engine {
  constructor(id, name, deviceType) {
    this.id = id;                     // 'webgpu' | 'wasm-simd' | 'webnn'
    this.name = name;
    this.deviceType = deviceType;     // 'GPU' | 'CPU-SIMD' | 'NPU/OS-DirectML'
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
}
