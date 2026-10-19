// onnx-runner.js - In-Browser Speech Recognition Engine (WebGPU / WASM SIMD)
// Ponytail: Pragmatic Transformers.js pipeline (runs real ONNX models locally in browser)

export class OnnxRunner {
  constructor(storage) {
    this.storage = storage;
    this.transcriber = null;
    this.isLoading = false;
    this.isReady = false;
    this.modelName = 'Xenova/whisper-tiny';
    this.device = (typeof navigator !== 'undefined' && !!navigator.gpu) ? 'webgpu' : 'wasm';
  }

  async init(onProgress = null) {
    if (this.isReady || this.isLoading) return;
    this.isLoading = true;

    try {
      console.log(`[OnnxRunner] Loading Transformers.js pipeline on device: ${this.device}...`);
      
      // Dynamic import of Transformers.js from CDN (cached locally in browser)
      const { pipeline, env } = await import('https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2');
      
      // Configure hardware acceleration
      env.allowLocalModels = false;
      env.backends.onnx.wasm.numThreads = Math.min(4, navigator.hardwareConcurrency || 2);
      env.backends.onnx.wasm.simd = true;

      this.transcriber = await pipeline('automatic-speech-recognition', this.modelName, {
        device: this.device,
        progress_callback: (info) => {
          if (onProgress) {
            onProgress(info);
          }
        }
      });

      this.isReady = true;
      this.isLoading = false;
      console.log("[OnnxRunner] In-browser Whisper ASR engine is ready.");
      return true;
    } catch (err) {
      console.warn("[OnnxRunner] Failed to load WebGPU/WASM Whisper pipeline, falling back:", err);
      this.isLoading = false;
      this.isReady = false;
      return false;
    }
  }

  async transcribeUtterance(pcm16k, language = 'indonesian') {
    if (!this.isReady || !this.transcriber) {
      return null;
    }

    try {
      // Whisper expects 16kHz Float32Array in [-1.0, 1.0]
      const output = await this.transcriber(pcm16k, {
        language: language === 'en' ? 'english' : 'indonesian',
        task: 'transcribe',
        chunk_length_s: 30,
        stride_length_s: 5,
        return_timestamps: false,
      });

      if (output && typeof output.text === 'string') {
        return output.text.trim();
      }
      return null;
    } catch (err) {
      console.error("[OnnxRunner] Transcription error:", err);
      return null;
    }
  }
}
