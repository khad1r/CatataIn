// onnx-runner.js - ONNX Runtime Web Manager (WebGPU / WASM Execution)

export class OnnxRunner {
  constructor(storage) {
    this.storage = storage;
    this.ort = null;
    this.zipformerSession = null;
    this.whisperSession = null;
    this.hasWebGpu = typeof navigator !== 'undefined' && !!navigator.gpu;
    this.isReady = false;
  }

  async init() {
    // Dynamically load onnxruntime-web if available globally or import
    if (typeof window !== 'undefined' && window.ort) {
      this.ort = window.ort;
    }

    if (this.ort) {
      this.ort.env.wasm.numThreads = Math.min(4, navigator.hardwareConcurrency || 2);
      this.ort.env.wasm.simd = true;
    }
  }

  getExecutionProvider() {
    return this.hasWebGpu ? 'webgpu' : 'wasm';
  }

  async loadPass1Zipformer(encoderBytes, decoderBytes, joinerBytes) {
    if (!this.ort) return false;
    try {
      const ep = this.getExecutionProvider();
      this.zipformerSession = {
        encoder: await this.ort.InferenceSession.create(encoderBytes, { executionProviders: [ep] }),
        decoder: await this.ort.InferenceSession.create(decoderBytes, { executionProviders: [ep] }),
        joiner: await this.ort.InferenceSession.create(joinerBytes, { executionProviders: [ep] }),
      };
      this.isReady = true;
      return true;
    } catch (err) {
      console.warn("Pass 1 Zipformer WebGPU load failed, falling back to CPU:", err);
      return false;
    }
  }

  async loadPass2Whisper(encoderBytes, decoderBytes) {
    if (!this.ort) return false;
    try {
      const ep = this.getExecutionProvider();
      this.whisperSession = {
        encoder: await this.ort.InferenceSession.create(encoderBytes, { executionProviders: [ep] }),
        decoder: await this.ort.InferenceSession.create(decoderBytes, { executionProviders: [ep] }),
      };
      return true;
    } catch (err) {
      console.warn("Pass 2 Whisper WebGPU load failed:", err);
      return false;
    }
  }

  async transcribeLiveChunk(pcm16kChunk) {
    if (!this.zipformerSession || !this.ort) {
      return null;
    }
    // Zipformer Pass 1 streaming inference
    try {
      const tensor = new this.ort.Tensor('float32', pcm16kChunk, [1, pcm16kChunk.length]);
      const encOut = await this.zipformerSession.encoder.run({ x: tensor });
      return encOut;
    } catch (err) {
      return null;
    }
  }

  async transcribeUtterance(pcm16kUtterance) {
    if (!this.whisperSession || !this.ort) {
      return null;
    }
    // Whisper Pass 2 offline utterance transcription
    try {
      const tensor = new this.ort.Tensor('float32', pcm16kUtterance, [1, pcm16kUtterance.length]);
      const encOut = await this.whisperSession.encoder.run({ audio: tensor });
      return encOut;
    } catch (err) {
      return null;
    }
  }
}
