// engines/pass2-webgpu.js - WebGPU Accelerated Finalizer Engine (Option A)
// Leverages GPU compute shaders for parallel execution with fast WebGPU Mel extraction & PSO pre-warming.

import { Pass2Engine, resolveWhisperLanguage, loadLocalTransformers } from './pass2-interface.js';
import { WebGpuMelExtractor } from './mel-spectrogram.js';

export class Pass2WebGpuEngine extends Pass2Engine {
  constructor() {
    super('webgpu', 'WebGPU (GPU Compute)', 'GPU');
    this.transcriber = null;
    this.currentModel = null;
    this.currentLanguage = 'id';
    this.melExtractor = null;
    this.TensorClass = null;
  }

  async init(modelName = 'Xenova/whisper-tiny', language = 'id', onProgress = null) {
    if (typeof navigator === 'undefined' || !navigator.gpu) {
      throw new Error("WebGPU tidak didukung atau belum diaktifkan pada peramban ini.");
    }

    this.currentModel = modelName;
    this.currentLanguage = language;

    const { pipeline, Tensor } = await loadLocalTransformers();
    this.TensorClass = Tensor;

    this.transcriber = await pipeline('automatic-speech-recognition', modelName, {
      device: 'webgpu',
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

    // Inisialisasi High-Performance WebGPU Mel Spectrogram Extractor
    // Bypasses the slow CPU-bound processor(pcm) loop
    const feFilters = this.transcriber?.processor?.feature_extractor?.config?.mel_filters;
    this.melExtractor = new WebGpuMelExtractor();
    await this.melExtractor.init(null, feFilters);

    // Jalankan pre-warming segera setelah model termuat
    await this.warmup();
    this.isReady = true;
  }

  async warmup() {
    if (!this.transcriber || !this.melExtractor || !this.TensorClass || !this.transcriber.model) {
      throw new Error('Pass2WebGPU belum terinisialisasi lengkap untuk warmup.');
    }

    // Dummy pass 0.5s audio nol untuk memaksa kompilasi Pipeline State Objects (PSO) di driver
    const dummyPcm = new Float32Array(8000);
    await this.melExtractor.extract(dummyPcm);

    const dummyFeatures = new this.TensorClass('float32', new Float32Array(80 * 3000), [1, 80, 3000]);
    await this.transcriber.model.generate({
      input_features: dummyFeatures,
      max_new_tokens: 1,
      temperature: 0.0,
      task: 'transcribe'
    });
  }

  async transcribe(pcm16k, options = {}) {
    if (!this.transcriber) {
      throw new Error("Pass2WebGPU belum diinisialisasi.");
    }

    if (!this.melExtractor || !this.TensorClass || !this.transcriber.model) {
      throw new Error('Pass2WebGPU: komponen WebGPU Mel/Model tidak siap.');
    }

    const whisperLang = resolveWhisperLanguage(options.language || this.currentLanguage);

    const start = performance.now();

    // FAST PATH: WebGPU Mel Extraction (< 25 ms) bypassing slow processor(pcm) (~1.121 ms)
    const t0Mel = performance.now();
    const { data } = await this.melExtractor.extract(pcm16k);
    const melLatencyMs = Math.round(performance.now() - t0Mel);

    const inputFeaturesTensor = new this.TensorClass('float32', data, [1, 80, 3000]);

    const genOpts = {
      input_features: inputFeaturesTensor,
      max_new_tokens: options.maxNewTokens || 64,
      num_beams: 1,
      temperature: 0.0,
      task: 'transcribe'
    };
    if (whisperLang) genOpts.language = whisperLang;

    const t0Model = performance.now();
    const outputTokens = await this.transcriber.model.generate(genOpts);
    const modelLatencyMs = Math.round(performance.now() - t0Model);

    const decoded = this.transcriber.tokenizer.batch_decode(outputTokens, { skip_special_tokens: true });
    const text = (decoded && decoded[0]) ? decoded[0].trim() : '';

    const latencyMs = Math.round(performance.now() - start);

    return {
      text,
      latencyMs,
      melLatencyMs,
      modelLatencyMs
    };
  }

  async dispose() {
    if (this.melExtractor) {
      this.melExtractor.dispose();
      this.melExtractor = null;
    }
    if (this.transcriber) {
      try {
        if (typeof this.transcriber.dispose === 'function') {
          await this.transcriber.dispose();
        } else if (this.transcriber.model && typeof this.transcriber.model.dispose === 'function') {
          await this.transcriber.model.dispose();
        }
      } catch (err) {
        console.error('[Pass2WebGPU] dispose error:', err);
      }
    }
    this.transcriber = null;
    this.TensorClass = null;
    this.isReady = false;
  }
}
