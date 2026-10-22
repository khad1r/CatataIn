// engines/hub.js - Pass 2 Engine Coordinator & Factory
// Provides unified engine switching with zero-refactoring architecture for the main application.

import { Pass2WebGpuEngine } from './pass2-webgpu.js';
import { Pass2WasmSimdEngine } from './pass2-wasm-simd.js';
import { Pass2WebNnEngine } from './pass2-webnn.js';
import { Pass2DirectOrtEngine } from './pass2-direct-ort.js';
import { Pass2WasmGgmlEngine } from './pass2-wasm-ggml.js';

export class Pass2Hub {
  constructor() {
    this.activeEngine = null;
    this.activeEngineId = 'webgpu';
    this.currentModel = 'Xenova/whisper-tiny';
    this.currentLanguage = 'id';
  }

  createEngineInstance(engineId) {
    switch (engineId) {
      case 'wasm-simd':
        return new Pass2WasmSimdEngine();
      case 'webnn':
        return new Pass2WebNnEngine();
      case 'direct-ort':
        return new Pass2DirectOrtEngine();
      case 'wasm-ggml':
        return new Pass2WasmGgmlEngine();
      case 'webgpu':
      default:
        return new Pass2WebGpuEngine();
    }
  }

  async setEngine(engineId, { model, language }, onProgress = null) {
    const targetModel = model || this.currentModel;
    const targetLang = language || this.currentLanguage;

    // Jika engine dan model sama dan sudah ready, tidak perlu re-init
    if (this.activeEngine &&
        this.activeEngineId === engineId &&
        this.currentModel === targetModel &&
        this.activeEngine.isReady) {
      return this.activeEngine;
    }

    if (this.activeEngine) {
      await this.activeEngine.dispose();
      this.activeEngine = null;
    }

    this.activeEngineId = engineId;
    this.currentModel = targetModel;
    this.currentLanguage = targetLang;

    if (targetModel === 'none') {
      return null;
    }

    const engine = this.createEngineInstance(engineId);
    await engine.init(targetModel, targetLang, onProgress);
    this.activeEngine = engine;

    return this.activeEngine;
  }

  async transcribe(pcm16k, options = {}) {
    if (!this.activeEngine || !this.activeEngine.isReady) {
      throw new Error(`Engine Pass 2 (${this.activeEngineId}) belum siap.`);
    }
    return await this.activeEngine.transcribe(pcm16k, options);
  }

  getActiveEngineInfo() {
    if (!this.activeEngine) {
      return {
        id: this.activeEngineId,
        name: 'None',
        deviceType: 'None',
        isReady: false
      };
    }
    return {
      id: this.activeEngine.id,
      name: this.activeEngine.name,
      deviceType: this.activeEngine.deviceType,
      isReady: this.activeEngine.isReady
    };
  }
}
