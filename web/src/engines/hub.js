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
        return new Pass2WebGpuEngine();
      default:
        throw new Error(`Engine Pass 2 tidak dikenal: ${engineId}`);
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
        available: false,
        isReady: false
      };
    }
    return {
      id: this.activeEngine.id,
      name: this.activeEngine.name,
      deviceType: this.activeEngine.deviceType,
      available: this.activeEngine ? this.activeEngine.available : false,
      isReady: this.activeEngine.isReady
    };
  }

  async runModelTestInWorker(engineId, modelName, onProgress = null) {
    if (typeof Worker === 'undefined' || engineId === 'direct-ort') {
      try {
        const engine = this.createEngineInstance(engineId);
        return await engine.testHardware(modelName, onProgress);
      } catch (err) {
        return {
          engineId,
          model: modelName,
          supported: false,
          initMs: 0,
          decodeMs: 0,
          rtf: Infinity,
          error: err.message || String(err)
        };
      }
    }

    return new Promise((resolve) => {
      let worker = null;
      let settled = false;

      const terminate = () => {
        if (worker) {
          try {
            worker.terminate();
          } catch (_) {}
          worker = null;
        }
      };

      const finish = (result) => {
        if (settled) return;
        settled = true;
        terminate();
        resolve(result);
      };

      try {
        worker = new Worker(new URL('../workers/readiness.worker.js', import.meta.url), { type: 'module' });
      } catch (err) {
        try {
          const engine = this.createEngineInstance(engineId);
          resolve(engine.testHardware(modelName, onProgress));
        } catch (fallbackErr) {
          resolve({
            engineId,
            model: modelName,
            supported: false,
            initMs: 0,
            decodeMs: 0,
            rtf: Infinity,
            error: fallbackErr.message || String(fallbackErr)
          });
        }
        return;
      }

      worker.onmessage = (e) => {
        const data = e.data || {};
        if (data.type === 'PROGRESS') {
          if (typeof onProgress === 'function') {
            try {
              onProgress(data.message);
            } catch (_) {}
          }
        } else if (data.type === 'RESULT') {
          finish(data.result);
        } else if (data.type === 'ERROR') {
          finish({
            engineId,
            model: modelName,
            supported: false,
            initMs: 0,
            decodeMs: 0,
            rtf: Infinity,
            error: data.error || 'Worker test error'
          });
        }
      };

      worker.onerror = (err) => {
        finish({
          engineId,
          model: modelName,
          supported: false,
          initMs: 0,
          decodeMs: 0,
          rtf: Infinity,
          error: (err && err.message) || 'Worker thread execution error'
        });
      };

      worker.postMessage({ type: 'TEST_MODEL', engineId, modelName });
    });
  }

  async runReadinessTests(onProgress = null) {
    const candidateEngines = ['webgpu', 'wasm-simd', 'direct-ort'];
    const results = [];

    const notify = (stepInfo) => {
      if (typeof onProgress === 'function') {
        try {
          onProgress(stepInfo);
        } catch (_) {}
      }
    };

    for (const engineId of candidateEngines) {
      const probe = this.createEngineInstance(engineId);
      if (!probe.available) continue;

      if (engineId === 'webgpu' && (typeof navigator !== 'undefined' && !navigator.gpu)) {
        results.push({
          engineId,
          engineName: probe.name,
          deviceType: probe.deviceType,
          model: 'Xenova/whisper-tiny',
          supported: false,
          initMs: 0,
          decodeMs: 0,
          rtf: Infinity,
          error: 'WebGPU tidak didukung di peramban ini'
        });
        continue;
      }

      if (engineId === 'direct-ort') {
        notify({ engineId, engineName: probe.name, model: 'Xenova/whisper-base' });
        const baseRes = await this.runModelTestInWorker(engineId, 'Xenova/whisper-base', (msg) => {
          notify({ engineId, engineName: probe.name, model: 'Xenova/whisper-base', progress: msg });
        });
        if (!baseRes.engineName) baseRes.engineName = probe.name;
        if (!baseRes.deviceType) baseRes.deviceType = probe.deviceType;
        results.push(baseRes);
        continue;
      }

      // 1. Uji Whisper Tiny
      notify({ engineId, engineName: probe.name, model: 'Xenova/whisper-tiny' });
      const tinyRes = await this.runModelTestInWorker(engineId, 'Xenova/whisper-tiny', (msg) => {
        notify({ engineId, engineName: probe.name, model: 'Xenova/whisper-tiny', progress: msg });
      });
      if (!tinyRes.engineName) tinyRes.engineName = probe.name;
      if (!tinyRes.deviceType) tinyRes.deviceType = probe.deviceType;
      results.push(tinyRes);

      // Gating: tiny < 0.8 -> lanjut base
      if (!tinyRes.supported || tinyRes.rtf >= 0.8) continue;

      // 2. Uji Whisper Base
      notify({ engineId, engineName: probe.name, model: 'Xenova/whisper-base' });
      const baseRes = await this.runModelTestInWorker(engineId, 'Xenova/whisper-base', (msg) => {
        notify({ engineId, engineName: probe.name, model: 'Xenova/whisper-base', progress: msg });
      });
      if (!baseRes.engineName) baseRes.engineName = probe.name;
      if (!baseRes.deviceType) baseRes.deviceType = probe.deviceType;
      results.push(baseRes);

      // Gating: base < 0.5 -> lanjut small (khusus non-wasm-simd karena wasm-simd rawan Acoustic Collapse & unduhan 460 MB)
      if (engineId !== 'wasm-simd' && baseRes.supported && baseRes.rtf < 0.5) {
        notify({ engineId, engineName: probe.name, model: 'Xenova/whisper-small' });
        const smallRes = await this.runModelTestInWorker(engineId, 'Xenova/whisper-small', (msg) => {
          notify({ engineId, engineName: probe.name, model: 'Xenova/whisper-small', progress: msg });
        });
        if (!smallRes.engineName) smallRes.engineName = probe.name;
        if (!smallRes.deviceType) smallRes.deviceType = probe.deviceType;
        results.push(smallRes);
      }
    }

    return {
      results,
      recommendation: this.computeRecommendation(results)
    };
  }

  computeRecommendation(results) {
    const valid = results.filter(r => r && r.supported && Number.isFinite(r.rtf));
    if (valid.length === 0) {
      return {
        pass2Engine: 'wasm-simd',
        whisperModel: 'Xenova/whisper-tiny',
        rtf: null,
        supported: false,
        slowHardwareWarning: true,
        warning: 'Tidak ada kandidat engine yang lolos uji kesiapan hardware.',
        reason: 'Tidak ada kandidat yang lolos uji; menggunakan konfigurasi minimal.'
      };
    }

    const modelRank = {
      'Xenova/whisper-small': 3,
      'Xenova/whisper-base': 2,
      'Xenova/whisper-tiny': 1
    };
    const enginePriority = {
      'webgpu': 2,
      'wasm-simd': 1,
      'direct-ort': 1
    };

    const passing = valid.filter(r => r.rtf < 0.8);
    if (passing.length === 0) {
      // Skenario D: Semua gagal atau RTF >= 0.8x
      const tinyCandidates = valid.filter(r => r.model === 'Xenova/whisper-tiny');
      const pool = tinyCandidates.length > 0 ? tinyCandidates : valid;
      pool.sort((a, b) => a.rtf - b.rtf);
      const fastest = pool[0];
      return {
        pass2Engine: fastest.engineId,
        whisperModel: fastest.model || 'Xenova/whisper-tiny',
        rtf: fastest.rtf,
        supported: false,
        slowHardwareWarning: true,
        warning: `Hardware terlalu lambat untuk inferensi real-time (RTF ${fastest.rtf}x >= 0.8x).`,
        reason: 'Hardware terlalu lambat untuk inferensi real-time.'
      };
    }

    let candidates = passing;
    const hasWebGpuPassing = passing.some(r => r.engineId === 'webgpu');

    // Jika ada kandidat WebGPU yang qualified, prioritaskan WebGPU di atas CPU (wasm-simd / direct-ort)
    if (hasWebGpuPassing) {
      candidates = candidates.filter(r => r.engineId === 'webgpu');
    }

    candidates.sort((a, b) => {
      const prioDiff = (enginePriority[b.engineId] || 0) - (enginePriority[a.engineId] || 0);
      if (prioDiff !== 0) return prioDiff;

      const rankDiff = (modelRank[b.model] || 0) - (modelRank[a.model] || 0);
      if (rankDiff !== 0) return rankDiff;

      return a.rtf - b.rtf;
    });

    const best = candidates[0];
    return {
      pass2Engine: best.engineId,
      whisperModel: best.model,
      rtf: best.rtf,
      supported: true,
      slowHardwareWarning: false
    };
  }
}
