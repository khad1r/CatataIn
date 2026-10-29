// engines/pass2-wasm-ggml.js - whisper.cpp WASM SIMD Finalizer Engine (Option E)
// Loads local Q8_0 GGML Whisper models (ggml-base-id-q8_0.bin / ggml-small-id-q8_0.bin)
// via Emscripten C-ABI (web/vendor/whisper-cpp/whisper.js + whisper.wasm).
// Zero Mock & Zero Silent Fallback compliant.

import { Pass2Engine } from './pass2-interface.js';

const GGML_CACHE_NAME = 'catetin-ggml-models-v1';

const LOCAL_GGML_MODELS = {
  'Xenova/whisper-base': '/models/ggml-base-id-q8_0.bin',
  '/models/ggml-base-id-q8_0.bin': '/models/ggml-base-id-q8_0.bin',
  'Xenova/whisper-small': '/models/ggml-small-id-q8_0.bin',
  '/models/ggml-small-id-q8_0.bin': '/models/ggml-small-id-q8_0.bin'
};

export class Pass2WasmGgmlEngine extends Pass2Engine {
  constructor() {
    super('wasm-ggml', 'whisper.cpp GGML WASM (Local Q8_0)', 'CPU-SIMD (GGML)');
    this.available = false;
    this.wasm = null;
    this.ctx = 0;
    this.modelPtr = 0;
    this.modelPath = null;
    this.language = 'id';
  }

  static resolveModelPath(modelId) {
    if (!modelId) return LOCAL_GGML_MODELS['Xenova/whisper-base'];
    if (LOCAL_GGML_MODELS[modelId]) return LOCAL_GGML_MODELS[modelId];
    if (String(modelId).includes('base')) return '/models/ggml-base-id-q8_0.bin';
    if (String(modelId).includes('small')) return '/models/ggml-small-id-q8_0.bin';
    return null;
  }

  async _loadRuntimeModule() {
    const isNode = typeof process !== 'undefined' && Boolean(process.versions?.node) && typeof window === 'undefined';
    if (isNode) {
      const fs = await import('node:fs');
      const path = await import('node:path');
      const { fileURLToPath, pathToFileURL } = await import('node:url');
      const currentDir = path.dirname(fileURLToPath(import.meta.url));
      const runtimePath = path.resolve(currentDir, '../../vendor/whisper-cpp/whisper.js');
      if (!fs.existsSync(runtimePath)) {
        this.available = false;
        this.isReady = false;
        throw new Error(
          'Engine Pass 2 whisper.cpp GGML WASM belum tersedia: biner web/vendor/whisper-cpp/whisper.wasm belum dikompilasi (jalankan scripts/build-whisper-wasm.ps1).'
        );
      }
      const mod = await import(pathToFileURL(runtimePath).href);
      const factory = mod.default || mod.createWhisperModule;
      return await factory();
    }

    const runtimeUrl = new URL('../../vendor/whisper-cpp/whisper.js', import.meta.url).href;
    let probeResp;
    try {
      probeResp = await fetch(runtimeUrl, { method: 'HEAD' });
    } catch (err) {
      this.available = false;
      this.isReady = false;
      throw new Error(
        `Engine Pass 2 whisper.cpp GGML WASM belum tersedia: gagal memeriksa runtime (${err.message}).`
      );
    }
    if (!probeResp.ok) {
      this.available = false;
      this.isReady = false;
      throw new Error(
        'Engine Pass 2 whisper.cpp GGML WASM belum tersedia: biner web/vendor/whisper-cpp/whisper.wasm belum dikompilasi (jalankan scripts/build-whisper-wasm.ps1).'
      );
    }
    const mod = await import(runtimeUrl);
    const factory = mod.default || mod.createWhisperModule;
    return await factory();
  }

  async _loadModelBytes(resolvedModelPath, onProgress = null) {
    const isNode = typeof process !== 'undefined' && Boolean(process.versions?.node) && typeof window === 'undefined';
    if (isNode) {
      const fs = await import('node:fs');
      const path = await import('node:path');
      const { fileURLToPath } = await import('node:url');
      const currentDir = path.dirname(fileURLToPath(import.meta.url));
      const diskPath = path.resolve(currentDir, '../../../dist', resolvedModelPath.replace(/^\//, ''));
      if (!fs.existsSync(diskPath)) {
        throw new Error(`Berkas model GGML lokal tidak ditemukan di: ${diskPath}`);
      }
      if (onProgress) onProgress({ status: 'loading', file: resolvedModelPath, progress: 50 });
      const buf = fs.readFileSync(diskPath);
      if (onProgress) onProgress({ status: 'ready', file: resolvedModelPath, progress: 100 });
      return new Uint8Array(buf.buffer, buf.byteOffset, buf.byteLength);
    }

    let response = null;
    let cache = null;
    if (typeof caches !== 'undefined') {
      try {
        cache = await caches.open(GGML_CACHE_NAME);
        response = await cache.match(resolvedModelPath);
      } catch (_) {
        cache = null;
      }
    }

    if (!response) {
      const netResp = await fetch(resolvedModelPath);
      if (!netResp.ok) {
        throw new Error(`Gagal mengunduh model GGML ${resolvedModelPath} (HTTP ${netResp.status}).`);
      }
      if (cache) {
        try {
          await cache.put(resolvedModelPath, netResp.clone());
        } catch (_) {
          // Abaikan QuotaExceededError tanpa memutus pemuatan model
        }
      }
      response = netResp;
    }

    const ab = await response.arrayBuffer();
    if (onProgress) onProgress({ status: 'ready', file: resolvedModelPath, progress: 100 });
    return new Uint8Array(ab);
  }

  async init(modelId = 'Xenova/whisper-base', language = 'id', onProgress = null) {
    await this.dispose();
    const resolvedPath = Pass2WasmGgmlEngine.resolveModelPath(modelId);
    if (!resolvedPath) {
      this.isReady = false;
      throw new Error(
        `Engine Pass 2 whisper.cpp GGML WASM tidak menyediakan bobot lokal untuk model "${modelId}" (hanya tersedia ggml-base-id-q8_0.bin dan ggml-small-id-q8_0.bin). Zero Silent Fallback aktif.`
      );
    }

    this.wasm = await this._loadRuntimeModule();
    const modelBytes = await this._loadModelBytes(resolvedPath, onProgress);

    this.modelPtr = this.wasm._malloc(modelBytes.byteLength);
    if (!this.modelPtr) {
      throw new Error('Gagal mengalokasikan memori WASM heap untuk model GGML.');
    }
    this.wasm.HEAPU8.set(modelBytes, this.modelPtr);

    this.ctx = this.wasm._whisper_init_from_buffer(this.modelPtr, modelBytes.byteLength);
    if (!this.ctx) {
      this.wasm._free(this.modelPtr);
      this.modelPtr = 0;
      throw new Error(`Gagal menginisialisasi konteks whisper.cpp dari buffer (${resolvedPath}).`);
    }

    this.modelPath = resolvedPath;
    this.language = language || 'id';
    this.available = true;
    this.isReady = true;
  }

  async warmup() {
    if (!this.isReady || !this.wasm || !this.ctx) {
      throw new Error('Engine Pass 2 whisper.cpp GGML WASM belum tersedia untuk warmup.');
    }
    const silentPcm = new Float32Array(8000);
    await this.transcribe(silentPcm, { language: this.language });
  }

  async transcribe(pcm16k, options = {}) {
    if (!this.isReady || !this.wasm || !this.ctx) {
      throw new Error('Engine Pass 2 whisper.cpp GGML WASM belum tersedia.');
    }
    if (!pcm16k || !(pcm16k instanceof Float32Array) || pcm16k.length === 0) {
      return { text: '', latencyMs: 0, rtf: 0, backend: this.backendName };
    }

    const t0 = performance.now();
    const nSamples = pcm16k.length;
    const pcmPtr = this.wasm._malloc(nSamples * 4);
    if (!pcmPtr) {
      throw new Error('Gagal mengalokasikan buffer audio PCM pada heap WASM whisper.cpp.');
    }

    try {
      this.wasm.HEAPF32.set(pcm16k, pcmPtr >> 2);
      const paramsPtr = this.wasm._whisper_full_default_params(0); // WHISPER_SAMPLING_GREEDY = 0
      const rc = this.wasm._whisper_full(this.ctx, paramsPtr, pcmPtr, nSamples);
      if (rc !== 0) {
        throw new Error(`Eksekusi _whisper_full gagal dengan kode status: ${rc}`);
      }

      const nSegments = this.wasm._whisper_full_n_segments(this.ctx);
      const parts = [];
      for (let i = 0; i < nSegments; i++) {
        const strPtr = this.wasm._whisper_full_get_segment_text(this.ctx, i);
        if (strPtr) {
          parts.push(this.wasm.UTF8ToString(strPtr));
        }
      }

      const latencyMs = Math.round(performance.now() - t0);
      const audioDurationSec = nSamples / 16000;
      const rtf = audioDurationSec > 0 ? Number(((latencyMs / 1000) / audioDurationSec).toFixed(3)) : 0;

      return {
        text: parts.join(' ').trim(),
        latencyMs,
        rtf,
        backend: this.backendName
      };
    } finally {
      this.wasm._free(pcmPtr);
    }
  }

  async dispose() {
    if (this.wasm) {
      if (this.ctx) {
        try {
          this.wasm._whisper_free(this.ctx);
        } catch (err) {
          console.warn('[GGML WASM] Gagal melepaskan konteks whisper.cpp:', err);
        }
        this.ctx = 0;
      }
      if (this.modelPtr) {
        try {
          this.wasm._free(this.modelPtr);
        } catch (err) {
          console.warn('[GGML WASM] Gagal membebaskan buffer model WASM:', err);
        }
        this.modelPtr = 0;
      }
    }
    this.isReady = false;
  }
}
