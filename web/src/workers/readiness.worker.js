// workers/readiness.worker.js - Dedicated Module Web Worker for Pass 2 Hardware Readiness Test
// Executes hardware readiness micro-benchmarks in an isolated thread to prevent memory bloat
// and ensure independent ONNX Runtime Web thread pool / WebGPU device states.

import { Pass2WebGpuEngine } from '../engines/pass2-webgpu.js';
import { Pass2WasmSimdEngine } from '../engines/pass2-wasm-simd.js';

function createEngine(engineId) {
  switch (engineId) {
    case 'webgpu':
      return new Pass2WebGpuEngine();
    case 'wasm-simd':
      return new Pass2WasmSimdEngine();
    default:
      return null;
  }
}

self.onmessage = async (e) => {
  const { type, engineId, modelName } = e.data || {};
  if (type === 'TEST_MODEL') {
    try {
      const engine = createEngine(engineId);
      if (!engine || !engine.available) {
        self.postMessage({
          type: 'RESULT',
          result: {
            engineId,
            model: modelName,
            available: false,
            supported: false,
            rtf: Infinity,
            reason: 'Belum tersedia'
          }
        });
        return;
      }

      const res = await engine.testHardware(modelName, (msg) => {
        self.postMessage({ type: 'PROGRESS', message: msg });
      });
      self.postMessage({ type: 'RESULT', result: res });
    } catch (err) {
      self.postMessage({
        type: 'ERROR',
        error: err.message || String(err)
      });
    }
  }
};
