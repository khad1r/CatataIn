// asr-worker.js - Dedicated Web Worker for In-Browser Speech Recognition (Pass 2)
// Offloads heavy neural inference off the main UI and audio threads.
// Coordinates pluggable Pass 2 engines (WebGPU, WASM SIMD, WebNN) via Pass2Hub.

import { Pass2Hub } from './engines/hub.js';

const hub = new Pass2Hub();
let isInitializing = false;
let isBusy = false;
let currentEngine = 'webgpu';
let currentModel = 'Xenova/whisper-tiny';
let currentLanguage = 'id';

self.onmessage = async (e) => {
  const { type, data } = e.data;

  if (type === 'config') {
    const { engine, model, language } = data;
    if (engine) currentEngine = engine;
    if (model) currentModel = model;
    if (language) currentLanguage = language;

    if (currentModel === 'none') {
      self.postMessage({
        type: 'status',
        data: { status: 'ready', model: 'none', engine: currentEngine, device: 'NONE' }
      });
      return;
    }

    if (!isInitializing) {
      isInitializing = true;
      self.postMessage({
        type: 'status',
        data: { status: 'loading', model: currentModel, engine: currentEngine }
      });

      try {
        await hub.setEngine(currentEngine, { model: currentModel, language: currentLanguage }, (info) => {
          self.postMessage({
            type: 'progress',
            data: {
              progress: Math.round(info.progress || 0),
              file: info.file
            }
          });
        });

        isInitializing = false;
        const info = hub.getActiveEngineInfo();
        self.postMessage({
          type: 'status',
          data: {
            status: 'ready',
            model: currentModel,
            engine: currentEngine,
            device: info.deviceType,
            engineName: info.name
          }
        });
      } catch (err) {
        isInitializing = false;
        console.error('[ASR Worker] Engine initialization error:', err);
        self.postMessage({
          type: 'status',
          data: { status: 'error', error: err.message, engine: currentEngine }
        });
      }
    }
  } else if (type === 'transcribe_interim') {
    // Pass 1: Streaming interim hypothesis (< 400ms)
    if (isBusy || !hub.activeEngine || !hub.activeEngine.isReady || currentModel === 'none') {
      return; // Skip interim if busy or engine not ready
    }

    const { pcm16k, speaker, durationS } = data;
    isBusy = true;
    try {
      const res = await hub.transcribe(pcm16k, {
        language: currentLanguage,
        maxNewTokens: 16
      });
      isBusy = false;
      if (res && res.text) {
        self.postMessage({
          type: 'interim_result',
          data: { speaker, text: res.text, durationS }
        });
      }
    } catch (_) {
      isBusy = false;
    }
  } else if (type === 'transcribe') {
    // Pass 2: High-accuracy turn finalization
    const { id, pcm16k, speaker, timeStr, language, durationS, contextPrompt } = data;
    const lang = language || currentLanguage;
    const audioDur = typeof durationS === 'number' ? durationS : (pcm16k ? pcm16k.length / 16000 : 0);

    if (currentModel === 'none') {
      self.postMessage({
        type: 'result',
        data: { id, text: '', speaker, timeStr, latencyMs: 0, audioDurationS: audioDur, rtf: '0.00' }
      });
      return;
    }

    if (!hub.activeEngine || !hub.activeEngine.isReady) {
      if (isInitializing) {
        self.postMessage({
          type: 'result',
          data: { id, text: '', speaker, timeStr, error: 'Engine sedang memuat', latencyMs: 0, audioDurationS: audioDur, rtf: '0.00' }
        });
        return;
      }
      try {
        await hub.setEngine(currentEngine, { model: currentModel, language: currentLanguage });
      } catch (err) {
        self.postMessage({
          type: 'result',
          data: { id, text: '', speaker, timeStr, error: err.message, latencyMs: 0, audioDurationS: audioDur, rtf: '0.00' }
        });
        return;
      }
    }

    isBusy = true;
    const startTime = performance.now();

    try {
      const res = await hub.transcribe(pcm16k, {
        speaker,
        timeStr,
        language: lang,
        contextPrompt,
        maxNewTokens: 64
      });

      isBusy = false;
      const latencyMs = res.latencyMs || Math.round(performance.now() - startTime);
      const rtf = audioDur > 0 ? (latencyMs / (audioDur * 1000)).toFixed(2) : '0.00';

      self.postMessage({
        type: 'result',
        data: {
          id,
          text: res.text || '',
          speaker,
          timeStr,
          latencyMs,
          audioDurationS: audioDur,
          rtf
        }
      });
    } catch (err) {
      isBusy = false;
      const latencyMs = Math.round(performance.now() - startTime);
      console.error('[ASR Worker] Finalize inference error:', err);
      self.postMessage({
        type: 'result',
        data: {
          id,
          text: '',
          speaker,
          timeStr,
          error: err.message,
          latencyMs,
          audioDurationS: audioDur,
          rtf: '0.00'
        }
      });
    }
  }
};
