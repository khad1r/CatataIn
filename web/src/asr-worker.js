// asr-worker.js - Dedicated Web Worker for In-Browser Speech Recognition (Pass 2)
// Offloads heavy neural inference off the main UI and audio threads.
// Coordinates pluggable Pass 2 engines via Pass2Hub with strict serial execution.

import { Pass2Hub } from './engines/hub.js';

const hub = new Pass2Hub();
let currentEngine = 'webgpu';
let currentModel = 'Xenova/whisper-tiny';
let currentLanguage = 'id';
let opQueue = Promise.resolve();

async function handleMessage(type, data) {
  if (type === 'config') {
    const { engine, model, language } = data;
    if (engine) currentEngine = engine;
    if (model) currentModel = model;
    if (language) currentLanguage = language;

    if (currentModel === 'none') {
      if (hub.activeEngine) {
        await hub.activeEngine.dispose();
        hub.activeEngine = null;
      }
      self.postMessage({
        type: 'status',
        data: { status: 'ready', model: 'none', engine: currentEngine, device: 'NONE' }
      });
      return;
    }

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
      console.error('[ASR Worker] Engine initialization error:', err);
      self.postMessage({
        type: 'status',
        data: { status: 'error', error: err.message, model: currentModel, engine: currentEngine }
      });
    }
  } else if (type === 'transcribe') {
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
      self.postMessage({
        type: 'result',
        data: { id, text: '', speaker, timeStr, error: `Engine Pass 2 (${currentEngine}) belum siap.`, latencyMs: 0, audioDurationS: audioDur, rtf: '0.00' }
      });
      return;
    }

    const startTime = performance.now();

    try {
      const res = await hub.transcribe(pcm16k, {
        speaker,
        timeStr,
        language: lang,
        contextPrompt,
        maxNewTokens: 64
      });

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
}

self.onmessage = (e) => {
  const { type, data } = e.data;
  opQueue = opQueue.then(() => handleMessage(type, data)).catch((err) => {
    console.error('[ASR Worker] Queue error:', err);
  });
};
