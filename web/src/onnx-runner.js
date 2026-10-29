// onnx-runner.js - Coordinator for Off-Thread ASR Web Workers (Pass 1 & Pass 2)
// Tracks task queue depth, inference latency, RTF, and emits telemetry events.

function prepareTransferablePcm(pcm16k) {
  if (!pcm16k || !pcm16k.buffer) return { pcm: pcm16k, transfer: [] };
  const standalone = (pcm16k.byteOffset === 0 && pcm16k.byteLength === pcm16k.buffer.byteLength)
    ? pcm16k
    : pcm16k.slice();
  return { pcm: standalone, transfer: [standalone.buffer] };
}

export class OnnxRunner {
  constructor() {
    this.worker = null;
    this.pendingTasks = new Map();
    this.taskIdCounter = 0;
    this.isReady = false;
    this.isLoading = false;
    this.status = 'idle'; // idle | loading | ready | transcribing | error
    this.lastPass2Error = null;
    this.device = 'unknown';

    // Model & Engine settings
    this.pass2Engine = 'webgpu';
    this.whisperModel = 'Xenova/whisper-tiny';
    this.language = 'id';
    this.transducerModel = 'id'; // 'id' | 'en' | 'none'
    this.transducerWorker = null;
    this.transducerReady = false;
    this.transducerStatus = 'idle'; // idle | loading | ready | disabled | error
    this.lastPass1Error = null;

    this._pass1Waiters = [];
    this._pass2Waiters = [];

    // Performance telemetry
    this.lastLatencyMs = 0;
    this.lastRtf = '0.00';
    this.totalInferences = 0;
    this.totalLatencyMs = 0;

    // Callbacks
    this.onStatusChange = null;  // (status, info) => void
    this.onProgress = null;      // (progressInfo) => void
    this.onTelemetry = null;     // (info) => void
    this.onLog = null;           // (cat: string, msg: string) => void
    this.onInterimResult = null; // ({ speaker, text, durationS, decodeTimeMs }) => void
    this.onTransducerStatus = null; // (status, model) => void

    this._initWorker();
    this._initTransducerWorker();
  }

  _settlePass1Waiters(err = null) {
    const waiters = this._pass1Waiters.splice(0);
    for (const w of waiters) {
      if (err) w.reject(err);
      else w.resolve();
    }
  }

  _settlePass2Waiters(err = null) {
    const waiters = this._pass2Waiters.splice(0);
    for (const w of waiters) {
      if (err) w.reject(err);
      else w.resolve();
    }
  }

  _rejectPendingTasks(reason) {
    const err = new Error(reason);
    for (const task of this.pendingTasks.values()) {
      task.reject(err);
    }
    this.pendingTasks.clear();
  }

  whenPass1Ready() {
    if (this.transducerModel === 'none' || this.transducerReady || this.transducerStatus === 'disabled') {
      return Promise.resolve();
    }
    if (this.transducerStatus === 'error') {
      return Promise.reject(new Error(this.lastPass1Error || 'Transducer Pass 1 gagal dimuat.'));
    }
    return new Promise((resolve, reject) => {
      this._pass1Waiters.push({ resolve, reject });
    });
  }

  whenPass2Ready() {
    if (this.whisperModel === 'none' || this.isReady) {
      return Promise.resolve();
    }
    if (this.status === 'error') {
      return Promise.reject(new Error(this.lastPass2Error || 'Engine Pass 2 gagal dimuat.'));
    }
    return new Promise((resolve, reject) => {
      this._pass2Waiters.push({ resolve, reject });
    });
  }

  _initWorker() {
    if (typeof Worker === 'undefined') {
      this.status = 'error';
      this.lastPass2Error = 'Web Workers tidak didukung di peramban ini.';
      return;
    }

    this.worker = new Worker('./src/asr-worker.js', { type: 'module' });

    this.worker.onmessage = (e) => {
      const { type, data } = e.data;

      if (type === 'status') {
        if (data.model && data.model !== this.whisperModel) return;
        if (data.engine && data.engine !== this.pass2Engine && data.model !== 'none') return;

        this.status = data.status;
        this.isReady = data.status === 'ready';
        this.isLoading = data.status === 'loading';
        if (data.device) this.device = data.device;
        if (data.engine) this.pass2Engine = data.engine;

        if (data.status === 'ready') {
          this.lastPass2Error = null;
          this._settlePass2Waiters(null);
        } else if (data.status === 'error') {
          this.lastPass2Error = data.error || 'Gagal inisialisasi Pass 2';
          this._settlePass2Waiters(new Error(this.lastPass2Error));
          this._rejectPendingTasks(this.lastPass2Error);
        }

        if (this.onStatusChange) {
          this.onStatusChange(data.status, data);
        }

        if (this.onTelemetry) {
          this.onTelemetry({
            status: this.status,
            model: this.whisperModel,
            device: this.device,
            queueDepth: this.pendingTasks.size,
            lastLatencyMs: this.lastLatencyMs,
            lastRtf: this.lastRtf
          });
        }

        if (this.onLog) {
          if (data.status === 'ready') {
            this.onLog('WHISPER', `Engine Pass 2 siap: ${data.engineName || data.engine} (${data.model}) [${data.device?.toUpperCase() || 'GPU'}]`);
          } else if (data.status === 'loading') {
            this.onLog('WHISPER', `Memuat Pass 2 (${data.engine || this.pass2Engine}): ${data.model}...`);
          } else if (data.status === 'error') {
            this.onLog('ERROR', `Gagal inisialisasi Pass 2 (${data.engine || this.pass2Engine}): ${data.error}`);
          }
        }
      } else if (type === 'progress') {
        if (this.onProgress) {
          this.onProgress(data);
        }
      } else if (type === 'result') {
        const { id, text, speaker, timeStr, latencyMs, rtf, error } = data;

        if (typeof latencyMs === 'number') this.lastLatencyMs = latencyMs;
        if (rtf) this.lastRtf = rtf;
        this.totalInferences++;
        this.totalLatencyMs += this.lastLatencyMs;

        const resolver = this.pendingTasks.get(id);
        if (resolver) {
          this.pendingTasks.delete(id);

          if (this.pendingTasks.size === 0 && this.status === 'transcribing') {
            this.status = 'ready';
          }

          if (this.onTelemetry) {
            this.onTelemetry({
              status: this.pendingTasks.size > 0 ? 'transcribing' : 'ready',
              model: this.whisperModel,
              device: this.device,
              queueDepth: this.pendingTasks.size,
              lastLatencyMs: this.lastLatencyMs,
              lastRtf: this.lastRtf,
              lastText: text
            });
          }

          if (this.onLog) {
            if (error) {
              this.onLog('ERROR', `Tugas #${id} gagal: ${error}`);
            } else {
              const shortText = text.length > 50 ? text.slice(0, 50) + '...' : text;
              this.onLog('WHISPER', `Tugas #${id} selesai (${latencyMs}ms, RTF: ${rtf}x) [${speaker}]: "${shortText}"`);
            }
          }

          if (error) {
            resolver.reject(new Error(error));
          } else {
            resolver.resolve(text);
          }
        }
      }
    };

    this.worker.onerror = (err) => {
      console.error('[OnnxRunner] Worker error:', err);
      this.status = 'error';
      this.isReady = false;
      this.isLoading = false;
      this.lastPass2Error = err.message || 'Worker runtime error';
      this._settlePass2Waiters(new Error(this.lastPass2Error));
      this._rejectPendingTasks(this.lastPass2Error);
      if (this.onLog) {
        this.onLog('ERROR', `Worker runtime error: ${this.lastPass2Error}`);
      }
    };
  }

  _initTransducerWorker() {
    if (typeof Worker === 'undefined') {
      this.transducerStatus = 'error';
      this.lastPass1Error = 'Web Workers tidak didukung di peramban ini.';
      return;
    }

    this.transducerWorker = new Worker('./src/sherpa/sherpa-transducer-worker.js');

    this.transducerWorker.onmessage = (e) => {
      const { type, data } = e.data;
      if (type === 'status') {
        this.transducerStatus = data.status;
        this.transducerReady = data.status === 'ready';
        if (data.status === 'ready' || data.status === 'disabled') {
          this.lastPass1Error = null;
          this._settlePass1Waiters(null);
        } else if (data.status === 'error') {
          this.lastPass1Error = data.message || 'Transducer Pass 1 error';
          this._settlePass1Waiters(new Error(this.lastPass1Error));
        }

        if (this.onTransducerStatus) {
          this.onTransducerStatus(this.transducerStatus, this.transducerModel);
        }

        if (this.onLog) {
          const cat = data.status === 'error' ? 'ERROR' : 'TRANSDUCER';
          this.onLog(cat, data.message || `Transducer status: ${data.status}`);
        }
      } else if (type === 'partial_result') {
        if (this.onInterimResult) {
          this.onInterimResult(data);
        }
      }
    };

    this.transducerWorker.onerror = (err) => {
      console.error('[OnnxRunner] Transducer Worker error:', err);
      this.transducerReady = false;
      this.transducerStatus = 'error';
      this.lastPass1Error = err.message || 'Transducer Worker runtime error';
      this._settlePass1Waiters(new Error(this.lastPass1Error));
      if (this.onTransducerStatus) {
        this.onTransducerStatus(this.transducerStatus, this.transducerModel);
      }
      if (this.onLog) {
        this.onLog('ERROR', `Transducer Worker error: ${this.lastPass1Error}`);
      }
    };

    this.transducerStatus = 'loading';
    if (this.onTransducerStatus) {
      this.onTransducerStatus(this.transducerStatus, this.transducerModel);
    }
    this.transducerWorker.postMessage({
      type: 'init',
      data: { language: this.transducerModel }
    });
  }

  updateConfig(config) {
    const nextEngine = config.pass2Engine !== undefined ? config.pass2Engine : this.pass2Engine;
    const nextModel = config.whisperModel !== undefined ? config.whisperModel : this.whisperModel;
    const nextLang = config.language !== undefined ? config.language : this.language;
    const nextTransducer = config.transducerModel !== undefined ? config.transducerModel : this.transducerModel;

    const pass1Changed = nextTransducer !== this.transducerModel || this.transducerStatus === 'error';
    const pass2Changed =
      nextEngine !== this.pass2Engine ||
      nextModel !== this.whisperModel ||
      nextLang !== this.language ||
      this.status === 'error' ||
      this.status === 'idle';

    this.pass2Engine = nextEngine;
    this.whisperModel = nextModel;
    this.language = nextLang;

    if (pass1Changed) {
      this.transducerModel = nextTransducer;
      this.transducerReady = false;
      this.lastPass1Error = null;
      this.transducerStatus = this.transducerModel === 'none' ? 'disabled' : 'loading';
      if (this.onTransducerStatus) {
        this.onTransducerStatus(this.transducerStatus, this.transducerModel);
      }
      if (this.transducerWorker) {
        this.transducerWorker.postMessage({
          type: 'init',
          data: { language: this.transducerModel }
        });
      }
    }

    if (pass2Changed && this.worker) {
      this._rejectPendingTasks('Konfigurasi engine Pass 2 diubah.');
      this.isReady = this.whisperModel === 'none';
      this.isLoading = this.whisperModel !== 'none';
      this.lastPass2Error = null;
      this.status = this.whisperModel === 'none' ? 'ready' : 'loading';
      this.worker.postMessage({
        type: 'config',
        data: {
          engine: this.pass2Engine,
          model: this.whisperModel,
          language: this.language
        }
      });
    }
  }

  feedTransducerChunk(pcm16k, speaker = 'Anda', durationS = 0) {
    if (!this.transducerWorker || !this.transducerReady || this.transducerModel === 'none') {
      return;
    }
    const { pcm, transfer } = prepareTransferablePcm(pcm16k);
    this.transducerWorker.postMessage({
      type: 'accept_chunk',
      data: {
        pcm16k: pcm,
        speaker,
        durationS
      }
    }, transfer);
  }

  resetTransducer() {
    if (this.transducerWorker && this.transducerReady) {
      this.transducerWorker.postMessage({ type: 'reset_stream' });
    }
  }

  async transcribeUtterance(pcm16k, speaker = 'Anda', timeStr = '') {
    if (!this.worker || this.whisperModel === 'none') {
      return null;
    }

    const id = ++this.taskIdCounter;
    const durationS = pcm16k.length / 16000;
    this.status = 'transcribing';

    return new Promise((resolve, reject) => {
      this.pendingTasks.set(id, { resolve, reject, durationS, queuedAt: performance.now() });

      if (this.onTelemetry) {
        this.onTelemetry({
          status: 'transcribing',
          model: this.whisperModel,
          device: this.device,
          queueDepth: this.pendingTasks.size,
          lastLatencyMs: this.lastLatencyMs,
          lastRtf: this.lastRtf
        });
      }

      if (this.onLog) {
        this.onLog('QUEUE', `Tugas #${id} masuk antrean (${durationS.toFixed(1)}s audio [${speaker}], kedalaman: ${this.pendingTasks.size})`);
      }

      const { pcm, transfer } = prepareTransferablePcm(pcm16k);
      this.worker.postMessage({
        type: 'transcribe',
        data: {
          id,
          pcm16k: pcm,
          speaker,
          timeStr,
          language: this.language,
          durationS
        }
      }, transfer);
    });
  }
}
