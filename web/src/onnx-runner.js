// onnx-runner.js - Coordinator for Off-Thread ASR Web Worker
// Keeps the main UI and WebAudio threads 100% smooth and responsive.
// Tracks task queue depth, inference latency, RTF, and emits telemetry events.

export class OnnxRunner {
  constructor() {
    this.worker = null;
    this.pendingTasks = new Map();
    this.taskIdCounter = 0;
    this.isReady = false;
    this.isLoading = false;
    this.status = 'idle'; // idle | loading | ready | transcribing | error
    this.device = 'unknown';

    // Model & Engine settings
    this.pass2Engine = 'webgpu'; // 'webgpu' | 'wasm-simd' | 'webnn'
    this.whisperModel = 'Xenova/whisper-tiny';
    this.language = 'id';
    this.transducerModel = 'id'; // 'id' | 'en' | 'none'
    this.transducerWorker = null;
    this.transducerReady = false;

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

    this._initWorker();
    this._initTransducerWorker();
  }

  _initWorker() {
    if (typeof Worker === 'undefined') {
      console.warn("[OnnxRunner] Web Workers not supported in this environment.");
      return;
    }

    try {
      this.worker = new Worker('./src/asr-worker.js', { type: 'module' });

      this.worker.onmessage = (e) => {
        const { type, data } = e.data;

        if (type === 'status') {
          this.status = data.status;
          this.isReady = data.status === 'ready';
          this.isLoading = data.status === 'loading';
          if (data.device) this.device = data.device;
          if (data.engine) this.pass2Engine = data.engine;

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
        } else if (type === 'interim_result') {
          if (this.onInterimResult) {
            this.onInterimResult(data);
          }
        } else if (type === 'result') {
          const { id, text, speaker, timeStr, latencyMs, audioDurationS, rtf, error } = data;

          if (typeof latencyMs === 'number') this.lastLatencyMs = latencyMs;
          if (rtf) this.lastRtf = rtf;
          this.totalInferences++;
          this.totalLatencyMs += this.lastLatencyMs;

          const resolver = this.pendingTasks.get(id);
          if (resolver) {
            this.pendingTasks.delete(id);

            // Update status back to ready if queue is empty
            if (this.pendingTasks.size === 0) {
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
              resolver.resolve(null);
            } else {
              resolver.resolve(text);
            }
          }
        }
      };

      this.worker.onerror = (err) => {
        console.error("[OnnxRunner] Worker error:", err);
        this.status = 'error';
        this.isLoading = false;
        if (this.onLog) {
          this.onLog('ERROR', `Worker runtime error: ${err.message || 'Unknown'}`);
        }
      };
    } catch (err) {
      console.error("[OnnxRunner] Failed to instantiate worker:", err);
      if (this.onLog) {
        this.onLog('ERROR', `Gagal membuat Web Worker: ${err.message}`);
      }
    }
  }

  _initTransducerWorker() {
    if (typeof Worker === 'undefined') return;

    try {
      this.transducerWorker = new Worker('./src/sherpa/sherpa-transducer-worker.js');

      this.transducerWorker.onmessage = (e) => {
        const { type, data } = e.data;
        if (type === 'status') {
          this.transducerReady = data.status === 'ready';
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
        console.error("[OnnxRunner] Transducer Worker error:", err);
        this.transducerReady = false;
        if (this.onLog) {
          this.onLog('ERROR', `Transducer Worker error: ${err.message || 'Unknown'}`);
        }
      };

      // Inisialisasi awal model Transducer
      this.transducerWorker.postMessage({
        type: 'init',
        data: { language: this.transducerModel }
      });
    } catch (err) {
      console.error("[OnnxRunner] Failed to instantiate Transducer Worker:", err);
      if (this.onLog) {
        this.onLog('ERROR', `Gagal membuat Transducer Worker: ${err.message}`);
      }
    }
  }

  updateConfig(config) {
    if (config.pass2Engine !== undefined) {
      this.pass2Engine = config.pass2Engine;
    }
    if (config.whisperModel !== undefined) {
      this.whisperModel = config.whisperModel;
    }
    if (config.language !== undefined) {
      this.language = config.language;
    }
    if (config.transducerModel !== undefined && config.transducerModel !== this.transducerModel) {
      this.transducerModel = config.transducerModel;
      if (this.transducerWorker) {
        this.transducerWorker.postMessage({
          type: 'init',
          data: { language: this.transducerModel }
        });
      }
    }

    if (this.worker) {
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
      // Fallback ke fast-decode jika Transducer dinonaktifkan
      if (this.transducerModel === 'none') {
        this.transcribeInterim(pcm16k, speaker, durationS);
      }
      return;
    }
    const canTransfer = pcm16k && pcm16k.buffer &&
      !(typeof WebAssembly !== 'undefined' && pcm16k.buffer instanceof WebAssembly.Memory) &&
      pcm16k.byteOffset === 0 &&
      pcm16k.byteLength === pcm16k.buffer.byteLength;
    const transfer = canTransfer ? [pcm16k.buffer] : [];

    this.transducerWorker.postMessage({
      type: 'accept_chunk',
      data: {
        pcm16k,
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

  transcribeInterim(pcm16k, speaker = 'Anda', durationS = 0) {
    if (!this.worker || this.whisperModel === 'none' || !this.isReady) {
      return;
    }
    // Safe zero-copy transfer if buffer is dedicated, or structured clone if subarray/view
    const canTransfer = pcm16k && pcm16k.buffer &&
      !(typeof WebAssembly !== 'undefined' && pcm16k.buffer instanceof WebAssembly.Memory) &&
      pcm16k.byteOffset === 0 &&
      pcm16k.byteLength === pcm16k.buffer.byteLength;
    const transfer = canTransfer ? [pcm16k.buffer] : [];

    this.worker.postMessage({
      type: 'transcribe_interim',
      data: {
        pcm16k,
        speaker,
        durationS
      }
    }, transfer);
  }

  async transcribeUtterance(pcm16k, speaker = 'Anda', timeStr = '') {
    if (!this.worker || this.whisperModel === 'none') {
      return null;
    }

    const id = ++this.taskIdCounter;
    const durationS = pcm16k.length / 16000;
    this.status = 'transcribing';

    return new Promise((resolve) => {
      this.pendingTasks.set(id, { resolve, durationS, queuedAt: performance.now() });

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

      // Safe zero-copy transfer of audio buffer to worker!
      const canTransfer = pcm16k && pcm16k.buffer &&
        !(typeof WebAssembly !== 'undefined' && pcm16k.buffer instanceof WebAssembly.Memory) &&
        pcm16k.byteOffset === 0 &&
        pcm16k.byteLength === pcm16k.buffer.byteLength;
      const transfer = canTransfer ? [pcm16k.buffer] : [];

      this.worker.postMessage({
        type: 'transcribe',
        data: {
          id,
          pcm16k,
          speaker,
          timeStr,
          language: this.language,
          durationS
        }
      }, transfer);
    });
  }
}
