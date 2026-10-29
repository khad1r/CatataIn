// audio-engine.js - WebAudio Capture & catetin.wasm DSP Driver
// Per-source resampler & segmenter isolation, zero fallback, zero dead buffers.

export class AudioEngine {
  constructor() {
    this.audioCtx = null;
    this.isRunning = false;
    this.isPaused = false;

    // Microphone state & isolated DSP context
    this.micStream = null;
    this.micCtx = null; // { stream, sourceNode, processorNode, resampler, segmenter, label, isSpeaking, turnSpeechSamples }
    this.micEnabled = false;

    // Multi-tab audio map: id -> { id, stream, sourceNode, processorNode, resampler, segmenter, label, trackLabel, isSpeaking, turnSpeechSamples }
    this.tabStreams = new Map();
    this.tabCounter = 0;

    // Silent gain node to keep ScriptProcessor alive without playing audio through speakers
    this.silentGainNode = null;

    // WASM Module & shared scratch pointers (used synchronously on the main audio callback thread)
    this.wasm = null;
    this.inBufPtr = 0;
    this.inBufCap = 8192;
    this.resampleBufPtr = 0;
    this.resampleBufCap = 16384;
    this.utteranceBufPtr = 0;
    this.utteranceBufCap = 160000; // 10s at 16kHz

    // Callbacks & Telemetry
    this.onVolume = null;          // (level: number) => void
    this.onUtteranceReady = null;  // (utterance: Float32Array, speakerLabel: string) => void
    this.onVadSpeaking = null;     // (isSpeaking: boolean, rms: number) => void
    this.onSpeechProgress = null;  // ({ speaker, durationS, rms, closed }) => void
    this.onTransducerChunk = null; // (chunkPcm: Float32Array, speaker: string, durationS: number) => void
    this.onTurnFinished = null;    // (speaker: string) => void
    this.onTabsChanged = null;     // (count: number) => void
    this.onLog = null;             // (cat: string, msg: string) => void

    this.isSpeaking = false;
    this.transducerOwner = null;

    // Cached typed array views over WASM memory to avoid per-tick heap allocations
    this._memF32 = null;
    this._memU8 = null;
    this.textEncoder = typeof TextEncoder !== 'undefined' ? new TextEncoder() : null;
    this.textDecoder = typeof TextDecoder !== 'undefined' ? new TextDecoder() : null;
  }

  _getMemF32() {
    if (!this.wasm || !this.wasm.memory) return null;
    if (!this._memF32 || this._memF32.buffer !== this.wasm.memory.buffer) {
      this._memF32 = new Float32Array(this.wasm.memory.buffer);
    }
    return this._memF32;
  }

  _getMemU8() {
    if (!this.wasm || !this.wasm.memory) return null;
    if (!this._memU8 || this._memU8.buffer !== this.wasm.memory.buffer) {
      this._memU8 = new Uint8Array(this.wasm.memory.buffer);
    }
    return this._memU8;
  }

  async initWasm(wasmPath = './catetin.wasm') {
    if (this.wasm) return;
    const response = await fetch(wasmPath);
    if (!response.ok) {
      throw new Error(`Gagal mengunduh ${wasmPath} (HTTP ${response.status})`);
    }
    const wasmModule = await WebAssembly.instantiateStreaming(response);
    this.wasm = wasmModule.instance.exports;

    // Allocate shared synchronous scratch buffers in WASM linear memory
    this.inBufPtr = this.wasm.catetin_alloc_f32(this.inBufCap);
    this.resampleBufPtr = this.wasm.catetin_alloc_f32(this.resampleBufCap);
    this.utteranceBufPtr = this.wasm.catetin_alloc_f32(this.utteranceBufCap);

    if (this.onLog) {
      this.onLog('INFO', 'catetin.wasm (C-ABI) berhasil dimuat.');
    }
  }

  async _ensureAudioContext() {
    if (!this.audioCtx || this.audioCtx.state === 'closed') {
      this.audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      this.silentGainNode = this.audioCtx.createGain();
      this.silentGainNode.gain.value = 0.0;
      this.silentGainNode.connect(this.audioCtx.destination);
    }
    if (this.audioCtx.state === 'suspended') {
      await this.audioCtx.resume();
    }
  }

  _createStreamDspContext(stream, label) {
    if (!this.wasm) {
      throw new Error('Modul DSP catetin.wasm belum dimuat.');
    }
    const resampler = this.wasm.catetin_resampler_new(this.audioCtx.sampleRate, 1, 16000);
    const segmenter = this.wasm.catetin_segmenter_new(16000);
    const ctx = {
      stream,
      label,
      resampler,
      segmenter,
      sourceNode: null,
      processorNode: null,
      isSpeaking: false,
      turnSpeechSamples: 0
    };
    this._setupStreamPipeline(ctx);
    return ctx;
  }

  _destroyStreamDspContext(ctx, flushRemaining = true) {
    if (!ctx) return;
    if (flushRemaining) {
      this._flushContextUtterance(ctx);
    }
    try {
      if (ctx.processorNode) ctx.processorNode.disconnect();
      if (ctx.sourceNode) ctx.sourceNode.disconnect();
      if (ctx.stream) ctx.stream.getTracks().forEach(t => t.stop());
    } catch (err) {
      console.error(`[AudioEngine] Error memutus node ${ctx.label}:`, err);
    }
    if (this.wasm) {
      if (ctx.resampler) {
        this.wasm.catetin_resampler_free(ctx.resampler);
        ctx.resampler = 0;
      }
      if (ctx.segmenter) {
        this.wasm.catetin_segmenter_free(ctx.segmenter);
        ctx.segmenter = 0;
      }
    }
    ctx.isSpeaking = false;
    ctx.turnSpeechSamples = 0;
  }

  _flushContextUtterance(ctx) {
    if (!ctx || !this.wasm || !ctx.segmenter) return;
    const utteranceLen = this.wasm.catetin_segmenter_flush(
      ctx.segmenter,
      this.utteranceBufPtr,
      this.utteranceBufCap
    );
    if (ctx.turnSpeechSamples > 0 && this.transducerOwner === ctx.label) {
      this.transducerOwner = null;
      if (this.onTurnFinished) {
        this.onTurnFinished(ctx.label);
      }
    }
    ctx.turnSpeechSamples = 0;
    ctx.isSpeaking = false;

    if (utteranceLen > 0 && this.onUtteranceReady) {
      const outOffset = this.utteranceBufPtr / 4;
      const latestMem = this._getMemF32();
      const utterance = new Float32Array(latestMem.subarray(outOffset, outOffset + utteranceLen));
      this.onUtteranceReady(utterance, ctx.label);
    }
  }

  _recomputeGlobalSpeaking(latestRms = 0) {
    const anyTabSpeaking = Array.from(this.tabStreams.values()).some(t => t.isSpeaking);
    const nowSpeaking = Boolean((this.micCtx && this.micCtx.isSpeaking) || anyTabSpeaking);
    if (nowSpeaking !== this.isSpeaking) {
      this.isSpeaking = nowSpeaking;
      if (this.onVadSpeaking) {
        this.onVadSpeaking(nowSpeaking, nowSpeaking ? latestRms : 0);
      }
    }
  }

  async startSession() {
    await this.initWasm();
    await this._ensureAudioContext();

    this.isRunning = true;
    this.isPaused = false;

    if (this.onLog) {
      this.onLog('AUDIO', `Sesi dimulai (SampleRate: ${this.audioCtx.sampleRate}Hz -> 16000Hz).`);
    }
  }

  async setMicEnabled(enabled) {
    if (enabled === this.micEnabled) return;

    if (enabled) {
      await this._ensureAudioContext();
      try {
        this.micStream = await navigator.mediaDevices.getUserMedia({
          audio: {
            echoCancellation: true,
            noiseSuppression: true,
            autoGainControl: true
          }
        });
        this.micCtx = this._createStreamDspContext(this.micStream, 'Anda');
        this.micEnabled = true;

        if (this.onLog) {
          this.onLog('AUDIO', 'Mikrofon lokal diaktifkan [Anda].');
        }
      } catch (err) {
        this.micEnabled = false;
        if (this.onLog) {
          this.onLog('ERROR', `Izin mikrofon ditolak/gagal: ${err.message}`);
        }
        throw err;
      }
    } else {
      this._destroyStreamDspContext(this.micCtx, true);
      this.micCtx = null;
      this.micStream = null;
      this.micEnabled = false;
      this._recomputeGlobalSpeaking(0);

      if (this.tabStreams.size === 0) {
        if (this.onVolume) this.onVolume(0);
        if (this.onSpeechProgress) this.onSpeechProgress({ speaker: 'Anda', durationS: 0, rms: 0, closed: true });
      }

      if (this.onLog) {
        this.onLog('AUDIO', 'Mikrofon dimatikan.');
      }
    }
  }

  async addTabAudio() {
    await this._ensureAudioContext();
    try {
      const stream = await navigator.mediaDevices.getDisplayMedia({
        video: true,
        audio: true
      });

      // Hentikan track video segera karena hanya audio tab yang dibutuhkan (#25)
      stream.getVideoTracks().forEach(t => {
        t.stop();
        stream.removeTrack(t);
      });

      const audioTrack = stream.getAudioTracks()[0];
      if (!audioTrack) {
        stream.getTracks().forEach(t => t.stop());
        throw new Error("Pilih opsi 'Share Tab Audio' agar suara tab tertangkap.");
      }

      const trackLabel = audioTrack.label || 'Tab Audio';

      // Cegah redundansi: jika tab yang sama sudah aktif, perbarui stream lama secara bersih
      for (const [id, entry] of this.tabStreams) {
        if (entry.trackLabel === trackLabel) {
          if (this.onLog) {
            this.onLog('STREAM', `Tab "${trackLabel}" sudah aktif, memperbarui stream.`);
          }
          this._destroyStreamDspContext(entry, false);
          this.tabStreams.delete(id);
          break;
        }
      }

      this.tabCounter++;
      const tabId = `tab-${this.tabCounter}`;
      const label = `Tab ${this.tabCounter}`;

      const ctx = this._createStreamDspContext(stream, label);
      ctx.id = tabId;
      ctx.trackLabel = trackLabel;

      this.tabStreams.set(tabId, ctx);

      audioTrack.onended = () => {
        this._removeTabAudio(tabId);
      };

      if (this.onTabsChanged) {
        this.onTabsChanged(this.tabStreams.size);
      }

      if (this.onLog) {
        this.onLog('AUDIO', `Tab Audio ditambahkan: ${label} ("${trackLabel}").`);
      }

      return label;
    } catch (err) {
      if (this.onLog && err.name !== 'NotAllowedError') {
        this.onLog('ERROR', `Gagal menambahkan tab audio: ${err.message}`);
      }
      throw err;
    }
  }

  _removeTabAudio(tabId) {
    const entry = this.tabStreams.get(tabId);
    if (!entry) return;

    // Flush hanya segmenter milik tab ini, tidak menyentuh mic (#16)
    this._destroyStreamDspContext(entry, true);
    this.tabStreams.delete(tabId);
    this._recomputeGlobalSpeaking(0);

    const hasActiveStreams = this.micEnabled || this.tabStreams.size > 0;
    if (!hasActiveStreams) {
      if (this.onVolume) this.onVolume(0);
      if (this.onSpeechProgress) this.onSpeechProgress({ speaker: entry.label, durationS: 0, rms: 0, closed: true });
    }

    if (this.onTabsChanged) {
      this.onTabsChanged(this.tabStreams.size);
    }

    if (this.onLog) {
      this.onLog('STREAM', `Tab ${entry.label} ditutup. Stream dibersihkan.${!hasActiveStreams ? ' Tidak ada stream aktif: VAD kembali ke IDLE.' : ''}`);
    }
  }

  _setupStreamPipeline(ctx) {
    const sourceNode = this.audioCtx.createMediaStreamSource(ctx.stream);
    const bufferSize = 4096;
    const processor = this.audioCtx.createScriptProcessor(bufferSize, 1, 1);

    ctx.sourceNode = sourceNode;
    ctx.processorNode = processor;

    processor.onaudioprocess = (event) => {
      if (!this.isRunning || this.isPaused || !ctx.resampler || !ctx.segmenter) return;

      const inputData = event.inputBuffer.getChannelData(0);
      const memF32 = this._getMemF32();
      if (!memF32) return;

      const inOffset = this.inBufPtr / 4;
      const len = Math.min(inputData.length, this.inBufCap);
      if (inputData.length <= this.inBufCap) {
        memF32.set(inputData, inOffset);
      } else {
        memF32.set(inputData.subarray(0, len), inOffset);
      }

      // Resample input -> 16kHz mono menggunakan resampler khusus stream ini
      const resampledLen = this.wasm.catetin_resampler_process(
        ctx.resampler,
        this.inBufPtr,
        len,
        this.resampleBufPtr,
        this.resampleBufCap
      );

      if (resampledLen <= 0) return;

      const rms = this.wasm.catetin_vad_frame_rms(0, this.resampleBufPtr, resampledLen);

      if (this.onVolume) {
        this.onVolume(Math.min(1.0, rms * 5.0));
      }

      // Feed segmenter khusus stream ini
      const utteranceLen = this.wasm.catetin_segmenter_feed(
        ctx.segmenter,
        this.resampleBufPtr,
        resampledLen,
        this.utteranceBufPtr,
        this.utteranceBufCap
      );

      const speaking = this.wasm.catetin_segmenter_is_speaking(ctx.segmenter) === 1;

      if (speaking) {
        if (!this.transducerOwner) {
          this.transducerOwner = ctx.label;
        }
        ctx.turnSpeechSamples += resampledLen;
        if (this.transducerOwner === ctx.label && this.onTransducerChunk) {
          const resOffset = this.resampleBufPtr / 4;
          const currentMem = this._getMemF32();
          const resampledView = currentMem.subarray(resOffset, resOffset + resampledLen);
          this.onTransducerChunk(resampledView, ctx.label, ctx.turnSpeechSamples / 16000);
        }
      } else {
        if (ctx.turnSpeechSamples > 0) {
          if (this.transducerOwner === ctx.label) {
            this.transducerOwner = null;
            if (this.onTurnFinished) {
              this.onTurnFinished(ctx.label);
            }
          }
        }
        ctx.turnSpeechSamples = 0;
      }

      if (speaking !== ctx.isSpeaking) {
        ctx.isSpeaking = speaking;
        this._recomputeGlobalSpeaking(rms);
        if (this.onLog) {
          this.onLog('VAD', speaking ? `Suara terdeteksi pada [${ctx.label}] (RMS: ${rms.toFixed(3)})` : `Suara berhenti pada [${ctx.label}].`);
        }
      }

      if (speaking) {
        const durS = ctx.turnSpeechSamples / 16000;
        if (this.onSpeechProgress) {
          this.onSpeechProgress({
            speaker: ctx.label,
            durationS: durS,
            rms
          });
        }

        // Potong turn lebih awal pada >= 4.5s jika terdapat penurunan energi (< 0.005)
        if (durS >= 4.5 && rms < 0.005) {
          const earlyLen = this.wasm.catetin_segmenter_flush(
            ctx.segmenter,
            this.utteranceBufPtr,
            this.utteranceBufCap
          );
          if (earlyLen > 0) {
            if (this.transducerOwner === ctx.label) {
              this.transducerOwner = null;
              if (this.onTurnFinished) this.onTurnFinished(ctx.label);
            }
            ctx.turnSpeechSamples = 0;
            const outOffset = this.utteranceBufPtr / 4;
            const latestMem = this._getMemF32();
            const utterance = new Float32Array(latestMem.subarray(outOffset, outOffset + earlyLen));

            if (this.onLog) {
              this.onLog('QUEUE', `Turn dipotong pada 4.5s [${ctx.label}] -> Kirim ke ASR Worker`);
            }

            if (this.onUtteranceReady) {
              this.onUtteranceReady(utterance, ctx.label);
            }
          }
        }
      }

      // Spoken turn endpointed by natural silence or 9s limit
      if (utteranceLen > 0) {
        if (this.transducerOwner === ctx.label) {
          this.transducerOwner = null;
          if (this.onTurnFinished) this.onTurnFinished(ctx.label);
        }
        ctx.turnSpeechSamples = 0;
        const outOffset = this.utteranceBufPtr / 4;
        const latestMem = this._getMemF32();
        const utterance = new Float32Array(latestMem.subarray(outOffset, outOffset + utteranceLen));

        if (this.onLog) {
          const durS = (utteranceLen / 16000).toFixed(1);
          this.onLog('QUEUE', `Turn selesai [${durS}s, ${ctx.label}] -> Kirim ke ASR Worker`);
        }

        if (this.onUtteranceReady) {
          this.onUtteranceReady(utterance, ctx.label);
        }
      }
    };

    sourceNode.connect(processor);
    processor.connect(this.silentGainNode);
  }

  pause() {
    this.isPaused = true;
    this.flushUtterance();
    if (this.onVadSpeaking) this.onVadSpeaking(false, 0);
    if (this.onVolume) this.onVolume(0);
    if (this.onLog) this.onLog('AUDIO', 'Sesi dijeda.');
  }

  resume() {
    this.isPaused = false;
    if (this.onLog) this.onLog('AUDIO', 'Sesi dilanjutkan.');
  }

  flushUtterance() {
    if (this.micCtx) {
      this._flushContextUtterance(this.micCtx);
    }
    for (const entry of this.tabStreams.values()) {
      this._flushContextUtterance(entry);
    }
    this._recomputeGlobalSpeaking(0);
  }

  stopSession() {
    this.isRunning = false;
    this.isPaused = false;

    if (this.micCtx) {
      this._destroyStreamDspContext(this.micCtx, true);
      this.micCtx = null;
    }
    this.micStream = null;
    this.micEnabled = false;

    for (const entry of this.tabStreams.values()) {
      this._destroyStreamDspContext(entry, true);
    }
    this.tabStreams.clear();
    this.isSpeaking = false;
    this.transducerOwner = null;
    if (this.audioCtx && this.audioCtx.state === 'running') {
      this.audioCtx.suspend().catch((err) => {
        console.error('[AudioEngine] Gagal men-suspend AudioContext:', err);
      });
    }

    if (this.onVadSpeaking) this.onVadSpeaking(false, 0);
    if (this.onVolume) this.onVolume(0);
    if (this.onTabsChanged) this.onTabsChanged(0);
    if (this.onLog) this.onLog('AUDIO', 'Sesi dihentikan seluruhnya.');
  }

  // Truncate runaway repetition loops in Whisper output (e.g. "kata, kata, kata...")
  cleanRepetitiveLoops(text) {
    if (!text || typeof text !== 'string') return '';
    const words = text.trim().split(/\s+/);
    if (words.length < 4) return text.trim();

    for (let n = 1; n <= 4; n++) {
      if (words.length < n * 3) continue;
      for (let i = 0; i <= words.length - n * 3; i++) {
        const c1 = words.slice(i, i + n).join(' ').toLowerCase();
        const c2 = words.slice(i + n, i + 2 * n).join(' ').toLowerCase();
        const c3 = words.slice(i + 2 * n, i + 3 * n).join(' ').toLowerCase();
        if (c1 === c2 && c2 === c3) {
          return words.slice(0, i + n).join(' ');
        }
      }
    }
    return text.trim();
  }

  // Hallucination filter powered by Rust WASM
  isHallucination(text) {
    if (!text || text.trim().length === 0) return true;
    if (!this.wasm || !this.wasm.catetin_is_hallucination) {
      throw new Error('Fungsi WASM catetin_is_hallucination tidak tersedia.');
    }
    if (!this.textEncoder) this.textEncoder = new TextEncoder();
    const bytes = this.textEncoder.encode(text);
    const ptr = this.wasm.catetin_alloc_u8(bytes.length);
    const memU8 = this._getMemU8();
    memU8.set(bytes, ptr);
    const result = this.wasm.catetin_is_hallucination(ptr, bytes.length) === 1;
    this.wasm.catetin_free_u8(ptr, bytes.length);
    return result;
  }

  // Text formatter powered by Rust WASM
  formatUtterance(text) {
    if (!text || !text.trim()) return '';
    if (!this.wasm || !this.wasm.catetin_format_utterance) {
      throw new Error('Fungsi WASM catetin_format_utterance tidak tersedia.');
    }
    if (!this.textEncoder) this.textEncoder = new TextEncoder();
    const bytes = this.textEncoder.encode(text);
    const inPtr = this.wasm.catetin_alloc_u8(bytes.length);
    const outCap = bytes.length * 2 + 128;
    const outPtr = this.wasm.catetin_alloc_u8(outCap);

    try {
      let memU8 = this._getMemU8();
      memU8.set(bytes, inPtr);

      const written = this.wasm.catetin_format_utterance(inPtr, bytes.length, 0, 0, outPtr, outCap);
      if (written <= 0) {
        return '';
      }
      if (!this.textDecoder) this.textDecoder = new TextDecoder();
      memU8 = this._getMemU8();
      return this.textDecoder.decode(memU8.subarray(outPtr, outPtr + written)).trim();
    } finally {
      this.wasm.catetin_free_u8(inPtr, bytes.length);
      this.wasm.catetin_free_u8(outPtr, outCap);
    }
  }
}
