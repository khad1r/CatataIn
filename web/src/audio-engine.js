// audio-engine.js - WebAudio Capture & catetin.wasm DSP Driver
// Supports multi-tab audio sharing, independent mic toggle, zero loopback echo,
// and real-time turn progress telemetry.

export class AudioEngine {
  constructor() {
    this.audioCtx = null;
    this.isRunning = false;
    this.isPaused = false;

    // Microphone state
    this.micStream = null;
    this.micSourceNode = null;
    this.micProcessorNode = null;
    this.micEnabled = false;

    // Multi-tab audio map: id -> { id, stream, sourceNode, processorNode, label, trackLabel }
    this.tabStreams = new Map();
    this.tabCounter = 0;

    // Silent gain node to keep ScriptProcessor alive without playing audio through speakers
    this.silentGainNode = null;

    // WASM Module & Handles
    this.wasm = null;
    this.resampler = null;
    this.segmenter = null;
    this.clusterer = null;

    // Memory buffer pointers
    this.inBufPtr = 0;
    this.inBufCap = 8192;
    this.resampleBufPtr = 0;
    this.resampleBufCap = 16384;
    this.utteranceBufPtr = 0;
    this.utteranceBufCap = 160000; // 10s at 16kHz

    // Callbacks & Telemetry
    this.onVolume = null;         // (level: number) => void
    this.onUtteranceReady = null; // (utterance: Float32Array, speakerLabel: string) => void
    this.onVadSpeaking = null;    // (isSpeaking: boolean, rms: number) => void
    this.onSpeechProgress = null; // ({ speaker, durationS, rms, closed }) => void
    this.onInterimSpeech = null;  // (interimPcm: Float32Array, speaker: string, durationS: number) => void
    this.onTransducerChunk = null; // (chunkPcm: Float32Array, speaker: string, durationS: number) => void
    this.onTurnFinished = null;    // (speaker: string) => void
    this.onTabsChanged = null;    // (count: number) => void
    this.onLog = null;            // (cat: string, msg: string) => void

    this.isSpeaking = false;
    this.lastActiveSpeaker = 'Anda';
    this.turnSpeechSamples = 0;
    this.stagingBuffer = new Float32Array(160000); // 10s pre-allocated contiguous staging buffer
    this.lastInterimTime = 0;

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
    try {
      let wasmModule;
      try {
        wasmModule = await WebAssembly.instantiateStreaming(fetch(wasmPath));
      } catch {
        const response = await fetch(wasmPath);
        const bytes = await response.arrayBuffer();
        wasmModule = await WebAssembly.instantiate(bytes);
      }

      this.wasm = wasmModule.instance.exports;

      // Allocate WASM memory
      this.inBufPtr = this.wasm.catetin_alloc_f32(this.inBufCap);
      this.resampleBufPtr = this.wasm.catetin_alloc_f32(this.resampleBufCap);
      this.utteranceBufPtr = this.wasm.catetin_alloc_f32(this.utteranceBufCap);

      // Initialize DSP instances
      this.segmenter = this.wasm.catetin_segmenter_new(16000);
      this.clusterer = this.wasm.catetin_clusterer_new(0.65, 0.10);

      if (this.onLog) {
        this.onLog('INFO', 'catetin.wasm (C-ABI) berhasil dimuat.');
      }
    } catch (err) {
      console.warn("[AudioEngine] Could not initialize catetin.wasm:", err);
      this.wasm = null;
      if (this.onLog) {
        this.onLog('ERROR', `Gagal inisialisasi catetin.wasm: ${err.message}`);
      }
    }
  }

  _ensureAudioContext() {
    if (!this.audioCtx) {
      this.audioCtx = new (window.AudioContext || window.webkitAudioContext)({
        sampleRate: 48000
      });
      // Zero-gain node keeps ScriptProcessor active without looping back to speakers
      this.silentGainNode = this.audioCtx.createGain();
      this.silentGainNode.gain.value = 0.0;
      this.silentGainNode.connect(this.audioCtx.destination);
    }
    if (this.audioCtx.state === 'suspended') {
      this.audioCtx.resume();
    }
  }

  async startSession() {
    this._ensureAudioContext();
    await this.initWasm();

    if (this.wasm && !this.resampler) {
      this.resampler = this.wasm.catetin_resampler_new(this.audioCtx.sampleRate, 1, 16000);
    }

    this.isRunning = true;
    this.isPaused = false;

    if (this.onLog) {
      this.onLog('AUDIO', `Sesi dimulai (SampleRate: ${this.audioCtx.sampleRate}Hz, Resampler 48k->16k).`);
    }
  }

  async setMicEnabled(enabled) {
    if (enabled === this.micEnabled) return;

    if (enabled) {
      this._ensureAudioContext();
      try {
        this.micStream = await navigator.mediaDevices.getUserMedia({
          audio: {
            echoCancellation: true,
            noiseSuppression: true,
            autoGainControl: true
          }
        });
        this._setupStreamPipeline(this.micStream, 'Anda', (proc, src) => {
          this.micProcessorNode = proc;
          this.micSourceNode = src;
        });
        this.micEnabled = true;
        this.lastActiveSpeaker = 'Anda';

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
      if (this.micStream) {
        this.micStream.getTracks().forEach(t => t.stop());
        this.micStream = null;
      }
      if (this.micProcessorNode) {
        try { this.micProcessorNode.disconnect(); } catch (_) {}
        this.micProcessorNode = null;
      }
      if (this.micSourceNode) {
        try { this.micSourceNode.disconnect(); } catch (_) {}
        this.micSourceNode = null;
      }
      this.micEnabled = false;

      // Check if any audio sources remain
      const hasStreams = this.tabStreams.size > 0;
      if (!hasStreams) {
        this.isSpeaking = false;
        this.turnSpeechSamples = 0;
        if (this.onVadSpeaking) this.onVadSpeaking(false, 0);
        if (this.onVolume) this.onVolume(0);
        if (this.onSpeechProgress) this.onSpeechProgress({ speaker: 'Anda', durationS: 0, rms: 0, closed: true });
      }

      if (this.onLog) {
        this.onLog('AUDIO', 'Mikrofon dimatikan.');
      }
    }
  }

  async addTabAudio() {
    this._ensureAudioContext();
    try {
      const stream = await navigator.mediaDevices.getDisplayMedia({
        video: true,
        audio: true
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
            this.onLog('STREAM', `Tab "${trackLabel}" sudah aktif, memperbarui stream tanpa redundansi.`);
          }
          try {
            entry.stream.getTracks().forEach(t => t.stop());
            if (entry.processorNode) entry.processorNode.disconnect();
            if (entry.sourceNode) entry.sourceNode.disconnect();
          } catch (_) {}
          this.tabStreams.delete(id);
          break;
        }
      }

      this.tabCounter++;
      const tabId = `tab-${this.tabCounter}`;
      const label = `Tab ${this.tabCounter}`;

      let processorNode = null;
      let sourceNode = null;
      this._setupStreamPipeline(stream, label, (proc, src) => {
        processorNode = proc;
        sourceNode = src;
      });

      this.tabStreams.set(tabId, {
        id: tabId,
        stream,
        label,
        trackLabel,
        processorNode,
        sourceNode
      });

      this.lastActiveSpeaker = label;

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
      console.warn("[AudioEngine] Tab audio sharing canceled or failed:", err);
      if (this.onLog && err.name !== 'NotAllowedError') {
        this.onLog('ERROR', `Gagal menambahkan tab audio: ${err.message}`);
      }
      throw err;
    }
  }

  _removeTabAudio(tabId) {
    const entry = this.tabStreams.get(tabId);
    if (!entry) return;

    // 1. Flush sisa buffered speech dari tab ini dan beri label speaker tab yang sesuai (BUKAN 'Anda'!)
    if (this.wasm && this.segmenter) {
      const utteranceLen = this.wasm.catetin_segmenter_flush(
        this.segmenter,
        this.utteranceBufPtr,
        this.utteranceBufCap
      );
      if (utteranceLen > 0 && this.onUtteranceReady) {
        const outOffset = this.utteranceBufPtr / 4;
        const memF32 = this._getMemF32();
        const utterance = new Float32Array(memF32.subarray(outOffset, outOffset + utteranceLen));
        this.onUtteranceReady(utterance, entry.label);
      }
    }

    this.turnSpeechSamples = 0;

    // 2. Putus audio nodes dan hentikan seluruh media track
    try {
      entry.stream.getTracks().forEach(t => t.stop());
      if (entry.processorNode) entry.processorNode.disconnect();
      if (entry.sourceNode) entry.sourceNode.disconnect();
    } catch (_) {}

    this.tabStreams.delete(tabId);

    // 3. Jika tidak ada sumber audio aktif yang tersisa, hentikan deteksi VAD dan kembali ke IDLE
    const hasActiveStreams = this.micEnabled || this.tabStreams.size > 0;
    if (!hasActiveStreams) {
      this.isSpeaking = false;
      if (this.onVadSpeaking) this.onVadSpeaking(false, 0);
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

  _setupStreamPipeline(stream, speakerLabel, onNodeCreated) {
    const sourceNode = this.audioCtx.createMediaStreamSource(stream);
    const bufferSize = 4096;
    const processor = this.audioCtx.createScriptProcessor(bufferSize, 1, 1);

    processor.onaudioprocess = (event) => {
      if (!this.isRunning || this.isPaused) return;

      const inputData = event.inputBuffer.getChannelData(0);

      if (this.wasm && this.resampler && this.segmenter) {
        // Native Rust WASM DSP Pipeline
        const memF32 = this._getMemF32();
        if (!memF32) return;

        const inOffset = this.inBufPtr / 4;
        const len = Math.min(inputData.length, this.inBufCap);
        if (inputData.length <= this.inBufCap) {
          memF32.set(inputData, inOffset);
        } else {
          memF32.set(inputData.subarray(0, len), inOffset);
        }

        // Resample input -> 16kHz mono
        const resampledLen = this.wasm.catetin_resampler_process(
          this.resampler,
          this.inBufPtr,
          len,
          this.resampleBufPtr,
          this.resampleBufCap
        );

        if (resampledLen > 0) {
          // RMS for VU meter and VAD
          const rms = this.wasm.catetin_vad_frame_rms
            ? this.wasm.catetin_vad_frame_rms(0, this.resampleBufPtr, resampledLen)
            : 0;

          if (this.onVolume) {
            this.onVolume(Math.min(1.0, rms * 5.0));
          }

          if (rms > 0.001) {
            this.lastActiveSpeaker = speakerLabel;
          }

          // Feed Rust SpeechSegmenter
          const utteranceLen = this.wasm.catetin_segmenter_feed(
            this.segmenter,
            this.resampleBufPtr,
            resampledLen,
            this.utteranceBufPtr,
            this.utteranceBufCap
          );

          // Update speaking status
          const speaking = this.wasm.catetin_segmenter_is_speaking(this.segmenter) === 1;

          if (speaking) {
            const resOffset = this.resampleBufPtr / 4;
            const resampledView = memF32.subarray(resOffset, resOffset + resampledLen);

            // Pre-staging directly into contiguous buffer (zero chunk-array reallocations)
            if (this.stagingBuffer.byteLength === 0) {
              this.stagingBuffer = new Float32Array(160000);
            }
            if (this.turnSpeechSamples + resampledLen > this.stagingBuffer.length) {
              const keep = Math.min(this.turnSpeechSamples, 56000);
              this.stagingBuffer.copyWithin(0, this.turnSpeechSamples - keep, this.turnSpeechSamples);
              this.turnSpeechSamples = keep;
            }
            this.stagingBuffer.set(resampledView, this.turnSpeechSamples);
            this.turnSpeechSamples += resampledLen;

            // Pass 1: Stream chunk immediately to Transducer (< 50ms) via zero-copy view
            if (this.onTransducerChunk) {
              this.onTransducerChunk(resampledView, speakerLabel, this.turnSpeechSamples / 16000);
            }
          } else {
            if (this.turnSpeechSamples > 0 && this.onTurnFinished) {
              this.onTurnFinished(speakerLabel);
            }
            this.turnSpeechSamples = 0;
            this.lastInterimTime = 0;
          }

          if (speaking !== this.isSpeaking) {
            this.isSpeaking = speaking;
            if (this.onVadSpeaking) this.onVadSpeaking(speaking, rms);
            if (this.onLog) {
              this.onLog('VAD', speaking ? `Suara terdeteksi pada [${speakerLabel}] (RMS: ${rms.toFixed(3)})` : 'Suara berhenti.');
            }
          }

          // Emit real-time turn progress for live transcript visualizer
          if (speaking) {
            const durS = this.turnSpeechSamples / 16000;
            if (this.onSpeechProgress) {
              this.onSpeechProgress({
                speaker: speakerLabel,
                durationS: durS,
                rms
              });
            }

            // Pass 1: Streaming interim hypothesis every 1.2s (sub-second live partials)
            const now = performance.now();
            if (durS >= 1.0 && (now - this.lastInterimTime) >= 1200 && this.onInterimSpeech) {
              this.lastInterimTime = now;
              const maxWindow = 56000; // max 3.5s window
              const startOffset = Math.max(0, this.turnSpeechSamples - maxWindow);
              const interimPcm = this.stagingBuffer.subarray(startOffset, this.turnSpeechSamples);
              this.onInterimSpeech(interimPcm, speakerLabel, durS);
            }

            // Optimasi responsivitas: Jika turn sudah mencapai >= 4.5s dan terdapat penurunan energi (< 0.005),
            // potong turn segera agar Whisper memproses potongan 4.5s tanpa menunggu 9 detik penuh
            if (durS >= 4.5 && rms < 0.005) {
              const earlyLen = this.wasm.catetin_segmenter_flush(
                this.segmenter,
                this.utteranceBufPtr,
                this.utteranceBufCap
              );
              if (earlyLen > 0) {
                if (this.onTurnFinished) this.onTurnFinished(speakerLabel);
                this.turnSpeechSamples = 0;
                this.lastInterimTime = 0;
                const outOffset = this.utteranceBufPtr / 4;
                const latestMem = this._getMemF32();
                const utterance = new Float32Array(latestMem.subarray(outOffset, outOffset + earlyLen));

                if (this.onLog) {
                  this.onLog('QUEUE', `Turn dipotong pada 4.5s [${speakerLabel}] -> Kirim ke ASR Worker`);
                }

                if (this.onUtteranceReady) {
                  this.onUtteranceReady(utterance, speakerLabel);
                }
              }
            }
          }

          // Spoken turn endpointed by natural silence or 9s limit
          if (utteranceLen > 0) {
            if (this.onTurnFinished) this.onTurnFinished(speakerLabel);
            this.turnSpeechSamples = 0;
            this.lastInterimTime = 0;
            const outOffset = this.utteranceBufPtr / 4;
            const latestMem = this._getMemF32();
            const utterance = new Float32Array(latestMem.subarray(outOffset, outOffset + utteranceLen));

            if (this.onLog) {
              const durS = (utteranceLen / 16000).toFixed(1);
              this.onLog('QUEUE', `Turn selesai [${durS}s, ${speakerLabel}] -> Kirim ke ASR Worker`);
            }

            if (this.onUtteranceReady) {
              this.onUtteranceReady(utterance, speakerLabel);
            }
          }
        }
      }
    };

    sourceNode.connect(processor);
    processor.connect(this.silentGainNode);

    if (onNodeCreated) onNodeCreated(processor, sourceNode);
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

  flushUtterance(speakerLabel = this.lastActiveSpeaker || 'Anda') {
    if (this.wasm && this.segmenter) {
      const utteranceLen = this.wasm.catetin_segmenter_flush(
        this.segmenter,
        this.utteranceBufPtr,
        this.utteranceBufCap
      );
      if (utteranceLen > 0) {
        this.turnSpeechSamples = 0;
        this.lastInterimTime = 0;
        const outOffset = this.utteranceBufPtr / 4;
        const latestMem = this._getMemF32();
        const utterance = new Float32Array(latestMem.subarray(outOffset, outOffset + utteranceLen));
        if (this.onUtteranceReady) {
          this.onUtteranceReady(utterance, speakerLabel);
        }
      }
    }
  }

  stopSession() {
    this.isRunning = false;
    this.isPaused = false;
    this.flushUtterance();

    // Stop mic
    if (this.micStream) {
      this.micStream.getTracks().forEach(t => t.stop());
      this.micStream = null;
    }
    if (this.micProcessorNode) {
      try { this.micProcessorNode.disconnect(); } catch (_) {}
      this.micProcessorNode = null;
    }
    if (this.micSourceNode) {
      try { this.micSourceNode.disconnect(); } catch (_) {}
      this.micSourceNode = null;
    }
    this.micEnabled = false;

    // Stop all tabs
    for (const [id, entry] of this.tabStreams) {
      try {
        entry.stream.getTracks().forEach(t => t.stop());
        if (entry.processorNode) entry.processorNode.disconnect();
        if (entry.sourceNode) entry.sourceNode.disconnect();
      } catch (_) {}
    }
    this.tabStreams.clear();
    this.isSpeaking = false;
    this.turnSpeechSamples = 0;

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
    if (this.wasm && this.wasm.catetin_is_hallucination) {
      if (!this.textEncoder) this.textEncoder = new TextEncoder();
      const bytes = this.textEncoder.encode(text);
      const ptr = this.wasm.catetin_alloc_u8(bytes.length);
      const memU8 = this._getMemU8();
      memU8.set(bytes, ptr);
      const result = this.wasm.catetin_is_hallucination(ptr, bytes.length) === 1;
      this.wasm.catetin_free_u8(ptr, bytes.length);
      return result;
    }
    return false;
  }

  // Text formatter powered by Rust WASM
  formatUtterance(text) {
    if (!text) return '';
    if (this.wasm && this.wasm.catetin_format_utterance) {
      try {
        if (!this.textEncoder) this.textEncoder = new TextEncoder();
        const bytes = this.textEncoder.encode(text);
        const inPtr = this.wasm.catetin_alloc_u8(bytes.length);
        const outCap = bytes.length * 2 + 128;
        const outPtr = this.wasm.catetin_alloc_u8(outCap);

        let memU8 = this._getMemU8();
        memU8.set(bytes, inPtr);

        // Rust C-ABI signature: catetin_format_utterance(in_ptr, in_len, prompt_ptr, prompt_len, out_ptr, out_cap)
        const written = this.wasm.catetin_format_utterance(inPtr, bytes.length, 0, 0, outPtr, outCap);
        let formatted = text.trim();
        if (written > 0) {
          if (!this.textDecoder) this.textDecoder = new TextDecoder();
          memU8 = this._getMemU8();
          formatted = this.textDecoder.decode(memU8.subarray(outPtr, outPtr + written)).trim();
        }

        this.wasm.catetin_free_u8(inPtr, bytes.length);
        this.wasm.catetin_free_u8(outPtr, outCap);
        return formatted || text.trim();
      } catch (_) {
        return text.trim();
      }
    }
    return text.trim();
  }
}
