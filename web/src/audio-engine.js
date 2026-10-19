// audio-engine.js - WebAudio Capture & catetin.wasm DSP Driver (Ponytail: native WebAudio + pure Rust WASM)

export class AudioEngine {
  constructor() {
    this.audioCtx = null;
    this.micStream = null;
    this.tabStream = null;
    this.processor = null;
    this.isRunning = false;
    this.isPaused = false;

    // WASM Module & DSP Handles
    this.wasm = null;
    this.resampler = null;
    this.segmenter = null;
    this.clusterer = null;

    // WASM Memory Pointers
    this.inBufPtr = 0;
    this.inBufCap = 8192;
    this.resampleBufPtr = 0;
    this.resampleBufCap = 16384;
    this.utteranceBufPtr = 0;
    this.utteranceBufCap = 160000; // 10s at 16kHz

    // Callbacks
    this.onVolume = null;         // (level: number) => void
    this.onUtteranceReady = null; // (utterance: Float32Array, isRemote: boolean) => void
    this.onVadSpeaking = null;    // (isSpeaking: boolean) => void
    this.isSpeaking = false;
  }

  async initWasm(wasmPath = './catetin.wasm') {
    if (this.wasm) return;
    try {
      const response = await fetch(wasmPath);
      let wasmModule;
      if (WebAssembly.instantiateStreaming) {
        try {
          wasmModule = await WebAssembly.instantiateStreaming(response);
        } catch {
          const bytes = await response.arrayBuffer();
          wasmModule = await WebAssembly.instantiate(bytes);
        }
      } else {
        const bytes = await response.arrayBuffer();
        wasmModule = await WebAssembly.instantiate(bytes);
      }

      this.wasm = wasmModule.instance.exports;

      // Allocate WASM memory buffers
      this.inBufPtr = this.wasm.catetin_alloc_f32(this.inBufCap);
      this.resampleBufPtr = this.wasm.catetin_alloc_f32(this.resampleBufCap);
      this.utteranceBufPtr = this.wasm.catetin_alloc_f32(this.utteranceBufCap);

      // Initialize DSP instances
      this.segmenter = this.wasm.catetin_segmenter_new(16000);
      this.clusterer = this.wasm.catetin_clusterer_new(0.65, 0.10);
      console.log("[AudioEngine] catetin.wasm DSP engine successfully initialized.");
    } catch (err) {
      console.warn("[AudioEngine] Could not initialize catetin.wasm, falling back to WebAudio JS DSP:", err);
      this.wasm = null;
    }
  }

  async startMic() {
    if (this.isRunning) return;

    await this.initWasm();

    this.audioCtx = new (window.AudioContext || window.webkitAudioContext)({
      sampleRate: 48000
    });

    if (this.wasm) {
      if (this.resampler) this.wasm.catetin_resampler_free(this.resampler);
      this.resampler = this.wasm.catetin_resampler_new(this.audioCtx.sampleRate, 1, 16000);
    }

    this.micStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true
      }
    });

    this._setupPipeline(this.micStream, false);
    this.isRunning = true;
    this.isPaused = false;
  }

  async addTabAudio() {
    try {
      this.tabStream = await navigator.mediaDevices.getDisplayMedia({
        video: true,
        audio: true
      });

      const audioTrack = this.tabStream.getAudioTracks()[0];
      if (!audioTrack) {
        this.tabStream.getTracks().forEach(t => t.stop());
        throw new Error("Pilih opsi 'Share Tab Audio' agar suara rapat terdengar.");
      }

      if (!this.audioCtx) {
        this.audioCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 48000 });
      }

      this._setupPipeline(this.tabStream, true);

      audioTrack.onended = () => {
        this.tabStream = null;
      };
    } catch (err) {
      console.warn("Tab audio sharing canceled or failed:", err);
      throw err;
    }
  }

  _setupPipeline(stream, isRemote) {
    const source = this.audioCtx.createMediaStreamSource(stream);
    const bufferSize = 4096;
    const processor = this.audioCtx.createScriptProcessor(bufferSize, 1, 1);

    processor.onaudioprocess = (event) => {
      if (!this.isRunning || this.isPaused) return;

      const inputData = event.inputBuffer.getChannelData(0);
      
      if (this.wasm && this.resampler && this.segmenter) {
        // --- 100% Native Rust WASM DSP Pipeline ---
        const memF32 = new Float32Array(this.wasm.memory.buffer);
        
        // 1. Copy WebAudio input into WASM buffer
        const len = Math.min(inputData.length, this.inBufCap);
        const inOffset = this.inBufPtr / 4;
        memF32.set(inputData.subarray(0, len), inOffset);

        // 2. Resample 48kHz -> 16kHz mono using Rust 5-tap anti-aliasing filter
        const resampledLen = this.wasm.catetin_resampler_process(
          this.resampler,
          this.inBufPtr,
          len,
          this.resampleBufPtr,
          this.resampleBufCap
        );

        if (resampledLen > 0) {
          // 3. RMS Volume calculation for VU meter
          const rms = this.wasm.catetin_vad_rms(this.resampleBufPtr, resampledLen);
          if (this.onVolume) {
            this.onVolume(Math.min(1.0, rms * 5.0));
          }

          // 4. Feed Rust SpeechSegmenter (adaptive noise floor + hangover + endpointing)
          const utteranceLen = this.wasm.catetin_segmenter_feed(
            this.segmenter,
            this.resampleBufPtr,
            resampledLen,
            this.utteranceBufPtr,
            this.utteranceBufCap
          );

          // 5. Update speaking state indicator
          const speaking = this.wasm.catetin_segmenter_is_speaking(this.segmenter) === 1;
          if (speaking !== this.isSpeaking) {
            this.isSpeaking = speaking;
            if (this.onVadSpeaking) this.onVadSpeaking(speaking);
          }

          // 6. Utterance endpoint reached!
          if (utteranceLen > 0) {
            const outOffset = this.utteranceBufPtr / 4;
            // Refresh memF32 in case memory grew
            const latestMem = new Float32Array(this.wasm.memory.buffer);
            const utterance = new Float32Array(utteranceLen);
            utterance.set(latestMem.subarray(outOffset, outOffset + utteranceLen));

            if (this.onUtteranceReady) {
              this.onUtteranceReady(utterance, isRemote);
            }
          }
        }
      } else {
        // --- Fallback Pure JS DSP ---
        let sum = 0;
        for (let i = 0; i < inputData.length; i++) sum += inputData[i] * inputData[i];
        const rms = Math.sqrt(sum / inputData.length);
        if (this.onVolume) this.onVolume(Math.min(1.0, rms * 5.0));

        const pcm16k = this._jsDownsample(inputData, this.audioCtx.sampleRate);
        this._jsProcessVad(pcm16k, isRemote);
      }
    };

    source.connect(processor);
    processor.connect(this.audioCtx.destination);
    this.processor = processor;
  }

  _jsDownsample(samples, srcRate) {
    if (srcRate === 16000) return samples;
    const ratio = srcRate / 16000;
    const outLength = Math.floor(samples.length / ratio);
    const result = new Float32Array(outLength);
    for (let i = 0; i < outLength; i++) {
      result[i] = samples[Math.floor(i * ratio)];
    }
    return result;
  }

  _jsProcessVad(samples, isRemote) {
    // Basic fallback VAD accumulator
    if (!this._jsBuf) this._jsBuf = [];
    for (let i = 0; i < samples.length; i++) {
      this._jsBuf.push(samples[i]);
    }
    if (this._jsBuf.length >= 16000 * 3) { // 3-second chunk
      const chunk = new Float32Array(this._jsBuf);
      this._jsBuf = [];
      if (this.onUtteranceReady) {
        this.onUtteranceReady(chunk, isRemote);
      }
    }
  }

  pause() {
    this.isPaused = true;
    this.flushUtterance();
    if (this.onVadSpeaking) this.onVadSpeaking(false);
    if (this.onVolume) this.onVolume(0);
  }

  resume() {
    this.isPaused = false;
  }

  flushUtterance() {
    if (this.wasm && this.segmenter) {
      const utteranceLen = this.wasm.catetin_segmenter_flush(
        this.segmenter,
        this.utteranceBufPtr,
        this.utteranceBufCap
      );
      if (utteranceLen > 0) {
        const outOffset = this.utteranceBufPtr / 4;
        const latestMem = new Float32Array(this.wasm.memory.buffer);
        const utterance = new Float32Array(utteranceLen);
        utterance.set(latestMem.subarray(outOffset, outOffset + utteranceLen));
        if (this.onUtteranceReady) {
          this.onUtteranceReady(utterance, false);
        }
      }
    }
  }

  stop() {
    this.isRunning = false;
    this.isPaused = false;
    this.flushUtterance();

    if (this.micStream) {
      this.micStream.getTracks().forEach(t => t.stop());
      this.micStream = null;
    }
    if (this.tabStream) {
      this.tabStream.getTracks().forEach(t => t.stop());
      this.tabStream = null;
    }
    if (this.audioCtx) {
      this.audioCtx.close();
      this.audioCtx = null;
    }
    if (this.onVolume) this.onVolume(0);
    if (this.onVadSpeaking) this.onVadSpeaking(false);
  }

  isHallucination(text) {
    if (!this.wasm) return false;
    const enc = new TextEncoder().encode(text);
    const ptr = this.wasm.catetin_alloc_u8(enc.length);
    new Uint8Array(this.wasm.memory.buffer).set(enc, ptr);
    const res = this.wasm.catetin_is_hallucination(ptr, enc.length) === 1;
    this.wasm.catetin_free_u8(ptr, enc.length);
    return res;
  }

  formatUtterance(text, prompt = '') {
    if (!this.wasm) return text.trim();
    const encText = new TextEncoder().encode(text);
    const encPrompt = new TextEncoder().encode(prompt);

    const ptrText = this.wasm.catetin_alloc_u8(encText.length);
    new Uint8Array(this.wasm.memory.buffer).set(encText, ptrText);

    const ptrPrompt = this.wasm.catetin_alloc_u8(encPrompt.length);
    new Uint8Array(this.wasm.memory.buffer).set(encPrompt, ptrPrompt);

    const outCap = encText.length + 64;
    const outPtr = this.wasm.catetin_alloc_u8(outCap);

    const written = this.wasm.catetin_format_utterance(
      ptrText, encText.length,
      ptrPrompt, encPrompt.length,
      outPtr, outCap
    );

    let result = text.trim();
    if (written > 0) {
      const slice = new Uint8Array(this.wasm.memory.buffer, outPtr, written);
      result = new TextDecoder().decode(slice);
    }

    this.wasm.catetin_free_u8(ptrText, encText.length);
    this.wasm.catetin_free_u8(ptrPrompt, encPrompt.length);
    this.wasm.catetin_free_u8(outPtr, outCap);

    return result;
  }
}
