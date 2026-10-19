// audio-engine.js - WebAudio Capture & catetin-wasm DSP Driver (Ponytail: native WebAudio + WASM)

export class AudioEngine {
  constructor() {
    this.audioCtx = null;
    this.micStream = null;
    this.tabStream = null;
    this.processor = null;
    this.isRunning = false;
    this.isPaused = false;

    // Callbacks
    this.onVolume = null;         // (level: number) => void
    this.onLiveChunk = null;      // (pcm16k: Float32Array) => void
    this.onUtteranceReady = null; // (utterance: Float32Array, isRemote: boolean) => void
    this.onVadSpeaking = null;    // (isSpeaking: boolean) => void

    // Internal VAD state (30ms at 16kHz = 480 samples)
    this.frameSize = 480;
    this.frameBuffer = [];
    this.speechBuffer = [];
    this.silenceFrames = 0;
    this.isSpeaking = false;
    this.energyThreshold = 0.015;
    this.maxSilenceFrames = 20; // ~600ms silence marks end of turn
  }

  async startMic() {
    if (this.isRunning) return;

    this.audioCtx = new (window.AudioContext || window.webkitAudioContext)({
      sampleRate: 48000
    });

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
        throw new Error("Pilih opsi 'Share Tab Audio' agar suara meeting tertangkap.");
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
    
    // ScriptProcessorNode (ponytail: native 4096 buffer, no build step needed)
    const bufferSize = 4096;
    const processor = this.audioCtx.createScriptProcessor(bufferSize, 1, 1);

    processor.onaudioprocess = (event) => {
      if (!this.isRunning || this.isPaused) return;

      const inputData = event.inputBuffer.getChannelData(0);
      
      // 1. Compute RMS Volume for VU Meter
      let sum = 0;
      for (let i = 0; i < inputData.length; i++) {
        sum += inputData[i] * inputData[i];
      }
      const rms = Math.sqrt(sum / inputData.length);
      if (this.onVolume) {
        this.onVolume(Math.min(1.0, rms * 5.0));
      }

      // 2. Downsample from audioCtx.sampleRate (48k / 44.1k) to 16kHz mono
      const pcm16k = this._downsampleTo16k(inputData, this.audioCtx.sampleRate);
      if (this.onLiveChunk) {
        this.onLiveChunk(pcm16k);
      }

      // 3. 30ms VAD Framing & Endpointing
      this._processVad(pcm16k, isRemote);
    };

    source.connect(processor);
    processor.connect(this.audioCtx.destination);
    this.processor = processor;
  }

  _downsampleTo16k(samples, srcRate) {
    if (srcRate === 16000) return samples;
    const ratio = srcRate / 16000;
    const outLength = Math.floor(samples.length / ratio);
    const result = new Float32Array(outLength);

    for (let i = 0; i < outLength; i++) {
      const srcIdx = Math.floor(i * ratio);
      result[i] = samples[srcIdx];
    }
    return result;
  }

  _processVad(samples, isRemote) {
    for (let i = 0; i < samples.length; i++) {
      this.frameBuffer.push(samples[i]);

      if (this.frameBuffer.length >= this.frameSize) {
        const frame = this.frameBuffer.splice(0, this.frameSize);
        
        let frameEnergy = 0;
        for (let j = 0; j < frame.length; j++) {
          frameEnergy += Math.abs(frame[j]);
        }
        frameEnergy /= frame.length;

        const isVoice = frameEnergy > this.energyThreshold;

        if (isVoice) {
          this.silenceFrames = 0;
          if (!this.isSpeaking) {
            this.isSpeaking = true;
            if (this.onVadSpeaking) this.onVadSpeaking(true);
          }
          this.speechBuffer.push(...frame);
        } else {
          if (this.isSpeaking) {
            this.silenceFrames++;
            this.speechBuffer.push(...frame);

            // Endpoint detected (> 600ms silence after speech)
            if (this.silenceFrames > this.maxSilenceFrames) {
              this.isSpeaking = false;
              if (this.onVadSpeaking) this.onVadSpeaking(false);

              // Minimum 0.5s speech length to filter coughs/clicks
              if (this.speechBuffer.length >= 8000) {
                const utterance = new Float32Array(this.speechBuffer);
                if (this.onUtteranceReady) {
                  this.onUtteranceReady(utterance, isRemote);
                }
              }
              this.speechBuffer = [];
              this.silenceFrames = 0;
            }
          }
        }
      }
    }
  }

  pause() {
    this.isPaused = true;
    this.frameBuffer = [];
    this.speechBuffer = [];
    this.isSpeaking = false;
    if (this.onVadSpeaking) this.onVadSpeaking(false);
    if (this.onVolume) this.onVolume(0);
  }

  resume() {
    this.isPaused = false;
  }

  stop() {
    this.isRunning = false;
    this.isPaused = false;
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
    this.frameBuffer = [];
    this.speechBuffer = [];
    if (this.onVolume) this.onVolume(0);
    if (this.onVadSpeaking) this.onVadSpeaking(false);
  }
}
