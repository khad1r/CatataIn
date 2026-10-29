// engines/pass2-webnn.js - WebNN OS-Native Finalizer Engine (Option B)
// DirectML on Windows, CoreML on macOS, or NPU.

import { Pass2Engine } from './pass2-interface.js';

export class Pass2WebNnEngine extends Pass2Engine {
  constructor() {
    super('webnn', 'WebNN (OS DirectML / NPU Native)', 'NPU/OS-DirectML');
    this.available = false;
  }

  async init(modelName = 'Xenova/whisper-tiny', language = 'id', onProgress = null) {
    throw new Error('Engine Pass 2 WebNN (DirectML Native) belum tersedia.');
  }

  async warmup() {
    throw new Error('Engine Pass 2 WebNN (DirectML Native) belum tersedia.');
  }

  async transcribe(pcm16k, options = {}) {
    throw new Error('Engine Pass 2 WebNN (DirectML Native) belum tersedia.');
  }

  async dispose() {
    this.isReady = false;
  }
}
