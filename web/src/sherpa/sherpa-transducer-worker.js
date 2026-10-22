// sherpa-transducer-worker.js - Pass 1 Streaming Transducer Dedicated Worker (WASM SIMD)
// Executes C-ABI SherpaOnnxOnlineRecognizer for Zipformer2 ID/EN models in real-time (< 50ms)

let recognizer = null;
let stream = null;
let isReady = false;
let currentLanguage = null;
let lastPartialText = '';

let resolveWasmInit;
const wasmInitPromise = new Promise((resolve) => {
  resolveWasmInit = resolve;
});

self.Module = {
  locateFile: function(fileName) {
    if (fileName.endsWith('.wasm')) {
      return './sherpa-onnx-wasm-main-asr.wasm';
    }
    return fileName;
  },
  onRuntimeInitialized: function() {
    resolveWasmInit();
  }
};

// Load Sherpa-ONNX Emscripten runtime and JavaScript API
importScripts('./sherpa-onnx-wasm-main-asr.js', './sherpa-onnx-asr.js');

async function initTransducer(lang = 'id') {
  if (lang === 'none') {
    if (stream) { stream.free(); stream = null; }
    if (recognizer) { recognizer.free(); recognizer = null; }
    isReady = false;
    currentLanguage = 'none';
    self.postMessage({ type: 'status', data: { status: 'disabled', message: 'Transducer dinonaktifkan.' } });
    return;
  }

  await wasmInitPromise;

  if (isReady && currentLanguage === lang) {
    return; // Already initialized for this language
  }

  isReady = false;
  self.postMessage({
    type: 'status',
    data: { status: 'loading', message: `Memuat model Transducer Streaming (${lang.toUpperCase()})...` }
  });

  const modelDir = lang === 'en'
    ? '/models/sherpa-onnx-streaming-zipformer-en-2023-06-26'
    : '/models/sherpa-onnx-streaming-zipformer2-id';

  const files = ['encoder.onnx', 'decoder.onnx', 'joiner.onnx', 'tokens.txt'];
  try {
    for (const f of files) {
      const url = `${modelDir}/${f}`;
      const res = await fetch(url);
      if (!res.ok) {
        throw new Error(`Gagal mengunduh berkas ${f} dari ${url} (${res.status})`);
      }
      const buf = await res.arrayBuffer();
      self.Module.FS.writeFile(`/${f}`, new Uint8Array(buf));
    }

    if (stream) { stream.free(); stream = null; }
    if (recognizer) { recognizer.free(); recognizer = null; }

    const config = {
      featConfig: { sampleRate: 16000, featureDim: 80 },
      modelConfig: {
        transducer: {
          encoder: '/encoder.onnx',
          decoder: '/decoder.onnx',
          joiner: '/joiner.onnx'
        },
        tokens: '/tokens.txt',
        numThreads: 1,
        provider: 'cpu',
        debug: 0
      },
      decodingMethod: 'greedy_search',
      maxActivePaths: 4,
      enableEndpoint: 1,
      rule1MinTrailingSilence: 2.4,
      rule2MinTrailingSilence: 1.2,
      rule3MinUtteranceLength: 20
    };

    recognizer = createOnlineRecognizer(self.Module, config);
    stream = recognizer.createStream();

    // Pre-warm with dummy frame (eliminates first-frame jitters)
    const dummy = new Float32Array(2560);
    stream.acceptWaveform(16000, dummy);
    while (recognizer.isReady(stream)) {
      recognizer.decode(stream);
    }
    recognizer.reset(stream);

    isReady = true;
    currentLanguage = lang;
    lastPartialText = '';

    self.postMessage({
      type: 'status',
      data: { status: 'ready', message: `Transducer Streaming ${lang.toUpperCase()} siap (< 50ms).` }
    });
  } catch (err) {
    isReady = false;
    self.postMessage({
      type: 'status',
      data: { status: 'error', message: `Transducer error: ${err.message}` }
    });
  }
}

self.onmessage = async (e) => {
  const { type, data } = e.data;

  if (type === 'init') {
    await initTransducer(data?.language || 'id');
  } else if (type === 'accept_chunk') {
    if (!isReady || !recognizer || !stream) return;

    const { pcm16k, speaker, durationS } = data;
    if (!pcm16k || pcm16k.length === 0) return;

    const t0 = performance.now();
    stream.acceptWaveform(16000, pcm16k);
    while (recognizer.isReady(stream)) {
      recognizer.decode(stream);
    }
    const decodeTimeMs = performance.now() - t0;

    const res = recognizer.getResult(stream);
    const text = (res && res.text) ? res.text.trim() : '';

    if (text && text !== lastPartialText) {
      lastPartialText = text;
      self.postMessage({
        type: 'partial_result',
        data: {
          text,
          speaker,
          durationS,
          decodeTimeMs
        }
      });
    }
  } else if (type === 'reset_stream') {
    if (isReady && recognizer && stream) {
      recognizer.reset(stream);
      lastPartialText = '';
    }
  }
};
