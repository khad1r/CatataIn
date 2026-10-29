// engines/pass2-direct-ort.js - Direct Local ONNX Runtime Web Finalizer Engine (Option D)
// High-performance direct C-ABI execution of INT8 quantized Whisper base models bypassing HuggingFace CDN.
// Supports both Browser (with CacheStorage) and Node.js environments.

import { Pass2Engine, resolveWhisperLanguage } from './pass2-interface.js';
import { computeCpuMelSpectrogram, WebGpuMelExtractor } from './mel-spectrogram.js';

const DIRECT_ORT_CACHE_NAME = 'catetin-direct-ort-models-v1';

const SPECIAL_TOKENS = {
  SOT: 50258,
  EOT: 50257,
  TRANSCRIBE: 50359,
  TRANSLATE: 50358,
  NO_TIMESTAMPS: 50363,
  NO_SPEECH: 50362
};

const LANGUAGE_CODE_TO_TOKEN = {
  'id': 50275,
  'indonesian': 50275,
  'en': 50259,
  'english': 50259,
  'auto': 50275
};

function base64ToBytes(b64) {
  if (!b64 || b64 === '=') return new Uint8Array(0);
  if (typeof Buffer !== 'undefined') {
    return Uint8Array.from(Buffer.from(b64, 'base64'));
  }
  try {
    const binString = atob(b64);
    const len = binString.length;
    const bytes = new Uint8Array(len);
    for (let i = 0; i < len; i++) {
      bytes[i] = binString.charCodeAt(i);
    }
    return bytes;
  } catch (err) {
    void err;
    return new Uint8Array(0);
  }
}

function parseTokens(text) {
  const lines = text.trim().split('\n');
  const map = new Map();
  for (const line of lines) {
    const lastSpace = line.lastIndexOf(' ');
    if (lastSpace === -1) continue;
    const b64 = line.substring(0, lastSpace);
    const id = parseInt(line.substring(lastSpace + 1), 10);
    map.set(id, base64ToBytes(b64));
  }
  return map;
}

function decodeTokens(tokenIds, tokenMap) {
  const byteArrays = [];
  let totalLen = 0;
  for (const id of tokenIds) {
    if (id >= SPECIAL_TOKENS.EOT) continue; // Abaikan special tokens & timestamps
    const bytes = tokenMap.get(id);
    if (bytes && bytes.length > 0) {
      byteArrays.push(bytes);
      totalLen += bytes.length;
    }
  }
  const merged = new Uint8Array(totalLen);
  let offset = 0;
  for (const b of byteArrays) {
    merged.set(b, offset);
    offset += b.length;
  }
  return new TextDecoder('utf-8', { fatal: false }).decode(merged).trim();
}

async function loadAssetBytes(subPath) {
  const isNode = typeof window === 'undefined' && typeof process !== 'undefined' && Boolean(process?.versions?.node);

  if (isNode) {
    const fs = await import('node:fs');
    const path = await import('node:path');
    const { fileURLToPath } = await import('node:url');

    const thisDir = path.dirname(fileURLToPath(import.meta.url));
    const projectRoot = path.resolve(thisDir, '../../..');

    const cleanPath = subPath.replace(/^\/+/, '');
    const candidates = [
      path.resolve(projectRoot, 'dist', cleanPath),
      path.resolve(projectRoot, cleanPath),
      path.resolve(projectRoot, 'web', cleanPath)
    ];

    for (const c of candidates) {
      if (fs.existsSync(c)) {
        return fs.promises.readFile(c);
      }
    }
    throw new Error(`[Direct ORT Node] File asset tidak ditemukan: ${subPath} (mencoba: ${candidates.join(', ')})`);
  }

  // Browser environment
  const url = subPath.startsWith('/') ? subPath : `/${subPath}`;
  let cache = null;
  if (typeof caches !== 'undefined') {
    try {
      cache = await caches.open(DIRECT_ORT_CACHE_NAME);
      const cachedResponse = await cache.match(url);
      if (cachedResponse && cachedResponse.ok) {
        return new Uint8Array(await cachedResponse.arrayBuffer());
      }
    } catch (cacheErr) {
      console.warn('[Direct ORT] Cache read dilewati:', cacheErr);
    }
  }

  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(`HTTP ${res.status} saat mengunduh aset Direct ORT: ${url}`);
  }

  if (cache) {
    try {
      await cache.put(url, res.clone());
    } catch (cachePutErr) {
      console.warn('[Direct ORT] Gagal menulis ke cache:', cachePutErr);
    }
  }

  const buf = await res.arrayBuffer();
  return new Uint8Array(buf);
}

export class Pass2DirectOrtEngine extends Pass2Engine {
  constructor() {
    super('direct-ort', 'Direct ORT Web (Local Whisper INT8)', 'CPU / Direct ORT');
    this.available = true;
    this.isReady = false;
    this.backendName = 'Direct ORT Web (C-ABI INT8)';

    this.wasm = null;
    this.encSession = null;
    this.decSession = null;
    this.encOpts = null;
    this.decOpts = null;
    this.tokenMap = null;
    this.melExtractor = null;
    this.currentLanguage = 'id';

    // Model WASM Heap pointers
    this.encModelPtr = null;
    this.decModelPtr = null;

    // Pre-allocated runtime WASM heap pointers
    this.encInNamesArr = null;
    this.encOutNamesArr = null;
    this.encInValuesArr = null;
    this.encOutValuesArr = null;
    this.melDimsPtr = null;

    this.decInNamesArr = null;
    this.decOutNamesArr = null;
    this.decInValuesArr = null;
    this.decOutValuesArr = null;
    this.cacheDimsPtr = null;
    this.offsetDimsPtr = null;
    this.offsetDataPtr = null;
    this.tokenDimsPtr = null;
    this.step1TokenDataPtr = null;
    this.infoPtr = null;
  }

  async init(modelId = 'Xenova/whisper-base', language = 'id', onProgress = null) {
    await this.dispose();

    // Zero Silent Fallback check: hanya model whisper-base lokal yang tersedia
    const isTinyOrSmall = typeof modelId === 'string' && (modelId.includes('tiny') || modelId.includes('small'));
    const isBase = typeof modelId === 'string' && (modelId.includes('base') || modelId.includes('sherpa-onnx-whisper-base'));
    if (isTinyOrSmall || !isBase) {
      this.isReady = false;
      throw new Error(`Direct ORT saat ini hanya menyediakan bobot lokal untuk whisper-base (diminta: ${modelId}). Zero Silent Fallback aktif.`);
    }

    this.currentLanguage = language || 'id';

    if (onProgress) onProgress({ status: 'progress', progress: 0.05, message: 'Memuat runtime ORT WASM SIMD...' });

    // 1. Inisialisasi runtime WASM SIMD C-ABI
    const isNode = typeof window === 'undefined' && typeof process !== 'undefined' && Boolean(process?.versions?.node);
    let ortWasmFactory;

    if (isNode) {
      const path = await import('node:path');
      const { fileURLToPath, pathToFileURL } = await import('node:url');
      const thisDir = path.dirname(fileURLToPath(import.meta.url));
      const jsepPath = path.resolve(thisDir, '../../vendor/transformers/ort-wasm-simd-threaded.jsep.mjs');
      const mod = await import(pathToFileURL(jsepPath).href);
      ortWasmFactory = mod.default;
    } else {
      const jsepUrl = new URL('../../vendor/transformers/ort-wasm-simd-threaded.jsep.mjs', import.meta.url).href;
      const mod = await import(jsepUrl);
      ortWasmFactory = mod.default;
    }

    this.wasm = await ortWasmFactory();
    this.wasm._OrtInit(1, 3); // 1 thread, log level 3 (quiet)

    // 2. Muat kamus token base-tokens.txt
    if (onProgress) onProgress({ status: 'progress', progress: 0.2, message: 'Memuat kamus token Whisper...' });
    const tokensBytes = await loadAssetBytes('models/sherpa-onnx-whisper-base/base-tokens.txt');
    const tokensText = new TextDecoder('utf-8', { fatal: false }).decode(tokensBytes);
    this.tokenMap = parseTokens(tokensText);

    // 3. Muat Encoder Model (base-encoder.int8.onnx)
    if (onProgress) onProgress({ status: 'progress', progress: 0.35, message: 'Memuat model base-encoder.int8.onnx...' });
    const encBytes = await loadAssetBytes('models/sherpa-onnx-whisper-base/base-encoder.int8.onnx');
    this.encModelPtr = this.wasm._malloc(encBytes.byteLength);
    this.wasm.HEAPU8.set(encBytes, this.encModelPtr);

    // Session options: graphOptLevel=3, cpuMemArena=1, memPattern=1, execMode=0, enableProfiling=0
    this.encOpts = this.wasm._OrtCreateSessionOptions(3, 1, 1, 0, 0, 0, 0, 3, 0, 0);
    this.encSession = await this.wasm._OrtCreateSession(this.encModelPtr, encBytes.byteLength, this.encOpts);
    if (!this.encSession) {
      throw new Error('Gagal membuat ONNX Runtime Session untuk base-encoder.');
    }

    // 4. Muat Decoder Model (base-decoder.int8.onnx)
    if (onProgress) onProgress({ status: 'progress', progress: 0.65, message: 'Memuat model base-decoder.int8.onnx...' });
    const decBytes = await loadAssetBytes('models/sherpa-onnx-whisper-base/base-decoder.int8.onnx');
    this.decModelPtr = this.wasm._malloc(decBytes.byteLength);
    this.wasm.HEAPU8.set(decBytes, this.decModelPtr);

    this.decOpts = this.wasm._OrtCreateSessionOptions(3, 1, 1, 0, 0, 0, 0, 3, 0, 0);
    this.decSession = await this.wasm._OrtCreateSession(this.decModelPtr, decBytes.byteLength, this.decOpts);
    if (!this.decSession) {
      throw new Error('Gagal membuat ONNX Runtime Session untuk base-decoder.');
    }

    // 5. Alokasi struktur pointer permanen C-ABI untuk eksekusi cepat
    this._allocateRuntimeStructures();

    // 6. Inisialisasi WebGpuMelExtractor jika tersedia di Browser
    if (typeof navigator !== 'undefined' && navigator.gpu) {
      try {
        this.melExtractor = new WebGpuMelExtractor();
        await this.melExtractor.init();
      } catch (gpuErr) {
        console.warn('[Direct ORT] WebGPU Mel tidak aktif, beralih ke CPU Mel:', gpuErr);
        this.melExtractor = null;
      }
    }

    // 7. Warmup model
    if (onProgress) onProgress({ status: 'progress', progress: 0.9, message: 'Menjalankan warmup pipeline...' });
    await this.warmup();

    this.isReady = true;
    if (onProgress) onProgress({ status: 'ready', progress: 1.0, message: 'Direct ORT Engine siap digunakan.' });
  }

  _allocateRuntimeStructures() {
    const wasm = this.wasm;

    // Encoder IO
    this.encInNamesArr = wasm._malloc(4);
    wasm.HEAP32[this.encInNamesArr >> 2] = wasm._OrtGetInputName(this.encSession, 0);

    this.encOutNamesArr = wasm._malloc(8);
    wasm.HEAP32[this.encOutNamesArr >> 2] = wasm._OrtGetOutputName(this.encSession, 0);
    wasm.HEAP32[(this.encOutNamesArr >> 2) + 1] = wasm._OrtGetOutputName(this.encSession, 1);

    this.encInValuesArr = wasm._malloc(4);
    this.encOutValuesArr = wasm._malloc(8);

    this.melDimsPtr = wasm._malloc(12);
    wasm.HEAP32[this.melDimsPtr >> 2] = 1;
    wasm.HEAP32[(this.melDimsPtr >> 2) + 1] = 80;
    wasm.HEAP32[(this.melDimsPtr >> 2) + 2] = 3000;

    // Decoder IO
    this.decInNamesArr = wasm._malloc(24);
    for (let i = 0; i < 6; i++) {
      wasm.HEAP32[(this.decInNamesArr >> 2) + i] = wasm._OrtGetInputName(this.decSession, i);
    }

    this.decOutNamesArr = wasm._malloc(12);
    for (let i = 0; i < 3; i++) {
      wasm.HEAP32[(this.decOutNamesArr >> 2) + i] = wasm._OrtGetOutputName(this.decSession, i);
    }

    this.decInValuesArr = wasm._malloc(24);
    this.decOutValuesArr = wasm._malloc(12);

    this.cacheDimsPtr = wasm._malloc(16);
    wasm.HEAP32[this.cacheDimsPtr >> 2] = 6;
    wasm.HEAP32[(this.cacheDimsPtr >> 2) + 1] = 1;
    wasm.HEAP32[(this.cacheDimsPtr >> 2) + 2] = 448;
    wasm.HEAP32[(this.cacheDimsPtr >> 2) + 3] = 512;

    this.offsetDimsPtr = wasm._malloc(4);
    wasm.HEAP32[this.offsetDimsPtr >> 2] = 1;
    this.offsetDataPtr = wasm._malloc(8);

    this.tokenDimsPtr = wasm._malloc(8);
    this.step1TokenDataPtr = wasm._malloc(8);
    this.infoPtr = wasm._malloc(16);
  }

  async warmup() {
    if (!this.encSession || !this.decSession) {
      throw new Error('Pass2DirectOrtEngine belum terinisialisasi lengkap untuk warmup.');
    }
    // Eksekusi dummy pass 0.1s audio untuk kompilasi kernel dan pemanasan heap
    const dummyPcm = new Float32Array(1600);
    await this.transcribe(dummyPcm, { maxNewTokens: 1 });
  }

  async transcribe(pcm16k, options = {}) {
    if (!this.encSession || !this.decSession) {
      throw new Error('Pass2DirectOrtEngine belum diinisialisasi.');
    }
    if (!pcm16k || !(pcm16k instanceof Float32Array) || pcm16k.length === 0) {
      throw new Error('Pass2DirectOrtEngine: transcribe membutuhkan Float32Array PCM 16kHz yang tidak kosong.');
    }

    const tStart = performance.now();
    const wasm = this.wasm;

    // 1. Ekstraksi Log-Mel Spectrogram (WebGPU jika ada, atau CPU)
    let melData;
    let melLatencyMs = 0;
    const tMelStart = performance.now();

    if (this.melExtractor && this.melExtractor.isReady) {
      const melRes = await this.melExtractor.extract(pcm16k);
      melData = melRes.data;
      melLatencyMs = melRes.latencyMs || Math.round(performance.now() - tMelStart);
    } else {
      const melRes = computeCpuMelSpectrogram(pcm16k);
      melData = melRes.data;
      melLatencyMs = melRes.latencyMs || Math.round(performance.now() - tMelStart);
    }

    const tDecodeStart = performance.now();

    // 2. Jalankan Encoder
    const melSize = 80 * 3000;
    const melDataPtr = wasm._malloc(melSize * 4);
    wasm.HEAPF32.set(melData, melDataPtr >> 2);

    const melTensor = wasm._OrtCreateTensor(1, melDataPtr, melSize * 4, this.melDimsPtr, 3, 1);
    wasm.HEAP32[this.encInValuesArr >> 2] = melTensor;
    wasm.HEAP32[this.encOutValuesArr >> 2] = 0;
    wasm.HEAP32[(this.encOutValuesArr >> 2) + 1] = 0;

    let crossK = 0;
    let crossV = 0;
    try {
      const encRet = await wasm._OrtRun(this.encSession, this.encInNamesArr, this.encInValuesArr, 1, this.encOutNamesArr, 2, this.encOutValuesArr, 0);
      if (encRet !== 0) throw new Error(`Encoder _OrtRun gagal dengan kode ${encRet}`);
      crossK = wasm.HEAP32[this.encOutValuesArr >> 2];
      crossV = wasm.HEAP32[(this.encOutValuesArr >> 2) + 1];
    } finally {
      wasm._OrtReleaseTensor(melTensor);
      wasm._free(melDataPtr);
    }

    // 3. Autoregressive Greedy Decoder Loop
    const generatedTokens = [];
    const maxTokens = options.maxNewTokens || 224;

    const langKey = options.language || this.currentLanguage || 'id';
    const langToken = LANGUAGE_CODE_TO_TOKEN[langKey.toLowerCase()] || SPECIAL_TOKENS.SOT;
    const taskToken = options.task === 'translate' ? SPECIAL_TOKENS.TRANSLATE : SPECIAL_TOKENS.TRANSCRIBE;

    // Prompt awal: [SOT(50258), LANG(50275), TRANSCRIBE(50359), NO_TIMESTAMPS(50363)]
    const promptTokens = [SPECIAL_TOKENS.SOT, langToken, taskToken, SPECIAL_TOKENS.NO_TIMESTAMPS];
    let curTokens = promptTokens;
    let offset = 0;

    // Inisialisasi K/V Cache: zeros [6, 1, 448, 512]
    const cacheSize = 6 * 1 * 448 * 512;
    const cacheBytes = cacheSize * 4;
    const kCachePtr = wasm._malloc(cacheBytes);
    const vCachePtr = wasm._malloc(cacheBytes);
    wasm.HEAPF32.fill(0, kCachePtr >> 2, (kCachePtr >> 2) + cacheSize);
    wasm.HEAPF32.fill(0, vCachePtr >> 2, (vCachePtr >> 2) + cacheSize);

    let curKTensor = wasm._OrtCreateTensor(1, kCachePtr, cacheBytes, this.cacheDimsPtr, 4, 1);
    let curVTensor = wasm._OrtCreateTensor(1, vCachePtr, cacheBytes, this.cacheDimsPtr, 4, 1);

    // Set cross K & V yang tetap sepanjang loop
    wasm.HEAP32[(this.decInValuesArr >> 2) + 3] = crossK;
    wasm.HEAP32[(this.decInValuesArr >> 2) + 4] = crossV;

    try {
      let step = 0;
      while (step < maxTokens) {
        const stepLen = curTokens.length;

        // Siapkan tensor tokens
        wasm.HEAP32[this.tokenDimsPtr >> 2] = 1;
        wasm.HEAP32[(this.tokenDimsPtr >> 2) + 1] = stepLen;

        let tokenDataPtr;
        let tokenBytes;
        if (stepLen === 1) {
          tokenDataPtr = this.step1TokenDataPtr;
          tokenBytes = 8;
          const big64 = new BigInt64Array([BigInt(curTokens[0])]);
          const u8 = new Uint8Array(big64.buffer, big64.byteOffset, 8);
          wasm.HEAPU8.set(u8, tokenDataPtr);
        } else {
          tokenBytes = stepLen * 8;
          tokenDataPtr = wasm._malloc(tokenBytes);
          const big64 = new BigInt64Array(stepLen);
          for (let i = 0; i < stepLen; i++) big64[i] = BigInt(curTokens[i]);
          const u8 = new Uint8Array(big64.buffer, big64.byteOffset, tokenBytes);
          wasm.HEAPU8.set(u8, tokenDataPtr);
        }

        const tokenTensor = wasm._OrtCreateTensor(7, tokenDataPtr, tokenBytes, this.tokenDimsPtr, 2, 1);

        // Siapkan tensor offset
        const offBig = new BigInt64Array([BigInt(offset)]);
        const offU8 = new Uint8Array(offBig.buffer, offBig.byteOffset, 8);
        wasm.HEAPU8.set(offU8, this.offsetDataPtr);
        const offsetTensor = wasm._OrtCreateTensor(7, this.offsetDataPtr, 8, this.offsetDimsPtr, 1, 1);

        let logitsTensor = 0;
        let nextKTensor = 0;
        let nextVTensor = 0;
        let nextToken = SPECIAL_TOKENS.EOT;

        try {
          // Pasang input tensor decoder
          wasm.HEAP32[this.decInValuesArr >> 2] = tokenTensor;
          wasm.HEAP32[(this.decInValuesArr >> 2) + 1] = curKTensor;
          wasm.HEAP32[(this.decInValuesArr >> 2) + 2] = curVTensor;
          wasm.HEAP32[(this.decInValuesArr >> 2) + 5] = offsetTensor;

          // Kosongkan slot output tensor
          wasm.HEAP32[this.decOutValuesArr >> 2] = 0;
          wasm.HEAP32[(this.decOutValuesArr >> 2) + 1] = 0;
          wasm.HEAP32[(this.decOutValuesArr >> 2) + 2] = 0;

          const decRet = await wasm._OrtRun(this.decSession, this.decInNamesArr, this.decInValuesArr, 6, this.decOutNamesArr, 3, this.decOutValuesArr, 0);
          if (decRet !== 0) throw new Error(`Decoder _OrtRun gagal dengan kode ${decRet} pada step ${step}`);

          logitsTensor = wasm.HEAP32[this.decOutValuesArr >> 2];
          nextKTensor = wasm.HEAP32[(this.decOutValuesArr >> 2) + 1];
          nextVTensor = wasm.HEAP32[(this.decOutValuesArr >> 2) + 2];

          // Ambil data pointer logits
          const code = wasm._OrtGetTensorData(logitsTensor, this.infoPtr, this.infoPtr + 4, this.infoPtr + 8, this.infoPtr + 12);
          if (code !== 0) throw new Error('Gagal mengambil data pointer logits');
          const logitsDataPtr = wasm.HEAP32[(this.infoPtr >> 2) + 1];
          const outDimsPtr = wasm.HEAP32[(this.infoPtr >> 2) + 2];
          wasm._OrtFree(outDimsPtr);

          // Logits token terakhir
          const lastPos = stepLen - 1;
          const baseIdx = (logitsDataPtr >> 2) + lastPos * 51865;

          // Greedy argmax (0..50257) - supresi token timestamp >= 50364 & special tokens > 50257
          let maxLogit = -Infinity;
          for (let i = 0; i <= SPECIAL_TOKENS.EOT; i++) {
            const v = wasm.HEAPF32[baseIdx + i];
            if (v > maxLogit) {
              maxLogit = v;
              nextToken = i;
            }
          }
        } finally {
          // Lepas tensor sementara step ini
          if (tokenTensor) wasm._OrtReleaseTensor(tokenTensor);
          if (offsetTensor) wasm._OrtReleaseTensor(offsetTensor);
          if (logitsTensor) wasm._OrtReleaseTensor(logitsTensor);
          if (stepLen > 1 && tokenDataPtr) {
            wasm._free(tokenDataPtr);
          }
        }

        // Ganti cache ke step berikutnya
        wasm._OrtReleaseTensor(curKTensor);
        wasm._OrtReleaseTensor(curVTensor);
        curKTensor = nextKTensor;
        curVTensor = nextVTensor;
        offset += stepLen;

        step++;

        if (nextToken === SPECIAL_TOKENS.EOT) {
          break;
        }

        generatedTokens.push(nextToken);
        curTokens = [nextToken];
      }
    } finally {
      // Lepas K/V cache terakhir dan cross K/V
      if (curKTensor) wasm._OrtReleaseTensor(curKTensor);
      if (curVTensor) wasm._OrtReleaseTensor(curVTensor);
      if (crossK) wasm._OrtReleaseTensor(crossK);
      if (crossV) wasm._OrtReleaseTensor(crossV);
      wasm._free(kCachePtr);
      wasm._free(vCachePtr);
    }

    const tEnd = performance.now();
    const decodeLatencyMs = Math.round(tEnd - tDecodeStart);
    const latencyMs = Math.round(tEnd - tStart);
    const audioDurationSec = pcm16k.length / 16000;
    const rtf = audioDurationSec > 0 ? Number(((latencyMs / 1000) / audioDurationSec).toFixed(3)) : 0;
    const text = decodeTokens(generatedTokens, this.tokenMap);

    return {
      text,
      latencyMs,
      melLatencyMs,
      decodeLatencyMs,
      rtf,
      numTokens: generatedTokens.length,
      backend: this.backendName
    };
  }

  async dispose() {
    if (this.melExtractor) {
      try {
        this.melExtractor.dispose();
      } catch (err) {
        console.warn('[Direct ORT] Gagal melepaskan melExtractor:', err);
      }
      this.melExtractor = null;
    }

    if (this.wasm) {
      const wasm = this.wasm;
      if (this.encSession) {
        try { wasm._OrtReleaseSession(this.encSession); } catch (err) { console.warn(err); }
        this.encSession = null;
      }
      if (this.decSession) {
        try { wasm._OrtReleaseSession(this.decSession); } catch (err) { console.warn(err); }
        this.decSession = null;
      }
      if (this.encOpts) {
        try { wasm._OrtReleaseSessionOptions(this.encOpts); } catch (err) { console.warn(err); }
        this.encOpts = null;
      }
      if (this.decOpts) {
        try { wasm._OrtReleaseSessionOptions(this.decOpts); } catch (err) { console.warn(err); }
        this.decOpts = null;
      }
      if (this.encModelPtr) {
        try { wasm._free(this.encModelPtr); } catch (err) { console.warn(err); }
        this.encModelPtr = null;
      }
      if (this.decModelPtr) {
        try { wasm._free(this.decModelPtr); } catch (err) { console.warn(err); }
        this.decModelPtr = null;
      }

      // Bebaskan string pointer dari _OrtGetInputName / _OrtGetOutputName sebelum _free array
      if (this.encInNamesArr) {
        const ptr = wasm.HEAP32[this.encInNamesArr >> 2];
        if (ptr) { try { wasm._OrtFree(ptr); } catch (err) { console.warn(err); } }
      }
      if (this.encOutNamesArr) {
        for (let i = 0; i < 2; i++) {
          const ptr = wasm.HEAP32[(this.encOutNamesArr >> 2) + i];
          if (ptr) { try { wasm._OrtFree(ptr); } catch (err) { console.warn(err); } }
        }
      }
      if (this.decInNamesArr) {
        for (let i = 0; i < 6; i++) {
          const ptr = wasm.HEAP32[(this.decInNamesArr >> 2) + i];
          if (ptr) { try { wasm._OrtFree(ptr); } catch (err) { console.warn(err); } }
        }
      }
      if (this.decOutNamesArr) {
        for (let i = 0; i < 3; i++) {
          const ptr = wasm.HEAP32[(this.decOutNamesArr >> 2) + i];
          if (ptr) { try { wasm._OrtFree(ptr); } catch (err) { console.warn(err); } }
        }
      }

      // Bebaskan runtime pointers
      const ptrs = [
        this.encInNamesArr, this.encOutNamesArr, this.encInValuesArr, this.encOutValuesArr, this.melDimsPtr,
        this.decInNamesArr, this.decOutNamesArr, this.decInValuesArr, this.decOutValuesArr,
        this.cacheDimsPtr, this.offsetDimsPtr, this.offsetDataPtr, this.tokenDimsPtr,
        this.step1TokenDataPtr, this.infoPtr
      ];
      for (const p of ptrs) {
        if (p) {
          try { wasm._free(p); } catch (err) { console.warn(err); }
        }
      }
      this.encInNamesArr = null;
      this.encOutNamesArr = null;
      this.encInValuesArr = null;
      this.encOutValuesArr = null;
      this.melDimsPtr = null;
      this.decInNamesArr = null;
      this.decOutNamesArr = null;
      this.decInValuesArr = null;
      this.decOutValuesArr = null;
      this.cacheDimsPtr = null;
      this.offsetDimsPtr = null;
      this.offsetDataPtr = null;
      this.tokenDimsPtr = null;
      this.step1TokenDataPtr = null;
      this.infoPtr = null;
    }

    this.isReady = false;
  }
}
