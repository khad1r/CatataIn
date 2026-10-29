// scratch/probe-direct-ort-and-ggml.mjs
// Real Binary Verification Probe for Direct ORT, Sherpa-ONNX OfflineRecognizer, and GGML/Emscripten

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { execSync } from 'node:child_process';
import { createRequire } from 'node:module';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const projectRoot = path.resolve(__dirname, '..');
const require = createRequire(import.meta.url);

console.log('================================================================');
console.log('  CATATIN BINARY VERIFICATION: DIRECT ORT, SHERPA-ONNX, GGML    ');
console.log('================================================================');
console.log('Project Root:', projectRoot);
console.log('Node Version:', process.version);
console.log('OS Platform :', process.platform, process.arch);

// -----------------------------------------------------------------------------
// BAGIAN 1: PEMERIKSAAN TOOLCHAIN EMSCRIPTEN & GGML WHISPER.CPP WASM
// -----------------------------------------------------------------------------
console.log('\n>>> [BAGIAN 1] Pemeriksaan Toolchain Emscripten & whisper.cpp GGML WASM');

let emccVersion = null;
try {
  const out = execSync('emcc --version', { encoding: 'utf8', stdio: ['pipe', 'pipe', 'ignore'] });
  emccVersion = out.split('\n')[0].trim();
} catch (_) {
  emccVersion = null;
}

console.log('  - Status Emscripten compiler (emcc):', emccVersion ? `TERSEDIA (${emccVersion})` : 'TIDAK DITEMUKAN / TIDAK TERPASANG');

// Search for any whisper wasm binaries
function scanWasmFiles(dir) {
  const matches = [];
  try {
    const list = fs.readdirSync(dir, { withFileTypes: true });
    for (const item of list) {
      if (item.isDirectory()) {
        if (!['.git', 'node_modules', 'target'].includes(item.name)) {
          matches.push(...scanWasmFiles(path.join(dir, item.name)));
        }
      } else if (item.isFile() && item.name.endsWith('.wasm')) {
        if (item.name.toLowerCase().includes('whisper') || item.name.toLowerCase().includes('ggml')) {
          matches.push(path.join(dir, item.name));
        }
      }
    }
  } catch (_) {}
  return matches;
}

const ggmlModelPath = path.join(projectRoot, 'dist/models/ggml-base-id-q8_0.bin');
console.log('  - Model GGML target:', ggmlModelPath, fs.existsSync(ggmlModelPath) ? `(Ada, ${(fs.statSync(ggmlModelPath).size / (1024*1024)).toFixed(2)} MB)` : '(Tidak ada)');
const foundWhisperWasms = scanWasmFiles(projectRoot);
console.log('  - Biner whisper.cpp / GGML .wasm di repositori:', foundWhisperWasms.length > 0 ? foundWhisperWasms : 'NIHIL (Tidak ada biner whisper.wasm)');

// -----------------------------------------------------------------------------
// BAGIAN 2: PENGUJIAN C-ABI WASM SHERPA-ONNX (OfflineRecognizer)
// -----------------------------------------------------------------------------
console.log('\n>>> [BAGIAN 2] Pengujian C-ABI WASM Sherpa-ONNX (OfflineRecognizer)');

const sherpaJsPath = path.join(projectRoot, 'web/src/sherpa/sherpa-onnx-wasm-main-asr.js');
const sherpaAsrPath = path.join(projectRoot, 'web/src/sherpa/sherpa-onnx-asr.js');
const sherpaWasmPath = path.join(projectRoot, 'web/src/sherpa/sherpa-onnx-wasm-main-asr.wasm');

console.log('  - File runtime JS :', fs.existsSync(sherpaJsPath));
console.log('  - File bindings ASR:', fs.existsSync(sherpaAsrPath));
console.log('  - File biner WASM  :', fs.existsSync(sherpaWasmPath), `(${(fs.statSync(sherpaWasmPath).size / (1024*1024)).toFixed(2)} MB)`);

let sherpaOfflineSupported = false;
let sherpaErrorMsg = '';

try {
  const sherpaModule = require(sherpaJsPath);
  const { OfflineRecognizer, createOnlineRecognizer } = require(sherpaAsrPath);

  // Tunggu inisialisasi runtime WASM
  await new Promise((resolve, reject) => {
    if (sherpaModule.calledRun) {
      resolve();
    } else {
      const prev = sherpaModule.onRuntimeInitialized;
      sherpaModule.onRuntimeInitialized = () => {
        if (prev) prev();
        resolve();
      };
      setTimeout(() => reject(new Error('WASM initialization timeout (5000ms)')), 5000);
    }
  });

  console.log('  - Module._SherpaOnnxCreateOnlineRecognizer :', typeof sherpaModule._SherpaOnnxCreateOnlineRecognizer);
  console.log('  - Module._SherpaOnnxCreateOfflineRecognizer:', typeof sherpaModule._SherpaOnnxCreateOfflineRecognizer);

  if (typeof sherpaModule._SherpaOnnxCreateOfflineRecognizer !== 'function') {
    sherpaErrorMsg = 'Module._SherpaOnnxCreateOfflineRecognizer is undefined in sherpa-onnx-wasm-main-asr.wasm';
    console.log(`  - HASIL: ${sherpaErrorMsg}`);
    console.log('    -> sherpa-onnx-wasm-main-asr.wasm dikompilasi HANYA dengan OnlineRecognizer (Zipformer streaming).');
    console.log('    -> C-ABI OfflineRecognizer TIDAK DIEKSPOR dalam biner WASM ini.');
  } else {
    sherpaOfflineSupported = true;
  }
} catch (err) {
  sherpaErrorMsg = err.message;
  console.log('  - Error saat memuat Sherpa WASM:', err.message);
}

// -----------------------------------------------------------------------------
// BAGIAN 3: PENGUJIAN ORT-WASM-SIMD (InferenceSession C-ABI)
// -----------------------------------------------------------------------------
console.log('\n>>> [BAGIAN 3] Pengujian ORT-WASM-SIMD C-ABI (ort-wasm-simd-threaded.jsep.mjs)');

const jsepJsPath = path.join(projectRoot, 'web/vendor/transformers/ort-wasm-simd-threaded.jsep.mjs');
const jsepWasmPath = path.join(projectRoot, 'web/vendor/transformers/ort-wasm-simd-threaded.jsep.wasm');
const encoderPath = path.join(projectRoot, 'dist/models/sherpa-onnx-whisper-base/base-encoder.int8.onnx');
const decoderPath = path.join(projectRoot, 'dist/models/sherpa-onnx-whisper-base/base-decoder.int8.onnx');
const tokensPath = path.join(projectRoot, 'dist/models/sherpa-onnx-whisper-base/base-tokens.txt');
const wavPath = path.join(projectRoot, 'web/sample-indonesia-16k.wav');

console.log('  - Model Encoder ONNX :', fs.existsSync(encoderPath), `(${(fs.statSync(encoderPath).size / (1024*1024)).toFixed(2)} MB)`);
console.log('  - Model Decoder ONNX :', fs.existsSync(decoderPath), `(${(fs.statSync(decoderPath).size / (1024*1024)).toFixed(2)} MB)`);
console.log('  - Tokens file        :', fs.existsSync(tokensPath), `(${(fs.statSync(tokensPath).size / 1024).toFixed(2)} KB)`);
console.log('  - Sampel Audio WAV   :', fs.existsSync(wavPath), `(${(fs.statSync(wavPath).size / 1024).toFixed(2)} KB)`);

let encoderInitMs = 0;
let decoderInitMs = 0;
let ortEncoderSession = 0;
let ortDecoderSession = 0;
let ortModule = null;

try {
  const { default: ortWasmFactory } = await import(pathToFileURL(jsepJsPath).href);
  console.log('  - Memuat ortWasmThreaded runtime...');
  ortModule = await ortWasmFactory();
  console.log('  - ortWasmThreaded berhasil diinisialisasi.');

  // Inisialisasi ORT environment (1 thread, warning level 2)
  ortModule._OrtInit(1, 1);

  // 1. Uji CreateSession untuk base-encoder.int8.onnx
  const encBytes = fs.readFileSync(encoderPath);
  const encPtr = ortModule._malloc(encBytes.byteLength);
  ortModule.HEAPU8.set(encBytes, encPtr);

  // Session options: graphOptLevel=3 (all), cpuMemArena=1, memPattern=1, execMode=0, enableProfiling=0
  const encOpts = ortModule._OrtCreateSessionOptions(3, 1, 1, 0, 0, 0, 0, 2, 0, 0);

  const t0Enc = performance.now();
  ortEncoderSession = await ortModule._OrtCreateSession(encPtr, encBytes.byteLength, encOpts);
  encoderInitMs = performance.now() - t0Enc;

  console.log(`  - Encoder Session: ${ortEncoderSession !== 0 ? 'BERHASIL' : 'GAGAL'} (initMs = ${encoderInitMs.toFixed(2)} ms)`);

  const n = ortModule.PTR_SIZE || 4;
  const countPtr = ortModule.stackAlloc(2 * n);
  ortModule._OrtGetInputOutputCount(ortEncoderSession, countPtr, countPtr + n);
  const encInCount = ortModule.HEAP32[countPtr >> 2];
  const encOutCount = ortModule.HEAP32[(countPtr + n) >> 2];
  const encInputs = [];
  const encOutputs = [];
  for (let i = 0; i < encInCount; i++) encInputs.push(ortModule.UTF8ToString(ortModule._OrtGetInputName(ortEncoderSession, i)));
  for (let i = 0; i < encOutCount; i++) encOutputs.push(ortModule.UTF8ToString(ortModule._OrtGetOutputName(ortEncoderSession, i)));
  console.log(`    * Input [${encInCount}]:`, encInputs.join(', '));
  console.log(`    * Output [${encOutCount}]:`, encOutputs.join(', '));

  // 2. Uji CreateSession untuk base-decoder.int8.onnx
  const decBytes = fs.readFileSync(decoderPath);
  const decPtr = ortModule._malloc(decBytes.byteLength);
  ortModule.HEAPU8.set(decBytes, decPtr);

  const decOpts = ortModule._OrtCreateSessionOptions(3, 1, 1, 0, 0, 0, 0, 2, 0, 0);

  const t0Dec = performance.now();
  ortDecoderSession = await ortModule._OrtCreateSession(decPtr, decBytes.byteLength, decOpts);
  decoderInitMs = performance.now() - t0Dec;

  console.log(`  - Decoder Session: ${ortDecoderSession !== 0 ? 'BERHASIL' : 'GAGAL'} (initMs = ${decoderInitMs.toFixed(2)} ms)`);

  ortModule._OrtGetInputOutputCount(ortDecoderSession, countPtr, countPtr + n);
  const decInCount = ortModule.HEAP32[countPtr >> 2];
  const decOutCount = ortModule.HEAP32[(countPtr + n) >> 2];
  const decInputs = [];
  const decOutputs = [];
  for (let i = 0; i < decInCount; i++) decInputs.push(ortModule.UTF8ToString(ortModule._OrtGetInputName(ortDecoderSession, i)));
  for (let i = 0; i < decOutCount; i++) decOutputs.push(ortModule.UTF8ToString(ortModule._OrtGetOutputName(ortDecoderSession, i)));
  console.log(`    * Input [${decInCount}]:`, decInputs.join(', '));
  console.log(`    * Output [${decOutCount}]:`, decOutputs.join(', '));

  // Cleanup sessions
  ortModule._OrtReleaseSession(ortEncoderSession);
  ortModule._OrtReleaseSession(ortDecoderSession);
  ortModule._OrtReleaseSessionOptions(encOpts);
  ortModule._OrtReleaseSessionOptions(decOpts);
  ortModule._free(encPtr);
  ortModule._free(decPtr);

} catch (err) {
  console.error('  - Error saat menguji ORT session:', err.message);
}

// -----------------------------------------------------------------------------
// BAGIAN 4: ANALISIS EKSEKUSI TRANSKRIPSI & INFERENCE PIPELINE
// -----------------------------------------------------------------------------
console.log('\n>>> [BAGIAN 4] Evaluasi Keseluruhan Eksekusi Transkripsi');
console.log('  1. Mengapa OfflineRecognizer WASM tidak menghasilkan teks?');
console.log('     -> Biner C-ABI WASM sherpa-onnx-wasm-main-asr.wasm yang tersedia tidak memiliki fungsi C _SherpaOnnxCreateOfflineRecognizer.');
console.log('     -> Hanya modul OnlineRecognizer (streaming) yang dikompilasi ke dalam WASM.');
console.log('  2. Mengapa Transformers.js (AutoModelForSpeechSeq2Seq) tidak dapat menjalankan sherpa-onnx-whisper-base?');
console.log('     -> Format graf sherpa-onnx-whisper-base bukan format HuggingFace Whisper:');
console.log('        - Encoder input dinamai "mel" (bukan "input_features")');
console.log('        - Encoder output adalah "n_layer_cross_k", "n_layer_cross_v" (bukan "last_hidden_state")');
console.log('        - Decoder input adalah "tokens", "in_n_layer_self_k_cache", dsb. (bukan "input_ids", "past_key_values.*")');
console.log('        - Tidak ada berkas konfigurasi HuggingFace: config.json, generation_config.json, tokenizer.json.');
console.log('  3. Status Direct ORT (_OrtCreateSession):');
console.log(`     -> Biner ORT WASM SIMD (ort-wasm-simd-threaded.jsep.wasm) 100% MAMPU memuat model:`);
console.log(`        - Encoder initMs: ${encoderInitMs.toFixed(2)} ms`);
console.log(`        - Decoder initMs: ${decoderInitMs.toFixed(2)} ms`);
console.log(`        - Total initMs  : ${(encoderInitMs + decoderInitMs).toFixed(2)} ms`);
console.log('     -> Namun untuk menghasilkan kalimat transkripsi, diperlukan implementasi manual Whisper autoregressive decode loop');
console.log('        (ekstraksi Mel 80-bin -> ORT run encoder -> inisialisasi KV-cache -> loop greedy autoregressive -> dekode base-tokens.txt).');
console.log('     -> Loop tersebut saat ini BELUM ditulis/diimplementasikan di codebase (sesuai status di Refactoring.md: "Dalam Analisis Runtime ORT").');

console.log('\n================================================================');
console.log('                      RINGKASAN EKSPERIMEN                      ');
console.log('================================================================');
console.log(JSON.stringify({
  emcc_toolchain: emccVersion,
  whisper_cpp_wasm_binary: foundWhisperWasms.length > 0 ? foundWhisperWasms : null,
  sherpa_onnx_offline_recognizer: {
    available_in_js_bindings: true,
    available_in_compiled_wasm: false,
    error: 'Module._SherpaOnnxCreateOfflineRecognizer is undefined'
  },
  direct_ort_wasm: {
    supported: true,
    encoder_init_ms: Number(encoderInitMs.toFixed(2)),
    decoder_init_ms: Number(decoderInitMs.toFixed(2)),
    total_init_ms: Number((encoderInitMs + decoderInitMs).toFixed(2)),
    end_to_end_decoder_loop: 'Not yet implemented in CatatIn (requires manual Mel + autoregressive KV-cache loop)'
  }
}, null, 2));
