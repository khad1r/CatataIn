// scratch/test-ort-decoder.mjs
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const projectRoot = path.resolve(__dirname, '..');

const decoderPath = path.join(projectRoot, 'dist/models/sherpa-onnx-whisper-base/base-decoder.int8.onnx');
const decoderBytes = fs.readFileSync(decoderPath);
console.log('Read decoder model:', (decoderBytes.length / (1024 * 1024)).toFixed(2), 'MB');

import ortWasmFactory from '../web/vendor/transformers/ort-wasm-simd-threaded.jsep.mjs';

try {
  console.log('Instantiating ortWasmThreaded...');
  const ort = await ortWasmFactory();
  console.log('ortWasmThreaded instantiated successfully.');

  ort._OrtInit(1, 1);

  // Allocate memory in WASM heap for 130 MB model bytes
  console.log('Allocating memory for decoder in WASM heap...');
  const modelPtr = ort._malloc(decoderBytes.byteLength);
  if (modelPtr === 0) {
    throw new Error('Failed to allocate memory in WASM heap for decoder');
  }
  ort.HEAPU8.set(decoderBytes, modelPtr);
  console.log('Model copied to WASM heap at', modelPtr);

  const sessionOptions = ort._OrtCreateSessionOptions(3, 1, 1, 0, 0, 0, 0, 2, 0, 0);

  console.log('Calling _OrtCreateSession for decoder...');
  const startInit = performance.now();
  const session = await ort._OrtCreateSession(modelPtr, decoderBytes.byteLength, sessionOptions);
  const initMs = performance.now() - startInit;
  console.log('Decoder Session created pointer:', session, 'in', initMs.toFixed(2), 'ms');

  if (session === 0) {
    throw new Error('Failed to create decoder session');
  }

  const n = ort.PTR_SIZE || 4;
  const countPtr = ort.stackAlloc(2 * n);
  ort._OrtGetInputOutputCount(session, countPtr, countPtr + n);
  const inCount = ort.HEAP32[countPtr >> 2];
  const outCount = ort.HEAP32[(countPtr + n) >> 2];
  console.log(`Decoder Input count: ${inCount}, Output count: ${outCount}`);

  for (let i = 0; i < inCount; i++) {
    const namePtr = ort._OrtGetInputName(session, i);
    const name = ort.UTF8ToString(namePtr);
    console.log(`Decoder Input [${i}]: ${name}`);
  }

  for (let i = 0; i < outCount; i++) {
    const namePtr = ort._OrtGetOutputName(session, i);
    const name = ort.UTF8ToString(namePtr);
    console.log(`Decoder Output [${i}]: ${name}`);
  }

  ort._OrtReleaseSession(session);
  ort._OrtReleaseSessionOptions(sessionOptions);
  ort._free(modelPtr);
  console.log('Successfully cleaned up decoder session.');
} catch (err) {
  console.error('Error during decoder session test:', err);
}
