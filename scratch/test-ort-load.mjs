// scratch/test-ort-load.mjs
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const projectRoot = path.resolve(__dirname, '..');

const encoderPath = path.join(projectRoot, 'dist/models/sherpa-onnx-whisper-base/base-encoder.int8.onnx');
const encoderBytes = fs.readFileSync(encoderPath);
console.log('Read encoder model:', encoderBytes.length, 'bytes');

import ortWasmFactory from '../web/vendor/transformers/ort-wasm-simd-threaded.jsep.mjs';

try {
  console.log('Instantiating ortWasmThreaded...');
  const ort = await ortWasmFactory();
  console.log('ortWasmThreaded instantiated successfully.');

  // Initialize ORT
  // _OrtInit(numThreads, loggingLevel)
  ort._OrtInit(1, 1);
  console.log('Ort initialized');

  // Allocate memory in WASM heap for model bytes
  const modelPtr = ort._malloc(encoderBytes.byteLength);
  ort.HEAPU8.set(encoderBytes, modelPtr);
  console.log('Model copied to WASM heap at', modelPtr);

  // Create session options
  // _OrtCreateSessionOptions(graphOptLevel, enableCpuMemArena, enableMemPattern, executionMode, enableProfiling, profileFilePrefix, logId, logSeverity, logVerbosity, optimizedModelFilePath)
  const sessionOptions = ort._OrtCreateSessionOptions(3, 1, 1, 0, 0, 0, 0, 2, 0, 0);
  console.log('Session options created:', sessionOptions);

  // Create session: _OrtCreateSession(modelPtr, modelLen, sessionOptionsPtr)
  console.log('Calling _OrtCreateSession...');
  const startInit = performance.now();
  const session = await ort._OrtCreateSession(modelPtr, encoderBytes.byteLength, sessionOptions);
  const initMs = performance.now() - startInit;
  console.log('Session created pointer:', session, 'in', initMs.toFixed(2), 'ms');

  if (session === 0) {
    throw new Error('Failed to create session (returned 0)');
  }

  // Get inputs/outputs count
  const n = ort.PTR_SIZE || 4;
  const countPtr = ort.stackAlloc(2 * n);
  ort._OrtGetInputOutputCount(session, countPtr, countPtr + n);
  const inCount = ort.HEAP32[countPtr >> 2];
  const outCount = ort.HEAP32[(countPtr + n) >> 2];
  console.log(`Input count: ${inCount}, Output count: ${outCount}`);

  for (let i = 0; i < inCount; i++) {
    const namePtr = ort._OrtGetInputName(session, i);
    const name = ort.UTF8ToString(namePtr);
    console.log(`Input [${i}]: ${name}`);
  }

  for (let i = 0; i < outCount; i++) {
    const namePtr = ort._OrtGetOutputName(session, i);
    const name = ort.UTF8ToString(namePtr);
    console.log(`Output [${i}]: ${name}`);
  }

  ort._OrtReleaseSession(session);
  ort._OrtReleaseSessionOptions(sessionOptions);
  ort._free(modelPtr);
  console.log('Successfully cleaned up encoder session.');
} catch (err) {
  console.error('Error during ORT session test:', err);
}
