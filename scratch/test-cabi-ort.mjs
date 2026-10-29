import fs from 'node:fs';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { computeCpuMelSpectrogram } from '../web/src/engines/mel-spectrogram.js';

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
  } catch (_) {
    return new Uint8Array(0);
  }
}

function loadTokens(tokensPath) {
  const content = fs.readFileSync(tokensPath, 'utf8');
  const lines = content.trim().split('\n');
  const tokenMap = new Map();
  for (const line of lines) {
    const lastSpace = line.lastIndexOf(' ');
    if (lastSpace === -1) continue;
    const b64 = line.substring(0, lastSpace);
    const id = parseInt(line.substring(lastSpace + 1), 10);
    tokenMap.set(id, base64ToBytes(b64));
  }
  return tokenMap;
}

function decodeTokens(tokenIds, tokenMap) {
  const byteArrays = [];
  let totalLen = 0;
  for (const id of tokenIds) {
    if (id >= 50257) continue; // special tokens
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

async function main() {
  const jsepJsPath = path.resolve('web/vendor/transformers/ort-wasm-simd-threaded.jsep.mjs');
  const encoderPath = path.resolve('dist/models/sherpa-onnx-whisper-base/base-encoder.int8.onnx');
  const decoderPath = path.resolve('dist/models/sherpa-onnx-whisper-base/base-decoder.int8.onnx');
  const tokensPath = path.resolve('dist/models/sherpa-onnx-whisper-base/base-tokens.txt');
  const audioPath = path.resolve('web/sample-indonesia-16k.wav');

  console.log('Loading tokens...');
  const tokenMap = loadTokens(tokensPath);
  console.log(`Loaded ${tokenMap.size} tokens`);

  console.log('Initializing ORT WASM...');
  const { default: ortWasmFactory } = await import(pathToFileURL(jsepJsPath).href);
  const wasm = await ortWasmFactory();
  wasm._OrtInit(1, 3); // 1 thread, log level 3

  // 1. Create Encoder Session
  console.log('Creating Encoder session...');
  const encBytes = fs.readFileSync(encoderPath);
  const encPtr = wasm._malloc(encBytes.byteLength);
  wasm.HEAPU8.set(encBytes, encPtr);
  const encOpts = wasm._OrtCreateSessionOptions(3, 1, 1, 0, 0, 0, 0, 3, 0, 0);
  const encSession = await wasm._OrtCreateSession(encPtr, encBytes.byteLength, encOpts);
  console.log('Encoder Session handle:', encSession);

  // 2. Create Decoder Session
  console.log('Creating Decoder session...');
  const decBytes = fs.readFileSync(decoderPath);
  const decPtr = wasm._malloc(decBytes.byteLength);
  wasm.HEAPU8.set(decBytes, decPtr);
  const decOpts = wasm._OrtCreateSessionOptions(3, 1, 1, 0, 0, 0, 0, 3, 0, 0);
  const decSession = await wasm._OrtCreateSession(decPtr, decBytes.byteLength, decOpts);
  console.log('Decoder Session handle:', decSession);

  // 3. Process Audio & Mel Spectrogram
  console.log('Processing audio...');
  const wavBuf = fs.readFileSync(audioPath);
  const dataIdx = wavBuf.indexOf(Buffer.from('data'));
  const pcmBuf = wavBuf.subarray(dataIdx + 8);
  const audioFloat32 = new Float32Array(pcmBuf.length / 2);
  for (let i = 0; i < audioFloat32.length; i++) {
    audioFloat32[i] = pcmBuf.readInt16LE(i * 2) / 32768.0;
  }
  console.log(`Audio loaded: ${audioFloat32.length} samples (${(audioFloat32.length / 16000).toFixed(2)}s)`);

  const tMel0 = performance.now();
  const mel = computeCpuMelSpectrogram(audioFloat32);
  const melMs = performance.now() - tMel0;
  console.log(`Mel computed in ${melMs.toFixed(2)} ms. Dims:`, mel.dims);

  // 4. Run Encoder
  console.log('Running Encoder...');
  const melDims = [1, 80, 3000];
  const melSize = 80 * 3000;
  const melDataPtr = wasm._malloc(melSize * 4);
  wasm.HEAPF32.set(mel.data, melDataPtr >> 2);

  const melDimsPtr = wasm.stackAlloc(3 * 4);
  wasm.HEAP32[melDimsPtr >> 2] = 1;
  wasm.HEAP32[(melDimsPtr >> 2) + 1] = 80;
  wasm.HEAP32[(melDimsPtr >> 2) + 2] = 3000;

  const melTensor = wasm._OrtCreateTensor(1, melDataPtr, melSize * 4, melDimsPtr, 3, 1);

  const encInNamePtr = wasm._OrtGetInputName(encSession, 0);
  const encOutName0Ptr = wasm._OrtGetOutputName(encSession, 0);
  const encOutName1Ptr = wasm._OrtGetOutputName(encSession, 1);

  const encInNamesArr = wasm.stackAlloc(4);
  wasm.HEAP32[encInNamesArr >> 2] = encInNamePtr;

  const encInValuesArr = wasm.stackAlloc(4);
  wasm.HEAP32[encInValuesArr >> 2] = melTensor;

  const encOutNamesArr = wasm.stackAlloc(8);
  wasm.HEAP32[encOutNamesArr >> 2] = encOutName0Ptr;
  wasm.HEAP32[(encOutNamesArr >> 2) + 1] = encOutName1Ptr;

  const encOutValuesArr = wasm.stackAlloc(8);
  wasm.HEAP32[encOutValuesArr >> 2] = 0;
  wasm.HEAP32[(encOutValuesArr >> 2) + 1] = 0;

  const tEnc0 = performance.now();
  const encRet = await wasm._OrtRun(encSession, encInNamesArr, encInValuesArr, 1, encOutNamesArr, 2, encOutValuesArr, 0);
  const encMs = performance.now() - tEnc0;
  if (encRet !== 0) throw new Error(`Encoder _OrtRun failed with code ${encRet}`);
  console.log(`Encoder finished in ${encMs.toFixed(2)} ms`);

  const crossK = wasm.HEAP32[encOutValuesArr >> 2];
  const crossV = wasm.HEAP32[(encOutValuesArr >> 2) + 1];

  // 5. Autoregressive Greedy Decoder Loop
  console.log('Starting Autoregressive Greedy Decoder Loop...');
  // Inputs:
  // 0: tokens (int64)
  // 1: in_n_layer_self_k_cache (float32 [6, 1, 448, 512])
  // 2: in_n_layer_self_v_cache (float32 [6, 1, 448, 512])
  // 3: n_layer_cross_k (from encoder)
  // 4: n_layer_cross_v (from encoder)
  // 5: offset (int64 [1])
  // Outputs:
  // 0: logits
  // 1: out_n_layer_self_k_cache
  // 2: out_n_layer_self_v_cache

  const decInNamesArr = wasm._malloc(6 * 4);
  for (let i = 0; i < 6; i++) {
    wasm.HEAP32[(decInNamesArr >> 2) + i] = wasm._OrtGetInputName(decSession, i);
  }

  const decOutNamesArr = wasm._malloc(3 * 4);
  for (let i = 0; i < 3; i++) {
    wasm.HEAP32[(decOutNamesArr >> 2) + i] = wasm._OrtGetOutputName(decSession, i);
  }

  const decInValuesArr = wasm._malloc(6 * 4);
  const decOutValuesArr = wasm._malloc(3 * 4);

  // Cross K and V stay constant
  wasm.HEAP32[(decInValuesArr >> 2) + 3] = crossK;
  wasm.HEAP32[(decInValuesArr >> 2) + 4] = crossV;

  // Initial prompt: [SOT(50258), LANG_ID(50275), TRANSCRIBE(50359), NO_TIMESTAMPS(50363)]
  const promptTokens = [50258, 50275, 50359, 50363];
  let curTokens = promptTokens;
  let offset = 0;

  // Initial self K/V cache: zeros [6, 1, 448, 512]
  const cacheSize = 6 * 1 * 448 * 512;
  const cacheBytes = cacheSize * 4;
  const kCachePtr = wasm._malloc(cacheBytes);
  const vCachePtr = wasm._malloc(cacheBytes);
  wasm.HEAPF32.fill(0, kCachePtr >> 2, (kCachePtr >> 2) + cacheSize);
  wasm.HEAPF32.fill(0, vCachePtr >> 2, (vCachePtr >> 2) + cacheSize);

  const cacheDimsPtr = wasm._malloc(4 * 4);
  wasm.HEAP32[cacheDimsPtr >> 2] = 6;
  wasm.HEAP32[(cacheDimsPtr >> 2) + 1] = 1;
  wasm.HEAP32[(cacheDimsPtr >> 2) + 2] = 448;
  wasm.HEAP32[(cacheDimsPtr >> 2) + 3] = 512;

  let curKTensor = wasm._OrtCreateTensor(1, kCachePtr, cacheBytes, cacheDimsPtr, 4, 1);
  let curVTensor = wasm._OrtCreateTensor(1, vCachePtr, cacheBytes, cacheDimsPtr, 4, 1);

  const generatedTokens = [];
  const maxTokens = 224;
  const tDec0 = performance.now();

  const EOT = 50257;

  // Pre-allocate buffer for offset tensor
  const offsetDimsPtr = wasm._malloc(1 * 4);
  wasm.HEAP32[offsetDimsPtr >> 2] = 1;
  const offsetDataPtr = wasm._malloc(8); // int64

  // Pre-allocate buffer for tokens dims & data
  const tokenDimsPtr = wasm._malloc(2 * 4);

  // Scratch info ptr for getting tensor data
  const infoPtr = wasm._malloc(16);

  let step = 0;
  while (step < maxTokens) {
    const stepLen = curTokens.length;

    // Create tokens tensor [1, stepLen], int64
    wasm.HEAP32[tokenDimsPtr >> 2] = 1;
    wasm.HEAP32[(tokenDimsPtr >> 2) + 1] = stepLen;

    const tokenBytes = stepLen * 8;
    const tokenDataPtr = wasm._malloc(tokenBytes);
    // Fill int64 little-endian
    const tokenBigInt = new BigInt64Array(stepLen);
    for (let i = 0; i < stepLen; i++) {
      tokenBigInt[i] = BigInt(curTokens[i]);
    }
    const tokenU8 = new Uint8Array(tokenBigInt.buffer, tokenBigInt.byteOffset, tokenBytes);
    wasm.HEAPU8.set(tokenU8, tokenDataPtr);

    const tokenTensor = wasm._OrtCreateTensor(7, tokenDataPtr, tokenBytes, tokenDimsPtr, 2, 1);

    // Create offset tensor [1], int64
    const offsetBigInt = new BigInt64Array([BigInt(offset)]);
    const offsetU8 = new Uint8Array(offsetBigInt.buffer, offsetBigInt.byteOffset, 8);
    wasm.HEAPU8.set(offsetU8, offsetDataPtr);
    const offsetTensor = wasm._OrtCreateTensor(7, offsetDataPtr, 8, offsetDimsPtr, 1, 1);

    // Assign inputs
    wasm.HEAP32[decInValuesArr >> 2] = tokenTensor;
    wasm.HEAP32[(decInValuesArr >> 2) + 1] = curKTensor;
    wasm.HEAP32[(decInValuesArr >> 2) + 2] = curVTensor;
    // index 3 & 4 already crossK and crossV
    wasm.HEAP32[(decInValuesArr >> 2) + 5] = offsetTensor;

    // Clear output slots
    wasm.HEAP32[decOutValuesArr >> 2] = 0;
    wasm.HEAP32[(decOutValuesArr >> 2) + 1] = 0;
    wasm.HEAP32[(decOutValuesArr >> 2) + 2] = 0;

    const decRet = await wasm._OrtRun(decSession, decInNamesArr, decInValuesArr, 6, decOutNamesArr, 3, decOutValuesArr, 0);
    if (decRet !== 0) throw new Error(`Decoder _OrtRun failed with code ${decRet} at step ${step}`);

    const logitsTensor = wasm.HEAP32[decOutValuesArr >> 2];
    const nextKTensor = wasm.HEAP32[(decOutValuesArr >> 2) + 1];
    const nextVTensor = wasm.HEAP32[(decOutValuesArr >> 2) + 2];

    // Read logits
    const code = wasm._OrtGetTensorData(logitsTensor, infoPtr, infoPtr + 4, infoPtr + 8, infoPtr + 12);
    if (code !== 0) throw new Error('Failed to get logits tensor data');
    const logitsDataPtr = wasm.HEAP32[(infoPtr >> 2) + 1];
    const outDimsPtr = wasm.HEAP32[(infoPtr >> 2) + 2];
    wasm._OrtFree(outDimsPtr);

    // Position of the last token
    const lastPos = stepLen - 1;
    const baseIdx = (logitsDataPtr >> 2) + lastPos * 51865;

    // Argmax (0..50257) - supresi token timestamp >= 50364 dan special tokens > 50257
    let maxLogit = -Infinity;
    let nextToken = EOT;
    for (let i = 0; i <= EOT; i++) {
      const v = wasm.HEAPF32[baseIdx + i];
      if (v > maxLogit) {
        maxLogit = v;
        nextToken = i;
      }
    }

    // Clean up temporary tensors for this step
    wasm._OrtReleaseTensor(tokenTensor);
    wasm._OrtReleaseTensor(offsetTensor);
    wasm._OrtReleaseTensor(logitsTensor);
    wasm._free(tokenDataPtr);

    // Release old cache tensors
    wasm._OrtReleaseTensor(curKTensor);
    wasm._OrtReleaseTensor(curVTensor);

    // Update state for next step
    curKTensor = nextKTensor;
    curVTensor = nextVTensor;
    offset += stepLen;

    step++;

    if (nextToken === EOT) {
      console.log(`Step ${step}: Encountered EOT token (${nextToken}). Stopping.`);
      break;
    }

    console.log(`Step ${step}: token ${nextToken} ("${decodeTokens([nextToken], tokenMap)}") [logit: ${maxLogit.toFixed(2)}]`);
    generatedTokens.push(nextToken);
    curTokens = [nextToken];
  }

  const decMs = performance.now() - tDec0;
  console.log(`Decoder loop finished: ${generatedTokens.length} tokens generated in ${decMs.toFixed(2)} ms (${(decMs / generatedTokens.length).toFixed(2)} ms/token)`);

  // Release last K/V cache tensors
  wasm._OrtReleaseTensor(curKTensor);
  wasm._OrtReleaseTensor(curVTensor);

  // Decode text
  const finalTranscription = decodeTokens(generatedTokens, tokenMap);
  console.log('========================================');
  console.log('RESULT TEXT:', `"${finalTranscription}"`);
  console.log('========================================');

  // Verify result
  if (finalTranscription.includes('Republik Indonesia') && finalTranscription.includes('Surabaya')) {
    console.log('SUCCESS: Real inference produced authentic Indonesian transcription:', `"${finalTranscription}"`);
  } else {
    console.log('UNEXPECTED RESULT:', `"${finalTranscription}"`);
  }

  // Cleanup sessions and allocations
  wasm._OrtReleaseTensor(melTensor);
  wasm._OrtReleaseTensor(crossK);
  wasm._OrtReleaseTensor(crossV);
  wasm._free(melDataPtr);
  wasm._free(kCachePtr);
  wasm._free(vCachePtr);
  wasm._free(cacheDimsPtr);
  wasm._free(offsetDimsPtr);
  wasm._free(offsetDataPtr);
  wasm._free(tokenDimsPtr);
  wasm._free(infoPtr);
  wasm._free(decInNamesArr);
  wasm._free(decOutNamesArr);
  wasm._free(decInValuesArr);
  wasm._free(decOutValuesArr);

  wasm._OrtReleaseSession(encSession);
  wasm._OrtReleaseSession(decSession);
  wasm._OrtReleaseSessionOptions(encOpts);
  wasm._OrtReleaseSessionOptions(decOpts);
  wasm._free(encPtr);
  wasm._free(decPtr);

  console.log('Done!');
}

main().catch(err => {
  console.error('Fatal error:', err);
  process.exit(1);
});
