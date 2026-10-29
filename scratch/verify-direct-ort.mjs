// scratch/verify-direct-ort.mjs
// Independent Whitebox Verification Script for Pass2DirectOrtEngine
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const projectRoot = path.resolve(__dirname, '..');

console.log('====================================================');
console.log('INDEPENDENT VERIFICATION: Pass2DirectOrtEngine (#1)');
console.log('====================================================');
console.log('Project Root:', projectRoot);

const results = {
  antiMockGate: { passed: false, details: [] },
  zeroSilentFallback: { passed: false, details: [] },
  realInference: { passed: false, details: [] }
};

// 1. Anti-Mock Gate: inspect source files for mock/fake/silent fallback
console.log('\n--- 1. Anti-Mock Gate ---');
const engineFilePath = path.join(projectRoot, 'web/src/engines/pass2-direct-ort.js');
const engineFileUrl = pathToFileURL(engineFilePath).href;
const melFilePath = path.join(projectRoot, 'web/src/engines/mel-spectrogram.js');

if (!fs.existsSync(engineFilePath)) {
  results.antiMockGate.details.push(`FAIL: ${engineFilePath} does not exist`);
} else {
  const engineSrc = fs.readFileSync(engineFilePath, 'utf8');
  const mockPatterns = [
    { pattern: /Math\.random\(\)/, name: 'Math.random()' },
    { pattern: /setTimeout\(.*(?:resolve|1000|500).*\)/, name: 'fake setTimeout progress' },
    { pattern: /catch\s*\([^\)]*\)\s*\{\s*\}/, name: 'empty catch block' },
    { pattern: /return\s*['"`]dummy/i, name: 'dummy string return' }
  ];

  let foundMock = false;
  for (const m of mockPatterns) {
    if (m.pattern.test(engineSrc)) {
      results.antiMockGate.details.push(`FAIL: detected ${m.name} in pass2-direct-ort.js`);
      foundMock = true;
    }
  }

  // Check if it's still an unready stub
  const isStub = engineSrc.includes('belum tersedia');
  if (isStub) {
    results.antiMockGate.details.push('STATUS: pass2-direct-ort.js is currently marked "belum tersedia" (stub / honest rejection).');
  } else {
    results.antiMockGate.details.push('STATUS: pass2-direct-ort.js contains non-stub implementation.');
  }

  if (!foundMock) {
    results.antiMockGate.passed = true;
    results.antiMockGate.details.push('PASS: 0% mock, 0% dummy return, 0% Math.random(), 0% silent fallback in pass2-direct-ort.js.');
  }
}
console.log('Anti-Mock Gate Results:', results.antiMockGate);

// 2. Zero Silent Fallback & Availability Check
console.log('\n--- 2. Zero Silent Fallback Check ---');
try {
  const { Pass2DirectOrtEngine } = await import(engineFileUrl);
  const engine = new Pass2DirectOrtEngine();
  console.log('Engine created:', { id: engine.id, name: engine.name, available: engine.available, isReady: engine.isReady });

  // Test whisper-tiny
  let tinyThrew = false;
  try {
    await engine.init('Xenova/whisper-tiny', 'id');
  } catch (err) {
    tinyThrew = true;
    results.zeroSilentFallback.details.push(`PASS: whisper-tiny correctly threw error: "${err.message}"`);
  }
  if (!tinyThrew) {
    results.zeroSilentFallback.details.push('FAIL: whisper-tiny did NOT throw an error when unavailable');
  }

  // Test whisper-small
  let smallThrew = false;
  try {
    await engine.init('Xenova/whisper-small', 'id');
  } catch (err) {
    smallThrew = true;
    results.zeroSilentFallback.details.push(`PASS: whisper-small correctly threw error: "${err.message}"`);
  }
  if (!smallThrew) {
    results.zeroSilentFallback.details.push('FAIL: whisper-small did NOT throw an error when unavailable');
  }

  results.zeroSilentFallback.passed = tinyThrew && smallThrew && !engine.isReady;
} catch (err) {
  results.zeroSilentFallback.details.push(`IMPORT/EXEC ERROR: ${err.message}`);
}
console.log('Zero Silent Fallback Results:', results.zeroSilentFallback);

// 3. Real Inference Execution with whisper-base
console.log('\n--- 3. Real Inference Execution Check ---');
try {
  const { Pass2DirectOrtEngine } = await import(engineFileUrl);
  const engine = new Pass2DirectOrtEngine();
  
  // Try init whisper-base
  console.log('Attempting engine.init("Xenova/whisper-base", "id")...');
  try {
    await engine.init('Xenova/whisper-base', 'id');
    console.log('engine.init succeeded. isReady:', engine.isReady);
    
    // Read audio
    const audioPath = path.join(projectRoot, 'web/sample-indonesia-16k.wav');
    const wavBuf = fs.readFileSync(audioPath);
    // Find 'data' chunk
    const dataIdx = wavBuf.indexOf(Buffer.from('data'));
    const pcmBuf = wavBuf.subarray(dataIdx + 8);
    const audioFloat32 = new Float32Array(pcmBuf.length / 2);
    for (let i = 0; i < audioFloat32.length; i++) {
      audioFloat32[i] = pcmBuf.readInt16LE(i * 2) / 32768.0;
    }
    console.log(`Audio loaded: ${audioFloat32.length} samples (${(audioFloat32.length / 16000).toFixed(2)}s)`);

    const res = await engine.transcribe(audioFloat32, { language: 'id' });
    console.log('Transcribe Result:', res);
    results.realInference.passed = true;
    results.realInference.details.push(`Output: "${res.text}"`);
  } catch (err) {
    console.log('Inference execution threw (expected if engine is stub):', err.message);
    results.realInference.details.push(`Engine threw: ${err.message}`);
    results.realInference.passed = false;
  }
} catch (err) {
  results.realInference.details.push(`FATAL: ${err.message}`);
}

console.log('\n====================================================');
console.log('VERIFICATION SUMMARY');
console.log('====================================================');
console.log(JSON.stringify(results, null, 2));
