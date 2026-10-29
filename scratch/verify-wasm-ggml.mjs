// scratch/verify-wasm-ggml.mjs
// Independent whitebox verification script for whisper.cpp WASM GGML adapter (#2)
// Strictly enforces AGENTS.md (Zero-Mock, Zero Silent Fallback, Ponytail)

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const projectRoot = path.resolve(__dirname, '..');

const results = {
  gate1: { name: 'Gate 1 (Anti-Mock & Static C-ABI Check)', passed: false, checks: [] },
  gate2: { name: 'Gate 2 (Zero Silent Fallback — Model Resolution)', passed: false, checks: [] },
  gate3: { name: 'Gate 3 (Honest Uncompiled Binary Check & Local Model Files)', passed: false, checks: [] }
};

function recordCheck(gate, checkName, passed, details = '') {
  gate.checks.push({ checkName, passed, details });
  if (!passed) {
    console.error(`❌ [${gate.name}] FAILED: ${checkName} - ${details}`);
  } else {
    console.log(`✅ [${gate.name}] PASSED: ${checkName} ${details ? `(${details})` : ''}`);
  }
}

async function runVerification() {
  console.log('================================================================');
  console.log('   VERIFIKASI INDEPENDEN WHITEBOX: PASS2-WASM-GGML (ADAPTER #2) ');
  console.log('================================================================\n');

  // --------------------------------------------------------------------------
  // GATE 1: Anti-Mock & Static C-ABI Check
  // --------------------------------------------------------------------------
  console.log('--- Memeriksa GATE 1: Anti-Mock & Static C-ABI Check ---');
  const engineJsPath = path.resolve(projectRoot, 'web/src/engines/pass2-wasm-ggml.js');
  const serveJsPath = path.resolve(projectRoot, 'scripts/serve.js');
  const buildPs1Path = path.resolve(projectRoot, 'scripts/build-whisper-wasm.ps1');

  if (!fs.existsSync(engineJsPath)) {
    recordCheck(results.gate1, 'pass2-wasm-ggml.js exists', false, `Berkas tidak ditemukan di: ${engineJsPath}`);
  } else {
    recordCheck(results.gate1, 'pass2-wasm-ggml.js exists', true, engineJsPath);
    const engineCode = fs.readFileSync(engineJsPath, 'utf8');

    // Anti-Mock checks
    const hasMathRandom = engineCode.includes('Math.random()');
    recordCheck(results.gate1, 'Zero Fake Telemetry (no Math.random())', !hasMathRandom, hasMathRandom ? 'Ditemukan Math.random()' : 'Bebas Math.random()');

    const hasSetTimeout = /setTimeout\s*\(/.test(engineCode);
    recordCheck(results.gate1, 'Zero Fake Timers (no setTimeout)', !hasSetTimeout, hasSetTimeout ? 'Ditemukan setTimeout' : 'Bebas setTimeout');

    const hasDummyString = /text:\s*['"](?:halo|test|testing|sample|dummy|mock)['"]/i.test(engineCode);
    recordCheck(results.gate1, 'Zero Dummy Strings (no mock text)', !hasDummyString, hasDummyString ? 'Ditemukan dummy string' : 'Bebas dummy string');

    // C-ABI calls
    const cAbiFunctions = [
      '_whisper_init_from_buffer',
      '_whisper_full',
      '_whisper_full_n_segments',
      '_whisper_full_get_segment_text',
      '_whisper_free'
    ];

    for (const fn of cAbiFunctions) {
      const containsFn = engineCode.includes(fn);
      recordCheck(results.gate1, `Calls C-ABI function ${fn}`, containsFn, containsFn ? 'Ditemukan pemanggilan' : 'TIDAK ditemukan');
    }
  }

  // Check serve.js MIME
  if (!fs.existsSync(serveJsPath)) {
    recordCheck(results.gate1, 'serve.js exists', false, `Berkas tidak ditemukan: ${serveJsPath}`);
  } else {
    const serveCode = fs.readFileSync(serveJsPath, 'utf8');
    const hasBinMime = serveCode.includes("'.bin': 'application/octet-stream'");
    recordCheck(results.gate1, "serve.js handles '.bin': 'application/octet-stream'", hasBinMime, hasBinMime ? 'MIME .bin terkonfigurasi' : 'MIME .bin hilang');
  }

  // Check scripts/build-whisper-wasm.ps1
  const buildPs1Exists = fs.existsSync(buildPs1Path);
  recordCheck(results.gate1, 'scripts/build-whisper-wasm.ps1 exists on disk', buildPs1Exists, buildPs1Exists ? buildPs1Path : 'File hilang');

  results.gate1.passed = results.gate1.checks.every(c => c.passed);
  console.log(`Hasil Gate 1: ${results.gate1.passed ? 'PASSED (100%)' : 'FAILED'}\n`);

  // --------------------------------------------------------------------------
  // GATE 2: Zero Silent Fallback — Model Resolution
  // --------------------------------------------------------------------------
  console.log('--- Memeriksa GATE 2: Zero Silent Fallback — Model Resolution ---');
  try {
    const engineModuleUrl = pathToFileURL(engineJsPath).href;
    const { Pass2WasmGgmlEngine } = await import(engineModuleUrl);

    // 1. Model resolution tests
    const baseResolved = Pass2WasmGgmlEngine.resolveModelPath('Xenova/whisper-base');
    const expectedBase = '/models/ggml-base-id-q8_0.bin';
    recordCheck(
      results.gate2,
      'resolveModelPath("Xenova/whisper-base")',
      baseResolved === expectedBase,
      `Hasil: ${baseResolved}, Diharapkan: ${expectedBase}`
    );

    const smallResolved = Pass2WasmGgmlEngine.resolveModelPath('Xenova/whisper-small');
    const expectedSmall = '/models/ggml-small-id-q8_0.bin';
    recordCheck(
      results.gate2,
      'resolveModelPath("Xenova/whisper-small")',
      smallResolved === expectedSmall,
      `Hasil: ${smallResolved}, Diharapkan: ${expectedSmall}`
    );

    const tinyResolved = Pass2WasmGgmlEngine.resolveModelPath('Xenova/whisper-tiny');
    recordCheck(
      results.gate2,
      'resolveModelPath("Xenova/whisper-tiny") === null',
      tinyResolved === null,
      `Hasil: ${tinyResolved}, Diharapkan: null`
    );

    // 2. Test explicit error on unprovided model (whisper-tiny)
    const engine = new Pass2WasmGgmlEngine();
    let threwExplicitError = false;
    let errorMessage = '';

    try {
      await engine.init('Xenova/whisper-tiny', 'id');
    } catch (err) {
      threwExplicitError = true;
      errorMessage = err.message || '';
    }

    const hasFallbackNotice = errorMessage.includes('Zero Silent Fallback aktif');
    recordCheck(
      results.gate2,
      'init("Xenova/whisper-tiny") melempar error Zero Silent Fallback',
      threwExplicitError && hasFallbackNotice,
      `threw: ${threwExplicitError}, pesan error: "${errorMessage}"`
    );

    recordCheck(
      results.gate2,
      'engine.isReady === false setelah penolakan',
      engine.isReady === false,
      `isReady: ${engine.isReady}`
    );
  } catch (err) {
    recordCheck(results.gate2, 'Modul Pass2WasmGgmlEngine dapat dimuat dan diuji', false, err.message);
  }

  results.gate2.passed = results.gate2.checks.every(c => c.passed);
  console.log(`Hasil Gate 2: ${results.gate2.passed ? 'PASSED (100%)' : 'FAILED'}\n`);

  // --------------------------------------------------------------------------
  // GATE 3: Honest Uncompiled Binary Check & Local Model Files
  // --------------------------------------------------------------------------
  console.log('--- Memeriksa GATE 3: Honest Uncompiled Binary Check & Local Model Files ---');
  const baseModelDist = path.resolve(projectRoot, 'dist/models/ggml-base-id-q8_0.bin');
  const smallModelDist = path.resolve(projectRoot, 'dist/models/ggml-small-id-q8_0.bin');

  const baseExists = fs.existsSync(baseModelDist);
  let baseSizeBytes = baseExists ? fs.statSync(baseModelDist).size : 0;
  const baseSizeMb = (baseSizeBytes / (1000 * 1000)).toFixed(2);
  recordCheck(
    results.gate3,
    'dist/models/ggml-base-id-q8_0.bin ada di disk (> 80 MB)',
    baseExists && baseSizeBytes > 80 * 1000 * 1000,
    `Ukuran: ${baseSizeMb} MB (${baseSizeBytes} bytes)`
  );

  const smallExists = fs.existsSync(smallModelDist);
  let smallSizeBytes = smallExists ? fs.statSync(smallModelDist).size : 0;
  const smallSizeMb = (smallSizeBytes / (1000 * 1000)).toFixed(2);
  recordCheck(
    results.gate3,
    'dist/models/ggml-small-id-q8_0.bin ada di disk (> 250 MB)',
    smallExists && smallSizeBytes > 250 * 1000 * 1000,
    `Ukuran: ${smallSizeMb} MB (${smallSizeBytes} bytes)`
  );

  // Honest uncompiled binary check:
  // When web/vendor/whisper-cpp/whisper.js is not compiled, engine.init('Xenova/whisper-base')
  // must throw an honest error mentioning scripts/build-whisper-wasm.ps1,
  // leaving engine.available === false and engine.isReady === false.
  try {
    const engineModuleUrl = pathToFileURL(engineJsPath).href;
    const { Pass2WasmGgmlEngine } = await import(engineModuleUrl);
    const engine = new Pass2WasmGgmlEngine();

    const runtimePath = path.resolve(projectRoot, 'web/vendor/whisper-cpp/whisper.js');
    const runtimeExists = fs.existsSync(runtimePath);

    if (!runtimeExists) {
      let honestError = false;
      let errMsg = '';
      try {
        await engine.init('Xenova/whisper-base', 'id');
      } catch (err) {
        honestError = true;
        errMsg = err.message || '';
      }

      const mentionsBuildScript = errMsg.includes('scripts/build-whisper-wasm.ps1');
      recordCheck(
        results.gate3,
        'init("Xenova/whisper-base") melempar error jujur yang menyebut scripts/build-whisper-wasm.ps1',
        honestError && mentionsBuildScript,
        `errMsg: "${errMsg}"`
      );

      recordCheck(
        results.gate3,
        'engine.available === false saat binary belum ada',
        engine.available === false,
        `available: ${engine.available}`
      );

      recordCheck(
        results.gate3,
        'engine.isReady === false saat binary belum ada',
        engine.isReady === false,
        `isReady: ${engine.isReady}`
      );
    } else {
      recordCheck(
        results.gate3,
        'Biner web/vendor/whisper-cpp/whisper.js sudah ada (melewati pengujian uncompiled)',
        true,
        'Binary sudah terkompilasi di disk'
      );
    }
  } catch (err) {
    recordCheck(results.gate3, 'Honest uncompiled binary check execution', false, err.message);
  }

  results.gate3.passed = results.gate3.checks.every(c => c.passed);
  console.log(`Hasil Gate 3: ${results.gate3.passed ? 'PASSED (100%)' : 'FAILED'}\n`);

  // --------------------------------------------------------------------------
  // Summary
  // --------------------------------------------------------------------------
  const allPassed = results.gate1.passed && results.gate2.passed && results.gate3.passed;
  console.log('================================================================');
  console.log(`   STATUS KESELURUHAN VERIFIKASI: ${allPassed ? '100% LULUS (ALL GATES PASSED)' : 'ADA KEGAGALAN (FAILED)'}`);
  console.log('================================================================');

  if (!allPassed) {
    process.exit(1);
  }
}

runVerification().catch(err => {
  console.error('Fatal verification error:', err);
  process.exit(1);
});
