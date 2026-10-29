import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const jsepJsPath = path.resolve('web/vendor/transformers/ort-wasm-simd-threaded.jsep.mjs');
const { default: ortWasmFactory } = await import(pathToFileURL(jsepJsPath).href);
const ort = await ortWasmFactory();
console.log('ORT keys:', Object.keys(ort).filter(k => k.startsWith('_Ort') || k.startsWith('Ort')));
