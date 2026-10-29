// sw.js - Service Worker for Catetin PWA Offline Caching
const CACHE = 'catetin-pwa-v12';
const CACHE_NAME = CACHE;
const ASSETS = [
  './',
  './index.html',
  './style.css',
  './manifest.json',
  './catetin.wasm',
  './src/app.js',
  './src/audio-engine.js',
  './src/onnx-runner.js',
  './src/asr-worker.js',
  './src/workers/readiness.worker.js',
  './src/engines/pass2-interface.js',
  './src/engines/pass2-webgpu.js',
  './src/engines/pass2-wasm-simd.js',
  './src/engines/pass2-webnn.js',
  './src/engines/hub.js',
  './src/engines/mel-spectrogram.js',
  './src/engines/pass2-direct-ort.js',
  './src/engines/pass2-wasm-ggml.js',
  './src/sherpa/sherpa-transducer-worker.js',
  './src/sherpa/sherpa-onnx-wasm-main-asr.js',
  './src/sherpa/sherpa-onnx-asr.js',
  './src/sherpa/sherpa-onnx-wasm-main-asr.wasm',
  './src/components/vu-meter.js',
  './src/components/transcript-view.js',
  './src/components/notes-view.js',
  './src/components/catetin-telemetry.js',
  './vendor/transformers/transformers.min.js',
  './vendor/transformers/ort-wasm-simd-threaded.jsep.mjs',
  './vendor/transformers/ort-wasm-simd-threaded.jsep.wasm'
];

self.addEventListener('install', (e) => {
  e.waitUntil(
    caches.open(CACHE)
      .then(c => c.addAll(ASSETS))
      .then(() => self.skipWaiting())
  );
});

self.addEventListener('activate', (e) => {
  e.waitUntil(
    caches.keys().then(keys => Promise.all(keys.filter(k => k !== CACHE).map(k => caches.delete(k))))
      .then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', (e) => {
  const url = new URL(e.request.url);
  if (e.request.method !== 'GET' || url.origin !== self.location.origin) return;

  const isVendorAsset = url.pathname.includes('/vendor/');
  const isAppFile = !isVendorAsset && (
                    url.pathname.endsWith('/') ||
                    url.pathname.endsWith('.html') ||
                    url.pathname.endsWith('.js') ||
                    url.pathname.endsWith('.css') ||
                    url.pathname.endsWith('.json'));

  if (isAppFile) {
    e.respondWith(
      fetch(e.request)
        .then(response => {
          if (response.ok) {
            const clone = response.clone();
            caches.open(CACHE).then(c => c.put(e.request, clone));
          }
          return response;
        })
        .catch(() => caches.match(e.request))
    );
  } else {
    e.respondWith(
      caches.match(e.request).then(cached => {
        if (cached) return cached;
        return fetch(e.request).then(response => {
          if (response.ok) {
            const clone = response.clone();
            caches.open(CACHE).then(c => c.put(e.request, clone));
          }
          return response;
        }).catch(() => Response.error());
      })
    );
  }
});
