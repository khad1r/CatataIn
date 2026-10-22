// sw.js - Service Worker for Catetin PWA Offline Caching
const CACHE = 'catetin-pwa-v8';
const ASSETS = [
  './',
  './index.html',
  './style.css',
  './manifest.json',
  './catetin.wasm',
  './src/app.js',
  './src/audio-engine.js',
  './src/opfs-storage.js',
  './src/onnx-runner.js',
  './src/asr-worker.js',
  './src/engines/pass2-interface.js',
  './src/engines/pass2-webgpu.js',
  './src/engines/pass2-wasm-simd.js',
  './src/engines/pass2-webnn.js',
  './src/engines/hub.js',
  './src/sherpa/sherpa-transducer-worker.js',
  './src/sherpa/sherpa-onnx-wasm-main-asr.js',
  './src/sherpa/sherpa-onnx-asr.js',
  './src/sherpa/sherpa-onnx-wasm-main-asr.wasm',
  './src/components/vu-meter.js',
  './src/components/transcript-view.js',
  './src/components/notes-view.js',
  './src/components/catetin-telemetry.js'
];

self.addEventListener('install', (e) => {
  e.waitUntil(
    caches.open(CACHE)
      .then(c => c.addAll(ASSETS))
      .then(() => self.skipWaiting())
      .catch(err => console.warn('[SW] Cache prefill notice:', err))
  );
});

self.addEventListener('activate', (e) => {
  e.waitUntil(
    caches.keys().then(keys => Promise.all(keys.filter(k => k !== CACHE).map(k => caches.delete(k))))
      .then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', (e) => {
  if (!e.request.url.startsWith('http')) return;

  e.respondWith(
    caches.match(e.request).then(cached => {
      if (cached) return cached;
      return fetch(e.request).then(response => {
        if (e.request.method === 'GET' && response && response.status === 200) {
          const clone = response.clone();
          caches.open(CACHE).then(c => c.put(e.request, clone)).catch(() => {});
        }
        return response;
      }).catch(err => {
        // Prevent unhandled promise rejection in FetchEvent
        return new Response(null, { status: 503, statusText: 'Network Unavailable' });
      });
    })
  );
});
