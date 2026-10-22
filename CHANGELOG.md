# Changelog

Semua perubahan penting pada proyek Catetin akan dicatat dalam berkas ini. Format mengacu pada [Keep a Changelog](https://keepachangelog.com/id/1.1.0/).

---

## [Unreleased]

### Ditambahkan
- **Implementasi State Machine Bootstrap & Eager Pre-warming Lifecycle (Tahap 1):**
  - Transisi status otomatis saat peramban dimuat: `BOOTSTRAPPING` $\to$ `WARMING_PASS1` $\to$ `WARMING_PASS2` $\to$ `SYSTEM_READY`.
  - Indikator kesiapan visual `#bootstrap-status` (`.bootstrap-badge`) pada UI dengan kunci interaksi `#btn-session` (`disabled`) sampai semua shader GPU/DirectML selesai dikompilasi (0ms first-utterance lag).
- **Optimasi Staging Buffer & Pipelining Asinkronik (Tahap 2 & 5):**
  - Buffer staging teralokasi statis 160.000 float (10 detik) pada `audio-engine.js`.
  - Penghapusan alokasi heap berulang dan pengaliran PCM simultan ke Pass 1 dan staging buffer (latensi transfer 0 ms saat turn selesai).
- **Upgrade Pass 2 ke Transformers.js v3 & StaticCache (Tahap 3):**
  - Migrasi `pass2-webgpu.js`, `pass2-wasm-simd.js`, dan `pass2-webnn.js` ke `@huggingface/transformers@3.3.3`.
  - Aktivasi alokasi statis KV-cache ($T_{\max} = 64$) dan konfigurasi split-dtype (`fp32` encoder, `q4` decoder) untuk eliminasi GPU reallocation stutter.
- **Ekstensi Adapter Low-Level Baru pada Pass2Hub (Tahap 4):**
  - `pass2-direct-ort.js`: Adapter ONNX Runtime Web murni dengan bobot INT8 terkuantisasi lokal.
  - `pass2-wasm-ggml.js`: Adapter whisper.cpp GGML WASM SIMD dengan model Q8_0 lokal (`ggml-base-id-q8_0.bin`).
  - Pendaftaran kedua adapter ke koordinator `Pass2Hub` dan modal Pengaturan tanpa refactoring lapisan aplikasi.
- **Likuidasi Bottleneck #1 Ekstraksi Mel-Spectrogram (`mel-spectrogram.js`):**
  - Implementasi `WebGpuMelExtractor` menggunakan compute shader WebGPU (WGSL) dengan Slaney 80-channel filterbank dan active-frame pruning.
  - Memotong latensi ekstraksi Mel dari baseline CPU ~1.121 ms menjadi **7 – 11 ms** (**46,6x lebih cepat**).
  - Menurunkan total latensi Pass 2 Whisper Tiny WebGPU ke **2.195 – 2.351 ms** (**RTF 0.384x**, ~2.6x lebih cepat dari durasi audio nyata).
- **Suite Benchmark Komparasi Empiris Utuh (`benchmark.html`):**
  - Pengujian head-to-head seluruh engine tanpa mock data menggunakan sampel audio nyata Bahasa Indonesia (`dist/sample-indonesia-16k.wav`, 6.12 detik, 97.903 sampel, 16kHz mono).
  - Hasil pengukuran performa empiris pasca-fix secara terpisah (Pass 1 Streaming vs Pass 2 Mandiri 10 Kombinasi):
    - **Pass 1 Streaming (Zipformer2 ID 70.9MB - WASM SIMD 1 Core)**: Decode 816.3 ms, TTFT 357.9 ms, RTF 0.133x, Teks: *"DARI REPUBLIK INDONESIA JAKARTA"*.
    - **Pass 2 Mandiri (Turn Finalizer - 10 Kombinasi Terpisah Tanpa Digabung)**:
      1. Option A (WebGPU) - Whisper Tiny (39MB) [Q4]: Decode **1.982 ms**, RTF **0.324x** (*"Republik Indonesia Jakarta Surabaya"*)
      2. Option C (WebNN) - Whisper Base (140MB) [DirectML]: Decode **2.978 ms**, RTF **0.487x** (*"Republik Indonesia Jakarta Surabaya"*)
      3. Option C (WebNN) - Whisper Tiny (39MB) [DirectML]: Decode **3.657 ms**, RTF **0.598x** (*"Republik Indonesia Jakarta Surabaya"*)
      4. Option A (WebGPU) - Whisper Base (140MB) [Q4]: Decode **3.912 ms**, RTF **0.639x** (*"Republik Indonesia Jakarta Surabaya"*)
      5. Option E (GGML WASM) - Whisper Tiny (39MB) [Q8]: Decode **4.549 ms**, RTF **0.743x** (*"Republik Indonesia Jakarta, Surabaya. Pancasila."*)
      6. Option B (WASM SIMD) - Whisper Tiny (39MB) [Q8]: Decode **5.173 ms**, RTF **0.845x** (*"Republik Indonesia Jakarta, Surabaya. Pancasila."*)
      7. Option D (Direct ORT) - Whisper Tiny (39MB) [Q8]: Decode **5.290 ms**, RTF **0.865x** (*"Republik Indonesia Jakarta, Surabaya. Pancasila."*)
      8. Option B (WASM SIMD) - Whisper Base (140MB) [Q8]: Decode **8.367 ms**, RTF **1.367x** (*"Pancasila"*)
      9. Option D (Direct ORT) - Whisper Base (140MB) [Q8]: Decode **10.569 ms**, RTF **1.727x** (*"Pancasila"*)
      10. Option E (GGML WASM) - Whisper Base (140MB) [Q8]: Decode **12.664 ms**, RTF **2.070x** (*"Pancasila"*)
- **Dokumentasi Spesifikasi Arsitektur Mendalam:**
  - `docs/BENCHMARK_AND_MODEL_EVALUATION.md`: Laporan komprehensif benchmark empiris, evaluasi model, pemisahan ranking Pass 1 vs Pass 2, serta analisis akar bottleneck hardware/software.
  - `docs/CATETIN_WEB_SYSTEM_ARCHITECTURE.md`: Arsitektur sistem web PWA Catetin, penyelesaian bottleneck Mel WGSL, dan matriks performa terpisah.
  - `docs/ASR_PIPELINE_LOWLEVEL_BOTTLENECK_ANALYSIS.md`: Pemetaan 8 pilar arsitektur Catetin, 5 irisan sinergi teknis, dan 6 tahap implementasi & verifikasi whitebox tanpa mock.
  - ADR #14: Eager State Machine Lifecycle & Zero-Lag Ready Indicator.
  - ADR #15: StaticCache, WebNN DirectML, dan Ekstensi Adapter Pass 2 Hub.

### Diubah
- `web/src/app.js`: Implementasi state machine eager pre-warming bootstrap dan sinkronisasi status UI.
- `web/src/audio-engine.js`: Pre-alokasi contiguous staging buffer dan optimasi transferable streaming.
- `web/src/engines/pass2-webgpu.js`: Migrasi Transformers.js v3 dengan static caching.
- `web/src/engines/pass2-webnn.js`: Upgrade ke Transformers.js v3 dengan DirectML/WebGPU pipeline.
- `web/src/engines/hub.js`: Dukungan registrasi 5 engine Pass 2 terpadu.
- `web/index.html` & `web/style.css`: Badge kesiapan sistem dan dropdown 5 backend Pass 2.
- Sinkronisasi seluruh berkas dari `web/` ke `dist/web/`.
  - Integrasi C-ABI `SherpaOnnxOnlineRecognizer` WebAssembly (`sherpa-onnx-wasm-main-asr.wasm`) untuk model streaming Zipformer2 Bahasa Indonesia (`sherpa-onnx-streaming-zipformer2-id`) dan Inggris (`sherpa-onnx-streaming-zipformer-en-2023-06-26`).
  - Pemrosesan frame audio 160ms secara live (< 50ms latensi) pada dedicated Web Worker (`sherpa-transducer-worker.js`) dengan zero-copy `ArrayBuffer` transfer list.
  - Rendering live turn verbatim real-time pada `<transcript-view>` (`⚡ [Speaker]: "kata..." [X.Xs]`) tanpa menunggu turn selesai.
  - Reset stream otomatis saat turn berakhir via VAD silence atau turn cutting (4.5s/9s) sebelum diserahkan ke Pass 2.
- **Arsitektur Modular Pass 2 Hub & Adapter Pattern:**
  - Kontrak interface baku `Pass2Engine` (`web/src/engines/pass2-interface.js`) dengan metode wajib `init()`, `warmup()`, `transcribe()`, dan `destroy()`.
  - Tiga adapter engine Pass 2 mandiri:
    1. `pass2-webgpu.js`: Akselerasi GPU compute shader via Transformers.js / ONNX Runtime Web.
    2. `pass2-wasm-simd.js`: Multi-threaded CPU SIMD murni (`SharedArrayBuffer`, ggml/ONNX), bebas overhead driver GPU.
    3. `pass2-webnn.js`: Akselerasi hardware native OS melalui W3C WebNN API (`navigator.ml` / DirectML).
  - Koordinator tunggal `Pass2Hub` (`web/src/engines/hub.js`) untuk peralihan engine dinamis tanpa refactoring pipeline UI.
  - Kontrak *pre-warming* pipeline: dummy pass 0.5s audio nol saat startup untuk eliminasi PSO compilation stall (latensi 2–5 detik pada turn pertama).
- **Diagnostik & Telemetri Terpadu:**
  - Penambahan Thread 3 (Pass 1 Streaming Transducer Worker) ke tab *Thread Stack & Topology* pada Custom Element `<catetin-telemetry>`.
  - Filter log diagnostik real-time berlabel `TRANSDUCER` dengan pewarnaan visual dedicated.
  - Opsi dropdown pemilihan Engine Pass 2 dan Model Transducer pada modal Pengaturan (`⚙️`).
- **Dokumentasi Keputusan Arsitektur:**
  - ADR #12: Streaming Transkripsi Live Pass 1, Restorasi Thread Stack Drawer, dan Disertasi WebGPU vs CLI.
  - ADR #13: Arsitektur Modular Pass 2 (Adapter Pattern Hub) dan Pipeline Pre-Warming.

### Diubah
- `web/src/audio-engine.js`: Memancarkan `onTransducerChunk` setiap frame audio resampled (16kHz) dan `onTurnFinished` saat VAD endpointed atau dynamic cut.
- `web/src/onnx-runner.js`: Mengelola siklus hidup dedicated Transducer Worker bersama Pass 2 Worker.
- `web/src/app.js`: Menghubungkan frame Transducer langsung ke runner dan memperbarui visualizer transkrip live.
- `web/sw.js`: Registrasi Service Worker `catetin-pwa-v8` mencakup runtime Sherpa WASM dan Transducer worker.

---

## [0.2.0] - 2026-10-18

### Ditambahkan
- Catetin Web PWA: Antarmuka berbasis Native Web Components tanpa framework JS runtime.
- Modul Rust WebAssembly `catetin.wasm`: Hann-window anti-aliasing resampler (48k -> 16k mono), kalkulasi RMS VU meter, dan `SpeechSegmenter`.
- Dukungan Multi-Tab Audio via `getDisplayMedia` dengan deduplikasi track.
- Penyimpanan model lokal via Origin Private File System (OPFS) dan File System Access API.
- Dynamic Turn Cutting (4.5s vs 9s) untuk menjaga responsivitas pipeline audio berkelanjutan (YouTube/video).

---

## [0.1.0] - 2026-10-15

### Ditambahkan
- Dual-pass ASR native Rust CLI: Pass 1 Sherpa-ONNX Zipformer RNN-T streaming (< 50ms) + Pass 2 Sherpa-ONNX Whisper offline.
- Pure ONNX Runtime direct DLL linking (menghapus total dependensi whisper.cpp / GGML C++).
- CLI minimalist single-config interactive menu (`catetin -c`).
- Dual audio capture (mikrofon lokal + loopback Windows WASAPI / Linux ALSA).
- AI Copilot assistant untuk ringkasan rapat dan tanya jawab instan.
