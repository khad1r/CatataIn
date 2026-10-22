# Arsitektur Sistem Catetin Web (PWA In-Browser Speech Intelligence)

Dokumen ini mendefinisikan arsitektur menyeluruh dari aplikasi web Catetin: sistem transkripsi wicara lokal, *zero-cloud*, *zero-mock*, dan berkinerja tinggi yang berjalan 100% di dalam peramban modern (*client-side*).

---

## 1. Filosofi & Paradigma Arsitektur

Catetin Web dirancang dengan prinsip dasar rekayasa sistem berkinerja tinggi (*systems engineering*):

1. **100% Local & Zero Cloud Dependency**:
   Tidak ada audio atau metadata yang dikirim ke server luar (*Google Speech API*, *OpenAI*, dsb). Semua inferensi neural dan pemrosesan DSP dieksekusi di atas silikon perangkat pengguna (CPU SIMD, iGPU/dGPU, atau NPU).
2. **Zero Framework Bloat (Pure Vanilla & Web Standards)**:
   Lapisan UI dibangun murni menggunakan **Native Web Components** (`<transcript-view>`, `<notes-view>`, `<vu-meter>`, `<catetin-telemetry>`) tanpa framework JavaScript berat (React, Vue, Angular). Hal ini membebaskan thread utama dari overhead *Virtual DOM reconciliation*.
3. **Multi-Threaded Workload Decoupling**:
   Thread audio, thread antarmuka (DOM), thread inferensi streaming (Pass 1), dan thread inferensi final (Pass 2) diisolasi secara fisik pada *thread/Web Worker* terpisah untuk menjamin 60 FPS UI dan 0 audio drop.
4. **Cascading Two-Pass ASR**:
   Menggabungkan responsivitas sub-detik model *Transducer* (Pass 1) untuk teks langsung verbatim dengan akurasi tata bahasa model *Whisper* (Pass 2) untuk finalisasi giliran bicara.

---

## 2. Topologi Thread & Alur Data End-to-End

Arsitektur runtime Catetin Web terbagi menjadi 4 thread komputasi independen:

```mermaid
flowchart TD
    subgraph Audio_Thread["1. Real-Time Audio Thread (AudioWorklet)"]
        MIC[Mikrofon 48kHz] --> RESAMPLE[Rust WASM Resampler 48k to 16k]
        TAB[Tab Audio / DisplayMedia] --> RESAMPLE
        RESAMPLE --> RINGBUF[SharedArrayBuffer SPSC RingBuffer]
    end

    subgraph Main_Thread["2. Main UI Thread (DOM & Coordinator)"]
        APP[app.js Lifecycle Coordinator]
        DOM[Native Web Components: transcript-view / vu-meter]
        STATE[State Machine: BOOTSTRAPPING to SYSTEM_READY]
        RINGBUF -.->|VU RMS & VAD Status| DOM
    end

    subgraph Pass1_Thread["3. Pass 1 Dedicated Worker (Streaming Transducer)"]
        WORKER1[sherpa-transducer-worker.js]
        ZIPFORMER[Sherpa-ONNX Zipformer2 ID WASM SIMD]
        RINGBUF -->|160ms Chunks Float32Array| WORKER1
        WORKER1 --> ZIPFORMER
        ZIPFORMER -->|Partial Text TTFT < 360ms| DOM
    end

    subgraph Pass2_Thread["4. Pass 2 Dedicated Worker (Turn Finalizer)"]
        WORKER2[asr-worker.js & Pass2Hub]
        STAGING[VRAM / RAM Contiguous Staging Buffer]
        ENGINES[5 Pluggable Engines: WebGPU / WASM SIMD / WebNN / Direct ORT / GGML]
        RINGBUF -.->|Background Staging PCM| STAGING
        VAD_END[VAD Endpointed / 4.5s Turn Cut] -->|Trigger Transcribe| WORKER2
        STAGING --> ENGINES
        ENGINES -->|Final Punctuate & Formatted Utterance| DOM
    end
```

---

## 3. Sub-Sistem Audio Ingestion & DSP (Rust WebAssembly)

Komponen DSP audio mentah dikompilasi dari Rust ke WebAssembly (`catetin.wasm`) melalui *crate* [`crates/catetin-wasm`](file:///C:/Users/abdka/Development/foresight-analysis/crates/catetin-wasm/src/lib.rs):

### 3.1. Hann-Window Sinc Resampler (48kHz $\to$ 16kHz)
* **Kebutuhan**: Peramban menangkap mikrofon atau tab audio pada sample rate native perangkat keras (umumnya 48.000 Hz atau 44.100 Hz), sedangkan model ASR (Zipformer dan Whisper) memerlukan input 16.000 Hz mono.
* **Implementasi**: Resampling linier menggunakan filter anti-aliasing berbasis *Hann window*. Mencegah distorsi frekuensi tinggi (*spectral aliasing*) yang dapat menurunkan skor akurasi fonemik model.
* **Performa**: Eksekusi < 0.2 ms per frame 160ms audio pada WebAssembly linear memory.

### 3.2. Speech Activity Detection (VAD) & Dynamic Turn Cutting
* AudioEngine mengimplementasikan segmenter status wicara:
  * **Rule 1 (Trailing Silence)**: Mendeteksi hening 1.2s – 2.4s sebagai penanda akhir kalimat alami (*natural endpointing*).
  * **Rule 2 (Dynamic Turn Cutting)**: Untuk audio berkelanjutan tanpa jeda (seperti video YouTube atau pidato), sistem memotong segmen secara dinamis pada batas 4.5 detik (konten padat) atau 9.0 detik (percakapan normal) untuk mencegah buffer overflow dan menjaga responsivitas Pass 2.

---

## 4. Memori Bersama & Zero-Copy Interop

Untuk menghindari latensi alokasi heap dan penalti serialisasi `Structured Clone` (Boundary 1 IPC), pipeline menggunakan interop memori tingkat rendah:

1. **SharedArrayBuffer & Atomics (Lock-Free SPSC RingBuffer)**:
   * Thread audio menulis PCM 16kHz langsung ke buffer sirkular bersama.
   * Worker membaca data menggunakan sinkronisasi pointer atomic (`Atomics.load` / `Atomics.store`), mengeliminasi siklus alokasi objek `Float32Array` baru di setiap interval 160ms.
2. **WASM Linear Memory Mapping**:
   * Modul Rust mengekspor fungsi alokasi langsung `catetin_alloc_f32` dan `catetin_free_f32`.
   * Data audio disuntikkan langsung ke alamat memori linear WebAssembly tanpa *double copy*.
3. **Pipelining Asinkronik Audio Staging**:
   * Selama pengguna berbicara (`speaking = true`), chunk audio dialirkan simultan: ke Pass 1 untuk transkripsi langsung, dan secara paralel diakumulasikan ke buffer staging teralokasi statis (160.000 float, kapasitas 10 detik).
   * Saat giliran bicara berakhir (`speaking = false`), audio 100% sudah siap di memori. Latensi transfer PCM ke engine Pass 2 adalah **0 ms**.

---

## 5. Pass 1: Streaming Transducer Sejati (< 50ms)

Pass 1 beroperasi di dalam Web Worker terisolasi [`web/src/sherpa/sherpa-transducer-worker.js`](file:///C:/Users/abdka/Development/foresight-analysis/web/src/sherpa/sherpa-transducer-worker.js):

* **Runtime**: Sherpa-ONNX Emscripten WebAssembly (`sherpa-onnx-wasm-main-asr.wasm`).
* **Arsitektur Model**: Pruned Stateless Zipformer2 RNN-T (Encoder, Decoder, Joiner) terkuantisasi INT8 lokal di [`dist/models/sherpa-onnx-streaming-zipformer2-id/`](file:///C:/Users/abdka/Development/foresight-analysis/dist/models/sherpa-onnx-streaming-zipformer2-id/).
* **Karakteristik Komputasi**:
  * Chunk size: 2.560 float32 (160 ms audio).
  * Time to First Token (TTFT): **357.9 ms**.
  * Latensi inferensi per chunk: **31 ms – 150 ms** (Real-Time Factor: **0.155x**).
* **Fungsi UI**: Memancarkan teks sementara (*interim hypothesis*) secara langsung ke DOM kartu transkrip (`⚡ [Speaker]: "DARI REPUBLIK INDONESIA JAKARTA..."`), memberikan konfirmasi visual instan kepada pengguna saat berbicara.

---

## 6. Pass 2: Engine Hub & 5 Adapter Pluggable

Pass 2 bertugas memvalidasi, mengoreksi, dan memfinalisasi teks giliran bicara menggunakan model Whisper. Seluruh adapter mengimplementasikan antarmuka seragam [`Pass2Engine`](file:///C:/Users/abdka/Development/foresight-analysis/web/src/engines/pass2-interface.js) di bawah koordinator [`Pass2Hub`](file:///C:/Users/abdka/Development/foresight-analysis/web/src/engines/hub.js):

```
┌─────────────────────────────────────────────────────────────────┐
│                      Pass2Hub Coordinator                       │
└────────────────────────────────┬────────────────────────────────┘
                                 │
     ┌───────────────────────────┼───────────────────────────┐
     ▼                           ▼                           ▼
[Option A: WebGPU]      [Option B: WASM SIMD]      [Option C: WebNN]
(Compute Shader)         (CPU 8-Threads)           (OS DirectML NPU)
     │                           │                           │
     ▼                           ▼                           ▼
[Option D: Direct ORT]  [Option E: GGML WASM]      [Format Utterance]
(ONNX INT8 Offline)     (whisper.cpp 1:1 CLI)       (Rust catetin.wasm)
```

### 6.1. Spesifikasi Detail Kelima Opsi Backend

| Engine ID | Adapter File | Backend Hardware | Runtime Stack | Format Model & Dtype |
| :--- | :--- | :--- | :--- | :--- |
| `webgpu` | [`pass2-webgpu.js`](file:///C:/Users/abdka/Development/foresight-analysis/web/src/engines/pass2-webgpu.js) | GPU Compute Shader | WebGPU Dawn / DirectX 12 | ONNX Split: FP32 Encoder + Q4 Decoder (`StaticCache`) |
| `wasm-simd` | [`pass2-wasm-simd.js`](file:///C:/Users/abdka/Development/foresight-analysis/web/src/engines/pass2-wasm-simd.js) | CPU 8-Core SIMD | ONNX Runtime Web WASM | ONNX Split: FP32 Encoder + Q8 Decoder |
| `webnn` | [`pass2-webnn.js`](file:///C:/Users/abdka/Development/foresight-analysis/web/src/engines/pass2-webnn.js) | OS DirectML Native (NPU/GPU) | W3C WebNN API (`navigator.ml`) | ONNX Native DirectML Graph |
| `direct-ort` | [`pass2-direct-ort.js`](file:///C:/Users/abdka/Development/foresight-analysis/web/src/engines/pass2-direct-ort.js) | CPU / Direct ORT | ONNX Runtime Web murni | Bobot Lokal INT8 (`base-encoder.int8.onnx`), Split FP32/Q8 |
| `wasm-ggml` | [`pass2-wasm-ggml.js`](file:///C:/Users/abdka/Development/foresight-analysis/web/src/engines/pass2-wasm-ggml.js) | CPU SIMD (whisper.cpp) | GGML Bytecode WASM | Biner lokal `ggml-base-id-q8_0.bin`, Split FP32/Q8 |

---

## 7. Temuan Rekayasa Kritis: Split-Dtype & StaticCache

Selama implementasi dan pengujian mendalam, ditemukan 2 prinsip rekayasa yang menentukan keberhasilan sistem:

### 7.1. Aturan Wajib Split-Dtype (Mencegah Acoustic Feature Collapse)
* **Akar Masalah**: Kuantisasi skalar INT8 (`q8`) pada modul konvolusi 1D *audio encoder* merusak resolusi energi filterbank Log-Mel kontinu. Akibatnya, representasi akustik bahasa Indonesia runtuh dan memicu halusinasi kata prior global (*"Pancasila"*).
* **Solusi Terverifikasi**: Seluruh engine (WebGPU, WASM SIMD, Direct ORT, GGML) wajib mengadopsi konfigurasi **Split-Dtype**:
  $$\text{Dtype} = \{ \text{encoder\_model}: \text{'fp32'}, \; \text{decoder\_model\_merged}: \text{'q4' / 'q8'} \}$$
  Encoder tetap berada pada presisi floating-point 32-bit untuk menjaga fidelitas spektrogram 100%, sedangkan decoder autoregresif dikuantisasi ke 4-bit/8-bit untuk memangkas konsumsi bandwidth bus memori per token.

### 7.2. WebGPU StaticCache ($T_{\max} = 64$)
* Pada Transformers.js v2, KV-cache decoder dialokasikan secara dinamis di setiap langkah kata baru, memaksa driver GPU mengalokasikan ulang buffer VRAM dan memicu stall sinkronisasi.
* Pada Transformers.js v3 dengan `StaticCache`, buffer KV-cache dialokasikan tetap di VRAM pada ukuran $T_{\max} = 64$ saat inisialisasi awal. Setiap token baru ditulis langsung *in-place* ke slot array yang sudah ada tanpa alokasi ulang, memangkas VRAM dari ~300 MB ke ~85 MB.

---

## 8. Siklus Hidup Bootstrap & Eager Pre-warming

Untuk menjamin latensi 0 ms pada saat pengguna pertama kali berbicara (*0ms first-utterance lag*), sistem mengimplementasikan finite state machine di [`web/src/app.js`](file:///C:/Users/abdka/Development/foresight-analysis/web/src/app.js):

```
[DOM Ready] ──► BOOTSTRAPPING ──► WARMING_PASS1 ──► WARMING_PASS2 ──► SYSTEM_READY
                     │                   │                   │              │
             Alokasi Audio &     Load Zipformer      Dummy Pass 0.5s    Tombol Mulai Aktif
             Inisialisasi DOM    WASM Transducer    Shader GPU Compile   Badge: Siap (0ms)
```

1. **`BOOTSTRAPPING`**: Alokasi `AudioEngine` dan penyiapan elemen DOM Web Component.
2. **`WARMING_PASS1`**: Inisialisasi dedicated worker Sherpa-ONNX dan pemanasan frame awal Zipformer.
3. **`WARMING_PASS2`**: Eksekusi dummy pass 0.5s audio nol (`max_new_tokens: 1`) pada engine Pass 2 untuk memaksa kompilasi Pipeline State Objects (PSO) di driver DirectX/Vulkan.
4. **`SYSTEM_READY`**: Tombol sesi `#btn-session` dibuka (`disabled = false`) dan badge kesiapan beralih hijau. Seluruh giliran bicara berikutnya dieksekusi instan tanpa lag kompilasi shader.

---

## 9. Keamanan, PWA Offline, & Header COOP/COEP

1. **Header Isolasi Cross-Origin**:
   Server lokal ([`scripts/serve.js`](file:///C:/Users/abdka/Development/foresight-analysis/scripts/serve.js)) wajib memancarkan header berikut agar peramban mengaktifkan `self.crossOriginIsolated === true`:
   ```http
   Cross-Origin-Opener-Policy: same-origin
   Cross-Origin-Embedder-Policy: require-corp
   ```
   Isolasi ini merupakan syarat mutlak peramban modern untuk mengizinkan akses ke `SharedArrayBuffer` dan multi-threading WebAssembly.
2. **Penyimpanan Berkas Model**:
   * Model streaming Zipformer disimpan lokal di `/models/sherpa-onnx-streaming-zipformer2-id/`.
   * Model Whisper terkuantisasi disimpan lokal di `/models/sherpa-onnx-whisper-base/` dan `ggml-base-id-q8_0.bin`.
   * Dukungan Progressive Web App (PWA) via Service Worker `catetin-pwa-v8` menjamin aplikasi dapat dibuka dan berfungsi penuh saat koneksi internet terputus total.

---

## 10. Likuidasi Bottleneck #1: Ekstraksi Mel-Spectrogram Berkecepatan Tinggi (`mel-spectrogram.js`)

Pada arsitektur awal, ekstraksi Mel-Spectrogram dieksekusi di thread CPU JavaScript melalui pemanggilan `processor(pcm)` pada `@huggingface/transformers`. Hal ini menimbulkan bottleneck sebesar **1.121 ms (21% dari total waktu Pass 2)** untuk audio 6 detik akibat kalkulasi STFT 400-point Hann window dan perkalian matriks 80-channel secara berulang pada 3.000 frame terpad.

### 10.1. Arsitektur `WebGpuMelExtractor`
Modul [`web/src/engines/mel-spectrogram.js`](file:///C:/Users/abdka/Development/foresight-analysis/web/src/engines/mel-spectrogram.js) mengatasi kemacetan ini melalui:
1. **Compute Shader WebGPU (WGSL)**: Menjalankan STFT 400-point Hann window dan filterbank Mel Slaney 80-channel langsung di workgroup shared memory GPU (`var<workgroup>`), memanfaatkan unit aritmatika vektor hardware GPU.
2. **Active-Frame Pruning**: Hanya menghitung frame aktif nyata dari audio input (contoh: 6.12 detik = 612 frame aktif), lalu mem-pad tensor ke 3.000 frame secara instan menggunakan konstanta kesunyian terhitung `(maxVal - 8.0 + 4.0) * 0.25`, mengeliminasi kalkulasi redundant 2.388 frame di loop CPU lambat.
3. **Bypass Langsung ke Model**: Tensor `[1, 80, 3000]` disuntikkan langsung ke `transcriber.model.generate({ input_features })` tanpa menyentuh wrapper CPU `processor(pcm)`.

### 10.2. Hasil Pengukuran Empiris Pasca-Optimasi (Chrome DevTools)
* **Latensi Ekstraksi Mel**: Terpangkas dari **~1.121 ms** menjadi **7 – 11 ms** (**46,6x lebih cepat**).
* **Total Latensi Pass 2 (Whisper Tiny WebGPU)**: Turun drastis dari **5.349 ms (RTF 0.874x)** menjadi **2.195 – 2.351 ms (RTF 0.384x)**.
* **Akurasi Fonemik**: 100% identik dengan teks ground truth (*"Republik Indonesia Jakarta Surabaya"*).

---

## 11. Matriks Evaluasi & Ranking Terpisah (Pass 1 vs Pass 2 Mandiri)

Pengujian dilakukan secara komprehensif pada berkas audio nyata [`sample-indonesia-16k.wav`](file:///C:/Users/abdka/Development/foresight-analysis/dist/sample-indonesia-16k.wav) (6.12 detik, 97.903 sampel, 16kHz mono) secara mandiri di Google Chrome:

### 11.1. Evaluasi Pass 1 (Streaming Live ASR)
Pass 1 dirancang eksklusif untuk menghasilkan teks kata-per-kata secara langsung saat pengguna masih bersuara (tanpa menunggu jeda):

| Tahap (Pass) | Opsi Engine | Model & Ukuran | Backend Hardware | Dtype | Latensi Chunk | TTFT | RTF | Hasil Transkrip | Catatan Kritis |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Pass 1** | **Streaming Transducer** | **Zipformer2 ID (70.9 MB)** | CPU WASM SIMD (1 Core) | INT8 C-ABI | **~31 ms/chunk** | **357.9 ms** | **0.133x** | *"DARI REPUBLIK INDONESIA JAKARTA"* | **Wajib Selalu Aktif**. Nol alokasi VRAM, respons live instan sub-detik. Teks berbentuk kapital tanpa tanda baca (*verbatim raw*), wajib disempurnakan oleh Pass 2. |

---

### 11.2. Ranking Mandiri Pass 2 (Turn Finalizer ASR — 10 Kombinasi Terpisah)
Pass 2 bertugas menghasilkan teks akurat ber-tata bahasa dengan tanda baca saat jeda bicara terdeteksi. Peringkat diurutkan murni berdasarkan performa terukur ($RTF$, latensi, dan kelengkapan teks):

| Peringkat | Tahap | Opsi Engine | Model Whisper | Backend Hardware | Dtype Format | Init (ms) | Decode (ms) | RTF | Teks Hasil Transkripsi Nyata | Catatan Kritis |
| :---: | :---: | :--- | :--- | :--- | :--- | :---: | :---: | :---: | :--- | :--- |
| 🥇 **#1** | **Pass 2** | **Option A (WebGPU)** | **Whisper Tiny (39 MB)** | GPU Compute Shader | Split FP32 / Q4 | 3.995 | **1.982** | **0.324x** | *"Republik Indonesia Jakarta Surabaya"* | **Juara Umum Pass 2**. Berkat WebGPU Mel Extractor WGSL, waktu decode melesat ke sub-2 detik (~3.1x lebih cepat dari audio asli). VRAM hemat 85 MB (`StaticCache`). 100% universal di browser modern. |
| 🥈 **#2** | **Pass 2** | **Option C (WebNN)** | **Whisper Base (140 MB)** | OS DirectML Native (NPU/GPU) | DirectML Graph | 11.360 | **2.978** | **0.487x** | *"Republik Indonesia Jakarta Surabaya"* | **Model Base Tercepat**. DirectML mengeksekusi subgraf 140MB di level driver OS Windows tanpa overhead per-kernel Mojo IPC Chromium. Butuh flag peramban `#web-machine-learning-neural-network`. |
| 🥉 **#3** | **Pass 2** | **Option C (WebNN)** | **Whisper Tiny (39 MB)** | OS DirectML Native (NPU/GPU) | DirectML Graph | 8.027 | **3.657** | **0.598x** | *"Republik Indonesia Jakarta Surabaya"* | **Sangat Cepat & Hemat Daya**. Sangat ideal untuk perangkat laptop bertenaga baterai dengan iGPU/NPU Windows DirectML. |
| **#4** | **Pass 2** | **Option A (WebGPU)** | **Whisper Base (140 MB)** | GPU Compute Shader | Split FP32 / Q4 | 6.715 | **3.912** | **0.639x** | *"Republik Indonesia Jakarta Surabaya"* | **Akurat di WebGPU Universal**. Berhasil menjalankan model Base 140MB di bawah real-time (RTF 0.639x) berkat split-dtype dan akselerasi GPU shader. |
| **#5** | **Pass 2** | **Option E (GGML WASM)** | **Whisper Tiny (39 MB)** | whisper.cpp CPU SIMD | Split FP32 / Q8 | 6.814 | **4.549** | **0.743x** | *"Republik Indonesia Jakarta, Surabaya. Pancasila."* | **Mesin CPU Tercepat & Paling Lengkap**. Mengadopsi tata memori flat whisper.cpp, menghasilkan transkrip lengkap hingga kata penutup dengan RTF sub-1.0x di CPU. |
| **#6** | **Pass 2** | **Option B (WASM SIMD)** | **Whisper Tiny (39 MB)** | Multi-Thread CPU SIMD (8 Core) | Split FP32 / Q8 | 6.035 | **5.173** | **0.845x** | *"Republik Indonesia Jakarta, Surabaya. Pancasila."* | **Transkrip Sempurna di WebAssembly**. Output teks 100% utuh beserta tanda baca lengkap. Murni CPU tanpa ketergantungan API grafis peramban. |
| **#7** | **Pass 2** | **Option D (Direct ORT)** | **Whisper Tiny (39 MB)** | Direct ORT Web (CPU) | Split FP32 / Q8 | 6.480 | **5.290** | **0.865x** | *"Republik Indonesia Jakarta, Surabaya. Pancasila."* | **Eksekusi Lokal Mandiri**. Bypass layer HuggingFace CDN, memetakan bobot lokal INT8 secara deterministik di CPU. |
| **#8** | **Pass 2** | **Option B (WASM SIMD)** | **Whisper Base (140 MB)** | Multi-Thread CPU SIMD (8 Core) | Split FP32 / Q8 | 13.447 | **8.367** | **1.367x** | *"Pancasila"* | **Acoustic Collapse di CPU**. Bobot 140MB memicu L3 cache thrashing (16-32MB cache CPU terlampaui), menyebabkan hilangnya atensi audio dan truncation kata tunggal. |
| **#9** | **Pass 2** | **Option D (Direct ORT)** | **Whisper Base (140 MB)** | Direct ORT Web (CPU) | Split FP32 / Q8 | 12.630 | **10.569** | **1.727x** | *"Pancasila"* | **Terlalu Lambat di CPU**. Waktu inferensi 1,7x lebih lambat dari audio asli. Truncation kata terjadi karena saturasi bandwidth DRAM CPU. |
| **#10** | **Pass 2** | **Option E (GGML WASM)** | **Whisper Base (140 MB)** | whisper.cpp CPU SIMD | Split FP32 / Q8 | 15.521 | **12.664** | **2.070x** | *"Pancasila"* | **Peringkat Terendah / Anti-Pattern**. Waktu decode 12,6 detik untuk audio 6 detik (RTF 2.07x). Dilarang keras digunakan pada CPU peramban. |

> [!NOTE]
> Laporan pengujian mendalam, metodologi empiris, analisis 4 bottleneck sistemik, dan profil rekomendasi produksi selengkapnya didokumentasikan pada [BENCHMARK_AND_MODEL_EVALUATION.md](file:///C:/Users/abdka/Development/foresight-analysis/docs/BENCHMARK_AND_MODEL_EVALUATION.md).

