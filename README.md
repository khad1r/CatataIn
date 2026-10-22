# Catetin 🎙️

> **Aplikasi Notulensi & Transkripsi Rapat Offline, Driverless, dan Dual-Pass ASR untuk Web (PWA), Desktop, dan Mobile.**  
> 100% Berjalan Lokal di Peramban Pengguna • Nol API Cloud • Nol Biaya Layanan Eksternal.

---

## Daftar Isi
1. [Ringkasan Arsitektur](#ringkasan-arsitektur)
2. [Arsitektur Dual-Pass ASR](#arsitektur-dual-pass-asr)
3. [Spesifikasi Model Pass 1 (Streaming Live)](#spesifikasi-model-pass-1-streaming-live)
4. [Spesifikasi Model & 5 Engine Pass 2 (Turn Finalizer)](#spesifikasi-model--5-engine-pass-2-turn-finalizer)
5. [Status Eksperimental: Option C (WebNN DirectML)](#status-eksperimental-option-c-webnn-directml)
6. [Tabel Evaluasi & Ranking Empiris Terpisah](#tabel-evaluasi--ranking-empiris-terpisah)
7. [Penyelesaian Bottleneck Mel-Spectrogram (WGSL)](#penyelesaian-bottleneck-mel-spectrogram-wgsl)
8. [Panduan Menjalankan Proyek](#panduan-menjalankan-proyek)
9. [Struktur Proyek](#struktur-proyek)
10. [Indeks Dokumentasi Teknis](#indeks-dokumentasi-teknis)

---

## Ringkasan Arsitektur

Catetin mengadopsi arsitektur **Dual-Pass Asynchronous ASR** yang memisahkan beban kerja antara responsivitas live instan saat pengguna berbicara (*Pass 1*) dan akurasi tata bahasa ber-tanda baca saat jeda bicara terdeteksi (*Pass 2*). Seluruh model diunduh satu kali ke penyimpanan lokal browser (Cache Storage / IndexedDB) dan dieksekusi secara lokal murni menggunakan akselerasi WebAssembly SIMD, WebGPU, atau WebNN DirectML.

```
[ Mikrofon / Loopback Audio 16kHz ]
                 │
                 ├──► [ Pass 1: Zipformer2 Streaming WASM ] ──► Teks Live Interim (Per 160ms)
                 │                                                (Kapital, 0ms latency)
                 │
            [ VAD Pause ]
                 │
                 └──► [ Pass 2: Whisper Turn Finalizer ] ──────► Teks Final Sempurna
                           ├── Option A: WebGPU (Default)         (Tata bahasa + Tanda baca)
                           ├── Option B: WASM SIMD Multi-Core
                           ├── Option C: WebNN DirectML [Eksperimental]
                           ├── Option D: Direct ORT Web
                           └── Option E: whisper.cpp GGML WASM
```

---

## Arsitektur Dual-Pass ASR

| Kategori | Pass 1 (Streaming Live ASR) | Pass 2 (Turn Finalizer ASR) |
| :--- | :--- | :--- |
| **Tujuan Arsitektur** | Responsivitas instan kata-per-kata ke layar saat berbicara | Akurasi maksimal, kapitalisasi, dan tanda baca saat hening |
| **Model Default** | **Zipformer2 ID (70.9 MB)** | **Whisper Tiny (39 MB) / Base (140 MB)** |
| **Pemicu Eksekusi** | Terus-menerus per chunk audio 160 ms (2.560 sampel) | Asinkron saat jeda bicara terdeteksi oleh VAD |
| **Format Bobot** | INT8 C-ABI WebAssembly | Split FP32 (Encoder) + Q4/Q8 (Decoder) |
| **Latensi Tipikal** | **~31 ms/chunk** (TTFT **357.9 ms**) | **1.982 – 5.173 ms** (RTF **0.324x – 0.845x**) |
| **Karakteristik Teks** | Teks mentah kapital tanpa tanda baca (*verbatim raw*) | Teks rapi ber-tata bahasa dengan koma & titik |

---

## Spesifikasi Model Pass 1 (Streaming Live)

### Model: Zipformer2 ID (70.9 MB)
* **Arsitektur**: Pruned Stateless Emformer / RNN-T Streaming Transducer bahasa Indonesia.
* **Backend Hardware**: WebAssembly SIMD (1 Core CPU) via C-ABI murni (`catetin-wasm`).
* **Kelebihan**:
  * Konsumsi VRAM **0 MB**.
  * Latensi komputasi per chunk 160 ms hanya **~31 ms** (RTF **0.133x**, ~7.5x lebih cepat dari suara asli).
  * Menghadirkan *Time-to-First-Token* (TTFT) instan **357.9 ms** saat pengguna mulai berbicara.
* **The Catch (Kelemahan & Batasan)**:
  * **Tanpa Tanda Baca & Huruf Kecil**: Hanya mengeluarkan huruf kapital (`DARI REPUBLIK INDONESIA JAKARTA`).
  * **Kosakata Terbatas**: Bergantung ketat pada kamus token RNN-T; rentan salah fonem pada kata serapan baru atau aksen daerah ekstrem.
  * **Wajib Didampingi Pass 2**: Tidak dapat digunakan sebagai transkripsi final dokumen tanpa koreksi turn finalizer.

---

## Spesifikasi Model & 5 Engine Pass 2 (Turn Finalizer)

Pass 2 menyediakan 5 opsi mesin inferensi modular di bawah kontrak antarmuka seragam `Pass2EngineInterface`:

### 1. Option A: WebGPU Shader Compute (Rekomendasi Default Universal)
* **Model**: Whisper Tiny (39 MB) / Whisper Base (140 MB).
* **Format Bobot**: Split FP32 (Encoder) / Q4 (Decoder) + `StaticCache` ($T_{\max} = 64$).
* **Akselerator**: WebGPU API via compute shader WGSL dengan custom `WebGpuMelExtractor`.
* **Kelebihan**:
  * **Juara Umum Pass 2**: Latensi decode Tiny hanya **1.982 ms (RTF 0.324x)**.
  * Hemat VRAM (~85 MB untuk Tiny, ~195 MB untuk Base).
  * Berjalan secara *out-of-the-box* di Google Chrome, Microsoft Edge, dan browser modern tanpa konfigurasi khusus.
* **The Catch**:
  * Menuntut GPU yang mendukung WebGPU.
  * Autoregressive loop sekuensial ($B = 1$) dibatasi oleh memori bandwidth transfer bus host-to-device.

---

### 2. Option B: WASM SIMD Multi-Core (Fallback CPU Deterministik)
* **Model**: Whisper Tiny (39 MB) / Whisper Base (140 MB).
* **Format Bobot**: Split FP32 / Q8.
* **Akselerator**: WebAssembly multi-threading SIMD 128-bit (8 Core CPU).
* **Kelebihan**:
  * Nol ketergantungan driver kartu grafis; berjalan deterministik di semua arsitektur CPU x86/ARM.
  * Menghasilkan teks transkripsi paling lengkap hingga tanda baca penutup (*"Republik Indonesia Jakarta, Surabaya. Pancasila."*).
* **The Catch**:
  * **Whisper Tiny**: Latensi decode **5.173 ms (RTF 0.845x)**; konsumsi baterai dan temperatur CPU lebih tinggi dibanding WebGPU.
  * **Whisper Base (Anti-Pattern / Dilarang)**: Bobot 140 MB melampaui kapasitas L3 cache CPU (16–32 MB). Terjadi *cache thrashing* masif yang memicu **Acoustic Collapse** (output terpotong menjadi hanya kata *"Pancasila"*) dan latensi membengkak ke **8.367 ms (RTF 1.367x)**.

---

### 3. Option C: WebNN DirectML (Akselerasi Driver OS — Tahap Eksperimental)
* **Model**: Whisper Base (140 MB) / Whisper Tiny (39 MB).
* **Format Bobot**: DirectML Graph compiled.
* **Akselerator**: W3C Web Neural Network API (`DirectML.dll` native Windows GPU/NPU).
* **Kelebihan**:
  * **Model Base Tercepat**: Waktu decode Base 140 MB hanya **2.978 ms (RTF 0.487x)**.
  * Subgraf dieksekusi langsung di level driver OS Windows, mengeliminasi overhead per-kernel Mojo IPC Chromium.
* **The Catch**:
  * **Wajib Mengaktifkan Flag Peramban**: Memerlukan `chrome://flags/#web-machine-learning-neural-network`.
  * **Tahap Eksperimental**: Implementasi spesifikasi WebNN masih dalam status draft W3C dan perilisan vendor bertahap.
  * Waktu kompilasi awal (*warmup / init*) lebih lama (8 – 11 detik) sebelum sesi pertama siap.

---

### 4. Option D: Direct ORT Web (Eksekusi Lokal Mandiri)
* **Model**: Whisper Tiny (39 MB) / Whisper Base (140 MB).
* **Format Bobot**: Model lokal INT8 (`base-encoder.int8.onnx` & `base-decoder.int8.onnx`).
* **Akselerator**: ONNX Runtime Web murni tanpa perantara Hugging Face CDN.
* **Kelebihan**:
  * Kemandirian aset 100% lokal tanpa resolver eksternal.
  * Menghasilkan teks lengkap pada Whisper Tiny (**5.290 ms, RTF 0.865x**).
* **The Catch**:
  * Eksekusi model Base pada CPU sangat lambat (**10.569 ms, RTF 1.727x**) dan mengalami pemotongan kata akibat saturasi DRAM.

---

### 5. Option E: whisper.cpp GGML WASM (Mesin CPU Tercepat)
* **Model**: Whisper Tiny (39 MB) / Whisper Base (140 MB).
* **Format Bobot**: GGML Q8_0 (`ggml-base-id-q8_0.bin`).
* **Akselerator**: C++ whisper.cpp terkompilasi ke WebAssembly SIMD.
* **Kelebihan**:
  * **Mesin CPU Tercepat untuk Tiny**: Waktu decode **4.549 ms (RTF 0.743x)**.
  * Arsitektur alokasi memori datar (*flat arena*) yang sangat minim alokasi heap dinamis JS.
* **The Catch**:
  * Whisper Base pada CPU mengalami kelambatan ekstrem (**12.664 ms, RTF 2.070x**) dan *acoustic collapse*. Hanya direkomendasikan untuk varian model Tiny.

---

## Status Eksperimental: Option C (WebNN DirectML)

> [!WARNING]
> **Option C (WebNN DirectML) berstatus EXPERIMENTAL.**  
> API WebNN belum diaktifkan secara default pada rilis stabil Google Chrome / Microsoft Edge.

### Cara Mengaktifkan WebNN di Peramban:
1. Buka peramban Google Chrome atau Microsoft Edge.
2. Akses tautan internal: `chrome://flags/#web-machine-learning-neural-network` (atau `edge://flags/#web-machine-learning-neural-network`).
3. Ubah pengaturan dari **Default** menjadi **Enabled**.
4. Muat ulang (*relaunch*) peramban Anda.
5. Pada modal pengaturan Catetin (`⚙️`), pilih backend **WebNN (DirectML Native)**.

Jika flag tidak aktif, runtime Catetin akan memberikan notifikasi peringatan pada konsol dan otomatis menyarankan *fallback* ke **Option A (WebGPU)** atau **Option B (WASM SIMD)**.

---

## Tabel Evaluasi & Ranking Empiris Terpisah

Pengujian empiris dilakukan pada berkas audio nyata [`dist/sample-indonesia-16k.wav`](file:///C:/Users/abdka/Development/foresight-analysis/dist/sample-indonesia-16k.wav) (6.12 detik, 97.903 sampel, 16kHz mono) secara terpisah:

### 1. Evaluasi Pass 1 (Streaming Live ASR)
| Tahap (Pass) | Opsi Engine | Model | Backend Hardware | Dtype | Latensi Chunk | TTFT | RTF | Hasil Transkrip | Catatan Kritis |
| :--- | :--- | :--- | :--- | :--- | :---: | :---: | :---: | :--- | :--- |
| **Pass 1** | **Transducer** | **Zipformer2 ID (70.9 MB)** | CPU WASM SIMD (1 Core) | INT8 | **~31 ms** | **357.9 ms** | **0.133x** | *"DARI REPUBLIK INDONESIA JAKARTA"* | **Wajib Selalu Aktif**. Nol VRAM, respons live instan sub-detik. Teks kapital mentah tanpa tanda baca. |

---

### 2. Ranking Mandiri Pass 2 (Turn Finalizer ASR — 10 Kombinasi Terpisah)
| Peringkat | PASS | OPTION | MODEL | HARDWARE | DTYPE | INIT (ms) | DECODE (ms) | RTF | HASIL TRANSKRIPSI | CATATAN KRITIS |
| :---: | :---: | :--- | :--- | :--- | :--- | :---: | :---: | :---: | :--- | :--- |
| 🥇 **#1** | **Pass 2** | **Option A** | **Whisper Tiny (39 MB)** | WebGPU Shader | Split FP32/Q4 | 3.995 | **1.982** | **0.324x** | *"Republik Indonesia Jakarta Surabaya"* | **Juara Umum Pass 2**. WGSL Mel Extractor memangkas decode ke sub-2 detik (~3.1x lebih cepat dari durasi audio). VRAM 85 MB. Universal tanpa flag. |
| 🥈 **#2** | **Pass 2** | **Option C** | **Whisper Base (140 MB)** | WebNN DirectML | DirectML Graph | 11.360 | **2.978** | **0.487x** | *"Republik Indonesia Jakarta Surabaya"* | **Model Base Tercepat**. DirectML bypass Mojo IPC Chromium. Butuh flag `#web-machine-learning-neural-network`. |
| 🥉 **#3** | **Pass 2** | **Option C** | **Whisper Tiny (39 MB)** | WebNN DirectML | DirectML Graph | 8.027 | **3.657** | **0.598x** | *"Republik Indonesia Jakarta Surabaya"* | **Hemat Daya Laptop**. Sangat efisien pada NPU/iGPU Windows DirectML. |
| **#4** | **Pass 2** | **Option A** | **Whisper Base (140 MB)** | WebGPU Shader | Split FP32/Q4 | 6.715 | **3.912** | **0.639x** | *"Republik Indonesia Jakarta Surabaya"* | **Akurat di WebGPU Universal**. Model Base 140 MB sukses dieksekusi di bawah real-time (RTF 0.639x). VRAM ~195 MB. |
| **#5** | **Pass 2** | **Option E** | **Whisper Tiny (39 MB)** | whisper.cpp WASM | Split FP32/Q8 | 6.814 | **4.549** | **0.743x** | *"Republik Indonesia Jakarta, Surabaya. Pancasila."* | **Mesin CPU Tercepat**. Flat memory whisper.cpp menghasilkan transkrip lengkap hingga kata penutup dengan RTF sub-1.0x di CPU. |
| **#6** | **Pass 2** | **Option B** | **Whisper Tiny (39 MB)** | WASM SIMD (8 Core) | Split FP32/Q8 | 6.035 | **5.173** | **0.845x** | *"Republik Indonesia Jakarta, Surabaya. Pancasila."* | **Transkripsi Sempurna di WASM**. Teks 100% utuh dengan tanda baca lengkap. Murni komputasi CPU tanpa ketergantungan API grafis. |
| **#7** | **Pass 2** | **Option D** | **Whisper Tiny (39 MB)** | Direct ORT Web | Split FP32/Q8 | 6.480 | **5.290** | **0.865x** | *"Republik Indonesia Jakarta, Surabaya. Pancasila."* | **Eksekusi Mandiri Tanpa CDN**. Bobot lokal INT8 deterministik di CPU. Teks lengkap. |
| **#8** | **Pass 2** | **Option B** | **Whisper Base (140 MB)** | WASM SIMD (8 Core) | Split FP32/Q8 | 13.447 | **8.367** | **1.367x** | *"Pancasila"* | **Acoustic Collapse di CPU**. Bobot 140 MB memicu L3 cache thrashing (16–32 MB cache terlampaui), menyebabkan truncation kata tunggal. |
| **#9** | **Pass 2** | **Option D** | **Whisper Base (140 MB)** | Direct ORT Web | Split FP32/Q8 | 12.630 | **10.569** | **1.727x** | *"Pancasila"* | **Terlalu Lambat di CPU**. Waktu decode 1,7x lebih lambat dari audio asli. Terpotong akibat saturasi DRAM CPU. |
| **#10** | **Pass 2** | **Option E** | **Whisper Base (140 MB)** | whisper.cpp WASM | Split FP32/Q8 | 15.521 | **12.664** | **2.070x** | *"Pancasila"* | **Peringkat Terendah / Anti-Pattern**. Waktu decode 12,6 detik untuk audio 6,1 detik (RTF 2.07x). Dilarang keras pada thread CPU peramban. |

---

## Penyelesaian Bottleneck Mel-Spectrogram (WGSL)

Sebelumnya, ekstraksi Mel-Spectrogram di JavaScript CPU menghabiskan **~1.121 ms (21% dari total waktu inferensi)** melalui perkalian matriks filterbank 80-channel sekuensial.

Dengan implementasi [`web/src/engines/mel-spectrogram.js`](file:///C:/Users/abdka/Development/foresight-analysis/web/src/engines/mel-spectrogram.js):
1. **Transfer Efisien**: Mengirim PCM 16kHz mentah langsung ke VRAM buffer (195 KB untuk audio 6 detik, efisiensi transfer bus 75%).
2. **Komputasi Paralel WGSL**: Eksekusi STFT Hann-window dan filterbank Mel 80-channel di workgroup shared memory GPU (`var<workgroup>`).
3. **Active-Frame Pruning**: Hanya menghitung 612 frame aktif riil dan menginstapad hingga 3.000 frame langsung di VRAM.
4. **Hasil**: Latensi Mel turun dari **1.121 ms** ke **7 – 11 ms** (**46,6x lebih cepat**), memangkas total latensi Pass 2 Option A Tiny ke **1.982 ms**.

---

## Panduan Menjalankan Proyek

### 1. Menjalankan Server Web PWA Lokal
Tidak memerlukan runtime berat. Cukup gunakan Node.js bawaan:

```bash
# Menjalankan HTTP daemon lokal dengan COOP/COEP headers aktif (dibutuhkan untuk SharedArrayBuffer & WebGPU)
node scripts/serve.js
```
Akses di peramban: `http://127.0.0.1:8085`

### 2. Menguji Benchmark Empiris Utuh
Buka halaman benchmark di peramban yang sama untuk menguji 11 kombinasi engine dan model secara langsung:
```
http://127.0.0.1:8085/benchmark.html
```
Klik tombol **"Jalankan Seluruh Benchmark"** untuk mengukur latensi dan akurasi pada audio uji `sample-indonesia-16k.wav`.

### 3. Kompilasi Modul Rust WebAssembly (`catetin-wasm`)
```bash
cd crates/catetin-wasm
wasm-pack build --target web --release
```

---

## Struktur Proyek

```
catetin/
├── crates/
│   ├── catetin-core/            # Engine headless Rust (WASAPI, AEC NLMS, VAD)
│   ├── catetin-cli/             # Antarmuka TUI terminal (ratatui)
│   └── catetin-wasm/            # Binding WebAssembly C-ABI untuk Pass 1 Zipformer2
│
├── web/                         # Frontend Web PWA Driverless
│   ├── index.html               # UI utama dengan bootstrap ready badge
│   ├── benchmark.html           # Suite pengujian empiris Pass 1 & Pass 2
│   ├── style.css                # Desain responsif modern
│   ├── src/
│   │   ├── app.js               # State machine bootstrap & integrasi UI
│   │   ├── audio-engine.js      # Staging buffer & pipeline audio 16kHz
│   │   ├── engines/
│   │   │   ├── hub.js           # Koordinator modular Pass 2 Hub
│   │   │   ├── pass2-interface.js # Kontrak seragam init, warmup, transcribe, dispose
│   │   │   ├── pass2-webgpu.js   # Option A: WebGPU Whisper Engine
│   │   │   ├── pass2-wasm-simd.js # Option B: WASM SIMD Whisper Engine
│   │   │   ├── pass2-webnn.js    # Option C: WebNN DirectML Engine [Eksperimental]
│   │   │   ├── pass2-direct-ort.js # Option D: Direct ONNX Runtime Web Engine
│   │   │   ├── pass2-wasm-ggml.js # Option E: whisper.cpp GGML WASM Engine
│   │   │   └── mel-spectrogram.js # WGSL GPU Compute Shader Mel Extractor
│   │   └── components/
│   │       ├── transcript-view.js # Virtual list tampilan transkripsi dual-pass
│   │       └── catetin-telemetry.js # Monitor real-time RTF, latensi, dan VRAM
│
├── docs/                        # Dokumentasi arsitektur dan spesifikasi teknis
│   ├── BENCHMARK_AND_MODEL_EVALUATION.md # Laporan komprehensif benchmark empiris
│   ├── CATETIN_WEB_SYSTEM_ARCHITECTURE.md # Arsitektur sistem web PWA menyeluruh
│   └── ASR_PIPELINE_LOWLEVEL_BOTTLENECK_ANALYSIS.md # Analisis bottleneck low-level & 8 pilar
│
├── dist/                        # Berkas build produksi & audio uji
│   └── sample-indonesia-16k.wav # Audio uji validasi fonemik 6.12 detik
│
├── scripts/
│   └── serve.js                 # Server lokal dengan dukungan COOP & COEP
├── DECISIONS.md                 # Rekaman keputusan arsitektur (ADR #1 s/d #15)
├── CHANGELOG.md                 # Riwayat perubahan proyek
└── README.md                    # Dokumentasi utama proyek ini
```

---

## Indeks Dokumentasi Teknis

* **[BENCHMARK_AND_MODEL_EVALUATION.md](file:///C:/Users/abdka/Development/foresight-analysis/docs/BENCHMARK_AND_MODEL_EVALUATION.md)**: Analisis mendalam hasil benchmark empiris seluruh engine dan model, rincian profiling bottleneck, dan panduan mitigasi.
* **[CATETIN_WEB_SYSTEM_ARCHITECTURE.md](file:///C:/Users/abdka/Development/foresight-analysis/docs/CATETIN_WEB_SYSTEM_ARCHITECTURE.md)**: Spesifikasi arsitektur Web PWA Catetin, implementasi state machine eager pre-warming, dan integrasi compute shader.
* **[ASR_PIPELINE_LOWLEVEL_BOTTLENECK_ANALYSIS.md](file:///C:/Users/abdka/Development/foresight-analysis/docs/ASR_PIPELINE_LOWLEVEL_BOTTLENECK_ANALYSIS.md)**: Bedah mendalam arsitektur low-level, 8 pilar preferensi sistem, dan panduan verifikasi whitebox tanpa mock data.
* **[DECISIONS.md](file:///C:/Users/abdka/Development/foresight-analysis/DECISIONS.md)**: Catatan keputusan arsitektur (Architecture Decision Records) dari ADR #1 hingga ADR #15.
