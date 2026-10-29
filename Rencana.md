# Rencana Riset & Eksperimen Masa Depan

Dokumen ini memuat rencana riset lanjutan, eksplorasi optimasi tingkat rendah, dan spesifikasi eksperimen engine masa depan yang saat ini masih berstatus *on hold* atau belum diimplementasikan.

> [!NOTE]
> Seluruh implementasi dasar pipeline CatatIn telah selesai dan terverifikasi:
> - **Pass 1:** Streaming Transducer Sherpa-ONNX Zipformer2 ID (latensi < 50ms).
> - **Pass 2 Hub:** Modular Adapter Pattern dengan kontrak baku `Pass2Engine`.
> - **Pass 2 Direct ORT INT8:** Runtime C-ABI ONNX Runtime Web mandiri dengan KV-cache autoregressive decode loop dan ekstraksi Mel CPU (`pass2-direct-ort.js`).
> - **Pass 2 whisper.cpp WASM GGML:** Adapter C-ABI Emscripten (`pass2-wasm-ggml.js`), model lokal Q8_0 (`ggml-base-id-q8_0.bin` & `ggml-small-id-q8_0.bin`), MIME `.bin` (`serve.js`), serta skrip kompilasi (`scripts/build-whisper-wasm.ps1`) selesai dan terverifikasi 100% whitebox.
> - **Ekstraksi Mel WGSL:** Compute shader WGSL mandiri (7–11 ms) menggantikan CPU (1.121 ms).
> - **Onboarding Hardware Readiness Test (Fase 5.1):** Pengujian micro-benchmark otomatis dan rekomendasi engine.
> - **Self-Host Transformers.js v3.3.3 (Fase 5.3):** Penyimpanan lokal penuh di `web/vendor/transformers/` bebas CDN.
> - **Cache Storage Transducer (Fase 5.4):** Cache 70 MB model Zipformer2 via `CacheStorage`.
>
> Riwayat lengkap seluruh pekerjaan yang telah selesai dapat dilihat di [CHANGELOG.md](./CHANGELOG.md).

---

## 1. Rencana Implementasi Nyata Engine Masa Depan

Dua opsi engine saat ini masih berstatus `(belum tersedia)` pada `Pass2Hub` (`pass2-wasm-ggml.js`, `pass2-webnn.js`) dan diimplementasikan secara terisolasi tanpa mengubah pipeline audio/UI:

| Dimensi | `pass2-wasm-ggml.js` | `pass2-webnn.js` |
|---|---|---|
| **Arsitektur** | whisper.cpp C++ via WASM SIMD multi-thread | W3C WebNN API graf komputasi native OS |
| **Backend Target** | CPU Cache L1/L2/L3 (pthreads + SIMD 128-bit) | DirectML (Windows) / NPU / dGPU |
| **Sumber Model** | `dist/models/ggml-base-id-q8_0.bin` (81,77 MB) & `ggml-small-id-q8_0.bin` (264,46 MB) | Model ONNX teroptimasi DirectML |
| **Kelebihan Utama** | 100% identik dengan modul CLI desktop; bebas bug driver GPU | Efisiensi daya tinggi; bebas Mojo IPC loop |
| **Status Saat Ini** | Adapter C-ABI JS, caching, MIME `.bin`, & skrip build selesai terverifikasi (pending eksekusi `build-whisper-wasm.ps1` via emsdk) | Backlog masa depan (menunggu kematangan W3C WebNN) |

### A. whisper.cpp WASM SIMD (`pass2-wasm-ggml.js`) — [ADAPTER SELESAI / PENDING BUILD EMSDK]
- **Tujuan**: Memberikan jalur eksekusi CPU multi-threaded yang konsisten 1:1 dengan aplikasi desktop CatatIn CLI.
- **Status Selesai**:
  1. Seluruh kode adapter JavaScript (`pass2-wasm-ggml.js`), alokasi C-ABI heap WASM (`_whisper_init_from_buffer`, `_whisper_full`), dan pembersihan memori deterministik telah selesai.
  2. Caching biner model via Cache Storage API (`catetin-ggml-models-v1`) dan endpoint `.bin` pada static web server (`scripts/serve.js`) telah siap dan terverifikasi.
  3. Bobot model kuantisasi lokal `dist/models/ggml-base-id-q8_0.bin` (81,77 MB) dan `dist/models/ggml-small-id-q8_0.bin` (264,46 MB) telah tersedia di disk.
  4. Skrip otomatisasi build Emscripten `scripts/build-whisper-wasm.ps1` telah siap; satu-satunya langkah manual yang tersisa hanyalah menjalankan `.\scripts\build-whisper-wasm.ps1` setelah `emsdk` dipasang di host.

### B. Native WebNN Graph (`pass2-webnn.js`) — [BACKLOG MASA DEPAN]
- **Tujuan**: Menjalankan model ASR langsung melalui akselerasi graf native OS (DirectML di Windows) untuk pemanfaatan NPU/GPU dengan konsumsi daya baterai minimal.
- **Tantangan & Pengawasan**:
  - API `navigator.ml` masih memerlukan flag eksplisit (`chrome://flags/#web-machine-learning-neural-network`).
  - Implementasi disimpan di backlog masa depan sambil menunggu kematangan dan standarisasi stabil W3C WebNN di peramban Chromium arus utama.

---

## 2. Area 1: Optimasi Loop Decoder Autoregresif GPU (On Hold)

### Permasalahan Fisik: Batas Latensi Sinkronisasi CPU ↔ GPU
Pada model autoregresif Whisper (batch size $B = 1$), setiap pembentukan 1 token baru menuntut siklus bolak-balik:
1. Browser JavaScript mendispatch kernel forward pass decoder ke GPU via Mojo IPC.
2. GPU menghitung logits token berikutnya.
3. Buffer logits ditransfer balik ke CPU melalui pemetaan asynchronous (`mapAsync`).
4. Main thread JS mengeksekusi operasi `argmax` / greedy search.
5. Token baru diinputkan kembali ke antrean GPU.

Transit driver DirectX 12/Dawn dan sinkronisasi buffer memicu *GPU idle bubble* sebesar **4–8 ms per token**. Pada kalimat 15 token, **60–120 ms** hilang murni untuk transit IPC.

### Opsi Eksperimen yang Ditahan (*On Hold*):

#### Opsi 1: Custom WGSL Compute Shader untuk Argmax
- Menulis compute shader WGSL untuk menjalankan operasi reduksi `argmax` langsung di GPU VRAM.
- Menghilangkan transfer buffer logits (~50.000 float per step) ke CPU; hanya indeks token skalar (4 byte) yang dikembalikan ke JS.

#### Opsi 2: Custom Loop Decoding di Level ORT / C-ABI
- Memindahkan seluruh decode loop ke dalam satu modul WebAssembly/C++ terkompilasi, sehingga seluruh sequence token dieksekusi dalam satu kali pemicuan tanpa interupsi per-token dari JavaScript.

#### Opsi 3: Speculative Decoding
- Menggunakan model kecil (Whisper Tiny) untuk memprediksi beberapa kandidat token secara paralel, kemudian memverifikasinya sekaligus menggunakan forward pass tunggal pada model Base.

> **Keputusan**: Seluruh opsi Area 1 saat ini berstatus **ON HOLD** (`pass2-direct-ort` telah selesai diimplementasikan dan diukur dengan RTF 1.135x, menunggu pembanding baseline dari `pass2-wasm-ggml`).

---

## 3. Daftar Pengawasan Dependensi (Watch List)

| Dependensi | Penggunaan | Status Risiko | Alasan Pemantauan |
|---|---|:---:|---|
| **WebNN (`navigator.ml`)** | `pass2-webnn.js` | 🔴 Tinggi | Bergantung pada flag browser; spesifikasi W3C masih dinamis; backend NPU belum stabil di seluruh OEM. |
| **Transformers.js v3** | `pass2-webgpu.js` | 🟡 Terkelola | Telah di-self-host pada v3.3.3 lokal; pantau rilis v4 untuk perbaikan WebGPU compute shader resmi. |
| **ONNX Runtime Web** | Eksekusi JSEP/WASM | 🟡 Terkelola | Dikunci sesuai versi dependensi Transformers.js lokal di `web/vendor/transformers/`. |
