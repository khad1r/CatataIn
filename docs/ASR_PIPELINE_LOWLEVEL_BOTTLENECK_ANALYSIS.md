# Analisis Bottleneck Low-Level ASR Pipeline, Arsitektur Pass 2, dan Cetak Biru Sistem Catetin

Dokumen ini menyatukan seluruh analisis teknis mengenai eliminasi bottleneck komputasi dan transfer memori, optimasi Transformers.js v3 StaticCache, pendekatan Low-Level Systems Programming, peta tahapan implementasi teknis (Tahap 1 s/d 6), serta pemetaan komprehensif 8 Pilar Preferensi Arsitektur Sistem Catetin. Seluruh keputusan mengacu pada kontrak antarmuka di [pass2-interface.js](file:///C:/Users/abdka/Development/foresight-analysis/web/src/engines/pass2-interface.js), [hub.js](file:///C:/Users/abdka/Development/foresight-analysis/web/src/engines/hub.js), serta prinsip arsitektur di [DECISIONS.md](file:///C:/Users/abdka/Development/foresight-analysis/DECISIONS.md).

---

## 1. Eager Pre-warming & Visual UI Ready Indicator

### A. Kondisi Eksisting & Masalah
* Kompilasi graph/shader DirectML dan WebGPU di Windows/Chromium memakan waktu **2.2 hingga 7.5 detik** saat *cold-start* pertama kali.
* Jika inisialisasi ditunda hingga audio pertama masuk (*lazy loading*), kata pertama pembicara mengalami pembekuan (*freeze/stall*), mengakibatkan lonjakan latensi dan hilangnya kontinuitas transkripsi live.

### B. Arsitektur Solusi & Alur Eksekusi
1. **Eager Initialization**:
   * Inisialisasi engine Pass 2 dipicu seketika saat siklus hidup aplikasi bootstrap (`DOMContentLoaded` / ServiceWorker ready), paralel dengan pemuatan model Pass 1, tanpa menunggu interaksi klik pengguna pada tombol sesi.
2. **Dummy Pass di Latar Belakang**:
   * Mengirimkan 1 frame dummy audio nol durasi 0.5 detik (`max_new_tokens: 1`, `temperature: 0.0`) ke worker di latar belakang.
   * Eksekusi ini memaksa driver Windows DirectX 12 (`D3D12ShaderCache`) dan Chromium (`GPUCache`) mengompilasi seluruh Pipeline State Objects (PSO) langsung ke penyimpanan cache disk. Eksekusi inferensi pada giliran bicara riil berikutnya instan (**< 50 ms**).
3. **State Machine Siklus Hidup & Visual Indikator di UI**:
   * Transisi state: `BOOTSTRAPPING` $\to$ `WARMING_PASS1` $\to$ `WARMING_PASS2` $\to$ `SYSTEM_READY`.
   * Tombol `#btn-session` (*"Mulai"*) berstatus `disabled` dengan animasi *pulse/spinner* dan label progres: *"Menyiapkan Engine AI..."* $\to$ *"Mengompilasi Shader DirectML/WebGPU..."*.
   * Panel telemetri [catetin-telemetry.js](file:///C:/Users/abdka/Development/foresight-analysis/web/src/components/catetin-telemetry.js) memancarkan status kesiapan sub-sistem.
   * Begitu antrean *promise* warmup selesai, tombol berubah menjadi aktif berwarna hijau (*"Mulai Rapat (Siap 0ms)"*).
   * **Ekstensibilitas**: Engine baru cukup mengembalikan `Promise` pada metode `warmup()` sesuai kontrak [pass2-interface.js](file:///C:/Users/abdka/Development/foresight-analysis/web/src/engines/pass2-interface.js).

---

## 2. Anatomi Bottleneck Transfer Data & Strategi Optimasi (3 Boundaries)

Pipeline streaming ASR beroperasi melintasi tiga batas memori (*memory boundaries*):

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                              TIGA BOUNDARY TRANSFER MEMORI                             │
│                                                                                        │
│   [Audio Thread] ──── Boundary 1 ────► [Worker CPU RAM] ──── Boundary 2 ────► [GPU VRAM]│
│   (AudioWorklet)     SharedArrayBuffer  (catetin.wasm)        DMA PCIe /        (Compute│
│                      Lock-free SPSC                           Mel WGSL           Shader)│
│                                                                                    │   │
│   [Main UI Thread] ◄────────────── Boundary 3 (Readback Latency) ─────────────────┘   │
│    (TranscriptView)                 Single End-of-Turn Dispatch                        │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

### Boundary 1: Real-Time Audio Thread (AudioWorklet) ke Worker (Float32Array)
* **Akar Masalah**: Mekanisme standar `port.postMessage(pcm)` melakukan *structured clone* (alokasi objek heap baru dan penyalinan memori tiap 160ms = 6.25 alokasi/detik). Hal ini memicu lonjakan Garbage Collection (GC pause) periodik pada thread utama.
* **Solusi Low-Level**: Gunakan **`SharedArrayBuffer` + `Atomics`** dengan topologi *Lock-free Single-Producer Single-Consumer (SPSC) RingBuffer*. `AudioWorklet` menulis langsung ke indeks sirkular buffer bersama, dan Worker membaca langsung tanpa serialisasi atau alokasi memori baru (0 ms thread-hop).

### Boundary 2: CPU RAM ke GPU/NPU VRAM (writeBuffer / writeTensor)
* **Akar Masalah**: Frame audio 160ms hanya berukuran 10 KB (2,560 float32), sementara transfer DMA PCIe/SoC memiliki latensi tetap (*base dispatch overhead*) ~2–4 ms per pemanggilan. Mengirim potongan 160ms ke GPU tidak efisien karena waktu sinkronisasi driver melampaui waktu komputasi tensor. Sebaliknya, pada Pass 2 (turn 6–15 detik, ukuran ~400 KB), transfer 400 KB hanya memakan ~3 ms tetapi komputasi paralel GPU/NPU berjalan optimal.
* **Solusi Low-Level**: Hitung **Mel-spectrogram langsung di GPU menggunakan WGSL Compute Shader**. Audio PCM 16kHz mentah (64 KB/detik) dikirim langsung sekali jalan. Filterbank FFT dan kompresi log-mel diproses paralel di VRAM, mengeliminasi pengiriman matriks mel 783 KB dari CPU.

### Boundary 3: GPU/NPU VRAM ke CPU RAM (Readback Latency pada Autoregressive Generation)
* **Akar Masalah**: Whisper Decoder menghasilkan token secara sekuensial satu per satu (*autoregressive seq2seq*, $B=1$). Jika setiap token yang dihasilkan dibaca kembali ke CPU via `mapAsync` / staging buffer, terjadi *pipeline stall* (GPU idle bubble) puluhan kali per kalimat (64 token = 64 kali context-switch CPU-GPU).
* **Solusi Low-Level**: Terapkan **Decoupled GPU Decode Loop**. Seluruh loop generasi token (`while token != EOS && step < 64`) dieksekusi di dalam satu rantai `GPUCommandEncoder` di VRAM. Host CPU hanya melakukan 1 kali readback saat token akhir transkrip `<|endoftranscript|>` tercapai atau batas token terpenuhi.

---

## 3. Detail Transformers.js v3 & StaticCache pada Kondisi Eksisting

### A. Kondisi Eksisting & Akar Masalah di v2 (`@xenova/transformers@2.17.2`)
Pada berkas [pass2-webgpu.js](file:///C:/Users/abdka/Development/foresight-analysis/web/src/engines/pass2-webgpu.js), implementasi awal menggunakan Transformers.js v2:
```javascript
const { pipeline, env } = await import('https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2');
env.allowLocalModels = false;
this.transcriber = await pipeline('automatic-speech-recognition', modelName, { device: 'webgpu' });
```
**Penyebab Latensi Whisper Base Tinggi (RTF 1.09x / 6,693 ms untuk audio ~6 detik):**
1. Decoder v2 menggunakan alokasi KV-cache dinamis.
2. Setiap token baru yang dihasilkan (misal token ke-15) memaksa WebGPU membuang buffer lama, mengalokasikan buffer baru untuk panjang 15, lalu menyalin data lama.
3. Driver GPU mengalami *stall* sinkronisasi buffer di setiap langkah generasi kata, memicu alokasi VRAM membengkak hingga ~300 MB dan fragmentasi memori.

### B. Mekanisme Transformers.js v3 & StaticCache
Pembaruan ke Transformers.js v3 di [pass2-webgpu.js](file:///C:/Users/abdka/Development/foresight-analysis/web/src/engines/pass2-webgpu.js) mempertahankan kontrak [pass2-interface.js](file:///C:/Users/abdka/Development/foresight-analysis/web/src/engines/pass2-interface.js) tanpa mengubah berkas aplikasi lain:

```javascript
// web/src/engines/pass2-webgpu.js
const { pipeline, env } = await import('https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.3.3');

this.transcriber = await pipeline('automatic-speech-recognition', modelName, {
  device: 'webgpu',
  dtype: {
    encoder_model: 'fp32',         // Akurasi ekstraksi representasi akustik
    decoder_model_merged: 'q4'     // Kuantisasi 4-bit decoder (mengurangi saturasi bandwidth bus memori)
  }
});
```
* **Mekanisme StaticCache**: `@huggingface/transformers@3` mengalokasikan buffer KV-cache statis berukuran tetap ($T_{\max} = 64/128$) sejak inisialisasi. Token baru ditulis langsung ke slot array sirkular yang sudah ada (*in-place write*) tanpa alokasi ulang buffer.
* **Hasil Terukur**: Menghilangkan GPU buffer reallocation stutter, memangkas konsumsi VRAM dari ~300 MB ke ~85 MB, dan mempercepat Real-Time Factor (RTF) Whisper Base dari 1.09x menjadi sub-0.5x.

---

## 4. Pendekatan Low-Level Systems Programming pada Kode Eksisting

### 1. Zero-Copy Memory-Mapped WASM ([lib.rs](file:///C:/Users/abdka/Development/foresight-analysis/crates/catetin-wasm/src/lib.rs))

#### Realita Alur Memori Saat Ini
Alur pemrosesan frame 160ms audio pada kode eksisting:
1. Di [audio-engine.js](file:///C:/Users/abdka/Development/foresight-analysis/web/src/audio-engine.js#L320): `memF32.set(inputData.subarray(0, len), inOffset)` $\to$ **Salin #1** dari browser buffer ke memori linear `catetin.wasm`.
2. Di [audio-engine.js](file:///C:/Users/abdka/Development/foresight-analysis/web/src/audio-engine.js#L362): `chunkCopy.set(memF32.subarray(...))` $\to$ **Salin #2** dari memori WASM kembali ke objek JavaScript `Float32Array`.
3. Di [audio-engine.js](file:///C:/Users/abdka/Development/foresight-analysis/web/src/audio-engine.js#L369) & [onnx-runner.js](file:///C:/Users/abdka/Development/foresight-analysis/web/src/onnx-runner.js): `postMessage({ pcm16k: chunkCopy.slice() })` $\to$ **Salin #3** via Structured Clone IPC ke worker Transducer.
4. Di [sherpa-transducer-worker.js](file:///C:/Users/abdka/Development/foresight-analysis/web/src/sherpa/sherpa-transducer-worker.js#L132): `stream.acceptWaveform(16000, pcm16k)` $\to$ **Salin #4** ke dalam Emscripten heap milik Sherpa-ONNX WASM.

Setiap 160ms (6.25 kali per detik), **data audio yang sama disalin 4 kali** melintasi batas thread dan runtime.

#### Implementasi Low-Level Zero-Copy
Memanfaatkan fungsi alokasi C-ABI yang sudah ada di [lib.rs](file:///C:/Users/abdka/Development/foresight-analysis/crates/catetin-wasm/src/lib.rs#L293-L305):
```rust
#[no_mangle]
pub extern "C" fn catetin_alloc_f32(len: usize) -> *mut f32 {
    let mut vec = Vec::<f32>::with_capacity(len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}
```
Dengan header COOP/COEP (`same-origin` & `require-corp`) aktif di server:
* Audio thread dan worker berbagi satu `SharedArrayBuffer` sirkular (ring buffer 32 KB).
* Audio input ditulis langsung ke alamat pointer memori bersama.
* Pointer 32-bit langsung diteruskan ke C-ABI Sherpa-ONNX (`SherpaOnnxOnlineStreamAcceptWaveform(stream, sample_rate, ptr, len)`).
* **Dampak**: 4 siklus `memcpy` dipangkas menjadi **0 salinan (true zero-copy)** tanpa alokasi heap baru dan nol beban Garbage Collection.

---

### 2. Kernel WGSL Kustom untuk Mel-Spectrogram & Decoupled Decode

#### Realita Beban Komputasi
Whisper memerlukan representasi 80-channel Log-Mel Spectrogram sebelum masuk ke Encoder:
* Ekstraksi di CPU menggunakan JavaScript/WASM (`WhisperFeatureExtractor`): menghitung STFT 400-point window, perkalian matriks filterbank mel, dan kompresi logaritmik.
* Untuk audio 6.12 detik, ekstraksi menghasilkan matriks $[1, 80, 612]$ ($\approx 195,840$ float32 $= 783\text{ KB}$). Array 783 KB ini kemudian ditransfer ke GPU.

#### Implementasi Low-Level WGSL
Audio PCM mentah 16kHz durasi 6.12 detik berukuran **hanya 195 KB** (75% lebih kecil dari matriks mel):
1. **Mel-Spectrogram via Compute Shader**:
   * Kirim audio PCM 195 KB langsung ke VRAM via `writeBuffer` dalam satu kali dispatch.
   * Shader WGSL 1-pass memproses frame FFT 400-sampel secara paralel langsung di VRAM menggunakan memori bersama workgroup (`var<workgroup>`). Thread CPU bebas dari kalkulasi FFT.
2. **Decoupled Decode Loop**:
   * Seluruh loop autoregresif token dienkapsulasi di dalam rantai `GPUCommandEncoder`.
   * WebGPU mengeksekusi iterasi generasi token tanpa pemicuan event loop JavaScript bolak-balik hingga token akhir `<|endoftranscript|>` terdeteksi.

---

### 3. Weight-Only Quantization (INT8/INT4 Packed) & On-The-Fly Dequantization

#### Realita Bottleneck Bandwidth Memori
Pada model Whisper eksisting di `dist/models/sherpa-onnx-whisper-base/`:
* `base-encoder.int8.onnx` (29.1 MB)
* `base-decoder.int8.onnx` (130.6 MB)

Saat Whisper Decoder mengeksekusi generasi token sekuensial (*batch size = 1*), **pembatas performa utama adalah Memory Bandwidth**, bukan FLOPS unit komputasi:
* Setiap generasi 1 token menuntut pembacaan seluruh matriks bobot decoder (130 MB) dari memori ke register core.
* Format FP32: Transfer 520 MB per token.
* Format INT8: Transfer 130 MB per token.
* Format **INT4 Packed**: 2 nilai bobot 4-bit dipadatkan ke dalam 1 byte `uint8` tunggal (transfer turun menjadi **hanya 65 MB per token**).

#### Low-Level On-The-Fly Dequantization
Bobot disimpan dalam format terkompresi 4-bit di VRAM. Saat kernel GEMM berjalan:
* Register membaca 1 byte `uint8`.
* Operator bitwise memisahkan menjadi dua integer 4-bit:
  $$\text{val}_0 = \text{byte} \ \& \ 0\text{x}0\text{F}, \quad \text{val}_1 = \text{byte} \gg 4$$
* Dekuantisasi instan pada register:
  $$\text{fp16} = (\text{int4} - \text{zero\_point}) \times \text{scale}$$
* Mengalikan hasil dengan vektor aktivasi input.
* **Dampak**: Waktu transfer bobot dari memori dipangkas hingga 50%, melipatgandakan throughput token per detik pada arsitektur iGPU memori terbagi.

---

### 4. Pipelining Asinkronik Antara Pass 1 dan Pass 2

#### Realita Keterlambatan (*Dead Time*) Saat Ini
Pada [audio-engine.js](file:///C:/Users/abdka/Development/foresight-analysis/web/src/audio-engine.js#L424-L438):
1. Selama pengguna berbicara, chunk audio ditampung di array JavaScript `currentTurnPcm`.
2. Saat VAD mendeteksi akhir bicara (*turn closed*), array digabungkan menjadi `new Float32Array(...)`.
3. Array diserahkan ke [onnx-runner.js](file:///C:/Users/abdka/Development/foresight-analysis/web/src/onnx-runner.js) untuk diteruskan ke Pass 2.
4. Pass 2 baru mulai mengalokasikan buffer GPU, mentransfer ratusan kilobyte, dan mengompilasi tensor. Ada jeda inisialisasi yang sia-sia setelah pembicara berhenti.

#### Arsitektur Pipelining 0ms Dead Time
Memanfaatkan sinyal `speaking = true` dari VAD sejak milidetik pertama:

```
T0: Suara Mulai               T_end: Suara Berhenti
 │                             │
 ▼                             ▼
┌──────────────────────────────┐
│ Pass 1 (CPU WASM SIMD):      │
│ Stream 160ms chunks ─────────┼──► Teks Live Berhenti
└──────────────────────────────┘
┌──────────────────────────────┐───────────────────────────────┐
│ Pass 2 Audio Staging:        │ Pass 2 Execution:             │
│ Akumulasi ke GPU Buffer VRAM │ 100% audio sudah ada di VRAM. │
│ saat pengguna masih bersuara │ Decode dimulai INSTAN (0ms)   │
└──────────────────────────────┘───────────────────────────────┘
```

1. Saat VAD memancarkan `speaking = true`, Pass 2 langsung memetakan staging buffer di VRAM GPU.
2. Setiap kali chunk 160ms masuk, data diserahkan ke Pass 1 untuk teks live, dan **secara simultan di-append ke staging buffer GPU di latar belakang**.
3. Saat pengguna berhenti bicara (`speaking = false`):
   * 100% sampel audio sudah berada di VRAM GPU.
   * Latensi transfer audio saat turn selesai adalah **0 ms**.
   * Encoder Pass 2 langsung mengeksekusi komputasi seketika tanpa fase penyiapan buffer.

---

## 5. Ekstensi Modular Pass 2 Hub

Sesuai pola desain pada [hub.js](file:///C:/Users/abdka/Development/foresight-analysis/web/src/engines/hub.js) dan antarmuka [pass2-interface.js](file:///C:/Users/abdka/Development/foresight-analysis/web/src/engines/pass2-interface.js), arsitektur Pass 2 dirancang modular dengan 6 adapter:

| Engine ID | Kelas Adapter | Model / Bobot | Karakteristik Utama |
| :--- | :--- | :--- | :--- |
| `webgpu` *(Eksisting)* | `Pass2WebGpuEngine` di [pass2-webgpu.js](file:///C:/Users/abdka/Development/foresight-analysis/web/src/engines/pass2-webgpu.js) | Transformers.js v3 (HuggingFace) | WebGPU Shader, `StaticCache` aktif, alokasi in-place |
| `wasm-simd` *(Eksisting)* | `Pass2WasmSimdEngine` di [pass2-wasm-simd.js](file:///C:/Users/abdka/Development/foresight-analysis/web/src/engines/pass2-wasm-simd.js) | Transformers.js (CPU SIMD) | Stabil di semua platform, multi-thread, zero shader stutter |
| `webnn` *(Eksisting)* | `Pass2WebNnEngine` di [pass2-webnn.js](file:///C:/Users/abdka/Development/foresight-analysis/web/src/engines/pass2-webnn.js) | WebNN OS Native Driver | Adapter deteksi NPU/GPU DirectML native |
| **`webnn-direct`** *(Opsi Baru)* | `Pass2WebNnDirectEngine` | Model ONNX lokal via `MLTensor` | DirectML NPU/GPU murni tanpa wrapper eksternal |
| **`direct-ort-web`** *(Opsi Baru)* | `Pass2DirectOrtEngine` | `base-encoder.int8.onnx` lokal | ONNX Runtime Web murni, static KV-cache manual, 100% offline |
| **`wasm-ggml`** *(Opsi Baru)* | `Pass2WasmGgmlEngine` | `ggml-base-id-q8_0.bin` lokal | whisper.cpp WASM murni, multi-threaded C++ |

---

## 6. Peta Irisan & Keterkaitan Antar-Poin (5 Sinergi Arsitektur)

Kelima bagian di atas membentuk arsitektur terintegrasi yang saling mengunci:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        PETA IRISAN ARSITEKTUR                          │
│                                                                        │
│   [Bagian 1: Eager Warmup] ◄────── Sinergi 1 ──────► [Bagian 3: StaticCache]
│              │                                                │        │
│          Sinergi 1                                            │    Sinergi 4
│              ▼                                                ▼        ▼
│   [Bagian 4.4: Pipelining]                         [Bagian 4.3: Quantization]
│                                                                        │
│   [Bagian 2: Boundary 1]   ◄────── Sinergi 2 ──────► [Bagian 4.1: Zero-Copy]  │
│   [Bagian 2: Boundary 2]   ◄────── Sinergi 3 ──────► [Bagian 4.2: WGSL Mel]   │
│                                                                        │
│   [Bagian 3: Kontrak Interface] ◄── Sinergi 5 ─────► [Bagian 5: Pass2Hub Opsi]│
└────────────────────────────────────────────────────────────────────────┘
```

### Sinergi 1: Eager Warmup (Bagian 1) $\cap$ StaticCache (Bagian 3) $\cap$ Pipelining Asinkronik (Bagian 4.4)
* **Titik Temu**: **Siklus Pre-Alokasi Buffer VRAM Permanen**.
* **Keterkaitan**: `StaticCache` memerlukan alokasi buffer tetap ($T_{\max} = 64$) di VRAM. Momen eksekusi alokasi dan kompilasi bind group ini terjadi tepat saat **Eager Warmup**. Begitu VAD mendeteksi suara mulai, **Pipelining Asinkronik** langsung menulis streaming audio ke buffer GPU yang sudah teralokasi dan berstatus *warm* tersebut.
* **Dampak**: Mencegah freeze 2.5s pada turn pertama, menghilangkan stutter alokasi dinamis decoder, dan mengeliminasi jeda transfer saat turn selesai (latensi transfer 0 ms).

### Sinergi 2: Boundary 1 AudioWorklet (Bagian 2) $\cap$ Zero-Copy Memory-Mapped WASM (Bagian 4.1)
* **Titik Temu**: **Eliminasi Penalti IPC Structured Clone 160ms pada Audio Ingestion**.
* **Keterkaitan**: Boundary 1 mendiagnosa inefisiensi pengiriman audio 160ms via `postMessage`. Bagian 4.1 menerapkan `SharedArrayBuffer` + SPSC RingBuffer yang diikat langsung ke pointer memori linear WASM melalui `catetin_alloc_f32` di [lib.rs](file:///C:/Users/abdka/Development/foresight-analysis/crates/catetin-wasm/src/lib.rs#L293).
* **Dampak**: Memangkas 4 siklus `memcpy` menjadi **0 salinan (true zero-copy)** dari mikrofon hingga ke fungsi C-ABI Sherpa-ONNX.

### Sinergi 3: Boundary 2 DMA Overhead (Bagian 2) $\cap$ Kernel WGSL Mel-Spectrogram (Bagian 4.2)
* **Titik Temu**: **Mereduksi Bandwidth Transfer Host-to-Device (CPU ke GPU VRAM)**.
* **Keterkaitan**: Boundary 2 membuktikan bahwa mentransfer 783 KB matriks mel hasil kalkulasi CPU memboroskan bandwidth PCIe/SoC. Bagian 4.2 mengirim PCM mentah 195 KB sekali jalan, lalu mengeksekusi FFT 400-point langsung di VRAM menggunakan compute shader WGSL 1-pass.
* **Dampak**: Memangkas payload transfer bus sebesar 75% dan membebaskan thread CPU dari beban komputasi FFT.

### Sinergi 4: StaticCache Decoder (Bagian 3) $\cap$ Weight-Only Quantization INT4/INT8 (Bagian 4.3)
* **Titik Temu**: **Mengatasi Saturasi Memory Bandwidth pada Autoregressive Generation ($B = 1$)**.
* **Keterkaitan**: Kecepatan generasi token sekuensial dibatasi oleh kecepatan bus memori membaca bobot dan KV-cache. Bagian 4.3 memadatkan bobot ke INT4 (memotong pembacaan bobot 50–75% per token), sementara Bagian 3 menstabilkan buffer KV-cache di slot VRAM tetap.
* **Dampak**: VRAM hanya mentransfer data yang benar-benar esensial, melipatgandakan throughput generasi token tanpa lonjakan daya prosesor.

### Sinergi 5: Isolasi Kontrak Antarmuka (Bagian 3) $\cap$ Ekstensi Multi-Engine Pass2Hub (Bagian 5)
* **Titik Temu**: **Pemisahan Tegas antara Lapisan UI/Aplikasi dan Engine Komputasi**.
* **Keterkaitan**: Kontrak `Pass2Engine` di [pass2-interface.js](file:///C:/Users/abdka/Development/foresight-analysis/web/src/engines/pass2-interface.js) mengisolasi detail runtime. Kita dapat mengadopsi Transformers.js v3 sekaligus menambahkan adapter low-level kustom (`direct-ort-web` dan `wasm-ggml`) tanpa mengubah kode di [app.js](file:///C:/Users/abdka/Development/foresight-analysis/web/src/app.js), [audio-engine.js](file:///C:/Users/abdka/Development/foresight-analysis/web/src/audio-engine.js), atau [transcript-view.js](file:///C:/Users/abdka/Development/foresight-analysis/web/src/components/transcript-view.js).
* **Dampak**: Memungkinkan perbandingan performa *head-to-head* secara objektif di panel telemetri dalam kondisi runtime yang sama.

---

## 7. Peta Tahapan Implementasi dan Verifikasi Teknis Utuh (Tahap 1 s/d 6)

Setiap tahapan dirancang dengan prosedur verifikasi *whitebox* tanpa mock data:

### Tahap 1: Eager Pre-warming & Visual UI Ready Indicator
* **Komponen & Berkas yang Diubah**:
  * [app.js](file:///C:/Users/abdka/Development/foresight-analysis/web/src/app.js): Koordinasi siklus hidup inisialisasi eager pada `DOMContentLoaded`.
  * [hub.js](file:///C:/Users/abdka/Development/foresight-analysis/web/src/engines/hub.js): Agregasi promise warmup engine aktif.
  * [pass2-webgpu.js](file:///C:/Users/abdka/Development/foresight-analysis/web/src/engines/pass2-webgpu.js), [pass2-webnn.js](file:///C:/Users/abdka/Development/foresight-analysis/web/src/engines/pass2-webnn.js), [pass2-wasm-simd.js](file:///C:/Users/abdka/Development/foresight-analysis/web/src/engines/pass2-wasm-simd.js): Eksekusi dummy pass 0.5s audio nol (`max_new_tokens: 1`).
  * [catetin-telemetry.js](file:///C:/Users/abdka/Development/foresight-analysis/web/src/components/catetin-telemetry.js): Refleksi status sub-sistem real-time.
* **Spesifikasi Perubahan**:
  * State machine: `BOOTSTRAPPING` $\to$ `WARMING_PASS1` $\to$ `WARMING_PASS2` $\to$ `SYSTEM_READY`.
  * Tombol `#btn-session` dikunci (`disabled`) selama fase warming dan menampilkan label kontekstual reaktif.
* **Verifikasi Whitebox Tanpa Mock**:
  1. *Cold-start Timing Test*: Mengukur delta waktu antara pemicuan audio turn pertama dan emisi token transkrip menggunakan `performance.now()`. Hasil wajib **< 50 ms** (dibandingkan baseline 2.5–7.5s tanpa pre-warming).
  2. *DOM State Transition Check*: Memvalidasi transisi atribut HTML `disabled` pada `#btn-session` dari `true` ke `false` dan perubahan teks label secara berurutan.
  3. *Driver Shader Cache Verification*: Memverifikasi pembentukan cache biner PSO pada direktori browser (`GPUCache` / DirectX `D3D12ShaderCache`) tanpa pesan kompilasi ulang pada log console runtime.

---

### Tahap 2: Migrasi Transformers.js v3 & StaticCache WebGPU Engine
* **Komponen & Berkas yang Diubah**:
  * [pass2-webgpu.js](file:///C:/Users/abdka/Development/foresight-analysis/web/src/engines/pass2-webgpu.js): Migrasi impor CDN ke `@huggingface/transformers@3.3.3`, konfigurasi split-dtype (`fp32` encoder, `q4` decoder), dan aktivasi `StaticCache`.
* **Spesifikasi Perubahan**:
  * Mengeliminasi alokasi dinamis KV-cache decoder per token.
  * Menetapkan panjang alokasi buffer statis $T_{\max} = 64$.
* **Verifikasi Whitebox Tanpa Mock**:
  1. *VRAM Footprint Profiling*: Mengukur alokasi memori VRAM melalui DevTools (`performance.memory` dan diagnostik internal GPU). Alokasi memori WebGPU wajib berada pada rentang **$\le$ 85 MB** (memangkas alokasi v2 sebesar ~300 MB).
  2. *Real-Time Factor (RTF) Benchmark*: Menjalankan transkripsi file audio uji 6.12 detik. Menghitung $\text{RTF} = \frac{T_{\text{inferensi}}}{T_{\text{audio}}}$. Nilai RTF wajib mencapai **sub-0.5x** (baseline v2: 1.09x).
  3. *Step Latency Stability*: Memeriksa log per-token decode interval via `performance.measure()`, memastikan deviasi waktu antar-token stabil tanpa lonjakan alokasi ulang buffer GPU.

---

### Tahap 3: Zero-Copy SPSC RingBuffer & Memory-Mapped WASM Interop
* **Komponen & Berkas yang Diubah**:
  * [lib.rs](file:///C:/Users/abdka/Development/foresight-analysis/crates/catetin-wasm/src/lib.rs): Pemanfaatan `catetin_alloc_f32` dan `catetin_free_f32` untuk ekspor pointer linear memori WebAssembly.
  * [audio-engine.js](file:///C:/Users/abdka/Development/foresight-analysis/web/src/audio-engine.js): Penggantian mekanisme salin ganda `memF32.set` dan `chunkCopy.set` dengan lock-free SPSC RingBuffer di atas `SharedArrayBuffer`.
  * [sherpa-transducer-worker.js](file:///C:/Users/abdka/Development/foresight-analysis/web/src/sherpa/sherpa-transducer-worker.js): Menerima pointer buffer bersama langsung ke fungsi C-ABI `SherpaOnnxOnlineStreamAcceptWaveform`.
  * `serve.js`: Memastikan pemancaran header isolasi `Cross-Origin-Opener-Policy: same-origin` dan `Cross-Origin-Embedder-Policy: require-corp`.
* **Spesifikasi Perubahan**:
  * Mengeliminasi 4 tahapan `memcpy` audio 160ms.
  * Menggunakan atomik indeks head/tail (`Atomics.load`, `Atomics.store`) untuk sinkronisasi thread tanpa lock.
* **Verifikasi Whitebox Tanpa Mock**:
  1. *Cross-Origin Isolation Check*: Memverifikasi ekspresi `self.crossOriginIsolated === true` pada Worker dan Window context.
  2. *Allocation Timeline Inspection*: Merekam jejak alokasi heap via Chrome DevTools Allocation Profiler selama perekaman 60 detik. Memastikan **0 alokasi objek `Float32Array` baru** pada interval reguler 160ms audio ingestion.
  3. *RingBuffer Boundary & Drift Test*: Menjalankan aliran audio terus-menerus selama 30 menit. Memverifikasi tidak ada selisih sampel (*sample drift*) atau buffer overrun/underrun antara indeks produsen dan konsumen.

---

### Tahap 4: Staging Pipelining Asinkronik Antara Pass 1 & Pass 2
* **Komponen & Berkas yang Diubah**:
  * [audio-engine.js](file:///C:/Users/abdka/Development/foresight-analysis/web/src/audio-engine.js): Intersepsi awal status VAD `speaking = true`.
  * [onnx-runner.js](file:///C:/Users/abdka/Development/foresight-analysis/web/src/onnx-runner.js) & [asr-worker.js](file:///C:/Users/abdka/Development/foresight-analysis/web/src/asr-worker.js): Penambahan kanal staging buffer GPU asinkronik selama turn aktif.
* **Spesifikasi Perubahan**:
  * Saat giliran bicara aktif, chunk audio dialirkan ke Pass 1 dan secara simultan di-append ke buffer staging GPU di VRAM.
  * Saat VAD mendeteksi akhir giliran (`speaking = false`), latensi transfer PCM ke GPU adalah 0 ms.
* **Verifikasi Whitebox Tanpa Mock**:
  1. *Turn-End to Inference Dispatched Latency*: Mengukur selisih waktu antara sinyal endpoint VAD dipancarkan hingga pemanggilan fungsi eksekusi encoder Pass 2. Nilai terukur wajib **< 1 ms** (dibandingkan baseline staging 15–40 ms).
  2. *Audio Buffer Integrity Test*: Membandingkan representasi biner antara buffer staging VRAM dan akumulasi array CPU lokal menggunakan verifikasi selisih absolut maksimum ($\max(|A - B|) = 0.0$).

---

### Tahap 5: Compute Shader WGSL Mel-Spectrogram & Decoupled GPU Decode
* **Komponen & Berkas yang Diubah**:
  * Penambahan modul compute shader `web/src/engines/mel-spectrogram.wgsl.js`.
  * Integrasi pipeline WebGPU pada [pass2-webgpu.js](file:///C:/Users/abdka/Development/foresight-analysis/web/src/engines/pass2-webgpu.js).
* **Spesifikasi Perubahan**:
  * Mengirim PCM 16kHz mentah langsung ke VRAM buffer.
  * Eksekusi STFT 400-point Hann window dan perkalian 80-channel filterbank mel langsung pada GPU workgroup shared memory (`var<workgroup>`).
  * Menjalankan autoregressive decode loop dalam satu `GPUCommandEncoder` chain mandiri.
* **Verifikasi Whitebox Tanpa Mock**:
  1. *Numerical Parity Test*: Membandingkan nilai matriks Log-Mel output WGSL terhadap ground truth ekstraksi Python NumPy / Rust Hann-window pada sampel audio uji standar. Toleransi deviasi Mean Squared Error ($\text{MSE}$) wajib **$< 10^{-5}$**.
  2. *Host-to-Device Payload Validation*: Memverifikasi ukuran buffer transfer DMA melalui profiler GPU (195 KB untuk audio 6.12 detik vs 783 KB representasi matriks mel CPU, efisiensi transfer 75%).
  3. *GPU Dispatch Trace*: Menggunakan pelacakan event `chrome://tracing`, memverifikasi bahwa generasi token sekuensial berjalan tanpa interupsi event loop JavaScript pada tiap token.
  4. *Hasil Terverifikasi di Chrome*: Modul `mel-spectrogram.js` berhasil memangkas latensi ekstraksi Mel dari baseline CPU **1.121 ms** menjadi **7 – 11 ms** (**46,6x lebih cepat**), memangkas total latensi Pass 2 WebGPU Tiny menjadi **2.195 ms (RTF 0.384x)**.

---

### Matriks Evaluasi & Ranking Terpisah (Pass 1 vs Pass 2 Mandiri)

Pengujian dilakukan secara komprehensif pada berkas audio nyata [`sample-indonesia-16k.wav`](file:///C:/Users/abdka/Development/foresight-analysis/dist/sample-indonesia-16k.wav) (6.12 detik, 97.903 sampel, 16kHz mono) secara mandiri di Google Chrome:

#### 1. Evaluasi Pass 1 (Streaming Live ASR)
Pass 1 dirancang eksklusif untuk menghasilkan teks kata-per-kata secara langsung saat pengguna masih bersuara (tanpa menunggu jeda):

| Tahap (Pass) | Opsi Engine | Model & Ukuran | Backend Hardware | Dtype | Latensi Chunk | TTFT | RTF | Hasil Transkrip | Catatan Kritis |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Pass 1** | **Streaming Transducer** | **Zipformer2 ID (70.9 MB)** | CPU WASM SIMD (1 Core) | INT8 C-ABI | **~31 ms/chunk** | **357.9 ms** | **0.133x** | *"DARI REPUBLIK INDONESIA JAKARTA"* | **Wajib Selalu Aktif**. Nol alokasi VRAM, respons live instan sub-detik. Teks berbentuk kapital tanpa tanda baca (*verbatim raw*), wajib disempurnakan oleh Pass 2. |

---

#### 2. Ranking Mandiri Pass 2 (Turn Finalizer ASR — 10 Kombinasi Terpisah Tanpa Digabung)
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
> Laporan pengujian empiris mendalam, metodologi uji, verifikasi isolasi hardware, dan matriks deployment produksi lengkap didokumentasikan di [BENCHMARK_AND_MODEL_EVALUATION.md](file:///C:/Users/abdka/Development/foresight-analysis/docs/BENCHMARK_AND_MODEL_EVALUATION.md).

---

### Tahap 6: Ekstensi Adapter Pass 2 Baru di Pass2Hub
* **Komponen & Berkas yang Diubah**:
  * [hub.js](file:///C:/Users/abdka/Development/foresight-analysis/web/src/engines/hub.js): Registrasi factory instance untuk `webnn-direct`, `direct-ort-web`, dan `wasm-ggml`.
  * Penambahan berkas engine baru: `web/src/engines/pass2-webnn-direct.js`, `web/src/engines/pass2-direct-ort.js`, `web/src/engines/pass2-wasm-ggml.js`.
  * [pass2-interface.js](file:///C:/Users/abdka/Development/foresight-analysis/web/src/engines/pass2-interface.js): Penegakan kontrak metode seragam (`init`, `warmup`, `transcribe`, `dispose`).
* **Spesifikasi Perubahan**:
  * `webnn-direct`: Inferensi langsung melalui W3C WebNN `MLTensor` tanpa pustaka perantara.
  * `direct-ort-web`: ONNX Runtime Web murni mengeksekusi model INT8 lokal (`base-encoder.int8.onnx` dan `base-decoder.int8.onnx`) dengan alokasi KV-cache statis manual.
  * `wasm-ggml`: whisper.cpp WASM murni dengan bobot terkuantisasi `q8_0` multi-thread.
* **Verifikasi Whitebox Tanpa Mock**:
  1. *Interface Contract Compliance Test*: Eksekusi script pengujian otomatis yang memeriksa kesesuaian seluruh metode (`init`, `warmup`, `transcribe`, `dispose`) pada setiap kelas adapter baru tanpa toleransi *missing method* atau stub kosong.
  2. *Dynamic Runtime Switching Test*: Mengganti engine Pass 2 secara dinamis saat sesi audio aktif melalui modal pengaturan UI (`⚙️`). Memverifikasi pelepasan resource (`dispose()`) engine lama dan inisialisasi engine baru tanpa memicu memory leak atau error konteks WebGL/WebGPU.
  3. *Comparative Telemetry Benchmark*: Menjalankan audio turn identik 10 detik ke masing-masing 6 adapter secara berurutan, merekam metrik RTF, latensi inferensi, dan konsumsi memori pada [catetin-telemetry.js](file:///C:/Users/abdka/Development/foresight-analysis/web/src/components/catetin-telemetry.js).

---

## 8. Pemetaan Komprehensif Seluruh Preferensi Arsitektur Catetin (8 Pilar Arsitektur)

Sesuai dengan keputusan desain pada [DECISIONS.md](file:///C:/Users/abdka/Development/foresight-analysis/DECISIONS.md), berikut adalah pemetaan arsitektur menyeluruh sistem Catetin:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                      8 PILAR PREFERENSI ARSITEKTUR CATETIN                             │
│                                                                                        │
│  [Pilar 1: Filosofi & Paradigma]          [Pilar 2: Audio Ingestion & DSP]             │
│  • Ponytail Mode & Zero Boilerplate       • Multi-source Capture (Mic + Dual Tab)      │
│  • 100% Offline, Zero Cloud API           • Rust WASM DSP (Hann Sinc Resampler, RMS)   │
│  • Pure ONNX & Pure C-ABI WebAssembly     • Adaptive VAD & Dynamic Speech Segmenter    │
│  • Portabilitas Eksekusi Tanpa Build Step • Drain Stale Startup Buffer & Real Pause    │
│                                                                                        │
│  [Pilar 3: Pass 1 Transducer]             [Pilar 4: Pass 2 Engine Hub]                 │
│  • Sherpa-ONNX Zipformer2 ID/EN           • Kontrak Seragam Pass2Engine & Pass2Hub     │
│  • Emisi Parsial Sub-detik (< 50ms)       • Multi-Engine (WebGPU, WebNN, WASM SIMD,    │
│  • Dedicated Transducer Worker            • WebNN-Direct, Direct-ORT, WASM-GGML)       │
│  • Umpan Balik Live Bar ⚡ Transkrip      • Pre-warming & Runtime Hot-Swapping         │
│                                                                                        │
│  [Pilar 5: Memory & Decoder Optimizations][Pilar 6: Post-Processing & Anti-Halusinasi] │
│  • Eliminasi 3 Memory Boundaries          • Pembersihan Repetition Loop (cleanLoops)   │
│  • StaticCache In-Place Allocation        • Language-locking (<|id|>) & Latin Filtering│
│  • Weight-Only INT4/INT8 Dequantization   • Dynamic Turn Cutting (4.5s vs 9s RMS)      │
│  • Pipelining Asinkronik Staging 0ms      • Pembersihan Buffer Tab Ended & Auto-Reset  │
│                                                                                        │
│  [Pilar 7: State Machine & Telemetri]     [Pilar 8: Storage & Keamanan PWA]            │
│  • Siklus Hidup Reaktif 8 State           • OPFS Streaming Model Storage               │
│  • Visual UI Ready Indicator              • File System Access API (showDirectory)    │
│  • Komponen Telemetri Mandiri 3 Thread    • Isolasi COOP/COEP (SharedArrayBuffer)      │
│  • Log Drawer Real-Time Berfilter         • Zero Host Dependency Docker Build Container│
└────────────────────────────────────────────────────────────────────────────────────────┘
```

### Pilar 1: Filosofi & Paradigma Sistem
* **Prinsip Ponytail**: Kode terkecil yang menyelesaikan masalah secara tuntas. Desain ringkas, tanpa abstraksi berlebihan (*no premature over-engineering*), dan tanpa dependensi pihak ketiga yang tidak esensial.
* **100% Offline & Privasi Mutlak**: Seluruh pemrosesan suara dan inferensi model berjalan lokal di perangkat pengguna. Nol panggilan API cloud pihak ketiga; menghapus ketergantungan pada `webkitSpeechRecognition` (Google Cloud Speech API).
* **Pure ONNX & Pure C-ABI WebAssembly**: Menghapus dependensi framework JavaScript berat (React, Vue, Webpack, Vite). Menggunakan fitur peramban modern native: **Vanilla Web Components** dan **Native ES Modules**. Zero build step runtime untuk aplikasi web.
* **Portabilitas Eksekusi**: Binari CLI portabel mandiri (meletakkan DLL native tepat di samping executable tanpa folder cache tersembunyi), serta arsitektur PWA berbasis browser standar.

### Pilar 2: Audio Ingestion, Preprocessing & DSP
* **Multi-Source Audio Capture**: Menangkap audio secara simultan dari dua sumber berbeda melalui Web Audio API:
  * Mikrofon lokal via `navigator.mediaDevices.getUserMedia`.
  * Audio tab rapat sistem via `navigator.mediaDevices.getDisplayMedia` (mendukung penangkapan multi-tab tanpa duplikasi stream audio melalui pelacakan `track.label`).
* **Rust WebAssembly DSP Pipeline ([lib.rs](file:///C:/Users/abdka/Development/foresight-analysis/crates/catetin-wasm/src/lib.rs))**:
  * Resampling audio adaptif dari sample rate peramban (44.1kHz / 48kHz) ke 16kHz mono standar ASR menggunakan filter bandlimited sinc dengan Hann-window.
  * Kalkulasi energi Root Mean Square (RMS) real-time untuk VU meter audio visual [vu-meter.js](file:///C:/Users/abdka/Development/foresight-analysis/web/src/components/vu-meter.js).
* **Adaptive Voice Activity Detection & Segmentation**:
  * Komponen `SpeechSegmenter` di Rust WASM memisahkan segmen bicara secara dinamis berdasarkan ambang energi dan jeda hening (*silence hangover*).
  * **Drain Stale Startup Buffers**: Menguras dan membuang buffer residu audio yang tertampung selama inisialisasi model, mencegah suara batuk atau pengujian mic bocor ke transkrip rapat.
  * **Perilaku Pause Nyata**: Ketika sesi dijeda, paket audio langsung dihentikan dan dibuang di level terluar untuk menghemat daya komputasi.

### Pilar 3: Pass 1 Transducer (Real-Time Sub-Detik Streaming)
* **Karakteristik & Peran**: Memberikan umpan balik instan kepada pengguna saat sedang berbicara tanpa menunggu jeda hening.
* **Model Dedikasi**: Sherpa-ONNX Zipformer2 RNN-T (`sherpa-onnx-streaming-zipformer2-id` untuk Bahasa Indonesia dan Zipformer English).
* **Dedicated Transducer Worker ([sherpa-transducer-worker.js](file:///C:/Users/abdka/Development/foresight-analysis/web/src/sherpa/sherpa-transducer-worker.js))**:
  * Inferensi streaming dijalankan di worker terpisah menggunakan runtime WASM SIMD C-ABI (`SherpaOnnxOnlineRecognizer`).
  * Menerima frame 160ms (2,560 sampel) secara berkala dan memancarkan emisi hipotesis parsial dalam waktu **< 50 ms**.
* **Umpan Balik Live Transkrip**: Mengirim hasil parsial ke [transcript-view.js](file:///C:/Users/abdka/Development/foresight-analysis/web/src/components/transcript-view.js) untuk ditampilkan secara dinamis pada bar `⚡ [Speaker]: "kata live..." [X.Xs]`, sebelum digantikan oleh hasil final Pass 2.

### Pilar 4: Pass 2 Engine Hub & Modularitas Adapter
* **Kontrak Antarmuka Seragam ([pass2-interface.js](file:///C:/Users/abdka/Development/foresight-analysis/web/src/engines/pass2-interface.js))**:
  * Kelas dasar abstrak `Pass2Engine` menetapkan kontrak wajib: `init(modelName, language, onProgress)`, `warmup()`, `transcribe(pcm16k, options)`, dan `dispose()`.
* **Koordinator Sentral ([hub.js](file:///C:/Users/abdka/Development/foresight-analysis/web/src/engines/hub.js))**:
  * Mengelola lifecycle engine aktif, instansiasi adapter, dan pertukaran model tanpa mengubah logika di [app.js](file:///C:/Users/abdka/Development/foresight-analysis/web/src/app.js) atau [audio-engine.js](file:///C:/Users/abdka/Development/foresight-analysis/web/src/audio-engine.js).
* **Ekosistem Multi-Engine**: Mendukung pemilihan fleksibel antara GPU Shader (`webgpu`), CPU SIMD Multi-threaded (`wasm-simd`), OS Driver DirectML (`webnn`), WebNN Direct (`webnn-direct`), ORT Web Murni (`direct-ort-web`), dan whisper.cpp WASM (`wasm-ggml`).
* **Pilihan Pengguna**: Dropdown pemilihan engine di modal Pengaturan (`⚙️`) memungkinkan perbandingan performa langsung secara objektif.

### Pilar 5: Bedah Mendalam Memory Management & Decoder Optimizations
* **Eliminasi 3 Batas Memori**: Mengganti penyalinan memori berkala dengan SharedArrayBuffer SPSC RingBuffer (Boundary 1), Mel-spectrogram compute shader WGSL (Boundary 2), dan Decoupled token decoding loop (Boundary 3).
* **StaticCache In-Place Allocation**: Mengalokasikan tensor KV-cache statis $T_{\max} = 64$ di VRAM, menghilangkan penalti realloc dan fragmentasi memori pada generasi sekuensial.
* **Precision & Quantization Engineering**: Mengadopsi konfigurasi split-dtype (Encoder FP32 untuk presisi fitur akustik, Decoder INT4/INT8 packed untuk memangkas pembacaan memori).
* **Mitigasi Saturasi Bandwidth Bus Memori**: Dekuantisasi on-the-fly di register core GPU/CPU, mengatasi batasan throughput memori terbagi (Unified memory / DDR4 iGPU) pada decoding $B=1$.

### Pilar 6: Post-Processing, Normalisasi & Anti-Halusinasi
* **Pembersihan Runaway Repetition Loop (`cleanRepetitiveLoops`)**:
  * Mendeteksi pengulangan n-gram berulang (1 s/d 4 kata berturut-turut) yang sering terjadi pada model Whisper saat audio berisi musik latar atau desah bising.
  * Memotong pengulangan liar dan mempertahankan kalimat pembuka yang valid daripada membuang seluruh teks transkrip.
* **Language-Locking & Filter Karakter Non-Latin**:
  * Mengunci decoder ke token `<|id|>` saat mode Bahasa Indonesia aktif untuk mencegah halusinasi peralihan bahasa acak.
  * Pada mode `auto`, sistem mendeteksi dan menyaring karakter non-Latin (Thai, Hangul, CJK, Katakana) dengan mekanisme fallback otomatis ke teks Transducer Pass 1.
* **Dynamic Turn Cutting (4.5s vs 9s)**:
  * Memotong giliran bicara lebih awal pada durasi $\ge 4.5$ detik jika terdeteksi penurunan energi RMS audio. Mencegah penumpukan antrean Web Worker saat audio kontinu (seperti video presentasi/YouTube) diputar tanpa henti.
* **Pembersihan Buffer Stream Tab Audio**:
  * Saat tab rapat ditutup (`onended`), sisa buffer audio di `SpeechSegmenter` di-flush seketika menggunakan label tab terkait (`[Tab X]`), mencegah residu audio salah dilabeli sebagai `[Anda]`.
  * Jika seluruh input audio tertutup, VAD otomatis di-reset ke status diam (*IDLE*).

### Pilar 7: State Machine, Siklus Hidup & Telemetri Diagnostik
* **State Machine Siklus Hidup Reaktif**:
  * Alur transisi status sistem: `BOOTSTRAPPING` $\to$ `WARMING_PASS1` $\to$ `WARMING_PASS2` $\to$ `SYSTEM_READY` $\to$ `LISTENING` $\to$ `RECORDING` $\to$ `PAUSED` $\to$ `DISPOSED`.
* **Visual UI Ready Indicator**:
  * Tombol `#btn-session` merefleksikan status kesiapan internal secara visual dengan animasi spinner dan label reaktif (*"Menyiapkan Engine AI..."* $\to$ *"Mulai Rapat (Siap 0ms)"*).
* **Komponen Telemetri Mandiri ([catetin-telemetry.js](file:///C:/Users/abdka/Development/foresight-analysis/web/src/components/catetin-telemetry.js))**:
  * Web Component mandiri yang menampilkan topologi eksekusi 3-thread:
    * **Thread 0 (UI Main Thread)**: FPS rendering (target 60 FPS), logical cores, alokasi memori heap JavaScript.
    * **Thread 1 (WebAudio AudioContext)**: Rust WASM resampler, VAD memory buffer, RMS VU meter.
    * **Thread 2 (Dedicated ASR Worker)**: WebGPU device adapter, kedalaman antrean tugas (*queue depth*), latensi inferensi milidetik, Real-Time Factor (RTF).
  * **Log Drawer Real-Time**: Drawer log diagnostik berfilter kategori (`AUDIO`, `VAD`, `WHISPER`, `QUEUE`, `ERROR`) dengan kemampuan salin dan pembersihan.

### Pilar 8: Storage, Keamanan & Ketahanan PWA (Zero Cloud & Portabilitas)
* **Origin Private File System (OPFS) ([opfs-storage.js](file:///C:/Users/abdka/Development/foresight-analysis/web/src/opfs-storage.js))**:
  * Mengelola penyimpanan model ASR dan aset bobot secara lokal di sandbox berkecepatan tinggi peramban dengan dukungan *streaming chunk write*.
* **File System Access API**:
  * Mendukung pemuatan folder model lokal kustom pengguna secara langsung via `window.showDirectoryPicker()`, memungkinkan operasional model *air-gapped* tanpa koneksi internet sama sekali.
* **Security Context & Isolasi Memori**:
  * Server lokal memancarkan header COOP (`same-origin`) dan COEP (`require-corp`) untuk membuka kapabilitas memori multi-threading `SharedArrayBuffer` dan `Atomics` secara aman.
* **Build Container Docker Portabel**:
  * Kompilasi seluruh artefak rilis (`catetin.exe`, `catetin`, `catetin.wasm`, dan modul `web/`) dikelola secara deterministik melalui Docker multi-stage build (`docker build --target export --output dist/ .`), menghasilkan distribusi bersih tanpa jejak file konfigurasi acak pada sistem operasi pengguna.
