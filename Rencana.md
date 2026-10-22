Sesuai permintaan Anda, berikut adalah rancangan ulang komprehensif (re-suggestion) dengan:

1. Penegasan Pass 1: Menghapus deviasi dan kembali 100% ke Streaming Transducer (Sherpa-ONNX Zipformer2 ID).
2. Komparasi Mendalam: Membedah kelebihan, kekurangan, dan karakteristik teknis dari #2.\* (WebGPU Optimization), #3.B (WebNN Native Graph), dan #4.2.B (Multi-
   Threaded WASM SIMD GGML).
3. Pemisahan Implementasi (Modular Adapter Pattern): Memisahkan ketiga opsi ke dalam modul terisolasi dengan kontrak interface seragam, sehingga ketika Anda
   memilih/mengunci salah satu opsi nanti, tidak ada refactoring pada pipeline utama (app.js, audio-engine.js, transcript-view.js, dan telemetry).
   ──────
4. Fondasi Pipeline: Pass 1 Kembali ke Desain Asli

Sesuai kesepakatan, deviasi Pass 1 ditutup. Pipeline Pass 1 dan Pass 2 dipisahkan secara tegas:

• Pass 1 (Streaming Transducer — Sherpa-ONNX Zipformer2 ID):
• Menerima chunk 160ms (2.560 sampel 16kHz) secara terus-menerus.
• Menghasilkan hipotesis kata parsial (interim partial text) secara sub-detik (< 50ms).
• Teks parsial live langsung mengalir ke bar UI: ⚡ [Speaker]: "kata sementara...".
• Non-autoregressive: Bebas dari loop decoder berat dan tidak membebani GPU/CPU.
• Pass 2 (Turn Finalizer — Whisper):
• Hanya menerima segmen audio utuh ketika VAD mendeteksi keheningan (silence ≥ 360ms) atau early cut (4.5s).
• Menggunakan salah satu dari 3 opsi engine di bawah untuk menghasilkan teks akurat, menyematkan tanda baca via catetin_format_utterance, lalu menggantikan
(replace) teks sementara Pass 1 menjadi Card transkrip permanen di DOM.
────── 2. Komparasi Komprehensif: #2._ vs #3.B vs #4.2.B
Berikut adalah perbandingan mendalam antar ketiga kandidat engine untuk Pass 2 (Turn Finalizer):
Dimensi Evaluasi | #2._ WebGPU Optimizations (Transformers.js v3… | #3.B WebNN API (W3C Standard via Direct… | #4.2.B Multi-Threaded WASM SIMD (whisper.cpp …
------------------|------------------------------------------------|------------------------------------------|------------------------------------------------
Arsitektur | Compute Shader WGSL → Dawn Engine → Mojo IPC → | Browser menyerahkan graf model utuh | Kode C++ GGML dikompilasi ke bytecode
Eksekusi | D3D12/Vulkan driver. | langsung ke API runtime OS native. | WebAssembly; instruksi CPU SIMD 128-bit
| | | native.
Overhead | Tinggi: 40–60 kernel launches per token | Sangat Rendah: 1 kali panggilan eksekusi | NOL (0 µs): Tidak ada driver/IPC. Komputasi
Dispatch / IPC | melewati IPC proses Chromium | subgraf per iterasi. | berjalan langsung di cache L1/L2/L3 CPU.
| (~150µs/dispatch). | |
Format & Sumber | File ONNX Hugging Face (onnx- | File ONNX teroptimasi DirectML / CoreML. | 100% IDENTIK DENGAN CLI: Menggunakan file
Model | community/whisper-tiny / base). Butuh download | | biner lokal ggml-base-id-q8_0.bin (81 MB) di
| terpisah. | | dist/models/.
Kuantisasi yang | Mixed precision: fp16 encoder, q4 / int8 | Bergantung driver OS (DirectML | Matang & Lengkap: q4_0, q5_0, q8_0, FP16
Didukung | decoder (MatMulNBits). | INT4/FP16, CoreML FP16). | (Format GGML/GGUF standar industri).
Karakteristik | Sangat cepat di encoder audio; decoder token | Cepat dan sangat hemat daya (terutama | Konsisten & Deterministik: ~20–30 token/detik
Throughput | agak tersendat pada iGPU/CPU-bound dispatch. | jika ada NPU atau discrete GPU). | pada CPU 8-core modern; latensi turn 4.5s ≈
| | | 300–500ms.
Shader | Ada: Perlu waktu kompilasi WGSL → HLSL/DXIL | Ada pada build graph: Context build 1 | TIDAK ADA: Bytecode WASM langsung di-JIT oleh
Compilation | (Wajib Pre-warming 1–3s saat init). | kali saat load model. | V8 saat modul diinstansiasi (0ms cold start).
(Cold-Start) | | |
Kesiapan Browser | Produksi Standar: Aktif default di Chrome, | Eksperimental / Transisi: Memerlukan | Universal 100%: Didukung di seluruh browser
(Tahun 2026) | Edge, Safari 18+. | flag browser (chrome://flags/#web- | modern sejak 2020 via SharedArrayBuffer +
| | machine-learning-neural-network). | SIMD.
Kesesuaian | Menengah (Menggunakan stack web terpisah dari | Tinggi pada Windows (Menggunakan | SANGAT TINGGI (1:1): Biner model dan algoritma
dengan Filosofi | biner CLI desktop). | DirectML yang sama dengan CLI). | decoding sama persis dengan modul CLI.
CLI | | |
────── 3. Rancangan Pemisahan Implementasi (Pluggable Adapter Pattern)

Agar Anda dapat menguji dan memilih salah satu opsi tanpa melakukan rewrite besar di masa depan, kita buat kontrak interface Pass 2 yang seragam. Pipeline
utama hanya berkomunikasi dengan interface ini.

                              ┌─────────────────────────────┐
                              │   Pipeline Utama (app.js)   │
                              └──────────────┬──────────────┘
                                             │
                                             ▼
                              ┌─────────────────────────────┐
                              │     Pass 2 Coordinator      │
                              │   (web/src/engines/hub.js)  │
                              └──────────────┬──────────────┘
                                             │
                 ┌───────────────────────────┼───────────────────────────┐
                 │ (Kontrak Sama)            │ (Kontrak Sama)            │ (Kontrak Sama)
                 ▼                           ▼                           ▼
    ┌─────────────────────────┐ ┌─────────────────────────┐ ┌─────────────────────────┐
    │         OPSI A          │ │         OPSI B          │ │         OPSI C          │
    │   Pass2WebGpuEngine     │ │    Pass2WebNnEngine     │ │   Pass2WasmSimdEngine   │
    │ (web/src/engines/       │ │ (web/src/engines/       │ │ (web/src/engines/       │
    │  pass2-webgpu.js)       │ │  pass2-webnn.js)        │ │  pass2-wasm-simd.js)    │
    ├─────────────────────────┤ ├─────────────────────────┤ ├─────────────────────────┤
    │ • Transformers.js v3    │ • navigator.ml DirectML   │ • whisper.cpp WASM        │
    │ • Split-Dtype (fp16/q4) │ • OS Graph Execution      │ • Model ggml-base-id-q8_0 │
    │ • Fused Shader Ops      │ • Zero Mojo IPC Loop      │ • Pthreads SharedMemory   │
    └─────────────────────────┘ └─────────────────────────┘ └─────────────────────────┘

### A. Kontrak Interface Bersama (Pass2Engine)

Semua engine wajib mengimplementasikan interface standar berikut:

    // web/src/engines/pass2-interface.js
    export class Pass2Engine {
      constructor(id, name, deviceType) {
        this.id = id;                     // 'webgpu' | 'webnn' | 'wasm-simd'
        this.name = name;
        this.deviceType = deviceType;     // 'GPU' | 'NPU/OS-DirectML' | 'CPU-SIMD'
        this.isReady = false;
      }

      // 1. Inisialisasi model & alokasi resource
      async init(modelPathOrId, onProgress) { throw new Error('Not implemented'); }

      // 2. Pre-warming wajib: Menghilangkan cold-start spike
      async warmup() { throw new Error('Not implemented'); }

      // 3. Transkripsi turn final (Audio 16kHz + context prompt 45 detik)
      async transcribe(pcm16k, { speaker, timeStr, contextPrompt, language }) { throw new Error('Not implemented'); }

      // 4. Pembersihan memory saat shutdown/ganti model
      async dispose() { throw new Error('Not implemented'); }
    }

──────

### B. Spesifikasi Teknis Tiap Modul Opsi

#### Modul Opsi A: pass2-webgpu.js

• File Terisolasi: web/src/engines/pass2-webgpu.js
• Library: @huggingface/transformers (v3) / ONNX Runtime Web.
• Fitur Utama:
• dtype: encoder_model: 'fp16', decoder_model_merged: 'q4' (menggunakan kernel MatMulNBits).
• Pre-warming: Saat init(), worker mengeksekusi buffer 0.5s dummy dengan max_new_tokens: 1 untuk memaksa Dawn mengompilasi PSO shader ke GPU driver cache.
• Kapan dipilih: Jika Anda ingin memanfaatkan akselerasi GPU murni dan ekosistem Hugging Face Web tanpa dependensi multi-threading CPU.

#### Modul Opsi B: pass2-webnn.js

• File Terisolasi: web/src/engines/pass2-webnn.js
• API: navigator.ml.createContext({ deviceType: 'gpu' | 'npu' }).
• Fitur Utama:
• Mengonversi graf ONNX Whisper ke WebNN Graph dan mengompilasinya via OS runtime (DirectML di Windows).
• Melompati layer penerjemahan WGSL shader dan Mojo IPC Chromium.
• Kapan dipilih: Jika Anda ingin kecepatan setara biner native Windows DirectML CLI desktop dengan konsumsi daya terendah, dengan asumsi flag browser aktif.

#### Modul Opsi C: pass2-wasm-simd.js

• File Terisolasi: web/src/engines/pass2-wasm-simd.js
• Runtime: whisper.cpp terkompilasi ke Emscripten WASM dengan -msimd128 -pthread.
• Fitur Utama:
• Membaca langsung model lokal ggml-base-id-q8_0.bin yang sudah ada di repo Catetin via OPFS / fetch stream.
• Multi-threading: Menggunakan pool Web Worker pthreads berbasis SharedArrayBuffer (header COOP/COEP sudah aktif di server).
• Zero driver lag: Sepenuhnya berjalan di CPU cache tanpa melewati API GPU browser.
• Kapan dipilih: Jika Anda memprioritaskan konsistensi 1:1 mutlak dengan CLI, deterministik di semua perangkat, dan bebas dari anomali driver GPU.
────── 4. Mekanisme Pemilihan di Pengaturan (User-Facing)

Di modal Pengaturan (⚙️), pengguna atau pengembang dapat memilih backend Pass 2 secara instan melalui dropdown:

    <label for="setting-pass2-backend">Engine Pass 2 (Turn Finalizer)</label>
    <select id="setting-pass2-backend" class="select-input">
      <option value="wasm-simd" selected>Multi-Thread CPU SIMD (whisper.cpp GGML q8_0 - 100% Mirip CLI)</option>
      <option value="webgpu">WebGPU Accelerated (Transformers.js v3 fp16/q4 - Grafika Browser)</option>
      <option value="webnn">WebNN OS-Native (DirectML / NPU - Eksperimental)</option>
    </select>

### Bagaimana Arsitektur Ini Melindungi Kode Anda dari Refactoring Ulang?

1. Koordinator Tunggal (hub.js): app.js tidak pernah memanggil WebGPU atau WASM secara langsung. app.js hanya memanggil coordinator.transcribe(pcm16k).
2. Isolasi Penuh:
   • Logika WebGPU hanya hidup di pass2-webgpu.js.
   • Logika WebNN hanya hidup di pass2-webnn.js.
   • Logika WASM GGML hanya hidup di pass2-wasm-simd.js.
3. Penguncian Opsi di Masa Depan:
   Ketika Anda nanti memutuskan: "Oke, kita kunci permanen ke Opsi C (WASM SIMD)", yang perlu dilakukan hanyalah menjadikan Opsi C sebagai default dan menghapus 2
   file adapter lainnya. Tidak ada 1 baris pun kode audio DSP, VAD, UI transkrip, atau telemetry yang perlu dirombak!
   ──────
4. Roadmap Eksekusi Bertahap

* [x] **Langkah 1 (Pass 1 Streaming Transducer Sejati)**:
  * Integrasi C-ABI Sherpa-ONNX ke WebAssembly untuk model Zipformer2 ID lokal (`dist/models/sherpa-onnx-streaming-zipformer2-id/`).
  * Frame audio 160ms dialirkan ke Transducer untuk teks live sub-detik (latensi chunk ~31 ms, TTFT 357.9 ms).
* [x] **Langkah 2 (Struktur Engine Hub & Pre-warming Contract)**:
  * Modul [pass2-interface.js](file:///C:/Users/abdka/Development/foresight-analysis/web/src/engines/pass2-interface.js) dan [hub.js](file:///C:/Users/abdka/Development/foresight-analysis/web/src/engines/hub.js) dengan kontrak wajib `init()`, `warmup()`, `transcribe()`, dan `dispose()`.
* [x] **Langkah 3 (Implementasi 5 Adapter Pass 2 Terpadu)**:
  * [pass2-webgpu.js](file:///C:/Users/abdka/Development/foresight-analysis/web/src/engines/pass2-webgpu.js): Transformers.js v3 split-dtype (`fp32`/`q4`) + `StaticCache`.
  * [pass2-wasm-simd.js](file:///C:/Users/abdka/Development/foresight-analysis/web/src/engines/pass2-wasm-simd.js): WASM multi-threading SIMD.
  * [pass2-webnn.js](file:///C:/Users/abdka/Development/foresight-analysis/web/src/engines/pass2-webnn.js): DirectML native graph OS (Eksperimental).
  * [pass2-direct-ort.js](file:///C:/Users/abdka/Development/foresight-analysis/web/src/engines/pass2-direct-ort.js): ONNX Runtime Web lokal murni zero-CDN.
  * [pass2-wasm-ggml.js](file:///C:/Users/abdka/Development/foresight-analysis/web/src/engines/pass2-wasm-ggml.js): whisper.cpp WASM model `ggml-base-id-q8_0.bin`.
* [x] **Langkah 4 (Uji Komparasi Empiris & Likuidasi Bottleneck #1)**:
  * Benchmark komparasi 11 kombinasi pada [benchmark.html](file:///C:/Users/abdka/Development/foresight-analysis/web/benchmark.html) menggunakan [`dist/sample-indonesia-16k.wav`](file:///C:/Users/abdka/Development/foresight-analysis/dist/sample-indonesia-16k.wav).
  * Pemindahan ekstraksi Mel Spectrogram ke WGSL compute shader ([mel-spectrogram.js](file:///C:/Users/abdka/Development/foresight-analysis/web/src/engines/mel-spectrogram.js)), memangkas latensi dari 1.121 ms menjadi 7–11 ms.

──────
5. Rencana Detail & Komprehensif: Perburuan Bottleneck Sistemik (Bottleneck Hunting Plan)

Fokus fase ini adalah **murni perburuan, isolasi, dan instrumentasi pengukuran akar masalah** di seluruh lapisan arsitektur sebelum tindakan optimasi dilakukan.

### 5.1. Peta 6 Area Perburuan (*Hunting Grounds*) & Sasaran Investigasi

```
[ Mikrofon / Loopback ]
         │
    (Area 2: VAD Silence Hangover Delay)
         │
         ▼
[ AudioWorklet ] ──(Area 4: Audio Buffer Marshalling & GC Churn)──► [ Main / Worker Heap ]
                                                                           │
         ┌─────────────────────────────────────────────────────────────────┴────────────────────────┐
         │                                                                                          │
         ▼                                                                                          ▼
(Area 1: Decoder CPU-GPU Sync Loop)                                                (Area 3: DRAM & L3 Cache Saturation)
(Area 5: KV-Cache Realloc & Shader JIT)                                            [ CPU WASM Multi-Core Engines ]
[ WebGPU / WebNN Compute Context ]                                                                 │
         │                                                                                          │
         └────────────────────────────────┬─────────────────────────────────────────────────────────┘
                                          │
                                          ▼
                         (Area 6: DOM Layout Reflow Thrashing)
                         [ Virtual Transcript View ]
```

---

### 5.2. Rencana Rinci Per Area Hunting

#### Area 1: Loop Autoregresif Decoder (Batas CPU ↔ GPU Sync Latency)
* **Target Komponen**: Loop `generate()` pada [pass2-webgpu.js](file:///C:/Users/abdka/Development/foresight-analysis/web/src/engines/pass2-webgpu.js) dan `@huggingface/transformers`.
* **Hipotesis Fisik**:
  * Pada batch size $B = 1$, pembentukan token bersifat sekuensial. Setiap 1 token baru menuntut siklus bolak-balik:
    1. Browser JavaScript mendispatch kernel forward pass decoder ke GPU via Mojo IPC.
    2. GPU menghitung logits token berikutnya.
    3. Buffer logits ditransfer balik ke CPU melalui pemetaan asynchronous (`mapAsync`).
    4. Main thread JS mengeksekusi operasi `argmax` / greedy search.
    5. Token baru diinputkan kembali ke antrean GPU.
  * Transit driver DirectX 12/Dawn dan sinkronisasi buffer memicu *GPU idle bubble* sebesar **4–8 ms per token**. Pada kalimat 15 token, **60–120 ms** hilang murni untuk transit IPC.
* **Metodologi & Alat Hunting**:
  * **Instrumen**: Pasang probe mikrodetik `performance.now()` presisi tinggi tepat sebelum dan sesudah eksekusi tiap token (`token_dispatch_start`, `token_gpu_resolve`, `token_argmax_done`).
  * **Tracing**: Ekspor trace `chrome://tracing` dengan kategori `gpu`, `v8.execute`, `mojom`, `toplevel`.
* **Indikator Keberhasilan Deteksi (Checklist)**:
  - [ ] Terdeteksi celah kosong (*gap*) $\ge 3\text{ ms}$ di mana GPU tidak mengeksekusi compute pipeline antara token $t$ dan $t+1$.
  - [ ] Waktu komputasi ALU GPU murni per-token terbukti $< 2\text{ ms}$, sementara waktu total iterasi loop di JS terukur $> 8\text{ ms}$.

---

#### Area 2: VAD Silence Hangover (Batas Persepsi Durasi / Dead-Time Hysteresis)
* **Target Komponen**: Algoritma VAD di [audio-engine.js](file:///C:/Users/abdka/Development/foresight-analysis/web/src/audio-engine.js) dan penghubung event ke [hub.js](file:///C:/Users/abdka/Development/foresight-analysis/web/src/engines/hub.js).
* **Hipotesis Fisik**:
  * Untuk mencegah kalimat terpotong di tengah jeda intonasi atau pernapasan normal, VAD menerapkan *hysteresis hangover* sebesar **500–800 ms**.
  * Akibatnya, meskipun komputasi Pass 2 hanya memakan waktu **1.982 ms**, pengguna mempersepsikan keterlambatan total **2.500–2.800 ms** sejak berhenti berbicara. Pass 2 tertahan dalam kondisi *dormant* selama jeda hening ini.
* **Metodologi & Alat Hunting**:
  * **Instrumen**: Logging penanda waktu audio riil:
    * $T_{\text{last\_speech}}$: Timestamp sampel PCM terakhir yang energinya $\ge \text{energyThreshold}$.
    * $T_{\text{vad\_trigger}}$: Timestamp event `onTurnComplete` dipicu oleh VAD.
    * $T_{\text{pass2\_entry}}$: Timestamp Pass 2 mulai menerima PCM buffer.
* **Indikator Keberhasilan Deteksi (Checklist)**:
  - [ ] Terukur selisih waktu $(T_{\text{vad\_trigger}} - T_{\text{last\_speech}}) \ge 500\text{ ms}$.
  - [ ] Memverifikasi apakah ada data audio bernilai hening (*zero-padded silence*) sepanjang 500–800 ms yang ikut dikirim dan diproses sia-sia oleh encoder Pass 2.

---

#### Area 3: Saturasi Bandwidth DRAM & Cache L3 (Batas Memory-Bound CPU WASM)
* **Target Komponen**: Modul WASM SIMD multi-thread [pass2-wasm-simd.js](file:///C:/Users/abdka/Development/foresight-analysis/web/src/engines/pass2-wasm-simd.js), [pass2-direct-ort.js](file:///C:/Users/abdka/Development/foresight-analysis/web/src/engines/pass2-direct-ort.js), dan [pass2-wasm-ggml.js](file:///C:/Users/abdka/Development/foresight-analysis/web/src/engines/pass2-wasm-ggml.js).
* **Hipotesis Fisik**:
  * Pada komputasi decoder autoregresif ($B = 1$), rasio aritmatika (Arithmetic Intensity / FLOPs per byte) sangat rendah.
  * Model Whisper Base (140 MB) melampaui kapasitas L3 Cache CPU (16–32 MB).
  * Pada setiap token baru, seluruh 140 MB bobot harus dipindahkan dari DRAM utama melalui memory bus ke core CPU. Untuk kalimat 15 token, volume transfer data ke CPU mencapai **$15 \times 140\text{ MB} = 2,1\text{ GB}$**.
  * Terjadi *bus starvation* yang melumpuhkan eksekusi SIMD dan menyebabkan degradasi akumulasi floating point (*acoustic collapse*). Sebaliknya, Whisper Tiny (39 MB) sebagian besar muat dalam hierarki cache L3.
* **Metodologi & Alat Hunting**:
  * **Instrumen**: Pengukuran waktu inferensi per token secara terisolasi pada CPU Worker WASM.
  * **Hardware Profiler**: Windows Performance Recorder / Intel VTune / Windows `typeperf "\Memory\Available MBytes"` dan cache miss counter.
* **Indikator Keberhasilan Deteksi (Checklist)**:
  - [ ] Latensi per-token Whisper Base pada CPU WASM terukur $> 150\text{ ms/token}$, sementara Whisper Tiny terukur $< 25\text{ ms/token}$ pada core yang sama.
  - [ ] Terdeteksi korelasi langsung antara lonjakan *L3 Cache Misses* dengan fenomena kegagalan atensi audio (*acoustic collapse*).

---

#### Area 4: Marshalling Buffer Audio & Garbage Collection Churn (Batas Heap V8)
* **Target Komponen**: Pipeline alokasi staging buffer di [audio-engine.js](file:///C:/Users/abdka/Development/foresight-analysis/web/src/audio-engine.js) dan transfer data ke Worker via `postMessage`.
* **Hipotesis Fisik**:
  * Saat turn selesai, pengambilan potongan PCM dari buffer 10 detik berpotensi mengeksekusi operasi `Float32Array.slice()` yang mengalokasikan memori baru di heap V8 (~640 KB per turn).
  * Jika transfer ke worker dilakukan tanpa *Transferable ArrayBuffer* (terjadi kloning terstruktur / *structured clone*), objek memori terduplikasi.
  * Pola alokasi berulang dalam rapat berdurasi 30–60 menit memicu *Minor GC (Scavenge)* atau *Major GC (Mark-Sweep)* secara periodik yang membekukan thread selama puluhan milidetik.
* **Metodologi & Alat Hunting**:
  * **Instrumen**: Monitor `performance.memory.usedJSHeapSize` sebelum dan sesudah transfer turn.
  * **DevTools**: Panel **Performance** Chrome dengan tab **Memory** dicentang; rekam sesi simulasi 3 menit berisi 20 giliran bicara berturut-turut.
* **Indikator Keberhasilan Deteksi (Checklist)**:
  - [ ] Terlihat pola grafik memori JS berbentuk gigi gergaji (*sawtooth pattern*) tajam dengan penurunan mendadak yang bertepatan dengan frame drop.
  - [ ] Terdeteksi event `Major GC` dengan durasi $\ge 15\text{ ms}$ tepat saat transkripsi sedang berlangsung.

---

#### Area 5: KV-Cache Dynamic Reallocation & Driver Shader JIT (Batas GPU VRAM)
* **Target Komponen**: Alokasi `StaticCache` pada [pass2-webgpu.js](file:///C:/Users/abdka/Development/foresight-analysis/web/src/engines/pass2-webgpu.js).
* **Hipotesis Fisik**:
  * Konfigurasi `StaticCache` dialokasikan dengan $T_{\max} = 64$.
  * Ketika giliran bicara panjang menghasilkan lebih dari 64 token, driver grafis terpaksa membuang buffer VRAM lama dan mengalokasikan buffer baru yang lebih besar di tengah-tengah loop decode.
  * Realokasi buffer VRAM memutus rantai antrean perintah GPU (*command buffer stall*) dan berpotensi memicu kompilasi ulang shader Pipeline State Object (PSO) di level driver DirectX 12.
* **Metodologi & Alat Hunting**:
  * **Instrumen**: Uji ketahanan (*stress test*) bertingkat menggunakan audio dengan panjang token bervariasi:
    * Kalimat Pendek (10–20 token)
    * Kalimat Menengah (40–60 token)
    * Kalimat Ambang Batas (64–70 token)
    * Kalimat Panjang (> 100 token)
  * Plot grafik waktu per-token terhadap indeks token ($1 \to N$).
* **Indikator Keberhasilan Deteksi (Checklist)**:
  - [ ] Waktu per-token melompat secara anomali ($\Delta t > 150\text{ ms}$) tepat saat melewati indeks token ke-64 atau kelipatannya.
  - [ ] Muncul event alokasi VRAM sekunder pada profiler GPU di tengah sesi aktif.

---

#### Area 6: DOM Layout Reflow Thrashing (Batas Render UI 60 FPS)
* **Target Komponen**: Komponen [transcript-view.js](file:///C:/Users/abdka/Development/foresight-analysis/web/src/components/transcript-view.js) dan pembaruan kontainer `#transcript-container`.
* **Hipotesis Fisik**:
  * Pass 1 mengalirkan teks interim setiap **160 ms**, sementara Pass 2 secara simultan menyisipkan blok transkripsi final.
  * Jika kode melakukan manipulasi DOM (seperti `.appendChild()` atau `.innerHTML`) kemudian langsung membaca properti dimensi layout geometris (seperti `.scrollTop`, `.scrollHeight`, atau `.getBoundingClientRect()`), browser dipaksa menghitung ulang layout seluruh halaman secara sinkron (*Forced Synchronous Layout*).
* **Metodologi & Alat Hunting**:
  * **Instrumen**: Logging durasi eksekusi method render pada `TranscriptView`.
  * **DevTools**: Panel **Performance** Chrome; aktifkan pelacakan *Rendering* dan filter event `Layout`.
* **Indikator Keberhasilan Deteksi (Checklist)**:
  - [ ] Muncul tanda segitiga merah pada DevTools: *"Forced reflow is a likely performance bottleneck"*.
  - [ ] Frame rate antarmuka turun di bawah 30 FPS saat transkripsi streaming sedang aktif mengalir.

---

### 5.3. Jadwal & Urutan Prioritas Eksekusi Hunting

| Prioritas | Area Hunting | Target Komponen | Dampak Potensial | Tingkat Kesulitan Profiling |
| :---: | :--- | :--- | :--- | :---: |
| **P1** | **Area 1: Decoder CPU-GPU Sync Loop** | `pass2-webgpu.js` | 60–120 ms murni transit IPC | Menengah (Chrome Tracing) |
| **P2** | **Area 2: VAD Silence Hangover Delay** | `audio-engine.js` | 500–800 ms persepsi jeda mati | Rendah (High-Res Timestamps) |
| **P3** | **Area 4: Buffer Marshalling & GC** | `audio-engine.js` $\to$ Worker | Micro-stutter 15–45 ms | Rendah (DevTools Memory Heap) |
| **P4** | **Area 5: KV-Cache Realloc & JIT** | StaticCache WebGPU | Spike 200–400 ms pada teks panjang | Menengah (Synthetic Audio Sweep) |
| **P5** | **Area 6: DOM Layout Reflow Thrashing** | `transcript-view.js` | Jitter antarmuka < 30 FPS | Rendah (DevTools Rendering Panel) |
| **P6** | **Area 3: DRAM & L3 Cache Saturation** | WASM SIMD Engines | Degradasi model Base | Tinggi (Hardware Performance Counters) |

──────
6. Rencana Implementasi: Fitur Uji Kesiapan Hardware Interaktif di UI

Fitur ini menyediakan pengujian empiris cepat (≤ 1.5 detik) yang terikat langsung dengan Opsi Pass 2 dan Model yang sedang dipilih pengguna di modal Pengaturan (⚙️). Tujuannya membuktikan secara transparan kemampuan fisik hardware pengguna tanpa slogan atau asumsi teoritis.

### 6.1. Prinsip Desain UX
* **Terikat Kontekstual**: Hanya menguji hardware dan model yang sedang aktif dipilih di dropdown (zero bloat, tidak mengunduh model lain yang tidak dipakai).
* **Indikator Status Aktif**: Menampilkan visual loading/pulsing yang jelas di UI saat pengujian hardware sedang berjalan agar pengguna tahu sistem sedang menguji mesin mereka.
* **Hasil Fungsional & Padat**: Cukup 1 baris badge status tanpa teks panjang bertele-tele.

### 6.2. Alur Pengujian per Opsi Pass 2
1. **Opsi A (WebGPU)**:
   * Cek `navigator.gpu` dan alokasi buffer VRAM statis.
   * Jalankan micro-inference 0.5 detik audio dummy di GPU.
   * Ukur RTF: Jika RTF < 0.8x $\to$ 🟢 **Mampu (RTF {rtf}x)**. Jika GPU context lost $\to$ 🔴 **GPU OOM / Tidak Didukung**.
2. **Opsi B & E (CPU WASM SIMD / whisper.cpp)**:
   * Cek `WebAssembly.validate()` (SIMD) dan thread pool.
   * Jalankan micro-inference 0.5 detik audio dummy di CPU.
   * Ukur RTF: Jika memilih model Base dan RTF > 1.0x $\to$ 🔴 **Hardware Terlalu Lambat (RTF {rtf}x). Rekomendasi: Gunakan Whisper Tiny**.
3. **Opsi C (WebNN DirectML)**:
   * Cek `navigator.ml?.createContext()`.
   * Jika flag belum aktif $\to$ ⚠️ **Flag WebNN Belum Aktif di Browser**.
4. **Opsi D (Direct ORT Web)**:
   * Cek alokasi ONNX Runtime Web provider CPU/WASM.

### 6.3. Spesifikasi Komponen & State UI
* **Elemen UI di Modal Pengaturan (`web/index.html`)**:
  * Tombol: `<button id="btn-test-hardware" class="btn-secondary">🔍 Uji Engine Ini</button>`
  * Badge Status: `<div id="hardware-test-status" class="hw-status-badge hidden"></div>`
* **State Transisi UI**:
  1. **Idle**: Tombol aktif `[ 🔍 Uji Engine Ini ]`, badge tersembunyi.
  2. **Testing (Indikasi Hardware Sedang Diuji)**:
     * Tombol berubah teks menjadi `[ ⏳ Menguji Hardware... ]` dan status `disabled`.
     * Badge status menampilkan animasi pulse warna kuning:
       `⏳ Menjalankan inferensi uji coba pada hardware Anda...`
  3. **Done (Hasil Evaluasi Objektif)**:
     * Tombol kembali aktif.
     * Badge menampilkan hasil sesuai performa riil:
       * 🟢 **Mampu (RTF {rtf}x)** — Hardware siap dan responsif.
       * 🔴 **Terlalu Lambat (RTF {rtf}x)** — Hardware tidak kuat untuk model ini. Ubah ke Whisper Tiny.
       * ⚠️ **Tidak Didukung** — Fitur grafis/API belum aktif di peramban ini.

### 6.4. Berkas yang Diubah
* [web/src/engines/pass2-interface.js](file:///C:/Users/abdka/Development/foresight-analysis/web/src/engines/pass2-interface.js): Penambahan kontrak method `testHardware(dummyPcm)`.
* [web/src/engines/hub.js](file:///C:/Users/abdka/Development/foresight-analysis/web/src/engines/hub.js): Dispatcher pengujian hardware ke engine yang sedang dipilih.
* [web/index.html](file:///C:/Users/abdka/Development/foresight-analysis/web/index.html) & [web/style.css](file:///C:/Users/abdka/Development/foresight-analysis/web/style.css): Tombol uji, badge status, dan keyframe animasi pulse testing.
* [web/src/app.js](file:///C:/Users/abdka/Development/foresight-analysis/web/src/app.js): Event listener tombol `#btn-test-hardware`, timer pengukuran, dan rendering state badge.
