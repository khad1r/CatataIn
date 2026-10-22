const fs = require('fs');
const path = require('path');

const filePath = path.join(__dirname, '..', 'Rencana.md');
let content = fs.readFileSync(filePath, 'utf8');

const targetIndex = content.indexOf('4. Roadmap Eksekusi Bertahap (Bila Diskusi Disepakati)');
if (targetIndex === -1) {
  console.error('Target not found!');
  process.exit(1);
}

// Find the line preceding '4. Roadmap Eksekusi Bertahap'
const before = content.substring(0, targetIndex);

const newSection = `4. Roadmap Eksekusi Bertahap

* [x] **Langkah 1 (Pass 1 Streaming Transducer Sejati)**:
  * Integrasi C-ABI Sherpa-ONNX ke WebAssembly untuk model Zipformer2 ID lokal (\`dist/models/sherpa-onnx-streaming-zipformer2-id/\`).
  * Frame audio 160ms dialirkan ke Transducer untuk teks live sub-detik (latensi chunk ~31 ms, TTFT 357.9 ms).
* [x] **Langkah 2 (Struktur Engine Hub & Pre-warming Contract)**:
  * Modul [pass2-interface.js](file:///C:/Users/abdka/Development/foresight-analysis/web/src/engines/pass2-interface.js) dan [hub.js](file:///C:/Users/abdka/Development/foresight-analysis/web/src/engines/hub.js) dengan kontrak wajib \`init()\`, \`warmup()\`, \`transcribe()\`, dan \`dispose()\`.
* [x] **Langkah 3 (Implementasi 5 Adapter Pass 2 Terpadu)**:
  * [pass2-webgpu.js](file:///C:/Users/abdka/Development/foresight-analysis/web/src/engines/pass2-webgpu.js): Transformers.js v3 split-dtype (\`fp32\`/\`q4\`) + \`StaticCache\`.
  * [pass2-wasm-simd.js](file:///C:/Users/abdka/Development/foresight-analysis/web/src/engines/pass2-wasm-simd.js): WASM multi-threading SIMD.
  * [pass2-webnn.js](file:///C:/Users/abdka/Development/foresight-analysis/web/src/engines/pass2-webnn.js): DirectML native graph OS (Eksperimental).
  * [pass2-direct-ort.js](file:///C:/Users/abdka/Development/foresight-analysis/web/src/engines/pass2-direct-ort.js): ONNX Runtime Web lokal murni zero-CDN.
  * [pass2-wasm-ggml.js](file:///C:/Users/abdka/Development/foresight-analysis/web/src/engines/pass2-wasm-ggml.js): whisper.cpp WASM model \`ggml-base-id-q8_0.bin\`.
* [x] **Langkah 4 (Uji Komparasi Empiris & Likuidasi Bottleneck #1)**:
  * Benchmark komparasi 11 kombinasi pada [benchmark.html](file:///C:/Users/abdka/Development/foresight-analysis/web/benchmark.html) menggunakan [\`dist/sample-indonesia-16k.wav\`](file:///C:/Users/abdka/Development/foresight-analysis/dist/sample-indonesia-16k.wav).
  * Pemindahan ekstraksi Mel Spectrogram ke WGSL compute shader ([mel-spectrogram.js](file:///C:/Users/abdka/Development/foresight-analysis/web/src/engines/mel-spectrogram.js)), memangkas latensi dari 1.121 ms menjadi 7–11 ms.

──────
5. Rencana Detail & Komprehensif: Perburuan Bottleneck Sistemik (Bottleneck Hunting Plan)

Fokus fase ini adalah **murni perburuan, isolasi, dan instrumentasi pengukuran akar masalah** di seluruh lapisan arsitektur sebelum tindakan optimasi dilakukan.

### 5.1. Peta 6 Area Perburuan (*Hunting Grounds*) & Sasaran Investigasi

\`\`\`
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
\`\`\`

---

### 5.2. Rencana Rinci Per Area Hunting

#### Area 1: Loop Autoregresif Decoder (Batas CPU ↔ GPU Sync Latency)
* **Target Komponen**: Loop \`generate()\` pada [pass2-webgpu.js](file:///C:/Users/abdka/Development/foresight-analysis/web/src/engines/pass2-webgpu.js) dan \`@huggingface/transformers\`.
* **Hipotesis Fisik**:
  * Pada batch size $B = 1$, pembentukan token bersifat sekuensial. Setiap 1 token baru menuntut siklus bolak-balik:
    1. Browser JavaScript mendispatch kernel forward pass decoder ke GPU via Mojo IPC.
    2. GPU menghitung logits token berikutnya.
    3. Buffer logits ditransfer balik ke CPU melalui pemetaan asynchronous (\`mapAsync\`).
    4. Main thread JS mengeksekusi operasi \`argmax\` / greedy search.
    5. Token baru diinputkan kembali ke antrean GPU.
  * Transit driver DirectX 12/Dawn dan sinkronisasi buffer memicu *GPU idle bubble* sebesar **4–8 ms per token**. Pada kalimat 15 token, **60–120 ms** hilang murni untuk transit IPC.
* **Metodologi & Alat Hunting**:
  * **Instrumen**: Pasang probe mikrodetik \`performance.now()\` presisi tinggi tepat sebelum dan sesudah eksekusi tiap token (\`token_dispatch_start\`, \`token_gpu_resolve\`, \`token_argmax_done\`).
  * **Tracing**: Ekspor trace \`chrome://tracing\` dengan kategori \`gpu\`, \`v8.execute\`, \`mojom\`, \`toplevel\`.
* **Indikator Keberhasilan Deteksi (Checklist)**:
  - [ ] Terdeteksi celah kosong (*gap*) $\\ge 3\\text{ ms}$ di mana GPU tidak mengeksekusi compute pipeline antara token $t$ dan $t+1$.
  - [ ] Waktu komputasi ALU GPU murni per-token terbukti $< 2\\text{ ms}$, sementara waktu total iterasi loop di JS terukur $> 8\\text{ ms}$.

---

#### Area 2: VAD Silence Hangover (Batas Persepsi Durasi / Dead-Time Hysteresis)
* **Target Komponen**: Algoritma VAD di [audio-engine.js](file:///C:/Users/abdka/Development/foresight-analysis/web/src/audio-engine.js) dan penghubung event ke [hub.js](file:///C:/Users/abdka/Development/foresight-analysis/web/src/engines/hub.js).
* **Hipotesis Fisik**:
  * Untuk mencegah kalimat terpotong di tengah jeda intonasi atau pernapasan normal, VAD menerapkan *hysteresis hangover* sebesar **500–800 ms**.
  * Akibatnya, meskipun komputasi Pass 2 hanya memakan waktu **1.982 ms**, pengguna mempersepsikan keterlambatan total **2.500–2.800 ms** sejak berhenti berbicara. Pass 2 tertahan dalam kondisi *dormant* selama jeda hening ini.
* **Metodologi & Alat Hunting**:
  * **Instrumen**: Logging penanda waktu audio riil:
    * $T_{\\text{last\\_speech}}$: Timestamp sampel PCM terakhir yang energinya $\\ge \\text{energyThreshold}$.
    * $T_{\\text{vad\\_trigger}}$: Timestamp event \`onTurnComplete\` dipicu oleh VAD.
    * $T_{\\text{pass2\\_entry}}$: Timestamp Pass 2 mulai menerima PCM buffer.
* **Indikator Keberhasilan Deteksi (Checklist)**:
  - [ ] Terukur selisih waktu $(T_{\\text{vad\\_trigger}} - T_{\\text{last\\_speech}}) \\ge 500\\text{ ms}$.
  - [ ] Memverifikasi apakah ada data audio bernilai hening (*zero-padded silence*) sepanjang 500–800 ms yang ikut dikirim dan diproses sia-sia oleh encoder Pass 2.

---

#### Area 3: Saturasi Bandwidth DRAM & Cache L3 (Batas Memory-Bound CPU WASM)
* **Target Komponen**: Modul WASM SIMD multi-thread [pass2-wasm-simd.js](file:///C:/Users/abdka/Development/foresight-analysis/web/src/engines/pass2-wasm-simd.js), [pass2-direct-ort.js](file:///C:/Users/abdka/Development/foresight-analysis/web/src/engines/pass2-direct-ort.js), dan [pass2-wasm-ggml.js](file:///C:/Users/abdka/Development/foresight-analysis/web/src/engines/pass2-wasm-ggml.js).
* **Hipotesis Fisik**:
  * Pada komputasi decoder autoregresif ($B = 1$), rasio aritmatika (Arithmetic Intensity / FLOPs per byte) sangat rendah.
  * Model Whisper Base (140 MB) melampaui kapasitas L3 Cache CPU (16–32 MB).
  * Pada setiap token baru, seluruh 140 MB bobot harus dipindahkan dari DRAM utama melalui memory bus ke core CPU. Untuk kalimat 15 token, volume transfer data ke CPU mencapai **$15 \\times 140\\text{ MB} = 2,1\\text{ GB}$**.
  * Terjadi *bus starvation* yang melumpuhkan eksekusi SIMD dan menyebabkan degradasi akumulasi floating point (*acoustic collapse*). Sebaliknya, Whisper Tiny (39 MB) sebagian besar muat dalam hierarki cache L3.
* **Metodologi & Alat Hunting**:
  * **Instrumen**: Pengukuran waktu inferensi per token secara terisolasi pada CPU Worker WASM.
  * **Hardware Profiler**: Windows Performance Recorder / Intel VTune / Windows \`typeperf "\\Memory\\Available MBytes"\` dan cache miss counter.
* **Indikator Keberhasilan Deteksi (Checklist)**:
  - [ ] Latensi per-token Whisper Base pada CPU WASM terukur $> 150\\text{ ms/token}$, sementara Whisper Tiny terukur $< 25\\text{ ms/token}$ pada core yang sama.
  - [ ] Terdeteksi korelasi langsung antara lonjakan *L3 Cache Misses* dengan fenomena kegagalan atensi audio (*acoustic collapse*).

---

#### Area 4: Marshalling Buffer Audio & Garbage Collection Churn (Batas Heap V8)
* **Target Komponen**: Pipeline alokasi staging buffer di [audio-engine.js](file:///C:/Users/abdka/Development/foresight-analysis/web/src/audio-engine.js) dan transfer data ke Worker via \`postMessage\`.
* **Hipotesis Fisik**:
  * Saat turn selesai, pengambilan potongan PCM dari buffer 10 detik berpotensi mengeksekusi operasi \`Float32Array.slice()\` yang mengalokasikan memori baru di heap V8 (~640 KB per turn).
  * Jika transfer ke worker dilakukan tanpa *Transferable ArrayBuffer* (terjadi kloning terstruktur / *structured clone*), objek memori terduplikasi.
  * Pola alokasi berulang dalam rapat berdurasi 30–60 menit memicu *Minor GC (Scavenge)* atau *Major GC (Mark-Sweep)* secara periodik yang membekukan thread selama puluhan milidetik.
* **Metodologi & Alat Hunting**:
  * **Instrumen**: Monitor \`performance.memory.usedJSHeapSize\` sebelum dan sesudah transfer turn.
  * **DevTools**: Panel **Performance** Chrome dengan tab **Memory** dicentang; rekam sesi simulasi 3 menit berisi 20 giliran bicara berturut-turut.
* **Indikator Keberhasilan Deteksi (Checklist)**:
  - [ ] Terlihat pola grafik memori JS berbentuk gigi gergaji (*sawtooth pattern*) tajam dengan penurunan mendadak yang bertepatan dengan frame drop.
  - [ ] Terdeteksi event \`Major GC\` dengan durasi $\\ge 15\\text{ ms}$ tepat saat transkripsi sedang berlangsung.

---

#### Area 5: KV-Cache Dynamic Reallocation & Driver Shader JIT (Batas GPU VRAM)
* **Target Komponen**: Alokasi \`StaticCache\` pada [pass2-webgpu.js](file:///C:/Users/abdka/Development/foresight-analysis/web/src/engines/pass2-webgpu.js).
* **Hipotesis Fisik**:
  * Konfigurasi \`StaticCache\` dialokasikan dengan $T_{\\max} = 64$.
  * Ketika giliran bicara panjang menghasilkan lebih dari 64 token, driver grafis terpaksa membuang buffer VRAM lama dan mengalokasikan buffer baru yang lebih besar di tengah-tengah loop decode.
  * Realokasi buffer VRAM memutus rantai antrean perintah GPU (*command buffer stall*) dan berpotensi memicu kompilasi ulang shader Pipeline State Object (PSO) di level driver DirectX 12.
* **Metodologi & Alat Hunting**:
  * **Instrumen**: Uji ketahanan (*stress test*) bertingkat menggunakan audio dengan panjang token bervariasi:
    * Kalimat Pendek (10–20 token)
    * Kalimat Menengah (40–60 token)
    * Kalimat Ambang Batas (64–70 token)
    * Kalimat Panjang (> 100 token)
  * Plot grafik waktu per-token terhadap indeks token ($1 \\to N$).
* **Indikator Keberhasilan Deteksi (Checklist)**:
  - [ ] Waktu per-token melompat secara anomali ($\\Delta t > 150\\text{ ms}$) tepat saat melewati indeks token ke-64 atau kelipatannya.
  - [ ] Muncul event alokasi VRAM sekunder pada profiler GPU di tengah sesi aktif.

---

#### Area 6: DOM Layout Reflow Thrashing (Batas Render UI 60 FPS)
* **Target Komponen**: Komponen [transcript-view.js](file:///C:/Users/abdka/Development/foresight-analysis/web/src/components/transcript-view.js) dan pembaruan kontainer \`#transcript-container\`.
* **Hipotesis Fisik**:
  * Pass 1 mengalirkan teks interim setiap **160 ms**, sementara Pass 2 secara simultan menyisipkan blok transkripsi final.
  * Jika kode melakukan manipulasi DOM (seperti \`.appendChild()\` atau \`.innerHTML\`) kemudian langsung membaca properti dimensi layout geometris (seperti \`.scrollTop\`, \`.scrollHeight\`, atau \`.getBoundingClientRect()\`), browser dipaksa menghitung ulang layout seluruh halaman secara sinkron (*Forced Synchronous Layout*).
* **Metodologi & Alat Hunting**:
  * **Instrumen**: Logging durasi eksekusi method render pada \`TranscriptView\`.
  * **DevTools**: Panel **Performance** Chrome; aktifkan pelacakan *Rendering* dan filter event \`Layout\`.
* **Indikator Keberhasilan Deteksi (Checklist)**:
  - [ ] Muncul tanda segitiga merah pada DevTools: *"Forced reflow is a likely performance bottleneck"*.
  - [ ] Frame rate antarmuka turun di bawah 30 FPS saat transkripsi streaming sedang aktif mengalir.

---

### 5.3. Jadwal & Urutan Prioritas Eksekusi Hunting

| Prioritas | Area Hunting | Target Komponen | Dampak Potensial | Tingkat Kesulitan Profiling |
| :---: | :--- | :--- | :--- | :---: |
| **P1** | **Area 1: Decoder CPU-GPU Sync Loop** | \`pass2-webgpu.js\` | 60–120 ms murni transit IPC | Menengah (Chrome Tracing) |
| **P2** | **Area 2: VAD Silence Hangover Delay** | \`audio-engine.js\` | 500–800 ms persepsi jeda mati | Rendah (High-Res Timestamps) |
| **P3** | **Area 4: Buffer Marshalling & GC** | \`audio-engine.js\` $\\to$ Worker | Micro-stutter 15–45 ms | Rendah (DevTools Memory Heap) |
| **P4** | **Area 5: KV-Cache Realloc & JIT** | StaticCache WebGPU | Spike 200–400 ms pada teks panjang | Menengah (Synthetic Audio Sweep) |
| **P5** | **Area 6: DOM Layout Reflow Thrashing** | \`transcript-view.js\` | Jitter antarmuka < 30 FPS | Rendah (DevTools Rendering Panel) |
| **P6** | **Area 3: DRAM & L3 Cache Saturation** | WASM SIMD Engines | Degradasi model Base | Tinggi (Hardware Performance Counters) |
`;

fs.writeFileSync(filePath, before + newSection, 'utf8');
console.log('Successfully updated Rencana.md!');
