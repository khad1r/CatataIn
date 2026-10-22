# Catetin - Architectural & Design Decisions (ADR)

Dokumen ini mencatat seluruh keputusan arsitektur, filosofi CLI flags, dan aturan desain penting proyek Catetin agar konsisten dan tidak terlupakan dalam iterasi selanjutnya.

---

## 1. Filosofi CLI Minimalis: Dilarang Menambah Flags! (Semua Opsi Wajib di `-c`)
- **Prinsip:** CLI Catetin dirancang sangat bersih dan bebas dari tumpukan flags (*no flag clutter*).
- **TIDAK ADA FLAGS untuk opsi-opsi konfigurasi individual**:
  - Dilarang menambahkan `--mode`, `--lang`, `--asr-lang`, `--assistant`, `--no-assistant`, `--verbose`, `--model`, `--transducer-model`, `--transducer-dir`, dll.
  - **SEMUA PENGATURAN DIJADIKAN SATU DI DALAM `-c`** (`catetin -c`).
- **Menu `-c` (Interactive Terminal Menu) Menyediakan 11 Pengaturan Lengkap:**
  1. Bahasa Tampilan UI (`id`, `en`)
  2. Bahasa Audio Rapat (`auto`, `id`, `en`)
  3. Mode Diarisasi (`Online`, `Tatap Muka / InPerson`, `Hybrid`)
  4. Model Whisper (Pass 2 Akurat)
  5. Model Transducer (Pass 1 Cepat < 50ms)
  6. Audio Input / Mikrofon
  7. Audio Loopback / Suara Rapat
  8. Peredam Bising (Noise Suppression)
  9. Panel Asisten AI Copilot (Aktif / Nonaktif)
  10. Model LLM Asisten
  11. Log Detail Verbose (Aktif / Nonaktif)
- **Flag `-c` Murni (Tanpa Argument `<path>`)**:
  - `-c` dan `--config` murni membuka menu konfigurasi interaktif. Tidak menerima argument `<path>`.
  - Berkas konfigurasi selalu baku di samping file executable (`config.json`), menjaga portabilitas 100%.

### Daftar Lengkap Perintah CLI Resmi (Hanya 5):
| Perintah | Fungsi |
|---|---|
| `catetin` | Jalankan transkripsi rapat langsung menggunakan konfigurasi tersimpan |
| `catetin -c` / `--config` | Buka menu wizard konfigurasi terminal interaktif (seluruh opsi diatur di sini) |
| `catetin check` / `--check` | Pengecekan diagnostik kesiapan sistem & model (*preflight check*) |
| `catetin setup` / `--setup` | Wizard interaktif persiapan & pengunduhan model/dependensi |
| `catetin -h` / `--help` | Tampilkan bantuan singkat |
| `catetin --version` | Tampilkan versi aplikasi |

---

## 2. Penempatan Native Runtime DLL (Portabilitas Penuh)
- **DLL Wajib Bersebelahan dengan Executable:**
  - `sherpa-onnx-c-api.dll` dan `onnxruntime.dll` diletakkan **langsung di samping berkas `catetin.exe`**.
  - **TIDAK MENGGUNAKAN** subfolder `runtime/` (`exe_dir/runtime/` atau `./runtime/`).
  - **TIDAK MENGGUNAKAN** system cache folder (`%LOCALAPPDATA%\catetin\runtime\`, `%APPDATA%`, atau `%TEMP%`).
- **Tujuan Distribusi:**
  - Memungkinkan distribusi aplikasi sebagai satu folder portabel (`.zip` untuk Windows / `.tar.gz` untuk Linux) atau via package manager (WinGet).
  - Tidak meninggalkan jejak file runtime di direktori cache sistem pengguna.

---

## 3. Arsitektur 100% Pure ONNX (Ditch GGML & whisper.cpp)
- **Penghapusan Total GGML / whisper.cpp:**
  - Dependensi `whisper-rs`, tautan C++ CMake, patch MinGW, dan file `.bin` dihapus 100%.
- **Eksekusi Inferensi:**
  - **Pass 1 (Streaming Transducer <50ms):** Sherpa-ONNX Zipformer RNN-T (CPU multithreaded AVX2).
  - **Pass 2 (Offline Whisper):** Sherpa-ONNX Whisper (DirectML GPU Windows / CPU multithreaded AVX2).
  - Ukuran binary Windows turun dari ~31.3 MB ke **26.1 MB** dengan performa latensi jauh lebih stabil dan cross-platform.

---

## 4. Dynamic Model Discovery (`discovery.rs`)
- Model dikenali berdasarkan inspeksi struktur file (bukan hardcoded path):
  - **Transducer:** Folder memiliki `*tokens*.txt`, `*encoder*.onnx`, `*decoder*.onnx`, `*joiner*.onnx`.
  - **Whisper:** Folder memiliki `*tokens*.txt`, `*encoder*.onnx`, `*decoder*.onnx` (tanpa joiner).
- Pengenalan bahasa otomatis dari nama folder:
  - Mengandung `-id`, `_id`, `indonesia` -> `language = "id"`
  - Mengandung `-en`, `_en`, `english` -> `language = "en"`
  - Model kustom yang ditaruh pengguna di `models/` langsung terdeteksi otomatis tanpa flags.

---

## 5. Modularitas Kode CLI (`catetin-cli`)
- File `crates/catetin-cli/src/main.rs` dipertahankan ringkas (< 500 baris) dan bertindak sebagai koordinator alur utama.
- Modul-modul terisolasi:
  - `console.rs`: Raw console mode Windows & Unix.
  - `kb.rs`: Non-blocking hotkey detection & interactive key reader.
  - `models.rs`: Registry preset model & downloader tersentralisasi.
  - `file_dialog.rs`: Native OS Save File Dialog (`comdlg32` / `zenity`).
  - `assistant.rs`: Klien AI Copilot (LLM summary & Q&A).
  - `wizard.rs`: Terminal configuration wizard & preflight diagnostic.
  - `tui.rs`: Real-time split-screen meeting dashboard & waveform VU meter.

---

## 6. Penanganan Audio Startup & Kebocoran Buffer Sebelum UI Terbuka
- **Audio Capture Baru Aktif Setelah Model Selesai Dimuat:**
  - Pemuatan model Whisper (Pass 2) dan Streaming Transducers (Pass 1) membutuhkan waktu startup ~20-25 detik.
  - Aliran mikrofon dan loopback (`DualAudioCapture`) **hanya diinisialisasi setelah seluruh model 100% siap di memori**.
- **Pembersihan Buffer (Drain Stale Startup Buffers):**
  - Tepat sebelum thread audio VAD (`catetin-audio-vad`) memasuki loop utamanya, seluruh paket yang mungkin sempat masuk di antrean mikrofon/loopback dibersihkan (`drain`).
  - Suara pengguna saat mengetes mic, batuk, atau kebisingan saat aplikasi masih loading tidak akan bocor ke transkrip rapat.
- **Perilaku Pause Nyata:**
  - Ketika rapat dijeda (`P` / `Space`), buffer audio baru langsung dibuang tanpa diproses VAD/ASR untuk menghemat CPU dan menjaga privasi pengguna.

---

## 7. Model Whisper & Streaming Transducer (Bahasa & Multi-Lingual)
- **Whisper Pass 2 (Akurat):**
  - Menggunakan model OpenAI Whisper Multilingual resmi INT8 ONNX (`sherpa-onnx-whisper-medium`, `base`, `small`, dll.).
  - Mendukung 99 bahasa (termasuk Bahasa Indonesia dan Bahasa Inggris).
  - Saat `asr_language` disetel ke `id` (via `catetin -c`), decoder dikunci ke `<|id|>` untuk mencegah halusinasi bahasa lain.
  - Saat disetel ke `auto`, dilengkapi filter penolak karakter non-Latin (Thai, Hangul, CJK, Katakana) dengan fallback otomatis ke hasil Transducer Pass 1.
- **Streaming Transducer Pass 1 (Cepat < 50ms):**
  - Menggunakan model dedicated Bahasa Indonesia (`sherpa-onnx-streaming-zipformer2-id` dari `spacewave`).
  - Menyediakan pengetikan langsung instan secara verbatim.

---

## 8. Standarisasi CI/CD GitHub Actions & Release Packaging
- **Perbaikan Nama Berkas Rilis:**
  - Binari Linux hasil build Docker bernama `catetin` (bukan `catetin-linux`).
  - Berkas WebAssembly hasil build Docker bernama `catetin.wasm` (bukan `catetin_wasm.wasm`).
- **Kelengkapan Bundle Windows:**
  - Paket `catetin-windows-x64.zip` wajib menyertakan `catetin.exe`, `sherpa-onnx-c-api.dll`, dan `onnxruntime.dll` agar portabel dan dapat langsung dijalankan tanpa error DLL hilang.
- **Aset Rilis Semantic Release:**
  - `.releaserc.json` menerbitkan 3 aset resmi di GitHub Releases:
    1. `catetin-windows-x64.zip`
    2. `catetin-linux-x64.tar.gz`
    3. `catetin-wasm.zip`
- **Smoke Test Windows CI:**
  - Alur `test-build.yml` memverifikasi eksekusi binari Windows (`catetin.exe --version` & `catetin.exe --help`) bersama DLL native sebelum pengujian dinyatakan lolos.

---

## 9. Arsitektur Catetin Web PWA (Prinsip Ponytail)
- **Zero Heavy Framework & Zero Build Step Runtime:**
  - Tidak menggunakan React, Vue, Webpack, atau tumpukan node_modules. Menggunakan fitur peramban native: **Native Web Components** (`<vu-meter>`, `<transcript-view>`, `<notes-view>`) dan **ES Modules**.
  - Bebas flow berbelit/wizard: Tampilan 1-layar langsung siap kerja.
- **Tipografi Bersih (Anti AI-Slop):**
  - Menggunakan font sistem (`system-ui`) dengan kontras tinggi. Hapus seluruh esai/paragraf pengantar panjang; label tombol maksimal 1-2 kata.
- **Pemisahan Kontrol Sesi & Mikrofon:**
  - Tombol **Mulai/Stop** mengatur status sesi perekaman secara global.
  - Tombol **Mic** dapat diaktifkan/dimatikan secara independen tanpa memutus sesi rapat.
- **Dukungan Multi-Tab Audio & Anti-Redundansi:**
  - Audio tab rapat (`getDisplayMedia`) dapat diklik berulang kali untuk menangkap banyak tab sekaligus.
  - Sistem mendeteksi `track.label` untuk mencegah registrasi tab yang sama secara redundan (otomatis memperbarui stream tanpa duplikasi audio).
- **Fleksibilitas Tampilan (Sembunyikan Panel Catatan):**
  - Opsi pengaturan untuk menyembunyikan panel catatan (`<notes-view>`) sehingga transkrip melebar 100% layar.

---

## 10. Implementasi Fitur Web PWA Tanpa Mock (End-to-End Real Implementation)
- **Koneksi catetin.wasm C-ABI Murni (Zero Import Dependencies):**
  - `crates/catetin-wasm/src/lib.rs` mengekspor fungsi C-ABI native murni tanpa pembungkus wasm-bindgen placeholder, sehingga modul WebAssembly dapat diinstansiasi secara instan tanpa pustaka loader eksternal.
  - Resampling 48k->16k mono menggunakan filter anti-aliasing Hann-window Rust, kalkulasi RMS VU meter, segmentasi turn bicara adaptif via `SpeechSegmenter`, dan filter halusinasi.
- **Offloading ASR ke Dedicated Web Worker (`asr-worker.js`):**
  - Inferensi berat Whisper dijalankan 100% pada Web Worker terpisah dengan transfer zero-copy `ArrayBuffer`.
  - Main thread UI dan pemrosesan WebAudio/WASM tetap berjalan mulus 60 FPS tanpa *frame drop* atau audio *glitch*.
  - Menghapus sepenuhnya ketergantungan pada `webkitSpeechRecognition` (Google Cloud Speech API) untuk menjaga privasi lokal dan kinerja konsisten.
- **Kebebasan Pemilihan Model (Pass 1 & Pass 2):**
  - Modal pengaturan 1 layar menyediakan kebebasan memilih model Whisper (Tiny ~39MB, Base ~140MB rekomendasi, Small ~460MB, custom HF ID, atau nonaktif) dan Transducer.
- **Penyimpanan OPFS & Impor Folder Lokal:**
  - `web/src/opfs-storage.js` mengelola model di Origin Private File System dengan dukungan streaming write dan impor berkas model ONNX langsung dari folder lokal pengguna via `window.showDirectoryPicker()`.
- **Build Container Docker (Zero Host Dependencies):**
  - Menghindari ketergantungan pada path cargo lokal di host pengguna. Seluruh binari (`catetin.exe`, `catetin`, `catetin.wasm`, serta folder `web/`) dikompilasi dan diekspor seragam melalui perintah `docker build --target export --output dist/ .`.

---

## 11. Telemetry Diagnostik Terpadu, Dynamic Turn Cutting, dan Pembersihan Stream Tab Audio
- **Komponen Telemetry Mandiri (`<catetin-telemetry>`):**
  - Menggabungkan seluruh metrik diagnostik ke dalam satu Custom Element native Web Component (`web/src/components/catetin-telemetry.js`).
  - Menampilkan topologi thread lengkap: Thread 0 (UI Main Thread, 60 FPS, logical cores, heap memory), Thread 1 (WebAudio AudioContext, Rust WASM resampler/VAD memory), dan Thread 2 (ASR Dedicated Worker, WebGPU adapter info, queue depth, RTF, latency ms).
  - Menyediakan log drawer real-time berfilter (AUDIO, VAD, WHISPER, QUEUE, ERROR) dengan fitur salin & bersihkan.
  - Komponen terisolasi 100%: mudah dihapus di kemudian hari hanya dengan menghapus satu baris tag `<catetin-telemetry>`.
- **Feedback Transkripsi Live Real-Time (`TranscriptView.setLiveTurn`):**
  - Begitu suara terdeteksi oleh VAD, feed transkrip langsung menampilkan bar indikator aktif (`⚡ [Speaker]: Mendengarkan... [X.Xs]`) secara real-time, menghilangkan kesan hening saat turn sedang diproses.
- **Dynamic Turn Cutting (4.5s vs 9s):**
  - Mencegah penumpukan antrean Web Worker saat audio video/YouTube diputar secara kontinu tanpa jeda 360ms: jika turn bicara mencapai >= 4.5s dan terdapat penurunan energi RMS, turn dipotong lebih awal sehingga inferensi Whisper berjalan jauh lebih cepat.
- **Pembersihan Stream Tab Audio & Pelabelan Speaker Akurat:**
  - Saat tab audio ditutup (`onended`), sisa buffer audio di `SpeechSegmenter` di-flush seketika menggunakan label tab tersebut (`[Tab X]`), mencegah residu audio tertag secara keliru sebagai `[Anda]`.
  - Jika seluruh stream audio tertutup (mic mati dan 0 tab), VAD di-reset secara instan ke status DIAM/IDLE dan sistem menghentikan listening.

---

## 12. Streaming Transkripsi Live Pass 1, Restorasi Thread Stack Drawer, dan Disertasi WebGPU vs CLI
- **Pass 1 Streaming Live Partial Transcription:**
  - `AudioEngine` mengakumulasi potongan audio 16kHz resampled (`currentTurnPcm`) selama status bicara aktif (`speaking = true`).
  - Setiap interval 1.2 detik, potongan sliding window (hingga 3.5s) dikirim via zero-copy `ArrayBuffer` ke `ASR Worker` (`type: transcribe_interim`).
  - Web Worker mengeksekusi *greedy decoding* cepat (`max_new_tokens: 16`), menghasilkan hipotesis kata langsung secara sub-detik (< 400ms) tanpa menunggu turn selesai.
  - `<transcript-view>` merender kata-kata live secara real-time pada bar `⚡ [Speaker]: "kata..." [X.Xs]`, lalu difinalisasi dengan transkrip penuh oleh Pass 2 saat turn selesai.
- **Pembersihan Runaway Repetition Loop Whisper (`cleanRepetitiveLoops`):**
  - Mencegah hilangnya kalimat transkrip valid saat Whisper mengalami pengulangan loop pada audio bising/musik: fungsi memotong pengulangan 1–4 kata berturut-turut dan mempertahankan teks pembuka yang valid daripada membuang seluruh kalimat.
- **Restorasi Thread Stack & Drawer Tab Telemetry:**
  - Metrik utama sistem (`CPU: 8 Cores`, `GPU: intel (gen-9)`, `RAM`, `VRAM: 2048MB`, `VAD`, `AUDIO`, `WHISPER`, `ANTREAN`, `PERF`) terpasang permanen pada bar utama `#1`.
  - Tombol bilah alat `Thread Stack ▾` dan tab drawer menyediakan inspeksi visual instan terhadap topologi 3 thread (Thread 0 Main UI, Thread 1 WebAudio/Rust WASM DSP, Thread 2 ASR Dedicated Worker).
- **Eliminasi Pemrosesan Panel Catatan Tersembunyi:**
  - Eksekusi `notesView.updateInsights(...)` (perhitungan frekuensi kata, stopwords, agenda, dan mutasi DOM) diguard secara ketat saat opsi "Sembunyikan Panel Catatan" aktif, menghemat siklus CPU utama 100%.
- **Disertasi Arsitektur WebGPU (Browser) vs Native CLI (whisper.cpp):**
  - **Memory & Bandwidth Bottleneck:** Intel Gen-9 iGPU (24 EUs) tidak memiliki VRAM dedicated, melainkan berbagi bus sistem DDR4 (~21–25 GB/s) dengan CPU dan OS. Model Whisper WebGPU dijalankan dalam bobot FP32/FP16 (tanpa kuantisasi GGML 4-bit/8-bit), membebani bandwidth memori terbagi.
  - **Overhead Dispatch & IPC Multi-Layer:** WebGPU Chromium berjalan melalui rantai abstraksi bertingkat: JS (`@xenova/transformers`) -> ONNX Runtime Web -> Dawn -> IPC Mojo -> Chrome GPU Process -> Direct3D 12 -> Intel User Mode Driver. Whisper adalah model *autoregressive seq2seq* yang memerlukan puluhan dispatch shader per token (800–1200 dispatch per turn). Latensi dispatch driver (~200µs × 1000 = ~250ms) mendominasi komputasi shader.
  - **Keunggulan Native CLI:** CLI mengeksekusi AVX2/FMA CPU SIMD terkompilasi langsung ke instruksi mesin x86-64 dengan bobot terkuantisasi `q8_0`/`q5_0` (~40–80 MB) yang sepenuhnya muat di cache CPU L1/L2/L3 (latensi < 10ns, bandwidth > 100 GB/s), tanpa driver kernel boundary atau IPC.

---

## 13. Arsitektur Modular Pass 2 (Adapter Pattern Hub) dan Pipeline Pre-Warming
- **Penegasan Pass 1 Tanpa Deviasi:**
  - Pass 1 difokuskan kembali sebagai Streaming Transducer (Sherpa-ONNX Zipformer2 ID) dengan emisi parsial sub-detik (< 50ms) per frame 160ms.
- **Modular Adapter Pattern Hub (`web/src/engines/`):**
  - Mengisolasi implementasi engine Pass 2 ke dalam 3 adapter mandiri di bawah kontrak interface seragam `Pass2Engine` (`pass2-interface.js`):
    - **Opsi A (`pass2-webgpu.js`):** Akselerasi GPU compute shader via Transformers.js / ONNX Runtime Web.
    - **Opsi B (`pass2-webnn.js`):** Akselerasi hardware native OS melalui W3C WebNN API (`navigator.ml` / DirectML).
    - **Opsi C (`pass2-wasm-simd.js`):** Multi-threaded CPU SIMD murni 8-thread (`SharedArrayBuffer`), bebas overhead driver GPU dan identik dengan biner CPU CLI.
  - Koordinator tunggal `Pass2Hub` (`hub.js`) mengelola instansiasi dan switching engine berdasarkan pengaturan pengguna tanpa mengubah 1 baris pun kode pipeline utama (`app.js`, `audio-engine.js`, `transcript-view.js`).
- **Pipeline Pre-Warming Terstandarisasi:**
  - Setiap engine mengimplementasikan metode `warmup()` wajib yang mengeksekusi 1 langkah dummy pass (audio nol 0.5s, `max_new_tokens: 1`) saat inisialisasi model.
  - Memaksa kompilasi Pipeline State Objects (PSO) dan alokasi memori sebelum rekaman dimulai, mengeliminasi lonjakan latensi (*cold start freeze*) 2–5 detik pada turn bicara pertama.
- **Pilihan Engine Pengguna:**
  - Opsi dropdown *"Engine Pass 2 (Turn Finalizer)"* ditambahkan ke modal Pengaturan (`⚙️`), memungkinkan perbandingan performa langsung antara WebGPU, WASM SIMD, dan WebNN secara real-time.

