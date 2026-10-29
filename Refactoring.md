# Refactoring & Technical Backlog `web/src`

Dasar: audit `web/src` dan `web/src/engines` serta analisis arsitektur ASR pipeline.
Prinsip: perubahan terkecil yang menuntaskan masalah (ponytail). Hapus dulu, tambah belakangan.

> [!NOTE]
> Seluruh riwayat pekerjaan refactoring & pengerasan sistem yang telah selesai dan terverifikasi (Fase 0 Sinkronisasi Dokumentasi & Audit Arsitektur, Fase 1 Bug Fixes, Fase 2 Zero Mock & Zero Fallback Engines, Fase 3 Pembersihan Kode Mati, Fase 4 Optimasi Transfer & Render, Fase 5.1 Onboarding Hardware Test, Fase 5.3 Self-Host Transformers.js, Fase 5.4 Cache Storage Model Transducer, Fase 6 Audit Fixes #36–#49, Implementasi Penuh Pass2DirectOrtEngine & CPU Mel Spectrogram, Implementasi Adapter C-ABI Pass2WasmGgmlEngine, serta Verifikasi Matriks Penuh 30 Kombinasi) telah dipindahkan dan didokumentasikan secara rinci di [CHANGELOG.md](./CHANGELOG.md).

---

## Prinsip Arsitektur Aktif

1. **Zero Silent Fallback (Tanpa Fallback Tersembunyi).** Gagal berarti error yang tampil jelas di UI, bukan pengalihan jalur secara diam-diam.
2. **Zero Mock (Tanpa Data Fiktif).** Kode, label diagnostik telemetri, dan angka performa harus merefleksikan operasi komputasi riil.
3. **5 Opsi Engine Pass 2 Tetap Terdaftar.** Seluruh opsi engine (`webgpu`, `wasm-simd`, `direct-ort`, `wasm-ggml`, `webnn`) tetap dipertahankan pada antarmuka. Engine `webgpu`, `wasm-simd`, dan `direct-ort` (khusus `whisper-base`) berstatus aktif/READY. Engine yang belum memiliki implementasi lokal penuh (`wasm-ggml`, `webnn`) berstatus `(belum tersedia)` dan melempar exception eksplisit saat dipilih.
4. **Isolasi Modular Pipeline.** Pipeline audio dan UI (`app.js`, `audio-engine.js`, `transcript-view.js`) tidak boleh bergantung pada detail implementasi engine tertentu. Seluruh interaksi diarahkan melalui koordinator `Pass2Hub` berbasis kontrak `Pass2Engine`.

---

## Backlog / Item yang Belum Dikerjakan

### 1. Status Implementasi Adapter Pass 2 Engine

1. **`pass2-wasm-ggml.js` (Adapter & Toolchain Selesai — Pending Eksekusi Kompilasi Emscripten)**:
   - Seluruh kode adapter JavaScript (`pass2-wasm-ggml.js`), alokasi C-ABI heap WASM (`_whisper_init_from_buffer`, `_whisper_full`), caching `catetin-ggml-models-v1`, endpoint server `.bin` (`scripts/serve.js`), serta model kuantisasi lokal `ggml-base-id-q8_0.bin` (81,77 MB) & `ggml-small-id-q8_0.bin` (264,46 MB) telah selesai diimplementasikan dan terverifikasi secara independen (Gate 1–3 100% PASS).
   - Skrip otomatisasi kompilasi `scripts/build-whisper-wasm.ps1` telah siap; satu-satunya langkah manual yang tersisa hanyalah menjalankan `.\scripts\build-whisper-wasm.ps1` setelah `emsdk` dipasang di mesin untuk memproduksi biner runtime `web/vendor/whisper-cpp/whisper.wasm`.
2. **`pass2-webnn.js` (Backlog Masa Depan)**:
   - Disimpan di backlog masa depan sambil menunggu kematangan dan standarisasi stabil W3C WebNN API (`navigator.ml`) serta adopsi umum tanpa flag eksperimental di browser Chromium.

---

### 2. Eksperimen Optimasi Performa Masa Depan

#### A. Area 1: GPU Argmax & Decoupled Decode Loop (On Hold)
- **Kondisi Saat Ini**: Ditahan (*on hold*).
- **Rencana Riset**: Membangun custom WGSL compute shader untuk operasi `argmax` langsung di GPU guna mengeliminasi latensi transit IPC CPU ↔ GPU (Mojo IPC) sebesar 4–8 ms per token pada decode loop Whisper.

#### B. Migrasi Audio Pipeline ke `AudioWorklet`
- **Kondisi Saat Ini**: Menggunakan `ScriptProcessorNode` (deprecated).
- **Rencana Riset**: Memindahkan tangkapan mikrofon dan loopback audio tab ke node `AudioWorklet` di thread audio berprioritas tinggi untuk mengeliminasi potensi audio glitch saat main thread sibuk.
