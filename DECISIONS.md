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
  - Bebas flow berbelit/wizard: Tampilan 1-layar langsung siap kerja (Header kontrol minimalis, 65% feed transkrip, 35% catatan topik & checklist).
- **Tipografi Bersih (Anti AI-Slop):**
  - Menggunakan font sistem (`system-ui`) dengan kontras tinggi. Hapus seluruh esai/paragraf pengantar panjang; label tombol maksimal 1-2 kata.
- **Penyimpanan Sandbox OPFS:**
  - Model ONNX disimpan di **Origin Private File System (OPFS)** browser untuk streaming zero-copy access handle tanpa overhead de-serialisasi database.
- **Dua Sumber Suara:**
  - Mikrofon lokal (`getUserMedia`) dilabeli sebagai `[Anda]`.
  - Audio tab rapat (`getDisplayMedia`) dilabeli sebagai `[Pembicara 1]`, `[2]`, dst.
- **Ekspor Dokumen:**
  - Tombol simpan langsung menghasilkan unduhan `notulensi.md` yang 100% kompatibel dengan format catatan CLI Catetin.
