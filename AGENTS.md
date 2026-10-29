# CatatIn Project Agent Rules & Workflow (`AGENTS.md`)

Aturan ini bersifat **WAJIB (`always_on`)** untuk setiap agent dan subagent yang bekerja di repositori `CatatIn`.

## 1. Filosofi Kode (`ponytail` + Zero-Mock)
1. **Zero Mock / Zero Fake Data**:
   - Dilarang keras membuat implementasi mock, dummy return, progress bar palsu, atau angka telemetri tiruan (`Math.random()`, `setTimeout` palsu, dsb.) agar fitur terlihat "jadi".
   - Jika suatu engine atau fitur belum diimplementasikan (`direct-ort`, `wasm-ggml`, `webnn`), tandai `this.available = false` dan lempar `new Error('... belum tersedia.')` secara jujur dan eksplisit.
2. **Zero Silent Fallback**:
   - Dilarang keras melakukan fallback diam-diam ke engine/metode lain saat inisialisasi, warmup, atau transkripsi gagal, serta dilarang membuat blok `catch {}` kosong yang menelan error.
3. **Ponytail (Smallest Complete Change)**:
   - Tulis perubahan terkecil yang menyelesaikan akar masalah secara tuntas tanpa abstraksi berlapis, dead code, atau spekulasi fitur masa depan.

## 2. Protokol Delegasi Multi-Subagent (Anti-Context-Poisoning & Anti-Halusinasi)
Untuk menjaga context window main agent tetap ringan dan mencegah *context poisoning* / halusinasi pada subagent:
1. **Granularitas Sempit & Non-Overlapping**:
   - Saat mendelegasikan tugas ke subagent (`invoke_subagent`), pecah pekerjaan menjadi unit-unit kecil yang **spesifik pada 1–4 berkas saja** dan **tidak saling tumpang-tindih (non-overlapping file scope)** antar subagent paralel.
   - Cantumkan daftar berkas yang boleh diubah (`RUANG LINGKUP BERKAS`) dan larang menyentuh berkas di luar daftar tersebut.
2. **Instruksi Baris & Spesifikasi Eksplisit (Zero-Ambiguity Prompting)**:
   - Jangan memberi instruksi kabur kepada subagent. Berikan lokasi fungsi/baris target dan spesifikasi logika/kontrak yang eksak.
   - Gunakan `Model: 'flash'` untuk tugas implementasi dan verifikasi yang sudah terdefinisi secara presisi.
3. **Pemisahan Ketat: Implementor ≠ Verifier (Zero-Trust Verification)**:
   - **Dilarang keras** menggunakan subagent yang sama untuk memverifikasi kode yang baru saja ditulisnya sendiri.
   - Setiap batch perubahan **WAJIB** diverifikasi oleh subagent verifikator independen (atau skrip pengujian whitebox Node.js terpisah) dengan cakupan verifikasi yang kecil, spesifik, dan menguji modul asli (bukan mock test) ditambah pemeriksaan `git diff` bahwa tidak ada jalan pintas/mock.

## 3. Tata Kelola Dokumentasi Proyek
1. **`CHANGELOG.md` Sebagai Satu-Satunya Arsip Selesai**:
   - Semua fitur, bugfix, dan fase refactoring yang **SUDAH SELESAI & TERVERIFIKASI** wajib dicatat di [`CHANGELOG.md`](./CHANGELOG.md).
2. **`Refactoring.md` & `Rencana.md` Tetap Ramping**:
   - Hapus rincian item yang sudah selesai dari [`Refactoring.md`](./Refactoring.md) dan [`Rencana.md`](./Rencana.md) segera setelah terverifikasi dan dipindahkan ke `CHANGELOG.md`. Kedua dokumen tersebut hanya boleh menyimpan prinsip arsitektur aktif dan backlog yang belum dikerjakan.
