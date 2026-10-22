const fs = require('fs');
const path = require('path');

const filePath = path.join(__dirname, '..', 'Rencana.md');
let content = fs.readFileSync(filePath, 'utf8');

const section6 = `
──────
6. Rencana Implementasi: Fitur Uji Kesiapan Hardware Interaktif di UI

Fitur ini menyediakan pengujian empiris cepat (≤ 1.5 detik) yang terikat langsung dengan Opsi Pass 2 dan Model yang sedang dipilih pengguna di modal Pengaturan (⚙️). Tujuannya membuktikan secara transparan kemampuan fisik hardware pengguna tanpa slogan atau asumsi teoritis.

### 6.1. Prinsip Desain UX
* **Terikat Kontekstual**: Hanya menguji hardware dan model yang sedang aktif dipilih di dropdown (zero bloat, tidak mengunduh model lain yang tidak dipakai).
* **Indikator Status Aktif**: Menampilkan visual loading/pulsing yang jelas di UI saat pengujian hardware sedang berjalan agar pengguna tahu sistem sedang menguji mesin mereka.
* **Hasil Fungsional & Padat**: Cukup 1 baris badge status tanpa teks panjang bertele-tele.

### 6.2. Alur Pengujian per Opsi Pass 2
1. **Opsi A (WebGPU)**:
   * Cek \`navigator.gpu\` dan alokasi buffer VRAM statis.
   * Jalankan micro-inference 0.5 detik audio dummy di GPU.
   * Ukur RTF: Jika RTF < 0.8x $\\to$ 🟢 **Mampu (RTF {rtf}x)**. Jika GPU context lost $\\to$ 🔴 **GPU OOM / Tidak Didukung**.
2. **Opsi B & E (CPU WASM SIMD / whisper.cpp)**:
   * Cek \`WebAssembly.validate()\` (SIMD) dan thread pool.
   * Jalankan micro-inference 0.5 detik audio dummy di CPU.
   * Ukur RTF: Jika memilih model Base dan RTF > 1.0x $\\to$ 🔴 **Hardware Terlalu Lambat (RTF {rtf}x). Rekomendasi: Gunakan Whisper Tiny**.
3. **Opsi C (WebNN DirectML)**:
   * Cek \`navigator.ml?.createContext()\`.
   * Jika flag belum aktif $\\to$ ⚠️ **Flag WebNN Belum Aktif di Browser**.
4. **Opsi D (Direct ORT Web)**:
   * Cek alokasi ONNX Runtime Web provider CPU/WASM.

### 6.3. Spesifikasi Komponen & State UI
* **Elemen UI di Modal Pengaturan (\`web/index.html\`)**:
  * Tombol: \`<button id="btn-test-hardware" class="btn-secondary">🔍 Uji Engine Ini</button>\`
  * Badge Status: \`<div id="hardware-test-status" class="hw-status-badge hidden"></div>\`
* **State Transisi UI**:
  1. **Idle**: Tombol aktif \`[ 🔍 Uji Engine Ini ]\`, badge tersembunyi.
  2. **Testing (Indikasi Hardware Sedang Diuji)**:
     * Tombol berubah teks menjadi \`[ ⏳ Menguji Hardware... ]\` dan status \`disabled\`.
     * Badge status menampilkan animasi pulse warna kuning:
       \`⏳ Menjalankan inferensi uji coba pada hardware Anda...\`
  3. **Done (Hasil Evaluasi Objektif)**:
     * Tombol kembali aktif.
     * Badge menampilkan hasil sesuai performa riil:
       * 🟢 **Mampu (RTF {rtf}x)** — Hardware siap dan responsif.
       * 🔴 **Terlalu Lambat (RTF {rtf}x)** — Hardware tidak kuat untuk model ini. Ubah ke Whisper Tiny.
       * ⚠️ **Tidak Didukung** — Fitur grafis/API belum aktif di peramban ini.

### 6.4. Berkas yang Diubah
* [web/src/engines/pass2-interface.js](file:///C:/Users/abdka/Development/foresight-analysis/web/src/engines/pass2-interface.js): Penambahan kontrak method \`testHardware(dummyPcm)\`.
* [web/src/engines/hub.js](file:///C:/Users/abdka/Development/foresight-analysis/web/src/engines/hub.js): Dispatcher pengujian hardware ke engine yang sedang dipilih.
* [web/index.html](file:///C:/Users/abdka/Development/foresight-analysis/web/index.html) & [web/style.css](file:///C:/Users/abdka/Development/foresight-analysis/web/style.css): Tombol uji, badge status, dan keyframe animasi pulse testing.
* [web/src/app.js](file:///C:/Users/abdka/Development/foresight-analysis/web/src/app.js): Event listener tombol \`#btn-test-hardware\`, timer pengukuran, dan rendering state badge.
`;

fs.writeFileSync(filePath, content.trimEnd() + '\n' + section6, 'utf8');
console.log('Successfully appended Section 6 to Rencana.md!');
