// app.js - Main Application Coordinator (Ponytail: clean, zero mocks, zero cloud API)

import { AudioEngine } from './audio-engine.js';
import { OnnxRunner } from './onnx-runner.js';
import './components/vu-meter.js';
import './components/transcript-view.js';
import './components/notes-view.js';
import './components/catetin-telemetry.js';

class CatetinApp {
  constructor() {
    this.audio = new AudioEngine();
    this.onnx = new OnnxRunner();

    this.isRecording = false;
    this.isPaused = false;
    this.micWanted = true;

    // System Lifecycle State
    this.state = 'BOOTSTRAPPING'; // BOOTSTRAPPING | WARMING_PASS1 | WARMING_PASS2 | SYSTEM_READY | ACTIVE_RECORDING

    // Load persisted settings
    this.settings = this.loadSettings();

    this.initDOM();
    this.initAudioHooks();
    this.initHotkeys();
    this.applySettings();

    // Jalankan bootstrap eager pre-warming
    this.bootstrapSystem();
  }

  loadSettings() {
    try {
      const saved = localStorage.getItem('catetin_settings');
      if (saved) {
        return JSON.parse(saved);
      }
    } catch (_) {}

    return {
      language: 'id',
      whisperModel: 'Xenova/whisper-base',
      transducerModel: 'id',
      customModel: '',
      hideNotes: false
    };
  }

  saveSettings() {
    try {
      localStorage.setItem('catetin_settings', JSON.stringify(this.settings));
    } catch (_) {}
  }

  initDOM() {
    this.btnSession = document.getElementById('btn-session');
    this.btnMic = document.getElementById('btn-mic');
    this.btnTab = document.getElementById('btn-tab');
    this.btnPause = document.getElementById('btn-pause');
    this.btnSave = document.getElementById('btn-save');
    this.btnClear = document.getElementById('btn-clear');
    this.btnSettings = document.getElementById('btn-settings');
    this.statusVad = document.getElementById('status-vad');
    this.bootstrapStatus = document.getElementById('bootstrap-status');

    this.mainLayout = document.getElementById('main-layout');
    this.vuMeter = document.querySelector('vu-meter');
    this.transcriptView = document.querySelector('transcript-view');
    this.notesView = document.querySelector('notes-view');
    this.telemetry = document.querySelector('catetin-telemetry');

    // Modal elements
    this.modalSettings = document.getElementById('modal-settings');
    this.btnCloseModal = document.getElementById('btn-close-modal');
    this.btnSaveSettings = document.getElementById('btn-save-settings');
    this.selectPass2Engine = document.getElementById('setting-pass2-engine');
    this.selectLanguage = document.getElementById('setting-language');
    this.selectWhisper = document.getElementById('setting-whisper');
    this.selectTransducer = document.getElementById('setting-transducer');
    this.groupCustomModel = document.getElementById('group-custom-model');
    this.inputCustomModel = document.getElementById('input-custom-model');
    this.checkboxHideNotes = document.getElementById('setting-hide-notes');
    this.modelStatusText = document.getElementById('model-status-text');

    // Event Listeners
    this.btnSession.addEventListener('click', () => this.toggleSession());
    this.btnMic.addEventListener('click', () => this.toggleMic());
    this.btnTab.addEventListener('click', () => this.addTabAudio());
    this.btnPause.addEventListener('click', () => this.togglePause());
    this.btnSave.addEventListener('click', () => this.saveMarkdown());
    this.btnClear.addEventListener('click', () => this.clearAll());
    this.btnSettings.addEventListener('click', () => this.openSettings());
    this.btnCloseModal.addEventListener('click', () => this.closeSettings());
    this.btnSaveSettings.addEventListener('click', () => this.saveSettingsFromModal());

    this.selectWhisper.addEventListener('change', () => {
      this.groupCustomModel.style.display =
        this.selectWhisper.value === 'custom' ? 'flex' : 'none';
    });

    this.checkboxHideNotes.addEventListener('change', () => {
      this.setNotesHidden(this.checkboxHideNotes.checked);
    });

    // Close modal on backdrop click
    this.modalSettings.addEventListener('click', (e) => {
      if (e.target === this.modalSettings) {
        this.closeSettings();
      }
    });
  }

  applySettings() {
    this.selectPass2Engine.value = this.settings.pass2Engine || 'webgpu';
    this.selectLanguage.value = this.settings.language || 'id';

    const isKnownWhisper = ['Xenova/whisper-tiny', 'Xenova/whisper-base', 'Xenova/whisper-small', 'none'].includes(this.settings.whisperModel);
    if (isKnownWhisper) {
      this.selectWhisper.value = this.settings.whisperModel;
      this.groupCustomModel.style.display = 'none';
    } else {
      this.selectWhisper.value = 'custom';
      this.inputCustomModel.value = this.settings.whisperModel || '';
      this.groupCustomModel.style.display = 'flex';
    }

    this.selectTransducer.value = this.settings.transducerModel || 'id';
    this.checkboxHideNotes.checked = !!this.settings.hideNotes;
    this.setNotesHidden(this.checkboxHideNotes.checked);

    // Apply to ASR Worker
    const activeModel = this.selectWhisper.value === 'custom'
      ? (this.inputCustomModel.value.trim() || 'Xenova/whisper-base')
      : this.selectWhisper.value;

    this.onnx.updateConfig({
      pass2Engine: this.selectPass2Engine.value,
      whisperModel: activeModel,
      language: this.settings.language,
      transducerModel: this.selectTransducer.value
    });
  }

  setBootstrapState(state, label) {
    this.state = state;
    if (this.bootstrapStatus) {
      this.bootstrapStatus.textContent = label;
      this.bootstrapStatus.className = 'bootstrap-badge ' + (state === 'SYSTEM_READY' ? 'ready' : 'warming');
    }
    if (this.telemetry) {
      this.telemetry.log('SYS', `[State] ${state}: ${label}`);
    }

    if (state === 'SYSTEM_READY') {
      this.btnSession.disabled = false;
      this.btnSession.textContent = this.isRecording ? 'Berhenti' : 'Mulai';
    } else if (!this.isRecording) {
      this.btnSession.disabled = true;
      this.btnSession.textContent = label;
    }
  }

  async bootstrapSystem() {
    this.setBootstrapState('BOOTSTRAPPING', 'Menyiapkan modul...');

    try {
      this.setBootstrapState('WARMING_PASS1', 'Memuat DSP & Transducer...');
      await this.audio.initWasm('./catetin.wasm');

      const pass1Promise = new Promise((resolve) => {
        if (this.onnx.transducerReady || this.settings.transducerModel === 'none') {
          resolve();
        } else {
          const checkP1 = setInterval(() => {
            if (this.onnx.transducerReady || this.settings.transducerModel === 'none') {
              clearInterval(checkP1);
              resolve();
            }
          }, 50);
        }
      });

      const pass2Promise = new Promise((resolve) => {
        if (this.onnx.isReady || this.settings.whisperModel === 'none') {
          resolve();
        } else {
          const checkP2 = setInterval(() => {
            if (this.onnx.isReady || this.settings.whisperModel === 'none') {
              clearInterval(checkP2);
              resolve();
            }
          }, 50);
        }
      });

      this.setBootstrapState('WARMING_PASS2', 'Pre-warming shader GPU...');
      await Promise.all([pass1Promise, pass2Promise]);

      this.setBootstrapState('SYSTEM_READY', 'Siap (0ms)');
    } catch (err) {
      console.error('[Bootstrap Error]', err);
      this.setBootstrapState('ERROR', 'Gagal inisialisasi AI');
    }
  }

  setNotesHidden(hidden) {
    if (hidden) {
      this.mainLayout.classList.add('notes-hidden');
    } else {
      this.mainLayout.classList.remove('notes-hidden');
      if (this.notesView && this.transcriptView) {
        this.notesView.updateInsights(this.transcriptView.getSegments());
      }
    }
  }

  openSettings() {
    this.applySettings();
    this.modalSettings.showModal();
  }

  closeSettings() {
    this.modalSettings.close();
  }

  saveSettingsFromModal() {
    const selectedWhisper = this.selectWhisper.value;
    const finalWhisperModel = selectedWhisper === 'custom'
      ? (this.inputCustomModel.value.trim() || 'Xenova/whisper-base')
      : selectedWhisper;

    this.settings = {
      pass2Engine: this.selectPass2Engine.value,
      language: this.selectLanguage.value,
      whisperModel: finalWhisperModel,
      transducerModel: this.selectTransducer.value,
      customModel: this.inputCustomModel.value.trim(),
      hideNotes: this.checkboxHideNotes.checked
    };

    this.saveSettings();
    this.applySettings();
    this.closeSettings();
  }

  initAudioHooks() {
    // Diagnostic logging & Telemetry bridges
    this.audio.onLog = (cat, msg) => {
      if (this.telemetry) this.telemetry.log(cat, msg);
    };

    this.onnx.onLog = (cat, msg) => {
      if (this.telemetry) this.telemetry.log(cat, msg);
    };

    this.onnx.onTelemetry = (info) => {
      if (this.telemetry) this.telemetry.updateWhisper(info);
    };

    this.audio.onVolume = (level) => {
      if (this.vuMeter) this.vuMeter.setLevel(level);
    };

    this.audio.onVadSpeaking = (speaking, rms = 0) => {
      if (this.statusVad) {
        if (speaking) {
          this.statusVad.classList.add('active');
        } else {
          this.statusVad.classList.remove('active');
        }
      }
      if (this.telemetry) {
        this.telemetry.updateVad(speaking, rms);
      }
    };

    // Real-time live turn indicator in transcript feed
    this.audio.onSpeechProgress = ({ speaker, durationS, rms, closed }) => {
      if (closed) {
        this.transcriptView.clearLiveTurn();
      } else if (!this.transcriptView.hasLiveText) {
        this.transcriptView.setLiveTurn(speaker, durationS);
      }
      if (this.telemetry) {
        this.telemetry.updateVad(!closed && durationS > 0, rms);
      }
    };

    // Pass 1: Streaming Transducer chunks (< 50ms live captions)
    this.audio.onTransducerChunk = (pcm16k, speaker, durationS) => {
      this.onnx.feedTransducerChunk(pcm16k, speaker, durationS);
    };

    this.audio.onTurnFinished = (speaker) => {
      this.onnx.resetTransducer();
    };

    // Pass 1 Fallback: Streaming interim speech fed to fast decoder if Transducer is disabled
    this.audio.onInterimSpeech = (pcm16k, speaker, durationS) => {
      this.onnx.transcribeInterim(pcm16k, speaker, durationS);
    };

    this.onnx.onInterimResult = ({ speaker, text, durationS, decodeTimeMs }) => {
      if (text && text.trim()) {
        const clean = text.replace(/\[.*?\]/g, '').trim();
        if (clean) {
          this.transcriptView.setLiveTurn(speaker, durationS, clean);
          if (this.telemetry) {
            const ms = typeof decodeTimeMs === 'number' ? ` (${decodeTimeMs.toFixed(1)}ms)` : '';
            this.telemetry.log('TRANSDUCER', `Live [${speaker}]: "${clean}"${ms}`);
          }
        }
      }
    };

    this.audio.onTabsChanged = (count) => {
      if (count > 0) {
        this.btnTab.textContent = `Tab Audio (${count})`;
        this.btnTab.classList.add('active');
      } else {
        this.btnTab.textContent = 'Tab Audio';
        this.btnTab.classList.remove('active');
      }

      if (this.telemetry) {
        this.telemetry.updateAudio(this.audio.micEnabled, Array.from(this.audio.tabStreams.values()));
      }
    };

    // Offloaded to Web Worker: never blocks UI or audio!
    this.audio.onUtteranceReady = async (utterance, speakerLabel) => {
      const timeStr = this._formatTime(new Date());
      const durationS = utterance.length / 16000;

      // Provide immediate visual feedback that final pass is running
      this.transcriptView.setLiveTurn(speakerLabel, durationS, 'Memproses transkrip final...');

      const rawText = await this.onnx.transcribeUtterance(utterance, speakerLabel, timeStr);
      if (rawText && rawText.trim()) {
        const cleanedText = this.audio.cleanRepetitiveLoops(rawText);
        if (this.audio.isHallucination(cleanedText)) {
          if (this.telemetry) this.telemetry.log('WHISPER', `Filter halusinasi mengabaikan: "${rawText.slice(0, 30)}..."`);
          this.transcriptView.clearLiveTurn();
          return;
        }
        const cleanText = this.audio.formatUtterance(cleanedText) || cleanedText.trim();
        if (cleanText) {
          this.transcriptView.clearLiveTurn();
          this.transcriptView.addSegment(speakerLabel, cleanText, timeStr);
          // Hanya proses update catatan jika panel catatan TIDAK disembunyikan
          if (!this.settings.hideNotes && this.notesView) {
            this.notesView.updateInsights(this.transcriptView.getSegments());
          }
          if (this.telemetry) {
            this.telemetry.log('STREAM', `Transkrip dirender [${speakerLabel}]: "${cleanText.slice(0, 50)}..."`);
          }
        } else {
          this.transcriptView.clearLiveTurn();
        }
      } else {
        this.transcriptView.clearLiveTurn();
      }
    };

    // Listen to ASR Worker progress/status
    this.onnx.onProgress = (info) => {
      if (this.modelStatusText) {
        this.modelStatusText.className = 'status-info loading';
        this.modelStatusText.textContent = `Mengunduh ${info.file || 'model'}: ${info.progress}%`;
      }
    };

    this.onnx.onStatusChange = (status, info) => {
      if (this.modelStatusText) {
        if (status === 'ready') {
          this.modelStatusText.className = 'status-info ready';
          this.modelStatusText.textContent = `Model aktif (${info.device?.toUpperCase() || 'WASM'}).`;
        } else if (status === 'loading') {
          this.modelStatusText.className = 'status-info loading';
          this.modelStatusText.textContent = 'Memuat model...';
        } else if (status === 'error') {
          this.modelStatusText.className = 'status-info';
          this.modelStatusText.textContent = `Error: ${info.error}`;
        }
      }
    };
  }

  initHotkeys() {
    window.addEventListener('keydown', (e) => {
      if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA' || e.target.tagName === 'SELECT') return;

      if (e.code === 'Space' || e.key === 'p' || e.key === 'P') {
        e.preventDefault();
        if (this.isRecording) this.togglePause();
      } else if (e.key === 'c' || e.key === 'C') {
        e.preventDefault();
        this.saveMarkdown();
      } else if (e.key === 'm' || e.key === 'M') {
        e.preventDefault();
        this.toggleMic();
      }
    });
  }

  async toggleSession() {
    if (!this.isRecording && this.state !== 'SYSTEM_READY') {
      console.warn("Sistem belum siap, masih dalam tahap bootstrap:", this.state);
      return;
    }

    if (this.isRecording) {
      this.audio.stopSession();
      this.isRecording = false;
      this.isPaused = false;
      this.btnSession.textContent = 'Mulai';
      this.btnSession.classList.remove('active');
      this.btnPause.disabled = true;
      this.transcriptView.clearLiveTurn();
      if (this.vuMeter) this.vuMeter.reset();
      if (this.telemetry) {
        this.telemetry.updateAudio(false, []);
        this.telemetry.updateVad(false, 0);
      }
      return;
    }

    try {
      await this.audio.startSession();

      // Start mic if desired
      if (this.micWanted) {
        try {
          await this.audio.setMicEnabled(true);
          this.btnMic.textContent = 'Mic: Aktif';
          this.btnMic.classList.remove('muted');
        } catch (micErr) {
          console.warn("Could not start microphone on session start:", micErr);
          this.btnMic.textContent = 'Mic: Mati';
          this.btnMic.classList.add('muted');
        }
      }

      this.isRecording = true;
      this.isPaused = false;
      this.btnSession.textContent = 'Stop';
      this.btnSession.classList.add('active');
      this.btnPause.disabled = false;

      if (this.telemetry) {
        this.telemetry.updateAudio(this.audio.micEnabled, Array.from(this.audio.tabStreams.values()));
      }
    } catch (err) {
      alert("Gagal memulai sesi audio: " + err.message);
    }
  }

  async toggleMic() {
    this.micWanted = !this.micWanted;

    if (this.isRecording) {
      try {
        await this.audio.setMicEnabled(this.micWanted);
      } catch (err) {
        alert("Gagal mengubah mikrofon: " + err.message);
        this.micWanted = false;
      }
    }

    if (this.micWanted) {
      this.btnMic.textContent = 'Mic: Aktif';
      this.btnMic.classList.remove('muted');
    } else {
      this.btnMic.textContent = 'Mic: Mati';
      this.btnMic.classList.add('muted');
    }

    if (this.telemetry) {
      this.telemetry.updateAudio(this.audio.micEnabled, Array.from(this.audio.tabStreams.values()));
    }
  }

  async addTabAudio() {
    try {
      const label = await this.audio.addTabAudio();
      console.log(`[App] Tab audio added: ${label}`);
      if (this.telemetry) {
        this.telemetry.updateAudio(this.audio.micEnabled, Array.from(this.audio.tabStreams.values()));
      }
    } catch (err) {
      if (err.name !== 'AbortError' && err.name !== 'NotAllowedError') {
        console.warn("Tab audio sharing canceled or failed:", err);
      }
    }
  }

  togglePause() {
    if (!this.isRecording) return;
    this.isPaused = !this.isPaused;
    if (this.isPaused) {
      this.audio.pause();
      this.btnPause.textContent = 'Lanjut';
      this.btnPause.classList.add('active');
      this.transcriptView.clearLiveTurn();
    } else {
      this.audio.resume();
      this.btnPause.textContent = 'Jeda';
      this.btnPause.classList.remove('active');
    }
  }

  saveMarkdown() {
    const segments = this.transcriptView.getSegments();
    if (segments.length === 0) {
      alert("Belum ada transkrip.");
      return;
    }

    const now = new Date();
    const ts = Math.floor(now.getTime() / 1000);
    const dateStr = now.toLocaleDateString('id-ID', { dateStyle: 'full' });

    let md = `# Catetin - Notulensi Rapat\n\n`;
    md += `*Tanggal: ${dateStr}*\n\n`;
    md += `## Ringkasan Eksekutif\n\n`;

    const summaryText = this.notesView.shadowRoot.getElementById('summary-text')?.textContent || '';
    md += summaryText ? `${summaryText}\n\n` : `*(Belum ada ringkasan)*\n\n`;

    md += `## Poin Keputusan & Tindak Lanjut\n\n`;
    const actionItems = Array.from(this.notesView.shadowRoot.querySelectorAll('.action-item')).map(el => el.textContent);
    if (actionItems.length > 0) {
      actionItems.forEach(item => { md += `- [ ] ${item}\n`; });
      md += '\n';
    } else {
      md += `*(Tidak ada poin tindakan khusus)*\n\n`;
    }

    md += `## Transkrip Lengkap\n\n`;
    segments.forEach(s => {
      md += `**${s.time} [${s.speaker}]**: ${s.text}\n\n`;
    });

    const blob = new Blob([md], { type: 'text/markdown;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `catetin-notulensi-${ts}.md`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }

  clearAll() {
    if (this.transcriptView.getSegments().length > 0) {
      if (!confirm("Hapus seluruh transkrip dan catatan rapat?")) return;
    }
    this.transcriptView.clear();
    this.notesView.clear();
    if (this.telemetry) {
      this.telemetry.clearLogs();
    }
  }

  _formatTime(date) {
    const pad = (n) => String(n).padStart(2, '0');
    return `[${pad(date.getHours())}:${pad(date.getMinutes())}:${pad(date.getSeconds())}]`;
  }
}

// Instantiate PWA
window.addEventListener('DOMContentLoaded', () => {
  window.app = new CatetinApp();
});
