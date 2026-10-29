// app.js - Main Application Coordinator

import { AudioEngine } from './audio-engine.js';
import { OnnxRunner } from './onnx-runner.js';
import { Pass2Hub } from './engines/hub.js';
import { READINESS_SCHEMA_VERSION } from './engines/pass2-interface.js';
import './components/vu-meter.js';
import './components/transcript-view.js';
import './components/notes-view.js';
import './components/catetin-telemetry.js';

const KNOWN_WHISPER_MODELS = [
  'Xenova/whisper-tiny',
  'Xenova/whisper-base',
  'Xenova/whisper-small',
  'none'
];

class CatetinApp {
  constructor() {
    this.audio = new AudioEngine();
    this.onnx = new OnnxRunner();

    this.isRecording = false;
    this.isPaused = false;
    this.micWanted = true;
    this.isOnboardingRunning = false;

    // System Lifecycle State: BOOTSTRAPPING | WARMING_PASS1 | WARMING_PASS2 | SYSTEM_READY | ERROR
    this.state = 'BOOTSTRAPPING';

    this.settings = this.loadSettings();

    this.initDOM();
    this.initAudioHooks();
    this.initHotkeys();
    this.applySettings();

    if (this.needsHardwareTest()) {
      this.runHardwareOnboarding();
    } else {
      this.bootstrapSystem();
    }
  }

  loadSettings() {
    try {
      const saved = localStorage.getItem('catetin_settings');
      if (saved) {
        return JSON.parse(saved);
      }
    } catch (err) {
      console.error('[App] Gagal membaca pengaturan dari localStorage:', err);
    }

    return {
      pass2Engine: 'webgpu',
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
    } catch (err) {
      console.error('[App] Gagal menyimpan pengaturan ke localStorage:', err);
    }
  }

  needsHardwareTest() {
    try {
      const raw = localStorage.getItem('catetin_hw_profile');
      if (!raw) return true;
      const profile = JSON.parse(raw);
      if (profile.schemaVersion !== READINESS_SCHEMA_VERSION) return true;
      if (profile.userAgent !== navigator.userAgent) return true;
      return false;
    } catch (_) {
      return true;
    }
  }

  async runHardwareOnboarding() {
    if (this.isOnboardingRunning) return;
    this.isOnboardingRunning = true;

    if (this.isRecording) {
      this.toggleSession();
    }
    if (!this.modalOnboarding) {
      this.isOnboardingRunning = false;
      return this.bootstrapSystem();
    }

    this.onboardingResultsList.innerHTML = '';
    this.onboardingRecommendation.style.display = 'none';
    this.onboardingWarning.style.display = 'none';
    this.onboardingWarning.classList.add('hidden');
    this.onboardingProgress.style.display = 'flex';
    const pulseDot = this.onboardingProgress?.querySelector('.pulse-dot');
    if (pulseDot) pulseDot.classList.remove('hidden');
    this.onboardingStepText.textContent = 'Menyiapkan pengujian hardware...';
    if (this.onboardingStatusText) {
      this.onboardingStatusText.textContent = 'Menyiapkan pengujian hardware...';
    }
    this.btnOnboardingStart.disabled = true;
    this.btnOnboardingManual.disabled = true;

    if (!this.modalOnboarding.open) {
      this.modalOnboarding.showModal();
    }

    try {
      const hub = new Pass2Hub();
      const { results, recommendation } = await hub.runReadinessTests(({ engineName, model }) => {
        const shortModel = model.replace('Xenova/', '');
        const progressMsg = `Menguji ${engineName} (${shortModel})...`;
        this.onboardingStepText.textContent = progressMsg;
        if (this.onboardingStatusText) {
          this.onboardingStatusText.textContent = progressMsg;
        }
      });

      this.onboardingProgress.style.display = 'none';

      for (const r of results) {
        const row = document.createElement('div');
        row.className = 'hw-row';
        const shortModel = r.model.replace('Xenova/', '');
        const badge = !r.supported ? '🔴' : (r.rtf < 0.8 ? '🟢' : '⚠️');
        const metric = r.supported ? `RTF ${r.rtf.toFixed(2)}x (${r.decodeMs}ms)` : (r.error || 'Gagal');
        const left = document.createElement('span');
        left.textContent = `${badge} ${r.engineId.toUpperCase()} · ${shortModel}`;
        const right = document.createElement('span');
        right.textContent = metric;
        row.appendChild(left);
        row.appendChild(right);
        this.onboardingResultsList.appendChild(row);
      }

      const recShortModel = recommendation.whisperModel.replace('Xenova/', '');
      const recRtfText = typeof recommendation.rtf === 'number' ? ` (RTF ${recommendation.rtf.toFixed(2)}x)` : '';
      this.onboardingRecommendation.textContent = `Rekomendasi: ${recommendation.pass2Engine.toUpperCase()} + ${recShortModel}${recRtfText}`;
      this.onboardingRecommendation.style.display = 'block';

      if (recommendation.slowHardwareWarning) {
        this.onboardingWarning.textContent = 'Hardware terlalu lambat untuk Whisper Base; menggunakan Whisper Tiny.';
        this.onboardingWarning.style.display = 'block';
        this.onboardingWarning.classList.remove('hidden');
      }

      this.settings.pass2Engine = recommendation.pass2Engine;
      this.settings.whisperModel = recommendation.whisperModel;
      this.saveSettings();

      try {
        localStorage.setItem('catetin_hw_profile', JSON.stringify({
          schemaVersion: READINESS_SCHEMA_VERSION,
          userAgent: navigator.userAgent,
          timestamp: Date.now(),
          recommendation,
          results
        }));
      } catch (err) {
        console.error('[App] Gagal menyimpan profil hardware:', err);
      }

      this.btnOnboardingStart.disabled = false;
    } catch (err) {
      console.error('[App] Pengujian hardware gagal:', err);
      if (this.onboardingWarning) {
        this.onboardingWarning.textContent = 'Pengujian perangkat keras gagal: ' + (err.message || String(err));
        this.onboardingWarning.classList.remove('hidden');
        this.onboardingWarning.style.display = 'block';
      }
      if (this.onboardingStatusText) {
        this.onboardingStatusText.textContent = 'Pengujian dihentikan karena kesalahan.';
      }
    } finally {
      this.isOnboardingRunning = false;
      const pulse = this.onboardingProgress?.querySelector('.pulse-dot');
      if (pulse) {
        pulse.classList.add('hidden');
      }
      if (this.btnOnboardingManual) {
        this.btnOnboardingManual.disabled = false;
      }
    }
  }

  _resolveSelectedWhisperModel() {
    if (this.selectWhisper.value === 'custom') {
      const custom = this.inputCustomModel.value.trim();
      if (!custom) {
        throw new Error('ID Model Hugging Face kustom tidak boleh kosong.');
      }
      return custom;
    }
    return this.selectWhisper.value;
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

    // Onboarding elements
    this.btnRetestHw = document.getElementById('btn-retest-hw');
    this.modalOnboarding = document.getElementById('modal-onboarding');
    this.onboardingProgress = document.getElementById('onboarding-progress');
    this.onboardingStepText = document.getElementById('onboarding-step-text');
    this.onboardingStatusText = this.onboardingStepText;
    this.onboardingResultsList = document.getElementById('onboarding-results-list');
    this.onboardingRecommendation = document.getElementById('onboarding-recommendation');
    this.onboardingWarning = document.getElementById('onboarding-warning');
    this.btnOnboardingManual = document.getElementById('btn-onboarding-manual');
    this.btnOnboardingStart = document.getElementById('btn-onboarding-start');

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

    if (this.btnRetestHw) {
      this.btnRetestHw.addEventListener('click', () => {
        if (this.state === 'RECORDING' || this.state === 'INITIALIZING' || this.state === 'STOPPING' || this.isOnboardingRunning) return;
        this.closeSettings();
        this.runHardwareOnboarding();
      });
    }
    if (this.modalOnboarding) {
      this.modalOnboarding.addEventListener('cancel', (e) => {
        e.preventDefault();
      });
    }
    if (this.btnOnboardingStart) {
      this.btnOnboardingStart.addEventListener('click', () => {
        this.modalOnboarding.close();
        this.applySettings();
        this.bootstrapSystem();
      });
    }
    if (this.btnOnboardingManual) {
      this.btnOnboardingManual.addEventListener('click', () => {
        this.modalOnboarding.close();
        this.openSettings();
      });
    }

    this.selectWhisper.addEventListener('change', () => {
      this.groupCustomModel.style.display =
        this.selectWhisper.value === 'custom' ? 'flex' : 'none';
    });

    this.checkboxHideNotes.addEventListener('change', () => {
      this.setNotesHidden(this.checkboxHideNotes.checked);
    });

    this.modalSettings.addEventListener('click', (e) => {
      if (e.target === this.modalSettings) {
        this.closeSettings();
      }
    });
  }

  applySettings() {
    this.selectPass2Engine.value = this.settings.pass2Engine || 'webgpu';
    this.selectLanguage.value = this.settings.language || 'id';

    if (KNOWN_WHISPER_MODELS.includes(this.settings.whisperModel)) {
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

    const activeModel = this._resolveSelectedWhisperModel();

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
    this.telemetry.log(state === 'ERROR' ? 'ERROR' : 'SYS', `[State] ${state}: ${label}`);

    if (state === 'SYSTEM_READY') {
      this.btnSession.disabled = false;
      this.btnSession.textContent = this.isRecording ? 'Stop' : 'Mulai';
    } else if (!this.isRecording) {
      this.btnSession.disabled = true;
      this.btnSession.textContent = state === 'ERROR' ? 'Error AI' : label;
    }
  }

  async bootstrapSystem() {
    this.setBootstrapState('BOOTSTRAPPING', 'Menyiapkan modul...');

    try {
      this.setBootstrapState('WARMING_PASS1', 'Memuat DSP & Transducer...');
      await Promise.all([
        this.audio.initWasm('./catetin.wasm'),
        this.onnx.whenPass1Ready()
      ]);

      const pass2Label = this.selectPass2Engine.value === 'webgpu'
        ? 'Pre-warming shader GPU...'
        : 'Menyiapkan engine Pass 2...';
      this.setBootstrapState('WARMING_PASS2', pass2Label);
      await this.onnx.whenPass2Ready();

      this.setBootstrapState('SYSTEM_READY', 'Siap (0ms)');
    } catch (err) {
      console.error('[Bootstrap Error]', err);
      this.setBootstrapState('ERROR', err.message || 'Gagal inisialisasi AI');
    }
  }

  setNotesHidden(hidden) {
    if (hidden) {
      this.mainLayout.classList.add('notes-hidden');
    } else {
      this.mainLayout.classList.remove('notes-hidden');
      this.notesView.updateInsights(this.transcriptView.getSegments());
    }
  }

  openSettings() {
    this.selectPass2Engine.value = this.settings.pass2Engine || 'webgpu';
    this.selectLanguage.value = this.settings.language || 'id';
    this.selectTransducer.value = this.settings.transducerModel || 'id';
    this.checkboxHideNotes.checked = !!this.settings.hideNotes;
    this.modalSettings.showModal();
  }

  closeSettings() {
    this.modalSettings.close();
    if (this.state === 'BOOTSTRAPPING' && !this.isOnboardingRunning) {
      this.bootstrapSystem();
    }
  }

  saveSettingsFromModal() {
    let finalWhisperModel;
    try {
      finalWhisperModel = this._resolveSelectedWhisperModel();
    } catch (err) {
      alert(err.message);
      return;
    }

    if (this.isRecording) {
      this.toggleSession();
    }

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
    this.modalSettings.close();
    if (this.state === 'BOOTSTRAPPING') {
      this.bootstrapSystem();
    }
  }

  initAudioHooks() {
    this.audio.onLog = (cat, msg) => {
      this.telemetry.log(cat, msg);
    };

    this.onnx.onLog = (cat, msg) => {
      this.telemetry.log(cat, msg);
    };

    this.onnx.onTelemetry = (info) => {
      this.telemetry.updateWhisper(info);
    };

    this.onnx.onTransducerStatus = (status, model) => {
      this.telemetry.updateTransducer(status, model);
    };

    this.audio.onVolume = (level) => {
      this.vuMeter.setLevel(level);
    };

    this.audio.onVadSpeaking = (speaking, rms = 0) => {
      if (speaking) {
        this.statusVad.classList.add('active');
      } else {
        this.statusVad.classList.remove('active');
      }
      this.telemetry.updateVad(speaking, rms);
    };

    this.audio.onSpeechProgress = ({ speaker, durationS, rms, closed }) => {
      if (closed) {
        this.transcriptView.clearLiveTurn();
      } else if (!this.transcriptView.hasLiveText) {
        this.transcriptView.setLiveTurn(speaker, durationS);
      }
      this.telemetry.updateVad(!closed && durationS > 0, rms);
    };

    // Pass 1: Streaming Transducer chunks (< 50ms live captions)
    this.audio.onTransducerChunk = (pcm16k, speaker, durationS) => {
      this.onnx.feedTransducerChunk(pcm16k, speaker, durationS);
    };

    this.audio.onTurnFinished = () => {
      this.onnx.resetTransducer();
    };

    this.onnx.onInterimResult = ({ speaker, text, durationS, decodeTimeMs }) => {
      if (text && text.trim()) {
        const clean = text.replace(/\[.*?\]/g, '').trim();
        if (clean) {
          this.transcriptView.setLiveTurn(speaker, durationS, clean);
          const ms = typeof decodeTimeMs === 'number' ? ` (${decodeTimeMs.toFixed(1)}ms)` : '';
          this.telemetry.log('TRANSDUCER', `Live [${speaker}]: "${clean}"${ms}`);
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

      this.telemetry.updateAudio(this.audio.micEnabled, Array.from(this.audio.tabStreams.values()));
    };

    this.audio.onUtteranceReady = async (utterance, speakerLabel) => {
      const timeStr = this._formatTime(new Date());
      const durationS = utterance.length / 16000;

      this.transcriptView.setLiveTurn(speakerLabel, durationS, 'Memproses transkrip final...');

      try {
        const rawText = await this.onnx.transcribeUtterance(utterance, speakerLabel, timeStr);
        if (!rawText || !rawText.trim()) {
          this.transcriptView.clearLiveTurn();
          return;
        }

        const cleanedText = this.audio.cleanRepetitiveLoops(rawText);
        if (this.audio.isHallucination(cleanedText)) {
          this.telemetry.log('WHISPER', `Filter halusinasi mengabaikan: "${rawText.slice(0, 30)}..."`);
          this.transcriptView.clearLiveTurn();
          return;
        }

        const cleanText = this.audio.formatUtterance(cleanedText);
        this.transcriptView.clearLiveTurn();
        if (cleanText) {
          this.transcriptView.addSegment(speakerLabel, cleanText, timeStr);
          if (!this.settings.hideNotes) {
            this.notesView.updateInsights(this.transcriptView.getSegments());
          }
          this.telemetry.log('STREAM', `Transkrip dirender [${speakerLabel}]: "${cleanText.slice(0, 50)}..."`);
        }
      } catch (err) {
        this.transcriptView.clearLiveTurn();
        this.telemetry.log('ERROR', `Gagal memproses ucapan [${speakerLabel}]: ${err.message}`);
      }
    };

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

    this.telemetry.updateTransducer(this.onnx.transducerStatus, this.onnx.transducerModel);
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
      console.warn('Sistem belum siap, masih dalam tahap bootstrap:', this.state);
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
      this.vuMeter.reset();
      this.telemetry.updateAudio(false, []);
      this.telemetry.updateVad(false, 0);
      return;
    }

    try {
      await this.audio.startSession();

      if (this.micWanted) {
        try {
          await this.audio.setMicEnabled(true);
          this.btnMic.textContent = 'Mic: Aktif';
          this.btnMic.classList.remove('muted');
        } catch (micErr) {
          console.warn('Tidak dapat mengaktifkan mikrofon saat sesi dimulai:', micErr);
          this.btnMic.textContent = 'Mic: Mati';
          this.btnMic.classList.add('muted');
        }
      }

      this.isRecording = true;
      this.isPaused = false;
      this.btnSession.textContent = 'Stop';
      this.btnSession.classList.add('active');
      this.btnPause.disabled = false;

      this.telemetry.updateAudio(this.audio.micEnabled, Array.from(this.audio.tabStreams.values()));
    } catch (err) {
      alert('Gagal memulai sesi audio: ' + err.message);
    }
  }

  async toggleMic() {
    this.micWanted = !this.micWanted;

    if (this.isRecording) {
      try {
        await this.audio.setMicEnabled(this.micWanted);
      } catch (err) {
        alert('Gagal mengubah mikrofon: ' + err.message);
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

    this.telemetry.updateAudio(this.audio.micEnabled, Array.from(this.audio.tabStreams.values()));
  }

  async addTabAudio() {
    try {
      await this.audio.addTabAudio();
      this.telemetry.updateAudio(this.audio.micEnabled, Array.from(this.audio.tabStreams.values()));
    } catch (err) {
      if (err.name !== 'AbortError' && err.name !== 'NotAllowedError') {
        console.warn('Pembagian audio tab dibatalkan atau gagal:', err);
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
      alert('Belum ada transkrip.');
      return;
    }

    const now = new Date();
    const ts = Math.floor(now.getTime() / 1000);
    const dateStr = now.toLocaleDateString('id-ID', { dateStyle: 'full' });

    let md = `# Catetin - Notulensi Rapat\n\n`;
    md += `*Tanggal: ${dateStr}*\n\n`;

    md += `## Topik Pembahasan\n\n`;
    const topics = this.notesView.topics || [];
    if (topics.length > 0) {
      topics.forEach(t => { md += `- ${t}\n`; });
      md += '\n';
    } else {
      md += `*(Tidak ada topik terdeteksi)*\n\n`;
    }

    md += `## Poin Keputusan & Tindak Lanjut\n\n`;
    const actionItems = this.notesView.actions || [];
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
      if (!confirm('Hapus seluruh transkrip dan catatan rapat?')) return;
    }
    this.transcriptView.clear();
    this.notesView.clear();
    this.telemetry.clearLogs();
  }

  _formatTime(date) {
    const pad = (n) => String(n).padStart(2, '0');
    return `[${pad(date.getHours())}:${pad(date.getMinutes())}:${pad(date.getSeconds())}]`;
  }
}

window.addEventListener('DOMContentLoaded', () => {
  window.app = new CatetinApp();

  if ('serviceWorker' in navigator) {
    navigator.serviceWorker.register('./sw.js').catch((err) => {
      console.warn('[SW] Gagal mendaftarkan Service Worker:', err);
    });
  }
});
