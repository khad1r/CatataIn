// app.js - Main Application Coordinator (Ponytail: native modular ES)

import { AudioEngine } from './audio-engine.js';
import { OpfsModelStorage } from './opfs-storage.js';
import { OnnxRunner } from './onnx-runner.js';
import './components/vu-meter.js';
import './components/transcript-view.js';
import './components/notes-view.js';

class CatetinApp {
  constructor() {
    this.audio = new AudioEngine();
    this.storage = new OpfsModelStorage();
    this.onnx = new OnnxRunner(this.storage);

    this.isRecording = false;
    this.isPaused = false;
    this.activeSpeaker = 'Anda';

    this.initDOM();
    this.initAudioHooks();
    this.initSpeechFallback();
    this.initHotkeys();
  }

  initDOM() {
    this.btnStart = document.getElementById('btn-start');
    this.btnTab = document.getElementById('btn-tab');
    this.btnPause = document.getElementById('btn-pause');
    this.btnSave = document.getElementById('btn-save');
    this.btnClear = document.getElementById('btn-clear');
    this.statusVad = document.getElementById('status-vad');

    this.vuMeter = document.querySelector('vu-meter');
    this.transcriptView = document.querySelector('transcript-view');
    this.notesView = document.querySelector('notes-view');

    this.btnStart.addEventListener('click', () => this.toggleRecord());
    this.btnTab.addEventListener('click', () => this.addTabAudio());
    this.btnPause.addEventListener('click', () => this.togglePause());
    this.btnSave.addEventListener('click', () => this.saveMarkdown());
    this.btnClear.addEventListener('click', () => this.clearAll());
  }

  initAudioHooks() {
    // 1. Live VU Meter Level
    this.audio.onVolume = (level) => {
      if (this.vuMeter) this.vuMeter.setLevel(level);
    };

    // 2. VAD Speaking indicator
    this.audio.onVadSpeaking = (speaking) => {
      if (this.statusVad) {
        if (speaking) {
          this.statusVad.classList.add('active');
        } else {
          this.statusVad.classList.remove('active');
        }
      }
    };

    // 3. Spoken Utterance Finished (Endpointed by VAD)
    this.audio.onUtteranceReady = async (utterance, isRemote) => {
      const speaker = isRemote ? 'Pembicara 1' : 'Anda';
      const timeStr = this._formatTime(new Date());

      // Attempt ONNX Pass 2 Whisper inference if model is ready
      let transcribedText = null;
      if (this.onnx.isReady) {
        transcribedText = await this.onnx.transcribeUtterance(utterance);
      }

      // If speech recognizer provided text via interim or ONNX
      if (this.lastFinalText) {
        this.transcriptView.addSegment(speaker, this.lastFinalText, timeStr);
        this.lastFinalText = '';
        this.transcriptView.setLiveText('');
        this.notesView.updateInsights(this.transcriptView.getSegments());
      }
    };
  }

  initSpeechFallback() {
    // Zero-config speech recognition when running standalone
    const SpeechRec = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (SpeechRec) {
      this.rec = new SpeechRec();
      this.rec.continuous = true;
      this.rec.interimResults = true;
      this.rec.lang = 'id-ID';

      this.rec.onresult = (e) => {
        if (this.isPaused) return;
        let interim = '';
        for (let i = e.resultIndex; i < e.results.length; i++) {
          if (e.results[i].isFinal) {
            this.lastFinalText = e.results[i][0].transcript.trim();
            const timeStr = this._formatTime(new Date());
            this.transcriptView.addSegment(this.activeSpeaker, this.lastFinalText, timeStr);
            this.transcriptView.setLiveText('');
            this.notesView.updateInsights(this.transcriptView.getSegments());
            this.lastFinalText = '';
          } else {
            interim += e.results[i][0].transcript;
          }
        }
        if (interim.trim()) {
          this.transcriptView.setLiveText(interim.trim());
        }
      };

      this.rec.onend = () => {
        if (this.isRecording && !this.isPaused) {
          try { this.rec.start(); } catch (_) {}
        }
      };
    }
  }

  initHotkeys() {
    window.addEventListener('keydown', (e) => {
      // Ignore if typing inside input
      if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;

      if (e.code === 'Space' || e.key === 'p' || e.key === 'P') {
        e.preventDefault();
        if (this.isRecording) this.togglePause();
      } else if (e.key === 'c' || e.key === 'C') {
        e.preventDefault();
        this.saveMarkdown();
      }
    });
  }

  async toggleRecord() {
    if (this.isRecording) {
      this.audio.stop();
      if (this.rec) {
        try { this.rec.stop(); } catch (_) {}
      }
      this.isRecording = false;
      this.isPaused = false;
      this.btnStart.textContent = '🎙 Mulai';
      this.btnStart.classList.remove('active');
      this.btnPause.disabled = true;
      this.transcriptView.setLiveText('');
      if (this.vuMeter) this.vuMeter.reset();
      return;
    }

    try {
      await this.audio.startMic();
      if (this.rec) {
        try { this.rec.start(); } catch (_) {}
      }
      this.isRecording = true;
      this.isPaused = false;
      this.btnStart.textContent = '⏹ Stop';
      this.btnStart.classList.add('active');
      this.btnPause.disabled = false;
    } catch (err) {
      alert("Gagal mengakses mikrofon: " + err.message);
    }
  }

  async addTabAudio() {
    try {
      await this.audio.addTabAudio();
      this.activeSpeaker = 'Pembicara 1';
      this.btnTab.classList.add('active');
    } catch (err) {
      console.warn("Tab audio sharing canceled:", err);
    }
  }

  togglePause() {
    if (!this.isRecording) return;
    this.isPaused = !this.isPaused;
    if (this.isPaused) {
      this.audio.pause();
      if (this.rec) {
        try { this.rec.stop(); } catch (_) {}
      }
      this.btnPause.textContent = '▶ Lanjut';
      this.btnPause.classList.add('active');
      this.transcriptView.setLiveText('');
    } else {
      this.audio.resume();
      if (this.rec) {
        try { this.rec.start(); } catch (_) {}
      }
      this.btnPause.textContent = '⏸ Jeda';
      this.btnPause.classList.remove('active');
    }
  }

  saveMarkdown() {
    const segments = this.transcriptView.getSegments();
    if (segments.length === 0) {
      alert("Belum ada transkrip untuk disimpan.");
      return;
    }

    const now = new Date();
    const ts = Math.floor(now.getTime() / 1000);
    const dateStr = now.toLocaleDateString('id-ID', { dateStyle: 'full' });

    let md = `# Catetin - Notulensi Rapat\n\n`;
    md += `- **Tanggal**: ${dateStr}\n`;
    md += `- **Total Percakapan**: ${segments.length}\n\n`;

    const topics = this.notesView.getTopics();
    const actions = this.notesView.getActions();

    if (topics.length > 0 || actions.length > 0) {
      md += `## Ringkasan\n\n`;
      if (topics.length > 0) {
        md += `### Topik\n`;
        topics.forEach(t => md += `- ${t}\n`);
        md += `\n`;
      }
      if (actions.length > 0) {
        md += `### Tindak Lanjut\n`;
        actions.forEach(a => md += `- [ ] ${a}\n`);
        md += `\n`;
      }
    }

    md += `## Transkrip Lengkap\n\n`;
    segments.forEach(s => {
      md += `- **${s.time} [${s.speaker}]**: ${s.text}\n`;
    });

    const blob = new Blob([md], { type: 'text/markdown;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `meeting_${ts}.md`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }

  clearAll() {
    if (confirm("Kosongkan transkrip dan catatan rapat?")) {
      this.transcriptView.clear();
      this.notesView.clear();
    }
  }

  _formatTime(d) {
    const pad = (n) => String(n).padStart(2, '0');
    return `[${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}]`;
  }
}

window.addEventListener('DOMContentLoaded', () => {
  window.app = new CatetinApp();

  // Register PWA service worker
  if ('serviceWorker' in navigator) {
    navigator.serviceWorker.register('./sw.js').catch(() => {});
  }
});
