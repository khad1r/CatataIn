// components/catetin-telemetry.js - High-Precision Diagnostic & Hardware Telemetry Component
// Displays real-time Thread Topology, CPU/GPU acceleration, Heap Memory, Audio Pipeline,
// VAD state, Whisper Queue/RTF/Latency, and filterable live event logs.
// Zero dependencies, 100% native Web Component. Easily removable in 1 tag.

export class CatetinTelemetry extends HTMLElement {
  constructor() {
    super();
    this.attachShadow({ mode: 'open' });

    this.logs = [];
    this.maxLogs = 200;
    this.activeFilter = 'ALL';
    this.activeTab = 'logs'; // 'logs' | 'stack'
    this.isDrawerOpen = false;
    this.isMinimized = false;

    // Hardware & Environment
    this.hw = {
      cpuCores: typeof navigator !== 'undefined' ? (navigator.hardwareConcurrency || 4) : 4,
      gpuAvailable: false,
      gpuVendor: 'Mencari...',
      gpuArch: '...',
      gpuDevice: '...',
      vramLimitMB: 0,
      heapUsedMB: '0.0',
      sysRamGB: typeof navigator !== 'undefined' && navigator.deviceMemory ? `${navigator.deviceMemory}GB` : 'Tidak diketahui',
      heapLimitMB: '0',
      audioCtxState: 'suspended',
      audioCtxSampleRate: 0,
      wasmMemoryMB: '0.0'
    };

    // Metrics state
    this.state = {
      vadSpeaking: false,
      rms: 0.0,
      micEnabled: false,
      tabStreams: [],
      whisperModel: 'Xenova/whisper-base',
      whisperDevice: 'webgpu',
      whisperStatus: 'idle', // idle | loading | transcribing | ready | error
      queueDepth: 0,
      lastLatencyMs: 0,
      lastRtf: '0.00',
      lastText: '',
      transducerStatus: 'idle',
      transducerModel: 'id',
      transducerFrames: 0,
      totalInferences: 0,
      avgLatencyMs: 0,
      turnDurationS: 0
    };

    this.render();
    this._initHardwareTelemetry();
    this._startPeriodicPolling();
  }

  async _initHardwareTelemetry() {
    // Detect WebGPU Adapter Info & Memory Limits
    if (typeof navigator !== 'undefined' && navigator.gpu) {
      try {
        const adapter = await navigator.gpu.requestAdapter();
        if (adapter) {
          this.hw.gpuAvailable = true;
          let info = {};
          if (adapter.requestAdapterInfo) {
            info = await adapter.requestAdapterInfo();
          } else if (adapter.info) {
            info = adapter.info;
          }
          this.hw.gpuVendor = info.vendor || 'Tidak diketahui';
          this.hw.gpuArch = info.architecture || '-';
          this.hw.gpuDevice = info.device || '-';

          // Extract VRAM limit from WebGPU limits
          if (adapter.limits && adapter.limits.maxBufferSize) {
            this.hw.vramLimitMB = Math.round(adapter.limits.maxBufferSize / 1048576);
          }
        } else {
          this.hw.gpuAvailable = false;
          this.hw.gpuVendor = 'Tidak tersedia';
          this.hw.gpuArch = '-';
        }
      } catch (err) {
        this.hw.gpuAvailable = false;
        this.hw.gpuVendor = 'Tidak tersedia';
        this.hw.gpuArch = '-';
      }
    } else {
      this.hw.gpuAvailable = false;
      this.hw.gpuVendor = 'Tidak tersedia';
      this.hw.gpuArch = '-';
    }

    this._updateHwDisplays();
  }

  _startPeriodicPolling() {
    // Update Memory & AudioContext states every 1.5 seconds
    setInterval(() => {
      if (typeof performance !== 'undefined' && performance.memory) {
        this.hw.heapUsedMB = (performance.memory.usedJSHeapSize / 1048576).toFixed(1);
        this.hw.heapLimitMB = (performance.memory.jsHeapSizeLimit / 1048576).toFixed(0);
      }

      if (window.app && window.app.audio) {
        if (window.app.audio.audioCtx) {
          this.hw.audioCtxState = window.app.audio.audioCtx.state;
          this.hw.audioCtxSampleRate = window.app.audio.audioCtx.sampleRate;
        }
        if (window.app.audio.wasm && window.app.audio.wasm.memory) {
          this.hw.wasmMemoryMB = (window.app.audio.wasm.memory.buffer.byteLength / 1048576).toFixed(2);
        }
      }

      this._updateHwDisplays();
    }, 1500);
  }

  render() {
    const style = document.createElement('style');
    style.textContent = `
      :host {
        display: block;
        width: 100%;
        background: #101012;
        border-top: 1px solid #27272a;
        color: #d4d4d8;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        font-size: 0.72rem;
        user-select: none;
        z-index: 9999;
      }

      .bar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 5px 10px;
        background: #111114;
        gap: 10px;
        flex-wrap: wrap;
      }

      .metrics-group {
        display: flex;
        align-items: center;
        gap: 8px;
        flex-wrap: wrap;
      }

      .metric-item {
        display: flex;
        align-items: center;
        gap: 4px;
        background: #18181b;
        padding: 2px 6px;
        border-radius: 4px;
        border: 1px solid #27272a;
      }

      .label {
        color: #71717a;
        font-weight: 700;
        text-transform: uppercase;
        font-size: 0.64rem;
      }

      .badge {
        display: inline-block;
        padding: 1px 5px;
        border-radius: 3px;
        font-weight: 700;
        font-size: 0.65rem;
        letter-spacing: 0.3px;
      }

      .badge-vad-speaking {
        background: #166534;
        color: #bbf7d0;
        animation: pulse 1s infinite;
      }

      .badge-vad-idle {
        background: #27272a;
        color: #a1a1aa;
      }

      .badge-status-idle {
        background: #27272a;
        color: #9ca3af;
      }

      .badge-status-loading {
        background: #854d0e;
        color: #fef08a;
      }

      .badge-status-busy {
        background: #0369a1;
        color: #e0f2fe;
        animation: pulse 1.2s infinite;
      }

      .badge-status-ready {
        background: #1e3a8a;
        color: #bfdbfe;
      }

      .badge-status-error {
        background: #991b1b;
        color: #fee2e2;
      }

      .badge-queue-warn {
        background: #9a3412;
        color: #ffedd5;
        font-weight: 800;
        animation: pulse 0.8s infinite;
      }

      @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
      }

      .val-cpu { color: #c084fc; font-weight: 600; }
      .val-gpu { color: #38bdf8; font-weight: 600; }
      .val-ram { color: #34d399; font-weight: 600; }
      .val-vram { color: #fbbf24; font-weight: 600; }

      .btn-toolbar {
        display: flex;
        align-items: center;
        gap: 6px;
      }

      .t-btn {
        background: #27272a;
        border: 1px solid #3f3f46;
        color: #f4f4f5;
        padding: 3px 8px;
        border-radius: 4px;
        cursor: pointer;
        font-family: inherit;
        font-size: 0.68rem;
        transition: background 0.15s, border-color 0.15s;
      }

      .t-btn:hover {
        background: #3f3f46;
        border-color: #71717a;
      }

      .t-btn.active {
        background: #0284c7;
        border-color: #38bdf8;
        color: #ffffff;
      }

      .drawer {
        display: none;
        max-height: 250px;
        background: #09090b;
        border-top: 1px solid #27272a;
        padding: 8px 12px;
        flex-direction: column;
        gap: 6px;
      }

      .drawer.open {
        display: flex;
      }

      .drawer-tabs {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding-bottom: 4px;
        border-bottom: 1px solid #1f1f23;
        gap: 8px;
        flex-wrap: wrap;
      }

      .tab-buttons {
        display: flex;
        gap: 4px;
      }

      .filters {
        display: flex;
        gap: 4px;
      }

      .tab-content {
        flex: 1;
        overflow-y: auto;
        max-height: 190px;
      }

      .log-feed {
        display: flex;
        flex-direction: column;
        gap: 2px;
        font-size: 0.71rem;
        line-height: 1.4;
      }

      .log-row {
        display: flex;
        gap: 8px;
        word-break: break-word;
      }

      .log-time {
        color: #71717a;
        flex-shrink: 0;
      }

      .log-cat {
        font-weight: 700;
        flex-shrink: 0;
        width: 82px;
      }

      .cat-AUDIO { color: #38bdf8; }
      .cat-VAD { color: #4ade80; }
      .cat-QUEUE { color: #fbbf24; }
      .cat-WHISPER { color: #c084fc; }
      .cat-TRANSDUCER { color: #f472b6; }
      .cat-STREAM { color: #67e8f9; }
      .cat-ERROR { color: #ef4444; font-weight: 800; }
      .cat-INFO { color: #a1a1aa; }

      .log-msg {
        color: #e4e4e7;
        flex: 1;
      }

      .empty-logs {
        color: #52525b;
        font-style: italic;
        padding: 8px 0;
      }

      /* Thread Stack & Topology Grid */
      .stack-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
        gap: 8px;
        padding: 4px 0;
      }

      .stack-card {
        background: #141417;
        border: 1px solid #27272a;
        border-radius: 5px;
        padding: 8px 10px;
        display: flex;
        flex-direction: column;
        gap: 4px;
      }

      .stack-title {
        color: #38bdf8;
        font-weight: 700;
        font-size: 0.74rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
        border-bottom: 1px solid #1f1f23;
        padding-bottom: 4px;
      }

      .stack-row {
        display: flex;
        justify-content: space-between;
        font-size: 0.69rem;
      }

      .stack-label {
        color: #71717a;
      }

      .stack-val {
        color: #f4f4f5;
        font-weight: 600;
      }
    `;

    this.shadowRoot.innerHTML = '';
    this.shadowRoot.appendChild(style);

    this.container = document.createElement('div');
    this.container.innerHTML = `
      <div class="bar">
        <div class="metrics-group">
          <!-- CPU (#1 Primary) -->
          <div class="metric-item">
            <span class="label">CPU:</span>
            <span id="val-cpu" class="val-cpu">${this.hw.cpuCores} Cores</span>
          </div>

          <!-- GPU (#1 Primary) -->
          <div class="metric-item">
            <span class="label">GPU:</span>
            <span id="val-gpu" class="val-gpu">${this.hw.gpuVendor}</span>
          </div>

          <!-- RAM (#1 Primary) -->
          <div class="metric-item">
            <span class="label">RAM:</span>
            <span id="val-ram" class="val-ram">${this.hw.heapUsedMB}MB / ${this.hw.sysRamGB}</span>
          </div>

          <!-- VRAM (#1 Primary) -->
          <div class="metric-item">
            <span class="label">VRAM:</span>
            <span id="val-vram" class="val-vram">${this.hw.vramLimitMB > 0 ? this.hw.vramLimitMB + 'MB' : 'Shared'}</span>
          </div>

          <!-- VAD & RMS -->
          <div class="metric-item">
            <span class="label">VAD:</span>
            <span id="badge-vad" class="badge badge-vad-idle">DIAM</span>
            <span id="val-rms" style="color:#a1a1aa">0.000</span>
          </div>

          <!-- Audio Streams -->
          <div class="metric-item">
            <span class="label">Audio:</span>
            <span id="val-audio">Mic: Mati | Tab: 0</span>
          </div>

          <!-- Pass 2 Whisper -->
          <div class="metric-item">
            <span class="label">Whisper:</span>
            <span id="badge-whisper" class="badge badge-status-idle">IDLE</span>
            <span id="val-device" style="color:#38bdf8">webgpu</span>
          </div>

          <!-- Antrean Queue -->
          <div class="metric-item">
            <span class="label">Antrean:</span>
            <span id="badge-queue" class="badge badge-status-idle">0</span>
          </div>

          <!-- Performance (Latency & RTF) -->
          <div class="metric-item">
            <span class="label">Perf:</span>
            <span id="val-perf" style="color:#fbbf24">0ms (RTF: 0.00x)</span>
          </div>
        </div>

        <div class="btn-toolbar">
          <button id="btn-toggle-stack" class="t-btn">Thread Stack ▾</button>
          <button id="btn-toggle-log" class="t-btn">Log (0) ▾</button>
          <button id="btn-collapse" class="t-btn" title="Minimalkan telemetry">_</button>
        </div>
      </div>

      <div id="drawer" class="drawer">
        <div class="drawer-tabs">
          <div class="tab-buttons">
            <button id="tab-btn-logs" class="t-btn active">Log Diagnostik</button>
            <button id="tab-btn-stack" class="t-btn">Thread Stack & Arsitektur</button>
          </div>

          <div id="log-controls" class="filters">
            <button class="t-btn active" data-filter="ALL">SEMUA</button>
            <button class="t-btn" data-filter="AUDIO">AUDIO</button>
            <button class="t-btn" data-filter="VAD">VAD</button>
            <button class="t-btn" data-filter="TRANSDUCER">TRANSDUCER</button>
            <button class="t-btn" data-filter="WHISPER">WHISPER</button>
            <button class="t-btn" data-filter="QUEUE">ANTREAN</button>
            <button class="t-btn" data-filter="ERROR">ERROR</button>
            <button id="btn-clear-log" class="t-btn" style="margin-left:8px">Bersihkan</button>
            <button id="btn-copy-log" class="t-btn">Salin</button>
          </div>
        </div>

        <div id="tab-content" class="tab-content">
          <!-- View 1: Logs -->
          <div id="view-logs" class="log-feed">
            <div class="empty-logs">Belum ada aktivitas diagnostik tercatat.</div>
          </div>

          <!-- View 2: Thread Stack & Topology -->
          <div id="view-stack" class="stack-grid" style="display: none;">
            <!-- Thread 0: UI Main Thread -->
            <div class="stack-card">
              <div class="stack-title">
                <span>Thread 0: UI Main Thread</span>
              </div>
              <div class="stack-row">
                <span class="stack-label">Fungsi:</span>
                <span class="stack-val">DOM, Web Components, VU Meter, Event Loop</span>
              </div>
              <div class="stack-row">
                <span class="stack-label">CPU Cores:</span>
                <span id="st-cores" class="stack-val">${this.hw.cpuCores} Logical Cores</span>
              </div>
              <div class="stack-row">
                <span class="stack-label">JS Heap Memory:</span>
                <span id="st-heap" class="stack-val">${this.hw.heapUsedMB} MB / ${this.hw.sysRamGB}</span>
              </div>
            </div>

            <!-- Thread 1: Audio Render Thread & Rust WASM -->
            <div class="stack-card">
              <div class="stack-title">
                <span>Thread 1: WebAudio & Rust WASM</span>
                <span id="st-audio-state" class="badge badge-vad-idle">SUSPENDED</span>
              </div>
              <div class="stack-row">
                <span class="stack-label">AudioContext:</span>
                <span id="st-sample-rate" class="stack-val">${this.hw.audioCtxSampleRate ? this.hw.audioCtxSampleRate + ' Hz' : '-'}</span>
              </div>
              <div class="stack-row">
                <span class="stack-label">catetin.wasm DSP:</span>
                <span id="st-dsp-info" class="stack-val">Hann Resampler (${this.hw.audioCtxSampleRate ? (this.hw.audioCtxSampleRate / 1000).toFixed(1).replace(/\.0$/, '') + 'k' : 'Native'}->16k), SpeechSegmenter</span>
              </div>
              <div class="stack-row">
                <span class="stack-label">WASM Heap:</span>
                <span id="st-wasm-mem" class="stack-val">${this.hw.wasmMemoryMB} MB</span>
              </div>
            </div>

            <!-- Thread 2: ASR Dedicated Web Worker -->
            <div class="stack-card">
              <div class="stack-title">
                <span>Thread 2: ASR Pass 2 Dedicated Worker</span>
                <span id="st-worker-status" class="badge badge-status-idle">IDLE</span>
              </div>
              <div class="stack-row">
                <span class="stack-label">Engine Hub:</span>
                <span class="stack-val">Pass2Hub (WebGPU / WASM / WebNN)</span>
              </div>
              <div class="stack-row">
                <span class="stack-label">Akselerasi GPU:</span>
                <span id="st-gpu-info" class="stack-val" style="color:#38bdf8">${this.hw.gpuVendor} (${this.hw.gpuArch})</span>
              </div>
              <div class="stack-row">
                <span class="stack-label">Buffer Transfer:</span>
                <span class="stack-val">Zero-copy ArrayBuffer TransferList</span>
              </div>
              <div class="stack-row">
                <span class="stack-label">Whisper Model:</span>
                <span id="st-whisper-model" class="stack-val">${this.state.whisperModel}</span>
              </div>
            </div>

            <!-- Thread 3: Pass 1 Streaming Transducer Dedicated Worker -->
            <div class="stack-card">
              <div class="stack-title">
                <span>Thread 3: Pass 1 Transducer Worker</span>
                <span id="st-transducer-status" class="badge badge-status-idle">IDLE</span>
              </div>
              <div class="stack-row">
                <span class="stack-label">Engine:</span>
                <span class="stack-val">Sherpa-ONNX Online Recognizer</span>
              </div>
              <div class="stack-row">
                <span class="stack-label">Model:</span>
                <span id="st-transducer-model" class="stack-val" style="color:#f472b6;">Zipformer2 ID (WASM SIMD)</span>
              </div>
              <div class="stack-row">
                <span class="stack-label">Input Audio:</span>
                <span class="stack-val">160ms Chunks Streaming</span>
              </div>
              <div class="stack-row">
                <span class="stack-label">Target Latensi:</span>
                <span class="stack-val" style="color:#22c55e;">Sub-detik (&lt; 50ms / frame)</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    `;

    this.shadowRoot.appendChild(this.container);

    this.valCpu = this.shadowRoot.getElementById('val-cpu');
    this.valGpu = this.shadowRoot.getElementById('val-gpu');
    this.valRam = this.shadowRoot.getElementById('val-ram');
    this.valVram = this.shadowRoot.getElementById('val-vram');

    this.badgeVad = this.shadowRoot.getElementById('badge-vad');
    this.valRms = this.shadowRoot.getElementById('val-rms');
    this.valAudio = this.shadowRoot.getElementById('val-audio');
    this.badgeWhisper = this.shadowRoot.getElementById('badge-whisper');
    this.valDevice = this.shadowRoot.getElementById('val-device');
    this.badgeQueue = this.shadowRoot.getElementById('badge-queue');
    this.valPerf = this.shadowRoot.getElementById('val-perf');

    this.drawer = this.shadowRoot.getElementById('drawer');
    this.viewLogs = this.shadowRoot.getElementById('view-logs');
    this.viewStack = this.shadowRoot.getElementById('view-stack');
    this.logControls = this.shadowRoot.getElementById('log-controls');

    this.btnToggleLog = this.shadowRoot.getElementById('btn-toggle-log');
    this.btnToggleStack = this.shadowRoot.getElementById('btn-toggle-stack');
    this.btnCollapse = this.shadowRoot.getElementById('btn-collapse');
    this.btnClearLog = this.shadowRoot.getElementById('btn-clear-log');
    this.btnCopyLog = this.shadowRoot.getElementById('btn-copy-log');

    this.tabBtnLogs = this.shadowRoot.getElementById('tab-btn-logs');
    this.tabBtnStack = this.shadowRoot.getElementById('tab-btn-stack');

    // Stack card values
    this.stCores = this.shadowRoot.getElementById('st-cores');
    this.stHeap = this.shadowRoot.getElementById('st-heap');
    this.stAudioState = this.shadowRoot.getElementById('st-audio-state');
    this.stSampleRate = this.shadowRoot.getElementById('st-sample-rate');
    this.stDspInfo = this.shadowRoot.getElementById('st-dsp-info');
    this.stWasmMem = this.shadowRoot.getElementById('st-wasm-mem');
    this.stWorkerStatus = this.shadowRoot.getElementById('st-worker-status');
    this.stGpuInfo = this.shadowRoot.getElementById('st-gpu-info');
    this.stWhisperModel = this.shadowRoot.getElementById('st-whisper-model');
    this.stTransducerStatus = this.shadowRoot.getElementById('st-transducer-status');
    this.stTransducerModel = this.shadowRoot.getElementById('st-transducer-model');

    // Event listeners
    this.btnToggleLog.addEventListener('click', () => {
      if (this.isDrawerOpen && this.activeTab === 'logs') {
        this.toggleDrawer(false);
      } else {
        this.activeTab = 'logs';
        this.switchTab('logs');
        this.toggleDrawer(true);
      }
    });

    this.btnToggleStack.addEventListener('click', () => {
      if (this.isDrawerOpen && this.activeTab === 'stack') {
        this.toggleDrawer(false);
      } else {
        this.activeTab = 'stack';
        this.switchTab('stack');
        this.toggleDrawer(true);
      }
    });

    this.tabBtnLogs.addEventListener('click', () => this.switchTab('logs'));
    this.tabBtnStack.addEventListener('click', () => this.switchTab('stack'));
    this.btnCollapse.addEventListener('click', () => this.toggleMinimize());
    this.btnClearLog.addEventListener('click', () => this.clearLogs());
    this.btnCopyLog.addEventListener('click', () => this.copyLogs());

    const filterBtns = this.shadowRoot.querySelectorAll('.filters .t-btn[data-filter]');
    filterBtns.forEach(btn => {
      btn.addEventListener('click', () => {
        filterBtns.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        this.activeFilter = btn.dataset.filter;
        this.renderLogFeed();
      });
    });
  }

  switchTab(tab) {
    this.activeTab = tab;
    if (tab === 'logs') {
      this.tabBtnLogs.classList.add('active');
      this.tabBtnStack.classList.remove('active');
      this.viewLogs.style.display = 'flex';
      this.viewStack.style.display = 'none';
      this.logControls.style.display = 'flex';
      this.scrollToBottom();
    } else {
      this.tabBtnLogs.classList.remove('active');
      this.tabBtnStack.classList.add('active');
      this.viewLogs.style.display = 'none';
      this.viewStack.style.display = 'grid';
      this.logControls.style.display = 'none';
    }
  }

  toggleDrawer(forceOpen = null) {
    if (typeof forceOpen === 'boolean') {
      this.isDrawerOpen = forceOpen;
    } else {
      this.isDrawerOpen = !this.isDrawerOpen;
    }

    if (this.isDrawerOpen) {
      this.drawer.classList.add('open');
      if (this.activeTab === 'logs') {
        this.btnToggleLog.classList.add('active');
        this.btnToggleStack.classList.remove('active');
      } else {
        this.btnToggleStack.classList.add('active');
        this.btnToggleLog.classList.remove('active');
      }
      this.scrollToBottom();
    } else {
      this.drawer.classList.remove('open');
      this.btnToggleLog.classList.remove('active');
      this.btnToggleStack.classList.remove('active');
    }
  }

  toggleMinimize() {
    this.isMinimized = !this.isMinimized;
    const metricsGroup = this.shadowRoot.querySelector('.metrics-group');
    if (this.isMinimized) {
      if (this.isDrawerOpen) this.toggleDrawer(false);
      metricsGroup.style.display = 'none';
      this.btnCollapse.textContent = '^';
      this.btnCollapse.title = 'Buka panel telemetry';
    } else {
      metricsGroup.style.display = 'flex';
      this.btnCollapse.textContent = '_';
      this.btnCollapse.title = 'Minimalkan telemetry';
    }
  }

  _updateHwDisplays() {
    if (this.valCpu) this.valCpu.textContent = `${this.hw.cpuCores} Cores`;
    if (this.valGpu) this.valGpu.textContent = `${this.hw.gpuVendor} (${this.hw.gpuArch})`;
    if (this.valRam) this.valRam.textContent = `${this.hw.heapUsedMB}MB / ${this.hw.sysRamGB}`;
    if (this.valVram) this.valVram.textContent = this.hw.vramLimitMB > 0 ? `${this.hw.vramLimitMB}MB` : 'Shared';

    if (this.stCores) this.stCores.textContent = `${this.hw.cpuCores} Logical Cores`;
    if (this.stHeap) this.stHeap.textContent = `${this.hw.heapUsedMB} MB / ${this.hw.sysRamGB}`;
    if (this.stAudioState) {
      this.stAudioState.textContent = (this.hw.audioCtxState || 'suspended').toUpperCase();
      this.stAudioState.className = this.hw.audioCtxState === 'running' ? 'badge badge-vad-speaking' : 'badge badge-vad-idle';
    }
    if (this.stSampleRate) this.stSampleRate.textContent = `${this.hw.audioCtxSampleRate ? this.hw.audioCtxSampleRate + ' Hz' : '-'}`;
    if (this.stDspInfo) {
      const rateK = this.hw.audioCtxSampleRate
        ? `${(this.hw.audioCtxSampleRate / 1000).toFixed(1).replace(/\.0$/, '')}k`
        : 'Native';
      this.stDspInfo.textContent = `Hann Resampler (${rateK}->16k), SpeechSegmenter`;
    }
    if (this.stWasmMem) this.stWasmMem.textContent = `${this.hw.wasmMemoryMB} MB`;
    if (this.stGpuInfo) this.stGpuInfo.textContent = `${this.hw.gpuVendor} (${this.hw.gpuArch})`;
  }

  log(category, message, meta = null) {
    const time = new Date().toTimeString().slice(0, 8);
    const entry = { time, category, message, meta };
    this.logs.push(entry);
    if (this.logs.length > this.maxLogs) {
      this.logs.shift();
    }

    this.btnToggleLog.textContent = `Log (${this.logs.length}) ${this.isDrawerOpen && this.activeTab === 'logs' ? '▴' : '▾'}`;

    if (this.viewLogs && (this.activeFilter === 'ALL' || this.activeFilter === category)) {
      const emptyEl = this.viewLogs.querySelector('.empty-logs');
      if (emptyEl) {
        emptyEl.remove();
      }

      const row = document.createElement('div');
      row.className = 'log-row';
      row.innerHTML = `
        <span class="log-time">${entry.time}</span>
        <span class="log-cat cat-${entry.category}">[${entry.category}]</span>
        <span class="log-msg">${escapeHtml(entry.message)}</span>
      `;
      this.viewLogs.appendChild(row);

      if (this.viewLogs.children.length > this.maxLogs) {
        this.viewLogs.firstElementChild?.remove();
      }

      if (this.activeTab === 'logs' && this.isDrawerOpen) {
        this.scrollToBottom();
      }
    }
  }

  clearLogs() {
    this.logs = [];
    this.renderLogFeed();
    this.btnToggleLog.textContent = `Log (0) ▾`;
  }

  copyLogs() {
    const text = this.logs
      .map(l => `[${l.time}] [${l.category}] ${l.message}`)
      .join('\n');
    navigator.clipboard.writeText(text).catch(err => console.error('[Telemetry] Gagal menyalin log:', err));
  }

  renderLogFeed() {
    if (!this.viewLogs) return;

    const filtered = this.activeFilter === 'ALL'
      ? this.logs
      : this.logs.filter(l => l.category === this.activeFilter);

    if (filtered.length === 0) {
      this.viewLogs.innerHTML = `<div class="empty-logs">Tidak ada log dalam kategori ${this.activeFilter}.</div>`;
      return;
    }

    this.viewLogs.innerHTML = filtered.map(l => `
      <div class="log-row">
        <span class="log-time">${l.time}</span>
        <span class="log-cat cat-${l.category}">[${l.category}]</span>
        <span class="log-msg">${escapeHtml(l.message)}</span>
      </div>
    `).join('');

    this.scrollToBottom();
  }

  scrollToBottom() {
    if (this.viewLogs) {
      this.viewLogs.scrollTop = this.viewLogs.scrollHeight;
    }
  }

  updateVad(speaking, rms = 0) {
    this.state.vadSpeaking = speaking;
    this.state.rms = rms;

    if (this.badgeVad) {
      if (speaking) {
        this.badgeVad.className = 'badge badge-vad-speaking';
        this.badgeVad.textContent = 'BICARA';
      } else {
        this.badgeVad.className = 'badge badge-vad-idle';
        this.badgeVad.textContent = 'DIAM';
      }
    }

    if (this.valRms) {
      this.valRms.textContent = rms.toFixed(3);
    }
  }

  updateAudio(micEnabled, tabStreams = []) {
    this.state.micEnabled = micEnabled;
    this.state.tabStreams = tabStreams;

    if (this.valAudio) {
      const micStr = micEnabled ? 'Mic: Aktif' : 'Mic: Mati';
      const tabCount = tabStreams.length;
      const tabStr = tabCount > 0
        ? `Tab: ${tabCount} (${tabStreams.map(t => t.label).join(', ')})`
        : 'Tab: 0';
      this.valAudio.textContent = `${micStr} | ${tabStr}`;
    }
  }

  updateTransducer(status, model) {
    if (status) this.state.transducerStatus = status;
    if (model) this.state.transducerModel = model;

    if (this.stTransducerStatus) {
      const s = this.state.transducerStatus || 'idle';
      this.stTransducerStatus.textContent = s.toUpperCase();
      if (s === 'ready') {
        this.stTransducerStatus.className = 'badge badge-status-ready';
      } else if (s === 'loading') {
        this.stTransducerStatus.className = 'badge badge-status-loading';
      } else if (s === 'error') {
        this.stTransducerStatus.className = 'badge badge-status-error';
      } else {
        this.stTransducerStatus.className = 'badge badge-status-idle';
      }
    }

    if (this.stTransducerModel) {
      const m = this.state.transducerModel;
      this.stTransducerModel.textContent = m === 'none'
        ? 'Nonaktif'
        : `Zipformer2 ${String(m).toUpperCase()} (WASM SIMD)`;
    }
  }

  updateWhisper(info) {
    if (info.status) this.state.whisperStatus = info.status;
    if (info.model) {
      this.state.whisperModel = info.model;
      if (this.stWhisperModel) this.stWhisperModel.textContent = info.model;
    }
    if (info.device) this.state.whisperDevice = info.device;
    if (typeof info.queueDepth === 'number') this.state.queueDepth = info.queueDepth;
    if (typeof info.lastLatencyMs === 'number') this.state.lastLatencyMs = info.lastLatencyMs;
    if (info.lastRtf) this.state.lastRtf = info.lastRtf;

    if (this.badgeWhisper) {
      const s = this.state.whisperStatus;
      if (s === 'transcribing') {
        this.badgeWhisper.className = 'badge badge-status-busy';
        this.badgeWhisper.textContent = 'INFERENSI';
      } else if (s === 'loading') {
        this.badgeWhisper.className = 'badge badge-status-loading';
        this.badgeWhisper.textContent = 'MEMUAT';
      } else if (s === 'ready') {
        this.badgeWhisper.className = 'badge badge-status-ready';
        this.badgeWhisper.textContent = 'SIAP';
      } else if (s === 'error') {
        this.badgeWhisper.className = 'badge badge-status-error';
        this.badgeWhisper.textContent = 'ERROR';
      } else {
        this.badgeWhisper.className = 'badge badge-status-idle';
        this.badgeWhisper.textContent = 'IDLE';
      }
    }

    if (this.stWorkerStatus) {
      this.stWorkerStatus.textContent = (this.state.whisperStatus || 'idle').toUpperCase();
      this.stWorkerStatus.className = this.state.whisperStatus === 'transcribing'
        ? 'badge badge-status-busy'
        : 'badge badge-status-idle';
    }

    if (this.valDevice) {
      const shortModel = (this.state.whisperModel || '').replace('Xenova/', '');
      this.valDevice.textContent = `${shortModel} (${this.state.whisperDevice})`;
    }

    if (this.badgeQueue) {
      const q = this.state.queueDepth;
      this.badgeQueue.textContent = q.toString();
      if (q > 2) {
        this.badgeQueue.className = 'badge badge-queue-warn';
      } else if (q > 0) {
        this.badgeQueue.className = 'badge badge-status-busy';
      } else {
        this.badgeQueue.className = 'badge badge-status-idle';
      }
    }

    if (this.valPerf) {
      this.valPerf.textContent = `${this.state.lastLatencyMs}ms (RTF: ${this.state.lastRtf}x)`;
    }
  }
}

function escapeHtml(str) {
  if (!str) return '';
  return str
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

customElements.define('catetin-telemetry', CatetinTelemetry);
