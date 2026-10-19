// components/vu-meter.js - Native Web Component for Audio Waveform Visualization

export class VuMeter extends HTMLElement {
  constructor() {
    super();
    this.attachShadow({ mode: 'open' });
    this.canvas = document.createElement('canvas');
    this.canvas.width = 160;
    this.canvas.height = 24;
    this.ctx = this.canvas.getContext('2d');
    this.history = new Array(24).fill(0.0);
    this.currentLevel = 0.0;
    this.animId = null;

    const style = document.createElement('style');
    style.textContent = `
      :host {
        display: inline-block;
        vertical-align: middle;
      }
      canvas {
        display: block;
        border-radius: 4px;
        background: #1e1e1e;
        border: 1px solid #333;
      }
    `;
    this.shadowRoot.appendChild(style);
    this.shadowRoot.appendChild(this.canvas);
  }

  connectedCallback() {
    this._startRenderLoop();
  }

  disconnectedCallback() {
    if (this.animId) cancelAnimationFrame(this.animId);
  }

  setLevel(level) {
    this.currentLevel = Math.max(0, Math.min(1, level));
  }

  reset() {
    this.currentLevel = 0;
    this.history.fill(0);
  }

  _startRenderLoop() {
    const loop = () => {
      this.history.push(this.currentLevel);
      if (this.history.length > 24) this.history.shift();

      const w = this.canvas.width;
      const h = this.canvas.height;
      this.ctx.clearRect(0, 0, w, h);

      const count = this.history.length;
      const barW = (w / count) - 1;

      for (let i = 0; i < count; i++) {
        const val = this.history[i];
        const barH = Math.max(2, val * h * 0.95);
        const x = i * (barW + 1);
        const y = (h - barH) / 2;

        const hue = 190 + (val * 70); // Cyan to Emerald
        this.ctx.fillStyle = `hsl(${hue}, 80%, 55%)`;
        this.ctx.fillRect(x, y, barW, barH);
      }

      this.animId = requestAnimationFrame(loop);
    };
    loop();
  }
}

customElements.define('vu-meter', VuMeter);
