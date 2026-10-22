// components/transcript-view.js - Native Web Component for Diarized Meeting Transcript
// Real-time live turn feedback with pulsing activity and clean speaker turns.

export class TranscriptView extends HTMLElement {
  constructor() {
    super();
    this.attachShadow({ mode: 'open' });
    this.segments = [];

    const style = document.createElement('style');
    style.textContent = `
      :host {
        display: flex;
        flex-direction: column;
        height: 100%;
        overflow: hidden;
        background: #181818;
        border: 1px solid #2d2d2d;
        border-radius: 6px;
      }
      .header {
        padding: 8px 12px;
        background: #202020;
        border-bottom: 1px solid #2d2d2d;
        font-size: 0.8rem;
        font-weight: 700;
        color: #888;
        letter-spacing: 0.5px;
      }
      .live-bar {
        display: none;
        padding: 8px 12px;
        background: rgba(56, 189, 248, 0.08);
        border-bottom: 1px solid rgba(56, 189, 248, 0.2);
        font-size: 0.85rem;
        color: #38bdf8;
        align-items: center;
        gap: 8px;
        flex-wrap: wrap;
      }
      .live-tag {
        font-weight: 700;
      }
      .live-pulse {
        animation: pulse 1.2s infinite;
        font-style: italic;
        color: #7dd3fc;
      }
      .live-dur {
        font-size: 0.75rem;
        color: #94a3b8;
        font-family: monospace;
      }
      @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.45; }
      }
      .feed {
        flex: 1;
        overflow-y: auto;
        padding: 12px;
        display: flex;
        flex-direction: column;
        gap: 10px;
      }
      .turn {
        display: flex;
        flex-direction: column;
        gap: 4px;
        line-height: 1.45;
      }
      .turn-meta {
        font-size: 0.75rem;
        font-family: monospace;
        display: flex;
        align-items: center;
        gap: 6px;
      }
      .time {
        color: #666;
      }
      .spk-you {
        color: #38bdf8;
        font-weight: 700;
      }
      .spk-remote {
        color: #fbbf24;
        font-weight: 700;
      }
      .turn-text {
        font-size: 0.95rem;
        color: #e5e5e5;
        white-space: pre-wrap;
        word-break: break-word;
      }
      .empty {
        margin: auto;
        color: #666;
        font-size: 0.85rem;
        text-align: center;
      }
    `;

    this.shadowRoot.appendChild(style);

    this.container = document.createElement('div');
    this.container.style.display = 'contents';

    const header = document.createElement('div');
    header.className = 'header';
    header.textContent = 'Transkrip';

    this.liveBar = document.createElement('div');
    this.liveBar.className = 'live-bar';
    this.liveBar.innerHTML = `<span class="live-text"></span>`;

    this.feed = document.createElement('div');
    this.feed.className = 'feed';
    this.feed.innerHTML = `<div class="empty">Belum ada transkrip.</div>`;

    this.shadowRoot.appendChild(header);
    this.shadowRoot.appendChild(this.liveBar);
    this.shadowRoot.appendChild(this.feed);
  }

  setLiveTurn(speaker, durationS, partialText = null) {
    const isYou = speaker.toLowerCase().includes('anda') || speaker.toLowerCase().includes('you');
    const spkClass = isYou ? 'spk-you' : 'spk-remote';
    const tag = `[${speaker}]`;
    const durStr = durationS > 0 ? ` [${durationS.toFixed(1)}s]` : '';

    if (partialText && partialText.trim()) {
      this.hasLiveText = true;
      this.liveBar.innerHTML = `
        <span class="${spkClass} live-tag">${tag}</span>
        <span class="live-text">"${escapeHtml(partialText.trim())}"</span>
        <span class="live-dur">${durStr}</span>
      `;
    } else {
      this.hasLiveText = false;
      this.liveBar.innerHTML = `
        <span class="${spkClass} live-tag">${tag}</span>
        <span class="live-text live-pulse">Mendengarkan...</span>
        <span class="live-dur">${durStr}</span>
      `;
    }
    this.liveBar.style.display = 'flex';
  }

  clearLiveTurn() {
    this.hasLiveText = false;
    this.liveBar.style.display = 'none';
    this.liveBar.innerHTML = `<span class="live-text"></span>`;
  }

  setLiveText(text) {
    if (!text || !text.trim()) {
      this.clearLiveTurn();
    } else {
      this.setLiveTurn('Live', 0, text);
    }
  }

  addSegment(speaker, text, timeStr) {
    this.clearLiveTurn();

    const empty = this.feed.querySelector('.empty');
    if (empty) empty.remove();

    const turn = document.createElement('div');
    turn.className = 'turn';

    const isYou = speaker.toLowerCase().includes('anda') || speaker.toLowerCase().includes('you');
    const spkClass = isYou ? 'spk-you' : 'spk-remote';

    turn.innerHTML = `
      <div class="turn-meta">
        <span class="time">${timeStr}</span>
        <span class="${spkClass}">[${speaker}]</span>
      </div>
      <div class="turn-text">${escapeHtml(text)}</div>
    `;

    this.feed.appendChild(turn);
    this.segments.push({ speaker, text, time: timeStr });

    // Auto-scroll to bottom
    this.feed.scrollTop = this.feed.scrollHeight;
  }

  clear() {
    this.segments = [];
    this.feed.innerHTML = `<div class="empty">Belum ada transkrip.</div>`;
    this.clearLiveTurn();
  }

  getSegments() {
    return this.segments;
  }

  getMarkdown() {
    if (this.segments.length === 0) return '';
    return this.segments
      .map(s => `${s.time} [${s.speaker}]: ${s.text}`)
      .join('\n\n');
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

customElements.define('transcript-view', TranscriptView);
