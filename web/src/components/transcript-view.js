// components/transcript-view.js - Native Web Component for Diarized Meeting Transcript
// Real-time live turn feedback with pulsing activity and clean speaker turns.

export class TranscriptView extends HTMLElement {
  constructor() {
    super();
    this.attachShadow({ mode: 'open' });
    this.segments = [];
    this.hasLiveText = false;

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

    const header = document.createElement('div');
    header.className = 'header';
    header.textContent = 'Transkrip';

    this.liveBar = document.createElement('div');
    this.liveBar.className = 'live-bar';

    this.liveTagEl = document.createElement('span');
    this.liveTextEl = document.createElement('span');
    this.liveDurEl = document.createElement('span');
    this.liveDurEl.className = 'live-dur';
    this.liveBar.append(this.liveTagEl, this.liveTextEl, this.liveDurEl);

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

    this.liveTagEl.className = `${spkClass} live-tag`;
    this.liveTagEl.textContent = tag;
    this.liveDurEl.textContent = durStr;

    if (partialText && partialText.trim()) {
      this.hasLiveText = true;
      this.liveTextEl.className = 'live-text';
      this.liveTextEl.textContent = `"${partialText.trim()}"`;
    } else {
      this.hasLiveText = false;
      this.liveTextEl.className = 'live-text live-pulse';
      this.liveTextEl.textContent = 'Mendengarkan...';
    }
    this.liveBar.style.display = 'flex';
  }

  clearLiveTurn() {
    this.hasLiveText = false;
    this.liveBar.style.display = 'none';
    this.liveTagEl.textContent = '';
    this.liveTextEl.textContent = '';
    this.liveDurEl.textContent = '';
  }

  addSegment(speaker, text, timeStr) {
    this.clearLiveTurn();

    const empty = this.feed.querySelector('.empty');
    if (empty) empty.remove();

    const turn = document.createElement('div');
    turn.className = 'turn';

    const isYou = speaker.toLowerCase().includes('anda') || speaker.toLowerCase().includes('you');
    const spkClass = isYou ? 'spk-you' : 'spk-remote';

    const nearBottom = (this.feed.scrollHeight - this.feed.scrollTop - this.feed.clientHeight) <= 48;

    turn.innerHTML = `
      <div class="turn-meta">
        <span class="time">${escapeHtml(timeStr)}</span>
        <span class="${spkClass}">[${escapeHtml(speaker)}]</span>
      </div>
      <div class="turn-text">${escapeHtml(text)}</div>
    `;

    this.feed.appendChild(turn);
    this.segments.push({ speaker, text, time: timeStr });

    if (nearBottom) {
      this.feed.scrollTop = this.feed.scrollHeight;
    }
  }

  clear() {
    this.segments = [];
    this.feed.innerHTML = `<div class="empty">Belum ada transkrip.</div>`;
    this.clearLiveTurn();
  }

  getSegments() {
    return this.segments;
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
