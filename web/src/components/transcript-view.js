// components/transcript-view.js - Native Web Component for Diarized Meeting Transcript

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
      }
      .live-tag {
        font-weight: 700;
        margin-right: 6px;
      }
      .live-text {
        color: #e0f2fe;
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
    header.textContent = 'TRANSKRIP';

    this.liveBar = document.createElement('div');
    this.liveBar.className = 'live-bar';
    this.liveBar.innerHTML = `<span class="live-tag">⚡ LIVE:</span><span class="live-text"></span>`;

    this.feed = document.createElement('div');
    this.feed.className = 'feed';
    this.feed.innerHTML = `<div class="empty">Belum ada percakapan. Tekan <strong>Mulai</strong> untuk mencatat rapat.</div>`;

    this.shadowRoot.appendChild(header);
    this.shadowRoot.appendChild(this.liveBar);
    this.shadowRoot.appendChild(this.feed);
  }

  setLiveText(text) {
    const liveTextSpan = this.liveBar.querySelector('.live-text');
    const clean = text ? text.trim() : '';
    if (clean) {
      liveTextSpan.textContent = `"${clean}"`;
      this.liveBar.style.display = 'block';
    } else {
      this.liveBar.style.display = 'none';
      liveTextSpan.textContent = '';
    }
  }

  addSegment(speaker, text, timeStr) {
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
      <div class="turn-text">${text}</div>
    `;

    this.feed.appendChild(turn);
    this.segments.push({ speaker, text, time: timeStr });

    // Auto-scroll to bottom
    this.feed.scrollTop = this.feed.scrollHeight;
  }

  clear() {
    this.segments = [];
    this.setLiveText('');
    this.feed.innerHTML = `<div class="empty">Belum ada percakapan. Tekan <strong>Mulai</strong> untuk mencatat rapat.</div>`;
  }

  getSegments() {
    return this.segments;
  }
}

customElements.define('transcript-view', TranscriptView);
