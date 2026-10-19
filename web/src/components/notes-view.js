// components/notes-view.js - Native Web Component for Meeting Topics & Action Items

export class NotesView extends HTMLElement {
  constructor() {
    super();
    this.attachShadow({ mode: 'open' });
    this.topics = [];
    this.actions = [];

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
      .content {
        flex: 1;
        overflow-y: auto;
        padding: 12px;
        display: flex;
        flex-direction: column;
        gap: 16px;
      }
      .section-title {
        font-size: 0.75rem;
        font-weight: 700;
        color: #38bdf8;
        letter-spacing: 0.5px;
        margin-bottom: 6px;
      }
      ul {
        list-style: none;
        margin: 0;
        padding: 0;
        display: flex;
        flex-direction: column;
        gap: 6px;
      }
      li {
        font-size: 0.85rem;
        color: #ccc;
        line-height: 1.4;
      }
      .topic-item::before {
        content: "•";
        color: #888;
        margin-right: 6px;
      }
      .action-item {
        display: flex;
        align-items: flex-start;
        gap: 6px;
      }
      .action-item input[type="checkbox"] {
        margin-top: 3px;
        accent-color: #38bdf8;
      }
      .empty-note {
        color: #666;
        font-size: 0.8rem;
      }
    `;

    this.shadowRoot.appendChild(style);

    const header = document.createElement('div');
    header.className = 'header';
    header.textContent = 'CATATAN';

    this.content = document.createElement('div');
    this.content.className = 'content';
    this.content.innerHTML = `
      <div>
        <div class="section-title">TOPIK</div>
        <ul id="topic-list"><li class="empty-note">Topik terdeteksi otomatis saat ada percakapan.</li></ul>
      </div>
      <div>
        <div class="section-title">TINDAK LANJUT</div>
        <ul id="action-list"><li class="empty-note">Action items akan muncul di sini.</li></ul>
      </div>
    `;

    this.shadowRoot.appendChild(header);
    this.shadowRoot.appendChild(this.content);
  }

  updateInsights(allSegments) {
    if (allSegments.length === 0) return;

    const actionKeywords = ['harus', 'nanti', 'besok', 'tolong', 'action', 'perlu', 'jadwalkan', 'siapkan', 'tugas', 'deadline', 'will', 'need to'];
    const topicKeywords = ['server', 'fitur', 'deployment', 'desain', 'database', 'anggaran', 'klien', 'timeline', 'evaluasi', 'api', 'model', 'rilis'];

    const newActions = [];
    const newTopics = [];
    const fullText = allSegments.map(s => s.text).join(' ').toLowerCase();

    allSegments.forEach(seg => {
      const lower = seg.text.toLowerCase();
      if (actionKeywords.some(kw => lower.includes(kw))) {
        if (!newActions.includes(seg.text) && newActions.length < 8) {
          newActions.push(seg.text);
        }
      }
    });

    topicKeywords.forEach(kw => {
      if (fullText.includes(kw) && !newTopics.includes(kw.toUpperCase())) {
        if (newTopics.length < 5) newTopics.push(kw.toUpperCase());
      }
    });

    this.topics = newTopics;
    this.actions = newActions;
    this._render();
  }

  _render() {
    const topicList = this.shadowRoot.getElementById('topic-list');
    const actionList = this.shadowRoot.getElementById('action-list');

    if (this.topics.length === 0) {
      topicList.innerHTML = `<li class="empty-note">Topik terdeteksi otomatis saat ada percakapan.</li>`;
    } else {
      topicList.innerHTML = this.topics.map(t => `<li class="topic-item">${t}</li>`).join('');
    }

    if (this.actions.length === 0) {
      actionList.innerHTML = `<li class="empty-note">Action items akan muncul di sini.</li>`;
    } else {
      actionList.innerHTML = this.actions.map(a => `
        <li class="action-item">
          <input type="checkbox">
          <span>${a}</span>
        </li>
      `).join('');
    }
  }

  clear() {
    this.topics = [];
    this.actions = [];
    this._render();
  }

  getTopics() {
    return this.topics;
  }

  getActions() {
    return this.actions;
  }
}

customElements.define('notes-view', NotesView);
