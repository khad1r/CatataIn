// opfs-storage.js - Origin Private File System Model Cache Manager (Ponytail: zero external dependencies)

export class OpfsModelStorage {
  constructor() {
    this.root = null;
    this.modelsDir = null;
    this.hasOpfs = typeof navigator !== 'undefined' && !!navigator.storage?.getDirectory;
  }

  async init() {
    if (this.hasOpfs && !this.modelsDir) {
      try {
        this.root = await navigator.storage.getDirectory();
        this.modelsDir = await this.root.getDirectoryHandle('models', { create: true });
      } catch (err) {
        console.warn("OPFS getDirectory error, falling back to IndexedDB:", err);
        this.hasOpfs = false;
      }
    }
  }

  async hasModel(filename) {
    await this.init();
    if (this.hasOpfs && this.modelsDir) {
      try {
        await this.modelsDir.getFileHandle(filename);
        return true;
      } catch {
        return false;
      }
    } else {
      return await this._idbHas(filename);
    }
  }

  async saveModel(filename, arrayBuffer) {
    await this.init();
    if (this.hasOpfs && this.modelsDir) {
      const fileHandle = await this.modelsDir.getFileHandle(filename, { create: true });
      const writable = await fileHandle.createWritable();
      await writable.write(arrayBuffer);
      await writable.close();
    } else {
      await this._idbSave(filename, arrayBuffer);
    }
  }

  async getModel(filename) {
    await this.init();
    if (this.hasOpfs && this.modelsDir) {
      const fileHandle = await this.modelsDir.getFileHandle(filename);
      const file = await fileHandle.getFile();
      return await file.arrayBuffer();
    } else {
      return await this._idbGet(filename);
    }
  }

  async listModels() {
    await this.init();
    const result = [];
    if (this.hasOpfs && this.modelsDir) {
      for await (const [name, handle] of this.modelsDir.entries()) {
        if (handle.kind === 'file') {
          result.push(name);
        }
      }
    }
    return result;
  }

  // --- IndexedDB Fallback (only if OPFS is not supported) ---
  _openIdb() {
    return new Promise((resolve, reject) => {
      const req = indexedDB.open('catetin_opfs_fallback', 1);
      req.onupgradeneeded = () => {
        req.result.createObjectStore('models');
      };
      req.onsuccess = () => resolve(req.result);
      req.onerror = () => reject(req.error);
    });
  }

  async _idbHas(filename) {
    const db = await this._openIdb();
    return new Promise((resolve) => {
      const tx = db.transaction('models', 'readonly');
      const store = tx.objectStore('models');
      const req = store.count(filename);
      req.onsuccess = () => resolve(req.result > 0);
      req.onerror = () => resolve(false);
    });
  }

  async _idbSave(filename, arrayBuffer) {
    const db = await this._openIdb();
    return new Promise((resolve, reject) => {
      const tx = db.transaction('models', 'readwrite');
      const store = tx.objectStore('models');
      store.put(arrayBuffer, filename);
      tx.oncomplete = () => resolve();
      tx.onerror = () => reject(tx.error);
    });
  }

  async _idbGet(filename) {
    const db = await this._openIdb();
    return new Promise((resolve, reject) => {
      const tx = db.transaction('models', 'readonly');
      const store = tx.objectStore('models');
      const req = store.get(filename);
      req.onsuccess = () => resolve(req.result);
      req.onerror = () => reject(req.error);
    });
  }
}
