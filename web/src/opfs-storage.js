// opfs-storage.js - Origin Private File System Model & Cache Manager (Ponytail: native browser OPFS)

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

  async saveModel(filename, arrayBufferOrStream) {
    await this.init();
    if (this.hasOpfs && this.modelsDir) {
      const fileHandle = await this.modelsDir.getFileHandle(filename, { create: true });
      const writable = await fileHandle.createWritable();
      if (arrayBufferOrStream instanceof ReadableStream) {
        await arrayBufferOrStream.pipeTo(writable);
      } else {
        await writable.write(arrayBufferOrStream);
        await writable.close();
      }
    } else {
      const buffer = arrayBufferOrStream instanceof ArrayBuffer
        ? arrayBufferOrStream
        : await new Response(arrayBufferOrStream).arrayBuffer();
      await this._idbSave(filename, buffer);
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

  async deleteModel(filename) {
    await this.init();
    if (this.hasOpfs && this.modelsDir) {
      try {
        await this.modelsDir.removeEntry(filename);
        return true;
      } catch {
        return false;
      }
    } else {
      return await this._idbDelete(filename);
    }
  }

  async listModels() {
    await this.init();
    const result = [];
    if (this.hasOpfs && this.modelsDir) {
      for await (const [name, handle] of this.modelsDir.entries()) {
        if (handle.kind === 'file') {
          const file = await handle.getFile();
          result.push({ name, size: file.size, lastModified: file.lastModified });
        }
      }
    }
    return result;
  }

  async importFromDirectory(dirHandle) {
    await this.init();
    const imported = [];
    for await (const [name, handle] of dirHandle.entries()) {
      if (handle.kind === 'file' && (name.endsWith('.onnx') || name.endsWith('.bin') || name.endsWith('.json') || name.endsWith('.txt'))) {
        const file = await handle.getFile();
        const buffer = await file.arrayBuffer();
        await this.saveModel(name, buffer);
        imported.push(name);
      }
    }
    return imported;
  }

  async getStorageUsage() {
    if (navigator.storage && navigator.storage.estimate) {
      const est = await navigator.storage.estimate();
      return {
        usage: est.usage || 0,
        quota: est.quota || 0,
        usageMB: ((est.usage || 0) / (1024 * 1024)).toFixed(1),
        quotaMB: ((est.quota || 0) / (1024 * 1024)).toFixed(1),
      };
    }
    return null;
  }

  // --- IndexedDB Fallback (only for older browsers without OPFS) ---
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

  async _idbDelete(filename) {
    const db = await this._openIdb();
    return new Promise((resolve, reject) => {
      const tx = db.transaction('models', 'readwrite');
      const store = tx.objectStore('models');
      const req = store.delete(filename);
      req.onsuccess = () => resolve(true);
      req.onerror = () => resolve(false);
    });
  }
}
