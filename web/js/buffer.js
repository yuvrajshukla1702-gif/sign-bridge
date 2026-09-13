/** Sliding window + prediction debounce (mirrors Python Module B). */

export class SequenceBuffer {
  constructor(windowSize = 30, featureDim = 147) {
    this.windowSize = windowSize;
    this.featureDim = featureDim;
    this.buf = [];
  }

  push(features) {
    const arr = features instanceof Float32Array ? features : Float32Array.from(features);
    this.buf.push(arr);
    if (this.buf.length > this.windowSize) this.buf.shift();
  }

  ready() {
    return this.buf.length === this.windowSize;
  }

  getWindow() {
    if (!this.ready()) return null;
    return this.buf.map((f) => Array.from(f));
  }

  clear() {
    this.buf = [];
  }
}

export class PredictionDebouncer {
  constructor({ threshold = 0.85, minConsecutive = 10, cooldownFrames = 15 } = {}) {
    this.threshold = threshold;
    this.minConsecutive = minConsecutive;
    this.cooldownFrames = cooldownFrames;
    this.currentLabel = null;
    this.streak = 0;
    this.cooldown = 0;
    this.lastEmitted = null;
  }

  update(label, confidence) {
    if (this.cooldown > 0) {
      this.cooldown -= 1;
      return null;
    }
    if (confidence < this.threshold) {
      this.currentLabel = null;
      this.streak = 0;
      return null;
    }
    if (label === this.currentLabel) this.streak += 1;
    else {
      this.currentLabel = label;
      this.streak = 1;
    }
    if (this.streak >= this.minConsecutive) {
      if (label === this.lastEmitted) {
        this.streak = 0;
        return null;
      }
      this.lastEmitted = label;
      this.streak = 0;
      this.cooldown = this.cooldownFrames;
      return label;
    }
    return null;
  }

  reset() {
    this.currentLabel = null;
    this.streak = 0;
    this.cooldown = 0;
    this.lastEmitted = null;
  }
}

export class SentenceBuffer {
  constructor({ idleMs = 2000, eosTokens = ["EOS", "FULL_STOP", "DONE"] } = {}) {
    this.idleMs = idleMs;
    this.eosTokens = new Set(eosTokens);
    this.tokens = [];
    this.lastAdd = 0;
  }

  add(token) {
    if (this.eosTokens.has(token)) return this.flush();
    this.tokens.push(token);
    this.lastAdd = Date.now();
    return null;
  }

  checkIdle() {
    if (!this.tokens.length) return null;
    if (Date.now() - this.lastAdd >= this.idleMs) return this.flush();
    return null;
  }

  flush() {
    if (!this.tokens.length) return null;
    const out = [...this.tokens];
    this.tokens = [];
    return out;
  }
}
