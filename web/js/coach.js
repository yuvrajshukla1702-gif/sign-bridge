/**
 * Local Learn Mode coach.
 * Letters (and phrases built on a letter handshape) are scored by the trained
 * A–Z MLP from /api/static-model: match_pct = P(target) * 100 — identical to
 * the backend's src/engine/feedback_evaluator.py. Template poses only provide
 * joint "nudge" hints, and score the phrases the model cannot cover.
 */

const FINGERS = ["thumb", "index", "middle", "ring", "pinky"];
const PARTS = ["base", "knuckle", "middle joint", "tip"];
const TIP_INDEX = { 0: "wrist" };
for (let i = 1; i <= 20; i++) TIP_INDEX[i] = `${FINGERS[Math.floor((i - 1) / 4)]} ${PARTS[(i - 1) % 4]}`;

export const HOLD_MS = 1100;
export const PASS_PCT = 70;

/** @type {Record<string, Float32Array>} */
let refs = {};

/** @type {{labels: string[], layers: {W: Float32Array, b: Float32Array, nIn: number, nOut: number}[], phrase: Record<string,string>} | null} */
let model = null;

function packLayers(layers) {
  return layers.map((L) => ({
    W: Float32Array.from(L.W.flat()),
    b: Float32Array.from(L.b),
    nOut: L.W.length,
    nIn: L.W[0].length,
  }));
}

export function loadStaticModel(json) {
  if (!json?.layers?.length) return false;
  model = {
    labels: json.labels.map(String),
    inputDim: json.input_dim || 63,
    phrase: json.phrase_handshapes || {},
    exemplars: Object.fromEntries(
      Object.entries(json.exemplars || {}).map(([k, rows]) => [k, rows.map((r) => Float32Array.from(r))])
    ),
    layers: packLayers(json.layers),
    digits: json.digits?.layers?.length
      ? {
          labels: json.digits.labels.map(String),
          inputDim: json.digits.input_dim || 63,
          layers: packLayers(json.digits.layers),
        }
      : null,
  };
  return true;
}

export function hasStaticModel() {
  return !!model;
}

const CHAINS = [
  [0, 1, 2, 3, 4],
  [0, 5, 6, 7, 8],
  [0, 9, 10, 11, 12],
  [0, 13, 14, 15, 16],
  [0, 17, 18, 19, 20],
];
const TIPS = [4, 8, 12, 16, 20];

/** Mirrors src/engine/hand_features.py model_features: 63 coords + 15 bend cosines + 10 tip distances. */
function modelFeatures(h) {
  const out = new Float32Array(88);
  out.set(h.subarray ? h.subarray(0, 63) : h.slice(0, 63));
  const pt = (i) => [h[i * 3], h[i * 3 + 1], h[i * 3 + 2]];
  let k = 63;
  for (const ch of CHAINS) {
    for (let j = 1; j < 4; j++) {
      const a = pt(ch[j - 1]), b = pt(ch[j]), c = pt(ch[j + 1]);
      const u = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
      const v = [c[0] - b[0], c[1] - b[1], c[2] - b[2]];
      out[k++] = (u[0] * v[0] + u[1] * v[1] + u[2] * v[2]) / (Math.hypot(...u) * Math.hypot(...v) + 1e-6);
    }
  }
  for (let i = 0; i < 5; i++) {
    for (let j = i + 1; j < 5; j++) {
      const a = pt(TIPS[i]), b = pt(TIPS[j]);
      out[k++] = Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
    }
  }
  return out;
}

function modelProba(x, net = model) {
  let a = net.inputDim > 63 ? modelFeatures(x) : x;
  net.layers.forEach((L, li) => {
    const out = new Float32Array(L.nOut);
    for (let o = 0; o < L.nOut; o++) {
      let s = L.b[o];
      const row = o * L.nIn;
      for (let i = 0; i < L.nIn; i++) s += L.W[row + i] * a[i];
      out[o] = li < net.layers.length - 1 && s < 0 ? 0 : s;
    }
    a = out;
  });
  let max = -Infinity;
  for (const v of a) if (v > max) max = v;
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    a[i] = Math.exp(a[i] - max);
    sum += a[i];
  }
  for (let i = 0; i < a.length; i++) a[i] /= sum;
  return a;
}

function modelLabelFor(target) {
  if (!model) return null;
  const t = String(target).trim();
  if (t.length === 1 && model.digits?.labels.includes(t)) return t;
  if (t.length === 1 && model.labels.includes(t.toUpperCase())) return t.toUpperCase();
  return model.phrase[t.toLowerCase()] || null;
}

/** Top-1 prediction from the trained model, or null if not loaded. */
export function predictLetter(hand63) {
  if (!model || !hand63) return null;
  const p = modelProba(hand63);
  let best = 0;
  for (let i = 1; i < p.length; i++) if (p[i] > p[best]) best = i;
  return { label: model.labels[best], conf: p[best] };
}

export function loadRefs(landmarksMap) {
  refs = {};
  for (const [k, v] of Object.entries(landmarksMap || {})) {
    refs[k] = Float32Array.from(v);
  }
  return Object.keys(refs).length;
}

export function hasRefs() {
  return Object.keys(refs).length > 0;
}

function resolveKey(target) {
  const raw = String(target).trim();
  if (refs[raw]) return raw;
  if (raw.length === 1 && refs[raw.toUpperCase()]) return raw.toUpperCase();
  if (refs[raw.toLowerCase()]) return raw.toLowerCase();
  if (refs[raw.toUpperCase()]) return raw.toUpperCase();
  return raw;
}

/**
 * @returns {{
 *   match_pct: number,
 *   tip: string,
 *   ok: boolean,
 *   worst: Array<{index:number,name:string,error:number}>,
 *   jointErrors: Float32Array,
 *   stepUnlock: number
 * }}
 */
export function evaluateLocal(target, hand63) {
  const key = resolveKey(target);
  const ref = refs[key];
  const mLabel = modelLabelFor(target);
  if (mLabel && hand63) return evaluateWithModel(mLabel, key, ref, hand63);
  if (!ref || !hand63) {
    return {
      match_pct: 0,
      tip: `No reference for “${target}”.`,
      ok: false,
      worst: [],
      jointErrors: new Float32Array(21),
      stepUnlock: 0,
    };
  }

  const live = hand63.length >= 63 ? hand63 : Float32Array.from(hand63);
  const livePts = [];
  const refPts = [];
  for (let i = 0; i < 21; i++) {
    livePts.push([live[i * 3], live[i * 3 + 1], live[i * 3 + 2]]);
    refPts.push([ref[i * 3], ref[i * 3 + 1], ref[i * 3 + 2]]);
  }

  const dists = new Float32Array(21);
  let meanD = 0;
  for (let i = 0; i < 21; i++) {
    const dx = livePts[i][0] - refPts[i][0];
    const dy = livePts[i][1] - refPts[i][1];
    const dz = livePts[i][2] - refPts[i][2];
    dists[i] = Math.hypot(dx, dy, dz);
    meanD += dists[i];
  }
  meanD /= 21;

  let match = Math.max(0, Math.min(100, (1 - meanD / 0.72) * 100));
  let normL = 0;
  let normR = 0;
  let dot = 0;
  for (let i = 0; i < 63; i++) {
    normL += live[i] * live[i];
    normR += ref[i] * ref[i];
    dot += live[i] * ref[i];
  }
  const cos = dot / (Math.sqrt(normL) * Math.sqrt(normR) + 1e-8);
  match = Math.max(0, Math.min(100, 0.35 * match + 0.65 * ((cos + 1) / 2) * 100));
  match = Math.min(100, match * 1.12);

  const order = [...Array(21).keys()].sort((a, b) => dists[b] - dists[a]);
  const worst = [];
  const tips = [];
  for (const i of order.slice(0, 3)) {
    const name = TIP_INDEX[i] || `joint ${i}`;
    worst.push({ index: i, name, error: dists[i] });
    const deltaY = livePts[i][1] - refPts[i][1];
    const deltaX = livePts[i][0] - refPts[i][0];
    if (dists[i] > 0.08) {
      const direction =
        Math.abs(deltaY) > Math.abs(deltaX)
          ? deltaY < 0
            ? "higher"
            : "lower"
          : deltaX > 0
            ? "outward"
            : "inward";
      tips.push(`Nudge your ${name} ${direction}.`);
    }
  }

  let tip;
  if (match >= 90) tip = `Locked in — hold still (${match.toFixed(0)}%).`;
  else if (match >= PASS_PCT) tip = tips[0] || `Almost perfect (${match.toFixed(0)}%). Hold…`;
  else if (match >= 50) tip = tips[0] || `Getting closer (${match.toFixed(0)}%). Curl / uncurl fingers.`;
  else tip = tips[0] || `Reshape toward “${key}” (${match.toFixed(0)}%).`;

  let stepUnlock = 0;
  if (match >= 28) stepUnlock = 1;
  if (match >= 48) stepUnlock = 2;
  if (match >= PASS_PCT) stepUnlock = 3;

  return {
    match_pct: Math.round(match * 10) / 10,
    tip,
    ok: match >= PASS_PCT,
    worst,
    jointErrors: dists,
    stepUnlock,
  };
}

function jointHints(ref, live) {
  const dists = new Float32Array(21);
  const worst = [];
  const tips = [];
  if (!ref) return { dists, worst, tips };
  for (let i = 0; i < 21; i++) {
    dists[i] = Math.hypot(
      live[i * 3] - ref[i * 3],
      live[i * 3 + 1] - ref[i * 3 + 1],
      live[i * 3 + 2] - ref[i * 3 + 2]
    );
  }
  const order = [...Array(21).keys()].sort((a, b) => dists[b] - dists[a]);
  for (const i of order.slice(0, 3)) {
    const name = TIP_INDEX[i] || `joint ${i}`;
    worst.push({ index: i, name, error: dists[i] });
    const dx = live[i * 3] - ref[i * 3];
    const dy = live[i * 3 + 1] - ref[i * 3 + 1];
    if (dists[i] > 0.08) {
      const dir = Math.abs(dy) > Math.abs(dx) ? (dy < 0 ? "higher" : "lower") : dx > 0 ? "outward" : "inward";
      tips.push(`Nudge your ${name} ${dir}.`);
    }
  }
  return { dists, worst, tips };
}

function nearestExemplar(label, live) {
  const bank = model.exemplars[label];
  if (!bank?.length) return null;
  let best = null;
  let bestD = Infinity;
  for (const ex of bank) {
    let d = 0;
    for (let i = 0; i < 63; i++) d += (ex[i] - live[i]) ** 2;
    if (d < bestD) {
      bestD = d;
      best = ex;
    }
  }
  return best;
}

function evaluateWithModel(label, key, ref, hand63) {
  const net = model.digits?.labels.includes(label) ? model.digits : model;
  const p = modelProba(hand63, net);
  const names = net.labels;
  const match = p[names.indexOf(label)] * 100;
  let top = 0;
  for (let i = 1; i < p.length; i++) if (p[i] > p[top]) top = i;
  const topLabel = names[top];
  const { dists, worst, tips } = jointHints(nearestExemplar(label, hand63) || ref, hand63);

  let tip;
  if (match >= 90) tip = `Locked in — hold still (${match.toFixed(0)}%).`;
  else if (match >= PASS_PCT) tip = `Good ${label} (${match.toFixed(0)}%). Hold…`;
  else if (topLabel !== label && p[top] >= 0.4)
    tip = `That reads as “${topLabel}”. ${tips[0] || `Reshape toward “${label}”.`}`;
  else tip = tips[0] || `Reshape toward “${label}” (${match.toFixed(0)}%).`;

  let stepUnlock = 0;
  if (match >= 15) stepUnlock = 1;
  if (match >= 40) stepUnlock = 2;
  if (match >= PASS_PCT) stepUnlock = 3;

  return {
    match_pct: Math.round(match * 10) / 10,
    tip,
    ok: match >= PASS_PCT,
    worst,
    jointErrors: dists,
    stepUnlock,
    predicted: topLabel,
    predicted_conf: p[top],
    scorer: "trained_mlp",
  };
}

export class HoldTracker {
  constructor(needMs = HOLD_MS, passPct = PASS_PCT) {
    this.needMs = needMs;
    this.passPct = passPct;
    this.heldMs = 0;
    this.lastTs = 0;
    this.completed = false;
  }

  reset() {
    this.heldMs = 0;
    this.lastTs = 0;
    this.completed = false;
  }

  update(matchPct, now = performance.now()) {
    if (this.completed) return { progress: 1, done: true, holding: false };
    if (!this.lastTs) this.lastTs = now;
    const dt = Math.min(80, now - this.lastTs);
    this.lastTs = now;

    if (matchPct >= this.passPct) {
      this.heldMs += dt;
    } else {
      this.heldMs = Math.max(0, this.heldMs - dt * 1.6);
    }

    const progress = Math.min(1, this.heldMs / this.needMs);
    if (progress >= 1) {
      this.completed = true;
      return { progress: 1, done: true, holding: true };
    }
    return { progress, done: false, holding: matchPct >= this.passPct };
  }
}
