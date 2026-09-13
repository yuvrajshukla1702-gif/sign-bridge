/**
 * Local Learn Mode coach — same scoring idea as Python FeedbackEvaluator,
 * runs on-device so match % updates every frame (no API lag).
 */

const TIP_INDEX = {
  4: "thumb tip",
  8: "index tip",
  12: "middle tip",
  16: "ring tip",
  20: "pinky tip",
};

export const HOLD_MS = 1100;
export const PASS_PCT = 72;

/** @type {Record<string, Float32Array>} */
let refs = {};

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
  if (refs[raw.toUpperCase()]) return raw.toUpperCase();
  if (refs[raw.toLowerCase()]) return raw.toLowerCase();
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

  // Progressive step unlock for the how-to checklist
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

/** Hold-to-master accumulator */
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

  /** @returns {{ progress: number, done: boolean, holding: boolean }} */
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
