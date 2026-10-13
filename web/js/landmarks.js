/**
 * Landmark feature extraction — must match Python src/tracking/landmarks.py
 * FEATURE_DIM = 147 (left hand 63 + right hand 63 + pose 21)
 */

export const FEATURE_DIM = 147;
export const HAND_LANDMARKS = 21;
export const POSE_SELECTED = [0, 11, 12, 13, 14, 15, 16];

/** aspect = frame width / height; must match src/engine/hand_features.py (hand63_aspect_v1). */
function normalizeHand(pts, aspect = 1) {
  if (!pts) return new Float32Array(HAND_LANDMARKS * 3);
  const wrist = pts[0];
  const centered = pts.map((p) => [
    (p[0] - wrist[0]) * aspect,
    p[1] - wrist[1],
    (p[2] - wrist[2]) * aspect,
  ]);
  let scale = 0;
  for (const p of centered) {
    const n = Math.hypot(p[0], p[1]);
    if (n > scale) scale = n;
  }
  if (scale < 1e-6) scale = 1;
  const out = new Float32Array(HAND_LANDMARKS * 3);
  let i = 0;
  for (const p of centered) {
    out[i++] = p[0] / scale;
    out[i++] = p[1] / scale;
    out[i++] = p[2] / scale;
  }
  return out;
}

function landmarksToArray(handLandmarks) {
  if (!handLandmarks) return null;
  return handLandmarks.map((lm) => [lm.x, lm.y, lm.z]);
}

/** Prefer right hand, else left — flat Float32Array(63). Works with Holistic or Hands results. */
export function extractPrimaryHand63(results, aspect = 1) {
  // Holistic
  let pts =
    landmarksToArray(results.rightHandLandmarks) ||
    landmarksToArray(results.leftHandLandmarks);
  // Hands solution
  if (!pts && results.multiHandLandmarks?.length) {
    const handed = results.multiHandedness || [];
    let idx = 0;
    for (let i = 0; i < handed.length; i++) {
      if ((handed[i].label || "").toLowerCase() === "right") {
        idx = i;
        break;
      }
    }
    pts = landmarksToArray(results.multiHandLandmarks[idx]);
  }
  if (!pts) return null;
  return normalizeHand(pts, aspect);
}

/** EMA smooth a 63-float hand vector */
export function smoothHand63(prev, next, alpha = 0.45) {
  if (!next) return prev;
  if (!prev) return Float32Array.from(next);
  const out = new Float32Array(next.length);
  for (let i = 0; i < next.length; i++) {
    out[i] = prev[i] * (1 - alpha) + next[i] * alpha;
  }
  return out;
}

function normalizePose(poseLandmarks) {
  const out = new Float32Array(POSE_SELECTED.length * 3);
  if (!poseLandmarks) return out;
  const pts = POSE_SELECTED.map((idx) => {
    const lm = poseLandmarks[idx];
    return [lm.x, lm.y, lm.z];
  });
  const mid = [
    (pts[1][0] + pts[2][0]) / 2,
    (pts[1][1] + pts[2][1]) / 2,
    (pts[1][2] + pts[2][2]) / 2,
  ];
  const centered = pts.map((p) => [p[0] - mid[0], p[1] - mid[1], p[2] - mid[2]]);
  let scale = 0;
  for (const p of centered) {
    const n = Math.hypot(p[0], p[1]);
    if (n > scale) scale = n;
  }
  if (scale < 1e-6) scale = 1;
  let i = 0;
  for (const p of centered) {
    out[i++] = p[0] / scale;
    out[i++] = p[1] / scale;
    out[i++] = p[2] / scale;
  }
  return out;
}

/** Build Float32Array length FEATURE_DIM from MediaPipe Holistic results. */
export function extractFeatures(results) {
  const left = normalizeHand(landmarksToArray(results.leftHandLandmarks));
  const right = normalizeHand(landmarksToArray(results.rightHandLandmarks));
  const pose = normalizePose(results.poseLandmarks);
  const features = new Float32Array(FEATURE_DIM);
  features.set(left, 0);
  features.set(right, HAND_LANDMARKS * 3);
  features.set(pose, HAND_LANDMARKS * 3 * 2);
  return features;
}
