"""Quick Sign Bridge benchmark: accuracy + latency (ms)."""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

BASE = "http://127.0.0.1:8000"


def http_json(method: str, path: str, body=None, timeout=30):
    data = None
    headers = {"Content-Type": "application/json"}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(BASE + path, data=data, headers=headers, method=method)
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as res:
        raw = res.read()
        status = res.status
    ms = (time.perf_counter() - t0) * 1000
    payload = json.loads(raw.decode("utf-8")) if raw else {}
    return status, payload, ms


def timed_local(fn, repeats=50):
    # warmup
    for _ in range(5):
        fn()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    return {
        "mean_ms": round(statistics.mean(times), 3),
        "p50_ms": round(statistics.median(times), 3),
        "p95_ms": round(sorted(times)[int(0.95 * (len(times) - 1))], 3),
        "min_ms": round(min(times), 3),
        "max_ms": round(max(times), 3),
        "n": repeats,
    }


def wait_health(max_s=40):
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < max_s:
        try:
            st, payload, ms = http_json("GET", "/api/health")
            if st == 200 and payload.get("ok"):
                return payload, ms
        except Exception:
            time.sleep(0.4)
    raise RuntimeError("API health timeout")


def main():
    report = {"ok": True, "tests": {}}

    health, health_ms = wait_health()
    report["health"] = {
        "model_loaded": health.get("model_loaded"),
        "static_ready": health.get("static_ready"),
        "latency_ms": round(health_ms, 2),
        "message": health.get("message"),
    }

    # Load refs
    st, refs_payload, refs_ms = http_json("GET", "/api/references")
    landmarks = refs_payload.get("landmarks", {})
    keys = sorted(landmarks.keys())
    report["references"] = {
        "count": len(keys),
        "latency_ms": round(refs_ms, 2),
        "sample_keys": keys[:8],
    }

    from src.engine.feedback_evaluator import FeedbackEvaluator
    from src.engine.static_classifier import StaticClassifier

    fe = FeedbackEvaluator()
    sc = StaticClassifier()

    # --- Accuracy: self-match (alphabet + numbers only for A–Z target) ---
    glyph_keys = [k for k in keys if len(k) == 1]
    phrase_keys = [k for k in keys if len(k) > 1]
    self_scores = []
    self_ok = 0
    for k in glyph_keys:
        hand = np.asarray(landmarks[k], dtype=np.float32)
        out = fe.evaluate(k, hand)
        self_scores.append(out["match_pct"])
        if out.get("ok"):
            self_ok += 1
    report["accuracy_self_match"] = {
        "labels": len(glyph_keys),
        "mean_match_pct": round(float(np.mean(self_scores)), 2) if self_scores else 0,
        "min_match_pct": round(float(np.min(self_scores)), 2) if self_scores else 0,
        "max_match_pct": round(float(np.max(self_scores)), 2) if self_scores else 0,
        "pass_rate_pct": round(100.0 * self_ok / max(1, len(glyph_keys)), 2),
        "threshold": 78.0,
        "note": "A–Z / 0–9 reference pose vs itself (ideal upper bound)",
    }

    phrase_scores = []
    phrase_ok = 0
    for k in phrase_keys:
        hand = np.asarray(landmarks[k], dtype=np.float32)
        out = fe.evaluate(k, hand)
        phrase_scores.append(out["match_pct"])
        if out.get("ok"):
            phrase_ok += 1
    report["accuracy_phrase_self_match"] = {
        "labels": len(phrase_keys),
        "mean_match_pct": round(float(np.mean(phrase_scores)), 2) if phrase_scores else 0,
        "pass_rate_pct": round(100.0 * phrase_ok / max(1, len(phrase_keys)), 2),
        "threshold": 78.0,
        "note": "Lesson 2 phrase handshapes vs themselves",
    }

    # --- Accuracy: noisy self (small jitter) still passes ---
    rng = np.random.default_rng(42)
    noisy_ok = 0
    noisy_scores = []
    for k in glyph_keys:
        hand = np.asarray(landmarks[k], dtype=np.float32).copy()
        hand = hand + rng.normal(0, 0.02, size=hand.shape).astype(np.float32)
        out = fe.evaluate(k, hand)
        noisy_scores.append(out["match_pct"])
        if out.get("ok"):
            noisy_ok += 1
    report["accuracy_noisy_self"] = {
        "noise_sigma": 0.02,
        "mean_match_pct": round(float(np.mean(noisy_scores)), 2) if noisy_scores else 0,
        "pass_rate_pct": round(100.0 * noisy_ok / max(1, len(glyph_keys)), 2),
        "note": "Same pose + small landmark jitter",
    }

    # --- Accuracy: discrimination (wrong letter should score lower / fail) ---
    wrong_fail = 0
    pairs = 0
    gap = []
    alphabet = [k for k in glyph_keys if k.isalpha()]
    for i, a in enumerate(alphabet):
        b = alphabet[(i + 7) % len(alphabet)]
        if a == b:
            continue
        hand_a = np.asarray(landmarks[a], dtype=np.float32)
        right = fe.evaluate(a, hand_a)["match_pct"]
        wrong = fe.evaluate(b, hand_a)["match_pct"]
        gap.append(right - wrong)
        pairs += 1
        if wrong < 78:
            wrong_fail += 1
    report["accuracy_discrimination"] = {
        "pairs": pairs,
        "wrong_label_fail_rate_pct": round(100.0 * wrong_fail / max(1, pairs), 2),
        "mean_score_gap_right_minus_wrong": round(float(np.mean(gap)), 2) if gap else 0,
        "note": "Hand for letter A scored against unrelated letter; higher gap is better",
    }

    # --- Static classifier top-1 on A–Z / 0–9 only ---
    if sc.ready:
        top1 = 0
        for k in glyph_keys:
            hand = np.asarray(landmarks[k], dtype=np.float32)
            label, conf = sc.predict(hand)
            if str(label).upper() == str(k).upper() or str(label) == str(k):
                top1 += 1
        report["accuracy_static_classifier"] = {
            "top1_pct": round(100.0 * top1 / max(1, len(glyph_keys)), 2),
            "labels": len(glyph_keys),
            "ready": True,
            "scope": "alphabet+numbers",
        }
    else:
        report["accuracy_static_classifier"] = {"ready": False}

    # --- Latency: local evaluate ---
    sample_key = "A" if "A" in landmarks else keys[0]
    sample_hand = np.asarray(landmarks[sample_key], dtype=np.float32)
    report["latency_local_evaluate_ms"] = timed_local(lambda: fe.evaluate(sample_key, sample_hand), 200)

    # --- Latency: API evaluate ---
    api_times = []
    for _ in range(30):
        _, out, ms = http_json(
            "POST",
            "/api/evaluate",
            {"target": sample_key, "hand": sample_hand.tolist()},
        )
        api_times.append(ms)
    report["latency_api_evaluate_ms"] = {
        "mean_ms": round(statistics.mean(api_times), 2),
        "p50_ms": round(statistics.median(api_times), 2),
        "p95_ms": round(sorted(api_times)[int(0.95 * (len(api_times) - 1))], 2),
        "min_ms": round(min(api_times), 2),
        "max_ms": round(max(api_times), 2),
        "n": len(api_times),
        "last_match_pct": out.get("match_pct"),
    }

    # --- Latency: API predict-static ---
    ps_times = []
    for _ in range(20):
        _, out, ms = http_json("POST", "/api/predict-static", {"hand": sample_hand.tolist()})
        ps_times.append(ms)
    report["latency_api_predict_static_ms"] = {
        "mean_ms": round(statistics.mean(ps_times), 2),
        "p50_ms": round(statistics.median(ps_times), 2),
        "p95_ms": round(sorted(ps_times)[int(0.95 * (len(ps_times) - 1))], 2),
        "n": len(ps_times),
        "last": out,
    }

    # --- Latency: sequence predict (may be unloaded) ---
    window = np.zeros((30, 147), dtype=np.float32).tolist()
    pr_times = []
    last_pr = None
    for _ in range(15):
        _, last_pr, ms = http_json("POST", "/api/predict", {"window": window})
        pr_times.append(ms)
    report["latency_api_predict_sequence_ms"] = {
        "mean_ms": round(statistics.mean(pr_times), 2),
        "p50_ms": round(statistics.median(pr_times), 2),
        "p95_ms": round(sorted(pr_times)[int(0.95 * (len(pr_times) - 1))], 2),
        "n": len(pr_times),
        "model_loaded": last_pr.get("model_loaded") if isinstance(last_pr, dict) else None,
        "note": last_pr.get("message") if isinstance(last_pr, dict) else None,
    }

    # --- Latency: other endpoints ---
    for name, method, path, body in [
        ("health", "GET", "/api/health", None),
        ("dictionary", "GET", "/api/dictionary?category=alphabet", None),
        ("sentences", "GET", "/api/sentences", None),
        ("instructions_A", "GET", "/api/instructions/A", None),
        ("polish", "POST", "/api/polish", {"tokens": ["thank_you"], "backend": "rule"}),
    ]:
        times = []
        for _ in range(15):
            _, _, ms = http_json(method, path, body)
            times.append(ms)
        report["tests"][name] = {
            "mean_ms": round(statistics.mean(times), 2),
            "p50_ms": round(statistics.median(times), 2),
            "p95_ms": round(sorted(times)[int(0.95 * (len(times) - 1))], 2),
        }

    # Estimated end-to-end learn loop (local coach path used in UI)
    report["estimated_learn_loop"] = {
        "landmark_extract_note": "Browser MediaPipe Hands ~15–40ms/frame (device-dependent, not measured here)",
        "local_score_ms": report["latency_local_evaluate_ms"]["mean_ms"],
        "ui_throttle_ms": 40,
        "typical_feedback_interval_ms": "local score every frame; UI paint ~40ms",
        "hold_to_master_ms": 1100,
    }

    out_path = ROOT / "docs" / "BENCHMARK_RESULTS.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
