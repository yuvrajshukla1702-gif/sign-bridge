"""
Sign Bridge — FastAPI backend for the web app.

Serves the frontend and exposes:
  GET  /api/health
  GET  /api/labels
  GET  /api/signs
  POST /api/predict   { "window": [[...147], ...] x 30 }
  POST /api/polish    { "tokens": ["I", "go", "hospital"] }
  POST /api/emergency { "key": "e" }
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from src.nlp.sentence import GrammarPolisher  # noqa: E402
from src.tracking.constants import FEATURE_DIM  # noqa: E402
from src.ui.emergency import DEFAULT_ACTIONS  # noqa: E402
from backend.livereload import router as livereload_router  # noqa: E402
from backend.auth import router as auth_router  # noqa: E402

WEB_DIR = ROOT / "web"
ASSETS_DIR = ROOT / "assets"
MODELS_DIR = ROOT / "models"
WINDOW_SIZE = 30

app = FastAPI(title="Sign Bridge API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(livereload_router)
app.include_router(auth_router)


@app.middleware("http")
async def disable_cache_for_ui(request, call_next):
    """So Cursor edits to HTML/CSS/JS show up immediately in the desktop app."""
    response = await call_next(request)
    path = request.url.path
    if (
        path == "/"
        or path.startswith("/css/")
        or path.startswith("/js/")
        or path.startswith("/assets/signs_dict/")
        or path.endswith(".html")
    ):
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
    return response

# ---------- Model state ----------
_predictor = None
_labels: List[str] = []
_model_loaded = False
_load_error: Optional[str] = None


def _default_labels() -> List[str]:
    return [
        "hello",
        "thank_you",
        "yes",
        "no",
        "water",
        "need_help",
        "hospital",
        "pain",
        "family",
        "emergency",
        "medical_help",
        "EOS",
    ]


def load_model() -> None:
    global _predictor, _labels, _model_loaded, _load_error
    model_path = MODELS_DIR / "sign_lstm.pt"
    labels_path = MODELS_DIR / "labels.json"

    if labels_path.exists():
        _labels = json.loads(labels_path.read_text(encoding="utf-8"))
    else:
        _labels = _default_labels()

    if not model_path.exists():
        _predictor = None
        _model_loaded = False
        _load_error = "No trained model yet — use Practice mode or train with train_model.py"
        return

    try:
        from src.model.lstm_model import SignPredictor

        _predictor = SignPredictor(model_path, _labels)
        _model_loaded = True
        _load_error = None
    except Exception as exc:  # noqa: BLE001
        _predictor = None
        _model_loaded = False
        _load_error = str(exc)


load_model()
_polisher = GrammarPolisher(backend="rule")

from src.engine.feedback_evaluator import FeedbackEvaluator  # noqa: E402
from src.engine.static_classifier import StaticClassifier  # noqa: E402

_static_clf = StaticClassifier()
_feedback = FeedbackEvaluator()


# ---------- Schemas ----------
class PredictRequest(BaseModel):
    window: List[List[float]] = Field(..., description="shape (30, feature_dim)")


class PredictResponse(BaseModel):
    label: str
    confidence: float
    model_loaded: bool
    message: Optional[str] = None


class PolishRequest(BaseModel):
    tokens: List[str]
    backend: str = "rule"


class PolishResponse(BaseModel):
    sentence: str
    tokens: List[str]


class EmergencyRequest(BaseModel):
    key: str = "e"


class EmergencyResponse(BaseModel):
    label: str
    spoken: str
    gloss: str


class StaticPredictRequest(BaseModel):
    hand: List[float] = Field(..., description="63 normalized hand coords")


class EvaluateRequest(BaseModel):
    target: str
    hand: List[float]


# ---------- API ----------
@app.get("/api/health")
def health() -> Dict[str, Any]:
    return {
        "ok": True,
        "app": "Sign Bridge",
        "modes": ["learn", "bridge"],
        "model_loaded": _model_loaded,
        "static_ready": _static_clf.ready,
        "labels": _labels,
        "static_labels": _static_clf.list_labels(),
        "feature_dim": FEATURE_DIM,
        "window_size": WINDOW_SIZE,
        "message": _load_error,
    }


@app.get("/api/labels")
def labels() -> Dict[str, Any]:
    return {"labels": _labels, "model_loaded": _model_loaded}


@app.post("/api/reload-model")
def reload_model() -> Dict[str, Any]:
    load_model()
    global _static_clf, _feedback
    _static_clf = StaticClassifier()
    _feedback = FeedbackEvaluator()
    return {
        "model_loaded": _model_loaded,
        "static_ready": _static_clf.ready,
        "labels": _labels,
        "message": _load_error,
    }


@app.post("/api/predict", response_model=PredictResponse)
def predict(body: PredictRequest) -> PredictResponse:
    if not _model_loaded or _predictor is None:
        return PredictResponse(
            label="—",
            confidence=0.0,
            model_loaded=False,
            message=_load_error or "Model not loaded",
        )

    arr = np.asarray(body.window, dtype=np.float32)
    if arr.ndim != 2:
        raise HTTPException(400, "window must be 2D (T, F)")
    if arr.shape[0] != WINDOW_SIZE:
        raise HTTPException(400, f"Expected {WINDOW_SIZE} frames, got {arr.shape[0]}")
    if arr.shape[1] != FEATURE_DIM:
        if arr.shape[1] < FEATURE_DIM:
            pad = np.zeros((arr.shape[0], FEATURE_DIM - arr.shape[1]), dtype=np.float32)
            arr = np.concatenate([arr, pad], axis=1)
        else:
            arr = arr[:, :FEATURE_DIM]

    label, conf = _predictor.predict(arr)
    return PredictResponse(label=label, confidence=conf, model_loaded=True)


@app.post("/api/predict-static")
def predict_static(body: StaticPredictRequest) -> Dict[str, Any]:
    if not _static_clf.ready:
        return {"label": "—", "confidence": 0.0, "ready": False}
    label, conf = _static_clf.predict(np.asarray(body.hand, dtype=np.float32))
    return {"label": label, "confidence": conf, "ready": True}


@app.post("/api/evaluate")
def evaluate(body: EvaluateRequest) -> Dict[str, Any]:
    return _feedback.evaluate(body.target, np.asarray(body.hand, dtype=np.float32))


@app.post("/api/polish", response_model=PolishResponse)
def polish(body: PolishRequest) -> PolishResponse:
    polisher = GrammarPolisher(backend=body.backend or "rule")
    sentence = polisher.polish(body.tokens)
    return PolishResponse(sentence=sentence, tokens=body.tokens)


@app.post("/api/emergency", response_model=EmergencyResponse)
def emergency(body: EmergencyRequest) -> EmergencyResponse:
    action = DEFAULT_ACTIONS.get(body.key.lower())
    if action is None:
        raise HTTPException(404, f"Unknown emergency key: {body.key}")
    return EmergencyResponse(label=action.label, spoken=action.spoken, gloss=action.gloss)


@app.get("/api/signs")
def list_signs() -> Dict[str, Any]:
    signs_dir = ASSETS_DIR / "signs"
    signs_dir.mkdir(parents=True, exist_ok=True)
    items = []
    for path in sorted(signs_dir.iterdir()):
        if path.suffix.lower() in {".gif", ".png", ".jpg", ".jpeg", ".webp", ".mp4"}:
            items.append(
                {
                    "key": path.stem.lower().replace("-", "_").replace(" ", "_"),
                    "url": f"/assets/signs/{path.name}",
                    "name": path.name,
                }
            )
    return {"signs": items}


@app.get("/api/dictionary")
def dictionary(q: str = "", category: str = "all") -> Dict[str, Any]:
    """Searchable sign dictionary: alphabet, numbers, phrases."""
    dict_root = ASSETS_DIR / "signs_dict"
    items: List[Dict[str, Any]] = []
    cats = ["alphabet", "numbers", "phrases"] if category == "all" else [category]
    for cat in cats:
        folder = dict_root / cat
        if not folder.exists():
            continue
        for path in sorted(folder.iterdir()):
            if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".gif", ".webp"}:
                continue
            key = path.stem
            if q and q.lower() not in key.lower() and q.lower() not in key.replace("_", " ").lower():
                continue
            items.append(
                {
                    "key": key,
                    "label": key.replace("_", " "),
                    "category": cat,
                    "url": f"/assets/signs_dict/{cat}/{path.name}",
                }
            )

    # Phrase assets often live under assets/signs/ (hello.png, thank_you.png, …)
    if category in ("all", "phrases"):
        signs_dir = ASSETS_DIR / "signs"
        sent_path = ROOT / "data" / "references" / "sentences.json"
        phrase_meta = {}
        if sent_path.exists():
            try:
                phrase_meta = json.loads(sent_path.read_text(encoding="utf-8")).get("phrases", {})
            except Exception:
                phrase_meta = {}
        existing = {it["key"] for it in items}
        if signs_dir.exists():
            for path in sorted(signs_dir.iterdir()):
                if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".gif", ".webp"}:
                    continue
                key = path.stem
                if key.lower() == "readme":
                    continue
                if key in existing:
                    continue
                label = phrase_meta.get(key, {}).get("title") or key.replace("_", " ")
                if q and q.lower() not in key.lower() and q.lower() not in label.lower():
                    continue
                items.append(
                    {
                        "key": key,
                        "label": label,
                        "category": "phrases",
                        "url": f"/assets/signs/{path.name}",
                    }
                )
                existing.add(key)

    return {"items": items, "count": len(items)}


@app.get("/api/instructions/{sign_key}")
def instructions(sign_key: str) -> Dict[str, Any]:
    """How-to steps for forming a letter/number/phrase (teaching content)."""
    key = sign_key.strip()

    # Phrase curriculum first (thank_you, water, …)
    sent_path = ROOT / "data" / "references" / "sentences.json"
    if sent_path.exists():
        try:
            sent = json.loads(sent_path.read_text(encoding="utf-8"))
            phrases = sent.get("phrases", {})
            pk = key if key in phrases else key.lower().replace(" ", "_")
            if pk in phrases:
                info = phrases[pk]
                return {
                    "key": pk,
                    "title": info.get("title", pk),
                    "steps": info.get("steps", []),
                    "remember": info.get("remember", ""),
                    "image_url": info.get("image") or f"/assets/signs/{pk}.png",
                    "kind": "phrase",
                }
        except Exception:
            pass

    path = ROOT / "data" / "references" / "sign_instructions.json"
    if not path.exists():
        raise HTTPException(404, "Instructions file missing — run scripts/generate_sign_instructions.py")
    data = json.loads(path.read_text(encoding="utf-8"))
    if key not in data:
        key_up = key.upper()
        key_low = key.lower()
        if key_up in data:
            key = key_up
        elif key_low in data:
            key = key_low
        else:
            raise HTTPException(404, f"No instructions for '{sign_key}'")
    info = data[key]
    folder = "alphabet" if key.isalpha() and len(key) == 1 else "numbers"
    return {
        "key": key,
        "title": info.get("title", key),
        "steps": info.get("steps", []),
        "remember": info.get("remember", ""),
        "image_url": f"/assets/signs_dict/{folder}/{key}.png",
        "kind": "static",
    }


@app.get("/api/references")
def references() -> Dict[str, Any]:
    """Canonical hand landmarks for on-device Learn Mode scoring (low latency)."""
    path = ROOT / "data" / "references" / "static_landmarks.json"
    if not path.exists():
        raise HTTPException(404, "static_landmarks.json missing")
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        "version": data.get("version", 1),
        "feature_dim": data.get("feature_dim", 63),
        "landmarks": data.get("landmarks", {}),
    }


@app.get("/api/sentences")
def sentences() -> Dict[str, Any]:
    """Sentence / phrase curriculum for Learning Hub."""
    path = ROOT / "data" / "references" / "sentences.json"
    if not path.exists():
        raise HTTPException(404, "sentences.json missing")
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        "version": data.get("version", 1),
        "phrases": data.get("phrases", {}),
        "sentences": data.get("sentences", []),
    }

# ---------- Static mounts (after API routes) ----------
@app.get("/")
def index() -> FileResponse:
    index_path = WEB_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(404, "web/index.html missing")
    return FileResponse(index_path)


if (WEB_DIR / "css").exists():
    app.mount("/css", StaticFiles(directory=str(WEB_DIR / "css")), name="css")
if (WEB_DIR / "js").exists():
    app.mount("/js", StaticFiles(directory=str(WEB_DIR / "js")), name="js")
if (ASSETS_DIR / "signs").exists():
    app.mount("/assets/signs", StaticFiles(directory=str(ASSETS_DIR / "signs")), name="signs")
if (ASSETS_DIR / "signs_dict").exists():
    app.mount(
        "/assets/signs_dict",
        StaticFiles(directory=str(ASSETS_DIR / "signs_dict")),
        name="signs_dict",
    )