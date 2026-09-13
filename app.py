"""
app.py — Streamlit dashboard for two-way Sign Language Communication Bridge.

Run:
  streamlit run app.py

Features:
  - Live webcam + landmark overlay
  - Sliding-window LSTM prediction with debounce
  - Sentence buffer + grammar polish + TTS
  - Voice → keyword → sign GIF panel
  - Emergency hotkeys (sidebar buttons)
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

ROOT = Path(__file__).resolve().parent
import sys

sys.path.insert(0, str(ROOT))

from src.audio.io import SignVisualizer, SpeechToText, TextToSpeech
from src.model.buffer import PredictionDebouncer, SequenceBuffer
from src.model.lstm_model import SignPredictor
from src.nlp.sentence import GrammarPolisher, SentenceBuffer
from src.tracking.landmarks import FEATURE_DIM, LandmarkExtractor
from src.ui.emergency import EmergencyController

st.set_page_config(
    page_title="Sign Language Bridge",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---- Dark-ish custom CSS ----
st.markdown(
    """
    <style>
    .stApp { background: linear-gradient(160deg, #0f1419 0%, #1a2332 50%, #0d1b2a 100%); }
    .flash-emergency {
        animation: pulse 0.6s ease-in-out infinite alternate;
        background: #8b0000; color: #fff; padding: 1rem 1.5rem;
        border-radius: 8px; font-size: 1.4rem; font-weight: 700; text-align: center;
    }
    @keyframes pulse { from { opacity: 0.7; } to { opacity: 1; } }
    .token-chip {
        display: inline-block; margin: 2px; padding: 4px 10px;
        background: #1b4332; color: #d8f3dc; border-radius: 6px; font-family: monospace;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_predictor():
    model_path = ROOT / "models" / "sign_lstm.pt"
    labels_path = ROOT / "models" / "labels.json"
    if not model_path.exists():
        return None, []
    labels = json.loads(labels_path.read_text(encoding="utf-8")) if labels_path.exists() else []
    if not labels:
        # Fallback from checkpoint
        import torch

        ckpt = torch.load(model_path, map_location="cpu")
        labels = ckpt.get("labels", [])
    return SignPredictor(model_path, labels), labels


def init_session() -> None:
    defaults = {
        "chat": [],
        "tokens": [],
        "last_sentence": "",
        "emergency_msg": None,
        "emergency_until": 0.0,
        "running": False,
        "voice_hits": [],
        "voice_text": "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def main() -> None:
    init_session()
    st.title("Sign Language ↔ Speech Bridge")
    st.caption("Real-time dynamic gestures · LLM sentence polish · two-way dialogue")

    with st.sidebar:
        st.header("Controls")
        camera_id = st.number_input("Camera index", 0, 4, 0)
        conf_th = st.slider("Confidence threshold", 0.5, 0.99, 0.85, 0.01)
        debounce_n = st.slider("Debounce frames", 5, 30, 10)
        idle_s = st.slider("Sentence idle (sec)", 1.0, 5.0, 2.0, 0.5)
        llm_backend = st.selectbox("Grammar backend", ["rule", "ollama", "groq"])
        st.divider()
        st.subheader("Emergency / Quick Actions")
        emergency = EmergencyController()
        tts = TextToSpeech()

        def trigger(action):
            st.session_state.emergency_msg = action.label
            st.session_state.emergency_until = time.time() + 4.0
            st.session_state.tokens.append(action.gloss)
            tts.speak_emergency(action.spoken)
            st.session_state.chat.append({"role": "emergency", "text": action.spoken})

        cols = st.columns(len(emergency.list_actions()))
        for col, action in zip(cols, emergency.list_actions()):
            if col.button(action.label, use_container_width=True):
                trigger(action)

        st.divider()
        run = st.toggle("Start webcam recognition", value=False)
        st.session_state.running = run

    # Emergency flash banner
    if st.session_state.emergency_msg and time.time() < st.session_state.emergency_until:
        st.markdown(
            f'<div class="flash-emergency">⚠ {st.session_state.emergency_msg}</div>',
            unsafe_allow_html=True,
        )

    col_video, col_side = st.columns([1.4, 1])

    predictor, labels = load_predictor()
    if predictor is None:
        st.warning(
            "No trained model at `models/sign_lstm.pt`. "
            "Collect data with `data_collection.py`, then run `train_model.py`."
        )

    with col_side:
        st.subheader("Detected word buffer")
        chips = " ".join(
            f'<span class="token-chip">{t}</span>' for t in st.session_state.tokens
        ) or "_empty_"
        st.markdown(chips, unsafe_allow_html=True)
        if st.button("Clear buffer"):
            st.session_state.tokens = []

        st.subheader("Chat history")
        for msg in st.session_state.chat[-12:]:
            role = msg["role"]
            st.markdown(f"**{role}:** {msg['text']}")

        st.divider()
        st.subheader("Speech → Sign")
        if st.button("Listen (microphone)"):
            try:
                stt = SpeechToText(backend="speechrecognition")
                with st.spinner("Listening…"):
                    text = stt.listen_once()
                st.session_state.voice_text = text
                viz = SignVisualizer(ROOT / "assets" / "signs")
                st.session_state.voice_hits = viz.match(text)
                if text:
                    st.session_state.chat.append({"role": "voice", "text": text})
            except Exception as exc:
                st.error(f"Microphone / STT error: {exc}")

        if st.session_state.voice_text:
            st.write(f"Heard: **{st.session_state.voice_text}**")
        if st.session_state.voice_hits:
            for key, path in st.session_state.voice_hits:
                st.caption(key)
                if path.suffix.lower() == ".gif":
                    st.image(str(path))
                elif path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
                    st.image(str(path))
                else:
                    st.write(path.name)
        else:
            st.info("Add GIF/PNG files under `assets/signs/` named like `water.gif`, `thank_you.gif`.")

    with col_video:
        st.subheader("Live camera")
        frame_slot = st.empty()
        pred_slot = st.empty()

        if not st.session_state.running:
            frame_slot.info("Enable **Start webcam recognition** in the sidebar.")
            return

        extractor = LandmarkExtractor()
        seq_buf = SequenceBuffer(window_size=30, feature_dim=FEATURE_DIM)
        debouncer = PredictionDebouncer(threshold=conf_th, min_consecutive=debounce_n)
        sent_buf = SentenceBuffer(idle_seconds=idle_s)
        # Sync tokens from session
        sent_buf.tokens = list(st.session_state.tokens)
        polisher = GrammarPolisher(backend=llm_backend)

        cap = cv2.VideoCapture(int(camera_id))
        if not cap.isOpened():
            st.error("Could not open webcam.")
            return

        # Limited loop so Streamlit stays responsive; user re-runs via toggle
        max_iters = 120
        try:
            for _ in range(max_iters):
                if not st.session_state.running:
                    break
                ok, frame = cap.read()
                if not ok:
                    break
                frame = cv2.flip(frame, 1)
                features, results = extractor.process(frame)
                vis = extractor.draw_landmarks(frame, results)
                seq_buf.push(features)

                label_txt = "…"
                if predictor is not None and seq_buf.ready():
                    window = seq_buf.get_window()
                    label, conf = predictor.predict(window)
                    label_txt = f"{label} ({conf:.2f})"
                    fired = debouncer.update(label, conf)
                    if fired:
                        eos = sent_buf.add(fired)
                        st.session_state.tokens = sent_buf.peek()
                        if eos:
                            sentence = polisher.polish(eos)
                            st.session_state.last_sentence = sentence
                            st.session_state.chat.append({"role": "sign", "text": sentence})
                            st.session_state.tokens = []
                            tts.speak(sentence)

                idle_flush = sent_buf.check_idle()
                if idle_flush:
                    sentence = polisher.polish(idle_flush)
                    st.session_state.last_sentence = sentence
                    st.session_state.chat.append({"role": "sign", "text": sentence})
                    st.session_state.tokens = []
                    tts.speak(sentence)

                rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
                frame_slot.image(rgb, channels="RGB", use_container_width=True)
                pred_slot.markdown(f"**Prediction:** `{label_txt}`")
                if st.session_state.last_sentence:
                    pred_slot.caption(f"Last sentence: {st.session_state.last_sentence}")
        finally:
            extractor.close()
            cap.release()


if __name__ == "__main__":
    main()
