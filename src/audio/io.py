"""
Module D helpers — Text-to-Speech and Speech-to-Text.
"""

from __future__ import annotations

import re
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Optional heavy deps — import lazily where possible


class TextToSpeech:
    """Offline TTS via pyttsx3 (fallback: print-only if unavailable)."""

    def __init__(self, rate: int = 175, volume: float = 1.0) -> None:
        self._engine = None
        try:
            import pyttsx3

            self._engine = pyttsx3.init()
            self._engine.setProperty("rate", rate)
            self._engine.setProperty("volume", volume)
        except Exception:
            self._engine = None

    def speak(self, text: str, block: bool = False) -> None:
        if not text:
            return
        if self._engine is None:
            print(f"[TTS] {text}")
            return

        def _run() -> None:
            self._engine.say(text)
            self._engine.runAndWait()

        if block:
            _run()
        else:
            threading.Thread(target=_run, daemon=True).start()

    def speak_emergency(self, text: str = "Emergency. Medical help needed.") -> None:
        """Higher priority — blocking announce."""
        self.speak(text, block=True)


class SpeechToText:
    """
    Microphone transcription.
    Prefer Whisper if installed; else SpeechRecognition + Google (needs network).
    """

    def __init__(self, backend: str = "speechrecognition", whisper_model: str = "base") -> None:
        self.backend = backend
        self._whisper = None
        if backend == "whisper":
            try:
                import whisper

                self._whisper = whisper.load_model(whisper_model)
            except Exception as exc:
                raise RuntimeError(f"Whisper unavailable: {exc}") from exc

    def listen_once(self, timeout: float = 5.0, phrase_time_limit: float = 8.0) -> str:
        import speech_recognition as sr

        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.4)
            audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)

        if self.backend == "whisper" and self._whisper is not None:
            # Write temp WAV then decode — SpeechRecognition AudioData → WAV bytes
            import io
            import tempfile
            import wave

            raw = audio.get_wav_data()
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                f.write(raw)
                path = f.name
            result = self._whisper.transcribe(path)
            return (result.get("text") or "").strip()

        try:
            return recognizer.recognize_google(audio)
        except sr.UnknownValueError:
            return ""
        except sr.RequestError as exc:
            raise RuntimeError(f"STT request failed: {exc}") from exc


class SignVisualizer:
    """
    Map spoken/transcribed keywords → GIF / image assets under assets/signs/.
    File naming: water.gif, thank_you.gif, need_help.gif, etc.
    """

    def __init__(self, assets_dir: str | Path = "assets/signs") -> None:
        self.assets_dir = Path(assets_dir)
        self.assets_dir.mkdir(parents=True, exist_ok=True)
        self.index = self._build_index()

    def _build_index(self) -> Dict[str, Path]:
        index: Dict[str, Path] = {}
        for path in self.assets_dir.glob("*"):
            if path.suffix.lower() in {".gif", ".png", ".jpg", ".jpeg", ".webp", ".mp4"}:
                key = path.stem.lower().replace("-", "_").replace(" ", "_")
                index[key] = path
        return index

    def refresh(self) -> None:
        self.index = self._build_index()

    def match(self, text: str) -> List[Tuple[str, Path]]:
        """Return ordered (keyword, path) hits found in text."""
        text_l = text.lower()
        # Prefer longer keys first (thank_you before thank)
        keys = sorted(self.index.keys(), key=len, reverse=True)
        hits: List[Tuple[str, Path]] = []
        remaining = text_l
        for key in keys:
            pattern = key.replace("_", r"[\s_]+")
            if re.search(pattern, remaining):
                hits.append((key, self.index[key]))
                remaining = re.sub(pattern, " ", remaining)
        return hits
