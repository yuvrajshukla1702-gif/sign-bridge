"""
Two-way assistive bridge: tokens ↔ polished speech ↔ sign visuals.
"""

from __future__ import annotations

from typing import List, Optional

from src.audio.io import SignVisualizer, TextToSpeech
from src.nlp.sentence import GrammarPolisher, SentenceBuffer
from src.ui.emergency import DEFAULT_ACTIONS, EmergencyAction, EmergencyController


class CommunicationBridge:
    def __init__(self, llm_backend: str = "rule", assets_dir: str = "assets/signs") -> None:
        self.sentence_buffer = SentenceBuffer(idle_seconds=2.0)
        self.polisher = GrammarPolisher(backend=llm_backend)
        self.tts = TextToSpeech()
        self.visualizer = SignVisualizer(assets_dir)
        self.emergency = EmergencyController()

    def add_token(self, token: str) -> Optional[List[str]]:
        return self.sentence_buffer.add(token)

    def check_idle(self) -> Optional[List[str]]:
        return self.sentence_buffer.check_idle()

    def polish(self, tokens: List[str], backend: Optional[str] = None) -> str:
        if backend:
            self.polisher = GrammarPolisher(backend=backend)
        return self.polisher.polish(tokens)

    def speak(self, text: str, emergency: bool = False) -> None:
        if emergency:
            self.tts.speak_emergency(text)
        else:
            self.tts.speak(text)

    def match_signs(self, text: str):
        return self.visualizer.match(text)

    def trigger_emergency(self, key: str) -> Optional[EmergencyAction]:
        return self.emergency.handle_key(key)

    def list_emergencies(self):
        return self.emergency.list_actions()
