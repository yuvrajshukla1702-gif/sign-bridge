"""
Module C — Sentence buffer + LLM grammar polishing.
Tokens accumulate until idle pause or end-of-sentence gesture.
"""

from __future__ import annotations

import os
import time
from typing import List, Optional

import requests


DEFAULT_SYSTEM_PROMPT = (
    "You convert sequences of American Sign Language gloss tokens into one "
    "natural, grammatically correct English sentence. "
    "Do not add facts that are not implied by the tokens. "
    "Reply with only the sentence."
)


class SentenceBuffer:
    """Accumulate detected sign tokens; flush on idle or EOS gesture."""

    def __init__(
        self,
        idle_seconds: float = 2.0,
        eos_tokens: Optional[List[str]] = None,
    ) -> None:
        self.idle_seconds = idle_seconds
        self.eos_tokens = set(eos_tokens or ["EOS", "FULL_STOP", "DONE"])
        self.tokens: List[str] = []
        self._last_add: float = 0.0

    def add(self, token: str) -> Optional[List[str]]:
        """
        Add a token. Returns the flushed token list if EOS gesture, else None.
        """
        if token in self.eos_tokens:
            return self.flush()
        self.tokens.append(token)
        self._last_add = time.time()
        return None

    def check_idle(self) -> Optional[List[str]]:
        """Call periodically; flush if idle pause exceeded and buffer non-empty."""
        if not self.tokens:
            return None
        if time.time() - self._last_add >= self.idle_seconds:
            return self.flush()
        return None

    def flush(self) -> Optional[List[str]]:
        if not self.tokens:
            return None
        out = list(self.tokens)
        self.tokens.clear()
        return out

    def peek(self) -> List[str]:
        return list(self.tokens)


class GrammarPolisher:
    """
    Turn gloss tokens into a natural sentence.
    Backends: 'ollama' (local), 'groq' (API), or 'rule' (offline fallback).
    """

    def __init__(
        self,
        backend: str = "rule",
        model: str = "llama3.2",
        groq_api_key: Optional[str] = None,
        ollama_url: str = "http://localhost:11434/api/generate",
    ) -> None:
        self.backend = backend.lower()
        self.model = model
        self.groq_api_key = groq_api_key or os.getenv("GROQ_API_KEY")
        self.ollama_url = ollama_url

    def polish(self, tokens: List[str]) -> str:
        if not tokens:
            return ""
        gloss = " ".join(tokens)
        if self.backend == "ollama":
            return self._ollama(gloss)
        if self.backend == "groq":
            return self._groq(gloss)
        return self._rule_based(tokens)

    def _prompt(self, gloss: str) -> str:
        return f'{DEFAULT_SYSTEM_PROMPT}\n\nTokens: [{gloss}]\nSentence:'

    def _ollama(self, gloss: str) -> str:
        try:
            r = requests.post(
                self.ollama_url,
                json={"model": self.model, "prompt": self._prompt(gloss), "stream": False},
                timeout=30,
            )
            r.raise_for_status()
            return r.json().get("response", gloss).strip().strip('"')
        except Exception:
            return self._rule_based(gloss.split())

    def _groq(self, gloss: str) -> str:
        if not self.groq_api_key:
            return self._rule_based(gloss.split())
        try:
            r = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.groq_api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model or "llama-3.1-8b-instant",
                    "messages": [
                        {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
                        {"role": "user", "content": f"Tokens: {gloss}"},
                    ],
                    "temperature": 0.2,
                },
                timeout=30,
            )
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"].strip().strip('"')
        except Exception:
            return self._rule_based(gloss.split())

    @staticmethod
    def _rule_based(tokens: List[str]) -> str:
        """Simple offline join — replace with LLM when available."""
        text = " ".join(t.lower() for t in tokens)
        # Light heuristics for demo
        replacements = {
            "i go hospital": "I am going to the hospital.",
            "need help": "I need help.",
            "need_help": "I need help.",
            "thank you": "Thank you.",
            "thank_you": "Thank you.",
            "want water": "I want water.",
            "medical help": "I need medical help.",
            "medical_help": "I need medical help.",
        }
        return replacements.get(text, text.capitalize() + ".")
