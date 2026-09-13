"""
Emergency / quick-action hotkeys.
High-confidence triggers that flash UI and play priority audio.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional


@dataclass
class EmergencyAction:
    key: str
    label: str
    spoken: str
    gloss: str  # also treat as a forced sign token when triggered


# Default quick actions (India emergency number 108 shown as example)
DEFAULT_ACTIONS: Dict[str, EmergencyAction] = {
    "e": EmergencyAction(
        key="e",
        label="EMERGENCY / CALL 108",
        spoken="Emergency. Please call one zero eight immediately.",
        gloss="EMERGENCY",
    ),
    "m": EmergencyAction(
        key="m",
        label="MEDICAL HELP",
        spoken="I need medical help right now.",
        gloss="MEDICAL_HELP",
    ),
    "h": EmergencyAction(
        key="h",
        label="NEED HELP",
        spoken="I need help.",
        gloss="NEED_HELP",
    ),
}


class EmergencyController:
    def __init__(
        self,
        actions: Optional[Dict[str, EmergencyAction]] = None,
        on_trigger: Optional[Callable[[EmergencyAction], None]] = None,
    ) -> None:
        self.actions = actions or DEFAULT_ACTIONS
        self.on_trigger = on_trigger
        self.last_triggered: Optional[EmergencyAction] = None

    def handle_key(self, key: str) -> Optional[EmergencyAction]:
        action = self.actions.get(key.lower())
        if action is None:
            return None
        self.last_triggered = action
        if self.on_trigger:
            self.on_trigger(action)
        return action

    def list_actions(self) -> List[EmergencyAction]:
        return list(self.actions.values())
