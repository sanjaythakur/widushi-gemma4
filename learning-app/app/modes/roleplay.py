"""RolePlayMode -- summative assessment via a simulated scenario.

The mode owns the scenario state (e.g. shopkeeper, doctor) and tracks
the running transcript. Each turn it stitches the system prompt, the
scenario brief, and the recent dialogue into a single prompt, then asks
Gemma to generate the next NPC line via the default ``/generate``
endpoint (audio in -> ``/audio/listen`` for the learner's spoken turn).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from app.events import Event
from app.modes.base import Mode
from app.modes.prompts import SYSTEM_PROMPT_HINGLISH
from app.services.gemma import GemmaReply

if TYPE_CHECKING:
    from app.orchestrator.fsm import Orchestrator

log = logging.getLogger(__name__)


_DEFAULT_SCENARIO = "shopkeeper"

_SCENARIO_BRIEFS: dict[str, str] = {
    "shopkeeper": (
        "You are a friendly Indian shopkeeper. The learner is a customer "
        "trying to buy fruit. Greet them, ask what they want, and respond "
        "naturally in short English sentences."
    ),
    "doctor": (
        "You are a calm doctor. The learner is a patient describing a small "
        "ailment. Ask one short follow-up question and give simple advice in "
        "English."
    ),
}


def _scenario_brief(scenario: str) -> str:
    return _SCENARIO_BRIEFS.get(scenario, _SCENARIO_BRIEFS[_DEFAULT_SCENARIO])


def _intro_text(scenario: str) -> str:
    if scenario == "doctor":
        return (
            "Let's role-play. I am a doctor; you are my patient. "
            "Tell me what is bothering you today."
        )
    return (
        "Let's role-play. I am a shopkeeper at a small fruit shop; you are "
        "my customer. Come on in, what would you like to buy today?"
    )


class RolePlayMode(Mode):
    """Scenario-driven summative practice (single-shot, no scoring yet)."""

    name = "roleplay"

    def __init__(self, *, scenario: str = _DEFAULT_SCENARIO) -> None:
        super().__init__()
        self._scenario = scenario
        self._transcript: list[str] = []

    def system_prompt(self) -> str:
        return SYSTEM_PROMPT_HINGLISH

    def mode_prompt(self, context: dict[str, Any] | None = None) -> str:
        scenario = self._scenario
        if context and context.get("scenario"):
            scenario = str(context["scenario"])
        return (
            f"You are in ROLEPLAY mode. Stay in character as a {scenario}. "
            "Reply in one or two short, natural English sentences. Never "
            "break the fourth wall."
        )

    async def on_enter(self, orch: Orchestrator) -> GemmaReply | None:
        await super().on_enter(orch)
        self._transcript = []
        intro = _intro_text(self._scenario)
        self._transcript.append(f"Shopkeeper: {intro}")
        return await orch.services.gemma.complete_with_tts(intro)

    async def run_thinking(
        self, orch: Orchestrator, event: Event
    ) -> GemmaReply:
        gemma = orch.services.gemma
        audio_bytes = event.payload.get("audio_bytes")

        if isinstance(audio_bytes, bytes):
            # Transcribe + reply via the existing /audio/listen path so we
            # both hear the learner's turn and get the NPC's next line.
            heard = await gemma.listen_audio_with_tts(
                audio_bytes,
                filename=str(event.payload.get("filename", "turn.wav")),
                content_type=str(event.payload.get("content_type", "audio/wav")),
            )
            user_text = "<audio>"
        else:
            user_text = str(event.payload.get("prompt", ""))
            prompt = self._build_prompt(user_text)
            heard = await gemma.complete_with_tts(prompt)

        self.record_turn(role="user", text=user_text)
        self.record_turn(role="assistant", text=heard.text)
        self._transcript.append(f"Learner: {user_text}")
        self._transcript.append(f"Shopkeeper: {heard.text}")
        return heard

    def _build_prompt(self, user_text: str) -> str:
        brief = _scenario_brief(self._scenario)
        history = "\n".join(self._transcript[-8:]) if self._transcript else ""
        layered = self.build_prompt(user_text, history=None)
        if history:
            return f"{layered}\n\n[SCENARIO]\n{brief}\n\n[TRANSCRIPT]\n{history}"
        return f"{layered}\n\n[SCENARIO]\n{brief}"

    @property
    def scenario(self) -> str:
        return self._scenario

    @property
    def transcript(self) -> list[str]:
        return list(self._transcript)


__all__ = ["RolePlayMode"]
