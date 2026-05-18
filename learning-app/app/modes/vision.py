"""VisionMode -- object-vocabulary practice from the camera.

Lifecycle:

1. ``on_enter`` greets the learner ("Hold up an object and tell me what
   you think it is.") via Piper TTS so they know to point the camera.
2. ``run_thinking`` snaps a frame from :class:`CameraSource`, sends the
   image + the learner's spoken guess to ``POST /vision/teach-object``,
   and returns the teaching reply ("Yes, this is milk. Say: I drink
   milk.").
3. After :attr:`target_turns` taught turns the mode requests a swap to
   :class:`RolePlayMode`.
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


_INTRO_TEXT = (
    "Hold an object in front of the camera and tell me what you think it is in English."
)


class VisionMode(Mode):
    """Camera + spoken guess -> tutor teaches the English noun."""

    name = "vision"

    target_turns: int = 2

    def __init__(self) -> None:
        super().__init__()
        self._turns_done: int = 0

    def system_prompt(self) -> str:
        return SYSTEM_PROMPT_HINGLISH

    def mode_prompt(self, context: dict[str, Any] | None = None) -> str:
        del context
        return (
            "You are in VISION mode. The learner holds an object up to the "
            "camera. Confirm or correct their guess, then prompt them to say "
            "a short English sentence using the word."
        )

    async def on_enter(self, orch: Orchestrator) -> GemmaReply | None:
        await super().on_enter(orch)
        self._turns_done = 0
        intro = await orch.services.gemma.complete_with_tts(_INTRO_TEXT)
        return intro

    async def run_thinking(
        self, orch: Orchestrator, event: Event
    ) -> GemmaReply:
        gemma = orch.services.gemma
        audio_bytes = event.payload.get("audio_bytes")

        if not isinstance(audio_bytes, bytes):
            log.debug("vision: no audio in payload; replaying intro")
            return await gemma.complete_with_tts(_INTRO_TEXT)

        image_bytes = await orch.services.camera.snapshot()
        log.info("vision: captured %d bytes from camera", len(image_bytes))

        reply = await gemma.vision_teach_object(
            image_bytes,
            audio_bytes,
            audio_filename=str(event.payload.get("filename", "guess.wav")),
            audio_content_type=str(event.payload.get("content_type", "audio/wav")),
        )

        self.record_turn(
            role="user",
            text=reply.transcript or "<audio guess>",
        )
        self.record_turn(
            role="assistant",
            text=reply.text,
            extras={"object": reply.object},
        )

        self._turns_done += 1
        log.info(
            "vision: taught turn %d/%d (object=%r)",
            self._turns_done,
            self.target_turns,
            reply.object,
        )
        if self._turns_done >= self.target_turns:
            self.request_mode_change(orch, "roleplay", auto_listen=False)
        return reply

    @property
    def turns_done(self) -> int:
        return self._turns_done


__all__ = ["VisionMode"]
