"""FreeConvoMode -- the device's default conversational mode after wake.

The learner can chat freely (Hindi / English / Hinglish). The
:func:`run_thinking` override calls ``POST /free-convo/turn`` on the
Gemma service which returns both a spoken reply and a ``start_learning``
intent flag. When ``start_learning`` is true the mode requests a swap
to :class:`VoiceMirrorMode` once the current reply finishes speaking.
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


class FreeConvoMode(Mode):
    """Open-ended chat that gates the switch into structured English practice."""

    name = "free_convo"

    def system_prompt(self) -> str:
        return SYSTEM_PROMPT_HINGLISH

    def mode_prompt(self, context: dict[str, Any] | None = None) -> str:
        del context
        return (
            "You are in FREE-CONVERSATION mode. Chat warmly with the learner. "
            "If they express any wish to learn or practise English, set "
            "start_learning=true in your JSON reply so the device can move "
            "into pronunciation practice."
        )

    async def run_thinking(
        self, orch: Orchestrator, event: Event
    ) -> GemmaReply:
        gemma = orch.services.gemma
        audio_bytes = event.payload.get("audio_bytes")

        if isinstance(audio_bytes, bytes):
            reply = await gemma.free_convo_turn(
                audio_bytes,
                filename=str(event.payload.get("filename", "turn.wav")),
                content_type=str(event.payload.get("content_type", "audio/wav")),
            )
            user_text = reply.transcript or "<audio>"
        else:
            # Keyboard / API path: no audio, fall back to plain chat.
            user_text = str(event.payload.get("prompt", ""))
            reply = await gemma.complete_with_tts(user_text)

        self.record_turn(role="user", text=user_text)
        self.record_turn(
            role="assistant",
            text=reply.text,
            extras={"start_learning": getattr(reply, "start_learning", False)},
        )

        if getattr(reply, "start_learning", False):
            log.info(
                "free_convo: start_learning=True; scheduling switch to voice_mirror"
            )
            # auto_listen=False because VoiceMirrorMode.on_enter speaks
            # its own intro ("Try saying: apple") and the SPEAKING ->
            # LISTENING follow-up of that intro arms the next listen.
            self.request_mode_change(orch, "voice_mirror", auto_listen=False)

        return reply


__all__ = ["FreeConvoMode"]
