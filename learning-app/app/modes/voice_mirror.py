"""VoiceMirrorMode -- pronunciation imitation.

Lifecycle:

1. ``on_enter`` calls ``POST /voice-mirror/suggest`` and returns the
   suggestion as the introductory :class:`GemmaReply` so the
   orchestrator speaks "Try saying: <word>" before listening.
2. ``run_thinking`` calls ``POST /voice-mirror/score`` for the
   learner's recorded attempt. The endpoint returns one of three
   verdicts (``praise`` / ``correct`` / ``retry``).
3. ``praise`` and ``correct`` count as a successful turn; ``retry``
   keeps the same target word. After :attr:`target_turns` successful
   turns the mode requests a swap to :class:`VisionMode`.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from app.events import Event
from app.modes.base import Mode
from app.modes.prompts import SYSTEM_PROMPT_HINGLISH
from app.services.gemma import GemmaReply, WordSuggestion

if TYPE_CHECKING:
    from app.orchestrator.fsm import Orchestrator

log = logging.getLogger(__name__)


class VoiceMirrorMode(Mode):
    """Pronunciation practice: suggest -> listen -> score -> next."""

    name = "voice_mirror"

    #: Number of successful (praise/correct) scored turns before swapping
    #: to :class:`VisionMode`. Two is the value the product spec calls out.
    target_turns: int = 2

    def __init__(self) -> None:
        super().__init__()
        self._current_word: str = ""
        self._turns_done: int = 0
        self._history: list[str] = []

    def system_prompt(self) -> str:
        return SYSTEM_PROMPT_HINGLISH

    def mode_prompt(self, context: dict[str, Any] | None = None) -> str:
        del context
        return (
            "You are in VOICE-MIRROR mode. The learner repeats a target word. "
            "Score their attempt and reply with praise, a gentle correction, "
            "or a calm retry. Keep replies to one short sentence."
        )

    async def on_enter(self, orch: Orchestrator) -> GemmaReply | None:
        await super().on_enter(orch)
        self._current_word = ""
        self._turns_done = 0
        self._history = []
        suggestion = await orch.services.gemma.voice_mirror_suggest(
            history=self._history
        )
        self._current_word = suggestion.word or ""
        if self._current_word:
            self._history.append(self._current_word)
        log.info("voice_mirror on_enter -> word=%r", self._current_word)
        return suggestion

    async def run_thinking(
        self, orch: Orchestrator, event: Event
    ) -> GemmaReply:
        gemma = orch.services.gemma
        audio_bytes = event.payload.get("audio_bytes")

        if not self._current_word:
            # Defensive: if we somehow lost the target word, ask Gemma for
            # a new one and play it back so the learner has something to try.
            log.warning("voice_mirror: no current_word; suggesting a fresh one")
            suggestion = await gemma.voice_mirror_suggest(history=self._history)
            self._current_word = suggestion.word or ""
            if self._current_word:
                self._history.append(self._current_word)
            self.record_turn(role="assistant", text=suggestion.text)
            return suggestion

        if not isinstance(audio_bytes, bytes):
            # Keyboard/API event without audio: replay the current cue
            # rather than scoring silence.
            log.debug("voice_mirror: no audio in payload; replaying current cue")
            replay_text = f"Try saying: {self._current_word}."
            return await gemma.complete_with_tts(replay_text)

        score = await gemma.voice_mirror_score(
            audio_bytes,
            target_word=self._current_word,
            filename=str(event.payload.get("filename", "attempt.wav")),
            content_type=str(event.payload.get("content_type", "audio/wav")),
        )
        verdict = (score.verdict or "retry").lower()

        self.record_turn(
            role="user",
            text=score.transcript or "<audio attempt>",
            extras={"target_word": self._current_word},
        )
        self.record_turn(
            role="assistant",
            text=score.text,
            extras={"verdict": verdict},
        )

        if verdict in ("praise", "correct"):
            self._turns_done += 1
            log.info(
                "voice_mirror: verdict=%s for %r (turns_done=%d/%d)",
                verdict,
                self._current_word,
                self._turns_done,
                self.target_turns,
            )
            if self._turns_done >= self.target_turns:
                # Hand off to VisionMode after this feedback line finishes
                # speaking. auto_listen=False because VisionMode.on_enter
                # speaks its own greeting which arms the next listen.
                self.request_mode_change(orch, "vision", auto_listen=False)
                return score

            # Move on to a fresh target word. We synthesise a combined
            # reply that says "<feedback> Now try: <next word>." so the
            # learner gets one coherent spoken turn instead of two.
            next_word = await gemma.voice_mirror_suggest(history=self._history)
            self._current_word = next_word.word or ""
            if self._current_word:
                self._history.append(self._current_word)
            combined_text = f"{score.text} Now {next_word.text}"
            # The new word's audio replaces the verbal-only feedback so
            # the learner hears the next cue rather than just praise.
            return GemmaReply(
                text=combined_text,
                audio_bytes=next_word.audio_bytes,
                audio_duration_ms=next_word.audio_duration_ms,
                voice=next_word.voice,
            )

        # retry: keep the same word, just play the coaching line.
        log.info(
            "voice_mirror: verdict=retry for %r (turns_done=%d/%d)",
            self._current_word,
            self._turns_done,
            self.target_turns,
        )
        return score

    # ------------------------------------------------------------------
    # Test helpers
    # ------------------------------------------------------------------

    @property
    def current_word(self) -> str:
        return self._current_word

    @property
    def turns_done(self) -> int:
        return self._turns_done


__all__ = ["VoiceMirrorMode"]
