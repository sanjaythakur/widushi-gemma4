"""VoiceMirrorMode -- pronunciation imitation.

Driven by an explicit sub-FSM (see ``phases/todo/1.png``):

::

                  on_enter
                     |
                     v
                  Prompt --(clip played)--> Listen
                                              |
                                          utterance
                                              |
                                              v
                                            Score
                              praise / correct / retry / silence
                                              |
                          +-------------------+--------------------+
                          |                   |                    |
                          v                   v                    v
                       Praise              Correct             CalmAndRetry
                    (face happy)       (face speaking)        (face worried)
                          |                   |                    |
                          v                   v                    v
                     NextPhrase            Listen              Listen
                          |
              more --|-- drill complete
              phrases|
                     v
                   Prompt                            DrillComplete --> MODE_CHANGE
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from app.events import Event, EventType
from app.modes.base import Mode
from app.modes.prompts import SYSTEM_PROMPT_HINGLISH
from app.modes.substate import ModeSubState, SubStateMachine
from app.services.gemma import GemmaReply
from app.state import AppState

if TYPE_CHECKING:
    from app.orchestrator.fsm import Orchestrator, Transition

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Substates
# ---------------------------------------------------------------------------


class PromptSub(ModeSubState):
    """A target word's cue is being prepared / spoken to the learner."""

    name = "voice_mirror.prompt"
    face_id = "speaking"


class ListenSub(ModeSubState):
    """We're recording the learner's attempt at the current target word."""

    name = "voice_mirror.listen"
    face_id = "listening"


class ScoreSub(ModeSubState):
    """Gemma is scoring the latest attempt; no user-visible audio yet."""

    name = "voice_mirror.score"
    face_id = "thinking"


class PraiseSub(ModeSubState):
    """High score: speak praise, then move on to the next phrase."""

    name = "voice_mirror.praise"
    face_id = "happy"


class CorrectSub(ModeSubState):
    """Mid score: speak a gentle correction; keep the same target word."""

    name = "voice_mirror.correct"
    face_id = "speaking"


class CalmAndRetrySub(ModeSubState):
    """Low score / long pause: speak a calming retry cue; keep the word."""

    name = "voice_mirror.calm_and_retry"
    face_id = "worried"


class NextPhraseSub(ModeSubState):
    """Transient bookkeeping state between Praise and the next Prompt."""

    name = "voice_mirror.next_phrase"
    face_id = "happy"


class DrillCompleteSub(ModeSubState):
    """Terminal substate: praise audio is playing, then we hand off to Vision."""

    name = "voice_mirror.drill_complete"
    face_id = "happy"


# ---------------------------------------------------------------------------
# Mode
# ---------------------------------------------------------------------------


class VoiceMirrorMode(Mode):
    """Pronunciation practice: suggest -> listen -> score -> next."""

    name = "voice_mirror"

    #: Number of successful (praise-only) scored turns before swapping to
    #: :class:`VisionMode`. Two is the value the product spec calls out.
    target_turns: int = 2

    def __init__(self) -> None:
        super().__init__()
        self._current_word: str = ""
        self._turns_done: int = 0
        self._history: list[str] = []

        # The substates are stateless singletons (just metadata); the
        # SubStateMachine owns which one is current.
        self._sub_prompt = PromptSub()
        self._sub_listen = ListenSub()
        self._sub_score = ScoreSub()
        self._sub_praise = PraiseSub()
        self._sub_correct = CorrectSub()
        self._sub_calm = CalmAndRetrySub()
        self._sub_next_phrase = NextPhraseSub()
        self._sub_drill_complete = DrillCompleteSub()
        self._substates = SubStateMachine()

    def system_prompt(self) -> str:
        return SYSTEM_PROMPT_HINGLISH

    def mode_prompt(self, context: dict[str, Any] | None = None) -> str:
        del context
        return (
            "You are in VOICE-MIRROR mode. The learner repeats a target word. "
            "Score their attempt and reply with praise, a gentle correction, "
            "or a calm retry. Keep replies to one short sentence."
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def on_enter(self, orch: Orchestrator) -> GemmaReply | None:
        await super().on_enter(orch)
        self._current_word = ""
        self._turns_done = 0
        self._history = []
        assert self._substates is not None  # set in __init__
        self._substates.reset()

        suggestion = await orch.services.gemma.voice_mirror_suggest(
            history=self._history
        )
        self._current_word = suggestion.word or ""
        if self._current_word:
            self._history.append(self._current_word)
        log.info("voice_mirror on_enter -> word=%r", self._current_word)

        # Cue audio will be played by the orchestrator's standard
        # THINKING -> SPEAKING path, so substate starts in Prompt.
        await self._substates.transition(self, orch, self._sub_prompt)
        return suggestion

    # ------------------------------------------------------------------
    # Sub-FSM event interception
    # ------------------------------------------------------------------

    def handle_event(
        self, orch: Orchestrator, event: Event
    ) -> Transition | None:
        """Drive the sub-FSM in lockstep with the outer AppState flow.

        The orchestrator invokes this synchronously from its event loop
        BEFORE consulting the default transition table, so we update the
        substate immediately and almost always return ``None`` to defer
        to the default table. The one exception is the silence CANCEL
        path, which we intercept to route to CalmAndRetry instead of
        dropping the session.
        """

        assert self._substates is not None
        machine = self._substates

        # Silence-triggered CANCEL from the wake-word source: stay in
        # SESSION, run a CalmAndRetry side-effect that re-prompts.
        if (
            event.type is EventType.CANCEL
            and orch.state is AppState.LISTENING
            and event.payload.get("reason") == "silence"
        ):
            log.info(
                "voice_mirror: silence CANCEL intercepted; routing to CalmAndRetry"
            )
            # Avoid circular import at module load time.
            from app.orchestrator.fsm import Transition as _Transition

            return _Transition(
                AppState.THINKING, side_effect=_calm_and_retry_side_effect
            )

        # Substate housekeeping for the happy path. None of these branches
        # affect the outer FSM -- we still return None so the
        # orchestrator's default table runs.
        current = machine.current

        if event.type is EventType.UTTERANCE_END and orch.state is AppState.LISTENING:
            # Learner finished their attempt; we're about to call Gemma.
            machine.set_current(self._sub_score)
            return None

        if event.type is EventType.PLAYBACK_DONE and orch.state is AppState.SPEAKING:
            # Whatever audio we just played has finished. Decide the next
            # substate based on what we were saying.
            if current in (self._sub_correct, self._sub_calm):
                # Same word, listen for another attempt.
                machine.set_current(self._sub_listen)
            elif current is self._sub_prompt:
                # First cue of a fresh word just finished playing.
                machine.set_current(self._sub_listen)
            elif current is self._sub_praise:
                # The combined "<praise>. Now try: <next-word>." audio
                # just finished; the next word's cue has effectively
                # been spoken, so we're now listening on the new word.
                machine.set_current(self._sub_listen)
            elif current is self._sub_drill_complete:
                # Praise audio for the last phrase finished. Nothing to
                # do here -- the orchestrator's SPEAKING side-effect
                # pops our pending MODE_CHANGE request and fires it.
                pass
            return None

        return None

    # ------------------------------------------------------------------
    # Gemma side effect
    # ------------------------------------------------------------------

    async def run_thinking(
        self, orch: Orchestrator, event: Event
    ) -> GemmaReply:
        assert self._substates is not None
        gemma = orch.services.gemma
        audio_bytes = event.payload.get("audio_bytes")

        if not self._current_word:
            # Defensive: if we somehow lost the target word, ask Gemma
            # for a new one and play it back as a fresh Prompt.
            log.warning("voice_mirror: no current_word; suggesting a fresh one")
            suggestion = await gemma.voice_mirror_suggest(history=self._history)
            self._current_word = suggestion.word or ""
            if self._current_word:
                self._history.append(self._current_word)
            self.record_turn(role="assistant", text=suggestion.text)
            await self._substates.transition(self, orch, self._sub_prompt)
            return suggestion

        if not isinstance(audio_bytes, bytes):
            # Keyboard/API event without audio: replay the current cue
            # rather than scoring silence.
            log.debug("voice_mirror: no audio in payload; replaying current cue")
            replay_text = f"Try saying: {self._current_word}."
            replay = await gemma.complete_with_tts(replay_text)
            await self._substates.transition(self, orch, self._sub_prompt)
            return replay

        # Score the attempt.
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

        if verdict == "praise":
            return await self._handle_praise(orch, score)
        if verdict == "correct":
            return await self._handle_correct(orch, score)
        return await self._handle_retry(orch, score)

    # ------------------------------------------------------------------
    # Verdict handlers
    # ------------------------------------------------------------------

    async def _handle_praise(
        self, orch: Orchestrator, score: GemmaReply
    ) -> GemmaReply:
        """High score: bump turn counter, then advance to NextPhrase.

        If we've hit ``target_turns`` we go through DrillCompleteSub and
        request a MODE_CHANGE to ``vision``. Otherwise we fetch the next
        word and return a combined "<praise>. Now <new-cue>." reply.
        """

        assert self._substates is not None
        gemma = orch.services.gemma
        self._turns_done += 1
        log.info(
            "voice_mirror: praise for %r (turns_done=%d/%d)",
            self._current_word,
            self._turns_done,
            self.target_turns,
        )

        # Substate hops: Score -> Praise (the praise audio belongs to
        # this substate).
        await self._substates.transition(self, orch, self._sub_praise)

        if self._turns_done >= self.target_turns:
            # Drill complete: the praise audio plays, then the
            # orchestrator pops our pending MODE_CHANGE and swaps to
            # VisionMode. auto_listen=False because VisionMode.on_enter
            # speaks its own greeting.
            self.request_mode_change(orch, "vision", auto_listen=False)
            await self._substates.transition(
                self, orch, self._sub_drill_complete
            )
            return score

        # More phrases: NextPhrase decides, then we go back to Prompt.
        await self._substates.transition(self, orch, self._sub_next_phrase)
        next_word = await gemma.voice_mirror_suggest(history=self._history)
        self._current_word = next_word.word or ""
        if self._current_word:
            self._history.append(self._current_word)
        combined_text = f"{score.text} Now {next_word.text}"
        # The new word's audio plays after praise so the learner hears
        # the next cue (the score.audio is discarded by design).
        combined = GemmaReply(
            text=combined_text,
            audio_bytes=next_word.audio_bytes,
            audio_duration_ms=next_word.audio_duration_ms,
            voice=next_word.voice,
        )
        # The combined audio IS the next cue, so by the time it starts
        # playing we're already in Prompt for the new word. handle_event
        # will move us to Listen when PLAYBACK_DONE fires.
        await self._substates.transition(self, orch, self._sub_prompt)
        return combined

    async def _handle_correct(
        self, orch: Orchestrator, score: GemmaReply
    ) -> GemmaReply:
        """Mid score: speak a gentle correction; keep the same word."""

        assert self._substates is not None
        log.info(
            "voice_mirror: correct for %r (turns_done=%d/%d)",
            self._current_word,
            self._turns_done,
            self.target_turns,
        )
        await self._substates.transition(self, orch, self._sub_correct)
        return score

    async def _handle_retry(
        self, orch: Orchestrator, score: GemmaReply
    ) -> GemmaReply:
        """Low score: speak a calming retry cue; keep the same word."""

        assert self._substates is not None
        log.info(
            "voice_mirror: retry for %r (turns_done=%d/%d)",
            self._current_word,
            self._turns_done,
            self.target_turns,
        )
        await self._substates.transition(self, orch, self._sub_calm)
        return score

    # ------------------------------------------------------------------
    # Test helpers / introspection
    # ------------------------------------------------------------------

    @property
    def current_word(self) -> str:
        return self._current_word

    @property
    def turns_done(self) -> int:
        return self._turns_done


# ---------------------------------------------------------------------------
# Side-effect coroutines
# ---------------------------------------------------------------------------


def _calm_and_retry_text(word: str) -> str:
    """Build the calming re-prompt spoken on silence / long-pause."""

    word = word.strip()
    if not word:
        return "Take a breath. Let's try again together."
    return f"Take a breath. Let's try again. Try saying: {word}."


async def _calm_and_retry_side_effect(orch: Orchestrator, event: Event) -> None:
    """Synthesise the CalmAndRetry re-prompt and post REPLY_READY.

    Invoked by the orchestrator when :meth:`VoiceMirrorMode.handle_event`
    returns a ``Transition(THINKING, side_effect=_calm_and_retry_...)``
    in response to a silence-flagged ``CANCEL``. The standard
    ``THINKING -> SPEAKING`` table entry then handles playback and the
    SPEAKING -> LISTENING follow-up re-arms the listener on the same word.
    """

    del event
    mode = orch.active_mode
    if not isinstance(mode, VoiceMirrorMode):
        log.warning(
            "calm_and_retry side effect ran without an active VoiceMirrorMode"
        )
        return

    if mode.substate_machine is not None:
        mode.substate_machine.set_current(mode._sub_calm)  # noqa: SLF001

    text = _calm_and_retry_text(mode.current_word)
    try:
        reply = await orch.services.gemma.complete_with_tts(text)
    except Exception:
        log.exception("voice_mirror: CalmAndRetry TTS failed; cancelling turn")
        await orch.queue.put(Event(EventType.CANCEL))
        return

    if not isinstance(reply.audio_bytes, bytes):
        log.error("voice_mirror: CalmAndRetry reply had no audio; cancelling")
        await orch.queue.put(Event(EventType.CANCEL))
        return

    await orch.queue.put(
        Event(
            EventType.REPLY_READY,
            payload={
                "text": reply.text,
                "audio_bytes": reply.audio_bytes,
                "audio_duration_ms": reply.audio_duration_ms,
                "voice": reply.voice,
            },
        )
    )


__all__ = ["VoiceMirrorMode"]
