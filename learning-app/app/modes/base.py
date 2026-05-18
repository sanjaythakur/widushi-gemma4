"""The :class:`Mode` contract.

Every interaction Mode (Tutor, Vision, Roleplay, VoiceMirror, ...) is a
sub-FSM that owns the *teaching behavior* for one activity. The
orchestrator (:class:`app.orchestrator.Orchestrator`) drives the outer
``SystemState`` and hands the active mode a chance to:

* Intercept transitions via :meth:`Mode.handle_event` (return a
  :class:`Transition` to override the default table, or ``None`` to
  defer).
* Run the THINKING side-effect against Gemma via
  :meth:`Mode.run_thinking`. The default implementation preserves the
  pre-hierarchy behavior exactly (just calls
  ``services.gemma.complete_with_tts`` / ``listen_audio_with_tts`` with
  the raw payload), so legacy callers do not need to register a mode.

Prompt architecture (per spec): ``SYSTEM PROMPT + MODE PROMPT`` plus an
``episode_history`` of past turns within the session. Today the
:class:`app.services.gemma.GemmaClient` API does not accept a separate
``system`` field; modes therefore expose the assembled prompt via
:meth:`Mode.build_prompt` for inspection and future-plumbing without
breaking the current HTTP contract.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from app.events import Event
from app.services.gemma import GemmaReply

if TYPE_CHECKING:  # avoid circular import; only used for type hints
    from app.orchestrator.fsm import Orchestrator, Transition

log = logging.getLogger(__name__)


@dataclass
class EpisodeTurn:
    """One user/assistant exchange inside an episode."""

    role: str  # "user" | "assistant"
    text: str
    extras: dict[str, Any] = field(default_factory=dict)


class Mode:
    """Base class for interaction modes.

    Subclasses override ``name``, ``system_prompt``, and optionally
    ``mode_prompt``, ``handle_event``, and ``run_thinking``.
    """

    name: str = "base"

    def __init__(self) -> None:
        self._episode_history: list[EpisodeTurn] = []
        self._pending_mode_change: dict[str, Any] | None = None

    # ------------------------------------------------------------------
    # Prompts
    # ------------------------------------------------------------------

    def system_prompt(self) -> str:
        """Return the SYSTEM-level prompt for this mode.

        Subclasses describe the persona / course-wide rules here (e.g.
        "Accept Hindi input, encourage English output").
        """

        return ""

    def mode_prompt(self, context: dict[str, Any] | None = None) -> str:
        """Return the MODE-level prompt for this mode.

        ``context`` lets the orchestrator pass per-turn data (e.g. the
        latest user utterance, current topic). Subclasses can ignore it.
        """

        del context
        return ""

    def build_prompt(
        self,
        user_text: str,
        *,
        context: dict[str, Any] | None = None,
        history: Sequence[EpisodeTurn] | None = None,
    ) -> str:
        """Assemble ``SYSTEM + MODE + HISTORY + USER`` into one string.

        Used for the text path today. Once the Gemma server accepts a
        separate ``system`` field, the orchestrator can plumb
        :meth:`system_prompt` through directly instead of concatenating.
        """

        parts: list[str] = []
        sys_prompt = self.system_prompt().strip()
        if sys_prompt:
            parts.append(f"[SYSTEM]\n{sys_prompt}")
        mode_prompt = self.mode_prompt(context).strip()
        if mode_prompt:
            parts.append(f"[MODE: {self.name}]\n{mode_prompt}")

        history_iter = history if history is not None else self._episode_history
        if history_iter:
            lines = [
                f"{turn.role.capitalize()}: {turn.text}" for turn in history_iter
            ]
            parts.append("[HISTORY]\n" + "\n".join(lines))

        parts.append(f"User: {user_text}")
        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    # Episode history
    # ------------------------------------------------------------------

    @property
    def episode_history(self) -> list[EpisodeTurn]:
        return self._episode_history

    def record_turn(
        self, *, role: str, text: str, extras: dict[str, Any] | None = None
    ) -> None:
        """Append one turn to ``episode_history``."""

        self._episode_history.append(
            EpisodeTurn(role=role, text=text, extras=dict(extras or {}))
        )

    def reset_history(self) -> None:
        self._episode_history.clear()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def on_enter(self, orch: Orchestrator) -> GemmaReply | None:
        """Called when this mode becomes active (SystemState IDLE -> SESSION,
        or a mid-session ``MODE_CHANGE``).

        May optionally return a :class:`GemmaReply` to be spoken as an
        introductory line BEFORE the learner is asked to speak. Returning
        ``None`` keeps the legacy behaviour (just listen). Default
        implementation only resets ``episode_history``.
        """

        del orch
        self.reset_history()
        self._pending_mode_change = None
        return None

    async def on_exit(self, orch: Orchestrator) -> None:
        """Called when leaving this mode (SESSION -> IDLE or mode swap)."""

        del orch

    # ------------------------------------------------------------------
    # Mid-session mode-change request (deferred swap)
    # ------------------------------------------------------------------

    def request_mode_change(
        self,
        orch: Orchestrator | None,
        target: str,
        *,
        auto_listen: bool = True,
    ) -> None:
        """Schedule a swap to ``target`` after the current SPEAKING ends.

        The orchestrator's ``_speaking_side_effect`` pops this stash after
        playing the active mode's reply audio and enqueues a
        ``MODE_CHANGE`` event, so the new mode's ``on_enter`` greeting
        plays cleanly after the current spoken line.

        ``orch`` is accepted for forward-compat (a future implementation
        may post the event directly) but is unused today.
        """

        del orch
        self._pending_mode_change = {
            "target_mode": target,
            "auto_listen": bool(auto_listen),
        }

    def pop_pending_mode_change(self) -> dict[str, Any] | None:
        """Return-and-clear the queued mode-change payload (or ``None``)."""

        pending = self._pending_mode_change
        self._pending_mode_change = None
        return pending

    # ------------------------------------------------------------------
    # Event interception
    # ------------------------------------------------------------------

    def handle_event(
        self, orch: Orchestrator, event: Event
    ) -> Transition | None:
        """Optionally override the default transition for ``event``.

        Returning ``None`` defers to the orchestrator's default table.
        Subclasses use this hook to insert mode-specific transitions
        (e.g. VoiceMirrorMode might add a SCORING sub-state).
        """

        del orch, event
        return None

    # ------------------------------------------------------------------
    # Gemma side-effect (the THINKING work)
    # ------------------------------------------------------------------

    async def run_thinking(
        self, orch: Orchestrator, event: Event
    ) -> GemmaReply:
        """Call Gemma for the user's utterance and return the reply.

        Default implementation mirrors the pre-hierarchy ``_call_gemma``
        exactly so legacy paths and existing tests keep working: it
        sends the raw ``prompt`` or ``audio_bytes`` from the
        ``UTTERANCE_END`` payload directly to ``services.gemma``. On
        success it appends both the user and assistant turns to
        :attr:`episode_history`.

        Modes that need richer prompting (system prompt injection,
        scenario context, etc.) override this method.
        """

        gemma = orch.services.gemma
        audio_bytes = event.payload.get("audio_bytes")

        if isinstance(audio_bytes, bytes):
            reply = await gemma.listen_audio_with_tts(
                audio_bytes,
                filename=str(event.payload.get("filename", "question.wav")),
                content_type=str(
                    event.payload.get("content_type", "audio/wav")
                ),
            )
            user_text = "<audio>"
        else:
            user_text = str(event.payload.get("prompt", ""))
            reply = await gemma.complete_with_tts(user_text)

        self.record_turn(role="user", text=user_text)
        self.record_turn(role="assistant", text=reply.text)
        return reply
