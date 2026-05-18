"""TutorMode: the default conversational tutor.

Wraps the existing turn flow (the flat FSM) with a Hinglish-aware system
prompt for the "English Speaking for Hindi speakers" course. Behavior on
the wire is unchanged from the pre-hierarchy code path so existing tests
in ``tests/test_fsm.py`` keep passing.

The :meth:`Mode.run_thinking` override is intentionally *not* mutating
the prompt sent to Gemma yet — :class:`app.services.gemma.GemmaClient`
posts ``{"prompt": prompt, "tts": ..., "voice": ...}`` and does not
accept a separate ``system`` field, so injecting SYSTEM + MODE into the
``prompt`` body would (a) break the existing ``complete_with_tts`` test
contract that asserts ``prompt == "hello"`` and (b) deserve a coordinated
update with the Gemma service in ``gemma-llama/``. The hook is wired
through the base ``Mode.run_thinking`` (which appends turns to
``episode_history`` on success) so once Gemma gets a ``system`` field
the swap is a one-line change here.
"""

from __future__ import annotations

from typing import Any

from app.modes.base import Mode
from app.modes.prompts import SYSTEM_PROMPT_HINGLISH


class TutorMode(Mode):
    """Standard teaching / explanation mode. Default mode of the device."""

    name = "tutor"

    def system_prompt(self) -> str:
        return SYSTEM_PROMPT_HINGLISH

    def mode_prompt(self, context: dict[str, Any] | None = None) -> str:
        del context
        return (
            "You are in TUTOR mode. The learner asks open-ended questions; "
            "you give short, encouraging answers and model one useful English "
            "phrase per turn."
        )


__all__ = ["TutorMode"]
