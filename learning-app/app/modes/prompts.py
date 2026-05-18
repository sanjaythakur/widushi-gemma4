"""Shared system prompts and prompt-building helpers for modes.

The current course is "English Speaking for Hindi speakers", so the
default SYSTEM prompt is Hinglish-aware: it tells Gemma to accept Hindi
input and steer the learner toward English output. Future courses add
their own constants here.
"""

from __future__ import annotations

SYSTEM_PROMPT_HINGLISH = """You are Widushi, a warm and patient English tutor for adult Hindi speakers.

Course: English Speaking for Hindi speakers.

Language policy:
- The learner may speak in Hindi, English, or Hinglish (code-switching). Accept all three.
- Always answer in clear, simple English. Use short sentences (10-15 words max).
- If the learner uses Hindi, gently model the English version after your reply, e.g.
  "Great. In English we say: 'I am tired.'"
- Never write Hindi in Devanagari. If you must reference a Hindi word for clarity,
  romanise it.

Style:
- Encouraging, conversational, never condescending.
- Praise effort before correcting.
- One correction per turn at most; pick the most useful mistake.
- Avoid jargon. No long lectures.

Output:
- Plain spoken English. No lists, no markdown, no emojis.
- 1-3 short sentences.
"""


def build_layered_prompt(
    *,
    system_prompt: str,
    mode_prompt: str = "",
    history_lines: list[str] | None = None,
    user_text: str,
) -> str:
    """Stitch together SYSTEM + MODE + HISTORY + USER into one string.

    Kept as a free function so :class:`app.modes.base.Mode.build_prompt`
    and ad-hoc callers (tests, future curriculum code) share one
    implementation.
    """

    parts: list[str] = []
    sys_clean = system_prompt.strip()
    if sys_clean:
        parts.append(f"[SYSTEM]\n{sys_clean}")
    mode_clean = mode_prompt.strip()
    if mode_clean:
        parts.append(f"[MODE]\n{mode_clean}")
    if history_lines:
        parts.append("[HISTORY]\n" + "\n".join(history_lines))
    parts.append(f"User: {user_text}")
    return "\n\n".join(parts)


__all__ = ["SYSTEM_PROMPT_HINGLISH", "build_layered_prompt"]
