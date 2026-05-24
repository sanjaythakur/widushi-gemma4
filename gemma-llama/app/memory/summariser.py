"""Episode + session summariser (PRD §Phase 2).

Owns the two small Gemma calls that land in Phase 2:

1. ``maybe_close_episode`` -- on ``X-Episode-Hint: close``, read the just-
   finished episode's turn transcript, ask Gemma for a <=2-sentence
   wrap-up, persist via :class:`SessionManager.close_episode`.
2. After every Nth episode close (cadence configurable via
   ``rolling_summary_every_n_closes``), fold the recent episode summaries
   plus the previous ``session.rolling_summary`` into a fresh rolling
   summary and persist via :class:`SessionManager.update_rolling_summary`.

Both calls go through the existing :class:`LlamaAdapter` (so they ride the
same prompt-prefix cache when the system prefix happens to match) but use
small ``max_tokens=128`` budgets and ``temperature=0.2`` so they stay
fast and deterministic.

Failure isolation: any :class:`LlamaServerError` is caught and logged at
``warning``; the user-visible mode response never 502s because the
summariser failed. The episode/session row simply stays with
``summary=NULL`` / unchanged ``rolling_summary``; the next close attempts
a fresh summary.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from ..errors import LlamaServerError
from ..llama_adapter import LlamaAdapter, extract_content, extract_usage
from .session_manager import Episode, Session, SessionManager, Turn

logger = logging.getLogger(__name__)


_ROLLING_COUNTER_KEY = "closed_episodes_since_rolling"

# Per-episode + per-rolling-summary char caps applied to the *input* we
# feed the summariser. The summary itself is bounded by max_tokens; this
# bounds the prompt so a runaway episode (200 turns) does not blow the
# context window.
_TURN_TEXT_MAX_CHARS = 240
_EPISODE_INPUT_CHAR_CAP = 3200
_ROLLING_INPUT_CHAR_CAP = 2400


@dataclass(slots=True)
class SummariserConfig:
    max_tokens: int = 128
    temperature: float = 0.2
    rolling_every_n_closes: int = 3


# ---------------------------------------------------------------------------
# Prompt assembly. Pure helpers -- no I/O, easy to eyeball.
# ---------------------------------------------------------------------------


def _short(text: str | None, max_chars: int = _TURN_TEXT_MAX_CHARS) -> str:
    if not text:
        return ""
    cleaned = " ".join(text.split())
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[: max_chars - 1].rstrip() + "\u2026"


def _format_turn_line(turn: Turn) -> str:
    role = "Learner" if turn.role == "user" else "Tutor"
    text = _short(turn.text)
    if not text and turn.media_kind:
        text = f"({turn.media_kind} input)"
    if not text:
        text = "(no transcript)"
    return f"- **{role}:** {text}"


def _trim_oldest_to_cap(lines: list[str], cap: int) -> list[str]:
    """Drop oldest lines until the joined text fits in ``cap`` chars."""
    out = list(lines)
    while out and len("\n".join(out)) > cap:
        out.pop(0)
    return out


def _build_episode_summary_messages(
    episode: Episode, turns: list[Turn]
) -> list[dict[str, str]]:
    """System + user message for the per-episode summariser call."""
    system = (
        "You are a concise note-taker for an English tutor. Given the "
        "transcript of a just-finished tutoring micro-episode, write a "
        "single short paragraph (one or two sentences, no more than 40 "
        "words) that captures: what the learner practised or asked, the "
        "outcome (success / stumble), and any concrete words or topics "
        "worth remembering for next time. Do not use lists, headings, "
        "code blocks, or markdown. Reply with the summary text only."
    )

    state = episode.state()
    words = state.get("words_drilled") if isinstance(state, dict) else None
    header_lines = [f"Mode: {episode.mode}"]
    if isinstance(words, list) and words:
        joined = ", ".join(str(w) for w in words)
        header_lines.append(f"Words drilled this episode: {joined}")

    turn_lines = [_format_turn_line(t) for t in turns]
    turn_lines = _trim_oldest_to_cap(turn_lines, _EPISODE_INPUT_CHAR_CAP)
    body = (
        "\n".join(header_lines)
        + "\n\nTurns (oldest first):\n"
        + ("\n".join(turn_lines) if turn_lines else "(no turns recorded)")
        + "\n\nWrite the summary now."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": body},
    ]


def _build_rolling_summary_messages(
    prev_rolling: str | None,
    recent_summaries: list[tuple[int, str, str]],
) -> list[dict[str, str]]:
    """System + user message for the session-level rolling summariser.

    ``recent_summaries`` is a list of ``(episode_id, mode, summary)``
    tuples in chronological order.
    """
    system = (
        "You are a concise note-taker keeping a running journal for an "
        "English tutor. Update the running session summary so it captures "
        "what the learner has worked on across the session: topics, words "
        "drilled, stumbles, and any explicit goals. Reply with one short "
        "paragraph (no more than 60 words). No lists, headings, code "
        "blocks, or markdown."
    )

    summary_lines = [
        f"- ({mode}) {_short(text, 320)}" for (_id, mode, text) in recent_summaries
    ]
    summary_lines = _trim_oldest_to_cap(summary_lines, _ROLLING_INPUT_CHAR_CAP)

    parts: list[str] = []
    if prev_rolling:
        parts.append(
            "Previous running summary:\n" + _short(prev_rolling, 800)
        )
    else:
        parts.append("Previous running summary: (none yet -- this is the first.)")
    parts.append(
        "Recent episode summaries (oldest first):\n"
        + ("\n".join(summary_lines) if summary_lines else "(none)")
    )
    parts.append("Write the updated running summary now.")
    body = "\n\n".join(parts)

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": body},
    ]


# ---------------------------------------------------------------------------
# EpisodeSummariser
# ---------------------------------------------------------------------------


class EpisodeSummariser:
    """LLM-driven episode + rolling-summary writer (PRD §Phase 2)."""

    def __init__(
        self,
        adapter: LlamaAdapter,
        session_manager: SessionManager,
        *,
        config: SummariserConfig | None = None,
    ) -> None:
        self.adapter = adapter
        self.session_manager = session_manager
        self.config = config or SummariserConfig()

    # ------------------------------------------------------------------
    # Public surface used by mode routers
    # ------------------------------------------------------------------

    async def maybe_close_episode(
        self,
        session: Session,
        episode: Episode,
        *,
        hint: str | None,
    ) -> str | None:
        """Close + summarise ``episode`` iff ``hint == "close"``.

        Returns the freshly-written ``episode.summary`` string on success
        so the router can echo it in the response, or ``None`` when the
        hint did not request a close, the episode was already closed, or
        the summariser call failed (we log + swallow in that case so the
        user-visible mode response is unaffected).
        """
        if hint is None or hint.strip().lower() != "close":
            return None
        if episode.ended_at is not None:
            logger.info(
                "episode %d already closed (ended_at=%s); skipping summariser",
                episode.id, episode.ended_at,
            )
            return None

        summary = await self._summarise_episode(episode)
        try:
            await self.session_manager.close_episode(
                episode.id, summary=summary
            )
        except Exception:  # noqa: BLE001
            logger.exception(
                "failed to persist episode close for episode_id=%d", episode.id
            )
            return summary

        # Reflect the in-memory state so a follow-on call inside the same
        # request would see the closed flag.
        episode.ended_at = "now"  # placeholder; routers do not read this
        episode.summary = summary

        if summary is not None:
            await self._maybe_update_rolling(session, episode)
        return summary

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _summarise_episode(self, episode: Episode) -> str | None:
        # Re-fetch so we observe the freshest ``state_json`` (e.g. the
        # word a /voice-mirror/score request just appended) without
        # depending on the router to mutate the in-memory dataclass.
        try:
            fresh = await self.session_manager.get_episode(episode.id)
        except Exception:  # noqa: BLE001
            logger.exception(
                "failed to refresh episode_id=%d; falling back to in-memory copy",
                episode.id,
            )
            fresh = episode
        episode_for_prompt = fresh or episode

        try:
            turns = await self.session_manager.episode_turns_full(episode.id)
        except Exception:  # noqa: BLE001
            logger.exception(
                "failed to load turns for episode_id=%d; skipping summary",
                episode.id,
            )
            return None
        if not turns:
            logger.info(
                "episode %d has no turns; writing empty summary placeholder",
                episode.id,
            )
            return None

        messages = _build_episode_summary_messages(episode_for_prompt, turns)
        return await self._summarise(
            messages,
            log_label=(
                f"episode_summary episode_id={episode.id} "
                f"mode={episode.mode}"
            ),
        )

    async def _maybe_update_rolling(
        self, session: Session, just_closed: Episode
    ) -> None:
        try:
            counter = await self.session_manager.bump_meta_counter(
                session.id, _ROLLING_COUNTER_KEY
            )
        except Exception:  # noqa: BLE001
            logger.exception(
                "failed to bump rolling counter for session_id=%d", session.id
            )
            return

        cadence = max(1, self.config.rolling_every_n_closes)
        if counter < cadence:
            logger.info(
                "rolling summary: session_id=%d counter=%d/%d -- skipping",
                session.id, counter, cadence,
            )
            return

        try:
            recent = await self.session_manager.closed_episode_summaries(
                session.id, limit=cadence
            )
        except Exception:  # noqa: BLE001
            logger.exception(
                "failed to load recent summaries for rolling update (session_id=%d)",
                session.id,
            )
            return

        if not recent:
            logger.info(
                "rolling summary: session_id=%d has no closed summaries yet",
                session.id,
            )
            await self._reset_rolling_counter(session.id)
            return

        # Refresh prev_rolling from the DB so we never overwrite a value
        # that was updated by another request mid-flight.
        prev_rolling = await self._fresh_rolling(session.id)
        messages = _build_rolling_summary_messages(prev_rolling, recent)
        rolling = await self._summarise(
            messages,
            log_label=f"rolling_summary session_id={session.id} counter={counter}",
        )
        if rolling is None:
            # Leave counter intact so the next close re-attempts the roll-up.
            return
        try:
            await self.session_manager.update_rolling_summary(session.id, rolling)
            await self._reset_rolling_counter(session.id)
        except Exception:  # noqa: BLE001
            logger.exception(
                "failed to persist rolling summary for session_id=%d", session.id
            )

    async def _fresh_rolling(self, session_id: int) -> str | None:
        row = await self.session_manager.db.afetchone(
            "SELECT rolling_summary FROM session WHERE id = ?", (session_id,)
        )
        if row is None:
            return None
        value = row["rolling_summary"]
        return str(value) if value else None

    async def _reset_rolling_counter(self, session_id: int) -> None:
        try:
            await self.session_manager.set_meta_counter(
                session_id, _ROLLING_COUNTER_KEY, 0
            )
        except Exception:  # noqa: BLE001
            logger.exception(
                "failed to reset rolling counter for session_id=%d", session_id
            )

    async def _summarise(
        self, messages: list[dict[str, str]], *, log_label: str
    ) -> str | None:
        """Run one chat completion and return the trimmed text, or None on failure."""
        started = time.perf_counter()
        try:
            response = await self.adapter.chat_completion(
                messages,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                thinking=False,
            )
        except LlamaServerError as exc:
            logger.warning("%s: llama call failed (%s); skipping", log_label, exc)
            return None
        except Exception:  # noqa: BLE001
            logger.exception("%s: unexpected error during summariser call", log_label)
            return None

        text = extract_content(response).strip()
        usage = extract_usage(response)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        logger.info(
            "%s ok in %.0fms prompt_tokens=%d completion_tokens=%d chars=%d",
            log_label,
            elapsed_ms,
            usage.get("prompt_tokens", 0),
            usage.get("completion_tokens", 0),
            len(text),
        )
        if not text:
            logger.warning("%s: empty completion; treating as no-summary", log_label)
            return None
        return text


__all__ = ["EpisodeSummariser", "SummariserConfig"]
