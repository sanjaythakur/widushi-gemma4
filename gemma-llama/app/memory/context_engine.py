"""Context engine: builds the per-request blocks that ride alongside the
mode-specific Jinja prompt.

Phase 1A owned the **learner block** (a compact Markdown blob assembled
from the YAML front-matter facts plus the verbatim ``## Notes for the
tutor`` section of the learner's profile). It sits at the head of the
system prompt so llama.cpp's prompt-prefix cache (`LLAMA_CACHE_PROMPT=
true`) keeps hitting across back-to-back turns under the same learner.

Phase 1B adds two more blocks, both sourced from the SQLite-backed
:class:`SessionManager` and rendered as a **user-message prelude**
(*not* injected into the system prompt) so the system prefix remains
byte-identical per learner:

* :meth:`build_working_block` -- last ``K_TURNS=3`` turns of the current
  episode (T0 in PRD §3.2).
* :meth:`build_episodic_block` -- raw-turn dump from up to N prior
  episodes in the same session, regardless of mode (T1, sans LLM
  summarisation -- that lands in Phase 2).

:meth:`build_user_prelude` composes both into a single Markdown blob
which routers concatenate in front of their existing user-instruction
text. When there is no history yet (fresh session / first turn) the
prelude is the empty string and behaviour collapses to Phase 1A.
"""
from __future__ import annotations

import logging
import re

from .learner_loader import DEFAULT_LEARNER_ID, LearnerLoader, LearnerProfile
from .session_manager import Episode, SessionManager, Turn

logger = logging.getLogger(__name__)


_NOTES_HEADING_RE = re.compile(
    r"^\s*##\s+Notes\s+for\s+the\s+tutor\s*$", re.IGNORECASE | re.MULTILINE
)


# ---------------------------------------------------------------------------
# Learner block (Phase 1A) -- unchanged renderers preserved verbatim.
# ---------------------------------------------------------------------------


def _extract_notes_section(body_markdown: str) -> str:
    """Return the verbatim text under ``## Notes for the tutor``."""
    if not body_markdown:
        return ""
    match = _NOTES_HEADING_RE.search(body_markdown)
    if not match:
        return ""
    start = match.end()
    tail = body_markdown[start:]
    next_heading = re.search(r"^\s*##\s+", tail, re.MULTILINE)
    section = tail[: next_heading.start()] if next_heading else tail
    return section.strip()


def _format_languages(profile: LearnerProfile) -> str:
    parts: list[str] = []
    if profile.l1:
        parts.append(f"native {profile.l1}")
    if profile.l1_secondary:
        parts.append(f"secondary {profile.l1_secondary}")
    if profile.l2_target:
        parts.append(f"learning {profile.l2_target}")
    if profile.proficiency:
        parts.append(f"level: {profile.proficiency}")
    return "; ".join(parts) if parts else "(unspecified)"


def _format_who(profile: LearnerProfile) -> str:
    bits: list[str] = []
    if profile.name:
        bits.append(profile.name)
    age_gender: list[str] = []
    if profile.age is not None:
        age_gender.append(str(profile.age))
    if profile.gender:
        age_gender.append(profile.gender)
    if age_gender:
        bits.append(f"({', '.join(age_gender)})")
    if profile.occupation:
        bits.append(f"-- {profile.occupation}")
    if profile.location:
        bits.append(f"in {profile.location}")
    return " ".join(bits).strip() or profile.learner_id


def _render_learner_block(profile: LearnerProfile) -> str:
    lines: list[str] = ["## Learner profile"]
    who = _format_who(profile)
    if who:
        lines.append(f"- **Who:** {who}.")
    languages = _format_languages(profile)
    lines.append(f"- **Languages:** {languages}.")
    if profile.interests:
        lines.append(f"- **Interests:** {', '.join(profile.interests)}.")
    notes = _extract_notes_section(profile.body_markdown)
    if notes:
        lines.append("")
        lines.append("### Notes for the tutor")
        lines.append(notes)
    return "\n".join(lines).strip()


# ---------------------------------------------------------------------------
# Phase 1B block renderers
# ---------------------------------------------------------------------------


# Per-block char budgets enforced by oldest-first truncation. We're not in
# a position to tokenise here (no in-process tokenizer), so PRD §3.3 caps
# are translated through the standard 4-chars-per-token heuristic.
_WORKING_BLOCK_CHAR_CAP = 800        # ~200 tokens text-only (PRD §3.3 T0)
_EPISODIC_BLOCK_CHAR_CAP = 1600      # ~400 tokens text-only (PRD §3.3 T1)
_TURN_TEXT_MAX_CHARS = 240           # keep individual lines readable; the
# point of episodic context is recency, not transcript completeness.


def _short(text: str | None, max_chars: int = _TURN_TEXT_MAX_CHARS) -> str:
    """Single-line, length-capped rendering of a turn's text."""
    if not text:
        return ""
    cleaned = " ".join(text.split())
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[: max_chars - 1].rstrip() + "\u2026"


def _format_turn_line(turn: Turn) -> str:
    """Render one turn as a Markdown bullet."""
    role = "Learner" if turn.role == "user" else "Tutor"
    text = _short(turn.text)
    if not text and turn.media_kind:
        text = f"({turn.media_kind} input)"
    if not text:
        text = "(no transcript)"
    return f"  - **{role}:** {text}"


def _trim_to_budget(lines: list[str], cap: int) -> list[str]:
    """Drop oldest entries until the joined text fits in ``cap`` chars.

    ``lines`` is expected to be in chronological order (oldest first);
    we trim from the front. Returns a new list.
    """
    out = list(lines)
    while out and len("\n".join(out)) > cap:
        out.pop(0)
    return out


# ---------------------------------------------------------------------------
# ContextEngine
# ---------------------------------------------------------------------------


class ContextEngine:
    """Assemble request-scoped context blocks for the prompt builder.

    Phase 1A constructed only the learner block. Phase 1B layers the
    episodic + working blocks on top, sourced from the SQLite-backed
    :class:`SessionManager`. Both phases share the same instance --
    callers that pre-date the memory tables simply skip the
    Phase-1B methods.
    """

    def __init__(
        self,
        loader: LearnerLoader,
        session_manager: SessionManager | None = None,
        *,
        working_k_turns: int = 3,
        episodic_max_episodes: int = 3,
        episodic_turns_per_episode: int = 2,
    ) -> None:
        self.loader = loader
        self.session_manager = session_manager
        self.working_k_turns = working_k_turns
        self.episodic_max_episodes = episodic_max_episodes
        self.episodic_turns_per_episode = episodic_turns_per_episode

    # ------------------------------------------------------------------
    # Phase 1A: learner block
    # ------------------------------------------------------------------

    def build_learner_block(
        self, learner_id: str | None
    ) -> tuple[str, str]:
        """Return ``(resolved_id, rendered_block)``."""
        resolved_id, profile = self.loader.resolve(learner_id)
        if profile is None:
            return resolved_id, ""
        block = _render_learner_block(profile)
        char_len = len(block)
        token_estimate = max(1, char_len // 4)
        logger.info(
            "learner_block resolved_id=%s chars=%d ~tokens=%d",
            resolved_id, char_len, token_estimate,
        )
        return resolved_id, block

    # ------------------------------------------------------------------
    # Phase 1B: episodic + working blocks (read-only, async)
    # ------------------------------------------------------------------

    async def build_working_block(self, episode_id: int) -> str:
        """Render the last ``K_TURNS`` of the current episode."""
        if self.session_manager is None:
            return ""
        turns = await self.session_manager.current_episode_turns(
            episode_id, limit=self.working_k_turns
        )
        if not turns:
            return ""
        lines = [_format_turn_line(t) for t in turns]
        lines = _trim_to_budget(lines, _WORKING_BLOCK_CHAR_CAP)
        if not lines:
            return ""
        header = f"### This turn so far (last {len(lines)} turns)"
        return "\n".join([header, *lines])

    async def build_episodic_block(
        self, session_id: int, current_episode_id: int | None
    ) -> str:
        """Render prior episodes -- summaries when available, raw tails otherwise.

        Phase 2 swap: each prior episode renders as its LLM-written
        ``summary`` when present (one bullet, very token-efficient). When
        the summary is still NULL (open episode, summariser hasn't run /
        failed), we fall back to the Phase 1B raw-tail rendering so the
        block degrades gracefully instead of going blank.

        We also prepend the session's ``rolling_summary`` (if any) as the
        first line of the block so cross-mode recall has a single
        compact anchor before the per-episode bullets.
        """
        if self.session_manager is None:
            return ""
        episodes = await self.session_manager.prior_episodes(
            session_id,
            exclude_episode_id=current_episode_id,
            limit=self.episodic_max_episodes,
        )

        rolling = await self._fresh_rolling_summary(session_id)

        sections: list[str] = []
        if rolling:
            sections.append(f"- **Session so far:** {rolling}")
        for episode in episodes:
            section = await self._render_episode_section(episode)
            if section:
                sections.append(section)

        if not sections:
            return ""

        sections = _trim_to_budget(sections, _EPISODIC_BLOCK_CHAR_CAP)
        if not sections:
            return ""

        header = "### Earlier in this session (summaries)"
        return "\n".join([header, *sections])

    async def _fresh_rolling_summary(self, session_id: int) -> str | None:
        assert self.session_manager is not None
        row = await self.session_manager.db.afetchone(
            "SELECT rolling_summary FROM session WHERE id = ?", (session_id,)
        )
        if row is None:
            return None
        value = row["rolling_summary"]
        text = str(value).strip() if value else ""
        return text or None

    async def _render_episode_section(self, episode: Episode) -> str:
        assert self.session_manager is not None
        state = episode.state()
        words = state.get("words_drilled") if isinstance(state, dict) else None

        if episode.summary:
            # Phase 2 fast path: one compact bullet per closed episode.
            line = (
                f"- **{episode.mode}** (episode #{episode.id}): "
                f"{episode.summary.strip()}"
            )
            if isinstance(words, list) and words:
                joined = ", ".join(str(w) for w in words)
                line += f" (words: {joined})"
            return line

        # Fallback: Phase 1B raw-tail rendering (open episode or summariser
        # failed). Keeps the prompt useful while the summariser catches up.
        turns = await self.session_manager.episode_turn_tail(
            episode.id, limit=self.episodic_turns_per_episode
        )
        body: list[str] = []
        body.append(
            f"- **{episode.mode}** (episode #{episode.id}, started {episode.started_at}):"
        )
        if isinstance(words, list) and words:
            joined = ", ".join(str(w) for w in words)
            body.append(f"  - words drilled: {joined}")
        for turn in turns:
            body.append(_format_turn_line(turn))
        if len(body) == 1:
            return ""
        return "\n".join(body)

    async def build_user_prelude(
        self, session_id: int, current_episode_id: int | None
    ) -> str:
        """Compose episodic + working blocks into a single prelude.

        Returns an empty string when there is no usable context (fresh
        session, first turn of a new episode with no priors). Routers
        guard on truthiness before prepending it to the user message.
        """
        if self.session_manager is None:
            return ""

        episodic = await self.build_episodic_block(session_id, current_episode_id)
        working = (
            await self.build_working_block(current_episode_id)
            if current_episode_id is not None
            else ""
        )
        sections = [s for s in (episodic, working) if s]
        if not sections:
            return ""

        prelude = "\n\n".join(["## Recent context (this session)", *sections]).strip()
        char_len = len(prelude)
        token_estimate = max(1, char_len // 4)
        logger.info(
            "user_prelude session_id=%d episode_id=%s chars=%d ~tokens=%d",
            session_id, current_episode_id, char_len, token_estimate,
        )
        return prelude


__all__ = ["ContextEngine", "DEFAULT_LEARNER_ID"]
