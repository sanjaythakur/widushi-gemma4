"""Session / episode / turn persistence (PRD §4.2, Phase 1B).

This module is the **only writer** to the SQLite store. Mode routers
funnel every request through :class:`SessionManager` to:

1. Resolve (or open) the active :class:`Session` for the current learner.
2. Resolve (or open) the active :class:`Episode` for ``(session, mode)``.
3. After inference, persist one or more :class:`Turn` rows.
4. (voice-mirror only) merge into ``episode.state_json`` so the next
   call can read back ``words_drilled``.

Concurrency assumption (PRD §8): single Pi 5, single learner, single
session in flight at a time. We hold one ``asyncio.Lock`` for *every*
DB call (see :mod:`.db`); this keeps things trivially correct without
introducing row-level locking. The lock makes the "open-or-resume"
read-then-insert race safe even when two requests arrive in the same
tick.
"""
from __future__ import annotations

import json
import logging
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

from .db import Database

logger = logging.getLogger(__name__)


Role = Literal["user", "assistant"]


# ---------------------------------------------------------------------------
# Row dataclasses. Thin -- routers and the context engine pass these
# around instead of raw ``sqlite3.Row`` so the field names are typed.
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Session:
    id: int
    learner_id: str
    started_at: str
    ended_at: str | None
    rolling_summary: str | None
    meta_json: str | None


@dataclass(slots=True)
class Episode:
    id: int
    session_id: int
    mode: str
    started_at: str
    ended_at: str | None
    summary: str | None
    state_json: str | None

    def state(self) -> dict[str, Any]:
        if not self.state_json:
            return {}
        try:
            parsed = json.loads(self.state_json)
        except json.JSONDecodeError:
            logger.warning("episode %d has invalid state_json; ignoring", self.id)
            return {}
        return parsed if isinstance(parsed, dict) else {}


@dataclass(slots=True)
class Turn:
    id: int
    episode_id: int
    ts: str
    role: Role
    text: str | None
    media_kind: str | None
    inference_ms: int | None
    usage_json: str | None


@dataclass(slots=True)
class EpisodeView:
    """Episode + its denormalised turn tail, for the ops endpoint."""
    episode: Episode
    turn_count: int
    recent_turns: list[Turn] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow_iso() -> str:
    """ISO-8601 UTC, trailing ``Z``, second precision."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _row_to_session(row: sqlite3.Row) -> Session:
    return Session(
        id=int(row["id"]),
        learner_id=str(row["learner_id"]),
        started_at=str(row["started_at"]),
        ended_at=row["ended_at"],
        rolling_summary=row["rolling_summary"],
        meta_json=row["meta_json"],
    )


def _row_to_episode(row: sqlite3.Row) -> Episode:
    return Episode(
        id=int(row["id"]),
        session_id=int(row["session_id"]),
        mode=str(row["mode"]),
        started_at=str(row["started_at"]),
        ended_at=row["ended_at"],
        summary=row["summary"],
        state_json=row["state_json"],
    )


def _row_to_turn(row: sqlite3.Row) -> Turn:
    return Turn(
        id=int(row["id"]),
        episode_id=int(row["episode_id"]),
        ts=str(row["ts"]),
        role=row["role"],
        text=row["text"],
        media_kind=row["media_kind"],
        inference_ms=row["inference_ms"],
        usage_json=row["usage_json"],
    )


def _parse_iso(ts: str) -> datetime:
    """Parse the ``..Z`` ISO strings we write back out."""
    try:
        if ts.endswith("Z"):
            return datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        return datetime.fromisoformat(ts)
    except ValueError:
        logger.warning("failed to parse timestamp %r; treating as epoch", ts)
        return datetime.fromtimestamp(0, tz=timezone.utc)


# ---------------------------------------------------------------------------
# SessionManager
# ---------------------------------------------------------------------------


class SessionManager:
    """Owns all writes to ``session`` / ``episode`` / ``turn``."""

    def __init__(
        self,
        db: Database,
        *,
        idle_timeout_seconds: int = 1800,
        working_k_turns: int = 3,
        episodic_max_episodes: int = 3,
        episodic_turns_per_episode: int = 2,
    ) -> None:
        self.db = db
        self.idle_timeout_seconds = idle_timeout_seconds
        self.working_k_turns = working_k_turns
        self.episodic_max_episodes = episodic_max_episodes
        self.episodic_turns_per_episode = episodic_turns_per_episode

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    async def open_or_resume_session(
        self,
        learner_id: str,
        hint_session_id: int | None = None,
    ) -> Session:
        """Return the active session for ``learner_id``.

        Resolution order:

        1. If ``hint_session_id`` is given AND that session is open AND
           it belongs to ``learner_id``, reuse it.
        2. Else, if the learner's most-recent open session had activity
           within ``idle_timeout_seconds``, reuse it.
        3. Else, open a brand-new session.

        The "activity" check looks at the latest of ``session.started_at``
        and the newest ``turn.ts`` under any episode in that session.
        """
        if hint_session_id is not None:
            session = await self._get_session(hint_session_id)
            if session is not None and session.learner_id == learner_id and session.ended_at is None:
                return session
            if session is not None:
                logger.info(
                    "ignoring hint session_id=%d (learner mismatch or already ended); opening fresh",
                    hint_session_id,
                )

        latest_open = await self._latest_open_session(learner_id)
        if latest_open is not None:
            last_activity_ts = await self._last_activity_iso(latest_open.id)
            anchor = last_activity_ts or latest_open.started_at
            age = (datetime.now(timezone.utc) - _parse_iso(anchor)).total_seconds()
            if age <= self.idle_timeout_seconds:
                return latest_open
            logger.info(
                "session %d idle for %.0fs (>%ds); closing and opening new",
                latest_open.id, age, self.idle_timeout_seconds,
            )
            await self.end_session(latest_open.id)

        return await self._create_session(learner_id)

    async def _get_session(self, session_id: int) -> Session | None:
        row = await self.db.afetchone(
            "SELECT * FROM session WHERE id = ?", (session_id,)
        )
        return _row_to_session(row) if row else None

    async def _latest_open_session(self, learner_id: str) -> Session | None:
        row = await self.db.afetchone(
            "SELECT * FROM session "
            "WHERE learner_id = ? AND ended_at IS NULL "
            "ORDER BY started_at DESC LIMIT 1",
            (learner_id,),
        )
        return _row_to_session(row) if row else None

    async def _last_activity_iso(self, session_id: int) -> str | None:
        row = await self.db.afetchone(
            "SELECT MAX(t.ts) AS last_ts "
            "FROM turn t JOIN episode e ON e.id = t.episode_id "
            "WHERE e.session_id = ?",
            (session_id,),
        )
        if row is None:
            return None
        last = row["last_ts"]
        return str(last) if last else None

    async def _create_session(self, learner_id: str) -> Session:
        now = _utcnow_iso()
        new_id = await self.db.aexecute(
            "INSERT INTO session(learner_id, started_at) VALUES (?, ?)",
            (learner_id, now),
        )
        logger.info("opened session id=%d learner_id=%s", new_id, learner_id)
        return Session(
            id=new_id,
            learner_id=learner_id,
            started_at=now,
            ended_at=None,
            rolling_summary=None,
            meta_json=None,
        )

    async def end_session(self, session_id: int) -> str:
        """Close the session and any still-open episodes; return ended_at."""
        now = _utcnow_iso()
        await self.db.aexecute(
            "UPDATE episode SET ended_at = ? "
            "WHERE session_id = ? AND ended_at IS NULL",
            (now, session_id),
        )
        await self.db.aexecute(
            "UPDATE session SET ended_at = ? WHERE id = ? AND ended_at IS NULL",
            (now, session_id),
        )
        logger.info("closed session id=%d at %s", session_id, now)
        return now

    # ------------------------------------------------------------------
    # Episode lifecycle (parallel-per-mode per design choice; PRD §5)
    # ------------------------------------------------------------------

    async def open_or_resume_episode(self, session_id: int, mode: str) -> Episode:
        """Return the in-flight episode for ``(session_id, mode)`` or insert one."""
        row = await self.db.afetchone(
            "SELECT * FROM episode "
            "WHERE session_id = ? AND mode = ? AND ended_at IS NULL "
            "ORDER BY started_at DESC LIMIT 1",
            (session_id, mode),
        )
        if row is not None:
            return _row_to_episode(row)

        now = _utcnow_iso()
        new_id = await self.db.aexecute(
            "INSERT INTO episode(session_id, mode, started_at, state_json) "
            "VALUES (?, ?, ?, ?)",
            (session_id, mode, now, "{}"),
        )
        logger.info(
            "opened episode id=%d session_id=%d mode=%s", new_id, session_id, mode
        )
        return Episode(
            id=new_id,
            session_id=session_id,
            mode=mode,
            started_at=now,
            ended_at=None,
            summary=None,
            state_json="{}",
        )

    async def close_episode(
        self, episode_id: int, *, summary: str | None
    ) -> str:
        """Mark ``episode`` closed (idempotent) and persist its summary.

        Sets ``ended_at`` only if the row is still open so re-running a
        close (e.g. duplicate ``X-Episode-Hint: close`` retry) preserves
        the original close time. ``summary`` is *always* written -- a
        late retry that finally succeeds in summarising overwrites a
        previous ``NULL``.
        """
        now = _utcnow_iso()
        await self.db.aexecute(
            "UPDATE episode "
            "SET summary = ?, "
            "    ended_at = COALESCE(ended_at, ?) "
            "WHERE id = ?",
            (summary, now, episode_id),
        )
        logger.info(
            "closed episode id=%d summary_chars=%d",
            episode_id, len(summary) if summary else 0,
        )
        return now

    async def update_rolling_summary(
        self, session_id: int, rolling_summary: str
    ) -> None:
        """Single-column UPDATE for ``session.rolling_summary``."""
        await self.db.aexecute(
            "UPDATE session SET rolling_summary = ? WHERE id = ?",
            (rolling_summary, session_id),
        )
        logger.info(
            "updated rolling summary for session_id=%d chars=%d",
            session_id, len(rolling_summary),
        )

    async def bump_meta_counter(self, session_id: int, key: str) -> int:
        """Atomically increment ``session.meta_json[key]`` and return new value."""

        def _txn(conn: sqlite3.Connection) -> int:
            row = conn.execute(
                "SELECT meta_json FROM session WHERE id = ?", (session_id,)
            ).fetchone()
            if row is None:
                raise ValueError(f"session {session_id} not found")
            current_raw = row["meta_json"] or "{}"
            try:
                current = json.loads(current_raw)
                if not isinstance(current, dict):
                    current = {}
            except json.JSONDecodeError:
                current = {}
            try:
                prev = int(current.get(key, 0))
            except (TypeError, ValueError):
                prev = 0
            new_value = prev + 1
            current[key] = new_value
            conn.execute(
                "UPDATE session SET meta_json = ? WHERE id = ?",
                (json.dumps(current, ensure_ascii=False), session_id),
            )
            return new_value

        return await self.db.arun_in_txn(_txn)

    async def set_meta_counter(
        self, session_id: int, key: str, value: int
    ) -> None:
        """Set ``session.meta_json[key] = value`` atomically."""

        def _txn(conn: sqlite3.Connection) -> None:
            row = conn.execute(
                "SELECT meta_json FROM session WHERE id = ?", (session_id,)
            ).fetchone()
            if row is None:
                raise ValueError(f"session {session_id} not found")
            current_raw = row["meta_json"] or "{}"
            try:
                current = json.loads(current_raw)
                if not isinstance(current, dict):
                    current = {}
            except json.JSONDecodeError:
                current = {}
            current[key] = int(value)
            conn.execute(
                "UPDATE session SET meta_json = ? WHERE id = ?",
                (json.dumps(current, ensure_ascii=False), session_id),
            )

        await self.db.arun_in_txn(_txn)

    async def update_episode_state(
        self,
        episode_id: int,
        mutator: Callable[[dict[str, Any]], dict[str, Any]],
    ) -> dict[str, Any]:
        """Read-modify-write ``episode.state_json`` atomically."""

        def _txn(conn: sqlite3.Connection) -> dict[str, Any]:
            row = conn.execute(
                "SELECT state_json FROM episode WHERE id = ?", (episode_id,)
            ).fetchone()
            if row is None:
                raise ValueError(f"episode {episode_id} not found")
            current_raw = row["state_json"] or "{}"
            try:
                current = json.loads(current_raw)
                if not isinstance(current, dict):
                    current = {}
            except json.JSONDecodeError:
                current = {}
            updated = mutator(current) or {}
            conn.execute(
                "UPDATE episode SET state_json = ? WHERE id = ?",
                (json.dumps(updated, ensure_ascii=False), episode_id),
            )
            return updated

        return await self.db.arun_in_txn(_txn)

    # ------------------------------------------------------------------
    # Turn writes
    # ------------------------------------------------------------------

    async def record_turn(
        self,
        episode_id: int,
        *,
        role: Role,
        text: str | None,
        media_kind: str | None = None,
        inference_ms: int | None = None,
        usage: dict[str, Any] | None = None,
    ) -> Turn:
        now = _utcnow_iso()
        usage_json = json.dumps(usage, ensure_ascii=False) if usage else None
        new_id = await self.db.aexecute(
            "INSERT INTO turn(episode_id, ts, role, text, media_kind, inference_ms, usage_json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (episode_id, now, role, text, media_kind, inference_ms, usage_json),
        )
        return Turn(
            id=new_id,
            episode_id=episode_id,
            ts=now,
            role=role,
            text=text,
            media_kind=media_kind,
            inference_ms=inference_ms,
            usage_json=usage_json,
        )

    # ------------------------------------------------------------------
    # Reads for the ContextEngine + ops endpoint
    # ------------------------------------------------------------------

    async def current_episode_turns(
        self, episode_id: int, *, limit: int
    ) -> list[Turn]:
        """Last ``limit`` turns in chronological order."""
        if limit <= 0:
            return []
        rows = await self.db.afetchall(
            "SELECT * FROM ( "
            "  SELECT * FROM turn WHERE episode_id = ? "
            "  ORDER BY ts DESC, id DESC LIMIT ? "
            ") sub ORDER BY ts ASC, id ASC",
            (episode_id, limit),
        )
        return [_row_to_turn(r) for r in rows]

    async def prior_episodes(
        self,
        session_id: int,
        *,
        exclude_episode_id: int | None,
        limit: int,
    ) -> list[Episode]:
        """Other episodes in this session, newest first, capped at ``limit``."""
        if limit <= 0:
            return []
        if exclude_episode_id is None:
            rows = await self.db.afetchall(
                "SELECT * FROM episode WHERE session_id = ? "
                "ORDER BY started_at DESC, id DESC LIMIT ?",
                (session_id, limit),
            )
        else:
            rows = await self.db.afetchall(
                "SELECT * FROM episode WHERE session_id = ? AND id != ? "
                "ORDER BY started_at DESC, id DESC LIMIT ?",
                (session_id, exclude_episode_id, limit),
            )
        return [_row_to_episode(r) for r in rows]

    async def episode_turn_tail(
        self, episode_id: int, *, limit: int
    ) -> list[Turn]:
        """Alias of :meth:`current_episode_turns`; kept for read-site clarity."""
        return await self.current_episode_turns(episode_id, limit=limit)

    async def get_episode(self, episode_id: int) -> Episode | None:
        """Return the freshest :class:`Episode` row, or ``None`` if missing."""
        row = await self.db.afetchone(
            "SELECT * FROM episode WHERE id = ?", (episode_id,)
        )
        return _row_to_episode(row) if row else None

    async def episode_turns_full(self, episode_id: int) -> list[Turn]:
        """Return every turn for ``episode_id`` in chronological order.

        Phase 2 summariser input. Bounded in practice by how many turns
        an episode accumulates before close; the per-line char cap in the
        summariser itself protects against runaways.
        """
        rows = await self.db.afetchall(
            "SELECT * FROM turn WHERE episode_id = ? ORDER BY ts ASC, id ASC",
            (episode_id,),
        )
        return [_row_to_turn(r) for r in rows]

    async def closed_episode_summaries(
        self, session_id: int, *, limit: int
    ) -> list[tuple[int, str, str]]:
        """Return the most recent ``limit`` ``(id, mode, summary)`` triples.

        Only rows whose ``summary`` is non-null are returned. Ordered
        chronologically (oldest first) so the rolling summariser sees a
        natural timeline.
        """
        if limit <= 0:
            return []
        rows = await self.db.afetchall(
            "SELECT * FROM ( "
            "  SELECT id, mode, summary, started_at FROM episode "
            "  WHERE session_id = ? AND summary IS NOT NULL "
            "  ORDER BY ended_at DESC, id DESC LIMIT ? "
            ") sub ORDER BY started_at ASC, id ASC",
            (session_id, limit),
        )
        return [(int(r["id"]), str(r["mode"]), str(r["summary"])) for r in rows]

    async def episode_turn_count(self, episode_id: int) -> int:
        row = await self.db.afetchone(
            "SELECT COUNT(*) AS n FROM turn WHERE episode_id = ?", (episode_id,)
        )
        return int(row["n"]) if row else 0

    async def get_session_view(
        self, session_id: int, *, episode_limit: int = 10, turn_tail: int = 5
    ) -> tuple[Session, list[EpisodeView]] | None:
        """Return session + denormalised episode list for ``GET /sessions/{id}``."""
        session = await self._get_session(session_id)
        if session is None:
            return None
        ep_rows = await self.db.afetchall(
            "SELECT * FROM episode WHERE session_id = ? "
            "ORDER BY started_at ASC, id ASC LIMIT ?",
            (session_id, episode_limit),
        )
        views: list[EpisodeView] = []
        for ep_row in ep_rows:
            episode = _row_to_episode(ep_row)
            count = await self.episode_turn_count(episode.id)
            tail = await self.episode_turn_tail(episode.id, limit=turn_tail)
            views.append(EpisodeView(episode=episode, turn_count=count, recent_turns=tail))
        return session, views


__all__ = [
    "Episode",
    "EpisodeView",
    "Session",
    "SessionManager",
    "Turn",
]
