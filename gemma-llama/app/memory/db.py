"""SQLite wrapper for the memory layer.

Phase 1B opens a single ``sqlite3.Connection`` against ``/data/widushi.db``
(bind-mounted from the host), enables WAL + foreign keys, and runs an
idempotent migration sequence keyed off ``PRAGMA user_version``. The
connection is held for the lifetime of the FastAPI process; all reads and
writes go through ``aexecute`` / ``afetchall`` / ``afetchone`` helpers
that wrap the sync sqlite3 calls in :func:`asyncio.to_thread` and a
process-wide :class:`asyncio.Lock`.

The lock is deliberately coarse: PRD §8 fixes the deployment shape as
**one learner, one session at a time on a Pi 5**, so concurrent writes
never happen in practice. The lock exists to keep ``sqlite3``'s
non-thread-safe connection objects sane across the asyncio thread pool,
not to gate contention.
"""
from __future__ import annotations

import asyncio
import logging
import sqlite3
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Migrations. One function per ``user_version`` bump. Each is invoked with
# the open sqlite3 connection and is responsible for issuing its own
# CREATE/ALTER statements. The runner sets ``PRAGMA user_version`` to the
# index+1 after each successful migration so reruns are no-ops.
# ---------------------------------------------------------------------------


def _v1_initial(conn: sqlite3.Connection) -> None:
    """Initial schema: sessions, episodes, turns + indexes (PRD §4.2)."""
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS session (
          id              INTEGER PRIMARY KEY AUTOINCREMENT,
          learner_id      TEXT NOT NULL,
          started_at      TEXT NOT NULL,
          ended_at        TEXT,
          rolling_summary TEXT,
          meta_json       TEXT
        );

        CREATE TABLE IF NOT EXISTS episode (
          id            INTEGER PRIMARY KEY AUTOINCREMENT,
          session_id    INTEGER NOT NULL REFERENCES session(id),
          mode          TEXT NOT NULL,
          started_at    TEXT NOT NULL,
          ended_at      TEXT,
          summary       TEXT,
          state_json    TEXT
        );

        CREATE TABLE IF NOT EXISTS turn (
          id             INTEGER PRIMARY KEY AUTOINCREMENT,
          episode_id     INTEGER NOT NULL REFERENCES episode(id),
          ts             TEXT NOT NULL,
          role           TEXT NOT NULL,
          text           TEXT,
          media_kind     TEXT,
          inference_ms   INTEGER,
          usage_json     TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_turn_episode_ts  ON turn(episode_id, ts);
        CREATE INDEX IF NOT EXISTS idx_episode_session  ON episode(session_id, started_at);
        CREATE INDEX IF NOT EXISTS idx_session_learner  ON session(learner_id, started_at);
        """
    )


_MIGRATIONS: list[Callable[[sqlite3.Connection], None]] = [_v1_initial]


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------


class Database:
    """Async-friendly wrapper around a single sqlite3 connection.

    Open with :meth:`start` (creates parent dirs, sets PRAGMAs, runs
    migrations). Close with :meth:`close`. All queries go through
    :meth:`aexecute` / :meth:`afetchall` / :meth:`afetchone` which take
    the write lock and dispatch the sync sqlite3 call to a worker thread.

    The connection uses ``isolation_level=None`` (autocommit); migrations
    that need transactional behaviour wrap their statements in
    ``BEGIN``/``COMMIT`` explicitly. The single-writer design means we
    do not need ``check_same_thread=False`` for correctness, but we set
    it so the asyncio thread pool can call us from any worker thread.
    """

    def __init__(self, sqlite_path: str | Path) -> None:
        self._path = Path(sqlite_path)
        self._conn: sqlite3.Connection | None = None
        self._lock = asyncio.Lock()

    @property
    def path(self) -> Path:
        return self._path

    async def start(self) -> None:
        """Open the connection, set PRAGMAs, and run migrations."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(self._open_sync)
        logger.info("sqlite ready: path=%s schema_version=%d", self._path, len(_MIGRATIONS))

    def _open_sync(self) -> None:
        conn = sqlite3.connect(
            str(self._path),
            isolation_level=None,
            check_same_thread=False,
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA synchronous=NORMAL")
        self._conn = conn
        self._run_migrations_sync()

    def _run_migrations_sync(self) -> None:
        assert self._conn is not None
        cursor = self._conn.execute("PRAGMA user_version")
        current = int(cursor.fetchone()[0])
        target = len(_MIGRATIONS)
        if current >= target:
            return
        for version in range(current, target):
            migration = _MIGRATIONS[version]
            logger.info(
                "applying sqlite migration v%d -> v%d (%s)",
                version, version + 1, migration.__name__,
            )
            migration(self._conn)
            self._conn.execute(f"PRAGMA user_version = {version + 1}")

    async def close(self) -> None:
        if self._conn is None:
            return
        conn, self._conn = self._conn, None
        await asyncio.to_thread(conn.close)

    # ------------------------------------------------------------------
    # Query helpers. All take the write lock; concurrent reads from the
    # async loop are serialised but each call still releases the GIL
    # inside the worker thread so the event loop keeps spinning.
    # ------------------------------------------------------------------

    def _require_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError("Database is not started")
        return self._conn

    async def aexecute(
        self, sql: str, params: Sequence[Any] | None = None
    ) -> int:
        """Run a write/DDL statement; returns ``lastrowid`` (0 if N/A)."""
        async with self._lock:
            return await asyncio.to_thread(self._execute_sync, sql, params or ())

    def _execute_sync(self, sql: str, params: Sequence[Any]) -> int:
        cursor = self._require_conn().execute(sql, params)
        return int(cursor.lastrowid or 0)

    async def afetchall(
        self, sql: str, params: Sequence[Any] | None = None
    ) -> list[sqlite3.Row]:
        async with self._lock:
            return await asyncio.to_thread(self._fetchall_sync, sql, params or ())

    def _fetchall_sync(self, sql: str, params: Sequence[Any]) -> list[sqlite3.Row]:
        cursor = self._require_conn().execute(sql, params)
        return list(cursor.fetchall())

    async def afetchone(
        self, sql: str, params: Sequence[Any] | None = None
    ) -> sqlite3.Row | None:
        async with self._lock:
            return await asyncio.to_thread(self._fetchone_sync, sql, params or ())

    def _fetchone_sync(self, sql: str, params: Sequence[Any]) -> sqlite3.Row | None:
        cursor = self._require_conn().execute(sql, params)
        return cursor.fetchone()

    async def arun_in_txn(self, fn: Callable[[sqlite3.Connection], Any]) -> Any:
        """Run a sync callable inside an exclusive transaction.

        Used for read-modify-write sequences (e.g. ``update_episode_state``)
        that must observe and mutate a single row atomically.
        """
        async with self._lock:
            return await asyncio.to_thread(self._run_in_txn_sync, fn)

    def _run_in_txn_sync(self, fn: Callable[[sqlite3.Connection], Any]) -> Any:
        conn = self._require_conn()
        conn.execute("BEGIN IMMEDIATE")
        try:
            result = fn(conn)
        except Exception:
            conn.execute("ROLLBACK")
            raise
        conn.execute("COMMIT")
        return result


__all__ = ["Database"]
