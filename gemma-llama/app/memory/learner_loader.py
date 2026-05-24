"""File-backed learner profile loader with mtime-invalidated cache.

A profile is a Markdown file under :attr:`LearnerLoader.profiles_dir` named
``<learner_id>.md``. The leading YAML front-matter (delimited by ``---``)
holds the structured facts; the trailing Markdown body holds the prose
sections (``## About the learner``, ``## Learning goals``,
``## Notes for the tutor``, ...).

Phase 1A ships exactly one canonical profile (``kalzy.md``) and treats
``"kalzy"`` as the default ``learner_id``. Future phases (onboarding) will
write to the same file shape; nothing else needs to change here.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


DEFAULT_LEARNER_ID = "kalzy"


@dataclass(frozen=True)
class LearnerProfile:
    """Parsed learner profile (YAML front-matter + Markdown body)."""

    learner_id: str
    name: str | None = None
    age: int | None = None
    gender: str | None = None
    location: str | None = None
    l1: str | None = None
    l1_secondary: str | None = None
    l2_target: str | None = None
    proficiency: str | None = None
    interests: list[str] = field(default_factory=list)
    occupation: str | None = None
    body_markdown: str = ""
    raw_front_matter: dict[str, Any] = field(default_factory=dict)


def _split_front_matter(text: str) -> tuple[dict[str, Any], str]:
    """Split a Markdown file with leading ``---`` YAML front-matter.

    Returns ``(front_matter_dict, body)``. If the file has no front-matter
    at all we return ``({}, text)`` so a profile that's just prose still
    loads (just without the structured facts).
    """
    stripped = text.lstrip("\ufeff")  # tolerate BOM
    if not stripped.startswith("---"):
        return {}, text

    # Drop the leading delimiter line, find the closing one.
    rest = stripped[3:]
    if rest.startswith("\n"):
        rest = rest[1:]
    end = rest.find("\n---")
    if end == -1:
        return {}, text

    front_raw = rest[:end]
    body = rest[end + 4 :]
    if body.startswith("\n"):
        body = body[1:]

    try:
        data = yaml.safe_load(front_raw) or {}
    except yaml.YAMLError as exc:
        logger.warning("learner profile front-matter YAML parse failed: %s", exc)
        return {}, body

    if not isinstance(data, dict):
        return {}, body
    return data, body


def _coerce_interests(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, str):
        return [s.strip() for s in value.split(",") if s.strip()]
    return [str(value)]


def _profile_from_file(learner_id: str, path: Path) -> LearnerProfile:
    text = path.read_text(encoding="utf-8")
    front, body = _split_front_matter(text)

    age_raw = front.get("age")
    try:
        age = int(age_raw) if age_raw is not None else None
    except (TypeError, ValueError):
        age = None

    return LearnerProfile(
        learner_id=str(front.get("learner_id") or learner_id),
        name=(str(front["name"]).strip() if front.get("name") else None),
        age=age,
        gender=(str(front["gender"]).strip() if front.get("gender") else None),
        location=(str(front["location"]).strip() if front.get("location") else None),
        l1=(str(front["l1"]).strip() if front.get("l1") else None),
        l1_secondary=(
            str(front["l1_secondary"]).strip() if front.get("l1_secondary") else None
        ),
        l2_target=(str(front["l2_target"]).strip() if front.get("l2_target") else None),
        proficiency=(
            str(front["proficiency"]).strip() if front.get("proficiency") else None
        ),
        interests=_coerce_interests(front.get("interests")),
        occupation=(
            str(front["occupation"]).strip() if front.get("occupation") else None
        ),
        body_markdown=body.strip(),
        raw_front_matter=front,
    )


class LearnerLoader:
    """Load ``<profiles_dir>/<learner_id>.md`` with an mtime-invalidated cache.

    The cache stores ``(mtime_ns, LearnerProfile)`` per ``learner_id``. On
    each :meth:`load` call we ``os.stat`` the file and re-read it iff the
    mtime moved. That means the operator can edit a profile on the host
    and the next request picks it up without restarting the api container.
    """

    def __init__(self, profiles_dir: Path) -> None:
        self.profiles_dir = Path(profiles_dir)
        self._cache: dict[str, tuple[int, LearnerProfile]] = {}
        self._lock = threading.Lock()
        self._missing_kalzy_logged = False

    def _path_for(self, learner_id: str) -> Path:
        return self.profiles_dir / f"{learner_id}.md"

    def load(self, learner_id: str) -> LearnerProfile | None:
        """Return the parsed profile for ``learner_id`` or ``None`` if missing."""
        path = self._path_for(learner_id)
        try:
            stat = path.stat()
        except FileNotFoundError:
            return None
        except OSError as exc:
            logger.warning("learner profile stat failed for %s: %s", path, exc)
            return None

        mtime_ns = stat.st_mtime_ns
        with self._lock:
            cached = self._cache.get(learner_id)
            if cached is not None and cached[0] == mtime_ns:
                return cached[1]

        try:
            profile = _profile_from_file(learner_id, path)
        except OSError as exc:
            logger.warning("learner profile read failed for %s: %s", path, exc)
            return None

        with self._lock:
            self._cache[learner_id] = (mtime_ns, profile)
        return profile

    def resolve(
        self, requested: str | None
    ) -> tuple[str, LearnerProfile | None]:
        """Resolve a (possibly missing or unknown) ``learner_id``.

        Order:
        1. ``requested`` if present and the file exists.
        2. Fallback to :data:`DEFAULT_LEARNER_ID` (``"kalzy"``).

        Returns ``(resolved_id, profile_or_None)``. The id is *always*
        returned (defaults to ``"kalzy"``) so the response contract stays
        stable even if the profile file is missing on disk.
        """
        if requested:
            requested = requested.strip()
        if requested:
            profile = self.load(requested)
            if profile is not None:
                return requested, profile
            if requested != DEFAULT_LEARNER_ID:
                logger.warning(
                    "learner_id %r not found under %s; falling back to %r",
                    requested,
                    self.profiles_dir,
                    DEFAULT_LEARNER_ID,
                )

        default = self.load(DEFAULT_LEARNER_ID)
        if default is None and not self._missing_kalzy_logged:
            logger.error(
                "default learner profile %r missing under %s; "
                "responses will use an empty learner_block",
                DEFAULT_LEARNER_ID,
                self.profiles_dir,
            )
            self._missing_kalzy_logged = True
        return DEFAULT_LEARNER_ID, default
