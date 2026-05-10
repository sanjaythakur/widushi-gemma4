"""Curated Piper voice "personalities".

The product surface is a short, opinionated list of named personalities so
non-technical callers can pick a voice without knowing the Piper voice ids.
``warm-academic`` is the default everywhere; the rest cover the common
tutor-app moods.

Each personality maps onto a Piper ONNX voice from
``rhasspy/piper-voices`` on Hugging Face. The ``onnx`` filename also
determines the matching ``.onnx.json`` config we read at runtime to surface
the correct sample rate.

Voices are downloaded once at api container startup by
``docker/download_voices.sh`` into ``${TTS_VOICES_DIR}`` (default
``/voices``).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from fastapi import HTTPException


@dataclass(frozen=True)
class PiperVoice:
    """A single Piper voice plus its product-facing metadata."""

    voice_id: str
    """The Piper voice key (e.g. ``en_US-lessac-medium``). Drives the file
    layout under ``rhasspy/piper-voices`` and the local cache filenames."""

    description: str
    """Short, human-readable label shown in ``GET /tts/voices``."""

    @property
    def language(self) -> str:
        """BCP-47-ish language tag (e.g. ``en_US``) parsed from ``voice_id``."""
        return self.voice_id.split("-", 1)[0] if "-" in self.voice_id else self.voice_id

    @property
    def onnx_filename(self) -> str:
        return f"{self.voice_id}.onnx"

    @property
    def config_filename(self) -> str:
        return f"{self.voice_id}.onnx.json"

    def hf_subdir(self) -> str:
        """Subdirectory under ``rhasspy/piper-voices`` that holds this voice.

        Layout (from upstream): ``<lang>/<lang>_<region>/<speaker>/<quality>/``.
        For ``en_US-lessac-medium`` -> ``en/en_US/lessac/medium``.
        """
        try:
            lang_region, speaker_quality = self.voice_id.split("-", 1)
            speaker, quality = speaker_quality.rsplit("-", 1)
        except ValueError as exc:
            raise ValueError(f"unparsable voice_id: {self.voice_id!r}") from exc
        lang = lang_region.split("_", 1)[0]
        return f"{lang}/{lang_region}/{speaker}/{quality}"


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

# NOTE: Keep this list short. Each voice is ~20-60 MiB and is downloaded into
# the api container's voices volume on first boot. Adding a personality also
# means the default ``TTS_VOICES_ENABLED`` widens, lengthening cold-start.
PERSONALITIES: dict[str, PiperVoice] = {
    "warm-academic": PiperVoice(
        voice_id="en_US-lessac-medium",
        description="Warm, clear academic narration. The default tutor voice.",
    ),
    "friendly-casual": PiperVoice(
        voice_id="en_US-hfc_female-medium",
        description="Friendly, conversational US English.",
    ),
    "neutral-news": PiperVoice(
        voice_id="en_GB-alan-medium",
        description="Neutral British news-reader cadence.",
    ),
    "energetic-kid": PiperVoice(
        voice_id="en_US-kathleen-low",
        description="Bright, energetic, kid-friendly delivery.",
    ),
    "calm-storyteller": PiperVoice(
        voice_id="en_GB-jenny_dioco-medium",
        description="Calm, measured storyteller voice.",
    ),
}

DEFAULT_PERSONALITY = "warm-academic"


def resolve_personality(name: str | None) -> tuple[str, PiperVoice]:
    """Look up a personality by id, falling back to the default.

    Raises ``HTTPException(422)`` when an unknown id is supplied so the error
    surface in the routers stays consistent.
    """
    key = (name or DEFAULT_PERSONALITY).strip().lower()
    if key not in PERSONALITIES:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Unknown TTS voice personality {key!r}. Available: "
                f"{sorted(PERSONALITIES)}."
            ),
        )
    return key, PERSONALITIES[key]


def voice_paths(voice: PiperVoice, voices_dir: Path) -> tuple[Path, Path]:
    """Return ``(model_path, config_path)`` for a voice in the local cache."""
    return voices_dir / voice.onnx_filename, voices_dir / voice.config_filename


def read_sample_rate(config_path: Path, default: int = 22050) -> int:
    """Parse the Piper ``.onnx.json`` config for ``audio.sample_rate``.

    Falls back to ``default`` if the file is missing or malformed -- the
    metadata is non-critical (only surfaced to clients for resampling) so we
    never want a parse error to take TTS down.
    """
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
        return int(data.get("audio", {}).get("sample_rate", default))
    except (OSError, ValueError, KeyError, TypeError):
        return default
