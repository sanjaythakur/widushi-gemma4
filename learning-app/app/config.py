"""Static configuration for the Widushi app.

Values here are intentionally simple module-level constants. Anything
that needs to change between Mac dev and Raspberry Pi deploy is read
from environment variables so the same image works in both places.
"""

from __future__ import annotations

import os
from pathlib import Path


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.lower() in {"1", "true", "yes", "on"}


# App layout — config.py lives at master-app/app/config.py, so the
# master app root is two parents up. The wider workspace root is one
# parent above that and contains sibling projects like openwakeword and
# pre-generated-clips.
MASTER_APP_ROOT: Path = Path(__file__).resolve().parent.parent
WORKSPACE_ROOT: Path = MASTER_APP_ROOT.parent

# The 3.5" TFT on the Pi is 480x320 in landscape. We render at this
# size on Mac too so the layout matches the deploy target.
SCREEN_WIDTH: int = 480
SCREEN_HEIGHT: int = 320
SCREEN_SIZE: tuple[int, int] = (SCREEN_WIDTH, SCREEN_HEIGHT)

# Target frame rate. Each pygame frame yields control back to the
# asyncio loop via ``await asyncio.sleep(1/FPS)`` so the orchestrator
# and uvicorn keep running.
FPS: int = 30
FRAME_INTERVAL: float = 1.0 / FPS

WINDOW_TITLE: str = "Widushi"

# FastAPI host/port for the local control plane. 8020 is chosen to
# stay clear of the Gemma service on 8010.
API_HOST: str = os.environ.get("WIDUSHI_API_HOST", "0.0.0.0")
API_PORT: int = int(os.environ.get("WIDUSHI_API_PORT", "8020"))

# Where the Gemma LLM service lives. The real httpx client lands in a
# later phase; the URL is plumbed now so callers can use it from day 1.
GEMMA_URL: str = os.environ.get("GEMMA_URL", "http://localhost:8010")
GEMMA_TIMEOUT_S: float = float(os.environ.get("GEMMA_TIMEOUT_S", "120"))
GEMMA_TTS_ENABLED: bool = _env_bool("GEMMA_TTS_ENABLED", default=True)
GEMMA_TTS_VOICE: str = os.environ.get("GEMMA_TTS_VOICE", "warm-academic")

# Pre-recorded TTS clips and the wake word model are produced by other
# sibling projects. We expose their paths here for later phases.
CLIPS_DIR: Path = WORKSPACE_ROOT / "pre-generated-clips" / "clips"
# Language subdirectory inside CLIPS_DIR. Swap to "hi" once Hindi clips
# have been generated under pre-generated-clips/clips/hi/.
CLIPS_LANG: str = os.environ.get("WIDUSHI_CLIPS_LANG", "en")
WAKE_WORD_MODEL: Path = Path(
    os.environ.get(
        "WAKE_WORD_MODEL",
        str(WORKSPACE_ROOT / "openwakeword" / "vDu_shee.onnx"),
    )
)
WAKE_WORD_ENABLED: bool = _env_bool("WAKE_WORD_ENABLED", default=True)
WAKE_WORD_THRESHOLD: float = float(os.environ.get("WAKE_WORD_THRESHOLD", "0.5"))
WAKE_WORD_DEBOUNCE_S: float = float(os.environ.get("WAKE_WORD_DEBOUNCE_S", "2.0"))
WAKE_WORD_FRAME_MS: int = int(os.environ.get("WAKE_WORD_FRAME_MS", "80"))

# Spoken-question capture after the wake word. Audio is recorded as
# mono int16 PCM at the wake-word sample rate, then wrapped as WAV for Gemma.
#
# RMS threshold above which a single 80 ms chunk is treated as "voiced".
# Tuned for a USB desk mic in a quiet-but-not-silent room (fan, distant
# AC, light typing). Crank it up if ambient noise keeps tripping the
# recorder, or down if a soft-spoken kid never registers as voice.
LISTENING_SILENCE_RMS_THRESHOLD: float = float(
    os.environ.get("LISTENING_SILENCE_RMS_THRESHOLD", "800")
)
# How many *consecutive* voiced chunks (80 ms each) the recorder must
# observe before it commits to "the user is speaking". A value of 2
# (~160 ms) filters single-chunk noise blips (fan, chair creak, tail of
# the listen_start clip echoing back through the mic) which would
# otherwise disarm the LISTENING_SILENCE_TIMEOUT_S no-voice fallback.
LISTENING_VOICE_ONSET_FRAMES: int = int(
    os.environ.get("LISTENING_VOICE_ONSET_FRAMES", "2")
)
# Length of trailing silence that ends a recording. Bumped from 1.0 s
# to 1.5 s because natural between-phrase pauses ("uh... one plus...
# one") routinely exceed 1 s and were being mistaken for end-of-turn.
LISTENING_TRAILING_SILENCE_S: float = float(
    os.environ.get("LISTENING_TRAILING_SILENCE_S", "1.5")
)
LISTENING_MIN_RECORDING_S: float = float(os.environ.get("LISTENING_MIN_RECORDING_S", "0.8"))
LISTENING_MAX_RECORDING_S: float = float(os.environ.get("LISTENING_MAX_RECORDING_S", "15.0"))
# How long LISTENING waits for the user to start speaking before giving
# up and emitting CANCEL (which the orchestrator wildcards back to
# IDLE). Applies on every LISTENING entry: after WAKE_DETECTED *and*
# after PLAYBACK_DONE (the SPEAKING -> LISTENING follow-up loopback).
LISTENING_SILENCE_TIMEOUT_S: float = float(
    os.environ.get("LISTENING_SILENCE_TIMEOUT_S", "5.0")
)

# Audio format used end-to-end (matches the pre-generated clips).
AUDIO_SAMPLE_RATE: int = 22050
AUDIO_CHANNELS: int = 1
