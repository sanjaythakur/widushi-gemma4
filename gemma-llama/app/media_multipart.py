"""Multipart upload prep for the image/audio mode endpoints.

All helpers are tuned for Raspberry Pi 5 (4 ARM cores, no GPU):

* Images are downscaled to fit Gemma 4's 896 px vision tile and re-encoded
  as JPEG (smaller payload, fewer pixels for the vision encoder).
* Audio is transcoded to mono 16 kHz WAV via a single ffmpeg pass -- the
  format llama.cpp's mtmd Gemma 4 audio encoder consumes natively.

Returned strings are always ``data:<mime>;base64,...`` URIs so the rest of
the codebase can keep using the existing OpenAI ``image_url`` /
``input_audio`` content-part shapes.
"""
from __future__ import annotations

import asyncio
import base64
import io
import logging
import os
import shutil
import tempfile
from pathlib import Path

from fastapi import HTTPException, UploadFile
from PIL import Image

logger = logging.getLogger(__name__)


def _resolve_image_edge() -> int:
    """Pick the longest-edge cap for uploaded images.

    Default 896 matches Gemma 4's native vision tile (no wasted pixels). Drop
    to 512 via ``IMAGE_MAX_EDGE=512`` when the camera feed is the bottleneck
    on Pi 5 -- handwriting / fine OCR loses detail, but a typical "what
    object is this?" classification is still accurate and prefill is ~3x
    faster. Bump only on hosts with a real GPU.
    """
    raw = os.environ.get("IMAGE_MAX_EDGE", "896").strip()
    try:
        value = int(raw)
    except ValueError:
        logger.warning("IMAGE_MAX_EDGE=%r is not an int; falling back to 896", raw)
        return 896
    # Hard floor / ceiling so a typo cannot disable the resize entirely.
    return max(64, min(value, 4096))


_MAX_IMAGE_EDGE = _resolve_image_edge()
_JPEG_QUALITY = int(os.environ.get("IMAGE_JPEG_QUALITY", "88") or 88)

# mtmd's Gemma 4 audio path expects mono 16 kHz PCM. Hard-coding both keeps
# the conversion cheap and predictable on the Pi.
_AUDIO_SAMPLE_RATE = 16000
_AUDIO_CHANNELS = 1

# Caps to protect the Pi from accidentally huge uploads.
MAX_IMAGE_BYTES = 10 * 1024 * 1024   # 10 MiB (only `/vision/teach-object`)
MAX_AUDIO_BYTES = 25 * 1024 * 1024   # 25 MiB (every audio-bearing endpoint)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


async def _read_capped(file: UploadFile, max_bytes: int, label: str) -> bytes:
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail=f"{label} upload is empty")
    if len(data) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"{label} upload is {len(data)} bytes; max is {max_bytes}",
        )
    return data


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _require_ffmpeg() -> str:
    path = shutil.which("ffmpeg")
    if not path:
        raise HTTPException(
            status_code=500,
            detail=(
                "ffmpeg binary not found in the api container. Rebuild the "
                "image so that docker/Dockerfile.api installs ffmpeg."
            ),
        )
    return path


async def _run(cmd: list[str], cwd: str | None = None) -> tuple[int, bytes, bytes]:
    """Run a subprocess off the event loop and return (rc, stdout, stderr)."""
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        cwd=cwd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    return proc.returncode or 0, stdout, stderr


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def read_image_upload(file: UploadFile) -> str:
    """Normalize an uploaded image to a JPEG ``data:`` URI sized for Gemma 4."""
    raw = await _read_capped(file, MAX_IMAGE_BYTES, "image")
    try:
        img = Image.open(io.BytesIO(raw))
        img.load()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"unreadable image: {exc}") from exc

    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    elif img.mode == "L":
        img = img.convert("RGB")

    longest = max(img.size)
    if longest > _MAX_IMAGE_EDGE:
        scale = _MAX_IMAGE_EDGE / longest
        new_size = (max(1, int(img.size[0] * scale)), max(1, int(img.size[1] * scale)))
        img = img.resize(new_size, Image.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=_JPEG_QUALITY, optimize=True)
    return f"data:image/jpeg;base64,{_b64(buf.getvalue())}"


async def read_audio_upload(file: UploadFile) -> str:
    """Transcode any uploaded audio to a mono 16 kHz WAV ``data:`` URI."""
    raw = await _read_capped(file, MAX_AUDIO_BYTES, "audio")
    ffmpeg = _require_ffmpeg()

    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "in.bin"
        dst = Path(tmp) / "out.wav"
        src.write_bytes(raw)

        rc, _, stderr = await _run([
            ffmpeg, "-hide_banner", "-loglevel", "error",
            "-y", "-i", str(src),
            "-ac", str(_AUDIO_CHANNELS),
            "-ar", str(_AUDIO_SAMPLE_RATE),
            "-f", "wav", str(dst),
        ])
        if rc != 0 or not dst.exists():
            msg = stderr.decode("utf-8", errors="replace").strip() or "unknown ffmpeg error"
            raise HTTPException(status_code=400, detail=f"ffmpeg audio decode failed: {msg}")
        wav = dst.read_bytes()

    return f"data:audio/wav;base64,{_b64(wav)}"
