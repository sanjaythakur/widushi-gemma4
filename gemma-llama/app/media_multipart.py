"""Multipart upload prep for the vision / audio / video tutor endpoints.

All helpers are tuned for Raspberry Pi 5 (4 ARM cores, no GPU):

* Images are downscaled to fit Gemma 4's 896 px vision tile and re-encoded as
  JPEG (smaller payload, fewer pixels for the vision encoder).
* Audio is transcoded to mono 16 kHz WAV via a single ffmpeg pass - the format
  llama.cpp's mtmd Gemma 4 audio encoder consumes natively.
* Video is decoded once with ffmpeg into N evenly-spaced JPEG frames plus an
  optional mono 16 kHz WAV track. One ffmpeg invocation per asset keeps the
  Pi's CPU budget reasonable.

Returned strings are always ``data:<mime>;base64,...`` URIs so the rest of the
codebase can keep using the existing OpenAI ``image_url`` / ``input_audio``
content-part shapes.
"""
from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import shutil
import tempfile
from pathlib import Path

from fastapi import HTTPException, UploadFile
from PIL import Image

logger = logging.getLogger(__name__)

# Gemma 4's vision tower tiles at 896 px. Anything larger is wasted bandwidth
# and Pi RAM; smaller is fine but loses detail in handwritten work.
_MAX_IMAGE_EDGE = 896
_JPEG_QUALITY = 88

# mtmd's Gemma 4 audio path expects mono 16 kHz PCM. Hard-coding both keeps
# the conversion cheap and predictable on the Pi.
_AUDIO_SAMPLE_RATE = 16000
_AUDIO_CHANNELS = 1

# Caps to protect the Pi from accidentally huge uploads.
MAX_IMAGE_BYTES = 10 * 1024 * 1024   # 10 MiB
MAX_AUDIO_BYTES = 25 * 1024 * 1024   # 25 MiB
MAX_VIDEO_BYTES = 75 * 1024 * 1024   # 75 MiB

# Frame budget for /video/analyze-process. Each frame goes through the
# vision projector (~64-86 s per slice on Pi 5 CPU at the default mmproj
# tile size!) so the cap here is the dominant driver of end-to-end latency.
# Defaults are deliberately conservative: 3 frames = ~3-4 min of preprocess
# on a Pi 5, 6 frames = ~6-8 min. Bump only on a host with a real GPU.
DEFAULT_VIDEO_FRAMES = 3
MAX_VIDEO_FRAMES = 6


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


async def extract_video_payload(
    file: UploadFile,
    n_frames: int = DEFAULT_VIDEO_FRAMES,
    include_audio: bool = True,
) -> tuple[list[str], str | None, dict]:
    """Sample ``n_frames`` evenly spaced JPEG frames + (optional) mono 16k WAV.

    Returns a tuple ``(image_data_urls, audio_data_url_or_None, meta)`` where
    ``meta`` carries the actual frame count and a probed duration in seconds
    (best-effort; ``None`` if ffprobe could not read it).
    """
    if n_frames < 1:
        raise HTTPException(status_code=422, detail="n_frames must be >= 1")
    if n_frames > MAX_VIDEO_FRAMES:
        raise HTTPException(
            status_code=422,
            detail=f"n_frames={n_frames} exceeds Pi 5 cap of {MAX_VIDEO_FRAMES}",
        )

    raw = await _read_capped(file, MAX_VIDEO_BYTES, "video")
    ffmpeg = _require_ffmpeg()
    ffprobe = shutil.which("ffprobe")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        src = tmp_path / "in.bin"
        src.write_bytes(raw)

        duration_s: float | None = None
        if ffprobe:
            rc, stdout, _ = await _run([
                ffprobe, "-hide_banner", "-loglevel", "error",
                "-print_format", "json",
                "-show_entries", "format=duration",
                str(src),
            ])
            if rc == 0:
                try:
                    duration_s = float(json.loads(stdout)["format"]["duration"])
                except (KeyError, ValueError, json.JSONDecodeError):
                    duration_s = None

        frames_pattern = tmp_path / "frame_%03d.jpg"
        # If we know the duration, sample at exactly n_frames evenly spaced
        # times; otherwise fall back to a uniform fps that yields ~n_frames for
        # a typical ~10 s clip (good enough for tutoring use-cases).
        if duration_s and duration_s > 0:
            fps = max(0.1, n_frames / max(duration_s, 0.1))
            vf = f"fps={fps:.4f},scale='min({_MAX_IMAGE_EDGE},iw)':-2"
        else:
            vf = f"fps=1,scale='min({_MAX_IMAGE_EDGE},iw)':-2"

        rc, _, stderr = await _run([
            ffmpeg, "-hide_banner", "-loglevel", "error",
            "-y", "-i", str(src),
            "-vf", vf,
            "-frames:v", str(n_frames),
            "-q:v", "3",
            str(frames_pattern),
        ])
        if rc != 0:
            msg = stderr.decode("utf-8", errors="replace").strip() or "unknown ffmpeg error"
            raise HTTPException(status_code=400, detail=f"ffmpeg frame extract failed: {msg}")

        frame_files = sorted(tmp_path.glob("frame_*.jpg"))
        if not frame_files:
            raise HTTPException(
                status_code=400,
                detail="ffmpeg produced no frames; is the upload a valid video?",
            )
        image_urls = [
            f"data:image/jpeg;base64,{_b64(p.read_bytes())}" for p in frame_files
        ]

        audio_url: str | None = None
        if include_audio:
            audio_dst = tmp_path / "audio.wav"
            rc, _, stderr = await _run([
                ffmpeg, "-hide_banner", "-loglevel", "error",
                "-y", "-i", str(src),
                "-vn",
                "-ac", str(_AUDIO_CHANNELS),
                "-ar", str(_AUDIO_SAMPLE_RATE),
                "-f", "wav", str(audio_dst),
            ])
            if rc == 0 and audio_dst.exists() and audio_dst.stat().st_size > 44:
                audio_url = f"data:audio/wav;base64,{_b64(audio_dst.read_bytes())}"
            else:
                logger.info(
                    "video has no usable audio track; continuing with frames only"
                )

    meta = {"frames_used": len(image_urls), "duration_s": duration_s}
    return image_urls, audio_url, meta
