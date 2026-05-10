"""Piper TTS subsystem.

Wraps the Pi-5-friendly Piper binary so every API endpoint can opt into
audio output. Core pieces:

* :mod:`app.tts.voices` -- frozen "personality" registry that maps user-facing
  ids (``warm-academic``, ``friendly-casual``, ...) to actual Piper voices.
* :mod:`app.tts.engine` -- async wrapper around the ``piper`` subprocess.
* :mod:`app.tts.sentences` -- streaming sentence buffer used to flush whole
  sentences to Piper while Gemma is still generating the next one.
* :mod:`app.tts.storage` -- TTL-managed scratch directory for the long-output
  ``audio_url`` shape.
* :mod:`app.tts.integration` -- helpers each router calls to attach audio to
  its JSON response or to splice ``{"type":"audio"}`` lines into an existing
  NDJSON stream.
* :mod:`app.tts.router` -- ``GET /tts/voices``, ``GET /tts/output/{id}.wav``,
  ``POST /tts/speak``.
"""
from __future__ import annotations

from .engine import PiperEngine, PiperError
from .integration import (
    TTSIntegrationError,
    attach_file_tts,
    attach_inline_tts,
    wrap_stream_with_tts,
)
from .voices import (
    DEFAULT_PERSONALITY,
    PERSONALITIES,
    PiperVoice,
    resolve_personality,
)

__all__ = [
    "DEFAULT_PERSONALITY",
    "PERSONALITIES",
    "PiperEngine",
    "PiperError",
    "PiperVoice",
    "TTSIntegrationError",
    "attach_file_tts",
    "attach_inline_tts",
    "resolve_personality",
    "wrap_stream_with_tts",
]
