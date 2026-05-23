"""Piper TTS subsystem.

Wraps the Pi-5-friendly Piper binary so every mode endpoint can opt into
audio output. Core pieces:

* :mod:`app.tts.voices` -- frozen "personality" registry that maps user-
  facing ids (``warm-academic``, ``friendly-casual``, ...) to actual Piper
  voices.
* :mod:`app.tts.engine` -- async wrapper around the ``piper`` subprocess.
* :mod:`app.tts.sentences` -- streaming sentence buffer used to flush whole
  sentences to Piper while Gemma is still generating the next one.
* :mod:`app.tts.storage` -- TTL-managed scratch directory for the
  ``audio_url`` shape returned by every mode endpoint.
* :mod:`app.tts.integration` -- helpers each router calls to attach audio
  to its JSON response (``attach_file_tts``) or to splice
  ``{"type":"audio"}`` lines into an existing NDJSON stream
  (``wrap_stream_with_tts``).
* :mod:`app.tts.router` -- ``GET /tts/output/{id}.wav`` (TTL-cached WAV
  served back to clients that received an ``audio_url`` from a mode
  endpoint).
"""
from __future__ import annotations

from .engine import PiperEngine, PiperError
from .integration import (
    TTSIntegrationError,
    attach_file_tts,
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
    "resolve_personality",
    "wrap_stream_with_tts",
]
