"""Helpers for recording spoken user utterances."""

from __future__ import annotations

import asyncio
import io
import wave


class UtteranceStopSignal:
    """Shared signal used by manual controls to finish active recording."""

    def __init__(self) -> None:
        self._event = asyncio.Event()
        self._active = False

    def activate(self) -> None:
        self._event.clear()
        self._active = True

    def deactivate(self) -> None:
        self._active = False
        self._event.clear()

    def request_stop(self) -> bool:
        if not self._active:
            return False
        self._event.set()
        return True

    @property
    def requested(self) -> bool:
        return self._event.is_set()


class FollowUpListenSignal:
    """Shared one-shot signal: FSM tells the wake-word source to start a
    follow-up listen without re-requiring the wake word.

    The FSM fires ``request()`` from the ``SPEAKING -> LISTENING``
    transition side effect on ``PLAYBACK_DONE``. The wake-word source
    polls ``consume()`` on each mic chunk and, when it returns ``True``,
    drops its wake-word frame buffer and enters recording immediately.
    """

    def __init__(self) -> None:
        self._pending = False

    def request(self) -> None:
        self._pending = True

    def consume(self) -> bool:
        if not self._pending:
            return False
        self._pending = False
        return True

    @property
    def pending(self) -> bool:
        return self._pending


def pcm16_mono_to_wav(pcm: bytes, *, sample_rate: int) -> bytes:
    """Wrap mono int16 PCM bytes in a WAV container."""

    out = io.BytesIO()
    with wave.open(out, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm)
    return out.getvalue()
