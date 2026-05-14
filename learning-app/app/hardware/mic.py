"""USB microphone source."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

from app import config

DEFAULT_SAMPLE_RATE = 16000
DEFAULT_CHUNK_MS = 80

# Sentinel so callers can pass ``device=None`` to force PortAudio's
# default and still distinguish that from "not specified, fall back to
# ``config.MIC_INPUT_DEVICE``".
_USE_CONFIG_DEFAULT: object = object()


class MicSource:
    def __init__(
        self,
        *,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        chunk_ms: int = DEFAULT_CHUNK_MS,
        device: int | str | None | object = _USE_CONFIG_DEFAULT,
        stub: bool = True,
    ) -> None:
        self.sample_rate = sample_rate
        self.chunk_ms = chunk_ms
        self.device = config.MIC_INPUT_DEVICE if device is _USE_CONFIG_DEFAULT else device
        self.stub = stub
        self._queue: asyncio.Queue[bytes] | None = None

    def drain(self) -> int:
        """Discard any chunks currently buffered in the capture queue.

        Used after playing a TTS clip through the same physical device
        as the mic: while the clip plays, sounddevice keeps capturing
        and the bounded queue fills up. Without draining, the recorder
        would start with up to ``maxsize`` chunks of self-captured clip
        audio. Returns the number of chunks dropped (handy for logs).
        """

        queue = self._queue
        if queue is None:
            return 0
        dropped = 0
        while not queue.empty():
            try:
                queue.get_nowait()
                dropped += 1
            except asyncio.QueueEmpty:
                break
        return dropped

    async def frames(self) -> AsyncIterator[bytes]:
        """Yield mono int16 PCM chunks.

        Stub mode keeps tests and headless development independent of audio
        hardware. Real capture uses sounddevice's callback thread and bridges
        chunks back into the asyncio loop through a queue.
        """

        if not self.stub:
            async for chunk in self._sounddevice_frames():
                yield chunk
            return

        chunk_samples = self._chunk_samples()
        chunk = b"\x00\x00" * chunk_samples
        delay = self.chunk_ms / 1000
        while True:
            await asyncio.sleep(delay)
            yield chunk

    def _chunk_samples(self) -> int:
        return int(self.sample_rate * self.chunk_ms / 1000)

    async def _sounddevice_frames(self) -> AsyncIterator[bytes]:
        try:
            import sounddevice as sd  # noqa: PLC0415 - optional runtime dependency
        except ImportError as exc:
            raise RuntimeError(
                "sounddevice is required for live microphone capture; "
                "install dependencies or run MicSource(stub=True)"
            ) from exc

        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=8)
        self._queue = queue
        chunk_samples = int(self.sample_rate * self.chunk_ms / 1000)
        blocksize = chunk_samples

        # Surface a clear log line so ALSA "no card" surprises are easy
        # to debug. ``self.device is None`` means PortAudio default →
        # ALSA ``default`` PCM (controlled by ``/etc/asound.conf``).
        import logging  # noqa: PLC0415 - kept local; mic is rarely imported
        logging.getLogger(__name__).info(
            "mic input: device=%r sample_rate=%d chunk_ms=%d",
            self.device, self.sample_rate, self.chunk_ms,
        )

        def callback(
            indata: Any,
            frames: int,
            time: Any,  # noqa: ARG001
            status: Any,
        ) -> None:
            if status:
                # sounddevice statuses are diagnostic; dropping the current
                # buffer is better than blocking its real-time callback.
                return
            if frames != blocksize:
                return
            data = bytes(indata)

            def put_nowait() -> None:
                if queue.full():
                    try:
                        queue.get_nowait()
                    except asyncio.QueueEmpty:
                        pass
                queue.put_nowait(data)

            loop.call_soon_threadsafe(put_nowait)

        try:
            with sd.RawInputStream(
                samplerate=self.sample_rate,
                blocksize=blocksize,
                channels=1,
                dtype="int16",
                device=self.device,
                callback=callback,
            ):
                while True:
                    yield await queue.get()
        finally:
            self._queue = None
