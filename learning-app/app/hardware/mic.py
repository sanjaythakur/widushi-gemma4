"""USB microphone source."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

DEFAULT_SAMPLE_RATE = 16000
DEFAULT_CHUNK_MS = 80


class MicSource:
    def __init__(
        self,
        *,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        chunk_ms: int = DEFAULT_CHUNK_MS,
        device: int | str | None = None,
        stub: bool = True,
    ) -> None:
        self.sample_rate = sample_rate
        self.chunk_ms = chunk_ms
        self.device = device
        self.stub = stub

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
        chunk_samples = int(self.sample_rate * self.chunk_ms / 1000)
        blocksize = chunk_samples

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
