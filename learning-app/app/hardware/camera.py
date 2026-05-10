"""USB camera source.

Phase 2 stub: returns a black frame as raw RGB bytes so callers don't
need numpy yet. Real implementation will use ``cv2`` or ``picamera2``.
"""

from __future__ import annotations

import asyncio


class CameraSource:
    def __init__(self, *, width: int = 320, height: int = 240) -> None:
        self.width = width
        self.height = height

    async def snapshot(self) -> bytes:
        """Return a single ``width x height x 3`` RGB byte buffer."""

        await asyncio.sleep(0)
        return b"\x00" * (self.width * self.height * 3)
