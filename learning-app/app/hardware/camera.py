"""USB / Pi camera source.

Returns a JPEG-encoded camera snapshot via :meth:`CameraSource.snapshot`.

Real-hardware path: tries `picamera2` first (the libcamera-based Pi 5
stack) and falls back to OpenCV's `VideoCapture` if installed. Both
imports happen lazily so the host dev environment (Mac, CI, headless
tests) does not need either package on the dependency tree.

Headless fallback: when neither backend can be initialised (or
``WIDUSHI_CAMERA_STUB=1`` is set), :meth:`snapshot` returns a tiny
pre-built JPEG placeholder so the rest of the pipeline keeps moving.
The gemma-llama service is tolerant of any decodable JPEG, so the
placeholder still exercises the multipart upload path end-to-end on
dev machines without a camera.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import os

log = logging.getLogger(__name__)

# A tiny 1x1 black JPEG. Base64-decoded once at import time so
# :meth:`snapshot` is just a memcpy on the fallback path.
_PLACEHOLDER_JPEG: bytes = base64.b64decode(
    "/9j/4AAQSkZJRgABAQEASABIAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0a"
    "HBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/2wBDAQkJCQwLDBgNDRgyIRwhMjIyMjIy"
    "MjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjL/wAARCAABAAEDASIA"
    "AhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAr/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/8QAFAEB"
    "AAAAAAAAAAAAAAAAAAAAAP/EABQRAQAAAAAAAAAAAAAAAAAAAAD/2gAMAwEAAhEDEQA/AL+AB//Z"
)


def _stub_requested() -> bool:
    return os.environ.get("WIDUSHI_CAMERA_STUB", "").strip().lower() in {
        "1", "true", "yes", "on",
    }


class CameraSource:
    """Captures JPEG snapshots from the device camera with a safe fallback."""

    def __init__(
        self,
        *,
        width: int = 640,
        height: int = 480,
        jpeg_quality: int = 85,
        stub: bool | None = None,
    ) -> None:
        self.width = width
        self.height = height
        self.jpeg_quality = jpeg_quality
        self._stub = bool(_stub_requested()) if stub is None else bool(stub)
        self._picam = None  # lazy picamera2.Picamera2
        self._cv2_cap = None  # lazy cv2.VideoCapture
        self._lock = asyncio.Lock()

    async def snapshot(self) -> bytes:
        """Capture one frame and return it as JPEG bytes.

        Never raises: every backend failure falls through to the
        placeholder JPEG with a log warning so the FSM can keep walking.
        """

        if self._stub:
            await asyncio.sleep(0)
            return _PLACEHOLDER_JPEG

        async with self._lock:
            jpeg = await asyncio.to_thread(self._capture_sync)
        if jpeg is None:
            return _PLACEHOLDER_JPEG
        return jpeg

    def _capture_sync(self) -> bytes | None:
        # Try picamera2 first (native Pi 5 backend).
        try:
            jpeg = self._capture_picamera2()
            if jpeg is not None:
                return jpeg
        except Exception:
            log.exception("picamera2 capture failed; falling back to cv2")

        try:
            jpeg = self._capture_cv2()
            if jpeg is not None:
                return jpeg
        except Exception:
            log.exception("cv2 capture failed; falling back to placeholder")

        log.warning("no camera backend available; returning placeholder JPEG")
        return None

    def _capture_picamera2(self) -> bytes | None:
        try:
            from picamera2 import Picamera2  # type: ignore[import-not-found]
        except Exception:
            return None
        try:
            from PIL import Image  # type: ignore[import-not-found]
        except Exception:
            log.debug("picamera2 present but PIL missing; cannot encode JPEG")
            return None

        if self._picam is None:
            picam = Picamera2()
            cfg = picam.create_still_configuration(
                main={"size": (self.width, self.height), "format": "RGB888"}
            )
            picam.configure(cfg)
            picam.start()
            self._picam = picam

        import io

        array = self._picam.capture_array()
        img = Image.fromarray(array)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=self.jpeg_quality, optimize=True)
        return buf.getvalue()

    def _capture_cv2(self) -> bytes | None:
        try:
            import cv2  # type: ignore[import-not-found]
        except Exception:
            return None

        if self._cv2_cap is None:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                cap.release()
                return None
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self._cv2_cap = cap

        ok, frame = self._cv2_cap.read()
        if not ok or frame is None:
            return None
        ok, buf = cv2.imencode(
            ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(self.jpeg_quality)]
        )
        if not ok:
            return None
        return bytes(buf.tobytes())

    def close(self) -> None:
        """Release the underlying camera handle if any."""

        if self._picam is not None:
            try:
                self._picam.stop()
                self._picam.close()
            except Exception:
                log.exception("picamera2 close failed")
            self._picam = None
        if self._cv2_cap is not None:
            try:
                self._cv2_cap.release()
            except Exception:
                log.exception("cv2 release failed")
            self._cv2_cap = None
