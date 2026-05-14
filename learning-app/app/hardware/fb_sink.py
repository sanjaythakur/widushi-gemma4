"""Direct-to-framebuffer sink for SPI TFT panels.

SDL2 has no ``fbcon``/``fbdev`` video driver, and the Pi 5 + ``fbtft``
SPI panel setup also exposes no ``/dev/dri`` device (so ``KMSDRM`` is
unavailable). The standard workaround is to render the pygame UI onto
an off-screen ``Surface`` (``SDL_VIDEODRIVER=dummy``) and then copy
each frame straight into ``/dev/fb0`` ourselves.

This module owns that copy. It mmaps the framebuffer once, and every
frame converts the RGB888 pygame surface to packed RGB565 — the
format the 16bpp ``fb_ili9486`` / MHS35 panel actually consumes —
before writing.

Set ``WIDUSHI_FB_DEVICE=/dev/fb0`` (or another ``fbN``) to enable.
With the variable unset, :meth:`FbSink.create_from_env` returns
``None`` and the UI loop falls back to plain ``pygame.display.flip()``
— which is what we want on Mac (cocoa window) and headless dev runs
(``SDL_VIDEODRIVER=dummy`` with no panel attached).
"""

from __future__ import annotations

import logging
import mmap
import os
from pathlib import Path

import numpy as np
import pygame

log = logging.getLogger(__name__)


class FbSink:
    """mmap-backed sink that writes pygame frames to a Linux framebuffer."""

    def __init__(self, device: str, width: int, height: int, bpp: int = 16) -> None:
        if bpp != 16:
            raise RuntimeError(
                f"fb {device} reports {bpp}bpp; only 16bpp RGB565 panels are supported"
            )
        self._device = device
        self._width = width
        self._height = height
        self._size = width * height * 2  # RGB565 = 2 bytes/pixel, tight stride
        self._fd = os.open(device, os.O_RDWR)
        try:
            self._mm = mmap.mmap(self._fd, self._size)
        except Exception:
            os.close(self._fd)
            raise

    @classmethod
    def create_from_env(cls) -> FbSink | None:
        """Build an ``FbSink`` from ``WIDUSHI_FB_DEVICE``, or return ``None``.

        Returning ``None`` is the no-op signal to the UI loop — Mac dev
        and headless tests follow this path and just rely on
        ``pygame.display.flip()``.
        """
        device = os.environ.get("WIDUSHI_FB_DEVICE")
        if not device:
            return None
        try:
            width, height, bpp = _read_fb_geometry(device)
        except FileNotFoundError:
            log.warning(
                "WIDUSHI_FB_DEVICE=%s but matching /sys/class/graphics entry "
                "is missing; falling back to no fb sink",
                device,
            )
            return None
        log.info("fb sink ready: %s %dx%d @ %dbpp", device, width, height, bpp)
        return cls(device, width, height, bpp)

    @property
    def size(self) -> tuple[int, int]:
        return (self._width, self._height)

    def push(self, surface: pygame.Surface) -> None:
        """Convert ``surface`` to RGB565 and copy it into the framebuffer.

        If the surface dimensions don't match the panel, we scale; that
        only happens on misconfiguration (e.g. the ``mhs35`` overlay
        without ``rotate=90``), in which case a slightly fuzzy picture
        beats a hard crash.
        """
        if surface.get_size() != (self._width, self._height):
            surface = pygame.transform.smoothscale(surface, (self._width, self._height))

        # array3d returns a copy (W, H, 3) uint8 RGB. The copy is cheap
        # (~460 KB at 480x320) and avoids the surface-locking footgun
        # that pixels3d carries.
        rgb = pygame.surfarray.array3d(surface)
        # pygame uses (x, y) indexing; framebuffers are row-major (y, x).
        rgb = np.transpose(rgb, (1, 0, 2))

        r = rgb[..., 0].astype(np.uint16) >> 3
        g = rgb[..., 1].astype(np.uint16) >> 2
        b = rgb[..., 2].astype(np.uint16) >> 3
        rgb565 = (r << 11) | (g << 5) | b

        self._mm.seek(0)
        self._mm.write(rgb565.tobytes())

    def close(self) -> None:
        try:
            self._mm.close()
        finally:
            os.close(self._fd)


def _read_fb_geometry(device: str) -> tuple[int, int, int]:
    """Look up width/height/bpp via ``/sys/class/graphics/<fbN>/``.

    We deliberately avoid ``FBIOGET_VSCREENINFO`` ioctls; sysfs is
    enough for fbtft panels which use tight stride and have no padding.
    """
    name = Path(device).name
    sysfs = Path("/sys/class/graphics") / name
    width, height = (
        int(x) for x in (sysfs / "virtual_size").read_text().strip().split(",")
    )
    bpp = int((sysfs / "bits_per_pixel").read_text().strip())
    return width, height, bpp
