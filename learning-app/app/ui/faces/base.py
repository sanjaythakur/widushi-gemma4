"""Shared geometry, palette, and base class for face renderers.

Every face is a small class with ``update(dt)`` (advance internal
timers) and ``draw(surface)`` (render to a pygame surface). Faces own
no global state and never call ``pygame.display.flip()`` themselves —
that's the job of :mod:`app.ui.loop`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

import pygame

from app import config


@dataclass(frozen=True)
class Palette:
    """Colors used across faces. Accent overrides per-state."""

    background: tuple[int, int, int] = (18, 20, 26)
    face: tuple[int, int, int] = (236, 224, 200)
    eye: tuple[int, int, int] = (28, 30, 38)
    mouth: tuple[int, int, int] = (28, 30, 38)
    text: tuple[int, int, int] = (200, 210, 224)
    accent_idle: tuple[int, int, int] = (170, 180, 200)
    accent_listening: tuple[int, int, int] = (66, 196, 188)  # teal
    accent_thinking: tuple[int, int, int] = (240, 180, 70)   # amber
    accent_speaking: tuple[int, int, int] = (240, 110, 110)  # coral


PALETTE = Palette()

CENTER: tuple[int, int] = (config.SCREEN_WIDTH // 2, config.SCREEN_HEIGHT // 2 - 18)
FACE_RADIUS: int = 96
EYE_OFFSET_X: int = 32
EYE_OFFSET_Y: int = -10
EYE_RADIUS: int = 12


class Face(Protocol):
    accent: tuple[int, int, int]

    def update(self, dt: float) -> None:
        ...

    def draw(self, surface: pygame.Surface) -> None:
        ...


class FaceBase:
    """Convenience base offering common drawing helpers."""

    accent: tuple[int, int, int] = PALETTE.accent_idle

    def __init__(self) -> None:
        self._t: float = 0.0
        self._font: pygame.font.Font | None = None

    def update(self, dt: float) -> None:
        self._t += dt

    def draw(self, surface: pygame.Surface) -> None:  # pragma: no cover - overridden
        raise NotImplementedError

    def _font_or_init(self, size: int = 16) -> pygame.font.Font:
        if self._font is None or self._font.get_height() != size:
            if not pygame.font.get_init():
                pygame.font.init()
            self._font = pygame.font.SysFont("Arial", size)
        return self._font

    def _draw_background(self, surface: pygame.Surface) -> None:
        surface.fill(PALETTE.background)
        # Thin accent ring underneath the face for a touch of state color.
        pygame.draw.circle(
            surface,
            self.accent,
            CENTER,
            FACE_RADIUS + 8,
            width=2,
        )

    def _draw_face_circle(
        self,
        surface: pygame.Surface,
        *,
        scale: float = 1.0,
        rotation_deg: float = 0.0,
    ) -> tuple[int, int]:
        """Draw the face oval and return its current center."""

        # Rotation is applied to eye/mouth positions by the caller; we
        # only render the base circle here. The rotation parameter is
        # kept for symmetry / future use.
        del rotation_deg
        radius = max(1, int(FACE_RADIUS * scale))
        pygame.draw.circle(surface, PALETTE.face, CENTER, radius)
        return CENTER

    def _eye_centers(self, *, rotation_deg: float = 0.0) -> tuple[
        tuple[int, int], tuple[int, int]
    ]:
        cx, cy = CENTER
        ey = cy + EYE_OFFSET_Y
        left = (cx - EYE_OFFSET_X, ey)
        right = (cx + EYE_OFFSET_X, ey)
        if rotation_deg == 0:
            return left, right
        rad = math.radians(rotation_deg)
        cos_r, sin_r = math.cos(rad), math.sin(rad)

        def rot(p: tuple[int, int]) -> tuple[int, int]:
            dx, dy = p[0] - cx, p[1] - cy
            return (
                int(cx + dx * cos_r - dy * sin_r),
                int(cy + dx * sin_r + dy * cos_r),
            )

        return rot(left), rot(right)
