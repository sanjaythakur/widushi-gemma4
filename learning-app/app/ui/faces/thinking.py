"""THINKING face: raised brows, eyes upper-left, orbital dots."""

from __future__ import annotations

import math

import pygame

from app import config
from app.ui.faces.base import (
    CENTER,
    EYE_RADIUS,
    FACE_RADIUS,
    PALETTE,
    FaceBase,
)

ORBIT_RADIUS = 22
ORBIT_PERIOD_S = 1.8
DOT_COUNT = 3


class ThinkingFace(FaceBase):
    accent = PALETTE.accent_thinking

    def draw(self, surface: pygame.Surface) -> None:
        self._draw_background(surface)
        self._draw_face_circle(surface)

        # Eyes glance upper-left: shift both eyes 4px up and left.
        left, right = self._eye_centers()
        gaze = (-4, -4)
        for eye in (left, right):
            ex, ey = eye[0] + gaze[0], eye[1] + gaze[1]
            pygame.draw.circle(surface, PALETTE.eye, (ex, ey), EYE_RADIUS)

        # Raised eyebrows: short arcs above each eye.
        for eye in (left, right):
            ex, ey = eye
            brow_rect = pygame.Rect(ex - 14, ey - 22, 28, 14)
            pygame.draw.arc(
                surface,
                PALETTE.eye,
                brow_rect,
                math.radians(20),
                math.radians(160),
                3,
            )

        # Subtle pursed mouth.
        cx, cy = CENTER
        pygame.draw.line(
            surface,
            PALETTE.mouth,
            (cx - 14, cy + 36),
            (cx + 14, cy + 36),
            width=3,
        )

        # Orbital dots in the corner — one full orbit per ORBIT_PERIOD_S.
        corner = (config.SCREEN_WIDTH - 40, 40)
        base_phase = (self._t / ORBIT_PERIOD_S) * (2 * math.pi)
        for i in range(DOT_COUNT):
            phase = base_phase + i * (2 * math.pi / DOT_COUNT)
            x = int(corner[0] + math.cos(phase) * ORBIT_RADIUS)
            y = int(corner[1] + math.sin(phase) * ORBIT_RADIUS)
            pygame.draw.circle(surface, self.accent, (x, y), 4)

        # Faint accent ring around the head emphasises the state.
        pygame.draw.circle(
            surface, self.accent, CENTER, FACE_RADIUS + 14, width=1
        )
