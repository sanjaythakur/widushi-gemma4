"""SPEAKING face: bright forward eyes, mouth animating on a sine wave."""

from __future__ import annotations

import math

import pygame

from app.ui.faces.base import (
    CENTER,
    EYE_RADIUS,
    PALETTE,
    FaceBase,
)

MOUTH_FREQ_HZ = 6.0
MOUTH_MAX_HEIGHT = 22
MOUTH_WIDTH = 44


class SpeakingFace(FaceBase):
    accent = PALETTE.accent_speaking

    def draw(self, surface: pygame.Surface) -> None:
        self._draw_background(surface)
        self._draw_face_circle(surface)

        # Bright forward-facing eyes (slightly bigger + accent ring).
        left, right = self._eye_centers()
        for eye in (left, right):
            pygame.draw.circle(surface, PALETTE.eye, eye, EYE_RADIUS + 1)
            pygame.draw.circle(surface, self.accent, eye, EYE_RADIUS + 4, width=1)

        # Mouth: ellipse whose vertical opening tracks |sin(2π f t)|.
        cx, cy = CENTER
        opening = abs(math.sin(self._t * 2 * math.pi * MOUTH_FREQ_HZ))
        height = max(2, int(MOUTH_MAX_HEIGHT * opening))
        rect = pygame.Rect(
            cx - MOUTH_WIDTH // 2,
            cy + 30 - height // 2,
            MOUTH_WIDTH,
            height,
        )
        pygame.draw.ellipse(surface, PALETTE.mouth, rect)
