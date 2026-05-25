"""HAPPY face: crescent eyes, wide smile, mint accent.

Shown by :class:`app.modes.voice_mirror.PraiseSub` (and the transient
``NextPhrase`` / ``DrillComplete`` substates) when the learner nailed
the target pronunciation.
"""

from __future__ import annotations

import math

import pygame

from app.ui.faces.base import (
    CENTER,
    EYE_OFFSET_X,
    EYE_RADIUS,
    PALETTE,
    FaceBase,
)


class HappyFace(FaceBase):
    accent = PALETTE.accent_happy

    def draw(self, surface: pygame.Surface) -> None:
        self._draw_background(surface)
        self._draw_face_circle(surface)

        # Crescent eyes: arcs facing downward suggest a smile up top.
        cx, cy = CENTER
        eye_y = cy - 10
        eye_half_w = EYE_RADIUS + 4
        for sign in (-1, 1):
            ex = cx + sign * EYE_OFFSET_X
            rect = pygame.Rect(
                ex - eye_half_w,
                eye_y - EYE_RADIUS,
                eye_half_w * 2,
                EYE_RADIUS * 2,
            )
            pygame.draw.arc(
                surface,
                PALETTE.eye,
                rect,
                math.radians(15),
                math.radians(165),
                width=3,
            )

        # Wide upturned mouth.
        mouth_w = 60
        mouth_h = 26
        mouth_rect = pygame.Rect(
            cx - mouth_w // 2,
            cy + 22,
            mouth_w,
            mouth_h,
        )
        pygame.draw.arc(
            surface,
            PALETTE.mouth,
            mouth_rect,
            math.radians(200),
            math.radians(340),
            width=4,
        )

        # Cheek blush — two small accent circles, slightly transparent.
        for sign in (-1, 1):
            blush_x = cx + sign * (EYE_OFFSET_X + 14)
            pygame.draw.circle(
                surface,
                self.accent,
                (blush_x, cy + 18),
                7,
            )
