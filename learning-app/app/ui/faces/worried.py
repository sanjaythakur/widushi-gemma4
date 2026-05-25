"""WORRIED face: drooped brows, small mouth, soft amber accent.

Shown by :class:`app.modes.voice_mirror.CalmAndRetrySub` when the
learner stalls or scores low and we want a gentle "take a breath, try
again" cue rather than the standard listening face.
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


class WorriedFace(FaceBase):
    accent = PALETTE.accent_worried

    def draw(self, surface: pygame.Surface) -> None:
        self._draw_background(surface)
        self._draw_face_circle(surface)

        # Eyes slightly downturned, looking at the user but soft.
        cx, cy = CENTER
        eye_y = cy - 6
        for sign in (-1, 1):
            ex = cx + sign * EYE_OFFSET_X
            pygame.draw.circle(
                surface, PALETTE.eye, (ex, eye_y), EYE_RADIUS - 1
            )

        # Drooped eyebrows: short tilted lines above each eye, inner
        # ends higher than outer ends to convey concern.
        brow_offset_y = 14
        for sign in (-1, 1):
            ex = cx + sign * EYE_OFFSET_X
            inner = (ex - sign * 10, eye_y - brow_offset_y - 4)
            outer = (ex + sign * 10, eye_y - brow_offset_y + 2)
            pygame.draw.line(
                surface, PALETTE.eye, inner, outer, width=3
            )

        # Small pursed mouth: a short arc that dips slightly downward.
        mouth_w = 28
        mouth_h = 12
        mouth_rect = pygame.Rect(
            cx - mouth_w // 2,
            cy + 30,
            mouth_w,
            mouth_h,
        )
        pygame.draw.arc(
            surface,
            PALETTE.mouth,
            mouth_rect,
            math.radians(20),
            math.radians(160),
            width=3,
        )

        # Subtle accent dot under the mouth -- a held breath.
        pygame.draw.circle(
            surface, self.accent, (cx, cy + 56), 3
        )
