"""LISTENING face: wide eyes, slight head tilt, animated sound bars."""

from __future__ import annotations

import math

import pygame

from app import config
from app.ui.faces.base import (
    CENTER,
    EYE_RADIUS,
    PALETTE,
    FaceBase,
)

HEAD_TILT_DEG = 8.0
NUM_BARS = 9
BAR_WIDTH = 6
BAR_GAP = 6
BAR_MAX_HEIGHT = 36


class ListeningFace(FaceBase):
    accent = PALETTE.accent_listening

    def draw(self, surface: pygame.Surface) -> None:
        self._draw_background(surface)
        self._draw_face_circle(surface)

        # Wide circular eyes, with positions rotated to fake the head tilt.
        left, right = self._eye_centers(rotation_deg=HEAD_TILT_DEG)
        eye_radius = int(EYE_RADIUS * 1.25)
        for eye in (left, right):
            pygame.draw.circle(surface, PALETTE.eye, eye, eye_radius)
            # Tiny highlight to keep them looking alive.
            pygame.draw.circle(
                surface,
                PALETTE.face,
                (eye[0] - 3, eye[1] - 3),
                max(1, eye_radius // 4),
            )

        # Sound-wave bars below the face — Phase 2 fakes amplitude with
        # offset sines so the bars feel responsive.
        cx, cy = CENTER
        bars_y = cy + 88
        total_w = NUM_BARS * BAR_WIDTH + (NUM_BARS - 1) * BAR_GAP
        start_x = cx - total_w // 2
        for i in range(NUM_BARS):
            phase = self._t * 6.0 + i * 0.7
            amp = (math.sin(phase) * 0.5 + 0.5) * 0.8 + 0.2
            h = int(BAR_MAX_HEIGHT * amp)
            rect = pygame.Rect(
                start_x + i * (BAR_WIDTH + BAR_GAP),
                bars_y - h // 2,
                BAR_WIDTH,
                h,
            )
            pygame.draw.rect(surface, self.accent, rect, border_radius=3)

        # Status caption.
        font = self._font_or_init(16)
        cap = font.render("listening", True, self.accent)
        surface.blit(
            cap,
            cap.get_rect(center=(config.SCREEN_WIDTH // 2, config.SCREEN_HEIGHT - 18)),
        )
