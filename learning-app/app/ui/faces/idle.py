"""IDLE face: half-closed eyes, slow breathing, periodic blink, clock."""

from __future__ import annotations

import math
import random
import time

import pygame

from app import config
from app.ui.faces.base import (
    CENTER,
    EYE_RADIUS,
    PALETTE,
    FaceBase,
)

BLINK_DURATION_S = 0.15
BREATH_PERIOD_S = 4.0
BREATH_AMPLITUDE_PX = 3


class IdleFace(FaceBase):
    accent = PALETTE.accent_idle

    def __init__(self) -> None:
        super().__init__()
        self._next_blink_at = self._t + random.uniform(3.0, 5.0)
        self._blink_started_at: float | None = None

    def update(self, dt: float) -> None:
        super().update(dt)
        if self._blink_started_at is None and self._t >= self._next_blink_at:
            self._blink_started_at = self._t
        if (
            self._blink_started_at is not None
            and self._t - self._blink_started_at >= BLINK_DURATION_S
        ):
            self._blink_started_at = None
            self._next_blink_at = self._t + random.uniform(3.0, 5.0)

    def draw(self, surface: pygame.Surface) -> None:
        self._draw_background(surface)

        # Breathing scale: ±BREATH_AMPLITUDE_PX baked into a scalar.
        breath = math.sin(self._t * (2 * math.pi / BREATH_PERIOD_S))
        scale = 1.0 + (BREATH_AMPLITUDE_PX / 100.0) * breath

        self._draw_face_circle(surface, scale=scale)

        # Eye height: half-closed by default (0.55), drops to ~0.1 mid-blink.
        openness = 0.55
        if self._blink_started_at is not None:
            phase = (self._t - self._blink_started_at) / BLINK_DURATION_S
            # 0->1 closes then opens via a triangle wave 1->0->1.
            tri = 1.0 - abs(phase * 2.0 - 1.0)
            openness = 0.55 - 0.45 * tri

        left, right = self._eye_centers()
        for eye in (left, right):
            ex, ey = eye
            rect = pygame.Rect(
                ex - EYE_RADIUS,
                int(ey - EYE_RADIUS * openness),
                EYE_RADIUS * 2,
                max(2, int(EYE_RADIUS * 2 * openness)),
            )
            pygame.draw.ellipse(surface, PALETTE.eye, rect)

        # Soft neutral mouth — a thin horizontal line.
        cx, cy = CENTER
        mouth_y = cy + 36
        pygame.draw.line(
            surface,
            PALETTE.mouth,
            (cx - 22, mouth_y),
            (cx + 22, mouth_y),
            width=3,
        )

        # Lower-third clock.
        clock_str = time.strftime("%H:%M")
        font = self._font_or_init(20)
        text_surface = font.render(clock_str, True, PALETTE.text)
        rect = text_surface.get_rect(
            center=(config.SCREEN_WIDTH // 2, config.SCREEN_HEIGHT - 24)
        )
        surface.blit(text_surface, rect)
