"""Display info.

The real rendering happens in ``app/ui/loop.py`` against a pygame
``Surface``; this module only carries metadata so non-UI code can ask
about the screen without importing pygame.
"""

from __future__ import annotations

from dataclasses import dataclass

from app import config


@dataclass(frozen=True)
class DisplayInfo:
    width: int = config.SCREEN_WIDTH
    height: int = config.SCREEN_HEIGHT
    fps: int = config.FPS
