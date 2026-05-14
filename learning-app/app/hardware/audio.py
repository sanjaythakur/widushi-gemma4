"""Shared audio-output wiring for pygame's process-global mixer.

``ClipPlayer`` and ``PiperClient`` both push WAVs through pygame's
mixer, which is a singleton owned by SDL. They share this helper so
the device selection (and any related diagnostics) live in exactly
one place.
"""

from __future__ import annotations

import logging
import os

import pygame

from app import config

log = logging.getLogger(__name__)


def init_pygame_mixer() -> None:
    """Initialise pygame's mixer, honouring ``WIDUSHI_OUTPUT_DEVICE``.

    Falls back to a default ``mixer.init()`` if pygame is too old to
    accept ``devicename``, or if SDL rejects the name (some ALSA
    builds disagree with pygame's enumeration). The fallback path
    still works because the container's ``SDL_AUDIODEV`` env var is
    consulted by SDL directly.
    """
    device = config.AUDIO_OUTPUT_DEVICE
    if device:
        try:
            pygame.mixer.init(devicename=device)
            log.info("pygame mixer using device=%r", device)
            return
        except (TypeError, pygame.error) as exc:
            log.warning(
                "pygame mixer rejected devicename=%r (%s); using default",
                device,
                exc,
            )
    pygame.mixer.init()
    log.info(
        "pygame mixer using ALSA default (SDL_AUDIODEV=%r)",
        os.environ.get("SDL_AUDIODEV"),
    )
