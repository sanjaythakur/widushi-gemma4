"""Keyboard-driven event source for FSM testing.

Lets a developer walk every transition by hand without needing a mic
or wake word. The pygame event pump runs in :mod:`app.ui.loop`, which
forwards ``KEYDOWN`` events into a per-source ``asyncio.Queue`` that
this source drains.

Bindings:
    SPACE  -> WAKE_DETECTED
    ENTER  -> UTTERANCE_END
    R      -> REPLY_READY
    D      -> PLAYBACK_DONE
    ESC    -> CANCEL  (second press triggers SHUTDOWN)
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable

import pygame

from app.events import Event, EventType
from app.input.sources import InputSource

log = logging.getLogger(__name__)


KEY_BINDINGS: dict[int, EventType] = {
    pygame.K_SPACE: EventType.WAKE_DETECTED,
    pygame.K_RETURN: EventType.UTTERANCE_END,
    pygame.K_KP_ENTER: EventType.UTTERANCE_END,
    pygame.K_r: EventType.REPLY_READY,
    pygame.K_d: EventType.PLAYBACK_DONE,
}


class KeyboardSource(InputSource):
    """Drains pygame KEYDOWN events from a queue and re-emits as Events."""

    def __init__(
        self,
        queue: asyncio.Queue[Event],
        key_events: asyncio.Queue[int],
        *,
        request_utterance_stop: Callable[[], bool] | None = None,
    ) -> None:
        super().__init__(queue)
        self._key_events = key_events
        self._request_utterance_stop = request_utterance_stop
        self._esc_seen = False

    async def run(self) -> None:
        while True:
            key = await self._key_events.get()
            event = self._translate(key)
            if event is None:
                continue
            log.info("keyboard -> %s", event.type.name)
            await self.emit(event)

    def _translate(self, key: int) -> Event | None:
        if key == pygame.K_ESCAPE:
            if self._esc_seen:
                return Event(EventType.SHUTDOWN)
            self._esc_seen = True
            return Event(EventType.CANCEL)
        if key in {pygame.K_RETURN, pygame.K_KP_ENTER} and self._request_utterance_stop:
            if self._request_utterance_stop():
                return None
        # Any other recognised key resets the double-ESC latch.
        self._esc_seen = False
        mapped = KEY_BINDINGS.get(key)
        if mapped is None:
            return None
        return Event(mapped)
