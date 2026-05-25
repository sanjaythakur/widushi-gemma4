"""Pygame UI loop running as an asyncio coroutine.

SDL requires that pygame run on the main thread, so the orchestrator,
uvicorn, and this loop all share the single asyncio event loop. Each
frame we:

1. drain pygame's event queue (so SDL stays responsive),
2. forward keyboard events to the keyboard input source,
3. tick + draw the face for the orchestrator's current state,
4. ``await asyncio.sleep(1/FPS)`` to yield back to other tasks.

Step 4 is what allows the orchestrator and FastAPI to make progress —
without that yield the FSM would never get the queue.
"""

from __future__ import annotations

import asyncio
import logging
import time

import pygame

from app import config
from app.events import Event, EventType
from app.hardware.fb_sink import FbSink
from app.orchestrator import Orchestrator
from app.ui.faces import APPSTATE_TO_FACE_ID, build_face_registry

log = logging.getLogger(__name__)


class UILoop:
    def __init__(
        self,
        screen: pygame.Surface,
        orchestrator: Orchestrator,
        key_events: asyncio.Queue[int],
        fb_sink: FbSink | None = None,
    ) -> None:
        self._screen = screen
        self._orchestrator = orchestrator
        self._key_events = key_events
        self._fb_sink = fb_sink
        self._faces = build_face_registry()
        self._last_tick: float | None = None
        self._stop = False

    def stop(self) -> None:
        self._stop = True

    async def run(self) -> None:
        log.info("ui loop started @ %d fps", config.FPS)
        try:
            while not self._stop:
                self._pump_events()
                self._tick_and_draw()
                await asyncio.sleep(config.FRAME_INTERVAL)
        finally:
            log.info("ui loop exiting")

    def _pump_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                # Window close = graceful shutdown.
                self._orchestrator.queue.put_nowait(Event(EventType.SHUTDOWN))
                self._stop = True
                return
            if event.type == pygame.KEYDOWN:
                # Hand the raw key code to the keyboard source; it
                # translates to FSM events.
                try:
                    self._key_events.put_nowait(event.key)
                except asyncio.QueueFull:
                    log.warning("key event queue full; dropping key=%s", event.key)

    def _tick_and_draw(self) -> None:
        now = time.monotonic()
        dt = 0.0 if self._last_tick is None else now - self._last_tick
        self._last_tick = now

        face = self._faces[self._pick_face_id()]
        face.update(dt)
        face.draw(self._screen)
        pygame.display.flip()

        # When SDL is rendering to a real window (cocoa/x11) ``flip``
        # is what shows the frame. With ``SDL_VIDEODRIVER=dummy`` on the
        # Pi the flip is a no-op, so we additionally push the surface
        # into ``/dev/fb0`` ourselves — that's how the SPI TFT lights up.
        if self._fb_sink is not None:
            self._fb_sink.push(self._screen)

    def _pick_face_id(self) -> str:
        """Active mode's substate face wins, else fall back to AppState."""

        mode = self._orchestrator.active_mode
        if mode is not None:
            substate = mode.current_substate
            if substate is not None and substate.face_id:
                face_id = substate.face_id
                if face_id in self._faces:
                    return face_id
                log.debug(
                    "unknown substate face_id=%r; falling back to AppState face",
                    face_id,
                )
        return APPSTATE_TO_FACE_ID[self._orchestrator.state]
