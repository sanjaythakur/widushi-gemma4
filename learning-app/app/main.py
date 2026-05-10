"""Process entrypoint for the Widushi app.

Initialises pygame on the main thread, wires the orchestrator, FastAPI
server, UI loop, and input sources into one asyncio ``TaskGroup``,
then ``asyncio.run()``s the lot. A ``SHUTDOWN`` event (sent on window
close, double-ESC, or SIGINT) drains the FSM and exits cleanly.
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal

import pygame

from app import config
from app.api.server import build_uvicorn_server
from app.events import Event, EventType
from app.input.keyboard import KeyboardSource
from app.input.utterance import FollowUpListenSignal, UtteranceStopSignal
from app.input.wakeword import WakeWordSource
from app.orchestrator import Orchestrator
from app.services import Services
from app.ui.loop import UILoop


def _configure_logging() -> None:
    level = os.environ.get("WIDUSHI_LOG", "INFO").upper()
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def _init_pygame() -> pygame.Surface:
    # Allow callers (Docker on Mac, CI) to force a headless backend
    # without code changes — SDL_VIDEODRIVER=dummy / SDL_AUDIODRIVER=dummy.
    pygame.init()
    pygame.display.set_caption(config.WINDOW_TITLE)
    return pygame.display.set_mode(config.SCREEN_SIZE)


async def amain() -> None:
    _configure_logging()
    log = logging.getLogger("app.main")

    screen = _init_pygame()
    queue: asyncio.Queue[Event] = asyncio.Queue()
    key_events: asyncio.Queue[int] = asyncio.Queue(maxsize=64)
    utterance_stop = UtteranceStopSignal()
    followup_listen = FollowUpListenSignal()

    services = Services.live()
    orchestrator = Orchestrator(queue, services, followup_listen=followup_listen)
    ui = UILoop(screen, orchestrator, key_events)
    keyboard = KeyboardSource(
        queue,
        key_events,
        request_utterance_stop=utterance_stop.request_stop,
    )
    wakeword = WakeWordSource(
        queue,
        stop_signal=utterance_stop,
        followup_signal=followup_listen,
        clips=services.clips,
    )
    server = build_uvicorn_server(
        orchestrator,
        request_utterance_stop=utterance_stop.request_stop,
    )

    # Translate SIGINT/SIGTERM to a graceful SHUTDOWN event.
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(
                sig,
                lambda: queue.put_nowait(Event(EventType.SHUTDOWN)),
            )
        except (NotImplementedError, RuntimeError):
            # Windows / restricted envs: fall back to default handler.
            pass

    log.info(
        "widushi up: api=http://%s:%d  display=%dx%d  fps=%d",
        config.API_HOST,
        config.API_PORT,
        config.SCREEN_WIDTH,
        config.SCREEN_HEIGHT,
        config.FPS,
    )

    # Plain create_task + gather (rather than TaskGroup) because we
    # want a graceful shutdown: when the FSM returns we ask uvicorn /
    # the UI to stop on their own, then cancel only the parked input
    # sources. TaskGroup's "cancel everything immediately" semantics
    # produce noisy lifespan tracebacks from starlette.
    fsm_task = asyncio.create_task(orchestrator.run(), name="orchestrator")
    keyboard_task = asyncio.create_task(keyboard.run(), name="keyboard")
    wakeword_task = asyncio.create_task(wakeword.run(), name="wakeword")
    server_task = asyncio.create_task(server.serve(), name="uvicorn")
    ui_task = asyncio.create_task(ui.run(), name="ui")

    try:
        await fsm_task
    finally:
        ui.stop()
        server.should_exit = True
        for t in (keyboard_task, wakeword_task):
            t.cancel()
        await asyncio.gather(
            ui_task,
            server_task,
            keyboard_task,
            wakeword_task,
            return_exceptions=True,
        )
        await services.gemma.aclose()
        await services.piper.aclose()
        await services.clips.aclose()
        await services.db.close()
        pygame.quit()
        log.info("widushi exited")


def main() -> None:
    asyncio.run(amain())


if __name__ == "__main__":
    main()
