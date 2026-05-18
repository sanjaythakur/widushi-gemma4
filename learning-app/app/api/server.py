"""FastAPI app and uvicorn factory.

Routes:
    GET  /health        -> liveness probe
    GET  /state         -> current orchestrator (per-turn) state
    GET  /system_state  -> current outer state (IDLE / SESSION / ERROR)
    GET  /mode          -> active mode + list of registered modes
    POST /events        -> enqueue an Event by ``EventType.name``
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app import config
from app.events import Event, EventType
from app.orchestrator import Orchestrator

log = logging.getLogger(__name__)


class EventIn(BaseModel):
    type: str = Field(..., description="EventType name, e.g. 'WAKE_DETECTED'")
    payload: dict[str, Any] = Field(default_factory=dict)


def build_api(
    orchestrator: Orchestrator,
    *,
    request_utterance_stop: Callable[[], bool] | None = None,
) -> FastAPI:
    api = FastAPI(title="widushi", version="0.2.0")

    @api.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @api.get("/state")
    async def get_state() -> dict[str, str]:
        return {"state": orchestrator.state.name}

    @api.get("/system_state")
    async def get_system_state() -> dict[str, str]:
        return {"system_state": orchestrator.system_state.name}

    @api.get("/mode")
    async def get_mode() -> dict[str, Any]:
        active = orchestrator.active_mode
        return {
            "active": active.name if active is not None else None,
            "registered": orchestrator.mode_registry.names(),
        }

    @api.post("/events", status_code=202)
    async def post_event(event_in: EventIn) -> dict[str, str]:
        try:
            event_type = EventType[event_in.type]
        except KeyError as exc:
            valid = ", ".join(t.name for t in EventType)
            raise HTTPException(
                status_code=400,
                detail=f"unknown event type {event_in.type!r}; valid: {valid}",
            ) from exc
        if event_type is EventType.UTTERANCE_END and request_utterance_stop:
            if request_utterance_stop():
                log.info("api -> UTTERANCE_END (manual recording stop)")
                return {"queued": event_type.name}
        await orchestrator.queue.put(Event(event_type, payload=event_in.payload))
        log.info("api -> %s", event_type.name)
        return {"queued": event_type.name}

    return api


def build_uvicorn_server(
    orchestrator: Orchestrator,
    *,
    request_utterance_stop: Callable[[], bool] | None = None,
) -> Any:
    """Construct a uvicorn Server bound to our existing asyncio loop.

    Imported lazily so non-runtime callers (tests, lint) don't need
    uvicorn installed unless they actually start the server.
    """

    import uvicorn  # noqa: PLC0415 — lazy by design

    cfg = uvicorn.Config(
        build_api(orchestrator, request_utterance_stop=request_utterance_stop),
        host=config.API_HOST,
        port=config.API_PORT,
        loop="asyncio",
        lifespan="on",
        log_level="info",
    )
    return uvicorn.Server(cfg)


__all__ = ["build_api", "build_uvicorn_server"]
