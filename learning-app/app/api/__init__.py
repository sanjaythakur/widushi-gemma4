"""FastAPI control plane for the Widushi app.

Hosted by the same uvicorn instance running inside the main asyncio
loop. Endpoints are intentionally minimal in Phase 2: enough to
inspect the FSM and inject test events without using the keyboard.
"""

from __future__ import annotations

from app.api.server import build_api

__all__ = ["build_api"]
