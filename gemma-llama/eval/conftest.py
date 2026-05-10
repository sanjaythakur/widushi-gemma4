"""Shared pytest fixtures for the evaluation suite."""
from __future__ import annotations

import os

import pytest
import pytest_asyncio

from .client import APIClient


@pytest_asyncio.fixture
async def client():
    async with APIClient() as c:
        try:
            health = await c.health()
        except Exception as exc:
            pytest.skip(f"API not reachable at {c.base_url}: {exc}")
        if not health.get("llama_server_reachable") and not os.environ.get("EVAL_ALLOW_DEGRADED"):
            pytest.skip("llama server not reachable; skipping live evaluation")
        yield c
