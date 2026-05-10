"""Widushi learning app — main process package.

This package implements the architecture described in
``phases/2-system-scaffolding.md``: a single Python process that hosts
a pygame UI, an asyncio FSM orchestrator, and a FastAPI server inside
one event loop, with hardware/services as pluggable async clients.
"""
