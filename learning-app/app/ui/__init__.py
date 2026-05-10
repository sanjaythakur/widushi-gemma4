"""Pygame UI package.

The renderer runs as an asyncio coroutine on the main thread (SDL
requirement). It selects which :class:`Face` to draw based on the
orchestrator's current state and forwards keyboard events to the
``KeyboardSource``.
"""
