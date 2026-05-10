"""Input sources that produce events for the orchestrator.

Each source is an ``InputSource`` that runs as its own asyncio task
and pushes ``Event`` objects into the shared queue. Adding a new input
(wake word ONNX, REST, GPIO button, …) means writing one more class
that implements the same protocol.
"""
