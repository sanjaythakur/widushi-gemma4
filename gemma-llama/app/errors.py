"""Domain exceptions for the gemma-llama service."""
from __future__ import annotations


class LlamaServerError(RuntimeError):
    """Raised when the llama.cpp server returns an error or is unreachable."""

    def __init__(self, message: str, *, status_code: int | None = None, body: str | None = None):
        super().__init__(message)
        self.status_code = status_code
        self.body = body
