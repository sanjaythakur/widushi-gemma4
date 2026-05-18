"""Async client for the llama.cpp OpenAI-compatible server.

Responsibilities:
    * Health checks (`GET /health`)
    * Non-streaming chat completions (`POST /v1/chat/completions`)
    * Streaming chat completions, normalised to NDJSON lines of the form
      ``{"type": "thinking" | "text", "content": "..."}``
    * Retries with exponential backoff on transient failures
    * Suppressing chain-of-thought when the caller passes ``thinking=False``:
      we both ask the server to disable it (``chat_template_kwargs.enable_thinking``
      plus the Qwen3-style ``/no_think`` directive) AND filter any ``<think>``
      / ``reasoning_content`` that the model emits anyway, so callers with
      ``thinking=False`` never see a ``[think]`` event.

Prompt-prefix caching:
    Every chat completion sets ``cache_prompt: true`` so llama.cpp's slot KV
    cache reuses any byte-identical prefix from the previous request on the
    same slot. This is what makes "keep system prompts byte-identical across
    calls" actually pay off: a tutor system prompt is encoded once per slot
    and skipped on every subsequent call until the prefix diverges. The flag
    is the default in modern llama.cpp builds; we set it explicitly so the
    behaviour is pinned across version bumps and obvious to anyone reading
    the payload.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from collections.abc import AsyncIterator
from copy import deepcopy
from typing import Any

import httpx

from .errors import LlamaServerError

logger = logging.getLogger(__name__)


def _cache_prompt_default() -> bool:
    """Resolve the default ``cache_prompt`` value once per process.

    Set ``LLAMA_CACHE_PROMPT=false`` to disable prefix caching globally (only
    useful for benchmarking the cold-start path).
    """
    raw = os.environ.get("LLAMA_CACHE_PROMPT", "true").strip().lower()
    return raw not in ("0", "false", "no", "off")


_CACHE_PROMPT_DEFAULT = _cache_prompt_default()


# ---------------------------------------------------------------------------
# Helper functions for picking apart chat-completion responses
# ---------------------------------------------------------------------------


def extract_content(response: dict[str, Any]) -> str:
    """Return the assistant message content for a completion response.

    Any inline ``<think>...</think>`` spans are stripped: reasoning is exposed
    separately via :func:`extract_reasoning`, and the visible ``text`` field
    should never contain raw think tags regardless of the ``thinking`` flag.
    """
    try:
        content = response["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError):
        return ""
    return _strip_think_blocks(content)


def _strip_think_blocks(text: str) -> str:
    """Remove every ``<think>...</think>`` span (and any unterminated tail)."""
    if "<think>" not in text:
        return text
    out: list[str] = []
    i = 0
    while i < len(text):
        start = text.find("<think>", i)
        if start == -1:
            out.append(text[i:])
            break
        out.append(text[i:start])
        end = text.find("</think>", start)
        if end == -1:
            # Unterminated think block - drop the rest.
            break
        i = end + len("</think>")
    return "".join(out).strip()


def extract_reasoning(response: dict[str, Any]) -> str:
    """Return the model's chain-of-thought, if any.

    llama.cpp surfaces reasoning under one of several keys depending on build:
    ``reasoning_content``, ``reasoning``, or inline ``<think>...</think>``
    blocks within ``content``. We probe each in order.
    """
    try:
        msg = response["choices"][0]["message"]
    except (KeyError, IndexError, TypeError):
        return ""

    for key in ("reasoning_content", "reasoning"):
        value = msg.get(key)
        if value:
            return value

    content = msg.get("content") or ""
    if "<think>" in content and "</think>" in content:
        start = content.index("<think>") + len("<think>")
        end = content.index("</think>")
        return content[start:end].strip()
    return ""


def extract_usage(response: dict[str, Any]) -> dict[str, int]:
    """Return ``usage`` stats with safe defaults when absent."""
    usage = response.get("usage") or {}
    return {
        "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
        "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
        "total_tokens": int(usage.get("total_tokens", 0) or 0),
        "tokens_predicted": int(
            usage.get("completion_tokens", usage.get("tokens_predicted", 0)) or 0
        ),
    }


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


def _apply_no_think(messages: list[dict[str, Any]], thinking: bool) -> list[dict[str, Any]]:
    """Append ``/no_think`` to the last user message when reasoning is off."""
    if thinking:
        return messages
    msgs = deepcopy(messages)
    for msg in reversed(msgs):
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, str):
            msg["content"] = content.rstrip() + "\n\n/no_think"
        elif isinstance(content, list):
            content.append({"type": "text", "text": "\n\n/no_think"})
        break
    return msgs


class LlamaAdapter:
    """Stateless wrapper around the llama.cpp OpenAI-compatible API."""

    def __init__(self, base_url: str, timeout: float = 300.0, retries: int = 3):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.retries = max(0, retries)
        self._client: httpx.AsyncClient | None = None

    async def start(self) -> None:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(self.timeout, connect=30.0),
            )

    async def stop(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            raise RuntimeError("LlamaAdapter.start() was not called")
        return self._client

    # --- health -----------------------------------------------------------

    async def health_check(self) -> bool:
        try:
            resp = await self.client.get("/health", timeout=5.0)
            return resp.status_code == 200
        except httpx.HTTPError as exc:
            logger.debug("llama health check failed: %s", exc)
            return False

    # --- non-streaming ----------------------------------------------------

    async def chat_completion(
        self,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int,
        temperature: float,
        response_format: dict[str, Any] | None = None,
        thinking: bool = True,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """POST ``/v1/chat/completions`` and return the parsed JSON response.

        ``_inference_time_ms`` is injected for observability.
        """
        payload: dict[str, Any] = {
            "messages": _apply_no_think(messages, thinking),
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
            # Opt-in to llama.cpp's slot-level prefix-cache reuse. See module
            # docstring; default is True, override with LLAMA_CACHE_PROMPT=false.
            "cache_prompt": _CACHE_PROMPT_DEFAULT,
        }
        if not thinking:
            # llama.cpp forwards chat_template_kwargs into the Jinja chat
            # template; templates that honour `enable_thinking` (Qwen3,
            # newer Gemma builds, etc.) will skip emitting <think> blocks
            # entirely. Harmless on templates that ignore the flag.
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        if response_format is not None:
            payload["response_format"] = response_format
        if extra:
            payload.update(extra)

        start = time.perf_counter()
        response = await self._post_with_retry("/v1/chat/completions", payload)
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        try:
            data = response.json()
        except json.JSONDecodeError as exc:
            raise LlamaServerError(
                f"llama.cpp returned non-JSON body: {exc}",
                status_code=response.status_code,
                body=response.text[:500],
            ) from exc

        data["_inference_time_ms"] = round(elapsed_ms, 2)
        return data

    # --- streaming --------------------------------------------------------

    async def chat_completion_stream(
        self,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int,
        temperature: float,
        thinking: bool = True,
        extra: dict[str, Any] | None = None,
    ) -> AsyncIterator[str]:
        """Yield NDJSON lines, one per token chunk, with type ``thinking`` or ``text``."""
        payload: dict[str, Any] = {
            "messages": _apply_no_think(messages, thinking),
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
            # Same prefix-cache opt-in as the non-streaming path; identical
            # system prompts across calls hit the slot KV cache and skip
            # re-encoding the tutor instructions.
            "cache_prompt": _CACHE_PROMPT_DEFAULT,
        }
        if not thinking:
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        if extra:
            payload.update(extra)

        in_think_block = False  # Tracks <think>...</think> spans inside content

        # Heavy multimodal prompts (12 video frames at 75 s/frame on Pi 5) can
        # take many minutes to start emitting tokens. Use a generous read
        # timeout for the streaming pipe so it does not collapse mid-encode;
        # connect/write/pool stay short so genuine network failures still
        # surface quickly.
        stream_timeout = httpx.Timeout(
            connect=30.0, read=None, write=60.0, pool=10.0,
        )
        async with self.client.stream(
            "POST", "/v1/chat/completions", json=payload, timeout=stream_timeout,
        ) as response:
            if response.status_code >= 400:
                body = (await response.aread()).decode("utf-8", errors="replace")
                raise LlamaServerError(
                    f"llama.cpp /v1/chat/completions returned {response.status_code}",
                    status_code=response.status_code,
                    body=body[:500],
                )

            async for raw_line in response.aiter_lines():
                if not raw_line:
                    continue
                if raw_line.startswith("data: "):
                    raw_line = raw_line[len("data: "):]
                if raw_line.strip() == "[DONE]":
                    break
                try:
                    chunk = json.loads(raw_line)
                except json.JSONDecodeError:
                    continue

                try:
                    delta = chunk["choices"][0]["delta"]
                except (KeyError, IndexError, TypeError):
                    continue

                # Some llama.cpp builds emit reasoning_content separately.
                reasoning = delta.get("reasoning_content") or delta.get("reasoning")
                if reasoning and thinking:
                    yield json.dumps({"type": "thinking", "content": reasoning}) + "\n"

                content = delta.get("content")
                if not content:
                    continue

                # Inline <think>...</think> support. We always consume the
                # tags (so they never leak into the visible text stream); we
                # only forward the inner reasoning when `thinking=True`.
                remaining = content
                while remaining:
                    if in_think_block:
                        end_idx = remaining.find("</think>")
                        if end_idx == -1:
                            if thinking:
                                yield json.dumps({"type": "thinking", "content": remaining}) + "\n"
                            remaining = ""
                        else:
                            piece = remaining[:end_idx]
                            if piece and thinking:
                                yield json.dumps({"type": "thinking", "content": piece}) + "\n"
                            remaining = remaining[end_idx + len("</think>"):]
                            in_think_block = False
                    else:
                        start_idx = remaining.find("<think>")
                        if start_idx == -1:
                            yield json.dumps({"type": "text", "content": remaining}) + "\n"
                            remaining = ""
                        else:
                            piece = remaining[:start_idx]
                            if piece:
                                yield json.dumps({"type": "text", "content": piece}) + "\n"
                            remaining = remaining[start_idx + len("<think>"):]
                            in_think_block = True

    # --- internals --------------------------------------------------------

    async def _post_with_retry(self, path: str, payload: dict[str, Any]) -> httpx.Response:
        attempt = 0
        last_exc: Exception | None = None
        while attempt <= self.retries:
            try:
                response = await self.client.post(path, json=payload)
                if response.status_code >= 500:
                    raise LlamaServerError(
                        f"llama.cpp {path} returned {response.status_code}",
                        status_code=response.status_code,
                        body=response.text[:500],
                    )
                if response.status_code >= 400:
                    # Client errors are not retried.
                    raise LlamaServerError(
                        f"llama.cpp {path} returned {response.status_code}",
                        status_code=response.status_code,
                        body=response.text[:500],
                    )
                return response
            except (httpx.TransportError, httpx.TimeoutException, LlamaServerError) as exc:
                last_exc = exc
                if isinstance(exc, LlamaServerError) and exc.status_code and exc.status_code < 500:
                    raise
                # Don't retry on timeouts. A timeout means llama.cpp is still
                # mid-inference (heavy multimodal prompts on Pi 5 take many
                # minutes); retrying would CANCEL the in-flight task and
                # restart the full prompt encode from zero, multiplying the
                # wall clock by 3x and confusing the user. Surface the
                # timeout immediately and let the caller bump
                # LLM_TIMEOUT_SECONDS instead.
                if isinstance(exc, httpx.TimeoutException):
                    logger.warning(
                        "llama %s timed out after %.0fs; not retrying (set "
                        "LLM_TIMEOUT_SECONDS higher for multimodal workloads).",
                        path, self.timeout,
                    )
                    raise LlamaServerError(
                        f"llama.cpp {path} timed out after {self.timeout:.0f}s. "
                        "Multimodal prompts on Pi 5 can take 5-15 min; bump "
                        "LLM_TIMEOUT_SECONDS or reduce n_frames."
                    ) from exc
                attempt += 1
                if attempt > self.retries:
                    break
                backoff = min(2 ** (attempt - 1), 10)
                logger.warning(
                    "llama %s attempt %d/%d failed: %s -- retrying in %ss",
                    path, attempt, self.retries, exc, backoff,
                )
                await asyncio.sleep(backoff)
        assert last_exc is not None
        if isinstance(last_exc, LlamaServerError):
            raise last_exc
        raise LlamaServerError(f"llama.cpp {path} unreachable: {last_exc}") from last_exc
