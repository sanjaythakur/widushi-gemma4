"""Glue the TTS engine + storage onto FastAPI's response shapes.

Two public helpers, each consumed by exactly one shape of router code:

* :func:`attach_file_tts` -- non-streaming mode endpoints (every one of the
  five Widushi routes that returns JSON). Writes a WAV to the cache and
  returns a dict containing the public ``audio_url``.
* :func:`wrap_stream_with_tts` -- the lone streaming endpoint
  (``/audio/listen`` with ``stream=true``). Wraps the upstream NDJSON
  iterator and interleaves ``{"type":"audio", ...}`` lines per completed
  sentence.
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
from collections.abc import AsyncIterator

from .engine import PiperEngine, PiperError
from .sentences import SentenceBuffer
from .storage import TTSStorage, concat_wavs, wav_duration_ms
from .voices import resolve_personality

logger = logging.getLogger(__name__)


class TTSIntegrationError(RuntimeError):
    """Raised when TTS attach fails for a reason callers should surface."""


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


async def _synth_all(
    text: str,
    *,
    engine: PiperEngine,
    voice: str,
    max_chars: int = 400,
) -> tuple[list[bytes], list[str]]:
    """Split ``text`` into sentences and synthesise each via Piper.

    Sentences are rendered serially so the Pi 5's CPU is not pegged by
    parallel piper invocations while a follow-up llama.cpp request might
    arrive. Returns ``(wav_blobs, sentences)`` so callers can attach
    metadata if they want.
    """
    buf = SentenceBuffer(max_chars=max_chars)
    sentences = buf.add(text)
    tail = buf.flush()
    if tail:
        sentences.append(tail)

    blobs: list[bytes] = []
    for sent in sentences:
        if not sent.strip():
            continue
        blob = await engine.synth(sent, voice)
        if blob:
            blobs.append(blob)
    return blobs, sentences


# ---------------------------------------------------------------------------
# file (URL) attach -- every non-streaming mode endpoint
# ---------------------------------------------------------------------------


async def attach_file_tts(
    text: str,
    *,
    engine: PiperEngine,
    storage: TTSStorage,
    voice: str | None,
    url_prefix: str = "/tts/output",
) -> dict[str, object]:
    """Render ``text``, persist the WAV, return the file-URL attachment.

    Resulting keys:

    ``audio_url``: relative URL playable via ``GET /tts/output/<id>.wav``.
    ``audio_duration_ms``: best-effort duration in milliseconds.
    ``voice``: resolved personality id.
    ``audio_error``: only set when synth failed; explains why.
    """
    key, _ = resolve_personality(voice)
    base: dict[str, object] = {"audio_url": None, "voice": key}
    try:
        blobs, _ = await _synth_all(text, engine=engine, voice=key)
        wav = concat_wavs(blobs)
        if not wav:
            return {**base, "audio_error": "TTS produced no audio"}
        audio_id, _ = storage.store(wav)
        return {
            **base,
            "audio_url": f"{url_prefix}/{audio_id}.wav",
            "audio_duration_ms": wav_duration_ms(wav),
        }
    except PiperError as exc:
        return {**base, "audio_error": str(exc)}


# ---------------------------------------------------------------------------
# streaming attach -- intersperse {"type":"audio"} lines into upstream NDJSON
# ---------------------------------------------------------------------------


_END_SENTINEL = object()


def wrap_stream_with_tts(
    upstream: AsyncIterator[str],
    *,
    engine: PiperEngine,
    voice: str | None,
    max_chars: int = 400,
) -> AsyncIterator[str]:
    """Forward upstream NDJSON lines, with sentence-level audio interleaved.

    Architecture (deliberately decoupled so token throughput is never blocked
    by audio synth latency):

    .. code::

        upstream --> [producer] --> out_queue --> client
                          |
                          v
                     sentence_queue --> [synthesizer] --> out_queue

    The producer pushes raw text/thinking/etc lines straight to ``out_queue``
    and additionally feeds completed sentences into ``sentence_queue``. The
    synthesizer consumes sentences, runs Piper, and pushes
    ``{"type":"audio", ...}`` lines onto ``out_queue``. The synthesizer is
    the sole writer of the terminal sentinel onto ``out_queue`` so the
    consumer side stays trivially correct.
    """
    key, _ = resolve_personality(voice)

    async def gen() -> AsyncIterator[str]:
        out_queue: asyncio.Queue = asyncio.Queue()
        sentence_queue: asyncio.Queue = asyncio.Queue(maxsize=8)
        producer_done = asyncio.Event()

        async def producer() -> None:
            buf = SentenceBuffer(max_chars=max_chars)
            try:
                async for line in upstream:
                    await out_queue.put(line)
                    text = _extract_text_delta(line)
                    if not text:
                        continue
                    for sent in buf.add(text):
                        await sentence_queue.put(sent)
                tail = buf.flush()
                if tail:
                    await sentence_queue.put(tail)
            except Exception as exc:  # noqa: BLE001
                await out_queue.put(
                    json.dumps({
                        "type": "error",
                        "content": f"upstream error: {type(exc).__name__}: {exc}",
                    }) + "\n"
                )
            finally:
                producer_done.set()
                # Sentinel for synthesizer.
                await sentence_queue.put(_END_SENTINEL)

        async def synthesizer() -> None:
            seq = 0
            sample_rate = engine.sample_rate_for(key)
            try:
                while True:
                    item = await sentence_queue.get()
                    if item is _END_SENTINEL:
                        break
                    sent: str = item
                    seq += 1
                    try:
                        wav = await engine.synth(sent, key)
                    except PiperError as exc:
                        await out_queue.put(
                            json.dumps({
                                "type": "audio_error",
                                "seq": seq,
                                "sentence": sent,
                                "content": str(exc),
                                "voice": key,
                            }) + "\n"
                        )
                        continue
                    if not wav:
                        continue
                    await out_queue.put(
                        json.dumps({
                            "type": "audio",
                            "seq": seq,
                            "sentence": sent,
                            "content": base64.b64encode(wav).decode("ascii"),
                            "mime": "audio/wav",
                            "sample_rate": sample_rate,
                            "voice": key,
                        }) + "\n"
                    )
            finally:
                # The synthesizer is the sole writer of the consumer-side
                # sentinel -- it has visibility into both the producer's
                # completion (via the sentence-queue sentinel) and any
                # in-flight synthesis.
                await out_queue.put(None)

        prod_task = asyncio.create_task(producer(), name="tts-stream-producer")
        synth_task = asyncio.create_task(synthesizer(), name="tts-stream-synth")
        try:
            while True:
                item = await out_queue.get()
                if item is None:
                    break
                yield item
        finally:
            for t in (prod_task, synth_task):
                if not t.done():
                    t.cancel()
            # Drain to prevent "Task was destroyed but it is pending"
            await asyncio.gather(prod_task, synth_task, return_exceptions=True)

    return gen()


def _extract_text_delta(line: str) -> str:
    """Return the text content of an upstream NDJSON line, ``""`` otherwise.

    We only feed ``type=text`` content into the sentence buffer; thinking
    output is forwarded but not spoken (it is the model's chain-of-thought
    and rarely useful as audio).
    """
    line = line.strip()
    if not line:
        return ""
    try:
        obj = json.loads(line)
    except json.JSONDecodeError:
        return ""
    if obj.get("type") != "text":
        return ""
    content = obj.get("content")
    return content if isinstance(content, str) else ""
