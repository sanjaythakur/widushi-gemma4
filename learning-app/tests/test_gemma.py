from __future__ import annotations

import httpx
import pytest

from app.services.gemma import GemmaClient


@pytest.mark.asyncio
async def test_listen_audio_posts_multipart_audio() -> None:
    seen: dict[str, object] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        body = await request.aread()
        seen["method"] = request.method
        seen["path"] = request.url.path
        seen["content_type"] = request.headers["content-type"]
        seen["body"] = body
        return httpx.Response(200, json={"text": "Gemma heard the question."})

    http = httpx.AsyncClient(
        base_url="http://gemma.test",
        transport=httpx.MockTransport(handler),
    )
    client = GemmaClient(url="http://gemma.test", client=http)

    try:
        reply = await client.listen_audio(
            b"RIFF fake wav",
            filename="question.wav",
            content_type="audio/wav",
        )
    finally:
        await client.aclose()
        await http.aclose()

    assert reply == "Gemma heard the question."
    assert seen["method"] == "POST"
    assert seen["path"] == "/audio/listen"
    assert str(seen["content_type"]).startswith("multipart/form-data")
    body = seen["body"]
    assert isinstance(body, bytes)
    assert b'name="audio"; filename="question.wav"' in body
    assert b"RIFF fake wav" in body
    assert b'name="stream"' in body


@pytest.mark.asyncio
async def test_listen_audio_with_tts_fetches_gemma_audio_url() -> None:
    seen: dict[str, object] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/audio/listen":
            body = await request.aread()
            seen["method"] = request.method
            seen["path"] = request.url.path
            seen["content_type"] = request.headers["content-type"]
            seen["body"] = body
            return httpx.Response(
                200,
                json={
                    "text": "Gemma heard the question.",
                    "audio_url": "/tts/output/reply.wav",
                    "audio_duration_ms": 1234.0,
                    "voice": "friendly-casual",
                },
            )
        if request.url.path == "/tts/output/reply.wav":
            seen["audio_fetch_path"] = request.url.path
            return httpx.Response(200, content=b"RIFF gemma wav")
        return httpx.Response(404)

    http = httpx.AsyncClient(
        base_url="http://gemma.test",
        transport=httpx.MockTransport(handler),
    )
    client = GemmaClient(
        url="http://gemma.test",
        client=http,
        tts_voice="friendly-casual",
    )

    try:
        reply = await client.listen_audio_with_tts(
            b"RIFF fake wav",
            filename="question.wav",
            content_type="audio/wav",
        )
    finally:
        await client.aclose()
        await http.aclose()

    assert reply.text == "Gemma heard the question."
    assert reply.audio_bytes == b"RIFF gemma wav"
    assert reply.audio_duration_ms == 1234.0
    assert reply.voice == "friendly-casual"
    assert seen["method"] == "POST"
    assert seen["path"] == "/audio/listen"
    assert seen["audio_fetch_path"] == "/tts/output/reply.wav"
    assert str(seen["content_type"]).startswith("multipart/form-data")
    body = seen["body"]
    assert isinstance(body, bytes)
    assert b'name="audio"; filename="question.wav"' in body
    assert b"RIFF fake wav" in body
    assert b'name="stream"' in body
    assert b"\r\nfalse\r\n" in body
    assert b'name="tts"' in body
    assert b"\r\ntrue\r\n" in body
    assert b'name="voice"' in body
    assert b"\r\nfriendly-casual\r\n" in body


@pytest.mark.asyncio
async def test_complete_with_tts_posts_generate_and_fetches_audio_url() -> None:
    seen: dict[str, object] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/generate":
            seen["json"] = await request.aread()
            return httpx.Response(
                200,
                json={
                    "text": "Hello from Gemma.",
                    "audio_url": "/tts/output/generate.wav",
                    "voice": "warm-academic",
                },
            )
        if request.url.path == "/tts/output/generate.wav":
            return httpx.Response(200, content=b"RIFF generated wav")
        return httpx.Response(404)

    http = httpx.AsyncClient(
        base_url="http://gemma.test",
        transport=httpx.MockTransport(handler),
    )
    client = GemmaClient(url="http://gemma.test", client=http)

    try:
        reply = await client.complete_with_tts("hello")
    finally:
        await client.aclose()
        await http.aclose()

    assert reply.text == "Hello from Gemma."
    assert reply.audio_bytes == b"RIFF generated wav"
    assert b'"tts":true' in seen["json"]
    assert b'"voice":"warm-academic"' in seen["json"]


@pytest.mark.asyncio
async def test_listen_audio_with_tts_raises_on_audio_error() -> None:
    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"text": "Gemma text only.", "audio_error": "TTS engine not ready"},
        )

    http = httpx.AsyncClient(
        base_url="http://gemma.test",
        transport=httpx.MockTransport(handler),
    )
    client = GemmaClient(url="http://gemma.test", client=http)

    try:
        with pytest.raises(RuntimeError, match="Gemma TTS failed"):
            await client.listen_audio_with_tts(b"RIFF fake wav")
    finally:
        await client.aclose()
        await http.aclose()
