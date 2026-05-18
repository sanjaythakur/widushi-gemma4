"""Gemma LLM client."""

from __future__ import annotations

import asyncio
import io
import logging
import wave
from dataclasses import dataclass
from urllib.parse import urljoin

import httpx

from app import config

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class GemmaReply:
    """Text plus optional Piper audio returned by the Gemma service."""

    text: str
    audio_bytes: bytes | None = None
    audio_duration_ms: float | None = None
    voice: str | None = None


@dataclass(frozen=True)
class FreeConvoReply(GemmaReply):
    """FreeConvoMode reply: chat text + ``start_learning`` intent flag."""

    start_learning: bool = False
    transcript: str | None = None


@dataclass(frozen=True)
class WordSuggestion(GemmaReply):
    """VoiceMirrorMode word suggestion (the spoken cue is in ``text``)."""

    word: str = ""
    example_sentence: str = ""
    ipa_hint: str | None = None


@dataclass(frozen=True)
class ScoreResult(GemmaReply):
    """VoiceMirrorMode score: spoken feedback + verdict for the orchestrator."""

    target_word: str = ""
    transcript: str | None = None
    verdict: str = "retry"


@dataclass(frozen=True)
class TeachReply(GemmaReply):
    """VisionMode teaching reply (object + spoken sentence + transcript)."""

    object: str | None = None
    transcript: str | None = None


class GemmaClient:
    """Minimal async client surface that the orchestrator can rely on."""

    def __init__(
        self,
        *,
        url: str = config.GEMMA_URL,
        timeout_s: float = config.GEMMA_TIMEOUT_S,
        latency_s: float = 0.6,
        stub: bool = False,
        client: httpx.AsyncClient | None = None,
        tts_enabled: bool = config.GEMMA_TTS_ENABLED,
        tts_voice: str = config.GEMMA_TTS_VOICE,
    ) -> None:
        self._url = url.rstrip("/")
        self._timeout_s = timeout_s
        self._latency_s = latency_s
        self._stub = stub
        self._client = client
        self._owns_client = client is None
        self._tts_enabled = tts_enabled
        self._tts_voice = tts_voice

    async def complete(self, prompt: str) -> str:
        """Return a completion for ``prompt``.

        Used as a fallback for manual/API text events. Spoken turns should
        prefer :meth:`listen_audio`.
        """

        log.debug("gemma.complete prompt=%r", prompt[:60])
        if self._stub:
            await asyncio.sleep(self._latency_s)
            snippet = prompt.strip().splitlines()[0][:40] if prompt.strip() else ""
            return f"(stub reply) heard: {snippet!r}"

        response = await self._http.post(
            "/generate",
            json={"prompt": prompt, "stream": False},
        )
        response.raise_for_status()
        payload = response.json()
        return str(payload.get("text", ""))

    async def complete_with_tts(self, prompt: str) -> GemmaReply:
        """Return a text completion with Gemma-hosted Piper audio attached."""

        log.debug("gemma.complete_with_tts prompt=%r", prompt[:60])
        if self._stub:
            await asyncio.sleep(self._latency_s)
            snippet = prompt.strip().splitlines()[0][:40] if prompt.strip() else ""
            return GemmaReply(
                text=f"(stub reply) heard: {snippet!r}",
                audio_bytes=_silence_wav(),
                audio_duration_ms=500.0,
                voice=self._tts_voice,
            )

        response = await self._http.post(
            "/generate",
            json={
                "prompt": prompt,
                "stream": False,
                "tts": self._tts_enabled,
                "voice": self._tts_voice,
            },
        )
        response.raise_for_status()
        return await self._reply_from_payload(response.json())

    async def listen_audio(
        self,
        audio_bytes: bytes,
        *,
        filename: str = "question.wav",
        content_type: str = "audio/wav",
    ) -> str:
        """Send a spoken question to Gemma's native audio endpoint."""

        log.debug("gemma.listen_audio bytes=%d filename=%s", len(audio_bytes), filename)
        if self._stub:
            await asyncio.sleep(self._latency_s)
            return f"(stub audio reply) received {len(audio_bytes)} bytes"

        response = await self._http.post(
            "/audio/listen",
            data={"stream": "false"},
            files={"audio": (filename, audio_bytes, content_type)},
        )
        response.raise_for_status()
        payload = response.json()
        return str(payload.get("text", ""))

    async def listen_audio_with_tts(
        self,
        audio_bytes: bytes,
        *,
        filename: str = "question.wav",
        content_type: str = "audio/wav",
    ) -> GemmaReply:
        """Send a spoken question and fetch Gemma-hosted Piper WAV output."""

        log.debug(
            "gemma.listen_audio_with_tts bytes=%d filename=%s",
            len(audio_bytes),
            filename,
        )
        if self._stub:
            await asyncio.sleep(self._latency_s)
            return GemmaReply(
                text=f"(stub audio reply) received {len(audio_bytes)} bytes",
                audio_bytes=_silence_wav(),
                audio_duration_ms=500.0,
                voice=self._tts_voice,
            )

        response = await self._http.post(
            "/audio/listen",
            data={
                "stream": "false",
                "tts": "true" if self._tts_enabled else "false",
                "voice": self._tts_voice,
            },
            files={"audio": (filename, audio_bytes, content_type)},
        )
        response.raise_for_status()
        return await self._reply_from_payload(response.json())

    # ------------------------------------------------------------------
    # Mode-specific endpoints
    # ------------------------------------------------------------------

    async def free_convo_turn(
        self,
        audio_bytes: bytes,
        *,
        filename: str = "turn.wav",
        content_type: str = "audio/wav",
    ) -> FreeConvoReply:
        """Single FreeConvoMode turn -> ``POST /free-convo/turn``."""

        log.debug("gemma.free_convo_turn bytes=%d", len(audio_bytes))
        if self._stub:
            await asyncio.sleep(self._latency_s)
            return FreeConvoReply(
                text="(stub) I'm here. Want to practise English?",
                audio_bytes=_silence_wav(),
                audio_duration_ms=500.0,
                voice=self._tts_voice,
                start_learning=False,
                transcript=None,
            )

        response = await self._http.post(
            "/free-convo/turn",
            data={
                "tts": "true" if self._tts_enabled else "false",
                "voice": self._tts_voice,
            },
            files={"audio": (filename, audio_bytes, content_type)},
        )
        response.raise_for_status()
        payload = response.json()
        base = await self._audio_from_payload(payload)
        return FreeConvoReply(
            text=base.text,
            audio_bytes=base.audio_bytes,
            audio_duration_ms=base.audio_duration_ms,
            voice=base.voice,
            start_learning=bool(payload.get("start_learning", False)),
            transcript=(
                str(payload["transcript"]) if payload.get("transcript") else None
            ),
        )

    async def voice_mirror_suggest(
        self,
        *,
        history: list[str] | None = None,
        level: str | None = None,
    ) -> WordSuggestion:
        """Ask Gemma for the next word to practise -> ``POST /voice-mirror/suggest``."""

        log.debug("gemma.voice_mirror_suggest history=%s level=%s", history, level)
        if self._stub:
            await asyncio.sleep(self._latency_s)
            return WordSuggestion(
                text="Try saying: apple. AP-uhl.",
                audio_bytes=_silence_wav(),
                audio_duration_ms=500.0,
                voice=self._tts_voice,
                word="apple",
                example_sentence="I like apples.",
                ipa_hint="AP-uhl",
            )

        body: dict[str, object] = {
            "tts": self._tts_enabled,
            "voice": self._tts_voice,
        }
        if history:
            body["history"] = list(history)
        if level:
            body["level"] = level

        response = await self._http.post("/voice-mirror/suggest", json=body)
        response.raise_for_status()
        payload = response.json()
        base = await self._audio_from_payload(payload, text_field="prompt_text")
        return WordSuggestion(
            text=base.text,
            audio_bytes=base.audio_bytes,
            audio_duration_ms=base.audio_duration_ms,
            voice=base.voice,
            word=str(payload.get("word", "")),
            example_sentence=str(payload.get("example_sentence", "")),
            ipa_hint=(
                str(payload["ipa_hint"]) if payload.get("ipa_hint") else None
            ),
        )

    async def voice_mirror_score(
        self,
        audio_bytes: bytes,
        *,
        target_word: str,
        filename: str = "attempt.wav",
        content_type: str = "audio/wav",
    ) -> ScoreResult:
        """Score a pronunciation attempt -> ``POST /voice-mirror/score``."""

        log.debug(
            "gemma.voice_mirror_score word=%r bytes=%d", target_word, len(audio_bytes)
        )
        if self._stub:
            await asyncio.sleep(self._latency_s)
            return ScoreResult(
                text=f"Nice try with '{target_word}'.",
                audio_bytes=_silence_wav(),
                audio_duration_ms=500.0,
                voice=self._tts_voice,
                target_word=target_word,
                transcript=target_word,
                verdict="praise",
            )

        response = await self._http.post(
            "/voice-mirror/score",
            data={
                "target_word": target_word,
                "tts": "true" if self._tts_enabled else "false",
                "voice": self._tts_voice,
            },
            files={"audio": (filename, audio_bytes, content_type)},
        )
        response.raise_for_status()
        payload = response.json()
        base = await self._audio_from_payload(payload, text_field="feedback_text")
        return ScoreResult(
            text=base.text,
            audio_bytes=base.audio_bytes,
            audio_duration_ms=base.audio_duration_ms,
            voice=base.voice,
            target_word=str(payload.get("target_word", target_word)),
            transcript=(
                str(payload["transcript"]) if payload.get("transcript") else None
            ),
            verdict=str(payload.get("verdict", "retry")).lower(),
        )

    async def vision_teach_object(
        self,
        image_bytes: bytes,
        audio_bytes: bytes,
        *,
        image_filename: str = "frame.jpg",
        image_content_type: str = "image/jpeg",
        audio_filename: str = "guess.wav",
        audio_content_type: str = "audio/wav",
    ) -> TeachReply:
        """Teach an English noun from a camera frame + spoken guess
        -> ``POST /vision/teach-object``.
        """

        log.debug(
            "gemma.vision_teach_object image=%d audio=%d",
            len(image_bytes),
            len(audio_bytes),
        )
        if self._stub:
            await asyncio.sleep(self._latency_s)
            return TeachReply(
                text="Yes, this is milk. Say: I drink milk.",
                audio_bytes=_silence_wav(),
                audio_duration_ms=500.0,
                voice=self._tts_voice,
                object="milk",
                transcript=None,
            )

        response = await self._http.post(
            "/vision/teach-object",
            data={
                "tts": "true" if self._tts_enabled else "false",
                "voice": self._tts_voice,
            },
            files={
                "image": (image_filename, image_bytes, image_content_type),
                "audio": (audio_filename, audio_bytes, audio_content_type),
            },
        )
        response.raise_for_status()
        payload = response.json()
        base = await self._audio_from_payload(payload)
        return TeachReply(
            text=base.text,
            audio_bytes=base.audio_bytes,
            audio_duration_ms=base.audio_duration_ms,
            voice=base.voice,
            object=(str(payload["object"]) if payload.get("object") else None),
            transcript=(
                str(payload["transcript"]) if payload.get("transcript") else None
            ),
        )

    async def _reply_from_payload(self, payload: dict[str, object]) -> GemmaReply:
        return await self._audio_from_payload(payload)

    async def _audio_from_payload(
        self, payload: dict[str, object], *, text_field: str = "text"
    ) -> GemmaReply:
        """Build a :class:`GemmaReply` from a TTS-attached JSON payload.

        ``text_field`` lets callers point at endpoint-specific text keys
        (``"prompt_text"`` for ``/voice-mirror/suggest``,
        ``"feedback_text"`` for ``/voice-mirror/score``).
        """

        text = str(payload.get(text_field, "") or payload.get("text", ""))
        audio_url = payload.get("audio_url")
        audio_error = payload.get("audio_error")
        voice = payload.get("voice")
        duration = payload.get("audio_duration_ms")

        if not self._tts_enabled:
            return GemmaReply(text=text)

        if audio_error:
            raise RuntimeError(f"Gemma TTS failed: {audio_error}")
        if not isinstance(audio_url, str) or not audio_url:
            raise RuntimeError("Gemma TTS response did not include audio_url")

        audio_response = await self._http.get(urljoin(f"{self._url}/", audio_url))
        audio_response.raise_for_status()
        return GemmaReply(
            text=text,
            audio_bytes=audio_response.content,
            audio_duration_ms=float(duration) if isinstance(duration, (int, float)) else None,
            voice=str(voice) if voice is not None else None,
        )

    @property
    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self._url,
                timeout=httpx.Timeout(self._timeout_s),
            )
        return self._client

    async def aclose(self) -> None:
        """Close the underlying HTTP session when owned by this client."""

        if self._client is not None and self._owns_client:
            await self._client.aclose()


def _silence_wav(len_s: float = 0.5) -> bytes:
    """Build a small mono int16 WAV for stubbed Gemma responses."""

    n_frames = max(1, int(len_s * config.AUDIO_SAMPLE_RATE))
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(config.AUDIO_CHANNELS)
        wav.setsampwidth(2)
        wav.setframerate(config.AUDIO_SAMPLE_RATE)
        wav.writeframes(b"\x00\x00" * n_frames)
    return buf.getvalue()
