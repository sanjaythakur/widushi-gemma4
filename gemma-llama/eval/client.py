"""Thin async HTTP client for hitting the running gemma-llama API."""
from __future__ import annotations

import mimetypes
import os
from pathlib import Path
from typing import Any

import httpx

DEFAULT_BASE_URL = os.environ.get(
    "GEMMA_LLAMA_BASE_URL", f"http://localhost:{os.environ.get('API_PORT', '8010')}"
)


class APIClient:
    def __init__(self, base_url: str = DEFAULT_BASE_URL, timeout: float = 300.0):
        self.base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=timeout)

    async def __aenter__(self) -> "APIClient":
        return self

    async def __aexit__(self, *_exc) -> None:
        await self.close()

    async def close(self) -> None:
        await self._client.aclose()

    async def health(self) -> dict[str, Any]:
        r = await self._client.get("/health")
        return r.json()

    async def classify(self, text: str, labels: list[str], multi_label: bool = False) -> dict[str, Any]:
        r = await self._client.post(
            "/classify",
            json={"text": text, "labels": labels, "multi_label": multi_label, "thinking": False},
        )
        r.raise_for_status()
        return r.json()

    async def extract(self, text: str, schema: dict[str, Any]) -> dict[str, Any]:
        r = await self._client.post(
            "/extract",
            json={"text": text, "schema": schema, "thinking": False},
        )
        r.raise_for_status()
        return r.json()

    async def summarize(self, text: str, style: str | None = None, max_sentences: int = 3) -> dict[str, Any]:
        r = await self._client.post(
            "/summarize",
            json={
                "text": text,
                "style": style,
                "max_sentences": max_sentences,
                "thinking": False,
            },
        )
        r.raise_for_status()
        return r.json()

    # ---- multimodal helpers ------------------------------------------------

    async def _post_multipart(
        self, path: str, file_field: str, file_path: Path, extra: dict[str, str] | None = None
    ) -> dict[str, Any]:
        mime, _ = mimetypes.guess_type(file_path.name)
        files = {file_field: (file_path.name, file_path.read_bytes(), mime or "application/octet-stream")}
        data = extra or {}
        r = await self._client.post(path, files=files, data=data)
        r.raise_for_status()
        return r.json()

    async def vision_explain(
        self, image: Path, *, subject: str | None = None, question: str | None = None
    ) -> dict[str, Any]:
        extra = {}
        if subject:
            extra["subject"] = subject
        if question:
            extra["question"] = question
        return await self._post_multipart("/vision/explain-work", "image", image, extra)

    async def audio_listen(self, audio: Path) -> dict[str, Any]:
        return await self._post_multipart("/audio/listen", "audio", audio)

    async def audio_transcribe(self, audio: Path) -> dict[str, Any]:
        return await self._post_multipart("/audio/transcribe", "audio", audio)

    async def audio_translate(
        self, audio: Path, *, target_language: str, source_language: str | None = None
    ) -> dict[str, Any]:
        extra: dict[str, str] = {"target_language": target_language}
        if source_language:
            extra["source_language"] = source_language
        return await self._post_multipart("/audio/translate", "audio", audio, extra)

    async def video_analyze(
        self,
        video: Path,
        *,
        task: str | None = None,
        n_frames: int = 6,
        include_audio: bool = True,
    ) -> dict[str, Any]:
        extra: dict[str, str] = {
            "n_frames": str(n_frames),
            "include_audio": "true" if include_audio else "false",
        }
        if task:
            extra["task"] = task
        return await self._post_multipart("/video/analyze-process", "video", video, extra)
