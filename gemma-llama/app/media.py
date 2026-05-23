"""Helpers to build OpenAI-style multimodal ``content`` parts.

The five surviving mode endpoints all use ``multipart/form-data`` and feed
:func:`build_user_content` with pre-built ``data:`` URIs from
:mod:`app.media_multipart`; we therefore no longer need the typed
``ImageInput`` JSON shape that the dropped ``/generate`` and ``/chat``
endpoints used.
"""
from __future__ import annotations

from typing import Any


def image_part_from_url(url: str) -> dict[str, Any]:
    """Build an OpenAI ``image_url`` content part from a fully-formed URL/URI.

    Used by the multipart endpoints, which already produced ``data:image/...``
    URIs in :mod:`app.media_multipart`.
    """
    return {"type": "image_url", "image_url": {"url": url}}


def audio_part_from_data_url(data_url: str) -> dict[str, Any]:
    """Build a llama.cpp/OpenAI ``input_audio`` content part.

    Expects a ``data:audio/<fmt>;base64,...`` URI (as produced by
    :func:`app.media_multipart.read_audio_upload`). The format hint is parsed
    out of the URI so the model gets ``"format": "wav"`` etc.
    """
    fmt = "wav"
    if data_url.startswith("data:audio/"):
        try:
            fmt = data_url.split("data:audio/", 1)[1].split(";", 1)[0] or "wav"
        except IndexError:
            fmt = "wav"
    payload = data_url.split(",", 1)[1] if "," in data_url else data_url
    return {
        "type": "input_audio",
        "input_audio": {"data": payload, "format": fmt},
    }


def build_user_content(
    text: str,
    *,
    image_urls: list[str] | None = None,
    audio_urls: list[str] | None = None,
) -> str | list[dict[str, Any]]:
    """Build the OpenAI ``content`` field for a user message.

    Per the Gemma 4 docs, visual/audio tokens MUST appear before the text
    prompt for the best multimodal reasoning. The ordering here is therefore
    images -> audio -> text.

    ``image_urls`` / ``audio_urls`` accept already-prepared ``data:`` URIs
    from :mod:`app.media_multipart`.
    """
    has_media = bool(image_urls) or bool(audio_urls)
    if not has_media:
        return text

    parts: list[dict[str, Any]] = []
    if image_urls:
        parts.extend(image_part_from_url(u) for u in image_urls)
    if audio_urls:
        parts.extend(audio_part_from_data_url(u) for u in audio_urls)
    if text:
        parts.append({"type": "text", "text": text})
    return parts
