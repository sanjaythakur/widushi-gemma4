"""Helpers to normalise image inputs into OpenAI ``image_url`` content parts.

Accepts:
    * ``http(s)://...`` URLs
    * ``data:image/...;base64,...`` URIs
    * Raw base64 strings (we wrap in ``data:image/<mime>;base64,...``)
"""
from __future__ import annotations

import base64
import binascii
import re
from typing import Any

from pydantic import BaseModel, Field

_ALLOWED_URL_SCHEMES = ("http://", "https://", "data:")
_DATA_URI_RE = re.compile(r"^data:(?P<mime>[^;,]+)(?:;[^,]+)?,(?P<data>.+)$", re.DOTALL)


class ImageInput(BaseModel):
    """One image attachment for a multimodal request."""

    url: str | None = Field(
        default=None,
        description="Either an http(s) URL or a data: URI.",
    )
    base64: str | None = Field(
        default=None,
        description="Raw base64 payload; combined with `mime_type` to form a data URI.",
    )
    mime_type: str = Field(
        default="image/png",
        description="MIME type used when the input is raw base64.",
    )

    def to_openai_part(self) -> dict[str, Any]:
        return {"type": "image_url", "image_url": {"url": _to_image_url(self)}}


def _to_image_url(image: ImageInput) -> str:
    if image.url and image.base64:
        raise ValueError("Provide either `url` or `base64`, not both")
    if image.url:
        if not image.url.startswith(_ALLOWED_URL_SCHEMES):
            raise ValueError(
                f"Image URL must start with one of {_ALLOWED_URL_SCHEMES}; got {image.url[:32]!r}"
            )
        return image.url
    if image.base64:
        cleaned = image.base64.strip()
        # Tolerate data: URIs accidentally pasted into the base64 field.
        m = _DATA_URI_RE.match(cleaned)
        if m:
            return cleaned
        try:
            base64.b64decode(cleaned, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError(f"Invalid base64 payload: {exc}") from exc
        return f"data:{image.mime_type};base64,{cleaned}"
    raise ValueError("ImageInput requires either `url` or `base64`")


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
    images: list[ImageInput] | None = None,
    *,
    image_urls: list[str] | None = None,
    audio_urls: list[str] | None = None,
) -> str | list[dict[str, Any]]:
    """Build the OpenAI ``content`` field for a user message.

    Per the Gemma 4 docs, visual/audio tokens MUST appear before the text
    prompt for the best multimodal reasoning. The ordering here is therefore
    images -> audio -> text.

    ``images`` is the typed :class:`ImageInput` form used by ``/chat`` and
    ``/generate``; ``image_urls`` and ``audio_urls`` accept already-prepared
    ``data:`` URIs from the multipart endpoints.
    """
    has_media = bool(images) or bool(image_urls) or bool(audio_urls)
    if not has_media:
        return text

    parts: list[dict[str, Any]] = []
    if images:
        parts.extend(img.to_openai_part() for img in images)
    if image_urls:
        parts.extend(image_part_from_url(u) for u in image_urls)
    if audio_urls:
        parts.extend(audio_part_from_data_url(u) for u in audio_urls)
    if text:
        parts.append({"type": "text", "text": text})
    return parts
