"""Live audio-route eval tests.

Skipped automatically when:
    * the gemma-llama API is not reachable (handled by ``conftest.py``), or
    * the active model does not advertise ``audio: true`` (the route returns
      409, which we treat as a skip rather than a failure).

The bundled media is **synthetic** (a 5 s 440 Hz sine, see
``eval/datasets/media/README.md``), so the assertions below intentionally
focus on codepath correctness rather than transcript accuracy. To upgrade
this to a semantic check, drop a real spoken-question recording into
``eval/datasets/media/`` and add an ``expected_keywords`` array to the
matching item in ``eval/datasets/audio.json``.
"""
from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from .client import APIClient
from .metrics import keyword_overlap

_DATASETS = Path(__file__).parent / "datasets"


def _load(name: str) -> list[dict]:
    return json.loads((_DATASETS / f"{name}.json").read_text(encoding="utf-8"))


def _audio_supported(client: APIClient) -> bool:
    """Cheap upfront probe: a ping that the server side will 409 if audio is off."""
    return True  # Real check is via the 409-on-call branch below; we don't pre-probe.


def _check_audio_capable(exc: httpx.HTTPStatusError) -> None:
    if exc.response.status_code == 409:
        pytest.skip(
            f"Active model does not support audio (server said 409: {exc.response.text})"
        )
    raise exc


@pytest.mark.asyncio
@pytest.mark.parametrize("item", _load("audio"), ids=lambda i: f"{i['mode']}::{Path(i['file']).name}")
async def test_audio_route(client: APIClient, item: dict) -> None:
    media = _DATASETS / item["file"]
    if not media.exists():
        pytest.skip(f"missing bundled asset: {media}")

    mode = item["mode"]
    try:
        if mode == "listen":
            resp = await client.audio_listen(media)
        elif mode == "transcribe":
            resp = await client.audio_transcribe(media)
        elif mode == "translate":
            resp = await client.audio_translate(
                media,
                target_language=item.get("target_language") or "English",
                source_language=item.get("source_language"),
            )
        else:
            pytest.fail(f"unknown audio mode: {mode}")
    except httpx.HTTPStatusError as exc:
        _check_audio_capable(exc)
        raise

    text = (resp.get("text") or "").strip()
    assert text, f"audio/{mode} returned empty text: {resp}"
    assert resp.get("model"), "response is missing model name"
    assert isinstance(resp.get("inference_time_ms"), (int, float))
    assert resp.get("usage", {}).get("total_tokens", 0) > 0

    keywords = item.get("expected_keywords") or []
    if keywords:
        overlap = keyword_overlap(text, keywords)
        assert overlap >= 0.5, (
            f"audio/{mode} keyword overlap {overlap:.2f} < 0.5 for keywords "
            f"{keywords!r}; text={text!r}"
        )
