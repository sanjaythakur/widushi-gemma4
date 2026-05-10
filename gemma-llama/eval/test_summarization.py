"""Pytest version of the summarization eval; gated on the live API."""
from __future__ import annotations

import pytest

from .metrics import summarization_metrics
from .runner import _load_dataset


@pytest.mark.asyncio
async def test_summarization_threshold(client):
    items = _load_dataset("summarization")
    preds, refs = [], []
    for item in items:
        resp = await client.summarize(
            item["text"], style=item.get("style"), max_sentences=item.get("max_sentences", 3)
        )
        preds.append(resp.get("summary", ""))
        refs.append(item["reference"])
    metrics = summarization_metrics(preds, refs)
    assert metrics["rouge1"] >= 0.2, metrics
