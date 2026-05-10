"""Pytest version of the extraction eval; gated on the live API."""
from __future__ import annotations

import pytest

from .metrics import extraction_metrics
from .runner import _load_dataset


@pytest.mark.asyncio
async def test_extraction_threshold(client):
    items = _load_dataset("extraction")
    preds, expected = [], []
    for item in items:
        resp = await client.extract(item["text"], item["schema"])
        preds.append(resp.get("data") or {})
        expected.append(item["expected"])
    metrics = extraction_metrics(preds, expected)
    assert metrics["json_valid_rate"] >= 0.8, metrics
