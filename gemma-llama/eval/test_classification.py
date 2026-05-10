"""Pytest version of the classification eval; gated on the live API."""
from __future__ import annotations

import pytest

from .metrics import classification_metrics
from .runner import _load_dataset, _normalise_labels


@pytest.mark.asyncio
async def test_classification_threshold(client):
    items = _load_dataset("classification")
    preds, expected = [], []
    for item in items:
        resp = await client.classify(item["text"], item["labels"], item.get("multi_label", False))
        preds.append(_normalise_labels(resp.get("labels")))
        expected.append(_normalise_labels(item["expected"]))
    metrics = classification_metrics(preds, expected)
    assert metrics["accuracy"] >= 0.5, metrics
