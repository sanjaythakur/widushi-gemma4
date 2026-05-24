"""Prompt-builder seam shared by every mode router.

Phase 1A only threads ``learner_block`` through the existing per-mode
Jinja templates. Phase 1B will grow ``episodic_block`` and
``working_block`` here, and Phase 4 will add ``curriculum_block``. We
land the seam now so the routers don't have to be touched again later.
"""
from __future__ import annotations

from ..prompts import _env  # noqa: F401  -- shared Jinja Environment


def render_with_learner(
    template_str: str, *, learner_block: str = "", **kwargs: object
) -> str:
    """Render a mode prompt string with the standard memory placeholders.

    Today this is a one-line shim around the per-mode ``render_*_prompt``
    helpers in :mod:`app.prompts`; we keep the helpers because the
    routers were already importing them. As Phase 1B and beyond add new
    block placeholders this is where the merging logic will grow.
    """
    template = _env.from_string(template_str)
    return template.render(learner_block=learner_block, **kwargs)


__all__ = ["render_with_learner"]
