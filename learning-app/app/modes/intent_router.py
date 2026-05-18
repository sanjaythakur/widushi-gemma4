"""IntentRouter: pick which Mode to enter on wake.

Post-wake the device always lands in :class:`FreeConvoMode`. Subsequent
mode swaps (FreeConvo -> VoiceMirror -> Vision -> Roleplay) happen
mid-session via ``EventType.MODE_CHANGE`` events that modes post from
their own ``run_thinking`` -- the router is *not* consulted for those.

Future work: replace the stub body with a tiny Gemma call that maps the
learner's first utterance to a non-default mode name. The signature is
intentionally compatible with that: it takes an :class:`Event` (so it
can read ``payload['prompt']`` or ``payload['audio_bytes']`` when
available) and returns a mode name.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable

from app.events import Event

log = logging.getLogger(__name__)


class IntentRouter:
    """Routes wake-time events to a mode name.

    ``known_modes`` is informational only — used to constrain real
    classifications once implemented. The stub does not consult it.
    """

    DEFAULT_MODE = "free_convo"

    def __init__(
        self,
        *,
        default: str = DEFAULT_MODE,
        known_modes: Iterable[str] | None = None,
    ) -> None:
        self._default = default
        self._known_modes = set(known_modes or ())

    @property
    def default(self) -> str:
        return self._default

    def classify(self, event: Event) -> str:
        """Return the mode name to enter for ``event``.

        Stub implementation: always returns the configured default
        (``"free_convo"`` for the device after wake). Logged at debug so
        live deployments can verify wiring without flooding logs.
        """

        log.debug(
            "intent_router stub: routing %s -> %s", event.type.name, self._default
        )
        return self._default


__all__ = ["IntentRouter"]
