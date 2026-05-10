"""Streaming sentence buffer for the TTS pipeline.

The wrapper feeds Gemma's text deltas in and yields whole sentences out the
moment they end. Sentences are split on terminal punctuation (``.!?``)
followed by whitespace or end-of-buffer; common abbreviations and decimal
numbers are masked first so we do not synthesise mid-word truncations.

The buffer also exposes a ``hard_flush`` based on a max character cap so
runaway output (no punctuation in 400 chars, say) still triggers timely
audio rather than silently growing forever.
"""
from __future__ import annotations

import re

# Conservative list of English abbreviations we never want to split on.
# Order matters only for readability; the regex is compiled once with all
# entries at module import.
_ABBREV = (
    "Mr", "Mrs", "Ms", "Dr", "Prof", "St", "Jr", "Sr",
    "vs", "etc", "i.e", "e.g", "cf", "approx",
    "Inc", "Ltd", "Co", "Corp",
    "U.S", "U.K", "U.N", "E.U",
    "a.m", "p.m",
)

_PROTECT_RE = re.compile(
    r"\b(" + "|".join(re.escape(a) for a in _ABBREV) + r")\."
)
_DECIMAL_RE = re.compile(r"(\d)\.(\d)")
_RESTORE_DOT = "\u0001"  # placeholder for protected dots
_SPLIT_RE = re.compile(r"(?<=[.!?])(?=\s|$)")


def _protect(text: str) -> str:
    text = _PROTECT_RE.sub(lambda m: m.group(1) + _RESTORE_DOT, text)
    text = _DECIMAL_RE.sub(lambda m: m.group(1) + _RESTORE_DOT + m.group(2), text)
    return text


def _restore(text: str) -> str:
    return text.replace(_RESTORE_DOT, ".")


class SentenceBuffer:
    """Accumulator that yields one sentence per ``add()`` flush boundary.

    Usage::

        buf = SentenceBuffer(max_chars=400)
        for sent in buf.add(delta):
            yield sent
        if (rem := buf.flush()):
            yield rem
    """

    def __init__(self, max_chars: int = 400) -> None:
        self.max_chars = max_chars
        self._buf = ""

    def add(self, chunk: str) -> list[str]:
        """Append ``chunk`` and return any sentences that completed.

        Sentences are returned in arrival order; the residual partial
        sentence stays in the buffer until a future ``add()`` or
        :meth:`flush` extracts it.
        """
        if not chunk:
            return []
        self._buf += chunk
        return self._drain()

    def flush(self) -> str:
        """Return whatever is buffered (possibly empty) and reset."""
        out, self._buf = self._buf, ""
        return out.strip()

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _drain(self) -> list[str]:
        sentences: list[str] = []

        while True:
            protected = _protect(self._buf)
            parts = _SPLIT_RE.split(protected, maxsplit=1)
            if len(parts) == 2 and parts[0]:
                sent = _restore(parts[0]).strip()
                # Keep the leading whitespace of the remainder out of the
                # next sentence so we do not emit "  Hello." style audio.
                self._buf = _restore(parts[1]).lstrip()
                if sent:
                    sentences.append(sent)
                continue

            # No sentence boundary; consider the hard-flush guard so
            # punctuation-free output still produces audio in bounded time.
            if len(self._buf) >= self.max_chars:
                # Flush at the last whitespace within the cap so we never
                # cut through a word.
                cut = self._buf.rfind(" ", 0, self.max_chars)
                if cut <= 0:
                    cut = self.max_chars
                head, tail = self._buf[:cut], self._buf[cut:]
                head = head.strip()
                self._buf = tail.lstrip()
                if head:
                    sentences.append(head)
                continue
            break

        return sentences
