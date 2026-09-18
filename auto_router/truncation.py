"""Did the provider say it ran out of output budget?

A route that stops because it exhausted its output budget has not answered the
request - but it answers with HTTP 200 and a well-formed body, so the router
used to record it as a success and hand the half-finished text to the client.

This is not a hypothetical. The 18 September held-out run measured it: on the
two hardest web-design tasks one route burned a 12,000-token budget without
finishing the page, while a route the design-arena evidence rated *lower*
finished the same prompt in about 2,200 tokens. No capability score predicts
it. Only the attempt does, which is exactly what an observation is for.

This module answers one question and refuses to guess:

* **Only the provider's own machine-readable stop flag.** Never the answer
  text. "it ends mid-sentence" is prose inference, it is wrong often enough to
  matter, and acting on it would mean reading content this codebase
  deliberately keeps out of every record.
* **Only values from a fixed vocabulary.** The label this returns is built from
  a known field name and a known stop word, so it can be written to the ledger
  without carrying any fragment of a provider body - which routinely echoes the
  prompt and sometimes the rejected credential.
* **Silence is not evidence.** A missing field, an empty ``choices`` list, a
  vendor word never seen before, a body that is not a mapping at all: none of
  them are truncation.

What the router does with the answer lives in ``server.py``; what it means for
the route lives nowhere. Nothing here edits a capability score.
"""

from __future__ import annotations

from typing import Any

#: The stop words that mean "the output budget ran out", across the shapes this
#: router speaks to: OpenAI chat completions (``length``), Anthropic messages
#: (``max_tokens``), and the words gateways pass through from an origin
#: provider. Compared case-folded; anything outside this set is not truncation.
LENGTH_STOP_WORDS = frozenset({
    "length",
    "max_tokens",
    "max_output_tokens",
    "maxtokens",
    "model_length",
    "output_limit",
    "token_limit",
})

#: Where a provider is allowed to say it, in the order they are trusted. The
#: per-choice fields come first because a gateway's top-level summary can lag
#: behind the choice it summarises.
CHOICE_FIELDS = ("finish_reason", "native_finish_reason")
TOP_LEVEL_FIELDS = ("stop_reason", "finish_reason")


def _stop_word(value: Any) -> str | None:
    """The canonical stop word, when the value is one this module recognises."""
    if not isinstance(value, str):
        return None
    word = value.strip().casefold()
    return word if word in LENGTH_STOP_WORDS else None


def truncation_label(payload: Any) -> str | None:
    """``"<field>:<stop word>"`` when the provider flagged a length stop, else ``None``.

    Both halves of the label come from the fixed vocabularies above, so no part
    of ``payload`` is ever reproduced in the returned string.
    """
    if not isinstance(payload, dict):
        return None
    choices = payload.get("choices")
    if isinstance(choices, list):
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            for field in CHOICE_FIELDS:
                word = _stop_word(choice.get(field))
                if word:
                    return f"{field}:{word}"
    for field in TOP_LEVEL_FIELDS:
        word = _stop_word(payload.get(field))
        if word:
            return f"{field}:{word}"
    return None


def label_for_stop_reason(value: Any, field: str = "finish_reason") -> str | None:
    """Same judgement for a stop word already pulled out of a stream.

    ``field`` must be one of the names above; anything else is refused rather
    than echoed, so a caller cannot smuggle provider text into the label.
    """
    if field not in CHOICE_FIELDS and field not in TOP_LEVEL_FIELDS:
        raise ValueError(f"unknown stop-reason field {field!r}")
    word = _stop_word(value)
    return f"{field}:{word}" if word else None
