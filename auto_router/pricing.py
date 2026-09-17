"""Turn observed provider `usage` blocks into USD.

Every provider reports cache accounting differently, so normalise first and
price second. Getting this wrong silently invalidates the whole cost comparison,
so unknown shapes are reported rather than guessed.
"""

from __future__ import annotations

from dataclasses import dataclass

from .catalog import ModelInfo


@dataclass
class Usage:
    """Normalised token accounting for one call."""

    uncached_input: int = 0
    cached_read: int = 0
    cache_write: int = 0
    output: int = 0

    @property
    def total_input(self) -> int:
        return self.uncached_input + self.cached_read + self.cache_write

    def to_dict(self) -> dict:
        return {
            "uncached_input": self.uncached_input,
            "cached_read": self.cached_read,
            "cache_write": self.cache_write,
            "output": self.output,
            "total_input": self.total_input,
        }


def parse_openai_usage(raw: dict) -> Usage:
    """OpenAI-compatible chat-completions `usage`.

    prompt_tokens is the TOTAL input on this API, with cached and written
    portions broken out underneath, so the uncached remainder is a subtraction.
    That is the opposite of Anthropic's convention below - mixing them up is the
    easiest way to double-count.
    """
    prompt = int(raw.get("prompt_tokens") or 0)
    details = raw.get("prompt_tokens_details") or {}
    cached = int(details.get("cached_tokens") or 0)
    written = int(details.get("cache_write_tokens") or 0)
    return Usage(
        uncached_input=max(0, prompt - cached - written),
        cached_read=cached,
        cache_write=written,
        output=int(raw.get("completion_tokens") or 0),
    )


def parse_anthropic_usage(raw: dict) -> Usage:
    """Anthropic Messages `usage`.

    Here input_tokens counts ONLY the tokens after the last cache breakpoint:
        total = input_tokens + cache_read_input_tokens + cache_creation_input_tokens
    """
    return Usage(
        uncached_input=int(raw.get("input_tokens") or 0),
        cached_read=int(raw.get("cache_read_input_tokens") or 0),
        cache_write=int(raw.get("cache_creation_input_tokens") or 0),
        output=int(raw.get("output_tokens") or 0),
    )


def cost_usd(model: ModelInfo, usage: Usage) -> float:
    """Actual cost of a call from its reported usage.

    Subscription and free models cost 0.0 at the margin; callers that need a
    comparable number use ``list_cost_usd`` with a list-price reference.
    """
    p = model.prices
    return (
        usage.uncached_input * p.input
        + usage.cached_read * p.read
        + usage.cache_write * p.write
        + usage.output * p.output
    ) / 1_000_000.0


def cache_hit_rate(usage: Usage) -> float:
    return usage.cached_read / usage.total_input if usage.total_input else 0.0
