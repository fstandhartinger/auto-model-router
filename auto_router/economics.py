"""Cache-aware cost model and success model.

Two questions decide every routing choice:

  1. What does this turn cost on model m, given what m still holds in cache?
  2. How likely is m to get the turn right?

Cost
----
A prompt of P tokens of which W are warm on m (cached and not yet expired):

    cost = W * [hit * read + (1-hit) * write]      # warm part, may still miss
         + (P - W) * write                         # new tokens are written to cache
         + O * output

``write`` is the input price on providers without a write premium, 1.25x input
on Anthropic's 5-minute cache. Below the provider's minimum cacheable prefix
nothing is cached and everything is plain input. ``hit`` is a measured
per-model property, not a constant: a cross-region inference profile can miss
most of the time even though the prefix is "warm".

Switching model abandons W, so the next turn on the new model pays P at the
write price. Staying pays W at the read price. With reads at ~0.1x input the
gap is roughly 10x on the prefix, which is why switching needs a reason.

Success
-------
``SuccessModel`` maps a benchmark capability score (0..100 for the category)
and a task difficulty (0..1) to a success probability with a logistic curve
fitted on a graded task set (see experiments/). Capability-per-dollar
reasoning then works for models that were never evaluated locally.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .catalog import ModelInfo


def is_warm(model: ModelInfo, last_used: float | None, now: float, safety_seconds: float = 15.0) -> bool:
    if last_used is None or model.cache.ttl_seconds <= 0:
        return False
    return now - last_used <= max(0.0, model.cache.ttl_seconds - safety_seconds)


def turn_cost(model: ModelInfo, prompt_tokens: int, warm_tokens: int, output_tokens: int,
              *, hit_rate: float | None = None) -> float:
    """Expected USD for one call. Subscription and free models return 0."""
    p = model.prices
    if p.is_free:
        return 0.0
    if prompt_tokens < model.cache.min_tokens:
        return (prompt_tokens * p.input + output_tokens * p.output) / 1e6
    hit = model.cache.hit_rate if hit_rate is None else hit_rate
    warm = max(0, min(warm_tokens, prompt_tokens))
    cold = prompt_tokens - warm
    warm_cost = warm * (hit * p.read + (1.0 - hit) * p.write)
    return (warm_cost + cold * p.write + output_tokens * p.output) / 1e6


def token_equivalent_cost(model: ModelInfo, prompt_tokens: int, warm_tokens: int,
                          output_tokens: int, list_prices: "ModelInfo | None" = None) -> float:
    """What a call would cost at list price. Used to measure subscription quota use."""
    ref = list_prices or model
    if ref.prices.is_free:
        return 0.0
    return turn_cost(ref, prompt_tokens, warm_tokens, output_tokens)


def horizon_cost(model: ModelInfo, prompt_tokens: int, warm_tokens: int, output_tokens: int,
                 turns: float, growth_per_turn: int) -> float:
    """Cost of this turn plus ``turns - 1`` follow-ups that read the grown prefix warm."""
    total = turn_cost(model, prompt_tokens, warm_tokens, output_tokens)
    remaining = max(0.0, turns - 1.0)
    if remaining:
        follow = turn_cost(model, prompt_tokens + growth_per_turn, prompt_tokens + output_tokens,
                           output_tokens)
        total += remaining * follow
    return total


@dataclass
class SuccessModel:
    """P(success) = sigmoid((capability - (offset + slope*difficulty)) / scale).

    Defaults were fitted on the graded task set in experiments/ (see
    EXPERIMENTS.md). ``per_category`` overrides let a category be harder or
    easier than the benchmark score suggests.
    """

    offset: float = 20.0
    slope: float = 50.0
    scale: float = 6.0
    floor: float = 0.02
    ceiling: float = 0.98
    per_category: dict[str, tuple[float, float, float]] = field(default_factory=dict)
    benchmaxxing_weight: float = 0.5
    #: 0..1. How far a capability score that rests on weak, derived or stale
    #: evidence is pulled toward the neutral prior before it is turned into a
    #: success probability.
    #:
    #: The default is deliberately non-zero: the requirement is that missing or
    #: stale evidence must actually *change* the decision, not merely be
    #: annotated after the fact. Set it to ``0`` to reproduce the behaviour
    #: from before evidence strengths existed, in which every capability number
    #: is trusted equally however thin its basis.
    evidence_discount: float = 0.35
    #: Direct measurements win over the curve: (model, category, bucket) -> p.
    measured: dict[tuple[str, str, str], float] = field(default_factory=dict)

    @staticmethod
    def bucket(difficulty: float) -> str:
        return "easy" if difficulty < 0.34 else "medium" if difficulty < 0.67 else "hard"

    def p(self, model: ModelInfo, category: str, difficulty: float) -> float:
        key = (model.measurement_key, category, self.bucket(difficulty))
        if key in self.measured:
            return self.measured[key]
        offset, slope, scale = self.per_category.get(category, (self.offset, self.slope, self.scale))
        cap = model.cap(category, benchmaxxing_weight=self.benchmaxxing_weight,
                        evidence_discount=self.evidence_discount)
        z = (cap - (offset + slope * difficulty)) / scale
        z = max(-50.0, min(50.0, z))
        value = 1.0 / (1.0 + math.exp(-z))
        return max(self.floor, min(self.ceiling, value))

    def difficulty_at(self, model: ModelInfo, category: str, p: float = 0.5) -> float:
        """Difficulty at which ``model`` succeeds with probability ``p`` (curve only)."""
        offset, slope, scale = self.per_category.get(category, (self.offset, self.slope, self.scale))
        cap = model.cap(category, benchmaxxing_weight=self.benchmaxxing_weight,
                        evidence_discount=self.evidence_discount)
        logit = math.log(p / (1 - p))
        return max(0.0, min(1.0, (cap - scale * logit - offset) / slope))
