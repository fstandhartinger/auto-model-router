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
import os
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
    #: Direct measurements: (model, category, bucket) -> p.
    measured: dict[tuple[str, str, str], float] = field(default_factory=dict)
    #: ``None`` (v1): a measured cell replaces the curve outright. A number (v2): the cell is
    #: evidence that is combined with the curve, which acts as a Beta prior worth this many
    #: pseudo-observations; the cell itself counts as ``measured_n`` observations. Small
    #: graded cells (4-6 tasks) then move the estimate without erasing the difference
    #: between models that the curve still sees.
    measured_prior_n: float | None = None
    measured_n: float = 6.0
    #: v2 only. Logit penalty per e-fold of (typical output tokens per task / request token
    #: budget), with the ratio clamped to ``length_clamp``. A model whose reasoning usually
    #: needs more tokens than the caller allows is cut off mid-answer, which no capability
    #: score reflects. 0 (v1) switches the term off.
    length_penalty: float = 0.0
    length_clamp: tuple[float, float] = (-2.0, 3.0)
    #: v2 only. True maps the logistic onto [floor, ceiling] (``floor + (ceiling - floor) * sigmoid``) instead of
    #: clipping it there, so two models never tie merely because both are above the ceiling.
    smooth_ceiling: bool = False
    #: Provenance of fitted parameters (file, hash, data); informational.
    fit_info: dict = field(default_factory=dict)

    @staticmethod
    def bucket(difficulty: float) -> str:
        return "easy" if difficulty < 0.34 else "medium" if difficulty < 0.67 else "hard"

    def curve(self, model: ModelInfo, category: str, difficulty: float,
              max_tokens: int | None = None) -> float:
        """The parametric success probability, ignoring measured cells."""
        offset, slope, scale = self.per_category.get(category, (self.offset, self.slope, self.scale))
        cap = model.cap(category, benchmaxxing_weight=self.benchmaxxing_weight,
                        evidence_discount=self.evidence_discount)
        z = (cap - (offset + slope * difficulty)) / scale
        if self.length_penalty and model.task_tokens:
            budget = min(model.max_output_tokens, max_tokens) if max_tokens else model.max_output_tokens
            ratio = math.log(max(model.task_tokens, 1.0) / max(budget, 1))
            z -= self.length_penalty * max(self.length_clamp[0], min(self.length_clamp[1], ratio))
        z = max(-50.0, min(50.0, z))
        value = 1.0 / (1.0 + math.exp(-z))
        if self.smooth_ceiling:
            return self.floor + (self.ceiling - self.floor) * value
        return max(self.floor, min(self.ceiling, value))

    def p(self, model: ModelInfo, category: str, difficulty: float,
          max_tokens: int | None = None) -> float:
        """Final probability; bucketed measured cells can break curve monotonicity."""
        value = self.curve(model, category, difficulty, max_tokens)
        key = (model.measurement_key, category, self.bucket(difficulty))
        if key in self.measured:
            if self.measured_prior_n is None:
                return self.measured[key]
            prior_n, obs_n = max(self.measured_prior_n, 0.0), max(self.measured_n, 0.0)
            if prior_n + obs_n > 0:
                value = (prior_n * value + obs_n * self.measured[key]) / (prior_n + obs_n)
        return value

    def difficulty_at(self, model: ModelInfo, category: str, p: float = 0.5) -> float:
        """Difficulty at which ``model`` succeeds with probability ``p`` (curve only)."""
        offset, slope, scale = self.per_category.get(category, (self.offset, self.slope, self.scale))
        cap = model.cap(category, benchmaxxing_weight=self.benchmaxxing_weight,
                        evidence_discount=self.evidence_discount)
        logit = math.log(p / (1 - p))
        return max(0.0, min(1.0, (cap - scale * logit - offset) / slope))


def success_v2(measured: dict[tuple[str, str, str], float] | None = None,
               path: "str | os.PathLike[str] | None" = None, *,
               allow_external_params: bool = False) -> SuccessModel:
    """The recalibrated success model (``auto_router/data/success_v2.json``).

    Same functional form as v1 (logistic in capability minus a difficulty-dependent
    offset) with refitted parameters, a ceiling that keeps strong models apart, a
    token-budget (truncation) term, and measured cells combined with the curve by Beta
    shrinkage instead of replacing it. ``SuccessModel()`` is unchanged (v1).
    Explicit paths resolve relative to the working directory and must stay inside
    the package, including symlinks, unless ``allow_external_params=True``.
    """
    import json
    import os
    from pathlib import Path

    file = Path(path) if path else Path(__file__).parent / "data" / "success_v2.json"
    target = file.resolve() if path else file
    if path and allow_external_params is not True:
        if not target.is_relative_to(Path(__file__).resolve().parent):
            raise ValueError("success v2 params must stay inside auto_router; "
                             "trusted operators may set allow_external_params=true")
    raw = json.loads(target.read_text())
    params = raw["params"]
    model = SuccessModel(
        offset=params["offset"], slope=params["slope"], scale=params["scale"],
        floor=params["floor"], ceiling=params["ceiling"],
        per_category={k: tuple(v) for k, v in params.get("per_category", {}).items()},
        length_penalty=params.get("length_penalty", 0.0), smooth_ceiling=params.get("smooth_ceiling", False),
        measured_prior_n=params.get("measured_prior_n"), measured_n=params.get("measured_n", 6.0),
        measured=dict(measured or {}))
    model.fit_info = {"file": str(file), "version": raw.get("version"), "fitted_on": raw.get("fitted_on")}
    return model
