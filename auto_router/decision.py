"""The routing decision record: four things kept strictly apart.

A router that stores "we picked model X and it cost $0.004" has already lost
the ability to tell you whether it was right. This module keeps four different
kinds of statement in four different places, and nothing merges them:

``TaskClassification``
    What the request *is*. A judgement about the task, produced by Jev or by
    the cautious fallback. Carries its own confidence and its own source. It
    never names a model.

``RouteSelection``
    What the router *decided*, and why: the candidates it priced, the evidence
    behind each one, the cache decision, the model it chose, the fallback it
    would use, and whether the safe fallback was triggered. Owned entirely by
    code in ``policies.py`` - Jev supplies inputs, never decisions.

``EstimatedOutcome``
    What the router *expected* before the call: expected cost, success
    probability, tokens. Always labelled ``estimate`` and always carrying the
    basis it was computed from.

``ObservedOutcome``
    What actually *happened*: status, latency, billed-token counts as reported
    by the provider, cost when - and only when - a price basis exists. An
    observation with no measurement basis is recorded as ``None``, never as a
    zero or as the estimate. A route that told us it ran out of output budget
    is recorded here as a failed attempt and nowhere else: an observation never
    writes back into a capability score.

``RoutingExplanation`` bundles the four and renders a compact dict. That dict
is the only thing that leaves the process (API, ledger, logs), and it holds no
prompt text, no response text, no system prompt, no tool arguments and no
credentials - only counts, identifiers, categories and numbers. The
``no prompt text`` property is enforced by a test, not by convention.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from .catalog import ModelInfo
from .verify import Verdict


def _r(value: float | None, digits: int = 4) -> float | None:
    return None if value is None else round(float(value), digits)


# ---------------------------------------------------------------------------
# 1. what the task is
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TaskClassification:
    """A judgement about the request. Contains no model name and no decision."""

    category: str
    difficulty: float
    source: str                      # "jev" | "fallback" | "tool-loop"
    category_confidence: float = 0.0
    difficulty_confidence: float = 0.0
    needs_tools: bool = False
    needs_vision: bool = False
    needs_long_context: bool = False
    follow_up: float = 0.5
    stakes_usd: float = 0.0
    classifier_model: str = ""
    latency_ms: float = 0.0
    #: Category the classifier named before the catalog mapped it onto a
    #: capability category, when the two differ.
    raw_category: str = ""

    def to_dict(self) -> dict:
        return {
            "category": self.category,
            "raw_category": self.raw_category or self.category,
            "difficulty": _r(self.difficulty, 3),
            "source": self.source,
            "classifier_model": self.classifier_model or None,
            "category_confidence": _r(self.category_confidence, 3),
            "difficulty_confidence": _r(self.difficulty_confidence, 3),
            "needs_tools": self.needs_tools,
            "needs_vision": self.needs_vision,
            "needs_long_context": self.needs_long_context,
            "follow_up": _r(self.follow_up, 3),
            "stakes_usd": _r(self.stakes_usd, 4),
            "latency_ms": _r(self.latency_ms, 1),
        }


# ---------------------------------------------------------------------------
# 2. what the router decided
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CandidateEvaluation:
    """One route the router priced for this turn."""

    model: str
    capability: float
    capability_basis: str
    evidence_strength: str
    evidence_stale: bool
    p_success: float
    est_call_usd: float | None
    est_expected_usd: float | None
    warm_tokens: int
    subscription: str | None = None
    rejected: str | None = None

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "capability": _r(self.capability, 2),
            "capability_basis": self.capability_basis,
            "evidence_strength": self.evidence_strength,
            "evidence_stale": self.evidence_stale,
            "estimated_p_success": _r(self.p_success, 3),
            "estimated_call_usd": _r(self.est_call_usd, 6),
            "estimated_expected_usd": _r(self.est_expected_usd, 6),
            "warm_tokens": self.warm_tokens,
            "subscription": self.subscription,
            "rejected": self.rejected,
        }


@dataclass(frozen=True)
class CacheDecision:
    """What the router believed about the provider-side prefix cache, and why.

    ``estimated_usd_avoided`` is populated only when a measurement basis
    exists: a cache read price and an observed or configured hit rate. When
    either is missing it stays ``None`` and ``basis`` says so. It is never
    inferred from a list price alone.
    """

    model: str
    status: str                       # warm | cold | too-short | no-cache
    prompt_tokens: int
    warm_tokens: int
    ttl_seconds: int
    min_tokens: int
    hit_rate: float | None
    hit_rate_basis: str
    #: sha256 prefix identifier, never the prompt itself.
    key_scope: str = ""
    estimated_tokens_read_warm: int = 0
    estimated_usd_avoided: float | None = None
    basis: str = "no measurement basis recorded"

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "status": self.status,
            "prompt_tokens": self.prompt_tokens,
            "warm_tokens": self.warm_tokens,
            "ttl_seconds": self.ttl_seconds,
            "min_cacheable_tokens": self.min_tokens,
            "hit_rate": _r(self.hit_rate, 3),
            "hit_rate_basis": self.hit_rate_basis,
            "key_scope": self.key_scope,
            "estimated_tokens_read_warm": self.estimated_tokens_read_warm,
            "estimated_usd_avoided": _r(self.estimated_usd_avoided, 6),
            "basis": self.basis,
        }


@dataclass(frozen=True)
class RouteSelection:
    """The decision itself. Made by code in ``policies.py``, never by Jev."""

    selected: str
    policy: str
    reason: str
    fallback: str | None = None
    switched_from: str | None = None
    considered: int = 0
    evidence_confidence: float = 1.0
    safe_fallback: str | None = None
    turn_start: bool = True

    def to_dict(self) -> dict:
        return {
            "selected": self.selected,
            "fallback": self.fallback,
            "policy": self.policy,
            "reason": self.reason,
            "switched_from": self.switched_from,
            "candidates_considered": self.considered,
            "evidence_confidence": _r(self.evidence_confidence, 3),
            "safe_fallback": self.safe_fallback,
            "turn_start": self.turn_start,
        }


# ---------------------------------------------------------------------------
# 3 and 4. expected versus happened
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class EstimatedOutcome:
    """Before the call. Always labelled as an estimate, never compared as cash."""

    model: str
    p_success: float
    cost_usd: float | None
    prompt_tokens: int
    output_tokens: int
    basis: str = "router cost model, list prices"

    def to_dict(self) -> dict:
        return {
            "kind": "estimate",
            "model": self.model,
            "p_success": _r(self.p_success, 3),
            "cost_usd": _r(self.cost_usd, 6),
            "prompt_tokens": self.prompt_tokens,
            "output_tokens": self.output_tokens,
            "basis": self.basis,
        }


@dataclass
class ObservedOutcome:
    """After the call. Only what the provider actually reported."""

    model: str
    #: ``truncated`` is a 200 the provider itself flagged as a length stop:
    #: real tokens, real cost, no finished answer. See ``truncation.py``.
    #: ``not_taken`` is a decision that was recorded but deliberately not acted
    #: on - the advisory record kept while subscription traffic is forwarded
    #: unchanged (see ``shim.advisory_passthrough``). It is an observation about
    #: the router, not about the model, and no capability follows from it.
    status: str = "pending"           # ok | truncated | not_taken | upstream_error | transport_error | pending
    http_status: int | None = None
    latency_ms: float | None = None
    uncached_input_tokens: int | None = None
    cached_read_tokens: int | None = None
    cache_write_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None
    cost_basis: str = "not measured"
    escalated_from: str | None = None
    attempts: int = 1
    #: Short error class name; never the upstream body, which can echo the prompt.
    error: str | None = None

    @property
    def cache_hit_rate(self) -> float | None:
        read = self.cached_read_tokens
        total = (read or 0) + (self.uncached_input_tokens or 0) + (self.cache_write_tokens or 0)
        return None if (read is None or not total) else read / total

    def to_dict(self) -> dict:
        return {
            "kind": "observed",
            "model": self.model,
            "status": self.status,
            "http_status": self.http_status,
            "latency_ms": _r(self.latency_ms, 1),
            "tokens": {
                "uncached_input": self.uncached_input_tokens,
                "cached_read": self.cached_read_tokens,
                "cache_write": self.cache_write_tokens,
                "output": self.output_tokens,
            },
            "observed_cache_hit_rate": _r(self.cache_hit_rate, 4),
            "cost_usd": _r(self.cost_usd, 6),
            "cost_basis": self.cost_basis,
            "escalated_from": self.escalated_from,
            "attempts": self.attempts,
            "error": self.error,
        }


# ---------------------------------------------------------------------------
# the bundle
# ---------------------------------------------------------------------------
@dataclass
class RoutingExplanation:
    """Classification, decision, estimate and observation, kept apart.

    ``to_dict`` is the wire format for ``/v1/router/decisions`` and the JSONL
    ledger. It is prompt-free by construction: every field is a count, an
    identifier, a category, a probability or a price.
    """

    conversation: str
    classification: TaskClassification
    selection: RouteSelection
    estimated: EstimatedOutcome
    cache: CacheDecision
    candidates: list[CandidateEvaluation] = field(default_factory=list)
    observed: ObservedOutcome | None = None
    #: What the answer judge said about the answer this decision produced, when
    #: it was asked at all. A fifth kind of statement, kept apart from the other
    #: four for the same reason they are kept apart from each other: it is a
    #: judgement about an *answer*, made after the fact by a model, and it must
    #: never be read as a measurement of the route.
    verification: Verdict | None = None
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    created_at: float = field(default_factory=time.time)
    #: Free-form notes the router wants surfaced (stale evidence, outages).
    notes: list[str] = field(default_factory=list)

    #: Candidates listed in full; the rest are summarised by count only.
    MAX_CANDIDATES = 12

    def to_dict(self) -> dict:
        candidates = sorted(
            self.candidates,
            key=lambda c: (c.est_expected_usd is None, c.est_expected_usd or 0.0))
        return {
            "id": self.id,
            "created_at": round(self.created_at, 3),
            "conversation": self.conversation,
            "classification": self.classification.to_dict(),
            "selection": self.selection.to_dict(),
            "cache": self.cache.to_dict(),
            "estimated_outcome": self.estimated.to_dict(),
            "observed_outcome": self.observed.to_dict() if self.observed else None,
            "verification": self.verification.to_dict() if self.verification else None,
            "candidates": [c.to_dict() for c in candidates[: self.MAX_CANDIDATES]],
            "candidates_omitted": max(0, len(candidates) - self.MAX_CANDIDATES),
            "notes": list(self.notes),
        }

    def summary(self) -> str:
        """One line for a log or a response header. No prompt text."""
        c, s = self.classification, self.selection
        bits = [f"{c.category}@{c.difficulty:.2f}({c.source})",
                f"-> {s.selected}",
                f"p={self.estimated.p_success:.2f}",
                f"est=${self.estimated.cost_usd:.5f}" if self.estimated.cost_usd is not None
                else "est=n/a",
                f"cache={self.cache.status}"]
        if s.safe_fallback:
            bits.append(f"safe-fallback:{s.safe_fallback}")
        return " ".join(bits)


def candidate_from(model: ModelInfo, category: str, *, p_success: float,
                   call_usd: float | None, expected_usd: float | None, warm_tokens: int,
                   evidence_discount: float = 0.0, rejected: str | None = None,
                   ) -> CandidateEvaluation:
    return CandidateEvaluation(
        model=model.name,
        capability=model.cap(category, evidence_discount=evidence_discount),
        capability_basis=model.evidence_basis(category),
        evidence_strength=model.evidence_strength(category),
        evidence_stale=model.evidence_stale,
        p_success=p_success,
        est_call_usd=call_usd,
        est_expected_usd=expected_usd,
        warm_tokens=warm_tokens,
        subscription=model.subscription,
        rejected=rejected,
    )
