"""Routing policies.

All policies share one interface so the simulator and the live router run the
exact same code:

    choice = policy.choose(conversation, request, ctx)       # before a user turn
    retry  = policy.on_failure(conversation, request, ctx, failed_model)

Decisions are made at *user-turn* boundaries. Inside an agent's tool loop
(the request ends with a tool result) the router stays on the model that
started the turn: switching mid-loop throws away a cache that is seconds old
and hands a half-finished plan to a different model.

Policies
--------
A  static      one fixed strong model.
B  naive       per turn, the cheapest model (list price) whose predicted success
               clears a bar. Ignores the cache.
C  ev_switch   B's target, but a downgrade only happens when the saving over a
               horizon beats P(target insufficient) x cost of redoing the turn.
D  escalate    start on the cheapest adequate model, escalate on a failure
               signal, remember the escalation; downgrade only when the
               escalated model's cache has expired or the saving over the
               horizon beats the risk.
E  D + subscription tier priced by quota pacing (quota.py).
F  expected    minimise expected total cost = call cost (cache-aware) +
               P(fail) x [P(detected) x retry cost + P(undetected) x stakes],
               over a horizon, with difficulty memory and the subscription tier.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .catalog import Catalog, ModelInfo
from .economics import SuccessModel, horizon_cost, is_warm, turn_cost
from .quota import QuotaDecision


# ---------------------------------------------------------------------------
# shared state
# ---------------------------------------------------------------------------
@dataclass
class TurnRequest:
    category: str
    difficulty: float                 # classifier estimate, 0..1
    prompt_tokens: int
    output_tokens: int                # estimate for the whole turn's model output
    now: float
    steps: int = 1                    # expected calls inside the turn (tool loop)
    growth_per_step: int = 1500       # prompt growth per call inside the loop
    needs_tools: bool = False
    needs_vision: bool = False
    follow_up: float = 0.5            # P(turn relies on the previous turn's context)
    stakes_usd: float = 2.0           # cost of an undetected wrong answer
    detect_prob: float = 0.6          # P(a failure is noticed and retried)
    remaining_turns: float = 3.0      # expected further user turns in the conversation
    difficulty_confidence: float = 1.0


@dataclass
class Conversation:
    current: str | None = None
    #: model -> (tokens cached, last call start time)
    warm: dict[str, tuple[int, float]] = field(default_factory=dict)
    #: difficulty memory: the conversation proved at least this hard
    floor: float = 0.0
    floor_set_at: float = 0.0
    turns: int = 0
    switches: int = 0
    escalations: int = 0

    def warm_tokens(self, model: ModelInfo, now: float) -> int:
        entry = self.warm.get(model.name)
        if not entry:
            return 0
        tokens, last = entry
        return tokens if is_warm(model, last, now) else 0

    def record_call(self, model: ModelInfo, prompt_tokens: int, output_tokens: int, now: float) -> None:
        if prompt_tokens >= model.cache.min_tokens and model.cache.ttl_seconds > 0:
            self.warm[model.name] = (prompt_tokens + output_tokens, now)
        if self.current is not None and self.current != model.name:
            self.switches += 1
        self.current = model.name


@dataclass
class Context:
    catalog: Catalog
    success: SuccessModel
    #: subscription name -> pacing decision
    quota: dict[str, QuotaDecision] = field(default_factory=dict)
    #: subscription name -> list-price reference model name (for shadow pricing)
    subscription_reference: dict[str, str] = field(default_factory=dict)


@dataclass
class Choice:
    model: str
    reason: str
    expected_cost: float = 0.0
    p_success: float = 0.0


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def turn_call_cost(model: ModelInfo, req: TurnRequest, warm_tokens: int, ctx: Context) -> float:
    """Dollar (or shadow-dollar) cost of running a whole user turn on ``model``."""
    effective = _priced(model, ctx)
    if effective is None:
        return math.inf
    total = 0.0
    prompt, warm = req.prompt_tokens, warm_tokens
    per_step_out = max(1, req.output_tokens // max(1, req.steps))
    for _ in range(max(1, req.steps)):
        total += turn_cost(effective, prompt, warm, per_step_out)
        warm = prompt + per_step_out
        prompt += req.growth_per_step
    return total


def _priced(model: ModelInfo, ctx: Context) -> ModelInfo | None:
    """Subscription models priced at multiplier x reference list price; None when closed."""
    if not model.subscription:
        return model
    decision = ctx.quota.get(model.subscription)
    if decision is None or not decision.open:
        return None
    ref_name = ctx.subscription_reference.get(model.name)
    ref = ctx.catalog.get(ref_name) if ref_name else None
    if ref is None or decision.multiplier == 0:
        return model  # free at the margin
    from .catalog import Prices
    p = ref.prices
    m = decision.multiplier
    scaled = Prices(p.input * m, p.output * m,
                    None if p.cache_read is None else p.cache_read * m,
                    None if p.cache_write is None else p.cache_write * m)
    return model.with_(prices=scaled, cache=ref.cache)


def candidates(req: TurnRequest, ctx: Context, *, allow_subscription: bool) -> list[ModelInfo]:
    pool = ctx.catalog.eligible(needs_vision=req.needs_vision, needs_tools=req.needs_tools,
                                prompt_tokens=req.prompt_tokens)
    if not pool:
        # Nothing claims enough context: keep the largest windows rather than failing the request.
        widest = max((m.context_tokens for m in ctx.catalog.all()), default=0)
        pool = [m for m in ctx.catalog.all() if m.context_tokens == widest]
    out = []
    for m in pool:
        if m.subscription:
            if not allow_subscription:
                continue
            if _priced(m, ctx) is None:
                continue
        out.append(m)
    return out


def list_blended(model: ModelInfo) -> float:
    return model.prices.input * 0.8 + model.prices.output * 0.2


def effective_difficulty(conv: Conversation, req: TurnRequest) -> float:
    """Classifier estimate, lifted by what this conversation already proved.

    The memory only applies to follow-up turns: a self-contained question in a
    hard conversation is judged on its own.
    """
    d = req.difficulty
    if conv.floor > d:
        d = d + (conv.floor - d) * req.follow_up
    return d


def strongest(models: list[ModelInfo], category: str, ctx: Context) -> ModelInfo:
    return max(models, key=lambda m: (m.cap(category, benchmaxxing_weight=ctx.success.benchmaxxing_weight),
                                      -list_blended(m)))


# ---------------------------------------------------------------------------
# policies
# ---------------------------------------------------------------------------
class Policy:
    name = "base"
    allow_subscription = False

    def choose(self, conv: Conversation, req: TurnRequest, ctx: Context) -> Choice:
        raise NotImplementedError

    def on_failure(self, conv: Conversation, req: TurnRequest, ctx: Context,
                   failed: str, tried: set[str]) -> Choice | None:
        """Default escalation: the cheapest model that is clearly more capable."""
        failed_m = ctx.catalog[failed]
        pool = [m for m in candidates(req, ctx, allow_subscription=self.allow_subscription)
                if m.name not in tried]
        better = [m for m in pool if m.cap(req.category) > failed_m.cap(req.category) + 1.0]
        if not better:
            return None
        d = max(effective_difficulty(conv, req), req.difficulty)
        target = min(better, key=lambda m: (turn_call_cost(m, req, conv.warm_tokens(m, req.now), ctx)
                                            / max(ctx.success.p(m, req.category, d), 0.05)))
        return Choice(target.name, f"escalate after failure on {failed}")


class StaticPolicy(Policy):
    name = "A_static"

    def __init__(self, model: str | None = None):
        self.model = model

    def choose(self, conv, req, ctx):
        pool = candidates(req, ctx, allow_subscription=False)
        if self.model and self.model in {m.name for m in pool}:
            return Choice(self.model, "static model")
        return Choice(strongest(pool, req.category, ctx).name, "static: strongest model")

    def on_failure(self, conv, req, ctx, failed, tried):
        return Choice(failed, "static: retry on the same model")


class NaivePolicy(Policy):
    name = "B_naive"

    def __init__(self, target_p: float = 0.8):
        self.target_p = target_p

    def target(self, req: TurnRequest, ctx: Context, d: float, allow_sub: bool = False) -> ModelInfo:
        pool = candidates(req, ctx, allow_subscription=allow_sub)
        ok = [m for m in pool if ctx.success.p(m, req.category, d) >= self.target_p]
        if not ok:
            return max(pool, key=lambda m: (ctx.success.p(m, req.category, d), -list_blended(m)))
        return min(ok, key=lambda m: (list_blended(m), -m.cap(req.category)))

    def choose(self, conv, req, ctx):
        m = self.target(req, ctx, req.difficulty)
        return Choice(m.name, f"cheapest model with p>={self.target_p:.2f}",
                      p_success=ctx.success.p(m, req.category, req.difficulty))


class EVSwitchPolicy(NaivePolicy):
    """The expected-value switch rule: escalate freely, downgrade only when it pays."""

    name = "C_ev_switch"

    def __init__(self, target_p: float = 0.8, horizon_turns: float = 4.0):
        super().__init__(target_p)
        self.horizon = horizon_turns

    def choose(self, conv, req, ctx):
        target = self.target(req, ctx, req.difficulty)
        current = ctx.catalog.get(conv.current)
        if current is None or current.name == target.name:
            return Choice(target.name, "first turn or already on target")
        if target.cap(req.category) > current.cap(req.category):
            return Choice(target.name, "escalation: capability outranks cache economics")
        return self._downgrade(conv, req, ctx, current, target, req.difficulty)

    def _downgrade(self, conv, req, ctx, current, target, d) -> Choice:
        out = req.output_tokens
        growth = req.growth_per_step * req.steps
        stay = horizon_cost(current, req.prompt_tokens, conv.warm_tokens(current, req.now), out,
                            self.horizon, growth)
        switch = horizon_cost(target, req.prompt_tokens, conv.warm_tokens(target, req.now), out,
                              self.horizon, growth)
        p_ok = ctx.success.p(target, req.category, d)
        redo = turn_cost(current, req.prompt_tokens, 0, out)
        risk = (1 - p_ok) * (redo + (1 - req.detect_prob) * req.stakes_usd)
        if stay - switch > risk:
            return Choice(target.name, f"downgrade pays: saves ${stay - switch:.4f} > risk ${risk:.4f}")
        return Choice(current.name, f"stay: saving ${stay - switch:.4f} <= risk ${risk:.4f}")


class EscalatePolicy(EVSwitchPolicy):
    """Start cheap, escalate on failure, remember it, downgrade only on expiry or clear savings."""

    name = "D_escalate"

    def __init__(self, start_p: float = 0.6, target_p: float = 0.8, horizon_turns: float = 4.0,
                 memory_half_life_s: float = 1800.0):
        super().__init__(target_p, horizon_turns)
        self.start_p = start_p
        self.half_life = memory_half_life_s

    def _decayed_floor(self, conv: Conversation, now: float) -> float:
        if conv.floor <= 0:
            return 0.0
        age = max(0.0, now - conv.floor_set_at)
        return conv.floor * 0.5 ** (age / self.half_life)

    def choose(self, conv, req, ctx):
        floor = self._decayed_floor(conv, req.now)
        d = req.difficulty + max(0.0, floor - req.difficulty) * req.follow_up
        pool = candidates(req, ctx, allow_subscription=self.allow_subscription)
        ok = [m for m in pool if ctx.success.p(m, req.category, d) >= self.start_p]
        cheapest = (min(ok, key=lambda m: (turn_call_cost(m, req, conv.warm_tokens(m, req.now), ctx),
                                           -m.cap(req.category)))
                    if ok else strongest(pool, req.category, ctx))
        current = ctx.catalog.get(conv.current)
        if current is None or current.name not in {m.name for m in pool}:
            return Choice(cheapest.name, f"start cheap (p>={self.start_p:.2f})")
        if current.name == cheapest.name:
            return Choice(current.name, "stay: already on the cheapest adequate model")
        if cheapest.cap(req.category) > current.cap(req.category):
            return Choice(cheapest.name, "current model below the start bar for this turn")
        if conv.warm_tokens(current, req.now) == 0:
            return Choice(cheapest.name, "downgrade: escalated model's cache has expired")
        return self._downgrade(conv, req, ctx, current, cheapest, d)

    def on_failure(self, conv, req, ctx, failed, tried):
        failed_m = ctx.catalog[failed]
        conv.floor = max(conv.floor, ctx.success.difficulty_at(failed_m, req.category, 0.5), req.difficulty)
        conv.floor_set_at = req.now
        conv.escalations += 1
        return super().on_failure(conv, req, ctx, failed, tried)


class EscalateSubscriptionPolicy(EscalatePolicy):
    name = "E_escalate_sub"
    allow_subscription = True


class ExpectedCostPolicy(EscalatePolicy):
    """Minimise expected cost of a turn including failure, retry, stakes and cache."""

    name = "F_expected"
    allow_subscription = True

    def __init__(self, horizon_turns: float = 3.0, memory_half_life_s: float = 1800.0,
                 switch_margin: float = 0.0):
        super().__init__(horizon_turns=horizon_turns, memory_half_life_s=memory_half_life_s)
        self.switch_margin = switch_margin

    def value(self, m: ModelInfo, conv: Conversation, req: TurnRequest, ctx: Context, d: float,
              pool: list[ModelInfo]) -> tuple[float, float]:
        warm = conv.warm_tokens(m, req.now)
        call = turn_call_cost(m, req, warm, ctx)
        p = ctx.success.p(m, req.category, d)
        stronger = [x for x in pool if x.cap(req.category) > m.cap(req.category) + 1.0]
        if stronger:
            retry_model = max(stronger, key=lambda x: ctx.success.p(x, req.category, d)
                              / max(turn_call_cost(x, req, conv.warm_tokens(x, req.now), ctx), 1e-6))
            p_retry = ctx.success.p(retry_model, req.category, d)
            retry = (turn_call_cost(retry_model, req, conv.warm_tokens(retry_model, req.now), ctx)
                     + (1 - p_retry) * (1 - req.detect_prob) * req.stakes_usd)
        else:
            retry = call + (1 - p) * req.stakes_usd
        fail = req.detect_prob * retry + (1 - req.detect_prob) * req.stakes_usd
        immediate = call + (1 - p) * fail
        # Future turns: the chosen model will be warm, others cold. Charge the
        # horizon at this model's warm follow-up price so cheap-cache models
        # are rewarded for staying power.
        if self.horizon > 1 and math.isfinite(call):
            follow_req = TurnRequest(**{**req.__dict__,
                                        "prompt_tokens": req.prompt_tokens + req.growth_per_step * req.steps})
            follow = turn_call_cost(m, follow_req, follow_req.prompt_tokens, ctx) + (1 - p) * fail
            immediate += (min(self.horizon, req.remaining_turns + 1) - 1) * follow * 0.5
        return immediate, p

    def choose(self, conv, req, ctx):
        floor = self._decayed_floor(conv, req.now)
        d = req.difficulty + max(0.0, floor - req.difficulty) * req.follow_up
        pool = candidates(req, ctx, allow_subscription=self.allow_subscription)
        scored = []
        for m in pool:
            v, p = self.value(m, conv, req, ctx, d, pool)
            if math.isfinite(v):
                scored.append((v, -m.cap(req.category), m, p))
        if not scored:
            m = strongest(pool, req.category, ctx)
            return Choice(m.name, "no finite option; strongest model")
        scored.sort(key=lambda t: (t[0], t[1]))
        best_v, _, best, best_p = scored[0]
        current = ctx.catalog.get(conv.current)
        if current is not None and current.name != best.name:
            cur = next((s for s in scored if s[2].name == current.name), None)
            if cur and cur[0] <= best_v * (1 + self.switch_margin):
                return Choice(current.name, "stay: switching gains less than the margin", cur[0], cur[3])
        return Choice(best.name, f"min expected cost ${best_v:.4f} (p={best_p:.2f})", best_v, best_p)


POLICIES = {
    "A_static": StaticPolicy,
    "B_naive": NaivePolicy,
    "C_ev_switch": EVSwitchPolicy,
    "D_escalate": EscalatePolicy,
    "E_escalate_sub": EscalateSubscriptionPolicy,
    "F_expected": ExpectedCostPolicy,
}
