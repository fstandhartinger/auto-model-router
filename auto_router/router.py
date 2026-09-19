"""Live routing: turn a request into a model choice, and learn from the outcome.

Shared by the HTTP server and the Claude Code shim. Holds per-conversation
state (current model, warm caches, difficulty memory) in memory.
"""

from __future__ import annotations

import logging
import math
import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field, replace
from typing import Any, Callable

from . import jev
from .cache_index import CalibratedEstimator, estimate_tokens, prefix_hashes
from .catalog import CATEGORIES, Catalog, ModelInfo
from .config import RouterConfig
from .decision import (
    CacheDecision,
    EstimatedOutcome,
    ObservedOutcome,
    RouteSelection,
    RoutingExplanation,
    TaskClassification,
    candidate_from,
)
from .economics import SuccessModel, turn_cost
from .ledger import RoutingLedger
from .outcome_memory import AvoidanceRule, OutcomeKey, OutcomeMemory, budget_bucket
from .policies import (POLICIES, Context, Conversation, ExpectedCostPolicy, Policy, TurnRequest,
                       turn_call_cost)
from .quota import (PacingRule, QuotaDecision, decide as quota_decide, from_budget_file,
                    from_codex_rollouts, from_command)
from .verify import (VerifyPolicy, Verdict, bump_floor, escalation_choice, not_verified,
                     retry_messages, verdict_from)

log = logging.getLogger("auto_router.router")


def _finite(value: float) -> float | None:
    return None if math.isinf(value) else value

def success_model_from_config(policy: dict) -> SuccessModel:
    """Logistic curve parameters and an optional measured success table.

    ``policy.success``: {offset, slope, scale, table: path}. The table is JSON
    ``[[model, category, bucket, p], ...]`` as written by experiments/calibrate.py.
    Measured rates beat benchmark-derived curves by a wide margin (see EXPERIMENTS.md).
    """
    import json
    conf = policy.get("success") or {}
    model = SuccessModel(**{k: conf[k] for k in ("offset", "slope", "scale", "evidence_discount")
                            if k in conf})
    table = conf.get("table")
    if table:
        with open(os.path.expanduser(table)) as fh:
            model.measured = {(m, c, b): float(p) for m, c, b, p in json.load(fh)}
    return model


#: Stakes score (0..1) -> dollars an undetected wrong answer is worth.
STAKES_USD = (0.05, 0.5, 3.0, 20.0)


def last_user_text(messages: list[dict]) -> str:
    for message in reversed(messages):
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            texts = [b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"]
            if texts:
                return "\n".join(texts)
    return ""


def is_tool_continuation(messages: list[dict]) -> bool:
    """True when the newest message feeds a tool result back (we are inside a tool loop)."""
    if not messages:
        return False
    last = messages[-1]
    if last.get("role") == "tool":
        return True
    content = last.get("content")
    if last.get("role") == "user" and isinstance(content, list):
        types = {b.get("type") for b in content if isinstance(b, dict)}
        return "tool_result" in types and "text" not in types
    return False


def recent_tool_errors(messages: list[dict], window: int = 6) -> int:
    """Consecutive failing tool results at the end of the conversation (tests failing, errors)."""
    markers = ("Traceback", "FAILED", "Error:", "error:", "AssertionError", "exit code 1", "is_error")
    count = 0
    for message in reversed(messages[-window * 2:]):
        content = message.get("content")
        blobs: list[str] = []
        if message.get("role") == "tool" and isinstance(content, str):
            blobs = [content]
        elif isinstance(content, list):
            for b in content:
                if isinstance(b, dict) and b.get("type") == "tool_result":
                    if b.get("is_error"):
                        blobs.append("is_error")
                    c = b.get("content")
                    blobs.append(c if isinstance(c, str) else str(c))
        if not blobs:
            if message.get("role") == "assistant":
                continue
            break
        if any(m in blob for blob in blobs for m in markers):
            count += 1
        else:
            break
    return count


@dataclass
class RouteResult:
    model: ModelInfo
    reason: str
    conversation_id: str
    turn_start: bool
    request: TurnRequest
    classification: jev.Classification | None
    started_at: float
    classification_ms: float = 0.0
    tried: set[str] = field(default_factory=set)
    #: Full separated decision record. See decision.py.
    explanation: RoutingExplanation | None = None
    #: Which requests this one is comparable to, for the observed-outcome
    #: memory. ``None`` for anything whose outcome says nothing about whether
    #: a route finishes (a launched job). See ``outcome_memory.py``.
    outcome_key: OutcomeKey | None = None

    @property
    def headers(self) -> dict[str, str]:
        head = {
            "X-Router-Model": self.model.name,
            "X-Router-Category": self.request.category,
            "X-Router-Difficulty": f"{self.request.difficulty:.2f}",
            "X-Router-Reason": self.reason[:600].replace("\n", " "),
            "X-Router-Conversation": self.conversation_id,
            "X-Router-Turn-Start": "true" if self.turn_start else "false",
        }
        if self.explanation is not None:
            head["X-Router-Decision"] = self.explanation.id
            head["X-Router-Evidence"] = f"{self.explanation.selection.evidence_confidence:.2f}"
            head["X-Router-Cache"] = self.explanation.cache.status
            if self.explanation.selection.safe_fallback:
                head["X-Router-Safe-Fallback"] = self.explanation.selection.safe_fallback
            verdict = self.explanation.verification
            if verdict is not None and verdict.verified and verdict.p_adequate is not None:
                head["X-Router-Verified"] = f"{verdict.p_adequate:.2f}"
                head["X-Router-Verify-Failure"] = verdict.failure
                if verdict.escalated_to:
                    head["X-Router-Verify-Escalated-To"] = verdict.escalated_to
        return head


class Router:
    def __init__(self, config: RouterConfig, policy: Policy | None = None,
                 success: SuccessModel | None = None,
                 classifier: Callable[[str, str], jev.Classification] | None = None,
                 quota_reader: Callable[[], dict[str, QuotaDecision]] | None = None,
                 ledger: "RoutingLedger | None" = None,
                 judge: Callable[..., jev.Judgement] | None = None):
        self.config = config
        name = (config.policy or {}).get("name") or os.environ.get("AUTO_ROUTER_POLICY", "F_expected")
        self.policy = policy or POLICIES[name]()
        #: Answer verification: the policy that decides which answers are
        #: checked, and the judge that checks them. A deployment without a Jev
        #: key keeps a policy object - so the configuration still reads the
        #: same - but no judge, and then nothing is ever checked and nothing is
        #: priced as if it were.
        self.verify = VerifyPolicy.from_config(config.policy or {})
        self.judge = judge or (jev.judge if os.environ.get("TYPESAFE_API_KEY") else None)
        if isinstance(self.policy, ExpectedCostPolicy) and self.policy.verify is None:
            self.policy.verify = self.verify
            self.policy.judge_available = self.judge is not None
        self.success = success or success_model_from_config(config.policy or {})
        cal = (config.policy or {}).get("jev_difficulty_calibration") or [0.0, 1.0]
        self.jev_offset, self.jev_scale = float(cal[0]), float(cal[1]) or 1.0
        self.classifier = classifier or (jev.classify if os.environ.get("TYPESAFE_API_KEY") else None)
        self.quota_reader = quota_reader or self._read_quota
        self.conversations: dict[str, Conversation] = {}
        self.estimator = CalibratedEstimator()
        self.escalate_after_tool_errors = int((config.policy or {}).get("escalate_after_tool_errors", 3))
        self._lock = threading.Lock()
        self._quota_cache: tuple[float, dict[str, QuotaDecision]] = (0.0, {})
        #: Recent decision records, newest last. Bounded so a long-running
        #: server cannot grow without limit.
        self.decisions: deque[RoutingExplanation] = deque(
            maxlen=int((config.policy or {}).get("decision_history", 200)))
        self.ledger = ledger or RoutingLedger.from_env()
        #: Observed length stops per route and comparable request, used to
        #: route around a route that repeatedly fails to finish.
        self.outcomes = OutcomeMemory(AvoidanceRule.from_config(config.policy or {}))

    # -- quota ---------------------------------------------------------------
    def _read_quota(self) -> dict[str, QuotaDecision]:
        out: dict[str, QuotaDecision] = {}
        for name, sub in (self.config.subscriptions or {}).items():
            rule = PacingRule(**{k: v for k, v in sub.items() if k in PacingRule.__dataclass_fields__})
            state = None
            key = sub.get("budget_key", name)
            if sub.get("usage_command"):
                # The operator's own reader is the truth; a budget file is a cache of it.
                state = from_command(list(sub["usage_command"]), key,
                                     ttl_s=float(sub.get("usage_command_ttl_s", 600)))
            if state is None and sub.get("budget_file"):
                state = from_budget_file(sub["budget_file"], key)
            if state is None and sub.get("codex_rollouts"):
                state = from_codex_rollouts(sub["codex_rollouts"])
            out[name] = quota_decide(state, rule)
        return out

    def quota(self) -> dict[str, QuotaDecision]:
        now = time.time()
        if now - self._quota_cache[0] > 60:
            self._quota_cache = (now, self.quota_reader())
        return self._quota_cache[1]

    def context(self) -> Context:
        refs: dict[str, str] = {}
        for entry in self.config.raw.get("models") or []:
            if entry.get("subscription") and entry.get("list_price_model"):
                refs[entry["name"]] = entry["list_price_model"]
        return Context(self.config.catalog, self.success, self.quota(), refs,
                       reference_catalog=self.config.catalog)

    # -- routing -------------------------------------------------------------
    def conversation_id(self, messages: list[dict], system: Any, tools: Any) -> str:
        hashes = prefix_hashes(messages[:1], system, tools)
        return hashes[0][:16] if hashes else "anonymous"

    def route(self, messages: list[dict], system: Any = None, tools: Any = None,
              max_tokens: int | None = None, now: float | None = None,
              exclude_subscriptions: bool = False) -> RouteResult:
        """Route one turn.

        ``exclude_subscriptions`` drops every plan route from the candidates.
        The gateway sets it for traffic that does not carry the plan's own
        login: a plan can only serve a request its official client
        authenticated itself, so offering it to anything else would be a
        decision nobody can carry out.
        """
        now = now or time.time()
        cid = self.conversation_id(messages, system, tools)
        with self._lock:
            conv = self.conversations.setdefault(cid, Conversation())
        prompt_tokens = self.estimator.estimate(estimate_tokens(messages, system, tools))
        continuation = is_tool_continuation(messages)
        ctx = self.context()
        if exclude_subscriptions:
            ctx = replace(ctx, catalog=Catalog([m for m in ctx.catalog.all() if not m.subscription]))

        if continuation and conv.current and conv.current in ctx.catalog:
            errors = recent_tool_errors(messages)
            base = TurnRequest(category="agentic", difficulty=conv.floor, prompt_tokens=prompt_tokens,
                               output_tokens=max_tokens or 2000, now=now, needs_tools=True)
            key = OutcomeKey(base.category, budget_bucket(max_tokens))
            if errors >= self.escalate_after_tool_errors:
                retry = self.policy.on_failure(conv, base, ctx, conv.current, {conv.current})
                if retry:
                    reason = f"{errors} failing tool results in a row: {retry.reason}"
                    return self._keyed(self._result(ctx, conv, cid, ctx.catalog[retry.model], reason,
                                                     base, None, now, 0.0, turn_start=False,
                                                     switched_from=conv.current), key)
            # Inside a tool loop the turn's model stays put - switching mid-loop
            # throws the cache away - so the memory is only fed here, not consulted.
            return self._keyed(self._result(ctx, conv, cid, ctx.catalog[conv.current],
                                            "inside a tool loop: stay on the turn's model", base,
                                            None, now, 0.0, turn_start=False), key)

        text = last_user_text(messages)
        started = time.perf_counter()
        cls = self.classifier(text, self._summary(messages, tools)) if self.classifier else None
        cls_ms = (time.perf_counter() - started) * 1000
        req = self._turn_request(cls, prompt_tokens, max_tokens, now, bool(tools), messages)
        choice = self.policy.choose(conv, req, ctx)
        key = OutcomeKey(req.category, budget_bucket(max_tokens))
        model, reason, memory = self._avoid_repeated_truncation(
            conv, req, ctx, key, ctx.catalog[choice.model], choice.reason)
        result = self._result(ctx, conv, cid, model, reason, req, cls, now,
                              cls_ms, turn_start=True, switched_from=conv.current,
                              prefix=self.conversation_id(messages, system, tools))
        if memory is not None and result.explanation is not None:
            note = memory.pop("note")
            result.explanation.selection = replace(result.explanation.selection,
                                                   truncation_memory=memory)
            result.explanation.notes.append(note)
        return self._keyed(result, key)

    @staticmethod
    def _keyed(result: RouteResult, key: OutcomeKey | None) -> RouteResult:
        result.outcome_key = key
        return result

    def _avoid_repeated_truncation(self, conv: Conversation, req: TurnRequest, ctx: Context,
                                   key: OutcomeKey, chosen: ModelInfo, reason: str
                                   ) -> tuple[ModelInfo, str, dict | None]:
        """Route around a route observed to run out of budget on comparable requests.

        Applied after the policy has chosen, so the policy, its estimate and
        every capability score stay exactly as they were: this only ever acts
        on *observed* outcomes, and only when ``OutcomeMemory`` finds they meet
        the configured evidence rule. The fallback is the policy's own best
        remaining route by expected cost among routes that are not flagged
        themselves; with no such route the choice stands and the record says
        so. Returns ``(model, reason, record-or-None)``.
        """
        record = self.outcomes.should_avoid(chosen.name, key, req.now)
        if record is None:
            return chosen, reason, None
        flagged = {chosen.name}
        usable = []
        for m, _call, _p, value in self.policy.evaluate(conv, req, ctx):
            if m.name == chosen.name or not math.isfinite(value):
                continue
            if self.outcomes.should_avoid(m.name, key, req.now) is not None:
                flagged.add(m.name)
                continue
            usable.append((value, m.name, m))
        memory = {"key": key.label(), "route": chosen.name, "basis": record.basis(),
                  "observed": record.observed, "truncated": record.truncated,
                  "flagged_routes": sorted(flagged)}
        if not usable:
            memory.update(applied=False, fallback=None,
                          note=(f"{chosen.name} repeatedly ran out of output budget on "
                                f"{key.label()}, but no usable unflagged route exists; "
                                f"the policy's choice stands."))
            return chosen, reason, memory
        value, _name, fallback = min(usable, key=lambda t: (t[0], t[1]))
        memory.update(applied=True, fallback=fallback.name,
                      note=(f"Avoided {chosen.name} on observed evidence: {record.basis()}."))
        return fallback, (f"avoid {chosen.name}: repeated observed length stops on {key.label()}; "
                          f"next usable route by expected cost (${value:.4f})"), memory

    def route_job(self, task: str, *, steps: int = 12, output_per_step: int = 1200,
                  force: str | None = None, now: float | None = None,
                  only: Callable[[ModelInfo], bool] | None = None) -> RouteResult:
        """Route a whole job rather than one turn: which *tool* should run this.

        The difference from :meth:`route` is the shape of the request, not the
        policy. A job handed to a coding agent is a long tool loop that starts
        cold - a new session, nothing cached anywhere - so it is priced as
        ``steps`` calls over a growing prefix, and no route gets credit for a
        warm cache it cannot have. The decision is recorded exactly like a turn
        decision, so a job and a turn are comparable in the same ledger.

        The dollar figures on the candidates are therefore an *estimate of a
        job*, built from the task text alone before any of it has run. They
        rank routes; they are not a measurement, and the observed outcome
        (``launcher``) records what really happened next to them.

        ``force`` names a route the operator insists on. The record is then
        built for *that* route - its cache state, its estimate, its candidate
        row - and says in one line which route the policy would have picked
        instead. An override the ledger describes as a routing decision would
        be the one lie that makes every later comparison worthless.
        """
        now = now or time.time()
        messages = [{"role": "user", "content": task}]
        # Truthful, and the only tool information available before the client
        # starts: every runnable route in this catalog is a coding agent with a
        # shell, a file editor and a reader.
        tools = [{"name": "shell"}, {"name": "edit"}, {"name": "read"}]
        ctx = self.context()
        if only is not None:
            # ``only`` narrows the candidates; price references stay reachable
            ctx = replace(ctx, catalog=Catalog([m for m in ctx.catalog.all() if only(m)]))
        cid = self.conversation_id(messages, None, tools)
        conv = Conversation()          # a launched job starts cold, always
        prompt_tokens = self.estimator.estimate(estimate_tokens(messages, None, tools))
        started = time.perf_counter()
        cls = self.classifier(task, self._job_summary(task, steps)) if self.classifier else None
        cls_ms = (time.perf_counter() - started) * 1000
        req = self._turn_request(cls, prompt_tokens, None, now, True, messages)
        req = replace(req, steps=max(1, steps), output_tokens=max(1, steps) * output_per_step,
                      remaining_turns=0.0)
        choice = self.policy.choose(conv, req, ctx)
        model = ctx.catalog[choice.model]
        reason = f"job of about {steps} calls: {choice.reason}"
        note = None
        if force and force != model.name:
            forced = ctx.catalog.get(force)
            if forced is None:
                raise KeyError(force)
            note = f"Route forced by the operator; the policy's own choice was {model.name}."
            reason = (f"operator override (--route {force}); "
                      f"the policy would have picked {model.name}")
            model = forced
        result = self._result(ctx, conv, cid, model, reason, req, cls, now, cls_ms, turn_start=True)
        if note and result.explanation is not None:
            result.explanation.notes.append(note)
        return result

    @staticmethod
    def _job_summary(task: str, steps: int) -> str:
        return (f"A whole coding-agent job, expected to take about {steps} model calls with a "
                f"shell, a file editor and a reader. Task length: {len(task)} characters.")

    # -- explanation ---------------------------------------------------------
    def _result(self, ctx: Context, conv: Conversation, cid: str, model: ModelInfo, reason: str,
                req: TurnRequest, cls: jev.Classification | None, now: float, cls_ms: float,
                *, turn_start: bool, switched_from: str | None = None,
                prefix: str = "") -> RouteResult:
        result = RouteResult(model, reason, cid, turn_start, req, cls, now, cls_ms)
        try:
            result.explanation = self.explain(ctx, conv, cid, model, reason, req, cls, now, cls_ms,
                                              turn_start=turn_start, switched_from=switched_from,
                                              prefix=prefix)
            self.decisions.append(result.explanation)
        except Exception:  # noqa: BLE001 - an explanation must never break routing
            log.exception("failed to build the routing explanation")
        return result

    def observe(self, result: RouteResult, outcome: ObservedOutcome) -> None:
        """Attach what actually happened. Estimates are never overwritten by it.

        The status - and only the status - also feeds the outcome memory, so a
        later comparable request can see that this route did not finish.
        """
        self.outcomes.record(outcome.model, result.outcome_key, outcome.status,
                             result.started_at)
        if result.explanation is None:
            return
        result.explanation.observed = outcome
        self.ledger.write(result.explanation)

    def explain(self, ctx: Context, conv: Conversation, cid: str, model: ModelInfo, reason: str,
                req: TurnRequest, cls: jev.Classification | None, now: float, cls_ms: float,
                *, turn_start: bool, switched_from: str | None = None,
                prefix: str = "") -> RoutingExplanation:
        """Build the compact, prompt-free decision record for one routing choice."""
        discount = ctx.success.evidence_discount
        scored = self.policy.evaluate(conv, req, ctx) if turn_start else []
        candidates = [
            candidate_from(m, req.category, p_success=p,
                           call_usd=None if math.isinf(call) else call,
                           expected_usd=None if math.isinf(value) else value,
                           warm_tokens=conv.warm_tokens(m, req.now),
                           evidence_discount=discount)
            for m, call, p, value in scored]
        if not any(c.model == model.name for c in candidates):
            candidates.append(candidate_from(
                model, req.category,
                p_success=ctx.success.p(model, req.category, req.difficulty),
                call_usd=_finite(turn_call_cost(model, req, conv.warm_tokens(model, req.now), ctx)),
                expected_usd=None, warm_tokens=conv.warm_tokens(model, req.now),
                evidence_discount=discount))

        chosen = next(c for c in candidates if c.model == model.name)
        ranked = sorted((c for c in candidates if c.model != model.name and c.est_expected_usd is not None),
                        key=lambda c: c.est_expected_usd)
        stronger = [c for c in candidates
                    if c.model != model.name and c.capability > chosen.capability]
        fallback = (min(stronger, key=lambda c: (c.est_expected_usd if c.est_expected_usd is not None
                                                 else float("inf"))).model
                    if stronger else (ranked[0].model if ranked else None))

        notes: list[str] = []
        safe_fallback = None
        if cls is not None and cls.failed:
            safe_fallback = "classifier-unavailable"
            notes.append("Jev was unreachable or unusable; cautious defaults were used.")
        elif cls is None and turn_start:
            safe_fallback = "classifier-disabled"
            notes.append("No classifier configured; category and difficulty are heuristic.")
        if chosen.evidence_stale:
            safe_fallback = safe_fallback or "stale-evidence"
            notes.append(f"Capability evidence for {model.name} is stale or absent "
                         f"({model.evidence.get('source', 'unknown')}).")
        if chosen.evidence_strength in ("none", "weak"):
            safe_fallback = safe_fallback or "weak-category-evidence"
            notes.append(f"No direct {req.category} evidence for {model.name}; "
                         f"basis: {chosen.capability_basis}.")
        if any(c.evidence_stale for c in candidates):
            notes.append("At least one candidate was priced from stale benchmark data.")

        classification = self._classification_record(cls, req, cls_ms, turn_start)
        cache = self._cache_decision(model, conv, req, prefix)
        evidence_confidence = self._evidence_confidence(cls, chosen)
        selection = RouteSelection(
            selected=model.name, policy=self.policy.name, reason=reason, fallback=fallback,
            switched_from=switched_from if switched_from != model.name else None,
            considered=len(candidates), evidence_confidence=evidence_confidence,
            safe_fallback=safe_fallback, turn_start=turn_start)
        estimated = EstimatedOutcome(
            model=model.name, p_success=chosen.p_success,
            cost_usd=chosen.est_call_usd, prompt_tokens=req.prompt_tokens,
            output_tokens=req.output_tokens,
            basis=("router cost model over configured list prices"
                   + ("; subscription shadow-priced by quota pacing" if model.subscription else "")))
        return RoutingExplanation(conversation=cid, classification=classification,
                                  selection=selection, estimated=estimated, cache=cache,
                                  candidates=candidates, notes=notes)

    @staticmethod
    def _classification_record(cls: jev.Classification | None, req: TurnRequest, cls_ms: float,
                               turn_start: bool) -> TaskClassification:
        if not turn_start:
            return TaskClassification(category=req.category, difficulty=req.difficulty,
                                      source="tool-loop", needs_tools=req.needs_tools,
                                      stakes_usd=req.stakes_usd, follow_up=req.follow_up)
        if cls is None:
            return TaskClassification(category=req.category, difficulty=req.difficulty,
                                      source="disabled", needs_tools=req.needs_tools,
                                      needs_vision=req.needs_vision, follow_up=req.follow_up,
                                      stakes_usd=req.stakes_usd, latency_ms=cls_ms)
        return TaskClassification(
            category=req.category, raw_category=cls.category, difficulty=req.difficulty,
            source=cls.source, category_confidence=cls.category_confidence,
            difficulty_confidence=cls.difficulty_confidence, needs_tools=req.needs_tools,
            needs_vision=req.needs_vision, needs_long_context=cls.needs_long_context > 0.5,
            follow_up=req.follow_up, stakes_usd=req.stakes_usd,
            classifier_model=cls.model, latency_ms=cls_ms)

    @staticmethod
    def _cache_decision(model: ModelInfo, conv: Conversation, req: TurnRequest,
                        prefix: str) -> CacheDecision:
        """What the router believed about the prefix cache, with the basis recorded.

        A saving is only ever quoted when both halves of the measurement exist:
        a published cache-read price and a hit rate that came from somewhere
        nameable. Otherwise the field stays None and says why.
        """
        warm = conv.warm_tokens(model, req.now)
        rules, prices = model.cache, model.prices
        if rules.ttl_seconds <= 0:
            status = "no-cache"
        elif req.prompt_tokens < rules.min_tokens:
            status = "too-short"
        elif warm > 0:
            status = "warm"
        else:
            status = "cold"

        measured = model.evidence.get("cache_hit_rate_basis")
        hit_basis = measured or ("provider default for the configured cache family "
                                 f"({rules.hit_rate:.2f}); measure it per route before trusting it")
        read_tokens = 0
        avoided: float | None = None
        basis = "cache not warm for this route"
        if status == "warm":
            read_tokens = int(min(warm, req.prompt_tokens) * rules.hit_rate)
            if prices.cache_read is None:
                basis = ("no cache-read price published for this route, so no saving is claimed")
            elif prices.is_free:
                basis = "route is free or subscription-priced; no cash saving to claim"
            else:
                cold = turn_cost(model, req.prompt_tokens, 0, req.output_tokens)
                hot = turn_cost(model, req.prompt_tokens, warm, req.output_tokens)
                avoided = max(0.0, cold - hot)
                basis = (f"estimate = cold turn cost - warm turn cost at the configured hit rate "
                         f"{rules.hit_rate:.2f} and published cache-read price "
                         f"{prices.cache_read} /Mtok; not a measured saving")
        return CacheDecision(model=model.name, status=status, prompt_tokens=req.prompt_tokens,
                             warm_tokens=warm, ttl_seconds=rules.ttl_seconds,
                             min_tokens=rules.min_tokens, hit_rate=rules.hit_rate,
                             hit_rate_basis=hit_basis, key_scope=prefix[:16],
                             estimated_tokens_read_warm=read_tokens,
                             estimated_usd_avoided=avoided, basis=basis)

    @staticmethod
    def _evidence_confidence(cls: jev.Classification | None, chosen) -> float:
        """How much of this decision rests on current, direct evidence (0..1).

        The two classifier confidences are combined with ``min``, not ``max``:
        the route choice depends on the category *and* the difficulty, so a
        confident category with an unknown difficulty is not a confident
        decision. An absent confidence is reported as 0 rather than being
        rounded up to "probably fine".
        """
        from .catalog import EVIDENCE_WEIGHT
        capability = EVIDENCE_WEIGHT.get(chosen.evidence_strength, 0.0)
        if cls is None:
            classifier = 0.0          # no classifier at all: nothing is evidenced
        elif cls.failed:
            classifier = 0.0
        else:
            classifier = min(cls.category_confidence, cls.difficulty_confidence)
        return round(min(1.0, max(0.0, 0.5 * capability + 0.5 * classifier)), 3)

    def _summary(self, messages: list[dict], tools: Any) -> str:
        users = sum(1 for m in messages if m.get("role") == "user")
        names = []
        for t in tools or []:
            names.append(t.get("name") or (t.get("function") or {}).get("name") or "?")
        previous = ""
        for message in reversed(messages[:-1]):
            if message.get("role") == "user":
                previous = last_user_text([message])[:400]
                if previous:
                    break
        return (f"{len(messages)} messages, {users} from the user. Tools: {', '.join(names)[:300] or 'none'}. "
                f"Previous user request: {previous or '(none)'}")

    def _turn_request(self, cls: jev.Classification | None, prompt_tokens: int, max_tokens: int | None,
                      now: float, has_tools: bool, messages: list[dict]) -> TurnRequest:
        request_chars = len(last_user_text(messages))
        if cls is None or cls.failed:
            return TurnRequest(category="agentic" if has_tools else "general", difficulty=0.5,
                               prompt_tokens=prompt_tokens, output_tokens=max_tokens or 1500, now=now,
                               needs_tools=has_tools, stakes_usd=STAKES_USD[2], difficulty_confidence=0.0,
                               request_chars=request_chars)
        stakes_idx = min(len(STAKES_USD) - 1, int(round(cls.stakes * (len(STAKES_USD) - 1))))
        agentic = has_tools and cls.needs_tools > 0.5
        difficulty = max(0.0, min(1.0, (cls.difficulty - self.jev_offset) / self.jev_scale))
        return TurnRequest(
            category=cls.category if cls.category in self.success_categories() else "general",
            difficulty=difficulty,
            prompt_tokens=prompt_tokens,
            output_tokens=min(max_tokens or 4000, 4000) if not agentic else 6000,
            now=now,
            steps=8 if agentic else 1,
            needs_tools=has_tools,
            needs_vision=cls.needs_vision > 0.5,
            follow_up=cls.follow_up,
            stakes_usd=STAKES_USD[stakes_idx],
            detect_prob=0.8 if agentic else 0.5,
            difficulty_confidence=cls.difficulty_confidence,
            needs_long_context=cls.needs_long_context > 0.5,
            request_chars=request_chars,
        )

    @staticmethod
    def success_categories() -> set[str]:
        from .catalog import CATEGORIES
        return set(CATEGORIES)

    # -- outcome -------------------------------------------------------------
    def commit(self, result: RouteResult, prompt_tokens: int | None = None, output_tokens: int = 0,
               raw_estimate: int | None = None) -> None:
        with self._lock:
            conv = self.conversations.setdefault(result.conversation_id, Conversation())
            tokens = prompt_tokens or result.request.prompt_tokens
            conv.record_call(result.model, tokens, output_tokens, result.started_at)
            if result.turn_start:
                conv.turns += 1
        if prompt_tokens and raw_estimate:
            self.estimator.observe(raw_estimate, prompt_tokens)

    # -- verification --------------------------------------------------------
    def check(self, result: RouteResult, request_text: str, answer: str) -> Verdict:
        """Ask the judge whether a cheap route's answer really answers the request.

        The request text is a parameter rather than something the router kept:
        a ``RouteResult`` is prompt-free by construction and stays that way.
        Whoever holds the answer - the HTTP server, the shim, an experiment -
        holds the request too, and passes both in.

        The verdict is attached to the decision record either way, including
        when the answer was not checked and why, so a reader of
        ``/v1/router/decisions`` can tell "the judge approved it" from "the
        judge was never asked".
        """
        carried = result.explanation.verification if result.explanation is not None else None
        if carried is not None and carried.escalated_to:
            # This route *is* the second attempt. The verdict that produced it
            # is the verdict that belongs on it, and grading a stronger model's
            # answer is the one thing this whole module refuses to do.
            return carried
        req = result.request
        applies, why = self.verify.applies(
            result.model, req.category, request_chars=req.request_chars or len(request_text),
            needs_long_context=req.needs_long_context,
            evidence_discount=self.success.evidence_discount,
            judge_available=self.judge is not None)
        if not applies:
            verdict = not_verified(why, model=result.model.name, category=req.category)
        elif not answer.strip():
            verdict = not_verified("the route returned no text to check",
                                   model=result.model.name, category=req.category)
        else:
            try:
                judgement = self.judge(request_text, answer, category=req.category)
            except Exception:  # noqa: BLE001 - a broken judge must not break the turn
                log.exception("the answer judge raised")
                verdict = not_verified("the judge raised an error",
                                       model=result.model.name, category=req.category)
            else:
                verdict = verdict_from(judgement, self.verify, req.category, result.model.name)
        self._attach_verdict(result, verdict)
        return verdict

    def escalate_after_verdict(self, result: RouteResult, verdict: Verdict,
                               messages: list[dict], answer: str
                               ) -> tuple[RouteResult | None, list[dict], Verdict]:
        """Route the second attempt at a turn the judge rejected.

        Raises the conversation's difficulty floor first, so the *next* turn in
        this conversation does not start below the level this one just proved
        it needs - the point Scott Shapiro made about a bad early pick
        cascading: without the memory, every turn of a chain repeats the same
        too-cheap choice and pays for it again.
        """
        if not verdict.escalate:
            return None, messages, verdict
        conv = self.conversations.setdefault(result.conversation_id, Conversation())
        with self._lock:
            bump_floor(conv, verdict, self.verify, result.request.now)
        retry = self._verified_retry(result, conv) or self.escalate(result)
        if retry is None:
            verdict = replace(verdict, reason="no stronger route available; the answer stands")
            self._attach_verdict(result, verdict)
            return None, messages, verdict
        verdict = replace(verdict, escalated_to=retry.model.name)
        self._attach_verdict(result, verdict)
        self._attach_verdict(retry, verdict)
        return retry, retry_messages(messages, answer, verdict, self.verify), verdict

    def _verified_retry(self, result: RouteResult, conv: Conversation) -> RouteResult | None:
        """A route clearly stronger than the one the judge rejected, if there is one."""
        ctx = self.context()
        tried = result.tried | {result.model.name}
        choice = escalation_choice(self.policy, conv, result.request, ctx, result.model.name,
                                   tried, self.verify)
        if choice is None:
            return None
        name, reason = choice
        retry = self._result(ctx, conv, result.conversation_id, ctx.catalog[name], reason,
                             result.request, result.classification, time.time(), 0.0,
                             turn_start=result.turn_start, switched_from=result.model.name)
        retry.tried = tried
        return self._keyed(retry, result.outcome_key)

    @staticmethod
    def _attach_verdict(result: RouteResult, verdict: Verdict) -> None:
        if result.explanation is not None:
            result.explanation.verification = verdict

    def escalate(self, result: RouteResult, *, availability: bool = False) -> RouteResult | None:
        """Pick a retry route after a failure signal.

        Two different failures are handled differently. A *capability* failure
        (the judge says the answer was inadequate, tool results keep failing)
        asks the policy for a clearly stronger model. An *availability* failure
        (the upstream returned 5xx, the connection broke) is not evidence that
        the model was too weak, so when no stronger model exists the router
        still moves to the next usable route rather than returning the error:
        that is the documented safe fallback. Returns None only when there is
        genuinely nowhere left to go.
        """
        conv = self.conversations.setdefault(result.conversation_id, Conversation())
        ctx = self.context()
        tried = result.tried | {result.model.name}
        retry = self.policy.on_failure(conv, result.request, ctx, result.model.name, tried)
        reason = retry.reason if retry else ""
        target = ctx.catalog.get(retry.model) if retry else None
        if target is None and availability:
            target, reason = self._next_available(conv, result.request, ctx, tried)
        if target is None:
            return None
        retry_result = self._result(ctx, conv, result.conversation_id, target, reason,
                                    result.request, result.classification, time.time(), 0.0,
                                    turn_start=result.turn_start,
                                    switched_from=result.model.name)
        retry_result.tried = tried
        return self._keyed(retry_result, result.outcome_key)

    def _next_available(self, conv: Conversation, req: TurnRequest, ctx: Context,
                        tried: set[str]) -> tuple[ModelInfo | None, str]:
        """Best remaining route by the policy's own ranking, ignoring capability order."""
        scored = [(value, m) for m, _call, _p, value in self.policy.evaluate(conv, req, ctx)
                  if m.name not in tried and not math.isinf(value)]
        if not scored:
            return None, ""
        value, model = min(scored, key=lambda pair: pair[0])
        return model, (f"safe fallback: {len(tried)} route(s) unavailable, "
                       f"next usable route by expected cost (${value:.4f})")

    @property
    def stats(self) -> dict:
        return {
            "policy": self.policy.name,
            "conversations": len(self.conversations),
            "switches": sum(c.switches for c in self.conversations.values()),
            "escalations": sum(c.escalations for c in self.conversations.values()),
            "quota": {k: v.__dict__ for k, v in self.quota().items()},
            "token_estimator": self.estimator.stats,
            "evidence_discount": self.success.evidence_discount,
            "decisions_retained": len(self.decisions),
            "ledger": self.ledger.stats,
            "truncation_memory": self.outcomes.stats,
            "stale_evidence_models": sorted(m.name for m in self.config.catalog.all()
                                            if m.evidence_stale),
            "models": [m.name for m in self.config.catalog.all()],
        }
