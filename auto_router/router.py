"""Live routing: turn a request into a model choice, and learn from the outcome.

Shared by the HTTP server and the Claude Code shim. Holds per-conversation
state (current model, warm caches, difficulty memory) in memory.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from . import jev
from .cache_index import CalibratedEstimator, estimate_tokens, prefix_hashes
from .catalog import ModelInfo
from .config import RouterConfig
from .economics import SuccessModel
from .policies import POLICIES, Context, Conversation, Policy, TurnRequest
from .quota import PacingRule, QuotaDecision, decide as quota_decide, from_budget_file, from_codex_rollouts

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

    @property
    def headers(self) -> dict[str, str]:
        return {
            "X-Router-Model": self.model.name,
            "X-Router-Category": self.request.category,
            "X-Router-Difficulty": f"{self.request.difficulty:.2f}",
            "X-Router-Reason": self.reason[:600].replace("\n", " "),
            "X-Router-Conversation": self.conversation_id,
            "X-Router-Turn-Start": "true" if self.turn_start else "false",
        }


class Router:
    def __init__(self, config: RouterConfig, policy: Policy | None = None,
                 success: SuccessModel | None = None,
                 classifier: Callable[[str, str], jev.Classification] | None = None,
                 quota_reader: Callable[[], dict[str, QuotaDecision]] | None = None):
        self.config = config
        name = (config.policy or {}).get("name") or os.environ.get("AUTO_ROUTER_POLICY", "F_expected")
        self.policy = policy or POLICIES[name]()
        self.success = success or SuccessModel()
        self.classifier = classifier or (jev.classify if os.environ.get("TYPESAFE_API_KEY") else None)
        self.quota_reader = quota_reader or self._read_quota
        self.conversations: dict[str, Conversation] = {}
        self.estimator = CalibratedEstimator()
        self.escalate_after_tool_errors = int((config.policy or {}).get("escalate_after_tool_errors", 3))
        self._lock = threading.Lock()
        self._quota_cache: tuple[float, dict[str, QuotaDecision]] = (0.0, {})

    # -- quota ---------------------------------------------------------------
    def _read_quota(self) -> dict[str, QuotaDecision]:
        out: dict[str, QuotaDecision] = {}
        for name, sub in (self.config.subscriptions or {}).items():
            rule = PacingRule(**{k: v for k, v in sub.items() if k in PacingRule.__dataclass_fields__})
            state = None
            if sub.get("budget_file"):
                state = from_budget_file(sub["budget_file"], sub.get("budget_key", name))
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
        return Context(self.config.catalog, self.success, self.quota(), refs)

    # -- routing -------------------------------------------------------------
    def conversation_id(self, messages: list[dict], system: Any, tools: Any) -> str:
        hashes = prefix_hashes(messages[:1], system, tools)
        return hashes[0][:16] if hashes else "anonymous"

    def route(self, messages: list[dict], system: Any = None, tools: Any = None,
              max_tokens: int | None = None, now: float | None = None) -> RouteResult:
        now = now or time.time()
        cid = self.conversation_id(messages, system, tools)
        with self._lock:
            conv = self.conversations.setdefault(cid, Conversation())
        prompt_tokens = self.estimator.estimate(estimate_tokens(messages, system, tools))
        continuation = is_tool_continuation(messages)
        ctx = self.context()

        if continuation and conv.current and conv.current in ctx.catalog:
            errors = recent_tool_errors(messages)
            base = TurnRequest(category="agentic", difficulty=conv.floor, prompt_tokens=prompt_tokens,
                               output_tokens=max_tokens or 2000, now=now, needs_tools=True)
            if errors >= self.escalate_after_tool_errors:
                retry = self.policy.on_failure(conv, base, ctx, conv.current, {conv.current})
                if retry:
                    return RouteResult(ctx.catalog[retry.model],
                                       f"{errors} failing tool results in a row: {retry.reason}",
                                       cid, False, base, None, now)
            return RouteResult(ctx.catalog[conv.current], "inside a tool loop: stay on the turn's model",
                               cid, False, base, None, now)

        text = last_user_text(messages)
        started = time.perf_counter()
        cls = self.classifier(text, self._summary(messages, tools)) if self.classifier else None
        cls_ms = (time.perf_counter() - started) * 1000
        req = self._turn_request(cls, prompt_tokens, max_tokens, now, bool(tools), messages)
        choice = self.policy.choose(conv, req, ctx)
        return RouteResult(ctx.catalog[choice.model], choice.reason, cid, True, req, cls, now, cls_ms)

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
        if cls is None or cls.failed:
            return TurnRequest(category="agentic" if has_tools else "general", difficulty=0.5,
                               prompt_tokens=prompt_tokens, output_tokens=max_tokens or 1500, now=now,
                               needs_tools=has_tools, stakes_usd=STAKES_USD[2], difficulty_confidence=0.0)
        stakes_idx = min(len(STAKES_USD) - 1, int(round(cls.stakes * (len(STAKES_USD) - 1))))
        agentic = has_tools and cls.needs_tools > 0.5
        return TurnRequest(
            category=cls.category if cls.category in self.success_categories() else "general",
            difficulty=cls.difficulty,
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

    def escalate(self, result: RouteResult) -> RouteResult | None:
        """Pick a retry model after a failure signal (upstream error, judge says inadequate)."""
        conv = self.conversations.setdefault(result.conversation_id, Conversation())
        ctx = self.context()
        tried = result.tried | {result.model.name}
        retry = self.policy.on_failure(conv, result.request, ctx, result.model.name, tried)
        if retry is None:
            return None
        return RouteResult(ctx.catalog[retry.model], retry.reason, result.conversation_id, result.turn_start,
                           result.request, result.classification, time.time(), 0.0, tried)

    @property
    def stats(self) -> dict:
        return {
            "policy": self.policy.name,
            "conversations": len(self.conversations),
            "switches": sum(c.switches for c in self.conversations.values()),
            "escalations": sum(c.escalations for c in self.conversations.values()),
            "quota": {k: v.__dict__ for k, v in self.quota().items()},
            "token_estimator": self.estimator.stats,
            "models": [m.name for m in self.config.catalog.all()],
        }
