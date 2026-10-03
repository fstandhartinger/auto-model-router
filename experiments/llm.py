"""Minimal OpenAI-compatible client for experiments: usage, cache fields, cost, latency.

Every *attempt* is appended to a JSONL ledger so spend can be audited and capped.
A call makes up to ``retries + 1`` attempts and each one is a row, with the same
``tag``, its 0-based ``attempt`` and ``final`` set on the last. A metered attempt
that cannot be accounted for (including response-processing errors or unusable
usage) may still have been billed upstream, so its row reserves the worst case
in ``list_cost_usd``. Failed responses retain usable reported billing as well. ``spent()`` counts the higher of
the two per row, and the budget is re-checked before every attempt.

For analysis: billed spend per tag is the sum of ``cost_usd`` over all its attempt
rows; the ``final`` row is the one the caller saw. A reservation row has
``cost_basis == RESERVATION_BASIS`` and ``cost_usd == 0``, so it never enters a
billed cash figure; it is resolved against provider receipts.
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import httpx

from auto_router.cache_index import estimate_tokens
from auto_router.catalog import ModelInfo
from auto_router.config import Provider, RouterConfig
from auto_router.pricing import parse_openai_usage

_ledger_lock = threading.Lock()

#: Measured billed/list ratio on the metered route (round 2: $0.832962 billed
#: against $0.416481 list over 70 calls). An unpriced attempt is reserved at
#: list price times this.
BILLED_TO_LIST_RATIO = 2.0
RESERVATION_BASIS = "unpriced attempt: reserved at worst case"


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class CallResult:
    ok: bool
    content: str = ""
    tool_calls: list[dict] = field(default_factory=list)
    finish_reason: str | None = None
    prompt_tokens: int = 0
    cached_tokens: int = 0
    cache_write_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    cost_usd: float = 0.0          # money actually charged (0 on free routes)
    list_cost_usd: float = 0.0     # what the call costs at the model's list price
    #: Where cost_usd came from. Never let a reported zero pass as a measurement
    #: without saying which zero it is.
    cost_basis: str = "not measured"
    routed_model: str | None = None
    latency_s: float = 0.0
    error: str | None = None
    raw_message: dict = field(default_factory=dict)


def _billed_cost(provider, usage_raw: dict, usage, model: ModelInfo,
                 list_cost: float) -> tuple[float, str]:
    """What this call actually cost, and on what basis.

    A gateway can report ``cost: 0`` while the money is charged somewhere else.
    OpenRouter does exactly that for a bring-your-own-key route: the top-level
    ``cost`` is 0 and the real figure sits in
    ``cost_details.upstream_inference_cost``. Taking the zero at face value
    printed "$0.0000" for an arm that had just spent sixteen cents - the exact
    mistake this project refuses to make elsewhere, made here.
    """
    details = usage_raw.get("cost_details") or {}
    upstream = details.get("upstream_inference_cost")
    if isinstance(upstream, (int, float)) and upstream > 0:
        return float(upstream), "gateway-reported upstream inference cost"
    reported = usage_raw.get("cost")
    if isinstance(reported, (int, float)) and reported > 0:
        return float(reported), "gateway-reported cost"
    if model.prices.is_free:
        return 0.0, "route configured as free"
    if isinstance(reported, (int, float)) and reported == 0 and usage_raw.get("is_byok"):
        # The gateway billed nothing because the key is the caller's own; the
        # provider still charged. List price is the honest stand-in, named.
        return list_cost, "gateway billed 0 (own key); priced at list instead"
    return list_cost, "priced at list (no billed figure reported)"


class Client:
    def __init__(self, config: RouterConfig, ledger: str | Path, budget_usd: float = 30.0,
                 list_prices: dict[str, ModelInfo] | None = None):
        self.config = config
        self.ledger = Path(ledger)
        self.ledger.parent.mkdir(parents=True, exist_ok=True)
        self.budget = budget_usd
        self.list_prices = list_prices or {}
        self.http = httpx.Client(timeout=httpx.Timeout(900.0, connect=30.0))
        #: when calling a router, price each call by the model it reports in X-Router-Model
        self.router_catalog = None
        #: per-model request extras from the config (e.g. a thinking budget)
        self.extras = {m["name"]: m["request_extra"] for m in (config.raw.get("models") or [])
                       if isinstance(m.get("request_extra"), dict)}

    def spent(self) -> float:
        if not self.ledger.exists():
            return 0.0
        total = 0.0
        for line in self.ledger.read_text().splitlines():
            try:
                row = json.loads(line)
                total += max(row.get("cost_usd", 0.0), row.get("list_cost_usd", 0.0))  # conservative
            except json.JSONDecodeError:
                pass
        return total

    def chat(self, model: ModelInfo, messages: list[dict], *, tools: list[dict] | None = None,
             max_tokens: int = 16000, tag: str = "", extra: dict | None = None,
             retries: int = 2) -> CallResult:
        provider: Provider = self.config.providers[model.provider]
        payload: dict[str, Any] = {"model": model.upstream_id, "messages": messages, "max_tokens": max_tokens}
        if tools:
            payload["tools"] = tools
        if provider.name == "openrouter":
            payload["usage"] = {"include": True}
        payload.update(self.extras.get(model.name) or {})
        payload.update(extra or {})
        headers = {"Content-Type": "application/json", **provider.extra_headers}
        if provider.api_key:
            headers["Authorization"] = f"Bearer {provider.api_key}"
        submitted_messages = payload.get("messages", messages)
        submitted_tools = payload.get("tools", tools)
        submitted_max_tokens = int(payload.get("max_tokens") or max_tokens)
        reserve = self._reservation(model, submitted_messages, submitted_tools, submitted_max_tokens)

        result = CallResult(ok=False)
        for attempt in range(retries + 1):
            # Before every attempt, not once per call: a retry can be billed too.
            if self.spent() >= self.budget:
                raise BudgetExceeded(f"spend {self.spent():.2f} >= budget {self.budget:.2f}")
            started = time.time()
            retry = False
            attempt_model = model
            attempt_reserve = reserve
            routed = None
            try:
                resp = self.http.post(f"{provider.base_url}/chat/completions", headers=headers, json=payload)
                latency = time.time() - started
                # Resolve the actual route before parsing either success or error bodies.
                routed = resp.headers.get("x-router-model")
                if routed and self.router_catalog is not None and routed in self.router_catalog:
                    attempt_model = self.router_catalog[routed]
                    attempt_reserve = self._reservation(
                        attempt_model, submitted_messages, submitted_tools, submitted_max_tokens)
                data = resp.json()
                failed = resp.status_code != 200 or not data.get("choices")
                retry = failed and resp.status_code in (429, 500, 502, 503, 504, 520, 524) and attempt < retries
                usage_raw = data.get("usage") or {}
                usage = parse_openai_usage(usage_raw)
                details = usage_raw.get("completion_tokens_details") or {}
                ref = self.list_prices.get(attempt_model.name, attempt_model)
                list_cost = (usage.uncached_input * ref.prices.input + usage.cached_read * ref.prices.read
                             + usage.cache_write * ref.prices.write + usage.output * ref.prices.output) / 1e6
                cost, cost_basis = _billed_cost(provider, usage_raw, usage, attempt_model, list_cost)
                result = CallResult(
                    ok=not failed, cost_basis=cost_basis, prompt_tokens=usage.total_input,
                    cached_tokens=usage.cached_read, cache_write_tokens=usage.cache_write,
                    output_tokens=usage.output, reasoning_tokens=int(details.get("reasoning_tokens") or 0),
                    cost_usd=cost, list_cost_usd=list_cost, latency_s=latency, routed_model=routed,
                )
                if failed:
                    err = json.dumps(data.get("error") or data)[:300]
                    result.error = f"HTTP {resp.status_code}: {err}"
                    # Keep reported billing, while reserving at least the worst case.
                    result = self._reserved(result, attempt_reserve)
                else:
                    choice = data["choices"][0]
                    message = choice.get("message") or {}
                    content = message.get("content") or ""
                    if isinstance(content, list):
                        content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
                    result.content = content
                    result.tool_calls = message.get("tool_calls") or []
                    result.finish_reason = choice.get("finish_reason")
                    result.raw_message = {k: v for k, v in message.items() if k in ("role", "content", "tool_calls")}
                    if not (cost > 0 or list_cost > 0):
                        # Nonempty usage is insufficient without usable accounting.
                        result = self._reserved(result, attempt_reserve)
            except (httpx.HTTPError, AttributeError, ValueError, TypeError, KeyError, IndexError) as exc:
                # A completed post may already be billed even if its body cannot be processed.
                result = self._reserved(CallResult(
                    ok=False, error=f"{type(exc).__name__}", latency_s=time.time() - started,
                    routed_model=routed), attempt_reserve)
                retry = attempt < retries
            # Exactly one append per dispatched attempt, including processing failures.
            self._log(attempt_model, tag, result, attempt=attempt, final=not retry)
            if not retry:
                break
            time.sleep((15 if result.error and result.error.startswith("HTTP ") else 5) * (attempt + 1))
        return result

    def _reservation(self, model: ModelInfo, messages: list[dict], tools: list[dict] | None,
                     max_tokens: int) -> float | None:
        """Worst-case list-cost reservation for an unpriced attempt; None on a free route."""
        if model.prices.is_free:
            return None
        ref = self.list_prices.get(model.name, model)
        prompt_estimate = estimate_tokens(messages, None, tools)
        return ((prompt_estimate * ref.prices.input + max_tokens * ref.prices.output)
                * BILLED_TO_LIST_RATIO / 1e6)

    @staticmethod
    def _reserved(result: CallResult, reserve: float | None) -> CallResult:
        """Retain reported accounting and reserve at least the worst case (0 on free routes)."""
        if reserve is None:
            return replace(result, cost_usd=0.0, list_cost_usd=0.0)
        return replace(result, list_cost_usd=max(result.list_cost_usd, reserve),
                       cost_basis=result.cost_basis if result.cost_usd > 0 else RESERVATION_BASIS)

    def _log(self, model: ModelInfo, tag: str, r: CallResult, *, attempt: int, final: bool) -> None:
        row = {"ts": time.time(), "model": model.name, "tag": tag, "attempt": attempt, "final": final,
               "ok": r.ok,
               "cost_basis": r.cost_basis, "prompt": r.prompt_tokens,
               "cached": r.cached_tokens, "cache_write": r.cache_write_tokens, "output": r.output_tokens,
               "reasoning": r.reasoning_tokens, "cost_usd": r.cost_usd, "list_cost_usd": r.list_cost_usd,
               "latency_s": round(r.latency_s, 2), "error": r.error}
        with _ledger_lock, self.ledger.open("a") as fh:
            fh.write(json.dumps(row) + "\n")
