"""IRP v0.3.0-draft suggest-only contract, independent of conversation state.

The cost/quality objective is a linear scalarisation of quality and request cost.
The coefficient of cost only increases with the preference: the two optimality
inequalities at any two preferences prove that the winner's cost cannot rise.
Predictions use the router's success curve and token appetite, not seller identity.
"""
from __future__ import annotations

import json
import math
import time
import uuid
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from .cache_index import estimate_tokens

NAMESPACE = "system1models.ai"
VERSION = "irp-0.3.0-linear-v1"
EFFORTS = {"none", "minimal", "low", "medium", "high", "xhigh", "max"}
PROBLEMS = {
    400: ("invalid-request", "Invalid request"),
    402: ("payment-required", "Payment required"),
    422: ("no-scorable-candidate", "No scorable candidate"),
    503: ("unavailable", "Temporarily unavailable"),
}


class Problem(Exception):
    def __init__(self, status: int, detail: str, retry_after: int | None = None):
        self.status, self.detail, self.retry_after = status, detail, retry_after
        super().__init__(detail)

    def response(self) -> JSONResponse:
        slug, title = PROBLEMS[self.status]
        headers = {"Retry-After": str(self.retry_after or 30)} if self.status == 503 else {}
        return JSONResponse({"type": f"urn:irp:problem:{slug}", "title": title,
                             "status": self.status, "detail": self.detail},
                            status_code=self.status, media_type="application/problem+json", headers=headers)


def obj(value: Any, name: str) -> dict:
    if not isinstance(value, dict):
        raise Problem(400, f"{name} must be an object.")
    return value


def integer(value: Any, name: str, low: int, high: int | None = 2**53-1) -> int:
    if type(value) is not int or value < low or (high is not None and value > high):
        raise Problem(400, f"{name} must be an integer in the allowed range.")
    return value


def number(value: Any, name: str) -> float:
    try:
        valid = type(value) in (int, float) and math.isfinite(value) and value >= 0
    except OverflowError:
        valid = False
    if not valid:
        raise Problem(400, f"{name} must be a finite nonnegative number.")
    return value


def validate_request(value: Any) -> dict:
    request = obj(value, "request")
    messages = request.get("messages")
    if not isinstance(messages, list) or not messages:
        raise Problem(400, "request.messages must be a nonempty array.")
    for message in messages:
        message = obj(message, "message")
        role = message.get("role")
        if role not in ("system", "developer", "user", "assistant", "tool", "function"):
            raise Problem(400, "Each message needs a valid Chat Completions role.")
        content = message.get("content")
        if not isinstance(content, (str, list)):
            if role != "assistant" or not (message.get("tool_calls") or message.get("function_call")):
                raise Problem(400, "Message content must be text or a content-part array.")
        if isinstance(content, list):
            for part in content:
                part = obj(part, "content part")
                kind = part.get("type")
                if not isinstance(kind, str) or not kind:
                    raise Problem(400, "Content parts need a type.")
                if kind == "text" and not isinstance(part.get("text"), str):
                    raise Problem(400, "Text content parts need text.")
        if role == "tool" and not isinstance(message.get("tool_call_id"), str):
            raise Problem(400, "Tool messages need tool_call_id.")
    for field in ("max_tokens", "max_completion_tokens", "n"):
        if field in request:
            integer(request[field], f"request.{field}", 1)
    if "tools" in request:
        if not isinstance(request["tools"], list):
            raise Problem(400, "request.tools must be an array.")
        for tool in request["tools"]:
            obj(tool, "tool")
    # model and stream are deliberately never inspected (SPEC §5.1).
    return request


def validate(body: Any) -> tuple[dict, int, list[dict]]:
    body = obj(body, "body")
    request = validate_request(body.get("request"))
    routing = obj(body.get("routing"), "routing")
    tradeoff = integer(routing.get("cost_quality_tradeoff", 5), "cost_quality_tradeoff", 0, 10)
    candidates = routing.get("candidates")
    if not isinstance(candidates, list) or not 1 <= len(candidates) <= 512:
        raise Problem(400, "routing.candidates must contain 1–512 candidates.")
    seen = set()
    clean = []
    for candidate in candidates:
        candidate = obj(candidate, "candidate")
        cid = candidate.get("id")
        if isinstance(cid, str):
            try:
                cid.encode("utf-8")
            except UnicodeEncodeError:
                raise Problem(400, "Candidate ids must be valid UTF-8 strings.")
        if not isinstance(cid, str) or not 1 <= len(cid) <= 128 or cid in seen:
            raise Problem(400, "Candidate ids must be unique strings of 1–128 characters.")
        seen.add(cid)
        model = candidate.get("model")
        if not isinstance(model, str) or not model:
            raise Problem(400, "Candidate model must be a nonempty string.")
        try:
            model.encode("utf-8")
        except UnicodeEncodeError:
            raise Problem(400, "Candidate models must be valid UTF-8 strings.")
        pricing = obj(candidate.get("pricing"), "pricing")
        pricing = {k: number(pricing.get(k), f"pricing.{k}") for k in ("input", "cache_read", "output")}
        usage = obj(candidate.get("expected_usage", {}), "expected_usage")
        cache = integer(usage.get("cache_read_tokens", 0), "cache_read_tokens", 0)
        clean.append({"id": cid, "model": model, "pricing": pricing, "cache_read_tokens": cache})
    return request, tradeoff, clean


def model_id(model, raw: dict | None = None) -> str:
    """Explicit cross-vendor id wins; provider-local bare names get stable ids."""
    explicit = (raw or {}).get("irp_model_id")
    if explicit:
        return explicit
    upstream = model.upstream_id
    return upstream if "/" in upstream else f"{NAMESPACE}/{model.name}"


def model_index(config) -> dict:
    entries = {e["name"]: e for e in config.raw.get("models", [])}
    result = {}
    for model in sorted(config.catalog.all(), key=lambda m: m.name):
        raw = entries.get(model.name, {})
        result.setdefault(model_id(model, raw), (model, raw))
    return result


def supported(config) -> dict:
    return {"object": "list", "data": [{"id": mid, "object": "model"} for mid in model_index(config)],
            "extra": {NAMESPACE: {"mode": "suggest-only", "protocol": "0.3.0-draft"}}}


def rank(router, body: Any) -> dict:
    request, tradeoff, candidates = validate(body)
    index = model_index(router.config)
    messages = request["messages"]
    # Include every supplied message, tool schema and modality in usage estimates.
    # A fixed estimator keeps these predictions independent of live proxy state.
    prompt = estimate_tokens(messages, None, request.get("tools"))
    from .router import last_user_text
    cls = router.classifier(last_user_text(messages), router._summary(messages, request.get("tools"))) if router.classifier else None
    budget = request.get("max_completion_tokens", request.get("max_tokens"))
    req = router._turn_request(cls, prompt, budget, time.time(), bool(request.get("tools")), messages)
    ranked = []
    for candidate in candidates:
        found = index.get(candidate["model"])
        if found is None:
            continue
        model, raw = found
        # Never replace caller prices with our provider's prices or merge sellers.
        cache = candidate["cache_read_tokens"]
        input_tokens = max(prompt, cache)  # known cache usage is a lower bound on the approximate prompt count
        output = max(0, int(req.output_tokens * model.output_appetite))
        output = min(output, model.max_output_tokens, budget if budget is not None else output)
        usage = {"input_tokens": input_tokens, "cache_read_tokens": cache, "output_tokens": output}
        prices = candidate["pricing"]
        output *= request.get("n", 1)
        usage["output_tokens"] = output
        try:
            cost = ((input_tokens - cache) * prices["input"] + cache * prices["cache_read"] + output * prices["output"]) / 1e6
        except OverflowError:
            raise Problem(400, "Candidate prices and usage exceed the supported numeric range.")
        if not math.isfinite(cost):
            raise Problem(400, "Candidate prices and usage exceed the supported numeric range.")
        quality = router.success.p(model, req.category, req.difficulty)
        entry = {"candidate_id": candidate["id"], "expected_quality": max(0.0, min(1.0, quality)),
                 "expected_cost_usd": cost, "expected_usage": usage}
        # Only explicitly documented support; never infer effort support from a model name.
        # Effort-specific prediction data is not available in this release.
        # Omit reasoning_effort instead of making an unsupported suggestion.
        ranked.append(entry)
    if not ranked:
        raise Problem(422, "None of the candidates use a model this router can score.")
    scale = max(e["expected_cost_usd"] for e in ranked) or 1.0
    t = tradeoff / 10
    ranked.sort(key=lambda e: (-(1 - t) * e["expected_quality"] + t * e["expected_cost_usd"] / scale,
                               e["expected_cost_usd"], -e["expected_quality"], e["candidate_id"]))
    return {"id": "rank_" + uuid.uuid4().hex, "object": "routing.ranking", "created": int(time.time()),
            "router": {"id": NAMESPACE + "/auto-model-router", "version": VERSION}, "ranked": ranked,
            "extra": {NAMESPACE: {"quality_basis": "configured success model; advisory predictions",
                                    "usage_basis": "character token estimate and configured output appetite",
                                    "classifier": "configured" if cls is not None else "local conservative prior",
                                    "router_fee_usd": 0, "mode": "suggest-only"}}}


def endpoints(get_router, guard=None) -> APIRouter:
    """Hosts retain their own auth/payment/limit policy through an optional guard.

    The guard runs before any classification, accepts Request, and raises Problem
    for existing payment (402) or temporary availability/limits (503) decisions.
    """
    routes = APIRouter()

    @routes.get("/v1/routing/models")
    async def models():
        try:
            router = get_router()
            return supported(router.config)
        except Problem as exc:
            return exc.response()
        except Exception:
            return Problem(503, "Model catalog is temporarily unavailable.", 30).response()

    @routes.post("/v1/routing/rank")
    async def ranking(request: Request):
        try:
            if guard:
                await guard(request)
            try:
                # Read incrementally so the limit also covers chunked uploads.
                chunks = bytearray()
                async for chunk in request.stream():
                    chunks.extend(chunk)
                    if len(chunks) > 2_000_000:
                        raise Problem(400, "Routing body exceeds the 2 MB service limit.")
                body = json.loads(chunks)
            except (ValueError, UnicodeDecodeError, RecursionError):
                raise Problem(400, "Body must be valid JSON.")
            validate(body)  # malformed requests never consume a classifier call
            import asyncio
            return JSONResponse(await asyncio.to_thread(rank, get_router(), body))
        except Problem as exc:
            return exc.response()
        except (ValueError, OverflowError, RecursionError):
            return Problem(400, "Routing input exceeds the supported numeric or nesting limits.").response()
        except Exception:
            # No provider URLs, credentials or prompt fragments in public errors.
            return Problem(503, "Ranking is temporarily unavailable.", 30).response()

    return routes
