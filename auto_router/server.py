"""FastAPI app.

Endpoints
---------
POST /v1/chat/completions   OpenAI-compatible, routed
POST /v1/messages           Anthropic-compatible, routed; subscription passthrough (see shim.py)
GET  /v1/models             configured models with prices and capability
GET  /v1/router/metrics     routing, cost and quota statistics
GET  /v1/router/decisions   recent decisions: classification, selection, estimate, observation
GET  /health
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from .config import Provider, RouterConfig, load_config
from .decision import ObservedOutcome
from .metrics import metrics
from .pricing import Usage, cost_usd, parse_openai_usage
from .router import RouteResult, Router
from .stream_translate import StreamOutcome, translate_stream
from .translate import (
    TranslationError,
    anthropic_error_sse,
    messages_to_openai,
    openai_response_to_anthropic,
    tool_choice_to_openai,
    tools_to_openai,
)
from .truncation import label_for_stop_reason, truncation_label

log = logging.getLogger("auto_router.server")

app = FastAPI(title="auto-model-router", version="0.2.0")

config: RouterConfig = load_config()
router = Router(config)
_client: httpx.AsyncClient | None = None
TIMEOUT = float(os.environ.get("AUTO_ROUTER_TIMEOUT_S", "600"))
MAX_ATTEMPTS = int(os.environ.get("AUTO_ROUTER_MAX_ATTEMPTS", "3"))


def truncation_retry_budget() -> int:
    """How many extra routes one unfinished answer may cost.

    Deliberately smaller than ``MAX_ATTEMPTS``, and deliberately read per
    request rather than at import. A length stop is a genuine failure, but
    unlike a 5xx it still returns a usable partial answer and it still bills
    for the tokens, so a caller who asked for a very small ``max_tokens`` must
    not have their bill multiplied by the retry budget. One extra route is
    what the observed case needed; ``0`` turns the retry off and leaves only
    the honest label behind.
    """
    try:
        return max(0, int(os.environ.get("AUTO_ROUTER_TRUNCATION_RETRIES", "1")))
    except ValueError:
        return 1


def client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(timeout=TIMEOUT)
    return _client


def provider_for(result: RouteResult) -> Provider:
    provider = config.providers.get(result.model.provider)
    if provider is None:
        raise HTTPException(500, f"model {result.model.name} has no configured provider")
    return provider


def provider_headers(provider: Provider) -> dict[str, str]:
    headers = {"Content-Type": "application/json", **provider.extra_headers}
    key = provider.api_key
    if key:
        headers["Authorization"] = f"Bearer {key}"
    return headers


@app.on_event("shutdown")
async def _shutdown() -> None:
    if _client is not None:
        await _client.aclose()


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "models": len(config.catalog.all()), "policy": router.policy.name}


@app.get("/v1/models")
async def list_models() -> dict:
    return {"object": "list", "data": [
        {"id": m.name, "object": "model", "owned_by": m.provider,
         "input_usd_per_mtok": m.prices.input, "output_usd_per_mtok": m.prices.output,
         "cache_read_usd_per_mtok": m.prices.cache_read, "cache_write_usd_per_mtok": m.prices.cache_write,
         "cache_ttl_seconds": m.cache.ttl_seconds, "capability": m.capability,
         "capability_basis": m.capability_basis, "capability_strength": m.capability_strength,
         "capability_source": m.capability_source, "evidence_stale": m.evidence_stale,
         "evidence": m.evidence,
         "benchmaxxing": m.benchmaxxing, "subscription": m.subscription}
        for m in config.catalog.all()]}


@app.get("/v1/router/metrics")
async def router_metrics() -> dict:
    return {**metrics.to_dict(), "router": router.stats}


@app.get("/v1/router/decisions")
async def router_decisions(limit: int = 20) -> dict:
    """Recent routing decisions, newest first.

    Each record keeps the classification, the route selection, the cache
    decision, the estimate and the observed outcome in separate objects, and
    contains no prompt or response text. See ``auto_router/decision.py``.
    """
    limit = max(1, min(int(limit), router.decisions.maxlen or 200))
    recent = list(router.decisions)[-limit:]
    return {"object": "list", "count": len(recent),
            "ledger": router.ledger.stats,
            "data": [d.to_dict() for d in reversed(recent)]}


async def _route(messages, system, tools, max_tokens) -> RouteResult:
    return await asyncio.to_thread(router.route, messages, system, tools, max_tokens)


# --------------------------------------------------------------------------
# OpenAI-compatible
# --------------------------------------------------------------------------
@app.post("/v1/chat/completions")
async def chat_completions(request: Request) -> Any:
    body = await request.json()
    messages = body.get("messages") or []
    if not messages:
        raise HTTPException(400, "messages is required")
    system = next((m.get("content") for m in messages if m.get("role") == "system"), None)
    convo = [m for m in messages if m.get("role") != "system"]
    result = await _route(convo, system, body.get("tools"), body.get("max_tokens"))

    if body.get("stream"):
        return StreamingResponse(_openai_stream(body, result), media_type="text/event-stream",
                                 headers=result.headers)

    last_error: tuple[int, Any] = (502, {"error": "no attempt made"})
    #: The first answer a provider told us it never finished, kept so that a
    #: turn where every route runs out of budget still returns what it did
    #: produce - exactly what the client got before truncation was noticed.
    unfinished: tuple[Any, RouteResult] | None = None
    truncation_retries = truncation_retry_budget()
    first = result
    for attempt in range(1, MAX_ATTEMPTS + 1):
        provider = provider_for(result)
        payload = {**body, "model": result.model.upstream_id}
        started = time.perf_counter()
        try:
            resp = await client().post(f"{provider.base_url}/chat/completions",
                                       headers=provider_headers(provider), json=payload)
            data = resp.json() if resp.content else {}
        except (httpx.HTTPError, json.JSONDecodeError) as exc:
            resp, data = None, {"error": type(exc).__name__}
        latency_ms = (time.perf_counter() - started) * 1000
        if resp is not None and resp.status_code == 200 and (data.get("choices") or []):
            usage = parse_openai_usage(data.get("usage") or {})
            # The provider's own machine-readable stop flag, never the prose.
            cut = truncation_label(data)
            # The tokens were really spent either way, so the call is committed
            # and metered either way: dropping a truncated attempt from the
            # books would under-report what the turn actually cost.
            router.commit(result, usage.total_input or None, usage.output)
            record_observed(result, "truncated" if cut else "ok", 200, latency_ms, usage,
                            attempt, first, error=cut)
            metrics.record(category=result.request.category, model=result.model,
                           classification_ms=result.classification_ms, usage=usage,
                           switched=False, escalated=bool(result.tried))
            if cut is None:
                return JSONResponse(data, headers=result.headers)
            # An answer the route says it never finished is a failed attempt,
            # not an answer. It says nothing about the model being too weak -
            # the measured case had the *higher*-rated route run out of budget -
            # so this takes the same sideways safe fallback a 5xx takes, and no
            # capability score is touched anywhere.
            metrics.record_error(result.model.name)
            if unfinished is None:
                unfinished = (data, result)
            if truncation_retries <= 0:
                break
            truncation_retries -= 1
            retry = router.escalate(result, availability=True)
            if retry is None:
                break
            result = retry
            continue
        metrics.record_error(result.model.name)
        status = resp.status_code if resp is not None else 502
        # Only a controlled label is kept. An upstream error body routinely
        # echoes part of the prompt, and some providers put the rejected API
        # key in the message, so no portion of it may reach the ledger.
        record_observed(result, "upstream_error" if resp is not None else "transport_error",
                        status, latency_ms, None, attempt, first,
                        error=error_label(resp, data))
        last_error = (status, data)
        # A 5xx or a broken connection says the route is unavailable, not that
        # the model was too weak, so the fallback is allowed to be sideways.
        retry = router.escalate(result, availability=resp is None or status >= 500)
        if retry is None:
            break
        result = retry
    if unfinished is not None:
        data, result = unfinished
        return JSONResponse(data, headers=result.headers)
    return JSONResponse(last_error[1], status_code=last_error[0], headers=result.headers)


def error_label(resp, data) -> str:
    """A short, controlled description of a failure.

    Deliberately derived from the HTTP status and, for a transport failure, the
    exception class name that ``chat_completions`` already put in ``data``.
    Never the upstream message: provider errors echo prompt fragments and
    sometimes the rejected credential itself.
    """
    if resp is None:
        kind = data.get("error") if isinstance(data, dict) else None
        return f"transport:{kind}" if isinstance(kind, str) and kind.isidentifier() else "transport"
    return f"http_{resp.status_code}"


def record_observed(result: RouteResult, status: str, http_status: int | None,
                    latency_ms: float | None, usage: Usage | None, attempt: int,
                    first: RouteResult, error: str | None = None) -> None:
    """Attach what actually happened to the routing decision record.

    Cost is filled in only when the route publishes prices; a free or
    subscription route reports ``None`` with the reason, never a zero that
    would later read as a measured saving. ``truncated`` is a 200 whose tokens
    are real and whose answer is not: the cost is measured as usual.
    """
    model = result.model
    if usage is None:
        cost, basis = None, f"no usage reported ({status})"
    elif model.subscription:
        cost, basis = None, f"subscription route {model.subscription}: no marginal cash cost"
    elif model.prices.is_free:
        cost, basis = None, "route configured as free: no cash cost to measure"
    else:
        cost, basis = cost_usd(model, usage), "provider-reported tokens x configured list prices"
    router.observe(result, ObservedOutcome(
        model=model.name, status=status, http_status=http_status, latency_ms=latency_ms,
        uncached_input_tokens=usage.uncached_input if usage else None,
        cached_read_tokens=usage.cached_read if usage else None,
        cache_write_tokens=usage.cache_write if usage else None,
        output_tokens=usage.output if usage else None,
        cost_usd=cost, cost_basis=basis, attempts=attempt,
        escalated_from=first.model.name if first.model.name != model.name else None,
        error=error))


async def _openai_stream(body: dict, result: RouteResult):
    provider = provider_for(result)
    payload = {**body, "model": result.model.upstream_id,
               "stream_options": {**(body.get("stream_options") or {}), "include_usage": True}}
    usage = Usage()
    finish_reason: str | None = None
    started = time.perf_counter()
    async with client().stream("POST", f"{provider.base_url}/chat/completions",
                               headers=provider_headers(provider), json=payload) as resp:
        if resp.status_code != 200:
            await resp.aread()
            metrics.record_error(result.model.name)
            record_observed(result, "upstream_error", resp.status_code,
                            (time.perf_counter() - started) * 1000, None, 1, result,
                            error=f"http_{resp.status_code}")
            yield f"data: {json.dumps({'error': {'message': 'upstream failed', 'code': resp.status_code}})}\n\n"
            return
        async for line in resp.aiter_lines():
            if line.startswith("data:"):
                chunk = line[5:].strip()
                if chunk and chunk != "[DONE]":
                    try:
                        parsed = json.loads(chunk)
                        if parsed.get("usage"):
                            usage = parse_openai_usage(parsed["usage"])
                        for choice in parsed.get("choices") or []:
                            # With several choices, a length stop on any of them
                            # is the honest verdict for the turn.
                            reason = choice.get("finish_reason") if isinstance(choice, dict) else None
                            if reason and (finish_reason is None
                                           or label_for_stop_reason(finish_reason) is None):
                                finish_reason = reason
                    except json.JSONDecodeError:
                        pass
            yield line + "\n"
    # The bytes are already on the wire, so a stream can only be recorded
    # honestly, never retried. See ``truncation.py``.
    cut = label_for_stop_reason(finish_reason)
    router.commit(result, usage.total_input or None, usage.output)
    record_observed(result, "truncated" if cut else "ok", 200,
                    (time.perf_counter() - started) * 1000, usage, 1, result, error=cut)
    if cut:
        metrics.record_error(result.model.name)
    metrics.record(category=result.request.category, model=result.model,
                   classification_ms=result.classification_ms, usage=usage)


# --------------------------------------------------------------------------
# Anthropic-compatible
# --------------------------------------------------------------------------
@app.post("/v1/messages")
async def anthropic_messages(request: Request) -> Any:
    from . import shim  # local import: shim depends on this module
    raw = await request.body()
    try:
        body = json.loads(raw)
    except json.JSONDecodeError:
        return JSONResponse({"type": "error", "error": {"type": "invalid_request_error",
                                                        "message": "body is not valid JSON"}}, status_code=400)
    messages = body.get("messages") or []
    if not messages:
        raise HTTPException(400, "messages is required")
    result = await _route(messages, body.get("system"), body.get("tools"), body.get("max_tokens"))
    if result.model.subscription == "claude":
        return await shim.subscription_passthrough(request, raw, body, result)
    return await proxy_openai_as_anthropic(body, result)


async def proxy_openai_as_anthropic(body: dict, result: RouteResult) -> Any:
    headers = result.headers
    try:
        openai_messages = messages_to_openai(body.get("messages") or [], body.get("system"))
        openai_tools = tools_to_openai(body.get("tools"))
        choice = tool_choice_to_openai(body.get("tool_choice"))
    except TranslationError as exc:
        detail = f"request translation failed: {exc}"
        return JSONResponse({"type": "error", "error": {"type": "invalid_request_error", "message": detail}},
                            status_code=400, headers=headers)

    provider = provider_for(result)
    payload: dict[str, Any] = {
        "model": result.model.upstream_id,
        "messages": openai_messages,
        "max_tokens": min(body.get("max_tokens") or 4096, result.model.max_output_tokens),
    }
    if body.get("temperature") is not None:
        payload["temperature"] = body["temperature"]
    if body.get("stop_sequences"):
        payload["stop"] = body["stop_sequences"]
    if openai_tools:
        payload["tools"] = openai_tools
    if choice is not None:
        payload["tool_choice"] = choice
    label = body.get("model") or result.model.name

    if body.get("stream"):
        payload["stream"] = True
        payload["stream_options"] = {"include_usage": True}
        return StreamingResponse(_anthropic_stream(payload, provider, result, label),
                                 media_type="text/event-stream", headers=headers)

    started = time.perf_counter()
    resp = await client().post(f"{provider.base_url}/chat/completions",
                               headers=provider_headers(provider), json=payload)
    latency_ms = (time.perf_counter() - started) * 1000
    if resp.status_code != 200:
        metrics.record_error(result.model.name)
        record_observed(result, "upstream_error", resp.status_code, latency_ms, None, 1, result,
                        error=f"http_{resp.status_code}")
        return JSONResponse({"type": "error", "error": {"type": "api_error",
                                                        "message": f"upstream returned {resp.status_code}"}},
                            status_code=resp.status_code, headers=headers)
    data = resp.json()
    try:
        message = openai_response_to_anthropic(data, label)
    except TranslationError as exc:
        metrics.record_error(result.model.name)
        return JSONResponse({"type": "error", "error": {"type": "api_error",
                                                        "message": f"response translation failed: {exc}"}},
                            status_code=502, headers=headers)
    usage = parse_openai_usage(data.get("usage") or {})
    cut = truncation_label(data)
    router.commit(result, usage.total_input or None, usage.output)
    record_observed(result, "truncated" if cut else "ok", 200, latency_ms, usage, 1, result,
                    error=cut)
    if cut:
        # This surface has no attempt loop to fall back through; the record is
        # still honest about what the route did.
        metrics.record_error(result.model.name)
    metrics.record(category=result.request.category, model=result.model,
                   classification_ms=result.classification_ms, usage=usage)
    return JSONResponse(message, headers=headers)


async def _anthropic_stream(payload: dict, provider: Provider, result: RouteResult, label: str):
    outcome = StreamOutcome()
    started = time.perf_counter()
    try:
        async with client().stream("POST", f"{provider.base_url}/chat/completions",
                                   headers=provider_headers(provider), json=payload) as resp:
            if resp.status_code != 200:
                raw = (await resp.aread()).decode(errors="replace")[:600]
                metrics.record_error(result.model.name)
                record_observed(result, "upstream_error", resp.status_code,
                                (time.perf_counter() - started) * 1000, None, 1, result,
                                error=f"http_{resp.status_code}")
                yield anthropic_error_sse(f"upstream returned {resp.status_code}: {raw}")
                return
            async for event in translate_stream(resp.aiter_lines(), label, outcome):
                yield event
    except TranslationError as exc:
        metrics.record_error(result.model.name)
        record_observed(result, "transport_error", None,
                        (time.perf_counter() - started) * 1000, None, 1, result,
                        error="TranslationError")
        yield anthropic_error_sse(f"stream translation failed: {exc}")
        return
    except Exception as exc:  # noqa: BLE001 - surfaced to the client, not swallowed
        metrics.record_error(result.model.name)
        record_observed(result, "transport_error", None,
                        (time.perf_counter() - started) * 1000, None, 1, result,
                        error=f"transport:{type(exc).__name__}")
        yield anthropic_error_sse(f"{type(exc).__name__}: {exc}")
        return
    # The provider's own word, not the Anthropic stop reason it was mapped to:
    # a stream with tool calls maps to "tool_use" even when the budget ran out.
    cut = label_for_stop_reason(outcome.finish_reason)
    router.commit(result, outcome.usage.total_input or None, outcome.usage.output)
    record_observed(result, "truncated" if cut else "ok", 200,
                    (time.perf_counter() - started) * 1000, outcome.usage, 1, result, error=cut)
    if cut:
        metrics.record_error(result.model.name)
    metrics.record(category=result.request.category, model=result.model,
                   classification_ms=result.classification_ms, usage=outcome.usage)


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"])
async def catch_all(path: str, request: Request) -> Response:
    """Claude Code also calls count_tokens and telemetry endpoints; pass them to Anthropic."""
    from . import shim
    body = await request.body()
    return await shim.passthrough(request, body, f"/{path}")
