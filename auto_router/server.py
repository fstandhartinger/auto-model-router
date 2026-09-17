"""FastAPI app.

Endpoints
---------
POST /v1/chat/completions   OpenAI-compatible, routed
POST /v1/messages           Anthropic-compatible, routed; subscription passthrough (see shim.py)
GET  /v1/models             configured models with prices and capability
GET  /v1/router/metrics     routing, cost and quota statistics
GET  /health
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from .config import Provider, RouterConfig, load_config
from .metrics import metrics
from .pricing import Usage, parse_openai_usage
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

log = logging.getLogger("auto_router.server")

app = FastAPI(title="auto-model-router", version="0.2.0")

config: RouterConfig = load_config()
router = Router(config)
_client: httpx.AsyncClient | None = None
TIMEOUT = float(os.environ.get("AUTO_ROUTER_TIMEOUT_S", "600"))
MAX_ATTEMPTS = int(os.environ.get("AUTO_ROUTER_MAX_ATTEMPTS", "3"))


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
         "benchmaxxing": m.benchmaxxing, "subscription": m.subscription}
        for m in config.catalog.all()]}


@app.get("/v1/router/metrics")
async def router_metrics() -> dict:
    return {**metrics.to_dict(), "router": router.stats}


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
    for _ in range(MAX_ATTEMPTS):
        provider = provider_for(result)
        payload = {**body, "model": result.model.upstream_id}
        try:
            resp = await client().post(f"{provider.base_url}/chat/completions",
                                       headers=provider_headers(provider), json=payload)
            data = resp.json() if resp.content else {}
        except (httpx.HTTPError, json.JSONDecodeError) as exc:
            resp, data = None, {"error": type(exc).__name__}
        if resp is not None and resp.status_code == 200 and (data.get("choices") or []):
            usage = parse_openai_usage(data.get("usage") or {})
            router.commit(result, usage.total_input or None, usage.output)
            metrics.record(category=result.request.category, model=result.model,
                           classification_ms=result.classification_ms, usage=usage,
                           switched=False, escalated=bool(result.tried))
            return JSONResponse(data, headers=result.headers)
        metrics.record_error(result.model.name)
        last_error = (resp.status_code if resp is not None else 502, data)
        retry = router.escalate(result)
        if retry is None:
            break
        result = retry
    return JSONResponse(last_error[1], status_code=last_error[0], headers=result.headers)


async def _openai_stream(body: dict, result: RouteResult):
    provider = provider_for(result)
    payload = {**body, "model": result.model.upstream_id,
               "stream_options": {**(body.get("stream_options") or {}), "include_usage": True}}
    usage = Usage()
    async with client().stream("POST", f"{provider.base_url}/chat/completions",
                               headers=provider_headers(provider), json=payload) as resp:
        if resp.status_code != 200:
            await resp.aread()
            metrics.record_error(result.model.name)
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
                    except json.JSONDecodeError:
                        pass
            yield line + "\n"
    router.commit(result, usage.total_input or None, usage.output)
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

    resp = await client().post(f"{provider.base_url}/chat/completions",
                               headers=provider_headers(provider), json=payload)
    if resp.status_code != 200:
        metrics.record_error(result.model.name)
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
    router.commit(result, usage.total_input or None, usage.output)
    metrics.record(category=result.request.category, model=result.model,
                   classification_ms=result.classification_ms, usage=usage)
    return JSONResponse(message, headers=headers)


async def _anthropic_stream(payload: dict, provider: Provider, result: RouteResult, label: str):
    outcome = StreamOutcome()
    try:
        async with client().stream("POST", f"{provider.base_url}/chat/completions",
                                   headers=provider_headers(provider), json=payload) as resp:
            if resp.status_code != 200:
                raw = (await resp.aread()).decode(errors="replace")[:600]
                metrics.record_error(result.model.name)
                yield anthropic_error_sse(f"upstream returned {resp.status_code}: {raw}")
                return
            async for event in translate_stream(resp.aiter_lines(), label, outcome):
                yield event
    except TranslationError as exc:
        metrics.record_error(result.model.name)
        yield anthropic_error_sse(f"stream translation failed: {exc}")
        return
    except Exception as exc:  # noqa: BLE001 - surfaced to the client, not swallowed
        metrics.record_error(result.model.name)
        yield anthropic_error_sse(f"{type(exc).__name__}: {exc}")
        return
    router.commit(result, outcome.usage.total_input or None, outcome.usage.output)
    metrics.record(category=result.request.category, model=result.model,
                   classification_ms=result.classification_ms, usage=outcome.usage)


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"])
async def catch_all(path: str, request: Request) -> Response:
    """Claude Code also calls count_tokens and telemetry endpoints; pass them to Anthropic."""
    from . import shim
    body = await request.body()
    return await shim.passthrough(request, body, f"/{path}")
