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
import uuid
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from .config import Provider, RouterConfig, for_http, load_config
from .decision import ObservedOutcome
from .metrics import metrics
from .outcome_memory import explicit_budget
from .policies import NoRouteAvailable
from .pricing import Usage, cost_usd, parse_openai_usage
from .router import RouteResult, Router
from .stream_translate import StreamOutcome, translate_stream
from .translate import (
    TranslationError,
    anthropic_error_sse,
    anthropic_sse_from_message,
    messages_to_openai,
    openai_response_to_anthropic,
    tool_choice_to_openai,
    tools_to_openai,
)
from .truncation import label_for_stop_reason, truncation_label

log = logging.getLogger("auto_router.server")

app = FastAPI(title="auto-model-router", version="0.2.0")

#: Routes only a launched client can reach are not endpoints, so the HTTP
#: surface never sees them. See ``config.for_http`` and ``launcher.py``.
config: RouterConfig = for_http(load_config())
router = Router(config)
_client: httpx.AsyncClient | None = None
TIMEOUT = float(os.environ.get("AUTO_ROUTER_TIMEOUT_S", "600"))
MAX_ATTEMPTS = int(os.environ.get("AUTO_ROUTER_MAX_ATTEMPTS", "3"))


def answer_text(data: dict) -> str:
    """The assistant text of an OpenAI-shaped completion, tool calls aside."""
    for choice in data.get("choices") or []:
        message = (choice or {}).get("message") or {}
        content = message.get("content")
        if isinstance(content, list):
            return "".join(p.get("text", "") for p in content if isinstance(p, dict))
        if isinstance(content, str):
            return content
    return ""


def request_text(messages: list[dict]) -> str:
    """The user's own last message: what the judge is asked to grade against."""
    for message in reversed(messages):
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, list):
            return "".join(p.get("text", "") for p in content if isinstance(p, dict))
        if isinstance(content, str):
            return content
    return ""


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


#: Rejected rather than coerced. Both the OpenAI and the Anthropic API define
#: ``max_tokens`` as a positive integer, so a float, a numeric string or a bool
#: is a malformed request, and guessing what the caller meant would be worse
#: than saying so: the value decides the ``outcome_memory`` budget bucket, and
#: everything that is not a positive int falls into ``"default"``, the bucket
#: that means *"no explicit budget, the provider's own default, which the
#: router does not know"*. Silently pooling a stated 12000-token budget with
#: that population mixes evidence the module documents as not comparable, and
#: can make the avoidance rule fire - or fail to fire - on it. The same value
#: also reaches ``TurnRequest.output_tokens`` and from there the cost
#: arithmetic, where a string would raise deep inside the policy instead.
MAX_TOKENS_ERROR = "max_tokens must be a positive integer"


def explicit_max_tokens(body: dict) -> int | None:
    """The caller's stated output budget, or ``None`` when it stated none.

    Raises :class:`ValueError` for a present-but-invalid value. ``None`` and an
    absent key both mean "no explicit budget" and are unchanged behaviour.
    """
    value = (body or {}).get("max_tokens")
    if value is None:
        return None
    budget = explicit_budget(value)
    if budget is None:
        raise ValueError(MAX_TOKENS_ERROR)
    return budget


async def _route(messages, system, tools, max_tokens, exclude_subscriptions: bool = False) -> RouteResult:
    try:
        return await asyncio.to_thread(router.route, messages, system, tools, max_tokens, None,
                                       exclude_subscriptions)
    except NoRouteAvailable as exc:
        # Not a bug and not a 500: the operator's own quota rules closed the
        # last open route. Say which state we are in, so the caller can wait
        # or configure a cheaper one.
        raise HTTPException(503, str(exc)) from exc


# --------------------------------------------------------------------------
# OpenAI-compatible
# --------------------------------------------------------------------------
@app.post("/v1/chat/completions")
async def chat_completions(request: Request) -> Any:
    body = await request.json()
    messages = body.get("messages") or []
    if not messages:
        raise HTTPException(400, "messages is required")
    try:
        max_tokens = explicit_max_tokens(body)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    system = next((m.get("content") for m in messages if m.get("role") == "system"), None)
    convo = [m for m in messages if m.get("role") != "system"]
    result = await _route(convo, system, body.get("tools"), max_tokens)

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
    prompt = request_text(convo)
    #: One verified escalation per turn. The second route is above the checked
    #: tier by construction, so this only ever binds when an operator has
    #: configured the whole catalog as cheap - and there a chain of judges
    #: would spend more on checking than on answering.
    verify_budget = 1
    #: The verdict that made this turn escalate, kept across the retry.
    checked = None
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
                answer = answer_text(data)
                verdict = await asyncio.to_thread(router.check, result, prompt, answer)
                # The escalated answer is reported with the verdict that caused
                # the escalation, not with the "not checked" verdict its own
                # stronger route earns: what the caller wants to know is why it
                # got a second answer.
                verdict = checked or verdict
                if verdict.escalate and verify_budget > 0:
                    verify_budget -= 1
                    retry, messages, verdict = await asyncio.to_thread(
                        router.escalate_after_verdict, result, verdict, body.get("messages") or [],
                        answer)
                    if retry is not None:
                        checked = verdict
                        metrics.record_error(result.model.name)
                        body = {**body, "messages": messages}
                        result = retry
                        continue
                return JSONResponse({**data, "x_router": {
                    "decision": result.explanation.id if result.explanation else None,
                    "model": result.model.name,
                    "escalated_from": first.model.name if first.model.name != result.model.name else None,
                    "verification": verdict.to_dict(),
                }}, headers=result.headers)
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


def stream_escalation_allowed() -> bool:
    """May a stream append a second, better answer after a failed check?

    Off by default, and it has to be. The first answer's bytes are already on
    the wire and cannot be taken back, so appending a second one changes what a
    plain OpenAI client sees: two answers concatenated in one message. A client
    that knows about this - the demo's playground collapses the first answer
    and streams the second below it - opts in per request with
    ``"x_router": {"stream_escalate": true}`` or by setting
    ``AUTO_ROUTER_STREAM_ESCALATE=1`` for the whole deployment. Everyone else
    still gets the verdict, in the final ``x_router`` event and in the decision
    record, and nothing appended to their answer.
    """
    return os.environ.get("AUTO_ROUTER_STREAM_ESCALATE", "0") not in ("", "0", "false", "no")


async def _stream_upstream(body: dict, result: RouteResult, state: dict):
    """Proxy one upstream stream and record it. Yields every line but ``[DONE]``.

    The sentinel is held back because the turn may not be over: when the answer
    judge rejects a cheap answer and the client asked for the second attempt,
    a second stream follows in the same response.
    """
    provider = provider_for(result)
    payload = {**body, "model": result.model.upstream_id,
               "stream_options": {**(body.get("stream_options") or {}), "include_usage": True}}
    usage = Usage()
    finish_reason: str | None = None
    text: list[str] = []
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
                if chunk == "[DONE]":
                    continue
                if chunk:
                    try:
                        parsed = json.loads(chunk)
                        if parsed.get("usage"):
                            usage = parse_openai_usage(parsed["usage"])
                        for choice in parsed.get("choices") or []:
                            if not isinstance(choice, dict):
                                continue
                            piece = (choice.get("delta") or {}).get("content")
                            if isinstance(piece, str):
                                text.append(piece)
                            # With several choices, a length stop on any of them
                            # is the honest verdict for the turn.
                            reason = choice.get("finish_reason")
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
    state.update(answered=cut is None, text="".join(text))


def _x_router_chunk(result: RouteResult, verdict, first: RouteResult, more: bool) -> str:
    """A chunk-shaped carrier for the router's own metadata.

    Shaped like a completion chunk with no choices - the same shape providers
    use for their final usage-only chunk - so a client that does not know about
    ``x_router`` reads it as an empty chunk instead of failing on it.
    """
    payload = {
        "object": "chat.completion.chunk",
        "model": result.model.name,
        "choices": [],
        "x_router": {
            "decision": result.explanation.id if result.explanation else None,
            "model": result.model.name,
            "escalated_from": first.model.name if first.model.name != result.model.name else None,
            "second_answer_follows": more,
            "verification": verdict.to_dict() if verdict is not None else None,
        },
    }
    return f"data: {json.dumps(payload)}\n\n"


async def _openai_stream(body: dict, result: RouteResult):
    convo = [m for m in (body.get("messages") or []) if m.get("role") != "system"]
    prompt = request_text(convo)
    opt_in = bool((body.get("x_router") or {}).get("stream_escalate")) or stream_escalation_allowed()
    first = result
    verdict = None
    state: dict = {}
    async for line in _stream_upstream(body, result, state):
        yield line
    if state.get("answered"):
        verdict = await asyncio.to_thread(router.check, result, prompt, state.get("text", ""))
        if verdict.escalate and opt_in:
            retry, messages, verdict = await asyncio.to_thread(
                router.escalate_after_verdict, result, verdict, body.get("messages") or [],
                state.get("text", ""))
            if retry is not None:
                metrics.record_error(result.model.name)
                yield _x_router_chunk(result, verdict, first, more=True)
                result = retry
                async for line in _stream_upstream({**body, "messages": messages}, retry, {}):
                    yield line
    yield _x_router_chunk(result, verdict, first, more=False)
    yield "data: [DONE]\n\n"


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
    try:
        max_tokens = explicit_max_tokens(body)
    except ValueError as exc:
        # Anthropic's own error envelope, so a client that already handles a
        # 400 from api.anthropic.com handles this one unchanged.
        return JSONResponse({"type": "error", "error": {"type": "invalid_request_error",
                                                        "message": str(exc)}}, status_code=400)
    from . import plan_auth
    kind = plan_auth.credential_kind(request.headers)
    from . import switch
    switch_file = switch.state_path(request.headers.get(switch.SWITCH_HEADER, "").strip())
    plan_reachable = kind == plan_auth.SUBSCRIPTION
    result = await _route(messages, body.get("system"), body.get("tools"), max_tokens,
                          exclude_subscriptions=not plan_reachable and switch_file is None)
    if result.model.subscription == "claude" and not plan_reachable:
        # A switch-mode session (see switch.py). The hook owns the decision at
        # the start of a prompt; here the plan can only win as an escalation,
        # when the cheap route is stuck inside a turn. The plan's login is not
        # on this request, so the gateway cannot serve it: it ends the turn and
        # asks the wrapper to resume the conversation in plan mode.
        if result.turn_start:
            result = await _route(messages, body.get("system"), body.get("tools"),
                                  max_tokens, exclude_subscriptions=True)
        else:
            return switch_to_plan(request, body, result, switch_file)
    if result.model.subscription == "claude":
        return await shim.subscription_passthrough(request, raw, body, result)
    if kind == plan_auth.SUBSCRIPTION and plan_auth.subscription_mode() == "passthrough_only":
        # The client is signed in with its own claude.ai login, so the plan
        # pays and Claude Code's own model choice stands. The router records
        # what it would have done and changes nothing. See shim.py.
        return await shim.advisory_passthrough(request, raw, result)
    return await proxy_openai_as_anthropic(body, result)


def switch_to_plan(request: Request, body: dict, result: RouteResult, path) -> Any:
    """End a cheap-mode turn and hand the conversation to the plan."""
    from . import switch
    reason = result.reason
    switch.write_state(path, {
        "session_id": request.headers.get("x-claude-code-session-id"),
        "prompt": ("Continue with the task. (auto-router moved this conversation to your Claude "
                   f"plan because the cheap route was stuck: {reason[:200]})"),
        "target": switch.PLAN, "plan_model": result.model.upstream_id, "reason": reason,
        "at": time.time(), "by": "gateway"})
    router.observe(result, ObservedOutcome(
        model=result.model.name, status="handed_over", http_status=200, latency_ms=0.0,
        error="switch mode: turn ended so the wrapper can resume it on the plan",
        cost_basis="not served here: the plan answers after the wrapper resumes the session"))
    notice = ("auto-router: the cheap route is stuck here, so this conversation continues on "
              f"your Claude plan in a moment. ({reason[:200]})")
    message = {"id": f"msg_{uuid.uuid4().hex[:24]}", "type": "message", "role": "assistant",
               "model": body.get("model") or result.model.name,
               "content": [{"type": "text", "text": notice}], "stop_reason": "end_turn",
               "stop_sequence": None, "usage": {"input_tokens": 0, "output_tokens": 0}}
    headers = {**result.headers, "X-Router-Switch": "plan"}
    if body.get("stream"):
        return StreamingResponse(iter(anthropic_sse_from_message(message)),
                                 media_type="text/event-stream", headers=headers)
    return JSONResponse(message, headers=headers)


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
