"""Claude Code in front of the router: the gateway path.

Point Claude Code at the router::

    ANTHROPIC_BASE_URL=http://127.0.0.1:8787 claude

This is the configuration Anthropic documents for an LLM gateway, and what
happens to billing depends on one thing only - whether a *gateway credential*
is set as well:

* No gateway credential: "a saved claude.ai login remains the active
  credential, so its usage limits and billing apply", and a gateway forwarding
  that traffic to Anthropic "must forward the OAuth capability in
  ``anthropic-beta``" (code.claude.com/docs/en/llm-gateway). The request is
  therefore forwarded byte for byte, headers included, and the turn is billed
  to the plan exactly as if the router were not in the path.
* ``ANTHROPIC_AUTH_TOKEN`` / ``ANTHROPIC_API_KEY`` / ``apiKeyHelper`` set: "the
  credential replaces the subscription login for that session, and the
  subscription's usage limits don't apply" (same page). The traffic is metered
  against that credential, and the router treats the Claude route like any
  other metered route.

Two guardrails live here, both about the first case:

1. A subscription credential is only ever forwarded to Anthropic
   (``plan_auth.check_upstream``). It is somebody's live Claude login; no
   routing decision is worth handing it to a third party.
2. On subscription traffic the router does not silently change what Claude Code
   asked for. By default it forwards and records the decision it *would* have
   made (``AUTO_ROUTER_SUBSCRIPTION_MODE``), because Anthropic "doesn't support
   routing Claude Code to non-Claude models through any gateway".

The credential is forwarded but never logged, stored, or put into an error
message; ``redact`` is applied to every header dict that reaches a log line.
This is for one person's own Claude Code sessions on their own plan: Anthropic
does not permit developers "to route requests through Free, Pro, or Max plan
credentials on behalf of their users" (code.claude.com/docs/en/legal-and-compliance).
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import TYPE_CHECKING, Any, AsyncIterator, Callable

import httpx
from fastapi import Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from . import plan_auth
from .decision import ObservedOutcome
from .truncation import truncation_label

if TYPE_CHECKING:
    from .router import RouteResult

log = logging.getLogger("auto_router.shim")

ANTHROPIC_UPSTREAM = os.environ.get("AUTO_ROUTER_ANTHROPIC_UPSTREAM", "https://api.anthropic.com")
SENSITIVE_HEADERS = {"authorization", "x-api-key", "proxy-authorization", "cookie"}

#: When true the router may replace the model Claude Code asked for with the
#: model the policy picked. Off by default, and on subscription traffic bounded
#: by ``AUTO_ROUTER_PLAN_MODELS``: pure passthrough is the configuration known
#: to preserve subscription billing, prompt caching and preserved thinking.
REWRITE_MODEL = os.environ.get("AUTO_ROUTER_REWRITE_MODEL", "false").strip().lower() in {"1", "true", "yes", "on"}

_upstream: httpx.AsyncClient | None = None


def redact(headers: dict[str, str]) -> dict[str, str]:
    return {k: (f"<redacted {len(v)} chars>" if k.lower() in SENSITIVE_HEADERS else v)
            for k, v in headers.items()}


def upstream() -> httpx.AsyncClient:
    global _upstream
    if _upstream is None:
        _upstream = httpx.AsyncClient(timeout=float(os.environ.get("AUTO_ROUTER_TIMEOUT_S", "600")))
    return _upstream


def upstream_base() -> str:
    """Read per call, so a test or a restart can change it without re-import."""
    return os.environ.get("AUTO_ROUTER_ANTHROPIC_UPSTREAM", ANTHROPIC_UPSTREAM).rstrip("/")


def forward_headers(request: Request) -> dict[str, str]:
    """Everything the client sent, minus the hop-by-hop headers.

    Deliberately an open list rather than an allowlist: "Claude Code gains
    capabilities over releases, and they arrive as new ``anthropic-beta``
    values, new request body fields, and occasionally new ``anthropic-*`` or
    ``x-claude-code-*`` headers" - a gateway pinned to an observed list breaks
    the next capability (code.claude.com/docs/en/llm-gateway-protocol).
    """
    skip = {"host", "content-length", "connection", "transfer-encoding", "accept-encoding"}
    return {k: v for k, v in request.headers.items() if k.lower() not in skip}


def _blocked(exc: plan_auth.UpstreamNotAllowed) -> JSONResponse:
    log.error("%s", exc)
    return JSONResponse({"type": "error", "error": {"type": "permission_error", "message": str(exc)}},
                        status_code=502)


async def passthrough(request: Request, body: bytes, path: str) -> Response:
    """Forward one request to Anthropic unchanged, streaming or not."""
    target = f"{upstream_base()}{path}"
    kind = plan_auth.credential_kind(request.headers)
    try:
        plan_auth.check_upstream(target, kind)
    except plan_auth.UpstreamNotAllowed as exc:
        return _blocked(exc)
    req = upstream().build_request(request.method, target,
                                   headers=forward_headers(request), content=body,
                                   params=dict(request.query_params))
    resp = await upstream().send(req, stream=True)
    drop = {"content-length", "transfer-encoding", "content-encoding", "connection"}
    out_headers = {k: v for k, v in resp.headers.items() if k.lower() not in drop}
    if resp.headers.get("content-type", "").startswith("text/event-stream"):
        async def body_iter():
            try:
                async for chunk in resp.aiter_raw():
                    yield chunk
            finally:
                await resp.aclose()
        return StreamingResponse(body_iter(), status_code=resp.status_code, headers=out_headers,
                                 media_type="text/event-stream")
    data = await resp.aread()
    await resp.aclose()
    return Response(content=data, status_code=resp.status_code, headers=out_headers)


def may_rewrite_model(target: str, kind: str) -> bool:
    """Whether the router may swap in ``target`` for what the client asked.

    On a metered credential this is an ordinary routing decision. On a
    subscription credential it is only allowed for a model the operator has
    stated their own plan includes, because a plan grants particular models: a
    gateway that upgrades every request to the strongest name it knows would be
    asking the plan for something the developer could not have selected in
    Claude Code themselves.
    """
    if not REWRITE_MODEL:
        return False
    if kind != plan_auth.SUBSCRIPTION:
        return True
    return target in plan_auth.plan_models()


#: One ``data:`` line of an Anthropic stream is small. A body that never sends
#: a newline is not one a stop flag can be read out of, so the unfinished line
#: is dropped rather than grown without bound.
MAX_SSE_LINE = 1 << 16


def _length_stop(payload: Any) -> str | None:
    """The plan's own length-stop flag in one message, or ``None``.

    Anthropic states the stop reason at the top level of a message and inside
    the ``delta`` of a stream's ``message_delta`` event. Both are read through
    ``truncation.truncation_label``, so the label can only ever be built from
    that module's fixed vocabularies - never from a fragment of the body.
    """
    if not isinstance(payload, dict):
        return None
    return truncation_label(payload) or truncation_label(payload.get("delta"))


def _length_stop_in(blob: bytes) -> str | None:
    """Same judgement for a body that arrived in one piece."""
    try:
        return _length_stop(json.loads(blob))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None


async def _watch_for_length_stop(source: AsyncIterator[bytes],
                                 record: Callable[[str | None], None]) -> AsyncIterator[bytes]:
    """Forward an SSE body byte for byte, and record the outcome when it ends.

    A stream says how it stopped last, so here the observation can only be made
    after the client already has the text - the same order
    ``server._anthropic_stream`` records a metered stream in. Only whole
    ``data:`` lines are parsed and only for the stop flag; at most one
    unfinished line is held, and nothing read here is kept.

    A stream can also stop without ever saying how, which is a third outcome
    and not a success: see the ``except`` below.
    """
    label: str | None = None
    partial = ""
    try:
        async for chunk in source:
            yield chunk
            if label is not None:
                continue
            partial += chunk.decode("utf-8", "replace")
            lines = partial.split("\n")
            partial = lines.pop()
            if len(partial) > MAX_SSE_LINE:
                partial = ""
            for line in lines:
                line = line.strip()
                if not line.startswith("data:"):
                    continue
                try:
                    found = _length_stop(json.loads(line[5:]))
                except json.JSONDecodeError:
                    continue
                if found:
                    label = found
                    break
    except BaseException as exc:  # noqa: BLE001 - re-raised; only the record is added
        # A stream that did not reach its end never said how it stopped, and
        # "no flag seen" is not "it finished". The headers were a 200, so
        # recording ``ok`` here would call a broken turn a success *and* add it
        # to the ``ok`` side of the denominator ``outcome_memory`` divides
        # truncations by. ``server._anthropic_stream`` already calls this a
        # ``transport_error``, which that memory deliberately does not count.
        # A flag read before the break still stands: it is the provider's own
        # word about the turn, which nothing later can unsay.
        record(label, f"transport:{type(exc).__name__}")
        raise
    record(label, None)


async def subscription_passthrough(request: Request, raw: bytes, body: dict,
                                   result: "RouteResult") -> Response:
    """The routed Claude turn: forward it, and book it against the plan."""
    from .server import router  # the shared router instance

    kind = plan_auth.credential_kind(request.headers)
    forward_body = raw
    headers = dict(result.headers)
    if body.get("model") != result.model.upstream_id and may_rewrite_model(result.model.upstream_id, kind):
        body["model"] = result.model.upstream_id
        forward_body = json.dumps(body).encode()
        headers["X-Router-Model-Rewritten"] = result.model.upstream_id
    log.info("subscription passthrough model=%s auth=%s headers=%s",
             result.model.name, kind, redact(dict(request.headers)))
    started = time.perf_counter()
    response = await passthrough(request, forward_body, "/v1/messages")
    served = response.status_code < 400
    if served:
        router.commit(result)

    def record(cut: str | None, broke: str | None = None) -> None:
        # A 200 the plan itself flagged as a length stop spent plan quota
        # without finishing the answer. Calling that ``ok`` would hide the
        # failure *and* pad the denominator the avoidance rule reads
        # (``outcome_memory.COUNTED_STATUSES`` counts ``ok`` and ``truncated``),
        # so this surface reads the same machine-readable flag every metered
        # surface already reads. The plan is still charged either way, which is
        # why ``commit`` above does not depend on it.
        #
        # ``ok`` is therefore the narrowest of the four: it means the response
        # was delivered to its end and the plan's own flag said nothing. A
        # stream that broke on the way (``broke``) is a ``transport_error``,
        # which the memory does not count - but a length stop already read
        # outranks it, because that one was really observed.
        if not served:
            status, error = "upstream_error", None
        elif cut:
            status, error = "truncated", cut
        elif broke:
            status, error = "transport_error", broke
        else:
            status, error = "ok", None
        router.observe(result, ObservedOutcome(
            model=result.model.name,
            status=status,
            # What the headers really said, even when the body did not arrive.
            http_status=response.status_code,
            latency_ms=(time.perf_counter() - started) * 1000,
            error=error,
            cost_basis="subscription route: no per-token charge; the plan's own usage limits apply"))

    if isinstance(response, StreamingResponse):
        response.body_iterator = _watch_for_length_stop(response.body_iterator, record)
    else:
        record(_length_stop_in(response.body) if served else None)
    for key, value in headers.items():
        response.headers[key] = value
    return response


async def advisory_passthrough(request: Request, raw: bytes, result: "RouteResult") -> Response:
    """Forward a subscription turn the router would have sent elsewhere.

    The default for subscription traffic. Claude Code behaves exactly as it
    does without a gateway - same model, same plan, same limits - while the
    decision record keeps what the policy would have chosen and marks it
    ``not_taken``. That is what makes "how much of this week's plan use could
    have gone to a cheaper route" answerable without changing a single answer
    on the way there.
    """
    from .server import router

    started = time.perf_counter()
    response = await passthrough(request, raw, "/v1/messages")
    router.observe(result, ObservedOutcome(
        model=result.model.name, status="not_taken", http_status=response.status_code,
        latency_ms=(time.perf_counter() - started) * 1000,
        error="subscription traffic forwarded unchanged (AUTO_ROUTER_SUBSCRIPTION_MODE=passthrough_only)",
        cost_basis="not taken: the turn was served by Claude Code's own subscription route"))
    response.headers["X-Router-Advisory-Model"] = result.model.name
    response.headers["X-Router-Subscription-Mode"] = "passthrough_only"
    return response
