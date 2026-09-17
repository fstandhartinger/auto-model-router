"""Claude Code subscription passthrough.

Point Claude Code at the router::

    ANTHROPIC_BASE_URL=http://127.0.0.1:8787 claude

When the policy picks a model tagged ``subscription: claude``, the request is
forwarded to Anthropic byte for byte with the client's own ``Authorization``
header, so it is billed against that user's plan exactly as if the router were
not there. Every other choice is translated to an OpenAI-compatible provider.

This is meant for one person's own Claude Code sessions on their own plan. It
must not be used to serve anyone else's traffic from a personal subscription.

The OAuth token is forwarded but never logged, stored, or put into an error
message; ``redact`` is applied to every header dict that reaches a log line.
"""

from __future__ import annotations

import json
import logging
import os
from typing import TYPE_CHECKING

import httpx
from fastapi import Request
from fastapi.responses import Response, StreamingResponse

if TYPE_CHECKING:
    from .router import RouteResult

log = logging.getLogger("auto_router.shim")

ANTHROPIC_UPSTREAM = os.environ.get("AUTO_ROUTER_ANTHROPIC_UPSTREAM", "https://api.anthropic.com")
SENSITIVE_HEADERS = {"authorization", "x-api-key", "proxy-authorization", "cookie"}

#: When true the router may replace the model Claude Code asked for with the
#: subscription model the policy picked (e.g. a cheaper tier). Off by default:
#: pure passthrough is the configuration known to preserve subscription billing.
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


def forward_headers(request: Request) -> dict[str, str]:
    skip = {"host", "content-length", "connection", "transfer-encoding", "accept-encoding"}
    return {k: v for k, v in request.headers.items() if k.lower() not in skip}


async def passthrough(request: Request, body: bytes, path: str) -> Response:
    req = upstream().build_request(request.method, f"{ANTHROPIC_UPSTREAM}{path}",
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


async def subscription_passthrough(request: Request, raw: bytes, body: dict, result: "RouteResult") -> Response:
    from .server import router  # the shared router instance
    forward_body = raw
    headers = dict(result.headers)
    if REWRITE_MODEL and body.get("model") != result.model.upstream_id:
        body["model"] = result.model.upstream_id
        forward_body = json.dumps(body).encode()
        headers["X-Router-Model-Rewritten"] = result.model.upstream_id
    log.info("subscription passthrough model=%s headers=%s", result.model.name, redact(dict(request.headers)))
    response = await passthrough(request, forward_body, "/v1/messages")
    router.commit(result)
    for key, value in headers.items():
        response.headers[key] = value
    return response
