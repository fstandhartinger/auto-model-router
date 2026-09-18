"""Which credential is on a request, and where that credential may go.

Claude Code reaches a gateway in one of two authentication modes, and the
difference decides who pays and what the gateway is allowed to do:

* **Subscription (claude.ai login).** The developer set ``ANTHROPIC_BASE_URL``
  and *no* gateway credential, so Claude Code keeps using its own saved login.
  Anthropic documents this case explicitly: "Setting only that variable,
  without a gateway credential, doesn't replace the subscription. Requests
  still route through the gateway, but a saved claude.ai login remains the
  active credential, so its usage limits and billing apply. Gateways that pass
  this traffic on to Anthropic must forward the OAuth capability in
  ``anthropic-beta``" (code.claude.com/docs/en/llm-gateway).
* **Gateway credential.** ``ANTHROPIC_AUTH_TOKEN``, ``ANTHROPIC_API_KEY`` or an
  ``apiKeyHelper`` is set, "the credential replaces the subscription login for
  that session, and the subscription's usage limits don't apply" (same page).
  That traffic is billed per token to whoever owns the credential.

Telling them apart is not guesswork: the gateway compatibility guide says that
on a claude.ai login the ``anthropic-beta`` header "also carries an OAuth
capability that the upstream requires, and stripping it fails those requests
with ``401``" (code.claude.com/docs/en/llm-gateway-protocol). So the capability
list on the request is the signal, with the token shape as a second opinion.

The one rule this module enforces mechanically: **a subscription credential
goes to Anthropic or nowhere.** A gateway is a piece of infrastructure holding
somebody's personal login in flight; sending that header to any other host
would hand a third party a live Claude account. Routing a turn to a cheaper
provider stays possible - the router then authenticates with *its own* key and
the client's ``Authorization`` header is never forwarded (see
``server.provider_headers``) - but the subscription header itself has exactly
one legitimate destination.
"""

from __future__ import annotations

import os
from typing import Iterable, Mapping
from urllib.parse import urlsplit

#: Authentication modes, in the vocabulary the docs use.
SUBSCRIPTION = "subscription"
GATEWAY_CREDENTIAL = "api_key"
NONE = "none"

#: Marker inside an ``anthropic-beta`` capability value that identifies
#: subscription (claude.ai login) traffic. Matched as a substring because the
#: capability is dated and the date changes; matching the whole string would
#: mean re-pinning this file on every rotation.
OAUTH_CAPABILITY_MARKER = "oauth"

#: Claude subscription access tokens observed in ``~/.claude/.credentials.json``.
#: Only ever compared against, never logged or stored.
OAUTH_TOKEN_PREFIXES = ("sk-ant-oat", "sk-ant-ort")

#: The only host a subscription credential may be forwarded to.
ANTHROPIC_HOSTS = frozenset({"api.anthropic.com"})


class UpstreamNotAllowed(RuntimeError):
    """Raised instead of forwarding a subscription credential off-Anthropic."""


def _bearer(headers: Mapping[str, str]) -> str:
    for name in ("authorization", "x-api-key"):
        value = _get(headers, name)
        if value:
            return value.split(" ", 1)[-1].strip()
    return ""


def _get(headers: Mapping[str, str], name: str) -> str:
    # Starlette headers are case-insensitive; a plain dict from a test is not.
    try:
        value = headers.get(name)
    except AttributeError:  # pragma: no cover - defensive
        return ""
    if value is None:
        for key, candidate in headers.items():
            if key.lower() == name:
                return candidate or ""
        return ""
    return value


def credential_kind(headers: Mapping[str, str]) -> str:
    """``subscription``, ``api_key`` or ``none`` for one inbound request.

    The OAuth capability in ``anthropic-beta`` is the documented signal and is
    checked first, because it is the half the upstream requires. A token shape
    check follows for clients that send the login without the capability.
    """
    betas = _get(headers, "anthropic-beta")
    if OAUTH_CAPABILITY_MARKER in betas.lower():
        return SUBSCRIPTION
    token = _bearer(headers)
    if not token:
        return NONE
    if token.startswith(OAUTH_TOKEN_PREFIXES):
        return SUBSCRIPTION
    return GATEWAY_CREDENTIAL


def allowed_hosts() -> frozenset[str]:
    """Hosts a subscription credential may reach.

    ``AUTO_ROUTER_ALLOW_UPSTREAM_HOSTS`` exists for tests and for a developer
    pointing the shim at a local recording proxy; it is a deliberate act, never
    a default.
    """
    extra = os.environ.get("AUTO_ROUTER_ALLOW_UPSTREAM_HOSTS", "")
    names = {h.strip().lower() for h in extra.split(",") if h.strip()}
    return frozenset(ANTHROPIC_HOSTS | names)


def check_upstream(url: str, kind: str, hosts: Iterable[str] | None = None) -> None:
    """Refuse to forward subscription traffic anywhere but Anthropic."""
    if kind != SUBSCRIPTION:
        return
    host = (urlsplit(url).hostname or "").lower()
    allowed = frozenset(hosts) if hosts is not None else allowed_hosts()
    if host not in allowed:
        raise UpstreamNotAllowed(
            f"refusing to forward a Claude subscription credential to {host or '(no host)'}; "
            f"allowed: {', '.join(sorted(allowed))}")


def subscription_mode() -> str:
    """What the gateway may do with subscription-authenticated traffic.

    ``passthrough_only`` (the default) forwards every such request to Anthropic
    unchanged and keeps the routing decision as an advisory record: the plan
    pays, Claude Code behaves exactly as it does without a gateway, and the
    ledger still shows what a router *would* have picked. ``route_others``
    additionally lets the policy serve a turn from a cheaper provider on the
    operator's own API key - which works, but is a configuration Anthropic
    states it "doesn't support" (code.claude.com/docs/en/llm-gateway), so it
    has to be asked for.
    """
    value = os.environ.get("AUTO_ROUTER_SUBSCRIPTION_MODE", "passthrough_only").strip().lower()
    return value if value in ("passthrough_only", "route_others") else "passthrough_only"


def plan_models() -> frozenset[str]:
    """Models the operator states their own plan includes.

    Only used to bound model rewriting. Empty (the default) means no rewrite is
    allowed on subscription traffic at all: swapping in a model the plan does
    not grant would be asking the plan for something the client could not have
    asked for itself.
    """
    raw = os.environ.get("AUTO_ROUTER_PLAN_MODELS", "")
    return frozenset(m.strip() for m in raw.split(",") if m.strip())
