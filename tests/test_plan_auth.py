"""Which credential is on a request, and what the gateway may do with it.

The rules under test come from Anthropic's own gateway documentation: a
claude.ai login stays the active credential when no gateway credential is set,
its usage limits and billing then apply, and a gateway forwarding that traffic
must pass the OAuth capability through. What this repository adds is the
refusal: that credential goes to Anthropic or nowhere.
"""

import json
import logging

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from auto_router import plan_auth, shim

#: Shaped like the real thing, valid nowhere. Never a real token, not even in
#: a fixture: a test file is the easiest place in a repository to leak one.
FAKE_OAUTH = "sk-ant-oat01-" + "A" * 40
FAKE_KEY = "sk-ant-api03-" + "B" * 40
OAUTH_BETAS = "oauth-2026-01-01,claude-code-20250219,fine-grained-tool-streaming-2025-05-14"


# --------------------------------------------------------------------------
# telling the two modes apart
# --------------------------------------------------------------------------
def test_the_oauth_capability_identifies_subscription_traffic():
    assert plan_auth.credential_kind({"anthropic-beta": OAUTH_BETAS}) == plan_auth.SUBSCRIPTION


def test_an_oauth_token_without_the_capability_is_still_subscription_traffic():
    assert plan_auth.credential_kind({"authorization": f"Bearer {FAKE_OAUTH}"}) == plan_auth.SUBSCRIPTION


def test_a_gateway_credential_is_not_subscription_traffic():
    headers = {"x-api-key": FAKE_KEY, "anthropic-beta": "claude-code-20250219"}
    assert plan_auth.credential_kind(headers) == plan_auth.GATEWAY_CREDENTIAL


def test_no_credential_at_all():
    assert plan_auth.credential_kind({"anthropic-version": "2023-06-01"}) == plan_auth.NONE


def test_header_lookup_is_case_insensitive():
    assert plan_auth.credential_kind({"Anthropic-Beta": OAUTH_BETAS.upper()}) == plan_auth.SUBSCRIPTION


# --------------------------------------------------------------------------
# where it may go
# --------------------------------------------------------------------------
def test_a_subscription_credential_may_only_go_to_anthropic():
    plan_auth.check_upstream("https://api.anthropic.com/v1/messages", plan_auth.SUBSCRIPTION)
    with pytest.raises(plan_auth.UpstreamNotAllowed, match="refusing to forward"):
        plan_auth.check_upstream("https://llm.example.invalid/v1/messages", plan_auth.SUBSCRIPTION)


def test_the_refusal_names_no_token():
    with pytest.raises(plan_auth.UpstreamNotAllowed) as exc:
        plan_auth.check_upstream("https://elsewhere.invalid/v1/messages", plan_auth.SUBSCRIPTION)
    assert "sk-ant" not in str(exc.value)


def test_a_metered_credential_is_the_operators_own_business():
    plan_auth.check_upstream("https://llm.example.invalid/v1/messages", plan_auth.GATEWAY_CREDENTIAL)


def test_extra_hosts_require_a_deliberate_act(monkeypatch):
    monkeypatch.setenv("AUTO_ROUTER_ALLOW_UPSTREAM_HOSTS", "127.0.0.1,localhost")
    plan_auth.check_upstream("http://127.0.0.1:9/v1/messages", plan_auth.SUBSCRIPTION)


# --------------------------------------------------------------------------
# what the router may change
# --------------------------------------------------------------------------
def test_subscription_mode_defaults_to_passthrough():
    assert plan_auth.subscription_mode() == "passthrough_only"


def test_an_unknown_mode_falls_back_to_the_safe_one(monkeypatch):
    monkeypatch.setenv("AUTO_ROUTER_SUBSCRIPTION_MODE", "whatever")
    assert plan_auth.subscription_mode() == "passthrough_only"


def test_model_rewriting_is_off_by_default(monkeypatch):
    monkeypatch.setattr(shim, "REWRITE_MODEL", False)
    assert not shim.may_rewrite_model("claude-opus-5", plan_auth.SUBSCRIPTION)
    assert not shim.may_rewrite_model("claude-opus-5", plan_auth.GATEWAY_CREDENTIAL)


def test_rewriting_subscription_traffic_needs_the_plan_to_include_the_model(monkeypatch):
    monkeypatch.setattr(shim, "REWRITE_MODEL", True)
    monkeypatch.setenv("AUTO_ROUTER_PLAN_MODELS", "claude-sonnet-5")
    assert shim.may_rewrite_model("claude-sonnet-5", plan_auth.SUBSCRIPTION)
    assert not shim.may_rewrite_model("claude-opus-5", plan_auth.SUBSCRIPTION)
    # On a metered credential the same swap is an ordinary routing decision.
    assert shim.may_rewrite_model("claude-opus-5", plan_auth.GATEWAY_CREDENTIAL)


# --------------------------------------------------------------------------
# the forwarding path itself
# --------------------------------------------------------------------------
def _app() -> FastAPI:
    app = FastAPI()

    @app.post("/v1/messages")
    async def messages(request: Request):          # noqa: ANN202 - test fixture
        return await shim.passthrough(request, await request.body(), "/v1/messages")

    return app


def test_forwarding_a_subscription_token_off_anthropic_is_refused(monkeypatch, caplog):
    monkeypatch.setenv("AUTO_ROUTER_ANTHROPIC_UPSTREAM", "https://not-anthropic.invalid")

    def explode(*_a, **_k):                        # the test fails loudly if we get this far
        raise AssertionError("the request must never reach an upstream client")

    monkeypatch.setattr(shim, "upstream", explode)
    with caplog.at_level(logging.ERROR, logger="auto_router.shim"):
        response = TestClient(_app()).post(
            "/v1/messages", json={"messages": []},
            headers={"authorization": f"Bearer {FAKE_OAUTH}", "anthropic-beta": OAUTH_BETAS})
    assert response.status_code == 502
    assert "refusing to forward" in response.json()["error"]["message"]
    assert "sk-ant" not in caplog.text and "sk-ant" not in json.dumps(response.json())


def test_headers_are_forwarded_as_an_open_list(monkeypatch):
    """The gateway guide: forward ``anthropic-beta`` verbatim, don't allowlist.

    Stripping the OAuth capability fails subscription requests with a 401, and
    pinning to today's list breaks the next capability Claude Code ships.
    """
    seen: dict = {}

    class FakeResponse:
        status_code = 200
        headers = {"content-type": "application/json"}

        async def aread(self):
            return b'{"ok": true}'

        async def aclose(self):
            return None

    class FakeClient:
        def build_request(self, method, url, headers=None, content=None, params=None):
            seen.update({"url": url, "headers": headers})
            return object()

        async def send(self, _request, stream=False):
            return FakeResponse()

    monkeypatch.setattr(shim, "upstream", lambda: FakeClient())
    response = TestClient(_app()).post(
        "/v1/messages", json={"messages": []},
        headers={"authorization": f"Bearer {FAKE_OAUTH}", "anthropic-beta": OAUTH_BETAS,
                 "anthropic-version": "2023-06-01", "x-claude-code-session-id": "abc",
                 "x-some-future-header": "keep me"})
    assert response.status_code == 200
    assert seen["url"] == "https://api.anthropic.com/v1/messages"
    forwarded = {k.lower(): v for k, v in seen["headers"].items()}
    assert forwarded["anthropic-beta"] == OAUTH_BETAS
    assert forwarded["anthropic-version"] == "2023-06-01"
    assert forwarded["x-claude-code-session-id"] == "abc"
    assert forwarded["x-some-future-header"] == "keep me"
    assert forwarded["authorization"] == f"Bearer {FAKE_OAUTH}"
    # Hop-by-hop headers are the only ones dropped.
    assert "content-length" not in forwarded and "connection" not in forwarded


# --------------------------------------------------------------------------
# the default for a whole subscription session
# --------------------------------------------------------------------------
def _server_with(tmp_path, monkeypatch):
    """A server whose only configured route is a cheap non-Claude one."""
    import importlib

    from auto_router import server as server_module

    config = tmp_path / "cfg.json"
    config.write_text(json.dumps({
        "providers": {"host": {"base_url": "https://llm.example.invalid/v1"}},
        "models": [{"name": "cheap", "provider": "host", "upstream_id": "vendor/cheap",
                    "free": True, "capability": {"general": 50, "agentic": 50}}]}))
    monkeypatch.setenv("AUTO_ROUTER_CONFIG", str(config))
    return importlib.reload(server_module)


@pytest.fixture
def restore_server():
    yield
    import importlib

    from auto_router import server as server_module
    importlib.reload(server_module)


def test_a_subscription_session_is_forwarded_even_when_a_cheaper_route_wins(
        tmp_path, monkeypatch, restore_server):
    """The default: the plan pays, Claude Code decides, the router only records.

    Anthropic "doesn't support routing Claude Code to non-Claude models through
    any gateway", so the cheaper route the policy found is written down as a
    decision that was *not taken* rather than acted on. Opting in is one
    environment variable, and it is the operator's call, not the router's.
    """
    server = _server_with(tmp_path, monkeypatch)

    class FakeResponse:
        status_code = 200
        headers = {"content-type": "application/json"}

        async def aread(self):
            return b'{"id": "abc123", "content": [{"type": "text", "text": "from anthropic"}]}'

        async def aclose(self):
            return None

    class FakeClient:
        def build_request(self, method, url, headers=None, content=None, params=None):
            assert url.startswith("https://api.anthropic.com")
            return object()

        async def send(self, _request, stream=False):
            return FakeResponse()

    monkeypatch.setattr(shim, "upstream", lambda: FakeClient())
    response = TestClient(server.app).post(
        "/v1/messages",
        json={"model": "claude-opus-5", "messages": [{"role": "user", "content": "hello"}],
              "max_tokens": 16},
        headers={"authorization": f"Bearer {FAKE_OAUTH}", "anthropic-beta": OAUTH_BETAS})

    assert response.status_code == 200
    assert response.json()["content"][0]["text"] == "from anthropic"
    assert response.headers["x-router-advisory-model"] == "cheap"
    assert response.headers["x-router-subscription-mode"] == "passthrough_only"
    record = server.router.decisions[-1]
    assert record.observed.status == "not_taken"
    assert record.observed.cost_usd is None
