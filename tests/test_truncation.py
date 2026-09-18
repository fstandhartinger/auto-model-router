"""An answer the provider says it never finished is not a successful answer.

The 18 September held-out run measured one route difference that no capability
score predicted: on the two hardest design tasks one route burned its whole
12,000-token output budget without finishing the page, while a route the
design-arena evidence rated *lower* answered the same prompt in about 2,200
tokens. The first route returned HTTP 200 with a well-formed body, so the
router recorded it as a success and handed the half-finished page to the
client.

These tests pin the four properties that fix asks for:

(a) a completion the provider explicitly marked as a length stop does not look
    successful in the decision record;
(b) an eligible alternate route is attempted through the *existing* safe
    fallback, not through a new mechanism;
(c) nothing about the prompt, the provider body or a credential reaches a
    decision record or the ledger because of it;
(d) an ordinary completion behaves exactly as it did before.

Everything here is deterministic and local: the upstream is an in-process
``httpx.MockTransport``, so no socket is opened and no provider is called.
"""

import json

import httpx
import pytest
from fastapi.testclient import TestClient

from auto_router import server
from auto_router.catalog import CacheRules, Catalog, ModelInfo, Prices
from auto_router.config import Provider, RouterConfig
from auto_router.decision import ObservedOutcome
from auto_router.ledger import RoutingLedger
from auto_router.router import Router
from auto_router.truncation import truncation_label

SECRET = "sk-" + "ant-api03-" + "T" * 24
PROMPT = "Build a responsive analytics dashboard with a collapsible sidebar. "
CACHE = CacheRules(ttl_seconds=300, min_tokens=1024, hit_rate=0.9)


# ---------------------------------------------------------------------------
# the detector: the provider's own flag, and nothing else
# ---------------------------------------------------------------------------
def _openai(finish, **extra):
    choice = {"index": 0, "finish_reason": finish,
              "message": {"role": "assistant", "content": "half a dash"}, **extra}
    return {"choices": [choice], "usage": {"prompt_tokens": 10, "completion_tokens": 12000}}


def test_an_openai_length_stop_is_a_truncation():
    assert truncation_label(_openai("length")) == "finish_reason:length"


def test_an_anthropic_max_tokens_stop_is_a_truncation():
    assert truncation_label({"type": "message", "stop_reason": "max_tokens"}) == \
        "stop_reason:max_tokens"


def test_a_vendor_native_length_stop_is_a_truncation():
    """Gateways pass the origin provider's own word through beside the mapped one."""
    body = _openai("stop", native_finish_reason="MAX_TOKENS")
    assert truncation_label(body) == "native_finish_reason:max_tokens"


@pytest.mark.parametrize("body", [
    _openai("stop"),
    _openai("tool_calls"),
    _openai("content_filter"),
    _openai(None),
    _openai("function_call"),
    {"choices": [{"index": 0, "message": {"content": "x"}}]},
    {"choices": []},
    {"stop_reason": "end_turn"},
    {},
    None,
    "not a dict",
    {"choices": "not a list"},
])
def test_anything_the_provider_did_not_explicitly_flag_is_not_a_truncation(body):
    assert truncation_label(body) is None


def test_an_answer_that_merely_reads_as_cut_off_is_not_a_truncation():
    """The one inference this module refuses to make."""
    body = {"choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant",
                                     "content": "<div class=\"sidebar\"><ul><li>Over"}}]}
    assert truncation_label(body) is None


def test_an_unknown_vendor_stop_word_is_not_a_truncation():
    assert truncation_label(_openai("eos")) is None
    # Built, never a literal: a key-shaped string in the tree trips the gate.
    fake_key = "sk-" + "live-" + "abcdefghijklmnop"
    assert truncation_label(_openai(f"ERROR: rejected key {fake_key}")) is None


def test_the_label_is_built_from_fixed_vocabulary_not_from_the_body():
    body = _openai("length")
    body["choices"][0]["message"]["content"] = f"partial page, key={SECRET}"
    body["error"] = {"message": f"context overflow for {SECRET}"}
    label = truncation_label(body)
    assert label == "finish_reason:length"
    assert SECRET not in label


# ---------------------------------------------------------------------------
# a routed turn, end to end, against a deterministic in-process upstream
# ---------------------------------------------------------------------------
def _model(name, inp, out, cap):
    return ModelInfo(name, "stub", f"stub/{name}", Prices(inp, out, inp / 10), CACHE,
                     capability={"coding": cap, "general": cap, "design": cap},
                     capability_basis={"coding": "aa_coding_index", "general": "ii",
                                       "design": "designarena_elo"},
                     capability_strength={"coding": "direct", "general": "direct",
                                          "design": "direct"})


def _complete(model, content="a finished page"):
    return {"id": "x", "object": "chat.completion", "model": model,
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": content}}],
            "usage": {"prompt_tokens": 4321, "completion_tokens": 2200,
                      "prompt_tokens_details": {"cached_tokens": 4000}}}


def _truncated(model):
    return {"id": "x", "object": "chat.completion", "model": model,
            "choices": [{"index": 0, "finish_reason": "length",
                         "message": {"role": "assistant",
                                     "content": f"half a page, key={SECRET}"}}],
            "usage": {"prompt_tokens": 4321, "completion_tokens": 12000,
                      "prompt_tokens_details": {"cached_tokens": 4000}}}


class Upstream:
    """Deterministic fake provider. Records every upstream model it was asked for."""

    def __init__(self, reply):
        self.reply = reply
        self.calls: list[str] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.calls.append(body["model"])
        return httpx.Response(200, json=self.reply(body["model"]))


@pytest.fixture
def harness(monkeypatch, tmp_path):
    """The real FastAPI app, a local catalog and an in-process upstream."""

    def build(reply, models=None, **env):
        # Equal capability on purpose: nothing in the catalog is "clearly
        # stronger", so the only route left is the documented safe fallback.
        models = models or [_model("twelve-k", 0.5, 2.0, 70), _model("other-route", 1.0, 4.0, 70)]
        config = RouterConfig(
            providers={"stub": Provider("stub", "http://127.0.0.1:1/v1", cache="openai")},
            catalog=Catalog(models),
            policy={"success": {"evidence_discount": 0.5}})
        ledger_path = tmp_path / "decisions.jsonl"
        router = Router(config, classifier=None, ledger=RoutingLedger(ledger_path))
        upstream = Upstream(reply)
        monkeypatch.setattr(server, "config", config)
        monkeypatch.setattr(server, "router", router)
        monkeypatch.setattr(server, "_client",
                            httpx.AsyncClient(transport=httpx.MockTransport(upstream)))
        for name, value in env.items():
            monkeypatch.setenv(name, value)
        return TestClient(server.app), router, upstream, ledger_path

    return build


def _ask(client, text=None, **extra):
    body = {"model": "auto", "max_tokens": 12000,
            "messages": [{"role": "user", "content": (text or PROMPT * 60) + f" key={SECRET}"}],
            **extra}
    return client.post("/v1/chat/completions", json=body)


def _observed(router):
    return [d.observed for d in router.decisions if d.observed is not None]


# -- (a) a length stop does not look successful -----------------------------
def test_a_truncated_completion_is_not_recorded_as_a_success(harness):
    client, router, _upstream, _ = harness(_truncated)
    _ask(client)
    statuses = [o.status for o in _observed(router)]
    assert "ok" not in statuses, f"a length stop was recorded as a success: {statuses}"
    assert statuses and all(s == "truncated" for s in statuses), statuses


def test_the_truncated_attempt_names_the_providers_own_flag(harness):
    client, router, _upstream, _ = harness(_truncated)
    _ask(client)
    first = _observed(router)[0]
    assert first.error == "finish_reason:length"
    assert first.http_status == 200, "the transport was fine; the answer was not"
    assert first.output_tokens == 12000, "the tokens really were spent and are still recorded"


# -- (b) the existing safe fallback retries an eligible alternate route ------
def test_a_truncated_completion_falls_back_to_an_eligible_alternate_route(harness):
    seen: list[str] = []

    def reply(model):
        seen.append(model)
        return _truncated(model) if len(seen) == 1 else _complete(model)

    client, router, upstream, _ = harness(reply)
    response = _ask(client)
    assert response.status_code == 200
    assert len(upstream.calls) == 2, f"no alternate route was attempted: {upstream.calls}"
    assert upstream.calls[0] != upstream.calls[1], "the same route was retried"
    assert response.json()["choices"][0]["message"]["content"] == "a finished page"
    assert response.headers["X-Router-Model"] != upstream.calls[0].split("/")[-1]


def test_the_retry_uses_the_documented_safe_fallback_semantics(harness):
    """Not a new mechanism: the same sideways fallback a 5xx already takes."""
    seen: list[str] = []

    def reply(model):
        seen.append(model)
        return _truncated(model) if len(seen) == 1 else _complete(model)

    client, router, _upstream, _ = harness(reply)
    _ask(client)
    retry = router.decisions[-1].selection
    assert retry.switched_from is not None
    assert "safe fallback" in retry.reason, retry.reason


def test_the_alternate_route_is_the_one_the_client_is_told_about(harness):
    seen: list[str] = []

    def reply(model):
        seen.append(model)
        return _truncated(model) if len(seen) == 1 else _complete(model)

    client, router, upstream, _ = harness(reply)
    response = _ask(client)
    chosen = response.headers["X-Router-Model"]
    assert upstream.calls[1].endswith(chosen)
    assert response.headers["X-Router-Decision"] == router.decisions[-1].id


def test_when_every_route_truncates_the_client_still_gets_the_first_answer(harness):
    """No worse than before the change: the original body, not an error."""
    client, router, upstream, _ = harness(_truncated)
    response = _ask(client)
    assert response.status_code == 200
    assert response.json()["choices"][0]["finish_reason"] == "length"
    assert 1 < len(upstream.calls) <= 3, f"unbounded retry: {upstream.calls}"
    assert response.headers["X-Router-Model"] == upstream.calls[0].split("/")[-1], \
        "the returned body and the returned headers describe the same attempt"


def test_the_last_remaining_route_is_not_retried_at_all(harness):
    client, _router, upstream, _ = harness(_truncated, models=[_model("only", 0.5, 2.0, 70)])
    response = _ask(client)
    assert response.status_code == 200
    assert upstream.calls == ["stub/only"], "nowhere to fall back to"


def test_the_truncation_retry_budget_is_one_extra_attempt_by_default(harness):
    client, _router, upstream, _ = harness(
        _truncated,
        models=[_model(n, 0.5, 2.0, 70) for n in ("a", "b", "c", "d")])
    _ask(client)
    assert len(upstream.calls) == 2, f"expected one extra attempt, got {upstream.calls}"


def test_the_truncation_retry_budget_is_configurable(harness):
    client, _router, upstream, _ = harness(
        _truncated,
        models=[_model(n, 0.5, 2.0, 70) for n in ("a", "b", "c", "d")],
        AUTO_ROUTER_TRUNCATION_RETRIES="0")
    _ask(client)
    assert len(upstream.calls) == 1, "retries off means the previous behaviour, minus the label"


# -- (c) nothing about the prompt, the body or a credential is recorded ------
def test_a_truncated_turn_records_no_prompt_text_body_or_credential(harness):
    client, router, _upstream, ledger_path = harness(_truncated)
    _ask(client)
    decisions = client.get("/v1/router/decisions?limit=10").json()
    ledger = ledger_path.read_text()
    for blob in (json.dumps(decisions), ledger):
        assert SECRET not in blob
        assert "responsive analytics" not in blob
        assert "half a page" not in blob
    assert ledger.strip(), "the ledger was written"
    records = decisions["data"]
    assert any(r["observed_outcome"]["status"] == "truncated" for r in records)
    assert all(r["observed_outcome"]["error"] in (None, "finish_reason:length")
               for r in records if r["observed_outcome"])


def test_a_truncation_invents_no_capability_score(harness):
    """The observation is recorded. The evidence is not edited."""
    client, router, _upstream, _ = harness(_truncated)
    before = {m.name: dict(m.capability) for m in router.config.catalog.all()}
    _ask(client)
    after = {m.name: dict(m.capability) for m in router.config.catalog.all()}
    assert before == after
    for record in (d.to_dict() for d in router.decisions):
        for candidate in record["candidates"]:
            assert candidate["capability_basis"] in ("designarena_elo", "aa_coding_index", "ii",
                                                     "config", "none")
            assert "truncat" not in json.dumps(candidate)


def test_the_classification_and_the_observation_stay_apart(harness):
    client, router, _upstream, _ = harness(_truncated)
    _ask(client)
    record = router.decisions[0].to_dict()
    assert "truncat" not in json.dumps(record["classification"])
    assert "truncat" not in json.dumps(record["estimated_outcome"])
    assert record["observed_outcome"]["status"] == "truncated"


# -- (d) an ordinary completion is untouched --------------------------------
def test_a_normal_completion_keeps_its_current_behaviour(harness):
    client, router, upstream, _ = harness(_complete)
    response = _ask(client)
    assert response.status_code == 200
    assert len(upstream.calls) == 1, upstream.calls
    assert response.json()["choices"][0]["message"]["content"] == "a finished page"
    observed = _observed(router)
    assert [o.status for o in observed] == ["ok"]
    assert observed[0].output_tokens == 2200 and observed[0].cached_read_tokens == 4000
    assert observed[0].cost_usd is not None


def test_a_tool_call_stop_is_still_a_success(harness):
    def reply(model):
        body = _complete(model, content=None)
        body["choices"][0]["finish_reason"] = "tool_calls"
        body["choices"][0]["message"]["tool_calls"] = [
            {"id": "c1", "type": "function",
             "function": {"name": "search", "arguments": "{}"}}]
        return body

    client, router, upstream, _ = harness(reply)
    assert _ask(client).status_code == 200
    assert len(upstream.calls) == 1
    assert [o.status for o in _observed(router)] == ["ok"]


def test_an_upstream_error_still_behaves_as_it_did(harness):
    """The 5xx path is unchanged: still upstream_error, still a sideways fallback."""
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(json.loads(request.content)["model"])
        if len(calls) == 1:
            return httpx.Response(503, json={"error": {"message": f"down, key={SECRET}"}})
        return httpx.Response(200, json=_complete(calls[-1]))

    client, router, _upstream, _ = harness(_complete)
    server._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    response = _ask(client)
    assert response.status_code == 200 and len(calls) == 2
    statuses = [o.status for o in _observed(router)]
    assert statuses == ["upstream_error", "ok"]
    assert _observed(router)[0].error == "http_503"


# -- the Anthropic-shaped surface -------------------------------------------
def test_the_anthropic_path_records_a_length_stop_as_truncated(harness):
    client, router, _upstream, _ = harness(_truncated)
    response = client.post("/v1/messages", json={
        "model": "auto", "max_tokens": 12000,
        "messages": [{"role": "user", "content": PROMPT * 60}]})
    assert response.status_code == 200, response.text
    assert response.json()["stop_reason"] == "max_tokens"
    assert [o.status for o in _observed(router)] == ["truncated"]
    assert _observed(router)[0].error == "finish_reason:length"


def test_the_anthropic_path_still_records_a_normal_stop_as_ok(harness):
    client, router, _upstream, _ = harness(_complete)
    response = client.post("/v1/messages", json={
        "model": "auto", "max_tokens": 4000,
        "messages": [{"role": "user", "content": PROMPT * 60}]})
    assert response.status_code == 200, response.text
    assert [o.status for o in _observed(router)] == ["ok"]


# -- the streaming surfaces: recorded, never retried ------------------------
def _sse(chunks):
    body = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"
    return httpx.Response(200, content=body.encode(),
                          headers={"Content-Type": "text/event-stream"})


LENGTH_CHUNKS = [
    {"choices": [{"index": 0, "delta": {"content": "half a page"}}]},
    {"choices": [{"index": 0, "delta": {}, "finish_reason": "length"}],
     "usage": {"prompt_tokens": 4321, "completion_tokens": 12000}},
]
STOP_CHUNKS = [
    {"choices": [{"index": 0, "delta": {"content": "a finished page"}}]},
    {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
     "usage": {"prompt_tokens": 4321, "completion_tokens": 2200}},
]


@pytest.mark.parametrize("path,chunks,expected", [
    ("/v1/chat/completions", LENGTH_CHUNKS, "truncated"),
    ("/v1/chat/completions", STOP_CHUNKS, "ok"),
    ("/v1/messages", LENGTH_CHUNKS, "truncated"),
    ("/v1/messages", STOP_CHUNKS, "ok"),
])
def test_a_stream_records_the_length_stop_it_saw(harness, path, chunks, expected):
    client, router, _upstream, _ = harness(_complete)
    server._client = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: _sse(chunks)))
    with client.stream("POST", path, json={
            "model": "auto", "max_tokens": 12000, "stream": True,
            "messages": [{"role": "user", "content": PROMPT * 60}]}) as response:
        assert response.status_code == 200
        streamed = "".join(response.iter_text())
    assert streamed, "the client still received the partial answer"
    assert [o.status for o in _observed(router)] == [expected]


def test_a_stream_that_used_tool_calls_and_ran_out_of_budget_is_still_truncated(harness):
    """The Anthropic stop reason alone cannot answer this: it maps to tool_use."""
    chunks = [
        {"choices": [{"index": 0, "delta": {"tool_calls": [
            {"index": 0, "id": "c1", "type": "function",
             "function": {"name": "search", "arguments": "{\"q\": \"a"}}]}}]},
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "length"}],
         "usage": {"prompt_tokens": 4321, "completion_tokens": 12000}},
    ]
    client, router, _upstream, _ = harness(_complete)
    server._client = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: _sse(chunks)))
    with client.stream("POST", "/v1/messages", json={
            "model": "auto", "max_tokens": 12000, "stream": True,
            "messages": [{"role": "user", "content": PROMPT * 60}]}) as response:
        streamed = "".join(response.iter_text())
    assert '"stop_reason": "tool_use"' in streamed, "the wire format is unchanged"
    assert [o.status for o in _observed(router)] == ["truncated"]
    assert _observed(router)[0].error == "finish_reason:length"


def test_a_length_stop_on_any_choice_truncates_the_stream(harness):
    """n>1: one finished choice does not make the turn finished."""
    chunks = [
        {"choices": [{"index": 0, "delta": {"content": "half"}, "finish_reason": "length"},
                     {"index": 1, "delta": {"content": "whole"}, "finish_reason": "stop"}],
         "usage": {"prompt_tokens": 4321, "completion_tokens": 12000}},
    ]
    client, router, _upstream, _ = harness(_complete)
    server._client = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: _sse(chunks)))
    with client.stream("POST", "/v1/chat/completions", json={
            "model": "auto", "max_tokens": 12000, "stream": True,
            "messages": [{"role": "user", "content": PROMPT * 60}]}) as response:
        list(response.iter_text())
    assert [o.status for o in _observed(router)] == ["truncated"]


def test_a_stream_is_never_retried(harness):
    """Bytes already on the wire cannot be taken back; only the record is fixed."""
    calls: list[str] = []

    def handler(request):
        calls.append(json.loads(request.content)["model"])
        return _sse(LENGTH_CHUNKS)

    client, _router, _upstream, _ = harness(_complete)
    server._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    with client.stream("POST", "/v1/chat/completions", json={
            "model": "auto", "max_tokens": 12000, "stream": True,
            "messages": [{"role": "user", "content": PROMPT * 60}]}) as response:
        list(response.iter_text())
    assert len(calls) == 1


# -- the record itself -------------------------------------------------------
def test_truncated_is_a_recognised_observed_status():
    outcome = ObservedOutcome(model="m", status="truncated", error="finish_reason:length")
    assert outcome.to_dict()["status"] == "truncated"
    assert outcome.to_dict()["error"] == "finish_reason:length"
