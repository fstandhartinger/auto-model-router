"""Quality check by intelligence threshold: grade answers of models below a reference.

The rule: when the answering model's intelligence (per category when both
sides have a direct score, the headline intelligence index otherwise) is below
the reference model's - GPT-5.6 Terra in the shipped example - the finished
answer is graded by a Jev-class judge, and a rejected answer is replaced by a
stronger route's answer before the coding agent sees anything.

Upstream and judge are local stand-ins; nothing opens a real socket.
"""

import json

import httpx
import pytest
from fastapi.testclient import TestClient

from auto_router import server
from auto_router.bench import BenchmarkClient
from auto_router.catalog import CacheRules, Catalog, ModelInfo, Prices
from auto_router.config import Provider, RouterConfig, resolve_reference
from auto_router.jev import Classification, Judgement
from auto_router.ledger import RoutingLedger
from auto_router.router import Router
from auto_router.verify import VerifyPolicy

CACHE = CacheRules(ttl_seconds=300, min_tokens=1024, hit_rate=0.9)
PROMPT = "Write a Python function median(xs) that returns the median of a list of numbers."
REFERENCE = {"reference_model": "gpt-5.6-terra::max", "resolved_id": "gpt-5.6-terra::max",
             "intelligence_index": 42.1, "source": "benchmark:network",
             "capability": {"coding": {"value": 76.7, "strength": "direct"},
                            "general": {"value": 63.15, "strength": "derived"}}}
THRESHOLD = {"reference_model": "gpt-5.6-terra::max", "value": 42.1}


def _model(name, inp, out, *, coding, intelligence, strength="direct"):
    # Blended prices above the cheap tier, so only the intelligence rule can apply.
    return ModelInfo(name, "stub", f"stub/{name}", Prices(inp, out, inp / 10), CACHE,
                     capability={"coding": coding, "general": coding},
                     capability_basis={"coding": "aa_coding_index", "general": "ii"},
                     capability_strength={"coding": strength, "general": "derived"},
                     intelligence_index=intelligence)


BELOW = _model("below-route", 2.0, 8.0, coding=60.0, intelligence=30.0)
BELOW2 = _model("below-route-2", 3.0, 12.0, coding=70.0, intelligence=38.0)
ABOVE = _model("above-route", 10.0, 50.0, coding=85.0, intelligence=52.0)

ANSWERS = {BELOW.upstream_id: "def median(xs): pass",
           BELOW2.upstream_id: "def median(xs): return xs[0]",
           ABOVE.upstream_id: "def median(xs): return sorted(xs)[len(xs)//2]"}


def _policy(**verify):
    return {"success": {"evidence_discount": 0.0},
            "verify": {"intelligence_threshold": dict(THRESHOLD), **verify}}


# -- the gate ------------------------------------------------------------------
def _gate_policy(reference=REFERENCE, **conf):
    return VerifyPolicy(intelligence_threshold={**THRESHOLD, **conf}, reference=reference)


def test_above_threshold_is_not_checked():
    ok, why, info = _gate_policy().gate(ABOVE, "coding")
    assert not ok and "not below the intelligence threshold" in why
    assert info["threshold_basis"] == "category:coding" and info["threshold"] == 76.7


def test_below_threshold_is_checked_per_category():
    ok, _why, info = _gate_policy().gate(BELOW, "coding")
    assert ok and info["rule"] == "intelligence-threshold"
    assert (info["intelligence"], info["threshold"]) == (60.0, 76.7)
    assert info["threshold_source"] == "benchmark:network"


def test_headline_index_when_a_side_has_no_direct_category_score():
    derived = _model("d", 2.0, 8.0, coding=90.0, intelligence=30.0, strength="derived")
    ok, _why, info = _gate_policy().gate(derived, "coding")
    assert ok and info["threshold_basis"] == "intelligence_index"
    assert (info["intelligence"], info["threshold"]) == (30.0, 42.1)
    # math has no direct reference score either way
    assert _gate_policy().gate(ABOVE, "math")[2]["threshold_basis"] == "intelligence_index"


def test_threshold_from_benchmark_data_vs_configured_fallback():
    fallback = _gate_policy(reference={"intelligence_index": None, "capability": {},
                                       "source": "none"}, value=35.0)
    below, info = fallback.intelligence_gate(BELOW, "coding")
    assert below and info["threshold"] == 35.0 and info["threshold_source"] == "configured-value"
    assert fallback.intelligence_gate(BELOW2, "coding")[0] is False
    nothing = _gate_policy(reference=None, value=None)
    below, info = nothing.intelligence_gate(BELOW, "coding")
    assert below is None and "no threshold" in info["note"]


def test_resolve_reference_from_data_and_from_the_configured_value(tmp_path):
    client = BenchmarkClient(base_url="http://127.0.0.1:9", cache_dir=tmp_path, offline=True,
                             snapshot=False)
    client._cache_path("/api/models/gpt-5.6-terra::max").write_text(json.dumps({"model": {
        "id": "gpt-5.6-terra::max", "benchmarks": {"aa_intelligence_index": 42.1,
                                                   "aa_coding_index": 76.7}}}))
    ref = resolve_reference(_policy(), client)
    assert ref["intelligence_index"] == 42.1 and ref["source"] == "benchmark:cache"
    assert ref["capability"]["coding"] == {"value": 76.7, "strength": "direct"}
    missing = resolve_reference({"verify": {"intelligence_threshold": {
        "reference_model": "not-there::x", "value": 40}}}, client)
    assert missing["intelligence_index"] == 40.0 and missing["source"] == "configured-value"
    assert "not-there::x" in missing["note"]
    assert resolve_reference({"verify": {"intelligence_threshold": {"enabled": False}}}, client) is None


def test_the_bundled_snapshot_carries_terra(tmp_path):
    client = BenchmarkClient(cache_dir=tmp_path, offline=True)
    ref = resolve_reference(_policy(), client)
    assert ref["resolved_id"] == "gpt-5.6-terra::max"
    assert ref["source"] == "benchmark:bundled-snapshot" and ref["intelligence_index"] > 0


def test_unknown_intelligence_is_checked_unless_configured_otherwise():
    unknown = ModelInfo("u", "stub", "stub/u", Prices(2, 8), CACHE, capability={})
    assert _gate_policy().gate(unknown, "coding")[0]
    assert not _gate_policy(unknown="skip").gate(unknown, "coding")[0]


def test_plans_and_disabled_rule_are_never_checked_by_it():
    plan = ModelInfo("plan", "stub", "stub/p", Prices.free(), CACHE, capability={"coding": 10},
                     intelligence_index=10.0, subscription="claude")
    assert not _gate_policy().gate(plan, "coding")[0]
    off = VerifyPolicy(intelligence_threshold={**THRESHOLD, "enabled": False}, reference=REFERENCE)
    assert not off.gate(BELOW, "coding")[0]


# -- over HTTP -------------------------------------------------------------------
def _completion(model, content, tool_call=False):
    message = {"role": "assistant", "content": content}
    if tool_call:
        message["tool_calls"] = [{"id": "c1", "type": "function",
                                  "function": {"name": "read", "arguments": "{}"}}]
    return {"id": "x", "object": "chat.completion", "model": model,
            "choices": [{"index": 0, "finish_reason": "tool_calls" if tool_call else "stop",
                         "message": message}],
            "usage": {"prompt_tokens": 200, "completion_tokens": 120}}


class Upstream:
    def __init__(self, tool_call=False):
        self.calls: list[dict] = []
        self.tool_call = tool_call

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.calls.append(body)
        answer = ANSWERS[body["model"]]
        if body.get("stream"):
            chunks = [
                {"id": "x", "object": "chat.completion.chunk", "model": body["model"],
                 "choices": [{"index": 0, "delta": {"content": answer}}]},
                {"id": "x", "object": "chat.completion.chunk", "model": body["model"],
                 "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                 "usage": {"prompt_tokens": 200, "completion_tokens": 120}},
            ]
            text = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"
            return httpx.Response(200, text=text, headers={"content-type": "text/event-stream"})
        return httpx.Response(200, json=_completion(body["model"], answer, self.tool_call))


class Judge:
    """Rejects every answer except the strongest route's (which it is never asked about)."""

    def __init__(self, p=0.05, failed=False):
        self.p, self.failed = p, failed
        self.calls: list[str] = []

    def __call__(self, request, response, **kw):
        self.calls.append(response)
        if self.failed:
            return Judgement(0.5, 0.2, failed=True)
        return Judgement(p_adequate=self.p, latency_s=0.3, failure="incomplete",
                         model="local-jev-class:test")


def _classifier(text, context):
    return Classification("coding", {}, 0.05, 1.0, 0.9, 0.0, 0.0, 0.2, 0.3, 0.01)


@pytest.fixture
def harness(monkeypatch, tmp_path):
    def build(judge=None, models=(BELOW, ABOVE), tool_call=False, **verify):
        config = RouterConfig(
            providers={"stub": Provider("stub", "http://127.0.0.1:1/v1", cache="openai")},
            catalog=Catalog(list(models)), policy=_policy(**verify),
            intelligence_reference=REFERENCE)
        ledger = RoutingLedger(tmp_path / "ledger.jsonl")
        router = Router(config, classifier=_classifier, judge=judge, ledger=ledger)
        upstream = Upstream(tool_call)
        monkeypatch.setattr(server, "config", config)
        monkeypatch.setattr(server, "router", router)
        monkeypatch.setattr(server, "_client",
                            httpx.AsyncClient(transport=httpx.MockTransport(upstream)))
        return TestClient(server.app), router, upstream, tmp_path / "ledger.jsonl"

    return build


def _ask(client, **extra):
    return client.post("/v1/chat/completions", json={
        "model": "auto", "max_tokens": 800,
        "messages": [{"role": "user", "content": PROMPT}], **extra})


def _ledger(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_http_above_threshold_is_returned_without_a_check(harness):
    judge = Judge()
    client, router, upstream, _ = harness(judge=judge, models=(ABOVE,))
    body = _ask(client).json()
    assert judge.calls == []
    assert body["x_router"]["verification"]["verified"] is False
    assert "intelligence threshold" in body["x_router"]["verification"]["reason"]


def test_http_below_threshold_pass_is_returned(harness):
    judge = Judge(p=0.95)
    client, router, upstream, ledger = harness(judge=judge)
    body = _ask(client).json()
    assert [c["model"] for c in upstream.calls] == [BELOW.upstream_id]
    assert body["choices"][0]["message"]["content"] == ANSWERS[BELOW.upstream_id]
    v = body["x_router"]["verification"]
    assert v["rule"] == "intelligence-threshold" and v["verified"] and not v["escalate"]
    assert v["intelligence"]["intelligence"] == 60.0 and v["intelligence"]["threshold"] == 76.7
    assert v["intelligence"]["reference_model"] == "gpt-5.6-terra::max"
    assert v["p_adequate"] == 0.95 and v["judge_latency_ms"] == 300.0
    assert v["judge_backend"] == "custom"
    events = [r for r in _ledger(ledger) if r.get("event") == "verification"]
    assert events and events[-1]["verification"]["rule"] == "intelligence-threshold"


def test_http_below_threshold_fail_returns_the_escalated_answer(harness):
    judge = Judge(p=0.05)
    client, router, upstream, ledger = harness(judge=judge)
    response = _ask(client)
    body = response.json()
    assert [c["model"] for c in upstream.calls] == [BELOW.upstream_id, ABOVE.upstream_id]
    assert body["choices"][0]["message"]["content"] == ANSWERS[ABOVE.upstream_id]
    v = body["x_router"]["verification"]
    assert v["escalate"] and v["escalated_to"] == ABOVE.name and v["escalations"] == 1
    assert body["x_router"]["escalated_from"] == BELOW.name
    assert response.headers["x-router-verify-escalated-to"] == ABOVE.name
    assert len(judge.calls) == 1, "the stronger route is not graded"
    logged = [r["verification"] for r in _ledger(ledger) if r.get("event") == "verification"]
    assert any(e["escalated_to"] == ABOVE.name and e["p_adequate"] == 0.05 for e in logged)


def test_http_judge_unavailable_returns_the_original_answer(harness):
    judge = Judge(failed=True)
    client, router, upstream, _ = harness(judge=judge)
    body = _ask(client).json()
    assert [c["model"] for c in upstream.calls] == [BELOW.upstream_id]
    assert body["choices"][0]["message"]["content"] == ANSWERS[BELOW.upstream_id]
    v = body["x_router"]["verification"]
    assert v["judge_failed"] and not v["escalate"]
    assert "original answer is returned" in v["reason"]


def test_http_no_judge_configured_is_logged_with_the_numbers(harness):
    client, router, upstream, _ = harness(judge=None)
    v = _ask(client).json()["x_router"]["verification"]
    assert not v["verified"] and v["reason"] == "no judge configured"
    assert v["intelligence"]["threshold"] == 76.7


def test_http_tool_call_steps_are_not_graded(harness):
    judge = Judge(p=0.01)
    client, router, upstream, _ = harness(judge=judge, tool_call=True)
    v = _ask(client).json()["x_router"]["verification"]
    assert judge.calls == [] and "tool-call step" in v["reason"]


def test_http_max_escalations_grades_a_still_below_second_route(harness):
    judge = Judge(p=0.05)
    client, router, upstream, _ = harness(judge=judge, models=(BELOW, BELOW2, ABOVE),
                                          max_escalations=2, min_capability_gain=4.0)
    body = _ask(client).json()
    models = [c["model"] for c in upstream.calls]
    assert models[0] == BELOW.upstream_id and models[-1] == ABOVE.upstream_id
    assert body["choices"][0]["message"]["content"] == ANSWERS[ABOVE.upstream_id]
    # escalation prefers a route that is not below the threshold when one exists
    assert models == [BELOW.upstream_id, ABOVE.upstream_id]


def test_escalation_count_is_bounded(harness):
    judge = Judge(p=0.05)
    client, router, upstream, _ = harness(judge=judge, models=(BELOW, BELOW2),
                                          max_escalations=1)
    body = _ask(client).json()
    assert len(upstream.calls) == 2 and len(judge.calls) == 1
    assert body["x_router"]["verification"]["escalations"] == 1


def _sse(response):
    return [json.loads(line[5:]) for line in response.text.splitlines()
            if line.startswith("data:") and line[5:].strip() not in ("", "[DONE]")]


def test_stream_below_threshold_is_buffered_and_replaced(harness):
    judge = Judge(p=0.05)
    client, router, upstream, _ = harness(judge=judge)
    response = _ask(client, stream=True)
    events = _sse(response)
    text = "".join((c.get("delta") or {}).get("content") or ""
                   for e in events for c in e.get("choices") or [])
    assert text == ANSWERS[ABOVE.upstream_id], "the rejected answer never reaches the client"
    meta = [e["x_router"] for e in events if "x_router" in e]
    assert len(meta) == 1 and meta[0]["verification"]["escalated_to"] == ABOVE.name
    assert meta[0]["second_answer_follows"] is False


def test_stream_below_threshold_pass_is_released_unchanged(harness):
    client, router, upstream, _ = harness(judge=Judge(p=0.9))
    text = "".join((c.get("delta") or {}).get("content") or ""
                   for e in _sse(_ask(client, stream=True)) for c in e.get("choices") or [])
    assert text == ANSWERS[BELOW.upstream_id]


def test_stream_buffering_can_be_switched_off(harness):
    judge = Judge(p=0.05)
    client, router, upstream, _ = harness(judge=judge, buffer_streams=False)
    assert not router.buffers_stream(router.route([{"role": "user", "content": PROMPT}]), PROMPT)
    text = "".join((c.get("delta") or {}).get("content") or ""
                   for e in _sse(_ask(client, stream=True)) for c in e.get("choices") or [])
    # the old stream-then-check behaviour: first answer on the wire, no second one
    assert text == ANSWERS[BELOW.upstream_id]


def _ask_anthropic(client, stream=False):
    return client.post("/v1/messages", json={
        "model": "claude-sonnet-5", "max_tokens": 800, "stream": stream,
        "messages": [{"role": "user", "content": PROMPT}]})


def test_anthropic_gateway_escalates_a_rejected_answer(harness):
    judge = Judge(p=0.05)
    client, router, upstream, _ = harness(judge=judge)
    response = _ask_anthropic(client)
    assert response.status_code == 200
    body = response.json()
    assert body["content"][0]["text"] == ANSWERS[ABOVE.upstream_id]
    assert [c["model"] for c in upstream.calls] == [BELOW.upstream_id, ABOVE.upstream_id]
    assert response.headers["x-router-verify-escalated-to"] == ABOVE.name


def test_anthropic_gateway_stream_is_buffered_and_replaced(harness):
    judge = Judge(p=0.05)
    client, router, upstream, _ = harness(judge=judge)
    response = _ask_anthropic(client, stream=True)
    deltas = [json.loads(line[5:]) for line in response.text.splitlines()
              if line.startswith("data:")]
    text = "".join(d["delta"].get("text", "") for d in deltas
                   if d.get("type") == "content_block_delta")
    assert text == ANSWERS[ABOVE.upstream_id]
    assert sum(1 for d in deltas if d.get("type") == "message_start") == 1


def test_a_second_route_still_below_the_threshold_is_graded_again(harness):
    judge = Judge(p=0.05)
    client, router, upstream, _ = harness(judge=judge, models=(BELOW, BELOW2),
                                          max_escalations=2)
    body = _ask(client).json()
    assert [c["model"] for c in upstream.calls] == [BELOW.upstream_id, BELOW2.upstream_id]
    assert judge.calls == [ANSWERS[BELOW.upstream_id], ANSWERS[BELOW2.upstream_id]]
    v = body["x_router"]["verification"]
    assert v["model"] == BELOW2.name and v["escalations"] == 1
    assert v["reason"] == "no stronger route available; the answer stands"
    assert body["choices"][0]["message"]["content"] == ANSWERS[BELOW2.upstream_id]
