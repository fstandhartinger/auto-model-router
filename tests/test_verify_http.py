"""Verify-and-escalate over the HTTP surface, against a local fake provider.

What the endpoint has to guarantee:

(a) a cheap answer the judge rejects is replaced by a second answer from a
    stronger route, and the caller is told why;
(b) the caller gets exactly one answer for a non-streamed request, whatever
    happened behind it - the escalation is not visible as two completions;
(c) a stream, whose bytes cannot be taken back, does not grow a second answer
    unless the client asked for one;
(d) an approved answer costs one judge call and nothing else.

The upstream is an in-process ``httpx.MockTransport`` and the judge is a local
function, so nothing here opens a socket or spends money.
"""

import json

import httpx
import pytest
from fastapi.testclient import TestClient

from auto_router import server
from auto_router.catalog import CacheRules, Catalog, ModelInfo, Prices
from auto_router.config import Provider, RouterConfig
from auto_router.jev import Classification, Judgement
from auto_router.router import Router

CACHE = CacheRules(ttl_seconds=300, min_tokens=1024, hit_rate=0.9)
PROMPT = "Write a Python function median(xs) that returns the median of a list of numbers."


def _model(name, inp, out, cap):
    return ModelInfo(name, "stub", f"stub/{name}", Prices(inp, out, inp / 10), CACHE,
                     capability={"coding": cap, "general": cap},
                     capability_basis={"coding": "aa_coding_index", "general": "ii"},
                     capability_strength={"coding": "direct", "general": "direct"})


CHEAP = _model("cheap-route", 0.05, 0.2, 45)
STRONG = _model("strong-route", 5.0, 20.0, 78)


def _completion(model, content):
    return {"id": "x", "object": "chat.completion", "model": model,
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": content}}],
            "usage": {"prompt_tokens": 200, "completion_tokens": 120}}


class Upstream:
    def __init__(self):
        self.calls: list[dict] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.calls.append(body)
        answer = "def median(xs): pass" if body["model"] == CHEAP.upstream_id \
            else "def median(xs): return sorted(xs)[len(xs)//2]"
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
        return httpx.Response(200, json=_completion(body["model"], answer))


class Judge:
    """P(adequate) by upstream route: the cheap answer fails, the strong one passes."""

    def __init__(self, p_cheap=0.04):
        self.p_cheap = p_cheap
        self.calls: list[tuple[str, str]] = []

    def __call__(self, request, response, **kw):
        self.calls.append((request, response))
        return Judgement(p_adequate=self.p_cheap, latency_s=0.6, failure="incomplete",
                         model="jev-test")


def _classifier(text, context):
    """An easy, self-contained coding request: exactly the tier the judge covers."""
    return Classification("coding", {}, 0.05, 1.0, 0.9, 0.0, 0.0, 0.2, 0.3, 0.01)


@pytest.fixture
def harness(monkeypatch, tmp_path):
    def build(judge=None, **env):
        config = RouterConfig(
            providers={"stub": Provider("stub", "http://127.0.0.1:1/v1", cache="openai")},
            catalog=Catalog([CHEAP, STRONG]),
            policy={"success": {"evidence_discount": 0.0}})
        router = Router(config, classifier=_classifier, judge=judge)
        upstream = Upstream()
        monkeypatch.setattr(server, "config", config)
        monkeypatch.setattr(server, "router", router)
        monkeypatch.setattr(server, "_client",
                            httpx.AsyncClient(transport=httpx.MockTransport(upstream)))
        for name, value in env.items():
            monkeypatch.setenv(name, value)
        return TestClient(server.app), router, upstream

    return build


def _ask(client, **extra):
    return client.post("/v1/chat/completions", json={
        "model": "auto", "max_tokens": 800,
        "messages": [{"role": "user", "content": PROMPT}], **extra})


# ---------------------------------------------------------------------------
def test_a_rejected_cheap_answer_is_replaced_by_the_stronger_routes_answer(harness):
    judge = Judge()
    client, router, upstream = harness(judge=judge)
    response = _ask(client)
    assert response.status_code == 200
    body = response.json()
    assert [c["model"] for c in upstream.calls] == [CHEAP.upstream_id, STRONG.upstream_id]
    assert "sorted(xs)" in body["choices"][0]["message"]["content"]
    assert len(body["choices"]) == 1
    router_meta = body["x_router"]
    assert router_meta["escalated_from"] == CHEAP.name
    assert router_meta["verification"]["escalate"] is True
    assert router_meta["verification"]["failure"] == "incomplete"
    assert response.headers["x-router-verify-escalated-to"] == STRONG.name
    # The judge is asked about the cheap answer only; the strong route is above
    # the tier it is allowed to grade.
    assert len(judge.calls) == 1
    assert judge.calls[0][0] == PROMPT


def test_an_approved_answer_is_returned_unchanged(harness):
    judge = Judge(p_cheap=0.96)
    client, router, upstream = harness(judge=judge)
    body = _ask(client).json()
    assert [c["model"] for c in upstream.calls] == [CHEAP.upstream_id]
    assert body["x_router"]["verification"]["escalate"] is False
    assert body["x_router"]["verification"]["p_adequate"] == 0.96


def test_without_a_judge_the_endpoint_behaves_exactly_as_before(harness):
    client, router, upstream = harness(judge=None)
    body = _ask(client).json()
    assert [c["model"] for c in upstream.calls] == [CHEAP.upstream_id]
    assert body["x_router"]["verification"]["verified"] is False
    assert "choices" in body


def test_a_stream_is_not_given_a_second_answer_unless_the_client_asked(harness):
    judge = Judge()
    client, router, upstream = harness(judge=judge)
    with client.stream("POST", "/v1/chat/completions", json={
            "model": "auto", "stream": True,
            "messages": [{"role": "user", "content": PROMPT}]}) as response:
        text = "".join(response.iter_text())
    assert [c["model"] for c in upstream.calls] == [CHEAP.upstream_id]
    assert "sorted(xs)" not in text
    meta = _last_x_router(text)
    assert meta["verification"]["escalate"] is True
    assert meta["second_answer_follows"] is False
    assert text.rstrip().endswith("data: [DONE]")


def test_a_client_that_opts_in_gets_the_second_answer_in_the_same_stream(harness):
    judge = Judge()
    client, router, upstream = harness(judge=judge)
    with client.stream("POST", "/v1/chat/completions", json={
            "model": "auto", "stream": True, "x_router": {"stream_escalate": True},
            "messages": [{"role": "user", "content": PROMPT}]}) as response:
        text = "".join(response.iter_text())
    assert [c["model"] for c in upstream.calls] == [CHEAP.upstream_id, STRONG.upstream_id]
    assert "sorted(xs)" in text
    assert text.count("second_answer_follows\": true") == 1
    assert text.rstrip().endswith("data: [DONE]")
    assert _last_x_router(text)["escalated_from"] == CHEAP.name


def _last_x_router(text: str) -> dict:
    """The last ``x_router`` chunk in an SSE body."""
    for line in reversed(text.splitlines()):
        if line.startswith("data: ") and "x_router" in line:
            return json.loads(line[6:])["x_router"]
    raise AssertionError("no x_router chunk in the stream")
