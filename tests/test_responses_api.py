"""Providers configured with ``api: responses`` are called on /v1/responses.

GPT-6 Luna rejects function tools with reasoning on /v1/chat/completions, so a
Claude Code turn routed there failed with HTTP 400. These tests pin the
conversion both ways and all three HTTP paths (chat, chat stream, Anthropic
messages with tools) against an in-process fake provider; no socket, no spend.
"""

import json

import httpx
import pytest
from fastapi.testclient import TestClient

from auto_router import responses_api, server
from auto_router.catalog import CacheRules, Catalog, ModelInfo, Prices
from auto_router.config import Provider, RouterConfig
from auto_router.jev import Classification
from auto_router.router import Router

CACHE = CacheRules(ttl_seconds=300, min_tokens=1024, hit_rate=0.9)
MODEL = ModelInfo("luna", "oai", "gpt-6-luna", Prices(0.1, 0.5, 0.01), CACHE,
                  capability={"coding": 60, "general": 60},
                  capability_basis={"coding": "aa_coding_index", "general": "ii"},
                  capability_strength={"coding": "direct", "general": "direct"})
TOOL = {"type": "function", "function": {"name": "Write", "description": "write a file",
                                         "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}}}


def _response(calls=False, status="completed", reason=None):
    output = [{"type": "reasoning", "id": "rs_1", "summary": []}]
    if calls:
        output.append({"type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "Write",
                       "arguments": '{"path": "roman.py"}'})
    else:
        output.append({"type": "message", "role": "assistant",
                       "content": [{"type": "output_text", "text": "done"}]})
    return {"id": "resp_1", "object": "response", "created_at": 1, "model": "gpt-6-luna",
            "status": status, "incomplete_details": {"reason": reason} if reason else None,
            "output": output,
            "usage": {"input_tokens": 1000, "input_tokens_details": {"cached_tokens": 600},
                      "output_tokens": 50, "output_tokens_details": {"reasoning_tokens": 20},
                      "total_tokens": 1050}}


def test_chat_request_becomes_a_responses_request():
    out = responses_api.chat_to_responses({
        "model": "gpt-6-luna", "max_completion_tokens": 900, "stream": True,
        "stream_options": {"include_usage": True}, "tools": [TOOL], "tool_choice": "auto",
        "messages": [
            {"role": "system", "content": "be terse"},
            {"role": "user", "content": [{"type": "text", "text": "write it"}]},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "call_1", "type": "function", "function": {"name": "Write", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "call_1", "content": "ok"},
        ]})
    assert out["instructions"] == "be terse"
    assert out["max_output_tokens"] == 900 and out["store"] is False
    assert "stream" not in out and "messages" not in out
    assert out["tools"][0]["name"] == "Write" and "function" not in out["tools"][0]
    assert out["input"] == [
        {"role": "user", "content": [{"type": "input_text", "text": "write it"}]},
        {"type": "function_call", "call_id": "call_1", "name": "Write", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "call_1", "output": "ok"},
    ]


def test_responses_result_becomes_a_chat_result_with_cache_usage():
    chat = responses_api.responses_to_chat(_response(calls=True))
    choice = chat["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["tool_calls"][0]["id"] == "call_1"
    assert chat["usage"]["prompt_tokens"] == 1000
    assert chat["usage"]["prompt_tokens_details"]["cached_tokens"] == 600
    cut = responses_api.responses_to_chat(_response(status="incomplete", reason="max_output_tokens"))
    assert cut["choices"][0]["finish_reason"] == "length"


class Upstream:
    def __init__(self, calls=False):
        self.paths: list[str] = []
        self.bodies: list[dict] = []
        self.calls = calls

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.paths.append(request.url.path)
        self.bodies.append(json.loads(request.content))
        if request.url.path.endswith("/chat/completions"):
            return httpx.Response(400, json={"error": {"message": "use /v1/responses"}})
        return httpx.Response(200, json=_response(calls=self.calls))


def _classifier(text, context):
    return Classification("coding", {}, 0.3, 1.0, 0.9, 0.0, 0.0, 0.2, 0.3, 0.01)


@pytest.fixture
def build(monkeypatch):
    def make(calls=False):
        config = RouterConfig(
            providers={"oai": Provider("oai", "http://127.0.0.1:1/v1", cache="openai", api="responses")},
            catalog=Catalog([MODEL]), policy={"success": {"evidence_discount": 0.0}})
        upstream = Upstream(calls)
        monkeypatch.setattr(server, "config", config)
        monkeypatch.setattr(server, "router", Router(config, classifier=_classifier))
        monkeypatch.setattr(server, "_client", httpx.AsyncClient(transport=httpx.MockTransport(upstream)))
        return TestClient(server.app), upstream
    return make


def test_chat_completions_surface_uses_the_responses_endpoint(build):
    client, upstream = build()
    r = client.post("/v1/chat/completions", json={
        "model": "auto", "max_tokens": 500, "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 200, r.text
    assert r.json()["choices"][0]["message"]["content"] == "done"
    assert upstream.paths == ["/v1/responses"]


def test_chat_stream_is_replayed_as_chat_sse(build):
    client, upstream = build()
    r = client.post("/v1/chat/completions", json={
        "model": "auto", "stream": True, "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 200
    assert '"content": "done"' in r.text and "data: [DONE]" in r.text
    assert upstream.paths == ["/v1/responses"]


@pytest.mark.parametrize("stream", [False, True])
def test_claude_code_tool_turn_reaches_the_responses_endpoint(build, stream):
    client, upstream = build(calls=True)
    r = client.post("/v1/messages", json={
        "model": "claude-opus-5-5", "max_tokens": 2000, "stream": stream,
        "system": "You are Claude Code.",
        "tools": [{"name": "Write", "description": "write a file",
                   "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}}}],
        "messages": [{"role": "user", "content": "write roman.py"}]})
    assert r.status_code == 200, r.text
    assert upstream.paths == ["/v1/responses"]
    assert upstream.bodies[0]["tools"][0]["name"] == "Write"
    assert "tool_use" in r.text and "roman.py" in r.text
