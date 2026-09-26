"""Free local models as routing targets, and a local Jev-class decision model.

Both talk to a tiny fake OpenAI-compatible server on loopback - the same
shape LM Studio, llama.cpp's ``llama-server`` and Ollama expose - so nothing
is downloaded and nothing leaves the machine.
"""

import json
import math

import httpx
import pytest
from fastapi.testclient import TestClient

from auto_router import jev, server
from auto_router.bench import BenchmarkClient, minimal_document
from auto_router.config import is_loopback, load_config
from auto_router.jev import FALLBACK, LocalJevClass, classifier_from_config, judge_from_config
from auto_router.router import Router
from tests.fake_http import allow_loopback, serve

QWEN = {"model": {
    "id": "qwen-like::medium", "benchmarks": {"aa_intelligence_index": 27.6,
                                              "aa_coding_index": 56.1},
    "token_efficiency": {"aa": {"tokens_per_task": {"value": {"output": 50000}}}},
    "offers": []}}
BONSAI = {"model": {"id": "bonsai::default", "benchmarks": {},
                    "offers": [{"platform": "OpenRouter", "provider": "SomeHost",
                                "input_per_1m": 0.075, "output_per_1m": 0.5}]}}


def _completion(model, content):
    return {"id": "x", "object": "chat.completion", "model": model,
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": content}}],
            "usage": {"prompt_tokens": 50, "completion_tokens": 10}}


def _local_llm(method, path, body):
    if path == "/v1/chat/completions":
        return 200, _completion(body["model"], "local answer")
    if path == "/v1/models":
        return 200, {"data": [{"id": "ternary-bonsai-2-27b"}]}
    return 404, {"error": "no"}


def _snapshot_client(tmp_path):
    client = BenchmarkClient(base_url="http://127.0.0.1:9", cache_dir=tmp_path, offline=True,
                             snapshot=False)
    client._cache_path("/api/models/qwen-like::medium").write_text(json.dumps(QWEN))
    client._cache_path("/api/models/bonsai::default").write_text(json.dumps(BONSAI))
    return client


def _local_config(tmp_path, base_url):
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps({
        "providers": {"llama-cpp": {"base_url": base_url + "/v1"}},
        "models": [{"name": "bonsai-2-local", "provider": "llama-cpp",
                    "upstream_id": "ternary-bonsai-2-27b", "bench_id": "bonsai::default",
                    "capability_like": "qwen-like::medium"}]}))
    return load_config(path, bench=_snapshot_client(tmp_path))


def test_loopback_providers_are_local():
    for url in ("http://127.0.0.1:1234/v1", "http://localhost:11434/v1", "http://127.0.0.1:8080/v1"):
        assert is_loopback(url)
    assert not is_loopback("https://api.tensorx.ai/v1")


def test_a_local_model_is_free_and_its_capability_is_marked_assumed(tmp_path):
    cfg = _local_config(tmp_path, "http://127.0.0.1:1234")
    bonsai = cfg.catalog["bonsai-2-local"]
    assert cfg.providers["llama-cpp"].local and cfg.providers["llama-cpp"].api_key is None
    # the hosted offer in the benchmark data is somebody else's price, not this machine's
    assert bonsai.prices.is_free and bonsai.local
    assert bonsai.capability["coding"] == 56.1 and bonsai.intelligence_index == 27.6
    assert bonsai.capability_assumed_from == "qwen-like::medium"
    assert all(b.startswith("assumed like qwen-like::medium") for b in bonsai.capability_basis.values())
    assert set(bonsai.capability_strength.values()) == {"weak"}
    assert bonsai.evidence["assumed_capability"] is True
    assert bonsai.capability_source == "assumed:qwen-like::medium"


def test_capability_like_without_data_leaves_nothing_assumed_as_measured(tmp_path):
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps({"providers": {"lm": {"base_url": "http://127.0.0.1:1234/v1"}},
                                "models": [{"name": "m", "provider": "lm",
                                            "capability_like": "not-in-data::x"}]}))
    m = load_config(path, bench=_snapshot_client(tmp_path)).catalog["m"]
    assert m.capability == {} and m.intelligence_index is None and m.evidence_stale
    assert m.evidence["capability_assumed_from"] == "not-in-data::x"


def test_a_local_openai_compatible_server_is_a_routing_target(tmp_path, monkeypatch):
    allow_loopback(monkeypatch)
    with serve(_local_llm) as (url, requests):
        cfg = _local_config(tmp_path, url)
        router = Router(cfg, classifier=None, judge=None)
        monkeypatch.setattr(server, "config", cfg)
        monkeypatch.setattr(server, "router", router)
        monkeypatch.setattr(server, "_client", None)
        with TestClient(server.app) as client:
            response = client.post("/v1/chat/completions", json={
                "model": "auto", "messages": [{"role": "user", "content": "hello"}]})
    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "local answer"
    assert response.headers["x-router-model"] == "bonsai-2-local"
    assert response.headers["x-router-capability-assumed-from"] == "qwen-like::medium"
    method, path, headers, body = requests[0]
    assert (method, path, body["model"]) == ("POST", "/v1/chat/completions", "ternary-bonsai-2-27b")
    assert "Authorization" not in headers, "a local server gets no key"
    record = router.decisions[-1].to_dict()
    assert record["observed_outcome"]["cost_basis"] == \
        "local model: ~zero cost, local electricity not counted"
    assert record["observed_outcome"]["cost_usd"] is None
    notes = " ".join(record["notes"])
    assert "Assumed capability" in notes and "qwen-like::medium" in notes
    assert "local electricity not counted" in notes
    assert record["candidates"][0]["capability_basis"].startswith("assumed like")


# -- local Jev-class decision model ------------------------------------------------
TOP = {"A": -0.1, "B": -2.5, " C": -4.0}


def _decider(method, path, body):
    if path == "/v1/completions":
        return 200, {"model": "jevk5-q4", "choices": [{"text": "A", "logprobs": {
            "top_logprobs": [TOP]}}]}
    if path == "/v1/chat/completions":
        return 200, {"model": "jevk5-q4", "choices": [{
            "message": {"role": "assistant", "content": "A"},
            "logprobs": {"content": [{"token": "A", "logprob": -0.1, "top_logprobs": [
                {"token": k, "logprob": v} for k, v in TOP.items()]}]}}]}
    return 404, {}


def _expected_first(n):
    letters = jev.LETTERS[:n]
    floor = min(TOP.values()) - 2.0
    logits = [TOP.get(x, TOP.get(" " + x, floor)) for x in letters]
    weights = [math.exp((x - max(logits)) / 1.532) for x in logits]
    return weights[0] / sum(weights)


@pytest.mark.parametrize("fmt", ["chatml", "chat"])
def test_local_jev_class_classifies_and_judges(monkeypatch, fmt):
    allow_loopback(monkeypatch)
    with serve(_decider) as (url, requests):
        model = LocalJevClass(url + "/v1", "jevk5", prompt_format=fmt)
        cls = model.classify("fix the off-by-one in my loop", "")
        verdict = model.judge("fix it", "here is the fix", category="coding")
    assert not cls.failed and cls.source == jev.LOCAL_SOURCE
    assert cls.category == "coding"            # option A of the category question
    assert cls.needs_tools == pytest.approx(_expected_first(2))
    assert cls.category_confidence == pytest.approx(_expected_first(len(jev.CATEGORY_OPTIONS)))
    assert 0 <= cls.difficulty <= 1 and cls.model == "jevk5-q4"
    assert not verdict.failed and verdict.p_adequate == pytest.approx(_expected_first(2))
    assert verdict.failure == "fine" and verdict.model == "local-jev-class:jevk5-q4"
    assert len(requests) == len(jev.QUESTIONS) + 2
    for _method, path, headers, body in requests:
        assert "Authorization" not in headers
        assert body["max_tokens"] == 1 and body["temperature"] == 0
        if fmt == "chatml":
            assert path == "/v1/completions" and "<think>" in body["prompt"]
            assert body["logprobs"] == 20
        else:
            assert path == "/v1/chat/completions" and body["top_logprobs"] == 20
            assert body["messages"][0]["content"] == jev.LOCAL_SYSTEM


def test_local_jev_class_without_logprobs_uses_the_letter(monkeypatch):
    allow_loopback(monkeypatch)

    def letter_only(method, path, body):
        return 200, {"choices": [{"text": "B"}]}

    with serve(letter_only) as (url, _requests):
        verdict = LocalJevClass(url + "/v1").judge("q", "a")
    assert verdict.p_adequate == 0.0 and not verdict.failed


def test_local_jev_class_outage_degrades_like_the_hosted_one():
    model = LocalJevClass("http://127.0.0.1:9/v1", timeout=0.3)
    assert model.classify("anything") is FALLBACK
    judged = model.judge("q", "a", category="coding")
    assert judged.failed and judged.p_adequate == 0.5


def test_backends_from_config(monkeypatch):
    local = {"backend": "local-jev", "base_url": "http://127.0.0.1:8081/v1", "model": "jevk5"}
    cls = classifier_from_config({"classifier": local})
    assert isinstance(cls, LocalJevClass) and cls.base_url == "http://127.0.0.1:8081/v1"
    judge = judge_from_config({"verify": {"judge": local}})
    assert isinstance(judge.__self__, LocalJevClass)
    same = judge_from_config({"classifier": local, "verify": {"judge": "same-as-classifier"}})
    assert same.__self__.model == "jevk5"
    # hosted needs the user's own key; without one there is no judge at all
    assert judge_from_config({"verify": {"judge": {"backend": "hosted"}}}) is None
    assert judge_from_config({}) is None
    monkeypatch.setenv("TYPESAFE_API_KEY", "x" * 12)
    assert judge_from_config({}) is jev.judge
    assert judge_from_config({"verify": {"judge": "none"}}) is None
    with pytest.raises(ValueError):
        judge_from_config({"verify": {"judge": "bogus"}})
    with pytest.raises(ValueError):
        LocalJevClass(prompt_format="raw")


def test_router_reports_the_local_judge_backend():
    from auto_router.catalog import Catalog
    from auto_router.config import RouterConfig
    cfg = RouterConfig(providers={}, catalog=Catalog([]), policy={
        "verify": {"judge": {"backend": "local-jev"}}})
    assert Router(cfg, classifier=None).judge_backend == jev.LOCAL_SOURCE


def test_minimal_document_roundtrip_for_local_alias():
    assert minimal_document(QWEN)["model"]["benchmarks"]["aa_intelligence_index"] == 27.6
    assert httpx is not None
