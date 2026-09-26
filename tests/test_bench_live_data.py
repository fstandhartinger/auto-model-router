"""Benchmark data: network, disk cache, stale cache and the bundled snapshot.

Also covers what the router takes from that data beyond capability - the
intelligence index, measured tokens per task (the model's token appetite and
the input to cost per task) - and the configurable model list. The network is
a fake Benchmark Heaven on loopback; nothing here leaves the machine.
"""

import json
import os
import time

import pytest

from auto_router import bench
from auto_router.bench import (BenchmarkClient, cost_per_task, intelligence_index,
                               minimal_document, task_tokens)
from auto_router.config import load_config
from auto_router.policies import Context, TurnRequest, turn_call_cost
from auto_router.economics import SuccessModel
from tests.fake_http import allow_loopback, serve

MODEL_ID = "example-model::max"
DOC = {"model": {
    "id": MODEL_ID, "family_key": "example-model", "display_name": "Example (max)",
    "benchmarks": {"aa_intelligence_index": 40.0, "aa_coding_index": 65.0},
    "aa_metadata": {"context_window_tokens": 400000, "license_url": "https://x.invalid"},
    "token_efficiency": {"aa": {"tokens_per_task": {
        "value": {"reasoning": 9000, "answer": 1000, "output": 10000},
        "collected_at": "2026-09-25", "basis": "measured", "scope": "aa_intelligence_index_benchmark",
        "source": "some upstream", "url": "https://upstream.invalid"}},
        "input_output_ratio": {"value": 20.0, "source": "a provider's usage statistics"}},
    "offers": [
        {"platform": "Direct", "provider": "Vendor", "input_per_1m": 2.0, "output_per_1m": 8.0,
         "cache_read_per_1m": 0.2, "notes": "long provider notes"},
        {"platform": "OpenRouter", "provider": "Ch" + "utes", "input_per_1m": 1.0,
         "output_per_1m": 4.0},
    ],
}}


def _bh(method, path, body):
    if path.startswith("/api/models/"):
        return 200, DOC
    if path.startswith("/api/benchmaxxing"):
        return 200, {"report": {"status": "scored", "score": 2.0}}
    if path == "/api/price-comparison":
        return 200, {"efficiency": {"global_io_ratio": {"value": 20.0}}}
    return 404, {"error": "not found"}


@pytest.fixture
def snapshot_file(tmp_path):
    path = tmp_path / "snap.json"
    path.write_text(json.dumps({
        "schema": 1, "source": "https://benchmarkheaven.com", "fetched_at": "2026-09-25T00:00:00Z",
        "fetched_at_epoch": time.time() - 3600,
        "documents": {f"/api/models/{MODEL_ID}": minimal_document(DOC)}}))
    return path


# -- the four provenance paths ------------------------------------------------
def test_network_then_fresh_cache(tmp_path, monkeypatch):
    allow_loopback(monkeypatch)
    with serve(_bh) as (url, requests):
        client = BenchmarkClient(base_url=url, cache_dir=tmp_path)
        doc, prov = client.model_with_provenance(MODEL_ID)
        assert prov.source == "network" and not prov.stale
        assert doc["id"] == MODEL_ID
        doc, prov = client.model_with_provenance(MODEL_ID)
        assert prov.source == "cache" and not prov.stale
        assert len(requests) == 1, "a fresh cache is not re-fetched"
        assert requests[0][1] == f"/api/models/{MODEL_ID}"


def test_stale_cache_when_the_api_is_down(tmp_path, monkeypatch):
    allow_loopback(monkeypatch)
    with serve(_bh) as (url, _requests):
        client = BenchmarkClient(base_url=url, cache_dir=tmp_path, timeout=0.5)
        client.model(MODEL_ID)
    two_days = time.time() - 2 * 24 * 3600
    for path in tmp_path.iterdir():
        os.utime(path, (two_days, two_days))
    # the server is gone now
    doc, prov = client.model_with_provenance(MODEL_ID)
    assert doc["id"] == MODEL_ID
    assert prov.source == "stale-cache" and prov.stale and prov.error


def test_bundled_snapshot_when_network_and_cache_are_unavailable(tmp_path, snapshot_file):
    client = BenchmarkClient(base_url="http://127.0.0.1:9", cache_dir=tmp_path, timeout=0.3,
                             snapshot=snapshot_file)
    doc, prov = client.model_with_provenance(MODEL_ID)
    assert prov.source == "bundled-snapshot"
    assert prov.stale and prov.usable
    assert prov.to_dict()["source"] == "bundled-snapshot"
    assert 3000 < prov.age_seconds < 4000
    assert doc["benchmarks"]["aa_intelligence_index"] == 40.0
    # an id the snapshot does not hold is still missing
    assert client.model_with_provenance("other::x")[1].source == "missing"


def test_snapshot_only_stands_in_for_its_own_origin(tmp_path, monkeypatch):
    monkeypatch.delenv("AUTO_ROUTER_BENCH_SNAPSHOT", raising=False)
    assert BenchmarkClient(cache_dir=tmp_path).snapshot_path == bench.SNAPSHOT_PATH
    assert BenchmarkClient(base_url="https://other.invalid", cache_dir=tmp_path).snapshot_path is None
    monkeypatch.setenv("AUTO_ROUTER_BENCH_SNAPSHOT", "0")
    assert BenchmarkClient(cache_dir=tmp_path).snapshot_path is None


def test_the_bundled_snapshot_is_minimal_and_attributed():
    snap = json.loads(bench.SNAPSHOT_PATH.read_text())
    assert snap["source"] == "https://benchmarkheaven.com"
    assert snap["fetched_at"] and snap["fetched_at_epoch"] > 0
    terra = snap["documents"]["/api/models/gpt-5.6-terra::max"]["model"]
    assert terra["benchmarks"]["aa_intelligence_index"] > 0
    for doc in snap["documents"].values():
        model = doc.get("model") or {}
        assert "notes" not in json.dumps(model.get("offers") or [])
        assert "input_output_ratio" not in json.dumps(model)


def test_default_client_falls_back_to_the_real_bundled_snapshot(tmp_path):
    client = BenchmarkClient(cache_dir=tmp_path, offline=True)
    doc, prov = client.model_with_provenance("gpt-5.6-terra::max")
    assert prov.source == "bundled-snapshot"
    assert intelligence_index(doc) is not None


# -- intelligence, tokens per task, cost per task -------------------------------
def test_task_tokens_and_cost_per_task():
    model = DOC["model"]
    assert intelligence_index(model) == 40.0
    assert task_tokens(model).output == 10000
    cost = cost_per_task(model, 2.0, 8.0, cache_read_per_1m=0.2, io_ratio=20.0, cache_hit_rate=0.5)
    # 10k out, 200k in, half of it read from cache
    expected = (100_000 * 2.0 + 100_000 * 0.2 + 10_000 * 8.0) / 1e6
    assert cost["usd_per_task"] == pytest.approx(expected)
    assert cost["measured_tokens"] and not cost["assumptions"]
    unmeasured = cost_per_task({"benchmarks": {}}, 1.0, 1.0)
    assert unmeasured["output_tokens"] == 1000 and len(unmeasured["assumptions"]) == 2
    stale = {"token_efficiency": {"aa": {"tokens_per_task": {"value": {"output": 5}, "stale": True}}}}
    assert task_tokens(stale) is None


def test_minimal_document_keeps_only_what_the_router_reads():
    doc = minimal_document(DOC)["model"]
    assert [o["provider"] for o in doc["offers"]] == ["Vendor"]
    assert "notes" not in doc["offers"][0]
    assert doc["aa_metadata"] == {"context_window_tokens": 400000}
    assert doc["token_efficiency"]["aa"]["tokens_per_task"]["value"] == {"output": 10000}
    assert "input_output_ratio" not in json.dumps(doc)


def test_global_io_ratio(tmp_path, monkeypatch):
    allow_loopback(monkeypatch)
    with serve(_bh) as (url, _requests):
        ratio, prov = BenchmarkClient(base_url=url, cache_dir=tmp_path).global_io_ratio()
    assert ratio == 20.0 and prov.source == "network"


# -- command line -----------------------------------------------------------------
def test_cli_show_and_refresh(tmp_path, monkeypatch, capsys):
    allow_loopback(monkeypatch)
    monkeypatch.setenv("AUTO_ROUTER_CACHE_DIR", str(tmp_path))
    with serve(_bh) as (url, requests):
        monkeypatch.setenv("AUTO_ROUTER_BENCH_URL", url)
        assert bench.main(["--refresh", MODEL_ID]) == 0
        assert f"{MODEL_ID}: network" in capsys.readouterr().out
        assert bench.main(["--show", MODEL_ID]) == 0
    shown = json.loads(capsys.readouterr().out)
    assert shown["intelligence_index"] == 40.0
    assert shown["task_tokens"]["output_tokens_per_task"] == 10000
    assert shown["cost_per_task"]["usd_per_task"] > 0
    assert shown["provenance"]["source"] == "cache"
    assert any(r[1] == "/api/price-comparison" for r in requests)


def test_cli_show_offline_uses_the_snapshot(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("AUTO_ROUTER_CACHE_DIR", str(tmp_path))
    assert bench.main(["--offline", "--show", "glm-5.3-flash::default"]) == 0
    shown = json.loads(capsys.readouterr().out)
    assert shown["provenance"]["source"] == "bundled-snapshot"


def test_write_snapshot(tmp_path, monkeypatch):
    allow_loopback(monkeypatch)
    monkeypatch.setenv("AUTO_ROUTER_CACHE_DIR", str(tmp_path))
    out = tmp_path / "snap.json"
    with serve(_bh) as (url, _requests):
        monkeypatch.setenv("AUTO_ROUTER_BENCH_URL", url)
        assert bench.main(["--refresh", MODEL_ID, "--write-snapshot", str(out)]) == 0
    snap = json.loads(out.read_text())
    assert set(snap["documents"]) == {f"/api/models/{MODEL_ID}",
                                      f"/api/benchmaxxing?report={MODEL_ID}",
                                      "/api/price-comparison"}
    assert "Ch" + "utes" not in out.read_text()


# -- config: token appetite, enabled list, write premium -----------------------
def _config(tmp_path, models, **extra):
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps({
        "providers": {"host": {"base_url": "https://example.invalid/v1", "cache": "openai"}},
        "models": models, **extra}))
    return path


def _doc_with_tokens(output):
    doc = json.loads(json.dumps(DOC))
    doc["model"]["token_efficiency"]["aa"]["tokens_per_task"]["value"]["output"] = output
    return doc


def _client_with_docs(tmp_path, docs):
    client = BenchmarkClient(base_url="http://127.0.0.1:9", cache_dir=tmp_path, offline=True,
                             snapshot=False)
    for bench_id, doc in docs.items():
        client._cache_path(f"/api/models/{bench_id}").write_text(json.dumps(doc))
    return client


def test_token_appetite_scales_expected_output(tmp_path):
    client = _client_with_docs(tmp_path, {"terse::x": _doc_with_tokens(5000),
                                          "verbose::x": _doc_with_tokens(20000)})
    cfg = load_config(_config(tmp_path, [
        {"name": "terse", "provider": "host", "bench_id": "terse::x"},
        {"name": "verbose", "provider": "host", "bench_id": "verbose::x"},
        {"name": "unmeasured", "provider": "host", "prices": {"input": 2, "output": 8}},
    ]), bench=client)
    terse, verbose = cfg.catalog["terse"], cfg.catalog["verbose"]
    assert terse.output_appetite == 0.4 and verbose.output_appetite == 1.6   # median 12,500
    assert cfg.catalog["unmeasured"].output_appetite == 1.0
    assert verbose.evidence["output_appetite"]["tokens_per_task"] == 20000
    ctx = Context(cfg.catalog, SuccessModel())
    req = TurnRequest("coding", 0.5, prompt_tokens=500, output_tokens=4000, now=0.0)
    # same prices, so the cost difference is the output share only
    out_price = terse.prices.output / 1e6
    diff = turn_call_cost(verbose, req, 0, ctx) - turn_call_cost(terse, req, 0, ctx)
    assert diff == pytest.approx(4000 * (1.6 - 0.4) * out_price)


def test_token_appetite_can_be_switched_off(tmp_path):
    client = _client_with_docs(tmp_path, {"a::x": _doc_with_tokens(5000),
                                          "b::x": _doc_with_tokens(20000)})
    cfg = load_config(_config(tmp_path, [
        {"name": "a", "provider": "host", "bench_id": "a::x"},
        {"name": "b", "provider": "host", "bench_id": "b::x"}],
        policy={"token_appetite": False}), bench=client)
    assert {m.output_appetite for m in cfg.catalog.all()} == {1.0}


def test_enabled_list_and_env_override(tmp_path, monkeypatch):
    models = [{"name": n, "provider": "host", "prices": {"input": 1, "output": 2}}
              for n in ("a", "b", "c")]
    cfg = load_config(_config(tmp_path, models, enabled=["a", "c"]), use_bench=False)
    assert sorted(m.name for m in cfg.catalog.all()) == ["a", "c"]
    monkeypatch.setenv("AUTO_ROUTER_MODELS", "b, c")
    cfg = load_config(_config(tmp_path, models, enabled=["a"]), use_bench=False)
    assert sorted(m.name for m in cfg.catalog.all()) == ["b", "c"]
    monkeypatch.setenv("AUTO_ROUTER_MODELS", "a,typo")
    with pytest.raises(ValueError, match="typo"):
        load_config(_config(tmp_path, models), use_bench=False)


def test_a_disabled_route_still_prices_a_plan_route(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTO_ROUTER_MODELS", "plan")
    cfg = load_config(_config(tmp_path, [
        {"name": "metered", "provider": "host", "prices": {"input": 4, "output": 20}},
        {"name": "plan", "provider": "host", "subscription": "claude",
         "list_price_model": "metered"}]), use_bench=False)
    assert [m.name for m in cfg.catalog.all()] == ["plan"]
    assert cfg.reference_models["metered"].prices.input == 4
    from auto_router.router import Router
    router = Router(cfg, classifier=None, judge=None)
    assert router.context().reference_catalog.get("metered") is not None


def test_cache_write_premium_can_be_dropped(tmp_path):
    doc = json.loads(json.dumps(DOC))
    doc["model"]["offers"][0]["cache_write_per_1m"] = 2.5
    client = _client_with_docs(tmp_path, {"w::x": doc})
    cfg = load_config(_config(tmp_path, [
        {"name": "premium", "provider": "host", "bench_id": "w::x",
         "bench_offer": {"platform": "Direct"}},
        {"name": "plain", "provider": "host", "bench_id": "w::x",
         "bench_offer": {"platform": "Direct"}, "cache_write_premium": False}]), bench=client)
    assert cfg.catalog["premium"].prices.write == 2.5
    assert cfg.catalog["plain"].prices.write == cfg.catalog["plain"].prices.input == 2.0


def test_example_models_file_loads_offline(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTO_ROUTER_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("AUTO_ROUTER_BENCH_OFFLINE", "1")
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg = load_config(os.path.join(root, "examples", "models.yaml"))
    names = {m.name for m in cfg.catalog.all()}
    assert {"opus-5.5-plan", "opus-5.5-api", "sonnet-5-plan", "sonnet-5-api", "gpt-6-luna",
            "gpt-6-astra", "glm-5.3-flash", "bonsai-2-local"} == names
    glm = cfg.catalog["glm-5.3-flash"]
    assert (glm.prices.input, glm.prices.read, glm.prices.output) == (0.20, 0.05, 0.50)
    assert glm.context_tokens == 1_048_576
    assert cfg.providers["tensorx"].base_url == "https://api.tensorx.ai/v1"
    assert cfg.providers["openai"].api_key_env == "OPENAI_API_KEY"
    bonsai = cfg.catalog["bonsai-2-local"]
    assert bonsai.prices.is_free and bonsai.local
    assert bonsai.capability_assumed_from == "qwen3.8-27b::medium"
    assert cfg.catalog["opus-5.5-plan"].subscription == "claude"
    ref = cfg.intelligence_reference
    assert ref["resolved_id"] == "gpt-5.6-terra::max"
    assert ref["source"] == "benchmark:bundled-snapshot"
    # every capability here came from the snapshot, so all of it is stale evidence
    assert all(m.evidence_stale for m in cfg.catalog.all())
