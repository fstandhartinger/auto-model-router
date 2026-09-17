import json

from auto_router.bench import BenchmarkClient, capability_from_model, endpoint_hit_rate, pick_offer
from auto_router.config import load_config

DOC = {"model": {
    "id": "example-model::max",
    "category_scores": {"cat_coding": 60.0, "cat_agentic": 50.0, "cat_science": 55.0, "cat_long_context": 45.0},
    "benchmarks": {"aa_math_index": 70.0, "aa_tau2": 0.8, "aa_intelligence_index": 40.0,
                   "aa_coding_index": 65.0},
    "aa_metadata": {"context_window_tokens": 400000},
    "offers": [
        {"platform": "Direct", "provider": "Vendor", "input_per_1m": 2.0, "output_per_1m": 8.0},
        {"platform": "OpenRouter", "provider": "HostA", "input_per_1m": 1.5, "output_per_1m": 6.0,
         "cache_read_per_1m": 0.15, "cache_write_per_1m": None},
    ],
}}


def test_capability_mapping():
    cap = capability_from_model(DOC["model"])
    assert cap["coding"] == 65.0, "headline index wins over the category score"
    assert cap["math"] == 70.0 and cap["long_context"] == 45.0 and cap["tool_use"] == 80.0
    assert cap["general"] == 60.0 and cap["agentic"] == 50.0


def test_offer_selection():
    assert pick_offer(DOC["model"], "Direct")["input_per_1m"] == 2.0
    assert pick_offer(DOC["model"])["cache_read_per_1m"] == 0.15   # prefers offers with cache prices


def test_hit_rate_is_token_weighted():
    stats = {"m": {"a": {"cache_hit_rate": {"value": 0.9, "total_tokens": 900}},
                   "b": {"cache_hit_rate": {"value": 0.0, "total_tokens": 100}}}}
    assert abs(endpoint_hit_rate(stats, "m") - 0.81) < 1e-9


def _client_with(tmp_path, offline):
    client = BenchmarkClient(base_url="http://127.0.0.1:9", cache_dir=tmp_path, offline=offline, timeout=0.5)
    client._cache_path("/api/models/example-model::max").write_text(json.dumps(DOC))
    client._cache_path("/api/benchmaxxing?report=example-model::max").write_text(
        json.dumps({"report": {"status": "scored", "score": 6.0}}))
    return client


def test_router_survives_an_unreachable_api(tmp_path):
    client = _client_with(tmp_path, offline=False)
    # make the cache look stale so the client tries the (unreachable) network first
    import os
    for p in tmp_path.iterdir():
        os.utime(p, (0, 0))
    assert client.model("example-model::max")["id"] == "example-model::max"
    assert client.errors, "the failed fetch is recorded"


def test_config_merges_bench_data_and_overrides(tmp_path):
    client = _client_with(tmp_path, offline=True)
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps({
        "providers": {"host": {"base_url": "https://example.invalid/v1", "api_key_env": "EXAMPLE_KEY",
                               "cache": "openai"}},
        "models": [
            {"name": "x", "provider": "host", "upstream_id": "vendor/x", "bench_id": "example-model::max",
             "bench_offer": {"platform": "OpenRouter", "provider": "HostA"}, "cache": {"hit_rate": 0.97}},
            {"name": "free-x", "provider": "host", "bench_id": "example-model::max", "free": True,
             "capability": {"coding": 40}},
        ]}))
    cfg = load_config(cfg_path, bench=client)
    x = cfg.catalog["x"]
    assert x.prices.input == 1.5 and x.prices.read == 0.15 and x.prices.write == 1.5
    assert x.cache.hit_rate == 0.97 and x.cache.ttl_seconds == 300
    assert x.context_tokens == 400000 and x.benchmaxxing == 6.0
    free = cfg.catalog["free-x"]
    assert free.prices.is_free and free.capability["coding"] == 40
