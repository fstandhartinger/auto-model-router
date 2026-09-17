"""Benchmark data client (Benchmark Heaven-compatible JSON API).

Pulls per-category capability, list prices per offer (including cache read and
cache write prices), context length, provider cache hit rates and a
benchmaxxing score for the models in the catalog.

Everything is cached on disk with a TTL. When the API is unreachable the last
good copy is used however old it is, and when there is no copy at all the
router falls back to what the provider config says. Routing must never fail
because a benchmark site is down.

Licensing note: some of the upstream numbers originate from third parties whose
terms restrict use in competing products. This client only reads what the API
serves; check the terms of the data you point it at before shipping a product.
"""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

log = logging.getLogger("auto_router.bench")

DEFAULT_BASE_URL = "https://benchmarkheaven.com"
DEFAULT_TTL_SECONDS = 24 * 3600


def _cache_dir() -> Path:
    root = os.environ.get("AUTO_ROUTER_CACHE_DIR") or os.path.join(
        os.path.expanduser("~"), ".cache", "auto-model-router")
    path = Path(root)
    path.mkdir(parents=True, exist_ok=True)
    return path


class BenchmarkClient:
    def __init__(self, base_url: str | None = None, ttl_seconds: int = DEFAULT_TTL_SECONDS,
                 timeout: float = 20.0, cache_dir: Path | None = None, offline: bool = False):
        self.base_url = (base_url or os.environ.get("AUTO_ROUTER_BENCH_URL") or DEFAULT_BASE_URL).rstrip("/")
        self.ttl = ttl_seconds
        self.timeout = timeout
        self.cache_dir = cache_dir or _cache_dir()
        self.offline = offline
        self.errors: list[str] = []

    # -- transport ---------------------------------------------------------
    def _cache_path(self, path: str) -> Path:
        safe = urllib.parse.quote(path, safe="")
        return self.cache_dir / f"{safe}.json"

    def get(self, path: str) -> Any | None:
        """GET a JSON document with a disk cache. Returns None when nothing is available."""
        cached = self._cache_path(path)
        if cached.exists() and (self.offline or time.time() - cached.stat().st_mtime < self.ttl):
            try:
                return json.loads(cached.read_text())
            except json.JSONDecodeError:
                pass
        if not self.offline:
            try:
                req = urllib.request.Request(f"{self.base_url}{path}",
                                             headers={"User-Agent": "auto-model-router/0.2"})
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    body = resp.read()
                data = json.loads(body)
                tmp = cached.with_suffix(".tmp")
                tmp.write_bytes(body)
                tmp.replace(cached)
                return data
            except Exception as exc:  # noqa: BLE001 - any failure means "use the stale copy"
                self.errors.append(f"{path}: {type(exc).__name__}")
                log.warning("benchmark API unavailable for %s (%s); using cached copy", path,
                            type(exc).__name__)
        if cached.exists():
            try:
                return json.loads(cached.read_text())
            except json.JSONDecodeError:
                return None
        return None

    # -- domain ------------------------------------------------------------
    def model(self, bench_id: str) -> dict | None:
        data = self.get(f"/api/models/{urllib.parse.quote(bench_id, safe=':')}")
        return (data or {}).get("model")

    def benchmaxxing(self, bench_id: str) -> float | None:
        data = self.get(f"/api/benchmaxxing?report={urllib.parse.quote(bench_id, safe=':')}")
        report = (data or {}).get("report") or {}
        if report.get("status") != "scored":
            return None
        score = report.get("score")
        return float(score) if isinstance(score, (int, float)) else None

    def endpoint_stats(self) -> dict:
        data = self.get("/api/price-comparison") or {}
        return ((data.get("efficiency") or {}).get("openrouter_endpoints")) or {}


def _pct(value: Any) -> float | None:
    if not isinstance(value, (int, float)):
        return None
    return float(value) * 100.0 if value <= 1.0 else float(value)


def capability_from_model(model: dict) -> dict[str, float]:
    """Map a model document onto the router's categories (roughly 0..100).

    Headline indexes (intelligence, coding, coding-agent, long-context
    reasoning) are preferred over per-variant category scores: the latter can
    rest on very few benchmarks for a given reasoning-effort variant and then
    rank a small model above a frontier one. Category scores are only a
    fallback. The success model is fitted per category, so the scales only need
    to be monotone, not identical.
    """
    cats = model.get("category_scores") or {}
    b = model.get("benchmarks") or {}

    def num(v: Any) -> float | None:
        return float(v) if isinstance(v, (int, float)) else None

    def first(*vals: Any) -> float | None:
        for v in vals:
            if isinstance(v, (int, float)):
                return float(v)
        return None

    ii = num(b.get("aa_intelligence_index"))
    general = ii * 1.5 if ii is not None else None
    tb = num(b.get("aa_terminalbench_v2_1"))
    tau2 = _pct(b.get("aa_tau2"))
    out: dict[str, float | None] = {
        "coding": first(b.get("aa_coding_index"), cats.get("cat_coding")),
        "agentic": first(b.get("aa_coding_agent_index"), tb * 72 if tb is not None else None,
                         cats.get("cat_agentic")),
        "math": first(b.get("aa_math_index"), general, cats.get("cat_science")),
        "knowledge": first(general, cats.get("cat_science"), _pct(b.get("aa_gpqa"))),
        "long_context": first(_pct(b.get("aa_lcr")), cats.get("cat_long_context")),
        "tool_use": first(tau2, general, cats.get("cat_agentic")),
    }
    known = [v for v in out.values() if v is not None]
    out["general"] = general if general is not None else (sum(known) / len(known) if known else None)
    return {k: round(v, 2) for k, v in out.items() if v is not None}


def pick_offer(model: dict, platform: str | None = None, provider: str | None = None) -> dict | None:
    """Choose the list-price offer that matches a configured endpoint.

    Exact platform+provider match first, then platform only, then the cheapest
    offer that publishes cache prices, then the cheapest offer.
    """
    offers = [o for o in (model.get("offers") or [])
              if isinstance(o.get("input_per_1m"), (int, float))]
    if not offers:
        return None

    def norm(s: Any) -> str:
        return str(s or "").strip().lower()

    if platform:
        exact = [o for o in offers if norm(o.get("platform")) == norm(platform)
                 and (not provider or norm(o.get("provider")) == norm(provider))]
        if exact:
            return min(exact, key=lambda o: o["input_per_1m"])
        loose = [o for o in offers if norm(o.get("platform")) == norm(platform)]
        if loose:
            return min(loose, key=lambda o: o["input_per_1m"])
    with_cache = [o for o in offers if isinstance(o.get("cache_read_per_1m"), (int, float))]
    pool = with_cache or offers
    return min(pool, key=lambda o: o["input_per_1m"] * 3 + (o.get("output_per_1m") or 0))


def context_length(model: dict) -> int | None:
    meta = model.get("aa_metadata") or {}
    ctx = meta.get("context_window_tokens")
    if isinstance(ctx, int) and ctx > 0:
        return ctx
    lengths = [o.get("context_length") for o in model.get("offers") or []
               if isinstance(o.get("context_length"), int)]
    return max(lengths) if lengths else None


def endpoint_hit_rate(stats: dict, or_model_id: str | None) -> float | None:
    """Workload-weighted cache hit rate across a model's public endpoints."""
    if not or_model_id:
        return None
    endpoints = stats.get(or_model_id) or {}
    num = den = 0.0
    for ep in endpoints.values():
        chr_ = (ep or {}).get("cache_hit_rate") or {}
        value, weight = chr_.get("value"), chr_.get("total_tokens") or 0
        if isinstance(value, (int, float)) and weight:
            num += value * weight
            den += weight
    return num / den if den else None
