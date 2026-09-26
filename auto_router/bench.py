"""Benchmark data client (Benchmark Heaven-compatible JSON API).

Pulls per-category capability, list prices per offer (including cache read and
cache write prices), context length, provider cache hit rates, the measured
output tokens per benchmark task (the input to Benchmark Heaven's cost per
task) and a benchmaxxing score for the models in the catalog.

Endpoints read (all public, no key; checked against benchmarkheaven.com on
25 Sep 2026):

* ``GET /api/models/{id}`` - one model document. ``id`` is a variant id
  (``gpt-5.6-terra::max``) or a family key (``gpt-5.6-terra``, which the site
  resolves to the family's first listed variant).
* ``GET /api/benchmaxxing?report={id}`` - the benchmaxxing report.
* ``GET /api/price-comparison`` - only ``efficiency.global_io_ratio``, the one
  input:output ratio the site applies to every model's cost per task.

Everything is cached on disk with a TTL. When the API is unreachable the last
good copy is used for at most seven days by default. With no usable copy the
client falls back to a small snapshot bundled with the package
(``auto_router/data/bench-snapshot.json``, provenance ``bundled-snapshot``,
always treated as stale), and only then to what the provider config says.
Routing must never fail because a benchmark site is down.

Command line::

    python -m auto_router.bench --show gpt-5.6-terra::max
    python -m auto_router.bench --refresh            # every bench id in $AUTO_ROUTER_CONFIG
    python -m auto_router.bench --refresh glm-5.3-flash::default

Licensing note: some of the upstream numbers originate from third parties whose
terms restrict use in competing products. This client only reads what the API
serves; check the terms of the data you point it at before shipping a product.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import sys
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger("auto_router.bench")

DEFAULT_BASE_URL = "https://benchmarkheaven.com"
DEFAULT_TTL_SECONDS = 24 * 3600
DEFAULT_MAX_STALE_SECONDS = 7 * 24 * 3600

#: The bundled last-resort copy of the documents the example configuration
#: needs. Regenerate with ``python -m auto_router.bench --write-snapshot``.
SNAPSHOT_PATH = Path(__file__).resolve().parent / "data" / "bench-snapshot.json"


@dataclass(frozen=True)
class Provenance:
    """Where one benchmark document came from and how old it is.

    ``source`` is one of ``network`` (fetched now), ``cache`` (fresh disk copy
    inside the TTL), ``stale-cache`` (older than the TTL but inside the
    stale limit, used because the API could not be reached),
    ``bundled-snapshot`` (the copy shipped with the package, used when neither
    the network nor the disk cache had anything) or ``missing`` (nothing
    usable). ``stale`` is what the router acts on: it lowers the confidence
    attached to every capability derived from this document.
    """

    origin: str
    path: str
    source: str
    age_seconds: float | None = None
    fetched_at: float | None = None
    error: str | None = None

    @property
    def stale(self) -> bool:
        return self.source in ("stale-cache", "bundled-snapshot", "missing")

    @property
    def usable(self) -> bool:
        return self.source != "missing"

    def to_dict(self) -> dict:
        return {"origin": self.origin, "path": self.path, "source": self.source,
                "age_seconds": None if self.age_seconds is None else round(self.age_seconds, 1),
                "stale": self.stale, "error": self.error}


def _cache_dir() -> Path:
    root = os.environ.get("AUTO_ROUTER_CACHE_DIR") or os.path.join(
        os.path.expanduser("~"), ".cache", "auto-model-router")
    path = Path(root)
    path.mkdir(parents=True, exist_ok=True)
    return path


_SNAPSHOTS: dict[str, dict | None] = {}


def load_snapshot(path: str | Path = SNAPSHOT_PATH) -> dict | None:
    """The bundled snapshot, parsed once per process. None when absent or unreadable."""
    key = str(path)
    if key not in _SNAPSHOTS:
        try:
            data = json.loads(Path(path).read_text())
            _SNAPSHOTS[key] = data if isinstance(data, dict) else None
        except (OSError, json.JSONDecodeError):
            _SNAPSHOTS[key] = None
    return _SNAPSHOTS[key]


def _snapshot_default() -> bool:
    return os.environ.get("AUTO_ROUTER_BENCH_SNAPSHOT", "1") not in ("0", "false", "no")


class BenchmarkClient:
    def __init__(self, base_url: str | None = None, ttl_seconds: int = DEFAULT_TTL_SECONDS,
                 timeout: float = 20.0, cache_dir: Path | None = None, offline: bool = False,
                 max_stale_seconds: int = DEFAULT_MAX_STALE_SECONDS,
                 snapshot: bool | str | Path | None = None):
        self.base_url = (base_url or os.environ.get("AUTO_ROUTER_BENCH_URL") or DEFAULT_BASE_URL).rstrip("/")
        self.ttl = ttl_seconds
        self.timeout = timeout
        self.cache_dir = cache_dir or _cache_dir()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_stale = max(0, max_stale_seconds)
        self.offline = offline
        self.errors: list[str] = []
        self._snapshot_warned: set[str] = set()
        #: Where the bundled last-resort copy comes from. By default it is only
        #: used for the origin it was taken from: a snapshot of one site's
        #: numbers must not stand in for a different API the operator chose.
        if snapshot is None:
            snapshot = _snapshot_default() and self.base_url == DEFAULT_BASE_URL
        self.snapshot_path: Path | None = (
            None if snapshot is False else SNAPSHOT_PATH if snapshot is True else Path(snapshot))

    # -- transport ---------------------------------------------------------
    def _cache_path(self, path: str) -> Path:
        safe = hashlib.sha256(f"{self.base_url}{path}".encode()).hexdigest()
        return self.cache_dir / f"{safe}.json"

    def fetch(self, path: str) -> tuple[Any | None, Provenance]:
        """GET a JSON document with a disk cache, reporting where the answer came from.

        Order: a fresh disk copy inside the TTL, then the network, then a stale
        disk copy up to ``max_stale``. Anything older is treated as missing so
        that a long outage degrades to the configured fallback instead of
        routing on last month's capability numbers.
        """
        cached = self._cache_path(path)
        age = time.time() - cached.stat().st_mtime if cached.exists() else float("inf")
        fresh_enough = cached.exists() and 0 <= age <= self.max_stale
        if fresh_enough and (self.offline or age < self.ttl):
            data = self._read(cached)
            if data is not None:
                # Offline mode lets an old copy be *used*; it does not make it
                # fresh. The source is decided by the age alone, so an offline
                # router still discounts what it is routing on.
                source = "cache" if age < self.ttl else "stale-cache"
                return data, Provenance(self.base_url, path, source, age, time.time() - age)
        error: str | None = None
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
                now = time.time()
                return data, Provenance(self.base_url, path, "network", 0.0, now)
            except Exception as exc:  # noqa: BLE001 - any failure means "use the stale copy"
                error = type(exc).__name__
                self.errors.append(f"{path}: {error}")
                log.warning("benchmark API unavailable for %s (%s); using cached copy", path,
                            error)
        if fresh_enough:
            data = self._read(cached)
            if data is not None:
                source = "cache" if age < self.ttl else "stale-cache"
                return data, Provenance(self.base_url, path, source, age, time.time() - age,
                                        error)
        reason = error or ("cache expired" if cached.exists() else "no cached copy")
        snap = self._snapshot(path)
        if snap is not None:
            data, fetched = snap
            if path not in self._snapshot_warned:
                self._snapshot_warned.add(path)
                log.warning("benchmark data for %s: no network and no usable cache; using the "
                            "bundled snapshot", path)
            snap_age = None if fetched is None else max(0.0, time.time() - fetched)
            return data, Provenance(self.base_url, path, "bundled-snapshot", snap_age, fetched,
                                    reason)
        return None, Provenance(self.base_url, path, "missing",
                                None if age == float("inf") else age, None, reason)

    def _snapshot(self, path: str) -> tuple[Any, float | None] | None:
        if self.snapshot_path is None:
            return None
        snap = load_snapshot(self.snapshot_path)
        if not snap:
            return None
        doc = (snap.get("documents") or {}).get(path)
        if doc is None:
            return None
        return doc, _number(snap.get("fetched_at_epoch"))

    @staticmethod
    def _read(cached: Path) -> Any | None:
        try:
            return json.loads(cached.read_text())
        except (json.JSONDecodeError, OSError):
            return None

    def get(self, path: str) -> Any | None:
        """``fetch`` without the provenance. Returns None when nothing is available."""
        return self.fetch(path)[0]

    # -- domain ------------------------------------------------------------
    def model_with_provenance(self, bench_id: str) -> tuple[dict | None, Provenance]:
        data, prov = self.fetch(f"/api/models/{urllib.parse.quote(bench_id, safe=':')}")
        return (data or {}).get("model"), prov

    def model(self, bench_id: str) -> dict | None:
        return self.model_with_provenance(bench_id)[0]

    def benchmaxxing_with_provenance(self, bench_id: str) -> tuple[float | None, Provenance]:
        """The benchmaxxing penalty and where it came from.

        It is fetched separately from the model document and can be stale while
        the model document is fresh, so it carries its own provenance rather
        than inheriting the document's.
        """
        data, prov = self.fetch(f"/api/benchmaxxing?report={urllib.parse.quote(bench_id, safe=':')}")
        report = (data or {}).get("report") or {}
        if report.get("status") != "scored":
            return None, prov
        return _number(report.get("score")), prov

    def benchmaxxing(self, bench_id: str) -> float | None:
        return self.benchmaxxing_with_provenance(bench_id)[0]

    def endpoint_stats(self) -> dict:
        data = self.get("/api/price-comparison") or {}
        return ((data.get("efficiency") or {}).get("openrouter_endpoints")) or {}

    def global_io_ratio(self) -> tuple[float | None, Provenance]:
        """The one input:output token ratio the site prices every model's task with."""
        data, prov = self.fetch("/api/price-comparison")
        ratio = ((data or {}).get("efficiency") or {}).get("global_io_ratio") or {}
        value = _number(ratio.get("value"))
        return (value if value is not None and value >= 0 else None), prov


def _number(value: Any) -> float | None:
    """A finite float, or None.

    Python's JSON decoder happily produces ``NaN`` and ``Infinity``, and both
    pass ``isinstance(value, float)``. A NaN capability silently poisons every
    comparison in the policy (``NaN > x`` is always False), so malformed
    upstream data is rejected here rather than ranked.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _count(value: Any) -> int:
    number = _number(value)
    return int(number) if number is not None else 0


def _pct(value: Any) -> float | None:
    number = _number(value)
    if number is None:
        return None
    return number * 100.0 if number <= 1.0 else number


@dataclass(frozen=True)
class CapabilityEvidence:
    """One category's capability score and exactly what it rests on.

    ``strength`` is ``direct`` when a benchmark measures this category,
    ``derived`` when it is computed from a neighbouring headline index, and
    ``weak`` when the number exists but rests on very little data (for example
    a design Elo from a handful of battles). The router discounts anything that
    is not ``direct`` rather than pretending all evidence is equal.
    """

    value: float
    basis: str
    strength: str = "direct"

    def to_dict(self) -> dict:
        return {"value": round(self.value, 2), "basis": self.basis, "strength": self.strength}


#: Elo is centred on 1200 and a 400-point gap is 10:1 odds. The router only
#: needs a monotone 0..100 axis comparable with the other category scores, so
#: an Elo is mapped affinely with 1200 == 50 and 400 Elo == 100 points.
DESIGN_ELO_CENTRE = 1200.0
DESIGN_ELO_PER_POINT = 4.0

#: Below this many recorded battles a design Elo is reported as weak evidence.
DESIGN_MIN_BATTLES = 200


def design_capability(model: dict) -> CapabilityEvidence | None:
    """Web/UI-design capability from the served design-arena Elo, or None.

    The router must not hard-code "model X is the design model". This reads
    whatever the configured benchmark API serves for that model and returns
    nothing at all when there is no design evidence, which is what triggers the
    documented fallback. ``frontend`` and ``fullstack`` boards are averaged
    when both are present.
    """
    arena = model.get("designarena")
    if not isinstance(arena, dict) or not arena:
        return None
    elos: list[float] = []
    battles = 0
    boards: list[str] = []
    for board in ("frontend", "fullstack"):
        entry = arena.get(board)
        if not isinstance(entry, dict):
            continue
        elo = _number(entry.get("elo"))
        if elo is None:
            continue
        elos.append(elo)
        boards.append(board)
        battles += _count(entry.get("battles"))
    if not elos:
        return None
    elo = sum(elos) / len(elos)
    value = max(0.0, min(100.0, 50.0 + (elo - DESIGN_ELO_CENTRE) / DESIGN_ELO_PER_POINT))
    strength = "direct" if battles >= DESIGN_MIN_BATTLES else "weak"
    basis = (f"designarena {'+'.join(boards)} elo {elo:.0f} over {battles} battles"
             f" -> 50+(elo-{DESIGN_ELO_CENTRE:.0f})/{DESIGN_ELO_PER_POINT:.0f}")
    return CapabilityEvidence(round(value, 2), basis, strength)


def capability_evidence(model: dict) -> dict[str, CapabilityEvidence]:
    """Map a model document onto the router's categories (roughly 0..100), with bases.

    Headline indexes (intelligence, coding, coding-agent, long-context
    reasoning) are preferred over per-variant category scores: the latter can
    rest on very few benchmarks for a given reasoning-effort variant and then
    rank a small model above a frontier one. Category scores are only a
    fallback. The success model is fitted per category, so the scales only need
    to be monotone, not identical.
    """
    cats = model.get("category_scores") or {}
    b = model.get("benchmarks") or {}

    def pick(*options: tuple[str, Any, str]) -> CapabilityEvidence | None:
        """First option whose value is a finite number. Each option is (basis, value, strength)."""
        for basis, value, strength in options:
            number = _number(value)
            if number is not None:
                return CapabilityEvidence(number, basis, strength)
        return None

    ii = _number(b.get("aa_intelligence_index"))
    general = ii * 1.5 if ii is not None else None
    general_basis = "aa_intelligence_index x 1.5"
    tb = _number(b.get("aa_terminalbench_v2_1"))
    tb_scaled = tb * 72 if tb is not None else None
    tau2 = _pct(b.get("aa_tau2"))

    out: dict[str, CapabilityEvidence | None] = {
        "coding": pick(("aa_coding_index", b.get("aa_coding_index"), "direct"),
                       ("category_scores.cat_coding", cats.get("cat_coding"), "direct")),
        "agentic": pick(("aa_coding_agent_index", b.get("aa_coding_agent_index"), "direct"),
                        ("aa_terminalbench_v2_1 x 72", tb_scaled, "derived"),
                        ("category_scores.cat_agentic", cats.get("cat_agentic"), "direct")),
        "math": pick(("aa_math_index", b.get("aa_math_index"), "direct"),
                     (general_basis, general, "derived"),
                     ("category_scores.cat_science", cats.get("cat_science"), "derived")),
        "knowledge": pick((general_basis, general, "derived"),
                          ("category_scores.cat_science", cats.get("cat_science"), "direct"),
                          ("aa_gpqa %", _pct(b.get("aa_gpqa")), "direct")),
        "long_context": pick(("aa_lcr %", _pct(b.get("aa_lcr")), "direct"),
                             ("category_scores.cat_long_context", cats.get("cat_long_context"),
                              "direct")),
        "tool_use": pick(("aa_tau2 %", tau2, "direct"),
                         (general_basis, general, "derived"),
                         ("category_scores.cat_agentic", cats.get("cat_agentic"), "derived")),
    }
    design = design_capability(model)
    if design is not None:
        out["design"] = design

    coding = out["coding"]
    long_ctx = out["long_context"]
    # Summarisation has no dedicated public benchmark in this feed. It is a
    # long-context reading job with a general-writing tail, so it is derived
    # from both and always reported as derived evidence, never as measured.
    parts = [e.value for e in (long_ctx, out["knowledge"]) if e is not None]
    if general is not None:
        parts.append(general)
    if parts:
        out["summarisation"] = CapabilityEvidence(
            round(sum(parts) / len(parts), 2),
            "mean(long_context, knowledge, " + general_basis + ")", "derived")

    known = [e.value for e in out.values() if e is not None]
    if general is not None:
        out["general"] = CapabilityEvidence(general, general_basis, "derived")
    elif known:
        out["general"] = CapabilityEvidence(round(sum(known) / len(known), 2),
                                            "mean of available category scores", "derived")
    # Design falls back to coding evidence only when there is no design data at
    # all; it is explicitly marked derived so the router discounts it.
    if "design" not in out and coding is not None:
        out["design"] = CapabilityEvidence(coding.value, "fallback: " + coding.basis, "derived")
    return {k: v for k, v in out.items() if v is not None}


def capability_from_model(model: dict) -> dict[str, float]:
    """Backwards-compatible view: category -> score, without the evidence bases."""
    return {k: round(e.value, 2) for k, e in capability_evidence(model).items()}


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


def intelligence_index(model: dict) -> float | None:
    """The headline intelligence number (``benchmarks.aa_intelligence_index``), or None."""
    return _number((model.get("benchmarks") or {}).get("aa_intelligence_index"))


@dataclass(frozen=True)
class TaskTokens:
    """Measured output tokens one benchmark task costs this model, with its basis.

    This is the model's own "token appetite": a reasoning model that writes
    40,000 tokens per task costs more per task than its list price suggests
    next to one that writes 5,000. Benchmark Heaven's cost per task is built
    from exactly this number (see ``cost_per_task``).
    """

    output: float
    collected_at: str | None = None
    basis: str | None = None
    scope: str | None = None

    def to_dict(self) -> dict:
        return {"output_tokens_per_task": round(self.output, 1), "collected_at": self.collected_at,
                "basis": self.basis, "scope": self.scope}


def task_tokens(model: dict) -> TaskTokens | None:
    """``token_efficiency.aa.tokens_per_task`` when present, fresh and positive."""
    aa = ((model.get("token_efficiency") or {}).get("aa")) or {}
    tokens = aa.get("tokens_per_task") or {}
    if not isinstance(tokens, dict) or tokens.get("stale"):
        return None
    output = _number((tokens.get("value") or {}).get("output"))
    if output is None or output <= 0:
        return None
    return TaskTokens(output, tokens.get("collected_at"), tokens.get("basis"), tokens.get("scope"))


#: What the site itself assumes when it has no measurement: a 1,000-token task
#: at a 10:1 input:output mix. Used only for display, never for routing.
FALLBACK_OUTPUT_TOKENS = 1000
FALLBACK_IO_RATIO = 10.0


def cost_per_task(model: dict, input_per_1m: float, output_per_1m: float,
                  cache_read_per_1m: float | None = None, io_ratio: float | None = None,
                  cache_hit_rate: float = 0.0) -> dict:
    """Benchmark Heaven-style cost of one benchmark task at the given prices.

    ``output = tokens per task``, ``input = output x io_ratio``; input tokens
    that hit the cache are priced at the cache-read price. Every fallback is
    named in ``assumptions``. A modelled workload, not a measured bill.
    """
    assumptions: list[str] = []
    tokens = task_tokens(model)
    out = tokens.output if tokens else float(FALLBACK_OUTPUT_TOKENS)
    if tokens is None:
        assumptions.append(f"no measured tokens per task: {FALLBACK_OUTPUT_TOKENS}-token task assumed")
    ratio = io_ratio if io_ratio is not None else FALLBACK_IO_RATIO
    if io_ratio is None:
        assumptions.append(f"no global input:output ratio: {FALLBACK_IO_RATIO:g}:1 assumed")
    inp = out * ratio
    hit = max(0.0, min(1.0, cache_hit_rate)) if cache_read_per_1m is not None else 0.0
    read_price = cache_read_per_1m if cache_read_per_1m is not None else input_per_1m
    usd = (inp * (1 - hit) * input_per_1m + inp * hit * read_price + out * output_per_1m) / 1e6
    return {"usd_per_task": round(usd, 6), "output_tokens": round(out, 1),
            "input_tokens": round(inp, 1), "io_ratio": ratio, "cache_hit_rate": hit,
            "measured_tokens": tokens is not None, "assumptions": assumptions}


# -- bundled snapshot ---------------------------------------------------------
#: Offer fields the router reads. Everything else (notes, source URLs, region
#: details) stays out of the snapshot.
_OFFER_FIELDS = ("platform", "provider", "input_per_1m", "output_per_1m", "cache_read_per_1m",
                 "cache_write_per_1m", "context_length")
#: Hosts whose names must not appear in files this repository ships.
_SNAPSHOT_EXCLUDED_HOSTS = tuple(bytes.fromhex(h).decode() for h in ("6368757465",))


def minimal_document(doc: dict) -> dict:
    """Only the fields the router uses: capability inputs, prices, context, task tokens."""
    model = dict(doc.get("model") or {})
    offers = []
    for offer in model.get("offers") or []:
        label = f"{offer.get('platform', '')} {offer.get('provider', '')}".lower()
        if any(host in label for host in _SNAPSHOT_EXCLUDED_HOSTS):
            continue
        if not isinstance(offer.get("input_per_1m"), (int, float)):
            continue
        offers.append({k: offer[k] for k in _OFFER_FIELDS if offer.get(k) is not None})
    out = {k: model[k] for k in ("id", "family_key", "display_name", "variant", "open_weights",
                                 "release_date", "benchmarks", "category_scores", "designarena")
           if model.get(k) is not None}
    ctx = (model.get("aa_metadata") or {}).get("context_window_tokens")
    if isinstance(ctx, int):
        out["aa_metadata"] = {"context_window_tokens": ctx}
    tokens = task_tokens(model)
    if tokens is not None:
        out["token_efficiency"] = {"aa": {"tokens_per_task": {
            "value": {"output": tokens.output}, "collected_at": tokens.collected_at,
            "basis": tokens.basis, "scope": tokens.scope}}}
    out["offers"] = offers
    return {"model": out}


def build_snapshot(client: "BenchmarkClient", bench_ids: list[str]) -> dict:
    """Fetch ``bench_ids`` live and keep the minimal fields. Raises if one is unavailable."""
    documents: dict[str, Any] = {}
    for bench_id in bench_ids:
        quoted = urllib.parse.quote(bench_id, safe=":")
        data, prov = client.fetch(f"/api/models/{quoted}")
        if not data or prov.source not in ("network", "cache"):
            raise RuntimeError(f"{bench_id}: no live document ({prov.source}, {prov.error})")
        documents[f"/api/models/{quoted}"] = minimal_document(data)
        report, _ = client.fetch(f"/api/benchmaxxing?report={quoted}")
        rep = (report or {}).get("report") or {}
        if rep.get("status") == "scored" and _number(rep.get("score")) is not None:
            documents[f"/api/benchmaxxing?report={quoted}"] = {
                "report": {"status": "scored", "score": _number(rep.get("score"))}}
    ratio, _ = client.global_io_ratio()
    if ratio is not None:
        documents["/api/price-comparison"] = {"efficiency": {"global_io_ratio": {"value": ratio}}}
    now = time.time()
    return {
        "schema": 1,
        "source": client.base_url,
        "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)),
        "fetched_at_epoch": round(now),
        "note": ("Last-resort copy used only when the benchmark API and the local disk cache "
                 "are both unavailable; provenance 'bundled-snapshot', always treated as stale. "
                 "Minimal fields only; regenerate with python -m auto_router.bench --write-snapshot."),
        "documents": documents,
    }


# -- command line -------------------------------------------------------------
def _config_bench_ids(path: str | None) -> list[str]:
    """Every bench id a config refers to: models, capability aliases, the verify reference."""
    if not path:
        return []
    from .config import _load_file
    raw = _load_file(path)
    ids: list[str] = []
    for entry in raw.get("models") or []:
        for key in ("bench_id", "capability_like"):
            if entry.get(key):
                ids.append(str(entry[key]))
    ref = (((raw.get("policy") or {}).get("verify") or {}).get("intelligence_threshold") or {})
    if isinstance(ref, dict) and ref.get("reference_model"):
        ids.append(str(ref["reference_model"]))
    return list(dict.fromkeys(ids))


def describe(client: "BenchmarkClient", bench_id: str) -> dict:
    """What the router would read about one model, for ``--show``."""
    doc, prov = client.model_with_provenance(bench_id)
    if not doc:
        return {"bench_id": bench_id, "provenance": prov.to_dict(), "found": False}
    ratio, ratio_prov = client.global_io_ratio()
    offer = pick_offer(doc)
    ev = capability_evidence(doc)
    tokens = task_tokens(doc)
    out: dict[str, Any] = {
        "bench_id": bench_id, "found": True, "id": doc.get("id"),
        "display_name": doc.get("display_name"), "provenance": prov.to_dict(),
        "intelligence_index": intelligence_index(doc),
        "capability": {k: e.to_dict() for k, e in ev.items()},
        "context_tokens": context_length(doc),
        "task_tokens": tokens.to_dict() if tokens else None,
        "global_io_ratio": {"value": ratio, "source": ratio_prov.source},
    }
    if offer:
        out["offer"] = {k: offer.get(k) for k in _OFFER_FIELDS}
        out["cost_per_task"] = cost_per_task(doc, float(offer["input_per_1m"]),
                                             float(offer.get("output_per_1m") or 0.0),
                                             offer.get("cache_read_per_1m"), ratio)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m auto_router.bench",
                                 description="Inspect or refresh the cached benchmark data.")
    ap.add_argument("--show", metavar="BENCH_ID", action="append", default=[],
                    help="print what the router reads for this model (repeatable)")
    ap.add_argument("--refresh", nargs="*", metavar="BENCH_ID",
                    help="re-fetch these ids, or every id in the config when none are given")
    ap.add_argument("--config", default=os.environ.get("AUTO_ROUTER_CONFIG"),
                    help="config file whose bench ids --refresh/--write-snapshot use")
    ap.add_argument("--write-snapshot", metavar="PATH", nargs="?", const=str(SNAPSHOT_PATH),
                    help="rebuild the bundled snapshot from live data")
    ap.add_argument("--offline", action="store_true", help="never touch the network")
    args = ap.parse_args(argv)
    if args.refresh is None and not args.show and args.write_snapshot is None:
        ap.print_help()
        return 2
    status = 0
    if args.refresh is not None or args.write_snapshot is not None:
        ids = list(args.refresh or []) or _config_bench_ids(args.config)
        if not ids:
            print("no bench ids: pass them or --config", file=sys.stderr)
            return 2
        live = BenchmarkClient(ttl_seconds=0, snapshot=False)
        if args.write_snapshot is not None:
            snap = build_snapshot(live, ids)
            Path(args.write_snapshot).parent.mkdir(parents=True, exist_ok=True)
            Path(args.write_snapshot).write_text(json.dumps(snap, indent=1, sort_keys=True) + "\n")
            print(f"wrote {len(snap['documents'])} documents to {args.write_snapshot}")
        else:
            for bench_id in ids:
                _, prov = live.model_with_provenance(bench_id)
                live.benchmaxxing_with_provenance(bench_id)
                print(f"{bench_id}: {prov.source}" + (f" ({prov.error})" if prov.error else ""))
                status = status or (0 if prov.source == "network" else 1)
            live.global_io_ratio()
    client = BenchmarkClient(offline=args.offline)
    for bench_id in args.show:
        info = describe(client, bench_id)
        print(json.dumps(info, indent=2, default=str))
        status = status or (0 if info.get("found") else 1)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
