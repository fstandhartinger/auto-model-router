"""Replay real agent sessions through the routing policies.

Inputs
  --traffic   output of traffic_stats.py (per-call metadata of real sessions: time,
              prefix tokens, output tokens, user-turn boundaries; no content)
  --matrix    output of run_matrix.py (graded success per model x category x difficulty,
              output verbosity, cache hit behaviour, latency)
  --scenario  pricing/quota scenario (see SCENARIOS)

Model of one user turn
  * The turn's calls are replayed with their real prefix sizes and gaps. Output tokens
    are scaled by the model's measured verbosity relative to the model that produced the
    log. Cache: a call reads what the previous call on the same model left warm, if the
    gap is within the TTL, with the measured hit rate; everything else is written.
  * Hidden difficulty of the turn comes from a proxy (calls in the turn); the policy only
    sees a noisy estimate (``--noise``).
  * Success is drawn from the measured matrix with common random numbers across
    policies (a turn that a weak model solves is also solved by a stronger one).
  * A failure is noticed with probability ``--detect``; the policy may then retry on
    another model, paying the turn again on a cold (or warm) cache. Unnoticed failures
    count as unsolved.

Outputs per policy: success rate, USD, USD per solved turn, subscription quota used
(list-price dollars and % of the weekly quota), switches, escalations, mean latency.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from auto_router.catalog import CacheRules, Catalog, ModelInfo, Prices  # noqa: E402
from auto_router.economics import SuccessModel  # noqa: E402
from auto_router.policies import (POLICIES, Choice, Context, Conversation, TurnRequest,  # noqa: E402
                                  turn_call_cost)
from auto_router.quota import PacingRule, QuotaDecision, QuotaState  # noqa: E402
from auto_router.quota import decide as quota_decide  # noqa: E402

LEVEL_VALUE = {"easy": 0.2, "medium": 0.5, "hard": 0.8}
#: name of the Opus 5 route in the matrix file
OPUS_MATRIX_NAME = "claude-opus-5-aiml"


# ---------------------------------------------------------------------------
# deployment catalogue used by the simulation
# ---------------------------------------------------------------------------
@dataclass
class SimModel:
    info: ModelInfo
    matrix_name: str
    verbosity: float = 1.0       # output tokens relative to the logged model
    call_latency_s: float = 20.0
    availability: float = 1.0    # share of turns the route is usable (e.g. utilisation caps)
    list_ref: ModelInfo | None = None   # list-price twin used to meter subscription quota


def anthropic_cache(hit: float) -> CacheRules:
    return CacheRules(ttl_seconds=300, min_tokens=1024, hit_rate=hit)


def openai_cache(hit: float, ttl: int = 300) -> CacheRules:
    return CacheRules(ttl_seconds=ttl, min_tokens=1024, hit_rate=hit)


def build_catalog(scenario: str, caps: dict[str, dict[str, float]], matrix_stats: dict) -> dict[str, SimModel]:
    """Prices are public list prices (Benchmark Heaven offers, 2026-09-17)."""
    def m(name, matrix_name, prices, cache, ctx, **kw):
        info = ModelInfo(name=name, provider="sim", upstream_id=name, prices=prices, cache=cache,
                         context_tokens=ctx, capability=caps.get(matrix_name, {}), tools=True,
                         subscription=kw.pop("subscription", None))
        st = matrix_stats.get(matrix_name, {})
        return SimModel(info, matrix_name, verbosity=st.get("verbosity", 1.0),
                        call_latency_s=st.get("call_latency_s", 20.0), **kw)

    opus_stats = matrix_stats.get(OPUS_MATRIX_NAME, {})
    opus_api = ModelInfo("claude-opus-5", "sim", "claude-opus-5", Prices(5, 25, 0.5, 6.25), anthropic_cache(0.99),
                         1_000_000, capability=caps.get("claude-opus-5-aiml", {}))
    models = {
        "qwen3.8-27b": m("qwen3.8-27b", "qwen3.8-27b", Prices(0.3, 2.4, 0.3), CacheRules(300, 1024, 0.0), 256_000),
        "dsv4-flash": m("dsv4-flash", "dsv4-flash", Prices(0.1, 0.4, 0.02), openai_cache(0.78), 1_000_000),
        "kimi-k3": m("kimi-k3", "kimi-k3", Prices(3.0, 15.0, 0.3), openai_cache(0.58), 1_000_000),
        "glm-5.3-flash": m("glm-5.3-flash", "glm-5.3-flash", Prices(0.2, 0.5, 0.05), openai_cache(0.92), 1_000_000),
        "gpt-5.6-luna": m("gpt-5.6-luna", "gpt-5.6-luna", Prices(0.2, 1.2, 0.02, 0.25), openai_cache(0.978), 1_000_000),
        "glm-5.3": m("glm-5.3", "glm-5.3", Prices(1.75, 4.5, 0.44), openai_cache(0.975), 1_000_000),
        "gpt-5.6-sol": m("gpt-5.6-sol", "gpt-5.6-sol-aiml", Prices(2.0, 10.0, 0.2, 2.5), openai_cache(0.988), 1_000_000),
        "claude-opus-5": SimModel(opus_api, OPUS_MATRIX_NAME,
                                  verbosity=opus_stats.get("verbosity", 1.0),
                                  call_latency_s=opus_stats.get("call_latency_s", 20)),
    }
    if scenario.startswith("ours"):
        # These three run on a free tier for us, usable while the host is below its utilisation cap.
        for name in ("qwen3.8-27b", "dsv4-flash", "kimi-k3"):
            sm = models[name]
            sm.list_ref = sm.info
            sm.info = sm.info.with_(prices=Prices.free())
            sm.availability = 0.9
            sm.info = sm.info.with_(cache=CacheRules(ttl_seconds=300, min_tokens=1024, hit_rate=0.85))
        # Claude Code on the plan keeps ~98 % cache reads across pauses of up to an hour (measured in
        # our logs), so the plan model gets a 1-hour cache; the API model keeps the 5-minute default.
        sub = opus_api.with_(name="claude-opus-5-sub", prices=Prices.free(), subscription="claude",
                             cache=CacheRules(ttl_seconds=3600, min_tokens=1024, hit_rate=0.99))
        models["claude-opus-5-sub"] = SimModel(sub, "claude-opus-5-aiml", models["claude-opus-5"].verbosity,
                                               models["claude-opus-5"].call_latency_s,
                                               list_ref=opus_api.with_(cache=sub.cache))
        luna = models["gpt-5.6-luna"]
        codex = luna.info.with_(name="codex-luna-sub", prices=Prices.free(), subscription="codex")
        models["codex-luna-sub"] = SimModel(codex, "gpt-5.6-luna", luna.verbosity, luna.call_latency_s,
                                            list_ref=luna.info)
    return models


SCENARIOS = {
    "list": "every model at public list price, no subscription",
    "ours": "our providers (free tier + metered) and the Claude plan paced live over the replayed week",
    "ours-closed": "our providers, Claude plan not used by the router",
}


# ---------------------------------------------------------------------------
# matrix -> ground truth and beliefs
# ---------------------------------------------------------------------------
def task_half(task_id: str) -> int:
    import hashlib
    return hashlib.sha256(task_id.encode()).digest()[0] % 2


def load_matrix(path: str, half: int | None = None) -> tuple[dict, dict]:
    rows = [json.loads(line) for line in open(path)]
    if half is not None:
        rows = [r for r in rows if task_half(r["task"]) == half]
    cells: dict = defaultdict(lambda: [0, 0])
    stats: dict = defaultdict(Counter)
    for r in rows:
        c = cells[(r["model"], r["category"], r["difficulty"])]
        c[0] += int(r["ok"])
        c[1] += 1
        if r["category"] in ("agentic", "coding"):
            s = stats[r["model"]]
            s["out"] += r["output_tokens"]
            s["calls"] += max(1, r["calls"])
            s["latency"] += r["latency_s"]
    per_call_out = {m: s["out"] / s["calls"] for m, s in stats.items()}
    ref = per_call_out.get("claude-opus-5-aiml") or statistics_median(list(per_call_out.values()))
    matrix_stats = {m: {"verbosity": per_call_out[m] / ref,
                        "call_latency_s": s["latency"] / s["calls"]} for m, s in stats.items()}
    return cells, matrix_stats


def statistics_median(values):
    values = sorted(values)
    return values[len(values) // 2] if values else 1.0


def truth_table(cells: dict, category_weights: dict[str, float], prior_strength: float = 3.0) -> dict:
    """P(success | model, level) for an agentic coding turn, smoothed towards the level mean."""
    models = {k[0] for k in cells}
    table = {}
    for level in LEVEL_VALUE:
        pooled_s = sum(cells[(m, c, level)][0] * w for m in models for c, w in category_weights.items())
        pooled_n = sum(cells[(m, c, level)][1] * w for m in models for c, w in category_weights.items())
        mean = pooled_s / pooled_n if pooled_n else 0.5
        for m in models:
            s = sum(cells[(m, c, level)][0] * w for c, w in category_weights.items())
            n = sum(cells[(m, c, level)][1] * w for c, w in category_weights.items())
            table[(m, level)] = (s + prior_strength * mean) / (n + prior_strength)
    return table


def fit_success_model(table: dict, sims: dict[str, SimModel], category: str) -> SuccessModel:
    """Logistic belief from benchmark capability, fitted to the matrix (what a router can know)."""
    best = None
    for offset in range(-40, 81, 4):
        for slope in (10, 20, 30, 40, 60, 80):
            for scale in (3, 5, 8, 12, 18):
                sm = SuccessModel(offset=offset, slope=slope, scale=scale, benchmaxxing_weight=0.0)
                loss = 0.0
                for s in sims.values():
                    for level, d in LEVEL_VALUE.items():
                        y = table.get((s.matrix_name, level))
                        if y is None:
                            continue
                        p = sm.p(s.info, category, d)
                        loss -= y * math.log(p) + (1 - y) * math.log(1 - p)
                if best is None or loss < best[0]:
                    best = (loss, offset, slope, scale)
    _, offset, slope, scale = best
    return SuccessModel(offset=offset, slope=slope, scale=scale, benchmaxxing_weight=0.0)


# ---------------------------------------------------------------------------
# traffic
# ---------------------------------------------------------------------------
@dataclass
class Turn:
    calls: list          # [t, prefix, read, output]
    level: str
    source: str


def load_sessions(path: str, sources: list[str], max_sessions: int, seed: int,
                  easy_max: int, medium_max: int) -> list[tuple[str, list[Turn]]]:
    data = json.load(open(path))
    rng = random.Random(seed)
    sessions = []
    for src in sources:
        for s in data[src]["_sessions"]:
            calls = s["calls"]
            bounds = sorted(set([0.0] + [t for t in s["user_turns"] if t > 0])) + [float("inf")]
            turns = []
            for a, b in zip(bounds, bounds[1:]):
                chunk = [c for c in calls if a <= c[0] < b]
                if not chunk:
                    continue
                n = len(chunk)
                level = "easy" if n <= easy_max else "medium" if n <= medium_max else "hard"
                turns.append(Turn(chunk, level, src))
            if turns:
                sessions.append((s.get("start", 0), src, turns))
    if max_sessions:
        rng.shuffle(sessions)
        sessions = sessions[:max_sessions]
    return sorted(sessions, key=lambda x: x[0])


# ---------------------------------------------------------------------------
# simulation
# ---------------------------------------------------------------------------
@dataclass
class Outcome:
    turns: int = 0
    solved: int = 0
    usd: float = 0.0
    quota_usd: float = 0.0
    switches: int = 0
    escalations: int = 0
    latency_s: float = 0.0
    calls: int = 0
    codex_usd: float = 0.0
    by_model: Counter = None

    def row(self, pct_per_usd: float) -> dict:
        return {"turns": self.turns, "success": round(self.solved / max(1, self.turns), 4),
                "usd": round(self.usd, 2), "usd_per_solved": round(self.usd / max(1, self.solved), 4),
                "quota_list_usd": round(self.quota_usd, 2),
                "quota_pct_week": round(self.quota_usd * pct_per_usd, 2),
                "codex_quota_list_usd": round(self.codex_usd, 2),
                "switches": self.switches, "escalations": self.escalations,
                "latency_s_per_turn": round(self.latency_s / max(1, self.turns), 1),
                "model_share": {k: round(v / max(1, sum(self.by_model.values())), 3)
                                for k, v in self.by_model.most_common(6)}}


def run_turn_calls(sim: SimModel, turn: Turn, conv: Conversation, start_offset: float, extra_prefix: int,
                   out: Outcome, t0: float) -> tuple[int, float]:
    """Charge all calls of a turn to ``sim``. Returns (tokens produced, end time)."""
    produced = 0
    now = t0
    for i, (t, prefix, _read, output) in enumerate(turn.calls):
        now = t0 + (t - turn.calls[0][0])
        prompt = prefix + extra_prefix
        warm = conv.warm_tokens(sim.info, now)
        warm = min(warm, prompt)
        o = max(1, int(output * sim.verbosity))
        p, hit = sim.info.prices, sim.info.cache.hit_rate
        if prompt < sim.info.cache.min_tokens:
            cost = (prompt * p.input + o * p.output) / 1e6
        else:
            cost = (warm * (hit * p.read + (1 - hit) * p.write) + (prompt - warm) * p.write + o * p.output) / 1e6
        out.usd += cost
        if sim.info.subscription and sim.list_ref is not None:
            lp, lh = sim.list_ref.prices, sim.list_ref.cache.hit_rate
            out.quota_usd += (warm * (lh * lp.read + (1 - lh) * lp.write) + (prompt - warm) * lp.write
                              + o * lp.output) / 1e6
        out.latency_s += sim.call_latency_s * max(0.3, sim.verbosity) ** 0.5
        out.calls += 1
        out.by_model[sim.info.name] += 1
        conv.record_call(sim.info, prompt, o, now)
        produced += o
    return produced, now


class QuotaMeter:
    """Weekly plan usage during the replay: background (interactive, not routed) + routed traffic."""

    def __init__(self, week_start: float, background: float, pct_per_usd: float, rule: PacingRule,
                 enabled: bool):
        self.week_start, self.background, self.pct_per_usd = week_start, background, pct_per_usd
        self.rule, self.enabled = rule, enabled
        self.routed_usd = 0.0
        self._cache: tuple[float, QuotaDecision] | None = None

    def used(self, now: float) -> float:
        elapsed = max(0.0, min(1.0, (now - self.week_start) / (7 * 86400)))
        return self.background * elapsed + self.routed_usd * self.pct_per_usd / 100.0

    def decision(self, now: float) -> dict[str, QuotaDecision]:
        if not self.enabled:
            return {}
        if self._cache is None or now - self._cache[0] > 300:
            state = QuotaState(week_used=self.used(now), week_resets_at=self.week_start + 7 * 86400,
                               measured_at=now)
            self._cache = (now, quota_decide(state, self.rule, now=now))
        return {"claude": self._cache[1]}


def today_model(src: str, available: dict) -> str:
    """What our agents use now: Claude Code on the plan, Codex on its plan (modelled as Luna at $0),
    OpenCode on the best free model."""
    if src == "claude":
        return "claude-opus-5-sub" if "claude-opus-5-sub" in available else "claude-opus-5"
    if src == "codex":
        return "codex-luna-sub" if "codex-luna-sub" in available else "gpt-5.6-luna"
    for name in ("kimi-k3", "dsv4-flash", "qwen3.8-27b", "glm-5.3-flash", "gpt-5.6-luna"):
        if name in available:
            return name
    return next((n for n, s in available.items() if not s.info.subscription), next(iter(available)))


def simulate(policy_name: str, sessions, sims: dict[str, SimModel], truth: dict, belief: SuccessModel,
             scenario: str, detect: float, noise: float, seed: int, stakes_mult: float, max_retries: int = 2,
             policy_kwargs: dict | None = None, background: float = 0.25, pct_per_usd: float = 1 / 38.45,
             rule: PacingRule = PacingRule()) -> Outcome:
    rng = random.Random(seed)
    out = Outcome(by_model=Counter())
    today = policy_name == "T_today"
    policy = POLICIES["A_static" if today else policy_name](**(policy_kwargs or {}))
    week_start = sessions[0][0] if sessions else 0.0
    meter = QuotaMeter(week_start, background, pct_per_usd, rule,
                       enabled=scenario.startswith("ours") and scenario != "ours-closed")
    opus_ref = (sims.get("claude-opus-5") or SimModel(ModelInfo("claude-opus-5", "sim", "claude-opus-5",
                Prices(5, 25, 0.5, 6.25), anthropic_cache(0.99), 1_000_000), "claude-opus-5-aiml")).info
    for start, src, turns in sessions:
        conv = Conversation()
        for index, turn in enumerate(turns):
            u_success, u_detect, u_noise = rng.random(), rng.random(), rng.gauss(0, 1)
            available = {n: s for n, s in sims.items() if rng.random() < s.availability}
            if src != "claude" or scenario == "ours-closed":
                available = {n: s for n, s in available.items() if s.info.subscription != "claude"}
            if src != "codex" or not today:
                available.pop("codex-luna-sub", None)
            if not available:  # every route drawn as unavailable: fall back to the full non-plan set
                available = {n: s for n, s in sims.items() if not s.info.subscription}
            t0 = start + turn.calls[0][0]
            quota = meter.decision(t0) if src == "claude" else {}
            if not today and src == "claude" and "claude-opus-5-sub" in available and not (
                    quota.get("claude") and quota["claude"].open):
                available.pop("claude-opus-5-sub")
            catalog = Catalog([s.info for s in available.values()])
            ctx = Context(catalog, belief, quota, {"claude-opus-5-sub": "claude-opus-5"})
            if "claude-opus-5" not in available:
                ctx.catalog = Catalog([s.info for s in available.values()] + [opus_ref])
                ctx.catalog._models.pop("claude-opus-5")  # reference only via subscription_reference
            true_d = LEVEL_VALUE[turn.level]
            est_d = max(0.0, min(1.0, true_d + noise * u_noise))
            first = turn.calls[0]
            growth = max(500, (turn.calls[-1][1] - first[1]) // max(1, len(turn.calls) - 1))
            req = TurnRequest(category="agentic", difficulty=est_d, prompt_tokens=first[1],
                              output_tokens=int(sum(c[3] for c in turn.calls)), now=t0,
                              steps=len(turn.calls), growth_per_step=int(growth), needs_tools=True,
                              follow_up=0.8, detect_prob=detect, remaining_turns=len(turns) - index - 1)
            req.stakes_usd = 0.5 + stakes_mult * turn_call_cost(opus_ref, req, 0, Context(Catalog([opus_ref]), belief))
            model = today_model(src, available) if today else policy.choose(conv, req, ctx).model
            tried: set[str] = set()
            extra = 0
            solved = False
            attempts = 0
            while True:
                sim = available[model]
                tried.add(model)
                before = out.quota_usd
                produced, _ = run_turn_calls(sim, turn, conv, 0.0, extra, out, t0)
                if sim.info.subscription == "claude":
                    meter.routed_usd += out.quota_usd - before
                elif sim.info.subscription == "codex":
                    out.codex_usd += out.quota_usd - before
                    out.quota_usd = before
                p = truth[(sim.matrix_name, turn.level)]
                if u_success < p:
                    solved = True
                    break
                if attempts >= max_retries or rng.random() >= detect:
                    break
                attempts += 1
                retry = (Choice(model, "retry") if today else
                         policy.on_failure(conv, req, ctx, model, tried if model not in tried else tried))
                if retry is None or retry.model not in available:
                    break
                out.escalations += int(retry.model != model)
                model = retry.model
                extra += produced
                u_success = rng.random() * 0.5 + u_success * 0.5  # retries are partially correlated
            out.turns += 1
            out.solved += int(solved)
        out.switches += conv.switches
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traffic", required=True)
    ap.add_argument("--matrix", required=True)
    ap.add_argument("--scenario", default="list", choices=list(SCENARIOS))
    ap.add_argument("--sources", default="claude,codex,opencode")
    ap.add_argument("--policies", default="T_today," + ",".join(POLICIES))
    ap.add_argument("--max-sessions", type=int, default=0)
    ap.add_argument("--detect", type=float, default=0.6)
    ap.add_argument("--noise", type=float, default=0.15)
    ap.add_argument("--stakes", type=float, default=1.0, help="x the turn's list cost on the strong model, + $0.5")
    ap.add_argument("--background", type=float, default=0.25, help="weekly plan share used interactively")
    ap.add_argument("--reserve", type=float, default=0.65)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--easy-max", type=int, default=8)
    ap.add_argument("--medium-max", type=int, default=40)
    ap.add_argument("--pct-per-usd", type=float, default=1 / 38.45)
    ap.add_argument("--belief", default="fitted", choices=["fitted", "measured"])
    ap.add_argument("--static-model", default="claude-opus-5")
    ap.add_argument("--exclude", default="", help="comma-separated model names to leave out of the deployment")
    ap.add_argument("--cv", action="store_true",
                    help="two-fold: beliefs from one half of the tasks, ground truth from the other")
    ap.add_argument("--out")
    args = ap.parse_args()

    caps_raw = json.load(open(Path(args.matrix).with_name("capabilities.json")))
    folds = [(0, 1), (1, 0)] if args.cv else [(None, None)]
    sessions = load_sessions(args.traffic, args.sources.split(","), args.max_sessions, args.seed,
                             args.easy_max, args.medium_max)
    fold_results = []
    for truth_half, belief_half in folds:
        fold_results.append(run_fold(args, caps_raw, sessions, truth_half, belief_half))
    results = merge_folds(fold_results)
    print(json.dumps(results, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=1))


def merge_folds(folds: list[dict]) -> dict:
    if len(folds) == 1:
        return folds[0]
    merged = {k: v for k, v in folds[0].items() if k != "policies"}
    merged["cv"] = True
    merged["policies"] = {}
    for name in folds[0]["policies"]:
        rows = [f["policies"][name] for f in folds]
        avg = {}
        for key, value in rows[0].items():
            if isinstance(value, (int, float)):
                avg[key] = round(sum(r[key] for r in rows) / len(rows), 4)
            else:
                avg[key] = value
        avg["usd_per_solved"] = round(avg["usd"] / max(1, avg["success"] * avg["turns"]), 4)
        merged["policies"][name] = avg
    return merged


def run_fold(args, caps_raw, sessions, truth_half, belief_half) -> dict:
    cells, mstats = load_matrix(args.matrix, truth_half)
    truth = truth_table(cells, {"agentic": 1.0, "coding": 0.5})
    belief_cells, _ = load_matrix(args.matrix, belief_half)
    belief_truth = truth_table(belief_cells, {"agentic": 1.0, "coding": 0.5})
    sims = build_catalog(args.scenario, caps_raw, mstats)
    for name in filter(None, args.exclude.split(",")):
        sims.pop(name, None)
    belief = fit_success_model(belief_truth, sims, "agentic")
    if args.belief == "measured":
        belief = replace(belief, measured={(n, "agentic", lvl): belief_truth[(s.matrix_name, lvl)]
                                           for n, s in sims.items() for lvl in LEVEL_VALUE})
    level_mix = Counter(t.level for _, _, ts in sessions for t in ts)
    results = {"scenario": args.scenario, "about": SCENARIOS[args.scenario], "detect": args.detect,
               "noise": args.noise, "stakes": args.stakes, "belief": args.belief, "background": args.background,
               "reserve": args.reserve,
               "belief_params": [belief.offset, belief.slope, belief.scale],
               "sessions": len(sessions), "level_mix": dict(level_mix), "policies": {}}
    for name in args.policies.split(","):
        kwargs = {"model": args.static_model} if name == "A_static" else {}
        if name == "T_today" and args.scenario == "list":
            continue
        o = simulate(name, sessions, sims, truth, belief, args.scenario, args.detect, args.noise, args.seed,
                     args.stakes, policy_kwargs=kwargs, background=args.background,
                     pct_per_usd=args.pct_per_usd, rule=PacingRule(weekly_reserve=args.reserve))
        results["policies"][name] = o.row(args.pct_per_usd)
    return results


if __name__ == "__main__":
    main()
