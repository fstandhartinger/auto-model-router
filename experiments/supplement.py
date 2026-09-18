"""The pre-registered held-out **supplement**: registration, runner, combined report.

The original 27-task held-out run is finished evidence and is never touched
again - not its task file, not its registration, not its ledger, not its
report. It cannot reach ten valid *paired* tasks in every category, mostly
because four of its twelve design rows are harness truncations.

This module adds a **second, separately pre-registered run** in its own
timestamped directory, and a report that combines the two while keeping them
distinguishable. Three things it does that the original registration did not:

**It freezes what it is comparing, not only what it asks.** The registration
records the routing policy's name and parameters and the identity of every
route in the catalog - provider, upstream id, prices, per-category capability,
whether the evidence behind it is stale. "The real normal routing policy" is
then a checkable fact rather than a claim, and the runner refuses to start if
any of it has changed without an amendment.

**It counts pairs, not rows.** A category's sample size is the number of tasks
where *both* arms were graded. A truncated answer costs its pair; it is never
quietly dropped from a denominator. See ``experiments.pairing``.

**It fails closed on the cumulative budget.** The guard adds the recorded prior
spend to every ledger it is given and refuses a batch whose worst case could
cross the absolute cap - before the batch, not after it.
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
from contextlib import contextmanager
import hashlib
import json
import math
import os
import random
import sys
import threading
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from auto_router.config import RouterConfig, load_config      # noqa: E402
from experiments import graders, pairing, sandbox             # noqa: E402
from experiments.heldout import CATEGORIES, digest, wilson    # noqa: E402
from experiments.llm import BudgetExceeded as _ClientBudgetExceeded  # noqa: E402
from experiments.tasks_supplement import build_tasks          # noqa: E402

SEED = 20260918_1030

#: The evaluation is paired, so this is ten *pairs*, not ten rows.
TARGET_VALID_PAIRS = 10

#: The original ledger's usable pairs per category, counted before this set was
#: written, against its weaker comparator (the free control). The supplement
#: needs at least the shortfall in every category, plus headroom where the
#: harness is known to lose rows.
ORIGINAL_VALID_PAIRS = {"design": 1, "coding": 4, "math": 6,
                        "research": 5, "summarisation": 4, "cache_repeat": 4}

#: How many tasks the supplement must carry per category. The shortfall plus
#: headroom; the design headroom is large on purpose, because on the original
#: set both free arms spent a 12,000-token budget without finishing the larger
#: pages and a pair needs *both* arms to finish.
SUPPLEMENT_MINIMUM = {"design": 18, "coding": 7, "math": 5,
                      "research": 6, "summarisation": 7, "cache_repeat": 7}

#: The identity of the original run this supplements. Recorded, never edited.
ORIGINAL_TASK_DIGEST = ("84731010531b266f4651d82455b90d17"
                        "c677f2e92e580aff13d03971a248a870")
ORIGINAL_TASK_COUNT = 27

ANALYSIS_PLAN = {
    "question": "Does the real normal routing policy solve held-out tasks at least as often as "
                "the ordinary single-route comparator, on at least ten valid paired tasks per "
                "category, and at what measured cost?",
    "design": "Paired. Every task is sent to the routing policy and to each comparator with an "
              "identical prompt, the same output budget and the same graders, in one run.",
    "unit_of_evidence": "the PAIR. A pair counts only when both of its rows were graded. A "
                        "truncated answer or an unavailable sandbox REMOVES its pair from the "
                        "pass-rate denominator - that is a real loss of evidence, not a neutral "
                        "one, and it is reported per arm and per category rather than absorbed. "
                        "Every category also carries the same counts under the opposite rule, "
                        "with truncation scored as a failure.",
    "primary_outcome": "valid pairs per category, and the paired pass-rate difference within them",
    "secondary_outcomes": ["discordant pairs and an exact sign test",
                           "measured cost per arm with its cost basis",
                           "observed latency", "route selected", "cache status",
                           "estimated versus observed cost"],
    "uncertainty": "Wilson 95% interval per category per arm, and the exact two-sided sign test "
                   "over discordant pairs. No aggregate is reported without the per-category "
                   "table beside it.",
    "stopping_rule": "The run stops when the task set is exhausted, the run cap is reached, or "
                     "the cumulative budget guard refuses the next call. A partial run reports "
                     "the categories it completed and names the rest.",
    "exclusions": "Only a HARNESS failure is excluded: the grader reports SANDBOX UNAVAILABLE, or "
                  "the answer hit the harness's output budget (TRUNCATED). A failed or refused "
                  "upstream call is a property of the route and is counted as a FAILURE.",
    "arms": {
        "router": "the real normal routing policy, as configured - the same code path the server "
                  "uses, choosing a route per task from evidence",
        "control": "the ordinary no-routing default: one capable route for everything, chosen "
                   "from the same capability data",
        "control-metered": "a second comparator, named on the command line, on a metered route, "
                           "so at least one arm has a cash figure at all. That figure is the "
                           "gateway's reported upstream inference cost - the best number the "
                           "provider gives - and it is NOT an invoice and is not reconciled "
                           "against one",
    },
    "claims_not_made": [
        "No cash saving is claimed from estimated or list-price arithmetic.",
        "No quality claim is made from a category with fewer than 10 valid PAIRS.",
        "The design grader is a structural proxy and is reported as such.",
        "Replay and simulation results are never combined with live results.",
        "A static single-route control is never described as the normal routing policy.",
    ],
}

#: Experiment files whose content decides an outcome in the supplement.
CODE_FILES = ("graders.py", "tasks_supplement.py", "sandbox_runner.py", "sandbox.py",
              "supplement.py", "pairing.py", "llm.py", "heldout_run.py", "heldout.py")

#: **Product** files that decide an outcome. Hashing only the experiment files
#: leaves the thing under test unhashed: ``run_live`` imports ``Router`` and
#: ``load_config``, and those choose the route an arm actually uses. An
#: independent reviewer pointed out that they could change while every
#: registered digest still matched. The whole package is hashed rather than a
#: hand-picked subset, because picking the subset is the same mistake again.
PRODUCT_PACKAGE = "auto_router"


class BudgetExceeded(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# the frozen identity of what is being measured
# ---------------------------------------------------------------------------
def load_frozen_config(path: str | Path) -> RouterConfig:
    return load_config(Path(path))


def policy_identity(config: RouterConfig) -> dict:
    """Exactly what "the real normal routing policy" resolves to, as data.

    Recorded before the run and re-checked before every later run, so a route
    silently gaining capability, losing its price or going stale cannot change
    the meaning of the arm without that being visible.
    """
    policy = config.policy or {}
    name = policy.get("name") or os.environ.get("AUTO_ROUTER_POLICY") or "F_expected"
    identity = {
        "policy_name": name,
        "policy_source": ("config" if policy.get("name")
                          else "environment" if os.environ.get("AUTO_ROUTER_POLICY")
                          else "environment default"),
        "policy_settings": {k: v for k, v in sorted(policy.items())
                            if k != "name" and isinstance(v, (int, float, str, bool, list))},
        "catalog": [
            {"name": m.name, "provider": m.provider, "upstream_id": m.upstream_id,
             "free": bool(m.prices.is_free), "subscription": bool(m.subscription),
             "prices_per_mtok": {"input": m.prices.input, "output": m.prices.output,
                                 "read": m.prices.read, "write": m.prices.write},
             "capability": {c: round(m.cap(c), 4) for c in
                            ("general", "coding", "math", "knowledge", "summarisation",
                             "design", "long_context", "tool_use", "agentic")},
             "evidence_stale": bool(m.evidence_stale)}
            for m in sorted(config.catalog.all(), key=lambda m: m.name)],
    }
    identity["sha256"] = hashlib.sha256(
        json.dumps(identity, sort_keys=True).encode()).hexdigest()
    return identity


def _code_digests() -> dict[str, str]:
    here = Path(__file__).resolve().parent
    return {name: digest(here / name) for name in CODE_FILES}


def _product_digests() -> dict[str, str]:
    root = Path(__file__).resolve().parent.parent / PRODUCT_PACKAGE
    return {f"{PRODUCT_PACKAGE}/{path.name}": digest(path)
            for path in sorted(root.glob("*.py"))}


def _repo_git_sha() -> str | None:
    import subprocess
    root = Path(__file__).resolve().parent.parent
    try:
        out = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or None
    except Exception:                                          # noqa: BLE001
        return None


def plan_digest() -> str:
    return hashlib.sha256(json.dumps(ANALYSIS_PLAN, sort_keys=True).encode()).hexdigest()


# ---------------------------------------------------------------------------
# pre-registration
# ---------------------------------------------------------------------------
def preregister(out_dir: Path, config_path: str | Path, seed: int = SEED) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks = build_tasks(random.Random(seed))
    task_file = out_dir / "tasks.jsonl"
    with task_file.open("w") as fh:
        for task in tasks:
            fh.write(json.dumps(task, sort_keys=True) + "\n")
    counts: dict[str, int] = {}
    for task in tasks:
        counts[task["category"]] = counts.get(task["category"], 0) + 1
    identity = policy_identity(load_frozen_config(config_path))
    record = {
        "registered_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "role": "supplement",
        "seed": seed,
        "task_file": task_file.name,
        "task_file_sha256": digest(task_file),
        "analysis_plan_sha256": plan_digest(),
        "code_sha256": _code_digests(),
        "product_sha256": _product_digests(),
        "repo_git_sha": _repo_git_sha(),
        "config_path": str(Path(config_path).resolve()),
        "config_sha256": digest(Path(config_path)),
        "repo_git_sha_note": ("the checkout at registration time. It is recorded, not enforced: "
                              "committing this run necessarily changes it, so a later mismatch "
                              "is expected and is reported rather than treated as drift."),
        "policy_identity": identity,
        "policy_identity_sha256": identity["sha256"],
        "task_count": len(tasks),
        "tasks_by_category": counts,
        "grader_by_category": {c: graders.GRADER_KIND[
            next(t["grader"] for t in tasks if t["category"] == c)] for c in counts},
        "task_ids": [t["id"] for t in tasks],
        "target_valid_pairs_per_category": TARGET_VALID_PAIRS,
        "original_valid_pairs_per_category": dict(ORIGINAL_VALID_PAIRS),
        "supplement_minimum_tasks_per_category": dict(SUPPLEMENT_MINIMUM),
        "supplements": {"directory": "runs/heldout",
                        "task_file_sha256": ORIGINAL_TASK_DIGEST,
                        "task_count": ORIGINAL_TASK_COUNT,
                        "note": "immutable; this run never writes to it"},
        "analysis_plan": ANALYSIS_PLAN,
        "note": ("Written before any model was called in this supplement. The runner refuses to "
                 "start unless the task file, the code that decides an outcome, the analysis plan "
                 "and the policy/catalog identity all still match. A self-certifying file can be "
                 "rewritten by whoever can write to it; the external anchors are the copies of "
                 "these digests in this run's EVIDENCE.md and in the git commit."),
    }
    (out_dir / "preregistration.json").write_text(json.dumps(record, indent=1))
    return record


def load_preregistration(out_dir: Path, *, config_path: str | Path | None = None,
                         strict: bool = True) -> dict:
    """Load the supplement registration and refuse to run on unrecorded drift."""
    out_dir = Path(out_dir)
    path = out_dir / "preregistration.json"
    if not path.exists():
        raise SystemExit(f"no supplement pre-registration in {out_dir}; run `preregister` first")
    record = json.loads(path.read_text())

    actual = digest(out_dir / record["task_file"])
    if actual != record["task_file_sha256"]:
        raise SystemExit(
            f"supplement task file digest changed since pre-registration\n"
            f"  registered: {record['task_file_sha256']}\n  actual:     {actual}")

    drift: list[str] = []
    if record.get("analysis_plan_sha256") not in (None, plan_digest()):
        drift.append("analysis plan")
    for name, expected in (record.get("code_sha256") or {}).items():
        current = _code_digests().get(name)
        if current and current != expected:
            drift.append(name)
    product_now = _product_digests()
    for name, expected in (record.get("product_sha256") or {}).items():
        if product_now.get(name) != expected:
            drift.append(name)
    for name in set(product_now) - set(record.get("product_sha256") or {}):
        if record.get("product_sha256"):
            drift.append(f"{name} (new file, not registered)")

    identity_path = config_path or record.get("config_path")
    # The config digest was recorded and then never checked - a reviewer's
    # finding. policy_identity covers the catalog and the policy block, not the
    # provider section or anything else the file carries.
    if identity_path and Path(identity_path).exists() and record.get("config_sha256"):
        if digest(Path(identity_path)) != record["config_sha256"]:
            drift.append("config file")
    if identity_path and Path(identity_path).exists():
        current = policy_identity(load_frozen_config(identity_path))
        if current["sha256"] != record.get("policy_identity_sha256"):
            drift.append("policy/catalog identity")
            record["policy_identity_now"] = current
    elif record.get("policy_identity_sha256") and strict:
        raise SystemExit(
            f"the registered config {identity_path!r} is missing, so the policy/catalog "
            "identity cannot be re-checked; refusing to run")

    record["drift"] = drift
    if drift and strict:
        raise SystemExit(
            "these decide an outcome and have changed since pre-registration: "
            + ", ".join(drift)
            + "\nAmend the registration explicitly (supplement.py amend --reason ...) so the "
              "change is on the record, then re-run.")
    return record


def amend(out_dir: Path, reason: str, config_path: str | Path | None = None) -> dict:
    path = Path(out_dir) / "preregistration.json"
    record = json.loads(path.read_text())
    identity = (policy_identity(load_frozen_config(config_path))
                if config_path else record.get("policy_identity"))
    record.setdefault("amendments", []).append({
        "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "reason": reason,
        "previous_code_sha256": record.get("code_sha256"),
        "previous_analysis_plan_sha256": record.get("analysis_plan_sha256"),
        "previous_policy_identity_sha256": record.get("policy_identity_sha256"),
        "previous_product_sha256": record.get("product_sha256"),
        "previous_repo_git_sha": record.get("repo_git_sha"),
    })
    record["code_sha256"] = _code_digests()
    record["product_sha256"] = _product_digests()
    record["repo_git_sha"] = _repo_git_sha()
    record["analysis_plan_sha256"] = plan_digest()
    if identity:
        record["policy_identity"] = identity
        record["policy_identity_sha256"] = identity["sha256"]
    path.write_text(json.dumps(record, indent=1))
    return record


def load_tasks(out_dir: Path) -> list[dict]:
    return [json.loads(line) for line
            in (Path(out_dir) / "tasks.jsonl").read_text().splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# the cumulative budget guard
# ---------------------------------------------------------------------------
class BudgetGuard:
    """Fails closed *before* a batch, on the cumulative total, not on one run.

    Cost per call is the higher of billed and list price - the same conservative
    rule the rest of the project uses - so an unpriced or promotional zero can
    never make the cap look further away than it is.
    """

    def __init__(self, prior_usd: float, cap_usd: float, ledgers: list[Path]):
        self.prior_usd = float(prior_usd)
        self.cap_usd = float(cap_usd)
        self.ledgers = [Path(p) for p in ledgers]
        self.unparsable = 0
        #: Worst case already authorised but not yet on any ledger. Without it,
        #: six workers each read the same remaining balance and all proceed -
        #: the check is atomic, the *spending* is not. An independent reviewer
        #: found this after the first repair.
        self.reserved_usd = 0.0
        self._lock = threading.Lock()

    def ledger_total(self) -> float:
        total, unparsable = 0.0, 0
        for path in self.ledgers:
            if not path.exists():
                continue
            for line in path.read_text().splitlines():
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    unparsable += 1
                    continue
                total += max(float(row.get("cost_usd") or 0.0),
                             float(row.get("list_cost_usd") or 0.0))
        self.unparsable = unparsable
        return total

    def total(self) -> float:
        return self.prior_usd + self.ledger_total() + self.reserved_usd

    def remaining(self) -> float:
        return self.cap_usd - self.total()

    def assert_headroom(self, projected_usd: float) -> float:
        """Refuse before spending when the worst case could cross the cap."""
        with self._lock:
            remaining = self.remaining()
            if self.unparsable:
                # A half-written cost row used to be counted as zero, which made
                # the headroom look *larger* than it was - the opposite of
                # failing closed. A reviewer found it. There is no safe amount
                # to assume for a call whose cost cannot be read.
                raise BudgetExceeded(
                    f"{self.unparsable} unreadable line(s) in the spend ledger; "
                    "the cost of those calls is unknown, so no further spending "
                    "is authorised")
            if projected_usd > remaining:
                raise BudgetExceeded(
                    f"projected {projected_usd:.4f} USD exceeds the remaining "
                    f"{remaining:.4f} USD of the {self.cap_usd:.2f} USD cap "
                    f"(recorded total {self.total():.4f})")
            self.reserved_usd += projected_usd
            return remaining

    def release(self, projected_usd: float) -> None:
        """Give back an authorisation once the call's real cost is on the ledger."""
        with self._lock:
            self.reserved_usd = max(0.0, self.reserved_usd - projected_usd)

    @contextmanager
    def reserve(self, projected_usd: float):
        """Authorise, run, and release even when the call raises."""
        self.assert_headroom(projected_usd)
        try:
            yield
        finally:
            self.release(projected_usd)

    def snapshot(self) -> dict:
        return {"prior_usd": round(self.prior_usd, 6),
                "ledgers_usd": round(self.ledger_total(), 6),
                "reserved_usd": round(self.reserved_usd, 6),
                "total_usd": round(self.total(), 6),
                "cap_usd": self.cap_usd,
                "remaining_usd": round(self.remaining(), 6),
                "unparsable_ledger_lines": self.unparsable,
                "accounting": "per call, the higher of billed and list-price cost"}


# ---------------------------------------------------------------------------
# the live run
# ---------------------------------------------------------------------------
def run_live(args) -> int:
    from experiments.heldout_run import OUTPUT_BUDGET, _one, control_model
    from experiments.llm import Client
    from auto_router.router import Router

    out_dir = Path(args.dir)
    prereg = load_preregistration(out_dir, config_path=args.config)
    tasks = load_tasks(out_dir)
    wanted = {c.strip() for c in args.categories.split(",") if c.strip()}
    tasks = [t for t in tasks if t["category"] in wanted]
    if args.limit:
        tasks = tasks[: args.limit]
    if args.budget <= 0:
        raise SystemExit("--budget must be a positive hard cap in USD")

    isolation = sandbox.preflight()
    if not isolation["available"]:
        raise SystemExit("Bubblewrap is unavailable; coding answers are never executed "
                         "unisolated. Refusing to run.")

    config = load_frozen_config(args.config)
    router = Router(config)
    if router.policy.name != prereg["policy_identity"]["policy_name"]:
        raise SystemExit(f"policy drift: registered {prereg['policy_identity']['policy_name']}, "
                         f"resolved {router.policy.name}")
    control = control_model(router, args.control)
    calls = out_dir / "calls.jsonl"
    guard = BudgetGuard(prior_usd=args.prior_spend, cap_usd=args.cap,
                        ledgers=[calls] + [Path(p) for p in (args.extra_ledger or [])])
    guard.assert_headroom(args.budget)
    client = Client(config, calls, budget_usd=args.budget)
    client.router_catalog = {m.name: m for m in config.catalog.all()}

    ledger = out_dir / "ledger.jsonl"
    done = set()
    if ledger.exists():
        for line in ledger.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                done.add((row["task_id"], row["arm"]))

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    label = args.control_label
    meta = {
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "role": "supplement",
        "supplement_preregistration_sha256": prereg["task_file_sha256"],
        "original_preregistration_sha256": ORIGINAL_TASK_DIGEST,
        "policy": router.policy.name,
        "policy_identity_sha256": prereg["policy_identity_sha256"],
        "control_model": control.name,
        "control_label": label,
        "control_is_metered": not control.prices.is_free,
        "control_prices_per_mtok": {"input": control.prices.input,
                                    "output": control.prices.output},
        "control_selected_by": ("named on the command line" if args.control
                                else "highest general capability in the catalog"),
        "arms": arms,
        "workers": args.workers,
        "run_budget_usd": args.budget,
        "output_budget_per_category": OUTPUT_BUDGET,
        "budget_before": guard.snapshot(),
        "sandbox": isolation,
    }
    #: One file per invocation, never overwritten, plus a `run-meta.json`
    #: pointing at the latest. A single rewritten meta file would quietly lose
    #: the settings of every earlier arm in the same directory.
    meta_path = out_dir / f"run-meta-{meta['started_at'].replace(':', '')}-{'-'.join(arms)}.json"
    _write_meta(out_dir, meta_path, meta)
    print(f"supplement: {len(tasks)} tasks x {arms}; control = {control.name} "
          f"({'metered' if not control.prices.is_free else 'FREE'}); policy = {router.policy.name}; "
          f"workers = {args.workers}; run cap ${args.budget:.2f}; "
          f"cumulative {guard.total():.4f}/{guard.cap_usd:.2f} USD")

    ledger_lock = threading.Lock()
    stopped: list[str] = []
    guarded_router = SerialisedRouter(router)

    def work(task: dict, arm: str) -> None:
        recorded = label if arm == "control" else arm
        if (task["id"], recorded) in done or stopped:
            return
        try:
            # Per-call worst case: the whole output budget at the most expensive
            # route in the catalog. Deliberately pessimistic - the guard has to
            # refuse before the money is spent, not after.
            with guard.reserve(_worst_case_usd(config, task, OUTPUT_BUDGET)):
                outcome = _one(task, arm, guarded_router, control, client, isolation)
        except (BudgetExceeded, _ClientBudgetExceeded) as exc:
            stopped.append(str(exc))
            return
        outcome.arm = recorded
        with ledger_lock, ledger.open("a") as fh:
            fh.write(json.dumps(asdict(outcome)) + "\n")
        mark = {True: "pass", False: "FAIL", None: "skip"}[outcome.passed]
        print(f"  {task['id']:<34} {recorded:<16} {outcome.model:<18} {mark:<5} "
              f"{outcome.detail[:60]}", flush=True)

    # Cache-eligible repeats stay sequential and in order: the point of the
    # category is that the second and later turns find a warm prefix, which a
    # thread pool would destroy.
    parallel = [t for t in tasks if t["category"] != "cache_repeat"]
    sequential = [t for t in tasks if t["category"] == "cache_repeat"]

    for arm in arms:
        if stopped:
            break
        with futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            list(pool.map(lambda t, a=arm: work(t, a), parallel))
        for task in sequential:
            if stopped:
                break
            work(task, arm)

    if stopped:
        print(f"\nstopped early: {stopped[0]}", file=sys.stderr)
    meta["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    meta["stopped_early"] = stopped[0] if stopped else None
    meta["budget_after"] = guard.snapshot()
    meta["run_spend_usd"] = round(client.spent(), 6)
    _write_meta(out_dir, meta_path, meta)
    print(f"\nspend on this run: ${client.spent():.4f} of ${args.budget:.2f}; "
          f"cumulative {guard.total():.4f}/{guard.cap_usd:.2f} USD")
    return 0


def _write_meta(out_dir: Path, meta_path: Path, meta: dict) -> None:
    payload = json.dumps(meta, indent=1)
    meta_path.write_text(payload)
    (out_dir / "run-meta.json").write_text(payload)


class SerialisedRouter:
    """One real ``Router``, safe to drive from several worker threads.

    The point of running the arms concurrently is wall-clock: one free route in
    this catalog spends five minutes on a single page, and the supplement needs
    a hundred and eighty calls. The point of keeping *one* router underneath is
    fidelity: a server keeps one router for all traffic, so its estimator
    calibration and its conversation memory carry across turns, and handing
    each worker its own router would be a different system from the one being
    measured.

    Only the router's own state transitions are serialised - choosing a route,
    committing the estimate, recording the observation. The provider call in
    between happens outside the lock, which is what makes the concurrency worth
    anything. Every attribute other than those three is the real router's.

    One honest consequence, recorded rather than hidden: the estimator is
    calibrated by whichever turns finish first, so the *order* in which
    estimates are refined is not reproducible across runs with more than one
    worker. Route choice, grading and cost are not affected by which task
    calibrated the estimator first; the ``estimated_*`` columns can move by a
    token or two.
    """

    def __init__(self, router):
        self._router = router
        self._lock = threading.Lock()

    def route(self, *a, **kw):
        with self._lock:
            return self._router.route(*a, **kw)

    def commit(self, *a, **kw):
        with self._lock:
            return self._router.commit(*a, **kw)

    def observe(self, *a, **kw):
        with self._lock:
            return self._router.observe(*a, **kw)

    def __getattr__(self, name):
        return getattr(self._router, name)


def _worst_case_usd(config: RouterConfig, task: dict, budgets: dict) -> float:
    """The most a single call for this task could possibly cost in this catalog."""
    output = budgets.get(task["category"], 4000)
    prompt = len(task["prompt"] + task["system"]) / 3.0
    worst = 0.0
    for model in config.catalog.all():
        worst = max(worst, (prompt * model.prices.input + output * model.prices.output) / 1e6)
    return worst


# ---------------------------------------------------------------------------
# the combined report
# ---------------------------------------------------------------------------
ROUTER_ARM = "router"

#: The non-inferiority margin for "at least as often". ``None`` because none was
#: pre-registered, and inventing one now - after the numbers are in - would be
#: the exact move this evaluation exists to avoid. With no margin, no category
#: can support the claim; what the data *do* support is stated per category.
NON_INFERIORITY_MARGIN: float | None = None
ARM_DESCRIPTIONS = {
    "router": "the real normal routing policy, choosing a route per task from evidence",
    "control": "the ordinary no-routing default: one fixed route for everything",
    "control-metered": ("a second fixed route for everything, added *after* the first control "
                        "turned out to be free - selected for being metered, not for being a "
                        "default anyone would configure"),
}


KNOWN_LIMITATIONS = [
    "The supplement's size and its per-category mix were chosen AFTER the original run's "
    "outcomes were known. That is what 'bring every category to ten valid pairs' requires, and "
    "it makes the supplement an adaptive sample rather than an independent one. Nothing about "
    "the rules, graders or arms moved; the count and the mix did.",
    "19 of the 22 supplement design tasks ask for a compact single component, not a full page. "
    "The original run's larger pages are exactly what the free routes failed to finish, so a "
    "design result here says little about full-page work. The three full-page tasks are kept and "
    "reported separately under `by_difficulty`.",
    "cache_repeat is eight questions over one shared warm prefix, run in order. Its pairs are "
    "observations, not independent tasks, and the category is flagged `independent_samples: "
    "false` everywhere it is reported.",
    "The original run's registration was amended three times AFTER outcomes were seen, including "
    "the rule that a truncated answer is excluded rather than failed, and a re-run of rows under "
    "a repaired grader and a repaired cost accounting. Its rows are pooled here, so every "
    "category also carries a supplement-only figure and a truncation-as-failure sensitivity.",
    "The original run froze no policy/catalog identity; only the supplement did. The combined "
    "router arm therefore carries a verified identity for 60 of its 87 rows.",
    "The metered comparator was added after the first control turned out to be free. It was "
    "selected for being metered, not for being what a sensible person would configure, and its "
    "quality column is not an independent second test of the same question.",
    "A measured cost here is the gateway's reported upstream inference cost, not an invoice. It "
    "is the best figure available from the provider and it is not reconciled against a bill.",
    "A free route's zero is a configured price, not a measurement that nothing was charged.",
    "The product-code digests, the config-digest check, the fail-closed budget guard and the "
    "manifest-based attempted universe were all added AFTER the 180 rows had been collected, in "
    "response to an independent review. They authenticate the checkout and protect the next run; "
    "they do not authenticate the run that produced this evidence. What does, weakly: the working "
    "tree was clean throughout, and the newest mtime under auto_router/ is 2026-09-18T08:27:44Z, "
    "nearly two hours before the first supplement call at 10:23:34Z.",
    "The paired-difference interval is a conservative product of two exact binomial intervals "
    "(how often the arms disagree, and which way), not an exact interval for the marginal risk "
    "difference. It is wider than an exact interval would be, which is the safe direction.",
    "A pre-registration file is self-certifying: whoever can write it can rewrite it and its "
    "digests in one edit. The external anchors are the copies of these digests in this run's "
    "EVIDENCE.md and in the git commit that publishes them.",
]


def _by_difficulty(rows, router_arm, comparator, category, manifest, difficulty):
    """The same pair counts, split by the difficulty declared at registration.

    Design averages over 22 compact components and 3 full pages; the average
    hides exactly the case the original run failed on.
    """
    out = {}
    for tier in ("easy", "medium", "hard"):
        ids = {t["id"] for t in manifest
               if t["category"] == category and difficulty.get(t["id"]) == tier}
        if not ids:
            continue
        subset = [r for r in rows if r["task_id"] in ids]
        entry = pairing.pair_accounting(
            subset, router_arm, comparator,
            manifest=[t for t in manifest if t["id"] in ids]).get(category, {})
        out[tier] = {k: entry.get(k) for k in
                     ("attempted_pairs", "valid_pairs", "invalid_truncated",
                      "router_passed", "comparator_passed")}
    return out


def _verdict(entry: dict, category: str) -> str:
    n = entry.get("valid_pairs", 0)
    if n < TARGET_VALID_PAIRS:
        return (f"{n} valid pairs: below the registered floor of {TARGET_VALID_PAIRS}. "
                "No quality statement either way.")
    b, c = entry["discordant"]
    independence = "" if entry.get("independent_samples", True) else \
        " NOTE: these tasks share one warm prefix and are not independent samples."
    lo, hi = entry["paired_difference_95"]
    span = (f"paired difference {entry['paired_difference']:+.3f} "
            f"(conservative 95% {lo:+.3f} to {hi:+.3f}, on {b + c} discordant pair(s) "
            f"out of {n})")
    if b == c == 0:
        return (f"{n} valid pairs, identical outcome on every one - the routing policy did as "
                f"well as the comparator and no better. That is the absence of a disagreement, "
                f"not equality: the conservative 95% interval still runs {lo:+.3f} to {hi:+.3f}."
                f"{independence}")
    if c > b:
        return (f"{n} valid pairs; the routing policy lost {c} pair(s) and won {b}. It is "
                f"BEHIND the comparator here, {span}, sign test p="
                f"{entry['sign_test_p']:.3f} - not significant at this size, and not evidence "
                f"of equivalence either.{independence}")
    return (f"{n} valid pairs; the routing policy won {b} pair(s) and lost {c}, {span}, "
            f"sign test p={entry['sign_test_p']:.3f}.{independence}")


def _manifest(directory: Path) -> list[dict]:
    """The registered task list of a run: the attempted universe, not the rows."""
    path = Path(directory) / "preregistration.json"
    if not path.exists():
        return []
    record = json.loads(path.read_text())
    task_file = Path(directory) / record.get("task_file", "tasks.jsonl")
    if not task_file.exists():
        return []
    return [{"id": t["id"], "category": t["category"], "difficulty": t.get("difficulty")}
            for t in (json.loads(line) for line in task_file.read_text().splitlines()
                      if line.strip())]


def _calls_usd_by_arm(directory: Path) -> dict[str, float]:
    """Every call, attributed to the arm whose tag it carries.

    ``tag`` is ``heldout:<arm>:<task-id>``. Without this the extra cost of
    discarded and re-run calls could only be shown as one aggregate, so an
    arm's displayed cost looked lower than what was spent to obtain its
    evidence - a reviewer's finding after the first repair.
    """
    out: dict[str, float] = {}
    for row in _read_jsonl(Path(directory) / "calls.jsonl"):
        parts = (row.get("tag") or "").split(":")
        arm = parts[1] if len(parts) > 2 and parts[0] == "heldout" else "untagged"
        out[arm] = out.get(arm, 0.0) + max(float(row.get("cost_usd") or 0.0),
                                           float(row.get("list_cost_usd") or 0.0))
    return out


def _calls_usd(directory: Path) -> float:
    """Every call the run made, including ones whose row was later discarded.

    Discarding a row from the analysis ledger does not refund the call. A
    reviewer pointed out that pricing an arm from its retained rows alone
    understates what the evaluation actually cost.
    """
    total = 0.0
    for row in _read_jsonl(Path(directory) / "calls.jsonl"):
        total += max(float(row.get("cost_usd") or 0.0), float(row.get("list_cost_usd") or 0.0))
    return total


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _registration_git_sha(directory: Path) -> str | None:
    path = Path(directory) / "preregistration.json"
    return json.loads(path.read_text()).get("repo_git_sha") if path.exists() else None


def _protocol_deviations(directory: Path) -> list[str]:
    """Where the run departed from the registered design, in its own words.

    The plan says every task goes to every arm "in one run". It did not: the
    metered comparator was a separate invocation minutes later, so provider
    load and cache state were not identical across arms. A reviewer found it in
    the invocation record - which is the point of keeping one.
    """
    invocations = _invocations(directory)
    out: list[str] = []
    arm_runs: dict[str, list[str]] = {}
    for inv in invocations:
        for arm in inv.get("arms") or []:
            label = inv.get("control_label") if arm == "control" else arm
            arm_runs.setdefault(label, []).append(inv.get("started_at") or "?")
    if len(invocations) > 1:
        later = [a for a, starts in arm_runs.items()
                 if starts and starts[0] != invocations[0].get("started_at")]
        for arm in sorted(later):
            out.append(
                f"the {arm} arm ran in a separate invocation ({arm_runs[arm][0]}), not in the "
                "same run as the routing arm. The registered design says one run with identical "
                "prompts; the prompts were identical, the provider load and cache state at the "
                "time were not.")
    return out


def _invocations(directory: Path) -> list[dict]:
    """What was actually run, read back from the per-invocation metadata.

    The registration cannot freeze the command line, so the next best thing is
    that every invocation leaves its own unrewritten record of the comparator,
    the arms, the categories and the worker count it used.
    """
    out = []
    for path in sorted(Path(directory).glob("run-meta-*.json")):
        meta = json.loads(path.read_text())
        out.append({k: meta.get(k) for k in
                    ("started_at", "finished_at", "arms", "workers", "control_model",
                     "control_label", "control_selected_by", "run_budget_usd",
                     "run_spend_usd", "stopped_early")})
    return out


def _read_rows(directory: Path) -> list[dict]:
    path = Path(directory) / "ledger.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _registration(directory: Path, role: str) -> dict:
    path = Path(directory) / "preregistration.json"
    record = json.loads(path.read_text()) if path.exists() else {}
    task_file = Path(directory) / record.get("task_file", "tasks.jsonl")
    intact = None
    if task_file.exists() and record.get("task_file_sha256"):
        intact = digest(task_file) == record["task_file_sha256"]
    return {
        "role": role,
        "directory": str(directory),
        "registered_at": record.get("registered_at"),
        "task_file_sha256": record.get("task_file_sha256"),
        "task_count": record.get("task_count"),
        "amendments": len(record.get("amendments") or []),
        "amendment_reasons": [a.get("reason") for a in (record.get("amendments") or [])],
        "task_file_digest_intact": intact,
        "policy_identity_sha256": record.get("policy_identity_sha256"),
    }


def combined_report(original_dir: Path, supplement_dir: Path,
                    out_path: Path | None = None,
                    task_dirs: list[Path] | None = None) -> dict:
    original_rows = _read_rows(original_dir)
    supplement_rows = _read_rows(supplement_dir)
    rows = original_rows + supplement_rows
    manifest = _manifest(original_dir) + _manifest(supplement_dir)
    difficulty = {t["id"]: t.get("difficulty") for t in manifest}

    arms_present = sorted({r["arm"] for r in rows}, key=lambda a: (a != ROUTER_ARM, a))
    comparators = [a for a in arms_present if a != ROUTER_ARM]

    per_category: dict[str, dict] = {c: {} for c in CATEGORIES}
    below: dict[str, list[str]] = {}
    floor_met: dict[str, list[str]] = {}
    for comparator in comparators:
        key = f"{ROUTER_ARM}_vs_{comparator}"
        accounting = pairing.pair_accounting(rows, ROUTER_ARM, comparator, manifest=manifest)
        supplement_only = pairing.pair_accounting(supplement_rows, ROUTER_ARM, comparator,
                                                  manifest=_manifest(supplement_dir))
        short, met = [], []
        for category in CATEGORIES:
            entry = dict(accounting.get(category, {}))
            b, c = pairing.mcnemar_discordant(entry) if entry else (0, 0)
            entry["discordant"] = [b, c]
            entry["sign_test_p"] = pairing.sign_test_p(b, c)
            entry["paired_difference"] = round((b - c) / entry["valid_pairs"], 4) \
                if entry.get("valid_pairs") else None
            entry["paired_difference_95"] = list(pairing.paired_difference_ci(entry)) \
                if entry.get("valid_pairs") else None
            n = entry.get("valid_pairs", 0)
            entry["router_pass_rate"] = round(entry.get("router_passed", 0) / n, 4) if n else None
            entry["comparator_pass_rate"] = (round(entry.get("comparator_passed", 0) / n, 4)
                                             if n else None)
            entry["router_wilson_95"] = [round(x, 4) for x in
                                         wilson(entry.get("router_passed", 0), n)] if n else None
            entry["comparator_wilson_95"] = [round(x, 4) for x in
                                             wilson(entry.get("comparator_passed", 0), n)] if n else None
            entry["meets_ten_valid_pairs"] = n >= TARGET_VALID_PAIRS
            # Reaching the sample floor is eligibility, not a result. A reviewer
            # was right that calling it "quality claim supported" reads as an
            # endorsement of a comparison the router does not actually win.
            entry["router_at_least_as_good"] = (
                entry.get("router_passed", 0) >= entry.get("comparator_passed", 0)) if n else None
            entry["supplement_only"] = supplement_only.get(category, {})
            entry["by_difficulty"] = _by_difficulty(rows, ROUTER_ARM, comparator,
                                                    category, manifest, difficulty)
            entry["verdict"] = _verdict(entry, category)
            if n < TARGET_VALID_PAIRS:
                short.append(category)
            else:
                met.append(category)
            per_category.setdefault(category, {})[key] = entry
        below[key] = short
        floor_met[key] = met

    calls_by_arm: dict[str, float] = {}
    for directory in (original_dir, supplement_dir):
        for arm, usd in _calls_usd_by_arm(directory).items():
            calls_by_arm[arm] = calls_by_arm.get(arm, 0.0) + usd
    arms: dict[str, dict] = {}
    for arm in arms_present:
        arm_rows = [r for r in rows if r["arm"] == arm]
        metered = [r["observed_cost_usd"] for r in arm_rows
                   if isinstance(r.get("observed_cost_usd"), (int, float))]
        arms[arm] = {
            "description": ARM_DESCRIPTIONS.get(arm, "an additional comparator"),
            "is_real_normal_policy": arm == ROUTER_ARM,
            "rows": len(arm_rows),
            "routes_used": sorted({r.get("model") for r in arm_rows if r.get("model")}),
            "cost_measurable": bool(metered),
            "measured_usd": round(sum(metered), 6) if metered else None,
            "metered_calls": len(metered),
            "unmetered_calls": len(arm_rows) - len(metered),
            "cost_bases": sorted({r.get("cost_basis") for r in arm_rows if r.get("cost_basis")}),
            "all_calls_usd": round(calls_by_arm.get(arm, 0.0), 6),
            "all_calls_note": ("every call tagged for this arm, including calls whose row was "
                               "later discarded and re-run; discarding a row does not refund it"),
        }

    exclusions = [r for r in rows if r.get("passed") is None]
    exclusions_by_arm: dict[str, dict[str, int]] = {}
    for row in exclusions:
        exclusions_by_arm.setdefault(row["arm"], {})
        exclusions_by_arm[row["arm"]][row["category"]] = \
            exclusions_by_arm[row["arm"]].get(row["category"], 0) + 1
    # "Supported" now means both things: enough pairs AND the router actually
    # matching the comparator. Either one alone is not the registered claim.
    # "At least as often" is a non-inferiority claim, and no margin for it was
    # pre-registered. Twelve concordant pairs are not evidence that the
    # population difference is non-negative; they are the absence of an observed
    # disagreement. Until a margin is registered in advance, no category can
    # support the claim, and the list stays empty by construction rather than by
    # accident.
    supported: list[str] = [] if NON_INFERIORITY_MARGIN is None else [
        c for c in CATEGORIES if per_category.get(c)
        and all(per_category[c][k].get("meets_ten_valid_pairs")
                and (per_category[c][k]["paired_difference_95"] or [-1, 1])[0]
                >= -NON_INFERIORITY_MARGIN
                for k in per_category[c])]
    original_reg = _registration(Path(original_dir), "original")
    supplement_reg = _registration(Path(supplement_dir), "supplement")
    spend = {
        "retained_rows_usd": round(sum(
            r["observed_cost_usd"] for r in rows
            if isinstance(r.get("observed_cost_usd"), (int, float))), 6),
        "all_calls_usd": round(_calls_usd(original_dir) + _calls_usd(supplement_dir), 6),
        "original_calls_usd": round(_calls_usd(original_dir), 6),
        "supplement_calls_usd": round(_calls_usd(supplement_dir), 6),
        "basis": ("retained rows: provider-reported cost of the calls still in the analysis "
                  "ledger. all calls: every call either run made, including calls whose row "
                  "was later discarded and re-run - discarding a row does not refund the call."),
    }
    spend["discarded_or_extra_usd"] = round(spend["all_calls_usd"] - spend["retained_rows_usd"], 6)

    data = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "registrations": [original_reg, supplement_reg],
        "rows_by_source": {"original": len(original_rows), "supplement": len(supplement_rows)},
        "arms": arms,
        "comparators": comparators,
        "target_valid_pairs_per_category": TARGET_VALID_PAIRS,
        "per_category": per_category,
        "categories_below_ten_valid_pairs": below,
        "categories_meeting_the_sample_floor": floor_met,
        "quality_claim_supported": supported,
        "non_inferiority_margin": NON_INFERIORITY_MARGIN,
        "quality_claim_definition": (
            "'at least as often' is a non-inferiority claim. It needs (a) at least 10 valid "
            "pairs and (b) a lower bound on the paired difference above a margin fixed in "
            "advance. No such margin was pre-registered, so no category supports the claim, "
            "whatever the pass counts look like. Reaching the pair floor is eligibility to "
            "look; an observed tie is the absence of a disagreement, not proof of equality."),
        "protocol_deviations": _protocol_deviations(supplement_dir),
        "repo_git_sha": {"at_registration": _registration_git_sha(supplement_dir),
                         "now": _repo_git_sha(),
                         "note": ("recorded, not enforced - publishing this run changes the SHA "
                                  "by construction, so a mismatch here is expected")},
        "policy_identity_frozen": {
            "original": bool(original_reg.get("policy_identity_sha256")),
            "supplement": bool(supplement_reg.get("policy_identity_sha256"))},
        "measured_spend": spend,
        "invocations": _invocations(supplement_dir),
        "excluded_rows": len(exclusions),
        "exclusions_by_arm": exclusions_by_arm,
        "exclusion_reasons": sorted({(r.get("detail") or "")[:60] for r in exclusions}),
        "claims_not_made": ANALYSIS_PLAN["claims_not_made"],
        "known_limitations": KNOWN_LIMITATIONS,
    }
    if out_path:
        Path(out_path).write_text(json.dumps(data, indent=1))
    return data


def format_combined(data: dict) -> str:
    lines = ["Combined held-out evidence - original run + pre-registered supplement", ""]
    for reg in data["registrations"]:
        lines.append(f"  {reg['role']:<11} {str(reg['task_file_sha256'])[:16]} "
                     f"registered {reg['registered_at']} "
                     f"{reg['task_count']} tasks, {reg['amendments']} amendment(s), "
                     f"digest intact: {reg['task_file_digest_intact']}")
    lines += ["", "Arms:"]
    for arm, info in data["arms"].items():
        cost = "no measurable cost (free routes)" if not info["cost_measurable"] \
            else f"measured ${info['measured_usd']:.4f} over {info['metered_calls']} call(s)"
        normal = "REAL NORMAL POLICY" if info["is_real_normal_policy"] else "comparator"
        lines.append(f"  {arm:<16} {normal:<18} {info['description']}")
        lines.append(f"  {'':<16} routes {', '.join(info['routes_used'])}; {cost}")
    for comparator in data["comparators"]:
        key = f"{ROUTER_ARM}_vs_{comparator}"
        lines += ["", f"router (real normal policy) vs {comparator} - valid PAIRS per category:"]
        header = (f"{'category':<14}{'attempted':>10}{'valid':>7}{'trunc':>7}{'sandbox':>8}"
                  f"{'incompl':>8}{'router':>8}{'compar':>8}{'b/c':>8}{'sign p':>8}  >=10?")
        lines += [header, "-" * len(header)]
        for category in CATEGORIES:
            e = data["per_category"][category][key]
            p = "n/a" if e["sign_test_p"] is None else f"{e['sign_test_p']:.3f}"
            rr = "n/a" if e["router_pass_rate"] is None else f"{e['router_pass_rate']:.2f}"
            cr = "n/a" if e["comparator_pass_rate"] is None else f"{e['comparator_pass_rate']:.2f}"
            lines.append(
                f"{category:<14}{e['attempted_pairs']:>10}{e['valid_pairs']:>7}"
                f"{e['invalid_truncated']:>7}{e['invalid_sandbox']:>8}{e['invalid_incomplete']:>8}"
                f"{rr:>8}{cr:>8}{str(e['discordant']):>8}{p:>8}  "
                f"{'yes' if e['meets_ten_valid_pairs'] else 'NO'}")
        short = data["categories_below_ten_valid_pairs"][key]
        if short:
            lines.append(f"  below {TARGET_VALID_PAIRS} valid pairs: " + ", ".join(short))
    lines += ["", "Verdict per category (registered question: does the routing policy solve "
                  "held-out tasks AT LEAST AS OFTEN as the comparator?):"]
    for comparator in data["comparators"]:
        key = f"{ROUTER_ARM}_vs_{comparator}"
        lines.append(f"  against {comparator}:")
        for category in CATEGORIES:
            lines.append(f"    {category:<14} {data['per_category'][category][key]['verdict']}")
    design = data["per_category"].get("design", {})
    if design:
        key = f"{ROUTER_ARM}_vs_{data['comparators'][0]}" if data["comparators"] else None
        tiers = design.get(key, {}).get("by_difficulty") or {}
        if tiers:
            lines += ["", "design by the difficulty declared at registration "
                          "(compact components are 'easy'/'medium', full pages are 'hard'):"]
            for tier, e in tiers.items():
                lines.append(f"    {tier:<7} attempted {e['attempted_pairs']:>2}, "
                             f"valid {e['valid_pairs']:>2}, truncated {e['invalid_truncated']:>2}, "
                             f"router {e['router_passed']}/{e['valid_pairs']}, "
                             f"comparator {e['comparator_passed']}/{e['valid_pairs']}")
    lines += ["", "Sensitivity - the same pairs with a TRUNCATED answer scored as a FAILURE "
                  "instead of excluded (the registered rule excludes it; that rule was written "
                  "after a truncation had already been seen):"]
    for comparator in data["comparators"]:
        key = f"{ROUTER_ARM}_vs_{comparator}"
        lines.append(f"  against {comparator}:")
        for category in CATEGORIES:
            e = data["per_category"][category][key]
            strict = e["if_truncation_counted_as_failure"]
            if strict["valid_pairs"] == e["valid_pairs"]:
                continue
            lines.append(
                f"    {category:<14} registered rule {e['router_passed']}/{e['valid_pairs']} vs "
                f"{e['comparator_passed']}/{e['valid_pairs']}   |   truncation-as-failure "
                f"{strict['router_passed']}/{strict['valid_pairs']} vs "
                f"{strict['comparator_passed']}/{strict['valid_pairs']}")
    lines += ["", "Supplement alone (the registration that was never amended before its run), "
                  "valid pairs, beside the pooled figure. Read this first: the supplement on its "
                  f"own does NOT reach {TARGET_VALID_PAIRS} valid pairs in most categories. The "
                  "floor is reached by pooling it with the original run, whose registration was "
                  "amended three times after outcomes were seen."]
    for comparator in data["comparators"]:
        key = f"{ROUTER_ARM}_vs_{comparator}"
        parts = []
        for category in CATEGORIES:
            e = data["per_category"][category][key]
            parts.append(f"{category} {e['supplement_only'].get('valid_pairs', 0)}/{e['valid_pairs']}")
        lines.append(f"  against {comparator}: " + ", ".join(parts))
    duplicates = {c: data["per_category"][c][f"{ROUTER_ARM}_vs_{k}"]["duplicate_task_ids"]
                  for c in CATEGORIES for k in data["comparators"]
                  if data["per_category"][c][f"{ROUTER_ARM}_vs_{k}"]["duplicate_task_ids"]}
    lines += ["", "Duplicate (task, arm) observations collapsed by last-write-wins: "
              + (json.dumps(duplicates) if duplicates else "none")]
    if data["invocations"]:
        lines += ["", "Invocations that produced the supplement's rows "
                      "(the registration cannot freeze a command line; each invocation leaves "
                      "its own record instead):"]
        for inv in data["invocations"]:
            lines.append(f"  {inv['started_at']} -> {inv['finished_at']}  arms={inv['arms']} "
                         f"workers={inv['workers']} control={inv['control_model']} "
                         f"({inv['control_label']}, {inv['control_selected_by']}) "
                         f"cap=${inv['run_budget_usd']} spent=${inv['run_spend_usd']} "
                         f"stopped_early={inv['stopped_early']}")
    if data["excluded_rows"]:
        lines += ["", f"Excluded rows, by arm ({data['excluded_rows']} in total). These are "
                      "TRUNCATED or SANDBOX UNAVAILABLE answers: they are removed from the "
                      "pair denominator, not scored as failures. Every category also carries "
                      "`if_truncation_counted_as_failure` for the opposite rule."]
        for arm, categories in sorted(data["exclusions_by_arm"].items()):
            lines.append(f"  {arm:<16} " + ", ".join(f"{c} {n}" for c, n in sorted(categories.items())))
        lines += [f"  - {reason}" for reason in data["exclusion_reasons"]]
    spend = data["measured_spend"]
    free_rows = sum(v["unmetered_calls"] for v in data["arms"].values())
    lines += ["", "Measured spend on the METERED arm (gateway-reported upstream inference cost, "
                  "not an invoice and not reconciled against one):",
              f"  retained analysis rows           ${spend['retained_rows_usd']:.4f}",
              f"  every metered call either run made ${spend['all_calls_usd']:.4f}",
              f"  difference (discarded/re-run)    ${spend['discarded_or_extra_usd']:.4f}",
              f"  The other {free_rows} rows are on routes configured at a zero price. That is a "
              "configuration fact; it is not a measurement that nothing was charged, and it is "
              "why no saving is computed anywhere in this report."]
    frozen = data["policy_identity_frozen"]
    lines += ["", "Policy/catalog identity: supplement "
              + ("frozen and re-verified" if frozen["supplement"] else "not frozen")
              + "; original " + ("frozen" if frozen["original"] else "not frozen")
              + ". The combined routing arm therefore carries a verified identity for the "
                "supplement's rows only."]
    if not data["quality_claim_supported"]:
        lines += ["", "No quality claim is supported in any category, and this is structural, "
                      "not an accident of the numbers: 'at least as often' is a non-inferiority "
                      "claim and **no non-inferiority margin was pre-registered**. Inventing one "
                      "now, with the results in hand, is the move this evaluation exists to "
                      "avoid. An observed tie on 12 pairs is the absence of a disagreement, not "
                      "proof of equality - the conservative paired interval on those categories "
                      "still admits a difference of roughly a quarter in either direction.",
                  f"What the {TARGET_VALID_PAIRS}-valid-pair floor does buy is a ceiling: a "
                  "routing advantage large enough to show up at these sample sizes is not "
                  "present in any category."]
    else:
        lines += ["", "Categories where the registered 'at least as often' claim is supported "
                  f"(>= {TARGET_VALID_PAIRS} valid pairs AND not behind the comparator, against "
                  "every comparator): " + ", ".join(data["quality_claim_supported"])]
    short = {k: v for k, v in data["categories_below_ten_valid_pairs"].items() if v}
    if not short:
        lines += ["Every category reaches the registered floor of "
                  f"{TARGET_VALID_PAIRS} valid pairs against every comparator."]
    if data.get("protocol_deviations"):
        lines += ["", "Protocol deviations from the registered design:"]
        lines += [f"  - {d}" for d in data["protocol_deviations"]]
    lines += ["", "Known limitations of this evidence:"] + \
             [f"  - {limitation}" for limitation in data["known_limitations"]]
    lines += ["", "Not claimed:"] + [f"  - {c}" for c in data["claims_not_made"]]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",
                        choices=["preregister", "verify", "run", "report", "amend", "identity"])
    parser.add_argument("--dir", type=Path, required=True, help="the supplement directory")
    parser.add_argument("--original", type=Path, help="the original held-out directory")
    parser.add_argument("--config", type=Path, help="router config")
    parser.add_argument("--reason", help="why the registration is being amended")
    parser.add_argument("--budget", type=float, default=0.0, help="hard USD cap for this run")
    parser.add_argument("--cap", type=float, default=30.0, help="absolute cumulative USD cap")
    parser.add_argument("--prior-spend", type=float, default=0.0,
                        help="USD already spent before this run, counted against --cap")
    parser.add_argument("--extra-ledger", action="append",
                        help="another call ledger to count against the cumulative cap")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--categories", default=",".join(CATEGORIES))
    parser.add_argument("--control")
    parser.add_argument("--control-label", default="control")
    parser.add_argument("--arms", default="router,control")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--out", type=Path, help="where to write the combined report")
    args = parser.parse_args(argv)

    if args.command == "preregister":
        if not args.config:
            raise SystemExit("--config is required: the registration freezes the policy and catalog")
        record = preregister(args.dir, args.config)
        print(json.dumps({k: v for k, v in record.items()
                          if k not in ("task_ids", "analysis_plan", "policy_identity")}, indent=1))
        return 0
    if args.command == "identity":
        print(json.dumps(policy_identity(load_frozen_config(args.config)), indent=1))
        return 0
    if args.command == "verify":
        record = load_preregistration(args.dir, config_path=args.config)
        print(f"supplement registration intact: {record['task_count']} tasks, "
              f"sha256 {record['task_file_sha256'][:16]}, "
              f"policy {record['policy_identity']['policy_name']} "
              f"identity {record['policy_identity_sha256'][:16]}, "
              f"{len(record.get('amendments') or [])} amendment(s)")
        return 0
    if args.command == "amend":
        if not args.reason:
            raise SystemExit("--reason is required: an amendment without a reason is a rewrite")
        print(json.dumps(amend(args.dir, args.reason, args.config).get("amendments"), indent=1))
        return 0
    if args.command == "report":
        if not args.original:
            raise SystemExit("--original is required for the combined report")
        data = combined_report(args.original, args.dir,
                               args.out or (args.dir / "combined-report.json"))
        print(format_combined(data))
        return 0
    if not args.config:
        raise SystemExit("--config is required for a live run")
    return run_live(args)


if __name__ == "__main__":
    raise SystemExit(main())
