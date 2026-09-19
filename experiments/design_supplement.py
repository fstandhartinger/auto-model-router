"""The 2026-09-19 design-floor supplement: registration, runner, combined design report.

The extended held-out run ``runs/heldout-extended-20260919T062948Z`` reached
ten valid pairs in five of its six categories. Design stopped at nine, because
five router design answers hit the registered 12,000-token output budget and the
registered rule excludes a truncated answer. That run - its registration, task
file, ledger, call log and report - is **immutable evidence**. Nothing here
writes to it, re-runs it, re-grades it or re-reads its outcomes differently.

This module is a **supplement, not a replacement**: two new design tasks
(``experiments/tasks_design_supplement.py``), the same two arms, registered
separately and before any call, in their own directory. Its registration pins
the historical run's files by sha256 as well, so the runner refuses to call a
model if the historical evidence has changed at all.

Four calls in a fixed order, one process, no retries: a failed call is recorded
as failed (the registered rule counts it as the route's failure), a truncated
answer is recorded as truncated, and neither is re-asked of any model. Before
every call a fail-closed guard adds the recorded spend of this supplement to
the worst case of that call and refuses if the sum could cross the cap.

Design answers are graded by ``graders.grade_design``: regular expressions over
the text. No design answer is executed, rendered or opened in a browser.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments import pairing                                 # noqa: E402
from experiments.heldout import digest, wilson                  # noqa: E402
from experiments.tasks_design_supplement import build_tasks     # noqa: E402

ROLE = "design-floor supplement"
ROUTER_ARM = "router"
CONTROL_ARM = "control-metered"
CONTROL_MODEL = "gpt-5.6-sol"
TARGET_VALID_PAIRS = 10

#: The hard cap on everything this supplement's calls may cost, in USD, on the
#: conservative basis max(billed, list) per call.
CAP_USD = 0.75

#: Every metered call of the historical run was billed at exactly 2x its list
#: price. The per-call worst case uses 2.5x so that a further surprise in the
#: same direction still cannot cross the cap.
BILLING_MULTIPLIER = 2.5

#: The registered output budget for design (``heldout_run.OUTPUT_BUDGET``),
#: repeated here so the registration states it; ``verify`` checks they agree.
DESIGN_OUTPUT_BUDGET = 12000

#: The exact order of the four calls. Sequential, one process.
RUN_ORDER = [
    ("ds2-design-easy-shortcut-card", ROUTER_ARM),
    ("ds2-design-easy-shortcut-card", CONTROL_ARM),
    ("ds2-design-easy-star-rating", ROUTER_ARM),
    ("ds2-design-easy-star-rating", CONTROL_ARM),
]

HISTORICAL_DIR_NAME = "heldout-extended-20260919T062948Z"

#: sha256 of every file of the historical run, taken on 2026-09-19 at 10:41Z,
#: before this supplement existed. They equal the digests the recovering session
#: recorded in its EVIDENCE §9 for the five files it listed.
HISTORICAL_FILES = {
    "calls.jsonl": "a372acc2a30ca9c94c5b918c8e4d589a0274ef23f2ad1f4477fa48718f26581b",
    "config-public-identity.json": "ece6bc366f88b4001854f23b9a84f8c8bdd9aee111a22f33f1f5c7e1f173f9ac",
    "ledger.jsonl": "71c52bdff8c8be36664196c544999aef4a26c7fbcc5e9c0dae8515574c6bf639",
    "preregistration.json": "8f72e59e224e2ae3b6176fb537afb5dcf37aaa5f59ed7f9bd2a41ad33210d561",
    "report.json": "1baba890110961bdb576b78263cec6338367890d7a02c888915365198b733cc7",
    "report.md": "9cf9b2cd531df8d5456a04df7dfb2d8a56bf0d6664596ad6acc727b2119c1f91",
    "report.txt": "32eb50ca9576385e98f3f9ae217c03dd928851d3d01b011fd4b461e509cf8f31",
    "run-meta.json": "6ec2b06c052ae09fcc7dced598dcc4ed47c345ab832e470edf06535a435f09cf",
    "run-meta.launch1-0646Z.json": "4e1034e0a86e5daca336d3202d87e832d2cc50247d34c91a1bbc75c83ea75852",
    "tasks.jsonl": "ff281eef4fb57eacadf28552a329b17db5209408a393990cb6a0990a76ec1fe9",
}
HISTORICAL_POLICY_IDENTITY = "123bd792bdeeaf0ab4d0206b1c75b1be465be7fe0fc1649dd0a3afce13a79ae8"

#: Historical design pairs whose deciding rule was not demanded by the prompt
#: (found by the predecessor's independent review, before this supplement).
#: Reported as a registered sensitivity only; the primary figure keeps them.
UNDEMANDED_RULE_PAIRS = ["design-easy-stat-tile", "design-medium-empty-state"]

ANALYSIS_PLAN = {
    "question": "Does the combined design evidence (historical extended run + this supplement) "
                "reach at least ten valid paired graded tasks, router vs control-metered "
                "gpt-5.6-sol, and what do those pairs show?",
    "role": "SUPPLEMENT to runs/heldout-extended-20260919T062948Z, not a replacement. The "
            "historical run is immutable; its files are pinned by sha256 in this registration "
            "and nothing in it is edited, re-run, re-graded or re-interpreted.",
    "arms": {ROUTER_ARM: "the real normal routing policy from the registered public config "
                         "(policy F_expected over the registered catalog); it may choose free routes",
             CONTROL_ARM: f"{CONTROL_MODEL}, metered, the same comparator as the historical run"},
    "tasks": "two new design tasks, ds2-design-easy-shortcut-card and ds2-design-easy-star-rating, "
             "whose every graded rule is demanded verbatim by the prompt",
    "run_order": [f"{t} / {a}" for t, a in RUN_ORDER],
    "execution": "sequential, one process, no retries of any kind (transport retries disabled), "
                 f"output budget {DESIGN_OUTPUT_BUDGET} tokens as registered for design",
    "cost_cap": f"hard {CAP_USD} USD over this supplement's calls, basis max(billed, list) per "
                f"call; before each call the recorded spend plus {BILLING_MULTIPLIER}x the "
                "list-price worst case of that call (full output budget, priciest catalog route, "
                "one token per prompt character) must stay within the cap, else no call is made",
    "exclusion_rule": "only a TRUNCATED answer (output tokens reached the budget) is excluded, "
                      "as in the historical run. A failed or refused upstream call is graded as "
                      "a failure of that route and its pair stays valid. A row that was never "
                      "attempted (guard refused) makes its pair incomplete",
    "combined_analysis": "pairs from both registrations are pooled by task id (ids are disjoint), "
                         "using experiments.pairing with both registered manifests as the "
                         "attempted universe. Reported: valid pairs, pass counts, Wilson 95% per "
                         "arm, discordant pairs, exact sign test, conservative paired-difference "
                         "interval, truncation-as-failure sensitivity, supplement-only and "
                         "historical-only figures, and the pooled figure without the two "
                         "historical pairs decided by an undemanded rule",
    "floor_rule": "the floor is met only if pooled valid design pairs >= 10",
    "difference_rule": "a difference is stated only if the floor is met AND the exact sign test "
                       "on discordant pairs gives p < 0.05; otherwise the report says no "
                       "difference was demonstrated",
    "claims_not_made": [
        "No equivalence or non-inferiority claim: no margin was registered.",
        "No quality claim below ten valid pairs.",
        "The design grader is a structural proxy, not a judgement of visual quality.",
        "No cash saving is claimed; the router's zero is a configured free price.",
        "The two supplement tasks are compact components and say nothing about full pages.",
        "The supplement was designed after the historical shortfall was known: it is an "
        "adaptive addition, recorded as such.",
    ],
}

#: Files whose content decides an outcome in this supplement.
CODE_FILES = ("design_supplement.py", "tasks_design_supplement.py", "graders.py",
              "heldout_run.py", "heldout.py", "llm.py", "pairing.py", "supplement.py",
              "evidence_verify.py")


# ---------------------------------------------------------------------------
# identity helpers
# ---------------------------------------------------------------------------
def plan_digest() -> str:
    return hashlib.sha256(json.dumps(ANALYSIS_PLAN, sort_keys=True).encode()).hexdigest()


def code_digests() -> dict[str, str]:
    here = Path(__file__).resolve().parent
    return {name: digest(here / name) for name in CODE_FILES}


def historical_digests(historical_dir: Path) -> dict[str, str | None]:
    return {name: (digest(Path(historical_dir) / name)
                   if (Path(historical_dir) / name).exists() else None)
            for name in HISTORICAL_FILES}


def write_tasks(path: Path) -> list[dict]:
    tasks = build_tasks()
    with Path(path).open("w") as fh:
        for task in tasks:
            fh.write(json.dumps(task, sort_keys=True) + "\n")
    return tasks


# ---------------------------------------------------------------------------
# pre-registration
# ---------------------------------------------------------------------------
def preregister(out_dir: Path, config_path: Path, historical_dir: Path) -> dict:
    from experiments import evidence_verify, supplement

    out_dir = Path(out_dir)
    if (out_dir / "preregistration.json").exists():
        raise SystemExit(f"{out_dir} is already registered; a registration is never rewritten")
    now = historical_digests(historical_dir)
    moved = [n for n, d in now.items() if d != HISTORICAL_FILES[n]]
    if moved:
        raise SystemExit(f"historical evidence differs from its pinned digests: {moved}")
    identity = supplement.policy_identity(supplement.load_frozen_config(config_path))
    if identity["sha256"] != HISTORICAL_POLICY_IDENTITY:
        raise SystemExit("the policy/catalog identity differs from the historical run's; "
                         "the router arm would not be the same arm")
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks = write_tasks(out_dir / "tasks.jsonl")
    public = evidence_verify.public_config_identity(config_path)
    record = {
        "registered_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "role": ROLE,
        "supplements": {"directory": f"runs/{HISTORICAL_DIR_NAME}",
                        "file_sha256": dict(HISTORICAL_FILES),
                        "note": "immutable; this supplement never writes to it and refuses to "
                                "run if any of these digests changes"},
        "task_file": "tasks.jsonl",
        "task_file_sha256": digest(out_dir / "tasks.jsonl"),
        "task_ids": [t["id"] for t in tasks],
        "task_count": len(tasks),
        "analysis_plan": ANALYSIS_PLAN,
        "analysis_plan_sha256": plan_digest(),
        "code_sha256": code_digests(),
        "product_sha256": supplement._product_digests(),
        "repo_git_sha": supplement._repo_git_sha(),
        "config_path": str(Path(config_path).resolve()),
        "config_sha256": digest(Path(config_path)),
        "public_config_identity_sha256": public["sha256"],
        "policy_identity": identity,
        "policy_identity_sha256": identity["sha256"],
        "arms": [ROUTER_ARM, CONTROL_ARM],
        "control_model": CONTROL_MODEL,
        "run_order": [list(x) for x in RUN_ORDER],
        "output_budget_tokens": DESIGN_OUTPUT_BUDGET,
        "cap_usd": CAP_USD,
        "billing_multiplier_for_worst_case": BILLING_MULTIPLIER,
        "target_valid_pairs": TARGET_VALID_PAIRS,
        "note": ("Written before any model call of this supplement. External anchors: the "
                 "copies of these digests in the job's EVIDENCE.md and in the git commit."),
    }
    (out_dir / "preregistration.json").write_text(json.dumps(record, indent=1))
    evidence_verify.freeze_public_config(out_dir, config_path)
    return record


def verify(out_dir: Path, config_path: Path | None, historical_dir: Path) -> list[tuple[str, bool, str]]:
    """Every check the runner requires, as (name, ok, detail). Never calls a model."""
    from experiments import evidence_verify, heldout_run, supplement

    out_dir = Path(out_dir)
    record = json.loads((out_dir / "preregistration.json").read_text())
    checks: list[tuple[str, bool, str]] = []

    def check(name, ok, detail=""):
        checks.append((name, bool(ok), detail))

    check("task file digest", digest(out_dir / record["task_file"]) == record["task_file_sha256"],
          record["task_file_sha256"][:16])
    rebuilt = [json.dumps(t, sort_keys=True) for t in build_tasks()]
    on_disk = [line for line in (out_dir / "tasks.jsonl").read_text().splitlines() if line]
    check("task file equals the deterministic builder", rebuilt == on_disk)
    check("analysis plan digest", record["analysis_plan_sha256"] == plan_digest())
    moved = [n for n, d in code_digests().items() if record["code_sha256"].get(n) != d]
    check("experiment code digests", not moved, ", ".join(moved) or f"{len(CODE_FILES)} files")
    product_now = supplement._product_digests()
    moved = sorted(n for n in set(product_now) | set(record["product_sha256"])
                   if product_now.get(n) != record["product_sha256"].get(n))
    check("product code digests", not moved, ", ".join(moved) or f"{len(product_now)} files")
    check("design output budget equals the registered harness value",
          heldout_run.OUTPUT_BUDGET["design"] == record["output_budget_tokens"]
          == DESIGN_OUTPUT_BUDGET)
    frozen = json.loads((out_dir / evidence_verify.PUBLIC_CONFIG_FILE).read_text())
    check("frozen public config identity matches the registration",
          frozen["public_identity"]["sha256"] == record["public_config_identity_sha256"])
    if config_path is not None:
        check("config file digest", digest(Path(config_path)) == record["config_sha256"])
        current = supplement.policy_identity(supplement.load_frozen_config(config_path))
        check("policy/catalog identity", current["sha256"] == record["policy_identity_sha256"],
              current["sha256"][:16])
        check("public config identity",
              evidence_verify.public_config_identity(config_path)["sha256"]
              == record["public_config_identity_sha256"])
    now = historical_digests(historical_dir)
    moved = [n for n, d in now.items() if d != record["supplements"]["file_sha256"].get(n)]
    check("historical run files unchanged", not moved,
          ", ".join(moved) or f"{len(now)} files byte-identical")
    check("no amendments", not record.get("amendments"))
    return checks


# ---------------------------------------------------------------------------
# the run
# ---------------------------------------------------------------------------
def _read_jsonl(path: Path) -> list[dict]:
    if not Path(path).exists():
        return []
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def spent_usd(calls_path: Path) -> float:
    """Cap-basis spend: per call the higher of billed and list. Fails on an unreadable line."""
    total = 0.0
    for line in Path(calls_path).read_text().splitlines() if Path(calls_path).exists() else []:
        if line.strip():
            row = json.loads(line)          # raises: an unreadable cost is not a zero
            total += max(float(row.get("cost_usd") or 0.0), float(row.get("list_cost_usd") or 0.0))
    return total


def worst_case_usd(config, task: dict) -> float:
    """The most one call for ``task`` could cost: full budget, priciest route, 1 token/char."""
    prompt_tokens = len(task["system"]) + len(task["prompt"])
    worst = max((prompt_tokens * m.prices.input + DESIGN_OUTPUT_BUDGET * m.prices.output) / 1e6
                for m in config.catalog.all())
    return worst * BILLING_MULTIPLIER


def call_tag(task_id: str, arm: str) -> str:
    # heldout_run._one tags a call with the arm it was *given*; the control arm
    # is called as "control" and recorded under its label.
    return f"heldout:{'control' if arm == CONTROL_ARM else arm}:{task_id}"


def run(out_dir: Path, config_path: Path, historical_dir: Path) -> int:
    from auto_router.config import load_config
    from auto_router.router import Router
    from experiments import heldout_run, llm, sandbox

    out_dir = Path(out_dir)
    failed = [c for c in verify(out_dir, config_path, historical_dir) if not c[1]]
    if failed:
        raise SystemExit("registration check failed, no model called: "
                         + "; ".join(f"{n} ({d})" for n, ok, d in failed))
    record = json.loads((out_dir / "preregistration.json").read_text())
    tasks = {t["id"]: t for t in _read_jsonl(out_dir / "tasks.jsonl")}
    config = load_config(Path(config_path))
    router = Router(config)
    if router.policy.name != record["policy_identity"]["policy_name"]:
        raise SystemExit(f"policy drift: {router.policy.name}")
    control = heldout_run.control_model(router, CONTROL_MODEL)

    class NoRetryClient(llm.Client):
        """Transport retries off: one request per row, whatever happens."""

        def chat(self, *a, **kw):
            kw["retries"] = 0
            return super().chat(*a, **kw)

    calls_path, ledger_path = out_dir / "calls.jsonl", out_dir / "ledger.jsonl"
    client = NoRetryClient(config, calls_path, budget_usd=CAP_USD)
    client.router_catalog = {m.name: m for m in config.catalog.all()}
    # Design answers are never executed; the preflight is recorded only because
    # heldout_run._one takes it. It gates coding tasks, of which there are none.
    isolation = sandbox.preflight()

    meta = {"started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "role": ROLE, "task_file_sha256": record["task_file_sha256"],
            "control_model": control.name, "run_order": record["run_order"],
            "cap_usd": CAP_USD, "spent_before_usd": round(spent_usd(calls_path), 6),
            "rows": []}
    meta_path = out_dir / f"run-meta-{meta['started_at'].replace(':', '')}.json"
    meta_path.write_text(json.dumps(meta, indent=1))
    print(f"{ROLE}: {len(RUN_ORDER)} rows; control {control.name}; policy {router.policy.name}; "
          f"cap ${CAP_USD:.2f}", flush=True)

    for task_id, arm in RUN_ORDER:
        task = tasks[task_id]
        rows = {(r["task_id"], r["arm"]) for r in _read_jsonl(ledger_path)}
        tags = {c.get("tag") for c in _read_jsonl(calls_path)}
        if (task_id, arm) in rows:
            meta["rows"].append({"row": [task_id, arm], "action": "already recorded, skipped"})
            continue
        if call_tag(task_id, arm) in tags:
            # A call without a ledger row: the process died mid-row. Never re-ask.
            meta["rows"].append({"row": [task_id, arm],
                                 "action": "call already made without a ledger row; NOT repeated"})
            continue
        spent = spent_usd(calls_path)
        worst = worst_case_usd(config, task)
        if spent + worst > CAP_USD:
            meta["rows"].append({"row": [task_id, arm], "action": "refused by the cap guard",
                                 "spent_usd": round(spent, 6), "worst_case_usd": round(worst, 6)})
            print(f"  {task_id} {arm}: refused, {spent:.4f} + {worst:.4f} > {CAP_USD}", flush=True)
            break
        outcome = heldout_run._one(task, "control" if arm == CONTROL_ARM else arm,
                                   router, control, client, isolation)
        outcome.arm = arm
        with ledger_path.open("a") as fh:
            fh.write(json.dumps(asdict(outcome)) + "\n")
        mark = {True: "pass", False: "FAIL", None: "excluded"}[outcome.passed]
        meta["rows"].append({"row": [task_id, arm], "action": "called", "model": outcome.model,
                             "outcome": mark, "output_tokens": outcome.output_tokens,
                             "worst_case_usd": round(worst, 6), "spent_before_usd": round(spent, 6)})
        meta_path.write_text(json.dumps(meta, indent=1))
        print(f"  {task_id:<32} {arm:<16} {outcome.model:<14} {mark:<8} "
              f"{outcome.output_tokens:>6} tok  {outcome.detail[:70]}", flush=True)

    meta["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    meta["spent_after_usd"] = round(spent_usd(calls_path), 6)
    meta_path.write_text(json.dumps(meta, indent=1))
    print(f"spend ${meta['spent_after_usd']:.6f} of ${CAP_USD:.2f}", flush=True)
    return 0


# ---------------------------------------------------------------------------
# read-back and the combined design report
# ---------------------------------------------------------------------------
def readback(out_dir: Path, historical_dir: Path) -> dict:
    """Integrity of both ledgers, straight from the files."""
    rows = _read_jsonl(Path(out_dir) / "ledger.jsonl")
    calls = _read_jsonl(Path(out_dir) / "calls.jsonl")
    keys = [(r["task_id"], r["arm"]) for r in rows]
    tags = [c.get("tag") for c in calls]
    registered = {tuple(x) for x in RUN_ORDER}
    return {
        "rows": len(rows),
        "duplicate_rows": sorted({k for k in keys if keys.count(k) > 1}),
        "unregistered_rows": sorted(set(keys) - registered),
        "missing_rows": sorted(registered - set(keys)),
        "calls": len(calls),
        "duplicate_call_tags": sorted({t for t in tags if tags.count(t) > 1}),
        "calls_without_row": sorted(t for t in set(tags)
                                    if t not in {call_tag(*k) for k in keys}),
        "failed_calls": [c.get("tag") for c in calls if not c.get("ok")],
        "billed_usd": round(sum(float(c.get("cost_usd") or 0) for c in calls), 6),
        "list_usd": round(sum(float(c.get("list_cost_usd") or 0) for c in calls), 6),
        "cap_basis_usd": round(spent_usd(Path(out_dir) / "calls.jsonl"), 6),
        "cap_usd": CAP_USD,
        "outcomes": [{"task_id": r["task_id"], "arm": r["arm"], "model": r["model"],
                      "passed": r["passed"], "output_tokens": r["output_tokens"],
                      "observed_cost_usd": r.get("observed_cost_usd"), "detail": r["detail"]}
                     for r in rows],
        "historical_files_unchanged": historical_digests(historical_dir) == HISTORICAL_FILES,
    }


def _manifest(directory: Path) -> list[dict]:
    return [{"id": t["id"], "category": t["category"]}
            for t in _read_jsonl(Path(directory) / "tasks.jsonl")]


def _summary(rows: list[dict], manifest: list[dict]) -> dict:
    entry = pairing.pair_accounting(rows, ROUTER_ARM, CONTROL_ARM,
                                    manifest=manifest).get("design", {})
    n = entry.get("valid_pairs", 0)
    b, c = pairing.mcnemar_discordant(entry) if entry else (0, 0)
    out = {k: entry.get(k) for k in
           ("attempted_pairs", "valid_pairs", "invalid_truncated", "invalid_incomplete",
            "invalid_other", "router_passed", "comparator_passed", "invalid_task_ids",
            "duplicate_observations", "unregistered_task_ids")}
    out.update({
        "discordant_router_only_vs_control_only": [b, c],
        "sign_test_p": pairing.sign_test_p(b, c),
        "router_wilson_95": [round(x, 4) for x in wilson(entry.get("router_passed", 0), n)] if n else None,
        "control_wilson_95": [round(x, 4) for x in wilson(entry.get("comparator_passed", 0), n)] if n else None,
        "paired_difference": round((b - c) / n, 4) if n else None,
        "paired_difference_95_conservative": list(pairing.paired_difference_ci(entry)) if n else None,
        "if_truncation_counted_as_failure": entry.get("if_truncation_counted_as_failure"),
    })
    return out


def combined_design_report(out_dir: Path, historical_dir: Path) -> dict:
    hist_rows = [r for r in _read_jsonl(Path(historical_dir) / "ledger.jsonl")
                 if r["category"] == "design"]
    hist_manifest = [t for t in _manifest(historical_dir) if t["category"] == "design"]
    supp_rows = _read_jsonl(Path(out_dir) / "ledger.jsonl")
    supp_manifest = _manifest(out_dir)
    overlap = {t["id"] for t in hist_manifest} & {t["id"] for t in supp_manifest}
    if overlap:
        raise SystemExit(f"task ids overlap between the registrations: {sorted(overlap)}")
    pooled = _summary(hist_rows + supp_rows, hist_manifest + supp_manifest)
    n = pooled["valid_pairs"]
    floor_met = n >= TARGET_VALID_PAIRS
    p = pooled["sign_test_p"]
    if not floor_met:
        answer = (f"floor NOT met: {n} valid design pairs, below {TARGET_VALID_PAIRS}. No quality "
                  "statement either way.")
    elif p is not None and p < 0.05:
        b, c = pooled["discordant_router_only_vs_control_only"]
        answer = (f"floor met ({n} pairs); a difference is demonstrated (sign test p={p:.3f}, "
                  f"{b} router-only vs {c} control-only passes).")
    else:
        answer = (f"floor met ({n} valid pairs); no difference demonstrated (sign test "
                  f"p={'n/a' if p is None else f'{p:.3f}'}). This is not equivalence: no margin "
                  "was registered and the paired interval is wide.")
    keep = [t for t in hist_manifest if t["id"] not in UNDEMANDED_RULE_PAIRS]
    data = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "historical_run": str(historical_dir),
        "historical_files_unchanged": historical_digests(historical_dir) == HISTORICAL_FILES,
        "supplement_run": str(out_dir),
        "arms": {ROUTER_ARM: "real normal routing policy", CONTROL_ARM: CONTROL_MODEL},
        "pooled": pooled,
        "historical_only": _summary(hist_rows, hist_manifest),
        "supplement_only": _summary(supp_rows, supp_manifest),
        "sensitivity_without_undemanded_rule_pairs": {
            "dropped": UNDEMANDED_RULE_PAIRS,
            **_summary([r for r in hist_rows if r["task_id"] not in UNDEMANDED_RULE_PAIRS]
                       + supp_rows, keep + supp_manifest)},
        "floor": TARGET_VALID_PAIRS,
        "floor_met": floor_met,
        "answer": answer,
        "difference_rule": ANALYSIS_PLAN["difference_rule"],
        "claims_not_made": ANALYSIS_PLAN["claims_not_made"],
    }
    return data


def format_markdown(data: dict) -> str:
    def row(name, s):
        rw = s["router_wilson_95"] or ["-", "-"]
        cw = s["control_wilson_95"] or ["-", "-"]
        p = "n/a" if s["sign_test_p"] is None else f"{s['sign_test_p']:.3f}"
        ci = s["paired_difference_95_conservative"] or ["-", "-"]
        vp = s["valid_pairs"]
        return (f"| {name} | {s['attempted_pairs']} | {vp} | {s['invalid_truncated']} | "
                f"{s['invalid_incomplete'] + s['invalid_other']} | {s['router_passed']}/{vp} "
                f"({rw[0]}–{rw[1]}) | {s['comparator_passed']}/{vp} ({cw[0]}–{cw[1]}) | "
                f"{s['discordant_router_only_vs_control_only'][0]}/"
                f"{s['discordant_router_only_vs_control_only'][1]} | {p} | {ci[0]} to {ci[1]} |")

    lines = ["# Combined design result — historical extended run + design-floor supplement", "",
             f"Historical run files unchanged: **{data['historical_files_unchanged']}**. "
             "The supplement adds two separately pre-registered tasks; it replaces nothing.", "",
             "| source | attempted | valid pairs | truncated | incomplete/other | router pass "
             "(Wilson 95 %) | control-metered pass (Wilson 95 %) | discordant r/c | sign p | "
             "paired diff (conservative 95 %) |",
             "|---|---:|---:|---:|---:|---|---|---|---|---|",
             row("historical only", data["historical_only"]),
             row("supplement only", data["supplement_only"]),
             row("**pooled**", data["pooled"]),
             row("pooled without 2 undemanded-rule pairs",
                 data["sensitivity_without_undemanded_rule_pairs"]), "",
             f"**Answer:** {data['answer']}", ""]
    strict = data["pooled"]["if_truncation_counted_as_failure"] or {}
    lines += [f"Sensitivity, truncation counted as failure: router {strict.get('router_passed')}/"
              f"{strict.get('valid_pairs')} vs control {strict.get('comparator_passed')}/"
              f"{strict.get('valid_pairs')}.", "", "Not claimed:"]
    lines += [f"- {c}" for c in data["claims_not_made"]]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["preregister", "verify", "run", "readback", "report"])
    parser.add_argument("--dir", type=Path, required=True)
    parser.add_argument("--historical", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    args = parser.parse_args(argv)
    if args.command == "preregister":
        if not args.config:
            raise SystemExit("--config is required")
        record = preregister(args.dir, args.config, args.historical)
        print(json.dumps({k: v for k, v in record.items()
                          if k not in ("analysis_plan", "policy_identity")}, indent=1))
        return 0
    if args.command == "verify":
        checks = verify(args.dir, args.config, args.historical)
        for name, ok, detail in checks:
            print(f"  {'PASS' if ok else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
        bad = sum(not ok for _, ok, _ in checks)
        print(f"{'PASS' if not bad else 'FAIL'} {len(checks) - bad}/{len(checks)}")
        return 1 if bad else 0
    if args.command == "run":
        if not args.config:
            raise SystemExit("--config is required")
        return run(args.dir, args.config, args.historical)
    if args.command == "readback":
        print(json.dumps(readback(args.dir, args.historical), indent=1))
        return 0
    data = combined_design_report(args.dir, args.historical)
    (args.dir / "combined-design-report.json").write_text(json.dumps(data, indent=1))
    text = format_markdown(data)
    (args.dir / "combined-design-report.md").write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
