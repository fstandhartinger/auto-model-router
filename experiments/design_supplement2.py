"""The second design-floor supplement: a credential-gated registration and runner.

The first supplement (``runs/design-supplement-20260919T1046Z``) lost both of
its control calls to HTTP 401 before any model ran: ``OPEN_ROUTER_API_KEY`` was
not set in the runner's environment, so the request carried no credential. A
401 like that is a **harness failure**, not a failure of the route, and it made
neither pair valid. Both earlier runs - the extended held-out run and the first
supplement - are **immutable evidence**; this module pins every file of both by
sha256 and never writes to them.

What is new here is the **credential preflight**. Before anything else the
runner checks, in its own process and environment, that the key variable of
every provider the two arms can reach is set. It records one boolean per
variable, never a value, and if any is missing it refuses before a client
exists, so no HTTP request or model call can happen. The same check is exposed
as the ``preflight`` command so the launching shell can be tested too.

Everything else follows the first supplement (``design_supplement``): two new
tasks (``tasks_design_supplement2``), the same two arms, four calls in a fixed
order in one process, no retries of any kind, a per-call worst-case cap guard,
and grading by ``graders.grade_design`` - regular expressions over the text.
No design answer is executed, rendered or opened in a browser.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments import design_supplement as ds1                 # noqa: E402
from experiments.heldout import digest                           # noqa: E402
from experiments.tasks_design_supplement2 import build_tasks     # noqa: E402

ROLE = "second design-floor supplement (credential-gated)"
ROUTER_ARM = ds1.ROUTER_ARM
CONTROL_ARM = ds1.CONTROL_ARM
CONTROL_MODEL = ds1.CONTROL_MODEL
TARGET_VALID_PAIRS = ds1.TARGET_VALID_PAIRS

#: Hard cap on everything this supplement's calls may cost, USD, basis
#: max(billed, list) per call.
CAP_USD = 0.75
BILLING_MULTIPLIER = ds1.BILLING_MULTIPLIER
DESIGN_OUTPUT_BUDGET = ds1.DESIGN_OUTPUT_BUDGET

RUN_ORDER = [
    ("ds3-design-easy-opening-hours", ROUTER_ARM),
    ("ds3-design-easy-opening-hours", CONTROL_ARM),
    ("ds3-design-easy-recipe-card", ROUTER_ARM),
    ("ds3-design-easy-recipe-card", CONTROL_ARM),
]

HISTORICAL_DIR_NAME = ds1.HISTORICAL_DIR_NAME
HISTORICAL_FILES = dict(ds1.HISTORICAL_FILES)
HISTORICAL_POLICY_IDENTITY = ds1.HISTORICAL_POLICY_IDENTITY

FIRST_SUPPLEMENT_DIR_NAME = "design-supplement-20260919T1046Z"
#: sha256 of every file of the first supplement, taken 2026-09-19T14:45Z before
#: this module existed (job directory: historical-digests-before.txt).
FIRST_SUPPLEMENT_FILES = {
    "READ-FIRST-harness-failure.md": "5d1e6412848581f5ee5042bf86d0df93e90ccda4737e98bc323f3c458218d830",
    "calls.jsonl": "079be5bf1132e54d20158d147d737cc5bcbaed0cfe9c952dd2a401c9c40e1702",
    "combined-design-report.json": "e16498ec877da6624a6354d8827f02b8068af467a11ce1febfb00a4be2dd56ea",
    "combined-design-report.md": "37141700b62d18fb65b6de1e485f18b9931b7feb5bb7c6ef191cfa4e501d9b0c",
    "config-public-identity.json": "ece6bc366f88b4001854f23b9a84f8c8bdd9aee111a22f33f1f5c7e1f173f9ac",
    "ledger.jsonl": "c294f59fb6c3a01548a3f1f7852059bb5a5dd9094804b1a9d219a4ab232eae4e",
    "preregistration.json": "52a969dcdebd033481ab334b3dcc26d38d645e597d0abb50e8ba655a1525e25a",
    "run-meta-2026-09-19T104656Z.json": "e868a612ef8120de9f0d1e3217cf1bfe5905aa896d4ecf5b61eef0542ad6d3b1",
    "tasks.jsonl": "a60052083562494bbe6c4e9e410858302508bc28737bf702d1cee97802efe9d2",
}

#: The error prefixes that mean the request was refused for its credential.
AUTH_FAILURE_PREFIXES = ("HTTP 401", "HTTP 403")
HARNESS_DETAIL = ("HARNESS FAILURE: the request was rejected for its credential before any "
                  "model ran (0 prompt and 0 output tokens); not a quality outcome")

ANALYSIS_PLAN = {
    "question": "Does the combined design evidence (extended held-out run + first supplement + "
                "this second supplement) reach at least ten valid paired graded tasks, router "
                "vs control-metered gpt-5.6-sol, and what do those pairs show?",
    "role": "SUPPLEMENT, not a replacement, to runs/heldout-extended-20260919T062948Z and "
            "runs/design-supplement-20260919T1046Z. Both are immutable; every file of both is "
            "pinned by sha256 here and nothing in them is edited, re-run or re-graded.",
    "arms": {ROUTER_ARM: "the real normal routing policy from the registered public config "
                         "(policy F_expected over the registered catalog); it may choose free routes",
             CONTROL_ARM: f"{CONTROL_MODEL}, metered via OpenRouter, the same comparator as the "
                          "historical run"},
    "tasks": "two new design tasks, ds3-design-easy-opening-hours and ds3-design-easy-recipe-card, "
             "whose every graded rule is demanded verbatim by the prompt",
    "credential_preflight": "before any client is built, in the runner's own process, the key "
                            "variable of every provider the catalog can route to (always including "
                            "the control's, OPEN_ROUTER_API_KEY) must be set and non-empty. Only "
                            "a boolean per variable is recorded. If any is missing the runner "
                            "writes a refusal record and exits: no HTTP request, no model call, "
                            "no ledger row",
    "run_order": [f"{t} / {a}" for t, a in RUN_ORDER],
    "execution": "sequential, one process, no retries of any kind (transport retries disabled; "
                 "a failed row is never re-asked of any model, by this or any later launch), "
                 f"output budget {DESIGN_OUTPUT_BUDGET} tokens as registered for design",
    "cost_cap": f"hard {CAP_USD} USD over this supplement's calls, basis max(billed, list) per "
                f"call; before each call the recorded spend plus {BILLING_MULTIPLIER}x the "
                "list-price worst case of that call (full output budget, priciest catalog route, "
                "one token per prompt character) must stay within the cap, else no call is made",
    "exclusion_rule": "a TRUNCATED answer (output tokens reached the budget) is excluded, as in "
                      "the historical run. A call rejected for its credential (HTTP 401/403 with "
                      "0 prompt and 0 output tokens) is a HARNESS FAILURE: excluded, never a "
                      "quality outcome, and the runner stops at once so no further call is made. "
                      "Any other failed or refused upstream call is graded as a failure of that "
                      "route and its pair stays valid. A row never attempted makes its pair "
                      "incomplete",
    "combined_analysis": "pairs from all three registrations are pooled by task id (ids are "
                         "disjoint) with experiments.pairing and all three manifests as the "
                         "attempted universe. The harness-failure rule is applied to rows of the "
                         "two supplements (the extended run has no credential failure, so it is "
                         "counted exactly as its registered report counts it); the first "
                         "supplement therefore contributes 0 valid pairs. Reported: valid pairs, "
                         "pass counts, Wilson 95% per arm, discordant pairs, exact sign test, "
                         "conservative paired-difference interval, truncation-as-failure "
                         "sensitivity, per-source figures, and the pooled figure without the two "
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
        "The supplement tasks are compact components and say nothing about full pages.",
        "Both supplements were designed after the historical shortfall was known: they are "
        "adaptive additions, recorded as such.",
    ],
}

CODE_FILES = ("design_supplement2.py", "tasks_design_supplement2.py", "design_supplement.py",
              "tasks_design_supplement.py", "graders.py", "heldout_run.py", "heldout.py",
              "llm.py", "pairing.py", "supplement.py", "evidence_verify.py")


# ---------------------------------------------------------------------------
# credential preflight
# ---------------------------------------------------------------------------
def credential_preflight(config, environ=None) -> dict:
    """Presence of every key variable the two arms can need. Booleans only.

    Reads each variable solely to test ``bool(value.strip())``; the value is
    never returned, stored, compared, hashed or printed.
    """
    environ = os.environ if environ is None else environ
    control = next((m for m in config.catalog.all() if m.name == CONTROL_MODEL), None)
    needed = {m.provider for m in config.catalog.all()}
    variables = {}
    for name in sorted(needed):
        provider = config.providers.get(name)
        var = provider.api_key_env if provider is not None else None
        variables[name] = {"variable": var,
                           "present": bool(var) and bool((environ.get(var) or "").strip())}
    control_ok = control is not None and variables.get(control.provider, {}).get("present", False)
    return {
        "checked_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "pid": os.getpid(),
        "control_model": CONTROL_MODEL,
        "control_provider": control.provider if control else None,
        "control_credential_present": bool(control_ok),
        "providers": variables,
        "ok": bool(control_ok) and all(v["present"] for v in variables.values()),
    }


def require_credentials(config, out_dir: Path | None = None, environ=None) -> dict:
    """Fail closed: raise before any client exists if a key variable is absent."""
    result = credential_preflight(config, environ)
    if not result["ok"]:
        missing = sorted(v["variable"] or f"<{p}: no variable configured>"
                         for p, v in result["providers"].items() if not v["present"])
        if out_dir is not None:
            stamp = result["checked_at"].replace(":", "")
            (Path(out_dir) / f"preflight-refused-{stamp}.json").write_text(
                json.dumps(result, indent=1))
        raise SystemExit("credential preflight FAILED, no model called: not set in this "
                         f"environment: {', '.join(missing)}")
    return result


# ---------------------------------------------------------------------------
# identity helpers
# ---------------------------------------------------------------------------
def plan_digest() -> str:
    return hashlib.sha256(json.dumps(ANALYSIS_PLAN, sort_keys=True).encode()).hexdigest()


def code_digests() -> dict[str, str]:
    here = Path(__file__).resolve().parent
    return {name: digest(here / name) for name in CODE_FILES}


def _digests(directory: Path, pinned: dict) -> dict[str, str | None]:
    directory = Path(directory)
    present = {p.name for p in directory.iterdir()} if directory.is_dir() else set()
    names = set(pinned) | present
    return {n: (digest(directory / n) if (directory / n).is_file() else None) for n in sorted(names)}


def pinned_state(runs_dir: Path) -> dict[str, bool]:
    """Whether each earlier run is byte-identical to its pin (no file added or removed)."""
    runs_dir = Path(runs_dir)
    return {
        HISTORICAL_DIR_NAME: _digests(runs_dir / HISTORICAL_DIR_NAME, HISTORICAL_FILES)
        == HISTORICAL_FILES,
        FIRST_SUPPLEMENT_DIR_NAME: _digests(runs_dir / FIRST_SUPPLEMENT_DIR_NAME,
                                            FIRST_SUPPLEMENT_FILES) == FIRST_SUPPLEMENT_FILES,
    }


def write_tasks(path: Path) -> list[dict]:
    tasks = build_tasks()
    with Path(path).open("w") as fh:
        for task in tasks:
            fh.write(json.dumps(task, sort_keys=True) + "\n")
    return tasks


# ---------------------------------------------------------------------------
# pre-registration
# ---------------------------------------------------------------------------
def preregister(out_dir: Path, config_path: Path, runs_dir: Path) -> dict:
    from experiments import evidence_verify, supplement

    out_dir = Path(out_dir)
    if (out_dir / "preregistration.json").exists():
        raise SystemExit(f"{out_dir} is already registered; a registration is never rewritten")
    moved = [n for n, ok in pinned_state(runs_dir).items() if not ok]
    if moved:
        raise SystemExit(f"earlier evidence differs from its pinned digests: {moved}")
    identity = supplement.policy_identity(supplement.load_frozen_config(config_path))
    if identity["sha256"] != HISTORICAL_POLICY_IDENTITY:
        raise SystemExit("the policy/catalog identity differs from the historical run's")
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks = write_tasks(out_dir / "tasks.jsonl")
    public = evidence_verify.public_config_identity(config_path)
    record = {
        "registered_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "role": ROLE,
        "supplements": {
            "directories": {f"runs/{HISTORICAL_DIR_NAME}": dict(HISTORICAL_FILES),
                            f"runs/{FIRST_SUPPLEMENT_DIR_NAME}": dict(FIRST_SUPPLEMENT_FILES)},
            "note": "immutable; this supplement never writes to them and refuses to run if any "
                    "file of either is changed, added or removed"},
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
        "no_retry": True,
        "note": ("Written before any model call of this supplement. External anchors: the "
                 "copies of these digests in the job's EVIDENCE.md and in the git commit."),
    }
    (out_dir / "preregistration.json").write_text(json.dumps(record, indent=1))
    evidence_verify.freeze_public_config(out_dir, config_path)
    return record


def verify(out_dir: Path, config_path: Path | None, runs_dir: Path) -> list[tuple[str, bool, str]]:
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
    check("analysis plan digest", record["analysis_plan_sha256"] == plan_digest()
          and record["analysis_plan"] == json.loads(json.dumps(ANALYSIS_PLAN)))
    check("run order and cap as registered",
          record["run_order"] == [list(x) for x in RUN_ORDER] and record["cap_usd"] == CAP_USD
          <= 0.75 and record.get("no_retry") is True)
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
        check("policy/catalog identity", current["sha256"] == record["policy_identity_sha256"]
              == HISTORICAL_POLICY_IDENTITY, current["sha256"][:16])
        check("public config identity",
              evidence_verify.public_config_identity(config_path)["sha256"]
              == record["public_config_identity_sha256"])
    pinned = record["supplements"]["directories"]
    for name, files in ((HISTORICAL_DIR_NAME, HISTORICAL_FILES),
                        (FIRST_SUPPLEMENT_DIR_NAME, FIRST_SUPPLEMENT_FILES)):
        now = _digests(Path(runs_dir) / name, files)
        moved = [n for n in now if now[n] != pinned[f"runs/{name}"].get(n)]
        check(f"earlier run unchanged: {name}", not moved and files == pinned[f"runs/{name}"],
              ", ".join(moved) or f"{len(now)} files byte-identical")
    check("no amendments", not record.get("amendments"))
    return checks


# ---------------------------------------------------------------------------
# the run
# ---------------------------------------------------------------------------
def is_harness_failure(call) -> bool:
    """A request refused for its credential before any model ran."""
    error = getattr(call, "error", None) if not isinstance(call, dict) else call.get("error")
    ok = getattr(call, "ok", None) if not isinstance(call, dict) else call.get("ok")
    if isinstance(call, dict):
        prompt, output = call.get("prompt", 0), call.get("output", 0)
    else:
        prompt, output = call.prompt_tokens, call.output_tokens
    return (not ok and (error or "").startswith(AUTH_FAILURE_PREFIXES)
            and not prompt and not output)


def run(out_dir: Path, config_path: Path, runs_dir: Path, environ=None) -> int:
    from auto_router.config import load_config
    from auto_router.router import Router
    from experiments import heldout_run, llm, sandbox

    out_dir = Path(out_dir)
    config = load_config(Path(config_path))
    # First, before anything that could open a connection: the credentials.
    preflight = require_credentials(config, out_dir, environ)
    failed = [c for c in verify(out_dir, config_path, runs_dir) if not c[1]]
    if failed:
        raise SystemExit("registration check failed, no model called: "
                         + "; ".join(f"{n} ({d})" for n, ok, d in failed))
    record = json.loads((out_dir / "preregistration.json").read_text())
    tasks = {t["id"]: t for t in ds1._read_jsonl(out_dir / "tasks.jsonl")}
    router = Router(config)
    if router.policy.name != record["policy_identity"]["policy_name"]:
        raise SystemExit(f"policy drift: {router.policy.name}")
    control = heldout_run.control_model(router, CONTROL_MODEL)

    calls_path, ledger_path = out_dir / "calls.jsonl", out_dir / "ledger.jsonl"
    if ds1._read_jsonl(calls_path) or ds1._read_jsonl(ledger_path):
        # One launch only. A second launch could only re-ask or continue after
        # outcomes are known; neither is registered.
        raise SystemExit("this supplement has already been launched; it is never relaunched")

    last_call = {}

    class NoRetryClient(llm.Client):
        """Transport retries off: one request per row, whatever happens."""

        def chat(self, *a, **kw):
            kw["retries"] = 0
            result = super().chat(*a, **kw)
            last_call["result"] = result
            return result

    client = NoRetryClient(config, calls_path, budget_usd=CAP_USD)
    client.router_catalog = {m.name: m for m in config.catalog.all()}
    # Design answers are never executed; heldout_run._one takes the preflight
    # only because coding tasks need it. There are none here.
    isolation = sandbox.preflight()

    meta = {"started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "role": ROLE, "task_file_sha256": record["task_file_sha256"],
            "credential_preflight": preflight,
            "control_model": control.name, "run_order": record["run_order"],
            "cap_usd": CAP_USD, "spent_before_usd": round(ds1.spent_usd(calls_path), 6),
            "rows": []}
    meta_path = out_dir / f"run-meta-{meta['started_at'].replace(':', '')}.json"
    meta_path.write_text(json.dumps(meta, indent=1))
    print(f"{ROLE}: {len(RUN_ORDER)} rows; control {control.name}; policy {router.policy.name}; "
          f"cap ${CAP_USD:.2f}; credential preflight ok", flush=True)

    for task_id, arm in RUN_ORDER:
        task = tasks[task_id]
        spent = ds1.spent_usd(calls_path)
        worst = ds1.worst_case_usd(config, task)
        if spent + worst > CAP_USD:
            meta["rows"].append({"row": [task_id, arm], "action": "refused by the cap guard",
                                 "spent_usd": round(spent, 6), "worst_case_usd": round(worst, 6)})
            print(f"  {task_id} {arm}: refused, {spent:.4f} + {worst:.4f} > {CAP_USD}", flush=True)
            break
        last_call.clear()
        outcome = heldout_run._one(task, "control" if arm == CONTROL_ARM else arm,
                                   router, control, client, isolation)
        outcome.arm = arm
        harness = "result" in last_call and is_harness_failure(last_call["result"])
        if harness:
            outcome.passed, outcome.detail = None, HARNESS_DETAIL
        with ledger_path.open("a") as fh:
            fh.write(json.dumps(asdict(outcome)) + "\n")
        mark = {True: "pass", False: "FAIL", None: "excluded"}[outcome.passed]
        meta["rows"].append({"row": [task_id, arm], "action": "called", "model": outcome.model,
                             "outcome": mark, "harness_failure": harness,
                             "output_tokens": outcome.output_tokens,
                             "worst_case_usd": round(worst, 6), "spent_before_usd": round(spent, 6)})
        meta_path.write_text(json.dumps(meta, indent=1))
        print(f"  {task_id:<32} {arm:<16} {outcome.model:<14} {mark:<8} "
              f"{outcome.output_tokens:>6} tok  {outcome.detail[:70]}", flush=True)
        if harness:
            meta["stopped"] = "harness failure (credential rejected); no further call made"
            print("  stopped: " + meta["stopped"], flush=True)
            break

    meta["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    meta["spent_after_usd"] = round(ds1.spent_usd(calls_path), 6)
    meta_path.write_text(json.dumps(meta, indent=1))
    print(f"spend ${meta['spent_after_usd']:.6f} of ${CAP_USD:.2f}", flush=True)
    return 0


# ---------------------------------------------------------------------------
# read-back and the combined design report
# ---------------------------------------------------------------------------
def readback(out_dir: Path, runs_dir: Path) -> dict:
    rows = ds1._read_jsonl(Path(out_dir) / "ledger.jsonl")
    calls = ds1._read_jsonl(Path(out_dir) / "calls.jsonl")
    keys = [(r["task_id"], r["arm"]) for r in rows]
    tags = [c.get("tag") for c in calls]
    registered = {tuple(x) for x in RUN_ORDER}
    paid = [c.get("tag") for c in calls
            if max(float(c.get("cost_usd") or 0), float(c.get("list_cost_usd") or 0)) > 0]
    return {
        "rows": len(rows),
        "duplicate_rows": sorted({k for k in keys if keys.count(k) > 1}),
        "unregistered_rows": sorted(set(keys) - registered),
        "missing_rows": sorted(registered - set(keys)),
        "calls": len(calls),
        "duplicate_call_tags": sorted({t for t in tags if tags.count(t) > 1}),
        "paid_calls": paid,
        "duplicate_paid_calls": sorted({t for t in paid if paid.count(t) > 1}),
        "calls_without_row": sorted(t for t in set(tags)
                                    if t not in {ds1.call_tag(*k) for k in keys}),
        "failed_calls": [c.get("tag") for c in calls if not c.get("ok")],
        "harness_failures": [c.get("tag") for c in calls if is_harness_failure(c)],
        "billed_usd": round(sum(float(c.get("cost_usd") or 0) for c in calls), 6),
        "list_usd": round(sum(float(c.get("list_cost_usd") or 0) for c in calls), 6),
        "cap_basis_usd": round(ds1.spent_usd(Path(out_dir) / "calls.jsonl"), 6),
        "cap_usd": CAP_USD,
        "outcomes": [{"task_id": r["task_id"], "arm": r["arm"], "model": r["model"],
                      "passed": r["passed"], "output_tokens": r["output_tokens"],
                      "observed_cost_usd": r.get("observed_cost_usd"), "detail": r["detail"]}
                     for r in rows],
        "earlier_runs_unchanged": pinned_state(runs_dir),
    }


def _harness_corrected(directory: Path) -> list[dict]:
    """Ledger rows with the registered harness-failure rule applied, in memory only."""
    calls = {c.get("tag"): c for c in ds1._read_jsonl(Path(directory) / "calls.jsonl")}
    out = []
    for row in ds1._read_jsonl(Path(directory) / "ledger.jsonl"):
        call = calls.get(ds1.call_tag(row["task_id"], row["arm"]))
        if call is not None and is_harness_failure(call):
            row = dict(row, passed=None, detail=HARNESS_DETAIL)
        out.append(row)
    return out


def combined_design_report(out_dir: Path, runs_dir: Path) -> dict:
    hist_dir = Path(runs_dir) / HISTORICAL_DIR_NAME
    s1_dir = Path(runs_dir) / FIRST_SUPPLEMENT_DIR_NAME
    hist_rows = [r for r in ds1._read_jsonl(hist_dir / "ledger.jsonl") if r["category"] == "design"]
    hist_manifest = [t for t in ds1._manifest(hist_dir) if t["category"] == "design"]
    s1_rows, s1_manifest = _harness_corrected(s1_dir), ds1._manifest(s1_dir)
    s2_rows, s2_manifest = _harness_corrected(out_dir), ds1._manifest(out_dir)
    ids = [t["id"] for t in hist_manifest + s1_manifest + s2_manifest]
    if len(ids) != len(set(ids)):
        raise SystemExit("task ids overlap between the registrations")
    pooled = ds1._summary(hist_rows + s1_rows + s2_rows, hist_manifest + s1_manifest + s2_manifest)
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
    drop = ds1.UNDEMANDED_RULE_PAIRS
    keep = [t for t in hist_manifest if t["id"] not in drop]
    return {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "sources": {"historical": str(hist_dir), "first_supplement": str(s1_dir),
                    "second_supplement": str(out_dir)},
        "earlier_runs_unchanged": pinned_state(runs_dir),
        "arms": {ROUTER_ARM: "real normal routing policy", CONTROL_ARM: CONTROL_MODEL},
        "pooled": pooled,
        "historical_only": ds1._summary(hist_rows, hist_manifest),
        "first_supplement_only": ds1._summary(s1_rows, s1_manifest),
        "second_supplement_only": ds1._summary(s2_rows, s2_manifest),
        "sensitivity_without_undemanded_rule_pairs": {
            "dropped": drop,
            **ds1._summary([r for r in hist_rows if r["task_id"] not in drop] + s1_rows + s2_rows,
                           keep + s1_manifest + s2_manifest)},
        "floor": TARGET_VALID_PAIRS,
        "floor_met": floor_met,
        "answer": answer,
        "difference_rule": ANALYSIS_PLAN["difference_rule"],
        "claims_not_made": ANALYSIS_PLAN["claims_not_made"],
    }


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

    unchanged = data["earlier_runs_unchanged"]
    lines = ["# Combined design result — extended run + two design-floor supplements", "",
             "Earlier runs byte-identical to their pins: "
             + ", ".join(f"{k} **{v}**" for k, v in unchanged.items()) + ". "
             "Each supplement adds separately pre-registered tasks; neither replaces anything. "
             "A credential rejection before any model ran counts as a harness failure "
             "(incomplete/other), never as a quality outcome.", "",
             "| source | attempted | valid pairs | truncated | incomplete/other/harness | router "
             "pass (Wilson 95 %) | control-metered pass (Wilson 95 %) | discordant r/c | sign p | "
             "paired diff (conservative 95 %) |",
             "|---|---:|---:|---:|---:|---|---|---|---|---|",
             row("historical only", data["historical_only"]),
             row("first supplement only", data["first_supplement_only"]),
             row("second supplement only", data["second_supplement_only"]),
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
    parser.add_argument("command",
                        choices=["preflight", "preregister", "verify", "run", "readback", "report"])
    parser.add_argument("--dir", type=Path)
    parser.add_argument("--runs", type=Path, required=True, help="the runs/ directory")
    parser.add_argument("--config", type=Path)
    args = parser.parse_args(argv)
    if args.command in ("preflight", "preregister", "run") and not args.config:
        raise SystemExit("--config is required")
    if args.command != "preflight" and not args.dir:
        raise SystemExit("--dir is required")
    if args.command == "preflight":
        from auto_router.config import load_config
        result = credential_preflight(load_config(args.config))
        print(json.dumps(result, indent=1))
        print("credential preflight " + ("PASS" if result["ok"] else "FAIL"))
        return 0 if result["ok"] else 1
    if args.command == "preregister":
        record = preregister(args.dir, args.config, args.runs)
        print(json.dumps({k: v for k, v in record.items()
                          if k not in ("analysis_plan", "policy_identity")}, indent=1))
        return 0
    if args.command == "verify":
        checks = verify(args.dir, args.config, args.runs)
        for name, ok, detail in checks:
            print(f"  {'PASS' if ok else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
        bad = sum(not ok for _, ok, _ in checks)
        print(f"{'PASS' if not bad else 'FAIL'} {len(checks) - bad}/{len(checks)}")
        return 1 if bad else 0
    if args.command == "run":
        return run(args.dir, args.config, args.runs)
    if args.command == "readback":
        print(json.dumps(readback(args.dir, args.runs), indent=1))
        return 0
    data = combined_design_report(args.dir, args.runs)
    (args.dir / "combined-design-report.json").write_text(json.dumps(data, indent=1))
    text = format_markdown(data)
    (args.dir / "combined-design-report.md").write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
