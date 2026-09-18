"""Runner for the pre-registered held-out evaluation.

Each task is sent to two arms with an identical prompt:

``router``   the configured policy makes the route choice per task.
``control``  the *normal* single-model behaviour: one capable model for
             everything, chosen from the catalog by capability, with no
             per-task switching. This is the ordinary default, not an
             intentionally expensive straw man, and its name is recorded in
             every row of the ledger and in the report.

Spend control
-------------
Every call goes through ``experiments.llm.Client``, which appends to a JSONL
ledger and refuses to start a call once the cap is reached. The cap is a hard
argument, has no default, and is enforced against the *higher* of billed and
list-price cost. A run that hits the cap stops and the report says which
categories were completed.

Isolation
---------
Generated coding answers are executed only through ``experiments.sandbox``
(Bubblewrap: no network, no home, private tmpfs). The runner refuses to grade
the coding category at all if ``sandbox.preflight()`` does not confirm real
isolation; those tasks are then recorded as excluded with the exact reason,
never silently run on the host.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from auto_router import jev                                    # noqa: E402
from auto_router.catalog import ModelInfo                      # noqa: E402
from auto_router.config import load_config                     # noqa: E402
from auto_router.decision import ObservedOutcome               # noqa: E402
from auto_router.policies import candidates                    # noqa: E402
from auto_router.router import Router                          # noqa: E402
from experiments import graders, sandbox                       # noqa: E402
from experiments.heldout import (                              # noqa: E402
    TaskOutcome,
    load_preregistration,
    load_tasks,
    report,
    format_report,
)
from experiments.llm import BudgetExceeded, Client             # noqa: E402


#: Output budget per category. A reasoning model spends most of its output
#: budget thinking before it answers, so a single flat cap either truncates a
#: long artifact or wastes money on a one-line answer. These are harness
#: parameters, not part of the pre-registered task definitions: they change how
#: much room an answer gets, never what is asked or how it is graded.
#:
#: Raised on 2026-09-18 after the first design call returned exactly 4000
#: output tokens - a hard truncation that failed every structural rule
#: including "is markup, not prose". That one row was discarded as a harness
#: artifact before any design result was read. The math and research rows
#: already collected peaked at 706 output tokens and were unaffected.
OUTPUT_BUDGET = {
    "design": 12000,
    "coding": 8000,
    "math": 6000,
    "summarisation": 4000,
    "research": 3000,
    "cache_repeat": 3000,
}
DEFAULT_OUTPUT_BUDGET = 4000


def output_budget(task: dict) -> int:
    return OUTPUT_BUDGET.get(task["category"], DEFAULT_OUTPUT_BUDGET)


#: How much of a grader's explanation is kept. Unlike the router's own decision
#: ledger, which carries no answer text at all, this experiment ledger keeps a
#: short excerpt so a failure can be diagnosed without re-running the model. It
#: is put through the same credential scrubber first and capped hard, and that
#: difference is deliberate and stated rather than accidental.
DETAIL_CHARS = 200


def _detail(text: str) -> str:
    return jev.scrub(text or "", DETAIL_CHARS)


def _error_label(error: str | None) -> str:
    """Only the shape of a provider error, never its message.

    Provider error bodies echo prompt fragments and sometimes the rejected key.
    """
    if not error:
        return "unknown"
    head = error.split(":")[0].strip()
    return head[:40] if head else "unknown"


def control_model(router: Router, name: str | None = None) -> ModelInfo:
    """The ordinary default: one capable route used for everything.

    Chosen by the same capability data the router uses, so the control is what
    a sensible person would configure if they were not routing at all - not the
    most expensive model available. Subscription routes are excluded because a
    plan is not what a fresh install would reach for.

    Note that this can legitimately land on a *free* route. When it does, the
    run can compare pass rates but there is no cash difference to measure, and
    the report has to say so rather than quietly implying a saving.
    """
    catalog = router.config.catalog
    if name:
        if name not in catalog:
            raise SystemExit(f"--control {name!r} is not in the catalog")
        return catalog[name]
    pool = [m for m in catalog.all() if not m.subscription] or catalog.all()
    return max(pool, key=lambda m: m.cap("general"))


def run_live(args) -> int:
    out_dir: Path = args.dir
    prereg = load_preregistration(out_dir)
    tasks = load_tasks(out_dir)
    wanted = {c.strip() for c in args.categories.split(",") if c.strip()}
    tasks = [t for t in tasks if t["category"] in wanted]
    if args.limit:
        tasks = tasks[: args.limit]
    if not args.config:
        raise SystemExit("--config is required for a live run")
    if args.budget <= 0:
        raise SystemExit("--budget must be a positive hard cap in USD")

    isolation = sandbox.preflight()
    if not isolation["available"]:
        print(f"! sandbox unavailable ({isolation.get('reason')}); "
              "coding answers will NOT be executed and are excluded", file=sys.stderr)

    config = load_config(args.config)
    router = Router(config)
    control = control_model(router, getattr(args, "control", None))
    client = Client(config, out_dir / "calls.jsonl", budget_usd=args.budget)
    client.router_catalog = {m.name: m for m in config.catalog.all()}

    ledger = out_dir / "ledger.jsonl"
    done = set()
    if ledger.exists():
        for line in ledger.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                done.add((row["task_id"], row["arm"]))

    meta = {"started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "preregistration_sha256": prereg["task_file_sha256"],
            "control_model": control.name,
            "control_label": args.control_label,
            "arms": args.arms,
            "control_is_metered": not control.prices.is_free,
            "control_prices_per_mtok": {"input": control.prices.input,
                                        "output": control.prices.output},
            "control_selected_by": ("named on the command line" if getattr(args, "control", None)
                                    else "highest general capability in the catalog"),
            "policy": router.policy.name,
            "routes": {m.name: {"free": m.prices.is_free,
                                "capability": m.capability,
                                "evidence_stale": m.evidence_stale}
                       for m in config.catalog.all()},
            "budget_usd": args.budget, "output_budget_per_category": OUTPUT_BUDGET,
            "sandbox": isolation}
    (out_dir / "run-meta.json").write_text(json.dumps(meta, indent=1))
    print(f"control arm = {control.name} ({'metered' if not control.prices.is_free else 'FREE'}); "
          f"router policy = {router.policy.name}; cap ${args.budget:.2f}; "
          f"sandbox available = {isolation['available']}")
    if control.prices.is_free:
        print("! the control route is free, so this run can compare pass rates but "
              "CANNOT measure a cash difference", file=sys.stderr)

    stopped = None
    label = args.control_label
    wanted_arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    for task in tasks:
        for arm in wanted_arms:
            recorded = label if arm == "control" else arm
            if (task["id"], recorded) in done:
                continue
            try:
                outcome = _one(task, arm, router, control, client, isolation)
                outcome.arm = recorded
            except BudgetExceeded as exc:
                stopped = str(exc)
                break
            with ledger.open("a") as fh:
                fh.write(json.dumps(asdict(outcome)) + "\n")
            mark = {True: "pass", False: "FAIL", None: "skip"}[outcome.passed]
            print(f"  {task['id']:<34} {recorded:<16} {outcome.model:<18} {mark:<5} "
                  f"{outcome.detail[:70]}")
        if stopped:
            break

    if stopped:
        print(f"\nstopped early: {stopped}", file=sys.stderr)
    print(f"\nspend on this run: ${client.spent():.4f} of ${args.budget:.2f}")
    print()
    print(format_report(report(out_dir)))
    return 0


def _one(task: dict, arm: str, router: Router, control: ModelInfo, client: Client,
         isolation: dict) -> TaskOutcome:
    messages = [{"role": "system", "content": task["system"]},
                {"role": "user", "content": task["prompt"]}]
    budget = output_budget(task)
    explanation = None
    if arm == "router":
        result = router.route(messages[1:], system=task["system"], max_tokens=budget)
        model, explanation = result.model, result.explanation
    else:
        result, model = None, control

    started = time.perf_counter()
    call = client.chat(model, messages, max_tokens=budget, tag=f"heldout:{arm}:{task['id']}")
    latency_ms = (time.perf_counter() - started) * 1000

    if result is not None:
        router.commit(result, call.prompt_tokens or None, call.output_tokens)
        router.observe(result, ObservedOutcome(
            model=model.name, status="ok" if call.ok else "upstream_error",
            latency_ms=latency_ms,
            uncached_input_tokens=max(0, call.prompt_tokens - call.cached_tokens),
            cached_read_tokens=call.cached_tokens, cache_write_tokens=call.cache_write_tokens,
            output_tokens=call.output_tokens,
            cost_usd=None if model.prices.is_free else call.cost_usd,
            cost_basis=("route configured as free: no cash cost to measure"
                        if model.prices.is_free
                        else "provider-reported tokens x configured list prices"),
            error=_error_label(call.error) if call.error else None))

    if not call.ok:
        # A refused or failed upstream call is a property of the route, not of
        # the harness, so it counts as a failure. Excluding it would let a flaky
        # provider drop its own failures out of its denominator.
        passed, detail = False, f"the route failed the call: {_error_label(call.error)}"
    elif call.output_tokens >= budget:
        # A truncated answer says nothing about the model's capability, so it is
        # excluded rather than scored as a failure.
        passed, detail = None, (f"TRUNCATED: hit the {budget}-token output budget; "
                                "excluded, not counted as a failure")
    elif task["grader"] == "coding" and not isolation["available"]:
        passed, detail = None, f"SANDBOX UNAVAILABLE: {isolation.get('reason')}"
    else:
        passed, detail = graders.grade(task, call.content)

    return TaskOutcome(
        task_id=task["id"], category=task["category"], arm=arm, passed=passed,
        grader=task["grader"], grader_kind=graders.GRADER_KIND[task["grader"]],
        detail=_detail(detail),
        model=model.name, latency_ms=round(latency_ms, 1),
        prompt_tokens=call.prompt_tokens, cached_tokens=call.cached_tokens,
        output_tokens=call.output_tokens,
        observed_cost_usd=None if model.prices.is_free else round(call.cost_usd, 8),
        cost_basis=call.cost_basis,
        estimated_cost_usd=(explanation.estimated.cost_usd if explanation else None),
        estimated_p_success=(explanation.estimated.p_success if explanation else None),
        cache_status=(explanation.cache.status if explanation else "n/a"),
        evidence_confidence=(explanation.selection.evidence_confidence if explanation else None),
        safe_fallback=(explanation.selection.safe_fallback if explanation else None),
        error=_error_label(call.error) if call.error else None, label="live")
