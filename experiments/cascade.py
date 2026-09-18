"""Chains of routed calls, with and without the answer judge.

Scott Shapiro's objection to a router benchmarked on one hop: "I'd want to see
how that holds when you chain three or four routed calls where a bad early
model pick cascades downstream. Single-hop latency flatters routers."

He is right, and the mechanism is worth stating precisely. In a chain, step
*i+1* works from step *i*'s output. A wrong early answer that nobody notices is
not one failure, it is the whole chain: every later step is diligently building
on something false. That is why a judge that only catches 80 % of failures
still changes the picture out of proportion to its catch rate - it removes
failures *early*, where they would otherwise be multiplied.

This is a simulation, and it is labelled as one everywhere it is reported. What
is measured and what is assumed:

  measured   per-model success by category and difficulty bucket (the 78-task
             matrix, ``runs/success.json``); the judge's catch rate and false
             alarm rate at the chosen threshold, and the rate at which the
             escalation target actually fixes a flagged answer
             (``verify_calibrate.py report``); per-model answer latency and
             judge latency from the same runs; list prices from the config.
  assumed    that a wrong step is fatal to the chain unless it is caught (no
             self-repair downstream), that steps are independent given the
             model and difficulty, and that the difficulty of a chain is the
             same at every step.

The first assumption is the load-bearing one, and it is the pessimistic reading
of Scott's point rather than a flattering one: it is what makes the *no-judge*
arm look bad. A reader who believes later steps often repair earlier mistakes
should read the gap as an upper bound.

    python experiments/cascade.py --config live.yaml \\
        --calibration runs/verify/report.json --latency runs/verify/latency.json \\
        --out runs/verify/cascade.json
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from auto_router import verify  # noqa: E402
from auto_router.config import load_config  # noqa: E402
from auto_router.economics import SuccessModel  # noqa: E402
from auto_router.policies import (Context, Conversation, ExpectedCostPolicy,  # noqa: E402
                                  TurnRequest, turn_call_cost)
from auto_router.router import success_model_from_config  # noqa: E402


def load_success(config, path: str | None) -> SuccessModel:
    success = success_model_from_config(config.policy or {})
    if path:
        rows = json.loads(Path(path).read_text())
        success.measured = {(m, c, b): float(p) for m, c, b, p in rows}
    return success


def arm_policy(verify_policy: verify.VerifyPolicy | None) -> ExpectedCostPolicy:
    return ExpectedCostPolicy(verify=verify_policy, judge_available=verify_policy is not None)


def run_chain(rng: random.Random, *, steps: int, category: str, difficulty: float,
              ctx: Context, policy: ExpectedCostPolicy, vp: verify.VerifyPolicy | None,
              latency: dict[str, float], calibration: dict, prompt_tokens: int,
              output_tokens: int, growth: int, force: str | None = None) -> dict:
    """One chain. Returns cost, seconds, whether every step was right, and why not."""
    conv = Conversation()
    cost = seconds = 0.0
    escalations = false_escalations = 0
    ok_chain = True
    failed_at: int | None = None
    now = 0.0
    for step in range(steps):
        req = TurnRequest(category=category, difficulty=difficulty,
                          prompt_tokens=prompt_tokens + step * growth,
                          output_tokens=output_tokens, now=now, follow_up=1.0,
                          remaining_turns=steps - step, request_chars=1200)
        name = force or policy.choose(conv, req, ctx).model
        model = ctx.catalog[name]
        cost += turn_call_cost(model, req, conv.warm_tokens(model, now), ctx)
        seconds += latency.get(name, 8.0)
        ok = rng.random() < ctx.success.p(model, category, difficulty)

        checked = vp is not None and policy.checks(model, req, ctx)
        if checked:
            cost += vp.judge_usd
            seconds += vp.judge_seconds
            flagged = (rng.random() < vp.catch(category)) if not ok \
                else (rng.random() < vp.false_flag(category))
            if flagged:
                escalations += 1
                false_escalations += ok
                retry = policy.on_failure(conv, req, ctx, name, {name})
                target = ctx.catalog.get(retry.model) if retry else None
                if target is not None:
                    cost += turn_call_cost(target, req, conv.warm_tokens(target, now), ctx)
                    seconds += latency.get(target.name, 12.0)
                    # The measured fix rate, not the curve: the calibration
                    # re-ran every flagged answer on this tier and graded it.
                    fix = calibration.get("fix_rate", ctx.success.p(target, category, difficulty))
                    ok = ok or rng.random() < fix
                    conv.floor = max(conv.floor, vp.difficulty_floor)
                    conv.floor_set_at = now
                    conv.record_call(target, req.prompt_tokens, output_tokens, now)
                    model = target
        conv.record_call(model, req.prompt_tokens, output_tokens, now)
        now += latency.get(model.name, 8.0)
        if not ok and ok_chain:
            ok_chain = False
            failed_at = step + 1
    return {"ok": ok_chain, "cost_usd": cost, "seconds": seconds, "escalations": escalations,
            "false_escalations": false_escalations, "failed_at": failed_at}


def summarise(rows: list[dict]) -> dict:
    n = len(rows)
    return {
        "chains": n,
        "chain_success_rate": round(sum(r["ok"] for r in rows) / n, 4),
        "mean_cost_usd": round(statistics.fmean(r["cost_usd"] for r in rows), 6),
        "mean_seconds": round(statistics.fmean(r["seconds"] for r in rows), 2),
        "escalations_per_chain": round(statistics.fmean(r["escalations"] for r in rows), 3),
        "false_escalations_per_chain": round(statistics.fmean(r["false_escalations"] for r in rows), 3),
        "median_first_failure_step": statistics.median(
            [r["failed_at"] for r in rows if r["failed_at"]]) if any(r["failed_at"] for r in rows) else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--success", default="")
    ap.add_argument("--calibration", required=True,
                    help="verify_calibrate.py report output; supplies thresholds and rates")
    ap.add_argument("--latency", default="", help='json: {"model": seconds_per_call}')
    ap.add_argument("--categories", default="coding,math")
    ap.add_argument("--difficulties", default="0.35,0.6,0.85")
    ap.add_argument("--steps", default="1,2,3,4")
    ap.add_argument("--trials", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=20260918)
    ap.add_argument("--frontier", default="", help="model name for the always-strongest control")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    config = load_config(args.config)
    success = load_success(config, args.success or None)
    ctx = Context(config.catalog, success, {}, {})
    report = json.loads(Path(args.calibration).read_text())
    latency = json.loads(Path(args.latency).read_text()) if args.latency else {}

    escalation = (report.get("escalation") or {}).get("clean") or {}
    per_category = report.get("per_category") or {}
    thresholds = {cat: data["chosen_threshold"] for cat, data in per_category.items()
                  if data.get("verified_tier")}
    catch, false_flag = {}, {}
    for cat, data in per_category.items():
        chosen = data.get("chosen_threshold")
        row = next((s for s in data.get("sweep", []) if s["threshold"] == chosen), None)
        if row and data.get("verified_tier"):
            catch[cat] = row["catch_rate"] or 0.0
            false_flag[cat] = row["false_flag_rate"] or 0.0
    vp = verify.with_measured(verify.VerifyPolicy(), thresholds=thresholds, catch_rate=catch,
                              false_flag_rate=false_flag)
    calibration = {"fix_rate": escalation.get("fix_rate_of_flagged_wrong")} \
        if escalation.get("fix_rate_of_flagged_wrong") is not None else {}

    results = []
    for category in args.categories.split(","):
        for difficulty in [float(x) for x in args.difficulties.split(",")]:
            for steps in [int(x) for x in args.steps.split(",")]:
                # Three arms, because "with the judge" changes two things at
                # once. ``checked_only`` routes exactly as the arm without a
                # judge does and only adds the check, which is the comparison
                # Scott's question is actually about; ``judge`` is the whole
                # system, where the check is also priced into the choice and
                # can move it to a different route.
                arms = {"no_judge": (arm_policy(None), None, None),
                        "checked_only": (arm_policy(None), vp, None),
                        "judge": (arm_policy(vp), vp, None)}
                if args.frontier:
                    arms["frontier_only"] = (arm_policy(None), None, args.frontier)
                for arm, (policy, arm_vp, force) in arms.items():
                    rng = random.Random(args.seed + steps * 13 + int(difficulty * 100))
                    rows = [run_chain(rng, steps=steps, category=category, difficulty=difficulty,
                                      ctx=ctx, policy=policy, vp=arm_vp, latency=latency,
                                      calibration=calibration, prompt_tokens=2500,
                                      output_tokens=700, growth=1200, force=force)
                            for _ in range(args.trials)]
                    results.append({"category": category, "difficulty": difficulty, "steps": steps,
                                    "arm": arm, **summarise(rows)})

    out = {"kind": "simulation",
           "note": ("Simulated chains. Per-model success, judge catch and false-alarm rates, the "
                    "escalation fix rate and the latencies are measured; the chain mechanics are "
                    "assumed. Not a measurement of end-to-end agent performance."),
           "thresholds": vp.thresholds, "catch_rate": vp.catch_rate,
           "false_flag_rate": vp.false_flag_rate, "fix_rate": calibration.get("fix_rate"),
           "trials_per_cell": args.trials, "results": results}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))

    print(f"{'cat':8s} {'d':>4s} {'steps':>5s} {'arm':12s} {'ok':>6s} {'$':>9s} {'s':>7s} {'esc':>5s}")
    for row in results:
        print(f"{row['category']:8s} {row['difficulty']:4.2f} {row['steps']:5d} {row['arm']:12s} "
              f"{row['chain_success_rate']:6.3f} {row['mean_cost_usd']:9.5f} "
              f"{row['mean_seconds']:7.1f} {row['escalations_per_chain']:5.2f}")


if __name__ == "__main__":
    main()
