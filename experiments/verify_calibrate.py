"""Calibrate the verify-and-escalate threshold, and measure what it buys.

The 17 September evaluation (EXPERIMENTS.md section 4) established that Jev can
tell a wrong cheap answer from a right one for self-contained coding and maths
requests. It did not establish *what to do about it*. This script closes that
gap on the same 78-task set, in three stages that are deliberately separate
files on disk so that none of them has to be re-run to redo the next:

  answers   cheap models answer the single-shot tasks; each answer is graded
            against ground truth and kept, so the judge and the threshold sweep
            can be redone for free.
  judge     Jev answers the typed adequacy question (``jev.verify_questions``)
            about each stored answer: P(adequate) plus the failure type.
  escalate  at a candidate threshold, every flagged answer is re-run on the
            escalation target and graded again - this is the only stage that
            spends real money, and it spends it only on flagged rows.
  report    sweeps thresholds over the stored judgements, picks one per
            category, and states what the escalation stage actually bought:
            extra solved tasks, extra dollars, extra seconds, false-escalation
            rate.

A false escalation is an *adequate* answer that was flagged. It costs a second
call and some seconds; it does not cost correctness. A missed failure costs the
user a wrong answer. The threshold is where those two meet, and because the two
are not equally bad the sweep reports both rather than collapsing them into one
score.

    python experiments/verify_calibrate.py answers --config live.yaml \\
        --models qwen3.8-27b,dsv4-flash,kimi-k3 --out runs/verify/answers.jsonl
    python experiments/verify_calibrate.py judge --answers runs/verify/answers.jsonl \\
        --out runs/verify/judged.jsonl
    python experiments/verify_calibrate.py escalate --judged runs/verify/judged.jsonl \\
        --config live.yaml --target gpt-5.6-luna --threshold 0.3 --out runs/verify/escalated.jsonl
    python experiments/verify_calibrate.py report --judged runs/verify/judged.jsonl \\
        --escalated runs/verify/escalated.jsonl --out runs/verify/report.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import tasks as T  # noqa: E402
from llm import BudgetExceeded, Client  # noqa: E402

from auto_router import jev, verify  # noqa: E402
from auto_router.config import load_config  # noqa: E402
from auto_router.economics import SuccessModel  # noqa: E402
from auto_router.policies import Context, Conversation, ExpectedCostPolicy, TurnRequest  # noqa: E402
from auto_router.router import success_model_from_config  # noqa: E402

#: Graders that score one answer to one prompt. A tool-loop or agent task has
#: no single "response" to hand a judge, and a long-context task hands it a
#: document it cannot see - both are measured here as the documented negative
#: control, not as candidates for the gate.
SINGLE_SHOT = ("humaneval", "stdio", "math")

#: How much of an answer the judge is shown. Same rule the runtime uses
#: (``jev.RESPONSE_CHARS``): head and tail, because a truncated answer's tail is
#: exactly the evidence that it was truncated.
ANSWER_CHARS = 7500


def load_tasks(path: str) -> list[dict]:
    rows = [json.loads(line) for line in open(path)]
    return [t for t in rows if t["grader"] in SINGLE_SHOT]


def grade(task: dict, answer: str) -> tuple[bool, str]:
    if task["grader"] == "humaneval":
        return T.grade_humaneval(task, answer)
    if task["grader"] == "stdio":
        return T.grade_stdio(task, T.extract_code(answer))
    return T.grade_math(task, answer)


def shrink(text: str) -> str:
    if len(text) <= ANSWER_CHARS:
        return text
    return text[: ANSWER_CHARS // 3] + "\n...\n" + text[-(ANSWER_CHARS * 2 // 3):]


def read_jsonl(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_jsonl(path: str | Path, rows: list[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("".join(json.dumps(r) + "\n" for r in rows))


_append_lock = __import__("threading").Lock()


def append_jsonl(path: str | Path, row: dict) -> None:
    """One finished row, on disk immediately.

    A run of a few hundred model calls takes long enough that it will be
    interrupted at least once; nothing that has already been paid for should
    have to be paid for twice.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with _append_lock, Path(path).open("a") as fh:
        fh.write(json.dumps(row) + "\n")


# ---------------------------------------------------------------------------
# 1. answers
# ---------------------------------------------------------------------------
def answers(args) -> None:
    cfg = load_config(args.config)
    client = Client(cfg, args.ledger, args.budget)
    tasks = load_tasks(args.tasks)
    done = {(r["model"], r["task"]) for r in read_jsonl(args.out)} if Path(args.out).exists() else set()
    pairs = [(m, t) for m in args.models.split(",") for t in tasks if (m, t["id"]) not in done]
    print(f"{len(pairs)} answers to generate ({len(done)} already on disk)")

    def one(pair):
        model_name, task = pair
        model = cfg.catalog[model_name]
        try:
            r = client.chat(model, [{"role": "system", "content": task["system"]},
                                    {"role": "user", "content": task["prompt"]}],
                            max_tokens=args.max_tokens, tag="verify-answer-" + task["id"])
        except BudgetExceeded as exc:
            return {"error": str(exc)}
        if not r.ok:
            return append_row({"model": model_name, "task": task["id"], "ok": False, "error": r.error})
        ok, why = grade(task, r.content)
        return append_row({"model": model_name, "task": task["id"], "category": task["category"],
                "level": task["difficulty"], "grader": task["grader"], "ok": True,
                "correct": ok, "why": why[:200], "answer": shrink(r.content),
                "truncated": r.finish_reason == "length", "latency_s": round(r.latency_s, 2),
                "cost_usd": r.cost_usd, "list_cost_usd": r.list_cost_usd,
                "cost_basis": r.cost_basis, "output_tokens": r.output_tokens})

    def append_row(row: dict) -> dict:
        append_jsonl(args.out, row)
        return row

    with ThreadPoolExecutor(args.workers) as pool:
        list(pool.map(one, pairs))
    good = [r for r in read_jsonl(args.out) if r.get("ok")]
    print(f"{len(good)} answers, {sum(not r['correct'] for r in good)} wrong, "
          f"${client.spent():.3f} spent")


# ---------------------------------------------------------------------------
# 2. judge
# ---------------------------------------------------------------------------
def judge(args) -> None:
    tasks = {t["id"]: t for t in load_tasks(args.tasks)}
    rows = [r for r in read_jsonl(args.answers) if r.get("ok")]
    print(f"judging {len(rows)} answers")

    def one(row):
        task = tasks[row["task"]]
        request = task["prompt"]
        if len(request) > jev.REQUEST_CHARS:
            request = request[: jev.REQUEST_CHARS]
        started = time.time()
        # The runtime asks with the category the classifier produced; here the
        # task's own label stands in, which is the same string for every task
        # the 17 Sep classification got right (78/78 plausible).
        j = jev.judge(request, row["answer"], category=row["category"])
        return {**{k: v for k, v in row.items() if k != "answer"},
                "p_adequate": j.p_adequate, "failure": j.failure,
                "failure_probs": j.failure_probs, "judge_failed": j.failed,
                "judge_latency_s": round(j.latency_s or (time.time() - started), 2),
                "judge_model": j.model, "judge_input_tokens": j.input_tokens,
                "judge_output_tokens": j.output_tokens}

    with ThreadPoolExecutor(args.workers) as pool:
        judged = list(pool.map(one, rows))
    write_jsonl(args.out, judged)
    failed = sum(r["judge_failed"] for r in judged)
    print(f"{len(judged)} judged, {failed} judge failures, "
          f"median {statistics.median([r['judge_latency_s'] for r in judged]):.2f}s")


# ---------------------------------------------------------------------------
# 3. escalate
# ---------------------------------------------------------------------------
def thresholds_from(spec: str) -> dict[str, float]:
    """``0.3`` or ``coding=0.25,math=0.6`` - one number, or one per category."""
    if "=" not in spec:
        return {"default": float(spec)}
    out = {}
    for part in spec.split(","):
        name, _, value = part.partition("=")
        out[name.strip()] = float(value)
    return out


def escalation_target(cfg, row: dict, policy: ExpectedCostPolicy, ctx: Context):
    """Where the *runtime* would send this turn, not a model named on the command line.

    A fixed ``--target`` would measure a route the router might never pick. The
    catalog here has a category-dependent answer - for maths the strongest
    route is a free one - and measuring the wrong one would make the whole
    escalation look more expensive than it is.
    """
    failed = cfg.catalog[row["model"]]
    req = TurnRequest(category=row["category"], difficulty=0.6, prompt_tokens=1200,
                      output_tokens=1200, now=0.0, request_chars=1200)
    choice = verify.escalation_choice(policy, Conversation(current=failed.name), req, ctx,
                                      failed.name, {failed.name}, verify.VerifyPolicy())
    if choice is not None:
        return cfg.catalog[choice[0]]
    retry = policy.on_failure(Conversation(current=failed.name), req, ctx, failed.name, {failed.name})
    return cfg.catalog.get(retry.model) if retry else None


def escalate(args) -> None:
    """Re-run every flagged answer where the router would send it, and grade it again.

    Both arms of the carry decision are measured when ``--carry both`` is
    given: the stronger model either sees the failed attempt or does not. The
    runtime default is chosen from this measurement rather than from taste.
    """
    cfg = load_config(args.config)
    client = Client(cfg, args.ledger, args.budget)
    tasks = {t["id"]: t for t in load_tasks(args.tasks)}
    success = success_model_from_config(cfg.policy or {})
    ctx = Context(cfg.catalog, success, {}, {})
    policy = ExpectedCostPolicy()
    thresholds = thresholds_from(args.threshold)
    judged = read_jsonl(args.judged)
    answers_by_key = {(r["model"], r["task"]): r for r in read_jsonl(args.answers)}
    flagged = []
    for row in judged:
        if row["judge_failed"] or row["category"] in verify.SKIP_CATEGORIES:
            continue
        limit = thresholds.get(row["category"], thresholds.get("default", 0.0))
        if row["p_adequate"] < limit:
            flagged.append(row)
    modes = ["clean", "carry"] if args.carry == "both" else [args.carry]
    done = {(r["model"], r["task"], r["mode"]) for r in read_jsonl(args.out)} \
        if Path(args.out).exists() else set()
    work = [(r, m) for r in flagged for m in modes if (r["model"], r["task"], m) not in done]
    print(f"{len(flagged)} flagged at {thresholds}; {len(work)} escalation calls to make")

    def one(item):
        row, mode = item
        task = tasks[row["task"]]
        target = cfg.catalog[args.target] if args.target else escalation_target(cfg, row, policy, ctx)
        if target is None:
            return {"model": row["model"], "task": row["task"], "mode": mode, "ok": False,
                    "error": "no stronger route available", "target": None}
        messages = [{"role": "system", "content": task["system"]},
                    {"role": "user", "content": task["prompt"]}]
        if mode == "carry":
            previous = answers_by_key[(row["model"], row["task"])]["answer"]
            messages = verify.retry_messages(
                messages, previous,
                verify.Verdict(verified=True, failure=row["failure"]),
                verify.VerifyPolicy(carry_failed_attempt=True))
        try:
            r = client.chat(target, messages, max_tokens=args.max_tokens,
                            tag=f"verify-escalate-{mode}-" + task["id"])
        except BudgetExceeded as exc:
            print(f"stopping: {exc}")
            return None
        if not r.ok:
            return append_jsonl_row(args.out, {
                "model": row["model"], "task": row["task"], "mode": mode, "ok": False,
                "error": r.error, "target": target.name})
        ok, why = grade(task, r.content)
        return append_jsonl_row(args.out, {
            "model": row["model"], "task": row["task"], "category": row["category"],
            "level": row["level"], "mode": mode, "ok": True, "target": target.name,
            "was_correct": row["correct"], "now_correct": ok, "why": why[:200],
            "p_adequate": row["p_adequate"], "failure": row["failure"],
            "latency_s": round(r.latency_s, 2), "cost_usd": r.cost_usd,
            "list_cost_usd": r.list_cost_usd, "cost_basis": r.cost_basis})

    with ThreadPoolExecutor(args.workers) as pool:
        list(pool.map(one, work))
    rows = read_jsonl(args.out) if Path(args.out).exists() else []
    print(f"{len(rows)} escalations on file, ${client.spent():.3f} spent in total")


def append_jsonl_row(path, row: dict) -> dict:
    append_jsonl(path, row)
    return row


# ---------------------------------------------------------------------------
# 4. report
# ---------------------------------------------------------------------------
def sweep(rows: list[dict], thresholds: list[float]) -> list[dict]:
    """Catch rate and false-flag rate at each threshold, for one row group."""
    wrong = [r for r in rows if not r["correct"]]
    right = [r for r in rows if r["correct"]]
    out = []
    for t in thresholds:
        caught = sum(1 for r in wrong if r["p_adequate"] < t)
        false_flags = sum(1 for r in right if r["p_adequate"] < t)
        out.append({
            "threshold": round(t, 3),
            "wrong_answers": len(wrong),
            "adequate_answers": len(right),
            "caught": caught,
            "catch_rate": round(caught / len(wrong), 3) if wrong else None,
            "false_flags": false_flags,
            "false_flag_rate": round(false_flags / len(right), 3) if right else None,
            "flagged_share": round((caught + false_flags) / len(rows), 3) if rows else None,
        })
    return out


def auc(rows: list[dict]) -> float | None:
    """P(a right answer scores above a wrong one). Ties count half."""
    wrong = [r["p_adequate"] for r in rows if not r["correct"]]
    right = [r["p_adequate"] for r in rows if r["correct"]]
    if not wrong or not right:
        return None
    wins = sum((a > b) + 0.5 * (a == b) for a in right for b in wrong)
    return round(wins / (len(right) * len(wrong)), 3)


def choose_threshold(sweeps: list[dict], max_false_flag: float) -> float:
    """The highest threshold whose false-flag rate stays inside the budget.

    Raising the threshold flags more answers: it catches more failures *and*
    escalates more answers that were fine. The operator's lever is how many
    unnecessary second calls they will pay for, so that - not a score - is what
    the choice is made on.
    """
    ok = [s for s in sweeps if (s["false_flag_rate"] or 0.0) <= max_false_flag]
    return max((s["threshold"] for s in ok), default=0.0)


def report(args) -> None:
    judged = [r for r in read_jsonl(args.judged) if not r["judge_failed"]]
    escalated = read_jsonl(args.escalated) if args.escalated and Path(args.escalated).exists() else []
    thresholds = [round(x / 100, 2) for x in range(5, 96, 5)]
    verifiable = [r for r in judged if r["category"] not in verify.SKIP_CATEGORIES]

    by_category: dict[str, dict] = {}
    for cat in sorted({r["category"] for r in judged}):
        rows = [r for r in judged if r["category"] == cat]
        sweeps = sweep(rows, thresholds)
        by_category[cat] = {
            "answers": len(rows),
            "wrong": sum(not r["correct"] for r in rows),
            "auc": auc(rows),
            "sweep": sweeps,
            "chosen_threshold": choose_threshold(sweeps, args.max_false_flag),
            "verified_tier": cat not in verify.SKIP_CATEGORIES,
        }
    overall = sweep(verifiable, thresholds)
    chosen = choose_threshold(overall, args.max_false_flag)

    # what the escalation stage bought, per carry mode
    escalation: dict[str, dict] = {}
    for mode in sorted({r["mode"] for r in escalated if r.get("ok")}):
        rows = [r for r in escalated if r.get("ok") and r["mode"] == mode]
        fixed = [r for r in rows if not r["was_correct"] and r["now_correct"]]
        broken = [r for r in rows if r["was_correct"] and not r["now_correct"]]
        unnecessary = [r for r in rows if r["was_correct"]]
        escalation[mode] = {
            "escalations": len(rows),
            "target": rows[0]["target"] if rows else None,
            "was_wrong": sum(not r["was_correct"] for r in rows),
            "fixed": len(fixed),
            "fix_rate_of_flagged_wrong": round(len(fixed) / max(1, sum(not r["was_correct"] for r in rows)), 3),
            "unnecessary": len(unnecessary),
            "broken_by_escalating": len(broken),
            "extra_cost_usd": round(sum(max(r["cost_usd"], r["list_cost_usd"]) for r in rows), 4),
            "extra_cost_usd_per_escalation": round(
                sum(max(r["cost_usd"], r["list_cost_usd"]) for r in rows) / max(1, len(rows)), 5),
            "extra_latency_s_median": round(statistics.median([r["latency_s"] for r in rows]), 1) if rows else None,
        }

    judge_latency = [r["judge_latency_s"] for r in read_jsonl(args.judged) if not r["judge_failed"]]
    out = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "answers_judged": len(judged),
        "models": sorted({r["model"] for r in judged}),
        "judge_latency_s": {
            "median": round(statistics.median(judge_latency), 2) if judge_latency else None,
            "p90": round(sorted(judge_latency)[int(0.9 * (len(judge_latency) - 1))], 2) if judge_latency else None,
        },
        "judge_failures": sum(1 for r in read_jsonl(args.judged) if r["judge_failed"]),
        "max_false_flag_rate_allowed": args.max_false_flag,
        "overall_sweep_self_contained": overall,
        "chosen_threshold_overall": chosen,
        "per_category": by_category,
        "escalation": escalation,
        "note": ("Rows of category "
                 + ", ".join(verify.SKIP_CATEGORIES)
                 + " are the negative control: the judge is not shown the document the answer "
                   "depends on, so its flags there measure blindness, not quality."),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in
                      ("answers_judged", "chosen_threshold_overall", "judge_latency_s")}, indent=2))
    for cat, data in by_category.items():
        best = next(s for s in data["sweep"] if s["threshold"] == data["chosen_threshold"]) \
            if data["chosen_threshold"] else None
        print(f"{cat:14s} n={data['answers']:3d} wrong={data['wrong']:3d} auc={data['auc']} "
              f"t={data['chosen_threshold']} "
              + (f"catch={best['catch_rate']} false={best['false_flag_rate']}" if best else ""))
    for mode, data in escalation.items():
        print(f"escalate[{mode}]: {data['fixed']}/{data['was_wrong']} wrong answers fixed, "
              f"{data['unnecessary']} unnecessary, ${data['extra_cost_usd']:.4f}, "
              f"+{data['extra_latency_s_median']}s median")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    tasks_default = "experiments/tasks/tasks.jsonl"

    a = sub.add_parser("answers")
    a.add_argument("--config", required=True)
    a.add_argument("--models", required=True)
    a.add_argument("--tasks", default=tasks_default)
    a.add_argument("--out", required=True)
    a.add_argument("--ledger", default="runs/verify/ledger.jsonl")
    a.add_argument("--budget", type=float, default=5.0)
    a.add_argument("--workers", type=int, default=8)
    a.add_argument("--max-tokens", type=int, default=12000)

    j = sub.add_parser("judge")
    j.add_argument("--answers", required=True)
    j.add_argument("--tasks", default=tasks_default)
    j.add_argument("--out", required=True)
    j.add_argument("--workers", type=int, default=8)

    e = sub.add_parser("escalate")
    e.add_argument("--judged", required=True)
    e.add_argument("--answers", required=True)
    e.add_argument("--config", required=True)
    e.add_argument("--tasks", default=tasks_default)
    e.add_argument("--target", default="", help="override; by default the router picks")
    e.add_argument("--threshold", default="0.3",
                   help="one number, or per category: coding=0.25,math=0.6")
    e.add_argument("--carry", choices=["clean", "carry", "both"], default="both")
    e.add_argument("--out", required=True)
    e.add_argument("--ledger", default="runs/verify/ledger.jsonl")
    e.add_argument("--budget", type=float, default=5.0)
    e.add_argument("--workers", type=int, default=6)
    e.add_argument("--max-tokens", type=int, default=12000)

    r = sub.add_parser("report")
    r.add_argument("--judged", required=True)
    r.add_argument("--escalated", default="")
    r.add_argument("--out", required=True)
    r.add_argument("--max-false-flag", type=float, default=0.05)

    args = ap.parse_args()
    {"answers": answers, "judge": judge, "escalate": escalate, "report": report}[args.cmd](args)


if __name__ == "__main__":
    main()
