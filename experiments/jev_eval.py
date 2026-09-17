"""Evaluate Jev as (1) the routing classifier and (2) an answer-adequacy judge.

(1) classify every task prompt; compare category and difficulty with the task labels,
    and check whether Jev's difficulty predicts which tasks a cheap model fails.
(2) generate answers with cheap models on the single-shot coding and math tasks, grade
    them, ask Jev "does this response fully and correctly address the request?", and
    report true/false positive rates at several thresholds.

    python experiments/jev_eval.py classify --out runs/jev_classify.jsonl
    python experiments/jev_eval.py judge --config my.local.yaml --models a,b --out runs/jev_judge.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import tasks as T  # noqa: E402
from llm import Client  # noqa: E402

from auto_router import jev  # noqa: E402
from auto_router.config import load_config  # noqa: E402


def task_request(task: dict) -> str:
    if task["grader"] in ("agent_humaneval", "agent_stdio"):
        body = task["instruction"]
        if "PROBLEM.md" in task.get("files", {}):
            body += "\n\nPROBLEM.md:\n" + task["files"]["PROBLEM.md"]
        elif "solution.py" in task.get("files", {}):
            body += "\n\nsolution.py:\n" + task["files"]["solution.py"]
        return body
    prompt = task["prompt"]
    if len(prompt) > 5000:  # long documents: head, tail and a size note
        prompt = prompt[:2000] + f"\n...[{len(prompt) // 4} tokens of document elided]...\n" + prompt[-2500:]
    return prompt


def classify(args):
    tasks = [json.loads(line) for line in open(args.tasks)]
    context = {"agentic": "Coding agent session with tools read_file, write_file, run_tests.",
               "tool_use": "Agent with order-management tools (find_customer, list_orders, refund, ...).",
               }

    def one(task):
        c = jev.classify(task_request(task), context.get(task["category"], ""))
        return {"task": task["id"], "category": task["category"], "level": task["difficulty"],
                "jev_category": c.category, "difficulty": c.difficulty, "confidence": c.difficulty_confidence,
                "needs_tools": c.needs_tools, "needs_long_context": c.needs_long_context, "stakes": c.stakes,
                "latency_s": round(c.latency_s, 2), "input_tokens": c.input_tokens,
                "output_tokens": c.output_tokens, "failed": c.failed}

    with ThreadPoolExecutor(8) as pool:
        rows = list(pool.map(one, tasks))
    Path(args.out).write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"{len(rows)} classified, {sum(r['failed'] for r in rows)} failures")


def judge(args):
    cfg = load_config(args.config)
    client = Client(cfg, args.ledger, args.budget)
    tasks = [json.loads(line) for line in open(args.tasks)
             if json.loads(line)["grader"] in ("humaneval", "stdio", "math")]

    def one(pair):
        model_name, task = pair
        model = cfg.catalog[model_name]
        r = client.chat(model, [{"role": "system", "content": task["system"]},
                                {"role": "user", "content": task["prompt"]}],
                        max_tokens=args.max_tokens, tag="judge-" + task["id"])
        if not r.ok:
            return None
        if task["grader"] == "humaneval":
            ok, _ = T.grade_humaneval(task, r.content)
        elif task["grader"] == "stdio":
            ok, _ = T.grade_stdio(task, T.extract_code(r.content))
        else:
            ok, _ = T.grade_math(task, r.content)
        answer = r.content if len(r.content) < 7500 else r.content[:2500] + "\n...\n" + r.content[-5000:]
        j = jev.judge(task["prompt"][:6000], answer)
        return {"model": model_name, "task": task["id"], "category": task["category"],
                "level": task["difficulty"], "correct": ok, "p_adequate": j.p_adequate,
                "judge_failed": j.failed, "judge_latency_s": round(j.latency_s, 2),
                "truncated": r.finish_reason == "length"}

    pairs = [(m, t) for m in args.models.split(",") for t in tasks]
    with ThreadPoolExecutor(args.workers) as pool:
        rows = [r for r in pool.map(one, pairs) if r]
    Path(args.out).write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"{len(rows)} judged")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("classify")
    c.add_argument("--tasks", default="experiments/tasks/tasks.jsonl")
    c.add_argument("--out", required=True)
    j = sub.add_parser("judge")
    j.add_argument("--tasks", default="experiments/tasks/tasks.jsonl")
    j.add_argument("--config", required=True)
    j.add_argument("--models", required=True)
    j.add_argument("--out", required=True)
    j.add_argument("--ledger", default="experiments/runs/ledger.jsonl")
    j.add_argument("--budget", type=float, default=30.0)
    j.add_argument("--workers", type=int, default=12)
    j.add_argument("--max-tokens", type=int, default=12000)
    args = ap.parse_args()
    {"classify": classify, "judge": judge}[args.cmd](args)
