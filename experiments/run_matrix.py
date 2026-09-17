"""Run the graded task set against candidate models; one JSONL row per (model, task).

Resumable: finished (model, task) pairs are skipped. Spend is capped through
the call ledger.

    python experiments/run_matrix.py --config my.local.yaml --models a,b --out experiments/runs/matrix.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import tasks as T  # noqa: E402
import toolenv  # noqa: E402
from llm import BudgetExceeded, Client  # noqa: E402

from auto_router.config import load_config  # noqa: E402

_write_lock = threading.Lock()


class Trace:
    """Aggregates usage over all calls of one task."""

    def __init__(self):
        self.calls = []

    def add(self, r):
        self.calls.append(r)
        return r

    def summary(self) -> dict:
        c = self.calls
        return {"calls": len(c), "prompt_tokens": sum(x.prompt_tokens for x in c),
                "cached_tokens": sum(x.cached_tokens for x in c),
                "cache_write_tokens": sum(x.cache_write_tokens for x in c),
                "output_tokens": sum(x.output_tokens for x in c),
                "reasoning_tokens": sum(x.reasoning_tokens for x in c),
                "cost_usd": round(sum(x.cost_usd for x in c), 6),
                "list_cost_usd": round(sum(x.list_cost_usd for x in c), 6),
                "latency_s": round(sum(x.latency_s for x in c), 1),
                "max_prompt_tokens": max((x.prompt_tokens for x in c), default=0),
                "routed_models": [x.routed_model for x in c if x.routed_model],
                "errors": [x.error for x in c if not x.ok][:3]}


def single_shot(client, model, task, trace, max_tokens):
    messages = [{"role": "system", "content": task["system"]}, {"role": "user", "content": task["prompt"]}]
    r = trace.add(client.chat(model, messages, max_tokens=max_tokens, tag=task["id"]))
    if not r.ok:
        return False, f"call failed: {r.error}", r.content
    if task["grader"] == "humaneval":
        ok, why = T.grade_humaneval(task, r.content)
    elif task["grader"] == "stdio":
        ok, why = T.grade_stdio(task, T.extract_code(r.content))
    else:
        ok, why = T.grade_math(task, r.content)
    if not ok and r.finish_reason == "length":
        why += " (truncated)"
    return ok, why, r.content


def tool_loop(client, model, task, trace, max_tokens, max_steps=25):
    world = toolenv.World.from_json(task["world"])
    messages = [{"role": "system", "content": task["system"]}, {"role": "user", "content": task["prompt"]}]
    for _ in range(max_steps):
        r = trace.add(client.chat(model, messages, tools=toolenv.TOOLS, max_tokens=max_tokens, tag=task["id"]))
        if not r.ok:
            return False, f"call failed: {r.error}", ""
        reply = {"role": "assistant", "content": r.content or None}
        if r.tool_calls:
            reply["tool_calls"] = r.tool_calls
        messages.append(reply)
        if not r.tool_calls:
            break
        finished = False
        for call in r.tool_calls:
            name = call["function"]["name"]
            try:
                args = json.loads(call["function"].get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            out = world.call(name, args if isinstance(args, dict) else {})
            messages.append({"role": "tool", "tool_call_id": call.get("id", ""), "content": out})
            finished = finished or name == "done"
        if finished:
            break
    ok, why = toolenv.grade(task, world.to_json())
    return ok, why, ""


TASK_DEADLINE_S = 480.0
FOLLOW_UP = ""


def agent_loop(client, model, task, trace, max_tokens, max_steps=30):
    deadline = time.time() + TASK_DEADLINE_S
    followed = False
    with tempfile.TemporaryDirectory() as ws:
        root = Path(ws)
        for rel, content in task["files"].items():
            (root / rel).parent.mkdir(parents=True, exist_ok=True)
            (root / rel).write_text(content)

        def run_visible() -> str:
            code = (root / "solution.py").read_text() if (root / "solution.py").exists() else ""
            if not code:
                return "solution.py does not exist yet"
            if task["grader"] == "agent_humaneval":
                ok, why = T.grade_humaneval(task["visible"], code)
            else:
                ok, why = T.grade_stdio(task, code, task["visible_tests"])
            return ("PASSED: " if ok else "FAILED: ") + why

        messages = [{"role": "system", "content": "You are a coding agent working in a small workspace through "
                                                  "tools. Work step by step; keep replies short."},
                    {"role": "user", "content": task["instruction"]}]
        for _ in range(max_steps):
            if time.time() > deadline:
                break
            r = trace.add(client.chat(model, messages, tools=T.AGENT_TOOLS, max_tokens=max_tokens, tag=task["id"]))
            if not r.ok:
                break
            reply = {"role": "assistant", "content": r.content or None}
            if r.tool_calls:
                reply["tool_calls"] = r.tool_calls
            messages.append(reply)
            if not r.tool_calls:
                messages.append({"role": "user", "content": "Continue using the tools; call done when finished."})
                continue
            finished = False
            for call in r.tool_calls:
                name = call["function"]["name"]
                try:
                    args = json.loads(call["function"].get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {}
                if not isinstance(args, dict):
                    args = {}
                if name == "list_files":
                    out = "\n".join(str(p.relative_to(root)) for p in root.rglob("*") if p.is_file())
                elif name == "read_file":
                    p = (root / str(args.get("path", ""))).resolve()
                    out = p.read_text()[:20000] if p.is_file() and root in p.parents else "no such file"
                elif name == "write_file":
                    p = (root / str(args.get("path", ""))).resolve()
                    if root in p.parents:
                        p.parent.mkdir(parents=True, exist_ok=True)
                        p.write_text(str(args.get("content", "")))
                        out = "written"
                    else:
                        out = "path outside workspace"
                elif name == "run_tests":
                    out = run_visible()
                elif name == "done":
                    out, finished = "ok", True
                else:
                    out = f"unknown tool {name}"
                messages.append({"role": "tool", "tool_call_id": call.get("id", ""), "content": out})
            if finished:
                if FOLLOW_UP and not followed:
                    followed = True
                    messages.append({"role": "user", "content": FOLLOW_UP})
                    continue
                break
        code = (root / "solution.py").read_text() if (root / "solution.py").exists() else ""
        if not code:
            return False, "no solution.py", ""
        if task["grader"] == "agent_humaneval":
            ok, why = T.grade_humaneval(task["hidden"], code)
        else:
            ok, why = T.grade_stdio(task, code)
        return ok, why, ""


def run_one(client, model, task, max_tokens):
    trace = Trace()
    started = time.time()
    try:
        if task["grader"] in ("humaneval", "stdio", "math"):
            ok, why, _ = single_shot(client, model, task, trace, max_tokens)
        elif task["grader"] == "toolenv":
            ok, why, _ = tool_loop(client, model, task, trace, max_tokens)
        else:
            ok, why, _ = agent_loop(client, model, task, trace, max_tokens)
    except BudgetExceeded:
        raise
    except Exception as exc:  # noqa: BLE001 - record and continue
        ok, why = False, f"harness error {type(exc).__name__}: {exc}"[:300]
    infra = bool(trace.calls) and all(not c.ok for c in trace.calls)
    return {"model": model.name, "task": task["id"], "category": task["category"], "difficulty": task["difficulty"],
            "ok": ok, "why": why[:300], "infra_error": infra or not trace.calls,
            "wall_s": round(time.time() - started, 1), **trace.summary()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--tasks", default="experiments/tasks/tasks.jsonl")
    ap.add_argument("--models", required=True)
    ap.add_argument("--out", default="experiments/runs/matrix.jsonl")
    ap.add_argument("--ledger", default="experiments/runs/ledger.jsonl")
    ap.add_argument("--budget", type=float, default=30.0)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--max-tokens", type=int, default=16000)
    ap.add_argument("--filter", default="", help="substring filter on task id")
    ap.add_argument("--time-budget", type=float, default=540, help="stop submitting after this many seconds")
    ap.add_argument("--retry-infra", action="store_true")
    ap.add_argument("--task-deadline", type=float, default=480.0, help="agent loops stop after this many seconds")
    ap.add_argument("--router-url", help="send every call to a running router (model 'auto') instead")
    ap.add_argument("--follow-up", default="", help="second user turn appended after the agent calls done")
    args = ap.parse_args()

    global TASK_DEADLINE_S, FOLLOW_UP
    TASK_DEADLINE_S = args.task_deadline
    FOLLOW_UP = args.follow_up
    cfg = load_config(args.config)
    client = Client(cfg, args.ledger, args.budget)
    if args.router_url:
        from auto_router.catalog import Catalog, ModelInfo, Prices
        from auto_router.config import Provider
        client.router_catalog = {m.name: m for m in cfg.catalog.all()}
        cfg.providers["router"] = Provider("router", args.router_url.rstrip("/"))
        auto = ModelInfo(name="auto", provider="router", upstream_id="auto", prices=Prices.free())
        cfg.catalog = Catalog(cfg.catalog.all() + [auto])
    tasks = [json.loads(line) for line in open(args.tasks)]
    tasks = [t for t in tasks if args.filter in t["id"]]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            row = json.loads(line)
            if args.retry_infra and row.get("infra_error"):
                continue
            done.add((row["model"], row["task"]))
    jobs = [(cfg.catalog[m], t) for t in tasks for m in args.models.split(",") if (m, t["id"]) not in done]
    print(f"{len(jobs)} jobs pending; spent so far ${client.spent():.3f}", flush=True)
    started = time.time()
    with ThreadPoolExecutor(args.workers) as pool:
        futures = {}
        it = iter(jobs)
        for _ in range(args.workers * 2):
            job = next(it, None)
            if job:
                futures[pool.submit(run_one, client, job[0], job[1], args.max_tokens)] = job
        while futures:
            for fut in as_completed(list(futures)):
                futures.pop(fut)
                try:
                    row = fut.result()
                except BudgetExceeded as exc:
                    print("BUDGET STOP:", exc, flush=True)
                    pool.shutdown(cancel_futures=True)
                    return
                with _write_lock, out.open("a") as fh:
                    fh.write(json.dumps(row) + "\n")
                print(f"{row['model']:16} {row['task'][:34]:34} {'OK ' if row['ok'] else 'no '} "
                      f"calls={row['calls']} in={row['prompt_tokens']} cached={row['cached_tokens']} "
                      f"out={row['output_tokens']} ${row['cost_usd']:.4f} {row['wall_s']}s {row['why'][:50]}",
                      flush=True)
                if time.time() - started < args.time_budget:
                    job = next(it, None)
                    if job:
                        futures[pool.submit(run_one, client, job[0], job[1], args.max_tokens)] = job
                break
    print(f"finished; spent ${client.spent():.3f}; {len(jobs) - len(done)} submitted", flush=True)


if __name__ == "__main__":
    main()
