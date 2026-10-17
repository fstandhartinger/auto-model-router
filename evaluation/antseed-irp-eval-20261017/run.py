#!/usr/bin/env python3
"""Generate one answer per (task, model) through OpenRouter and grade it.

  OPEN_ROUTER_API_KEY=... python run.py TASKS.jsonl PRICES.json ANSWERS.jsonl RESULTS.csv [--only-smoke]

Default reasoning settings, max_tokens 8000, one call per (task, model); transport errors (connection errors,
timeouts, HTTP 502/503/504) are retried at most twice, nothing else is. Truncated, empty and errored answers fail.
Spending stops dispatching new calls at USD 18 (max of OpenRouter's `usage.cost` and the list-price estimate, see OR_LIST). ANSWERS.jsonl (the answer
texts) stays local; RESULTS.csv is the public record. Re-running resumes: finished (task, model) pairs are skipped.
"""
import concurrent.futures as cf
import csv
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request

import grade

URL = "https://openrouter.ai/api/v1/chat/completions"
MAX_TOKENS = 8000
STOP_AT_USD = 18.0
WORKERS = 8
FIELDS = ["task", "model", "passed", "prompt_tokens", "completion_tokens", "cached_tokens", "cost_antseed",
          "cost_antseed_median", "cost_openrouter", "cost_openrouter_listed_est", "latency_s", "finish_reason"]
#: OpenRouter list price, USD per 1M tokens (input, output), read from /api/v1/models on 2026-10-17. Used only as a
#: floor for the spend guard: some routes (OpenAI) report usage.cost = 0, so the guard takes max(reported, listed).
OR_LIST = {"claude-opus-5": (5.0, 25.0), "gpt-5.6-sol": (4.0, 20.0), "gpt-5.6-luna": (0.2, 1.2), "kimi-k3": (0.304, 11.936),
           "glm-5.3-flash": (0.0238, 0.3966), "deepseek-v4.1-flash": (0.0153, 0.594)}
lock = threading.Lock()
spent = [0.0]


def call(model_id, prompt, key):
    body = json.dumps({"model": model_id, "messages": [{"role": "user", "content": prompt}], "max_tokens": MAX_TOKENS,
                       "usage": {"include": True}}).encode()
    req = urllib.request.Request(URL, body, {"Authorization": "Bearer " + key, "Content-Type": "application/json"})
    last = None
    for attempt in range(3):
        started = time.time()
        try:
            with urllib.request.urlopen(req, timeout=900) as r:
                data = json.load(r)
            return data, time.time() - started, attempt, None
        except urllib.error.HTTPError as e:
            last = f"HTTP {e.code}: {e.read()[:300]!r}"
            if e.code not in (502, 503, 504):
                return None, time.time() - started, attempt, last
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
            last = f"transport: {type(e).__name__}: {e}"
    return None, 0.0, 2, last


def one(task, key_name, info, key):
    with lock:
        if spent[0] >= STOP_AT_USD:
            return None
    data, latency, attempts, error = call(info["openrouter_id"], task["prompt"], key)
    rec = {"task": task["id"], "model": key_name, "latency_s": round(latency, 3), "attempts": attempts + 1, "error": error}
    if data and data.get("choices"):
        ch = data["choices"][0]
        u = data.get("usage") or {}
        rec.update(content=(ch.get("message") or {}).get("content") or "", finish_reason=ch.get("finish_reason"),
                   prompt_tokens=u.get("prompt_tokens", 0), completion_tokens=u.get("completion_tokens", 0),
                   cached_tokens=(u.get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0,
                   reasoning_tokens=(u.get("completion_tokens_details") or {}).get("reasoning_tokens", 0) or 0,
                   cost_openrouter=u.get("cost") or 0.0, provider=data.get("provider"), generation_id=data.get("id"))
    elif data:
        rec.update(error=error or "no choices: " + json.dumps(data)[:300])
    pi, po = OR_LIST[key_name]
    rec["cost_listed_est"] = (rec.get("prompt_tokens", 0) * pi + rec.get("completion_tokens", 0) * po) / 1e6
    with lock:
        spent[0] += max(rec.get("cost_openrouter", 0.0), rec["cost_listed_est"])
    return rec


def public_row(rec, task, prices):
    p = prices["models"][rec["model"]]
    passed, _ = (False, "error") if rec.get("error") or "content" not in rec else grade.grade(task, rec["content"], rec["finish_reason"])
    def cost(basis):
        pr = p[basis]
        pt, cached = rec.get("prompt_tokens", 0), rec.get("cached_tokens", 0)
        return ((pt - cached) * pr["input"] + cached * pr["cache_read"] + rec.get("completion_tokens", 0) * pr["output"]) / 1e6
    return {"task": rec["task"], "model": rec["model"], "passed": int(passed), "prompt_tokens": rec.get("prompt_tokens", 0),
            "completion_tokens": rec.get("completion_tokens", 0), "cached_tokens": rec.get("cached_tokens", 0),
            "cost_antseed": f"{cost('cheapest'):.9f}", "cost_antseed_median": f"{cost('median'):.9f}",
            "cost_openrouter": f"{rec.get('cost_openrouter', 0.0):.9f}",
            "cost_openrouter_listed_est": f"{rec.get('cost_listed_est', 0.0):.9f}", "latency_s": rec["latency_s"],
            "finish_reason": rec.get("finish_reason") or "error"}


def main(tasks_path, prices_path, answers_path, results_path, smoke=False):
    key = os.environ["OPEN_ROUTER_API_KEY"]
    tasks = [json.loads(l) for l in open(tasks_path)]
    prices = json.load(open(prices_path))
    done = {}
    if os.path.exists(answers_path):
        for l in open(answers_path):
            r = json.loads(l)
            done[(r["task"], r["model"])] = r
            spent[0] += max(r.get("cost_openrouter", 0.0), r.get("cost_listed_est", 0.0))
    jobs = [(t, m) for t in tasks for m in prices["models"] if (t["id"], m) not in done]
    if smoke:
        jobs = jobs[:0]
    with open(answers_path, "a") as out, cf.ThreadPoolExecutor(WORKERS) as pool:
        futs = [pool.submit(one, t, m, prices["models"][m], key) for t, m in jobs]
        for n, f in enumerate(cf.as_completed(futs), 1):
            rec = f.result()
            if rec:
                out.write(json.dumps(rec) + "\n")
                out.flush()
                done[(rec["task"], rec["model"])] = rec
            if n % 20 == 0:
                print(f"{n}/{len(jobs)} calls, spent ${spent[0]:.4f}", flush=True)
    by_id = {t["id"]: t for t in tasks}
    missing = [(t["id"], m) for t in tasks for m in prices["models"] if (t["id"], m) not in done]
    with open(results_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for t in tasks:
            for m in prices["models"]:
                if (t["id"], m) in done:
                    w.writerow(public_row(done[(t["id"], m)], by_id[t["id"]], prices))
    print(f"done; spent ${spent[0]:.4f}; not run (budget stop): {len(missing)}")


if __name__ == "__main__":
    main(*sys.argv[1:5], smoke="--only-smoke" in sys.argv)
