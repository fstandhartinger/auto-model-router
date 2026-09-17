"""Derive a traffic profile from local agent logs. Metadata and counts only.

Reads Claude Code project logs, Codex rollout logs and the OpenCode SQLite
database, and prints aggregate statistics: API calls per session, calls per
user turn, prefix sizes, output sizes, gaps between calls and between user
turns (for cache expiry), cache read share, and model mix. No message text is
read into the output.

    python experiments/traffic_stats.py --days 7 --out traffic.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sqlite3
import statistics
import time
from collections import Counter, defaultdict
from datetime import datetime


def ts(value: str) -> float:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def pct(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    return values[min(len(values) - 1, int(q * len(values)))]


def summarise(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    return {"n": len(values), "mean": round(statistics.mean(values), 1), "p10": round(pct(values, .1), 1),
            "p50": round(pct(values, .5), 1), "p90": round(pct(values, .9), 1), "max": round(max(values), 1)}


# ---------------------------------------------------------------------------
def claude_sessions(root: str, since: float) -> list[dict]:
    sessions = []
    for path in glob.glob(os.path.join(os.path.expanduser(root), "**", "*.jsonl"), recursive=True):
        if os.path.getmtime(path) < since:
            continue
        calls: dict[str, dict] = {}
        user_turns: list[float] = []
        with open(path, errors="replace") as fh:
            for line in fh:
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                kind = d.get("type")
                stamp = d.get("timestamp")
                if not stamp:
                    continue
                t = ts(stamp)
                if t < since:
                    continue
                message = d.get("message") or {}
                if kind == "assistant" and message.get("usage"):
                    key = message.get("id") or d.get("requestId") or stamp
                    if key in calls:
                        calls[key]["tools"] += sum(1 for b in message.get("content") or []
                                                   if isinstance(b, dict) and b.get("type") == "tool_use")
                        continue
                    u = message["usage"]
                    calls[key] = {"t": t, "model": message.get("model"),
                                  "uncached": u.get("input_tokens") or 0,
                                  "read": u.get("cache_read_input_tokens") or 0,
                                  "write": u.get("cache_creation_input_tokens") or 0,
                                  "output": u.get("output_tokens") or 0,
                                  "tools": sum(1 for b in message.get("content") or []
                                               if isinstance(b, dict) and b.get("type") == "tool_use")}
                elif kind == "user" and not d.get("isMeta"):
                    content = message.get("content")
                    is_text = isinstance(content, str) or (isinstance(content, list) and any(
                        isinstance(b, dict) and b.get("type") == "text" for b in content))
                    is_tool = isinstance(content, list) and any(
                        isinstance(b, dict) and b.get("type") == "tool_result" for b in content)
                    if is_text and not is_tool:
                        user_turns.append(t)
        if calls:
            sessions.append({"source": "claude", "calls": sorted(calls.values(), key=lambda c: c["t"]),
                             "user_turns": sorted(user_turns)})
    return sessions


def codex_sessions(root: str, since: float) -> list[dict]:
    sessions = []
    for path in glob.glob(os.path.join(os.path.expanduser(root), "**", "rollout-*.jsonl"), recursive=True):
        if os.path.getmtime(path) < since:
            continue
        calls, user_turns, model = [], [], None
        with open(path, errors="replace") as fh:
            for line in fh:
                if '"token_count"' not in line and '"turn_context"' not in line and '"user_message"' not in line:
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                stamp = d.get("timestamp")
                if not stamp or ts(stamp) < since:
                    continue
                p = d.get("payload") or {}
                if d.get("type") == "turn_context":
                    model = p.get("model") or model
                elif p.get("type") == "user_message":
                    user_turns.append(ts(stamp))
                elif p.get("type") == "token_count" and (p.get("info") or {}).get("last_token_usage"):
                    u = p["info"]["last_token_usage"]
                    inp, cached = u.get("input_tokens") or 0, u.get("cached_input_tokens") or 0
                    calls.append({"t": ts(stamp), "model": model, "uncached": inp - cached, "read": cached,
                                  "write": 0, "output": u.get("output_tokens") or 0, "tools": 0})
        if calls:
            sessions.append({"source": "codex", "calls": calls, "user_turns": sorted(user_turns)})
    return sessions


def opencode_sessions(db: str, since: float) -> list[dict]:
    path = os.path.expanduser(db)
    if not os.path.exists(path):
        return []
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    by_session: dict[str, dict] = defaultdict(lambda: {"source": "opencode", "calls": [], "user_turns": []})
    for sid, created, data in con.execute(
            "select session_id, time_created, data from message where time_created > ? order by time_created",
            (int(since * 1000),)):
        try:
            d = json.loads(data)
        except json.JSONDecodeError:
            continue
        if d.get("role") == "user":
            by_session[sid]["user_turns"].append(created / 1000)
            continue
        tok = d.get("tokens") or {}
        cache = tok.get("cache") or {}
        if not (tok.get("input") or cache.get("read") or tok.get("output")):
            continue
        by_session[sid]["calls"].append({"t": created / 1000, "model": d.get("modelID"),
                                         "uncached": tok.get("input") or 0, "read": cache.get("read") or 0,
                                         "write": cache.get("write") or 0, "output": tok.get("output") or 0,
                                         "tools": 0})
    con.close()
    return [s for s in by_session.values() if s["calls"]]


# ---------------------------------------------------------------------------
def profile(sessions: list[dict]) -> dict:
    prefix, output, call_gaps, turn_gaps, calls_per_turn, calls_per_session, turns_per_session = \
        [], [], [], [], [], [], []
    read_tokens = total_input = 0
    models: Counter = Counter()
    gap_over = Counter()
    for s in sessions:
        calls = s["calls"]
        calls_per_session.append(len(calls))
        turns = s["user_turns"] or [calls[0]["t"]]
        turns_per_session.append(len(turns))
        for i, c in enumerate(calls):
            p = c["uncached"] + c["read"] + c["write"]
            prefix.append(p)
            output.append(c["output"])
            read_tokens += c["read"]
            total_input += p
            models[c["model"] or "?"] += 1
            if i:
                gap = c["t"] - calls[i - 1]["t"]
                call_gaps.append(gap)
                for limit in (300, 600, 3600):
                    gap_over[limit] += gap > limit
        # calls per user turn: calls between consecutive user-turn timestamps
        boundaries = turns + [float("inf")]
        for a, b in zip(boundaries, boundaries[1:]):
            n = sum(1 for c in calls if a <= c["t"] < b)
            if n:
                calls_per_turn.append(n)
        for a, b in zip(turns, turns[1:]):
            turn_gaps.append(b - a)
    n_gaps = max(1, len(call_gaps))
    return {
        "sessions": len(sessions),
        "api_calls": len(prefix),
        "calls_per_session": summarise(calls_per_session),
        "user_turns_per_session": summarise(turns_per_session),
        "calls_per_user_turn": summarise(calls_per_turn),
        "prefix_tokens": summarise(prefix),
        "output_tokens": summarise(output),
        "gap_between_calls_s": summarise(call_gaps),
        "gap_between_user_turns_s": summarise(turn_gaps),
        "share_of_call_gaps_over": {f"{k}s": round(v / n_gaps, 4) for k, v in gap_over.items()},
        "cache_read_share_of_input": round(read_tokens / total_input, 4) if total_input else 0.0,
        "input_tokens_total": total_input,
        "output_tokens_total": int(sum(output)),
        "models": dict(models.most_common(12)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=float, default=7)
    ap.add_argument("--claude", default="~/.claude/projects")
    ap.add_argument("--codex", default="~/.codex/sessions")
    ap.add_argument("--opencode", default="~/.local/share/opencode/opencode.db")
    ap.add_argument("--out")
    args = ap.parse_args()
    since = time.time() - args.days * 86400
    data = {"days": args.days, "generated_at": time.time()}
    all_sessions = []
    for name, loader, src in (("claude", claude_sessions, args.claude), ("codex", codex_sessions, args.codex),
                              ("opencode", opencode_sessions, args.opencode)):
        sessions = loader(src, since)
        all_sessions += sessions
        data[name] = profile(sessions)
        # keep raw per-call metadata for the simulator (numbers only)
        data[name]["_sessions"] = [
            {"calls": [[round(c["t"] - s["calls"][0]["t"], 1), c["uncached"] + c["read"] + c["write"], c["read"],
                        c["output"]] for c in s["calls"]],
             "user_turns": [round(t - s["calls"][0]["t"], 1) for t in s["user_turns"]],
             "model": Counter(c["model"] for c in s["calls"]).most_common(1)[0][0],
             "start": round(s["calls"][0]["t"])}
            for s in sessions]
    data["all"] = profile(all_sessions)
    printable = {k: ({kk: vv for kk, vv in v.items() if kk != "_sessions"} if isinstance(v, dict) else v)
                 for k, v in data.items()}
    print(json.dumps(printable, indent=1))
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(data, fh)


if __name__ == "__main__":
    main()
