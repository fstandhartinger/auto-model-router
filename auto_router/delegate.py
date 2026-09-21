"""MCP tools for handing bounded work from a strong planner to routed workers.

The planner's client remains unchanged. Workers are launched through the normal
job router, never through a subscription route, and the response distinguishes
the router's pre-run cost estimate from measured cost (which most agent CLIs do
not report).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "2025-06-18"
MAX_OUTPUT = 12000
MAX_PARALLEL = 8

_COMMON_PROPERTIES = {
    "context": {"type": "string", "description": "only the task-local context the worker needs"},
    "cwd": {"type": "string", "description": "working directory (default: the server's)"},
    "tier": {"type": "string", "enum": ["cheap", "auto", "strong"], "default": "cheap",
             "description": "cheap prefers a low-cost worker unless the router judges the task hard; auto uses expected cost; strong uses the most capable non-plan worker"},
    "timeout_s": {"type": "integer", "minimum": 1, "maximum": 7200, "default": 900},
}

DELEGATE_TOOL = {
    "name": "delegate",
    "description": (
        "Run a self-contained sub-task on routed worker agents. Use cheap for mechanical work; "
        "the router may choose a stronger worker when the task is hard. parallel>1 asks independent "
        "workers for results. Give only task-local context and verify their output before using it."),
    "inputSchema": {
        "type": "object",
        "properties": {
            "task": {"type": "string", "description": "complete, self-contained worker brief"},
            **_COMMON_PROPERTIES,
            "parallel": {"type": "integer", "minimum": 1, "maximum": MAX_PARALLEL, "default": 1},
        },
        "required": ["task"],
        "additionalProperties": False,
    },
}

DELEGATE_MANY_TOOL = {
    "name": "delegate_many",
    "description": (
        "Run different independent worker briefs concurrently. Keep dependencies with the planner; "
        "use this only when every task can finish without another worker's result."),
    "inputSchema": {
        "type": "object",
        "properties": {
            "tasks": {"type": "array", "minItems": 1, "maxItems": 32,
                      "items": {"type": "string"},
                      "description": "self-contained worker briefs"},
            **_COMMON_PROPERTIES,
            "parallel": {"type": "integer", "minimum": 1, "maximum": MAX_PARALLEL, "default": 4},
        },
        "required": ["tasks"],
        "additionalProperties": False,
    },
}

TOOLS = [DELEGATE_TOOL, DELEGATE_MANY_TOOL]


def _brief(task: str, context: str | None) -> str:
    task = task.strip()
    if not context or not context.strip():
        return task
    return f"{task}\n\nTask-local context:\n{context.strip()}"


def launcher_argv(task: str, cwd: str | None, tier: str = "cheap") -> list[str]:
    argv = [sys.executable, "-m", "auto_router.launcher", "--no-plans", "--json",
            "--tier", tier]
    if cwd:
        argv += ["--cwd", cwd]
    return argv + [task]


def _decision(stderr: str) -> dict[str, Any]:
    for line in reversed((stderr or "").splitlines()):
        try:
            value = json.loads(line)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(value, dict) and "route" in value:
            return value
    return {}


def run_delegate(task: str, context: str | None = None, cwd: str | None = None,
                 tier: str = "cheap", timeout_s: int = 900,
                 run: Callable[..., Any] = subprocess.run) -> dict[str, Any]:
    """Run one worker and return a truthful, machine-readable result."""
    started = time.perf_counter()
    if not task.strip():
        return {"ok": False, "error": "empty task"}
    if cwd and not Path(cwd).is_dir():
        return {"ok": False, "error": f"no such directory: {cwd}"}
    if tier not in {"cheap", "auto", "strong"}:
        return {"ok": False, "error": f"unknown tier: {tier}"}
    timeout_s = max(1, min(int(timeout_s), 7200))
    full_brief = _brief(task, context)
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(ROOT), env.get("PYTHONPATH", "")) if p)
    try:
        proc = run(launcher_argv(full_brief, cwd, tier), cwd=cwd or None, env=env,
                   capture_output=True, text=True, timeout=timeout_s, stdin=subprocess.DEVNULL)
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"worker timed out after {timeout_s}s",
                "tier": tier, "wall_time_s": round(time.perf_counter() - started, 3),
                "brief_chars": len(full_brief), "brief_tokens_estimate": (len(full_brief) + 3) // 4}
    decision = _decision(proc.stderr or "")
    output = (proc.stdout or "").strip()
    truncated = len(output) > MAX_OUTPUT
    if truncated:
        output = "[... earlier output cut ...]\n" + output[-MAX_OUTPUT:]
    estimate = ((decision.get("decision") or {}).get("estimated_outcome") or {}).get("cost_usd")
    return {
        "ok": proc.returncode == 0,
        "model": decision.get("route"),
        "tier": tier,
        "result": output or "(no output)",
        "exit_code": proc.returncode,
        "wall_time_s": round(time.perf_counter() - started, 3),
        "cost_usd": None,
        "cost_basis": "not measured: the launched agent CLI did not report token usage",
        "estimated_cost_usd": estimate,
        "estimated_cost_basis": "router estimate before the worker ran" if estimate is not None else None,
        "brief_chars": len(full_brief),
        "brief_tokens_estimate": (len(full_brief) + 3) // 4,
        "output_truncated": truncated,
    }


def run_many(tasks: list[str], *, context: str | None = None, cwd: str | None = None,
             tier: str = "cheap", parallel: int = 4, timeout_s: int = 900,
             runner: Callable[..., dict[str, Any]] = run_delegate) -> dict[str, Any]:
    if not tasks or any(not isinstance(task, str) or not task.strip() for task in tasks):
        return {"ok": False, "error": "tasks must contain non-empty strings"}
    workers = max(1, min(int(parallel), MAX_PARALLEL, len(tasks)))
    started = time.perf_counter()
    results: list[dict[str, Any] | None] = [None] * len(tasks)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = {pool.submit(runner, task, context, cwd, tier, timeout_s): i
                   for i, task in enumerate(tasks)}
        for future in as_completed(pending):
            i = pending[future]
            try:
                results[i] = future.result()
            except Exception as exc:  # a broken worker must not lose its siblings
                results[i] = {"ok": False, "error": f"worker failed: {type(exc).__name__}: {exc}"}
    finished = [r or {"ok": False, "error": "worker produced no result"} for r in results]
    known_costs = [r.get("cost_usd") for r in finished if r.get("cost_usd") is not None]
    estimates = [r.get("estimated_cost_usd") for r in finished
                 if r.get("estimated_cost_usd") is not None]
    return {
        "ok": all(r.get("ok") for r in finished),
        "results": finished,
        "wall_time_s": round(time.perf_counter() - started, 3),
        "parallel": workers,
        "cost_usd": round(sum(known_costs), 8) if len(known_costs) == len(finished) else None,
        "estimated_cost_usd": round(sum(estimates), 8) if len(estimates) == len(finished) else None,
        "brief_tokens_estimate": sum(int(r.get("brief_tokens_estimate") or 0) for r in finished),
    }


def _text_result(data: dict[str, Any]) -> tuple[str, bool, dict[str, Any]]:
    return json.dumps(data, ensure_ascii=False, indent=2), not bool(data.get("ok")), data


def _call(name: str, arguments: dict) -> tuple[str, bool, dict[str, Any]]:
    common = {
        "context": arguments.get("context"), "cwd": arguments.get("cwd"),
        "tier": arguments.get("tier") or "cheap",
        "timeout_s": int(arguments.get("timeout_s") or 900),
    }
    if name == "delegate":
        copies = max(1, min(int(arguments.get("parallel") or 1), MAX_PARALLEL))
        if copies == 1:
            return _text_result(run_delegate(str(arguments.get("task") or ""), **common))
        return _text_result(run_many([str(arguments.get("task") or "")] * copies,
                                     parallel=copies, **common))
    if name == "delegate_many":
        return _text_result(run_many(list(arguments.get("tasks") or []),
                                     parallel=int(arguments.get("parallel") or 4), **common))
    return json.dumps({"error": f"unknown tool {name!r}"}), True, {}


def handle(message: dict, run_tool: Callable[[str, dict], tuple[str, bool, dict]]) -> dict | None:
    method, mid = message.get("method"), message.get("id")
    if mid is None:
        return None
    if method == "initialize":
        version = (message.get("params") or {}).get("protocolVersion") or PROTOCOL
        result = {"protocolVersion": version, "capabilities": {"tools": {}},
                  "serverInfo": {"name": "auto-router-delegate", "version": "0.2.0"}}
    elif method == "ping":
        result = {}
    elif method == "tools/list":
        result = {"tools": TOOLS}
    elif method == "tools/call":
        params = message.get("params") or {}
        if params.get("name") not in {tool["name"] for tool in TOOLS}:
            return {"jsonrpc": "2.0", "id": mid,
                    "error": {"code": -32602, "message": f"unknown tool {params.get('name')!r}"}}
        text, is_error, structured = run_tool(params["name"], params.get("arguments") or {})
        result = {"content": [{"type": "text", "text": text}], "isError": is_error,
                  "structuredContent": structured}
    else:
        return {"jsonrpc": "2.0", "id": mid,
                "error": {"code": -32601, "message": f"no method {method}"}}
    return {"jsonrpc": "2.0", "id": mid, "result": result}


def main(stdin=sys.stdin, stdout=sys.stdout) -> int:
    for line in stdin:
        line = line.strip()
        if not line:
            continue
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            reply = {"jsonrpc": "2.0", "id": None,
                     "error": {"code": -32700, "message": "parse error"}}
        else:
            reply = handle(message, _call)
        if reply is not None:
            stdout.write(json.dumps(reply) + "\n")
            stdout.flush()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
