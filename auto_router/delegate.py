"""A ``delegate`` tool for Claude Code and Codex: the plan orchestrates, cheap routes do the legwork.

This is the mode that needs nothing from anyone's terms but the ordinary use of
two features both clients document: MCP servers and tools. The official client
runs unmodified, signed in with its own plan, talking to its own vendor - the
router is never in that path. What the router adds is a tool the plan model can
*choose* to call::

    delegate(task="write unit tests for parser.py and run them", cwd="/repo")

The call runs the job launcher (``launcher.py``) restricted to non-plan routes:
it picks the cheapest route expected to do the job, starts that route's own
agent CLI in ``cwd``, and returns what it printed. The plan model then reads
the result and checks it, which is the part it is good at, and the output
tokens and tool-loop turns of the sub-task are spent somewhere else.

Register it (stdio MCP server)::

    claude mcp add auto-router-delegate -- python -m auto_router.delegate
    codex mcp add auto-router-delegate -- python -m auto_router.delegate

It needs ``AUTO_ROUTER_CONFIG`` pointing at a configuration whose cheap routes
have ``runner`` blocks (see ``examples/launcher.example.yaml``).

The protocol handled here is the small stdio subset of MCP a tool server needs
(``initialize``, ``tools/list``, ``tools/call``, ``ping``), newline-delimited
JSON-RPC 2.0, with no dependency beyond the standard library.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "2025-06-18"
MAX_OUTPUT = 12000

TOOL = {
    "name": "delegate",
    "description": (
        "Hand a self-contained sub-task to a cheaper model instead of doing it yourself. "
        "Good for: writing or extending tests, boilerplate, mechanical refactors across files, "
        "summarising files or logs, first drafts. The sub-agent works in `cwd` with its own "
        "shell and editor and cannot see this conversation, so state everything it needs "
        "(files, constraints, how to verify). It returns the sub-agent's final output. "
        "Check the result (read the diff, run the tests) before relying on it."),
    "inputSchema": {
        "type": "object",
        "properties": {
            "task": {"type": "string", "description": "complete, self-contained instructions"},
            "cwd": {"type": "string", "description": "working directory (default: the server's)"},
            "timeout_s": {"type": "integer", "description": "wall-clock limit, default 900"},
        },
        "required": ["task"],
        "additionalProperties": False,
    },
}


def launcher_argv(task: str, cwd: str | None) -> list[str]:
    argv = [sys.executable, "-m", "auto_router.launcher", "--no-plans"]
    if cwd:
        argv += ["--cwd", cwd]
    return argv + [task]


def run_delegate(task: str, cwd: str | None = None, timeout_s: int = 900,
                 run: Callable[..., Any] = subprocess.run) -> tuple[str, bool]:
    """Run one delegated job; returns (text for the model, is_error)."""
    if not task.strip():
        return "delegate: empty task", True
    if cwd and not Path(cwd).is_dir():
        return f"delegate: no such directory: {cwd}", True
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(ROOT), env.get("PYTHONPATH", "")) if p)
    try:
        proc = run(launcher_argv(task, cwd), cwd=cwd or None, env=env, capture_output=True,
                   text=True, timeout=timeout_s, stdin=subprocess.DEVNULL)
    except subprocess.TimeoutExpired:
        return f"delegate: the sub-agent did not finish within {timeout_s}s", True
    route = next((line for line in (proc.stderr or "").splitlines() if line.strip()), "")
    output = (proc.stdout or "").strip()
    if len(output) > MAX_OUTPUT:
        output = "[... earlier output cut ...]\n" + output[-MAX_OUTPUT:]
    text = f"route: {route}\nexit code: {proc.returncode}\n\n{output or '(no output)'}"
    return text, proc.returncode != 0


def handle(message: dict, run_tool: Callable[[dict], tuple[str, bool]]) -> dict | None:
    method = message.get("method")
    mid = message.get("id")
    if mid is None:                      # a notification: nothing to answer
        return None
    if method == "initialize":
        version = (message.get("params") or {}).get("protocolVersion") or PROTOCOL
        result = {"protocolVersion": version, "capabilities": {"tools": {}},
                  "serverInfo": {"name": "auto-router-delegate", "version": "0.1.0"}}
    elif method == "ping":
        result = {}
    elif method == "tools/list":
        result = {"tools": [TOOL]}
    elif method == "tools/call":
        params = message.get("params") or {}
        if params.get("name") != TOOL["name"]:
            return {"jsonrpc": "2.0", "id": mid,
                    "error": {"code": -32602, "message": f"unknown tool {params.get('name')!r}"}}
        text, is_error = run_tool(params.get("arguments") or {})
        result = {"content": [{"type": "text", "text": text}], "isError": is_error}
    else:
        return {"jsonrpc": "2.0", "id": mid, "error": {"code": -32601, "message": f"no method {method}"}}
    return {"jsonrpc": "2.0", "id": mid, "result": result}


def _call(arguments: dict) -> tuple[str, bool]:
    return run_delegate(str(arguments.get("task") or ""), arguments.get("cwd"),
                        int(arguments.get("timeout_s") or 900))


def main(stdin=sys.stdin, stdout=sys.stdout) -> int:
    for line in stdin:
        line = line.strip()
        if not line:
            continue
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            reply = {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "parse error"}}
        else:
            reply = handle(message, _call)
        if reply is not None:
            stdout.write(json.dumps(reply) + "\n")
            stdout.flush()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
