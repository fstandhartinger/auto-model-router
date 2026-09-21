#!/usr/bin/env python3
"""Install the delegation skill and stdio MCP entry for one supported agent."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SERVER = shutil.which("auto-router-delegate") or str(Path.home() / ".local/bin/auto-router-delegate")
CONFIG = str(Path.home() / ".auto-router/launcher.yaml")


def copy_skill(target: Path) -> None:
    source = ROOT / "skills" / "plan-with-cheap-workers"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(source, target)


def update_json(path: Path, key: str, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(path.read_text()) if path.exists() else {}
    bucket = data.setdefault(key, {})
    bucket["auto-router-delegate"] = value
    path.write_text(json.dumps(data, indent=2) + "\n")


def install(tool: str) -> None:
    home = Path.home()
    if tool == "claude":
        copy_skill(home / ".claude/skills/plan-with-cheap-workers")
        subprocess.run(["claude", "mcp", "remove", "auto-router-delegate", "--scope", "user"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run(["claude", "mcp", "add", "--scope", "user", "--env",
                        f"AUTO_ROUTER_CONFIG={CONFIG}", "auto-router-delegate", "--", SERVER],
                       check=True)
    elif tool == "codex":
        copy_skill(home / ".agents/skills/plan-with-cheap-workers")
        subprocess.run(["codex", "mcp", "remove", "auto-router-delegate"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run(["codex", "mcp", "add", "--env", f"AUTO_ROUTER_CONFIG={CONFIG}",
                        "auto-router-delegate", "--", SERVER], check=True)
    elif tool == "opencode":
        copy_skill(home / ".config/opencode/skill/plan-with-cheap-workers")
        update_json(home / ".config/opencode/opencode.json", "mcp", {
            "type": "local", "command": [SERVER],
            "environment": {"AUTO_ROUTER_CONFIG": CONFIG}, "enabled": True})
    elif tool == "cursor":
        rules = Path.cwd() / ".cursor/rules"
        rules.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / "integrations/cursor-plan-with-cheap-workers.mdc",
                     rules / "plan-with-cheap-workers.mdc")
        update_json(home / ".cursor/mcp.json", "mcpServers", {
            "command": SERVER, "args": [], "env": {"AUTO_ROUTER_CONFIG": CONFIG}})
    else:
        raise SystemExit(f"unsupported tool: {tool}")
    print(f"Installed auto-router delegation for {tool}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("tool", choices=("claude", "codex", "opencode", "cursor"))
    install(parser.parse_args().tool)
