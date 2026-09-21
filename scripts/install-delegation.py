#!/usr/bin/env python3
"""Install the delegation skill and stdio MCP entry for one supported agent.

Conservative by design, because it edits another program's global settings:

- It never overwrites. An existing MCP entry, skill directory, Cursor rule or
  command link that differs from what would be written is left alone and
  reported; ``--force`` replaces it after moving the old one aside to a
  timestamped ``.bak-...`` copy.
- JSON settings are written atomically (a temporary file, then a rename), and
  a file that is not plain JSON (for example JSONC with comments) is refused,
  not rewritten.
- ``AUTO_ROUTER_CONFIG`` is written into the MCP entry only when you name a
  configuration (``--config``, or the variable already set in your shell) and
  that file exists. Otherwise the entry carries no configuration path and the
  server reads the variable from the environment it is started in.
- The Cursor rule is project-local, so it is written only with ``--project``.
- It reads and writes no credential.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAME = "auto-router-delegate"
SKILL = "plan-with-cheap-workers"


class InstallError(RuntimeError):
    pass


def server_path() -> str:
    return shutil.which(NAME) or str(Path.home() / ".local/bin" / NAME)


def resolve_config(value: str | None) -> str | None:
    """The configuration path to record, or None to record none.

    A named file must exist: an MCP entry pointing at a missing file makes
    every delegate call fail, and it would override the variable you set.
    """
    value = value or os.environ.get("AUTO_ROUTER_CONFIG")
    if not value:
        return None
    path = Path(os.path.expanduser(value)).resolve()
    if not path.is_file():
        raise InstallError(f"configuration {str(path)!r} does not exist; create it first "
                           f"(see examples/launcher.example.yaml) or leave --config out")
    return str(path)


def _backup(path: Path) -> Path:
    target = path.with_name(f"{path.name}.bak-{time.strftime('%Y%m%dT%H%M%S')}")
    n = 0
    while target.exists() or target.is_symlink():
        n += 1
        target = path.with_name(f"{path.name}.bak-{time.strftime('%Y%m%dT%H%M%S')}-{n}")
    path.rename(target)
    return target


def _same_tree(a: Path, b: Path) -> bool:
    files_a = sorted(p.relative_to(a) for p in a.rglob("*") if p.is_file())
    files_b = sorted(p.relative_to(b) for p in b.rglob("*") if p.is_file())
    return files_a == files_b and all((a / r).read_bytes() == (b / r).read_bytes() for r in files_a)


def _copy_plain(source: Path, target: Path) -> None:
    """Copy contents only, with default modes (the source may be read-only)."""
    target.mkdir()
    for path in sorted(source.rglob("*")):
        dest = target / path.relative_to(source)
        if path.is_dir():
            dest.mkdir()
        else:
            shutil.copyfile(path, dest)


def copy_skill(target: Path, *, force: bool = False) -> str:
    """Copy the skill; never deletes a different existing copy without --force."""
    source = ROOT / "skills" / SKILL
    if target.is_symlink() or target.exists():
        if target.is_dir() and not target.is_symlink() and _same_tree(source, target):
            return f"skill already current at {target}"
        if not force:
            raise InstallError(f"{target} exists and differs; rerun with --force to replace it "
                               f"(the old copy is kept as a .bak- directory)")
        moved = _backup(target)
        note = f" (previous copy moved to {moved})"
    else:
        note = ""
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{SKILL}-", dir=target.parent))
    try:
        _copy_plain(source, staging / "skill")
        (staging / "skill").rename(target)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return f"skill installed at {target}{note}"


def _write_atomic(path: Path, text: str) -> None:
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write(text)
        if path.exists():
            shutil.copymode(path, tmp)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def update_json(path: Path, key: str, value: dict, *, force: bool = False) -> str:
    """Add this server under ``key`` without touching any other entry."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            raise InstallError(f"{path} is not plain JSON ({exc}); add the entry by hand") from exc
        if not isinstance(data, dict):
            raise InstallError(f"{path} does not hold a JSON object; add the entry by hand")
    else:
        data = {}
    bucket = data.setdefault(key, {})
    if not isinstance(bucket, dict):
        raise InstallError(f"{path}: {key!r} is not an object; add the entry by hand")
    current = bucket.get(NAME)
    if current == value:
        return f"{path}: entry already current"
    note = ""
    if current is not None:
        if not force:
            raise InstallError(f"{path} already has a different {NAME!r} entry; rerun with "
                               f"--force to replace it (the file is backed up first)")
        backup = path.with_name(f"{path.name}.bak-{time.strftime('%Y%m%dT%H%M%S')}")
        shutil.copy2(path, backup)
        note = f" (previous file saved as {backup})"
    bucket[NAME] = value
    _write_atomic(path, json.dumps(data, indent=2) + "\n")
    return f"{path}: entry written{note}"


def _cli_entry(cli: str, config: str | None, server: str, *, force: bool,
               run=subprocess.run) -> str:
    """Register through the agent's own CLI; an existing entry is kept unless --force."""
    exists = run([cli, "mcp", "get", NAME], stdout=subprocess.DEVNULL,
                 stderr=subprocess.DEVNULL).returncode == 0
    if exists and not force:
        raise InstallError(f"{cli} already has an MCP server named {NAME!r}; inspect it with "
                           f"`{cli} mcp get {NAME}` and rerun with --force to replace it")
    if exists:
        remove = [cli, "mcp", "remove", NAME] + (["--scope", "user"] if cli == "claude" else [])
        run(remove, check=True)
    env = ["--env", f"AUTO_ROUTER_CONFIG={config}"] if config else []
    scope = ["--scope", "user"] if cli == "claude" else []
    run([cli, "mcp", "add", *scope, *env, NAME, "--", server], check=True)
    return f"{cli}: MCP entry {'replaced' if exists else 'added'}"


def install(tool: str, *, config: str | None = None, project: str | None = None,
            force: bool = False, server: str | None = None, run=subprocess.run) -> list[str]:
    home = Path.home()
    server = server or server_path()
    config = resolve_config(config)
    done: list[str] = []
    if tool == "claude":
        done.append(copy_skill(home / ".claude/skills" / SKILL, force=force))
        done.append(_cli_entry("claude", config, server, force=force, run=run))
    elif tool == "codex":
        done.append(copy_skill(home / ".agents/skills" / SKILL, force=force))
        done.append(_cli_entry("codex", config, server, force=force, run=run))
    elif tool == "opencode":
        done.append(copy_skill(home / ".config/opencode/skill" / SKILL, force=force))
        entry = {"type": "local", "command": [server], "enabled": True}
        if config:
            entry["environment"] = {"AUTO_ROUTER_CONFIG": config}
        done.append(update_json(home / ".config/opencode/opencode.json", "mcp", entry, force=force))
    elif tool == "cursor":
        entry = {"command": server, "args": []}
        if config:
            entry["env"] = {"AUTO_ROUTER_CONFIG": config}
        done.append(update_json(home / ".cursor/mcp.json", "mcpServers", entry, force=force))
        if project:
            rules = Path(project).resolve() / ".cursor/rules"
            rule = rules / f"{SKILL}.mdc"
            source = ROOT / "integrations" / f"cursor-{SKILL}.mdc"
            if rule.exists() and rule.read_bytes() != source.read_bytes() and not force:
                raise InstallError(f"{rule} exists and differs; rerun with --force to replace it")
            rules.mkdir(parents=True, exist_ok=True)
            if rule.exists() and rule.read_bytes() != source.read_bytes():
                done.append(f"previous rule moved to {_backup(rule)}")
            _write_atomic(rule, source.read_text())
            done.append(f"cursor rule written to {rule}")
        else:
            done.append("cursor rule not written (it is per project: pass --project DIR)")
    else:
        raise InstallError(f"unsupported tool: {tool}")
    if not config:
        done.append("no AUTO_ROUTER_CONFIG recorded: the server reads it from its environment")
    return done


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("tool", choices=("claude", "codex", "opencode", "cursor"))
    parser.add_argument("--config", help="launcher configuration to record in the MCP entry "
                                         "(default: $AUTO_ROUTER_CONFIG if set; must exist)")
    parser.add_argument("--project", help="Cursor only: project directory for the rule file")
    parser.add_argument("--server", help="path of the auto-router-delegate command to register "
                                         "(default: the one on PATH, else ~/.local/bin)")
    parser.add_argument("--force", action="store_true",
                        help="replace differing existing entries after backing them up")
    args = parser.parse_args(argv)
    try:
        for line in install(args.tool, config=args.config, project=args.project,
                            force=args.force, server=args.server):
            print(line)
    except (InstallError, subprocess.CalledProcessError, FileNotFoundError) as exc:
        print(f"install-delegation: {exc}", file=sys.stderr)
        return 1
    print(f"Installed auto-router delegation for {args.tool}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
