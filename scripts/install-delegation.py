#!/usr/bin/env python3
"""Install the delegation skill and stdio MCP entry for one supported agent.

Agents: claude, codex, opencode, cursor, openclaw, hermes, copilot.

Conservative by design, because it edits another program's global settings:

- It never overwrites. An existing MCP entry, skill directory, Cursor rule or
  command link that differs from what would be written is left alone and
  reported; ``--force`` replaces it after moving the old one aside to a
  timestamped ``.bak-...`` copy.
- JSON settings are written atomically (a temporary file, then a rename), and
  a file that is not plain JSON (for example JSONC or JSON5 with comments) is
  refused, not rewritten; the refusal prints the entry to paste by hand.
- Hermes keeps its settings in YAML. The entry is inserted as text below
  ``mcp_servers:`` (or appended as a new block), so comments and layout stay
  as they were; the result is parsed again and must equal the old settings
  plus this one entry, and the old file is backed up first. Anything else
  (an inline ``mcp_servers: {}``, a differing entry even with ``--force``) is
  refused with the snippet to paste.
- ``AUTO_ROUTER_CONFIG`` is written into the MCP entry only when you name a
  configuration (``--config``, or the variable already set in your shell) and
  that file exists. Otherwise the entry carries no configuration path and the
  server reads the variable from the environment it is started in.
- Every refusal is decided before anything is written, so a refused install
  leaves the skill, the settings, the MCP entry and the rule as they were.
- The Cursor rule is project-local, so it is written only with ``--project``.
  So are Copilot's ``.vscode/mcp.json`` and ``.github/copilot-instructions.md``;
  the latter is only ever created, never replaced, not even with ``--force``.
- It reads and writes no credential.
"""

from __future__ import annotations

import argparse
import json
import os
import re
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


def check_skill(target: Path, *, force: bool = False) -> bool:
    """True if the skill at ``target`` is current; raises if copy_skill would refuse."""
    if not (target.is_symlink() or target.exists()):
        return False
    if target.is_dir() and not target.is_symlink() and _same_tree(ROOT / "skills" / SKILL, target):
        return True
    if not force:
        raise InstallError(f"{target} exists and differs; rerun with --force to replace it "
                           f"(the old copy is kept as a .bak- directory)")
    return False


def copy_skill(target: Path, *, force: bool = False) -> str:
    """Copy the skill; never deletes a different existing copy without --force."""
    source = ROOT / "skills" / SKILL
    if check_skill(target, force=force):
        return f"skill already current at {target}"
    if target.is_symlink() or target.exists():
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


def _nest(keys: tuple[str, ...], value) -> dict:
    for k in reversed(keys):
        value = {k: value}
    return value


def update_json(path: Path, key: str | tuple[str, ...], value: dict, *, force: bool = False,
                write: bool = True) -> str:
    """Add this server under ``key`` (a name or a path of names) without touching any other entry.

    With ``write=False`` it only raises what a write would refuse.
    """
    keys = (key,) if isinstance(key, str) else tuple(key)
    where = ".".join(keys)
    snippet = json.dumps(_nest(keys + (NAME,), value), indent=2)
    if path.exists():
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            raise InstallError(f"{path} is not plain JSON ({exc}); add the entry by hand:\n"
                               f"{snippet}") from exc
        if not isinstance(data, dict):
            raise InstallError(f"{path} does not hold a JSON object; add the entry by hand")
    else:
        data = {}
    bucket = data
    for k in keys:
        bucket = bucket.setdefault(k, {})
        if not isinstance(bucket, dict):
            raise InstallError(f"{path}: {where!r} is not an object; add the entry by hand")
    current = bucket.get(NAME)
    if current == value:
        return f"{path}: entry already current"
    if current is not None and not force:
        raise InstallError(f"{path} already has a different {NAME!r} entry; rerun with "
                           f"--force to replace it (the file is backed up first)")
    if not write:
        return f"{path}: entry can be written"
    path.parent.mkdir(parents=True, exist_ok=True)
    note = ""
    if current is not None:
        backup = path.with_name(f"{path.name}.bak-{time.strftime('%Y%m%dT%H%M%S')}")
        shutil.copy2(path, backup)
        note = f" (previous file saved as {backup})"
    bucket[NAME] = value
    _write_atomic(path, json.dumps(data, indent=2) + "\n")
    return f"{path}: entry written{note}"


def _yaml_entry(server: str, config: str | None) -> str:
    """The Hermes entry as YAML text; strings are JSON-quoted, which YAML reads as-is."""
    lines = [f"{NAME}:", f"  command: {json.dumps(server)}", "  args: []"]
    if config:
        lines += ["  env:", f"    AUTO_ROUTER_CONFIG: {json.dumps(config)}"]
    return "\n".join(lines) + "\n"


def update_yaml(path: Path, entry: str, *, write: bool = True) -> str:
    """Add ``entry`` (one ``NAME: ...`` mapping) under the top-level ``mcp_servers``.

    Format-preserving: the text goes in right below a block-style
    ``mcp_servers:`` line, indented like its first child, or is appended as a
    new ``mcp_servers:`` block. The result must parse to the old settings plus
    exactly this entry. There is no ``--force``: replacing an entry in YAML
    without a round-tripping parser could lose comments, so a differing entry
    is left for you to edit.
    """
    import yaml

    snippet = "mcp_servers:\n" + "".join("  " + line + "\n" for line in entry.splitlines())
    by_hand = f"; add the entry by hand:\n{snippet}"
    want = yaml.safe_load(entry)[NAME]
    text = path.read_text() if path.exists() else ""
    try:
        before = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise InstallError(f"{path} is not valid YAML ({exc})") from exc
    if before is None:  # empty, or only comments
        before = {}
    if not isinstance(before, dict):
        raise InstallError(f"{path} does not hold a YAML mapping{by_hand}")
    section = before.get("mcp_servers")
    if section is not None and not isinstance(section, dict):
        raise InstallError(f"{path}: 'mcp_servers' is not a mapping{by_hand}")
    if section and NAME in section:
        if section[NAME] == want:
            return f"{path}: entry already current"
        raise InstallError(f"{path} already has a different mcp_servers.{NAME} entry; edit it by "
                           f"hand (YAML is not rewritten, even with --force). Wanted:\n{snippet}")
    lines = text.splitlines()
    idx = next((i for i, line in enumerate(lines)
                if re.match(r"^mcp_servers:[ \t]*(#.*)?$", line)), None)
    if idx is None and "mcp_servers" in before:
        raise InstallError(f"{path}: 'mcp_servers' is not a plain block mapping{by_hand}")
    if idx is None:
        new = (text.rstrip("\n") + "\n\n" if text.strip() else "") + snippet
    else:
        indent = "  "
        for line in lines[idx + 1:]:
            if line.strip() and not line.lstrip().startswith("#"):
                indent = re.match(r"^([ \t]*)", line).group(1) or "  "
                break
        block = [indent + line for line in entry.splitlines()]
        new = "\n".join(lines[:idx + 1] + block + lines[idx + 1:]) + "\n"
    try:
        after = yaml.safe_load(new)
    except yaml.YAMLError:
        after = None
    if after != {**before, "mcp_servers": {**(section or {}), NAME: want}}:
        raise InstallError(f"{path}: the entry cannot be inserted without changing other "
                           f"settings{by_hand}")
    if not write:
        return f"{path}: entry can be written"
    note = " (new file)"
    if path.exists():
        backup = path.with_name(f"{path.name}.bak-{time.strftime('%Y%m%dT%H%M%S')}")
        shutil.copy2(path, backup)
        note = f" (previous file saved as {backup})"
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_atomic(path, new)
    return f"{path}: entry written{note}"


def copilot_instructions() -> str:
    """The Cursor rule's text, without its Cursor-only front matter."""
    rule = (ROOT / "integrations" / f"cursor-{SKILL}.mdc").read_text()
    body = rule.split("---", 2)[2].strip() if rule.startswith("---") else rule.strip()
    return f"# Delegating to cheap workers (auto-router)\n\n{body}\n"


def check_cli_entry(cli: str, *, force: bool, run=subprocess.run) -> bool:
    """Whether the agent's CLI has an entry already; raises if it may not be replaced."""
    exists = run([cli, "mcp", "get", NAME], stdout=subprocess.DEVNULL,
                 stderr=subprocess.DEVNULL).returncode == 0
    if exists and not force:
        raise InstallError(f"{cli} already has an MCP server named {NAME!r}; inspect it with "
                           f"`{cli} mcp get {NAME}` and rerun with --force to replace it")
    return exists


def _cli_entry(cli: str, config: str | None, server: str, *, exists: bool,
               run=subprocess.run) -> str:
    """Register through the agent's own CLI, replacing the entry checked to exist."""
    if exists:
        remove = [cli, "mcp", "remove", NAME] + (["--scope", "user"] if cli == "claude" else [])
        run(remove, check=True)
    env = ["--env", f"AUTO_ROUTER_CONFIG={config}"] if config else []
    scope = ["--scope", "user"] if cli == "claude" else []
    run([cli, "mcp", "add", *scope, *env, NAME, "--", server], check=True)
    return f"{cli}: MCP entry {'replaced' if exists else 'added'}"


def _state_dir(var: str, default: str) -> Path:
    value = os.environ.get(var)
    return Path(os.path.expanduser(value)) if value else Path.home() / default


def openclaw_paths() -> tuple[Path, Path]:
    """OpenClaw's settings file and managed skills directory (docs.openclaw.ai)."""
    state = _state_dir("OPENCLAW_STATE_DIR", ".openclaw")
    config = os.environ.get("OPENCLAW_CONFIG_PATH")
    return (Path(os.path.expanduser(config)) if config else state / "openclaw.json"), state / "skills"


def install(tool: str, *, config: str | None = None, project: str | None = None,
            force: bool = False, server: str | None = None, run=subprocess.run) -> list[str]:
    home = Path.home()
    server = server or server_path()
    config = resolve_config(config)
    done: list[str] = []
    # Each branch checks every refusal first, then writes.
    if tool in ("claude", "codex"):
        skill = home / (".claude/skills" if tool == "claude" else ".agents/skills") / SKILL
        check_skill(skill, force=force)
        exists = check_cli_entry(tool, force=force, run=run)
        done.append(copy_skill(skill, force=force))
        done.append(_cli_entry(tool, config, server, exists=exists, run=run))
    elif tool == "opencode":
        skill = home / ".config/opencode/skill" / SKILL
        settings = home / ".config/opencode/opencode.json"
        entry = {"type": "local", "command": [server], "enabled": True}
        if config:
            entry["environment"] = {"AUTO_ROUTER_CONFIG": config}
        check_skill(skill, force=force)
        update_json(settings, "mcp", entry, force=force, write=False)
        done.append(copy_skill(skill, force=force))
        done.append(update_json(settings, "mcp", entry, force=force))
    elif tool == "cursor":
        settings = home / ".cursor/mcp.json"
        entry = {"command": server, "args": []}
        if config:
            entry["env"] = {"AUTO_ROUTER_CONFIG": config}
        update_json(settings, "mcpServers", entry, force=force, write=False)
        if project:
            rules = Path(project).resolve() / ".cursor/rules"
            rule = rules / f"{SKILL}.mdc"
            source = ROOT / "integrations" / f"cursor-{SKILL}.mdc"
            if rule.exists() and rule.read_bytes() != source.read_bytes() and not force:
                raise InstallError(f"{rule} exists and differs; rerun with --force to replace it")
        done.append(update_json(settings, "mcpServers", entry, force=force))
        if project:
            rules.mkdir(parents=True, exist_ok=True)
            if rule.exists() and rule.read_bytes() != source.read_bytes():
                done.append(f"previous rule moved to {_backup(rule)}")
            _write_atomic(rule, source.read_text())
            done.append(f"cursor rule written to {rule}")
        else:
            done.append("cursor rule not written (it is per project: pass --project DIR)")
    elif tool == "openclaw":
        # ``mcp.servers.<name>`` with command/args/env and ``<state-dir>/skills/<name>/SKILL.md``,
        # from docs.openclaw.ai (gateway/config-extensions, cli/mcp/registry, tools/skills),
        # checked 26 Sep 2026. Not run against a real OpenClaw. The file is JSON5: one
        # with comments is refused and the entry printed (or use `openclaw mcp set`).
        settings, skills = openclaw_paths()
        skill = skills / SKILL
        entry = {"command": server, "args": []}
        if config:
            entry["env"] = {"AUTO_ROUTER_CONFIG": config}
        check_skill(skill, force=force)
        try:
            update_json(settings, ("mcp", "servers"), entry, force=force, write=False)
        except InstallError as exc:
            if "not plain JSON" not in str(exc):
                raise
            raise InstallError(f"{exc}\nor run: openclaw mcp set {NAME} "
                               f"'{json.dumps(entry)}'") from exc
        done.append(copy_skill(skill, force=force))
        done.append(update_json(settings, ("mcp", "servers"), entry, force=force))
        done.append("openclaw: restart the OpenClaw gateway so it loads the server")
    elif tool == "hermes":
        # ``mcp_servers.<name>`` (command/args/env) in config.yaml and
        # ``skills/<name>/SKILL.md`` under HERMES_HOME, from Hermes Agent's docs
        # (user-guide/features/mcp.md and skills.md, hermes-agent commit b20cc5f,
        # 1 Sep 2026). Not run against a real Hermes.
        base = _state_dir("HERMES_HOME", ".hermes")
        skill = base / "skills" / SKILL
        settings = base / "config.yaml"
        entry = _yaml_entry(server, config)
        check_skill(skill, force=force)
        update_yaml(settings, entry, write=False)
        done.append(copy_skill(skill, force=force))
        done.append(update_yaml(settings, entry))
        # Hermes starts stdio servers with PATH, HOME, USER, LANG and a few more
        # plus the entry's ``env`` (tools/mcp_tool.py, _build_safe_env).
        done.append("hermes: the server sees only PATH, HOME and a few other variables plus the "
                    "entry's env" + ("" if config else ", so pass --config: AUTO_ROUTER_CONFIG "
                    "from your shell does not reach it") + "; API-key variables your routes name "
                    "must be added under env by you (this installer writes no credential)")
    elif tool == "copilot":
        # Copilot CLI: ``mcpServers`` in ~/.copilot/mcp-config.json, type "local" with
        # ``tools`` (docs.github.com, "Adding MCP servers for GitHub Copilot CLI").
        # VS Code: ``servers`` in .vscode/mcp.json, type "stdio"
        # (code.visualstudio.com/docs/copilot/reference/mcp-configuration). Both
        # checked 26 Sep 2026; COPILOT_HOME is honoured as the CLI's config directory.
        env = {"AUTO_ROUTER_CONFIG": config} if config else None
        if project:
            base = Path(project).resolve()
            settings, key = base / ".vscode/mcp.json", "servers"
            entry = {"type": "stdio", "command": server, "args": []}
        else:
            settings, key = _state_dir("COPILOT_HOME", ".copilot") / "mcp-config.json", "mcpServers"
            entry = {"type": "local", "command": server, "args": [], "tools": ["*"]}
        if env:
            entry["env"] = env
        update_json(settings, key, entry, force=force, write=False)
        done.append(update_json(settings, key, entry, force=force))
        if project:
            notes = base / ".github/copilot-instructions.md"
            text = copilot_instructions()
            if not notes.exists():
                notes.parent.mkdir(parents=True, exist_ok=True)
                _write_atomic(notes, text)
                done.append(f"copilot instructions written to {notes}")
            elif notes.read_text() == text:
                done.append(f"copilot instructions already current at {notes}")
            else:
                done.append(f"{notes} exists and is never replaced; add this yourself if you "
                            f"want it:\n{text}")
        else:
            done.append("VS Code and copilot-instructions.md not written (they are per project: "
                        "pass --project DIR)")
    else:
        raise InstallError(f"unsupported tool: {tool}")
    if not config and tool != "hermes":
        done.append("no AUTO_ROUTER_CONFIG recorded: the server reads it from its environment")
    return done


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("tool", choices=("claude", "codex", "opencode", "cursor",
                                         "openclaw", "hermes", "copilot"))
    parser.add_argument("--config", help="launcher configuration to record in the MCP entry "
                                         "(default: $AUTO_ROUTER_CONFIG if set; must exist)")
    parser.add_argument("--project", help="Cursor: project directory for the rule file; Copilot: "
                                          "write .vscode/mcp.json and .github/copilot-instructions.md "
                                          "there instead of the CLI's user config")
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
