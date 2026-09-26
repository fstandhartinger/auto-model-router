"""Point coding harnesses at the local router, with a backup and an undo for every edit.

    python -m auto_router.harness detect [--json]
    python -m auto_router.harness configure --harness claude-code,codex,... [--dry-run]
    python -m auto_router.harness undo [--harness ...] [--dry-run]
    python -m auto_router.harness status

Rules this module keeps, because it edits other programs' settings:

* **No key values.** Entries name the local router URL, the launcher command
  and, where a harness supports it, an environment variable *name*. Nothing
  here reads a credential.
* **Backups.** Before the first edit of a file, a copy is kept next to it as
  ``<name>.auto-router-bak-<timestamp>``. Every change is recorded in
  ``~/.auto-router/install-manifest.json``.
* **Never overwrite.** An existing entry with the same name that differs is
  left alone and reported, unless ``--force`` (the file is backed up first).
* **Only formats we can round-trip.** JSON with comments (JSONC/JSON5) is not
  rewritten: the snippet is saved under ``~/.auto-router/snippets`` and the
  step is reported as manual. TOML and YAML are edited by appending or
  inserting a block between marker comments, then re-parsed to prove the file
  is still valid and says what we meant.
* **Undo.** If a file still has the content we wrote, the backup is restored
  (or the file removed if we created it). If it changed since, only our own
  entries or marker block are removed and the backup is kept.

What is verified: the file formats for Claude Code, Codex, OpenCode, Cursor,
VS Code, Copilot CLI and OpenClaw were taken from their vendors' current docs
(25 Sep 2026); Hermes Agent's from its own documentation (v0.21.0). The
installer's tests run against temporary HOME directories; none of this has
been run against every real client.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

HARNESSES = ("claude-code", "codex", "opencode", "copilot", "cursor", "openclaw", "hermes")
MCP_NAME = "auto-router-delegate"
PROVIDER_ID = "autorouter"
MODEL_ID = "auto"
SKILL = "plan-with-cheap-workers"
BEGIN = "# >>> auto-router (added by the auto-router installer; `auto-router uninstall` removes it)"
END = "# <<< auto-router"
REPO_ROOT = Path(__file__).resolve().parents[1]


class HarnessError(RuntimeError):
    pass


# ------------------------------------------------------------------ context

@dataclass
class Ctx:
    home: Path
    os_name: str                  # linux | macos | windows
    router_url: str = "http://127.0.0.1:8787"
    launcher: str = ""            # the auto-router launcher command (absolute path)
    state_dir: Path | None = None
    env: dict[str, str] = field(default_factory=dict)
    dry_run: bool = False
    force: bool = False
    claude_gateway: bool = False  # opt-in: ANTHROPIC_BASE_URL in Claude Code settings
    project: Path | None = None   # Cursor rule target
    delegate: bool = True         # MCP delegate tool (POSIX only)
    log: list[str] = field(default_factory=list)
    which: Callable[[str], "str | None"] = shutil.which
    run: Callable[..., Any] = subprocess.run

    def __post_init__(self) -> None:
        self.state_dir = self.state_dir or self.home / ".auto-router"
        if not self.launcher:
            name = "auto-router.cmd" if self.os_name == "windows" else "auto-router"
            sub = "bin" if self.os_name == "windows" else ""
            self.launcher = str((self.state_dir / sub / name) if sub else self.home / ".local/bin" / name)

    @classmethod
    def from_env(cls, **kw: Any) -> "Ctx":
        os_name = {"darwin": "macos"}.get(platform.system().lower(), platform.system().lower())
        home = Path(os.environ.get("AUTO_ROUTER_TEST_HOME") or Path.home())
        state = os.environ.get("AUTO_ROUTER_HOME")
        return cls(home=home, os_name=os_name, env=dict(os.environ),
                   state_dir=Path(state) if state else None, **kw)

    @property
    def v1(self) -> str:
        return self.router_url.rstrip("/") + "/v1"

    def say(self, line: str) -> None:
        self.log.append(line)

    # ---------------------------------------------------------- per-OS paths
    def claude_dir(self) -> Path:
        return Path(self.env["CLAUDE_CONFIG_DIR"]) if self.env.get("CLAUDE_CONFIG_DIR") else self.home / ".claude"

    def codex_dir(self) -> Path:
        return Path(self.env["CODEX_HOME"]) if self.env.get("CODEX_HOME") else self.home / ".codex"

    def opencode_file(self) -> Path:
        if self.env.get("OPENCODE_CONFIG"):
            return Path(self.env["OPENCODE_CONFIG"])
        base = self.home / ".config" / "opencode"
        for name in ("opencode.json", "opencode.jsonc"):
            if (base / name).exists():
                return base / name
        return base / "opencode.json"

    def copilot_dir(self) -> Path:
        return Path(self.env["COPILOT_HOME"]) if self.env.get("COPILOT_HOME") else self.home / ".copilot"

    def vscode_user_dir(self) -> Path:
        if self.os_name == "windows":
            appdata = self.env.get("APPDATA") or str(self.home / "AppData" / "Roaming")
            return Path(appdata) / "Code" / "User"
        if self.os_name == "macos":
            return self.home / "Library" / "Application Support" / "Code" / "User"
        return self.home / ".config" / "Code" / "User"

    def openclaw_file(self) -> Path:
        if self.env.get("OPENCLAW_CONFIG_PATH"):
            return Path(self.env["OPENCLAW_CONFIG_PATH"])
        base = Path(self.env["OPENCLAW_STATE_DIR"]) if self.env.get("OPENCLAW_STATE_DIR") else (
            Path(self.env["OPENCLAW_HOME"]) / ".openclaw" if self.env.get("OPENCLAW_HOME") else self.home / ".openclaw")
        return base / "openclaw.json"

    def hermes_dir(self) -> Path:
        return Path(self.env["HERMES_HOME"]) if self.env.get("HERMES_HOME") else self.home / ".hermes"

    def manifest_path(self) -> Path:
        return self.state_dir / "install-manifest.json"

    def snippets(self) -> Path:
        return self.state_dir / "snippets"


# ---------------------------------------------------------------- detection

def detect(ctx: Ctx, which: Callable[[str], str | None] = shutil.which) -> dict[str, dict]:
    """Which harnesses are installed: a binary on PATH or their config directory."""
    def found(bins: list[str], paths: list[Path]) -> dict:
        b = [x for x in bins if which(x)]
        p = [str(x) for x in paths if x.exists()]
        return {"installed": bool(b or p), "binaries": b, "paths": p}

    vs_ext = ctx.home / (".vscode" if ctx.os_name != "windows" else ".vscode") / "extensions"
    copilot_ext = [str(p) for p in vs_ext.glob("github.copilot-chat-*")] if vs_ext.exists() else []
    out = {
        "claude-code": found(["claude"], [ctx.claude_dir(), ctx.home / ".claude.json"]),
        "codex": found(["codex"], [ctx.codex_dir()]),
        "opencode": found(["opencode"], [ctx.opencode_file().parent]),
        "copilot": found(["copilot", "code"], [ctx.copilot_dir()]),
        "cursor": found(["cursor", "cursor-agent"], [ctx.home / ".cursor"]),
        "openclaw": found(["openclaw"], [ctx.openclaw_file().parent]),
        "hermes": found(["hermes"], [ctx.hermes_dir()]),
    }
    out["copilot"]["vscode_copilot_chat_extension"] = copilot_ext
    out["copilot"]["cli"] = bool(which("copilot")) or ctx.copilot_dir().exists()
    out["copilot"]["vscode"] = bool(which("code")) or ctx.vscode_user_dir().exists()
    return out


# ----------------------------------------------------------------- manifest

def _sha(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def load_manifest(ctx: Ctx) -> dict:
    p = ctx.manifest_path()
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {"version": 1, "changes": []}


def save_manifest(ctx: Ctx, manifest: dict) -> None:
    if ctx.dry_run:
        return
    p = ctx.manifest_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    _write_atomic(p, json.dumps(manifest, indent=2) + "\n")


def _missing_dirs(d: Path) -> list[str]:
    """Directories that do not exist yet, deepest first (removed again on undo if empty)."""
    out = []
    while not d.exists() and d != d.parent:
        out.append(str(d))
        d = d.parent
    return out


def _rmdirs(dirs: list[str]) -> None:
    for d in dirs:
        try:
            Path(d).rmdir()
        except OSError:
            break


def _write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.auto-router-tmp")
    tmp.write_text(text, encoding="utf-8")
    if path.exists():
        shutil.copymode(path, tmp)
    os.replace(tmp, path)


class Editor:
    """Records one harness's file edits: backup first, then write, then log the change."""

    def __init__(self, ctx: Ctx, harness: str, manifest: dict):
        self.ctx, self.harness, self.manifest = ctx, harness, manifest
        self._backed: dict[str, str | None] = {}

    def _change(self, path: Path) -> dict:
        for c in self.manifest["changes"]:
            if c["path"] == str(path) and c["harness"] == self.harness and c.get("open"):
                return c
        existed = path.exists()
        backup = None
        if existed and not self.ctx.dry_run:
            backup = path.with_name(f"{path.name}.auto-router-bak-{time.strftime('%Y%m%dT%H%M%S')}")
            n = 0
            while backup.exists():
                n += 1
                backup = path.with_name(f"{path.name}.auto-router-bak-{time.strftime('%Y%m%dT%H%M%S')}-{n}")
            shutil.copy2(path, backup)
        c = {"harness": self.harness, "path": str(path), "created": not existed,
             "created_dirs": _missing_dirs(path.parent),
             "backup": str(backup) if backup else None, "entries": [], "open": True,
             "time": time.strftime("%Y-%m-%dT%H:%M:%S")}
        self.manifest["changes"].append(c)
        return c

    def write(self, path: Path, text: str, entry: dict) -> None:
        if self.ctx.dry_run:
            self.ctx.say(f"[dry-run] would edit {path}: {entry['describe']}")
            return
        c = self._change(path)
        _write_atomic(path, text)
        c["entries"].append({k: v for k, v in entry.items()})
        c["sha_after"] = _sha(path)
        self.ctx.say(f"edited {path}: {entry['describe']}" + (f" (backup: {c['backup']})" if c["backup"] else " (new file)"))

    def close(self) -> None:
        for c in self.manifest["changes"]:
            c.pop("open", None)


# ------------------------------------------------------------- JSON helpers

def _read_json(path: Path) -> dict | None:
    """The object in ``path``; {} if absent; None if it is not plain JSON (comments etc.)."""
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        return {}
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _get(d: dict, keys: list[str]) -> Any:
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def json_set(ed: Editor, path: Path, keys: list[str], value: Any, describe: str,
             snippet_name: str) -> str:
    """Set ``keys`` in the JSON file to ``value`` unless something different is there."""
    ctx = ed.ctx
    data = _read_json(path)
    if data is None:
        return manual(ctx, snippet_name, _nest(keys, value), path,
                      f"{path} is not plain JSON (comments or JSON5?); merge the snippet by hand")
    current = _get(data, keys)
    if current == value:
        ctx.say(f"{path}: {describe} already present")
        return "current"
    if current is not None and not ctx.force:
        raise HarnessError(f"{path} already has a different {'.'.join(keys)}; left alone "
                           f"(rerun with --force to replace it after a backup)")
    node = data
    for k in keys[:-1]:
        if not isinstance(node.get(k), dict):
            if k in node and not ctx.force:
                raise HarnessError(f"{path}: {k!r} is not an object; left alone")
            node[k] = {}
        node = node[k]
    node[keys[-1]] = value
    ed.write(path, json.dumps(data, indent=2) + "\n",
             {"kind": "json", "keys": keys, "describe": describe})
    return "written"


def _nest(keys: list[str], value: Any) -> dict:
    out: Any = value
    for k in reversed(keys):
        out = {k: out}
    return out


def manual(ctx: Ctx, name: str, content: Any, target: Path | str, why: str) -> str:
    """Save a snippet the user merges by hand, and say so."""
    path = ctx.snippets() / name
    text = content if isinstance(content, str) else json.dumps(content, indent=2) + "\n"
    if ctx.dry_run:
        ctx.say(f"[dry-run] would save snippet {path} for {target}: {why}")
    else:
        _write_atomic(path, text)
        ctx.say(f"MANUAL STEP for {target}: {why}. Snippet: {path}")
    return "manual"


# ---------------------------------------------------------- text-block helpers

def _strip_block(text: str) -> str:
    pattern = re.compile(r"\n?[ \t]*" + re.escape(BEGIN) + r".*?" + re.escape(END) + r"[^\n]*\n?", re.S)
    return pattern.sub("\n", text).rstrip("\n") + ("\n" if text.strip() else "")


def _has_block(text: str) -> bool:
    return BEGIN in text


def toml_append(ed: Editor, path: Path, table_header_re: str, block: str, describe: str,
                check: Callable[[dict], bool]) -> str:
    """Append a marker-delimited TOML block unless the table already exists."""
    ctx = ed.ctx
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    if _has_block(text):
        ctx.say(f"{path}: {describe} already present")
        return "current"
    if re.search(table_header_re, text, re.M):
        if not ctx.force:
            raise HarnessError(f"{path} already defines {describe}; left alone (remove it or use --force)")
        raise HarnessError(f"{path} already defines {describe}; --force cannot merge TOML tables "
                           f"safely - remove that table by hand, then rerun")
    new = (text.rstrip("\n") + "\n\n" if text.strip() else "") + f"{BEGIN}\n{block.rstrip()}\n{END}\n"
    _validate_toml(new, check, path)
    ed.write(path, new, {"kind": "block", "describe": describe})
    return "written"


def _validate_toml(text: str, check: Callable[[dict], bool], path: Path) -> None:
    try:
        import tomllib  # Python 3.11+
    except ModuleNotFoundError:  # pragma: no cover - 3.10: skip the re-parse
        return
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise HarnessError(f"{path}: the edited file would not parse ({exc}); nothing written") from exc
    if not check(data):
        raise HarnessError(f"{path}: the edited file does not contain the expected entry; nothing written")


def yaml_insert(ed: Editor, path: Path, top_key: str, entry_name: str, entry_yaml: str,
                describe: str) -> str:
    """Put ``entry_name: ...`` under the top-level mapping ``top_key`` of a YAML file.

    Comments are preserved: the entry goes in as text between marker comments,
    right below ``top_key:`` (or as a new ``top_key:`` block at the end), and
    the result is parsed again to prove it means what we intended.
    """
    ctx = ed.ctx
    try:
        import yaml
    except ModuleNotFoundError:
        if ctx.dry_run:
            ctx.say(f"[dry-run] would add {describe} to {path}")
            return "dry-run"
        raise HarnessError("PyYAML is missing; cannot edit YAML safely")
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    try:
        before = yaml.safe_load(text) or {}
    except yaml.YAMLError as exc:
        raise HarnessError(f"{path} is not valid YAML ({exc}); left alone") from exc
    if not isinstance(before, dict):
        raise HarnessError(f"{path} does not hold a mapping; left alone")
    want = yaml.safe_load(entry_yaml)
    section = before.get(top_key)
    if isinstance(section, dict) and entry_name in section:
        if section[entry_name] == want:
            ctx.say(f"{path}: {describe} already present")
            return "current"
        raise HarnessError(f"{path} already has {top_key}.{entry_name}; left alone")
    if section is not None and not isinstance(section, dict):
        raise HarnessError(f"{path}: {top_key} is not a mapping; left alone")

    lines = text.splitlines()
    idx = next((i for i, l in enumerate(lines) if re.match(rf"^{re.escape(top_key)}:\s*(#.*)?$", l)), None)
    if idx is None and top_key in before:           # e.g. `providers: {}` inline
        raise HarnessError(f"{path}: {top_key} is written inline; add the entry by hand")
    if idx is None:
        body = [BEGIN, f"{top_key}:"] + ["  " + l for l in [f"{entry_name}:"] +
                                         ["  " + x for x in entry_yaml.rstrip().splitlines()]] + [END]
        new = (text.rstrip("\n") + "\n\n" if text.strip() else "") + "\n".join(body) + "\n"
    else:
        indent = "  "
        for l in lines[idx + 1:]:
            if l.strip() and not l.lstrip().startswith("#"):
                m = re.match(r"^([ \t]+)", l)
                if m:
                    indent = m.group(1)
                break
        block = [indent + BEGIN, indent + f"{entry_name}:"] + \
                [indent + "  " + x for x in entry_yaml.rstrip().splitlines()] + [indent + END]
        new = "\n".join(lines[:idx + 1] + block + lines[idx + 1:]) + "\n"
    try:
        after = yaml.safe_load(new) or {}
    except yaml.YAMLError as exc:
        raise HarnessError(f"{path}: inserting the entry would break the YAML ({exc}); nothing written") from exc
    expected = dict(before)
    expected[top_key] = {**(section or {}), entry_name: want}
    if after != expected:
        raise HarnessError(f"{path}: the edited file would not mean what was intended; nothing written")
    ed.write(path, new, {"kind": "block", "describe": describe})
    return "written"


# ------------------------------------------------------------ CLI helpers

def cli_add(ed: Editor, binary: str, add: list[str], get: list[str], remove: list[str],
            snippet: str, target: str) -> str:
    """Register through the harness's own CLI; recorded so undo runs ``remove``."""
    ctx = ed.ctx
    line = " ".join(add) + "\n"
    if not ctx.which(binary):
        return manual(ctx, snippet, line, target, f"`{binary}` is not on PATH; run the command in the snippet")
    if ctx.dry_run:
        ctx.say(f"[dry-run] would run: {line.strip()}")
        return "dry-run"
    # The CLI writes under HOME: run it against the HOME being configured.
    env = {**os.environ, **ctx.env, "HOME": str(ctx.home), "USERPROFILE": str(ctx.home)}
    exists = ctx.run(get, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env).returncode == 0
    if exists:
        ctx.say(f"{binary}: an MCP server named {MCP_NAME} is already registered; left alone "
                f"(inspect with `{' '.join(get)}`)")
        return "current"
    out = ctx.run(add, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, env=env)
    if out.returncode != 0:
        raise HarnessError(f"`{line.strip()}` failed: {(out.stderr or '').strip()[:200]}")
    ed.manifest["changes"].append({"harness": ed.harness, "path": target, "kind": "cli",
                                   "undo": remove, "home": str(ctx.home), "created": True, "backup": None,
                                   "entries": [{"kind": "cli", "describe": line.strip()}],
                                   "time": time.strftime("%Y-%m-%dT%H:%M:%S")})
    ctx.say(f"ran: {line.strip()}")
    return "written"


# ---------------------------------------------------------- per-harness steps

def _mcp_command(ctx: Ctx) -> tuple[str, list[str]]:
    return ctx.launcher, ["delegate"]


def _copy_skill(ed: Editor, target: Path) -> str:
    ctx = ed.ctx
    source = REPO_ROOT / "skills" / SKILL
    if not source.exists():
        ctx.say(f"skill source {source} missing; skipped")
        return "skipped"
    if target.exists():
        same = all((target / p.relative_to(source)).is_file() and
                   (target / p.relative_to(source)).read_bytes() == p.read_bytes()
                   for p in source.rglob("*") if p.is_file())
        if same:
            ctx.say(f"skill already current at {target}")
            return "current"
        if not ctx.force:
            raise HarnessError(f"{target} exists and differs; left alone (use --force)")
    if ctx.dry_run:
        ctx.say(f"[dry-run] would copy skill to {target}")
        return "dry-run"
    backup = None
    created_dirs = _missing_dirs(target.parent)
    if target.exists():
        backup = target.with_name(f"{target.name}.auto-router-bak-{time.strftime('%Y%m%dT%H%M%S')}")
        target.rename(backup)
    shutil.copytree(source, target)
    ed.manifest["changes"].append({"harness": ed.harness, "path": str(target), "created": backup is None,
                                   "created_dirs": created_dirs,
                                   "backup": str(backup) if backup else None, "kind": "dir",
                                   "entries": [{"kind": "dir", "describe": "skill"}],
                                   "time": time.strftime("%Y-%m-%dT%H:%M:%S")})
    ctx.say(f"skill copied to {target}")
    return "written"


def configure_claude(ed: Editor) -> list[str]:
    ctx = ed.ctx
    res = []
    if ctx.delegate:
        cmd, args = _mcp_command(ctx)
        # User-scope MCP servers live in ~/.claude.json, which Claude Code itself
        # rewrites while it runs; `claude mcp add` is the documented way in.
        add = ["claude", "mcp", "add", "--scope", "user", MCP_NAME, "--", cmd, *args]
        res.append(cli_add(ed, "claude", add, ["claude", "mcp", "get", MCP_NAME],
                           ["claude", "mcp", "remove", "--scope", "user", MCP_NAME],
                           "claude-code-mcp.sh", "~/.claude.json (user-scope MCP)"))
        res.append(_copy_skill(ed, ctx.claude_dir() / "skills" / SKILL))
    if ctx.claude_gateway:
        # Opt-in only: the plan's login then passes through the local router
        # (forwarded, never stored). Own login, own machine.
        res.append(json_set(ed, ctx.claude_dir() / "settings.json", ["env", "ANTHROPIC_BASE_URL"],
                            ctx.router_url.rstrip("/"),
                            "env.ANTHROPIC_BASE_URL -> local router (opt-in gateway pass-through)",
                            "claude-settings.json"))
    ctx.say("Claude Code: switch mode needs no file change - start it with `auto-router switch`.")
    return res


def configure_codex(ed: Editor) -> list[str]:
    ctx = ed.ctx
    res = []
    if ctx.delegate:
        cmd, args = _mcp_command(ctx)
        block = (f"[mcp_servers.{MCP_NAME}]\ncommand = {json.dumps(cmd)}\n"
                 f"args = {json.dumps(args)}\n")
        res.append(toml_append(ed, ctx.codex_dir() / "config.toml",
                               rf"^\[mcp_servers\.\"?{re.escape(MCP_NAME)}\"?\]", block,
                               f"[mcp_servers.{MCP_NAME}]",
                               lambda d: _get(d, ["mcp_servers", MCP_NAME, "command"]) == cmd))
        res.append(_copy_skill(ed, ctx.home / ".agents" / "skills" / SKILL))
    ctx.say("Codex: jobs go through `auto-router run` (route-run). The openai_base_url proxy is "
            "not configured: the router serves no Responses API, and the ChatGPT-login proxy path "
            "is documented by OpenAI but unverified here.")
    return res


def configure_opencode(ed: Editor) -> list[str]:
    ctx = ed.ctx
    path = ctx.opencode_file()
    provider = {"npm": "@ai-sdk/openai-compatible", "name": "Auto Router (local)",
                "options": {"baseURL": ctx.v1}, "models": {MODEL_ID: {"name": "auto (router picks)"}}}
    res = [json_set(ed, path, ["provider", PROVIDER_ID], provider,
                    f"provider.{PROVIDER_ID} -> {ctx.v1}", "opencode-provider.json")]
    if ctx.delegate:
        cmd, args = _mcp_command(ctx)
        res.append(json_set(ed, path, ["mcp", MCP_NAME],
                            {"type": "local", "command": [cmd, *args], "enabled": True},
                            f"mcp.{MCP_NAME}", "opencode-mcp.json"))
        res.append(_copy_skill(ed, path.parent / "skill" / SKILL))
    ctx.say(f"OpenCode: pick the model `{PROVIDER_ID}/{MODEL_ID}` (/models) or set \"model\" yourself.")
    return res


def configure_copilot(ed: Editor) -> list[str]:
    ctx = ed.ctx
    res = []
    cmd, args = _mcp_command(ctx)
    if ctx.delegate:
        res.append(json_set(ed, ctx.copilot_dir() / "mcp-config.json", ["mcpServers", MCP_NAME],
                            {"type": "local", "command": cmd, "args": args, "tools": ["*"]},
                            f"Copilot CLI mcpServers.{MCP_NAME}", "copilot-cli-mcp.json"))
        vs = ctx.vscode_user_dir()
        if vs.exists() or (ctx.home / ".vscode").exists():
            res.append(json_set(ed, vs / "mcp.json", ["servers", MCP_NAME],
                                {"type": "stdio", "command": cmd, "args": args},
                                f"VS Code servers.{MCP_NAME}", "vscode-mcp.json"))
    # VS Code BYOK: the Custom Endpoint provider is added through the Language
    # Models editor, which then opens chatLanguageModels.json itself; its path
    # is not documented, so this stays a manual step.
    byok = [{"name": "Auto Router (local)", "vendor": "customendpoint",
             "apiKey": "${input:autoRouterKey}", "apiType": "chat-completions",
             "models": [{"id": MODEL_ID, "name": "auto (router picks)",
                         "url": f"{ctx.v1}/chat/completions", "toolCalling": True,
                         "vision": False, "maxInputTokens": 120000, "maxOutputTokens": 16000}]}]
    res.append(manual(ctx, "vscode-chatLanguageModels.json", byok, "VS Code Copilot Chat",
                      "Chat: Manage Language Models > Add Models > Custom Endpoint, then paste this "
                      "(any placeholder works as the key; the local router checks none)"))
    ctx.say("Copilot CLI: run `auto-router copilot` - it sets COPILOT_PROVIDER_BASE_URL="
            f"{ctx.v1} and COPILOT_MODEL={MODEL_ID} for that process only.")
    return res


def configure_cursor(ed: Editor) -> list[str]:
    ctx = ed.ctx
    res = []
    if ctx.delegate:
        cmd, args = _mcp_command(ctx)
        res.append(json_set(ed, ctx.home / ".cursor" / "mcp.json", ["mcpServers", MCP_NAME],
                            {"command": cmd, "args": args}, f"mcpServers.{MCP_NAME}", "cursor-mcp.json"))
        if ctx.project:
            rule = ctx.project / ".cursor" / "rules" / f"{SKILL}.mdc"
            source = (REPO_ROOT / "integrations" / f"cursor-{SKILL}.mdc").read_text(encoding="utf-8")
            if rule.exists() and rule.read_text(encoding="utf-8") != source and not ctx.force:
                raise HarnessError(f"{rule} exists and differs; left alone")
            if rule.exists() and rule.read_text(encoding="utf-8") == source:
                res.append("current")
            else:
                ed.write(rule, source, {"kind": "file", "describe": "Cursor rule"})
                res.append("written")
    ctx.say("Cursor: 'Override OpenAI Base URL' is not set - Cursor sends those requests from its "
            "own servers, which cannot reach 127.0.0.1. Use the MCP delegate tool instead.")
    return res


def configure_openclaw(ed: Editor) -> list[str]:
    ctx = ed.ctx
    path = ctx.openclaw_file()
    provider = {"baseUrl": ctx.v1, "api": "openai-completions",
                "models": [{"id": MODEL_ID, "name": "auto (router picks)"}]}
    res = [json_set(ed, path, ["models", "providers", PROVIDER_ID], provider,
                    f"models.providers.{PROVIDER_ID}", "openclaw-provider.json5")]
    if ctx.delegate:
        cmd, args = _mcp_command(ctx)
        res.append(json_set(ed, path, ["mcp", "servers", MCP_NAME], {"command": cmd, "args": args},
                            f"mcp.servers.{MCP_NAME}", "openclaw-mcp.json5"))
    ctx.say(f"OpenClaw: select `{PROVIDER_ID}/{MODEL_ID}` as agents.defaults.model.primary to use it "
            "(not changed for you). Format from docs.openclaw.ai; not tested against a real OpenClaw.")
    return res


def configure_hermes(ed: Editor) -> list[str]:
    ctx = ed.ctx
    path = ctx.hermes_dir() / "config.yaml"
    res = [yaml_insert(ed, path, "providers", PROVIDER_ID,
                       f"api: {ctx.v1}\ndiscover_models: false\nmodels:\n  - {MODEL_ID}\n",
                       f"providers.{PROVIDER_ID}")]
    if ctx.delegate:
        cmd, args = _mcp_command(ctx)
        res.append(yaml_insert(ed, path, "mcp_servers", MCP_NAME,
                               f"command: {json.dumps(cmd)}\nargs: {json.dumps(args)}\n",
                               f"mcp_servers.{MCP_NAME}"))
    ctx.say(f"Hermes: switch with `/model {MODEL_ID}` on provider {PROVIDER_ID} (not made the default for you).")
    return res


CONFIGURE: dict[str, Callable[[Editor], list[str]]] = {
    "claude-code": configure_claude, "codex": configure_codex, "opencode": configure_opencode,
    "copilot": configure_copilot, "cursor": configure_cursor, "openclaw": configure_openclaw,
    "hermes": configure_hermes,
}


def configure(ctx: Ctx, harnesses: list[str]) -> dict[str, Any]:
    manifest = load_manifest(ctx)
    results: dict[str, Any] = {}
    for h in harnesses:
        if h not in CONFIGURE:
            raise HarnessError(f"unknown harness {h!r}; choose from {', '.join(HARNESSES)}")
        ed = Editor(ctx, h, manifest)
        try:
            results[h] = CONFIGURE[h](ed)
        except HarnessError as exc:
            results[h] = f"refused: {exc}"
            ctx.say(f"{h}: {exc}")
        finally:
            ed.close()
    save_manifest(ctx, manifest)
    return results


# --------------------------------------------------------------------- undo

def _remove_json(path: Path, keys: list[str]) -> bool:
    data = _read_json(path)
    if not data:
        return False
    node = data
    for k in keys[:-1]:
        node = node.get(k) if isinstance(node, dict) else None
        if node is None:
            return False
    if not isinstance(node, dict) or keys[-1] not in node:
        return False
    del node[keys[-1]]
    _write_atomic(path, json.dumps(data, indent=2) + "\n")
    return True


def undo(ctx: Ctx, harnesses: list[str] | None = None) -> list[str]:
    manifest = load_manifest(ctx)
    keep, done = [], []
    for c in reversed(manifest["changes"]):
        if harnesses and c["harness"] not in harnesses:
            keep.append(c)
            continue
        path = Path(c["path"])
        backup = Path(c["backup"]) if c.get("backup") else None
        if ctx.dry_run:
            done.append(f"[dry-run] would undo {c['harness']}: {path}")
            keep.append(c)
            continue
        if c.get("kind") == "cli":
            home = c.get("home") or str(ctx.home)
            out = ctx.run(c["undo"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                          env={**os.environ, "HOME": home, "USERPROFILE": home})
            done.append(f"ran {' '.join(c['undo'])}" + ("" if out.returncode == 0 else " (it failed; remove the entry by hand)"))
            continue
        if c.get("kind") == "dir":
            if path.exists():
                shutil.rmtree(path)
            if backup and backup.exists():
                backup.rename(path)
            _rmdirs(c.get("created_dirs") or [])
            done.append(f"removed {path}" + (f", restored {backup.name}" if backup else ""))
            continue
        current = _sha(path)
        if current is None and c["created"]:
            done.append(f"{path} already gone")
        elif current == c.get("sha_after"):
            if backup and backup.exists():
                shutil.copy2(backup, path)
                backup.unlink()
                done.append(f"restored {path} from its backup")
            elif c["created"]:
                path.unlink()
                _rmdirs(c.get("created_dirs") or [])
                done.append(f"removed {path} (the installer created it)")
        else:
            # Changed since we wrote it: remove only our own entries.
            removed = False
            text = path.read_text(encoding="utf-8") if path.exists() else ""
            if _has_block(text):
                _write_atomic(path, _strip_block(text))
                removed = True
            for e in c.get("entries", []):
                if e.get("kind") == "json":
                    removed = _remove_json(path, e["keys"]) or removed
            done.append(f"{path} changed since install: removed only the router entries"
                        + ("" if removed else " (none left)")
                        + (f"; backup kept at {backup}" if backup else ""))
    manifest["changes"] = list(reversed(keep))
    save_manifest(ctx, manifest)
    return done


# ---------------------------------------------------------------------- CLI

def _split(value: str | None) -> list[str]:
    if not value:
        return []
    items = [x.strip() for x in value.split(",") if x.strip()]
    return list(HARNESSES) if items == ["all"] else items


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Configure coding harnesses for the local router.")
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("detect")
    d.add_argument("--json", action="store_true")
    for name in ("configure", "undo"):
        s = sub.add_parser(name)
        s.add_argument("--harness", default="" if name == "undo" else "all")
        s.add_argument("--dry-run", action="store_true")
        if name == "configure":
            s.add_argument("--force", action="store_true")
            s.add_argument("--claude-gateway", action="store_true",
                           help="opt-in: set ANTHROPIC_BASE_URL in Claude Code's settings.json "
                                "(your own login passes through the local router; own machine only)")
            s.add_argument("--project", help="Cursor: project directory for the rule")
            s.add_argument("--no-delegate", action="store_true", help="skip the MCP delegate tool")
            s.add_argument("--router-url", default=os.environ.get("AUTO_ROUTER_URL", "http://127.0.0.1:8787"))
            s.add_argument("--launcher", default="")
    sub.add_parser("status")
    args = p.parse_args(argv)

    if args.cmd == "detect":
        ctx = Ctx.from_env()
        found = detect(ctx)
        if args.json:
            print(json.dumps(found, indent=2))
        else:
            for h, info in found.items():
                print(f"{h:12} {'yes' if info['installed'] else 'no':4} {', '.join(info['binaries'] + info['paths'])}")
        return 0
    if args.cmd == "status":
        ctx = Ctx.from_env()
        for c in load_manifest(ctx)["changes"]:
            print(f"{c['harness']:12} {c['path']}  backup={c.get('backup')}")
        return 0
    if args.cmd == "configure":
        ctx = Ctx.from_env(dry_run=args.dry_run, force=args.force, claude_gateway=args.claude_gateway,
                           project=Path(args.project).resolve() if args.project else None,
                           delegate=not args.no_delegate, router_url=args.router_url,
                           launcher=args.launcher)
        results = configure(ctx, _split(args.harness))
        print("\n".join(ctx.log))
        return 1 if any(isinstance(v, str) and v.startswith("refused") for v in results.values()) else 0
    ctx = Ctx.from_env(dry_run=args.dry_run)
    for line in undo(ctx, _split(args.harness) or None):
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
