"""Harness configuration in temporary HOME directories: edits, backups, refusals, undo."""
import json
import subprocess
import tomllib
from pathlib import Path

import pytest
import yaml

from auto_router import harness as H


class FakeRun:
    """Records CLI calls; `mcp get` fails (no entry) unless told otherwise."""

    def __init__(self, existing=False, fail_add=False):
        self.calls, self.existing, self.fail_add = [], existing, fail_add

    def __call__(self, cmd, **kw):
        self.calls.append((cmd, kw.get("env", {}).get("HOME")))
        rc = 0
        if cmd[1:3] == ["mcp", "get"]:
            rc = 0 if self.existing else 1
        if cmd[1:3] == ["mcp", "add"] and self.fail_add:
            rc = 1
        return subprocess.CompletedProcess(cmd, rc, "", "boom" if rc else "")


def make_ctx(tmp_path, os_name="linux", which=lambda b: None, run=None, **kw):
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    return H.Ctx(home=home, os_name=os_name, env={}, launcher="/opt/ar/auto-router",
                 which=which, run=run or FakeRun(), **kw)


def snapshot(root: Path) -> dict:
    return {str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()
            and ".auto-router" not in p.parts}


def test_configure_all_then_undo_restores_everything(tmp_path):
    ctx = make_ctx(tmp_path)
    home = ctx.home
    (home / ".codex").mkdir()
    (home / ".codex/config.toml").write_text('model = "gpt-6-astra"\n\n[mcp_servers.other]\ncommand = "x"\n')
    (home / ".config/opencode").mkdir(parents=True)
    (home / ".config/opencode/opencode.json").write_text(json.dumps({"model": "x/y", "mcp": {"other": {}}}))
    (home / ".hermes").mkdir()
    (home / ".hermes/config.yaml").write_text("# my hermes\nmodel:\n  default: foo\nproviders:\n  mine:\n    api: http://h/v1\n")
    before = snapshot(home)

    results = H.configure(ctx, list(H.HARNESSES))
    assert not any(str(r).startswith("refused") for r in results.values()), results

    toml = tomllib.loads((home / ".codex/config.toml").read_text())
    assert toml["mcp_servers"]["auto-router-delegate"] == {"command": "/opt/ar/auto-router", "args": ["delegate"]}
    assert toml["mcp_servers"]["other"] == {"command": "x"} and toml["model"] == "gpt-6-astra"

    oc = json.loads((home / ".config/opencode/opencode.json").read_text())
    assert oc["provider"]["autorouter"]["options"] == {"baseURL": "http://127.0.0.1:8787/v1"}
    assert oc["mcp"]["auto-router-delegate"]["command"] == ["/opt/ar/auto-router", "delegate"]
    assert oc["model"] == "x/y" and "other" in oc["mcp"]

    hermes_text = (home / ".hermes/config.yaml").read_text()
    hermes = yaml.safe_load(hermes_text)
    assert "# my hermes" in hermes_text
    assert hermes["providers"]["autorouter"]["api"] == "http://127.0.0.1:8787/v1"
    assert hermes["providers"]["mine"] == {"api": "http://h/v1"}
    assert hermes["mcp_servers"]["auto-router-delegate"]["args"] == ["delegate"]
    assert hermes["model"] == {"default": "foo"}

    assert json.loads((home / ".cursor/mcp.json").read_text())["mcpServers"]["auto-router-delegate"]
    assert json.loads((home / ".copilot/mcp-config.json").read_text())["mcpServers"]["auto-router-delegate"]["type"] == "local"
    oclaw = json.loads((home / ".openclaw/openclaw.json").read_text())
    assert oclaw["models"]["providers"]["autorouter"]["api"] == "openai-completions"
    assert (home / ".claude/skills/plan-with-cheap-workers/SKILL.md").exists()
    # Claude's MCP entry needs the `claude` CLI; without it the command is saved as a manual step.
    assert (home / ".auto-router/snippets/claude-code-mcp.sh").read_text().startswith("claude mcp add --scope user")
    assert (home / ".auto-router/snippets/vscode-chatLanguageModels.json").exists()
    # No gateway unless asked for.
    assert not (home / ".claude/settings.json").exists()

    # backups exist for files that existed
    assert list((home / ".codex").glob("config.toml.auto-router-bak-*"))
    assert list((home / ".hermes").glob("config.yaml.auto-router-bak-*"))

    H.undo(ctx)
    after = snapshot(home)
    assert after == before, set(after) ^ set(before)
    assert not (home / ".cursor").exists() and not (home / ".openclaw").exists()


def test_no_key_values_are_written(tmp_path):
    ctx = make_ctx(tmp_path)
    ctx.env = {"OPENROUTER_API_KEY": "sk-secret-value-should-not-appear"}
    H.configure(ctx, list(H.HARNESSES))
    for p in ctx.home.rglob("*"):
        if p.is_file():
            assert "sk-secret" not in p.read_text(errors="replace"), p


def test_existing_different_entry_is_refused_and_file_untouched(tmp_path):
    ctx = make_ctx(tmp_path)
    f = ctx.home / ".cursor/mcp.json"
    f.parent.mkdir(parents=True)
    f.write_text(json.dumps({"mcpServers": {"auto-router-delegate": {"command": "something-else"}}}))
    before = f.read_bytes()
    res = H.configure(ctx, ["cursor"])
    assert res["cursor"].startswith("refused")
    assert f.read_bytes() == before
    assert not list(f.parent.glob("*.auto-router-bak-*"))


def test_force_replaces_after_backup_and_undo_restores(tmp_path):
    ctx = make_ctx(tmp_path, force=True)
    f = ctx.home / ".cursor/mcp.json"
    f.parent.mkdir(parents=True)
    original = json.dumps({"mcpServers": {"auto-router-delegate": {"command": "old"}}})
    f.write_text(original)
    H.configure(ctx, ["cursor"])
    assert json.loads(f.read_text())["mcpServers"]["auto-router-delegate"]["command"] == "/opt/ar/auto-router"
    H.undo(ctx)
    assert f.read_text() == original


def test_jsonc_is_not_rewritten_but_snippet_saved(tmp_path):
    ctx = make_ctx(tmp_path)
    f = ctx.home / ".config/opencode/opencode.jsonc"
    f.parent.mkdir(parents=True)
    f.write_text('{\n  // my comment\n  "model": "a/b"\n}\n')
    before = f.read_bytes()
    res = H.configure(ctx, ["opencode"])
    assert res["opencode"][0] == "manual"
    assert f.read_bytes() == before
    assert (ctx.home / ".auto-router/snippets/opencode-provider.json").exists()


def test_rerun_is_idempotent(tmp_path):
    ctx = make_ctx(tmp_path)
    H.configure(ctx, ["codex", "opencode", "hermes", "cursor"])
    first = snapshot(ctx.home)
    res = H.configure(ctx, ["codex", "opencode", "hermes", "cursor"])
    assert snapshot(ctx.home) == first
    assert res["codex"][0] == "current" and res["opencode"][0] == "current"


def test_undo_after_user_edit_removes_only_our_entries(tmp_path):
    ctx = make_ctx(tmp_path)
    H.configure(ctx, ["opencode", "codex"])
    oc = ctx.home / ".config/opencode/opencode.json"
    data = json.loads(oc.read_text())
    data["theme"] = "dark"
    oc.write_text(json.dumps(data))
    toml = ctx.home / ".codex/config.toml"
    toml.write_text(toml.read_text() + '\nmodel = "mine"\n')
    H.undo(ctx)
    left = json.loads(oc.read_text())
    assert left.get("theme") == "dark"
    assert "autorouter" not in left.get("provider", {}) and "auto-router-delegate" not in left.get("mcp", {})
    assert tomllib.loads(toml.read_text()) == {"model": "mine"}


def test_claude_gateway_is_opt_in_and_reversible(tmp_path):
    ctx = make_ctx(tmp_path, claude_gateway=True)
    s = ctx.home / ".claude/settings.json"
    s.parent.mkdir(parents=True)
    s.write_text(json.dumps({"env": {"FOO": "1"}, "model": "opus"}))
    H.configure(ctx, ["claude-code"])
    data = json.loads(s.read_text())
    assert data["env"] == {"FOO": "1", "ANTHROPIC_BASE_URL": "http://127.0.0.1:8787"}
    assert "ANTHROPIC_API_KEY" not in data["env"] and "ANTHROPIC_AUTH_TOKEN" not in data["env"]
    H.undo(ctx)
    assert json.loads(s.read_text()) == {"env": {"FOO": "1"}, "model": "opus"}


def test_claude_mcp_uses_the_cli_with_the_configured_home_and_undo_removes(tmp_path):
    run = FakeRun()
    ctx = make_ctx(tmp_path, which=lambda b: "/usr/bin/" + b if b == "claude" else None, run=run)
    H.configure(ctx, ["claude-code"])
    add = [c for c in run.calls if c[0][1:3] == ["mcp", "add"]]
    assert add and add[0][0] == ["claude", "mcp", "add", "--scope", "user", "auto-router-delegate", "--",
                                 "/opt/ar/auto-router", "delegate"]
    assert add[0][1] == str(ctx.home)
    H.undo(ctx)
    assert run.calls[-1][0] == ["claude", "mcp", "remove", "--scope", "user", "auto-router-delegate"]
    assert run.calls[-1][1] == str(ctx.home)


def test_claude_existing_mcp_entry_is_left_alone(tmp_path):
    run = FakeRun(existing=True)
    ctx = make_ctx(tmp_path, which=lambda b: "/usr/bin/claude", run=run)
    H.configure(ctx, ["claude-code"])
    assert not [c for c in run.calls if c[0][1:3] == ["mcp", "add"]]


def test_dry_run_writes_nothing(tmp_path):
    ctx = make_ctx(tmp_path, dry_run=True, claude_gateway=True)
    (ctx.home / ".hermes").mkdir()
    (ctx.home / ".hermes/config.yaml").write_text("model: {}\n")
    before = snapshot(ctx.home)
    H.configure(ctx, list(H.HARNESSES))
    assert snapshot(ctx.home) == before
    assert not (ctx.home / ".auto-router").exists()
    assert any(l.startswith("[dry-run]") for l in ctx.log)


def test_hermes_inline_mapping_is_refused(tmp_path):
    ctx = make_ctx(tmp_path)
    f = ctx.home / ".hermes/config.yaml"
    f.parent.mkdir()
    f.write_text("providers: {x: {api: 'http://a'}}\n")
    res = H.configure(ctx, ["hermes"])
    assert res["hermes"].startswith("refused")
    assert f.read_text() == "providers: {x: {api: 'http://a'}}\n"


def test_windows_paths(tmp_path):
    ctx = make_ctx(tmp_path, os_name="windows")
    ctx.env = {"APPDATA": str(ctx.home / "AppData/Roaming")}
    assert ctx.vscode_user_dir() == ctx.home / "AppData/Roaming/Code/User"
    assert ctx.opencode_file() == ctx.home / ".config/opencode/opencode.json"
    mac = make_ctx(tmp_path, os_name="macos")
    assert mac.vscode_user_dir() == mac.home / "Library/Application Support/Code/User"


def test_vscode_mcp_only_when_vscode_present(tmp_path):
    ctx = make_ctx(tmp_path)
    H.configure(ctx, ["copilot"])
    assert not (ctx.home / ".config/Code/User/mcp.json").exists()
    (ctx.home / ".config/Code/User").mkdir(parents=True)
    H.configure(ctx, ["copilot"])
    data = json.loads((ctx.home / ".config/Code/User/mcp.json").read_text())
    assert data["servers"]["auto-router-delegate"] == {"type": "stdio", "command": "/opt/ar/auto-router",
                                                       "args": ["delegate"]}


def test_cursor_rule_only_with_project(tmp_path):
    proj = tmp_path / "proj"
    proj.mkdir()
    ctx = make_ctx(tmp_path, project=proj)
    H.configure(ctx, ["cursor"])
    rule = proj / ".cursor/rules/plan-with-cheap-workers.mdc"
    assert rule.exists()
    H.undo(ctx)
    assert not rule.exists()


def test_detect(tmp_path):
    ctx = make_ctx(tmp_path)
    (ctx.home / ".hermes").mkdir()
    found = H.detect(ctx, which=lambda b: "/bin/codex" if b == "codex" else None)
    assert found["codex"]["installed"] and found["hermes"]["installed"]
    assert not found["openclaw"]["installed"]


def test_no_delegate_skips_mcp(tmp_path):
    ctx = make_ctx(tmp_path, delegate=False)
    H.configure(ctx, ["opencode", "cursor", "codex"])
    oc = json.loads((ctx.home / ".config/opencode/opencode.json").read_text())
    assert "mcp" not in oc and "autorouter" in oc["provider"]
    assert not (ctx.home / ".cursor/mcp.json").exists() and not (ctx.home / ".codex/config.toml").exists()


def test_cli_main_detect(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("AUTO_ROUTER_TEST_HOME", str(tmp_path))
    assert H.main(["detect", "--json"]) == 0
    assert "claude-code" in json.loads(capsys.readouterr().out)
