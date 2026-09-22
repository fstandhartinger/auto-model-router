"""The delegation installer, exercised only against a disposable fake HOME.

Nothing here runs a real agent CLI, touches the network or the real home
directory: HOME points into ``tmp_path`` and the ``claude``/``codex`` calls go
to a recorder.
"""

import importlib.util
import json
import os
import subprocess
import types
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts" / "install-delegation.py"
SHELL = Path(__file__).parents[1] / "scripts" / "install-delegation.sh"
SPEC = importlib.util.spec_from_file_location("install_delegation", SCRIPT)
install_delegation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(install_delegation)
InstallError = install_delegation.InstallError


@pytest.fixture
def home(tmp_path, monkeypatch):
    fake = tmp_path / "home"
    fake.mkdir()
    monkeypatch.setenv("HOME", str(fake))
    monkeypatch.delenv("AUTO_ROUTER_CONFIG", raising=False)
    return fake


class Recorder:
    """Stands in for ``subprocess.run`` of the agent CLIs."""

    def __init__(self, existing=False):
        self.calls, self.existing = [], existing

    def __call__(self, argv, **kwargs):
        self.calls.append(list(argv))
        code = 0 if (argv[2] != "get" or self.existing) else 1
        if kwargs.get("check") and code:
            raise subprocess.CalledProcessError(code, argv)
        return types.SimpleNamespace(returncode=code)


def test_json_config_update_preserves_existing_servers(tmp_path):
    path = tmp_path / "mcp.json"
    path.write_text(json.dumps({"mcpServers": {"existing": {"command": "x"}}, "other": 1}))
    install_delegation.update_json(path, "mcpServers", {"command": "delegate", "args": []})
    data = json.loads(path.read_text())
    assert data["other"] == 1 and data["mcpServers"]["existing"]["command"] == "x"
    assert data["mcpServers"]["auto-router-delegate"]["command"] == "delegate"


def test_a_differing_entry_is_kept_unless_forced_and_then_backed_up(tmp_path):
    path = tmp_path / "mcp.json"
    original = {"mcpServers": {"auto-router-delegate": {"command": "mine"}}}
    path.write_text(json.dumps(original))
    with pytest.raises(InstallError, match="--force"):
        install_delegation.update_json(path, "mcpServers", {"command": "new"})
    assert json.loads(path.read_text()) == original
    install_delegation.update_json(path, "mcpServers", {"command": "new"}, force=True)
    assert json.loads(path.read_text())["mcpServers"]["auto-router-delegate"] == {"command": "new"}
    backups = list(tmp_path.glob("mcp.json.bak-*"))
    assert len(backups) == 1 and json.loads(backups[0].read_text()) == original


def test_jsonc_and_odd_shapes_are_refused_without_writing(tmp_path):
    path = tmp_path / "opencode.json"
    jsonc = '{\n  // my comment\n  "mcp": {}\n}\n'
    path.write_text(jsonc)
    with pytest.raises(InstallError, match="not plain JSON"):
        install_delegation.update_json(path, "mcp", {"type": "local"})
    assert path.read_text() == jsonc
    path.write_text('{"mcp": []}')
    with pytest.raises(InstallError, match="not an object"):
        install_delegation.update_json(path, "mcp", {"type": "local"})
    path.write_text("[]")
    with pytest.raises(InstallError, match="JSON object"):
        install_delegation.update_json(path, "mcp", {"type": "local"})


def test_json_write_is_atomic(tmp_path, monkeypatch):
    path = tmp_path / "mcp.json"
    path.write_text('{"keep": 1}')

    def fail(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(install_delegation.os, "replace", fail)
    with pytest.raises(OSError):
        install_delegation.update_json(path, "mcpServers", {"command": "x"})
    assert path.read_text() == '{"keep": 1}'
    assert [p.name for p in tmp_path.iterdir()] == ["mcp.json"]


def test_copy_skill_never_deletes_a_different_copy_without_force(tmp_path):
    target = tmp_path / "skill"
    target.mkdir()
    (target / "mine").write_text("my edits")
    with pytest.raises(InstallError, match="--force"):
        install_delegation.copy_skill(target)
    assert (target / "mine").read_text() == "my edits"
    install_delegation.copy_skill(target, force=True)
    assert (target / "SKILL.md").exists() and not (target / "mine").exists()
    backups = list(tmp_path.glob("skill.bak-*"))
    assert len(backups) == 1 and (backups[0] / "mine").read_text() == "my edits"
    assert "already current" in install_delegation.copy_skill(target)


def test_config_is_recorded_only_when_named_and_existing(home, tmp_path, monkeypatch):
    assert install_delegation.resolve_config(None) is None
    with pytest.raises(InstallError, match="does not exist"):
        install_delegation.resolve_config(str(tmp_path / "missing.yaml"))
    cfg = tmp_path / "mine.yaml"
    cfg.write_text("models: []\n")
    monkeypatch.setenv("AUTO_ROUTER_CONFIG", str(cfg))
    assert install_delegation.resolve_config(None) == str(cfg.resolve())


def test_claude_install_without_config_writes_no_config_path(home):
    run = Recorder()
    install_delegation.install("claude", server="/opt/delegate", run=run)
    add = run.calls[-1]
    assert add[:3] == ["claude", "mcp", "add"] and "--env" not in add
    assert add[-2:] == ["--", "/opt/delegate"]
    assert not any(c[2] == "remove" for c in run.calls)
    assert (home / ".claude/skills/plan-with-cheap-workers/SKILL.md").exists()


def test_claude_install_respects_the_users_config(home, tmp_path, monkeypatch):
    cfg = tmp_path / "router.yaml"
    cfg.write_text("models: []\n")
    monkeypatch.setenv("AUTO_ROUTER_CONFIG", str(cfg))
    run = Recorder()
    install_delegation.install("codex", server="/opt/delegate", run=run)
    assert f"AUTO_ROUTER_CONFIG={cfg.resolve()}" in run.calls[-1]
    assert ".auto-router/launcher.yaml" not in json.dumps(run.calls)


def test_an_existing_cli_entry_is_not_replaced_without_force(home):
    run = Recorder(existing=True)
    with pytest.raises(InstallError, match="--force"):
        install_delegation.install("claude", server="/opt/delegate", run=run)
    assert [c[2] for c in run.calls] == ["get"]
    assert not (home / ".claude").exists(), "a refused install copies no skill"
    run = Recorder(existing=True)
    install_delegation.install("claude", server="/opt/delegate", force=True, run=run)
    assert [c[2] for c in run.calls] == ["get", "remove", "add"]


def test_opencode_and_cursor_entries_leave_other_settings_alone(home, tmp_path):
    oc = home / ".config/opencode/opencode.json"
    oc.parent.mkdir(parents=True)
    oc.write_text(json.dumps({"mcp": {"other": {"type": "local"}}, "theme": "x"}))
    install_delegation.install("opencode", server="/opt/delegate")
    data = json.loads(oc.read_text())
    assert data["theme"] == "x" and data["mcp"]["other"] == {"type": "local"}
    assert data["mcp"]["auto-router-delegate"] == {"type": "local", "command": ["/opt/delegate"],
                                                   "enabled": True}
    out = install_delegation.install("cursor", server="/opt/delegate")
    assert json.loads((home / ".cursor/mcp.json").read_text())["mcpServers"]["auto-router-delegate"] \
        == {"command": "/opt/delegate", "args": []}
    assert any("not written" in line for line in out)
    assert not (Path.cwd() / ".cursor").exists()
    project = tmp_path / "project"
    project.mkdir()
    install_delegation.install("cursor", server="/opt/delegate", project=str(project))
    assert (project / ".cursor/rules/plan-with-cheap-workers.mdc").exists()


def test_the_installer_never_touches_credentials(home, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-must-not-appear")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-must-not-appear")
    run = Recorder()
    install_delegation.install("claude", server="/opt/delegate", run=run)
    install_delegation.install("opencode", server="/opt/delegate")
    written = "".join(p.read_text() for p in home.rglob("*") if p.is_file())
    assert "sk-must-not-appear" not in written + json.dumps(run.calls)


def _stub_bin(tmp_path):
    """A PATH on which git, python3 and ln only record that they were called."""
    stub = tmp_path / "stub-bin"
    stub.mkdir()
    log = tmp_path / "calls.log"
    for name in ("git", "python3", "ln", "pip"):
        (stub / name).write_text(f'#!/bin/sh\necho "{name} $*" >> "{log}"\nexit 1\n')
        (stub / name).chmod(0o755)
    return stub, log


@pytest.mark.parametrize("args", [["claude"], ["claude", "main"], ["claude", "v1.0"],
                                  ["claude", "7a7bc31"], ["nothing", "0" * 40]])
def test_the_shell_installer_refuses_an_unpinned_ref_before_any_side_effect(home, tmp_path, args):
    stub, log = _stub_bin(tmp_path)
    proc = subprocess.run(["/bin/sh", str(SHELL), *args], capture_output=True, text=True,
                          env={"HOME": str(home), "PATH": f"{stub}:/usr/bin:/bin"}, timeout=30)
    assert proc.returncode == 2
    assert not log.exists(), "the installer ran a command before validating its arguments"
    assert list(home.iterdir()) == []


def test_the_shell_installer_is_pinned_and_never_force_links():
    text = SHELL.read_text()
    assert "pull" not in text and "ln -sf" not in text
    assert "checkout -q --detach FETCH_HEAD" in text and "rev-parse HEAD" in text
