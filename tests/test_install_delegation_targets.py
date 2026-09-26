"""The OpenClaw, Hermes and Copilot targets of the delegation installer.

Same rules as tests/test_install_delegation.py: HOME and every agent's own
home-directory variable point into ``tmp_path``; no agent CLI is run.
"""

import importlib.util
import json
from pathlib import Path

import pytest
import yaml

SCRIPT = Path(__file__).parents[1] / "scripts" / "install-delegation.py"
SPEC = importlib.util.spec_from_file_location("install_delegation_targets", SCRIPT)
install_delegation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(install_delegation)
InstallError = install_delegation.InstallError
NAME = "auto-router-delegate"
SKILL = "plan-with-cheap-workers"


@pytest.fixture
def home(tmp_path, monkeypatch):
    fake = tmp_path / "home"
    fake.mkdir()
    monkeypatch.setenv("HOME", str(fake))
    for var in ("AUTO_ROUTER_CONFIG", "HERMES_HOME", "COPILOT_HOME", "OPENCLAW_STATE_DIR",
                "OPENCLAW_CONFIG_PATH", "OPENCLAW_HOME"):
        monkeypatch.delenv(var, raising=False)
    return fake


def install(tool, **kw):
    return install_delegation.install(tool, server="/opt/delegate", **kw)


# ------------------------------------------------------------------ openclaw

def test_openclaw_writes_mcp_servers_and_the_managed_skill(home):
    settings = home / ".openclaw/openclaw.json"
    settings.parent.mkdir()
    settings.write_text(json.dumps({"agents": {"defaults": {"model": "x"}},
                                    "mcp": {"servers": {"other": {"command": "uvx"}}}}))
    out = install("openclaw")
    data = json.loads(settings.read_text())
    assert data["agents"] == {"defaults": {"model": "x"}}
    assert data["mcp"]["servers"] == {"other": {"command": "uvx"},
                                      NAME: {"command": "/opt/delegate", "args": []}}
    assert (home / ".openclaw/skills" / SKILL / "SKILL.md").exists()
    assert any("restart the OpenClaw gateway" in line for line in out)
    assert any("already current" in line for line in install("openclaw"))


def test_openclaw_honours_its_state_dir_and_records_a_named_config(home, tmp_path, monkeypatch):
    state = tmp_path / "claw-state"
    monkeypatch.setenv("OPENCLAW_STATE_DIR", str(state))
    cfg = tmp_path / "router.yaml"
    cfg.write_text("models: []\n")
    install("openclaw", config=str(cfg))
    entry = json.loads((state / "openclaw.json").read_text())["mcp"]["servers"][NAME]
    assert entry["env"] == {"AUTO_ROUTER_CONFIG": str(cfg.resolve())}
    assert (state / "skills" / SKILL / "SKILL.md").exists()
    assert not (home / ".openclaw").exists()


def test_openclaw_json5_is_refused_with_a_paste_ready_entry_and_nothing_written(home):
    settings = home / ".openclaw/openclaw.json"
    settings.parent.mkdir()
    json5 = "{\n  // mine\n  agents: {},\n}\n"
    settings.write_text(json5)
    with pytest.raises(InstallError) as err:
        install("openclaw")
    assert "not plain JSON" in str(err.value) and f"openclaw mcp set {NAME}" in str(err.value)
    assert '"servers"' in str(err.value)
    assert settings.read_text() == json5
    assert sorted(p.name for p in settings.parent.iterdir()) == ["openclaw.json"], "no skill"


def test_openclaw_differing_entry_needs_force_and_is_backed_up(home):
    settings = home / ".openclaw/openclaw.json"
    settings.parent.mkdir()
    original = {"mcp": {"servers": {NAME: {"command": "/mine"}}}}
    settings.write_text(json.dumps(original))
    with pytest.raises(InstallError, match="--force"):
        install("openclaw")
    assert json.loads(settings.read_text()) == original
    assert not (home / ".openclaw/skills").exists()
    install("openclaw", force=True)
    assert json.loads(settings.read_text())["mcp"]["servers"][NAME]["command"] == "/opt/delegate"
    backups = list(settings.parent.glob("openclaw.json.bak-*"))
    assert len(backups) == 1 and json.loads(backups[0].read_text()) == original


# -------------------------------------------------------------------- hermes

HERMES_CONFIG = """\
# my Hermes settings
model:
  default: some-model   # keep this comment
providers:
  mine:
    api: http://localhost:1234/v1
"""


def test_hermes_appends_mcp_servers_and_keeps_every_comment(home):
    settings = home / ".hermes/config.yaml"
    settings.parent.mkdir()
    settings.write_text(HERMES_CONFIG)
    out = install("hermes")
    text = settings.read_text()
    assert text.startswith(HERMES_CONFIG), "the existing text is kept byte for byte"
    data = yaml.safe_load(text)
    assert data["mcp_servers"] == {NAME: {"command": "/opt/delegate", "args": []}}
    assert data["providers"] == {"mine": {"api": "http://localhost:1234/v1"}}
    backups = list(settings.parent.glob("config.yaml.bak-*"))
    assert len(backups) == 1 and backups[0].read_text() == HERMES_CONFIG
    assert (home / ".hermes/skills" / SKILL / "SKILL.md").exists()
    assert any("pass --config" in line for line in out)
    again = install("hermes")
    assert any("already current" in line for line in again)
    assert settings.read_text() == text


def test_hermes_inserts_below_an_existing_mcp_servers_block(home, tmp_path, monkeypatch):
    base = tmp_path / "hermes-home"
    base.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(base))
    settings = base / "config.yaml"
    settings.write_text("mcp_servers:   # tools\n    time:\n        command: uvx\n"
                        "        args: [\"mcp-server-time\"]\nother: 1\n")
    cfg = tmp_path / "router.yaml"
    cfg.write_text("models: []\n")
    out = install("hermes", config=str(cfg))
    text = settings.read_text()
    assert "mcp_servers:   # tools\n    auto-router-delegate:\n" in text
    data = yaml.safe_load(text)
    assert data["other"] == 1
    assert data["mcp_servers"]["time"] == {"command": "uvx", "args": ["mcp-server-time"]}
    assert data["mcp_servers"][NAME] == {"command": "/opt/delegate", "args": [],
                                         "env": {"AUTO_ROUTER_CONFIG": str(cfg.resolve())}}
    assert (base / "skills" / SKILL / "SKILL.md").exists() and not (home / ".hermes").exists()
    assert not any("pass --config" in line for line in out)


def test_hermes_creates_a_missing_config(home):
    install("hermes")
    assert yaml.safe_load((home / ".hermes/config.yaml").read_text()) == {
        "mcp_servers": {NAME: {"command": "/opt/delegate", "args": []}}}


@pytest.mark.parametrize("text, reason", [
    ("mcp_servers: {}\n", "not a plain block mapping"),
    ("mcp_servers: []\n", "not a mapping"),
    ("- a\n- b\n", "does not hold a YAML mapping"),
    ("model: [unclosed\n", "not valid YAML"),
    (f"mcp_servers:\n  {NAME}:\n    command: /mine\n", "edit it by hand"),
])
def test_hermes_refuses_what_it_cannot_edit_safely_and_writes_nothing(home, text, reason):
    settings = home / ".hermes/config.yaml"
    settings.parent.mkdir()
    settings.write_text(text)
    for force in (False, True):
        with pytest.raises(InstallError, match=reason) as err:
            install("hermes", force=force)
        assert settings.read_text() == text
        assert sorted(p.name for p in settings.parent.iterdir()) == ["config.yaml"]
    if "mapping" in reason or "by hand" in reason:
        assert f"mcp_servers:\n  {NAME}:\n" in str(err.value), "the snippet to paste is printed"


def test_hermes_refused_skill_leaves_the_yaml_alone(home):
    settings = home / ".hermes/config.yaml"
    settings.parent.mkdir()
    settings.write_text(HERMES_CONFIG)
    skill = home / ".hermes/skills" / SKILL
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text("mine\n")
    with pytest.raises(InstallError, match="exists and differs"):
        install("hermes")
    assert settings.read_text() == HERMES_CONFIG


# ------------------------------------------------------------------- copilot

def test_copilot_default_writes_the_cli_user_config_only(home):
    settings = home / ".copilot/mcp-config.json"
    settings.parent.mkdir()
    settings.write_text(json.dumps({"mcpServers": {"playwright": {"type": "local"}}}))
    out = install("copilot")
    assert json.loads(settings.read_text())["mcpServers"] == {
        "playwright": {"type": "local"},
        NAME: {"type": "local", "command": "/opt/delegate", "args": [], "tools": ["*"]}}
    assert any("pass --project DIR" in line for line in out)
    assert sorted(p.name for p in home.rglob("*")) == [".copilot", "mcp-config.json"]


def test_copilot_honours_copilot_home(home, tmp_path, monkeypatch):
    monkeypatch.setenv("COPILOT_HOME", str(tmp_path / "cp"))
    install("copilot")
    assert NAME in json.loads((tmp_path / "cp/mcp-config.json").read_text())["mcpServers"]
    assert not (home / ".copilot").exists()


def test_copilot_project_writes_vscode_servers_and_new_instructions(home, tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    cfg = tmp_path / "router.yaml"
    cfg.write_text("models: []\n")
    install("copilot", project=str(project), config=str(cfg))
    assert json.loads((project / ".vscode/mcp.json").read_text()) == {"servers": {NAME: {
        "type": "stdio", "command": "/opt/delegate", "args": [],
        "env": {"AUTO_ROUTER_CONFIG": str(cfg.resolve())}}}}
    notes = project / ".github/copilot-instructions.md"
    assert "delegate_many" in notes.read_text() and "alwaysApply" not in notes.read_text()
    assert not (home / ".copilot").exists(), "--project does not touch the user config"
    assert any("already current" in line
               for line in install("copilot", project=str(project), config=str(cfg)))


def test_copilot_instructions_are_never_replaced_even_with_force(home, tmp_path):
    project = tmp_path / "project"
    notes = project / ".github/copilot-instructions.md"
    notes.parent.mkdir(parents=True)
    notes.write_text("my own instructions\n")
    out = install("copilot", project=str(project), force=True)
    assert notes.read_text() == "my own instructions\n"
    assert sorted(p.name for p in notes.parent.iterdir()) == ["copilot-instructions.md"]
    assert any("never replaced" in line and "delegate_many" in line for line in out)


def test_copilot_jsonc_workspace_config_is_refused(home, tmp_path):
    project = tmp_path / "project"
    vscode = project / ".vscode/mcp.json"
    vscode.parent.mkdir(parents=True)
    jsonc = '{\n  // servers\n  "servers": {}\n}\n'
    vscode.write_text(jsonc)
    with pytest.raises(InstallError, match="not plain JSON"):
        install("copilot", project=str(project))
    assert vscode.read_text() == jsonc and not (project / ".github").exists()


def test_new_targets_never_touch_credentials(home, tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-must-not-appear")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-must-not-appear")
    project = tmp_path / "project"
    project.mkdir()
    for tool in ("openclaw", "hermes", "copilot"):
        install(tool)
    install("copilot", project=str(project))
    written = "".join(p.read_text() for root in (home, project)
                      for p in root.rglob("*") if p.is_file())
    assert "sk-must-not-appear" not in written
