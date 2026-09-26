"""Both installer entry points, run end to end against fakes in a disposable HOME.

``scripts/install-delegation.sh`` and ``scripts/install-delegation.py`` run as
real processes through their own argument parsing and control flow. Every
external command they call is a local fake that appends its arguments to a
log and then imitates just enough of the real tool:

- ``git`` "fetches" by copying ``scripts/``, ``skills/`` and ``integrations/``
  from this checkout; it never contacts a remote.
- ``python3 -m venv`` makes a directory whose ``pip`` only records the call
  (and drops a stub ``auto-router-delegate``) and whose ``python`` is this
  interpreter, so the real Python installer runs.
- ``claude`` and ``codex`` keep their MCP entries as files under ``tmp_path``.
- ``cursor``, ``opencode``, ``openclaw``, ``hermes`` and ``copilot`` exist only
  to prove the installer never calls them.

HOME is a directory in ``tmp_path`` and PATH holds only the fakes and the
system directories. This proves the installers' control flow, not that they
work with the real git, pip, Claude Code or Codex CLIs: those are untested.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SHELL = ROOT / "scripts" / "install-delegation.sh"
PYTHON_INSTALLER = ROOT / "scripts" / "install-delegation.py"
SHA = "0123456789abcdef0123456789abcdef01234567"
NAME = "auto-router-delegate"
URL = "https://github.com/fstandhartinger/auto-model-router.git"

FAKE_GIT = r"""#!/bin/sh
echo "git $*" >> "$FAKE_LOG"
repo=.
if [ "$1" = "-C" ]; then repo=$2; shift 2; fi
if [ "$1" = "-c" ]; then shift 2; fi
case "$1" in
  init) mkdir -p "$3/.git" ;;
  remote) ;;
  status) [ -z "${FAKE_GIT_DIRTY:-}" ] || echo " M scripts/local-edit" ;;
  fetch) cp -R "$FAKE_SOURCE/scripts" "$FAKE_SOURCE/skills" "$FAKE_SOURCE/integrations" "$repo/"
         chmod -R u+w "$repo"
         echo "$6" > "$repo/.git/FETCH_HEAD" ;;
  checkout) cp "$repo/.git/FETCH_HEAD" "$repo/.git/HEAD" ;;
  rev-parse) if [ -n "${FAKE_GIT_HEAD:-}" ]; then echo "$FAKE_GIT_HEAD"; else cat "$repo/.git/HEAD"; fi ;;
  *) exit 1 ;;
esac
"""

FAKE_PYTHON3 = r"""#!/bin/sh
echo "python3 $*" >> "$FAKE_LOG"
[ "$1" = "-m" ] && [ "$2" = "venv" ] || exit 1
mkdir -p "$3/bin"
cp "$FAKE_BIN/venv-pip" "$3/bin/pip"
printf '#!/bin/sh\nexec "%s" "$@"\n' "$FAKE_REAL_PYTHON" > "$3/bin/python"
chmod +x "$3/bin/pip" "$3/bin/python"
"""

FAKE_PIP = r"""#!/bin/sh
echo "pip $*" >> "$FAKE_LOG"
[ "$1" = "install" ] || exit 1
printf '#!/bin/sh\necho "auto-router-delegate $*" >> "$FAKE_LOG"\n' > "$(dirname "$0")/auto-router-delegate"
chmod +x "$(dirname "$0")/auto-router-delegate"
"""

# claude|codex mcp get NAME / add [--scope S] [--env K=V] NAME -- CMD / remove NAME [--scope S]
FAKE_AGENT_CLI = r"""#!/bin/sh
tool=$(basename "$0")
echo "$tool $*" >> "$FAKE_LOG"
store="$FAKE_STATE/$tool"
mkdir -p "$store"
[ "$1" = "mcp" ] || exit 2
action=$2; shift 2
case "$action" in
  get) [ -f "$store/$1" ] ;;
  remove) [ -f "$store/$1" ] && rm "$store/$1" ;;
  add) line="$*"; name=
       while [ $# -gt 0 ]; do
         case "$1" in --scope|--env) shift 2 ;; --) break ;; *) name=$1; shift ;; esac
       done
       [ -n "$name" ] && [ ! -f "$store/$name" ] && echo "$line" > "$store/$name" ;;
  *) exit 2 ;;
esac
"""

NEVER_CALLED = r"""#!/bin/sh
echo "$(basename "$0") $*" >> "$FAKE_LOG"
exit 1
"""


@pytest.fixture
def fake(tmp_path):
    """A fake HOME, a PATH of recording fakes, and helpers to run and inspect."""
    home = tmp_path / "home"
    home.mkdir()
    bin_dir = tmp_path / "fake-bin"
    bin_dir.mkdir()
    state = tmp_path / "fake-state"
    log = tmp_path / "calls.log"
    for name, body in {"git": FAKE_GIT, "python3": FAKE_PYTHON3, "venv-pip": FAKE_PIP,
                       "claude": FAKE_AGENT_CLI, "codex": FAKE_AGENT_CLI,
                       "cursor": NEVER_CALLED, "opencode": NEVER_CALLED,
                       "openclaw": NEVER_CALLED, "hermes": NEVER_CALLED,
                       "copilot": NEVER_CALLED}.items():
        (bin_dir / name).write_text(body)
        (bin_dir / name).chmod(0o755)
    env = {"HOME": str(home), "PATH": f"{bin_dir}:/usr/bin:/bin", "LANG": "C.UTF-8",
           "FAKE_LOG": str(log), "FAKE_BIN": str(bin_dir), "FAKE_STATE": str(state),
           "FAKE_SOURCE": str(ROOT), "FAKE_REAL_PYTHON": sys.executable}

    class Fake:
        pass

    f = Fake()
    f.home, f.state, f.log, f.env, f.tmp = home, state, log, env, tmp_path
    f.base = home / ".auto-router"
    f.link = home / ".local/bin" / NAME

    def run(*argv, extra=None, entry="sh"):
        cmd = (["/bin/sh", str(SHELL)] if entry == "sh"
               else [sys.executable, str(PYTHON_INSTALLER)]) + list(argv)
        return subprocess.run(cmd, capture_output=True, text=True, timeout=60,
                              env={**env, **(extra or {})}, cwd=tmp_path)

    def calls():
        return log.read_text().splitlines() if log.exists() else []

    def home_files():
        return sorted(str(p.relative_to(home)) for p in home.rglob("*")
                      if (p.is_file() or p.is_symlink()) and ".auto-router/src/" not in str(p))

    f.run, f.calls, f.home_files = run, calls, home_files
    return f


def _tools(calls):
    return [c.split(" ")[0] for c in calls]


def test_shell_installer_for_claude_runs_its_whole_path_against_fakes(fake):
    proc = fake.run("claude", SHA)
    assert proc.returncode == 0, proc.stderr
    repo = fake.base / "src"
    assert fake.calls() == [
        f"git init -q {repo}",
        f"git -C {repo} remote add origin {URL}",
        f"git -C {repo} fetch -q --depth 1 origin {SHA}",
        f"git -C {repo} -c advice.detachedHead=false checkout -q --detach FETCH_HEAD",
        f"git -C {repo} rev-parse HEAD",
        f"python3 -m venv {fake.base / 'venv'}",
        f"pip install -q {repo}",
        f"claude mcp get {NAME}",
        f"claude mcp add --scope user {NAME} -- {fake.link}",
    ]
    assert os.readlink(fake.link) == str(fake.base / "venv/bin" / NAME)
    assert (fake.state / "claude" / NAME).read_text().strip() == f"--scope user {NAME} -- {fake.link}"
    assert "Installed auto-router delegation for claude." in proc.stdout
    assert "no AUTO_ROUTER_CONFIG recorded" in proc.stdout
    skill = "plan-with-cheap-workers"
    assert (fake.home / ".claude/skills" / skill / "SKILL.md").read_bytes() == \
        (ROOT / "skills" / skill / "SKILL.md").read_bytes()
    # nothing else in the fake HOME: no global git, pip or agent configuration
    assert fake.home_files() == sorted([
        ".auto-router/venv/bin/auto-router-delegate",
        ".auto-router/venv/bin/pip", ".auto-router/venv/bin/python",
        ".local/bin/auto-router-delegate",
        *(f".claude/skills/{skill}/{p.relative_to(ROOT / 'skills' / skill)}"
          for p in (ROOT / "skills" / skill).rglob("*") if p.is_file()),
    ])
    assert not any(p.name.startswith((".gitconfig", ".claude.json", ".codex"))
                   for p in fake.home.iterdir())


def test_a_second_shell_install_fails_closed_and_force_replaces(fake):
    assert fake.run("claude", SHA).returncode == 0
    entry = (fake.state / "claude" / NAME).read_text()
    first = len(fake.calls())

    again = fake.run("claude", SHA)
    assert again.returncode == 1
    assert f"claude already has an MCP server named '{NAME}'" in again.stderr
    rerun = fake.calls()[first:]
    assert f"git -C {fake.base / 'src'} status --porcelain" in rerun
    assert not any(c.startswith(("git init", "git -C " + str(fake.base / "src") + " remote"))
                   for c in rerun)
    assert not any(c.startswith("python3") for c in rerun), "the existing venv is reused"
    assert [c for c in rerun if c.startswith("claude")] == [f"claude mcp get {NAME}"]
    assert (fake.state / "claude" / NAME).read_text() == entry

    second = len(fake.calls())
    forced = fake.run("claude", SHA, "--force")
    assert forced.returncode == 0, forced.stderr
    assert [c for c in fake.calls()[second:] if c.startswith("claude")] == [
        f"claude mcp get {NAME}", f"claude mcp remove {NAME} --scope user",
        f"claude mcp add --scope user {NAME} -- {fake.link}"]
    assert "skill already current" in forced.stdout


def test_shell_installer_for_codex_records_an_existing_config(fake):
    cfg = fake.tmp / "launcher.yaml"
    cfg.write_text("models: []\n")
    proc = fake.run("codex", SHA, "--config", str(cfg))
    assert proc.returncode == 0, proc.stderr
    assert fake.calls()[-1] == f"codex mcp add --env AUTO_ROUTER_CONFIG={cfg} {NAME} -- {fake.link}"
    assert (fake.home / ".agents/skills/plan-with-cheap-workers/SKILL.md").exists()
    assert "claude" not in _tools(fake.calls())


def test_shell_installer_for_cursor_and_opencode_writes_json_and_calls_no_cli(fake):
    project = fake.tmp / "project"
    project.mkdir()
    assert fake.run("cursor", SHA, "--project", str(project)).returncode == 0
    oc = fake.home / ".config/opencode/opencode.json"
    oc.parent.mkdir(parents=True)
    oc.write_text(json.dumps({"theme": "mine", "mcp": {"other": {"type": "local"}}}))
    assert fake.run("opencode", SHA).returncode == 0
    assert json.loads((fake.home / ".cursor/mcp.json").read_text()) == {
        "mcpServers": {NAME: {"command": str(fake.link), "args": []}}}
    assert (project / ".cursor/rules/plan-with-cheap-workers.mdc").read_bytes() == \
        (ROOT / "integrations/cursor-plan-with-cheap-workers.mdc").read_bytes()
    assert json.loads(oc.read_text()) == {"theme": "mine", "mcp": {
        "other": {"type": "local"},
        NAME: {"type": "local", "command": [str(fake.link)], "enabled": True}}}
    tools = set(_tools(fake.calls()))
    assert tools == {"git", "python3", "pip"}, "cursor, opencode, claude and codex were never run"


def test_shell_installer_for_openclaw_hermes_and_copilot_edits_files_and_calls_no_cli(fake):
    import yaml

    hermes = fake.home / ".hermes/config.yaml"
    hermes.parent.mkdir(parents=True)
    hermes.write_text("# mine\nmodel:\n  default: x\n")
    project = fake.tmp / "project"
    project.mkdir()
    for argv in (("openclaw",), ("hermes",), ("copilot",), ("copilot", "--project", str(project))):
        proc = fake.run(argv[0], SHA, *argv[1:])
        assert proc.returncode == 0, proc.stderr
        assert f"Installed auto-router delegation for {argv[0]}." in proc.stdout
    link = str(fake.link)
    assert json.loads((fake.home / ".openclaw/openclaw.json").read_text()) == {
        "mcp": {"servers": {NAME: {"command": link, "args": []}}}}
    assert hermes.read_text().startswith("# mine\nmodel:\n  default: x\n")
    assert yaml.safe_load(hermes.read_text())["mcp_servers"] == {NAME: {"command": link, "args": []}}
    assert json.loads((fake.home / ".copilot/mcp-config.json").read_text()) == {"mcpServers": {
        NAME: {"type": "local", "command": link, "args": [], "tools": ["*"]}}}
    assert json.loads((project / ".vscode/mcp.json").read_text()) == {"servers": {
        NAME: {"type": "stdio", "command": link, "args": []}}}
    assert (project / ".github/copilot-instructions.md").exists()
    for skills in (".openclaw/skills", ".hermes/skills"):
        assert (fake.home / skills / "plan-with-cheap-workers/SKILL.md").read_bytes() == \
            (ROOT / "skills/plan-with-cheap-workers/SKILL.md").read_bytes()
    assert set(_tools(fake.calls())) == {"git", "python3", "pip"}, "no agent CLI was run"


def test_a_refused_hermes_entry_leaves_the_yaml_and_skills_alone(fake):
    hermes = fake.home / ".hermes/config.yaml"
    hermes.parent.mkdir(parents=True)
    text = f"mcp_servers:\n  {NAME}:\n    command: /mine\n"
    hermes.write_text(text)
    proc = fake.run("hermes", "--server", "/opt/delegate", "--force", entry="py")
    assert proc.returncode == 1 and "edit it by hand" in proc.stderr
    assert "command: \"/opt/delegate\"" in proc.stderr, "the snippet to paste is printed"
    assert hermes.read_text() == text
    assert sorted(p.name for p in hermes.parent.iterdir()) == ["config.yaml"]


def test_a_checkout_with_local_changes_is_left_alone(fake):
    (fake.base / "src/.git").mkdir(parents=True)
    proc = fake.run("claude", SHA, extra={"FAKE_GIT_DIRTY": "1"})
    assert proc.returncode == 1 and "has local changes" in proc.stderr
    assert fake.calls() == [f"git -C {fake.base / 'src'} status --porcelain"]
    assert not fake.link.exists() and not fake.state.exists()


def test_a_fetched_commit_that_does_not_match_the_pin_stops_before_pip(fake):
    proc = fake.run("claude", SHA, extra={"FAKE_GIT_HEAD": "f" * 40})
    assert proc.returncode == 1 and "does not match" in proc.stderr
    assert _tools(fake.calls()) == ["git"] * 5
    assert not (fake.base / "venv").exists() and not fake.link.exists()


def test_a_foreign_command_link_is_not_replaced(fake):
    fake.link.parent.mkdir(parents=True)
    fake.link.symlink_to("/opt/someone-else/auto-router-delegate")
    proc = fake.run("claude", SHA)
    assert proc.returncode == 1 and "is not this install" in proc.stderr
    assert os.readlink(fake.link) == "/opt/someone-else/auto-router-delegate"
    assert "claude" not in _tools(fake.calls())


def test_a_missing_config_file_stops_before_any_agent_entry(fake):
    proc = fake.run("claude", SHA, "--config", str(fake.tmp / "missing.yaml"))
    assert proc.returncode == 1 and "does not exist" in proc.stderr
    assert "claude" not in _tools(fake.calls())
    assert not (fake.home / ".claude").exists()


def test_the_python_entry_point_refuses_jsonc_and_leaves_it_byte_identical(fake):
    oc = fake.home / ".config/opencode/opencode.json"
    oc.parent.mkdir(parents=True)
    jsonc = '{\n  // mine\n  "mcp": {}\n}\n'
    oc.write_text(jsonc)
    proc = fake.run("opencode", "--server", "/opt/delegate", entry="py")
    assert proc.returncode == 1 and "not plain JSON" in proc.stderr
    assert oc.read_text() == jsonc
    assert sorted(p.name for p in oc.parent.iterdir()) == ["opencode.json"], "no skill copied"
    assert not fake.calls()


@pytest.mark.parametrize("tool", ["claude", "codex"])
def test_a_refused_cli_entry_leaves_no_skill_behind(fake, tool):
    store = fake.state / tool
    store.mkdir(parents=True)
    (store / NAME).write_text("someone else's server\n")
    proc = fake.run(tool, "--server", "/opt/delegate", entry="py")
    assert proc.returncode == 1 and "already has an MCP server" in proc.stderr
    assert fake.calls() == [f"{tool} mcp get {NAME}"]
    assert (store / NAME).read_text() == "someone else's server\n"
    assert list(fake.home.iterdir()) == [], "the refused install wrote nothing to HOME"


def test_a_refused_skill_leaves_the_cli_entry_alone(fake):
    skill = fake.home / ".claude/skills/plan-with-cheap-workers"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text("mine\n")
    proc = fake.run("claude", "--server", "/opt/delegate", entry="py")
    assert proc.returncode == 1 and "exists and differs" in proc.stderr
    assert not fake.calls() and not fake.state.exists()
    assert [p.name for p in skill.iterdir()] == ["SKILL.md"]


def test_a_refused_opencode_entry_leaves_no_skill_behind(fake):
    oc = fake.home / ".config/opencode/opencode.json"
    oc.parent.mkdir(parents=True)
    oc.write_text(json.dumps({"mcp": {NAME: {"type": "local", "command": ["/other"]}}}))
    before = oc.read_bytes()
    proc = fake.run("opencode", "--server", "/opt/delegate", entry="py")
    assert proc.returncode == 1 and "different 'auto-router-delegate' entry" in proc.stderr
    assert oc.read_bytes() == before
    assert sorted(p.name for p in oc.parent.iterdir()) == ["opencode.json"]


def test_a_refused_cursor_rule_leaves_the_global_settings_alone(fake):
    project = fake.tmp / "project"
    rule = project / ".cursor/rules/plan-with-cheap-workers.mdc"
    rule.parent.mkdir(parents=True)
    rule.write_text("my own rule\n")
    proc = fake.run("cursor", "--server", "/opt/delegate", "--project", str(project), entry="py")
    assert proc.returncode == 1 and "exists and differs" in proc.stderr
    assert rule.read_text() == "my own rule\n"
    assert list(fake.home.iterdir()) == [], "no ~/.cursor/mcp.json written"


def test_the_python_entry_point_rejects_an_unknown_tool_through_argparse(fake):
    proc = fake.run("vscode", entry="py")
    assert proc.returncode == 2 and "invalid choice" in proc.stderr
    assert not fake.calls() and list(fake.home.iterdir()) == []


def test_the_python_entry_point_registers_through_the_fake_cli(fake):
    proc = fake.run("claude", "--server", "/opt/delegate", entry="py")
    assert proc.returncode == 0, proc.stderr
    assert fake.calls() == [f"claude mcp get {NAME}",
                            f"claude mcp add --scope user {NAME} -- /opt/delegate"]
