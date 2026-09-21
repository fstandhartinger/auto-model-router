import importlib.util
import json
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "install-delegation.py"
SPEC = importlib.util.spec_from_file_location("install_delegation", SCRIPT)
install_delegation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(install_delegation)


def test_json_config_update_preserves_existing_servers(tmp_path):
    path = tmp_path / "mcp.json"
    path.write_text(json.dumps({"mcpServers": {"existing": {"command": "x"}}, "other": 1}))
    install_delegation.update_json(path, "mcpServers", {"command": "delegate", "args": []})
    data = json.loads(path.read_text())
    assert data["other"] == 1 and data["mcpServers"]["existing"]["command"] == "x"
    assert data["mcpServers"]["auto-router-delegate"]["command"] == "delegate"


def test_copy_skill_replaces_stale_copy(tmp_path, monkeypatch):
    target = tmp_path / "skill"
    target.mkdir()
    (target / "stale").write_text("old")
    install_delegation.copy_skill(target)
    assert (target / "SKILL.md").exists() and not (target / "stale").exists()
