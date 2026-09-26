"""The Python half of the installers, in temporary HOMEs with mocked hardware and endpoints."""
import io
import json
from pathlib import Path

import pytest
import yaml

from auto_router import installer as I
from auto_router import smoke
from auto_router.hardware import GPU, Hardware

ROOT = Path(__file__).resolve().parents[1]


def hw(ram=16, gpus=()):
    return Hardware(os="linux", wsl=False, arch="x86_64", ram_gb=ram, gpus=list(gpus))


def args(tmp_path, *extra):
    return I.build_parser().parse_args(["--home", str(tmp_path / "home"), "--src", str(ROOT),
                                        "--launcher", "/opt/ar/auto-router", *extra])


def install(tmp_path, *extra, env=None, endpoints=(), machine=None, asker=None):
    (tmp_path / "home").mkdir(exist_ok=True)
    a = args(tmp_path, *extra)
    a.json = True
    return I.run_install(a, env=env if env is not None else {}, endpoints=list(endpoints),
                         hw=machine or hw(), asker=asker or I.Asker(False, a.yes), which=lambda b: None)


def test_key_names_only():
    env = {"OPENROUTER_API_KEY": "sk-or-verysecret", "TYPESAFE_API_KEY": ""}
    k = I.key_names(env)
    assert k["OPENROUTER_API_KEY"] is True and k["TYPESAFE_API_KEY"] is False
    assert "verysecret" not in json.dumps(k)


def test_cloud_install_writes_valid_config_with_key_names(tmp_path):
    env = {"OPENROUTER_API_KEY": "sk-or-secret-1", "TENSORX_API_KEY": "tx-secret-2"}
    s = install(tmp_path, "--yes", "--models", "cloud", "--harness", "opencode", env=env)
    cfg_path = tmp_path / "home/.auto-router/config.yaml"
    text = cfg_path.read_text()
    assert "secret" not in text
    cfg = yaml.safe_load(text)
    names = [m["name"] for m in cfg["models"]]
    for want in ("claude-opus-5.5", "claude-sonnet-5", "gpt-6-luna", "gpt-6-astra", "glm-5.3-flash",
                 "glm-5.3-flash-tensorx"):
        assert want in names
    assert set(cfg["enabled"]) == set(names)
    assert cfg["providers"]["tensorx"]["api_key_env"] == "TENSORX_API_KEY"
    assert cfg["policy"]["classifier"]["backend"] == "heuristic"
    assert I.validate_router_config(cfg_path) is None
    assert s["keys_set"] == ["OPENROUTER_API_KEY", "TENSORX_API_KEY"]
    assert "secret" not in (tmp_path / "home/.auto-router/install-summary.json").read_text()
    assert yaml.safe_load((tmp_path / "home/.auto-router/launcher.yaml").read_text())["models"]


def test_enabled_lists_only_routes_with_keys(tmp_path):
    install(tmp_path, "--yes", "--models", "cloud", "--harness", "none", env={"TENSORX_API_KEY": "x"})
    cfg = yaml.safe_load((tmp_path / "home/.auto-router/config.yaml").read_text())
    assert cfg["enabled"] == ["glm-5.3-flash-tensorx"]


def test_running_local_endpoints_are_configured(tmp_path):
    eps = [{"url": "http://127.0.0.1:1234/v1", "port": 1234, "label": "LM Studio",
            "models": ["ternary-bonsai-2-27b", "jevk5-4b-v0.2-q8_0"]}]
    s = install(tmp_path, "--yes", "--models", "bonsai,jev-local", "--harness", "none", endpoints=eps)
    cfg = yaml.safe_load((tmp_path / "home/.auto-router/config.yaml").read_text())
    assert cfg["providers"]["local-bonsai"]["base_url"] == "http://127.0.0.1:1234/v1"
    bonsai = next(m for m in cfg["models"] if m["name"] == "bonsai-2-27b")
    assert bonsai["free"] is True and bonsai["capability_like"] == "qwen3.8-27b::medium"
    assert cfg["policy"]["classifier"]["backend"] in ("local-jev", "heuristic")
    assert s["jev"]["model"] == "jevk5-4b-v0.2-q8_0"


def test_no_download_without_consent(tmp_path, monkeypatch):
    called = []
    monkeypatch.setattr(I, "download", lambda *a, **k: called.append(a))
    s = install(tmp_path, "--models", "bonsai,jev-local", "--harness", "none", "--with-bonsai",
                "--with-jev-local", machine=hw(32, [GPU("nvidia", "RTX 4090", 24.0)]))
    assert called == []
    assert s["bonsai"] is None and s["jev"] is None
    assert any("not downloaded" in l for l in s["log"])


def test_download_with_yes_writes_start_scripts(tmp_path, monkeypatch):
    called = []
    monkeypatch.setattr(I, "download", lambda repo, file, dest, **k: called.append((repo, file)))
    s = install(tmp_path, "--yes", "--models", "bonsai,jev-local", "--harness", "none", "--with-bonsai",
                "--with-jev-local", machine=hw(32, [GPU("nvidia", "RTX 4090", 24.0)]))
    assert ("prism-ml/Ternary-Bonsai-2-27B-gguf", "Ternary-Bonsai-2-27B-PQ2_0.gguf") in called
    assert ("alibiserikbay/JevK5-GGUF", "jevk5-4b-v0.2-Q8_0.gguf") in called
    bonsai = (tmp_path / "home/.auto-router/bin/start-bonsai").read_text()
    assert "PrismML" in bonsai and "--port 8081" in bonsai
    jev = (tmp_path / "home/.auto-router/bin/start-jev-local").read_text()
    assert "--port 8082" in jev and "-ngl 99" in jev
    cfg = yaml.safe_load((tmp_path / "home/.auto-router/config.yaml").read_text())
    assert cfg["providers"]["local-bonsai"]["base_url"] == "http://127.0.0.1:8081/v1"
    assert s["jev"]["temperature"] == 1.532


def test_interactive_answers(tmp_path):
    answers = io.StringIO("cloud,subscription\nopencode,claude-code\nn\n")
    s = install(tmp_path, asker=I.Asker(True, False, answers))
    assert s["groups"] == ["cloud", "subscription"]
    assert set(s["harnesses"]) == {"opencode", "claude-code"}
    assert s["claude_gateway"] is False
    cfg = yaml.safe_load((tmp_path / "home/.auto-router/config.yaml").read_text())
    assert any(m.get("subscription") == "claude" for m in cfg["models"])


def test_gateway_only_with_explicit_flag(tmp_path):
    install(tmp_path, "--yes", "--models", "subscription", "--harness", "claude-code")
    assert not (tmp_path / "home/.claude/settings.json").exists()
    install(tmp_path, "--yes", "--models", "subscription", "--harness", "claude-code", "--claude-gateway")
    data = json.loads((tmp_path / "home/.claude/settings.json").read_text())
    assert data["env"]["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:8787"


def test_hand_edited_config_is_kept(tmp_path):
    cfg = tmp_path / "home/.auto-router/config.yaml"
    cfg.parent.mkdir(parents=True)
    cfg.write_text("providers: {}\nmodels: []\n")
    install(tmp_path, "--yes", "--models", "cloud", "--harness", "none")
    assert cfg.read_text() == "providers: {}\nmodels: []\n"
    assert (cfg.parent / "config.yaml.new").exists()


def test_dry_run_changes_nothing(tmp_path):
    install(tmp_path, "--yes", "--dry-run", "--models", "all", "--harness", "all", "--claude-gateway")
    assert not any((tmp_path / "home").rglob("*"))


def test_uninstall_via_main(tmp_path, monkeypatch):
    install(tmp_path, "--yes", "--models", "cloud", "--harness", "opencode,cursor")
    assert (tmp_path / "home/.cursor/mcp.json").exists()
    assert I.main(["--uninstall", "--home", str(tmp_path / "home")]) == 0
    assert not (tmp_path / "home/.cursor").exists()
    assert not (tmp_path / "home/.config/opencode/opencode.json").exists()


def test_unknown_group_is_an_error(tmp_path):
    with pytest.raises(SystemExit):
        install(tmp_path, "--yes", "--models", "cloud,gpt5", "--harness", "none")


def test_unknown_classifier_backend_falls_back(tmp_path, monkeypatch):
    from auto_router import jev

    def strict(policy):
        if ((policy or {}).get("classifier") or {}).get("backend") == "local-jev":
            raise ValueError("unknown classifier backend 'local-jev'")
        return None
    monkeypatch.setattr(jev, "classifier_from_config", strict)
    eps = [{"url": "http://127.0.0.1:8082/v1", "port": 8082, "label": "x", "models": ["jevk5"]}]
    s = install(tmp_path, "--yes", "--models", "jev-local", "--harness", "none", endpoints=eps)
    cfg = yaml.safe_load((tmp_path / "home/.auto-router/config.yaml").read_text())
    assert cfg["policy"]["classifier"]["backend"] == "heuristic" and s["classifier"] == "heuristic"


def test_probe_local_uses_only_loopback():
    seen = []

    def fetch(url, timeout):
        seen.append(url)
        if ":1234/" in url:
            return {"data": [{"id": "ternary-bonsai-2-27b"}]}
        raise OSError("refused")
    found = I.probe_local(fetch=fetch)
    assert all(u.startswith("http://127.0.0.1:") for u in seen)
    assert found == [{"url": "http://127.0.0.1:1234/v1", "port": 1234, "label": "LM Studio",
                      "models": ["ternary-bonsai-2-27b"]}]


# ------------------------------------------------------------------ doctor

def test_doctor_dry_run_without_keys(tmp_path, monkeypatch, capsys):
    install(tmp_path, "--yes", "--models", "cloud", "--harness", "opencode", env={"OPENROUTER_API_KEY": "x"})
    home = tmp_path / "home"
    monkeypatch.setenv("AUTO_ROUTER_TEST_HOME", str(home))
    monkeypatch.setenv("AUTO_ROUTER_HOME", str(home / ".auto-router"))
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("AUTO_ROUTER_MODELS", raising=False)
    rc = smoke.main(["--config", str(home / ".auto-router/config.yaml"), "--url", "http://127.0.0.1:9",
                     "--no-start", "--json"])
    out = json.loads(capsys.readouterr().out)
    by = {(c["area"], c["name"]): c for c in out["checks"]}
    assert by[("route", "glm-5.3-flash")]["detail"] == "OPENROUTER_API_KEY is not set"
    assert by[("router", "http://127.0.0.1:9")]["status"] == "fail"
    assert any(c["area"] == "harness" and c["status"] == "ok" for c in out["checks"])
    assert rc == 1


def test_doctor_routes_dry_run_with_key(tmp_path, monkeypatch):
    raw = {"providers": {"or": {"base_url": "https://example.invalid/v1", "api_key_env": "K"}},
           "models": [{"name": "a", "provider": "or", "upstream_id": "x/a"}]}
    rep = smoke.Report()
    smoke.check_routes(raw, {"K": "dummy"}, rep, live=False)
    assert rep.items[0]["status"] == "dry-run"
    rep = smoke.Report()
    smoke.check_routes(raw, {"K": "dummy", "AUTO_ROUTER_MODELS": "b"}, rep, live=False)
    assert rep.items == []


def test_doctor_starts_a_temporary_router(tmp_path, monkeypatch):
    install(tmp_path, "--yes", "--models", "cloud", "--harness", "none", env={})
    cfg = tmp_path / "home/.auto-router/config.yaml"
    monkeypatch.setenv("AUTO_ROUTER_BENCH_OFFLINE", "1")
    monkeypatch.setenv("AUTO_ROUTER_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.chdir(ROOT)
    rep = smoke.Report()
    proc = smoke.check_router("http://127.0.0.1:9", cfg, rep, start=True)
    try:
        # Offline, bench-priced models cannot load, so the temporary router may
        # refuse to start; either outcome must be reported, never hang.
        assert rep.items and rep.items[0]["area"] == "router"
    finally:
        if proc is not None:
            proc.terminate()
            proc.wait(10)
