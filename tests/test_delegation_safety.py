"""Regression tests for the delegation-layer review of 7a7bc31 (findings F1-F8, F12).

Every worker here is a fake: a small shell script written into ``tmp_path``
that prints what it saw, sleeps, or leaves a background process behind. No
agent CLI, provider or network is involved.
"""

import io
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from auto_router import delegate, procs
from auto_router.catalog import Catalog, ModelInfo, Prices
from auto_router.jev import Classification
from auto_router.config import RouterConfig
from auto_router.launcher import (BASE_ENV_ALLOW, EnvPolicy, LauncherError, build_command,
                                  build_parser, check_cwd, execute)
from auto_router.quota import QuotaDecision
from auto_router.router import Router

ROOT = Path(__file__).resolve().parents[1]


def _script(path: Path, body: str) -> str:
    path.write_text("#!/bin/sh\n" + body)
    path.chmod(0o755)
    return str(path)


def _write_config(tmp_path: Path, worker: str, *, extra: str = "", plan_budget: bool = False) -> Path:
    budget = tmp_path / "no-such-budget.json"
    if plan_budget:
        budget.write_text(json.dumps({"generated_at": time.time(), "claude": {
            "week_percent": 5, "session_percent": 5, "week_resets_at": time.time() + 3 * 86400}}))
    cfg = tmp_path / "launcher.yaml"
    cfg.write_text(f"""
providers:
  local: {{base_url: http://127.0.0.1:9/v1, cache: generic}}
  anthropic: {{base_url: http://127.0.0.1:9/v1, api: anthropic, cache: anthropic}}
subscriptions:
  claude: {{budget_file: {budget}, clear_env: [ANTHROPIC_API_KEY]}}
policy: {{name: F_expected, classifier: {{backend: heuristic}}}}
{extra}
models:
  - name: plan-strong
    provider: anthropic
    upstream_id: opus
    subscription: claude
    capability: {{coding: 90, agentic: 90, general: 90}}
    runner: {{cmd: [{worker}, "{{task}}"]}}
  - name: cheap-notools
    provider: local
    upstream_id: tiny
    free: true
    tools: false
    context_tokens: 2000
    capability: {{coding: 10, agentic: 10, general: 10}}
    runner: {{cmd: [{worker}, "{{task}}"]}}
  - name: worker
    provider: local
    upstream_id: mid
    tools: true
    prices: {{input: 0.2, output: 0.8}}
    capability: {{coding: 50, agentic: 50, general: 50}}
    runner: {{cmd: [{worker}, "{{task}}"], timeout_s: 60}}
""")
    return cfg


@pytest.fixture
def hermetic(monkeypatch, tmp_path):
    """No ledger, no benchmark fetch, no hosted classifier, a delegation root in tmp."""
    for name in ("AUTO_ROUTER_LEDGER", "TYPESAFE_API_KEY", "AUTO_ROUTER_POLICY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("AUTO_ROUTER_BENCH_OFFLINE", "1")
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.setenv("AUTO_ROUTER_DELEGATE_ROOT", str(project))
    return project


def _launch(cfg: Path, *args: str, env_extra: dict | None = None) -> subprocess.CompletedProcess:
    env = {**os.environ, "AUTO_ROUTER_CONFIG": str(cfg), "PYTHONPATH": str(ROOT),
           **(env_extra or {})}
    return subprocess.run([sys.executable, "-m", "auto_router.launcher", *args],
                          capture_output=True, text=True, env=env, timeout=60)


# --------------------------------------------------------------------------
# F1 / F12: a tier stays inside the policy's eligible routes
# --------------------------------------------------------------------------
@pytest.mark.parametrize("tier", ["cheap", "strong"])
def test_a_tier_never_launches_a_closed_plan_or_a_route_without_tools(hermetic, tmp_path, tier):
    cfg = _write_config(tmp_path, _script(tmp_path / "w.sh", "echo ok\n"))
    proc = _launch(cfg, "--dry-run", "--tier", tier, "rename variable x to y in util.py")
    assert proc.returncode == 0, proc.stderr
    doc = json.loads(proc.stdout)
    # plan-strong's quota is closed (no budget file) and cheap-notools has no
    # tools: the only route the policy allows is `worker`.
    assert doc["route"] == "worker"
    assert "operator override" not in doc["reason"]
    assert doc["tier"] == tier
    notes = " ".join((doc.get("decision") or {}).get("notes") or [])
    assert "forced by the operator" not in notes


def test_a_strong_tier_can_pick_an_open_plan_and_says_it_was_the_tier(hermetic, tmp_path):
    cfg = _write_config(tmp_path, _script(tmp_path / "w.sh", "echo ok\n"), plan_budget=True)
    doc = json.loads(_launch(cfg, "--dry-run", "--tier", "strong", "rename x").stdout)
    assert doc["route"] == "plan-strong"
    assert "operator override" not in doc["reason"]
    # The delegate path never reaches a plan, whatever the tier.
    doc = json.loads(_launch(cfg, "--dry-run", "--no-plans", "--tier", "strong", "rename x").stdout)
    assert doc["route"] == "worker"


def _router(*models, difficulty=0.1, quota=None):
    calls = []

    def classifier(text, summary):
        calls.append(text)
        return Classification("coding", {}, difficulty, 1.0, 0.9, 0.0, 0.0, 0.2, 0.3, 0.01)

    router = Router(RouterConfig(providers={}, catalog=Catalog(list(models))),
                    classifier=classifier, quota_reader=lambda: quota or {})
    return router, calls


def _route(name, price, cap, **kw):
    return ModelInfo(name=name, provider="p", upstream_id=name, prices=Prices(price, price * 4),
                     capability={"coding": cap, "agentic": cap, "general": cap},
                     runner={"cmd": ["x", "{task}"]}, **kw)


def test_tier_selection_is_one_routing_pass_recorded_as_a_tier():
    router, calls = _router(_route("cheap", 0.1, 30), _route("dear", 5.0, 90))
    auto = router.route_job("rename x", tier="auto")
    calls.clear()
    strong = router.route_job("rename x", tier="strong")
    assert calls == ["rename x"], "the classifier ran more than once for one launch"
    assert strong.model.name == "dear"
    if auto.model.name != "dear":
        assert strong.reason.startswith("worker tier strong")
        assert any("not an operator override" in n for n in strong.explanation.notes)
    with pytest.raises(ValueError):
        router.route_job("x", tier="bogus")


def test_cheap_tier_skips_routes_the_policy_filters_out():
    notools = _route("notools", 0.0, 10, tools=False)
    tiny = _route("tiny-context", 0.0, 10, context_tokens=10)
    plan = _route("plan", 0.0, 95, subscription="claude")
    mid = _route("mid", 0.2, 50)
    router, _ = _router(notools, tiny, plan, mid,
                        quota={"claude": QuotaDecision(False, 1.0, 0.95, "hard stop")})
    result = router.route_job("rename x", tier="cheap")
    assert result.model.name == "mid"
    router, _ = _router(notools, tiny, plan, mid, difficulty=0.9,
                        quota={"claude": QuotaDecision(False, 1.0, 0.95, "hard stop")})
    assert router.route_job("hard", tier="strong").model.name == "mid"


# --------------------------------------------------------------------------
# F2: a timeout ends the whole process tree, at both boundaries
# --------------------------------------------------------------------------
LINGERING_WORKER = """
( sleep 1.5; echo late > "$MARK" ) &
sleep 30
"""


def _alive(pids):
    return [p for p in pids if Path(f"/proc/{p}").exists()
            and Path(f"/proc/{p}/stat").read_text().split(") ")[1][0] not in "ZX"]


def test_launcher_timeout_kills_the_agent_and_its_background_children(tmp_path, monkeypatch):
    mark = tmp_path / "late-write"
    worker = _script(tmp_path / "w.sh", LINGERING_WORKER)
    route = ModelInfo(name="w", provider="p", upstream_id="w", prices=Prices.free(),
                      capability={"coding": 50}, runner={"cmd": [worker, "{task}"], "timeout_s": 0.5,
                                                         "env": {"MARK": str(mark)}})
    started = time.monotonic()
    outcome = execute(build_command(route, "t"), cwd=str(tmp_path))
    assert outcome.timed_out and outcome.exit_code == 124
    assert time.monotonic() - started < 10
    time.sleep(2.5)
    assert not mark.exists(), "a descendant of the timed-out agent kept writing"


def test_a_background_process_left_by_a_finished_agent_is_stopped(tmp_path):
    mark = tmp_path / "late-write"
    worker = _script(tmp_path / "w.sh", '( sleep 1.5; echo late > "$MARK" ) >/dev/null 2>&1 &\necho done\n')
    route = ModelInfo(name="w", provider="p", upstream_id="w", prices=Prices.free(),
                      capability={"coding": 50}, runner={"cmd": [worker, "{task}"], "timeout_s": 20,
                                                         "env": {"MARK": str(mark)}})
    outcome = execute(build_command(route, "t"), cwd=str(tmp_path), capture=True)
    assert outcome.exit_code == 0 and "done" in outcome.stdout
    time.sleep(2.5)
    assert not mark.exists()


def test_delegate_timeout_stops_launcher_agent_and_grandchildren(hermetic, tmp_path, monkeypatch):
    mark = hermetic / "late-write"
    worker = _script(tmp_path / "w.sh", LINGERING_WORKER.replace('"$MARK"', f'"{mark}"'))
    cfg = _write_config(tmp_path, worker)
    monkeypatch.setenv("AUTO_ROUTER_CONFIG", str(cfg))
    started = time.monotonic()
    result = delegate.run_delegate("SLEEP then write", cwd=str(hermetic), tier="auto", timeout_s=1)
    assert not result["ok"] and "timed out" in result["error"]
    assert time.monotonic() - started < 15
    time.sleep(2.5)
    assert not mark.exists(), "the delegated worker outlived its timeout"


def test_procs_run_leaves_no_member_of_the_session_alive(tmp_path):
    script = _script(tmp_path / "tree.sh", "sleep 30 &\nsleep 30 &\nsleep 30\n")
    seen = {}
    real_terminate = procs.terminate

    def spy(proc, **kw):
        seen["pids"] = procs.members(proc.pid, scope=kw["scope"])
        return real_terminate(proc, **kw)

    procs.terminate, saved = spy, procs.terminate
    try:
        with pytest.raises(subprocess.TimeoutExpired):
            procs.run([script], timeout=0.5, scope="session")
    finally:
        procs.terminate = saved
    assert len(seen["pids"]) >= 3
    assert _alive(seen["pids"]) == []


# --------------------------------------------------------------------------
# F3: workers get an allowlisted environment and a bounded cwd
# --------------------------------------------------------------------------
def test_a_worker_does_not_inherit_the_planner_environment():
    source = {"PATH": "/usr/bin", "HOME": "/h", "FAKE_PROVIDER_SECRET": "sk-fake",
              "AWS_SECRET_ACCESS_KEY": "x", "SSH_AUTH_SOCK": "/s", "OPENAI_API_KEY": "sk"}
    route = ModelInfo(name="w", provider="p", upstream_id="w", prices=Prices.free(),
                      capability={}, runner={"cmd": ["cli", "{task}"], "env_pass": ["OPENAI_API_KEY"]})
    cmd = build_command(route, "t", environ=source)
    assert cmd.env == {"PATH": "/usr/bin", "HOME": "/h", "OPENAI_API_KEY": "sk"}
    assert set(BASE_ENV_ALLOW).isdisjoint({"SSH_AUTH_SOCK", "XDG_RUNTIME_DIR", "DBUS_SESSION_BUS_ADDRESS"})
    # The operator can widen the list for every route, or restore the old
    # behaviour explicitly.
    wide = build_command(route, "t", environ=source, env_policy=EnvPolicy(allow=("SSH_AUTH_SOCK",)))
    assert wide.env == {"SSH_AUTH_SOCK": "/s", "OPENAI_API_KEY": "sk"}
    legacy = build_command(route, "t", environ=source, env_policy=EnvPolicy(inherit=True))
    assert legacy.env["FAKE_PROVIDER_SECRET"] == "sk-fake"


def test_env_policy_reads_the_launcher_section():
    cfg = RouterConfig(providers={}, catalog=Catalog([]),
                       raw={"launcher": {"env_allow": ["HTTPS_PROXY"], "inherit_env": False}})
    policy = EnvPolicy.from_config(cfg)
    assert "HTTPS_PROXY" in policy.allow and "PATH" in policy.allow and not policy.inherit
    assert EnvPolicy.from_config(RouterConfig(providers={}, catalog=Catalog([]))) == EnvPolicy()


def test_planner_secret_never_reaches_a_delegated_worker(hermetic, tmp_path, monkeypatch):
    worker = _script(tmp_path / "w.sh", 'echo "SECRET_SEEN=${FAKE_PROVIDER_SECRET:-<none>}"\n'
                                        'echo "PATH_SET=${PATH:+yes}"\n')
    monkeypatch.setenv("AUTO_ROUTER_CONFIG", str(_write_config(tmp_path, worker)))
    monkeypatch.setenv("FAKE_PROVIDER_SECRET", "sk-fake-planner-env-123")
    for tier in ("auto", "cheap", "strong"):
        result = delegate.run_delegate("rename x", cwd=str(hermetic), tier=tier)
        assert result["ok"], result
        assert "SECRET_SEEN=<none>" in result["result"] and "PATH_SET=yes" in result["result"]
        assert "sk-fake-planner-env-123" not in json.dumps(result)


def test_delegation_cwd_must_stay_inside_the_root(hermetic, tmp_path, monkeypatch):
    inside = hermetic / "sub"
    inside.mkdir()
    assert delegate.resolve_cwd(str(inside)) == str(inside.resolve())
    assert delegate.resolve_cwd(None) == str(hermetic.resolve())
    for outside in (str(tmp_path), "/", str(hermetic / ".." / "..")):
        with pytest.raises(delegate.ArgumentError, match="outside"):
            delegate.resolve_cwd(outside)
    link = hermetic / "escape"
    link.symlink_to(tmp_path)
    with pytest.raises(delegate.ArgumentError, match="outside"):
        delegate.resolve_cwd(str(link))
    for broad in ("/", str(Path.home()), str(Path.home().parent)):
        monkeypatch.setenv("AUTO_ROUTER_DELEGATE_ROOT", broad)
        with pytest.raises(delegate.ArgumentError, match="too broad"):
            delegate.resolve_cwd(None)


def test_launcher_cwd_root_is_enforced(tmp_path):
    root = tmp_path / "root"
    (root / "a").mkdir(parents=True)
    check_cwd(str(root / "a"), root.resolve())
    with pytest.raises(LauncherError, match="outside launcher.cwd_root"):
        check_cwd(str(tmp_path), root.resolve())


# --------------------------------------------------------------------------
# F4 / F6: the brief is data, and bad arguments are structured errors
# --------------------------------------------------------------------------
@pytest.mark.parametrize("brief", ["--help", "--list", "--route=plan-strong", "-", "--"])
def test_a_brief_that_looks_like_an_option_stays_the_task(brief):
    argv = delegate.launcher_argv(brief, "/tmp", "cheap")
    assert argv[-2:] == ["--", brief]
    args = build_parser().parse_args(argv[3:])
    assert args.task == brief and not args.list and args.route is None


def test_option_like_briefs_reach_the_worker_verbatim(hermetic, tmp_path, monkeypatch):
    worker = _script(tmp_path / "w.sh", 'printf "got:%s\\n" "$1"\n')
    monkeypatch.setenv("AUTO_ROUTER_CONFIG", str(_write_config(tmp_path, worker)))
    for brief in ("--help", "--list", "--route=plan-strong"):
        result = delegate.run_delegate(brief, cwd=str(hermetic), tier="auto")
        assert result["ok"] and result["result"] == f"got:{brief}", result
        assert result["model"] == "worker"


def _rpc(*messages):
    out = io.StringIO()
    delegate.main(io.StringIO("\n".join(m if isinstance(m, str) else json.dumps(m)
                                        for m in messages) + "\n"), out)
    return [json.loads(line) for line in out.getvalue().splitlines()]


BAD_ARGUMENTS = [
    ("delegate", {"task": "x", "timeout_s": "soon"}, "timeout_s must be an integer"),
    ("delegate", {"task": "x", "timeout_s": True}, "timeout_s must be an integer"),
    ("delegate", {"task": "x", "timeout_s": 0}, "between"),
    ("delegate", {"task": "x", "timeout_s": 99999}, "between"),
    ("delegate", {"task": "x", "parallel": 9}, "between"),
    ("delegate", {"task": "x", "parallel": 2.5}, "integer"),
    ("delegate", {"task": "x", "tier": "free"}, "one of"),
    ("delegate", {"task": ""}, "empty"),
    ("delegate", {"task": 5}, "string"),
    ("delegate", {"task": "x", "context": ["a"]}, "string"),
    ("delegate", {"task": "x", "rm": True}, "unknown argument"),
    ("delegate", {}, "missing required"),
    ("delegate", ["task"], "must be an object"),
    ("delegate", {"task": "x" * (delegate.MAX_TASK_CHARS + 1)}, "longer than"),
    ("delegate_many", {"tasks": "abc"}, "must be a list"),
    ("delegate_many", {"tasks": []}, "items"),
    ("delegate_many", {"tasks": ["a"] * 33}, "items"),
    ("delegate_many", {"tasks": ["a", 3]}, "tasks[1] must be a string"),
    ("delegate_many", {"tasks": ["a", " "]}, "tasks[1] must not be empty"),
    ("delegate_many", {"tasks": ["a"], "parallel": "4"}, "integer"),
]


@pytest.mark.parametrize("name,arguments,message", BAD_ARGUMENTS)
def test_malformed_arguments_get_a_structured_error_and_the_server_stays_up(
        hermetic, monkeypatch, name, arguments, message):
    launched = []
    monkeypatch.setattr(delegate, "run_delegate", lambda *a, **k: launched.append(a) or {"ok": True})
    monkeypatch.setattr(delegate, "run_many", lambda *a, **k: launched.append(a) or {"ok": True})
    reply, pong = _rpc({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                        "params": {"name": name, "arguments": arguments}},
                       {"jsonrpc": "2.0", "id": 2, "method": "ping"})
    assert reply["result"]["isError"] is True
    assert reply["result"]["structuredContent"]["ok"] is False
    assert message in reply["result"]["structuredContent"]["error"]
    assert pong == {"jsonrpc": "2.0", "id": 2, "result": {}}
    assert launched == []


def test_an_exception_inside_a_tool_is_an_error_reply_not_a_dead_server(hermetic, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("secret detail")

    monkeypatch.setattr(delegate, "run_delegate", boom)
    reply, pong = _rpc({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                        "params": {"name": "delegate", "arguments": {"task": "x"}}},
                       {"jsonrpc": "2.0", "id": 2, "method": "ping"})
    assert reply["error"]["code"] == -32603 and "secret detail" not in reply["error"]["message"]
    assert pong == {"jsonrpc": "2.0", "id": 2, "result": {}}


def test_non_object_messages_and_params_are_rejected_cleanly(hermetic):
    replies = _rpc("[1, 2]", "\"text\"",
                   {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": ["delegate"]},
                   {"jsonrpc": "2.0", "id": 2, "method": "ping"})
    assert replies[0]["error"]["code"] == -32600 and replies[1]["error"]["code"] == -32600
    assert replies[2]["error"]["code"] == -32602
    assert replies[3]["result"] == {}


def test_valid_calls_are_normalised_before_running(hermetic, monkeypatch):
    seen = {}
    monkeypatch.setattr(delegate, "run_many", lambda tasks, **k: seen.update(tasks=tasks, **k) or {"ok": True})
    _rpc({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
          "params": {"name": "delegate_many", "arguments": {"tasks": ["a", "b"], "parallel": 2}}})
    assert seen["tasks"] == ["a", "b"] and seen["parallel"] == 2
    assert seen["tier"] == "cheap" and seen["timeout_s"] == 900
    assert seen["cwd"] == str(hermetic.resolve())


def test_the_server_stops_running_workers_when_it_is_told_to_stop(monkeypatch):
    called = []
    monkeypatch.setattr(procs, "terminate_all", lambda: called.append(True))
    with pytest.raises(SystemExit):
        delegate._exit_on_signal(15, None)
    assert called


# --------------------------------------------------------------------------
# F5: concurrent workers never share a tree
# --------------------------------------------------------------------------
def test_parallel_workers_each_get_their_own_copy_and_cwd_is_untouched(hermetic):
    (hermetic / "shared.txt").write_text("original\n")
    (hermetic / ".git").mkdir()
    (hermetic / ".git" / "HEAD").write_text("ref")
    cwds = []

    def worker(task, context, cwd, tier, timeout_s):
        cwds.append(cwd)
        Path(cwd, "shared.txt").write_text(f"edited by {task}\n")
        Path(cwd, f"{task}.new").write_text("new\n")
        return {"ok": True, "result": task}

    result = delegate.run_many(["one", "two", "three"], cwd=str(hermetic), parallel=3, runner=worker)
    assert result["ok"] and result["isolation"] == "per-worker copy"
    assert len(set(cwds)) == 3 and str(hermetic) not in cwds
    assert (hermetic / "shared.txt").read_text() == "original\n"
    assert not list(hermetic.glob("*.new"))
    for task, item in zip(["one", "two", "three"], result["results"]):
        changes = item["changes"]
        assert changes["modified"] == ["shared.txt"] and changes["added"] == [f"{task}.new"]
        assert f"+edited by {task}" in changes["patch"] and "-original" in changes["patch"]
        assert Path(item["workspace"], "shared.txt").read_text() == f"edited by {task}\n"
        assert not Path(item["workspace"], ".git").exists()


def test_parallel_duplicates_of_one_brief_are_isolated_too(hermetic, monkeypatch):
    seen = {}

    def fake_many(tasks, **kw):
        seen.update(tasks=tasks, **kw)
        return {"ok": True}

    monkeypatch.setattr(delegate, "run_many", fake_many)
    _rpc({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
          "params": {"name": "delegate", "arguments": {"task": "x", "parallel": 3}}})
    assert seen["tasks"] == ["x", "x", "x"] and seen["parallel"] == 3


def test_unchanged_worker_copies_are_removed(hermetic):
    (hermetic / "a.txt").write_text("a")
    result = delegate.run_many(["one", "two"], cwd=str(hermetic),
                               runner=lambda *a: {"ok": True})
    assert all(r["workspace"] is None for r in result["results"])


def test_a_tree_too_large_to_copy_is_refused(hermetic, monkeypatch):
    (hermetic / "big.bin").write_bytes(b"x" * 2048)
    monkeypatch.setattr(delegate, "COPY_LIMIT_BYTES", 1024)
    result = delegate.run_many(["one", "two"], cwd=str(hermetic), runner=lambda *a: {"ok": True})
    assert not result["ok"] and "too large" in result["error"]


def test_a_single_worker_runs_in_place(hermetic):
    cwds = []
    delegate.run_many(["only"], cwd=str(hermetic), runner=lambda t, c, cwd, *r: cwds.append(cwd) or {"ok": True})
    assert cwds == [str(hermetic)]


def test_cheap_tier_alone_would_have_taken_the_route_without_tools():
    """Guards the test above: without the tools filter the cheapest route is `notools`."""
    notools = _route("notools", 0.0, 10, tools=False)
    mid = _route("mid", 0.2, 50)
    router, _ = _router(notools, mid)
    assert router.route_job("rename x", tier="cheap").request.needs_tools
    router, _ = _router(_route("notools", 0.0, 10), mid)
    assert router.route_job("rename x", tier="cheap").model.name == "notools"
