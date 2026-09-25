"""Regression tests for the delegation-layer review of 7a7bc31 (findings F1-F8, F12).

Every worker here is a fake: a small shell script written into ``tmp_path``
that prints what it saw, sleeps, or leaves a background process behind. No
agent CLI, provider or network is involved.
"""

import io
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from auto_router import delegate, procs
from auto_router import launcher as launcher_mod
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
    stopping = threading.Event()
    monkeypatch.setattr(delegate, "_STOPPING", stopping)
    monkeypatch.setattr(procs, "terminate_all", lambda: called.append(stopping.is_set()))
    with pytest.raises(SystemExit) as exc:
        delegate._exit_on_signal(15, None)
    assert exc.value.code == 128 + 15
    assert called == [True], "queued briefs must be barred before the running ones are stopped"
    # A second signal leaves the cleanup the first one began alone.
    assert delegate._exit_on_signal(1, None) is None
    assert called == [True]


def test_a_second_stop_signal_does_not_abort_the_cleanup_the_first_began(
        hermetic, tmp_path, monkeypatch):
    # Real signals through the real handler: the first arrives while two
    # workers run, the second while the interrupted run deletes its copies.
    # No worker is left to write to them, so they must all still go.
    (hermetic / "a.txt").write_text("a\n")
    monkeypatch.setattr(delegate.tempfile, "tempdir", str(tmp_path))
    stopping = threading.Event()
    monkeypatch.setattr(delegate, "_STOPPING", stopping)
    both = threading.Barrier(2)
    real_rmtree = delegate.shutil.rmtree
    second = []

    def runner(task, context, cwd, tier, timeout_s):
        both.wait(timeout=10)
        if task == "one":
            os.kill(os.getpid(), signal.SIGTERM)
        stopping.wait(timeout=10)  # set by the handler, in the main thread
        return {"ok": True}

    def rmtree(path, *a, **k):
        if not second:
            second.append(path)
            os.kill(os.getpid(), signal.SIGTERM)
        return real_rmtree(path, *a, **k)

    monkeypatch.setattr(delegate.shutil, "rmtree", rmtree)
    previous = signal.signal(signal.SIGTERM, delegate._exit_on_signal)
    try:
        with pytest.raises(SystemExit) as exc:
            delegate.run_many(["one", "two"], cwd=str(hermetic), parallel=2, runner=runner)
    finally:
        signal.signal(signal.SIGTERM, previous)
    assert exc.value.code == 128 + signal.SIGTERM
    assert len(second) == 1, "the second signal must arrive during the cleanup"
    assert not list(tmp_path.glob("auto-router-delegate-*")), "the worker copies were left behind"


SIGNAL_WHILE_LOCKED = """
import os, signal, sys
from auto_router import delegate, launcher, procs
handler = {"delegate": delegate._exit_on_signal, "launcher": launcher._exit_on_signal}[sys.argv[1]]
point, pidfile = sys.argv[2], sys.argv[3]

class Tripwire(dict):
    # Sends the stop signal from inside procs' own "with _LIVE_LOCK:" blocks,
    # on the main thread, which is where Python then runs the handler.
    fired = False

    def _fire(self):
        if not Tripwire.fired:
            Tripwire.fired = True
            os.kill(os.getpid(), signal.SIGTERM)

    def __setitem__(self, proc, scope):
        super().__setitem__(proc, scope)
        with open(pidfile, "w") as fh:
            fh.write(str(proc.pid))
        if point == "register":
            self._fire()

    def pop(self, *a):
        out = super().pop(*a)
        if point == "unregister":
            self._fire()
        return out

    def items(self):
        if point == "snapshot":
            self._fire()
        return super().items()

procs._LIVE = Tripwire()
signal.signal(signal.SIGTERM, handler)
if point == "snapshot":
    procs.terminate_all()  # as main()'s finally does when the server stops
else:
    procs.run(["sleep", "0" if point == "unregister" else "30"], timeout=60)
sys.exit(0)
"""


@pytest.mark.parametrize("server", ["delegate", "launcher"])
@pytest.mark.parametrize("point", ["register", "unregister", "snapshot"])
def test_a_stop_signal_while_the_main_thread_holds_the_job_lock_does_not_deadlock(
        tmp_path, server, point):
    # The first signal's handler calls terminate_all(), which takes the job
    # lock; the main thread may already hold it (registering, unregistering or
    # listing a job), and Python runs the handler on that same thread.
    driver, pidfile = tmp_path / "driver.py", tmp_path / "worker.pid"
    driver.write_text(SIGNAL_WHILE_LOCKED)
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": str(tmp_path),
           "PYTHONPATH": str(ROOT)}
    child = subprocess.Popen([sys.executable, str(driver), server, point, str(pidfile)],
                             env=env, cwd=tmp_path)
    worker = None
    try:
        try:
            code = child.wait(timeout=15)
        except subprocess.TimeoutExpired:
            code = None
        if pidfile.exists():
            worker = int(pidfile.read_text())
        assert code == 128 + signal.SIGTERM, "the stop handler deadlocked on the job lock"
        if worker is not None:
            assert not procs.members(worker, scope="session", children=False), \
                "the job registered when the signal arrived was left running"
    finally:
        child.kill()
        child.wait()
        if worker is not None:
            try:
                os.killpg(worker, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass


SIGNAL_BEFORE_REGISTERING = """
import os, signal, subprocess, sys
server, signame, pidfile, worker = sys.argv[1:5]

class Spawned(subprocess.Popen):
    # Sends the stop signal once the job's process exists but before
    # procs.run has registered it: as Popen returns, on the main thread,
    # which is where Python then runs the handler.
    fired = False

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        if not Spawned.fired:
            Spawned.fired = True
            with open(pidfile, "w") as fh:
                fh.write(str(self.pid))
            os.kill(os.getpid(), getattr(signal, "SIG" + signame))

subprocess.Popen = Spawned
if server == "delegate":
    from auto_router import delegate
    delegate.launcher_argv = lambda task, cwd, tier="cheap": [worker]
    sys.exit(delegate.main())
from auto_router import launcher
sys.exit(launcher.main(["--route", "worker", "--quiet", "--", "task"]))
"""


@pytest.mark.parametrize("server", ["delegate", "launcher"])
@pytest.mark.parametrize("signame", ["INT", "TERM", "HUP"])
def test_a_stop_signal_before_the_job_is_registered_still_ends_it(
        hermetic, tmp_path, server, signame):
    # The real server or route-run in a child process with its own handlers;
    # the signal lands between the job's start and its registration, where
    # neither the caller's cleanup nor terminate_all() could see the job.
    driver, pidfile = tmp_path / "driver.py", tmp_path / "worker.pid"
    driver.write_text(SIGNAL_BEFORE_REGISTERING)
    worker = _script(tmp_path / "w.sh", "exec sleep 30\n")
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": str(tmp_path),
           "PYTHONPATH": str(ROOT), "AUTO_ROUTER_DELEGATE_ROOT": str(hermetic),
           "AUTO_ROUTER_BENCH_OFFLINE": "1",
           "AUTO_ROUTER_CONFIG": str(_write_config(tmp_path, worker))}
    child = subprocess.Popen(
        [sys.executable, str(driver), server, signame, str(pidfile), worker],
        cwd=hermetic, env=env, text=True, stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    pid = None
    try:
        child.stdin.write(json.dumps({
            "jsonrpc": "2.0", "id": 1, "method": "tools/call",
            "params": {"name": "delegate", "arguments": {"task": "t"}}}) + "\n")
        child.stdin.close()
        code = child.wait(timeout=30)
        time.sleep(0.2)
        assert pidfile.exists(), "the job never started"
        pid = int(pidfile.read_text())
        assert not _alive([pid]), "the job started just before the signal was left running"
        sig = getattr(signal, "SIG" + signame)
        assert code == (-sig if sig == signal.SIGINT else 128 + sig)
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()
        if pid is not None:
            try:
                os.killpg(pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass


def test_starting_a_job_leaves_the_stop_handlers_as_they_were(tmp_path):
    def handler(signum, frame):
        pass

    previous = {sig: signal.signal(sig, handler) for sig in procs._STOP_SIGNALS}
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    try:
        assert procs.run(["true"], timeout=10).returncode == 0
        with pytest.raises(FileNotFoundError):
            procs.run([str(tmp_path / "missing")], timeout=10)
        assert signal.getsignal(signal.SIGINT) is handler
        assert signal.getsignal(signal.SIGTERM) is handler
        assert signal.getsignal(signal.SIGHUP) is signal.SIG_IGN
    finally:
        for sig, old in previous.items():
            signal.signal(sig, old)


LEFTOVER = """
# $1 marks, $2 the signals (comma-separated) the first SIGTERMs are answered
# with, one each, $3 the caller. Answers the cleanup's SIGTERMs with stop
# signals to the caller and otherwise ignores them, so only a SIGKILL ends it.
n=0
trap 'n=$((n+1)); s=$(echo "$2," | cut -d, -f$n); mkdir -p "$1/second"; [ -z "$s" ] || kill -$s $3' TERM
echo $$ > "$1/.p" && mv "$1/.p" "$1/leftover-pid"
while :; do sleep 0.05; done
"""

EXITING_JOB = """
# $1 marks, $2 signal: leave the leftover behind, then exit at once.
"$(dirname "$0")/leftover.sh" "$1" "$2" $PPID </dev/null >/dev/null 2>&1 &
while [ ! -e "$1/leftover-pid" ]; do sleep 0.02; done
"""

SIGNAL_DURING_LEFTOVER_CLEANUP = """
import sys
job, marks, signame, server = sys.argv[1:5]
if server == "delegate":
    from auto_router import delegate
    delegate.launcher_argv = lambda task, cwd, tier="cheap": [job, marks, signame]
    sys.exit(delegate.main())
from auto_router import launcher
sys.exit(launcher.main(["--route", "worker", "--quiet", "--", "task"]))
"""


def _exiting_job(tmp_path: Path) -> tuple[str, Path]:
    marks = tmp_path / "marks"
    marks.mkdir()
    _script(tmp_path / "leftover.sh", LEFTOVER)
    return _script(tmp_path / "job.sh", EXITING_JOB), marks


def _leftover_pid(marks: Path) -> int | None:
    path = marks / "leftover-pid"
    return int(path.read_text()) if path.exists() else None


def _kill_group(pid: int | None) -> None:
    for kill in (os.killpg, os.kill):
        try:
            if pid is not None:
                kill(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass


@pytest.mark.parametrize("server", ["delegate", "launcher"])
@pytest.mark.parametrize("signame", ["INT", "TERM", "HUP", "INT,TERM"])
def test_a_stop_signal_while_an_exited_jobs_leftover_is_stopped_still_ends_it(
        hermetic, tmp_path, server, signame):
    # The real server or route-run in a child process with its own handlers.
    # The job exits at once and leaves a process behind that only SIGKILL
    # ends; that process answers the leftover cleanup's SIGTERM with the stop
    # signal, so it lands after the job is gone, while its leftover is stopped.
    # "INT,TERM": a SIGTERM follows while the Ctrl-C's own cleanup runs.
    job, marks = _exiting_job(tmp_path)
    agent = _script(tmp_path / "agent.sh", f"exec {job} {marks} {signame}\n")
    driver = tmp_path / "driver.py"
    driver.write_text(SIGNAL_DURING_LEFTOVER_CLEANUP)
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": str(tmp_path),
           "PYTHONPATH": str(ROOT), "AUTO_ROUTER_DELEGATE_ROOT": str(hermetic),
           "AUTO_ROUTER_BENCH_OFFLINE": "1",
           "AUTO_ROUTER_CONFIG": str(_write_config(tmp_path, agent))}
    # Not part of the job: must come through untouched.
    bystander = subprocess.Popen(["sleep", "60"], start_new_session=True)
    child = subprocess.Popen(
        [sys.executable, str(driver), job, str(marks), signame, server],
        cwd=hermetic, env=env, text=True, stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        child.stdin.write(json.dumps({
            "jsonrpc": "2.0", "id": 1, "method": "tools/call",
            "params": {"name": "delegate", "arguments": {"task": "t"}}}) + "\n")
        child.stdin.close()
        code = child.wait(timeout=60)
        time.sleep(0.2)
        pid = _leftover_pid(marks)
        assert pid is not None, "the job never left its process behind"
        assert (marks / "second").exists(), "the stop signal never came"
        assert not _alive([pid]), "the leftover of the exited job was left running"
        assert bystander.poll() is None, "a process outside the job was signalled"
        # Both ignore a stop signal during a Ctrl-C's cleanup.
        sig = getattr(signal, "SIG" + signame.split(",")[0])
        assert code == (-sig if sig == signal.SIGINT else 128 + sig)
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()
        _kill_group(_leftover_pid(marks))
        bystander.kill()
        bystander.wait()


class _Stop(Exception):
    pass


@pytest.mark.parametrize("scope", ["session", "group"])
@pytest.mark.parametrize("handler_ends_jobs", [False, True])
def test_procs_run_ends_an_exited_jobs_leftover_when_a_stop_signal_interrupts(
        tmp_path, scope, handler_ends_jobs):
    # In-process: a SIGTERM handler that raises (like Ctrl-C), or that first
    # calls terminate_all() (like the servers' SIGTERM/SIGHUP handlers).
    job, marks = _exiting_job(tmp_path)

    def handler(signum, frame):
        if handler_ends_jobs:
            procs.terminate_all()
        raise _Stop

    bystander = subprocess.Popen(["sleep", "60"], start_new_session=True)
    previous = signal.signal(signal.SIGTERM, handler)
    try:
        with pytest.raises(_Stop):
            procs.run([job, str(marks), "TERM"], timeout=30, scope=scope, grace_s=1.0)
        assert (marks / "second").exists(), "the stop signal never came"
        assert not _alive([_leftover_pid(marks)]), "the leftover was left running"
        assert procs._LIVE == {}, "a finished job stayed registered"
        assert bystander.poll() is None, "a process outside the job was signalled"
    finally:
        signal.signal(signal.SIGTERM, previous)
        _kill_group(_leftover_pid(marks))
        bystander.kill()
        bystander.wait()


def test_procs_run_a_second_stop_that_ends_jobs_still_reaches_an_exited_jobs_leftover(tmp_path):
    # In-process: the first SIGTERM raises in the leftover cleanup, the second
    # one lands in the except-block's pass and ends the job via terminate_all(),
    # which sees it only because it is still registered.
    job, marks = _exiting_job(tmp_path)
    calls = []

    def handler(signum, frame):
        calls.append(signum)
        if len(calls) > 1:
            procs.terminate_all()
        raise _Stop

    bystander = subprocess.Popen(["sleep", "60"], start_new_session=True)
    previous = signal.signal(signal.SIGTERM, handler)
    try:
        with pytest.raises(_Stop):
            procs.run([job, str(marks), "TERM,TERM"], timeout=30, scope="group", grace_s=1.0)
        assert len(calls) == 2, "the second stop signal never came"
        assert not _alive([_leftover_pid(marks)]), "the leftover was left running"
        assert procs._LIVE == {}, "a finished job stayed registered"
        assert bystander.poll() is None, "a process outside the job was signalled"
    finally:
        signal.signal(signal.SIGTERM, previous)
        _kill_group(_leftover_pid(marks))
        bystander.kill()
        bystander.wait()


SLOW_TO_STOP_AGENT = """
# $1 marks, $2 the signal the first SIGTERM is answered with (- for none).
# Needs 1.2 s after a SIGTERM to shut down cleanly and marks when it has; it
# leaves a process in its group that only SIGKILL ends.
sh -c 'trap "" TERM; echo $$ > "$0/.l" && mv "$0/.l" "$0/leftover-pid"
       while :; do sleep 0.05; done' "$1" </dev/null >/dev/null 2>&1 &
trap 'trap "" TERM; mkdir "$1/term"; [ "$2" = - ] || kill -$2 $PPID
      sleep 1.2; touch "$1/clean"; exit 0' TERM
echo $$ > "$1/.a" && mv "$1/.a" "$1/agent-pid"
while :; do sleep 0.05; done
"""


@pytest.mark.parametrize("first, second", [
    ("INT", "TERM"), ("INT", "HUP"), ("INT", "-"), ("TERM", "-"), ("HUP", "-")])
def test_a_stop_signal_after_ctrl_c_does_not_cut_route_runs_cleanup_short(
        hermetic, tmp_path, first, second):
    # route-run in a child process with a real config and a live agent. The
    # test sends the first signal; the agent answers the cleanup's SIGTERM with
    # the second, so it lands while the agent is being stopped.
    marks = tmp_path / "marks"
    marks.mkdir()
    slow = _script(tmp_path / "slow.sh", SLOW_TO_STOP_AGENT)
    agent = _script(tmp_path / "agent.sh", f"exec {slow} {marks} {second}\n")
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": str(tmp_path),
           "PYTHONPATH": str(ROOT), "AUTO_ROUTER_CONFIG": str(_write_config(tmp_path, agent)),
           "AUTO_ROUTER_BENCH_OFFLINE": "1"}
    bystander = subprocess.Popen(["sleep", "60"], start_new_session=True)
    run = subprocess.Popen([sys.executable, "-m", "auto_router.launcher", "--route", "worker",
                            "--quiet", "--", "task"], cwd=hermetic, env=env,
                           stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL)
    pids = []
    try:
        deadline = time.monotonic() + 30
        while not ((marks / "agent-pid").exists() and (marks / "leftover-pid").exists()):
            assert time.monotonic() < deadline and run.poll() is None, "the agent never started"
            time.sleep(0.02)
        pids = [int((marks / name).read_text()) for name in ("agent-pid", "leftover-pid")]
        os.kill(run.pid, getattr(signal, "SIG" + first))
        code = run.wait(timeout=60)
        time.sleep(0.2)
        assert (marks / "term").exists(), "the agent was never asked to stop"
        assert not _alive(pids), "the agent or its leftover was left running"
        assert bystander.poll() is None, "a process outside the job was signalled"
        if first == "INT":
            # The Ctrl-C's grace period (procs.GRACE_S) was not cut short.
            assert (marks / "clean").exists(), "the agent's own shutdown was cut short"
        else:
            # Without a Ctrl-C a SIGTERM/SIGHUP still ends route-run promptly.
            assert not (marks / "clean").exists(), "the stop waited out the agent"
        sig = getattr(signal, "SIG" + first)
        assert code == (-sig if sig == signal.SIGINT else 128 + sig)
    finally:
        if run.poll() is None:
            run.kill()
            run.wait()
        for pid in pids:
            _kill_group(pid)
        bystander.kill()
        bystander.wait()


def test_route_runs_stop_handler_leaves_a_ctrl_c_cleanup_alone(monkeypatch):
    called = []
    monkeypatch.setattr(procs, "terminate_all", lambda: called.append(1))
    monkeypatch.setattr(launcher_mod, "_INTERRUPTED", threading.Event())
    with pytest.raises(SystemExit) as exc:
        launcher_mod._exit_on_signal(signal.SIGTERM, None)
    assert exc.value.code == 128 + signal.SIGTERM and called == [1]
    with pytest.raises(KeyboardInterrupt):
        launcher_mod._interrupt_once(signal.SIGINT, None)
    assert launcher_mod._exit_on_signal(signal.SIGTERM, None) is None
    assert launcher_mod._exit_on_signal(signal.SIGHUP, None) is None
    assert called == [1]


def test_a_stop_signal_after_ctrl_c_still_restores_the_callers_handlers(monkeypatch):
    # route-run called in-process: a Ctrl-C ends the run, and a SIGHUP/SIGTERM
    # arrives while main() is putting the caller's handlers back.
    def caller(signum, frame):
        pass

    real_signal = signal.signal
    previous = {sig: real_signal(sig, caller) for sig in procs._STOP_SIGNALS}
    fired = []

    def restoring(sig, handler):
        old = real_signal(sig, handler)
        if handler is caller and not fired:
            # The other stop signal still has route-run's handler here.
            fired.append(signal.SIGHUP if sig == signal.SIGTERM else signal.SIGTERM)
            signal.raise_signal(fired[0])
        return old

    monkeypatch.setattr(procs, "terminate_all", lambda: None)
    monkeypatch.setattr(launcher_mod, "_main", lambda args: signal.raise_signal(signal.SIGINT))
    monkeypatch.setattr(signal, "signal", restoring)
    try:
        with pytest.raises(KeyboardInterrupt):
            launcher_mod.main(["task"])
        assert fired, "no stop signal came during the restore"
        assert all(signal.getsignal(sig) is caller for sig in procs._STOP_SIGNALS)
    finally:
        for sig, old in previous.items():
            real_signal(sig, old)


def test_stopping_a_reaped_job_with_nothing_left_signals_nothing(monkeypatch):
    # Its pid is free again once no group or session holds it: a group signal
    # to it could reach whoever got the pid next.
    proc = subprocess.Popen(["true"], start_new_session=True)
    proc.wait()
    sent = []
    monkeypatch.setattr(os, "killpg", lambda *a: sent.append(("killpg", *a)))
    monkeypatch.setattr(os, "kill", lambda *a: sent.append(("kill", *a)))
    procs.terminate(proc, scope="session", grace_s=0.1)
    assert sent == []


def test_without_proc_a_reaped_jobs_group_is_still_signalled(monkeypatch):
    # No /proc: whether anything is left cannot be told, so the group signal
    # its leftovers depend on is still sent.
    proc = subprocess.Popen(["true"], start_new_session=True)
    proc.wait()
    sent = []
    monkeypatch.setattr(procs, "_table", lambda: None)
    monkeypatch.setattr(os, "killpg", lambda *a: sent.append(a))
    procs.terminate(proc, scope="session", grace_s=0.1)
    assert (proc.pid, signal.SIGTERM) in sent


def test_a_server_run_does_not_inherit_an_earlier_runs_stop(monkeypatch):
    # main() called again in the same process after a stop: queued briefs are
    # started again, and a stop signal still ends this run.
    stopping = threading.Event()
    stopping.set()
    monkeypatch.setattr(delegate, "_STOPPING", stopping)
    monkeypatch.setattr(procs, "terminate_all", lambda: None)
    seen = []

    def serve(stdin, stdout):
        seen.append(delegate._unless_stopping(lambda: {"ok": True}))
        delegate._exit_on_signal(signal.SIGTERM, None)
        return 0

    monkeypatch.setattr(delegate, "_serve", serve)
    with pytest.raises(SystemExit):
        delegate.main(io.StringIO(), io.StringIO())
    assert seen == [{"ok": True}]


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


STOPPABLE_WORKER = """
echo edited > "$2/edited.txt"
sleep 20 &
echo "$$ $!" > "$1/.pids" && mv "$1/.pids" "$1/started-$(basename "$2")"
wait
"""

SIGNALLED_SERVER = """
import sys
from auto_router import delegate
worker, marks = sys.argv[1], sys.argv[2]
delegate.launcher_argv = lambda task, cwd, tier="cheap": [worker, marks, cwd]
sys.exit(delegate.main())
"""


@pytest.mark.parametrize("signum, returncode", [
    (signal.SIGTERM, 128 + signal.SIGTERM),  # the handler's SystemExit
    (signal.SIGHUP, 128 + signal.SIGHUP),
    (signal.SIGINT, -signal.SIGINT),  # Ctrl-C: still a KeyboardInterrupt
])
def test_a_stop_signal_during_parallel_work_stops_every_worker_and_removes_the_copies(
        hermetic, tmp_path, signum, returncode):
    # The real server in a child process, the real run_delegate/procs path, a
    # fake worker: two briefs, one at a time, so the second is still queued
    # when the signal arrives during the first.
    (hermetic / "a.txt").write_text("a\n")
    marks, scratch_tmp = tmp_path / "marks", tmp_path / "tmp"
    marks.mkdir()
    scratch_tmp.mkdir()
    worker = _script(tmp_path / "w.sh", STOPPABLE_WORKER)
    driver = tmp_path / "server.py"
    driver.write_text(SIGNALLED_SERVER)
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": str(tmp_path),
           "PYTHONPATH": str(ROOT), "AUTO_ROUTER_DELEGATE_ROOT": str(hermetic),
           "AUTO_ROUTER_BENCH_OFFLINE": "1", "TMPDIR": str(scratch_tmp)}
    server = subprocess.Popen([sys.executable, str(driver), worker, str(marks)], cwd=hermetic,
                              env=env, text=True, stdin=subprocess.PIPE,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        server.stdin.write(json.dumps({
            "jsonrpc": "2.0", "id": 1, "method": "tools/call",
            "params": {"name": "delegate_many",
                       "arguments": {"tasks": ["one", "two"], "parallel": 1}}}) + "\n")
        server.stdin.flush()
        deadline = time.monotonic() + 30
        while not list(marks.glob("started-*")) and time.monotonic() < deadline:
            time.sleep(0.05)
        first = sorted(p.name for p in marks.glob("started-*"))
        assert len(first) == 1, "the first worker never started"
        assert list(scratch_tmp.glob("auto-router-delegate-*/worker-*/edited.txt"))
        server.send_signal(signum)
        out, _ = server.communicate(timeout=60)
    finally:
        if server.poll() is None:
            server.kill()
            server.wait()
    # The exit status still names the signal, and no reply claims a result.
    assert server.returncode == returncode
    assert out == ""
    # The queued brief never got a worker, and the one that ran is gone with
    # its background child: nothing was alive when the copies were removed.
    assert sorted(p.name for p in marks.glob("started-*")) == first
    pids = [int(p) for f in marks.glob("started-*") for p in f.read_text().split()]
    assert len(pids) == 2 and not _alive(pids)
    assert not list(scratch_tmp.iterdir()), "the worker copies were left behind"


INTERRUPTING_WORKER = """
# $1 marks, $2 lead|quiet, $3 workers to wait for, $4 the signal a SIGTERM answers with
echo $$ > "$1/.p$$" && mv "$1/.p$$" "$1/pid-$$"
# The cleanup's SIGTERM is answered once (by whichever worker gets it first)
# with a second signal to the process cleaning up, and otherwise ignored, so
# only a SIGKILL ends this worker.
trap 'if mkdir "$1/second" 2>/dev/null; then kill -$4 $PPID; fi' TERM
if [ "$2" = lead ]; then
    while [ "$(ls "$1" | grep -c '^pid-')" -lt "$3" ]; do sleep 0.02; done
    sleep 0.3
    kill -INT $PPID  # the first Ctrl-C
fi
while :; do sleep 0.05; done
"""

INTERRUPTED_SERVER = """
import sys
from auto_router import delegate
worker, marks, second = sys.argv[1], sys.argv[2], sys.argv[3]
workers = sys.argv[4]
delegate.launcher_argv = lambda task, cwd, tier="cheap": [worker, marks, task, workers, second]
sys.exit(delegate.main())
"""


def _worker_pids(marks: Path) -> list[int]:
    return [int(p.name[4:]) for p in marks.glob("pid-*")]


def _kill_workers(marks: Path) -> None:
    for pid in _worker_pids(marks):
        for kill in (os.killpg, os.kill):
            try:
                kill(pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass


@pytest.mark.parametrize("tool, second", [
    ("delegate_many", "INT"), ("delegate_many", "TERM"), ("delegate", "INT")])
def test_a_second_signal_after_ctrl_c_does_not_cut_the_servers_cleanup_short(
        hermetic, tmp_path, tool, second):
    # The real server in a child process, the real run_many/run_delegate/procs
    # path, workers that only SIGKILL ends. The first Ctrl-C comes from a
    # worker; the second signal from the worker that the cleanup's SIGTERM
    # reaches first, so it lands while the cleanup is under way.
    (hermetic / "a.txt").write_text("a\n")
    marks, scratch_tmp = tmp_path / "marks", tmp_path / "tmp"
    marks.mkdir()
    scratch_tmp.mkdir()
    worker = _script(tmp_path / "w.sh", INTERRUPTING_WORKER)
    driver = tmp_path / "server.py"
    driver.write_text(INTERRUPTED_SERVER)
    arguments = ({"tasks": ["lead", "quiet"], "parallel": 2} if tool == "delegate_many"
                 else {"task": "lead"})
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": str(tmp_path),
           "PYTHONPATH": str(ROOT), "AUTO_ROUTER_DELEGATE_ROOT": str(hermetic),
           "AUTO_ROUTER_BENCH_OFFLINE": "1", "TMPDIR": str(scratch_tmp)}
    server = subprocess.Popen(
        [sys.executable, str(driver), worker, str(marks), second,
         "2" if tool == "delegate_many" else "1"],
        cwd=hermetic, env=env, text=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE)
    try:
        server.stdin.write(json.dumps({
            "jsonrpc": "2.0", "id": 1, "method": "tools/call",
            "params": {"name": tool, "arguments": arguments}}) + "\n")
        server.stdin.flush()
        out, _ = server.communicate(timeout=60)
        time.sleep(0.2)
        pids = _worker_pids(marks)
        assert (marks / "second").exists(), "the second signal never came"
        assert len(pids) == (2 if tool == "delegate_many" else 1)
        assert not _alive(pids), "a worker was left running"
        assert not list(scratch_tmp.iterdir()), "the worker copies were left behind"
        assert server.returncode == -signal.SIGINT and out == ""
    finally:
        if server.poll() is None:
            server.kill()
            server.wait()
        _kill_workers(marks)


def test_a_second_ctrl_c_does_not_cut_route_runs_cleanup_short(hermetic, tmp_path):
    # route-run in a child process with a real config; the agent sends the
    # first Ctrl-C and answers the cleanup's SIGTERM with the second one.
    marks = tmp_path / "marks"
    marks.mkdir()
    worker = _script(tmp_path / "w.sh", INTERRUPTING_WORKER)
    agent = _script(tmp_path / "agent.sh", f'exec {worker} {marks} lead 1 INT\n')
    cfg = _write_config(tmp_path, agent)
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": str(tmp_path),
           "PYTHONPATH": str(ROOT), "AUTO_ROUTER_CONFIG": str(cfg),
           "AUTO_ROUTER_BENCH_OFFLINE": "1"}
    run = subprocess.Popen([sys.executable, "-m", "auto_router.launcher", "--route", "worker",
                            "--quiet", "--", "task"], cwd=hermetic, env=env,
                           stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL)  # an agent left running holds pipes open
    try:
        run.wait(timeout=60)
        time.sleep(0.2)
        pids = _worker_pids(marks)
        assert (marks / "second").exists(), "the second Ctrl-C never came"
        assert len(pids) == 1
        assert not _alive(pids), "the agent was left running"
        assert run.returncode == -signal.SIGINT
    finally:
        if run.poll() is None:
            run.kill()
            run.wait()
        _kill_workers(marks)


def test_ctrl_c_is_a_keyboard_interrupt_once_and_then_ignored(monkeypatch):
    stopping = threading.Event()
    monkeypatch.setattr(delegate, "_STOPPING", stopping)
    called = []
    monkeypatch.setattr(procs, "terminate_all", lambda: called.append(1))
    with pytest.raises(KeyboardInterrupt):
        delegate._exit_on_signal(signal.SIGINT, None)
    assert stopping.is_set() and not called, "the interrupted code does its own cleanup"
    assert delegate._exit_on_signal(signal.SIGINT, None) is None
    assert delegate._exit_on_signal(signal.SIGTERM, None) is None
    assert called == []
    monkeypatch.setattr(launcher_mod, "_INTERRUPTED", threading.Event())
    with pytest.raises(KeyboardInterrupt):
        launcher_mod._interrupt_once(signal.SIGINT, None)
    assert launcher_mod._interrupt_once(signal.SIGINT, None) is None


@pytest.mark.parametrize("server", ["delegate", "launcher"])
def test_ctrl_c_the_process_was_started_to_ignore_stays_ignored(monkeypatch, server):
    seen = []

    def body(*_a):
        seen.append(signal.getsignal(signal.SIGINT))
        return 0

    previous = signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        if server == "delegate":
            monkeypatch.setattr(delegate, "_serve", body)
            delegate.main(io.StringIO(), io.StringIO())
        else:
            monkeypatch.setattr(launcher_mod, "_main", body)
            launcher_mod.main(["--list"])
        assert seen == [signal.SIG_IGN]
        assert signal.getsignal(signal.SIGINT) is signal.SIG_IGN
    finally:
        signal.signal(signal.SIGINT, previous)


def test_an_interrupted_run_stops_its_other_workers_before_removing_the_copies(
        hermetic, tmp_path, monkeypatch):
    (hermetic / "a.txt").write_text("a\n")
    monkeypatch.setattr(delegate.tempfile, "tempdir", str(tmp_path))
    sleeper = _script(tmp_path / "sleep.sh", 'echo "$$" > "$1"\nexec sleep 30\n')
    pid_file = tmp_path / "pid"
    alive_at_cleanup = []
    real_rmtree = delegate.shutil.rmtree

    def rmtree(path, *a, **k):
        alive_at_cleanup.append(bool(_alive([int(pid_file.read_text())])))
        return real_rmtree(path, *a, **k)

    def runner(task, context, cwd, tier, timeout_s):
        if task == "sleeps":
            return {"ok": True, "p": procs.run([sleeper, str(pid_file)], timeout=60).returncode}
        while not pid_file.exists() or not pid_file.read_text().strip():
            time.sleep(0.02)
        raise KeyboardInterrupt  # Ctrl-C arriving while the other worker runs

    monkeypatch.setattr(delegate.shutil, "rmtree", rmtree)
    with pytest.raises(KeyboardInterrupt):
        delegate.run_many(["sleeps", "interrupted"], cwd=str(hermetic), parallel=2, runner=runner)
    # The copies go one directory at a time (the owner record last); the
    # worker was gone before the first of them.
    assert alive_at_cleanup and not any(alive_at_cleanup)
    assert not _alive([int(pid_file.read_text())])
    assert not list(tmp_path.glob("auto-router-delegate-*"))


def test_ctrl_c_with_briefs_still_queued_cleans_up_without_waiting_for_them(
        hermetic, tmp_path, monkeypatch):
    # Ctrl-C reaches the main thread while it waits; one worker runs, one
    # brief is queued. The cancelled brief must neither start nor hold up the
    # cleanup until STOP_WAIT_S runs out.
    (hermetic / "a.txt").write_text("a\n")
    monkeypatch.setattr(delegate.tempfile, "tempdir", str(tmp_path))
    sleeper = _script(tmp_path / "sleep.sh", 'echo "$$" > "$1"\nexec sleep 30\n')
    pid_file = tmp_path / "pid"
    ran = []

    def runner(task, context, cwd, tier, timeout_s):
        ran.append(task)
        return {"ok": True, "p": procs.run([sleeper, str(pid_file)], timeout=60).returncode}

    def interrupted(futures):
        while not pid_file.exists() or not pid_file.read_text().strip():
            time.sleep(0.02)
        raise KeyboardInterrupt
        yield  # pragma: no cover - a generator, like as_completed

    monkeypatch.setattr(delegate, "as_completed", interrupted)
    began = time.monotonic()
    with pytest.raises(KeyboardInterrupt):
        delegate.run_many(["runs", "queued"], cwd=str(hermetic), parallel=1, runner=runner)
    assert time.monotonic() - began < delegate.STOP_WAIT_S / 3
    assert ran == ["runs"]
    assert not _alive([int(pid_file.read_text())])
    assert not list(tmp_path.glob("auto-router-delegate-*"))


def test_copies_a_worker_may_still_write_to_are_not_deleted(hermetic, tmp_path, monkeypatch):
    (hermetic / "a.txt").write_text("a\n")
    monkeypatch.setattr(delegate.tempfile, "tempdir", str(tmp_path))
    monkeypatch.setattr(delegate, "STOP_WAIT_S", 0.3)
    release, stuck = threading.Event(), threading.Event()

    def runner(task, context, cwd, tier, timeout_s):
        if task == "stuck":  # not under procs.run, so nothing can stop it
            stuck.set()
            release.wait(30)
            return {"ok": True}
        stuck.wait(30)
        raise KeyboardInterrupt

    try:
        with pytest.raises(KeyboardInterrupt):
            delegate.run_many(["stuck", "interrupted"], cwd=str(hermetic), parallel=2,
                              runner=runner)
        assert list(tmp_path.glob("auto-router-delegate-*/worker-1/a.txt"))
    finally:
        release.set()
        for leftover in tmp_path.glob("auto-router-delegate-*"):
            shutil.rmtree(leftover)


def test_a_tree_too_large_to_copy_is_refused(hermetic, monkeypatch):
    (hermetic / "big.bin").write_bytes(b"x" * 2048)
    monkeypatch.setattr(delegate, "COPY_LIMIT_BYTES", 1024)
    result = delegate.run_many(["one", "two"], cwd=str(hermetic), runner=lambda *a: {"ok": True})
    assert not result["ok"] and "too large" in result["error"]


# --------------------------------------------------------------------------
# Symlinks in per-worker copies (known limit of b7bda45, closed here)
# --------------------------------------------------------------------------
def _links_project(hermetic: Path) -> Path:
    """A project whose links point inside and outside it; returns the outside file."""
    outside = hermetic.parent / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("outside\n")
    (hermetic / "shared.txt").write_text("original\n")
    (hermetic / "sub").mkdir()
    (hermetic / "sub" / "real.txt").write_text("real\n")
    (hermetic / "abs_link").symlink_to(outside / "secret.txt")          # absolute, outside
    (hermetic / "up_link").symlink_to("../outside/secret.txt")          # .. climbs out
    (hermetic / "dir_up").symlink_to("../outside")                      # directory, outside
    (hermetic / "chain").symlink_to("abs_link")                         # in-tree hop, ends outside
    (hermetic / "abs_inside").symlink_to(hermetic / "shared.txt")       # absolute into the original
    (hermetic / "sneaky").symlink_to(f"../{hermetic.name}/shared.txt")  # out and back by name
    (hermetic / "in_tree").symlink_to("sub/real.txt")                   # safe
    (hermetic / "sub" / "back").symlink_to("../shared.txt")             # safe, climbs within
    (hermetic / "sub_link").symlink_to("sub")                           # safe directory link
    return outside / "secret.txt"


ESCAPING = ["abs_inside", "abs_link", "chain", "dir_up", "sneaky", "up_link"]


def test_a_copy_keeps_safe_links_and_drops_links_that_leave_it(hermetic, tmp_path):
    _links_project(hermetic)
    skipped = delegate._copy(hermetic, tmp_path / "copy")
    copy = tmp_path / "copy"
    assert [line.split(" ")[0] for line in skipped] == ESCAPING
    assert all("leads outside the copy" in line for line in skipped)
    for name in ESCAPING:
        assert not os.path.lexists(copy / name)
    assert os.readlink(copy / "in_tree") == "sub/real.txt"
    assert os.readlink(copy / "sub" / "back") == "../shared.txt"
    assert os.readlink(copy / "sub_link") == "sub"
    assert (copy / "in_tree").read_text() == "real\n"
    assert (copy / "sub_link" / "real.txt").resolve() == (copy / "sub" / "real.txt").resolve()
    assert (copy / "shared.txt").read_text() == "original\n" and not (copy / "shared.txt").is_symlink()


def test_a_worker_writing_through_links_in_its_copy_cannot_reach_outside(hermetic):
    secret = _links_project(hermetic)

    def worker(task, context, cwd, tier, timeout_s):
        for name in ["dir_up/secret.txt"] + ESCAPING + ["in_tree", "sub/back"]:
            try:  # follows a link if one is there
                Path(cwd, name).write_text(f"written by {task}\n")
            except (FileNotFoundError, NotADirectoryError):
                pass  # a relative link that dangles from the copy's location
        return {"ok": True}

    result = delegate.run_many(["one", "two"], cwd=str(hermetic), runner=worker)
    assert result["ok"], result
    assert secret.read_text() == "outside\n"
    assert sorted(p.name for p in secret.parent.iterdir()) == ["secret.txt"]
    assert (hermetic / "shared.txt").read_text() == "original\n"
    assert (hermetic / "sub" / "real.txt").read_text() == "real\n"
    assert [line.split(" ")[0] for line in result["copy_skipped"]] == ESCAPING
    for item in result["results"]:
        work = Path(item["workspace"])
        # the safe links still work inside the copy, and the edits land there
        assert (work / "sub" / "real.txt").read_text() == "written by " + ("one\n" if work.name == "worker-1" else "two\n")
        assert (work / "shared.txt").read_text() == (work / "sub" / "real.txt").read_text()
        assert set(ESCAPING) <= set(item["changes"]["added"])
        assert item["changes"]["modified"] == ["shared.txt", "sub/real.txt"]
    shutil.rmtree(Path(result["results"][0]["workspace"]).parent, ignore_errors=True)


def test_writing_through_every_name_in_a_copy_beside_the_original_stays_inside(hermetic, tmp_path):
    """The copy sits at the original's depth, so ``..`` links would really reach ``outside``."""
    secret = _links_project(hermetic)
    copy = tmp_path / "copy"
    delegate._copy(hermetic, copy)
    for name in ["dir_up/secret.txt"] + ESCAPING + ["in_tree", "sub/back"]:
        try:
            (copy / name).write_text("written in the copy\n")
        except (FileNotFoundError, NotADirectoryError):
            pass
    assert secret.read_text() == "outside\n"
    assert sorted(p.name for p in secret.parent.iterdir()) == ["secret.txt"]
    assert (hermetic / "shared.txt").read_text() == "original\n"
    assert (hermetic / "sub" / "real.txt").read_text() == "real\n"
    for name in ESCAPING:  # each became a plain file of the copy
        assert (copy / name).is_file() and not (copy / name).is_symlink()
    assert (copy / "sub" / "real.txt").read_text() == "written in the copy\n"


def test_a_file_swapped_for_a_link_during_the_copy_is_not_followed(hermetic, tmp_path, monkeypatch):
    secret = _links_project(hermetic)
    (hermetic / "victim.txt").write_text("victim\n")
    real_open = os.open

    def racing_open(path, flags, *args, **kwargs):
        if path == "victim.txt":  # the swap lands between listing and opening
            os.unlink(hermetic / "victim.txt")
            os.symlink(secret, hermetic / "victim.txt")
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(delegate.os, "open", racing_open)
    skipped = delegate._copy(hermetic, tmp_path / "copy")
    assert "victim.txt (changed or unreadable during the copy)" in skipped
    assert not os.path.lexists(tmp_path / "copy" / "victim.txt")


def test_fifos_are_skipped_and_do_not_block_the_copy_or_the_diff(hermetic, tmp_path):
    (hermetic / "a.txt").write_text("a\n")
    os.mkfifo(hermetic / "pipe")
    skipped = delegate._copy(hermetic, tmp_path / "base")
    assert skipped == ["pipe (not a regular file)"]
    delegate._copy(tmp_path / "base", tmp_path / "work")
    os.mkfifo(tmp_path / "work" / "worker-pipe")
    changes = delegate.diff_trees(tmp_path / "base", tmp_path / "work")
    assert changes["added"] == ["worker-pipe"] and changes["binary_changed"] == ["worker-pipe"]


def test_a_directory_link_a_worker_adds_is_reported_not_hidden(hermetic, tmp_path):
    """os.walk lists a link to a directory as a directory and never enters it;
    the diff must still show the link, most of all one that leads out of the copy."""
    outside = tmp_path / "outside"
    outside.mkdir()
    (hermetic / "a.txt").write_text("a\n")
    (hermetic / "sub").mkdir()
    (hermetic / "sub" / "f.txt").write_text("f\n")
    (hermetic / "sub_link").symlink_to("sub")
    delegate._copy(hermetic, tmp_path / "base")
    delegate._copy(tmp_path / "base", tmp_path / "work")
    work = tmp_path / "work"
    (work / "escape").symlink_to(outside)              # added, absolute, outside
    (work / "sub_link").unlink()
    (work / "sub_link").symlink_to("../outside")       # retargeted out of the copy
    shutil.rmtree(work / "sub")
    (work / "sub").symlink_to("sub_link")              # a directory replaced by a link
    changes = delegate.diff_trees(tmp_path / "base", work)
    assert changes["added"] == ["escape", "sub"]
    assert changes["modified"] == ["sub_link"]
    assert changes["deleted"] == ["sub/f.txt"]
    assert f"+-> {outside}" in changes["patch"]
    assert "--> sub\n+-> ../outside" in changes["patch"] and "+-> sub_link" in changes["patch"]
    assert changes["links_leaving_copy"] == ["escape", "sub", "sub_link"]


def test_a_copy_whose_only_change_is_a_directory_link_is_kept_and_flagged(hermetic, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (hermetic / "a.txt").write_text("a\n")

    def worker(task, context, cwd, tier, timeout_s):
        if task == "link":
            Path(cwd, "home").symlink_to(outside)
        return {"ok": True}

    result = delegate.run_many(["link", "idle"], cwd=str(hermetic), runner=worker)
    linked, idle = result["results"]
    assert linked["workspace"] and linked["changes"]["added"] == ["home"]
    assert linked["changes"]["links_leaving_copy"] == ["home"]
    assert idle["workspace"] is None and idle["changes"]["links_leaving_copy"] == []
    assert not os.path.lexists(hermetic / "home")
    shutil.rmtree(Path(linked["workspace"]).parent, ignore_errors=True)


def test_unchanged_links_in_a_copy_are_neither_changes_nor_flagged(hermetic, tmp_path):
    _links_project(hermetic)
    delegate._copy(hermetic, tmp_path / "base")
    delegate._copy(tmp_path / "base", tmp_path / "work")
    changes = delegate.diff_trees(tmp_path / "base", tmp_path / "work")
    assert changes["added"] == changes["modified"] == changes["deleted"] == []
    assert changes["links_leaving_copy"] == [] and changes["patch"] == ""


def test_a_huge_file_a_worker_writes_is_listed_but_never_read_into_memory(hermetic, tmp_path):
    """The copy is capped on the way in; what a worker writes into it was read whole."""
    import tracemalloc
    (hermetic / "a.txt").write_text("a\n")
    (hermetic / "grown.log").write_text("line\n" * 1000)
    delegate._copy(hermetic, tmp_path / "base")
    delegate._copy(tmp_path / "base", tmp_path / "work")
    work = tmp_path / "work"
    (work / "huge.log").write_text("x" * 99 + "\n" * 1 + ("y" * 99 + "\n") * 80_000)  # 8 MB
    with (work / "grown.log").open("a") as out:
        out.write("z" * 8_000_000 + "\n")
    (work / "a.txt").write_text("b\n")
    tracemalloc.start()
    try:
        changes = delegate.diff_trees(tmp_path / "base", work)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert changes["added"] == ["huge.log"] and changes["modified"] == ["a.txt", "grown.log"]
    assert changes["not_diffed"] == ["grown.log", "huge.log"]
    assert "-a\n+b\n" in changes["patch"] and "huge.log" not in changes["patch"]
    assert "grown.log" not in changes["patch"]
    assert peak < 4_000_000, peak


def test_the_patch_stops_growing_once_its_budget_is_spent(hermetic, tmp_path, monkeypatch):
    for i in range(40):
        (hermetic / f"f{i:02}.txt").write_text("old\n")
    delegate._copy(hermetic, tmp_path / "base")
    delegate._copy(tmp_path / "base", tmp_path / "work")
    for i in range(40):
        (tmp_path / "work" / f"f{i:02}.txt").write_text("new " * 250 + "\n")
    calls = []
    real = delegate.difflib.unified_diff
    monkeypatch.setattr(delegate.difflib, "unified_diff",
                        lambda *a, **k: calls.append(a[2]) or real(*a, **k))
    changes = delegate.diff_trees(tmp_path / "base", tmp_path / "work", limit=5000)
    assert len(changes["modified"]) == 40 and changes["patch_truncated"]
    assert len(changes["patch"]) == 5000
    assert len(calls) < 10, len(calls)  # later files are listed, not read and diffed
    assert changes["not_diffed"] == [f"f{i:02}.txt" for i in range(len(calls), 40)]


def test_a_worker_copy_holding_a_huge_file_is_kept_and_reported(hermetic):
    (hermetic / "a.txt").write_text("a\n")

    def worker(task, context, cwd, tier, timeout_s):
        if task == "big":
            Path(cwd, "dump.txt").write_text("d" * 3_000_000)
        return {"ok": True}

    result = delegate.run_many(["big", "idle"], cwd=str(hermetic), runner=worker)
    big, idle = result["results"]
    assert big["workspace"] and big["changes"]["added"] == ["dump.txt"]
    assert big["changes"]["not_diffed"] == ["dump.txt"] and big["changes"]["patch"] == ""
    assert idle["workspace"] is None and idle["changes"]["not_diffed"] == []
    shutil.rmtree(Path(big["workspace"]).parent, ignore_errors=True)


def test_a_read_only_source_gives_an_editable_copy(hermetic, tmp_path):
    (hermetic / "sub").mkdir()
    (hermetic / "sub" / "f.txt").write_text("f\n")
    (hermetic / "sub" / "f.txt").chmod(0o444)
    (hermetic / "sub").chmod(0o555)
    try:
        delegate._copy(hermetic, tmp_path / "copy")
    finally:
        (hermetic / "sub").chmod(0o755)
    (tmp_path / "copy" / "sub" / "f.txt").write_text("edited\n")
    (tmp_path / "copy" / "sub" / "new.txt").write_text("new\n")
    shutil.rmtree(tmp_path / "copy")


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


# --------------------------------------------------------------------------
# Copies left by a server that died (SIGKILL) or kept for a worker it could
# not stop are reclaimed by a later server once no one can write to them.

WRITING_WORKER = """
# $1 marks, $2 the copy it works in: keeps writing there until killed
echo $$ > "$1/.p$$" && mv "$1/.p$$" "$1/pid-$$"
while :; do date +%s%N > "$2/written.txt"; sleep 0.05; done
"""

WRITING_SERVER = """
import sys
from auto_router import delegate
worker, marks = sys.argv[1], sys.argv[2]
delegate.launcher_argv = lambda task, cwd, tier="cheap": [worker, marks, cwd]
sys.exit(delegate.main())
"""


def _server_env(hermetic: Path, tmp_path: Path, scratch_tmp: Path) -> dict:
    return {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": str(tmp_path),
            "PYTHONPATH": str(ROOT), "AUTO_ROUTER_DELEGATE_ROOT": str(hermetic),
            "AUTO_ROUTER_BENCH_OFFLINE": "1", "TMPDIR": str(scratch_tmp)}


def _killed_server_copies(hermetic: Path, tmp_path: Path) -> tuple[Path, Path, dict]:
    """Start a real server on two briefs of a fake worker that writes into its
    copy until killed, SIGKILL the server once both write, return the copies."""
    (hermetic / "a.txt").write_text("a\n")
    marks, scratch_tmp = tmp_path / "marks", tmp_path / "tmp"
    marks.mkdir()
    scratch_tmp.mkdir()
    worker = _script(tmp_path / "w.sh", WRITING_WORKER)
    driver = tmp_path / "server.py"
    driver.write_text(WRITING_SERVER)
    env = _server_env(hermetic, tmp_path, scratch_tmp)
    server = subprocess.Popen([sys.executable, str(driver), worker, str(marks)], cwd=hermetic,
                              env=env, text=True, stdin=subprocess.PIPE,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        server.stdin.write(json.dumps({
            "jsonrpc": "2.0", "id": 1, "method": "tools/call",
            "params": {"name": "delegate_many",
                       "arguments": {"tasks": ["one", "two"], "parallel": 2}}}) + "\n")
        server.stdin.flush()
        deadline = time.monotonic() + 30
        while (len(list(scratch_tmp.glob("auto-router-delegate-*/worker-*/written.txt"))) < 2
               and time.monotonic() < deadline):
            time.sleep(0.05)
        server.kill()  # SIGKILL: no cleanup of any kind runs
    finally:
        if server.poll() is None:
            server.kill()
        server.communicate(timeout=30)
    copies = list(scratch_tmp.glob("auto-router-delegate-*"))
    assert len(copies) == 1 and len(_worker_pids(marks)) == 2
    return copies[0], marks, env


def _wait_gone(pids: list[int]) -> None:
    """Wait until none of ``pids`` is a live process (reaped or a zombie)."""
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        states = []
        for pid in pids:
            try:
                states.append(Path(f"/proc/{pid}/stat").read_text().split(") ")[1][0])
            except (OSError, IndexError):
                pass
        if all(state in "ZX" for state in states):
            return
        time.sleep(0.05)
    raise AssertionError(f"still running: {pids}")


def _start_server(hermetic: Path, env: dict) -> str:
    """Start a fresh server, let it serve nothing, return what it said on stderr."""
    proc = subprocess.run([sys.executable, "-m", "auto_router.delegate"], cwd=hermetic, env=env,
                          input="", capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stderr
    return proc.stderr


def test_copies_of_a_killed_server_are_reclaimed_once_their_workers_are_gone(
        hermetic, tmp_path):
    scratch, marks, env = _killed_server_copies(hermetic, tmp_path)
    try:
        _kill_workers(marks)
        _wait_gone(_worker_pids(marks))
        assert not _alive(_worker_pids(marks))
        assert (scratch / "worker-1" / "written.txt").exists(), "the killed server left its copies"
        said = _start_server(hermetic, env)
        assert not scratch.exists(), "a later server did not reclaim the dead server's copies"
        assert f"removed {scratch}" in said
    finally:
        _kill_workers(marks)


def test_copies_a_worker_of_a_killed_server_still_writes_to_are_kept(hermetic, tmp_path):
    scratch, marks, env = _killed_server_copies(hermetic, tmp_path)
    try:
        # The workers outlived the server (their sessions are their own) and
        # keep writing into their copies: a later server must leave them be.
        assert len(_alive(_worker_pids(marks))) == 2
        said = _start_server(hermetic, env)
        assert (scratch / "worker-1" / "a.txt").exists() and (scratch / "worker-2").is_dir()
        assert f"kept {scratch}" in said and "in use by pid" in said
        before = (scratch / "worker-1" / "written.txt").read_text()
        time.sleep(0.3)
        assert (scratch / "worker-1" / "written.txt").read_text() != before, "worker stopped"
        # Once they are gone, the next server reclaims the copies.
        _kill_workers(marks)
        _wait_gone(_worker_pids(marks))
        _start_server(hermetic, env)
        assert not scratch.exists()
    finally:
        _kill_workers(marks)


def _dead_owner() -> tuple[int, int]:
    """The pid and start time of a process that has ended."""
    proc = subprocess.Popen(["sleep", "30"])
    start = procs.start_time(proc.pid)
    proc.kill()
    proc.wait()
    return proc.pid, start


def _left_copy(tmp_path: Path, name: str = "auto-router-delegate-left", **record) -> Path:
    scratch = tmp_path / name
    (scratch / "worker-1").mkdir(parents=True)
    (scratch / "worker-1" / "a.txt").write_text("a\n")
    pid, start = _dead_owner()
    fields = {"path": str(scratch), "pid": pid, "start": start,
              "boot_id": delegate._boot_id(), "pid_ns": delegate._pid_ns(), **record}
    (scratch / delegate.OWNER_RECORD).write_text(json.dumps(fields))
    return scratch


def test_a_copy_whose_server_and_workers_are_gone_is_reclaimed(tmp_path):
    scratch = _left_copy(tmp_path)
    assert delegate.sweep_copies(str(tmp_path)) == [
        f"removed {scratch}: its server and every worker had ended"]
    assert not scratch.exists()


def test_copies_kept_for_an_unstoppable_worker_stay_while_their_server_runs(
        hermetic, tmp_path, monkeypatch):
    # Same run as test_copies_a_worker_may_still_write_to_are_not_deleted: the
    # stuck worker is a thread of this very process, the server, which the
    # process scan cannot tell apart from the sweeping server. Only the owner
    # record's pid and start time keep these copies.
    (hermetic / "a.txt").write_text("a\n")
    monkeypatch.setattr(delegate.tempfile, "tempdir", str(tmp_path))
    monkeypatch.setattr(delegate, "STOP_WAIT_S", 0.3)
    release, stuck = threading.Event(), threading.Event()

    def runner(task, context, cwd, tier, timeout_s):
        if task == "stuck":
            stuck.set()
            release.wait(30)
            return {"ok": True}
        stuck.wait(30)
        raise KeyboardInterrupt

    try:
        with pytest.raises(KeyboardInterrupt):
            delegate.run_many(["stuck", "interrupted"], cwd=str(hermetic), parallel=2,
                              runner=runner)
        [scratch] = tmp_path.glob("auto-router-delegate-*")
        assert delegate.sweep_copies(str(tmp_path)) == [
            f"kept {scratch}: its server (pid {os.getpid()}) is still running"]
        assert (scratch / "worker-1" / "a.txt").exists()
        # The record the server left names it; once the server is gone and
        # nothing uses the copies, they are reclaimed.
        record = json.loads((scratch / delegate.OWNER_RECORD).read_text())
        assert record["pid"] == os.getpid() and record["path"] == str(scratch)
        record["pid"], record["start"] = _dead_owner()
        (scratch / delegate.OWNER_RECORD).write_text(json.dumps(record))
        release.set()
        assert delegate.sweep_copies(str(tmp_path))[0].startswith("removed")
        assert not scratch.exists()
    finally:
        release.set()
        for leftover in tmp_path.glob("auto-router-delegate-*"):
            shutil.rmtree(leftover)


def test_the_sweep_never_deletes_what_it_cannot_prove_is_an_orphan(tmp_path):
    untagged = tmp_path / "auto-router-delegate-untagged"
    (untagged / "worker-1").mkdir(parents=True)
    garbled = _left_copy(tmp_path, "auto-router-delegate-garbled")
    (garbled / delegate.OWNER_RECORD).write_text('{"pid": ')
    moved = _left_copy(tmp_path, "auto-router-delegate-moved", path=str(tmp_path / "elsewhere"))
    other_boot = _left_copy(tmp_path, "auto-router-delegate-boot", boot_id="another-boot")
    other_ns = _left_copy(tmp_path, "auto-router-delegate-ns", pid_ns="pid:[1]")
    no_start = _left_copy(tmp_path, "auto-router-delegate-nostart", start=None)
    linked = tmp_path / "auto-router-delegate-link"
    linked.symlink_to(_left_copy(tmp_path, "not-ours"))
    foreign = _left_copy(tmp_path, "someone-elses-dir")
    lines = delegate.sweep_copies(str(tmp_path))
    kept = {line.split(":")[0] for line in lines}
    assert kept == {f"kept {p}" for p in (untagged, garbled, moved, other_boot, other_ns,
                                          no_start, linked)}
    for path in (untagged, garbled, moved, other_boot, other_ns, no_start, linked, foreign,
                 tmp_path / "not-ours"):
        assert path.exists()
    assert any("no owner record" in line for line in lines)
    assert any("before the last reboot" in line for line in lines)
    assert any("another pid namespace" in line for line in lines)


def test_copies_a_result_handed_over_are_never_reclaimed(hermetic, tmp_path, monkeypatch):
    monkeypatch.setattr(delegate.tempfile, "tempdir", str(tmp_path))

    def runner(task, context, cwd, tier, timeout_s):
        Path(cwd, f"{task}.txt").write_text(task)
        return {"ok": True}

    result = delegate.run_many(["one", "two"], cwd=str(hermetic), runner=runner)
    workspaces = [Path(r["workspace"]) for r in result["results"]]
    assert all(w.exists() for w in workspaces)
    assert not (workspaces[0].parent / delegate.OWNER_RECORD).exists()
    assert "no owner record" in delegate.sweep_copies(str(tmp_path))[0]
    assert all(w.exists() for w in workspaces)


def test_a_copy_held_by_a_live_process_or_an_uninspectable_one_is_kept(tmp_path, monkeypatch):
    scratch = _left_copy(tmp_path)
    # A process of this user holding a file inside open, with its working
    # directory elsewhere.
    holder = subprocess.Popen([sys.executable, "-c",
                               "import sys,time; f=open(sys.argv[1]); print(1, flush=True); "
                               "time.sleep(30)", str(scratch / "worker-1" / "a.txt")],
                              cwd="/", stdout=subprocess.PIPE, text=True)
    try:
        holder.stdout.readline()
        assert delegate.sweep_copies(str(tmp_path)) == [
            f"kept {scratch}: in use by pid {holder.pid}"]
    finally:
        holder.kill()
        holder.wait()
    # Processes the scan cannot read that started after the dead server, or
    # no /proc at all: kept as well.
    monkeypatch.setattr(procs, "holders", lambda inodes, since: (set(), {4242}))
    assert "cannot be inspected started after its server: pid 4242" in \
        delegate.sweep_copies(str(tmp_path))[0]
    monkeypatch.setattr(procs, "holders", lambda inodes, since: None)
    assert "/proc unreadable" in delegate.sweep_copies(str(tmp_path))[0]
    assert (scratch / "worker-1" / "a.txt").exists()


def test_an_interrupted_reclaim_leaves_the_owner_record_for_the_next_one(tmp_path, monkeypatch):
    scratch = _left_copy(tmp_path)
    (scratch / "base").mkdir()
    real_rmtree = delegate.shutil.rmtree

    def rmtree(path, *a, **k):
        real_rmtree(path, *a, **k)
        raise KeyboardInterrupt  # stopped after the first directory

    real_iterdir = Path.iterdir
    # The record sorts first, so a deletion in directory order reaches it first.
    monkeypatch.setattr(Path, "iterdir", lambda self: iter(sorted(real_iterdir(self))))
    monkeypatch.setattr(delegate.shutil, "rmtree", rmtree)
    with pytest.raises(KeyboardInterrupt):
        delegate.sweep_copies(str(tmp_path))
    assert (scratch / delegate.OWNER_RECORD).exists()
    monkeypatch.setattr(delegate.shutil, "rmtree", real_rmtree)
    assert delegate.sweep_copies(str(tmp_path))[0].startswith("removed")
    assert not scratch.exists()


def test_a_locked_owner_record_keeps_the_copy_even_if_no_process_is_seen(tmp_path, monkeypatch):
    # A server in a pid namespace this /proc does not show still holds the
    # record's lock; the lock alone keeps its copies.
    scratch = _left_copy(tmp_path)
    locker = subprocess.Popen([sys.executable, "-c",
                               "import fcntl,sys,time; f=open(sys.argv[1]); "
                               "fcntl.flock(f, fcntl.LOCK_EX); print(1, flush=True); time.sleep(30)",
                               str(scratch / delegate.OWNER_RECORD)],
                              stdout=subprocess.PIPE, text=True)
    monkeypatch.setattr(procs, "holders", lambda inodes, since: (set(), set()))
    try:
        locker.stdout.readline()
        assert delegate.sweep_copies(str(tmp_path)) == [
            f"kept {scratch}: its owner record is locked by a running process"]
    finally:
        locker.kill()
        locker.wait()
    assert delegate.sweep_copies(str(tmp_path))[0].startswith("removed")
