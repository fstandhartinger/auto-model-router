"""The job launcher: which tool runs a job, and how it is started.

Two things are worth guarding here. The mechanical one: a task must reach the
tool, exactly once and as data, never as shell input. The one that costs money
if it breaks: a route that claims to run on a flat-rate plan must actually
start its client in plan mode, which means the credential variables that would
silently move the run onto per-token billing are cleared for the child.
"""

import json
import sys

import pytest

from auto_router.catalog import Catalog, ModelInfo, Prices
from auto_router.config import RouterConfig, load_config
from auto_router.launcher import (
    Command,
    LauncherError,
    build_command,
    build_parser,
    decision_document,
    execute,
    launcher_config,
    main,
    observe,
    read_task,
)
from auto_router.policies import NoRouteAvailable
from auto_router.quota import QuotaDecision
from auto_router.router import Router

RUNNER = {"cmd": ["some-cli", "run", "--task", "{task}"], "timeout_s": 60}


def model(name, runner=RUNNER, **kw):
    return ModelInfo(name=name, provider="p", upstream_id=name, prices=Prices.free(),
                     capability={"general": 50, "coding": 50, "agentic": 50},
                     runner=runner, **kw)


def config(*models) -> RouterConfig:
    return RouterConfig(providers={}, catalog=Catalog(list(models)))


# --------------------------------------------------------------------------
# building the command
# --------------------------------------------------------------------------
def test_task_is_substituted_as_one_argument():
    cmd = build_command(model("a"), "fix the parser; rm -rf /", environ={})
    assert cmd.argv == ["some-cli", "run", "--task", "fix the parser; rm -rf /"]
    assert cmd.stdin_text is None


def test_task_on_stdin_needs_no_placeholder():
    cmd = build_command(model("a", runner={"cmd": ["cli", "-p"], "stdin": True}), "hello", environ={})
    assert cmd.argv == ["cli", "-p"] and cmd.stdin_text == "hello"


def test_a_task_that_would_never_reach_the_tool_is_an_error():
    with pytest.raises(LauncherError, match="never reach the tool"):
        build_command(model("a", runner={"cmd": ["cli", "-p"]}), "hello", environ={})


def test_empty_runner_is_an_error():
    with pytest.raises(LauncherError, match="non-empty list"):
        build_command(model("a", runner={"cmd": []}), "hello", environ={})


def test_plan_credentials_are_cleared_for_the_child_and_recorded():
    route = model("plan", subscription="claude",
                  runner={"cmd": ["cli", "{task}"], "clear_env": ["PLAN_KEY", "PLAN_TOKEN"]})
    cmd = build_command(route, "t", environ={"PLAN_KEY": "sk-secret", "HOME": "/home/x"})
    assert cmd.env["PLAN_KEY"] == "" and cmd.env["PLAN_TOKEN"] == ""
    assert cmd.env["HOME"] == "/home/x"
    # Only the variable that was actually set is reported as cleared, and the
    # value itself is nowhere in the record.
    assert cmd.cleared == ["PLAN_KEY"]
    assert "sk-secret" not in json.dumps(decision_document(_result_for(route, cmd), cmd))


def test_clear_env_falls_back_to_the_subscription_default():
    route = model("plan", subscription="claude", runner={"cmd": ["cli", "{task}"]})
    cmd = build_command(route, "t", environ={"PLAN_KEY": "x"},
                        default_clear={"claude": ["PLAN_KEY"]})
    assert cmd.env["PLAN_KEY"] == "" and cmd.cleared == ["PLAN_KEY"]


def test_env_from_copies_by_name_only(monkeypatch):
    monkeypatch.setenv("SOURCE_VALUE", "v")
    route = model("m", runner={"cmd": ["cli", "{task}"], "env_from": {"TARGET": "SOURCE_VALUE"},
                               "env": {"FLAG": "1"}})
    cmd = build_command(route, "t", environ={})
    assert cmd.env["TARGET"] == "v" and cmd.env["FLAG"] == "1"


def test_display_elides_the_task():
    cmd = build_command(model("a"), "a private task description", environ={})
    assert "private" not in cmd.display
    assert "<task>" in cmd.display and cmd.display.startswith("some-cli run --task")


# --------------------------------------------------------------------------
# choosing the route
# --------------------------------------------------------------------------
def test_only_routes_with_a_runner_can_be_launched():
    cfg = config(model("with-runner"), model("http-only", runner=None))
    assert [m.name for m in launcher_config(cfg).catalog.all()] == ["with-runner"]


def test_a_configuration_with_nothing_to_launch_says_so():
    with pytest.raises(LauncherError, match="nothing to launch"):
        launcher_config(config(model("http-only", runner=None)))


def test_job_routing_starts_cold_and_records_a_decision():
    cfg = config(model("free-worker"),
                 ModelInfo(name="metered", provider="p", upstream_id="metered",
                           prices=Prices(5.0, 25.0), capability={"general": 70, "coding": 70},
                           runner=RUNNER))
    router = Router(cfg)
    result = router.route_job("rename a variable in one file", steps=6)
    record = result.explanation.to_dict()
    # A launched job is priced as a whole job that starts cold: no route gets
    # credit for a cache it cannot have, and the output estimate is the job's,
    # not one turn's.
    assert record["cache"]["warm_tokens"] == 0
    assert record["cache"]["status"] in ("cold", "too-short")
    assert record["estimated_outcome"]["output_tokens"] == 6 * 1200
    assert record["selection"]["candidates_considered"] == 2


def test_equal_capability_sends_the_job_to_the_cheaper_route():
    cfg = config(model("free-worker"),
                 ModelInfo(name="metered", provider="p", upstream_id="metered",
                           prices=Prices(5.0, 25.0),
                           capability={"general": 50, "coding": 50, "agentic": 50},
                           runner=RUNNER))
    assert Router(cfg).route_job("rename a variable", steps=6).model.name == "free-worker"


def test_nothing_available_is_a_sentence_not_a_traceback():
    cfg = config(model("plan", subscription="claude"))
    router = Router(cfg, quota_reader=lambda: {
        "claude": QuotaDecision(False, 1.0, 0.95, "over the hard stop")})
    with pytest.raises(NoRouteAvailable, match="no route is available"):
        router.route_job("a task", steps=4)


def test_a_closed_plan_is_not_launched():
    plan = model("plan", subscription="claude")
    cfg = config(plan, model("free-worker"))
    closed = Router(cfg, quota_reader=lambda: {
        "claude": QuotaDecision(False, 1.0, 0.95, "weekly use 95% at or above hard stop 80%")})
    assert closed.route_job("anything", steps=4).model.name == "free-worker"


def test_forcing_a_route_is_recorded_as_an_override():
    cfg = config(model("free-worker"), model("other"))
    router = Router(cfg)
    chosen = router.route_job("a task", steps=4).model.name
    other = next(m.name for m in cfg.catalog.all() if m.name != chosen)
    forced = Router(cfg).route_job("a task", steps=4, force=other)
    record = forced.explanation.to_dict()
    assert forced.model.name == other
    assert record["selection"]["selected"] == other
    # The whole record is about the route that will actually run, not a mix.
    assert record["estimated_outcome"]["model"] == other
    assert record["cache"]["model"] == other
    assert chosen in record["notes"][-1]


def test_worker_tier_flag_is_available():
    parser = build_parser()
    assert parser.parse_args(["--tier", "cheap", "task"]).tier == "cheap"
    assert parser.parse_args(["--tier", "strong", "task"]).tier == "strong"


# --------------------------------------------------------------------------
# running it
# --------------------------------------------------------------------------
def _result_for(route, _cmd):
    """A decision for a single-route catalog, for record-shape assertions."""
    router = Router(config(route), quota_reader=lambda: {
        "claude": QuotaDecision(True, 0.0, 0.1, "plenty of slack")})
    return router.route_job("a task", steps=2)


def test_execute_runs_the_child_and_reports_its_exit_code(tmp_path):
    route = model("m", runner={"cmd": [sys.executable, "-c", "import sys; sys.exit(3)"],
                               "stdin": True})
    outcome = execute(build_command(route, "t"), cwd=str(tmp_path))
    assert outcome.exit_code == 3 and not outcome.timed_out


def test_execute_passes_the_task_on_stdin():
    route = model("m", runner={"cmd": [sys.executable, "-c", "import sys; print(sys.stdin.read().strip())"],
                               "stdin": True})
    outcome = execute(build_command(route, "the task"), capture=True)
    assert outcome.stdout.strip() == "the task"


def test_execute_times_out_without_hanging_the_launcher():
    route = model("m", runner={"cmd": [sys.executable, "-c", "import time; time.sleep(30)"],
                               "stdin": True, "timeout_s": 0.5})
    outcome = execute(build_command(route, "t"))
    assert outcome.timed_out and outcome.exit_code == 124


def test_a_missing_client_is_a_clear_error():
    route = model("m", runner={"cmd": ["definitely-not-installed-xyz", "{task}"]})
    with pytest.raises(LauncherError, match="not installed"):
        execute(build_command(route, "t"))


def test_outcome_records_status_without_inventing_a_cost(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTO_ROUTER_LEDGER", str(tmp_path / "ledger.jsonl"))
    route = model("plan", subscription="claude")
    router = Router(config(route), quota_reader=lambda: {
        "claude": QuotaDecision(True, 0.0, 0.1, "plenty of slack")})
    result = router.route_job("a task", steps=2)
    cmd = build_command(route, "a task", environ={})
    observe(router, result, execute(build_command(
        model("m", runner={"cmd": [sys.executable, "-c", "pass"], "stdin": True}), "t")), cmd)
    observed = result.explanation.observed.to_dict()
    assert observed["status"] == "ok"
    assert observed["cost_usd"] is None
    assert "no per-token charge" in observed["cost_basis"]
    assert observed["tokens"]["output"] is None


def test_a_failed_run_is_recorded_as_a_failure():
    route = model("m", runner={"cmd": [sys.executable, "-c", "raise SystemExit(2)"], "stdin": True})
    router = Router(config(route))
    result = router.route_job("a task", steps=2)
    cmd = build_command(route, "a task", environ={})
    observe(router, result, execute(cmd), cmd)
    assert result.explanation.observed.status == "transport_error"
    assert result.explanation.observed.error == "exit_2"


# --------------------------------------------------------------------------
# the command line
# --------------------------------------------------------------------------
def _write_config(tmp_path, cmd):
    path = tmp_path / "launcher.json"
    path.write_text(json.dumps({
        "providers": {"host": {"base_url": "https://example.invalid/v1"}},
        "models": [{"name": "only-route", "provider": "host", "free": True,
                    "capability": {"general": 50},
                    "runner": {"cmd": cmd, "stdin": True}}]}))
    return path


def test_dry_run_decides_and_runs_nothing(tmp_path, capsys):
    path = _write_config(tmp_path, [sys.executable, "-c", "raise SystemExit(9)"])
    assert main(["--config", str(path), "--dry-run", "a small task"]) == 0
    doc = json.loads(capsys.readouterr().out)
    assert doc["route"] == "only-route" and "decision" in doc


def test_main_returns_the_child_exit_code(tmp_path):
    path = _write_config(tmp_path, [sys.executable, "-c", "raise SystemExit(7)"])
    assert main(["--config", str(path), "--quiet", "a small task"]) == 7


def test_main_without_a_task_fails_cleanly(tmp_path, monkeypatch, capsys):
    path = _write_config(tmp_path, [sys.executable, "-c", "pass"])
    monkeypatch.setattr("sys.stdin", type("S", (), {"read": staticmethod(lambda: "  ")})())
    assert main(["--config", str(path), "-"]) == 2
    assert "no task given" in capsys.readouterr().err


def test_list_shows_the_runnable_routes(tmp_path, capsys):
    path = _write_config(tmp_path, ["some-cli"])
    assert main(["--config", str(path), "--list"]) == 0
    assert "only-route" in capsys.readouterr().out


def test_read_task_accepts_stdin(monkeypatch):
    monkeypatch.setattr("sys.stdin", type("S", (), {"read": staticmethod(lambda: " task \n")})())
    assert read_task("-") == "task"


def test_parser_documents_the_flags():
    help_text = build_parser().format_help()
    for flag in ("--dry-run", "--route", "--steps", "--list"):
        assert flag in help_text


# --------------------------------------------------------------------------
# what belongs on which surface
# --------------------------------------------------------------------------
def test_a_plan_reachable_only_through_its_own_cli_is_not_an_http_route():
    """Found live: the gateway offered a *different* vendor's plan for a turn.

    Nothing broke, because subscription traffic is forwarded unchanged, but the
    decision was meaningless: that plan can only be reached by launching its
    own client, so no HTTP request can be served from it. It belongs to the
    launcher and to no other surface.
    """
    from auto_router.config import for_http

    cfg = config(model("free-worker"),
                 model("other-vendors-plan", subscription="other", launch_only=True))
    assert [m.name for m in for_http(cfg).catalog.all()] == ["free-worker"]
    # The launcher still sees both: launching is exactly how that plan is used.
    assert len(launcher_config(cfg).catalog.all()) == 2


def test_an_http_only_route_is_not_launchable_and_a_launch_only_route_is():
    cfg = config(model("http-only", runner=None), model("launchable"))
    assert [m.name for m in launcher_config(cfg).catalog.all()] == ["launchable"]
