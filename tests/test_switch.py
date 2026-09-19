"""Switch mode: one Claude Code conversation moving between cheap mode and plan mode.

What is under test is the plumbing that decides and carries out a switch; the
routing policy itself is tested elsewhere. The two properties that matter most:
plan mode never has the router in its path, and nothing the router does can
block the user's prompt when the router itself is broken.
"""

import io
import json
import types

import pytest
from fastapi.testclient import TestClient

from auto_router import switch


def _result(name, subscription=None, upstream="vendor/x", difficulty=0.5, rows=None,
            turn_start=True, reason="because"):
    model = types.SimpleNamespace(name=name, subscription=subscription, upstream_id=upstream,
                                  launch_only=False)
    explanation = None
    if rows is not None:
        explanation = types.SimpleNamespace(to_dict=lambda: {"candidates": rows})
    return types.SimpleNamespace(model=model, request=types.SimpleNamespace(difficulty=difficulty),
                                 reason=reason, explanation=explanation, turn_start=turn_start,
                                 headers={}, conversation_id="c")


# --------------------------------------------------------------------------
# deciding
# --------------------------------------------------------------------------
def test_a_forcing_prefix_wins_and_is_stripped():
    d = switch.decide("~plan  refactor the parser", switch.CHEAP, lambda t: pytest.fail("not asked"))
    assert (d.target, d.forced, d.prompt) == (switch.PLAN, True, "refactor the parser")
    d = switch.decide("~cheap rename x", switch.PLAN, None)
    assert (d.target, d.prompt) == (switch.CHEAP, "rename x")
    # only as a separate word
    assert switch.strip_force("~planning ahead")[1] is None


def test_a_broken_router_never_moves_or_blocks_the_session():
    def boom(_):
        raise RuntimeError("no route")
    for current in (switch.CHEAP, switch.PLAN):
        d = switch.decide("anything", current, boom)
        assert d.target == current
        assert "router unavailable" in d.reason
    assert switch.decide("anything", switch.PLAN, None).target == switch.PLAN


def test_the_plan_is_chosen_when_the_policy_picks_a_claude_plan_route():
    d = switch.decide("hard", switch.CHEAP, lambda t: _result("plan-claude", "claude", "claude-opus-5"))
    assert (d.target, d.plan_model) == (switch.PLAN, "claude-opus-5")


def test_a_tie_between_plan_and_a_free_route_goes_to_the_free_route():
    rows = [{"model": "plan-claude", "subscription": "claude", "estimated_expected_usd": 0.0,
             "estimated_p_success": 1.0},
            {"model": "free", "subscription": None, "estimated_expected_usd": 0.0004,
             "estimated_p_success": 1.0}]
    d = switch.decide("rename x", switch.CHEAP, lambda t: _result("plan-claude", "claude", rows=rows))
    assert (d.target, d.model) == (switch.CHEAP, "free")
    # a real gap in success probability is not a tie
    rows[1]["estimated_p_success"] = 0.6
    d = switch.decide("hard", switch.CHEAP, lambda t: _result("plan-claude", "claude", rows=rows))
    assert d.target == switch.PLAN


def test_leaving_the_plan_needs_a_clearly_easy_prompt():
    cheap = lambda t: _result("free", difficulty=0.5)
    assert switch.decide("medium", switch.PLAN, cheap, sticky=0.35).target == switch.PLAN
    easy = lambda t: _result("free", difficulty=0.1)
    assert switch.decide("easy", switch.PLAN, easy, sticky=0.35).target == switch.CHEAP


def test_only_routes_a_mode_can_serve_are_candidates():
    ns = types.SimpleNamespace
    assert switch.switch_candidate(ns(subscription="claude", launch_only=False))
    assert not switch.switch_candidate(ns(subscription="codex", launch_only=True))
    assert switch.switch_candidate(ns(subscription=None, launch_only=False))
    assert not switch.switch_candidate(ns(subscription=None, launch_only=True))   # a price reference


# --------------------------------------------------------------------------
# the hook
# --------------------------------------------------------------------------
def _hook(monkeypatch, tmp_path, mode, prompt, route):
    state = tmp_path / "abc.json"
    monkeypatch.setenv("AUTO_ROUTER_SWITCH_STATE", str(state))
    monkeypatch.setenv("AUTO_ROUTER_SWITCH_MODE", mode)
    out = io.StringIO()
    event = {"session_id": "s-1", "prompt": prompt, "transcript_path": "/dev/null"}
    assert switch.hook_main(io.StringIO(json.dumps(event)), out, lambda: route) == 0
    return state, out.getvalue()


def test_the_hook_does_nothing_outside_the_wrapper(monkeypatch):
    monkeypatch.delenv("AUTO_ROUTER_SWITCH_STATE", raising=False)
    out = io.StringIO()
    assert switch.hook_main(io.StringIO("{}"), out, lambda: pytest.fail("not asked")) == 0
    assert out.getvalue() == ""


def test_the_hook_lets_a_prompt_through_when_the_mode_is_right(monkeypatch, tmp_path):
    state, out = _hook(monkeypatch, tmp_path, switch.CHEAP, "rename x", lambda t: _result("free"))
    assert out == "" and not state.exists()
    assert state.with_suffix(".session").read_text() == "s-1"


def test_the_hook_blocks_and_records_a_switch(monkeypatch, tmp_path):
    state, out = _hook(monkeypatch, tmp_path, switch.CHEAP, "design the cache",
                       lambda t: _result("plan-claude", "claude", "claude-opus-5"))
    reply = json.loads(out)
    assert reply["decision"] == "block" and "your Claude plan" in reply["reason"]
    saved = json.loads(state.read_text())
    assert saved == {**saved, "session_id": "s-1", "prompt": "design the cache",
                     "target": switch.PLAN, "plan_model": "claude-opus-5"}


def test_the_prompt_a_session_was_resumed_for_is_not_routed_again(monkeypatch, tmp_path):
    # Otherwise a forced switch bounces straight back: the resumed prompt no
    # longer carries its ~plan prefix and would be decided afresh.
    monkeypatch.setenv("AUTO_ROUTER_SWITCH_RESUMED", switch.prompt_digest("append beta"))
    state, out = _hook(monkeypatch, tmp_path, switch.PLAN, "append beta",
                       lambda t: pytest.fail("not asked"))
    assert out == "" and not state.exists()


def test_a_forced_prompt_needs_no_router(monkeypatch, tmp_path):
    def broken_factory():
        raise FileNotFoundError("config")
    state = tmp_path / "abc.json"
    monkeypatch.setenv("AUTO_ROUTER_SWITCH_STATE", str(state))
    monkeypatch.setenv("AUTO_ROUTER_SWITCH_MODE", switch.CHEAP)
    out = io.StringIO()
    event = {"session_id": "s-1", "prompt": "~plan go"}
    switch.hook_main(io.StringIO(json.dumps(event)), out, broken_factory)
    assert json.loads(out.getvalue())["decision"] == "block"


def test_a_broken_configuration_lets_the_prompt_through(monkeypatch, tmp_path):
    def broken_factory():
        raise FileNotFoundError("config")
    state, out = _hook(monkeypatch, tmp_path, switch.CHEAP, "anything", None)
    monkeypatch.setenv("AUTO_ROUTER_SWITCH_DIR", str(tmp_path))
    out = io.StringIO()
    switch.hook_main(io.StringIO(json.dumps({"session_id": "s", "prompt": "hi"})), out, broken_factory)
    assert out.getvalue() == "" and not state.exists()


def test_slash_commands_are_never_routed(monkeypatch, tmp_path):
    state, out = _hook(monkeypatch, tmp_path, switch.CHEAP, "/model opus",
                       lambda t: pytest.fail("not asked"))
    assert out == "" and not state.exists()


# --------------------------------------------------------------------------
# the wrapper
# --------------------------------------------------------------------------
def test_plan_mode_has_no_router_and_no_credential_variables(tmp_path):
    base = {"ANTHROPIC_API_KEY": "k", "ANTHROPIC_AUTH_TOKEN": "t", "ANTHROPIC_BASE_URL": "http://x",
            "CLAUDE_CODE_CHILD_SESSION": "1", "PATH": "/bin"}
    env = switch.mode_env(switch.PLAN, base, "http://127.0.0.1:8787", tmp_path / "ab.json")
    assert not {"ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL",
                "CLAUDE_CODE_CHILD_SESSION", "ANTHROPIC_CUSTOM_HEADERS"} & set(env)
    assert env["AUTO_ROUTER_SWITCH_MODE"] == switch.PLAN


def test_cheap_mode_points_at_the_gateway_with_its_own_credential(tmp_path):
    base = {"ANTHROPIC_API_KEY": "k", "ANTHROPIC_CUSTOM_HEADERS": "X-Team: a"}
    env = switch.mode_env(switch.CHEAP, base, "http://127.0.0.1:8787", tmp_path / "ab12.json")
    assert env["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:8787"
    assert env["ANTHROPIC_AUTH_TOKEN"] == "auto-router-local"
    assert "ANTHROPIC_API_KEY" not in env
    assert env["ANTHROPIC_CUSTOM_HEADERS"] == "X-Team: a\nx-auto-router-switch: ab12"


def test_the_initial_prompt_is_split_off_and_not_repeated_on_resume(tmp_path):
    assert switch.split_prompt(["-p", "--model", "sonnet", "fix it"]) == (["-p", "--model", "sonnet"], "fix it")
    assert switch.split_prompt(["--model", "sonnet"]) == (["--model", "sonnet"], None)
    assert switch.split_prompt(["fix it"]) == ([], "fix it")
    s = tmp_path / "s.json"
    first = switch.claude_argv("claude", s, switch.CHEAP, ["-p"], None, None, "fix it")
    again = switch.claude_argv("claude", s, switch.PLAN, ["-p"], None, ("sid", "fix it"), "fix it")
    assert first[-1] == "fix it" and again.count("fix it") == 1 and again[-3:] == ["--resume", "sid", "fix it"]


def test_a_resumed_switch_replaces_the_users_own_resume_flag(tmp_path):
    argv = switch.claude_argv("claude", tmp_path / "s", switch.PLAN, ["-p", "--resume", "old", "-c"],
                              None, ("new", "go"))
    assert argv.count("--resume") == 1 and "old" not in argv and "-c" not in argv
    assert argv[-3:] == ["--resume", "new", "go"]


def test_state_path_only_accepts_our_own_tokens(monkeypatch, tmp_path):
    monkeypatch.setenv("AUTO_ROUTER_SWITCH_DIR", str(tmp_path))
    assert switch.state_path("00ff") == tmp_path / "00ff.json"
    for bad in ("", "../etc/passwd", "ABC", "a" * 65):
        assert switch.state_path(bad) is None


def test_the_wrapper_resumes_the_same_session_in_the_other_mode(monkeypatch, tmp_path):
    monkeypatch.setenv("AUTO_ROUTER_SWITCH_DIR", str(tmp_path))
    monkeypatch.setattr(switch, "gateway_up", lambda url: True)
    launches = []

    class Child:
        def __init__(self, argv, env):
            launches.append((argv, env))
            self.code = None
            if len(launches) == 1:        # the hook asks for the plan on the first prompt
                switch.write_state(__import__("pathlib").Path(env["AUTO_ROUTER_SWITCH_STATE"]), {
                    "session_id": "s-9", "prompt": "design it", "target": switch.PLAN,
                    "plan_model": "claude-opus-5"})

        def poll(self):
            return self.code

        def terminate(self):
            self.code = -15

        def wait(self):
            if self.code is None:
                self.code = 0 if len(launches) > 1 else -15
            return self.code

    code = switch.run(["--permission-mode", "acceptEdits"], start=switch.CHEAP,
                      spawn=lambda argv, env: Child(argv, env), poll_s=0.01, settle_s=0.0)
    assert code == 0 and len(launches) == 2
    first, second = launches
    assert first[1]["AUTO_ROUTER_SWITCH_MODE"] == switch.CHEAP and "ANTHROPIC_BASE_URL" in first[1]
    argv, env = second
    assert env["AUTO_ROUTER_SWITCH_MODE"] == switch.PLAN and "ANTHROPIC_BASE_URL" not in env
    assert argv[-3:] == ["--resume", "s-9", "design it"]
    assert env["AUTO_ROUTER_SWITCH_RESUMED"] == switch.prompt_digest("design it")
    assert argv[argv.index("--model") + 1] == "claude-opus-5"
    assert list(tmp_path.iterdir()) == []          # nothing left behind


def test_a_gateway_switch_without_a_session_id_uses_the_one_the_hook_saw(monkeypatch, tmp_path):
    monkeypatch.setenv("AUTO_ROUTER_SWITCH_DIR", str(tmp_path))
    monkeypatch.setattr(switch, "gateway_up", lambda url: True)
    launches = []

    class Child:
        def __init__(self, argv, env):
            launches.append(argv)
            state = __import__("pathlib").Path(env["AUTO_ROUTER_SWITCH_STATE"])
            if len(launches) == 1:
                state.with_suffix(".session").write_text("s-7")
                switch.write_state(state, {"session_id": None, "prompt": "Continue",
                                           "target": switch.PLAN, "plan_model": None})

        def poll(self):
            return None if len(launches) == 1 else 0

        def terminate(self):
            pass

        def wait(self):
            return 0

    switch.run([], spawn=lambda argv, env: Child(argv, env), poll_s=0.01, settle_s=0.0)
    assert launches[1][-3:] == ["--resume", "s-7", "Continue"]


# --------------------------------------------------------------------------
# the gateway's half: escalating a stuck cheap turn
# --------------------------------------------------------------------------
@pytest.fixture
def gateway(tmp_path, monkeypatch):
    import importlib

    from auto_router import server as server_module
    config = tmp_path / "cfg.json"
    config.write_text(json.dumps({
        "providers": {"host": {"base_url": "https://llm.example.invalid/v1"},
                      "anthropic": {"base_url": "https://api.anthropic.com/v1", "api": "anthropic"}},
        "models": [{"name": "cheap", "provider": "host", "upstream_id": "vendor/cheap", "free": True,
                    "capability": {"general": 50, "agentic": 50}},
                   {"name": "plan-claude", "provider": "anthropic", "upstream_id": "claude-opus-5",
                    "subscription": "claude", "capability": {"general": 90, "agentic": 90}}]}))
    monkeypatch.setenv("AUTO_ROUTER_CONFIG", str(config))
    monkeypatch.setenv("AUTO_ROUTER_SWITCH_DIR", str(tmp_path / "switch"))
    server = importlib.reload(server_module)
    yield server
    importlib.reload(server_module)


def test_a_stuck_cheap_turn_in_switch_mode_is_handed_to_the_plan(gateway, monkeypatch, tmp_path):
    seen = {}

    def fake_route(messages, system, tools, max_tokens, now=None, exclude_subscriptions=False):
        seen["exclude"] = exclude_subscriptions
        return _result("plan-claude", "claude", "claude-opus-5", turn_start=False,
                       reason="3 failing tool results in a row")

    monkeypatch.setattr(gateway.router, "route", fake_route)
    monkeypatch.setattr(gateway.router, "observe", lambda *a, **k: None)
    response = TestClient(gateway.app).post(
        "/v1/messages",
        json={"model": "claude-sonnet-5", "max_tokens": 16,
              "messages": [{"role": "user", "content": "fix it"}]},
        headers={"authorization": "Bearer auto-router-local", "x-claude-code-session-id": "s-5",
                 "x-auto-router-switch": "beef"})
    assert response.status_code == 200
    assert seen["exclude"] is False
    body = response.json()
    assert body["stop_reason"] == "end_turn" and "Claude plan" in body["content"][0]["text"]
    saved = json.loads((tmp_path / "switch" / "beef.json").read_text())
    assert (saved["session_id"], saved["target"], saved["by"]) == ("s-5", "plan", "gateway")


def test_without_the_switch_header_a_gateway_credential_never_reaches_the_plan(gateway, monkeypatch):
    seen = {}

    def fake_route(messages, system, tools, max_tokens, now=None, exclude_subscriptions=False):
        seen["exclude"] = exclude_subscriptions
        raise gateway.NoRouteAvailable("stop here")

    monkeypatch.setattr(gateway.router, "route", fake_route)
    TestClient(gateway.app).post(
        "/v1/messages", json={"model": "m", "max_tokens": 8, "messages": [{"role": "user", "content": "x"}]},
        headers={"authorization": "Bearer auto-router-local"})
    assert seen["exclude"] is True
