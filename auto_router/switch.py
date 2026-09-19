"""Switch mode: Claude Code on cheap routes by default, on your own plan when it matters.

Claude Code has two documented ways to reach a model, and this module moves one
conversation between them, a prompt at a time:

* **cheap mode** - ``ANTHROPIC_BASE_URL`` points at this router and a gateway
  credential is set, so "the credential replaces the subscription login for
  that session, and the subscription's usage limits don't apply"
  (code.claude.com/docs/en/llm-gateway). The router answers each request from
  free or metered routes on your own provider keys. Your plan is not touched.
* **plan mode** - Claude Code runs with no base URL and no credential variable,
  signed in with your own claude.ai login, and talks to Anthropic directly. The
  router is not in the path at all, so it never sees the plan's login.

A ``UserPromptSubmit`` hook asks the router which of the two should answer each
prompt *before* any model sees it. When the answer is the other mode, the hook
blocks the prompt ("prevents the prompt from being processed and erases it from
context", code.claude.com/docs/en/hooks), the wrapper stops Claude Code and
starts it again in the other mode with ``--resume <session> "<the prompt>"``.
Claude Code keeps the transcript locally, so the conversation continues where
it was, including what the other mode did.

What a switch costs: a restart of a second or two, and the new side reads the
whole conversation cold - on the plan that is one cache write of the full
context. The hook therefore does not switch back from the plan for a prompt
that is only borderline easy (``AUTO_ROUTER_SWITCH_STICKY``).

Entry points::

    python -m auto_router.switch [--start cheap|plan] [--plan-model opus] [claude args...]
    python -m auto_router.switch hook       # the UserPromptSubmit hook (JSON on stdin)
    python -m auto_router.switch decide "a prompt"   # print the decision, run nothing

Typing ``~plan`` or ``~cheap`` at the start of a prompt forces the mode for it.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

CHEAP = "cheap"
PLAN = "plan"
#: ``!`` would be Claude Code's own shell mode, ``@`` a file mention, ``/`` a command.
FORCE_PREFIXES = {"~plan": PLAN, "~cheap": CHEAP}

#: Variables that would move a plan-mode session onto per-token billing or away
#: from Anthropic. Cleared for plan mode, as the launcher does.
PLAN_CLEAR = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL")
#: Set by Claude Code for its own child processes; it switches transcript
#: saving off, and without a transcript there is nothing to resume.
CHILD_MARKERS = ("CLAUDE_CODE_CHILD_SESSION",)

ROOT = Path(__file__).resolve().parents[1]

#: Header the wrapper adds to every cheap-mode request (``ANTHROPIC_CUSTOM_HEADERS``)
#: so the gateway can hand the conversation to the plan when the cheap route is
#: stuck. The value is a random token naming a file in :func:`switch_dir`.
SWITCH_HEADER = "x-auto-router-switch"


def switch_dir() -> Path:
    return Path(os.environ.get("AUTO_ROUTER_SWITCH_DIR")
                or Path.home() / ".cache" / "auto-router" / "switch")


def state_path(token: str) -> Path | None:
    """The switch file for ``token``, or None when the token is not one of ours."""
    if not token or len(token) > 64 or not all(c in "0123456789abcdef" for c in token):
        return None
    return switch_dir() / f"{token}.json"


def write_state(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload))
    tmp.replace(path)


@dataclass
class Decision:
    target: str
    model: str
    reason: str
    difficulty: float | None = None
    forced: bool = False
    prompt: str = ""
    plan_model: str | None = None


def strip_force(prompt: str) -> tuple[str, str | None]:
    head = prompt.lstrip()
    for prefix, mode in FORCE_PREFIXES.items():
        if head.lower().startswith(prefix) and (len(head) == len(prefix) or head[len(prefix)].isspace()):
            return head[len(prefix):].lstrip(), mode
    return prompt, None


def sticky_threshold() -> float:
    try:
        return float(os.environ.get("AUTO_ROUTER_SWITCH_STICKY", "0.35"))
    except ValueError:
        return 0.35


def decide(prompt: str, current: str, route_job: Callable[[str], object] | None,
           sticky: float | None = None) -> Decision:
    """Which mode should answer ``prompt``, given the mode the session is in.

    ``route_job`` is :meth:`Router.route_job` (or a stand-in in tests). Any
    failure keeps the current mode: a hook that blocks the user's prompt
    because the router had a bad moment would be worse than no router.
    """
    text, forced = strip_force(prompt)
    if forced:
        return Decision(forced, "(forced)", f"forced with ~{forced}", forced=True, prompt=text)
    if route_job is None:
        return Decision(current, "(none)", "no router configuration; staying put", prompt=text)
    try:
        result = route_job(text)
    except Exception as exc:  # NoRouteAvailable, network, config: never block on it
        return Decision(current, "(error)", f"router unavailable, staying in {current} mode: {exc}",
                        prompt=text)
    model = result.model
    difficulty = getattr(result.request, "difficulty", None)
    target = PLAN if model.subscription == "claude" else CHEAP
    reason = result.reason
    if target == PLAN:
        alternative = cheap_equivalent(result)
        if alternative:
            return Decision(CHEAP, alternative, f"{alternative} is expected to do this as well as the plan; "
                            "the plan's limits are kept for work that needs them",
                            difficulty=difficulty, prompt=text)
    threshold = sticky_threshold() if sticky is None else sticky
    if current == PLAN and target == CHEAP and difficulty is not None and difficulty >= threshold:
        return Decision(PLAN, model.name,
                        f"staying on the plan: {model.name} would do, but difficulty {difficulty:.2f} "
                        f"is not clearly easy (< {threshold:.2f}) and switching back costs a cold cache",
                        difficulty=difficulty, prompt=text, plan_model=None)
    return Decision(target, model.name, reason, difficulty=difficulty, prompt=text,
                    plan_model=model.upstream_id if target == PLAN else None)


#: How close a cheap route has to come to the plan to be preferred: expected
#: cost within this many dollars, success probability within this many points.
TIE_USD = 0.01
TIE_P = 0.02


def cheap_equivalent(result) -> str | None:
    """A non-plan candidate the policy rates as good as the chosen plan route.

    With plenty of headroom the pacing rule prices a plan at zero ("use it, it
    would expire unused"), so a trivial prompt ties between the plan and a free
    model and the plan wins on order alone. In switch mode that would spend a
    restart and a cold cache on nothing; the tie goes to the side already
    serving the conversation's cheap work.
    """
    explanation = getattr(result, "explanation", None)
    if explanation is None:
        return None
    rows = explanation.to_dict().get("candidates") or []
    chosen = next((r for r in rows if r.get("model") == result.model.name), None)
    if not chosen:
        return None
    for row in rows:
        if row.get("subscription") or row.get("rejected"):
            continue
        if (row.get("estimated_expected_usd", 1e9) <= chosen.get("estimated_expected_usd", 0) + TIE_USD
                and row.get("estimated_p_success", 0) >= chosen.get("estimated_p_success", 1) - TIE_P):
            return row.get("model")
    return None


def _router_route_job() -> Callable[[str], object] | None:
    """A job-level router over the configured catalog, or None without a config."""
    if not os.environ.get("AUTO_ROUTER_CONFIG"):
        return None
    from .config import load_config
    from .router import Router

    router = Router(load_config())
    steps = int(os.environ.get("AUTO_ROUTER_SWITCH_STEPS", "8"))
    return lambda text: router.route_job(text, steps=steps, only=switch_candidate)


def switch_candidate(model) -> bool:
    """Routes one of the two modes can actually serve.

    Plan mode serves the Claude plan routes; cheap mode serves whatever the
    gateway can answer over HTTP without a plan. A launch-only route (another
    vendor's CLI, a list-price reference) is neither.
    """
    if model.subscription:
        return model.subscription == "claude"
    return not model.launch_only


# --------------------------------------------------------------------------
# the hook
# --------------------------------------------------------------------------
def hook_main(stdin=sys.stdin, stdout=sys.stdout,
              route_job_factory: Callable[[], Callable[[str], object] | None] = _router_route_job) -> int:
    """The ``UserPromptSubmit`` hook. Fails open: an error lets the prompt through."""
    try:
        return _hook(stdin, stdout, route_job_factory)
    except Exception:  # never block a prompt on our own bug
        import traceback
        switch_dir().mkdir(parents=True, exist_ok=True)
        with open(switch_dir() / "hook-errors.log", "a") as fh:
            fh.write(f"{time.ctime()}\n{traceback.format_exc()}\n")
        return 0


def _hook(stdin, stdout, route_job_factory) -> int:
    state = os.environ.get("AUTO_ROUTER_SWITCH_STATE")
    current = os.environ.get("AUTO_ROUTER_SWITCH_MODE")
    if not state or current not in (CHEAP, PLAN):
        return 0                       # not started by the wrapper: do nothing
    event = json.load(stdin)
    prompt = event.get("prompt") or ""
    if not prompt.strip() or prompt.lstrip().startswith("/"):
        return 0                       # slash commands and empty prompts are Claude Code's own
    if os.environ.get("AUTO_ROUTER_SWITCH_RESUMED") == prompt_digest(prompt):
        return 0                       # the prompt this session was just switched for
    if event.get("session_id"):
        Path(state).with_suffix(".session").write_text(event["session_id"])
    decision = decide(prompt, current, _lazy(route_job_factory))
    _log(decision, current, event)
    if decision.target == current:
        return 0                       # a forcing prefix for the current mode is left in place
    payload = {"session_id": event.get("session_id"), "prompt": decision.prompt,
               "target": decision.target, "plan_model": decision.plan_model,
               "reason": decision.reason, "at": time.time()}
    write_state(Path(state), payload)
    label = "your Claude plan" if decision.target == PLAN else "the cheap routes"
    json.dump({"decision": "block", "suppressOriginalPrompt": True,
               "reason": f"auto-router: switching this conversation to {label} "
                         f"({decision.reason[:160]}). It resumes in a moment with your prompt."}, stdout)
    return 0


def _lazy(factory):
    """Build the router only when a decision needs it; a forced prompt never does."""
    def route(text):
        real = factory()
        if real is None:
            raise RuntimeError("no router configuration (AUTO_ROUTER_CONFIG)")
        return real(text)
    return route


def prompt_digest(prompt: str) -> str:
    return hashlib.sha256(prompt.encode()).hexdigest()


def _log(decision: Decision, current: str, event: dict) -> None:
    path = os.environ.get("AUTO_ROUTER_SWITCH_LOG")
    if not path:
        return
    record = {"at": time.time(), "session": event.get("session_id"), "from": current,
              **{k: v for k, v in asdict(decision).items() if k != "prompt"}}
    with open(path, "a") as fh:
        fh.write(json.dumps(record) + "\n")


# --------------------------------------------------------------------------
# the wrapper
# --------------------------------------------------------------------------
def hook_command() -> str:
    code = (f"import sys; sys.path.insert(0, {str(ROOT)!r}); "
            "from auto_router.switch import hook_main; raise SystemExit(hook_main())")
    return f"{sys.executable} -c {json.dumps(code)}"


def settings_file(directory: Path) -> Path:
    path = directory / "switch-settings.json"
    path.write_text(json.dumps({"hooks": {"UserPromptSubmit": [
        {"hooks": [{"type": "command", "command": hook_command(), "timeout": 30}]}]}}))
    return path


def mode_env(mode: str, base: dict[str, str], gateway: str, state: Path) -> dict[str, str]:
    env = {k: v for k, v in base.items() if k not in CHILD_MARKERS}
    if mode == PLAN:
        for name in PLAN_CLEAR:
            env.pop(name, None)
    else:
        env.pop("ANTHROPIC_API_KEY", None)
        env["ANTHROPIC_BASE_URL"] = gateway
        env["ANTHROPIC_AUTH_TOKEN"] = base.get("AUTO_ROUTER_GATEWAY_TOKEN") or "auto-router-local"
        header = f"{SWITCH_HEADER}: {state.stem}"
        extra = base.get("ANTHROPIC_CUSTOM_HEADERS")
        env["ANTHROPIC_CUSTOM_HEADERS"] = f"{extra}\n{header}" if extra else header
    env["AUTO_ROUTER_SWITCH_MODE"] = mode
    env["AUTO_ROUTER_SWITCH_STATE"] = str(state)
    return env


def claude_argv(claude: str, settings: Path, mode: str, user_args: list[str],
                plan_model: str | None, resume: tuple[str, str] | None,
                prompt: str | None = None) -> list[str]:
    argv = [claude, "--settings", str(settings)]
    if mode == PLAN and plan_model and "--model" not in user_args:
        argv += ["--model", plan_model]
    if resume:
        user_args = without_resume(user_args)
    argv += user_args
    if resume:
        session, text = resume
        argv += ["--resume", session, text]
    elif prompt:
        argv.append(prompt)
    return argv


def without_resume(args: list[str]) -> list[str]:
    """Drop the user's own ``--resume``/``--continue``: a switch names the session itself."""
    out, skip = [], False
    for arg in args:
        if skip:
            skip = False
            continue
        if arg in ("--resume", "-r"):
            skip = True
            continue
        if arg in ("--continue", "-c") or arg.startswith("--resume="):
            continue
        out.append(arg)
    return out


def split_prompt(args: list[str]) -> tuple[list[str], str | None]:
    """Separate a trailing prompt from Claude Code's options.

    The last argument is taken as the prompt when it is not an option and does
    not follow one (``--model sonnet`` ends in a value, not a prompt). A resumed
    session gets its prompt from the switch, so the original must not be
    passed a second time.
    """
    if args and not args[-1].startswith("-") and (len(args) == 1 or not args[-2].startswith("-")):
        return args[:-1], args[-1]
    return args, None


def gateway_up(url: str) -> bool:
    try:
        with urllib.request.urlopen(f"{url.rstrip('/')}/health", timeout=3) as resp:
            return resp.status == 200
    except Exception:
        return False


def run(user_args: list[str], start: str = CHEAP, plan_model: str | None = None,
        prompt: str | None = None,
        gateway: str = "http://127.0.0.1:8787", claude: str = "claude",
        spawn=subprocess.Popen, poll_s: float = 0.3, settle_s: float = 0.8) -> int:
    exe = shutil.which(claude) or claude
    for name in ("AUTO_ROUTER_CONFIG", "AUTO_ROUTER_LEDGER", "AUTO_ROUTER_SWITCH_LOG"):
        if os.environ.get(name):       # the hook runs in Claude Code's working directory
            os.environ[name] = os.path.abspath(os.path.expanduser(os.environ[name]))
    workdir = Path(tempfile.mkdtemp(prefix="auto-router-switch-"))
    state = state_path(secrets.token_hex(16))
    state.parent.mkdir(parents=True, exist_ok=True)
    settings = settings_file(workdir)
    mode, resume, code = start, None, 0
    if mode == CHEAP and not gateway_up(gateway):
        print(f"auto-router: no router answering at {gateway}; starting on your plan instead",
              file=sys.stderr)
        mode = PLAN
    previous = signal.signal(signal.SIGINT, signal.SIG_IGN)  # Ctrl-C belongs to Claude Code
    try:
        while True:
            print(f"auto-router: {describe(mode, gateway)}", file=sys.stderr)
            env = mode_env(mode, dict(os.environ), gateway, state)
            if resume:
                env["AUTO_ROUTER_SWITCH_RESUMED"] = prompt_digest(resume[1])
            child = spawn(claude_argv(exe, settings, mode, user_args, plan_model, resume, prompt), env=env)
            watcher = threading.Thread(target=_watch, args=(child, state, poll_s, settle_s), daemon=True)
            watcher.start()
            code = child.wait()
            if not state.exists():
                return code
            switch = json.loads(state.read_text())
            state.unlink()
            session_file = state.with_suffix(".session")
            if not switch.get("session_id") and session_file.exists():
                switch["session_id"] = session_file.read_text().strip()
            if not switch.get("session_id"):
                print("auto-router: no session to resume; stopping", file=sys.stderr)
                return code
            if switch["target"] == CHEAP and not gateway_up(gateway):
                print(f"auto-router: the router at {gateway} is down; staying on your plan",
                      file=sys.stderr)
                switch["target"] = PLAN
            mode = switch["target"]
            plan_model = switch.get("plan_model") or plan_model
            resume = (switch["session_id"], switch["prompt"])
    finally:
        signal.signal(signal.SIGINT, previous)
        shutil.rmtree(workdir, ignore_errors=True)
        for leftover in (state, state.with_suffix(".session"), state.with_suffix(".tmp")):
            leftover.unlink(missing_ok=True)


def describe(mode: str, gateway: str) -> str:
    if mode == PLAN:
        return "plan mode - Claude Code signed in with your plan, talking to Anthropic directly"
    return f"cheap mode - Claude Code through the router at {gateway}, on your own provider keys"


def _watch(child, state: Path, poll_s: float, settle_s: float) -> None:
    while child.poll() is None:
        if state.exists():
            time.sleep(settle_s)       # let Claude Code draw the block message
            if child.poll() is None:
                child.terminate()
            return
        time.sleep(poll_s)


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args[:1] == ["hook"]:
        return hook_main()
    if args[:1] == ["decide"]:
        prompt = " ".join(args[1:]) or sys.stdin.read()
        current = os.environ.get("AUTO_ROUTER_SWITCH_MODE", CHEAP)
        print(json.dumps(asdict(decide(prompt, current, _lazy(_router_route_job))), indent=2))
        return 0
    start, plan_model, prompt = CHEAP, None, None
    gateway = os.environ.get("AUTO_ROUTER_URL", "http://127.0.0.1:8787")
    rest: list[str] = []
    it = iter(args)
    for arg in it:
        if arg == "--start":
            start = next(it)
        elif arg == "--plan-model":
            plan_model = next(it)
        elif arg == "--gateway":
            gateway = next(it)
        elif arg == "--prompt":
            prompt = next(it)
        else:
            rest.append(arg)
    if start not in (CHEAP, PLAN):
        print("auto-router: --start is cheap or plan", file=sys.stderr)
        return 2
    if prompt is None:
        rest, prompt = split_prompt(rest)
    return run(rest, start=start, plan_model=plan_model, gateway=gateway, prompt=prompt)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
