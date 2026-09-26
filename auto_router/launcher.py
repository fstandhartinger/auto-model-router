"""``route-run``: pick the tool for a job, then start that tool's own client.

The router's other surface is an HTTP endpoint that answers a request itself.
This one answers a different question - *which program should do this work* -
and then gets out of the way::

    route-run "add a retry with backoff to the upload client and run the tests"

It classifies the task, prices every configured route for a job of this shape,
picks one, and executes that route's own command line: a subscription coding
agent through its official CLI, a metered API model through whichever client
you use for those, a free model through a local runner. The task text and the
child's output flow through unchanged; nothing intercepts, rewrites or
inspects the traffic between that client and its provider.

That property is the point. A flat-rate plan is sold for use through its own
client, and the launcher keeps it that way: the official binary runs, signed in
the way its vendor documents, with its own credentials, and the only thing the
router contributes is the choice of *which* binary to start and with which
model flag. Nothing here reads, stores or forwards anybody's login.

What it records is the same four-part decision record the HTTP surface writes
(``decision.py``), so a launched job and a routed turn sit in one ledger:
what the task was, which routes were considered and why one won, what the run
was estimated to cost, and what actually happened - exit status and wall time,
which is all an external client reports back. A subscription run carries no
per-token cost to measure, and none is invented.

Configuration is the ordinary model config with a ``runner`` block per route::

    models:
      - name: plan-strong
        provider: anthropic
        subscription: claude
        runner:
          cmd: [claude, -p, --model, opus]
          stdin: true
          clear_env: [ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN]

``clear_env`` matters more than it looks: for both major coding subscriptions,
a credential variable left in the environment silently moves the run from the
plan to per-token billing. The launcher clears the named variables for the
child process only, and records that it did.

The child does **not** inherit the launcher's environment. It gets an
allowlist - :data:`BASE_ENV_ALLOW` (locale, ``PATH``, ``HOME`` and the like,
no credential), plus the names a route asks for with ``env_pass`` or
``env_from``, plus anything the operator adds for every route::

    launcher:
      env_allow: [HTTPS_PROXY]     # passed to every launched route
      cwd_root: ~/code             # --cwd must lie inside this directory
      inherit_env: false           # true = pre-0.4 behaviour: the whole environment

A worker is often a cheap third-party model with a shell; whatever is in its
environment is one prompt injection away from being printed. So a key only
reaches the route that names it.

The child runs in a process group of its own, and a timeout, ``Ctrl-C`` or a
``SIGTERM`` to the launcher ends that whole group, not just the child
(:mod:`auto_router.procs`).
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from . import procs
from .catalog import Catalog, ModelInfo
from .config import RouterConfig, load_config
from .decision import ObservedOutcome
from .policies import NoRouteAvailable
from .router import RouteResult, Router

#: Placeholder replaced with the task text in a runner's argument list.
TASK_PLACEHOLDER = "{task}"

#: Variables every launched route receives, when set. None of them carries a
#: credential: they locate the user's own files and tools and set the locale.
#: Deliberately absent: provider keys, ``SSH_AUTH_SOCK``, ``DBUS_*`` and
#: ``XDG_RUNTIME_DIR`` (both reach a keyring), cloud and proxy settings.
BASE_ENV_ALLOW = (
    "PATH", "HOME", "USER", "LOGNAME", "SHELL", "TERM", "COLORTERM", "NO_COLOR", "TZ", "TMPDIR",
    "LANG", "LANGUAGE", "LC_ALL", "LC_CTYPE", "LC_MESSAGES", "LC_NUMERIC", "LC_TIME",
    "XDG_CONFIG_HOME", "XDG_DATA_HOME", "XDG_CACHE_HOME", "XDG_STATE_HOME",
    "AUTO_ROUTER_DELEGATE_COPY",  # the copy a delegated worker runs in (delegate.COPY_ENV)
)


@dataclass(frozen=True)
class EnvPolicy:
    """Which of the launcher's variables a launched route may see."""

    allow: tuple[str, ...] = BASE_ENV_ALLOW
    inherit: bool = False

    @classmethod
    def from_config(cls, config: RouterConfig) -> "EnvPolicy":
        section = (config.raw or {}).get("launcher") or {}
        extra = [str(name) for name in section.get("env_allow") or []]
        return cls(allow=tuple(dict.fromkeys([*BASE_ENV_ALLOW, *extra])),
                   inherit=bool(section.get("inherit_env", False)))


def cwd_root(config: RouterConfig) -> Path | None:
    value = ((config.raw or {}).get("launcher") or {}).get("cwd_root")
    return Path(os.path.expanduser(str(value))).resolve() if value else None


def check_cwd(cwd: str | None, root: Path | None) -> None:
    """Refuse a working directory outside the configured root."""
    if cwd and not Path(cwd).is_dir():
        raise LauncherError(f"--cwd {cwd!r} is not a directory")
    if root is None:
        return
    target = Path(cwd or os.getcwd()).resolve()
    if target != root and root not in target.parents:
        raise LauncherError(f"working directory {str(target)!r} is outside launcher.cwd_root "
                            f"({str(root)!r})")


class LauncherError(RuntimeError):
    pass


@dataclass
class Command:
    """A fully resolved child process, ready to run."""

    argv: list[str]
    env: dict[str, str]
    stdin_text: str | None
    timeout_s: float
    cleared: list[str] = field(default_factory=list)
    task: str = ""

    @property
    def display(self) -> str:
        """The command line with the task elided: it can be long, and it is the user's."""
        parts = [("<task>" if self.task and self.task in a else a) for a in self.argv]
        return " ".join(shlex.quote(p) for p in parts)


@dataclass
class RunOutcome:
    exit_code: int
    duration_s: float
    timed_out: bool = False
    stdout: str | None = None


def runnable(config: RouterConfig) -> list[ModelInfo]:
    return [m for m in config.catalog.all() if m.runner]


def launcher_config(config: RouterConfig) -> RouterConfig:
    """The same configuration, restricted to routes that can run a job.

    A route without a ``runner`` is reachable over HTTP only; offering it as
    the answer to "which tool should run this" would be a decision nobody can
    execute.
    """
    models = runnable(config)
    if not models:
        raise LauncherError(
            "no route in this configuration has a runner block, so there is nothing to launch; "
            "see examples/launcher.example.yaml")
    return replace(config, catalog=Catalog(models))


def build_command(model: ModelInfo, task: str, *, cwd: str | None = None,
                  environ: dict[str, str] | None = None,
                  default_clear: dict[str, list[str]] | None = None,
                  env_policy: EnvPolicy | None = None) -> Command:
    """Resolve a route's ``runner`` block into a child process.

    ``cmd`` is a literal argument list - never a shell string - so a task that
    contains quotes, newlines or a stray ``$(...)`` is data, not code.

    The child's environment is built from an allowlist (:class:`EnvPolicy`),
    never copied wholesale, unless the operator set ``inherit_env``.
    """
    spec: dict[str, Any] = dict(model.runner or {})
    argv_template = spec.get("cmd")
    if not isinstance(argv_template, list) or not argv_template:
        raise LauncherError(f"route {model.name!r}: runner.cmd must be a non-empty list")
    use_stdin = bool(spec.get("stdin"))
    argv = [a.replace(TASK_PLACEHOLDER, task) if isinstance(a, str) else str(a)
            for a in argv_template]
    if not use_stdin and not any(TASK_PLACEHOLDER in str(a) for a in argv_template):
        raise LauncherError(
            f"route {model.name!r}: the task would never reach the tool - put {TASK_PLACEHOLDER} "
            f"in runner.cmd or set runner.stdin: true")

    source = dict(os.environ if environ is None else environ)
    policy = env_policy or EnvPolicy()
    if policy.inherit:
        env = source
    else:
        wanted = [*policy.allow, *(str(n) for n in spec.get("env_pass") or [])]
        env = {name: source[name] for name in wanted if name in source}
    clear = list(spec.get("clear_env") or (default_clear or {}).get(model.subscription or "", []))
    cleared = [name for name in clear if source.get(name)]
    for name in clear:
        # Emptied rather than deleted: an empty value is what both major
        # coding CLIs read as "no key here, use the signed-in plan", while an
        # absent variable can be re-derived from a helper or a keychain.
        env[name] = ""
    for name, value in (spec.get("env") or {}).items():
        env[str(name)] = str(value)
    for name, source in (spec.get("env_from") or {}).items():
        # Names only, never literal secrets in a config file.
        if os.environ.get(str(source)) is not None:
            env[str(name)] = os.environ[str(source)]
    if cwd:
        env["PWD"] = cwd
    return Command(argv=argv, env=env, stdin_text=task if use_stdin else None,
                   timeout_s=float(spec.get("timeout_s", 3600)), cleared=cleared, task=task)


def execute(command: Command, *, cwd: str | None = None, capture: bool = False) -> RunOutcome:
    """Run the child, streaming its output unless the caller wants it back.

    The child and everything it starts share a process group; a timeout ends
    all of them, so no writer outlives the job it belonged to.
    """
    started = time.time()
    try:
        proc = procs.run(
            command.argv, env=command.env, cwd=cwd, scope="group",
            input=command.stdin_text if command.stdin_text is not None else "",
            text=True, timeout=command.timeout_s, capture_output=capture)
    except subprocess.TimeoutExpired:
        return RunOutcome(exit_code=124, duration_s=time.time() - started, timed_out=True)
    except FileNotFoundError as exc:
        raise LauncherError(f"{command.argv[0]!r} is not installed or not on PATH") from exc
    if capture and proc.stderr:
        sys.stderr.write(proc.stderr)
    return RunOutcome(exit_code=proc.returncode, duration_s=time.time() - started,
                      stdout=proc.stdout if capture else None)


def observe(router: Router, result: RouteResult, outcome: RunOutcome, command: Command) -> None:
    """Record what the launched client did - which is exit status and time.

    An external client reports no token counts back to the launcher, so none
    are stored. A subscription run has no per-token charge to measure in the
    first place; a metered run has one, but it is the client's invoice, not
    something this process saw, and a number nobody measured is never written
    into a ledger that other numbers are compared against.
    """
    status = "ok" if outcome.exit_code == 0 and not outcome.timed_out else "transport_error"
    error = None
    if outcome.timed_out:
        error = f"timeout after {command.timeout_s:.0f}s"
    elif outcome.exit_code != 0:
        error = f"exit_{outcome.exit_code}"
    basis = ("subscription route: no per-token charge; the plan's own usage limits apply"
             if result.model.subscription else
             "launched client: no usage reported back to the launcher, so no cost is claimed")
    router.observe(result, ObservedOutcome(
        model=result.model.name, status=status, latency_ms=outcome.duration_s * 1000,
        error=error, cost_basis=basis))


def decision_document(result: RouteResult, command: Command,
                      outcome: RunOutcome | None = None) -> dict:
    doc: dict[str, Any] = {
        "route": result.model.name,
        "subscription": result.model.subscription,
        "reason": result.reason,
        "command": command.display,
        "cleared_env": command.cleared,
        "env_names": sorted(command.env),
    }
    if result.explanation is not None:
        doc["decision"] = result.explanation.to_dict()
    if outcome is not None:
        doc["outcome"] = {"exit_code": outcome.exit_code, "duration_s": round(outcome.duration_s, 2),
                          "timed_out": outcome.timed_out}
    return doc


def summary_line(result: RouteResult, command: Command) -> str:
    plan = f" on the {result.model.subscription} plan" if result.model.subscription else ""
    return (f"route-run: {result.model.name}{plan} -> {command.display}\n"
            f"           {result.reason}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="route-run",
        description="Ask the router which tool should do a job, then start that tool.")
    p.add_argument("task", nargs="?", help="the task, or - to read it from stdin")
    p.add_argument("--config", help="router config (default: $AUTO_ROUTER_CONFIG)")
    p.add_argument("--route", help="skip the decision and use this route by name")
    p.add_argument("--cwd", help="working directory for the launched client")
    p.add_argument("--steps", type=int, default=12,
                   help="expected model calls in the job; shapes the cost estimate (default: 12)")
    p.add_argument("--dry-run", action="store_true", help="decide and print, run nothing")
    p.add_argument("--json", action="store_true",
                   help="write the decision record as JSON (stdout for --dry-run, else stderr)")
    p.add_argument("--list", action="store_true", help="list the routes that can run a job")
    p.add_argument("--no-plans", action="store_true",
                   help="never pick a subscription route (used by the delegate tool, so a "
                        "plan session hands work only to cheaper routes)")
    p.add_argument("--tier", choices=("cheap", "auto", "strong"), default="auto",
                   help="worker tier, applied among the routes the policy allows (tools, "
                        "context, quota): cheap prefers the lowest price for easy work, auto "
                        "keeps the policy's choice, strong prefers the most capable")
    p.add_argument("--quiet", action="store_true", help="no summary line on stderr")
    return p


def read_task(value: str | None) -> str:
    text = sys.stdin.read() if value in (None, "-") else value
    text = text.strip()
    if not text:
        raise LauncherError("no task given")
    return text


#: Set by the first Ctrl-C of a :func:`main` run.
_INTERRUPTED = threading.Event()


def _exit_on_signal(signum, _frame):
    # After a Ctrl-C the agent is already being stopped, with the full grace
    # period (procs.run); ending it here would cut that short to terminate_all's
    # and skip what main() still does on the way out, so it is ignored, as in
    # the delegate server.
    if _INTERRUPTED.is_set():
        return
    # The agent's process group is ended on the way out instead of being left
    # running without a supervisor.
    procs.terminate_all()
    raise SystemExit(128 + signum)


def _interrupt_once(_signum, _frame):
    # Ctrl-C ends the agent's process group (SIGTERM, a grace period, SIGKILL;
    # procs.run). A second Ctrl-C raised inside that grace period would cut it
    # short and leave an agent that ignores SIGTERM running, so it is ignored.
    if _INTERRUPTED.is_set():
        return
    _INTERRUPTED.set()
    raise KeyboardInterrupt


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    previous = {}
    _INTERRUPTED.clear()
    handlers = {signal.SIGTERM: _exit_on_signal, signal.SIGHUP: _exit_on_signal,
                signal.SIGINT: _interrupt_once}
    for sig, handler in handlers.items():
        # A Ctrl-C route-run was started to ignore stays ignored.
        if sig == signal.SIGINT and signal.getsignal(sig) is signal.SIG_IGN:
            continue
        try:
            previous[sig] = signal.signal(sig, handler)
        except ValueError:  # not the main thread; nothing to install
            pass
    try:
        return _main(args)
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def _main(args: argparse.Namespace) -> int:
    try:
        config = load_config(args.config) if args.config else load_config()
        lconfig = launcher_config(config)
        if args.list:
            for m in sorted(lconfig.catalog.all(), key=lambda m: m.name):
                plan = f"  [{m.subscription} plan]" if m.subscription else ""
                print(f"{m.name}{plan}\n    {' '.join(map(str, (m.runner or {}).get('cmd', [])))}")
            return 0

        task = read_task(args.task)
        if args.route and lconfig.catalog.get(args.route) is None:
            raise LauncherError(f"no runnable route named {args.route!r}")
        check_cwd(args.cwd, cwd_root(config))
        router = Router(lconfig)
        only = (lambda m: not m.subscription) if args.no_plans else None
        # One routing pass. A tier only chooses among the routes the policy
        # itself allows, so it can never reopen a closed plan or pick a route
        # without tools, and it is recorded as a tier, not an override.
        result = router.route_job(task, steps=args.steps, force=args.route, only=only,
                                  tier=args.tier)

        default_clear = {name: list(sub.get("clear_env") or [])
                         for name, sub in (config.subscriptions or {}).items()}
        command = build_command(result.model, task, cwd=args.cwd, default_clear=default_clear,
                                env_policy=EnvPolicy.from_config(config))
        if not args.quiet:
            print(summary_line(result, command), file=sys.stderr)
        if args.dry_run:
            doc = decision_document(result, command)
            doc["tier"] = args.tier
            print(json.dumps(doc, indent=2))
            return 0

        outcome = execute(command, cwd=args.cwd)
        observe(router, result, outcome, command)
        if args.json:
            doc = decision_document(result, command, outcome)
            doc["tier"] = args.tier
            print(json.dumps(doc), file=sys.stderr)
        return outcome.exit_code
    except NoRouteAvailable as exc:
        print(f"route-run: {exc}", file=sys.stderr)
        return 3
    except LauncherError as exc:
        print(f"route-run: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover - module entry point
    raise SystemExit(main())
