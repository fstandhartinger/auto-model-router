"""Flat-rate subscription quota as a routing tier.

A subscription (e.g. a Claude or ChatGPT plan used through its official CLI)
has zero marginal dollar cost, but the quota is shared with the people who work
interactively on the same account. Burning it on background traffic is only
free until someone needs it. So the router prices subscription calls with a
*shadow price*:

    multiplier = 0     plenty of projected slack: use it, it would expire unused
    multiplier -> 1    projected weekly use approaches the reserve line
    closed             projected use above the reserve, used above the hard stop,
                       or the short (session) window nearly full

and ``effective cost = multiplier * list-price cost``. The router therefore
prefers the subscription while it is genuinely spare and stops well before it
would hurt interactive work.

Projection: ``used + rate * remaining`` where ``rate = used / elapsed`` over the
current weekly window. A late-week exception raises the reserve line when the
window resets soon and a lot is left (unused quota is lost at reset).

Only use this for your own agents on your own subscription, through the
vendor's official client, and within the vendor's terms. Never route other
people's traffic through a personal plan.
"""

from __future__ import annotations

import glob
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

WEEK_SECONDS = 7 * 24 * 3600


@dataclass(frozen=True)
class QuotaState:
    week_used: float                 # 0..1
    week_resets_at: float | None     # epoch seconds
    session_used: float | None = None  # 0..1, short rolling window
    measured_at: float = 0.0
    source: str = "unknown"


@dataclass(frozen=True)
class PacingRule:
    weekly_reserve: float = 0.65     # projected end-of-week use the router may fill up to
    hard_stop: float = 0.80          # never route when current weekly use is at or above this
    session_stop: float = 0.70       # never push the short window above this
    ramp: float = 0.15               # width of the band where the shadow price rises 0 -> 1
    late_week_hours: float = 36.0    # "use it or lose it" window before reset
    late_week_reserve: float = 0.85
    max_staleness_s: float = 4 * 3600


@dataclass(frozen=True)
class QuotaDecision:
    open: bool
    multiplier: float
    projected: float | None
    reason: str


def _to_epoch(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def decide(state: QuotaState | None, rule: PacingRule = PacingRule(), now: float | None = None) -> QuotaDecision:
    now = now or time.time()
    if state is None:
        return QuotaDecision(False, 1.0, None, "no quota measurement: subscription closed")
    if state.measured_at and now - state.measured_at > rule.max_staleness_s:
        return QuotaDecision(False, 1.0, None, "quota measurement is stale: subscription closed")
    used = state.week_used
    if used >= rule.hard_stop:
        return QuotaDecision(False, 1.0, used, f"weekly use {used:.0%} at or above hard stop {rule.hard_stop:.0%}")
    if state.session_used is not None and state.session_used >= rule.session_stop:
        return QuotaDecision(False, 1.0, used, f"session window {state.session_used:.0%} too full")

    reserve = rule.weekly_reserve
    projected = used
    if state.week_resets_at:
        remaining = max(0.0, state.week_resets_at - now)
        elapsed = max(3600.0, WEEK_SECONDS - remaining)
        projected = used + used / elapsed * remaining
        if remaining <= rule.late_week_hours * 3600:
            reserve = max(reserve, rule.late_week_reserve)
    if projected >= reserve:
        return QuotaDecision(False, 1.0, projected,
                             f"projected weekly use {projected:.0%} >= reserve {reserve:.0%}")
    slack = reserve - projected
    multiplier = 0.0 if slack >= rule.ramp else 1.0 - slack / rule.ramp
    return QuotaDecision(True, round(multiplier, 3), projected,
                         f"projected {projected:.0%} of week, reserve {reserve:.0%}: price x{multiplier:.2f}")


# -- readers ----------------------------------------------------------------
def _state_from_document(data: dict, name: str, *, measured_at: float, source: str) -> QuotaState | None:
    """``{name: {week_percent, session_percent, week_resets_at}, generated_at}``."""
    entry = data.get(name) or {}
    week = entry.get("week_percent", entry.get("percent"))
    if not isinstance(week, (int, float)):
        return None
    session = entry.get("session_percent")
    return QuotaState(
        week_used=float(week) / 100.0,
        week_resets_at=_to_epoch(entry.get("week_resets_at") or entry.get("resets_at")),
        session_used=float(session) / 100.0 if isinstance(session, (int, float)) else None,
        measured_at=_to_epoch(data.get("generated_at")) or measured_at,
        source=source,
    )


def from_budget_file(path: str | os.PathLike, name: str) -> QuotaState | None:
    """Generic budget JSON written by whatever measures the plan on this machine."""
    p = Path(path).expanduser()
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text())
    except json.JSONDecodeError:
        return None
    return _state_from_document(data, name, measured_at=p.stat().st_mtime, source=str(p))


#: Command results, so a per-turn routing decision does not spawn a usage
#: reader: {(command, plan name) -> (read_at, state)}. The plan name belongs in
#: the key: one reader usually reports every plan on the machine, and caching
#: by command alone would hand the first plan's usage to the second.
_command_cache: dict[tuple[tuple[str, ...], str], tuple[float, QuotaState | None]] = {}


def from_command(command: list[str], name: str, *, ttl_s: float = 600.0,
                 timeout_s: float = 30.0) -> QuotaState | None:
    """Ask the plan's own usage reader, in the same JSON shape as the budget file.

    The router has no business reading anyone's login token, so it never asks a
    vendor how full a plan is - it runs *your* reader and parses percentages.
    On this machine that is the client's own usage command; in a container it
    might be a script that echoes a cached number. A reader that fails, times
    out or prints something unparseable yields ``None``, which closes the
    subscription tier rather than opening it on a guess (see ``decide``).
    """
    import subprocess

    key = (tuple(command), name)
    now = time.time()
    cached = _command_cache.get(key)
    if cached and now - cached[0] < ttl_s:
        return cached[1]
    state: QuotaState | None = None
    try:
        proc = subprocess.run(command, capture_output=True, text=True, timeout=timeout_s)
        if proc.returncode == 0 and proc.stdout.strip():
            state = _state_from_document(json.loads(proc.stdout), name,
                                         measured_at=now, source=f"command:{command[0]}")
    except (OSError, ValueError, subprocess.SubprocessError):
        state = None
    _command_cache[key] = (now, state)
    return state


def from_codex_rollouts(root: str | os.PathLike = "~/.codex/sessions", max_files: int = 40) -> QuotaState | None:
    """Latest ``rate_limits`` block the Codex CLI wrote into its local rollout logs."""
    files = sorted(glob.glob(os.path.join(os.path.expanduser(str(root)), "**", "rollout-*.jsonl"),
                             recursive=True), key=os.path.getmtime, reverse=True)[:max_files]
    for path in files:
        best = None
        with open(path, errors="replace") as fh:
            for line in fh:
                if '"rate_limits"' not in line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                limits = _find_key(event, "rate_limits")
                primary = (limits or {}).get("primary") or {}
                if isinstance(primary.get("used_percent"), (int, float)):
                    best = (primary, event)
        if best:
            primary, event = best
            return QuotaState(
                week_used=primary["used_percent"] / 100.0,
                week_resets_at=_to_epoch(primary.get("resets_at")),
                measured_at=_to_epoch(event.get("timestamp")) or os.path.getmtime(path),
                source=path,
            )
    return None


def _find_key(obj: Any, key: str) -> Any:
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for v in obj.values():
            found = _find_key(v, key)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for v in obj:
            found = _find_key(v, key)
            if found is not None:
                return found
    return None
