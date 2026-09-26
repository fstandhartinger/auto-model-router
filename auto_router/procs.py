"""Run a launched agent so that a timeout ends *everything* it started.

An agent CLI is a program that starts programs: a shell, a test runner, a
language server, sometimes another agent. ``subprocess.run(timeout=...)``
kills only the direct child, so on a timeout the rest keep running - and keep
writing into the working tree - after the caller has reported the job as
over. Found in review: a delegated worker that "timed out after 1 s" wrote a
file five seconds later.

Here the child starts in a process group of its own (``scope="group"``, used
by the launcher, so an interactive ``route-run`` stays in the user's session)
or in a new session (``scope="session"``, used by the delegate server, whose
launcher then contains its agent in a group *inside* that session). On a
timeout, an interrupt or a ``SystemExit`` every process in that group or
session, plus any descendant still linked by parent id, gets ``SIGTERM``, then
``SIGKILL`` after a short grace period, and the call returns only once none of
them is left alive. A stop signal that arrives while a job is being started
waits until the job is registered, so it ends that job too; one that arrives
while the leftover of an already-exited job is being stopped ends that
leftover too, because the job stays registered until it is gone.

The limit is the kernel's: a descendant that calls ``setsid()`` *and* whose
parent has already exited is no longer linked to the job in any way a process
can see. Containing that needs a cgroup or a sandbox, which is the operator's
choice (README, "Use it as a subagent layer").
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
import time
from typing import Any

#: Seconds between SIGTERM and SIGKILL.
GRACE_S = 2.0

#: Jobs currently running under :func:`run`, so a supervisor that is itself
#: told to stop can end them (:func:`terminate_all`).
_LIVE: dict[subprocess.Popen, str] = {}
#: Reentrant because a stop-signal handler calls :func:`terminate_all` on the
#: main thread, which may be inside one of this lock's own blocks at that
#: moment; a plain Lock deadlocked there.
_LIVE_LOCK = threading.RLock()


def _table() -> dict[int, tuple[str, int, int, int]] | None:
    """pid -> (state, ppid, pgid, sid) from /proc, or None where there is none."""
    try:
        names = os.listdir("/proc")
    except OSError:
        return None
    out: dict[int, tuple[str, int, int, int]] = {}
    for name in names:
        if not name.isdigit():
            continue
        try:
            with open(f"/proc/{name}/stat", "rb") as fh:
                raw = fh.read().decode("utf-8", "replace")
            fields = raw[raw.rfind(")") + 2:].split()
            out[int(name)] = (fields[0], int(fields[1]), int(fields[2]), int(fields[3]))
        except (OSError, IndexError, ValueError):
            continue
    return out


def members(root: int, *, scope: str, children: bool = True) -> set[int] | None:
    """Live processes belonging to the job rooted at ``root`` (None: cannot tell).

    ``children`` also follows parent links from ``root``; it is only safe
    while ``root`` has not been reaped, because a reaped pid can be reused.
    (A pid that is still some group's or session's id is never reused, so the
    group and session match stays safe.)
    """
    table = _table()
    if table is None:
        return None
    found = {pid for pid, (_, _, pgid, sid) in table.items()
             if pgid == root or (scope == "session" and sid == root)}
    frontier = (found | {root}) if children else set(found)
    while frontier:
        kids = {pid for pid, (_, ppid, _, _) in table.items()
                if ppid in frontier and pid not in found}
        found |= kids
        frontier = kids
    found.discard(os.getpid())
    return {pid for pid in found if table.get(pid, ("Z",))[0] not in ("Z", "X")}


def start_time(pid: int) -> int | None:
    """When ``pid`` started, in clock ticks since boot (None: no such process).

    With the boot id, this tells a process from a later one given the same pid.
    """
    try:
        with open(f"/proc/{pid}/stat", "rb") as fh:
            raw = fh.read().decode("utf-8", "replace")
        return int(raw[raw.rfind(")") + 2:].split()[19])
    except (OSError, IndexError, ValueError):
        return None


def _references(base: str, majors: dict[tuple[int, int], int]) -> set[tuple[int, int]]:
    """``(st_dev, st_ino)`` of what the process at ``base`` (``/proc/<pid>``) holds."""
    refs = set()
    for link in [f"{base}/cwd", f"{base}/root", f"{base}/exe",
                 *(f"{base}/fd/{fd}" for fd in os.listdir(f"{base}/fd"))]:
        try:
            st = os.stat(link)
        except FileNotFoundError:  # a closed fd, a deleted file
            continue
        refs.add((st.st_dev, st.st_ino))
    with open(f"{base}/maps") as fh:
        for line in fh:
            parts = line.split()
            if len(parts) >= 6 and parts[4] != "0":
                major, minor = parts[3].split(":")
                dev = majors.get((int(major, 16), int(minor, 16)))
                if dev is not None:
                    refs.add((dev, int(parts[4])))
    return refs


def holders(inodes: set[tuple[int, int]], *, since: int,
            environ: bytes | None = None) -> tuple[set[int], set[int]] | None:
    """Live processes that hold one of ``inodes`` (``(st_dev, st_ino)``) open.

    Held means as working or root directory, executable, open file or mapped
    file. With ``environ`` (``b"NAME=value"``), a process started at or after
    ``since`` whose environment, as it was started, has that entry counts as
    holding too: it was started by one that had it, so it may write into what
    the entry names even once it has detached and holds nothing there.
    Returns ``(holding, unreadable)``: the pids seen holding one, and the
    pids started at or after ``since`` (clock ticks since boot) whose
    references cannot be read - another user's or a non-dumpable process.
    None when ``/proc`` cannot be read at all. Only processes this ``/proc``
    shows are seen.
    """
    try:
        names = os.listdir("/proc")
    except OSError:
        return None
    majors = {(os.major(dev), os.minor(dev)): dev for dev, _ in inodes}
    holding: set[int] = set()
    unreadable: set[int] = set()
    for name in names:
        if not name.isdigit() or int(name) == os.getpid():
            continue
        pid, base = int(name), f"/proc/{name}"
        started: int | None = None
        try:
            with open(f"{base}/stat", "rb") as fh:
                raw = fh.read().decode("utf-8", "replace")
            fields = raw[raw.rfind(")") + 2:].split()
            if fields[0] in ("Z", "X"):
                continue
            started = int(fields[19])
            refs = _references(base, majors)
            marked = False
            if environ is not None and started >= since:
                with open(f"{base}/environ", "rb") as fh:
                    marked = environ in fh.read().split(b"\0")
        except (FileNotFoundError, ProcessLookupError):
            continue  # gone meanwhile
        except (OSError, IndexError, ValueError):
            if started is None or started >= since:
                unreadable.add(pid)
            continue
        if marked or refs & inodes:
            holding.add(pid)
    return holding, unreadable


def _signal(root: int, targets: set[int] | None, sig: int) -> None:
    try:
        os.killpg(root, sig)
    except (ProcessLookupError, PermissionError):
        pass
    for pid in targets or ():
        try:
            os.kill(pid, sig)
        except (ProcessLookupError, PermissionError):
            pass


def terminate(proc: subprocess.Popen, *, scope: str, grace_s: float = GRACE_S) -> None:
    """End the whole job: TERM, a grace period, then KILL; reap the direct child."""
    if proc.returncode is not None and members(proc.pid, scope=scope, children=False) == set():
        # Reaped and nothing left in its group or session: its pid may be
        # anyone's by now, so nothing is signalled. (Without /proc nothing is
        # known and the group is signalled as before.)
        return
    for sig, wait_s in ((signal.SIGTERM, grace_s), (signal.SIGKILL, 5.0)):
        reaped = proc.returncode is not None
        _signal(proc.pid, members(proc.pid, scope=scope, children=not reaped), sig)
        deadline = time.monotonic() + wait_s
        while time.monotonic() < deadline:
            proc.poll()
            left = members(proc.pid, scope=scope, children=False)
            if proc.returncode is not None and not left:
                break
            time.sleep(0.02)
        if proc.returncode is not None and not members(proc.pid, scope=scope, children=False):
            return
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:  # pragma: no cover - SIGKILL cannot be ignored
        pass


def terminate_all() -> None:
    """End every job started by :func:`run` that is still running."""
    with _LIVE_LOCK:
        live = list(_LIVE.items())
    for proc, scope in live:
        terminate(proc, scope=scope, grace_s=0.5)


#: The signals this package's entry points stop on (launcher, delegate).
_STOP_SIGNALS = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)


class _HeldSignals:
    """Lets a stop signal wait while a job is started and registered.

    Between the fork and the registration in ``_LIVE`` a handler that raises
    (Ctrl-C, the servers' SIGTERM/SIGHUP handlers) would leave the job running
    where neither :func:`run`'s own cleanup nor :func:`terminate_all` sees it.
    So, on the main thread (the only one Python runs handlers on), each Python
    handler of a stop signal is replaced by this object, which notes the
    signal while held and hands it to the replaced handler once released.
    Blocking the signals instead would not do: the child inherits the mask.
    """

    def __init__(self) -> None:
        self.previous: dict[int, Any] = {}
        self.pending: list[int] = []
        self.holding = True

    def __call__(self, signum: int, frame: Any) -> None:
        if self.holding:
            self.pending.append(signum)
        else:
            self.previous[signum](signum, frame)

    def hold(self) -> None:
        if threading.current_thread() is not threading.main_thread():
            return
        for sig in _STOP_SIGNALS:
            if callable(signal.getsignal(sig)):
                self.previous[sig] = signal.signal(sig, self)

    def release(self) -> None:
        """Put the handlers back, then deliver what came meanwhile, in order."""
        self.holding = False
        for sig, handler in self.previous.items():
            signal.signal(sig, handler)
        for signum in self.pending:
            signal.raise_signal(signum)


def run(argv: list[str], *, timeout: float | None, scope: str = "session",
        input: str | None = None, capture_output: bool = False, text: bool = True,
        stdin: Any = None, cwd: str | None = None, env: dict[str, str] | None = None,
        grace_s: float = GRACE_S) -> subprocess.CompletedProcess:
    """``subprocess.run`` with the timeout applied to the whole job.

    Raises ``subprocess.TimeoutExpired`` only after every contained process is
    gone.
    """
    if scope not in ("session", "group"):
        raise ValueError(f"unknown scope {scope!r}")
    kwargs: dict[str, Any] = {}
    if scope == "session":
        kwargs["start_new_session"] = True
    elif sys.version_info >= (3, 11):
        kwargs["process_group"] = 0
    else:  # pragma: no cover - Python 3.10
        kwargs["preexec_fn"] = os.setpgrp
    if input is not None:
        stdin = subprocess.PIPE
    pipe = subprocess.PIPE if capture_output else None
    held = _HeldSignals()
    try:
        held.hold()
        proc = subprocess.Popen(argv, stdin=stdin, stdout=pipe, stderr=pipe, cwd=cwd, env=env,
                                text=text, **kwargs)
        with _LIVE_LOCK:
            _LIVE[proc] = scope
    except BaseException:
        held.release()
        raise
    try:
        held.release()
        out, err = proc.communicate(input, timeout=timeout)
        # The direct child finished, but it may have left a background process
        # behind in its group; that one is as much a stray writer as a hung
        # child. The job stays registered and inside this try while that is
        # stopped, so a stop signal meanwhile still ends it (terminate_all or
        # the except-block below) instead of cutting the cleanup short.
        if members(proc.pid, scope=scope, children=False):
            terminate(proc, scope=scope, grace_s=grace_s)
    except subprocess.TimeoutExpired:
        terminate(proc, scope=scope, grace_s=grace_s)
        raise subprocess.TimeoutExpired(argv, timeout) from None
    except BaseException:
        # Ctrl-C, SIGTERM turned into SystemExit, a crash in the caller: the
        # job must not outlive the process that was supervising it.
        terminate(proc, scope=scope, grace_s=grace_s)
        raise
    finally:
        with _LIVE_LOCK:
            _LIVE.pop(proc, None)
    return subprocess.CompletedProcess(argv, proc.returncode, out, err)
