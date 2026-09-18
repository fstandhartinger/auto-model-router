"""Run model-produced code under Bubblewrap, with no network and no home.

Why this exists
---------------
Grading a coding answer means executing text a model wrote. That must not
touch the machine it is graded on. Docker is not available here (the socket is
deliberately not reachable, and working around that would be the wrong fix), so
this uses ``bwrap`` with unprivileged user namespaces instead.

What the sandbox gives the code
-------------------------------
* a read-only view of ``/usr``, ``/bin``, ``/lib``, ``/lib64`` and the handful
  of ``/etc`` files Python needs to start;
* a private tmpfs for ``/tmp``, ``$HOME`` and the working directory, so nothing
  it writes survives or is visible to anything else;
* its own PID, IPC, UTS, cgroup, mount and **network** namespace: there is no
  loopback and no route out, so a generated answer cannot call an API, exfiltrate
  anything, or reach the router under test;
* an empty environment apart from ``PATH``, ``HOME``, ``LANG`` and
  ``PYTHONHASHSEED`` - no API keys, no subscription tokens;
* ``--die-with-parent`` and ``--new-session`` so it cannot outlive the harness
  or take over the controlling terminal;
* CPU-time, address-space, file-size and process-count limits applied in the
  child before exec, and a wall-clock timeout enforced by the parent.

What it does not give
---------------------
Bubblewrap is a namespace sandbox, not a VM. It does not defend against a local
kernel exploit.

It also does not hide the host's ``/usr`` from the code: the read-only binds are
what the interpreter needs to start, and anything else installed under ``/usr``
is readable. The code cannot send what it reads anywhere - there is no network -
and its output goes only to the grader, so the exposure is bounded by what the
grader then does with that output. If you point this at genuinely hostile code
rather than at benchmark answers, build a minimal root filesystem first and bind
only the interpreter.

It is the right tool for grading benchmark answers; it is not a reason to
execute code from an untrusted third party on a machine that matters.

``preflight()`` reports whether the sandbox works here, so a harness can record
an exact limitation instead of silently running code unisolated. Nothing in
this module ever falls back to running unsandboxed.
"""

from __future__ import annotations

import os
import resource
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

BWRAP = shutil.which("bwrap") or "/usr/bin/bwrap"

#: Read-only host paths the child needs to start an interpreter. Missing paths
#: are skipped rather than failing: layouts differ between distributions.
READ_ONLY = ("/usr", "/bin", "/sbin", "/lib", "/lib32", "/lib64",
             "/etc/alternatives", "/etc/ssl/certs", "/etc/localtime")


@dataclass
class Limits:
    wall_seconds: float = 20.0
    cpu_seconds: int = 15
    address_space_mb: int = 2048
    output_file_mb: int = 32
    #: Headroom above the user's *current* process count. RLIMIT_NPROC is
    #: per-uid and system-wide, so a flat cap below what the machine is already
    #: running makes ``bwrap`` fail with EAGAIN before the child ever starts.
    extra_processes: int = 256
    #: Bytes of stdout/stderr kept. Long output is truncated, never streamed.
    max_output_bytes: int = 200_000


@dataclass
class SandboxResult:
    ok: bool
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool = False
    seconds: float = 0.0
    #: Populated when the sandbox itself could not be used.
    unavailable: str | None = None
    argv: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"ok": self.ok, "returncode": self.returncode, "timed_out": self.timed_out,
                "seconds": round(self.seconds, 3), "unavailable": self.unavailable,
                "stdout_bytes": len(self.stdout), "stderr_bytes": len(self.stderr)}


def _process_cap(headroom: int) -> int | None:
    """A fork-bomb cap that is above what this user already runs, or None.

    Returns None when the current count cannot be read or the existing hard
    limit is already lower, in which case nothing is changed.
    """
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NPROC)
    except OSError:
        return None
    running = _own_threads()
    if running is None:
        return None
    cap = running + max(16, headroom)
    if hard != resource.RLIM_INFINITY and cap >= hard:
        # No cap we could install would be both effective and permitted.
        return None
    return cap


def _own_threads() -> int | None:
    """Threads belonging to this real uid.

    RLIMIT_NPROC is counted per real uid, so the right baseline is what *this
    user* is running - not the machine's total, which would make the cap far
    too loose, and not the process count, which would make it far too tight on
    a host with threaded services.
    """
    uid = os.getuid()
    total = 0
    try:
        entries = os.listdir("/proc")
    except OSError:
        return None
    for entry in entries:
        if not entry.isdigit():
            continue
        try:
            if os.stat(f"/proc/{entry}").st_uid != uid:
                continue
            with open(f"/proc/{entry}/status") as handle:
                for line in handle:
                    if line.startswith("Threads:"):
                        total += int(line.split()[1])
                        break
        except (OSError, ValueError, IndexError):
            continue
    return total or None


def _rlimits(limits: Limits):
    def apply() -> None:
        resource.setrlimit(resource.RLIMIT_CPU, (limits.cpu_seconds, limits.cpu_seconds + 1))
        space = limits.address_space_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (space, space))
        size = limits.output_file_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_FSIZE, (size, size))
        cap = _process_cap(limits.extra_processes)
        if cap:
            resource.setrlimit(resource.RLIMIT_NPROC, (cap, cap))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        os.setsid()
    return apply


def build_argv(command: list[str], workdir: Path, limits: Limits,
               ro_binds: dict[str, str] | None = None) -> list[str]:
    """The exact bwrap command line, so a run can be reproduced and audited."""
    argv = [BWRAP]
    for path in READ_ONLY:
        if Path(path).exists():
            argv += ["--ro-bind", path, path]
    for source, target in (ro_binds or {}).items():
        argv += ["--ro-bind", source, target]
    argv += [
        "--bind", str(workdir), "/work",
        "--tmpfs", "/tmp",
        "--tmpfs", "/home",
        "--proc", "/proc",
        "--dev", "/dev",
        "--chdir", "/work",
        "--unshare-all",          # user, ipc, pid, net, uts, cgroup, mount
        "--die-with-parent",
        "--new-session",
        "--cap-drop", "ALL",
        "--clearenv",
        "--setenv", "PATH", "/usr/local/bin:/usr/bin:/bin",
        "--setenv", "HOME", "/work",
        "--setenv", "TMPDIR", "/tmp",
        "--setenv", "LANG", "C.UTF-8",
        "--setenv", "PYTHONHASHSEED", "0",
        "--setenv", "PYTHONDONTWRITEBYTECODE", "1",
        "--",
    ]
    return argv + command


class UnsafePath(ValueError):
    """A requested file name would land outside the private work directory."""


def _safe_target(workdir: Path, name: str) -> Path:
    """Resolve ``name`` strictly beneath ``workdir``.

    ``files={"../../etc/x": ...}`` or an absolute path would otherwise be
    written on the host, with the harness's privileges, *before* the sandbox
    starts - so the sandbox would never see it and could not stop it.
    """
    candidate = Path(name)
    if candidate.is_absolute() or any(part == ".." for part in candidate.parts):
        raise UnsafePath(f"{name!r} must be a relative path inside the work directory")
    target = (workdir / candidate).resolve()
    root = workdir.resolve()
    if target != root and root not in target.parents:
        raise UnsafePath(f"{name!r} resolves outside the work directory")
    return target


def run(command: list[str], *, files: dict[str, str] | None = None, stdin: str = "",
        limits: Limits | None = None, ro_binds: dict[str, str] | None = None) -> SandboxResult:
    """Run ``command`` in a fresh sandbox. ``files`` are written into /work first."""
    limits = limits or Limits()
    if not Path(BWRAP).exists():
        return SandboxResult(False, -1, "", "", unavailable="bwrap is not installed")
    with tempfile.TemporaryDirectory(prefix="auto-router-sandbox-") as tmp:
        workdir = Path(tmp)
        try:
            for name, content in (files or {}).items():
                target = _safe_target(workdir, name)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content)
        except UnsafePath as exc:
            return SandboxResult(False, -1, "", "", unavailable=str(exc))
        argv = build_argv(command, workdir, limits, ro_binds)
        import time
        started = time.perf_counter()
        # Output goes to files inside the work directory, not to pipes. A pipe
        # is buffered in the harness's memory and RLIMIT_FSIZE does not apply to
        # it, so a program that prints continuously for the whole timeout can
        # exhaust the harness. Written to a file, the same program hits
        # RLIMIT_FSIZE and dies.
        out_path, err_path = workdir / ".stdout", workdir / ".stderr"
        try:
            with out_path.open("wb") as out, err_path.open("wb") as err:
                proc = subprocess.run(argv, input=stdin.encode(), stdout=out, stderr=err,
                                      timeout=limits.wall_seconds, preexec_fn=_rlimits(limits))
            returncode, timed_out = proc.returncode, False
        except subprocess.TimeoutExpired:
            returncode, timed_out = -9, True
        except OSError as exc:
            return SandboxResult(False, -1, "", "", unavailable=f"{type(exc).__name__}: {exc}",
                                 argv=argv)
        seconds = time.perf_counter() - started
        stdout = _read_capped(out_path, limits)
        stderr = _read_capped(err_path, limits)
        if timed_out:
            return SandboxResult(False, -9, stdout, stderr, timed_out=True,
                                 seconds=limits.wall_seconds, argv=argv)
        if returncode != 0 and ("setting up uid map" in stderr
                                or "Creating new namespace" in stderr):
            # bwrap itself refused (no user namespaces, restricted kernel): this
            # is an unavailable sandbox, not a failing program.
            return SandboxResult(False, returncode, "", stderr,
                                 unavailable=stderr.strip()[:200], argv=argv)
        return SandboxResult(returncode == 0, returncode, stdout, stderr, seconds=seconds,
                             argv=argv)


def _read_capped(path: Path, limits: Limits) -> str:
    """At most ``max_output_bytes`` from a capture file, never the whole thing."""
    try:
        with path.open("rb") as handle:
            return handle.read(limits.max_output_bytes).decode("utf-8", "replace")
    except OSError:
        return ""


def _text(value, limits: Limits) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        value = value.decode("utf-8", "replace")
    return value[: limits.max_output_bytes]


def run_python(code: str, *, stdin: str = "", limits: Limits | None = None,
               extra_files: dict[str, str] | None = None) -> SandboxResult:
    """Run one Python program. The interpreter comes from the read-only host bind."""
    files = {"main.py": code, **(extra_files or {})}
    return run(["/usr/bin/python3", "-I", "-S", "main.py"], files=files, stdin=stdin,
               limits=limits)


def preflight() -> dict:
    """Prove the sandbox works here, or say exactly why it does not.

    A harness records this before grading anything, so "we could not isolate
    execution" is never confused with "the answers were wrong".
    """
    probe = (
        "import os, socket, json\n"
        "out = {}\n"
        "try:\n"
        "    socket.create_connection(('1.1.1.1', 53), timeout=2); out['network'] = 'reachable'\n"
        "except Exception as exc:\n"
        "    out['network'] = 'blocked:' + type(exc).__name__\n"
        "out['home_readable'] = os.path.exists(%r)\n"
        "out['secret_env'] = [k for k in os.environ if 'KEY' in k or 'TOKEN' in k]\n"
        "out['cwd'] = os.getcwd()\n"
        "out['writable_work'] = os.access('/work', os.W_OK)\n"
        "print(json.dumps(out))\n" % (os.path.expanduser("~/.bashrc"),)
    )
    result = run_python(probe, limits=Limits(wall_seconds=30))
    report = {"bwrap": BWRAP, "available": False, "checks": {}, "raw": result.to_dict()}
    if result.unavailable:
        report["reason"] = result.unavailable
        return report
    if not result.ok:
        report["reason"] = (result.stderr or result.stdout).strip()[:300]
        return report
    import json as _json
    try:
        checks = _json.loads(result.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        report["reason"] = "probe produced no parsable output"
        return report
    report["checks"] = checks
    report["available"] = (
        checks.get("network", "").startswith("blocked")
        and checks.get("home_readable") is False
        and not checks.get("secret_env")
    )
    if not report["available"]:
        report["reason"] = "sandbox ran but did not isolate: " + _json.dumps(checks)
    return report


if __name__ == "__main__":
    import json as _json
    print(_json.dumps(preflight(), indent=1))
