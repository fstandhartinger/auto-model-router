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
kernel exploit. It is the right tool for grading benchmark answers; it is not a
reason to execute code from an untrusted third party on a machine that matters.

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
        # Field 4 of /proc/loadavg is "running/total" *threads* system-wide.
        # RLIMIT_NPROC counts threads too, so a cap based on the process count
        # alone is far too low on a machine with threaded services.
        total = int(open("/proc/loadavg").read().split()[3].split("/")[1])
        soft, hard = resource.getrlimit(resource.RLIMIT_NPROC)
    except (OSError, IndexError, ValueError):
        return None
    cap = total + max(16, headroom)
    if hard != resource.RLIM_INFINITY and cap >= hard:
        return None
    return cap


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


def run(command: list[str], *, files: dict[str, str] | None = None, stdin: str = "",
        limits: Limits | None = None, ro_binds: dict[str, str] | None = None) -> SandboxResult:
    """Run ``command`` in a fresh sandbox. ``files`` are written into /work first."""
    limits = limits or Limits()
    if not Path(BWRAP).exists():
        return SandboxResult(False, -1, "", "", unavailable="bwrap is not installed")
    with tempfile.TemporaryDirectory(prefix="auto-router-sandbox-") as tmp:
        workdir = Path(tmp)
        for name, content in (files or {}).items():
            target = workdir / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content)
        argv = build_argv(command, workdir, limits, ro_binds)
        import time
        started = time.perf_counter()
        try:
            proc = subprocess.run(argv, input=stdin, capture_output=True, text=True,
                                  timeout=limits.wall_seconds, preexec_fn=_rlimits(limits))
        except subprocess.TimeoutExpired as exc:
            return SandboxResult(False, -9, _text(exc.stdout, limits), _text(exc.stderr, limits),
                                 timed_out=True, seconds=limits.wall_seconds, argv=argv)
        except OSError as exc:
            return SandboxResult(False, -1, "", "", unavailable=f"{type(exc).__name__}: {exc}",
                                 argv=argv)
        seconds = time.perf_counter() - started
        stderr = _text(proc.stderr, limits)
        if proc.returncode != 0 and "bwrap:" in stderr and "No such file" not in stderr:
            # bwrap itself refused (no user namespaces, restricted kernel): this
            # is an unavailable sandbox, not a failing program.
            if "setting up uid map" in stderr or "Creating new namespace" in stderr:
                return SandboxResult(False, proc.returncode, "", stderr,
                                     unavailable=stderr.strip()[:200], argv=argv)
        return SandboxResult(proc.returncode == 0, proc.returncode, _text(proc.stdout, limits),
                             stderr, seconds=seconds, argv=argv)


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
