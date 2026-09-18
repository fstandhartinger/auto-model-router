"""The execution sandbox used to grade model-produced code.

These are the properties that make it safe to run a generated answer at all.
If any of them stops holding, grading must stop rather than fall back to
running the code unisolated.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments import sandbox  # noqa: E402

pytestmark = pytest.mark.skipif(not Path(sandbox.BWRAP).exists(), reason="bwrap is not installed")


@pytest.fixture(scope="module")
def available():
    report = sandbox.preflight()
    if not report["available"]:
        pytest.skip(f"sandbox unavailable: {report.get('reason')}")
    return report


def test_preflight_reports_real_isolation(available):
    checks = available["checks"]
    assert checks["network"].startswith("blocked")
    assert checks["home_readable"] is False
    assert checks["secret_env"] == []
    assert checks["cwd"] == "/work"


def test_a_plain_program_runs_and_returns_output(available):
    result = sandbox.run_python("print(2 + 2)")
    assert result.ok and result.stdout.strip() == "4"


def test_the_network_is_unreachable(available):
    result = sandbox.run_python(
        "import socket\n"
        "try:\n"
        "    socket.create_connection(('1.1.1.1', 53), timeout=3)\n"
        "    print('REACHED')\n"
        "except OSError as exc:\n"
        "    print('BLOCKED', type(exc).__name__)\n")
    assert "BLOCKED" in result.stdout and "REACHED" not in result.stdout


def test_loopback_is_unreachable_too(available):
    """The router under test runs on loopback; a graded answer must not see it."""
    result = sandbox.run_python(
        "import socket\n"
        "s = socket.socket()\n"
        "s.settimeout(2)\n"
        "print('OPEN' if s.connect_ex(('127.0.0.1', 22)) == 0 else 'CLOSED')\n")
    assert "OPEN" not in result.stdout


def test_the_home_directory_is_not_visible(available):
    result = sandbox.run_python(
        "import os\n"
        "print('ENTRIES', sorted(os.listdir('/home')))\n"
        "print('BASHRC', os.path.exists(os.path.expanduser('~/.bashrc')))\n")
    assert "BASHRC False" in result.stdout
    assert "ENTRIES []" in result.stdout


def test_no_credentials_are_inherited(available, monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "sk-should-never-appear")
    monkeypatch.setenv("SOME_SUBSCRIPTION_TOKEN", "oat-should-never-appear")
    result = sandbox.run_python("import os; print(sorted(os.environ))")
    assert "should-never-appear" not in result.stdout
    assert "TYPESAFE_API_KEY" not in result.stdout
    assert "SOME_SUBSCRIPTION_TOKEN" not in result.stdout


def test_the_host_filesystem_is_read_only(available, tmp_path):
    marker = tmp_path / "must-not-change.txt"
    marker.write_text("original")
    result = sandbox.run_python(
        "import pathlib\n"
        "try:\n"
        "    pathlib.Path('/usr/pwned').write_text('x'); print('WROTE')\n"
        "except OSError as exc:\n"
        "    print('READONLY', type(exc).__name__)\n")
    assert "READONLY" in result.stdout and "WROTE" not in result.stdout
    assert marker.read_text() == "original"


def test_work_is_private_and_does_not_leak_out(available):
    first = sandbox.run_python("open('/work/state', 'w').write('from the first run'); print('ok')")
    assert first.ok
    second = sandbox.run_python("import os; print('LEAKED' if os.path.exists('/work/state') else 'CLEAN')")
    assert "CLEAN" in second.stdout


def test_a_wall_clock_timeout_is_enforced(available):
    result = sandbox.run_python("import time\nwhile True: time.sleep(0.1)\n",
                                limits=sandbox.Limits(wall_seconds=3))
    assert result.timed_out and not result.ok


def test_a_cpu_burner_is_stopped(available):
    result = sandbox.run_python("while True: pass",
                                limits=sandbox.Limits(wall_seconds=20, cpu_seconds=2))
    assert not result.ok


def test_a_memory_hog_is_stopped(available):
    result = sandbox.run_python("x = bytearray(1024 * 1024 * 1024 * 4)\nprint('ALLOCATED')",
                                limits=sandbox.Limits(address_space_mb=256))
    assert "ALLOCATED" not in result.stdout and not result.ok


def test_extra_files_are_available_to_the_program(available):
    result = sandbox.run_python("print(open('data.txt').read().strip())",
                                extra_files={"data.txt": "hidden test input"})
    assert result.stdout.strip() == "hidden test input"


def test_stdin_is_delivered(available):
    result = sandbox.run_python("import sys; print(sum(int(x) for x in sys.stdin.read().split()))",
                                stdin="1 2 3 4")
    assert result.stdout.strip() == "10"


def test_a_failing_program_is_reported_not_raised(available):
    result = sandbox.run_python("raise SystemExit(3)")
    assert not result.ok and result.returncode == 3


def test_oversized_output_is_truncated(available):
    limits = sandbox.Limits(max_output_bytes=5_000)
    result = sandbox.run_python("print('z' * 500000)", limits=limits)
    assert len(result.stdout) <= limits.max_output_bytes


def test_the_command_line_is_recorded_for_audit(available):
    result = sandbox.run_python("print(1)")
    assert "--unshare-all" in result.argv
    assert "--die-with-parent" in result.argv
    assert "--clearenv" in result.argv
    assert result.argv.count("--bind") == 1, "only the private work directory is writable"


def test_a_missing_bwrap_is_reported_not_bypassed(monkeypatch):
    """The harness must never silently run generated code unisolated."""
    monkeypatch.setattr(sandbox, "BWRAP", "/nonexistent/bwrap")
    result = sandbox.run_python("print('should not run')")
    assert not result.ok and result.unavailable
    assert "should not run" not in result.stdout


def test_the_process_cap_is_above_this_user_and_below_the_machine():
    """Two ways to get RLIMIT_NPROC wrong, both of which happened here.

    Too low and ``bwrap`` fails with EAGAIN before the child ever starts,
    because the limit is per real uid and system-wide. Based on the *machine's*
    total thread count instead of this user's, the headroom is so large that the
    fork-bomb cap does nothing.
    """
    cap = sandbox._process_cap(256)
    own = sandbox._own_threads()
    if cap is None or own is None:
        pytest.skip("no usable process limit on this host")
    assert cap > own, "a cap below what this user already runs breaks bwrap outright"
    assert cap <= own + 512, "the cap must still bound a fork bomb"


def test_own_thread_count_is_smaller_than_the_machine_total():
    own = sandbox._own_threads()
    if own is None:
        pytest.skip("/proc is not readable")
    machine = int(open("/proc/loadavg").read().split()[3].split("/")[1])
    assert 0 < own <= machine


def test_a_file_outside_the_work_directory_is_refused(tmp_path):
    """Files are written by the harness before the sandbox starts, so a
    traversing name would be written on the host with harness privileges."""
    victim = tmp_path / "must-not-exist.txt"
    for name in ("../escape.txt", "../../etc/passwd", "/etc/passwd", "a/../../escape.txt"):
        result = sandbox.run_python("print(1)", extra_files={name: "x"})
        assert not result.ok and result.unavailable, name
        assert "outside" in result.unavailable or "relative path" in result.unavailable
    assert not victim.exists()


def test_a_nested_relative_file_is_still_allowed(available):
    result = sandbox.run_python("print(open('pkg/data.txt').read().strip())",
                                extra_files={"pkg/data.txt": "nested ok"})
    assert result.stdout.strip() == "nested ok"


def test_a_program_that_floods_stdout_does_not_exhaust_the_harness(available):
    """Captured output goes to a file inside the sandbox, where RLIMIT_FSIZE
    applies; a pipe would be buffered in the harness's own memory instead."""
    limits = sandbox.Limits(wall_seconds=15, output_file_mb=2, max_output_bytes=50_000)
    result = sandbox.run_python(
        "import sys\nline = 'x' * 4096\nwhile True:\n    sys.stdout.write(line)\n",
        limits=limits)
    assert not result.ok
    assert len(result.stdout) <= limits.max_output_bytes
