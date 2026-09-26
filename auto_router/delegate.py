"""MCP tools for handing bounded work from a strong planner to routed workers.

The planner's client remains unchanged. Workers are launched through the normal
job router, never through a subscription route, and the response distinguishes
the router's pre-run cost estimate from measured cost. No launched agent CLI
reports token usage back to the launcher, so ``cost_usd`` is always ``null``.

Boundaries the server enforces, because a worker is usually a cheap
third-party model with a shell:

- **Arguments** are validated against the tool schema before anything runs; a
  malformed call gets a structured error and the server keeps serving.
- **The brief is data.** It follows ``--`` on the launcher's command line, so a
  brief that starts with ``-`` cannot become a launcher option.
- **Working directory.** ``cwd`` must lie inside the delegation root:
  ``AUTO_ROUTER_DELEGATE_ROOT``, or else the directory the server was started
  in (the project the planner's client opened). A root of ``/``, ``$HOME`` or
  a parent of ``$HOME`` is refused as too broad.
- **Environment.** The launcher starts the worker with an allowlist, not the
  planner's environment (``launcher.EnvPolicy``).
- **Time.** ``timeout_s`` ends the launcher's whole session - the launcher,
  the agent and everything the agent started - not just the direct child.
- **Concurrency.** More than one worker never shares a tree: each gets a
  disposable copy of ``cwd``, and the result carries the copy's path and a
  diff against the state the copies started from. Nothing is applied to
  ``cwd`` and nothing a worker wrote is executed by this server; the planner
  reviews the diff and applies what it accepts. A symlink that would lead
  out of a copy is left out of it (``_copy``), so editing a file in the copy
  cannot write through a link into the original tree; a link a worker adds
  is reported in the diff, and ``links_leaving_copy`` names the ones that lead
  out of its copy. The diff reads at most ``DIFF_FILE_LIMIT_BYTES`` of any
  changed file and stops reading once the patch is full; the rest are listed
  in ``not_diffed``. A single worker runs in
  ``cwd`` itself. Tool calls are served one at a time.
- **Copies left behind** by a server that was killed, or kept because a
  worker could not be stopped, carry an owner record. A later server, when it
  starts, stops the workers such a server left running (nobody can collect
  their results any more), then deletes the copies once no process holds
  anything inside them or still carries the copy's ``COPY_ENV`` variable
  (:func:`sweep_copies`), and names what it stops and keeps.
"""

from __future__ import annotations

import difflib
import fcntl
import filecmp
import json
import os
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, wait
from pathlib import Path
from typing import Any, Callable

from . import procs

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "2025-06-18"
MAX_OUTPUT = 12000
MAX_PATCH = 60000
#: A changed file larger than this is listed in a worker's changes but never
#: read into memory for the patch; nor is any file once the patch is full.
DIFF_FILE_LIMIT_BYTES = 1024 * 1024
MAX_PARALLEL = 8
MAX_TASKS = 32
MAX_TASK_CHARS = 100_000
MAX_CONTEXT_CHARS = 200_000
#: A per-worker copy larger than this is refused rather than made eight times.
COPY_LIMIT_BYTES = int(os.environ.get("AUTO_ROUTER_DELEGATE_COPY_LIMIT_MB", "200")) * 1024 * 1024
COPY_LIMIT_FILES = 50_000
#: How long an interrupted run_many keeps stopping its workers before it gives
#: up and leaves their copies in place rather than delete them under a writer.
STOP_WAIT_S = 30.0
#: Every run's scratch directory is ``$TMPDIR/COPY_PREFIX*``.
COPY_PREFIX = "auto-router-delegate-"
#: Written into a scratch directory when it is made and removed when a result
#: hands its copies over: who made it (pid, start time, boot, pid namespace).
#: The server holds an exclusive ``flock`` on it for as long as it runs.
OWNER_RECORD = ".auto-router-owner.json"
#: Set to the scratch directory for every worker that runs in one of its
#: copies. Whatever the worker starts inherits it, even a process that detached
#: (``setsid()``), left the copy and holds nothing inside it, so the sweep can
#: still tell that such a process may write into the copy by path.
COPY_ENV = "AUTO_ROUTER_DELEGATE_COPY"
#: Left out of per-worker copies: large, machine-specific or version control.
COPY_IGNORE = (".git", ".hg", ".svn", ".venv", "venv", "node_modules", "__pycache__",
               ".mypy_cache", ".pytest_cache", ".ruff_cache", ".tox")

_COMMON_PROPERTIES = {
    "context": {"type": "string", "maxLength": MAX_CONTEXT_CHARS,
                "description": "only the task-local context the worker needs"},
    "cwd": {"type": "string",
            "description": "working directory inside the delegation root (default: the root)"},
    "tier": {"type": "string", "enum": ["cheap", "auto", "strong"], "default": "cheap",
             "description": "applied among the routes the policy allows: cheap prefers a low-cost "
                            "worker unless the router judges the task hard; auto keeps the policy's "
                            "choice; strong prefers the most capable non-plan worker"},
    "timeout_s": {"type": "integer", "minimum": 1, "maximum": 7200, "default": 900},
}

DELEGATE_TOOL = {
    "name": "delegate",
    "description": (
        "Run a self-contained sub-task on routed worker agents. Use cheap for mechanical work; "
        "the router may choose a stronger worker when the task is hard. parallel>1 runs "
        "independent attempts, each in its own disposable copy of cwd, and returns each copy's "
        "diff; nothing is applied to cwd. Give only task-local context and verify worker output "
        "before using it."),
    "inputSchema": {
        "type": "object",
        "properties": {
            "task": {"type": "string", "minLength": 1, "maxLength": MAX_TASK_CHARS,
                     "description": "complete, self-contained worker brief"},
            **_COMMON_PROPERTIES,
            "parallel": {"type": "integer", "minimum": 1, "maximum": MAX_PARALLEL, "default": 1},
        },
        "required": ["task"],
        "additionalProperties": False,
    },
}

DELEGATE_MANY_TOOL = {
    "name": "delegate_many",
    "description": (
        "Run different independent worker briefs concurrently. With more than one brief, each "
        "worker gets its own disposable copy of cwd and the result carries its diff; nothing is "
        "applied to cwd. Keep dependencies with the planner; use this only when every task can "
        "finish without another worker's result."),
    "inputSchema": {
        "type": "object",
        "properties": {
            "tasks": {"type": "array", "minItems": 1, "maxItems": MAX_TASKS,
                      "items": {"type": "string", "minLength": 1, "maxLength": MAX_TASK_CHARS},
                      "description": "self-contained worker briefs"},
            **_COMMON_PROPERTIES,
            "parallel": {"type": "integer", "minimum": 1, "maximum": MAX_PARALLEL, "default": 4},
        },
        "required": ["tasks"],
        "additionalProperties": False,
    },
}

TOOLS = [DELEGATE_TOOL, DELEGATE_MANY_TOOL]


class ArgumentError(ValueError):
    """A tool call whose arguments do not match the tool's schema."""


# ---------------------------------------------------------------------------
# argument validation
# ---------------------------------------------------------------------------
def _check(name: str, value: Any, spec: dict) -> Any:
    kind = spec.get("type")
    if kind == "string":
        if not isinstance(value, str):
            raise ArgumentError(f"{name} must be a string")
        if len(value.strip()) < spec.get("minLength", 0):
            raise ArgumentError(f"{name} must not be empty")
        if "maxLength" in spec and len(value) > spec["maxLength"]:
            raise ArgumentError(f"{name} is longer than {spec['maxLength']} characters")
        if "enum" in spec and value not in spec["enum"]:
            raise ArgumentError(f"{name} must be one of {', '.join(spec['enum'])}")
    elif kind == "integer":
        if isinstance(value, bool) or not isinstance(value, int):
            raise ArgumentError(f"{name} must be an integer")
        if not spec.get("minimum", value) <= value <= spec.get("maximum", value):
            raise ArgumentError(f"{name} must be between {spec['minimum']} and {spec['maximum']}")
    elif kind == "array":
        if not isinstance(value, list):
            raise ArgumentError(f"{name} must be a list")
        if not spec.get("minItems", 0) <= len(value) <= spec.get("maxItems", len(value)):
            raise ArgumentError(f"{name} must have {spec.get('minItems', 0)} to "
                                f"{spec.get('maxItems')} items")
        for i, item in enumerate(value):
            _check(f"{name}[{i}]", item, spec.get("items") or {})
    return value


def validate(tool: dict, arguments: Any) -> dict[str, Any]:
    """Arguments checked against ``tool``'s input schema, defaults filled in."""
    schema = tool["inputSchema"]
    if not isinstance(arguments, dict):
        raise ArgumentError("arguments must be an object")
    unknown = sorted(set(arguments) - set(schema["properties"]))
    if unknown:
        raise ArgumentError(f"unknown argument(s): {', '.join(unknown)}")
    missing = [key for key in schema["required"] if arguments.get(key) is None]
    if missing:
        raise ArgumentError(f"missing required argument(s): {', '.join(missing)}")
    out: dict[str, Any] = {}
    for key, spec in schema["properties"].items():
        value = arguments.get(key)
        out[key] = spec.get("default") if value is None else _check(key, value, spec)
    return out


# ---------------------------------------------------------------------------
# boundaries: working directory and environment
# ---------------------------------------------------------------------------
_START_DIR = os.getcwd()


def delegation_root() -> Path:
    """The directory every ``cwd`` must lie in; refused when it is too broad."""
    raw = os.environ.get("AUTO_ROUTER_DELEGATE_ROOT") or _START_DIR
    root = Path(os.path.expanduser(raw)).resolve()
    home = Path.home().resolve()
    if root == Path(root.anchor) or root == home or root in home.parents:
        raise ArgumentError(
            f"delegation root {str(root)!r} is too broad; start the server in a project directory "
            f"or set AUTO_ROUTER_DELEGATE_ROOT to one")
    return root


def resolve_cwd(cwd: str | None) -> str:
    root = delegation_root()
    target = Path(cwd).expanduser().resolve() if cwd else root
    if target != root and root not in target.parents:
        raise ArgumentError(f"cwd {str(target)!r} is outside the delegation root {str(root)!r}")
    if not target.is_dir():
        raise ArgumentError(f"no such directory: {str(target)!r}")
    return str(target)


def launcher_env(environ: dict[str, str] | None = None) -> dict[str, str]:
    """The launcher process's environment.

    The launcher is this package's own code: it needs its configuration, the
    quota readers and the classifier key the configuration names. The worker
    it starts receives only the launcher's allowlist (``EnvPolicy``); nothing
    here is passed on to the worker unless the configuration names it.
    """
    env = dict(os.environ if environ is None else environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(ROOT), env.get("PYTHONPATH", "")) if p)
    return env


# ---------------------------------------------------------------------------
# running workers
# ---------------------------------------------------------------------------
def _brief(task: str, context: str | None) -> str:
    task = task.strip()
    if not context or not context.strip():
        return task
    return f"{task}\n\nTask-local context:\n{context.strip()}"


def launcher_argv(task: str, cwd: str | None, tier: str = "cheap") -> list[str]:
    argv = [sys.executable, "-m", "auto_router.launcher", "--no-plans", "--json",
            "--tier", tier]
    if cwd:
        argv += ["--cwd", cwd]
    # ``--`` ends the options: a brief such as "--list" is a task, not a flag.
    return argv + ["--", task]


def _decision(stderr: str) -> dict[str, Any]:
    for line in reversed((stderr or "").splitlines()):
        try:
            value = json.loads(line)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(value, dict) and "route" in value:
            return value
    return {}


def _run_launcher(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess:
    """Run the launcher in a session of its own, so a timeout ends all of it."""
    return procs.run(argv, scope="session", **kwargs)


def run_delegate(task: str, context: str | None = None, cwd: str | None = None,
                 tier: str = "cheap", timeout_s: int = 900,
                 run: Callable[..., Any] = _run_launcher) -> dict[str, Any]:
    """Run one worker and return a truthful, machine-readable result."""
    started = time.perf_counter()
    if not isinstance(task, str) or not task.strip():
        return {"ok": False, "error": "empty task"}
    if cwd and not Path(cwd).is_dir():
        return {"ok": False, "error": f"no such directory: {cwd}"}
    if tier not in {"cheap", "auto", "strong"}:
        return {"ok": False, "error": f"unknown tier: {tier}"}
    if isinstance(timeout_s, bool) or not isinstance(timeout_s, int):
        return {"ok": False, "error": "timeout_s must be an integer"}
    timeout_s = max(1, min(timeout_s, 7200))
    full_brief = _brief(task, context)
    env = launcher_env()
    if cwd and Path(cwd).parent.name.startswith(COPY_PREFIX):
        env[COPY_ENV] = str(Path(cwd).parent)
    try:
        proc = run(launcher_argv(full_brief, cwd, tier), cwd=cwd or None, env=env,
                   capture_output=True, text=True, timeout=timeout_s, stdin=subprocess.DEVNULL)
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"worker timed out after {timeout_s}s; the worker and "
                                      f"every process it started were stopped",
                "tier": tier, "wall_time_s": round(time.perf_counter() - started, 3),
                "brief_chars": len(full_brief), "brief_tokens_estimate": (len(full_brief) + 3) // 4}
    decision = _decision(proc.stderr or "")
    output = (proc.stdout or "").strip()
    truncated = len(output) > MAX_OUTPUT
    if truncated:
        output = "[... earlier output cut ...]\n" + output[-MAX_OUTPUT:]
    estimate = ((decision.get("decision") or {}).get("estimated_outcome") or {}).get("cost_usd")
    result = {
        "ok": proc.returncode == 0,
        "model": decision.get("route"),
        "tier": tier,
        "result": output or "(no output)",
        "exit_code": proc.returncode,
        "wall_time_s": round(time.perf_counter() - started, 3),
        "cost_usd": None,
        "cost_basis": "not measured: the launched agent CLI does not report token usage",
        "estimated_cost_usd": estimate,
        "estimated_cost_basis": "router estimate before the worker ran" if estimate is not None else None,
        "brief_chars": len(full_brief),
        "brief_tokens_estimate": (len(full_brief) + 3) // 4,
        "output_truncated": truncated,
    }
    if proc.returncode != 0 and not decision:
        # The launcher refused before any worker started (bad config, a
        # --cwd outside launcher.cwd_root, no route): say why.
        result["error"] = ((proc.stderr or "").strip().splitlines() or ["launcher failed"])[-1][:500]
    return result


def _tree(root: Path) -> dict[str, Path]:
    """Every entry of ``root`` that is not a directory, links to directories included.

    ``os.walk`` lists a symlink to a directory among the directories and does
    not enter it; left there it would vanish from the diff, so a worker could
    add ``home -> ~`` to its copy unreported. It is listed as an entry of its
    own and, like every link, never followed.
    """
    out: dict[str, Path] = {}
    for dirpath, dirnames, filenames in os.walk(root):
        kept = [d for d in dirnames if d not in COPY_IGNORE]
        dirnames[:] = [d for d in kept if not os.path.islink(os.path.join(dirpath, d))]
        for name in filenames + [d for d in kept if d not in dirnames]:
            path = Path(dirpath) / name
            out[str(path.relative_to(root))] = path
    return out


def _copy_size(source: Path) -> tuple[int, int]:
    files = size = 0
    for path in _tree(source).values():
        files += 1
        try:
            size += path.lstat().st_size
        except OSError:
            pass
        if files > COPY_LIMIT_FILES or size > COPY_LIMIT_BYTES:
            break
    return files, size


def _inside(root: str, path: str) -> bool:
    return path == root or path.startswith(root.rstrip(os.sep) + os.sep)


def _link_escapes(root: str, link: str) -> bool:
    """Whether the symlink ``link`` in the tree ``root`` can lead out of it.

    Kept only if its target is relative, stays inside ``root`` as written
    (``a/../b``), and still stays inside once every link on the way is
    followed. An absolute target is refused even when it names a path inside
    the tree: in a copy it would point back at the original.
    """
    target = os.readlink(link)
    if os.path.isabs(target):
        return True
    written = os.path.normpath(os.path.join(os.path.dirname(link), target))
    return not _inside(root, written) or not _inside(root, os.path.realpath(link))


def _copy_entries(src_fd: int, dest: Path, rel: str, skipped: list[str]) -> None:
    """Copy the directory open at ``src_fd`` into the new directory ``dest``.

    Every entry is opened relative to its parent's descriptor with
    ``O_NOFOLLOW``, so an entry swapped for a symlink while the copy runs is
    not followed. Symlinks are recreated as they are (they are vetted on the
    finished copy); FIFOs, sockets and devices are skipped.
    """
    with os.scandir(src_fd) as entries:
        names = sorted((e.name, e) for e in entries)
    for name, entry in names:
        where = f"{rel}{name}"
        if name in COPY_IGNORE:
            continue
        if entry.is_symlink():
            os.symlink(os.readlink(name, dir_fd=src_fd), dest / name)
        elif entry.is_dir(follow_symlinks=False):
            try:
                fd = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=src_fd)
            except OSError:
                skipped.append(f"{where}/ (changed or unreadable during the copy)")
                continue
            try:
                (dest / name).mkdir(mode=0o700)
                _copy_entries(fd, dest / name, f"{where}/", skipped)
                (dest / name).chmod(stat.S_IMODE(os.fstat(fd).st_mode) | 0o700)
            finally:
                os.close(fd)
        elif entry.is_file(follow_symlinks=False):
            try:
                fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=src_fd)
            except OSError:
                skipped.append(f"{where} (changed or unreadable during the copy)")
                continue
            try:
                info = os.fstat(fd)
                if not stat.S_ISREG(info.st_mode):
                    skipped.append(f"{where} (not a regular file)")
                    continue
                with open(dest / name, "xb") as out, os.fdopen(os.dup(fd), "rb") as src:
                    shutil.copyfileobj(src, out)
                    os.fchmod(out.fileno(), stat.S_IMODE(info.st_mode) & 0o777 | 0o600)
                os.utime(dest / name, ns=(info.st_atime_ns, info.st_mtime_ns))
            finally:
                os.close(fd)
        else:
            skipped.append(f"{where} (not a regular file)")


def _copy(source: Path, target: Path) -> list[str]:
    """Make ``target`` a private, self-contained copy of ``source``.

    A symlink that could lead out of the copy - an absolute target, a ``..``
    that climbs above the copy, or a chain that ends outside it - is left out,
    so a worker editing ``shared/config`` in its copy cannot write through it
    into the original tree or anywhere else. The check runs on the finished
    copy, inside a fresh ``0700`` directory no one else writes to, before any
    worker starts; a source changing during the copy cannot slip a link past
    it. Links that stay inside the copy are kept. Copied files and directories
    are made owner-writable, since the copy exists to be edited. Returns the
    left-out entries, relative to ``source``, with the reason.
    """
    skipped: list[str] = []
    fd = os.open(source, os.O_RDONLY | os.O_DIRECTORY)
    try:
        target.mkdir(mode=0o700)
        _copy_entries(fd, target, "", skipped)
    finally:
        os.close(fd)
    root = os.path.realpath(target)
    escaping = []
    for dirpath, dirnames, filenames in os.walk(root):
        links = [n for n in dirnames + filenames if os.path.islink(os.path.join(dirpath, n))]
        dirnames[:] = [n for n in dirnames if n not in links]
        escaping += [p for p in (os.path.join(dirpath, n) for n in links) if _link_escapes(root, p)]
    for path in escaping:  # all judged first, so the verdict on a chain does not depend on order
        skipped.append(f"{os.path.relpath(path, root)} (symlink to "
                       f"{os.readlink(path)!r} leads outside the copy)")
        os.unlink(path)
    return sorted(skipped)


def _same_entry(a: Path, b: Path) -> bool:
    """Links compare by target, not by the file they reach (it is diffed itself)."""
    if a.is_symlink() or b.is_symlink():
        return a.is_symlink() and b.is_symlink() and os.readlink(a) == os.readlink(b)
    return filecmp.cmp(a, b, shallow=False)


def diff_trees(before: Path, after: Path, limit: int = MAX_PATCH) -> dict[str, Any]:
    """What a worker changed in its copy: file lists and a unified diff."""
    old, new = _tree(before), _tree(after)
    added = sorted(set(new) - set(old))
    deleted = sorted(set(old) - set(new))
    modified = sorted(rel for rel in set(old) & set(new) if not _same_entry(old[rel], new[rel]))
    chunks: list[str] = []
    binary: list[str] = []
    not_diffed: list[str] = []
    size = 0
    for rel in sorted({*added, *deleted, *modified}):
        if size > limit:  # the patch is full: list the rest, do not read them
            not_diffed.append(rel)
            continue
        texts = [_diff_text(tree.get(rel)) for tree in (old, new)]
        if "too large" in texts:
            not_diffed.append(rel)
            continue
        if "binary" in texts:
            binary.append(rel)
            continue
        for chunk in difflib.unified_diff(texts[0], texts[1], f"a/{rel}", f"b/{rel}"):
            chunks.append(chunk)
            size += len(chunk)
    patch = "".join(chunks)
    # A link the worker added or retargeted that leads out of its copy: the
    # planner applying this copy would import a path into someone else's tree.
    root = os.path.realpath(after)
    leaving = [rel for rel in sorted({*added, *modified})
               if new[rel].is_symlink() and _link_escapes(root, os.path.join(root, rel))]
    return {"added": added, "modified": modified, "deleted": deleted, "binary_changed": binary,
            "links_leaving_copy": leaving, "not_diffed": not_diffed, "patch": patch[:limit],
            "patch_truncated": len(patch) > limit}


def _diff_text(path: Path | None) -> list[str] | str:
    """The lines of ``path`` for the patch, or why it has none: "binary" or "too large".

    At most ``DIFF_FILE_LIMIT_BYTES`` + 1 bytes are read, so a worker that
    writes a huge file into its copy cannot make the server hold it in memory.
    """
    if path is None:
        return []
    if path.is_symlink():
        return [f"-> {os.readlink(path)}\n"]
    if not stat.S_ISREG(path.lstat().st_mode):  # a FIFO would block the read
        return "binary"
    try:
        with open(path, "rb") as src:
            data = src.read(DIFF_FILE_LIMIT_BYTES + 1)
    except OSError:
        return "binary"
    if len(data) > DIFF_FILE_LIMIT_BYTES:
        return "too large"
    try:  # newlines as read_text() gives them
        text = data.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")
    except UnicodeDecodeError:
        return "binary"
    return text.splitlines(keepends=True)


def _boot_id() -> str | None:
    try:
        return Path("/proc/sys/kernel/random/boot_id").read_text().strip() or None
    except OSError:
        return None


def _pid_ns() -> str | None:
    try:
        return os.readlink("/proc/self/ns/pid")
    except OSError:
        return None


def _tag(scratch: Path) -> int:
    """Record this server as the owner of ``scratch``; returns the locked fd."""
    fd = os.open(scratch / OWNER_RECORD, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        os.write(fd, json.dumps({"path": str(scratch), "pid": os.getpid(),
                                 "start": procs.start_time(os.getpid()), "boot_id": _boot_id(),
                                 "pid_ns": _pid_ns()}).encode())
    except BaseException:
        os.close(fd)
        raise
    return fd


def _remove_copies(scratch: Path) -> None:
    """Delete ``scratch``, its owner record last, so an interrupted deletion
    leaves a directory a later server still recognises and finishes."""
    try:
        entries = list(scratch.iterdir())
    except OSError:
        entries = []
    for entry in entries:
        if entry.name == OWNER_RECORD:
            continue
        if entry.is_dir() and not entry.is_symlink():
            shutil.rmtree(entry, ignore_errors=True)
        else:
            entry.unlink(missing_ok=True)
    (scratch / OWNER_RECORD).unlink(missing_ok=True)
    shutil.rmtree(scratch, ignore_errors=True)


def _marker(scratch: Path) -> bytes:
    """The environment entry every worker in ``scratch`` is started with."""
    return f"{COPY_ENV}={scratch}".encode()


def _orphan(scratch: Path) -> tuple[bool, str, int | None, set[int]]:
    """(reclaimable, why not, the locked record fd, pids stopped) for one
    scratch directory.

    Reclaimable only if all of: a directory (not a link) this user owns, with
    an owner record naming this very path, made in this boot and pid
    namespace; its owner is gone (no process with the recorded pid and start
    time, and the record's lock is free); and no live process holds anything
    inside it (working or root directory, executable, open or mapped file)
    and none started after its owner carries ``COPY_ENV`` naming it, while
    every process that cannot be inspected started before its owner did.

    Once its owner is gone and the record's lock is held, the processes that
    started after the owner and carry ``COPY_ENV`` naming this very path are
    stopped first (:func:`procs.stop_marked`): they are that server's workers
    or what they started, and no reply will ever carry their results. A server
    that is itself part of such a job (:func:`_part_of`) stops none.
    """
    try:
        st = os.lstat(scratch)
    except OSError as exc:
        return False, f"cannot inspect it ({exc.strerror})", None, set()
    if not stat.S_ISDIR(st.st_mode) or st.st_uid != os.getuid():
        return False, "not a directory of this user", None, set()
    try:
        fd = os.open(scratch / OWNER_RECORD, os.O_RDONLY | os.O_NOFOLLOW)
    except FileNotFoundError:
        return False, "no owner record (copies a result handed over, or not ours)", None, set()
    except OSError as exc:
        return False, f"owner record unreadable ({exc.strerror})", None, set()
    try:
        record = json.loads(os.read(fd, 65536) or b"null")
        pid, start = record["pid"], record["start"]
        if (record["path"] != str(scratch) or isinstance(pid, bool) or not isinstance(pid, int)
                or isinstance(start, bool) or not isinstance(start, int)):
            raise ValueError
    except (OSError, ValueError, TypeError, KeyError):
        os.close(fd)
        return False, "owner record not understood", None, set()
    if record.get("boot_id") != _boot_id() or _boot_id() is None:
        why = "made before the last reboot or on another machine"
    elif record.get("pid_ns") != _pid_ns() or _pid_ns() is None:
        why = "made in another pid namespace (container or sandbox)"
    elif procs.start_time(pid) == start:
        why = f"its server (pid {pid}) is still running"
    else:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            why = "its owner record is locked by a running process"
        else:
            why = ""
    stopped: set[int] = set()
    try:
        if not why and not _part_of(scratch):
            stopped = procs.stop_marked(_marker(scratch), since=start)
        if not why:
            why = _in_use(scratch, since=start)
    except BaseException:
        os.close(fd)
        raise
    if why:
        os.close(fd)
        return False, why, None, stopped
    return True, "", fd, stopped


def _part_of(scratch: Path) -> bool:
    """Whether this server runs inside one of ``scratch``'s jobs: it carries
    the copy's ``COPY_ENV`` entry, or descends from a process started with it.
    Such a server stops none of that job's processes (they include its own
    ancestors)."""
    return os.environ.get(COPY_ENV) == str(scratch) or procs.lineage_carries(_marker(scratch))


def _in_use(scratch: Path, *, since: int) -> str:
    """Why a process may still write into ``scratch`` ("" if none can)."""
    inodes = set()
    try:
        for top, dirs, files in os.walk(scratch, onerror=_raise):
            for name in (top, *(os.path.join(top, n) for n in dirs + files)):
                entry = os.lstat(name)
                inodes.add((entry.st_dev, entry.st_ino))
    except OSError as exc:
        return f"cannot list it ({exc.strerror})"
    seen = procs.holders(inodes, since=since, environ=_marker(scratch))
    if seen is None:
        return "cannot tell which processes use it (/proc unreadable)"
    if seen[0]:
        return "in use by pid " + ", ".join(map(str, sorted(seen[0])))
    if seen[1]:
        return ("processes that cannot be inspected started after its server: pid "
                + ", ".join(map(str, sorted(seen[1]))))
    return ""


def _raise(exc: OSError) -> None:
    raise exc


def sweep_copies(tmpdir: str | None = None) -> list[str]:
    """Stop the workers a dead server left running, and delete the scratch
    directories it left in ``tmpdir`` (default ``$TMPDIR``) that no process can
    still write to (:func:`_orphan`); keep every other ``COPY_PREFIX*`` entry.
    Returns a line per entry, and one before it for the workers it stopped.
    """
    lines = []
    base = Path(tmpdir or tempfile.gettempdir())
    try:
        entries = sorted(base.glob(COPY_PREFIX + "*"))
    except OSError:
        return lines
    for scratch in entries:
        ok, why, fd, stopped = _orphan(scratch)
        if stopped:
            lines.append(f"stopped {scratch}: pid {', '.join(map(str, sorted(stopped)))} still "
                         f"worked for its server, which had ended")
        if not ok:
            lines.append(f"kept {scratch}: {why}")
            continue
        try:
            _remove_copies(scratch)
        finally:
            os.close(fd)
        lines.append(f"removed {scratch}: its server and every worker had ended")
    return lines


#: Set by the server's stop-signal handler before it stops the running workers,
#: so that a pool thread freed by that stop does not start a queued brief.
#: Cleared when :func:`main` starts, so it belongs to one server run.
_STOPPING = threading.Event()


def _unless_stopping(runner: Callable[..., dict[str, Any]], *args: Any) -> dict[str, Any]:
    if _STOPPING.is_set():
        return {"ok": False, "error": "not started: the server was told to stop"}
    return runner(*args)


def _stop_workers(futures: dict) -> bool:
    """Stop every worker still running; True once none is, False after STOP_WAIT_S."""
    deadline = time.monotonic() + STOP_WAIT_S
    # A queued brief cancelled by the shutdown never runs, but wait() only
    # counts it as done once a pool thread has seen it, which may be never.
    live = [future for future in futures if not future.cancelled()]
    while True:
        procs.terminate_all()
        if not wait(live, timeout=0.2).not_done:
            return True
        if time.monotonic() >= deadline:
            return False


def run_many(tasks: list[str], *, context: str | None = None, cwd: str | None = None,
             tier: str = "cheap", parallel: int = 4, timeout_s: int = 900,
             runner: Callable[..., dict[str, Any]] = run_delegate) -> dict[str, Any]:
    """Run several workers; with more than one, each works in its own copy of ``cwd``."""
    if (not isinstance(tasks, list) or not tasks or len(tasks) > MAX_TASKS
            or any(not isinstance(task, str) or not task.strip() for task in tasks)):
        return {"ok": False, "error": f"tasks must be a list of 1 to {MAX_TASKS} non-empty strings"}
    if isinstance(parallel, bool) or not isinstance(parallel, int):
        return {"ok": False, "error": "parallel must be an integer"}
    workers = max(1, min(parallel, MAX_PARALLEL, len(tasks)))
    started = time.perf_counter()
    source = Path(cwd or os.getcwd()).resolve()
    isolated = len(tasks) > 1
    scratch: Path | None = None
    skipped: list[str] = []
    workdirs: list[str | None] = [cwd] * len(tasks)
    if isolated:
        files, size = _copy_size(source)
        if files > COPY_LIMIT_FILES or size > COPY_LIMIT_BYTES:
            return {"ok": False, "error": (
                f"{source} is too large to copy once per worker (limit {COPY_LIMIT_FILES} files, "
                f"{COPY_LIMIT_BYTES // 2**20} MB, without {', '.join(COPY_IGNORE[:6])}...); "
                f"use a smaller cwd or run one worker at a time")}
    results: list[dict[str, Any] | None] = [None] * len(tasks)
    pool: ThreadPoolExecutor | None = None
    pending: dict = {}
    owner: int | None = None
    try:
        if isolated:
            scratch = Path(tempfile.mkdtemp(prefix=COPY_PREFIX))
            try:
                owner = _tag(scratch)
                skipped = _copy(source, scratch / "base")
                for i in range(len(tasks)):
                    _copy(scratch / "base", scratch / f"worker-{i + 1}")
                    workdirs[i] = str(scratch / f"worker-{i + 1}")
            except OSError as exc:
                _remove_copies(scratch)
                return {"ok": False, "error": f"could not copy {source} for the workers: {exc}"}
        pool = ThreadPoolExecutor(max_workers=workers)
        for i, task in enumerate(tasks):
            pending[pool.submit(_unless_stopping, runner, task, context, workdirs[i], tier,
                                timeout_s)] = i
        for future in as_completed(pending):
            i = pending[future]
            try:
                results[i] = future.result()
            except Exception as exc:  # a broken worker must not lose its siblings
                results[i] = {"ok": False, "error": f"worker failed: {type(exc).__name__}: {exc}"}
        pool.shutdown()
        finished = [r or {"ok": False, "error": "worker produced no result"} for r in results]
        if isolated and scratch is not None:
            for i, item in enumerate(finished):
                workdir = Path(workdirs[i] or "")
                try:
                    changes = diff_trees(scratch / "base", workdir)
                except OSError as exc:
                    changes = {"error": f"could not diff the worker copy: {exc}"}
                changed = any(changes.get(k) for k in ("added", "modified", "deleted"))
                if not changed and not changes.get("error"):
                    shutil.rmtree(workdir, ignore_errors=True)
                finished[i] = {**item, "workspace": str(workdir) if changed else None,
                               "changes": changes}
            shutil.rmtree(scratch / "base", ignore_errors=True)
            if not any(r.get("workspace") for r in finished):
                _remove_copies(scratch)
            else:
                # Handed over: the result names these copies, so they are the
                # planner's now and no later server may reclaim them.
                (scratch / OWNER_RECORD).unlink(missing_ok=True)
    except BaseException:
        # Interrupted: the server's stop signal (raised here as SystemExit),
        # Ctrl-C or a crash. No result will ever name these copies, so they go
        # too - but only once no worker can still be writing into them. Queued
        # briefs are not started; the exception goes on unchanged. Copies
        # kept for a worker that could not be stopped keep their owner record,
        # so a later server reclaims them once nothing uses them (sweep_copies).
        stopped = True
        if pool is not None:
            pool.shutdown(wait=False, cancel_futures=True)
            stopped = _stop_workers(pending)
        if scratch is not None and stopped:
            _remove_copies(scratch)
        raise
    finally:
        if owner is not None:
            os.close(owner)
    known_costs = [r.get("cost_usd") for r in finished if r.get("cost_usd") is not None]
    estimates = [r.get("estimated_cost_usd") for r in finished
                 if r.get("estimated_cost_usd") is not None]
    out = {
        "ok": all(r.get("ok") for r in finished),
        "results": finished,
        "wall_time_s": round(time.perf_counter() - started, 3),
        "parallel": workers,
        "isolation": "per-worker copy" if isolated else "in place (one worker)",
        "cost_usd": round(sum(known_costs), 8) if len(known_costs) == len(finished) else None,
        "estimated_cost_usd": round(sum(estimates), 8) if len(estimates) == len(finished) else None,
        "brief_tokens_estimate": sum(int(r.get("brief_tokens_estimate") or 0) for r in finished),
    }
    if skipped:
        out["copy_skipped"] = skipped
    if isolated:
        out["note"] = ("Each worker ran in its own copy of cwd; nothing was applied to cwd. Review "
                       "each result's changes and apply what you accept.")
    return out


# ---------------------------------------------------------------------------
# MCP
# ---------------------------------------------------------------------------
def _text_result(data: dict[str, Any]) -> tuple[str, bool, dict[str, Any]]:
    return json.dumps(data, ensure_ascii=False, indent=2), not bool(data.get("ok")), data


def _call(name: str, arguments: Any) -> tuple[str, bool, dict[str, Any]]:
    tool = next((t for t in TOOLS if t["name"] == name), None)
    if tool is None:
        return _text_result({"ok": False, "error": f"unknown tool {name!r}"})
    try:
        args = validate(tool, arguments)
        cwd = resolve_cwd(args["cwd"])
    except ArgumentError as exc:
        return _text_result({"ok": False, "error": f"invalid arguments: {exc}"})
    common = {"context": args["context"], "cwd": cwd, "tier": args["tier"],
              "timeout_s": args["timeout_s"]}
    if name == "delegate":
        if args["parallel"] == 1:
            return _text_result(run_delegate(args["task"], **common))
        return _text_result(run_many([args["task"]] * args["parallel"],
                                     parallel=args["parallel"], **common))
    return _text_result(run_many(args["tasks"], parallel=args["parallel"], **common))


def handle(message: Any, run_tool: Callable[[str, Any], tuple[str, bool, dict]]) -> dict | None:
    if not isinstance(message, dict):
        return {"jsonrpc": "2.0", "id": None,
                "error": {"code": -32600, "message": "invalid request: not an object"}}
    method, mid = message.get("method"), message.get("id")
    if mid is None:
        return None
    if method == "initialize":
        params = message.get("params") if isinstance(message.get("params"), dict) else {}
        version = params.get("protocolVersion") or PROTOCOL
        result = {"protocolVersion": version, "capabilities": {"tools": {}},
                  "serverInfo": {"name": "auto-router-delegate", "version": "0.4.0"}}
    elif method == "ping":
        result = {}
    elif method == "tools/list":
        result = {"tools": TOOLS}
    elif method == "tools/call":
        params = message.get("params")
        if not isinstance(params, dict) or params.get("name") not in {t["name"] for t in TOOLS}:
            name = params.get("name") if isinstance(params, dict) else None
            return {"jsonrpc": "2.0", "id": mid,
                    "error": {"code": -32602, "message": f"unknown tool {name!r}"}}
        arguments = params.get("arguments")
        try:
            text, is_error, structured = run_tool(params["name"], {} if arguments is None else arguments)
        except Exception as exc:  # noqa: BLE001 - one bad call must not end the server
            return {"jsonrpc": "2.0", "id": mid,
                    "error": {"code": -32603, "message": f"internal error: {type(exc).__name__}"}}
        result = {"content": [{"type": "text", "text": text}], "isError": is_error,
                  "structuredContent": structured}
    else:
        return {"jsonrpc": "2.0", "id": mid,
                "error": {"code": -32601, "message": f"no method {method}"}}
    return {"jsonrpc": "2.0", "id": mid, "result": result}


def _exit_on_signal(signum, _frame):
    # Running workers (possibly in pool threads) are stopped before the server
    # goes, instead of carrying on unsupervised; queued briefs are not started.
    # A second signal changes nothing: raising again would abort the cleanup
    # the first one began and leave the worker copies behind.
    if _STOPPING.is_set():
        return
    _STOPPING.set()
    if signum == signal.SIGINT:
        # Ctrl-C stays a KeyboardInterrupt; the code it interrupts stops its
        # workers (procs.run, run_many) and removes their copies.
        raise KeyboardInterrupt
    procs.terminate_all()
    raise SystemExit(128 + signum)


def main(stdin=sys.stdin, stdout=sys.stdout) -> int:
    previous = {}
    _STOPPING.clear()
    if threading.current_thread() is threading.main_thread():
        for sig in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
            # A Ctrl-C the server was started to ignore stays ignored.
            if sig != signal.SIGINT or signal.getsignal(sig) is not signal.SIG_IGN:
                previous[sig] = signal.signal(sig, _exit_on_signal)
    try:
        _report_sweep()
        return _serve(stdin, stdout)
    finally:
        procs.terminate_all()
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def _report_sweep() -> None:
    """Reclaim what dead servers left behind; say on stderr what was kept."""
    try:
        lines = sweep_copies()
    except Exception as exc:  # noqa: BLE001 - never a reason not to serve
        lines = [f"could not check for copies left behind: {type(exc).__name__}: {exc}"]
    for line in lines:
        print(f"auto-router-delegate: {line}", file=sys.stderr, flush=True)


def _serve(stdin, stdout) -> int:
    for line in stdin:
        line = line.strip()
        if not line:
            continue
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            reply = {"jsonrpc": "2.0", "id": None,
                     "error": {"code": -32700, "message": "parse error"}}
        else:
            reply = handle(message, _call)
        if reply is not None:
            stdout.write(json.dumps(reply) + "\n")
            stdout.flush()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
