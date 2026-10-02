"""Both installer entry points, run end to end against fakes in a disposable HOME.

``scripts/install-delegation.sh`` and ``scripts/install-delegation.py`` run as
real processes through their own argument parsing and control flow. Every
external command they call is a local fake that appends its arguments to a
log and then imitates just enough of the real tool:

- ``git`` "fetches" by copying ``scripts/``, ``skills/``, ``integrations/``
  and ``.gitignore`` from this checkout; it never contacts a remote, and the
  commit, ``FETCH_HEAD`` and ``HEAD`` are only files under ``.git``.
  It models readable regular files only: a source holding anything else, such
  as a symlink, makes the fetch fail before it copies anything, rather than
  leave that entry out of the commit. A source file or directory it cannot
  read also makes the fetch fail and leaves the checkout as it was, where real
  git would still fetch the committed file. The fetch copies into a staging
  directory under ``.git`` and gives the owner read, write and search on the
  copies there, as a copy of another user's file is ours but keeps its mode
  bits. Before it writes to the checkout it checks that every staged path fits:
  a file where the checkout has a directory, a directory where it has a file, a
  symlink in the way, an entry this user cannot write, or a directory it cannot
  list makes the fetch fail and leaves the checkout and ``FETCH_*`` as they
  were, where real git would replace an ignored entry in its way or write into
  a directory it can search but not list. Once those checks pass, the fetch
  copies into the checkout and then checks that every file it fetched is there
  with the staged bytes. A copy that fails, for a reason the checks do not
  cover such as a full disk, or that ends without a fetched file, can leave the
  checkout partly updated: the fetch then says so, rather than that it fetched
  nothing, and leaves ``FETCH_*`` as they were, but it does not roll the
  checkout back. ``FETCH_HEAD`` and ``FETCH_TREE`` are published as one pair:
  ``FETCH_HEAD`` is written first, in place, so a read-only one fails the fetch
  as it does in real git, and the tree is renamed into place last; a failure in
  either step puts the previous ``FETCH_HEAD`` back, says that publishing the
  pair failed (not that the copy did) and leaves the previous pair, though the
  checkout itself is already updated. Should the previous ``FETCH_HEAD`` not be
  restorable, a ``FETCH_PENDING`` marker stays and ``checkout`` refuses the pair
  until a fetch publishes a new one; no other failure while publishing is
  modelled. The fetch lists the files it copied, separated by NUL so that
  any file name survives, as the commit's tree, and ``checkout`` records a hash
  of each of them, looked up by name, plus of any file an earlier checkout
  tracked, as the index, and then moves ``HEAD``; a listed file that is missing
  or unreadable fails the checkout and leaves the index and ``HEAD`` as they
  were, so no fetched file drops out of the index. The index is written to a
  temporary file, flushed and closed with the result checked, and renamed over
  the old one, and ``HEAD`` moves only after that: a write that fails (a full
  disk, a file size limit) or is interrupted fails the checkout and leaves the
  index whole, as real git does. A fetched file is tracked
  even if ``.gitignore`` matches it, as a file added with ``git add -f`` is;
  any other file in the checkout, such as ignored ``build/`` and
  ``*.egg-info/`` output, stays untracked across repeated checkouts.
  ``status --porcelain`` compares against the index, as real git does for these
  cases: a changed tracked file is `` M``, a missing one `` D``, and a new file
  is ``??`` (a directory holding no tracked file as ``?? dir/``) unless the
  ``.gitignore`` ignores it. Only the pattern forms that file uses are
  understood (``name``, ``*.ext``, ``dir/``, ``a/b/``); there are no negations,
  nested ignore files, staging or real objects, and status prints paths
  unquoted where real git quotes unusual names.
- ``python3 -m venv`` makes a directory whose ``pip`` records the call, copies
  any existing ``build/lib`` of the source into the venv's ``site-packages``,
  as setuptools keeps stale files there in the wheel, leaves ``build/`` and
  ``*.egg-info/`` in the directory it installs from, as an in-place
  setuptools build does, and drops a stub ``auto-router-delegate``; its
  ``python`` is this interpreter, so the real Python installer runs.
- ``claude`` and ``codex`` keep their MCP entries as files under ``tmp_path``.
  ``claude`` parses ``--env`` as Claude Code 2.1.284 does: the flag takes every
  following argument up to the next option, so it must come after the name.
- ``cursor``, ``opencode``, ``openclaw``, ``hermes`` and ``copilot`` exist only
  to prove the installer never calls them.

HOME is a directory in ``tmp_path`` and PATH holds only the fakes and the
system directories. This proves the installers' control flow, not that they
work with the real git, pip, Claude Code or Codex CLIs: those are untested.
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SHELL = ROOT / "scripts" / "install-delegation.sh"
PYTHON_INSTALLER = ROOT / "scripts" / "install-delegation.py"
SHA = "0123456789abcdef0123456789abcdef01234567"
NAME = "auto-router-delegate"
URL = "https://github.com/fstandhartinger/auto-model-router.git"

FAKE_GIT = r"""#!/bin/sh
echo "git $*" >> "$FAKE_LOG"
repo=.
if [ "$1" = "-C" ]; then repo=$2; shift 2; fi
if [ "$1" = "-c" ]; then shift 2; fi
case "$1" in
  init) mkdir -p "$3/.git" ;;
  remote) ;;
  status) [ -z "${FAKE_GIT_DIRTY:-}" ] || echo " M scripts/local-edit"
          exec "$FAKE_REAL_PYTHON" "$FAKE_BIN/git-index.py" status "$repo" ;;
  fetch) other=$(cd "$FAKE_SOURCE" && find scripts skills integrations .gitignore ! -type f ! -type d) \
           || { echo "fake git: cannot read the source tree, fetched nothing" >&2; exit 1; }
         if [ -n "$other" ]; then
           printf '%s\n' "fake git: unsupported source entry, only regular files are modelled: $other" >&2
           exit 1
         fi
         # Copy into a staging directory first, so a source file that cannot be
         # read fails the fetch and leaves the checkout and FETCH_* as they were.
         # A copy of another user's file is ours but keeps its mode bits, so give
         # the owner read, write and search before listing or publishing it.
         stage="$repo/.git/fetch-stage"
         rm -rf "$stage" "$stage.list" "$stage.tree" && mkdir "$stage" || exit 1
         if cp -R "$FAKE_SOURCE/scripts" "$FAKE_SOURCE/skills" "$FAKE_SOURCE/integrations" \
                  "$FAKE_SOURCE/.gitignore" "$stage/" && chmod -R u+rwX "$stage" &&
            (cd "$stage" && find scripts skills integrations .gitignore -type f -print0) > "$stage.list" &&
            sort -z "$stage.list" > "$stage.tree"; then
           rm -f "$stage.list"
         else
           chmod -R u+rwX "$stage"; rm -rf "$stage" "$stage.list" "$stage.tree"
           echo "fake git: cannot copy the source tree, fetched nothing" >&2
           exit 1
         fi
         # Touch the checkout only once every staged path is known to fit in it.
         "$FAKE_REAL_PYTHON" "$FAKE_BIN/git-preflight.py" "$stage" "$repo" ||
           { rm -rf "$stage" "$stage.tree"; exit 1; }
         # A copy that ends without every listed file in the checkout is not a fetch.
         if ! { cp -R "$stage/." "$repo/" &&
                "$FAKE_REAL_PYTHON" "$FAKE_BIN/git-published.py" "$stage" "$repo"; }; then
           rm -rf "$stage" "$stage.tree"
           echo "fake git: copying into the checkout failed part way, it may be partly updated" >&2
           exit 1
         fi
         # FETCH_HEAD and FETCH_TREE are one fetch's pair. Keep the previous
         # FETCH_HEAD aside and mark the pair pending, write FETCH_HEAD in place
         # (it fails if the file is read-only, as real git does) and only then
         # rename the new tree over FETCH_TREE. If either step fails, put the
         # previous FETCH_HEAD back; FETCH_TREE is untouched until the last step.
         head="$repo/.git/FETCH_HEAD"
         if ! { { [ ! -e "$head" ] || cp "$head" "$stage.oldhead"; } &&
                : > "$repo/.git/FETCH_PENDING"; }; then
           rm -rf "$stage" "$stage.tree" "$stage.oldhead" "$repo/.git/FETCH_PENDING"
           echo "fake git: publishing FETCH_HEAD and FETCH_TREE failed, the previous pair is kept," \
                "the checkout is already updated" >&2
           exit 1
         fi
         if echo "$6" > "$head" && mv "$stage.tree" "$repo/.git/FETCH_TREE"; then
           rm -rf "$stage" "$stage.oldhead" "$repo/.git/FETCH_PENDING"
         else
           restored=1
           if [ -e "$stage.oldhead" ]; then
             cmp -s "$stage.oldhead" "$head" || cat "$stage.oldhead" > "$head" || restored=
           else
             rm -f "$head" || restored=
           fi
           rm -rf "$stage" "$stage.tree" "$stage.oldhead"
           if [ -n "$restored" ]; then
             rm -f "$repo/.git/FETCH_PENDING"
             echo "fake git: publishing FETCH_HEAD and FETCH_TREE failed, the previous pair is kept," \
                  "the checkout is already updated" >&2
           else
             echo "fake git: publishing FETCH_HEAD and FETCH_TREE failed and the previous FETCH_HEAD" \
                  "could not be put back, checkout will refuse them" >&2
           fi
           exit 1
         fi ;;
  checkout) [ ! -e "$repo/.git/FETCH_PENDING" ] ||
              { echo "fake git: FETCH_HEAD and FETCH_TREE are not a pair from one fetch" >&2; exit 1; }
            "$FAKE_REAL_PYTHON" "$FAKE_BIN/git-index.py" record "$repo" &&
            cp "$repo/.git/FETCH_HEAD" "$repo/.git/HEAD" ;;
  rev-parse) if [ -n "${FAKE_GIT_HEAD:-}" ]; then echo "$FAKE_GIT_HEAD"; else cat "$repo/.git/HEAD"; fi ;;
  *) exit 1 ;;
esac
"""

# git-preflight.py STAGE REPO: refuse a fetch whose staged paths cp could not put
# in place in the checkout, before anything there changes.
FAKE_GIT_PREFLIGHT = r"""
import os, sys

stage, repo = sys.argv[1:]


def fits(rel, is_dir):
    dest = os.path.join(repo, rel)
    if os.path.islink(dest):
        return False
    if not os.path.lexists(dest):
        return True
    if is_dir:
        # Read too: a directory this user cannot list hides what is put in it
        # from the index walk and from the installer's copy of the checkout.
        return os.path.isdir(dest) and os.access(dest, os.R_OK | os.W_OK | os.X_OK)
    return os.path.isfile(dest) and os.access(dest, os.W_OK)


def check(rel, is_dir):
    if not fits(rel, is_dir):
        sys.stderr.buffer.write(b"fake git: the checkout cannot take a fetched path, "
                                b"fetched nothing: " + os.fsencode(rel) + b"\n")
        sys.exit(1)


def unreadable(error):
    raise error


# Top down, so each parent is known to be a searchable directory, or absent,
# before anything below it is looked up.
check(".", True)
for top, dirs, names in os.walk(stage, onerror=unreadable):
    here = os.path.relpath(top, stage)
    for name in sorted(dirs):
        check(os.path.normpath(os.path.join(here, name)), True)
    for name in sorted(names):
        check(os.path.normpath(os.path.join(here, name)), False)
"""

# git-published.py STAGE REPO: after the copy into the checkout, check that every
# file STAGE.tree lists is there as a regular file with its staged bytes, so a
# copy that left one out is not passed off as a complete fetch.
FAKE_GIT_PUBLISHED = r"""
import os, sys

stage, repo = (os.fsencode(a) for a in sys.argv[1:])


def published(rel):
    dest = os.path.join(repo, rel)
    try:
        if os.path.islink(dest) or not os.path.isfile(dest):
            return False
        with open(os.path.join(stage, rel), "rb") as staged, open(dest, "rb") as copied:
            return staged.read() == copied.read()
    except OSError:
        return False


with open(stage + b".tree", "rb") as tree:
    for rel in (p for p in tree.read().split(b"\0") if p):
        if not published(rel):
            sys.stderr.buffer.write(b"fake git: the checkout lacks a fetched file: " + rel + b"\n")
            sys.exit(1)
"""

# git-index.py record|status REPO: the fake git's index and its status --porcelain.
FAKE_GIT_INDEX = r"""
import fnmatch, hashlib, json, os, signal, sys

mode, repo = sys.argv[1:]
index_file = os.path.join(repo, ".git", "fake-index.json")


def digest(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def files():
    for top, dirs, names in os.walk(repo):
        dirs[:] = sorted(d for d in dirs if not (top == repo and d == ".git"))
        for name in sorted(names):
            path = os.path.join(top, name)
            yield os.path.relpath(path, repo), digest(path)


def ignored(rel, patterns):
    parts = rel.split("/")
    for pat in patterns:
        dir_only, pat = pat.endswith("/"), pat.rstrip("/")
        for i in range(len(parts) - dir_only):
            name = "/".join(parts[:i + 1]) if "/" in pat else parts[i]
            if fnmatch.fnmatchcase(name, pat.lstrip("/")):
                return True
    return False


index = json.load(open(index_file)) if os.path.exists(index_file) else {}
try:
    patterns = [l.strip() for l in open(os.path.join(repo, ".gitignore"))
                if l.strip() and not l.startswith("#")]
except FileNotFoundError:
    patterns = []
if mode == "record":
    # The commit tracks every file the fetch copied, whether or not .gitignore
    # matches it, and a tracked file stays tracked; anything else in the
    # checkout, such as ignored build output, stays out of the index.
    # FETCH_TREE is NUL-separated, so a name holding a newline is one path.
    # Each fetched file is hashed by name, not looked for in the walk, which
    # skips a directory it cannot list: one that is missing or unreadable fails
    # the checkout, before the index or HEAD changes, rather than drop out of
    # the index.
    with open(os.path.join(repo, ".git", "FETCH_TREE"), "rb") as tree:
        fetched = {os.fsdecode(p) for p in tree.read().split(b"\0") if p}
    recorded = {p: h for p, h in files() if p in index and p not in fetched}
    for p in sorted(fetched):
        path = os.path.join(repo, p)
        try:
            if os.path.islink(path) or not os.path.isfile(path):
                raise FileNotFoundError(path)
            recorded[p] = digest(path)
        except OSError:
            sys.stderr.buffer.write(b"fake git: the checkout lacks a fetched file, index and "
                                    b"HEAD left as they were: " + os.fsencode(p) + b"\n")
            sys.exit(1)
    # Publish the index atomically: a temp file next to it, flushed and closed
    # with the result checked, then renamed over it. A failed or interrupted
    # write leaves the previous index whole and HEAD, which the caller moves
    # only after this exits 0, where it was.
    temp = index_file + ".tmp"

    def abandon(*_):
        try:
            os.unlink(temp)
        except FileNotFoundError:
            pass
        sys.exit(1)

    signal.signal(signal.SIGTERM, abandon)
    try:
        with open(temp, "w") as out:
            json.dump(recorded, out)
            out.flush()
            os.fsync(out.fileno())
        os.replace(temp, index_file)
    except OSError as error:
        sys.stderr.write(f"fake git: cannot write the index, index and HEAD left as they were: {error}\n")
        abandon()
    sys.exit()
tracked_dirs = {"/".join(p.split("/")[:i]) for p in index for i in range(1, p.count("/") + 1)}
current = dict(files())
out = [f" M {p}" for p in sorted(index) if p in current and current[p] != index[p]]
out += [f" D {p}" for p in sorted(index) if p not in current]
untracked = set()
for p in current:
    if p in index or ignored(p, patterns):
        continue
    parts = p.split("/")
    top = next(("/".join(parts[:i]) for i in range(1, len(parts))
                if "/".join(parts[:i]) not in tracked_dirs), None)
    untracked.add(top + "/" if top else p)
out += [f"?? {p}" for p in sorted(untracked)]
print("\n".join(out), end="\n" if out else "")
"""

FAKE_PYTHON3 = r"""#!/bin/sh
echo "python3 $*" >> "$FAKE_LOG"
[ "$1" = "-m" ] && [ "$2" = "venv" ] || exit 1
mkdir -p "$3/bin"
cp "$FAKE_BIN/venv-pip" "$3/bin/pip"
printf '#!/bin/sh\nexec "%s" "$@"\n' "$FAKE_REAL_PYTHON" > "$3/bin/python"
chmod +x "$3/bin/pip" "$3/bin/python"
"""

FAKE_PIP = r"""#!/bin/sh
echo "pip $*" >> "$FAKE_LOG"
[ "$1" = "install" ] || exit 1
for src; do :; done
site="$(dirname "$0")/../lib/site-packages"
if [ -d "$src/build/lib" ]; then mkdir -p "$site"; cp -R "$src/build/lib/." "$site/"; fi
mkdir -p "$src/build/lib" "$src/auto_model_router.egg-info"
printf '#!/bin/sh\necho "auto-router-delegate $*" >> "$FAKE_LOG"\n' > "$(dirname "$0")/auto-router-delegate"
chmod +x "$(dirname "$0")/auto-router-delegate"
"""

# claude|codex mcp get NAME / add [--scope S] [--env K=V] NAME -- CMD / remove NAME [--scope S]
FAKE_AGENT_CLI = r"""#!/bin/sh
tool=$(basename "$0")
echo "$tool $*" >> "$FAKE_LOG"
store="$FAKE_STATE/$tool"
mkdir -p "$store"
[ "$1" = "mcp" ] || exit 2
action=$2; shift 2
case "$action" in
  get) [ -f "$store/$1" ] ;;
  remove) [ -f "$store/$1" ] && rm "$store/$1" ;;
  add) line="$*"; name=
       while [ $# -gt 0 ]; do
         case "$1" in
           --scope) shift 2 ;;
           --env) shift
                  if [ "$tool" = claude ]; then
                    while [ $# -gt 0 ]; do case "$1" in -*) break ;; *) shift ;; esac; done
                  else shift; fi ;;
           --) break ;;
           *) name=$1; shift ;;
         esac
       done
       [ -n "$name" ] || { echo "error: missing required argument" >&2; exit 1; }
       [ ! -f "$store/$name" ] && echo "$line" > "$store/$name" ;;
  *) exit 2 ;;
esac
"""

NEVER_CALLED = r"""#!/bin/sh
echo "$(basename "$0") $*" >> "$FAKE_LOG"
exit 1
"""


@pytest.fixture
def fake(tmp_path):
    """A fake HOME, a PATH of recording fakes, and helpers to run and inspect."""
    home = tmp_path / "home"
    home.mkdir()
    bin_dir = tmp_path / "fake-bin"
    bin_dir.mkdir()
    state = tmp_path / "fake-state"
    log = tmp_path / "calls.log"
    for name, body in {"git": FAKE_GIT, "git-index.py": FAKE_GIT_INDEX,
                       "git-preflight.py": FAKE_GIT_PREFLIGHT,
                       "git-published.py": FAKE_GIT_PUBLISHED,
                       "python3": FAKE_PYTHON3, "venv-pip": FAKE_PIP,
                       "claude": FAKE_AGENT_CLI, "codex": FAKE_AGENT_CLI,
                       "cursor": NEVER_CALLED, "opencode": NEVER_CALLED,
                       "openclaw": NEVER_CALLED, "hermes": NEVER_CALLED,
                       "copilot": NEVER_CALLED}.items():
        (bin_dir / name).write_text(body)
        (bin_dir / name).chmod(0o755)
    env = {"HOME": str(home), "PATH": f"{bin_dir}:/usr/bin:/bin", "LANG": "C.UTF-8",
           "FAKE_LOG": str(log), "FAKE_BIN": str(bin_dir), "FAKE_STATE": str(state),
           "FAKE_SOURCE": str(ROOT), "FAKE_REAL_PYTHON": sys.executable,
           "TMPDIR": str(tmp_path / "tmp")}
    (tmp_path / "tmp").mkdir()

    class Fake:
        pass

    f = Fake()
    f.home, f.state, f.log, f.env, f.tmp = home, state, log, env, tmp_path
    f.base = home / ".auto-router"
    f.link = home / ".local/bin" / NAME

    def run(*argv, extra=None, entry="sh"):
        cmd = (["/bin/sh", str(SHELL)] if entry == "sh"
               else [sys.executable, str(PYTHON_INSTALLER)]) + list(argv)
        return subprocess.run(cmd, capture_output=True, text=True, timeout=60,
                              env={**env, **(extra or {})}, cwd=tmp_path)

    def calls():
        return log.read_text().splitlines() if log.exists() else []

    def home_files():
        return sorted(str(p.relative_to(home)) for p in home.rglob("*")
                      if (p.is_file() or p.is_symlink()) and ".auto-router/src/" not in str(p))

    def built():
        """Every directory pip installed from."""
        return [c.split(" ")[-1] for c in calls() if c.startswith("pip install")]

    def status():
        proc = subprocess.run(["git", "-C", str(f.base / "src"), "status", "--porcelain"],
                              env=env, capture_output=True, text=True)
        assert proc.returncode == 0, proc.stderr
        return proc.stdout

    f.run, f.calls, f.home_files, f.built, f.status = run, calls, home_files, built, status
    return f


def _tools(calls):
    return [c.split(" ")[0] for c in calls]


def test_shell_installer_for_claude_runs_its_whole_path_against_fakes(fake):
    proc = fake.run("claude", SHA)
    assert proc.returncode == 0, proc.stderr
    repo = fake.base / "src"
    assert fake.calls() == [
        f"git init -q {repo}",
        f"git -C {repo} remote add origin {URL}",
        f"git -C {repo} fetch -q --depth 1 origin {SHA}",
        f"git -C {repo} -c advice.detachedHead=false checkout -q --detach FETCH_HEAD",
        f"git -C {repo} rev-parse HEAD",
        f"python3 -m venv {fake.base / 'venv'}",
        f"pip install -q {fake.built()[0]}",
        f"claude mcp get {NAME}",
        f"claude mcp add --scope user {NAME} -- {fake.link}",
    ]
    assert os.readlink(fake.link) == str(fake.base / "venv/bin" / NAME)
    assert (fake.state / "claude" / NAME).read_text().strip() == f"--scope user {NAME} -- {fake.link}"
    assert "Installed auto-router delegation for claude." in proc.stdout
    assert "no AUTO_ROUTER_CONFIG recorded" in proc.stdout
    skill = "plan-with-cheap-workers"
    assert (fake.home / ".claude/skills" / skill / "SKILL.md").read_bytes() == \
        (ROOT / "skills" / skill / "SKILL.md").read_bytes()
    # nothing else in the fake HOME: no global git, pip or agent configuration
    assert fake.home_files() == sorted([
        ".auto-router/venv/bin/auto-router-delegate",
        ".auto-router/venv/bin/pip", ".auto-router/venv/bin/python",
        ".local/bin/auto-router-delegate",
        *(f".claude/skills/{skill}/{p.relative_to(ROOT / 'skills' / skill)}"
          for p in (ROOT / "skills" / skill).rglob("*") if p.is_file()),
    ])
    assert not any(p.name.startswith((".gitconfig", ".claude.json", ".codex"))
                   for p in fake.home.iterdir())


def test_a_second_shell_install_fails_closed_and_force_replaces(fake):
    assert fake.run("claude", SHA).returncode == 0
    entry = (fake.state / "claude" / NAME).read_text()
    first = len(fake.calls())

    again = fake.run("claude", SHA)
    assert again.returncode == 1
    assert f"claude already has an MCP server named '{NAME}'" in again.stderr
    rerun = fake.calls()[first:]
    assert f"git -C {fake.base / 'src'} status --porcelain" in rerun
    assert not any(c.startswith(("git init", "git -C " + str(fake.base / "src") + " remote"))
                   for c in rerun)
    assert not any(c.startswith("python3") for c in rerun), "the existing venv is reused"
    assert [c for c in rerun if c.startswith("claude")] == [f"claude mcp get {NAME}"]
    assert (fake.state / "claude" / NAME).read_text() == entry

    second = len(fake.calls())
    forced = fake.run("claude", SHA, "--force")
    assert forced.returncode == 0, forced.stderr
    assert [c for c in fake.calls()[second:] if c.startswith("claude")] == [
        f"claude mcp get {NAME}", f"claude mcp remove {NAME} --scope user",
        f"claude mcp add --scope user {NAME} -- {fake.link}"]
    assert "skill already current" in forced.stdout


def test_the_package_is_built_outside_the_checkout_so_it_stays_clean(fake):
    assert fake.run("claude", SHA).returncode == 0
    repo = fake.base / "src"
    [built] = fake.built()
    assert not Path(built).is_relative_to(repo) and Path(built).is_relative_to(fake.tmp / "tmp")
    assert not Path(built).exists() and list((fake.tmp / "tmp").iterdir()) == [], \
        "the temporary copy is removed"
    assert sorted(p.name for p in repo.iterdir()) == [
        ".git", ".gitignore", "integrations", "scripts", "skills"]
    assert fake.status() == ""


def test_the_git_fake_reports_edits_and_honours_the_ignore_file(fake):
    assert fake.run("claude", SHA).returncode == 0
    repo = fake.base / "src"
    (repo / "build/lib/auto_router").mkdir(parents=True)
    (repo / "build/lib/auto_router/stale.py").write_text("STALE = 1\n")
    (repo / "auto_model_router.egg-info").mkdir()
    (repo / "scripts/__pycache__").mkdir()
    (repo / "scripts/__pycache__/x.pyc").write_bytes(b"")
    assert fake.status() == "", "ignored build output is not listed, as with real git"
    (repo / "notes.txt").write_text("mine\n")
    (repo / "extra/deep").mkdir(parents=True)
    (repo / "extra/deep/a.txt").write_text("a\n")
    (repo / "skills/new.md").write_text("new\n")
    (repo / "integrations/cursor-plan-with-cheap-workers.mdc").unlink()
    with open(repo / "scripts/install-delegation.py", "a") as f:
        f.write("# local edit\n")
    assert fake.status().splitlines() == [
        " M scripts/install-delegation.py",
        " D integrations/cursor-plan-with-cheap-workers.mdc",
        "?? extra/", "?? notes.txt", "?? skills/new.md"]


def test_a_tracked_edit_in_the_checkout_makes_the_next_install_fail_closed(fake):
    assert fake.run("claude", SHA).returncode == 0
    with open(fake.base / "src/scripts/install-delegation.py", "a") as f:
        f.write("\n# accidental checkout edit\n")
    first = len(fake.calls())
    again = fake.run("claude", SHA, "--force")
    assert again.returncode == 1 and "has local changes" in again.stderr
    assert fake.calls()[first:] == [f"git -C {fake.base / 'src'} status --porcelain"]


def test_stale_ignored_build_output_in_the_checkout_is_not_installed(fake):
    assert fake.run("claude", SHA).returncode == 0
    repo = fake.base / "src"
    stale = repo / "build/lib/auto_router/stale_module.py"
    stale.parent.mkdir(parents=True)
    stale.write_text("STALE = 1\n")
    (repo / "auto_model_router.egg-info").mkdir()
    (repo / "auto_model_router.egg-info/SOURCES.txt").write_text("auto_router/stale_module.py\n")
    assert fake.status() == "", "git ignores it, so the local-changes check passes"
    again = fake.run("claude", SHA, "--force")
    assert again.returncode == 0, again.stderr
    assert len(fake.built()) == 2
    site = fake.base / "venv/lib/site-packages"
    assert not (site / "auto_router/stale_module.py").exists(), "the copy's build/ was dropped"
    assert stale.read_text() == "STALE = 1\n", "the checkout itself is left alone"
    assert (repo / "auto_model_router.egg-info/SOURCES.txt").exists()


def test_ignored_build_output_stays_untracked_across_a_reinstall(fake):
    assert fake.run("claude", SHA).returncode == 0
    repo = fake.base / "src"
    stale = repo / "build/lib/auto_router/stale_module.py"
    stale.parent.mkdir(parents=True)
    stale.write_text("STALE = 1\n")
    (repo / "auto_model_router.egg-info").mkdir()
    (repo / "auto_model_router.egg-info/SOURCES.txt").write_text("auto_router/stale_module.py\n")
    again = fake.run("claude", SHA, "--force")
    assert again.returncode == 0, again.stderr
    stale.write_text("STALE = 2\n")
    (repo / "auto_model_router.egg-info/SOURCES.txt").unlink()
    assert fake.status() == "", "the checkout did not start tracking ignored files"
    third = fake.run("claude", SHA, "--force")
    assert third.returncode == 0, third.stderr
    assert "has local changes" not in third.stderr
    assert len(fake.built()) == 3


def _upstream(fake):
    """A copy of what the fake fetches, to add commits to."""
    upstream = fake.tmp / "upstream"
    for name in ("scripts", "skills", "integrations"):
        shutil.copytree(ROOT / name, upstream / name)
    shutil.copy2(ROOT / ".gitignore", upstream / ".gitignore")
    fake.env["FAKE_SOURCE"] = str(upstream)
    return upstream


def _change(path, change):
    if change == " M":
        with open(path, "a") as f:
            f.write("# local edit\n")
    else:
        path.unlink()


@pytest.mark.parametrize("change", [" M", " D"])
def test_a_fetched_file_the_ignore_file_matches_is_still_tracked(fake, change):
    rel = "scripts/remote_tracked.local.yaml"
    (_upstream(fake) / rel).write_text("upstream: 1\n")
    assert fake.run("claude", SHA).returncode == 0
    assert fake.status() == ""
    _change(fake.base / "src" / rel, change)
    assert fake.status() == f"{change} {rel}\n", "real git reports it: the commit tracks it"
    again = fake.run("claude", SHA, "--force")
    assert again.returncode == 1 and "has local changes" in again.stderr


@pytest.mark.parametrize("rel,rule", [("scripts/remote_tracked.local.yaml", None),
                                      ("scripts/remote_tracked.py", "scripts/remote_tracked.py")])
def test_an_ignored_file_a_later_commit_tracks_is_tracked_after_the_reinstall(fake, rel, rule):
    upstream = _upstream(fake)
    assert fake.run("claude", SHA).returncode == 0
    if rule:
        with open(upstream / ".gitignore", "a") as f:
            f.write(f"\n{rule}\n")
    (upstream / rel).write_text("UPSTREAM = 1\n")
    again = fake.run("claude", SHA, "--force")
    assert again.returncode == 0, again.stderr
    assert fake.status() == ""
    _change(fake.base / "src" / rel, " M")
    assert fake.status() == f" M {rel}\n"
    third = fake.run("claude", SHA, "--force")
    assert third.returncode == 1 and "has local changes" in third.stderr


@pytest.mark.parametrize("rel", ["scripts/two\nlines.py", "scripts/two\nlines.local.yaml",
                                 "scripts/carriage\rreturn.py"])
def test_a_fetched_file_with_a_line_break_in_its_name_is_tracked(fake, rel):
    (_upstream(fake) / rel).write_text("UPSTREAM = 1\n")
    assert fake.run("claude", SHA).returncode == 0
    repo = fake.base / "src"
    assert rel in json.loads((repo / ".git/fake-index.json").read_text())
    assert fake.status() == ""
    again = fake.run("claude", SHA, "--force")
    assert again.returncode == 0, again.stderr
    _change(repo / rel, " M")
    assert fake.status().startswith(" M scripts/"), "real git reports it, quoted"
    third = fake.run("claude", SHA, "--force")
    assert third.returncode == 1 and "has local changes" in third.stderr


@pytest.mark.parametrize("rel,target", [("scripts/link.py", "install-delegation.py"),
                                        ("scripts/link.local.yaml", "install-delegation.py"),
                                        ("scripts/dangling.py", "missing.py"),
                                        ("scripts/link\\name.py", "install-delegation.py")])
def test_a_fetched_symlink_stops_the_fake_fetch_before_it_copies_anything(fake, rel, target):
    upstream = _upstream(fake)
    (upstream / rel).symlink_to(target)
    refused = fake.run("claude", SHA)
    assert refused.returncode == 1
    assert f"fake git: unsupported source entry, only regular files are modelled: {rel}\n" \
        in refused.stderr, "the name exactly as it is, backslashes and all"
    assert "has local changes" not in refused.stderr
    assert not (fake.base / "src/scripts").exists() and fake.built() == []
    (upstream / rel).unlink()
    again = fake.run("claude", SHA)
    assert again.returncode == 0, again.stderr
    assert fake.status() == ""


# Root reads a mode 000 file, so these sources are only unreadable to others.
needs_non_root = pytest.mark.skipif(os.geteuid() == 0, reason="root can read a mode 000 source")


def _fetched_nothing(fake):
    repo = fake.base / "src"
    return [p.name for p in repo.iterdir()] == [".git"] and not any((repo / ".git").iterdir())


@needs_non_root
@pytest.mark.parametrize("rel", ["scripts/unreadable.py", "scripts/unreadable.local.yaml"])
def test_an_unreadable_source_file_fails_the_fake_fetch_without_a_partial_copy(fake, rel):
    upstream = _upstream(fake)
    (upstream / rel).write_text("upstream: 1\n")
    (upstream / rel).chmod(0)
    refused = fake.run("claude", SHA)
    (upstream / rel).chmod(0o644)
    assert refused.returncode == 1
    assert "fake git: cannot copy the source tree, fetched nothing" in refused.stderr
    assert "has local changes" not in refused.stderr
    assert _fetched_nothing(fake) and fake.built() == []
    again = fake.run("claude", SHA)
    assert again.returncode == 0, again.stderr
    repo = fake.base / "src"
    assert rel in json.loads((repo / ".git/fake-index.json").read_text())
    _change(repo / rel, " M")
    assert fake.status() == f" M {rel}\n"
    third = fake.run("claude", SHA, "--force")
    assert third.returncode == 1 and "has local changes" in third.stderr


@needs_non_root
def test_an_unreadable_source_file_leaves_an_earlier_checkout_as_it_was(fake):
    upstream = _upstream(fake)
    rel = "scripts/unreadable.local.yaml"
    (upstream / rel).write_text("upstream: 1\n")
    assert fake.run("claude", SHA).returncode == 0
    repo = fake.base / "src"
    before = {p: p.read_bytes() for p in repo.rglob("*") if p.is_file()}
    (upstream / "scripts/next.py").write_text("NEXT = 1\n")
    (upstream / rel).write_text("upstream: 2\n")
    (upstream / rel).chmod(0)
    refused = fake.run("claude", SHA, "--force")
    (upstream / rel).chmod(0o644)
    assert refused.returncode == 1
    assert "fake git: cannot copy the source tree, fetched nothing" in refused.stderr
    assert {p: p.read_bytes() for p in repo.rglob("*") if p.is_file()} == before
    assert len(fake.built()) == 1


@needs_non_root
def test_an_unreadable_source_directory_stops_the_fake_fetch_before_it_copies_anything(fake):
    private = _upstream(fake) / "scripts/private"
    private.mkdir()
    (private / "link.py").symlink_to("../install-delegation.py")
    private.chmod(0)
    refused = fake.run("claude", SHA)
    private.chmod(0o755)
    assert refused.returncode == 1
    assert "fake git: cannot read the source tree, fetched nothing" in refused.stderr
    assert _fetched_nothing(fake) and fake.built() == []
    hidden = fake.run("claude", SHA)
    assert hidden.returncode == 1
    assert "only regular files are modelled: scripts/private/link.py\n" in hidden.stderr


def _tree(repo):
    """Every path in the checkout, ``.git`` included, with the bytes of each file."""
    return {p: p.read_bytes() if p.is_file() else None for p in repo.rglob("*")}


def _fetched_in_full(upstream, repo):
    return all((repo / p.relative_to(upstream)).read_bytes() == p.read_bytes()
               for p in upstream.rglob("*") if p.is_file())


@pytest.mark.parametrize("local", ["directory", "file", pytest.param("read-only file",
                                                                     marks=needs_non_root)])
def test_a_fetched_path_the_checkout_cannot_take_stops_the_fetch_before_it_writes(fake, local):
    upstream = _upstream(fake)
    assert fake.run("claude", SHA).returncode == 0
    repo = fake.base / "src"
    rel = "scripts/new.local.yaml"
    if local == "directory":
        (repo / rel).mkdir()
        (repo / rel / "cache.bin").write_bytes(b"IGNORED CACHE\n")
        (upstream / rel).write_text("upstream: 2\n")
    elif local == "file":
        (repo / rel).write_text("local: 1\n")
        (upstream / rel).mkdir()
        (upstream / rel / "inside.py").write_text("INSIDE = 1\n")
    else:
        (repo / rel).write_text("local: 1\n")
        (repo / rel).chmod(0o444)
        (upstream / rel).write_text("upstream: 2\n")
    with open(upstream / "scripts/install-delegation.py", "a") as f:
        f.write("# newer upstream revision\n")
    assert fake.status() == "", "the ignored local entry is not a local change"
    before = _tree(repo)
    for _ in range(2):
        refused = fake.run("claude", SHA, "--force")
        assert refused.returncode == 1
        assert f"fake git: the checkout cannot take a fetched path, fetched nothing: {rel}\n" \
            in refused.stderr
        assert "has local changes" not in refused.stderr
        assert _tree(repo) == before, "no checkout file, FETCH_* or index entry changed"
        assert fake.status() == "" and len(fake.built()) == 1
    if (repo / rel).is_dir():
        shutil.rmtree(repo / rel)
    else:
        (repo / rel).unlink()
    again = fake.run("claude", SHA, "--force")
    assert again.returncode == 0, again.stderr
    assert _fetched_in_full(upstream, repo)
    assert fake.status() == "" and len(fake.built()) == 2


# A copy of another user's file belongs to whoever copies it but keeps its mode
# bits. This cp gives each staged copy listed in $FAKE_FOREIGN the mode it then
# has, since a test cannot make a source that another user owns without root.
FOREIGN_CP = r"""#!/bin/sh
/bin/cp "$@" || exit
for dest; do :; done
case "$dest" in
  */.git/fetch-stage/) while read -r mode rel; do chmod "$mode" "$dest$rel"; done < "$FAKE_FOREIGN" ;;
esac
"""


@pytest.mark.parametrize("rel,mode", [("scripts/readable.py", "0044"), ("scripts/private", "0055")])
def test_a_source_another_user_owns_but_lets_us_read_is_fetched_in_full(fake, rel, mode):
    upstream = _upstream(fake)
    (upstream / "scripts/readable.py").write_text("READABLE = 1\n")
    (upstream / "scripts/private").mkdir()
    (upstream / "scripts/private/inside.py").write_text("INSIDE = 1\n")
    cp = Path(fake.env["FAKE_BIN"]) / "cp"
    cp.write_text(FOREIGN_CP)
    cp.chmod(0o755)
    (fake.tmp / "foreign").write_text(f"{mode} {rel}\n")
    fake.env["FAKE_FOREIGN"] = str(fake.tmp / "foreign")
    first = fake.run("claude", SHA)
    assert first.returncode == 0, first.stderr
    with open(upstream / "scripts/install-delegation.py", "a") as f:
        f.write("# newer upstream revision\n")
    again = fake.run("claude", SHA, "--force")
    assert again.returncode == 0, again.stderr
    repo = fake.base / "src"
    assert _fetched_in_full(upstream, repo)
    assert fake.status() == "" and len(fake.built()) == 2
    assert not list((repo / ".git").glob("fetch-stage*"))


@needs_non_root
def test_a_fetched_path_in_a_directory_we_cannot_list_stops_the_fetch_before_it_writes(fake):
    upstream = _upstream(fake)
    assert fake.run("claude", SHA).returncode == 0
    repo = fake.base / "src"
    rel = "scripts/subtree.local.yaml"
    (repo / rel).mkdir()
    (repo / rel).chmod(0o300)
    (upstream / rel).mkdir()
    (upstream / rel / "inside.py").write_text("UPSTREAM = 2\n")
    with open(upstream / "scripts/install-delegation.py", "a") as f:
        f.write("# newer upstream revision\n")
    assert fake.status() == "", "the ignored local directory is not a local change"
    before = _tree(repo)
    for _ in range(2):
        refused = fake.run("claude", SHA, "--force")
        assert refused.returncode == 1
        assert refused.stderr == \
            f"fake git: the checkout cannot take a fetched path, fetched nothing: {rel}\n"
        assert _tree(repo) == before, "no checkout file, FETCH_* or index entry changed"
        assert not os.path.lexists(repo / rel / "inside.py")
        assert (repo / rel).stat().st_mode & 0o777 == 0o300
        assert fake.status() == "" and len(fake.built()) == 1
    (repo / rel).chmod(0o700)
    again = fake.run("claude", SHA, "--force")
    assert again.returncode == 0, again.stderr
    assert _fetched_in_full(upstream, repo)
    assert set(json.loads((repo / ".git/fake-index.json").read_text())) == \
        {str(p.relative_to(upstream)) for p in upstream.rglob("*") if p.is_file()}
    assert fake.status() == "" and len(fake.built()) == 2
    (repo / rel / "inside.py").write_text("LOCAL KEEP ME\n")
    assert fake.status() == f" M {rel}/inside.py\n", "real git tracks the fetched file"
    kept = fake.run("claude", SHA, "--force")
    assert kept.returncode == 1 and "has local changes" in kept.stderr
    assert (repo / rel / "inside.py").read_text() == "LOCAL KEEP ME\n" and len(fake.built()) == 2


# This cp loses the staged file named in $FAKE_LOST just before it copies the
# stage into the checkout, as if it vanished while cp ran: cp copies the rest
# and succeeds.
LOSING_CP = r"""#!/bin/sh
case "$2" in
  */.git/fetch-stage/.) rm -f "${2%.}$FAKE_LOST" ;;
esac
exec /bin/cp "$@"
"""


def test_a_copy_into_the_checkout_that_leaves_a_fetched_file_out_is_reported(fake):
    upstream = _upstream(fake)
    assert fake.run("claude", SHA).returncode == 0
    repo = fake.base / "src"
    rel = "scripts/probe.py"
    (upstream / rel).write_text("PROBE = 1\n")
    with open(upstream / "scripts/install-delegation.py", "a") as f:
        f.write("# newer upstream revision\n")
    cp = Path(fake.env["FAKE_BIN"]) / "cp"
    cp.write_text(LOSING_CP)
    cp.chmod(0o755)
    fake.env["FAKE_LOST"] = rel
    git_dir = {p.name: p.read_bytes() for p in (repo / ".git").iterdir()}
    lost = fake.run("claude", SHA, "--force")
    assert lost.returncode == 1
    assert lost.stderr == (
        f"fake git: the checkout lacks a fetched file: {rel}\n"
        "fake git: copying into the checkout failed part way, it may be partly updated\n")
    assert {p.name: p.read_bytes() for p in (repo / ".git").iterdir()} == git_dir, \
        "FETCH_*, HEAD and the index are as they were and no stage is left"
    assert not (repo / rel).exists() and len(fake.built()) == 1
    assert fake.status() == " M scripts/install-delegation.py\n", "the partial update shows"
    retry = fake.run("claude", SHA, "--force")
    assert retry.returncode == 1 and "has local changes" in retry.stderr
    assert len(fake.built()) == 1


@pytest.mark.parametrize("loss", ["missing", pytest.param("unlisted", marks=needs_non_root)])
def test_the_fake_checkout_records_every_fetched_file_or_fails(fake, loss):
    assert fake.run("claude", SHA).returncode == 0
    repo = fake.base / "src"
    (repo / ".git/FETCH_HEAD").write_text("f" * 40 + "\n")
    before = {n: (repo / ".git" / n).read_bytes() for n in ("HEAD", "fake-index.json")}
    if loss == "missing":
        (repo / "scripts/install-delegation.py").unlink()
    else:
        (repo / "scripts").chmod(0o300)
    checkout = subprocess.run(["git", "-C", str(repo), "checkout", "-q", "--detach", "FETCH_HEAD"],
                              env=fake.env, capture_output=True)
    (repo / "scripts").chmod(0o755)
    after = {n: (repo / ".git" / n).read_bytes() for n in ("HEAD", "fake-index.json")}
    if loss == "missing":
        assert checkout.returncode == 1
        assert checkout.stderr == b"fake git: the checkout lacks a fetched file, index and HEAD " \
                                  b"left as they were: scripts/install-delegation.py\n"
        assert after == before
    else:
        assert checkout.returncode == 0, checkout.stderr
        fetched = {p for p in (repo / ".git/FETCH_TREE").read_bytes().split(b"\0") if p}
        assert {os.fsencode(p) for p in json.loads(after["fake-index.json"])} == fetched, \
            "a file the walk cannot reach is still in the index"
        assert after["HEAD"] == b"f" * 40 + b"\n"


SHA2 = "fedcba9876543210fedcba9876543210fedcba98"

# Stands in for the interpreter that runs git-index.py. For the record step only
# it either caps file size at 512 bytes with SIGXFSZ ignored, so a write past
# that fails with EFBIG as on a full disk ($FAKE_RECORD=fsize), or sends the
# process SIGTERM the moment it opens an index file for writing
# ($FAKE_RECORD=term). Every other call goes straight to the real interpreter.
RECORD_PYTHON = r"""#!/bin/sh
[ "$2" = record ] || exec "$FAKE_REAL" "$@"
exec "$FAKE_REAL" -c '
import builtins, os, resource, runpy, signal, sys
script, *rest = sys.argv[1:]
sys.argv = [script] + rest
if os.environ["FAKE_RECORD"] == "fsize":
    signal.signal(signal.SIGXFSZ, signal.SIG_IGN)
    resource.setrlimit(resource.RLIMIT_FSIZE, (512, 512))
else:
    real_open = builtins.open
    def hook(file, mode="r", *a, **k):
        opened = real_open(file, mode, *a, **k)
        if "w" in mode and os.path.basename(os.fspath(file)).startswith("fake-index.json"):
            os.kill(os.getpid(), signal.SIGTERM)
        return opened
    builtins.open = hook
runpy.run_path(script, run_name="__main__")
' "$@"
"""


def _record_with(fake, how):
    wrapper = fake.tmp / "record-python"
    wrapper.write_text(RECORD_PYTHON)
    wrapper.chmod(0o755)
    fake.env.update(FAKE_REAL=sys.executable, FAKE_REAL_PYTHON=str(wrapper), FAKE_RECORD=how)


def _record_normally(fake):
    fake.env["FAKE_REAL_PYTHON"] = sys.executable
    del fake.env["FAKE_RECORD"]


def _git_dir(repo):
    return {p.name: p.read_bytes() for p in (repo / ".git").iterdir()}


def _installed_then_upstream_adds(fake):
    """A first install, then a newer upstream commit that adds an ignored file."""
    upstream = _upstream(fake)
    assert fake.run("claude", SHA).returncode == 0
    (upstream / "scripts/added.local.yaml").write_text("upstream: 2\n")
    return fake.base / "src"


@pytest.mark.parametrize("how", ["fsize", "term"])
def test_a_failed_or_interrupted_index_write_leaves_the_index_and_head_alone(fake, how):
    repo = _installed_then_upstream_adds(fake)
    before = _git_dir(repo)
    assert len(before["fake-index.json"]) > 512, "the cap would stop a rewrite of it"
    _record_with(fake, how)
    failed = fake.run("claude", SHA2, "--force")
    assert failed.returncode != 0, "a failed index write is not an installed checkout"
    assert "installed successfully" not in failed.stdout
    after = _git_dir(repo)
    assert after["fake-index.json"] == before["fake-index.json"]
    assert after["HEAD"] == before["HEAD"]
    assert not [n for n in after if "tmp" in n or n.startswith("fetch-stage")]
    assert json.loads(after["fake-index.json"]), "the index still parses"
    assert fake.status() == "", "status still runs against the old index"
    _record_normally(fake)
    again = fake.run("claude", SHA2, "--force")
    assert again.returncode == 0, again.stderr
    assert "scripts/added.local.yaml" in json.loads((repo / ".git/fake-index.json").read_text())
    assert (repo / ".git/HEAD").read_text() == SHA2 + "\n"
    assert fake.status() == ""


@needs_non_root
def test_a_fetch_whose_fetch_head_cannot_be_written_keeps_the_previous_pair(fake):
    repo = _installed_then_upstream_adds(fake)
    before = _git_dir(repo)
    (repo / ".git/FETCH_HEAD").chmod(0o444)
    refused = fake.run("claude", SHA2, "--force")
    assert refused.returncode == 1
    assert "publishing FETCH_HEAD and FETCH_TREE failed, the previous pair is kept" in refused.stderr
    assert "copying into the checkout failed" not in refused.stderr
    assert _git_dir(repo) == before, "FETCH_HEAD, FETCH_TREE, HEAD and the index are as they were"
    assert len(fake.built()) == 1
    direct = subprocess.run(["git", "-C", str(repo), "checkout", "-q", "--detach", "FETCH_HEAD"],
                            env=fake.env, capture_output=True)
    assert direct.returncode == 0, direct.stderr
    assert "scripts/added.local.yaml" not in json.loads((repo / ".git/fake-index.json").read_text()), \
        "no new path is indexed under the old HEAD"
    assert (repo / ".git/HEAD").read_bytes() == before["HEAD"]
    (repo / ".git/FETCH_HEAD").chmod(0o644)
    again = fake.run("claude", SHA2, "--force")
    assert again.returncode == 0, again.stderr
    assert "scripts/added.local.yaml" in json.loads((repo / ".git/fake-index.json").read_text())


FAILING_MV = r"""#!/bin/sh
case "$2" in */FETCH_TREE) exit 1 ;; esac
exec /bin/mv "$@"
"""


def test_a_fetch_that_cannot_promote_its_tree_puts_the_previous_fetch_head_back(fake):
    repo = _installed_then_upstream_adds(fake)
    before = _git_dir(repo)
    mv = Path(fake.env["FAKE_BIN"]) / "mv"
    mv.write_text(FAILING_MV)
    mv.chmod(0o755)
    refused = fake.run("claude", SHA2, "--force")
    assert refused.returncode == 1
    assert "publishing FETCH_HEAD and FETCH_TREE failed, the previous pair is kept" in refused.stderr
    assert _git_dir(repo) == before


def test_a_checkout_refuses_a_fetch_head_and_tree_that_are_not_one_fetch(fake):
    assert fake.run("claude", SHA).returncode == 0
    repo = fake.base / "src"
    (repo / ".git/FETCH_HEAD").write_text(SHA2 + "\n")
    (repo / ".git/FETCH_PENDING").write_text("")
    before = _git_dir(repo)
    checkout = subprocess.run(["git", "-C", str(repo), "checkout", "-q", "--detach", "FETCH_HEAD"],
                              env=fake.env, capture_output=True)
    assert checkout.returncode == 1
    assert b"are not a pair from one fetch" in checkout.stderr
    assert _git_dir(repo) == before


def test_a_second_install_into_the_same_home_is_not_refused(fake):
    project = fake.tmp / "project"
    project.mkdir()
    assert fake.run("cursor", SHA, "--project", str(project)).returncode == 0
    again = fake.run("cursor", SHA, "--project", str(project))
    assert again.returncode == 0, again.stderr
    assert "has local changes" not in again.stderr
    other = fake.run("codex", SHA)
    assert other.returncode == 0, other.stderr
    assert len(fake.built()) == 3


def test_shell_installer_for_claude_puts_the_config_after_the_name(fake):
    cfg = fake.tmp / "launcher.yaml"
    cfg.write_text("models: []\n")
    proc = fake.run("claude", SHA, "--config", str(cfg))
    assert proc.returncode == 0, proc.stderr
    add = f"--scope user {NAME} --env AUTO_ROUTER_CONFIG={cfg} -- {fake.link}"
    assert fake.calls()[-1] == f"claude mcp add {add}"
    assert (fake.state / "claude" / NAME).read_text().strip() == add


def test_shell_installer_for_codex_records_an_existing_config(fake):
    cfg = fake.tmp / "launcher.yaml"
    cfg.write_text("models: []\n")
    proc = fake.run("codex", SHA, "--config", str(cfg))
    assert proc.returncode == 0, proc.stderr
    assert fake.calls()[-1] == f"codex mcp add --env AUTO_ROUTER_CONFIG={cfg} {NAME} -- {fake.link}"
    assert (fake.home / ".agents/skills/plan-with-cheap-workers/SKILL.md").exists()
    assert "claude" not in _tools(fake.calls())


def test_shell_installer_for_cursor_and_opencode_writes_json_and_calls_no_cli(fake):
    project = fake.tmp / "project"
    project.mkdir()
    assert fake.run("cursor", SHA, "--project", str(project)).returncode == 0
    oc = fake.home / ".config/opencode/opencode.json"
    oc.parent.mkdir(parents=True)
    oc.write_text(json.dumps({"theme": "mine", "mcp": {"other": {"type": "local"}}}))
    assert fake.run("opencode", SHA).returncode == 0
    assert json.loads((fake.home / ".cursor/mcp.json").read_text()) == {
        "mcpServers": {NAME: {"command": str(fake.link), "args": []}}}
    assert (project / ".cursor/rules/plan-with-cheap-workers.mdc").read_bytes() == \
        (ROOT / "integrations/cursor-plan-with-cheap-workers.mdc").read_bytes()
    assert json.loads(oc.read_text()) == {"theme": "mine", "mcp": {
        "other": {"type": "local"},
        NAME: {"type": "local", "command": [str(fake.link)], "enabled": True}}}
    tools = set(_tools(fake.calls()))
    assert tools == {"git", "python3", "pip"}, "cursor, opencode, claude and codex were never run"


def test_shell_installer_for_openclaw_hermes_and_copilot_edits_files_and_calls_no_cli(fake):
    import yaml

    hermes = fake.home / ".hermes/config.yaml"
    hermes.parent.mkdir(parents=True)
    hermes.write_text("# mine\nmodel:\n  default: x\n")
    project = fake.tmp / "project"
    project.mkdir()
    for argv in (("openclaw",), ("hermes",), ("copilot",), ("copilot", "--project", str(project))):
        proc = fake.run(argv[0], SHA, *argv[1:])
        assert proc.returncode == 0, proc.stderr
        assert f"Installed auto-router delegation for {argv[0]}." in proc.stdout
    link = str(fake.link)
    assert json.loads((fake.home / ".openclaw/openclaw.json").read_text()) == {
        "mcp": {"servers": {NAME: {"command": link, "args": []}}}}
    assert hermes.read_text().startswith("# mine\nmodel:\n  default: x\n")
    assert yaml.safe_load(hermes.read_text())["mcp_servers"] == {NAME: {"command": link, "args": []}}
    assert json.loads((fake.home / ".copilot/mcp-config.json").read_text()) == {"mcpServers": {
        NAME: {"type": "local", "command": link, "args": [], "tools": ["*"]}}}
    assert json.loads((project / ".vscode/mcp.json").read_text()) == {"servers": {
        NAME: {"type": "stdio", "command": link, "args": []}}}
    assert (project / ".github/copilot-instructions.md").exists()
    for skills in (".openclaw/skills", ".hermes/skills"):
        assert (fake.home / skills / "plan-with-cheap-workers/SKILL.md").read_bytes() == \
            (ROOT / "skills/plan-with-cheap-workers/SKILL.md").read_bytes()
    assert set(_tools(fake.calls())) == {"git", "python3", "pip"}, "no agent CLI was run"


def test_a_refused_hermes_entry_leaves_the_yaml_and_skills_alone(fake):
    hermes = fake.home / ".hermes/config.yaml"
    hermes.parent.mkdir(parents=True)
    text = f"mcp_servers:\n  {NAME}:\n    command: /mine\n"
    hermes.write_text(text)
    proc = fake.run("hermes", "--server", "/opt/delegate", "--force", entry="py")
    assert proc.returncode == 1 and "edit it by hand" in proc.stderr
    assert "command: \"/opt/delegate\"" in proc.stderr, "the snippet to paste is printed"
    assert hermes.read_text() == text
    assert sorted(p.name for p in hermes.parent.iterdir()) == ["config.yaml"]


def test_a_checkout_with_local_changes_is_left_alone(fake):
    (fake.base / "src/.git").mkdir(parents=True)
    proc = fake.run("claude", SHA, extra={"FAKE_GIT_DIRTY": "1"})
    assert proc.returncode == 1 and "has local changes" in proc.stderr
    assert fake.calls() == [f"git -C {fake.base / 'src'} status --porcelain"]
    assert not fake.link.exists() and not fake.state.exists()


def test_a_fetched_commit_that_does_not_match_the_pin_stops_before_pip(fake):
    proc = fake.run("claude", SHA, extra={"FAKE_GIT_HEAD": "f" * 40})
    assert proc.returncode == 1 and "does not match" in proc.stderr
    assert _tools(fake.calls()) == ["git"] * 5
    assert not (fake.base / "venv").exists() and not fake.link.exists()


def test_a_foreign_command_link_is_not_replaced(fake):
    fake.link.parent.mkdir(parents=True)
    fake.link.symlink_to("/opt/someone-else/auto-router-delegate")
    proc = fake.run("claude", SHA)
    assert proc.returncode == 1 and "is not this install" in proc.stderr
    assert os.readlink(fake.link) == "/opt/someone-else/auto-router-delegate"
    assert "claude" not in _tools(fake.calls())


def test_a_missing_config_file_stops_before_any_agent_entry(fake):
    proc = fake.run("claude", SHA, "--config", str(fake.tmp / "missing.yaml"))
    assert proc.returncode == 1 and "does not exist" in proc.stderr
    assert "claude" not in _tools(fake.calls())
    assert not (fake.home / ".claude").exists()


def test_the_python_entry_point_refuses_jsonc_and_leaves_it_byte_identical(fake):
    oc = fake.home / ".config/opencode/opencode.json"
    oc.parent.mkdir(parents=True)
    jsonc = '{\n  // mine\n  "mcp": {}\n}\n'
    oc.write_text(jsonc)
    proc = fake.run("opencode", "--server", "/opt/delegate", entry="py")
    assert proc.returncode == 1 and "not plain JSON" in proc.stderr
    assert oc.read_text() == jsonc
    assert sorted(p.name for p in oc.parent.iterdir()) == ["opencode.json"], "no skill copied"
    assert not fake.calls()


@pytest.mark.parametrize("tool", ["claude", "codex"])
def test_a_refused_cli_entry_leaves_no_skill_behind(fake, tool):
    store = fake.state / tool
    store.mkdir(parents=True)
    (store / NAME).write_text("someone else's server\n")
    proc = fake.run(tool, "--server", "/opt/delegate", entry="py")
    assert proc.returncode == 1 and "already has an MCP server" in proc.stderr
    assert fake.calls() == [f"{tool} mcp get {NAME}"]
    assert (store / NAME).read_text() == "someone else's server\n"
    assert list(fake.home.iterdir()) == [], "the refused install wrote nothing to HOME"


def test_a_refused_skill_leaves_the_cli_entry_alone(fake):
    skill = fake.home / ".claude/skills/plan-with-cheap-workers"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text("mine\n")
    proc = fake.run("claude", "--server", "/opt/delegate", entry="py")
    assert proc.returncode == 1 and "exists and differs" in proc.stderr
    assert not fake.calls() and not fake.state.exists()
    assert [p.name for p in skill.iterdir()] == ["SKILL.md"]


def test_a_refused_opencode_entry_leaves_no_skill_behind(fake):
    oc = fake.home / ".config/opencode/opencode.json"
    oc.parent.mkdir(parents=True)
    oc.write_text(json.dumps({"mcp": {NAME: {"type": "local", "command": ["/other"]}}}))
    before = oc.read_bytes()
    proc = fake.run("opencode", "--server", "/opt/delegate", entry="py")
    assert proc.returncode == 1 and "different 'auto-router-delegate' entry" in proc.stderr
    assert oc.read_bytes() == before
    assert sorted(p.name for p in oc.parent.iterdir()) == ["opencode.json"]


def test_a_refused_cursor_rule_leaves_the_global_settings_alone(fake):
    project = fake.tmp / "project"
    rule = project / ".cursor/rules/plan-with-cheap-workers.mdc"
    rule.parent.mkdir(parents=True)
    rule.write_text("my own rule\n")
    proc = fake.run("cursor", "--server", "/opt/delegate", "--project", str(project), entry="py")
    assert proc.returncode == 1 and "exists and differs" in proc.stderr
    assert rule.read_text() == "my own rule\n"
    assert list(fake.home.iterdir()) == [], "no ~/.cursor/mcp.json written"


def test_the_python_entry_point_rejects_an_unknown_tool_through_argparse(fake):
    proc = fake.run("vscode", entry="py")
    assert proc.returncode == 2 and "invalid choice" in proc.stderr
    assert not fake.calls() and list(fake.home.iterdir()) == []


def test_the_python_entry_point_registers_through_the_fake_cli(fake):
    proc = fake.run("claude", "--server", "/opt/delegate", entry="py")
    assert proc.returncode == 0, proc.stderr
    assert fake.calls() == [f"claude mcp get {NAME}",
                            f"claude mcp add --scope user {NAME} -- /opt/delegate"]
