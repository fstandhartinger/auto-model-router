#!/bin/sh
# One-command installer for the auto model router: Linux, macOS and WSL.
#
#   curl -fsSL https://raw.githubusercontent.com/fstandhartinger/auto-model-router/main/scripts/install.sh | sh
#   sh install.sh --yes --models cloud,jev-local --harness opencode,codex
#   sh install.sh --dry-run --models all --harness all      # show every change, make none
#   sh install.sh --uninstall                               # undo harness edits, remove the install
#
# What it does, and nothing else:
#   * fetches github.com/fstandhartinger/auto-model-router (MIT) at --ref
#     (default main) into ~/.auto-router/src
#   * makes a virtualenv in ~/.auto-router/venv and installs the router's
#     dependencies into it (no sudo, nothing system-wide)
#   * writes ~/.auto-router/config.yaml and launcher.yaml (keys by variable NAME only)
#   * writes one launcher, ~/.local/bin/auto-router
#   * configures the harnesses you choose; every edited file is backed up next
#     to itself (*.auto-router-bak-*) and recorded, and --uninstall restores it
#   * downloads a local model only with --with-bonsai / --with-jev-local, after
#     showing its size and getting a yes (or --yes)
#
# It never reads, prints or writes an API key value. Read it before you run it.
#
# Options:
#   --yes, -y              no questions (use the flags below or the defaults)
#   --models LIST          cloud,subscription,bonsai,jev-local | all | none
#   --harness LIST         claude-code,codex,opencode,copilot,cursor,openclaw,hermes | all | none
#   --classifier NAME      auto (default) | jev-local | hosted | heuristic | laya
#   --with-bonsai          download Bonsai 2 if no local endpoint serves it (asks first)
#   --with-jev-local       download the picked Jev-class model likewise (asks first)
#   --claude-gateway       opt-in: Claude Code's ANTHROPIC_BASE_URL -> local router, with
#                          your own login (own login, own machine; see TERMS.md)
#   --project DIR          Cursor: also write the project rule into DIR/.cursor/rules
#   --no-delegate          skip the MCP delegate tool
#   --ref REF              branch, tag or commit to install (default: main)
#   --port N               router port (default 8787)
#   --force                replace differing harness entries (after a backup)
#   --dry-run              print what would change; change nothing
#   --uninstall [--purge]  undo harness edits, remove launcher, venv and code
#                          (--purge also removes config, models and backups dir)
#   --update               fetch --ref and reinstall dependencies; keep all config
#   --json                 machine-readable summary (for coding agents)
#
# Environment: AUTO_ROUTER_HOME (~/.auto-router), AUTO_ROUTER_BIN (~/.local/bin),
# AUTO_ROUTER_PORT, AUTO_ROUTER_REPO, AUTO_ROUTER_REF.

set -eu

REPO=${AUTO_ROUTER_REPO:-https://github.com/fstandhartinger/auto-model-router.git}
REF=${AUTO_ROUTER_REF:-main}
ROOT=${AUTO_ROUTER_HOME:-$HOME/.auto-router}
BIN=${AUTO_ROUTER_BIN:-$HOME/.local/bin}
PORT=${AUTO_ROUTER_PORT:-8787}
MARK="# auto-router launcher (written by install.sh)"

YES=0 DRY=0 UNINSTALL=0 PURGE=0 UPDATE=0 FORCE=0 JSON=0
MODELS="" HARNESS="" CLASSIFIER=auto PROJECT="" EXTRA=""

say() { [ "$JSON" = 1 ] && return 0; printf '  %s\n' "$*"; }
die() { printf '\n  install.sh: %s\n\n' "$*" >&2; exit 1; }
need_arg() { [ "$#" -ge 2 ] || die "$1 needs a value"; }

while [ "$#" -gt 0 ]; do
  case "$1" in
    -y|--yes) YES=1 ;;
    --dry-run) DRY=1 ;;
    --uninstall) UNINSTALL=1 ;;
    --purge) PURGE=1 ;;
    --update) UPDATE=1 ;;
    --force) FORCE=1 ;;
    --json) JSON=1 ;;
    --models) need_arg "$@"; MODELS=$2; shift ;;
    --models=*) MODELS=${1#*=} ;;
    --harness|--harnesses) need_arg "$@"; HARNESS=$2; shift ;;
    --harness=*|--harnesses=*) HARNESS=${1#*=} ;;
    --classifier) need_arg "$@"; CLASSIFIER=$2; shift ;;
    --classifier=*) CLASSIFIER=${1#*=} ;;
    --ref) need_arg "$@"; REF=$2; shift ;;
    --ref=*) REF=${1#*=} ;;
    --port) need_arg "$@"; PORT=$2; shift ;;
    --port=*) PORT=${1#*=} ;;
    --project) need_arg "$@"; PROJECT=$2; shift ;;
    --project=*) PROJECT=${1#*=} ;;
    --with-bonsai|--with-jev-local|--claude-gateway|--no-delegate) EXTRA="$EXTRA $1" ;;
    -h|--help) sed -n '2,50p' "$0" 2>/dev/null || true; exit 0 ;;
    *) die "unknown option $1 (see --help)" ;;
  esac
  shift
done

case "$PORT" in ''|*[!0-9]*) die "--port must be a number" ;; esac
case "$REF" in *[!A-Za-z0-9._/-]*|-*) die "--ref may only contain letters, digits and . _ / -" ;; esac

SRC="$ROOT/src"
VENV="$ROOT/venv"
LAUNCHER="$BIN/auto-router"

# ------------------------------------------------------------------ platform
OS=$(uname -s)
case "$OS" in
  Linux) PLATFORM=linux; grep -qi microsoft /proc/version 2>/dev/null && PLATFORM=wsl ;;
  Darwin) PLATFORM=macos ;;
  MINGW*|MSYS*|CYGWIN*) die "this is a POSIX shell on Windows; use scripts/install.ps1 in PowerShell, or run this inside WSL" ;;
  *) PLATFORM=other ;;
esac

[ "$JSON" = 1 ] || printf '\n  auto-model-router installer (%s)\n  ------------------------------\n' "$PLATFORM"
[ "$DRY" = 1 ] && say "DRY RUN: nothing will be written, downloaded or edited."

find_python() {
  for c in python3.13 python3.12 python3.11 python3.10 python3; do
    command -v "$c" >/dev/null 2>&1 || continue
    if "$c" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then
      echo "$c"; return 0
    fi
  done
  return 1
}

# ----------------------------------------------------------------- uninstall
if [ "$UNINSTALL" = 1 ]; then
  if [ -x "$VENV/bin/python" ] && [ -d "$SRC/auto_router" ]; then
    set -- --uninstall
    [ "$DRY" = 1 ] && set -- "$@" --dry-run
    PYTHONPATH="$SRC" AUTO_ROUTER_HOME="$ROOT" "$VENV/bin/python" -m auto_router.installer "$@"
  else
    say "no virtualenv at $VENV: harness edits (if any) are listed in $ROOT/install-manifest.json"
  fi
  if [ -f "$LAUNCHER" ] && grep -qF "$MARK" "$LAUNCHER"; then
    if [ "$DRY" = 1 ]; then say "[dry-run] would remove $LAUNCHER"; else rm -f "$LAUNCHER"; say "removed $LAUNCHER"; fi
  fi
  for d in "$VENV" "$SRC"; do
    [ -e "$d" ] || continue
    if [ "$DRY" = 1 ]; then say "[dry-run] would remove $d"; else rm -rf "$d"; say "removed $d"; fi
  done
  if [ "$PURGE" = 1 ] && [ -d "$ROOT" ]; then
    if [ "$DRY" = 1 ]; then say "[dry-run] would remove $ROOT (config, models, snippets)"; else rm -rf "$ROOT"; say "removed $ROOT"; fi
  elif [ -d "$ROOT" ]; then
    say "kept $ROOT (your config, downloaded models, manifest); --purge removes it"
  fi
  exit 0
fi

# -------------------------------------------------------------- requirements
PYTHON=$(find_python) || die "Python 3.10 or newer is needed (python3 not found or too old)."
say "python: $($PYTHON --version 2>&1)"
if ! "$PYTHON" -c 'import venv, ensurepip' >/dev/null 2>&1; then
  die "Python's venv/ensurepip module is missing (Debian/Ubuntu: the python3-venv package)."
fi

fetch_code() {  # $1 = target directory
  if command -v git >/dev/null 2>&1; then
    if [ -d "$1/.git" ]; then
      git -C "$1" fetch --quiet --depth 1 origin "$REF" && git -C "$1" -c advice.detachedHead=false checkout --quiet FETCH_HEAD
    else
      rm -rf "$1"
      git init --quiet "$1"
      git -C "$1" remote add origin "$REPO"
      git -C "$1" fetch --quiet --depth 1 origin "$REF" && git -C "$1" -c advice.detachedHead=false checkout --quiet FETCH_HEAD
    fi
    say "code: $REPO @ $REF ($(git -C "$1" rev-parse --short HEAD))"
    return 0
  fi
  # No git: the GitHub tarball of the same ref.
  base=$(printf '%s' "$REPO" | sed -e 's#\.git$##' -e 's#^https://github.com/#https://codeload.github.com/#')
  url="$base/tar.gz/$REF"
  tmp="$1.download.tgz"
  if command -v curl >/dev/null 2>&1; then curl -fsSL "$url" -o "$tmp"
  elif command -v wget >/dev/null 2>&1; then wget -q "$url" -O "$tmp"
  else die "need git, curl or wget to fetch the code"; fi
  rm -rf "$1"; mkdir -p "$1"
  tar -xzf "$tmp" -C "$1" --strip-components=1
  rm -f "$tmp"
  say "code: $url (no git; tarball)"
}

# -------------------------------------------------------- the Python half args
set --
[ "$YES" = 1 ] && set -- "$@" --yes
[ "$DRY" = 1 ] && set -- "$@" --dry-run
[ "$FORCE" = 1 ] && set -- "$@" --force
[ "$JSON" = 1 ] && set -- "$@" --json
[ -n "$MODELS" ] && set -- "$@" --models "$MODELS"
[ -n "$HARNESS" ] && set -- "$@" --harness "$HARNESS"
[ -n "$PROJECT" ] && set -- "$@" --project "$PROJECT"
set -- "$@" --classifier "$CLASSIFIER" --router-url "http://127.0.0.1:$PORT" --launcher "$LAUNCHER"
# shellcheck disable=SC2086
[ -n "$EXTRA" ] && set -- "$@" $EXTRA

interactive=0
if [ "$YES" = 0 ] && [ "$JSON" = 0 ] && [ -t 1 ] && (: </dev/tty) 2>/dev/null; then interactive=1; fi

# ------------------------------------------------------------------- dry run
if [ "$DRY" = 1 ]; then
  if [ -d "$SRC/auto_router" ]; then
    code="$SRC"
  else
    code=$(mktemp -d "${TMPDIR:-/tmp}/auto-router-dry.XXXXXX")
    trap 'rm -rf "$code"' EXIT
    fetch_code "$code"
  fi
  say "[dry-run] would install into $ROOT (code, venv, config) and write $LAUNCHER"
  py="$PYTHON"; [ -x "$VENV/bin/python" ] && py="$VENV/bin/python"
  set -- "$@" --src "$SRC"
  if [ "$interactive" = 1 ]; then
    PYTHONPATH="$code" AUTO_ROUTER_HOME="$ROOT" "$py" -m auto_router.installer "$@" </dev/tty
  else
    PYTHONPATH="$code" AUTO_ROUTER_HOME="$ROOT" "$py" -m auto_router.installer "$@"
  fi
  exit $?
fi

# ------------------------------------------------------------------ the code
mkdir -p "$ROOT" "$BIN"
fetch_code "$SRC"

# ------------------------------------------------------------- the virtualenv
if [ ! -x "$VENV/bin/python" ]; then
  say "creating a virtualenv in $VENV"
  "$PYTHON" -m venv "$VENV" || die "could not create a virtualenv"
fi
say "installing dependencies (this can take a minute)"
"$VENV/bin/python" -m pip install --quiet --disable-pip-version-check -r "$SRC/requirements.txt" \
  || die "installing the router's dependencies failed"
if [ "$CLASSIFIER" = laya ]; then
  say "installing the Laya classifier (CPU torch, about 1.7 GB of weights on first use)"
  case "$PLATFORM" in
    linux|wsl) "$VENV/bin/python" -m pip install --quiet --index-url https://download.pytorch.org/whl/cpu 'torch>=2.0' ;;
    *) "$VENV/bin/python" -m pip install --quiet 'torch>=2.0' ;;
  esac
  "$VENV/bin/python" -m pip install --quiet 'laya>=0.3.4'
fi

# ----------------------------------------------------------------- launcher
if [ -e "$LAUNCHER" ] && ! grep -qF "$MARK" "$LAUNCHER" 2>/dev/null; then
  if [ "$FORCE" = 1 ]; then
    cp -p "$LAUNCHER" "$LAUNCHER.auto-router-bak-$(date +%Y%m%dT%H%M%S)"
    say "backed up the existing $LAUNCHER"
  else
    die "$LAUNCHER exists and was not written by this installer; move it away or use --force"
  fi
fi
cat > "$LAUNCHER" <<LAUNCHER
#!/bin/sh
$MARK
#   auto-router                 start the router on http://127.0.0.1:$PORT/v1
#   auto-router doctor [--live] check config, router, classifier, routes and harness files
#   auto-router check           one routed request through a running router
#   auto-router switch [args]   Claude Code: cheap routes by default, your plan when needed
#   auto-router claude [args]   Claude Code through the router with its own gateway credential
#   auto-router run "<task>"    job-level: pick Codex / Claude / opencode for a whole job, start it
#   auto-router delegate        MCP server: a plan session hands sub-tasks to cheap models
#   auto-router copilot [args]  GitHub Copilot CLI with the router as its BYOK provider
#   auto-router hardware        what this machine can run locally
#   auto-router uninstall       undo harness edits and remove the install
#   auto-router update          fetch the latest code, keep the config
set -eu
ROOT="$ROOT"; SRC="$SRC"; VENV="$VENV"
export PYTHONPATH="\$SRC\${PYTHONPATH:+:\$PYTHONPATH}" AUTO_ROUTER_HOME="\$ROOT"
export AUTO_ROUTER_CACHE_DIR="\${AUTO_ROUTER_CACHE_DIR:-\$ROOT/cache}"
URL="http://\${AUTO_ROUTER_HOST:-127.0.0.1}:\${AUTO_ROUTER_PORT:-$PORT}"
cmd="\${1:-serve}"; [ "\$#" -gt 0 ] && shift
case "\$cmd" in
  serve)
    export AUTO_ROUTER_CONFIG="\${AUTO_ROUTER_CONFIG:-\$ROOT/config.yaml}"
    cd "\$SRC"
    exec "\$VENV/bin/python" -m uvicorn auto_router.server:app --host "\${AUTO_ROUTER_HOST:-127.0.0.1}" --port "\${AUTO_ROUTER_PORT:-$PORT}" "\$@" ;;
  doctor|smoke)
    export AUTO_ROUTER_CONFIG="\${AUTO_ROUTER_CONFIG:-\$ROOT/config.yaml}" AUTO_ROUTER_URL="\$URL"
    exec "\$VENV/bin/python" -m auto_router.smoke "\$@" ;;
  check)
    export AUTO_ROUTER_CONFIG="\${AUTO_ROUTER_CONFIG:-\$ROOT/config.yaml}" AUTO_ROUTER_URL="\$URL"
    exec "\$VENV/bin/python" -m auto_router.smoke --no-start --live "\$@" ;;
  switch)
    export AUTO_ROUTER_CONFIG="\${AUTO_ROUTER_CONFIG:-\$ROOT/config.yaml}" AUTO_ROUTER_URL="\$URL"
    exec "\$VENV/bin/python" -m auto_router.switch "\$@" ;;
  claude)
    # A gateway credential: "the credential replaces the subscription login for that
    # session, and the subscription's usage limits don't apply" (Anthropic's docs).
    ANTHROPIC_BASE_URL="\$URL" ANTHROPIC_API_KEY="\${AUTO_ROUTER_CLAUDE_KEY:-local-router}" exec claude "\$@" ;;
  run)
    export AUTO_ROUTER_CONFIG="\${AUTO_ROUTER_LAUNCHER_CONFIG:-\$ROOT/launcher.yaml}"
    exec "\$VENV/bin/python" "\$SRC/scripts/route-run" "\$@" ;;
  delegate)
    export AUTO_ROUTER_CONFIG="\${AUTO_ROUTER_LAUNCHER_CONFIG:-\$ROOT/launcher.yaml}"
    exec "\$VENV/bin/python" -m auto_router.delegate ;;
  copilot)
    COPILOT_PROVIDER_BASE_URL="\$URL/v1" COPILOT_PROVIDER_TYPE=openai COPILOT_MODEL="\${COPILOT_MODEL:-auto}" exec copilot "\$@" ;;
  hardware)
    exec "\$VENV/bin/python" -m auto_router.hardware "\$@" ;;
  harness)
    exec "\$VENV/bin/python" -m auto_router.harness "\$@" ;;
  uninstall)
    exec sh "\$SRC/scripts/install.sh" --uninstall "\$@" ;;
  update)
    exec sh "\$SRC/scripts/install.sh" --update "\$@" ;;
  -h|--help|help)
    sed -n '3,14p' "\$0" ;;
  *)
    echo "auto-router: unknown command \$cmd (try: auto-router help)" >&2; exit 2 ;;
esac
LAUNCHER
chmod +x "$LAUNCHER"
say "launcher: $LAUNCHER"

if [ "$UPDATE" = 1 ]; then
  say "updated; configuration left as it was"
  exit 0
fi

# ----------------------------------------------------- configure (Python half)
set -- "$@" --src "$SRC"
status=0
if [ "$interactive" = 1 ]; then
  PYTHONPATH="$SRC" AUTO_ROUTER_HOME="$ROOT" "$VENV/bin/python" -m auto_router.installer "$@" </dev/tty || status=$?
else
  PYTHONPATH="$SRC" AUTO_ROUTER_HOME="$ROOT" "$VENV/bin/python" -m auto_router.installer "$@" || status=$?
fi

if [ "$JSON" = 0 ]; then
  printf '\n  Installed.\n\n'
  say "start the router:   auto-router            (http://127.0.0.1:$PORT/v1)"
  say "check everything:   auto-router doctor     (add --live to send one tiny request per route)"
  say "Claude Code:        auto-router switch     (cheap by default, your plan when needed)"
  say "whole jobs:         auto-router run --dry-run \"a task\""
  say "undo everything:    auto-router uninstall"
  case ":$PATH:" in *":$BIN:"*) ;; *) say "note: $BIN is not on your PATH; add it or call $LAUNCHER" ;; esac
  printf '\n'
fi
exit "$status"
