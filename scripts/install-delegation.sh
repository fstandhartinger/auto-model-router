#!/bin/sh
# Install the delegate MCP server and skill for one agent, from a pinned commit.
#
#   sh install-delegation.sh claude|codex|opencode|cursor <commit-sha> [install-delegation.py options]
#
# The commit is required and must be a full 40-character SHA you have
# reviewed: the script never installs whatever a branch points at today. It
# checks the commit out detached into $AUTO_ROUTER_HOME (default
# ~/.auto-router), refuses an existing checkout with local changes, installs
# it into its own virtualenv, and links ~/.local/bin/auto-router-delegate only
# if that name is free or already this link. Everything else - the MCP entry
# and the skill - is done by install-delegation.py, which never overwrites a
# differing entry without --force. No credential is read or written.
set -eu

tool=${1:-}
ref=${2:-}
case "$tool" in claude|codex|opencode|cursor) ;; *)
  echo "usage: install-delegation.sh claude|codex|opencode|cursor <commit-sha> [--config FILE] [--project DIR] [--force]" >&2
  exit 2
esac
if ! printf '%s' "$ref" | grep -Eq '^[0-9a-f]{40}$'; then
  echo "install-delegation: give the full 40-character commit SHA to install (a reviewed commit, not a branch)" >&2
  exit 2
fi
shift 2

base=${AUTO_ROUTER_HOME:-"$HOME/.auto-router"}
repo="$base/src"
url=https://github.com/fstandhartinger/auto-model-router.git
if [ -e "$repo" ] && [ ! -d "$repo/.git" ]; then
  echo "install-delegation: $repo exists and is not a git checkout; move it away first" >&2
  exit 1
fi
if [ ! -d "$repo/.git" ]; then
  mkdir -p "$base"
  git init -q "$repo"
  git -C "$repo" remote add origin "$url"
elif [ -n "$(git -C "$repo" status --porcelain)" ]; then
  echo "install-delegation: $repo has local changes; not touching it" >&2
  exit 1
fi
git -C "$repo" fetch -q --depth 1 origin "$ref"
git -C "$repo" -c advice.detachedHead=false checkout -q --detach FETCH_HEAD
if [ "$(git -C "$repo" rev-parse HEAD)" != "$ref" ]; then
  echo "install-delegation: checked-out commit does not match $ref" >&2
  exit 1
fi

[ -x "$base/venv/bin/python" ] || python3 -m venv "$base/venv"
"$base/venv/bin/pip" install -q "$repo"

link="$HOME/.local/bin/auto-router-delegate"
target="$base/venv/bin/auto-router-delegate"
mkdir -p "$HOME/.local/bin"
if [ -L "$link" ] && [ "$(readlink "$link")" = "$target" ]; then
  :
elif [ -e "$link" ] || [ -L "$link" ]; then
  echo "install-delegation: $link already exists and is not this install; leaving it alone" >&2
  exit 1
else
  ln -s "$target" "$link"
fi
"$base/venv/bin/python" "$repo/scripts/install-delegation.py" "$tool" --server "$link" "$@"
