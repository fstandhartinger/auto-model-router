#!/bin/sh
set -eu

tool=${1:-}
case "$tool" in claude|codex|opencode|cursor) ;; *)
  echo "usage: install-delegation.sh claude|codex|opencode|cursor" >&2
  exit 2
esac

base=${AUTO_ROUTER_HOME:-"$HOME/.auto-router"}
repo="$base/src"
if [ ! -d "$repo/.git" ]; then
  git clone --depth 1 https://github.com/fstandhartinger/auto-model-router.git "$repo"
else
  git -C "$repo" pull --ff-only
fi
python3 -m venv "$base/venv"
"$base/venv/bin/pip" install -q "$repo"
mkdir -p "$HOME/.local/bin"
ln -sf "$base/venv/bin/auto-router-delegate" "$HOME/.local/bin/auto-router-delegate"
"$base/venv/bin/python" "$repo/scripts/install-delegation.py" "$tool"
