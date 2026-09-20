#!/bin/sh
set -eu

PYTHON_BIN="${PYTHON_BIN:-python3}"
case "$(uname -s)" in
  Linux) "$PYTHON_BIN" -m pip install --index-url https://download.pytorch.org/whl/cpu 'torch>=2.0' ;;
  *) "$PYTHON_BIN" -m pip install 'torch>=2.0' ;;
esac
"$PYTHON_BIN" -m pip install 'auto-model-router[local] @ git+https://github.com/fstandhartinger/auto-model-router.git'
printf '%s\n' 'Installed. Copy examples/config.example.yaml, then run: auto-model-router'
