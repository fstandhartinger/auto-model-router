"""Runs the documented local HTTP smoke test as part of the suite.

The implementation lives in ``scripts/smoke_http.py`` so it can also be run on
its own. It binds only to loopback on ephemeral ports, keeps everything in a
temporary directory, and tears both processes down before it returns.
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "smoke_http.py"


@pytest.mark.skipif(shutil.which("uvicorn") is None and not (REPO / "auto_router").exists(),
                    reason="server dependencies are not installed")
def test_local_http_smoke_and_outage_paths(tmp_path):
    report = tmp_path / "smoke.json"
    env = {**os.environ, "AUTO_ROUTER_SMOKE_REPORT": str(report), "TYPESAFE_API_KEY": "",
           "PYTHONPATH": str(REPO)}
    proc = subprocess.run([sys.executable, str(SCRIPT)], cwd=REPO, env=env,
                          capture_output=True, text=True, timeout=600)
    detail = proc.stdout + proc.stderr
    assert report.exists(), f"smoke test produced no report\n{detail}"
    data = json.loads(report.read_text())
    failed = [c for c in data["checks"] if not c["ok"]]
    assert not failed, "failed checks: " + json.dumps(failed, indent=1) + "\n" + detail
    assert proc.returncode == 0, detail
    # Guard against an empty or truncated report silently passing.
    assert data["passed"] >= 24, json.dumps(data, indent=1)
    # The point of the exercise: the server is gone again.
    assert any(c["check"] == "no service left behind" and c["ok"] for c in data["checks"])
