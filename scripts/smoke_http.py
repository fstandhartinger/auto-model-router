"""Isolated local HTTP smoke test, including the outage and fallback paths.

What it does, all on loopback and all in a temporary directory:

1. Starts a stub OpenAI-compatible upstream on an ephemeral port. It reports
   token usage, and can be told to fail so the escalation path is exercised.
2. Writes a throwaway router config pointing at that stub, with a benchmark
   API base URL that does not resolve and no ``TYPESAFE_API_KEY``, so the
   benchmark-outage and classifier-outage fallbacks are what is actually
   under test.
3. Starts the router itself on a second ephemeral port as a real subprocess
   and drives it over HTTP.
4. Checks health, the model list, a routed completion, the metrics, and the
   decision records, then asserts that nothing leaked prompt text.
5. Tears both processes down in a ``finally`` and verifies that neither port
   is still accepting connections, so no service is left behind.

Run it directly (``python scripts/smoke_http.py``) or through
``tests/test_http_smoke.py``, which simply calls ``main``.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from contextlib import closing
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SECRET = "sk-" + "ant-api03-" + "S" * 24   # built, never a literal: see the gate
PROMPT = "Draft a responsive dashboard layout with a sidebar. "

#: Stub upstream. Kept in this file so the smoke test is a single artifact.
STUB = '''
import json, os
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI()
STATE = {"calls": 0, "fail_next": int(os.environ.get("STUB_FAIL_FIRST", "0"))}


@app.get("/healthz")
async def healthz():
    return {"calls": STATE["calls"]}


@app.post("/v1/chat/completions")
async def completions(request: Request):
    body = await request.json()
    STATE["calls"] += 1
    if STATE["fail_next"] > 0:
        STATE["fail_next"] -= 1
        return JSONResponse({"error": {"message": "stub is failing on purpose"}}, status_code=503)
    return {
        "id": "stub-1",
        "object": "chat.completion",
        "model": body.get("model"),
        "choices": [{"index": 0, "finish_reason": "stop",
                     "message": {"role": "assistant", "content": "stub answer"}}],
        "usage": {"prompt_tokens": 4321, "completion_tokens": 128,
                  "prompt_tokens_details": {"cached_tokens": 4000}},
    }
'''


def free_port() -> int:
    with closing(socket.socket()) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def port_open(port: int, timeout: float = 0.4) -> bool:
    with closing(socket.socket()) as sock:
        sock.settimeout(timeout)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def wait_for(port: int, deadline: float = 40.0, proc: subprocess.Popen | None = None) -> None:
    start = time.time()
    while time.time() - start < deadline:
        if port_open(port):
            return
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(f"process exited early with code {proc.returncode}")
        time.sleep(0.15)
    raise TimeoutError(f"nothing listening on 127.0.0.1:{port} after {deadline:.0f}s")


def request(url: str, payload: dict | None = None, timeout: float = 30.0):
    data = json.dumps(payload).encode() if payload is not None else None
    headers = {"Content-Type": "application/json"} if data else {}
    req = urllib.request.Request(url, data=data, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read() or b"{}"), _headers(resp)
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read() or b"{}"), _headers(exc)


def _headers(response) -> dict:
    """Header names are case-insensitive on the wire; uvicorn sends them lowercase."""
    return {k.title(): v for k, v in response.headers.items()}


def spawn(module_dir: Path, app: str, port: int, env: dict) -> subprocess.Popen:
    return subprocess.Popen(
        [sys.executable, "-m", "uvicorn", app, "--host", "127.0.0.1", "--port", str(port),
         "--log-level", "warning"],
        cwd=module_dir, env={**os.environ, **env},
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


def stop(proc: subprocess.Popen | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10)


def check(ok: bool, label: str, detail: str = "") -> dict:
    line = f"{'PASS' if ok else 'FAIL'}  {label}"
    print(line + (f"  ({detail})" if detail else ""), flush=True)
    return {"check": label, "ok": bool(ok), "detail": detail}


def main(argv: list[str] | None = None) -> int:
    results: list[dict] = []
    tmp = Path(tempfile.mkdtemp(prefix="auto-router-smoke-"))
    stub_port, router_port = free_port(), free_port()
    stub = router = None
    try:
        (tmp / "stub_upstream.py").write_text(STUB)
        # An unroutable benchmark host and an empty cache directory: the config
        # must build from its own numbers alone.
        config = {
            "providers": {"stub": {"base_url": f"http://127.0.0.1:{stub_port}/v1",
                                   "cache": "openai"}},
            "policy": {"name": "F_expected", "success": {"evidence_discount": 0.5}},
            "models": [
                {"name": "smoke-cheap", "provider": "stub", "upstream_id": "stub/cheap",
                 "prices": {"input": 0.1, "output": 0.4, "cache_read": 0.01},
                 "capability": {"coding": 45, "design": 40, "general": 45}},
                {"name": "smoke-design", "provider": "stub", "upstream_id": "stub/design",
                 "prices": {"input": 3.0, "output": 15.0, "cache_read": 0.3},
                 "capability": {"coding": 70, "design": 82, "general": 65}},
                {"name": "smoke-strong", "provider": "stub", "upstream_id": "stub/strong",
                 "prices": {"input": 5.0, "output": 25.0, "cache_read": 0.5},
                 "capability": {"coding": 88, "design": 50, "general": 80}},
            ],
        }
        config_path = tmp / "smoke-config.json"
        config_path.write_text(json.dumps(config))
        ledger = tmp / "decisions.jsonl"

        stub = spawn(tmp, "stub_upstream:app", stub_port, {"STUB_FAIL_FIRST": "0",
                                                           "PYTHONPATH": str(tmp)})
        wait_for(stub_port, proc=stub)

        router = spawn(REPO, "auto_router.server:app", router_port, {
            "AUTO_ROUTER_CONFIG": str(config_path),
            "AUTO_ROUTER_CACHE_DIR": str(tmp / "bench-cache"),
            # Deliberately unroutable: the benchmark-outage fallback is under test.
            "AUTO_ROUTER_BENCH_URL": "http://127.0.0.1:1/bench-is-down",
            "AUTO_ROUTER_LEDGER": str(ledger),
            "AUTO_ROUTER_MAX_ATTEMPTS": "3",
            # No classifier key: the cautious-default path is under test.
            "TYPESAFE_API_KEY": "",
            "PYTHONPATH": str(REPO),
        })
        wait_for(router_port, proc=router)
        base = f"http://127.0.0.1:{router_port}"

        status, health, _ = request(f"{base}/health")
        results.append(check(status == 200 and health.get("status") == "ok",
                             "health endpoint answers", json.dumps(health)))
        results.append(check(health.get("models") == 3,
                             "catalog built with the benchmark API unreachable",
                             f"{health.get('models')} models"))

        status, models, _ = request(f"{base}/v1/models")
        names = {m["id"] for m in models.get("data", [])}
        results.append(check(status == 200 and names == {"smoke-cheap", "smoke-design",
                                                         "smoke-strong"},
                             "model list served", ", ".join(sorted(names))))
        results.append(check(all(m["capability_source"] in ("config", "bench+config")
                                 for m in models["data"]),
                             "capability falls back to the config when the API is down"))

        body = {"model": "auto", "max_tokens": 900, "messages": [
            {"role": "user", "content": PROMPT * 60 + f" my api_key={SECRET}"}]}
        status, completion, headers = request(f"{base}/v1/chat/completions", body)
        results.append(check(status == 200 and completion["choices"][0]["message"]["content"]
                             == "stub answer", "routed completion returns the upstream answer"))
        chosen = headers.get("X-Router-Model")
        results.append(check(chosen in names, "a route was selected", str(chosen)))
        results.append(check(bool(headers.get("X-Router-Decision")),
                             "decision id is returned on the response"))
        results.append(check(headers.get("X-Router-Safe-Fallback") == "classifier-unavailable"
                             or headers.get("X-Router-Safe-Fallback") == "classifier-disabled",
                             "classifier outage is declared in the response headers",
                             str(headers.get("X-Router-Safe-Fallback"))))

        status, decisions, _ = request(f"{base}/v1/router/decisions?limit=5")
        record = (decisions.get("data") or [{}])[0]
        results.append(check(status == 200 and record.get("selection", {}).get("selected") == chosen,
                             "decision record matches the response header"))
        results.append(check(record.get("estimated_outcome", {}).get("kind") == "estimate"
                             and record.get("observed_outcome", {}).get("kind") == "observed",
                             "estimate and observation are separate objects"))
        observed = record.get("observed_outcome") or {}
        results.append(check(observed.get("tokens", {}).get("cached_read") == 4000
                             and observed.get("tokens", {}).get("output") == 128,
                             "provider-reported tokens are recorded as observed",
                             json.dumps(observed.get("tokens"))))
        results.append(check(observed.get("observed_cache_hit_rate") is not None,
                             "cache hit rate is measured, not assumed",
                             str(observed.get("observed_cache_hit_rate"))))

        blob = json.dumps(decisions)
        results.append(check(SECRET not in blob and "responsive dashboard" not in blob,
                             "no prompt text or credential in the decision API"))

        status, metrics, _ = request(f"{base}/v1/router/metrics")
        results.append(check(status == 200 and metrics.get("total_requests") == 1,
                             "metrics counted the request"))
        results.append(check(metrics["router"]["ledger"]["written"] >= 1,
                             "the routing ledger was written"))

        # -- outage and fallback: the upstream fails, the router escalates ---
        request(f"http://127.0.0.1:{stub_port}/healthz")
        stop(stub)
        stub = spawn(tmp, "stub_upstream:app", stub_port,
                     {"STUB_FAIL_FIRST": "1", "PYTHONPATH": str(tmp)})
        wait_for(stub_port, proc=stub)
        body["messages"] = [{"role": "user", "content": "Prove that the sum of two odds is even. " * 90}]
        status, completion, headers = request(f"{base}/v1/chat/completions", body)
        results.append(check(status == 200,
                             "a failing upstream is survived by escalating", f"HTTP {status}"))
        status, decisions, _ = request(f"{base}/v1/router/decisions?limit=5")
        records = decisions.get("data") or []
        escalated = [r for r in records
                     if (r.get("observed_outcome") or {}).get("status") == "upstream_error"]
        results.append(check(bool(escalated), "the failed attempt is recorded as observed",
                             f"{len(escalated)} failed attempt(s) recorded"))
        results.append(check(all(r["selection"]["fallback"] for r in records if r["selection"]
                                 ["turn_start"]),
                             "every turn-start decision names a fallback route"))

        lines = [json.loads(line) for line in ledger.read_text().splitlines() if line.strip()]
        results.append(check(len(lines) >= 2, "ledger is machine-readable JSONL",
                             f"{len(lines)} records"))
        results.append(check(SECRET not in ledger.read_text()
                             and "responsive dashboard" not in ledger.read_text(),
                             "no prompt text or credential in the ledger file"))

        # -- outage and fallback: the classifier is configured but unreachable
        # The first phase ran with no key at all, which exercises "disabled".
        # This phase sets a key and points the endpoint at an unroutable host,
        # which is what an actual Jev outage looks like from inside the server.
        stop(router)
        router = spawn(REPO, "auto_router.server:app", router_port, {
            "AUTO_ROUTER_CONFIG": str(config_path),
            "AUTO_ROUTER_CACHE_DIR": str(tmp / "bench-cache"),
            "AUTO_ROUTER_BENCH_URL": "http://127.0.0.1:1/bench-is-down",
            "AUTO_ROUTER_LEDGER": str(ledger),
            # Unroutable on purpose: this is a classifier outage, not a missing key.
            "AUTO_ROUTER_JEV_URL": "http://127.0.0.1:1/jev-is-down",
            "AUTO_ROUTER_JEV_ATTEMPTS": "1",
            "TYPESAFE_API_KEY": "smoke-test-placeholder-not-a-real-key",
            "PYTHONPATH": str(REPO),
        })
        wait_for(router_port, proc=router)
        body["messages"] = [{"role": "user", "content": "Summarise this changelog. " * 90}]
        started = time.perf_counter()
        status, completion, headers = request(f"{base}/v1/chat/completions", body)
        elapsed = time.perf_counter() - started
        results.append(check(status == 200,
                             "a classifier outage does not break routing", f"HTTP {status}"))
        results.append(check(headers.get("X-Router-Safe-Fallback") == "classifier-unavailable",
                             "the classifier outage is named as a safe fallback",
                             str(headers.get("X-Router-Safe-Fallback"))))
        results.append(check(float(headers.get("X-Router-Evidence", "1")) < 1.0,
                             "the outage lowers the recorded evidence confidence",
                             str(headers.get("X-Router-Evidence"))))
        results.append(check(elapsed < 60,
                             "the outage fails fast instead of hanging the request",
                             f"{elapsed:.1f}s"))
    finally:
        stop(router)
        stop(stub)

    left_behind = [p for p in (router_port, stub_port) if port_open(p, timeout=0.3)]
    results.append(check(not left_behind, "no service left behind",
                         f"ports still open: {left_behind}" if left_behind else "both ports closed"))

    failed = [r for r in results if not r["ok"]]
    report = {"checks": results, "passed": len(results) - len(failed), "failed": len(failed),
              "tmpdir": str(tmp)}
    out = os.environ.get("AUTO_ROUTER_SMOKE_REPORT")
    if out:
        Path(out).write_text(json.dumps(report, indent=1))
    print(f"\n{len(results) - len(failed)}/{len(results)} checks passed; artifacts in {tmp}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
