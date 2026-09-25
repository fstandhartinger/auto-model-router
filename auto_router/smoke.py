"""`auto-router doctor`: check an installation end to end.

    python -m auto_router.smoke [--config FILE] [--url URL] [--live] [--json]

1. **Router.** Pings ``/health`` at ``--url``. If nothing answers, it starts a
   temporary router on a free loopback port with your config, and stops it
   again at the end.
2. **Classifier.** Builds the classifier ``policy.classifier`` names. A local
   Jev-class endpoint is pinged; hosted Jev only checks that
   ``TYPESAFE_API_KEY`` is set (by name).
3. **Routes.** For every enabled route: is its key variable set, is a local
   endpoint reachable. With ``--live`` it also sends one tiny request
   (``max_tokens: 5``) to each route's provider and one ``model: auto``
   request through the router - those cost a fraction of a cent on metered
   routes. Without ``--live`` nothing billable is sent (dry run).
4. **Harnesses.** Every file the installer edited still holds its entry;
   Claude Code's opt-in gateway is flagged if the router is not running,
   because Claude Code then cannot reach any model.

Key values are never printed.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from . import harness


def _get(url: str, timeout: float = 3.0) -> Any:
    with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310 - local or configured URL
        return json.load(resp)


def _post(url: str, body: dict, headers: dict | None = None, timeout: float = 60.0) -> tuple[int, Any, dict]:
    req = urllib.request.Request(url, json.dumps(body).encode(),
                                 {"Content-Type": "application/json", **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            return resp.status, json.load(resp), dict(resp.headers)
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()[:300].decode(errors="replace"), {}


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Report:
    def __init__(self) -> None:
        self.items: list[dict] = []

    def add(self, area: str, name: str, status: str, detail: str = "") -> None:
        self.items.append({"area": area, "name": name, "status": status, "detail": detail})

    @property
    def failed(self) -> bool:
        return any(i["status"] == "fail" for i in self.items)


def load_raw(path: Path) -> dict:
    import yaml
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def enabled_models(raw: dict, env: dict[str, str]) -> list[dict]:
    models = [m for m in raw.get("models") or [] if isinstance(m, dict)]
    only = env.get("AUTO_ROUTER_MODELS")
    names = [x.strip() for x in only.split(",")] if only else raw.get("enabled")
    if names:
        models = [m for m in models if m.get("name") in names]
    return models


def check_router(url: str, config: Path, rep: Report, *, start: bool) -> subprocess.Popen | None:
    try:
        health = _get(url.rstrip("/") + "/health")
        rep.add("router", url, "ok", f"running: {health}")
        return None
    except Exception as exc:
        if not start:
            rep.add("router", url, "fail", f"not answering ({exc.__class__.__name__}); start it with `auto-router`")
            return None
    port = _free_port()
    env = {**os.environ, "AUTO_ROUTER_CONFIG": str(config)}
    proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "auto_router.server:app", "--host", "127.0.0.1",
                             "--port", str(port), "--log-level", "warning"], env=env,
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    temp = f"http://127.0.0.1:{port}"
    for _ in range(120):
        if proc.poll() is not None:
            err = (proc.stderr.read() if proc.stderr else "")[-400:]
            rep.add("router", temp, "fail", f"a temporary router did not start: {err.strip()}")
            return None
        try:
            health = _get(temp + "/health", 1.0)
            rep.add("router", temp, "ok", f"not running at {url}; started a temporary router for this check: {health}")
            proc.url = temp  # type: ignore[attr-defined]
            return proc
        except Exception:
            time.sleep(0.5)
    proc.terminate()
    rep.add("router", temp, "fail", "a temporary router did not answer within 60 s")
    return None


def check_classifier(raw: dict, env: dict[str, str], rep: Report) -> None:
    cfg = (raw.get("policy") or {}).get("classifier") or {}
    backend = str(cfg.get("backend") or ("hosted" if env.get("TYPESAFE_API_KEY") else "heuristic")).lower()
    try:
        from . import jev
        jev.classifier_from_config(raw.get("policy") or {})
    except Exception as exc:
        rep.add("classifier", backend, "fail", f"this router cannot build it: {exc}")
        return
    if backend in ("local-jev", "jev-local", "local-jev-class", "openai-compatible"):
        base = str(cfg.get("base_url") or "http://127.0.0.1:8080/v1").rstrip("/")
        try:
            ids = [m.get("id") for m in _get(base + "/models").get("data", [])]
            want = cfg.get("model")
            rep.add("classifier", backend, "ok" if not want or want in ids or not ids else "warn",
                    f"{base} serves {ids}" + ("" if not want or want in ids else f"; configured model {want!r} not listed"))
        except Exception as exc:
            rep.add("classifier", backend, "fail", f"{base} not reachable ({exc.__class__.__name__}); "
                    "start it (~/.auto-router/bin/start-jev-local) - routing falls back cautiously meanwhile")
    elif backend in ("hosted", "jev"):
        rep.add("classifier", backend, "ok" if env.get("TYPESAFE_API_KEY") else "fail",
                "TYPESAFE_API_KEY is " + ("set" if env.get("TYPESAFE_API_KEY") else "not set"))
    elif backend == "local":
        rep.add("classifier", backend, "ok", "Laya in-process (first request downloads its weights)")
    else:
        rep.add("classifier", backend, "warn", "no classifier model: routing uses the heuristic")


def check_routes(raw: dict, env: dict[str, str], rep: Report, *, live: bool) -> None:
    providers = raw.get("providers") or {}
    for m in enabled_models(raw, env):
        name, prov = m.get("name"), providers.get(m.get("provider")) or {}
        base = str(prov.get("base_url") or "").rstrip("/")
        if m.get("subscription"):
            rep.add("route", name, "ok", "subscription route: reached through its own CLI (switch mode / route-run), not tested here")
            continue
        local = prov.get("local") or "127.0.0.1" in base or "localhost" in base
        key_env = prov.get("api_key_env")
        if not local and key_env and not env.get(key_env):
            rep.add("route", name, "fail", f"{key_env} is not set")
            continue
        if prov.get("api") == "anthropic":
            rep.add("route", name, "ok", "Anthropic Messages provider: not probed")
            continue
        if not (live or local):
            rep.add("route", name, "dry-run", f"would POST {base}/chat/completions model={m.get('upstream_id')} "
                    f"max_tokens=5 (use --live to send)")
            continue
        headers = {"Authorization": f"Bearer {env[key_env]}"} if key_env and env.get(key_env) else {}
        status, data, _ = _post(base + "/chat/completions",
                                {"model": m.get("upstream_id"), "max_tokens": 5,
                                 "messages": [{"role": "user", "content": "Reply OK."}]}, headers, 120)
        rep.add("route", name, "ok" if status == 200 else "fail",
                f"HTTP {status}" + ("" if status == 200 else f": {str(data)[:160]}"))


def check_through_router(url: str, rep: Report, *, live: bool) -> None:
    if not live:
        rep.add("router", "auto request", "dry-run", "would POST /v1/chat/completions model=auto (use --live)")
        return
    status, data, headers = _post(url.rstrip("/") + "/v1/chat/completions",
                                  {"model": "auto", "max_tokens": 20,
                                   "messages": [{"role": "user", "content": "Reply with the single word OK."}]})
    rep.add("router", "auto request", "ok" if status == 200 else "fail",
            f"HTTP {status}, chosen {headers.get('X-Router-Model')}" if status == 200 else f"HTTP {status}: {str(data)[:200]}")


def check_harnesses(ctx: harness.Ctx, rep: Report, router_up: bool) -> None:
    manifest = harness.load_manifest(ctx)
    if not manifest["changes"]:
        rep.add("harness", "-", "warn", "the installer has configured no harness yet")
    for c in manifest["changes"]:
        path = Path(c["path"])
        if c.get("kind") == "cli":
            rep.add("harness", c["harness"], "ok", f"registered via CLI: {c['entries'][0]['describe']}")
            continue
        if not path.exists():
            rep.add("harness", c["harness"], "fail", f"{path} is gone")
            continue
        if c.get("kind") == "dir":
            rep.add("harness", c["harness"], "ok", f"{path} present")
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        ok = True
        for e in c.get("entries", []):
            if e.get("kind") == "json":
                data = harness._read_json(path)
                ok = ok and data is not None and harness._get(data, e["keys"]) is not None
            elif e.get("kind") == "block":
                ok = ok and harness.BEGIN in text
        rep.add("harness", c["harness"], "ok" if ok else "fail",
                f"{path}: " + ("entries present" if ok else "an installer entry is missing (edited since?)"))
    settings = ctx.claude_dir() / "settings.json"
    data = harness._read_json(settings) if settings.exists() else {}
    base = harness._get(data or {}, ["env", "ANTHROPIC_BASE_URL"])
    if base:
        rep.add("harness", "claude-code gateway", "ok" if router_up else "fail",
                f"ANTHROPIC_BASE_URL={base}" + ("" if router_up else
                                                "; the router is not running, so Claude Code cannot reach a model"))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Check the auto-router installation.")
    ctx = harness.Ctx.from_env()
    p.add_argument("--config", default=os.environ.get("AUTO_ROUTER_CONFIG") or str(ctx.state_dir / "config.yaml"))
    p.add_argument("--url", default=os.environ.get("AUTO_ROUTER_URL", "http://127.0.0.1:8787"))
    p.add_argument("--live", action="store_true", help="send one tiny request per route (costs a fraction of a cent)")
    p.add_argument("--no-start", action="store_true", help="do not start a temporary router")
    p.add_argument("--json", action="store_true")
    args = p.parse_args(argv)
    env = dict(os.environ)
    rep = Report()
    config = Path(args.config).expanduser()
    if not config.exists():
        rep.add("config", str(config), "fail", "missing; run the installer first")
        raw: dict = {}
    else:
        raw = load_raw(config)
        rep.add("config", str(config), "ok", f"{len(raw.get('models') or [])} models, "
                f"{len(enabled_models(raw, env))} enabled")
    proc = check_router(args.url, config, rep, start=not args.no_start and config.exists())
    url = getattr(proc, "url", args.url)
    router_up = any(i["area"] == "router" and i["status"] == "ok" for i in rep.items)
    try:
        if raw:
            check_classifier(raw, env, rep)
            check_routes(raw, env, rep, live=args.live)
            if router_up:
                check_through_router(url, rep, live=args.live)
        check_harnesses(ctx, rep, router_up=router_up and proc is None)
    finally:
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(10)
            except subprocess.TimeoutExpired:
                proc.kill()
    if args.json:
        print(json.dumps({"ok": not rep.failed, "checks": rep.items}, indent=2))
    else:
        for i in rep.items:
            print(f"  [{i['status']:7}] {i['area']:10} {i['name']}: {i['detail']}")
        print("  OK" if not rep.failed else "  Some checks failed (see above).")
    return 1 if rep.failed else 0


if __name__ == "__main__":
    sys.exit(main())
