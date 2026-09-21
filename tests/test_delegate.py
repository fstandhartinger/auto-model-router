"""The delegation MCP tools: a plan session hands bounded work to routed workers."""

import io
import json
import subprocess
import time
import types

from auto_router import delegate


def _rpc(*messages):
    out = io.StringIO()
    delegate.main(io.StringIO("\n".join(json.dumps(m) for m in messages) + "\n"), out)
    return [json.loads(line) for line in out.getvalue().splitlines()]


def test_the_server_lists_both_tools_and_speaks_mcp():
    init, listed = _rpc(
        {"jsonrpc": "2.0", "id": 1, "method": "initialize",
         "params": {"protocolVersion": "2025-06-18", "capabilities": {}}},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
    assert init["result"]["protocolVersion"] == "2025-06-18"
    assert [tool["name"] for tool in listed["result"]["tools"]] == ["delegate", "delegate_many"]


def test_unknown_methods_and_tools_are_errors_not_crashes():
    bad_method, bad_tool = _rpc(
        {"jsonrpc": "2.0", "id": 1, "method": "resources/list"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/call",
         "params": {"name": "rm", "arguments": {}}})
    assert bad_method["error"]["code"] == -32601
    assert bad_tool["error"]["code"] == -32602


def _fake_process(returncode=0, stdout="done", route="cheap-worker", estimate=0.012):
    record = {"route": route, "decision": {"estimated_outcome": {"cost_usd": estimate}}}
    return types.SimpleNamespace(returncode=returncode, stdout=stdout,
                                 stderr="route summary\n" + json.dumps(record) + "\n")


def test_a_worker_never_uses_a_plan_and_reports_model_cost_basis_and_brief(tmp_path):
    seen = {}

    def fake_run(argv, **kwargs):
        seen.update(argv=argv, cwd=kwargs["cwd"], stdin=kwargs["stdin"])
        return _fake_process()

    result = delegate.run_delegate("write tests", "only parser.py", str(tmp_path), run=fake_run)
    assert "--no-plans" in seen["argv"] and seen["argv"][-1].endswith("only parser.py")
    assert seen["cwd"] == str(tmp_path) and seen["stdin"] is subprocess.DEVNULL
    assert result["ok"] and result["model"] == "cheap-worker"
    assert result["cost_usd"] is None and result["estimated_cost_usd"] == 0.012
    assert "not measured" in result["cost_basis"] and result["brief_tokens_estimate"] > 0


def test_failures_timeouts_and_invalid_inputs_are_structured(tmp_path):
    result = delegate.run_delegate("x", cwd=str(tmp_path), run=lambda *a, **k: _fake_process(3, ""))
    assert not result["ok"] and result["exit_code"] == 3

    def slow(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], 1)

    timed = delegate.run_delegate("x", cwd=str(tmp_path), timeout_s=1, run=slow)
    assert not timed["ok"] and "timed out" in timed["error"]
    assert not delegate.run_delegate("x", cwd="/no/such/dir")["ok"]
    assert not delegate.run_delegate("  ")["ok"]
    assert not delegate.run_delegate("x", tier="impossible")["ok"]


def test_long_output_keeps_the_end():
    result = delegate.run_delegate("x", run=lambda *a, **k: _fake_process(stdout="a" * 20000 + "THE END"))
    assert result["result"].endswith("THE END") and result["output_truncated"]


def test_delegate_many_runs_concurrently_preserves_order_and_sums_overhead():
    def worker(task, context, cwd, tier, timeout_s):
        time.sleep(0.03 if task == "one" else 0.01)
        return {"ok": True, "result": task, "cost_usd": None,
                "estimated_cost_usd": 0.1, "brief_tokens_estimate": len(task)}

    started = time.perf_counter()
    result = delegate.run_many(["one", "two"], parallel=2, runner=worker)
    assert time.perf_counter() - started < 0.055
    assert [item["result"] for item in result["results"]] == ["one", "two"]
    assert result["estimated_cost_usd"] == 0.2 and result["cost_usd"] is None
    assert result["brief_tokens_estimate"] == 6


def test_tool_call_returns_structured_content(monkeypatch):
    monkeypatch.setattr(delegate, "run_delegate", lambda *a, **k: {
        "ok": True, "model": "worker", "result": "ok", "cost_usd": 0.0})
    response = _rpc({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                     "params": {"name": "delegate", "arguments": {"task": "x"}}})[0]
    assert response["result"]["structuredContent"]["model"] == "worker"
    assert not response["result"]["isError"]
