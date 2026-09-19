"""The delegate MCP tool: a plan session hands sub-tasks to cheaper routes."""

import io
import json
import subprocess
import types

from auto_router import delegate


def _rpc(*messages):
    out = io.StringIO()
    delegate.main(io.StringIO("\n".join(json.dumps(m) for m in messages) + "\n"), out)
    return [json.loads(line) for line in out.getvalue().splitlines()]


def test_the_server_speaks_the_mcp_handshake_and_lists_one_tool():
    init, listed = _rpc({"jsonrpc": "2.0", "id": 1, "method": "initialize",
                         "params": {"protocolVersion": "2025-06-18", "capabilities": {}}},
                        {"jsonrpc": "2.0", "method": "notifications/initialized"},
                        {"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
    assert init["result"]["protocolVersion"] == "2025-06-18"
    assert "tools" in init["result"]["capabilities"]
    assert [t["name"] for t in listed["result"]["tools"]] == ["delegate"]


def test_unknown_methods_and_tools_are_errors_not_crashes():
    bad_method, bad_tool = _rpc({"jsonrpc": "2.0", "id": 1, "method": "resources/list"},
                                {"jsonrpc": "2.0", "id": 2, "method": "tools/call",
                                 "params": {"name": "rm", "arguments": {}}})
    assert bad_method["error"]["code"] == -32601
    assert bad_tool["error"]["code"] == -32602


def test_a_delegated_job_never_uses_a_plan_route(tmp_path):
    seen = {}

    def fake_run(argv, **kw):
        seen.update(argv=argv, cwd=kw["cwd"], stdin=kw["stdin"])
        return types.SimpleNamespace(returncode=0, stdout="all 4 tests pass\n",
                                     stderr="route-run: kimi-k3 (free) ...\n")

    text, is_error = delegate.run_delegate("write tests", str(tmp_path), run=fake_run)
    assert "--no-plans" in seen["argv"] and seen["argv"][-1] == "write tests"
    assert seen["cwd"] == str(tmp_path) and seen["stdin"] is subprocess.DEVNULL
    assert not is_error and "all 4 tests pass" in text and "kimi-k3" in text


def test_failures_and_timeouts_are_reported_to_the_model(tmp_path):
    def failing(argv, **kw):
        return types.SimpleNamespace(returncode=3, stdout="", stderr="route-run: no route\n")

    def slow(argv, **kw):
        raise subprocess.TimeoutExpired(argv, 1)

    assert delegate.run_delegate("x", str(tmp_path), run=failing)[1] is True
    text, is_error = delegate.run_delegate("x", str(tmp_path), timeout_s=1, run=slow)
    assert is_error and "did not finish" in text
    assert delegate.run_delegate("x", "/no/such/dir")[1] is True
    assert delegate.run_delegate("  ")[1] is True


def test_long_output_keeps_the_end():
    def chatty(argv, **kw):
        return types.SimpleNamespace(returncode=0, stdout="a" * 20000 + "THE END", stderr="")

    text, _ = delegate.run_delegate("x", run=chatty)
    assert text.endswith("THE END") and len(text) < delegate.MAX_OUTPUT + 200
