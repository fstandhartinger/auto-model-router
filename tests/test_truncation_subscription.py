"""A plan turn that ran out of output budget is not a successful plan turn.

``truncation.py`` reads the provider's own machine-readable stop flag, and
every metered surface in ``server.py`` records a length stop as ``truncated``.
The subscription surface did not: ``shim.subscription_passthrough`` recorded
``ok`` for every response below HTTP 400 without ever looking at the flag. Two
things went wrong at once, and the second is the worse one:

* the failure disappeared - a turn that spent plan quota without finishing the
  answer read exactly like one that finished;
* the same turn was added to the ``ok`` side of the denominator
  ``outcome_memory`` divides truncations by (it counts ``ok`` and ``truncated``
  and nothing else), so the observed truncation rate was *diluted* rather than
  merely incomplete - and that rate is what the avoidance evidence rests on.

These tests pin the repaired behaviour on both shapes a plan answer arrives in
- one JSON body, and the SSE stream Claude Code actually uses - and pin what
must not change with it: the bytes on the wire, the plan being charged either
way, the ledger's freedom from prompt, answer and credential text, and the
separation between what was observed and what was only estimated.

Nothing here opens a socket: the Anthropic upstream is an in-process fake.
"""

import importlib
import json
import time

import httpx
import pytest
from fastapi.testclient import TestClient

from auto_router import shim
from auto_router.outcome_memory import OutcomeKey, budget_bucket

#: Shaped like the real thing, valid nowhere. Never a real token, not even in
#: a fixture: a test file is the easiest place in a repository to leak one.
FAKE_OAUTH = "sk-ant-oat01-" + "A" * 40
OAUTH_BETAS = "oauth-2026-01-01,claude-code-20250219"
PROMPT = "Build a responsive analytics dashboard with a collapsible sidebar. "
ANSWER = "<!doctype html><html><body><aside>half a page and then it stops"
BUDGET = 12000


# ---------------------------------------------------------------------------
# a gateway whose only route is the plan, and a fake Anthropic behind it
# ---------------------------------------------------------------------------
@pytest.fixture
def gateway(tmp_path, monkeypatch):
    from auto_router import server as server_module

    # A plan with plenty of measured slack, so the quota tier is open and the
    # only configured route really is the plan (quota.py closes it otherwise).
    budget = tmp_path / "budget.json"
    budget.write_text(json.dumps({
        "generated_at": time.time(),
        "claude": {"week_percent": 5, "session_percent": 5,
                   "week_resets_at": time.time() + 3 * 24 * 3600}}))
    config = tmp_path / "cfg.json"
    config.write_text(json.dumps({
        "providers": {"anthropic": {"base_url": "https://api.anthropic.com/v1",
                                    "api": "anthropic"}},
        "subscriptions": {"claude": {"budget_file": str(budget)}},
        "models": [{"name": "plan-claude", "provider": "anthropic",
                    "upstream_id": "claude-opus-5", "subscription": "claude",
                    "capability": {"general": 90, "coding": 90, "agentic": 90}}]}))
    monkeypatch.setenv("AUTO_ROUTER_CONFIG", str(config))
    monkeypatch.setenv("AUTO_ROUTER_LEDGER", str(tmp_path / "ledger.jsonl"))
    server = importlib.reload(server_module)
    yield server
    importlib.reload(server_module)


class _Body:
    """A complete Anthropic message, delivered in one piece."""

    def __init__(self, payload: dict, status: int = 200):
        self.status_code = status
        self.headers = {"content-type": "application/json"}
        self._payload = json.dumps(payload).encode()

    async def aread(self):
        return self._payload

    async def aclose(self):
        return None


class _Stream:
    """An Anthropic SSE body, delivered in the chunks a socket would deliver."""

    def __init__(self, chunks: list[bytes], status: int = 200):
        self.status_code = status
        self.headers = {"content-type": "text/event-stream"}
        self.chunks = chunks

    async def aiter_raw(self):
        for chunk in self.chunks:
            yield chunk

    async def aclose(self):
        return None


class _Upstream:
    """Just enough of ``httpx.AsyncClient`` to stand in for api.anthropic.com."""

    def __init__(self, response):
        self.response = response
        self.urls: list[str] = []

    def build_request(self, method, url, headers=None, content=None, params=None):
        self.urls.append(url)
        return object()

    async def send(self, _request, stream=False):
        return self.response


def _message(stop_reason: str) -> dict:
    # No message id: the publication gate allows Anthropic's own id prefix only
    # on the one line of the translator that has to produce it, and nothing on
    # this path reads the field anyway.
    return {"type": "message", "role": "assistant",
            "model": "claude-opus-5", "stop_reason": stop_reason, "stop_sequence": None,
            "content": [{"type": "text", "text": ANSWER}],
            "usage": {"input_tokens": 4321, "output_tokens": BUDGET}}


def _sse(stop_reason: str) -> list[bytes]:
    """The event sequence Anthropic streams, stop reason last, as it really is."""
    events = [
        {"type": "message_start",
         "message": {**_message(stop_reason), "content": [], "stop_reason": None,
                     "usage": {"input_tokens": 4321, "output_tokens": 0}}},
        {"type": "content_block_start", "index": 0,
         "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0,
         "delta": {"type": "text_delta", "text": ANSWER}},
        {"type": "content_block_stop", "index": 0},
        {"type": "message_delta", "delta": {"stop_reason": stop_reason, "stop_sequence": None},
         "usage": {"output_tokens": BUDGET}},
        {"type": "message_stop"},
    ]
    return [f"event: {e['type']}\ndata: {json.dumps(e)}\n\n".encode() for e in events]


def _call(gateway, monkeypatch, response, *, stream=False):
    """One subscription-authenticated turn. Returns (status, body bytes, upstream)."""
    upstream = _Upstream(response)
    monkeypatch.setattr(shim, "upstream", lambda: upstream)
    client = TestClient(gateway.app)
    payload = {"model": "claude-opus-5", "max_tokens": BUDGET, "stream": stream,
               "messages": [{"role": "user", "content": PROMPT * 40}]}
    headers = {"authorization": f"Bearer {FAKE_OAUTH}", "anthropic-beta": OAUTH_BETAS}
    if stream:
        with client.stream("POST", "/v1/messages", json=payload, headers=headers) as resp:
            body = b"".join(resp.iter_bytes())
            return resp.status_code, body, upstream
    resp = client.post("/v1/messages", json=payload, headers=headers)
    return resp.status_code, resp.content, upstream


def _record(gateway):
    return gateway.router.decisions[-1]


def _shape(name):
    """Both shapes a plan answer arrives in, for a parametrised test."""
    return {"json": (lambda stop: _Body(_message(stop)), False),
            "stream": (lambda stop: _Stream(_sse(stop)), True)}[name]


# ---------------------------------------------------------------------------
# reading the flag at all
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("shape", ["json", "stream"])
def test_a_plan_answer_that_ran_out_of_budget_is_not_recorded_as_a_success(
        gateway, monkeypatch, shape):
    """The defect itself: HTTP 200 plus ``max_tokens`` used to read as ``ok``."""
    build, stream = _shape(shape)
    status, _body, upstream = _call(gateway, monkeypatch, build("max_tokens"), stream=stream)

    assert status == 200
    assert upstream.urls == ["https://api.anthropic.com/v1/messages"]
    observed = _record(gateway).observed
    assert observed.status == "truncated"
    assert observed.error == "stop_reason:max_tokens"
    assert observed.model == "plan-claude"


@pytest.mark.parametrize("shape", ["json", "stream"])
def test_a_plan_answer_that_finished_is_still_recorded_as_ok(gateway, monkeypatch, shape):
    """The unchanged half: an ordinary turn reads exactly as it did before."""
    build, stream = _shape(shape)
    status, _body, _upstream = _call(gateway, monkeypatch, build("end_turn"), stream=stream)

    assert status == 200
    observed = _record(gateway).observed
    assert observed.status == "ok"
    assert observed.error is None


@pytest.mark.parametrize("stop", ["tool_use", "stop_sequence", "refusal"])
def test_a_plan_stop_reason_that_is_not_a_length_stop_is_not_a_truncation(
        gateway, monkeypatch, stop):
    """Only the length vocabulary counts; a tool call is a finished turn."""
    _status, _body, _upstream = _call(gateway, monkeypatch, _Body(_message(stop)))
    assert _record(gateway).observed.status == "ok"


def test_an_answer_that_merely_reads_as_cut_off_is_not_a_truncation(gateway, monkeypatch):
    """Prose inference is refused here too: the flag decides, never the text."""
    message = {**_message("end_turn"),
               "content": [{"type": "text", "text": "the dashboard should therefore"}]}
    _status, _body, _upstream = _call(gateway, monkeypatch, _Body(message))
    assert _record(gateway).observed.status == "ok"


# ---------------------------------------------------------------------------
# the dilution this was actually about
# ---------------------------------------------------------------------------
def test_a_plan_length_stop_counts_as_one_in_the_outcome_memory(gateway, monkeypatch):
    """What the avoidance evidence reads: one observation, and it is a truncation.

    Before the fix the same turn landed in the memory as ``ok``, which both
    hid the truncation and enlarged the denominator its rate is measured
    against - so the recorded rate was wrong in both directions at once.
    """
    _call(gateway, monkeypatch, _Stream(_sse("max_tokens")), stream=True)

    key = OutcomeKey(category=_record(gateway).classification.category,
                     budget=budget_bucket(BUDGET))
    evidence = gateway.router.outcomes.evidence("plan-claude", key)
    assert (evidence.observed, evidence.truncated) == (1, 1)
    assert evidence.rate == 1.0


def test_a_finished_plan_turn_still_counts_on_the_other_side(gateway, monkeypatch):
    """The denominator is not simply emptied: an ``ok`` is still an observation."""
    _call(gateway, monkeypatch, _Stream(_sse("end_turn")), stream=True)

    key = OutcomeKey(category=_record(gateway).classification.category,
                     budget=budget_bucket(BUDGET))
    evidence = gateway.router.outcomes.evidence("plan-claude", key)
    assert (evidence.observed, evidence.truncated) == (1, 0)


# ---------------------------------------------------------------------------
# what must not change with it
# ---------------------------------------------------------------------------
def test_the_client_still_gets_the_upstream_bytes_unchanged(gateway, monkeypatch):
    """Reading the stop flag must not edit, reorder or buffer away the stream."""
    chunks = _sse("max_tokens")
    _status, body, _upstream = _call(gateway, monkeypatch, _Stream(chunks), stream=True)
    assert body == b"".join(chunks)


def test_a_stop_flag_split_across_two_chunks_is_still_read(gateway, monkeypatch):
    """A socket does not respect event boundaries, so neither may the reader."""
    joined = b"".join(_sse("max_tokens"))
    cut = len(joined) - 40
    _status, body, _upstream = _call(
        gateway, monkeypatch, _Stream([joined[:cut], joined[cut:]]), stream=True)

    assert body == joined
    assert _record(gateway).observed.status == "truncated"


def test_the_truncated_plan_turn_is_still_charged_to_the_plan(gateway, monkeypatch):
    """The tokens were really spent, so the turn is still committed and still free."""
    committed: list = []
    monkeypatch.setattr(gateway.router, "commit",
                        lambda result, *a, **k: committed.append(result.model.name))

    _call(gateway, monkeypatch, _Body(_message("max_tokens")))

    assert committed == ["plan-claude"]
    observed = _record(gateway).observed
    assert observed.cost_usd is None
    assert "no per-token charge" in observed.cost_basis


def test_a_truncated_plan_turn_invents_no_estimate_and_no_capability_score(gateway, monkeypatch):
    """Observation and estimate stay apart; nothing here edits a score."""
    plan = gateway.router.config.catalog.get("plan-claude")
    before = dict(plan.capability)
    _call(gateway, monkeypatch, _Stream(_sse("max_tokens")), stream=True)

    record = _record(gateway)
    assert record.observed.status == "truncated"
    assert record.estimated.p_success is not None      # the estimate still stands
    assert record.observed.output_tokens is None       # and nothing was inferred into it
    assert gateway.router.config.catalog.get("plan-claude").capability == before


def test_a_truncated_plan_turn_writes_no_prompt_answer_or_credential(gateway, monkeypatch, tmp_path):
    """The label is built from fixed vocabulary, so the ledger stays prompt-free."""
    _call(gateway, monkeypatch, _Stream(_sse("max_tokens")), stream=True)

    written = (tmp_path / "ledger.jsonl").read_text()
    assert '"status":"truncated"' in written
    assert "stop_reason:max_tokens" in written
    for leak in (PROMPT.strip(), ANSWER, FAKE_OAUTH, "sk-ant"):
        assert leak not in written


def test_an_error_from_the_plan_still_reads_as_an_upstream_error(gateway, monkeypatch):
    """A 4xx is not a length stop, and carries no label into the record."""
    status, _body, _upstream = _call(
        gateway, monkeypatch, _Body({"type": "error", "error": {"type": "overloaded_error"}}, 529))

    assert status == 529
    observed = _record(gateway).observed
    assert (observed.status, observed.http_status, observed.error) == ("upstream_error", 529, None)


# ---------------------------------------------------------------------------
# a stream that never reached its end
# ---------------------------------------------------------------------------
class _BrokenStream(_Stream):
    """200 headers, then the socket dies part way through the body."""

    def __init__(self, chunks: list[bytes], at: int):
        super().__init__(chunks)
        self.at = at

    async def aiter_raw(self):
        for index, chunk in enumerate(self.chunks):
            if index == self.at:
                raise httpx.ReadError("connection reset by peer")
            yield chunk


def _broken_call(gateway, monkeypatch, response):
    """One streamed turn whose body fails, with the client's exception kept."""
    try:
        _call(gateway, monkeypatch, response, stream=True)
    except Exception as exc:      # noqa: BLE001 - the failure is the subject here
        return type(exc).__name__
    raise AssertionError("the broken stream did not fail the client")


def test_a_plan_stream_that_dies_mid_body_is_not_recorded_as_a_success(gateway, monkeypatch):
    """A stream that stopped early said nothing; absence of a flag is not ``ok``.

    The headers really were 200, so the old reading was "no length stop, so the
    turn succeeded" - for a turn the client received in pieces or not at all.
    ``server._anthropic_stream`` has always called this ``transport_error``.
    """
    _broken_call(gateway, monkeypatch, _BrokenStream(_sse("end_turn"), at=3))

    observed = _record(gateway).observed
    assert observed.status == "transport_error"
    assert observed.error == "transport:ReadError"    # a class name, never a body
    assert observed.http_status == 200                # what the headers really said


def test_a_broken_plan_stream_is_kept_out_of_the_outcome_memory(gateway, monkeypatch):
    """The worse half of the same error: it must not pad the denominator.

    ``outcome_memory`` counts ``ok`` and ``truncated`` and divides one by the
    other, so a transport failure banked as ``ok`` would lower the observed
    truncation rate the avoidance rule reads.
    """
    _broken_call(gateway, monkeypatch, _BrokenStream(_sse("max_tokens"), at=2))

    key = OutcomeKey(category=_record(gateway).classification.category,
                     budget=budget_bucket(BUDGET))
    assert gateway.router.outcomes.evidence("plan-claude", key) is None
    assert gateway.router.outcomes.stats["observations"] == 0


def test_a_length_stop_already_read_survives_a_broken_stream(gateway, monkeypatch):
    """The provider's own flag outranks the transport: it was really observed."""
    chunks = _sse("max_tokens")
    _broken_call(gateway, monkeypatch, _BrokenStream(chunks, at=len(chunks) - 1))

    observed = _record(gateway).observed
    assert (observed.status, observed.error) == ("truncated", "stop_reason:max_tokens")


# ---------------------------------------------------------------------------
# the reader itself
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("payload,expected", [
    ({"stop_reason": "max_tokens"}, "stop_reason:max_tokens"),
    ({"type": "message_delta", "delta": {"stop_reason": "max_tokens"}}, "stop_reason:max_tokens"),
    ({"stop_reason": "end_turn"}, None),
    ({"type": "message_delta", "delta": {"stop_reason": "tool_use"}}, None),
    ({"type": "message_stop"}, None),
    ({"delta": "not a mapping"}, None),
    ({}, None),
    ("not a mapping at all", None),
    (None, None),
])
def test_the_length_stop_reader_answers_only_what_it_was_told(payload, expected):
    assert shim._length_stop(payload) == expected


@pytest.mark.parametrize("blob", [b"", b"not json", b"[]", b'{"stop_reason": "end_turn"}',
                                  b"\xff\xfe not utf-8"])
def test_a_body_that_says_nothing_is_not_a_truncation(blob):
    """Silence is not evidence - a malformed body may not become a length stop."""
    assert shim._length_stop_in(blob) is None
