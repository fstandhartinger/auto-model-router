from __future__ import annotations

import json
import logging

import pytest

from auto_router.pricing import Usage
from auto_router.shim import redact
from auto_router.stream_translate import StreamOutcome, translate_stream
from auto_router.translate import (
    TranslationError,
    anthropic_sse_from_message,
    messages_to_openai,
    openai_response_to_anthropic,
    tool_choice_to_openai,
    tools_to_openai,
)

FAKE_OAUTH = "sk-ant-oat01-" + "A" * 100


# --------------------------------------------------------------------------
# request translation
# --------------------------------------------------------------------------
def test_system_and_text_messages():
    got = messages_to_openai([{"role": "user", "content": "hi"}], system="be terse")
    assert got == [{"role": "system", "content": "be terse"},
                   {"role": "user", "content": "hi"}]


def test_structured_system_blocks_are_joined():
    got = messages_to_openai(
        [{"role": "user", "content": "hi"}],
        system=[{"type": "text", "text": "a"}, {"type": "text", "text": "b"}],
    )
    assert got[0]["content"] == "a\nb"


def test_tool_use_becomes_openai_tool_call():
    got = messages_to_openai([
        {"role": "user", "content": "read the file"},
        {"role": "assistant", "content": [
            {"type": "tool_use", "id": "toolu_1", "name": "Read",
             "input": {"path": "/tmp/x"}}]},
    ])
    call = got[-1]["tool_calls"][0]
    assert call["id"] == "toolu_1"
    assert call["function"]["name"] == "Read"
    assert json.loads(call["function"]["arguments"]) == {"path": "/tmp/x"}


def test_tool_result_precedes_following_user_text():
    got = messages_to_openai([
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "toolu_1", "content": "file body"},
            {"type": "text", "text": "now summarise"},
        ]},
    ])
    assert [m["role"] for m in got] == ["tool", "user"]
    assert got[0]["tool_call_id"] == "toolu_1"
    assert got[0]["content"] == "file body"


def test_structured_tool_result_content_is_flattened():
    got = messages_to_openai([
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "t1",
             "content": [{"type": "text", "text": "line1"},
                         {"type": "text", "text": "line2"}]}]},
    ])
    assert got[0]["content"] == "line1\nline2"


def test_thinking_blocks_are_dropped_not_rejected():
    got = messages_to_openai([
        {"role": "assistant", "content": [
            {"type": "thinking", "thinking": "hmm", "signature": "x"},
            {"type": "text", "text": "answer"}]}])
    assert got[0]["content"] == "answer"


def test_base64_image_becomes_data_url():
    got = messages_to_openai([{"role": "user", "content": [
        {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                     "data": "QUJD"}}]}])
    assert got[0]["content"][0]["image_url"]["url"] == "data:image/png;base64,QUJD"


def test_unknown_block_type_fails_loudly():
    with pytest.raises(TranslationError):
        messages_to_openai([{"role": "user", "content": [{"type": "server_tool_use"}]}])


def test_tool_result_without_id_fails_loudly():
    with pytest.raises(TranslationError):
        messages_to_openai([{"role": "user", "content": [
            {"type": "tool_result", "content": "x"}]}])


def test_nameless_tool_fails_loudly():
    with pytest.raises(TranslationError):
        tools_to_openai([{"type": "web_search_20250305"}])


def test_tools_and_tool_choice_translation():
    got = tools_to_openai([{"name": "Read", "description": "read a file",
                            "input_schema": {"type": "object"}}])
    assert got[0]["function"]["name"] == "Read"
    assert tool_choice_to_openai({"type": "tool", "name": "Read"}) == {
        "type": "function", "function": {"name": "Read"}}
    assert tool_choice_to_openai("any") == "required"
    assert tool_choice_to_openai({"type": "auto"}) == "auto"


# --------------------------------------------------------------------------
# response translation
# --------------------------------------------------------------------------
def test_openai_response_with_tool_call_maps_to_tool_use():
    message = openai_response_to_anthropic({
        "id": "chatcmpl-1",
        "choices": [{"finish_reason": "tool_calls", "message": {
            "role": "assistant", "content": None,
            "tool_calls": [{"id": "call_1", "type": "function",
                            "function": {"name": "Read",
                                         "arguments": '{"path":"/tmp/x"}'}}]}}],
        "usage": {"prompt_tokens": 1000, "completion_tokens": 20,
                  "prompt_tokens_details": {"cached_tokens": 900}},
    }, "claude-sonnet-5")
    assert message["stop_reason"] == "tool_use"
    block = message["content"][0]
    assert block["type"] == "tool_use"
    assert block["input"] == {"path": "/tmp/x"}
    assert message["usage"]["cache_read_input_tokens"] == 900
    assert message["usage"]["input_tokens"] == 100


def test_unparseable_tool_arguments_fail_loudly():
    with pytest.raises(TranslationError):
        openai_response_to_anthropic({
            "choices": [{"finish_reason": "tool_calls", "message": {
                "tool_calls": [{"id": "c1", "function": {
                    "name": "Read", "arguments": "{not json"}}]}}]},
            "m")


def test_empty_choices_fail_loudly():
    with pytest.raises(TranslationError):
        openai_response_to_anthropic({"choices": []}, "m")


def test_replayed_sse_has_full_event_sequence():
    message = openai_response_to_anthropic({
        "choices": [{"finish_reason": "stop",
                     "message": {"role": "assistant", "content": "hello"}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 2},
    }, "claude-sonnet-5")
    events = [e.split("\n")[0].replace("event: ", "")
              for e in anthropic_sse_from_message(message)]
    assert events == ["message_start", "content_block_start", "content_block_delta",
                      "content_block_stop", "message_delta", "message_stop"]


# --------------------------------------------------------------------------
# streaming translation
# --------------------------------------------------------------------------
async def _lines(chunks: list[dict]):
    for chunk in chunks:
        yield "data: " + json.dumps(chunk)
    yield "data: [DONE]"


def _parse(events: list[str]) -> list[dict]:
    out = []
    for raw in events:
        for line in raw.splitlines():
            if line.startswith("data: "):
                out.append(json.loads(line[6:]))
    return out


@pytest.mark.asyncio
async def test_stream_text_deltas():
    outcome = StreamOutcome()
    chunks = [
        {"choices": [{"delta": {"content": "Hel"}}]},
        {"choices": [{"delta": {"content": "lo"}}]},
        {"choices": [{"delta": {}, "finish_reason": "stop"}]},
        {"choices": [], "usage": {"prompt_tokens": 12, "completion_tokens": 2}},
    ]
    events = [e async for e in translate_stream(_lines(chunks), "claude-sonnet-5", outcome)]
    parsed = _parse(events)
    text = "".join(p["delta"]["text"] for p in parsed
                   if p.get("type") == "content_block_delta")
    assert text == "Hello"
    assert outcome.stop_reason == "end_turn"
    assert outcome.usage.output == 2
    assert parsed[-1]["type"] == "message_stop"


@pytest.mark.asyncio
async def test_stream_tool_call_becomes_input_json_delta():
    outcome = StreamOutcome()
    chunks = [
        {"choices": [{"delta": {"tool_calls": [
            {"index": 0, "id": "call_1", "function": {"name": "Read", "arguments": ""}}]}}]},
        {"choices": [{"delta": {"tool_calls": [
            {"index": 0, "function": {"arguments": '{"pa'}}]}}]},
        {"choices": [{"delta": {"tool_calls": [
            {"index": 0, "function": {"arguments": 'th":"/tmp/x"}'}}]}}]},
        {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
    ]
    events = [e async for e in translate_stream(_lines(chunks), "claude-sonnet-5", outcome)]
    parsed = _parse(events)
    starts = [p for p in parsed if p.get("type") == "content_block_start"]
    assert starts[0]["content_block"]["type"] == "tool_use"
    assert starts[0]["content_block"]["name"] == "Read"
    joined = "".join(p["delta"]["partial_json"] for p in parsed
                     if p.get("type") == "content_block_delta")
    assert json.loads(joined) == {"path": "/tmp/x"}
    assert outcome.stop_reason == "tool_use"


@pytest.mark.asyncio
async def test_stream_mixed_text_then_tool_uses_distinct_indices():
    outcome = StreamOutcome()
    chunks = [
        {"choices": [{"delta": {"content": "let me look"}}]},
        {"choices": [{"delta": {"tool_calls": [
            {"index": 0, "id": "c1", "function": {"name": "Read", "arguments": "{}"}}]}}]},
        {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
    ]
    events = [e async for e in translate_stream(_lines(chunks), "m", outcome)]
    parsed = _parse(events)
    indices = {p["index"] for p in parsed if p.get("type") == "content_block_start"}
    assert indices == {0, 1}


@pytest.mark.asyncio
async def test_stream_preserves_cache_usage_in_message_start():
    outcome = StreamOutcome()
    chunks = [
        {"choices": [], "usage": {"prompt_tokens": 5000, "completion_tokens": 0,
                                  "prompt_tokens_details": {"cached_tokens": 4800}}},
        {"choices": [{"delta": {"content": "ok"}, "finish_reason": "stop"}]},
    ]
    events = [e async for e in translate_stream(_lines(chunks), "m", outcome)]
    start = _parse(events)[0]
    assert start["message"]["usage"]["cache_read_input_tokens"] == 4800
    assert start["message"]["usage"]["input_tokens"] == 200


# --------------------------------------------------------------------------
# credential hygiene
# --------------------------------------------------------------------------
def test_redact_hides_the_oauth_token():
    out = redact({"Authorization": f"Bearer {FAKE_OAUTH}", "x-app": "cli"})
    assert "sk-ant" not in json.dumps(out)
    assert out["x-app"] == "cli"


def test_redact_covers_all_sensitive_header_names():
    out = redact({"x-api-key": FAKE_OAUTH, "Cookie": "a=b",
                  "proxy-authorization": "Basic xyz"})
    assert "sk-ant" not in json.dumps(out)
    assert "a=b" not in json.dumps(out)


def test_shim_logging_never_emits_the_token(caplog):
    log = logging.getLogger("auto_router.shim")
    with caplog.at_level(logging.INFO, logger="auto_router.shim"):
        log.info("headers=%s", redact({"authorization": f"Bearer {FAKE_OAUTH}"}))
    assert "sk-ant" not in caplog.text
