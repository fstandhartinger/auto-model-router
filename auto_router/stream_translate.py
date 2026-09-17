"""Streaming translation: OpenAI chat-completion SSE -> Anthropic Messages SSE.

Handles text and tool calls incrementally so Claude Code gets real
time-to-first-token instead of a buffered replay. OpenAI streams tool-call
arguments as string fragments under delta.tool_calls[i].function.arguments;
Anthropic wants those as input_json_delta inside a tool_use content block. The
index spaces are different (OpenAI indexes tool calls, Anthropic indexes all
content blocks), so they are mapped explicitly.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from typing import AsyncIterator

from .pricing import Usage
from .translate import STOP_REASON_MAP, TranslationError, _sse


@dataclass
class _ToolAccumulator:
    block_index: int
    call_id: str
    name: str = ""
    started: bool = False


@dataclass
class StreamOutcome:
    usage: Usage = field(default_factory=Usage)
    stop_reason: str = "end_turn"
    saw_content: bool = False


async def translate_stream(
    lines: AsyncIterator[str],
    model_name: str,
    outcome: StreamOutcome,
) -> AsyncIterator[str]:
    message_id = f"msg_{uuid.uuid4().hex[:24]}"
    next_index = 0
    text_block: int | None = None
    tools: dict[int, _ToolAccumulator] = {}
    started_message = False
    finish_reason: str | None = None

    def start_message(usage: Usage) -> str:
        return _sse("message_start", {
            "type": "message_start",
            "message": {
                "id": message_id, "type": "message", "role": "assistant",
                "content": [], "model": model_name,
                "stop_reason": None, "stop_sequence": None,
                "usage": {
                    "input_tokens": usage.uncached_input,
                    "cache_read_input_tokens": usage.cached_read,
                    "cache_creation_input_tokens": usage.cache_write,
                    "output_tokens": 0,
                },
            },
        })

    async for line in lines:
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if not payload or payload == "[DONE]":
            continue
        try:
            chunk = json.loads(payload)
        except json.JSONDecodeError:
            continue

        if chunk.get("usage"):
            from .pricing import parse_openai_usage
            outcome.usage = parse_openai_usage(chunk["usage"])

        choices = chunk.get("choices") or []
        if not choices:
            continue
        choice = choices[0]
        delta = choice.get("delta") or {}
        if choice.get("finish_reason"):
            finish_reason = choice["finish_reason"]

        if not started_message:
            started_message = True
            yield start_message(outcome.usage)

        content = delta.get("content")
        if content:
            outcome.saw_content = True
            if text_block is None:
                text_block = next_index
                next_index += 1
                yield _sse("content_block_start", {
                    "type": "content_block_start", "index": text_block,
                    "content_block": {"type": "text", "text": ""},
                })
            yield _sse("content_block_delta", {
                "type": "content_block_delta", "index": text_block,
                "delta": {"type": "text_delta", "text": content},
            })

        for call in delta.get("tool_calls") or []:
            outcome.saw_content = True
            oi = int(call.get("index", 0))
            acc = tools.get(oi)
            if acc is None:
                acc = _ToolAccumulator(
                    block_index=next_index,
                    call_id=call.get("id") or f"toolu_{uuid.uuid4().hex[:24]}",
                )
                next_index += 1
                tools[oi] = acc
            fn = call.get("function") or {}
            if fn.get("name"):
                acc.name = fn["name"]
            if not acc.started and acc.name:
                acc.started = True
                yield _sse("content_block_start", {
                    "type": "content_block_start", "index": acc.block_index,
                    "content_block": {"type": "tool_use", "id": acc.call_id,
                                      "name": acc.name, "input": {}},
                })
            args = fn.get("arguments")
            if args:
                if not acc.started:
                    raise TranslationError(
                        "upstream streamed tool arguments before a tool name; "
                        "cannot build a valid tool_use block"
                    )
                yield _sse("content_block_delta", {
                    "type": "content_block_delta", "index": acc.block_index,
                    "delta": {"type": "input_json_delta", "partial_json": args},
                })

    if not started_message:
        yield start_message(outcome.usage)

    if text_block is not None:
        yield _sse("content_block_stop", {"type": "content_block_stop", "index": text_block})
    for acc in tools.values():
        if acc.started:
            yield _sse("content_block_stop",
                       {"type": "content_block_stop", "index": acc.block_index})

    outcome.stop_reason = ("tool_use" if tools
                           else STOP_REASON_MAP.get(finish_reason or "stop", "end_turn"))
    yield _sse("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": outcome.stop_reason, "stop_sequence": None},
        "usage": {"output_tokens": outcome.usage.output},
    })
    yield _sse("message_stop", {"type": "message_stop"})
