"""Anthropic Messages <-> OpenAI Chat Completions translation.

Three details matter here and differ from most Anthropic-compatible proxies:

  * usage is carried through the streaming path. Many proxies emit
    output_tokens: 0 in message_delta, which makes measured cost comparisons
    impossible.
  * tool_use is streamed as proper content_block_start + input_json_delta so
    Claude Code's incremental JSON parser sees the shape it expects.
  * translation failures raise TranslationError instead of degrading to an
    empty message. A silently dropped tool call looks like a model that simply
    chose not to act, which is the worst possible failure mode in an agent loop.
"""

from __future__ import annotations

import json
import uuid
from typing import Any, Iterator

STOP_REASON_MAP = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
    "content_filter": "end_turn",
    "function_call": "tool_use",
}


class TranslationError(RuntimeError):
    """Raised when a request or response cannot be faithfully translated."""


def extract_system_text(system: Any) -> str | None:
    if system is None:
        return None
    if isinstance(system, str):
        return system or None
    if isinstance(system, list):
        parts = [b.get("text") for b in system
                 if isinstance(b, dict) and isinstance(b.get("text"), str)]
        return "\n".join(parts) or None
    raise TranslationError(f"unsupported system block type: {type(system).__name__}")


def tools_to_openai(tools: Any) -> list[dict] | None:
    if not tools:
        return None
    out: list[dict] = []
    for tool in tools:
        if not isinstance(tool, dict):
            raise TranslationError(f"tool entry is {type(tool).__name__}, expected object")
        name = tool.get("name")
        if not name:
            # Server-side tools (web_search etc.) arrive without a name and have
            # no OpenAI equivalent. Dropping them silently would change model
            # behaviour, so refuse.
            raise TranslationError(f"tool without a name cannot be translated: {tool.get('type')}")
        fn: dict[str, Any] = {"name": name}
        if tool.get("description"):
            fn["description"] = tool["description"]
        schema = tool.get("input_schema") or tool.get("parameters")
        if schema:
            fn["parameters"] = schema
        out.append({"type": "function", "function": fn})
    return out or None


def tool_choice_to_openai(choice: Any) -> Any:
    if choice is None:
        return None
    if isinstance(choice, str):
        return {"auto": "auto", "any": "required", "none": "none"}.get(choice.lower())
    if isinstance(choice, dict):
        ctype = choice.get("type")
        if ctype == "tool" and choice.get("name"):
            return {"type": "function", "function": {"name": choice["name"]}}
        return {"auto": "auto", "any": "required", "none": "none", "tool": "required"}.get(ctype)
    return None


def _coerce_tool_result(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text") or "")
            elif isinstance(block, dict):
                parts.append(json.dumps(block))
            else:
                parts.append(str(block))
        return "\n".join(parts)
    return json.dumps(content)


def messages_to_openai(messages: list[dict], system: Any = None) -> list[dict]:
    out: list[dict] = []
    system_text = extract_system_text(system)
    if system_text:
        out.append({"role": "system", "content": system_text})

    for message in messages:
        role = message.get("role")
        content = message.get("content")

        if isinstance(content, str):
            out.append({"role": role, "content": content})
            continue
        if not isinstance(content, list):
            raise TranslationError(f"message content is {type(content).__name__}")

        parts: list[dict] = []
        tool_calls: list[dict] = []
        tool_results: list[dict] = []

        for block in content:
            if not isinstance(block, dict):
                raise TranslationError(f"content block is {type(block).__name__}")
            btype = block.get("type")
            if btype == "text":
                if block.get("text"):
                    parts.append({"type": "text", "text": block["text"]})
            elif btype == "thinking":
                # Thinking blocks have no OpenAI representation. They are
                # replayed context, not instructions, so dropping them is safe
                # and is what every Anthropic-compatible proxy does.
                continue
            elif btype == "image":
                source = block.get("source") or {}
                stype = source.get("type")
                if stype == "base64":
                    media = source.get("media_type") or "image/png"
                    data = source.get("data")
                    if not isinstance(data, str) or not data.strip():
                        raise TranslationError("base64 image block has no data")
                    parts.append({"type": "image_url",
                                  "image_url": {"url": f"data:{media};base64,{data}"}})
                elif stype == "url":
                    parts.append({"type": "image_url", "image_url": {"url": source["url"]}})
                else:
                    raise TranslationError(f"unsupported image source type: {stype}")
            elif btype == "tool_use":
                tool_calls.append({
                    "id": block.get("id") or f"call_{uuid.uuid4().hex[:24]}",
                    "type": "function",
                    "function": {
                        "name": block.get("name") or "",
                        "arguments": json.dumps(block.get("input") or {}),
                    },
                })
            elif btype == "tool_result":
                tool_use_id = block.get("tool_use_id")
                if not tool_use_id:
                    raise TranslationError("tool_result without tool_use_id")
                tool_results.append({
                    "role": "tool",
                    "tool_call_id": tool_use_id,
                    "content": _coerce_tool_result(block.get("content")),
                })
            else:
                raise TranslationError(f"unsupported content block type: {btype!r}")

        has_image = any(p.get("type") == "image_url" for p in parts)
        text = "\n".join(p["text"] for p in parts if p.get("type") == "text").strip()
        body: Any = parts if has_image else text

        if role == "assistant":
            if text or has_image or tool_calls:
                m: dict[str, Any] = {"role": "assistant", "content": body or None}
                if tool_calls:
                    m["tool_calls"] = tool_calls
                out.append(m)
        else:
            # tool results must precede the user text that follows them
            out.extend(tool_results)
            if text or has_image:
                out.append({"role": "user", "content": body})

    if not out:
        raise TranslationError("translation produced no messages")
    return out


def openai_response_to_anthropic(data: dict, model_name: str) -> dict:
    choices = data.get("choices") or []
    if not choices:
        raise TranslationError("upstream response contained no choices")
    message = choices[0].get("message") or {}
    finish = choices[0].get("finish_reason")
    tool_calls = message.get("tool_calls") or []

    blocks: list[dict] = []
    text = message.get("content")
    if isinstance(text, str) and text:
        blocks.append({"type": "text", "text": text})
    for call in tool_calls:
        fn = call.get("function") or {}
        raw = fn.get("arguments")
        if isinstance(raw, str):
            try:
                args = json.loads(raw) if raw.strip() else {}
            except json.JSONDecodeError as exc:
                raise TranslationError(
                    f"tool call {fn.get('name')!r} had unparseable arguments: {exc}"
                ) from exc
        elif isinstance(raw, dict):
            args = raw
        else:
            args = {}
        blocks.append({
            "type": "tool_use",
            "id": call.get("id") or f"toolu_{uuid.uuid4().hex[:24]}",
            "name": fn.get("name") or "",
            "input": args,
        })
    if not blocks:
        blocks.append({"type": "text", "text": ""})

    usage = data.get("usage") or {}
    details = usage.get("prompt_tokens_details") or {}
    cached = int(details.get("cached_tokens") or 0)
    written = int(details.get("cache_write_tokens") or 0)
    prompt = int(usage.get("prompt_tokens") or 0)

    return {
        "id": data.get("id") or f"msg_{uuid.uuid4().hex[:24]}",
        "type": "message",
        "role": "assistant",
        "content": blocks,
        "model": model_name,
        "stop_reason": "tool_use" if tool_calls else STOP_REASON_MAP.get(finish, "end_turn"),
        "stop_sequence": None,
        "usage": {
            "input_tokens": max(0, prompt - cached - written),
            "cache_read_input_tokens": cached,
            "cache_creation_input_tokens": written,
            "output_tokens": int(usage.get("completion_tokens") or 0),
        },
    }


def _sse(event: str, payload: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(payload)}\n\n"


def anthropic_sse_from_message(message: dict) -> Iterator[str]:
    """Replay a complete Anthropic message as a well-formed SSE stream.

    Used when the upstream call had to be non-streaming (tool calls) but the
    client asked for a stream. Claude Code accepts this because the event
    sequence and the usage figures are the real ones.
    """
    usage = message.get("usage") or {}
    yield _sse("message_start", {
        "type": "message_start",
        "message": {
            "id": message.get("id"),
            "type": "message",
            "role": "assistant",
            "content": [],
            "model": message.get("model"),
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {
                "input_tokens": usage.get("input_tokens", 0),
                "cache_read_input_tokens": usage.get("cache_read_input_tokens", 0),
                "cache_creation_input_tokens": usage.get("cache_creation_input_tokens", 0),
                "output_tokens": 0,
            },
        },
    })

    for index, block in enumerate(message.get("content") or []):
        btype = block.get("type")
        if btype == "text":
            yield _sse("content_block_start", {
                "type": "content_block_start", "index": index,
                "content_block": {"type": "text", "text": ""},
            })
            if block.get("text"):
                yield _sse("content_block_delta", {
                    "type": "content_block_delta", "index": index,
                    "delta": {"type": "text_delta", "text": block["text"]},
                })
        elif btype == "tool_use":
            yield _sse("content_block_start", {
                "type": "content_block_start", "index": index,
                "content_block": {"type": "tool_use", "id": block.get("id"),
                                  "name": block.get("name"), "input": {}},
            })
            yield _sse("content_block_delta", {
                "type": "content_block_delta", "index": index,
                "delta": {"type": "input_json_delta",
                          "partial_json": json.dumps(block.get("input") or {})},
            })
        else:
            raise TranslationError(f"cannot stream content block of type {btype!r}")
        yield _sse("content_block_stop", {"type": "content_block_stop", "index": index})

    yield _sse("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": message.get("stop_reason"),
                  "stop_sequence": message.get("stop_sequence")},
        "usage": {"output_tokens": usage.get("output_tokens", 0)},
    })
    yield _sse("message_stop", {"type": "message_stop"})


def anthropic_error_sse(message: str, error_type: str = "api_error") -> str:
    """Surface a failure inside an already-open stream.

    Claude Code shows this to the user. Ending the stream cleanly instead would
    render as an empty assistant turn, hiding the fault.
    """
    return _sse("error", {"type": "error",
                          "error": {"type": error_type, "message": message}})
