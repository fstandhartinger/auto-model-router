"""OpenAI's Responses API behind the router's chat-completions interface.

Some OpenAI models refuse function tools on ``/v1/chat/completions`` while
reasoning is on. GPT-6 Luna answers a Claude Code turn there with HTTP 400
("Function tools with reasoning_effort are not supported ... use
/v1/responses"). Turning reasoning off would route coding turns to a weaker
model than the benchmark row the router priced, so the router speaks
``/v1/responses`` to such a provider instead (``api: responses``).

Everything else in the router keeps speaking chat completions. This module
converts in both directions and nothing more:

* ``chat_to_responses`` - a chat-completions request body as a Responses body.
* ``responses_to_chat`` - a Responses result as a chat-completions result, so
  usage parsing, truncation labels and translation to Anthropic are unchanged.
* ``chat_sse`` - a finished chat-completions result as the SSE lines a
  streaming call would have produced. Responses calls are made unstreamed and
  replayed; the client sees the same event shapes, just in one burst.

``store`` is always false: the router never asks a provider to keep a copy.
"""
from __future__ import annotations

import json
from typing import Any

#: Chat-completions fields with the same meaning on the Responses API.
_PASS = ("temperature", "top_p", "parallel_tool_calls", "user", "metadata", "service_tier")


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(p.get("text", "") for p in content
                       if isinstance(p, dict) and p.get("type") in ("text", "input_text", "output_text"))
    return ""


def _user_content(content: Any) -> Any:
    if not isinstance(content, list):
        return content if isinstance(content, str) else _text(content)
    parts = []
    for p in content:
        if not isinstance(p, dict):
            continue
        if p.get("type") == "text":
            parts.append({"type": "input_text", "text": p.get("text", "")})
        elif p.get("type") == "image_url":
            url = p.get("image_url")
            url = url.get("url") if isinstance(url, dict) else url
            if url:
                parts.append({"type": "input_image", "image_url": url})
    return parts


def chat_to_responses(payload: dict) -> dict:
    instructions: list[str] = []
    items: list[dict] = []
    for m in payload.get("messages") or []:
        role = m.get("role")
        if role in ("system", "developer"):
            instructions.append(_text(m.get("content")))
        elif role == "user":
            items.append({"role": "user", "content": _user_content(m.get("content"))})
        elif role == "assistant":
            text = _text(m.get("content"))
            if text:
                items.append({"role": "assistant", "content": text})
            for call in m.get("tool_calls") or []:
                fn = call.get("function") or {}
                items.append({"type": "function_call", "call_id": call.get("id"),
                              "name": fn.get("name"), "arguments": fn.get("arguments") or "{}"})
        elif role == "tool":
            items.append({"type": "function_call_output", "call_id": m.get("tool_call_id"),
                          "output": _text(m.get("content"))})
    out: dict[str, Any] = {"model": payload.get("model"), "input": items, "store": False}
    if instructions:
        out["instructions"] = "\n\n".join(i for i in instructions if i)
    tools = []
    for t in payload.get("tools") or []:
        fn = t.get("function") if t.get("type") == "function" else None
        if fn:
            tools.append({"type": "function", "name": fn.get("name"),
                          "description": fn.get("description") or "",
                          "parameters": fn.get("parameters") or {"type": "object", "properties": {}},
                          "strict": False})
    if tools:
        out["tools"] = tools
    choice = payload.get("tool_choice")
    if isinstance(choice, str):
        out["tool_choice"] = choice
    elif isinstance(choice, dict) and (choice.get("function") or {}).get("name"):
        out["tool_choice"] = {"type": "function", "name": choice["function"]["name"]}
    budget = payload.get("max_output_tokens") or payload.get("max_completion_tokens") or payload.get("max_tokens")
    if budget:
        out["max_output_tokens"] = budget
    if payload.get("reasoning_effort"):
        out["reasoning"] = {"effort": payload["reasoning_effort"]}
    for key in _PASS:
        if key in payload:
            out[key] = payload[key]
    return out


def responses_to_chat(data: dict) -> dict:
    text: list[str] = []
    calls: list[dict] = []
    for item in data.get("output") or []:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "message":
            text.extend(p.get("text", "") for p in item.get("content") or []
                        if isinstance(p, dict) and p.get("type") == "output_text")
        elif item.get("type") == "function_call":
            calls.append({"id": item.get("call_id") or item.get("id"), "type": "function",
                          "function": {"name": item.get("name"), "arguments": item.get("arguments") or "{}"}})
    reason = ((data.get("incomplete_details") or {}).get("reason") if data.get("status") == "incomplete" else None)
    if reason == "max_output_tokens":
        finish = "length"
    elif reason == "content_filter":
        finish = "content_filter"
    elif calls:
        finish = "tool_calls"
    else:
        finish = "stop"
    message: dict[str, Any] = {"role": "assistant", "content": "".join(text) or None}
    if calls:
        message["tool_calls"] = calls
    raw = data.get("usage") or {}
    usage = {"prompt_tokens": raw.get("input_tokens") or 0,
             "completion_tokens": raw.get("output_tokens") or 0,
             "total_tokens": raw.get("total_tokens") or 0,
             "prompt_tokens_details": {"cached_tokens": (raw.get("input_tokens_details") or {}).get("cached_tokens") or 0},
             "completion_tokens_details": {"reasoning_tokens":
                                           (raw.get("output_tokens_details") or {}).get("reasoning_tokens") or 0}}
    return {"id": data.get("id"), "object": "chat.completion", "created": data.get("created_at"),
            "model": data.get("model"),
            "choices": [{"index": 0, "message": message, "finish_reason": finish}],
            "usage": usage}


def chat_sse(chat: dict) -> list[str]:
    """``data:`` lines equivalent to streaming ``chat`` (usage chunk included)."""
    base = {"id": chat.get("id"), "object": "chat.completion.chunk", "created": chat.get("created"),
            "model": chat.get("model")}
    choice = (chat.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    chunks: list[dict] = [{**base, "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]}]
    if message.get("content"):
        chunks.append({**base, "choices": [{"index": 0, "delta": {"content": message["content"]},
                                            "finish_reason": None}]})
    for i, call in enumerate(message.get("tool_calls") or []):
        chunks.append({**base, "choices": [{"index": 0, "delta": {"tool_calls": [{**call, "index": i}]},
                                            "finish_reason": None}]})
    chunks.append({**base, "choices": [{"index": 0, "delta": {}, "finish_reason": choice.get("finish_reason")}]})
    chunks.append({**base, "choices": [], "usage": chat.get("usage")})
    return [f"data: {json.dumps(c)}" for c in chunks] + ["data: [DONE]"]
