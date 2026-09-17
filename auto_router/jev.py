"""Jev (TypeSafe System One) as the routing classifier and adequacy judge.

Jev answers typed questions with calibrated probabilities instead of text:

* ``classify``: category (choice), difficulty (score), needs tools / vision /
  long context (noul), is this a follow-up relying on the previous turn (noul),
  and how costly a wrong answer would be (score). One call, questions evaluated
  in parallel.
* ``judge``: "does this response fully and correctly address the request?"
  (noul). Used as a cheap failure signal for escalation when no test or tool
  result is available.

Only a compact summary of the conversation is sent (head and tail of the last
user message, tool names, sizes), never the full transcript.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

ENDPOINT = os.environ.get("AUTO_ROUTER_JEV_URL", "https://api.typesafe.ai/v1/systemone")
MODEL = os.environ.get("AUTO_ROUTER_JEV_MODEL", "jev-latest")

CATEGORY_OPTIONS = {
    "coding": "Write, change, review or debug code in a single self-contained step.",
    "agentic": "Multi-step work in a codebase or system: explore files, run commands, iterate on test results.",
    "math": "A calculation, proof or quantitative puzzle with a definite answer.",
    "knowledge": "Factual, scientific or domain knowledge questions; explanations.",
    "long_context": "Find or synthesise information inside a long provided document or log.",
    "tool_use": "Operate external tools or APIs to reach a specific end state (not primarily coding).",
    "general": "Chit-chat, writing, rephrasing, summaries and anything else.",
}

QUESTIONS = {
    "category": {
        "type": "choice",
        "instructions": "What kind of work does the assistant need to do to handle `request`?",
        "criteria": CATEGORY_OPTIONS,
    },
    "difficulty": {
        "type": "score",
        "instructions": ("How much reasoning capability does a model need to handle `request` correctly on "
                         "the first attempt, given `context`?"),
        "criteria": [
            "Trivial: greeting, acknowledgement, one-sentence fact, obvious one-line edit",
            "Easy: routine single step, rename, lookup, small function, standard explanation",
            "Moderate: implement or debug one component with clear requirements, multi-step reasoning",
            "Hard: subtle bugs, concurrency, multi-file design, olympiad-style math, ambiguous requirements",
            "Frontier: research-grade problems where only the strongest models succeed",
        ],
    },
    "needs_tools": {
        "type": "noul",
        "instructions": "Will handling `request` require calling tools (running code, reading files, APIs)?",
        "criteria": {"true": "Tools or command execution are needed", "false": "A direct answer suffices"},
    },
    "needs_vision": {
        "type": "noul",
        "instructions": "Does `request` include or refer to an image the assistant must look at?",
        "criteria": {"true": "An image must be inspected", "false": "Text only"},
    },
    "needs_long_context": {
        "type": "noul",
        "instructions": "Does handling `request` require reading more than about 50 pages of provided text?",
        "criteria": {"true": "Very long input must be read", "false": "Short or moderate input"},
    },
    "follow_up": {
        "type": "noul",
        "instructions": ("Is `request` a follow-up that relies on the previous turn's work or decisions "
                         "described in `context`?"),
        "criteria": {"true": "Builds on or refers to earlier turns", "false": "Self-contained"},
    },
    "stakes": {
        "type": "score",
        "instructions": "How costly would it be if the answer to `request` were subtly wrong and nobody noticed?",
        "criteria": [
            "Negligible: chit-chat or throwaway output",
            "Low: easy to spot and redo",
            "Medium: wastes a work session or needs a later fix",
            "High: breaks production, loses data or money, or misleads a decision",
        ],
    },
}

JUDGE_QUESTION = {
    "adequate": {
        "type": "noul",
        "instructions": "Does `response` fully and correctly address `request`?",
        "criteria": {
            "true": "Complete, correct, follows every instruction in the request",
            "false": "Wrong, incomplete, evasive, truncated, or ignores part of the request",
        },
    },
}


@dataclass
class Classification:
    category: str
    category_probs: dict[str, float]
    difficulty: float          # 0..1
    difficulty_confidence: float
    needs_tools: float
    needs_vision: float
    needs_long_context: float
    follow_up: float
    stakes: float              # 0..1
    latency_s: float
    input_tokens: int = 0
    output_tokens: int = 0
    failed: bool = False
    raw: dict = field(default_factory=dict, repr=False)


FALLBACK = Classification("general", {}, 0.5, 0.0, 0.5, 0.0, 0.0, 0.5, 0.5, 0.0, failed=True)


def _post(state: dict, questions: dict, api_key: str | None, timeout: float) -> tuple[dict, float]:
    key = api_key or os.environ.get("TYPESAFE_API_KEY")
    if not key:
        raise RuntimeError("TYPESAFE_API_KEY is not set")
    body = json.dumps({"model": MODEL, "state": state, "questions": questions}).encode()
    req = urllib.request.Request(ENDPOINT, data=body, headers={
        "Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    started = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = json.loads(resp.read())
    return payload, time.time() - started


def _score01(answer: dict, levels: int) -> float:
    score = answer.get("score")
    if isinstance(score, (int, float)):
        return max(0.0, min(1.0, float(score) / (levels - 1)))
    return 0.5


def classify(request: str, context: str = "", *, api_key: str | None = None,
             timeout: float = 20.0) -> Classification:
    """Classify one user turn. Never raises: failures return a cautious default."""
    try:
        payload, latency = _post({"request": request[:6000], "context": context[:2000] or "(new conversation)"},
                                 QUESTIONS, api_key, timeout)
        a = payload["answers"]
        usage = payload.get("usage") or {}
        return Classification(
            category=a["category"]["choice"],
            category_probs=a["category"].get("probabilities") or {},
            difficulty=_score01(a["difficulty"], len(QUESTIONS["difficulty"]["criteria"])),
            difficulty_confidence=float(a["difficulty"].get("confidence") or 0.0),
            needs_tools=float(a["needs_tools"]["noul"]),
            needs_vision=float(a["needs_vision"]["noul"]),
            needs_long_context=float(a["needs_long_context"]["noul"]),
            follow_up=float(a["follow_up"]["noul"]),
            stakes=_score01(a["stakes"], len(QUESTIONS["stakes"]["criteria"])),
            latency_s=latency,
            input_tokens=int(usage.get("input_tokens") or 0),
            output_tokens=int(usage.get("output_tokens") or 0),
            raw=a,
        )
    except (urllib.error.URLError, KeyError, TypeError, ValueError, RuntimeError, TimeoutError, OSError):
        return FALLBACK


@dataclass
class Judgement:
    p_adequate: float
    latency_s: float
    failed: bool = False


def judge(request: str, response: str, *, api_key: str | None = None, timeout: float = 20.0) -> Judgement:
    """P(response adequately answers request). On error returns 0.5 and failed=True."""
    try:
        payload, latency = _post({"request": request[:6000], "response": response[:8000]},
                                 JUDGE_QUESTION, api_key, timeout)
        return Judgement(float(payload["answers"]["adequate"]["noul"]), latency)
    except (urllib.error.URLError, KeyError, TypeError, ValueError, RuntimeError, TimeoutError, OSError):
        return Judgement(0.5, 0.0, failed=True)
