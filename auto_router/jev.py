"""Jev (TypeSafe System One) as the routing classifier and adequacy judge.

Jev answers typed questions with calibrated probabilities instead of text:

* ``classify``: category (choice), difficulty (score), needs tools / vision /
  long context (noul), is this a follow-up relying on the previous turn (noul),
  and how costly a wrong answer would be (score). One call, questions evaluated
  in parallel.
* ``judge``: "does this response fully and correctly address the request?"
  (noul). Used as a cheap failure signal for escalation when no test or tool
  result is available.

Only a compact summary of the conversation is sent (the truncated last user
message plus tool names and sizes), never the full transcript.

Privacy contract
----------------
``scrub()`` runs on every field before it leaves the process, and always before
truncation, so a secret cannot survive by sitting past the cut. It removes
private-key blocks, ``Authorization``-style strings, ``key=``/``token:``-style
assignments, credentials embedded in URLs, a list of well-known secret shapes
(see ``SECRET_SHAPES``) and the literal value of any environment variable whose
*name* looks credential-like.

This is best effort, not a guarantee. Arbitrary sensitive text with no
recognisable shape - a customer's name, a private document - is not detected
and must not be put in the request in the first place. ``AUTO_ROUTER_JEV_MAX_CHARS``
caps how much of any field is sent at all.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

ENDPOINT = os.environ.get("AUTO_ROUTER_JEV_URL", "https://api.typesafe.ai/v1/systemone")
MODEL = os.environ.get("AUTO_ROUTER_JEV_MODEL", "jev-latest")

#: How much of each field is sent at all. The documented budget is 64k tokens
#: for state plus every question, and 32k for state plus the longest question
#: (docs.typesafe.ai/models); these character caps stay far inside it, and the
#: jaggedness notes warn that accuracy drops in a large state full of
#: irrelevant detail. Lower them if your prompts carry sensitive material.
REQUEST_CHARS = int(os.environ.get("AUTO_ROUTER_JEV_REQUEST_CHARS", "6000"))
CONTEXT_CHARS = int(os.environ.get("AUTO_ROUTER_JEV_CONTEXT_CHARS", "2000"))
RESPONSE_CHARS = int(os.environ.get("AUTO_ROUTER_JEV_RESPONSE_CHARS", "8000"))

#: Bounded retry for the documented rate-limit and transient server statuses.
RETRY_STATUS = frozenset({429, 500, 502, 503, 504})
MAX_ATTEMPTS = int(os.environ.get("AUTO_ROUTER_JEV_ATTEMPTS", "3"))
BASE_BACKOFF_S = 0.25
MAX_BACKOFF_S = 5.0

CATEGORY_OPTIONS = {
    "coding": "Write, change, review or debug code in a single self-contained step.",
    "agentic": "Multi-step work in a codebase or system: explore files, run commands, iterate on test results.",
    "math": "A calculation, proof or quantitative puzzle with a definite answer.",
    "knowledge": "Factual, scientific or domain knowledge questions; explanations.",
    "long_context": "Find or synthesise information inside a long provided document or log.",
    "tool_use": "Operate external tools or APIs to reach a specific end state (not primarily coding).",
    "design": "Produce or critique a user-facing web or app interface: layout, styling, "
              "components, visual hierarchy, front-end markup and CSS.",
    "summarisation": "Condense or rewrite text that is already provided into a shorter form.",
    "general": "Chit-chat, open writing, and anything that fits no other option.",
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
    """What Jev thinks the task *is*. Deliberately separate from the route choice.

    Every field is a judgement about the request, never a decision: the routing
    policy in ``policies.py`` owns every decision and can be read and tested
    without Jev in the loop.
    """

    category: str
    category_probs: dict[str, float]
    difficulty: float                      # 0..1
    difficulty_confidence: float           # 0..1, derived from the score distribution
    needs_tools: float
    needs_vision: float
    needs_long_context: float
    follow_up: float
    stakes: float                          # 0..1
    #: 0..1, derived from the choice distribution. Noul answers carry none.
    category_confidence: float = 0.0
    latency_s: float = 0.0
    #: Versioned model id that actually answered, as reported by the API.
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    failed: bool = False
    raw: dict = field(default_factory=dict, repr=False)

    @property
    def source(self) -> str:
        return "fallback" if self.failed else "jev"

    def traits(self) -> dict:
        """Compact, prompt-free view of the classification for the explanation."""
        return {
            "source": self.source,
            "model": self.model or None,
            "category": self.category,
            "category_confidence": round(self.category_confidence, 3),
            "difficulty": round(self.difficulty, 3),
            "difficulty_confidence": round(self.difficulty_confidence, 3),
            "needs_tools": round(self.needs_tools, 3),
            "needs_vision": round(self.needs_vision, 3),
            "needs_long_context": round(self.needs_long_context, 3),
            "follow_up": round(self.follow_up, 3),
            "stakes": round(self.stakes, 3),
            "latency_ms": round(self.latency_s * 1000, 1),
        }


#: Returned whenever the classifier is unavailable or answers unusably. Middle
#: difficulty, middle stakes, no confidence: the policy sees "I do not know"
#: rather than a confident wrong answer.
FALLBACK = Classification(
    category="general", category_probs={}, difficulty=0.5, difficulty_confidence=0.0,
    needs_tools=0.5, needs_vision=0.0, needs_long_context=0.0, follow_up=0.5, stakes=0.5,
    failed=True)


#: Environment variable names whose *value* is removed wherever it appears.
SECRET_ENV_NAME = re.compile(r"(?i)(key|token|secret|password|passwd|credential|api[_-]?secret)")

#: Minimum length before an environment value is treated as a secret. Short
#: values ("1", "true", a two-letter region) would redact ordinary prose.
MIN_ENV_SECRET_LEN = 8

#: Well-known credential shapes. Best effort: a secret with no recognisable
#: shape is not detected, which is why the caller must not send one.
SECRET_SHAPES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("private-key-block", re.compile(
        r"-----BEGIN [^-]*PRIVATE KEY-----.*?(?:-----END [^-]*PRIVATE KEY-----|$)", re.S)),
    ("authorization-header", re.compile(r"(?i)\b(?:bearer|basic)\s+[A-Za-z0-9._~+/=-]{8,}")),
    ("anthropic-key", re.compile(r"sk-ant-[A-Za-z0-9._-]{16,}")),
    ("openai-key", re.compile(r"\bsk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{20,}")),
    ("github-token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{16,}")),
    ("github-fine-grained", re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}")),
    ("slack-token", re.compile(r"\bxox[abposr]-[A-Za-z0-9-]{10,}")),
    ("google-api-key", re.compile(r"\bAIza[A-Za-z0-9_-]{30,}")),
    ("aws-access-key-id", re.compile(r"\b(?:AKIA|ASIA|AGPA|AIDA|AROA)[0-9A-Z]{16}\b")),
    ("stripe-key", re.compile(r"\b[sr]k_(?:live|test)_[A-Za-z0-9]{16,}")),
    ("hugging-face-token", re.compile(r"\bhf_[A-Za-z0-9]{20,}")),
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}")),
    ("pem-certificate-body", re.compile(r"-----BEGIN CERTIFICATE-----.*?(?:-----END CERTIFICATE-----|$)", re.S)),
)

#: ``name = value`` / ``"name": value`` where the name looks credential-like.
#:
#: The match deliberately starts at the credential word rather than at the
#: start of the identifier: an unbounded ``[\w.-]*`` in front of a literal
#: backtracks quadratically over a long unbroken run of word characters, which
#: a large pasted prompt easily contains. Any prefix (``api_`` in ``api_key``)
#: is simply left in place, which is the same visible result.
_ASSIGNMENT = re.compile(
    r"(?i)((?:key|token|secret|password|passwd|credential)[\w.-]{0,48}"
    r"[\"\']?[ \t]*[:=][ \t]*)"
    r"(?!\[REDACTED\])(?:\"[^\"\n]*\"|\'[^\'\n]*\'|[^\s,;}\]]+)")

#: ``https://user:password@host``
_URL_CREDENTIALS = re.compile(r"(?i)\b([a-z][a-z0-9+.-]*)://[^\s/@]+:[^\s/@]+@")

REDACTED = "[REDACTED]"


def scrub(text: str, limit: int) -> str:
    """Remove recognisable credentials, then truncate to ``limit`` characters.

    Order matters: redaction runs on the *whole* string first, so a secret
    cannot escape by sitting past the truncation point, and a redaction marker
    may itself be cut in half (harmless - it carries no secret).
    """
    if not text:
        return ""
    # Assignments first: ``API_KEY=sk-...`` then reads as one redaction instead
    # of a shape redaction the assignment rule would chop a second time.
    text = _ASSIGNMENT.sub(r"\1" + REDACTED, text)
    for _name, pattern in SECRET_SHAPES:
        text = pattern.sub(REDACTED, text)
    text = _URL_CREDENTIALS.sub(r"\1://" + REDACTED + "@", text)
    for name, value in os.environ.items():
        if len(value) >= MIN_ENV_SECRET_LEN and SECRET_ENV_NAME.search(name):
            text = text.replace(value, REDACTED)
    return text[:limit]


def scrub_report(text: str) -> list[str]:
    """Which secret shapes were found. For tests and privacy evidence, not for logs."""
    found = [name for name, pattern in SECRET_SHAPES if pattern.search(text)]
    if _ASSIGNMENT.search(text):
        found.append("credential-assignment")
    if _URL_CREDENTIALS.search(text):
        found.append("url-credentials")
    for name, value in os.environ.items():
        if len(value) >= MIN_ENV_SECRET_LEN and SECRET_ENV_NAME.search(name) and value in text:
            found.append("environment-value")
            break
    return found


def _retry_after_seconds(exc: urllib.error.HTTPError, attempt: int) -> float:
    """Seconds to wait before retrying. Honours ``retry-after`` when the API sends it.

    TypeSafe documents 429 responses carrying ``retry-after`` (docs.typesafe.ai/models).
    Anything unparseable falls back to bounded exponential backoff.
    """
    header = ""
    try:
        header = (exc.headers.get("retry-after") or "").strip()
    except AttributeError:
        header = ""
    if header:
        try:
            return max(0.0, min(MAX_BACKOFF_S, float(header)))
        except ValueError:
            pass  # HTTP-date form: not worth parsing for a sub-second router hop
    return min(MAX_BACKOFF_S, BASE_BACKOFF_S * (2 ** attempt))


def _post(state: dict, questions: dict, api_key: str | None, timeout: float,
          *, attempts: int = MAX_ATTEMPTS, sleep=time.sleep) -> tuple[dict, float]:
    """One evaluation call. Retries 429 and 5xx within a bounded budget.

    Raises on final failure; callers turn that into a cautious fallback rather
    than letting a classifier outage break routing.
    """
    key = api_key or os.environ.get("TYPESAFE_API_KEY")
    if not key:
        raise RuntimeError("TYPESAFE_API_KEY is not set")
    body = json.dumps({"model": MODEL, "state": state, "questions": questions}).encode()
    started = time.time()
    last: Exception = RuntimeError("no attempt made")
    for attempt in range(max(1, attempts)):
        req = urllib.request.Request(ENDPOINT, data=body, headers={
            "Authorization": f"Bearer {key}", "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read()), time.time() - started
        except urllib.error.HTTPError as exc:
            last = exc
            if exc.code not in RETRY_STATUS or attempt == attempts - 1:
                raise
            sleep(_retry_after_seconds(exc, attempt))
    raise last


def _score01(answer: dict, levels: int) -> float:
    """Map a Score answer onto 0..1.

    The API returns a probability-weighted value across ``levels`` ordered
    criteria (docs.typesafe.ai/api). Jev's own jaggedness notes warn against
    reading an exact magnitude out of an interpolated score, so this is only
    ever used as a monotone signal and is then recalibrated against measured
    outcomes (``policy.jev_difficulty_calibration``).
    """
    score = answer.get("score")
    if isinstance(score, (int, float)) and levels > 1:
        return max(0.0, min(1.0, float(score) / (levels - 1)))
    return 0.5


def _confidence(answer: dict) -> float:
    """Choice/Score answers carry ``confidence``; Noul answers do not."""
    value = answer.get("confidence")
    return max(0.0, min(1.0, float(value))) if isinstance(value, (int, float)) else 0.0


def classify(request: str, context: str = "", *, api_key: str | None = None,
             timeout: float = 20.0) -> Classification:
    """Classify one user turn. Never raises: failures return a cautious default."""
    try:
        state = {"request": scrub(request, REQUEST_CHARS),
                 "context": scrub(context, CONTEXT_CHARS) or "(new conversation)"}
        payload, latency = _post(state, QUESTIONS, api_key, timeout)
        a = payload["answers"]
        usage = payload.get("usage") or {}
        return Classification(
            category=a["category"]["choice"],
            category_probs=a["category"].get("probabilities") or {},
            category_confidence=_confidence(a["category"]),
            difficulty=_score01(a["difficulty"], len(QUESTIONS["difficulty"]["criteria"])),
            difficulty_confidence=_confidence(a["difficulty"]),
            needs_tools=float(a["needs_tools"]["noul"]),
            needs_vision=float(a["needs_vision"]["noul"]),
            needs_long_context=float(a["needs_long_context"]["noul"]),
            follow_up=float(a["follow_up"]["noul"]),
            stakes=_score01(a["stakes"], len(QUESTIONS["stakes"]["criteria"])),
            latency_s=latency,
            model=str(payload.get("model") or MODEL),
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
        state = {"request": scrub(request, REQUEST_CHARS), "response": scrub(response, RESPONSE_CHARS)}
        payload, latency = _post(state, JUDGE_QUESTION, api_key, timeout)
        return Judgement(float(payload["answers"]["adequate"]["noul"]), latency)
    except (urllib.error.URLError, KeyError, TypeError, ValueError, RuntimeError, TimeoutError, OSError):
        return Judgement(0.5, 0.0, failed=True)
