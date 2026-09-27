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
import math
import os
import re
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

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

#: What "adequate" means for each kind of work. The generic wording above is
#: what the 17 Sep evaluation measured; these sharpen it per category so the
#: judge is told what a *coding* answer has to contain rather than being left
#: to infer it. The category is the router's own classification, so a request
#: it could not classify simply falls back to ``general``.
ADEQUACY_CRITERIA: dict[str, dict[str, str]] = {
    "coding": {
        "true": "The code is complete and runnable as given, implements what was asked, keeps the "
                "requested names and signatures, and handles the edge cases the request mentions.",
        "false": "Wrong output for some input, missing or stubbed parts, a different signature than "
                 "asked for, a described solution instead of the code, or an unfinished block.",
    },
    "math": {
        "true": "The final answer is correct and stated explicitly, and the steps shown actually "
                "support it.",
        "false": "The final answer is wrong or missing, a step contradicts the result, or the "
                 "response stops before reaching an answer.",
    },
    "knowledge": {
        "true": "Factually correct, answers the question that was asked, and states the parts it "
                "is unsure about.",
        "false": "Factually wrong, answers a neighbouring question, or invents a specific that the "
                 "request would need to be right.",
    },
    "summarisation": {
        "true": "Covers the source's main points, adds nothing that is not in it, and obeys the "
                "requested length and form.",
        "false": "Drops a main point, adds facts the source does not contain, or ignores the "
                 "requested length or form.",
    },
    "design": {
        "true": "Delivers the artefact that was asked for (markup, layout or critique), covers every "
                "element the request lists, and is self-contained where the request says so.",
        "false": "Missing elements the request lists, prose where an artefact was asked for, or "
                 "markup that would not render on its own.",
    },
    "tool_use": {
        "true": "Reaches the end state the request describes and names the calls it would make.",
        "false": "Stops before the end state, invents a tool, or describes intent instead of acting.",
    },
    "agentic": {
        "true": "Completes every step the request asks for and reports the result of each.",
        "false": "Leaves a step unfinished, reports a result it did not reach, or stops at a plan.",
    },
    "general": {
        "true": "Complete, correct, follows every instruction in the request.",
        "false": "Wrong, incomplete, evasive, truncated, or ignores part of the request.",
    },
}

#: Why an answer is inadequate. ``fine`` is deliberately one of the options:
#: forcing a failure type on an adequate answer would make the choice useless
#: as a label for the ones that really failed.
FAILURE_OPTIONS = {
    "fine": "The response answers the request; there is nothing substantial to fix.",
    "wrong": "The response is confidently incorrect: it answers, and the answer is not right.",
    "incomplete": "The response is on the right track but stops short: missing steps, missing "
                  "parts, or cut off.",
    "off_topic": "The response does not address what was asked, or answers a different question.",
}


def verify_questions(category: str = "general") -> dict:
    """The typed adequacy question pair for one category.

    Two questions in one call (the API evaluates them in parallel, so this
    costs one round trip): the adequacy Noul that the threshold is compared
    against, and a Choice naming the failure type, which is what a user
    interface can show and what makes a false escalation legible afterwards.
    """
    criteria = ADEQUACY_CRITERIA.get(category) or ADEQUACY_CRITERIA["general"]
    return {
        "adequate": {
            "type": "noul",
            "instructions": "Does `response` fully and correctly address `request`?",
            "criteria": dict(criteria),
        },
        "failure": {
            "type": "choice",
            "instructions": "If `response` falls short of `request`, what kind of failure is it?",
            "criteria": dict(FAILURE_OPTIONS),
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
    source_name: str = "jev"

    @property
    def source(self) -> str:
        return "fallback" if self.failed else self.source_name

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


class LocalLayaClassifier:
    """Lazy CPU-only Laya classifier with the same result shape as hosted Jev.

    The optional dependency is imported only on the first request.  The model
    weights are then fetched by Hugging Face once and cached in the user's
    normal model cache.  Import, download, and inference failures degrade to
    ``FALLBACK`` just like a hosted classifier outage does.
    """

    def __init__(self, model: str = "convaiinnovations/laya", threads: int = 4):
        self.model = model
        self.threads = max(1, int(threads))
        self._agent = None
        self._load_lock = threading.Lock()

    def _load(self):
        if self._agent is None:
            with self._load_lock:
                if self._agent is None:
                    import laya
                    import torch
                    torch.set_num_threads(self.threads)
                    self._agent = laya.load(self.model)
        return self._agent

    def __call__(self, request: str, context: str = "") -> Classification:
        started = time.perf_counter()
        try:
            state = {"request": scrub(request, REQUEST_CHARS),
                     "context": scrub(context, CONTEXT_CHARS) or "(new conversation)"}
            payload = self._load().predict(state, QUESTIONS)
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
                latency_s=time.perf_counter() - started,
                model=str(payload.get("model") or self.model),
                input_tokens=int(usage.get("input_tokens") or 0),
                output_tokens=int(usage.get("output_tokens") or 0),
                raw=a,
                source_name="local-laya",
            )
        except Exception:  # backend/model failures must not take down the routed LLM call
            return FALLBACK


#: Letters a local decision model answers with, in option order.
LETTERS = "ABCDEFGHIJKLMNOP"

#: System prompt of the open JevK5 v0.2 recipe (Apache-2.0,
#: huggingface.co/alibiserikbay/JevK5-GGUF): one option letter, nothing else.
LOCAL_SYSTEM = ("Apply the supplied criterion to the supplied evidence. Choose exactly one listed "
                "option. Respond with only its uppercase letter, with no explanation or reasoning.")

#: Softmax temperature over the option-letter log-probabilities. 1.532 is the
#: value the JevK5 v0.2 card publishes for its calibration; another model
#: should be given its own (``temperature`` in the config).
LOCAL_TEMPERATURE = 1.532

#: Name the local backend reports in decision records.
LOCAL_SOURCE = "local-jev-class"


def _options(question: dict) -> list[tuple[str, str]]:
    """(option id, description) pairs in the order the letters are assigned."""
    kind = question.get("type")
    criteria = question.get("criteria") or {}
    if kind == "noul":
        return [("true", str(criteria.get("true") or "The proposition is true.")),
                ("false", str(criteria.get("false") or "The proposition is false."))]
    if kind == "score":
        return [(str(i), str(text)) for i, text in enumerate(criteria)]
    return [(str(k), str(v)) for k, v in criteria.items()]


class LocalJevClass:
    """A local open Jev-class decision model behind an OpenAI-compatible endpoint.

    Serves the same typed questions as hosted Jev (``classify`` and ``judge``)
    from a model on the user's own machine: llama.cpp's ``llama-server`` or LM
    Studio serving, for example, JevK5 v0.2 GGUF (Apache-2.0) or decider-4b
    (Apache-2.0). Nothing leaves the machine and no key is needed.

    Each question is one call that asks for a single option letter and reads
    the letters' log-probabilities (the JevK5 recipe's one-pass readout);
    probabilities are a temperature softmax over them. A server that returns no
    log-probabilities still works, with the generated letter taken as certain,
    which is coarser and is reported as confidence 0.

    ``prompt_format``:

    * ``chatml`` (default) - ``POST /completions`` with the raw ChatML prompt
      and an empty thinking block, exactly as the recipe prompts. Right for
      Qwen-based decision models on llama.cpp and LM Studio.
    * ``chat`` - ``POST /chat/completions`` with ``logprobs``; for servers
      without a raw completions endpoint (Ollama's OpenAI layer, for example).
    """

    def __init__(self, base_url: str = "http://127.0.0.1:8080/v1", model: str = "jevk5",
                 *, api_key_env: str | None = None, temperature: float = LOCAL_TEMPERATURE,
                 prompt_format: str = "chatml", timeout: float = 30.0, top_logprobs: int = 20,
                 parallel: int = 4):
        if prompt_format not in ("chatml", "chat"):
            raise ValueError(f"unknown prompt_format {prompt_format!r}; use chatml or chat")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key_env = api_key_env
        self.temperature = max(1e-3, float(temperature))
        self.prompt_format = prompt_format
        self.timeout = timeout
        self.top_logprobs = max(2, int(top_logprobs))
        self.parallel = max(1, int(parallel))

    # -- transport -----------------------------------------------------------
    def _post(self, path: str, body: dict, timeout: float) -> dict:
        headers = {"Content-Type": "application/json"}
        key = os.environ.get(self.api_key_env) if self.api_key_env else None
        if key:
            headers["Authorization"] = f"Bearer {key}"
        req = urllib.request.Request(self.base_url + path, data=json.dumps(body).encode(),
                                     headers=headers)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())

    def _letter_logprobs(self, user: str, timeout: float) -> tuple[dict[str, float], str, str]:
        """Top log-probabilities of the first answer token, the text, and the model id."""
        if self.prompt_format == "chatml":
            prompt = (f"<|im_start|>system\n{LOCAL_SYSTEM}<|im_end|>\n<|im_start|>user\n{user}"
                      "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
            data = self._post("/completions", {
                "model": self.model, "prompt": prompt, "max_tokens": 1, "temperature": 0,
                "logprobs": self.top_logprobs, "cache_prompt": False}, timeout)
            choice = (data.get("choices") or [{}])[0]
            text = str(choice.get("text") or "")
        else:
            data = self._post("/chat/completions", {
                "model": self.model, "max_tokens": 1, "temperature": 0, "logprobs": True,
                "top_logprobs": self.top_logprobs,
                "chat_template_kwargs": {"enable_thinking": False},
                "messages": [{"role": "system", "content": LOCAL_SYSTEM},
                             {"role": "user", "content": user}]}, timeout)
            choice = (data.get("choices") or [{}])[0]
            text = str(((choice.get("message") or {}).get("content")) or "")
        return _top_logprobs(choice.get("logprobs")), text, str(data.get("model") or self.model)

    # -- one typed question ----------------------------------------------------
    def ask(self, state: dict, question: dict, timeout: float | None = None) -> tuple[dict, str]:
        """Answer one typed question in the hosted API's answer shape."""
        options = _options(question)
        if not 2 <= len(options) <= len(LETTERS):
            raise ValueError(f"a local decision model needs 2..{len(LETTERS)} options")
        user = json.dumps({
            "evidence": state, "criterion": question.get("instructions", ""),
            "options": [{"letter": LETTERS[i], "description": f"{oid}: {text}"}
                        for i, (oid, text) in enumerate(options)]})
        top, text, model = self._letter_logprobs(user, timeout or self.timeout)
        letters = LETTERS[:len(options)]
        confidence_known = any(letter in top for letter in letters)
        if confidence_known:
            floor = min(top.values()) - 2.0
            logits = [top.get(letter, floor) for letter in letters]
            peak = max(logits)
            weights = [math.exp((x - peak) / self.temperature) for x in logits]
        else:
            first = text.strip()[:1].upper()
            if first not in letters:
                raise ValueError("the local decision model answered with no option letter")
            weights = [1.0 if letter == first else 0.0 for letter in letters]
        total = sum(weights)
        probs = {oid: w / total for (oid, _), w in zip(options, weights)}
        best = max(probs, key=probs.get)
        confidence = probs[best] if confidence_known else 0.0
        kind = question.get("type")
        if kind == "noul":
            return {"noul": probs["true"]}, model
        if kind == "score":
            score = sum(int(oid) * p for oid, p in probs.items())
            return {"score": score, "confidence": confidence, "probabilities": probs}, model
        return {"choice": best, "probabilities": probs, "confidence": confidence}, model

    def evaluate(self, state: dict, questions: dict, timeout: float | None = None) -> dict:
        """All questions, in parallel up to ``parallel`` (llama-server ``--parallel``)."""
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=min(self.parallel, len(questions))) as pool:
            futures = {name: pool.submit(self.ask, state, q, timeout)
                       for name, q in questions.items()}
            results = {name: f.result() for name, f in futures.items()}
        model = next((m for _a, m in results.values()), self.model)
        return {"answers": {name: a for name, (a, _m) in results.items()}, "model": model}

    # -- the two jobs ----------------------------------------------------------
    def classify(self, request: str, context: str = "") -> Classification:
        started = time.perf_counter()
        try:
            state = {"request": scrub(request, REQUEST_CHARS),
                     "context": scrub(context, CONTEXT_CHARS) or "(new conversation)"}
            payload = self.evaluate(state, QUESTIONS)
            a = payload["answers"]
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
                latency_s=time.perf_counter() - started,
                model=str(payload.get("model") or self.model),
                raw=a,
                source_name=LOCAL_SOURCE,
            )
        except Exception:  # noqa: BLE001 - a local model outage must not break routing
            return FALLBACK

    __call__ = classify

    def judge(self, request: str, response: str, *, api_key: str | None = None,
              timeout: float | None = None, category: str = "") -> Judgement:
        """Same contract as the hosted ``judge``: failures return 0.5 with ``failed``."""
        started = time.perf_counter()
        try:
            state = {"request": scrub(request, REQUEST_CHARS),
                     "response": scrub(response, RESPONSE_CHARS)}
            questions = verify_questions(category) if category else JUDGE_QUESTION
            payload = self.evaluate(state, questions, timeout)
            answers = payload["answers"]
            failure_answer = answers.get("failure") or {}
            choice = failure_answer.get("choice")
            return Judgement(
                p_adequate=float(answers["adequate"]["noul"]),
                latency_s=time.perf_counter() - started,
                failure=str(choice) if choice in FAILURE_OPTIONS else "unknown",
                failure_probs={k: float(v) for k, v in
                               (failure_answer.get("probabilities") or {}).items()},
                model=f"{LOCAL_SOURCE}:{payload.get('model') or self.model}",
            )
        except Exception:  # noqa: BLE001 - same contract as the hosted judge
            return Judgement(0.5, time.perf_counter() - started, failed=True)


def _top_logprobs(logprobs: Any) -> dict[str, float]:
    """Letter -> log-probability of the first generated token, across response shapes.

    Chat completions carry ``content[0].top_logprobs`` as a list of
    ``{token, logprob}``; legacy completions carry ``top_logprobs[0]`` as a
    ``{token: logprob}`` map; llama.cpp's completions endpoint may use either.
    A token with surrounding whitespace counts as its letter.
    """
    out: dict[str, float] = {}

    def add(token: Any, value: Any) -> None:
        if not isinstance(token, str) or not isinstance(value, (int, float)):
            return
        key = token.strip()
        if key and math.isfinite(float(value)):
            out[key] = max(out.get(key, -math.inf), float(value))

    if not isinstance(logprobs, dict):
        return out
    content = logprobs.get("content")
    if isinstance(content, list) and content:
        for item in (content[0] or {}).get("top_logprobs") or []:
            if isinstance(item, dict):
                add(item.get("token"), item.get("logprob"))
    top = logprobs.get("top_logprobs")
    if isinstance(top, list) and top and isinstance(top[0], dict):
        first = top[0]
        if "token" in first and "logprob" in first:
            for item in top:
                if isinstance(item, dict):
                    add(item.get("token"), item.get("logprob"))
        else:
            for token, value in first.items():
                add(token, value)
    return out


def local_from_config(cfg: dict) -> LocalJevClass:
    """A ``LocalJevClass`` from a ``classifier`` or ``verify.judge`` config block."""
    return LocalJevClass(
        str(cfg.get("base_url") or "http://127.0.0.1:8080/v1"),
        str(cfg.get("model") or "jevk5"),
        api_key_env=cfg.get("api_key_env"),
        temperature=float(cfg.get("temperature") or LOCAL_TEMPERATURE),
        prompt_format=str(cfg.get("prompt_format") or "chatml"),
        timeout=float(cfg.get("timeout_s") or 30.0),
        top_logprobs=int(cfg.get("top_logprobs") or 20),
        parallel=int(cfg.get("parallel") or 4))


LOCAL_BACKENDS = {"local-jev", "jev-local", "local-jev-class", "openai-compatible"}


def classifier_from_config(policy: dict | None):
    """Build the selected classifier backend from ``policy.classifier``.

    ``local`` uses Laya on CPU, ``local-jev`` a local open Jev-class model
    behind an OpenAI-compatible endpoint (see ``LocalJevClass``),
    ``local-route-head`` the small ONNX routing model (see ``route_head.py``), ``hosted``
    uses the TypeSafe/Jev API with the user's own key, and ``heuristic``
    disables model inference. Returning ``None`` preserves the router's
    existing cautious heuristic path.
    """
    cfg = (policy or {}).get("classifier") or {}
    backend = str(cfg.get("backend") or "").lower()
    if not backend:
        return classify if os.environ.get("TYPESAFE_API_KEY") else None
    if backend == "local":
        return LocalLayaClassifier(str(cfg.get("model") or "convaiinnovations/laya"),
                                   int(cfg.get("threads") or 4))
    if backend in LOCAL_BACKENDS:
        return local_from_config(cfg)
    if backend in {"local-route-head", "route-head"}:
        from .route_head import DEFAULT_MODEL, LocalRouteHeadClassifier
        return LocalRouteHeadClassifier(str(cfg.get("model") or DEFAULT_MODEL),
                                        str(cfg.get("variant") or "fp16"), int(cfg.get("threads") or 2))
    if backend in {"hosted", "jev"}:
        return classify
    if backend in {"heuristic", "none", "disabled"}:
        return None
    raise ValueError(f"unknown classifier backend {backend!r}; "
                     "use local, local-jev, local-route-head, hosted, or heuristic")


def judge_from_config(policy: dict | None):
    """The answer judge from ``policy.verify.judge``, or None when there is none.

    ``hosted`` is Jev with the user's own ``TYPESAFE_API_KEY`` (and is None
    without one); ``local-jev`` a local open Jev-class model; ``same-as-classifier``
    reuses the ``local-jev`` classifier block; ``none`` switches judging off.
    Unset keeps the old default: hosted when the key is set.
    """
    verify = (policy or {}).get("verify") or {}
    cfg = verify.get("judge") or {}
    if isinstance(cfg, str):
        cfg = {"backend": cfg}
    backend = str(cfg.get("backend") or "").lower()
    if backend == "same-as-classifier":
        cls_cfg = (policy or {}).get("classifier") or {}
        if str(cls_cfg.get("backend") or "").lower() in LOCAL_BACKENDS:
            return local_from_config(cls_cfg).judge
        backend = str(cls_cfg.get("backend") or "").lower()
    if not backend or backend in {"hosted", "jev"}:
        return judge if os.environ.get("TYPESAFE_API_KEY") else None
    if backend in LOCAL_BACKENDS:
        return local_from_config(cfg).judge
    if backend in {"none", "disabled", "heuristic", "local"}:
        return None
    raise ValueError(f"unknown judge backend {backend!r}; use hosted, local-jev, or none")


@dataclass
class Judgement:
    """P(the answer is adequate), and what kind of failure it is if not.

    ``failure`` is ``"fine"`` whenever the judge sees nothing substantial to
    fix, and ``"unknown"`` when the failure question was not asked or could not
    be read - never a guess. A judge that could not answer at all returns
    ``p_adequate = 0.5`` with ``failed=True``, which is below no sensible
    threshold and above none either: an outage must not silently escalate every
    turn, and must not silently approve one.
    """

    p_adequate: float
    latency_s: float
    failed: bool = False
    failure: str = "unknown"
    failure_probs: dict[str, float] = field(default_factory=dict)
    #: Versioned model id that answered, as reported by the API.
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0


def judge(request: str, response: str, *, api_key: str | None = None, timeout: float = 20.0,
          category: str = "") -> Judgement:
    """P(response adequately answers request). On error returns 0.5 and failed=True.

    With ``category`` set, the adequacy criteria are the task-specific ones
    from ``ADEQUACY_CRITERIA`` and a second question names the failure type.
    Without it the question is the generic one the 17 Sep evaluation measured,
    so old calibrations stay comparable.
    """
    try:
        state = {"request": scrub(request, REQUEST_CHARS), "response": scrub(response, RESPONSE_CHARS)}
        questions = verify_questions(category) if category else JUDGE_QUESTION
        payload, latency = _post(state, questions, api_key, timeout)
        answers = payload["answers"]
        usage = payload.get("usage") or {}
        failure_answer = answers.get("failure") or {}
        choice = failure_answer.get("choice")
        return Judgement(
            p_adequate=float(answers["adequate"]["noul"]),
            latency_s=latency,
            failure=str(choice) if choice in FAILURE_OPTIONS else "unknown",
            failure_probs={k: float(v) for k, v in (failure_answer.get("probabilities") or {}).items()},
            model=str(payload.get("model") or MODEL),
            input_tokens=int(usage.get("input_tokens") or 0),
            output_tokens=int(usage.get("output_tokens") or 0),
        )
    except (urllib.error.URLError, KeyError, TypeError, ValueError, RuntimeError, TimeoutError, OSError):
        return Judgement(0.5, 0.0, failed=True)
