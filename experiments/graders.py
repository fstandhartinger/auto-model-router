"""Deterministic graders for the held-out evaluation.

Every grader is a pure function of (task, answer) and returns
``(passed, detail)``. No grader calls a model, so a run can be regraded later
from the stored answers and will produce exactly the same verdicts.

Honesty about what each one measures
------------------------------------
``coding``   executes the answer against hidden tests in the Bubblewrap
             sandbox. This is a real pass/fail.
``math``     compares a normalised final answer against the known value. Real
             pass/fail.
``research`` checks that the required factual token appears and that a named
             common confusion does not. Real pass/fail on a narrow question.
``design``   a **structural proxy**, not a judgement of whether the design is
             good. It checks that the answer is real, self-contained markup
             that satisfies the requirements the prompt actually stated
             (a media query, semantic landmarks, a labelled control, no
             external dependency). A model can satisfy the rubric and still
             produce something ugly; the rubric cannot be satisfied by prose
             that merely describes a page.
``summary``  three checks a summary must pass to be usable: it is inside the
             requested length, it retains the facts the prompt asked for, and
             it invents no number that is absent from the source. It does not
             measure elegance.

Wherever a grader is a proxy it says so in ``detail``, and the report carries
that label through, so a per-category number is never read as more than it is.
"""

from __future__ import annotations

import math
import re
from typing import Any

from . import sandbox


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------
CODE_FENCE = re.compile(r"```(?:[a-zA-Z0-9_+-]*)\n(.*?)```", re.S)


def extract_code(text: str, language: str = "python") -> str:
    """Longest fenced block, else the whole answer."""
    blocks = CODE_FENCE.findall(text or "")
    if not blocks:
        return (text or "").strip()
    typed = re.findall(r"```(?:%s)\n(.*?)```" % language, text or "", re.S)
    pool = typed or blocks
    return max(pool, key=len).strip()


def normalise_answer(value: str) -> str:
    value = (value or "").strip().lower()
    value = re.sub(r"[\s,]+", "", value)
    value = re.sub(r"^[\$\(\[]+|[\)\]\.\!]+$", "", value)
    return value


FINAL = re.compile(r"(?:final answer|answer)\s*[:=]?\s*\**\s*([^\n]{1,80})", re.I)


def final_answer(text: str) -> str | None:
    matches = FINAL.findall(text or "")
    if matches:
        return matches[-1].strip().strip("*` ")
    numbers = re.findall(r"-?\d+(?:\.\d+)?", text or "")
    return numbers[-1] if numbers else None


def _close(a: str, b: str, atol: float) -> bool:
    try:
        return math.isclose(float(a), float(b), abs_tol=atol)
    except ValueError:
        return False


# ---------------------------------------------------------------------------
# graders
# ---------------------------------------------------------------------------
def grade_coding(task: dict, answer: str) -> tuple[bool, str]:
    """Execute the answer against hidden tests inside the sandbox."""
    code = extract_code(answer)
    if not code:
        return False, "no code in the answer"
    harness = task["hidden_tests"]
    result = sandbox.run_python(code + "\n\n" + harness,
                                limits=sandbox.Limits(wall_seconds=task.get("timeout_s", 20)))
    if result.unavailable:
        return False, f"SANDBOX UNAVAILABLE: {result.unavailable}"
    if result.timed_out:
        return False, "timed out in the sandbox"
    if not result.ok:
        return False, f"exit {result.returncode}: {(result.stderr or '').strip()[-200:]}"
    passed = "ALL_TESTS_PASSED" in result.stdout
    return passed, "hidden tests passed" if passed else f"output: {result.stdout.strip()[-200:]}"


NUMBER_IN = re.compile(r"-?\d+(?:\.\d+)?")


def grade_math(task: dict, answer: str) -> tuple[bool, str]:
    """Compare the model's own stated final answer with the known value.

    Only the final-answer line is searched, never the whole response, so a
    model cannot pass by mentioning the right number somewhere in its working.
    A trailing unit ("68 euro") is tolerated; a different number is not.
    """
    got = final_answer(answer)
    if got is None:
        return False, "no final answer found"
    expected = str(task["expected"])
    atol = task.get("atol", 0.0)
    if normalise_answer(got) == normalise_answer(expected):
        return True, f"exact match ({got})"
    for candidate in NUMBER_IN.findall(got)[:3]:
        if candidate == expected or _close(candidate, expected, atol):
            return True, f"matched {candidate} in final answer {got!r}"
    return False, f"got {got!r}, expected {expected!r}"


def grade_research(task: dict, answer: str) -> tuple[bool, str]:
    text = (answer or "").lower()
    missing = [t for t in task["must_contain"] if t.lower() not in text]
    wrong = [t for t in task.get("must_not_contain", []) if t.lower() in text]
    if missing:
        return False, f"missing: {missing}"
    if wrong:
        return False, f"contains a known confusion: {wrong}"
    return True, "required facts present, no known confusion"


def grade_design(task: dict, answer: str) -> tuple[bool, str]:
    """Structural proxy: did the answer produce markup meeting the stated rules?"""
    code = extract_code(answer, "html")
    low = code.lower()
    checks: dict[str, bool] = {
        "is markup, not prose": "<html" in low or "<!doctype" in low or "<section" in low,
        "self-contained (no external stylesheet or script)":
            not re.search(r'<(?:link[^>]+href|script[^>]+src)=["\']https?://', low),
    }
    for rule in task["rules"]:
        checks[rule["label"]] = bool(re.search(rule["pattern"], low, re.S))
    failed = [name for name, ok in checks.items() if not ok]
    detail = ("STRUCTURAL PROXY - all rules met" if not failed
              else f"STRUCTURAL PROXY - failed: {failed}")
    return not failed, detail


NUMBER = re.compile(r"\b\d+(?:[.,]\d+)?\b")


def grade_summary(task: dict, answer: str) -> tuple[bool, str]:
    text = (answer or "").strip()
    words = len(text.split())
    problems = []
    if not (task["min_words"] <= words <= task["max_words"]):
        problems.append(f"length {words} outside {task['min_words']}-{task['max_words']}")
    missing = [k for k in task["must_retain"] if k.lower() not in text.lower()]
    if missing:
        problems.append(f"dropped required facts: {missing}")
    source_numbers = {n.replace(",", "") for n in NUMBER.findall(task["source"])}
    invented = sorted({n.replace(",", "") for n in NUMBER.findall(text)} - source_numbers)
    # A summary may restate a year or a count from the source; anything else is
    # a number the model made up, which is the failure mode that matters.
    if invented:
        problems.append(f"numbers not in the source: {invented}")
    return not problems, "; ".join(problems) or "length, retention and no invented numbers all pass"


def grade_cache_repeat(task: dict, answer: str) -> tuple[bool, str]:
    """A repeat task is still graded on its answer; cache behaviour is measured separately."""
    return grade_research(task, answer)


GRADERS = {
    "coding": grade_coding,
    "math": grade_math,
    "research": grade_research,
    "design": grade_design,
    "summarisation": grade_summary,
    "cache_repeat": grade_cache_repeat,
}

#: Which graders are a real pass/fail and which are an explicit proxy.
GRADER_KIND = {
    "coding": "executed",
    "math": "exact",
    "research": "exact",
    "design": "structural-proxy",
    "summarisation": "rubric",
    "cache_repeat": "exact",
}


def grade(task: dict, answer: str) -> tuple[bool, str]:
    return GRADERS[task["grader"]](task, answer)
