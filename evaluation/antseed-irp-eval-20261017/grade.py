"""Deterministic graders for tasks-public.jsonl. Pure functions of (task, answer text); no model is ever called.

Coding runs the answer against the dataset's own tests in the repo's bubblewrap sandbox (experiments/sandbox.py,
nonce-verified verdict, see experiments/graders.py). Everything else compares a stated final answer with the key.
"""
import json
import re
import sys
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

#: The dataset tests need numpy only for np.allclose / ndarray; the sandbox interpreter has no site-packages.
NUMPY_SHIM = '''
import sys, types, math
np = types.ModuleType("numpy")
class _Arr(list):
    dtype = None
np.ndarray = _Arr
np.float64 = np.float32 = "float"
def _flat(x):
    if isinstance(x, (list, tuple, set)):
        for i in x:
            yield from _flat(i)
    else:
        yield x
def _allclose(a, b, rtol=1e-05, atol=1e-08):
    fa, fb = list(_flat(a)), list(_flat(b))
    return len(fa) == len(fb) and all(math.isclose(x, y, rel_tol=rtol, abs_tol=atol) for x, y in zip(fa, fb))
np.allclose = _allclose
sys.modules["numpy"] = np
'''


def grade_code(task, answer):
    from experiments import graders
    return graders.grade_coding({"hidden_tests": NUMPY_SHIM + task["check"], "timeout_s": 60}, answer)


def _num(text):
    """Parse '12', '-3.5', '3/4', '\\frac{3}{4}', '$1,200$', '12 euro' to a Fraction, else None."""
    s = (text or "").strip().strip("*`$ ").replace("\\dfrac", "\\frac").replace("\\!", "").replace(",", "")
    s = re.sub(r"\\text\{[^}]*\}|\\%|%", "", s).strip().rstrip(".")
    m = re.fullmatch(r"(-?)\\frac\{(\d+)\}\{(\d+)\}", s) or re.fullmatch(r"(-?)(\d+)\s*/\s*(\d+)", s)
    if m and int(m.group(3)):
        return Fraction(int(m.group(2)), int(m.group(3))) * (-1 if m.group(1) else 1)
    m = re.match(r"^\s*(-?\d+(?:\.\d+)?)(?:\s*[A-Za-z€$°]{0,12}\.?)?$", s)
    return Fraction(m.group(1)) if m else None


FINAL = re.compile(r"final answer\s*[:=]?\s*(.+)", re.I)


def final_line(answer):
    hits = FINAL.findall(answer or "")
    return hits[-1].strip().strip("*` ").strip() if hits else None


def grade_number(task, answer):
    got = final_line(answer)
    if got is None:
        return False, "no 'Final answer:' line"
    a, b = _num(got), _num(str(task["expected"]))
    if a is None or b is None:
        return False, f"unparsable: {got!r}"
    tol = Fraction(str(task.get("tol", 0)))
    return (abs(a - b) <= tol), f"got {got!r}, expected {task['expected']!r}"


def grade_string(task, answer):
    got = final_line(answer)
    if got is None:
        return False, "no 'Final answer:' line"
    norm = lambda x: re.sub(r"\s+", " ", x.strip().strip("*`'\".").strip()).lower()
    return norm(got) == norm(str(task["expected"])), f"got {got!r}, expected {task['expected']!r}"


def grade_letter(task, answer):
    hits = re.findall(r"answer\s*[:=]?\s*\**\(?([A-J])\b", answer or "", re.I)
    if not hits:
        return False, "no 'Answer: <letter>' line"
    return hits[-1].upper() == task["expected"], f"got {hits[-1].upper()}, expected {task['expected']}"


def _json_candidates(answer):
    fenced = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", answer or "", re.S)
    if fenced:
        yield fenced[-1]
    text, depth, start = answer or "", 0, None
    found = []
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth:
            depth -= 1
            if depth == 0:
                found.append(text[start:i + 1])
    if found:
        yield found[-1]


def _same(a, b):
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(a - b) < 0.005
    return a == b


def grade_json(task, answer):
    for cand in _json_candidates(answer):
        try:
            got = json.loads(cand)
        except ValueError:
            continue
        return _same(got, task["expected"]), f"got {cand[:120]!r}"
    return False, "no JSON object"


GRADERS = {"code": grade_code, "number": grade_number, "string": grade_string, "letter": grade_letter,
           "json": grade_json}


def grade(task, answer, finish_reason="stop"):
    if finish_reason == "length":
        return False, "truncated (finish_reason=length)"
    if not (answer or "").strip():
        return False, "empty answer"
    return GRADERS[task["grader"]](task, answer)
