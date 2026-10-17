"""Deterministic graders for wave 3 (identical to grade_w2.py including its amendment A2). Wave-1 graders (number / string / letter / json) are reused unchanged from ../grade.py;
this file adds the LiveCodeBench grader ("lcb"): stdin/stdout programs and LeetCode-style classes are run against the
dataset's own tests (at most 30 per problem: all public ones plus evenly spaced private ones) in the repo's bubblewrap
sandbox with the same nonce-verified verdict as wave 1 (experiments/sandbox_runner.py). No model is ever called.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import grade as g1  # noqa: E402  (also puts the repo root on sys.path)
from experiments import graders, sandbox  # noqa: E402

# "from builtins import *" last: `from math import *` would otherwise shadow the builtin pow() (3-argument pow broke every answer to
# coding-lcb6-leetcode-3739; found after registration, see README amendment A2)
IMPORTS = ("from typing import *\nfrom collections import *\nfrom itertools import *\nfrom functools import *\nfrom heapq import *\n"
           "from bisect import *\nfrom math import *\nfrom builtins import *\nimport math, sys, re, string, heapq, bisect, itertools, functools, collections\n")

STDIO_CHECK = '''import io, json, sys
TESTS = json.loads(%r)

def run_one(inp):
    real_in, real_out = sys.stdin, sys.stdout
    sys.stdin = io.TextIOWrapper(io.BytesIO(inp.encode()), encoding="utf-8")
    sink = io.BytesIO()
    sys.stdout = io.TextIOWrapper(sink, encoding="utf-8", write_through=True)
    try:
        try:
            exec(compile(SRC, "submission.py", "exec"), {"__name__": "__main__"})
        except SystemExit:
            pass
        sys.stdout.flush()
        return sink.getvalue().decode("utf-8", "replace")
    finally:
        sys.stdin, sys.stdout = real_in, real_out

def same(got, want):
    a, b = got.split(), want.split()
    if a == b:
        return True
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        if x != y:
            try:
                if abs(float(x) - float(y)) > 1e-6 * max(1.0, abs(float(y))):
                    return False
            except ValueError:
                return False
    return True

for k, t in enumerate(TESTS):
    got = run_one(t["input"])
    assert same(got, t["output"]), "test %%d: got %%r want %%r" %% (k, got[:80], t["output"][:80])
'''

FUNC_CHECK = '''import json
TESTS = json.loads(%r)
FUNC = %r

def norm(x):
    return json.loads(json.dumps(x))

def same(a, b):
    if isinstance(a, float) or isinstance(b, float):
        return isinstance(a, (int, float)) and isinstance(b, (int, float)) and abs(a - b) <= 1e-5 * max(1.0, abs(b))
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return a == b

for k, t in enumerate(TESTS):
    args = [json.loads(l) for l in t["input"].split("\\n") if l.strip()]
    got = norm(getattr(Solution(), FUNC)(*args))
    assert same(got, json.loads(t["output"])), "test %%d" %% k
'''



def grade_lcb(task, answer):
    code = graders.extract_code(answer)
    if not code:
        return False, "no code in the answer"
    tests = json.dumps(task["tests"], sort_keys=True)
    if task["mode"] == "stdin":
        solution, checks = "SRC = " + repr(code) + "\n", STDIO_CHECK % tests
    else:
        solution, checks = IMPORTS + code + "\n", FUNC_CHECK % (tests, task["func_name"])
    import secrets
    nonce = secrets.token_hex(16)
    result = sandbox.run_python(graders.RUNNER.read_text(), stdin=nonce + "\n", extra_files={"solution.py": solution, "checks.py": checks},
                                limits=sandbox.Limits(wall_seconds=70, cpu_seconds=60))
    if result.unavailable:
        raise RuntimeError("SANDBOX UNAVAILABLE: " + result.unavailable)
    if result.timed_out:
        return False, "timed out in the sandbox"
    if f"VERDICT {nonce}" not in result.stdout:
        reason = {10: "no nonce", 11: "raised while loading", 12: "a test failed", 13: "a test raised"}.get(result.returncode, f"exit {result.returncode}")
        return False, f"{reason}: {(result.stderr or result.stdout or '').strip()[-200:]}"
    return True, "all tests passed (verified by run nonce)"


GRADERS = {**g1.GRADERS, "lcb": grade_lcb}


def grade(task, answer, finish_reason="stop"):
    if finish_reason == "length":
        return False, "truncated (finish_reason=length)"
    if not (answer or "").strip():
        return False, "empty answer"
    return GRADERS[task["grader"]](task, answer)
