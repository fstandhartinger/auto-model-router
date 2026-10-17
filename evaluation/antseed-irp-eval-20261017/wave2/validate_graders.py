"""Self-test of the wave-2 graders. Non-coding tasks: an answer built from the key must pass and a wrong one must fail.
Coding tasks: a stub must fail on every task (so the harness does not pass empty answers), and hand-written reference
solutions for REF_TASKS must pass (this proves both harness modes, stdin and class, end to end). LiveCodeBench ships no
reference solutions, so the hidden tests of the other coding tasks are trusted as published; this is stated in the README.
usage: validate_graders.py TASKS_FULL.jsonl"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import grade_w2 as grade

T = [json.loads(l) for l in open(sys.argv[1])]
REF = {
    "coding-lcb6-atcoder-abc392_a": None, "coding-lcb6-atcoder-abc400_a": None, "coding-lcb6-leetcode-3736": None,
}
bad = 0
for t in T:
    if t["grader"] == "lcb":
        stub = "```python\ndef nothing(): pass\n```" if t["mode"] == "stdin" else "```python\nclass Solution:\n    pass\n```"
        ok, d = grade.grade(t, stub)
        if ok:
            print("BAD stub passes", t["id"]); bad += 1
        continue
    e = t["expected"]
    ans = {"json": "```json\n" + json.dumps(e) + "\n```", "letter": "blah\nAnswer: " + str(e)}.get(t["grader"], "x\nFinal answer: **" + str(e) + "**")
    neg = {"json": '{"a":1}', "letter": "Answer: Z", "number": "Final answer: 123456789", "string": "Final answer: nope"}[t["grader"]]
    ok, d = grade.grade(t, ans)
    ok2, _ = grade.grade(t, neg)
    if not ok or ok2:
        print("BAD", t["id"], ok, ok2, d); bad += 1
refs = json.load(open(Path(__file__).resolve().parent / "reference-solutions.json"))
byid = {t["id"]: t for t in T}
for tid, src in refs.items():
    ok, d = grade.grade(byid[tid], "```python\n" + src + "\n```")
    print("reference", tid, ok, "" if ok else d)
    bad += 0 if ok else 1
    ok2, _ = grade.grade(byid[tid], "```python\n" + src + "\nprint(0)\n```" if byid[tid]["mode"] == "stdin" else "```python\nclass Solution:\n    pass\n```")
    if ok2:
        print("BAD mutated reference passes", tid); bad += 1
print("bad", bad)
sys.exit(1 if bad else 0)
