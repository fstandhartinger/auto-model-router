"""Graded task set: build (from public datasets + generators) and grade.

Categories x difficulty:
  coding        easy: HumanEval+   medium: CodeContests 1300-1700   hard: CodeContests >= 2100
  math          easy: GSM8K        medium: MATH-500 level 5          hard: AIME 2025
  tool_use      generated order-management world (toolenv.py), easy / medium / hard
  long_context  generated haystacks: single needle ~12k / two-hop ~40k / aggregation ~80k tokens
  agentic       tool loop in a scratch dir: fix a seeded bug (HumanEval+) / solve CodeContests medium / hard

Datasets are downloaded at build time and not redistributed with the repo.

    python experiments/tasks.py build --out experiments/tasks/tasks.jsonl --per-cell 6
"""

from __future__ import annotations

import argparse
import ast
import gzip
import json
import os
import random
import re
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import toolenv  # noqa: E402

CACHE = Path(os.environ.get("AUTO_ROUTER_DATA", Path.home() / ".cache" / "auto-model-router" / "datasets"))
HF_ROWS = "https://datasets-server.huggingface.co/rows?dataset={ds}&config={cfg}&split={split}&offset={off}&length={n}"
HUMANEVAL_PLUS = "https://github.com/evalplus/humanevalplus_release/releases/download/v0.1.10/HumanEvalPlus.jsonl.gz"


# ---------------------------------------------------------------------------
# download helpers
# ---------------------------------------------------------------------------
def fetch(url: str, name: str) -> Path:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / name
    if not path.exists():
        req = urllib.request.Request(url, headers={"User-Agent": "auto-model-router-experiments"})
        with urllib.request.urlopen(req, timeout=300) as resp:
            path.write_bytes(resp.read())
    return path


def hf_rows(ds: str, cfg: str, split: str, total: int) -> list[dict]:
    rows: list[dict] = []
    for off in range(0, total, 100):
        name = f"{ds.replace('/', '__')}-{cfg}-{split}-{off}.json"
        path = fetch(HF_ROWS.format(ds=ds, cfg=cfg, split=split, off=off, n=100), name)
        rows += [r["row"] for r in json.loads(path.read_text())["rows"]]
    return rows


# ---------------------------------------------------------------------------
# sandboxed execution
# ---------------------------------------------------------------------------
def run_python(code: str, stdin: str = "", timeout: float = 10.0, cwd: str | None = None) -> tuple[int, str, str]:
    """Run untrusted code in a subprocess without network (user+net namespace when available)."""
    with tempfile.TemporaryDirectory() as tmp:
        script = Path(cwd or tmp) / "_run.py"
        script.write_text(code)
        cmd = [sys.executable, str(script)]
        if _unshare_ok():
            cmd = ["unshare", "-rn"] + cmd
        try:
            p = subprocess.run(cmd, input=stdin, capture_output=True, text=True, timeout=timeout,
                               cwd=cwd or tmp, preexec_fn=_limits)
            return p.returncode, p.stdout, p.stderr[-2000:]
        except subprocess.TimeoutExpired:
            return -9, "", "timeout"


def _limits():
    import resource
    resource.setrlimit(resource.RLIMIT_AS, (2 << 30, 2 << 30))


_UNSHARE: bool | None = None


def _unshare_ok() -> bool:
    global _UNSHARE
    if _UNSHARE is None:
        try:
            _UNSHARE = subprocess.run(["unshare", "-rn", "true"], capture_output=True, timeout=5).returncode == 0
        except (OSError, subprocess.TimeoutExpired):
            _UNSHARE = False
    return _UNSHARE


def extract_code(text: str) -> str:
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, re.S)
    if blocks:
        return max(blocks, key=len)
    return text


# ---------------------------------------------------------------------------
# coding
# ---------------------------------------------------------------------------
def humaneval_cases(row: dict, limit: int = 60) -> list[list]:
    inputs = (row["base_input"] + row["plus_input"])[:limit]
    return inputs


def humaneval_expected(row: dict, inputs: list[list]) -> list | None:
    code = row["prompt"] + row["canonical_solution"] + "\nimport json,sys\n" + \
        f"inputs = json.loads(sys.stdin.read())\nprint(json.dumps([repr({row['entry_point']}(*a)) for a in inputs]))\n"
    rc, out, _ = run_python(code, json.dumps(inputs), timeout=30)
    if rc != 0:
        return None
    return json.loads(out)


def grade_humaneval(task: dict, text: str) -> tuple[bool, str]:
    code = extract_code(text)
    harness = code + "\nimport json,sys\n" + \
        f"inputs = json.loads(sys.stdin.read())\nout=[]\nfor a in inputs:\n" \
        f"    try:\n        out.append(repr({task['entry_point']}(*a)))\n    except Exception as e:\n" \
        f"        out.append('EXC')\nprint(json.dumps(out))\n"
    rc, out, err = run_python(harness, json.dumps(task["inputs"]), timeout=30)
    if rc != 0:
        return False, f"crash: {err[-200:]}"
    try:
        got = json.loads(out.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        return False, "no output"
    bad = sum(1 for g, e in zip(got, task["expected"]) if g != e and not _close(g, e, task.get("atol", 0)))
    return bad == 0, f"{bad} of {len(got)} cases wrong"


def _close(a: str, b: str, atol: float) -> bool:
    try:
        return abs(float(a) - float(b)) <= max(atol, 1e-6)
    except ValueError:
        return False


def grade_stdio(task: dict, code: str, tests: list[dict] | None = None) -> tuple[bool, str]:
    tests = tests if tests is not None else task["hidden_tests"]
    for i, t in enumerate(tests):
        rc, out, err = run_python(code, t["input"], timeout=task.get("time_limit", 10))
        if rc != 0:
            return False, f"test {i}: {'timeout' if rc == -9 else 'runtime error'}"
        if out.split() != t["output"].split():
            return False, f"test {i}: wrong answer"
    return True, f"{len(tests)} tests passed"


def build_contests(rows: list[dict], lo: int, hi: int, n: int, rng: random.Random, hidden: int = 25) -> list[dict]:
    pool = [r for r in rows if r["cf_rating"] and lo <= r["cf_rating"] <= hi
            and 3 in r["solutions"]["language"] and len(r["description"]) < 6000]
    rng.shuffle(pool)
    out = []
    for r in pool:
        tests = [{"input": i, "output": o} for i, o in zip(r["public_tests"]["input"], r["public_tests"]["output"])]
        gen = [{"input": i, "output": o} for i, o in zip(r["generated_tests"]["input"], r["generated_tests"]["output"])]
        private = [{"input": i, "output": o} for i, o in zip(r["private_tests"]["input"], r["private_tests"]["output"])]
        hidden_tests = (private + gen)[:hidden]
        refs = [s for s, lang in zip(r["solutions"]["solution"], r["solutions"]["language"]) if lang == 3]
        task = {"time_limit": 10, "hidden_tests": tests + hidden_tests}
        # keep only problems whose checker is exact-match: a reference solution must pass
        if not any(grade_stdio(task, ref)[0] for ref in refs[:3]):
            continue
        out.append({"name": r["name"], "rating": r["cf_rating"], "description": r["description"],
                    "public_tests": tests, "hidden_tests": tests + hidden_tests, "time_limit": 10})
        if len(out) >= n:
            break
    return out


# ---------------------------------------------------------------------------
# math
# ---------------------------------------------------------------------------
def normalise_answer(s: str) -> str:
    s = s.strip().strip("$").strip()
    s = re.sub(r"\\(left|right|!|,|;|:)", "", s)
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac").replace("^\\circ", "").replace("^{\\circ}", "")
    s = s.replace("{,}", "")
    s = re.sub(r"\\(?:text|mbox|mathrm)\{\s*[A-Za-z ]+\}\s*$", "", s)  # trailing units: 100\text{ pounds}
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    s = s.replace(" ", "").rstrip(".").replace("\\%", "").replace("%", "").replace("\\$", "").replace("$", "")
    s = re.sub(r"(?<=\d),(?=\d{3}\b)", "", s)
    if re.fullmatch(r"-?\d+\.0+", s):  # 20.00 -> 20
        s = s.split(".")[0]
    return s


def final_answer(text: str) -> str | None:
    m = re.findall(r"\\boxed\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}", text)
    if m:
        return m[-1]
    m = re.findall(r"(?i)final answer\s*[:：]?\s*(.+)", text)
    if m:
        return m[-1].strip()
    return None


def grade_math(task: dict, text: str) -> tuple[bool, str]:
    ans = final_answer(text)
    if ans is None:
        return False, "no final answer"
    ok = normalise_answer(ans) == normalise_answer(task["answer"])
    return ok, f"got {ans[:40]!r} want {task['answer']!r}"


# ---------------------------------------------------------------------------
# long context
# ---------------------------------------------------------------------------
CITIES = ["Porto", "Tallinn", "Graz", "Ghent", "Lyon", "Turin", "Bergen", "Brno", "Cork", "Split", "Malmo", "Bonn"]
DEPTS = ["billing", "logistics", "research", "legal", "support", "design", "security", "finance"]
WORDS = ("quarterly review notes mention that the rollout slipped by a week because the vendor changed the "
         "invoice format again and nobody updated the parser before the holiday freeze").split()


def filler_line(rng: random.Random, i: int) -> str:
    return (f"[{i:05d}] Employee E-{rng.randint(1000, 9999)} in {rng.choice(DEPTS)} ({rng.choice(CITIES)}): "
            + " ".join(rng.choice(WORDS) for _ in range(rng.randint(10, 22))) + ".")


def build_long_context(level: str, seed: int) -> dict:
    rng = random.Random(seed)
    target_tokens = {"easy": 12_000, "medium": 40_000, "hard": 80_000}[level]
    lines, tokens = [], 0
    while tokens < target_tokens:
        line = filler_line(rng, len(lines))
        lines.append(line)
        tokens += len(line) // 4
    if level == "easy":
        code = str(rng.randint(10000, 99999))
        vault = rng.choice(["Aster", "Birch", "Cedar", "Dahlia", "Elm"])
        lines.insert(rng.randrange(len(lines)), f"NOTE: the access code for vault {vault} was changed to {code}.")
        for other in {"Aster", "Birch", "Cedar", "Dahlia", "Elm"} - {vault}:
            lines.insert(rng.randrange(len(lines)), f"NOTE: vault {other} is scheduled for inspection next month.")
        question = f"What is the current access code for vault {vault}? Answer with the number only."
        answer = code
    elif level == "medium":
        project = rng.choice(["Falcon", "Heron", "Kestrel", "Osprey"])
        lead = f"E-{rng.randint(1000, 9999)}"
        room = str(rng.randint(100, 999))
        decoy = f"E-{rng.randint(1000, 9999)}"
        third = len(lines) // 3
        lines.insert(rng.randrange(0, third), f"ORG: Project {project} is led by employee {lead}; {decoy} is deputy.")
        lines.insert(rng.randrange(2 * third, len(lines)), f"DIRECTORY: {lead} sits in room {room}.")
        lines.insert(rng.randrange(third, 2 * third), f"DIRECTORY: {decoy} sits in room {rng.randint(100, 999)}.")
        lines.insert(rng.randrange(len(lines)), f"DIRECTORY (outdated, 2019): {lead} sat in room {rng.randint(100, 999)}.")
        question = (f"In which room does the current lead of Project {project} sit? Ignore outdated entries. "
                    f"Answer with the room number only.")
        answer = room
    else:
        customer = f"C-{rng.randint(10, 99)}"
        total = 0
        for _ in range(rng.randint(9, 14)):
            amount = rng.randint(20, 900)
            month = rng.choice(["2026-03", "2026-03", "2026-04", "2026-02"])
            cat = rng.choice(["travel", "travel", "office", "meals"])
            if month == "2026-03" and cat == "travel":
                total += amount
            line = f"TXN {month}-{rng.randint(10, 28)} customer={customer} category={cat} amount_usd={amount}"
            lines.insert(rng.randrange(len(lines)), line)
        for _ in range(30):
            other = f"C-{rng.randint(10, 99)}"
            if other == customer:
                continue
            lines.insert(rng.randrange(len(lines)), f"TXN 2026-03-{rng.randint(10, 28)} customer={other} "
                                                     f"category=travel amount_usd={rng.randint(20, 900)}")
        lines.insert(rng.randrange(len(lines)), f"VOID: all transactions of {customer} dated 2026-03-99 are void.")
        question = (f"What is the total amount_usd of all transactions of customer {customer} in March 2026 "
                    f"(dates 2026-03-*) with category=travel? Answer with the integer only.")
        answer = str(total)
    doc = "\n".join(lines)
    prompt = f"<document>\n{doc}\n</document>\n\n{question}\nPut the final answer in \\boxed{{}}."
    return {"prompt": prompt, "answer": answer}


# ---------------------------------------------------------------------------
# agentic coding
# ---------------------------------------------------------------------------
AGENT_TOOLS = [
    {"type": "function", "function": {"name": "list_files", "description": "List files in the workspace.",
     "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "read_file", "description": "Read a file.",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "write_file", "description": "Create or overwrite a file.",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                    "required": ["path", "content"]}}},
    {"type": "function", "function": {"name": "run_tests", "description": "Run the visible tests; returns the report.",
     "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "done", "description": "Finish. Call when the tests pass.",
     "parameters": {"type": "object", "properties": {}}}},
]

MUTATIONS = [(ast.Lt, ast.LtE), (ast.LtE, ast.Lt), (ast.Gt, ast.GtE), (ast.GtE, ast.Gt), (ast.Add, ast.Sub),
             (ast.Sub, ast.Add), (ast.Eq, ast.NotEq), (ast.And, ast.Or)]


def mutate(source: str, rng: random.Random) -> str | None:
    tree = ast.parse(source)
    sites = []
    for node in ast.walk(tree):
        for attr in ("op", "ops"):
            val = getattr(node, attr, None)
            ops = val if isinstance(val, list) else [val] if val is not None else []
            for i, op in enumerate(ops):
                for a, b in MUTATIONS:
                    if type(op) is a:
                        sites.append((node, attr, i, b))
        if isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(node.value, bool) \
                and node.value in (0, 1, 2):
            sites.append((node, "value", None, None))
    if not sites:
        return None
    node, attr, i, b = rng.choice(sites)
    if attr == "value":
        node.value = node.value + 1
    elif attr == "ops":
        node.ops[i] = b()
    else:
        node.op = b()
    return ast.unparse(tree)


def build_bugfix(row: dict, rng: random.Random) -> dict | None:
    full = row["prompt"] + row["canonical_solution"]
    inputs = humaneval_cases(row)
    expected = humaneval_expected(row, inputs)
    if expected is None:
        return None
    for _ in range(20):
        mutant = mutate(full, rng)
        if not mutant:
            return None
        task = {"entry_point": row["entry_point"], "inputs": row["base_input"],
                "expected": expected[:len(row["base_input"])]}
        visible_ok, _ = grade_humaneval(task, mutant)
        hidden_ok, _ = grade_humaneval({"entry_point": row["entry_point"], "inputs": inputs, "expected": expected},
                                       mutant)
        if not visible_ok and not hidden_ok:
            return {"files": {"solution.py": mutant},
                    "visible": {"entry_point": row["entry_point"], "inputs": row["base_input"],
                                "expected": expected[:len(row["base_input"])]},
                    "hidden": {"entry_point": row["entry_point"], "inputs": inputs, "expected": expected},
                    "instruction": (f"`solution.py` contains `{row['entry_point']}` which has a bug: some tests "
                                    f"fail. Find and fix the bug without changing the function's signature. Use "
                                    f"run_tests to check, then call done.")}
    return None


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------
SYSTEM_CODE = "You are an expert programmer. Reply with one complete Python code block."
SYSTEM_MATH = "Solve the problem. Show brief reasoning, then give the final answer in \\boxed{}."


def build(out: Path, per_cell: int, seed: int) -> None:
    rng = random.Random(seed)
    tasks: list[dict] = []

    he_rows = [json.loads(line) for line in gzip.open(fetch(HUMANEVAL_PLUS, "HumanEvalPlus.jsonl.gz"), "rt")]
    rng.shuffle(he_rows)
    count = 0
    for row in he_rows:
        inputs = humaneval_cases(row)
        expected = humaneval_expected(row, inputs)
        if expected is None:
            continue
        tasks.append({"id": f"coding-easy-{row['task_id']}", "category": "coding", "difficulty": "easy",
                      "grader": "humaneval", "system": SYSTEM_CODE,
                      "prompt": "Complete this function. Return the full function (with imports) in a code block.\n\n"
                                + row["prompt"],
                      "entry_point": row["entry_point"], "inputs": inputs, "expected": expected,
                      "atol": row.get("atol") or 0})
        count += 1
        if count >= per_cell:
            break

    cc_rows = hf_rows("deepmind/code_contests", "default", "test", 165)
    medium = build_contests(cc_rows, 1300, 1700, per_cell + 4, rng)
    hard = build_contests(cc_rows, 2100, 3500, per_cell + 4, rng)
    for level, probs in (("medium", medium[:per_cell]), ("hard", hard[:per_cell])):
        for p in probs:
            tasks.append({"id": f"coding-{level}-{p['name'][:12]}", "category": "coding", "difficulty": level,
                          "grader": "stdio", "system": SYSTEM_CODE, "rating": p["rating"],
                          "prompt": "Solve this competitive programming problem in Python 3. Read from stdin, "
                                    "write to stdout. Reply with one code block.\n\n" + p["description"],
                          "hidden_tests": p["hidden_tests"], "time_limit": p["time_limit"]})

    gsm = hf_rows("openai/gsm8k", "main", "test", 200)
    rng.shuffle(gsm)
    for r in gsm[:per_cell]:
        tasks.append({"id": f"math-easy-{abs(hash(r['question'])) % 10**8}", "category": "math",
                      "difficulty": "easy", "grader": "math", "system": SYSTEM_MATH, "prompt": r["question"],
                      "answer": r["answer"].split("####")[-1].strip()})
    m500 = [r for r in hf_rows("HuggingFaceH4/MATH-500", "default", "test", 500)
            if r["level"] == 5 and len(r["answer"]) <= 12 and "\\" not in r["answer"].replace("\\frac", "")]
    rng.shuffle(m500)
    for r in m500[:per_cell]:
        tasks.append({"id": f"math-medium-{r['unique_id']}", "category": "math", "difficulty": "medium",
                      "grader": "math", "system": SYSTEM_MATH, "prompt": r["problem"], "answer": r["answer"]})
    aime = hf_rows("math-ai/aime25", "default", "test", 30)
    rng.shuffle(aime)
    for r in aime[:per_cell]:
        tasks.append({"id": f"math-hard-aime25-{r['id']}", "category": "math", "difficulty": "hard",
                      "grader": "math", "system": SYSTEM_MATH, "prompt": r["problem"], "answer": r["answer"]})

    for level in ("easy", "medium", "hard"):
        for i in range(per_cell):
            t = toolenv.build_task(level, seed * 100 + i * 17 + len(level))
            tasks.append({"id": f"tool_use-{level}-{i}", "category": "tool_use", "difficulty": level,
                          "grader": "toolenv", "system": "You operate the order system through the tools. "
                          "Be precise and follow the instructions exactly.", **t})

    lc_n = max(2, per_cell * 2 // 3)
    for level in ("easy", "medium", "hard"):
        for i in range(lc_n):
            t = build_long_context(level, seed * 1000 + i)
            tasks.append({"id": f"long_context-{level}-{i}", "category": "long_context", "difficulty": level,
                          "grader": "math", "system": "Answer from the document only.", **t})

    ag_n = max(2, per_cell * 2 // 3)
    count = 0
    for row in he_rows[per_cell:]:
        bug = build_bugfix(row, rng)
        if bug is None:
            continue
        tasks.append({"id": f"agentic-easy-{row['task_id']}", "category": "agentic", "difficulty": "easy",
                      "grader": "agent_humaneval", **bug})
        count += 1
        if count >= ag_n:
            break
    for level, probs in (("medium", medium[per_cell:per_cell + ag_n]), ("hard", hard[per_cell:per_cell + ag_n])):
        for p in probs:
            tasks.append({"id": f"agentic-{level}-{p['name'][:12]}", "category": "agentic", "difficulty": level,
                          "grader": "agent_stdio", "rating": p["rating"],
                          "files": {"PROBLEM.md": p["description"],
                                    "tests/public.json": json.dumps(p["public_tests"])},
                          "instruction": "Solve the problem in PROBLEM.md: write `solution.py` (Python 3, stdin -> "
                                         "stdout). run_tests runs the public examples. Hidden tests will also be "
                                         "run, so handle edge cases and efficiency. Call done when finished.",
                          "visible_tests": p["public_tests"], "hidden_tests": p["hidden_tests"],
                          "time_limit": p["time_limit"]})

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fh:
        for t in tasks:
            fh.write(json.dumps(t) + "\n")
    from collections import Counter
    print(Counter((t["category"], t["difficulty"]) for t in tasks))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--out", default="experiments/tasks/tasks.jsonl")
    b.add_argument("--per-cell", type=int, default=6)
    b.add_argument("--seed", type=int, default=17)
    args = ap.parse_args()
    build(Path(args.out), args.per_cell, args.seed)
