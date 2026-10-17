#!/usr/bin/env python3
"""Build tasks-public.jsonl: 60 tasks, seeded, fixed order. Needs network + `pandas huggingface_hub`.

  coding    15  HumanEval+ (Apache-2.0) x7, MBPP+ (Apache-2.0) x8     hidden tests run in the sandbox
  math      15  GSM8K (MIT) x5, MATH level 4-5 (MIT) x5, AIME 2024 (MIT) x5   exact final answer
  knowledge 15  MMLU-Pro (MIT), 10-option multiple choice               exact letter
  everyday  15  synthetic, generated here with a seeded RNG (CC0)       deterministic checks

Disjoint from the calibration tasks: HumanEval+ ids the earlier seeded shuffle could reach are excluded, GSM8K
rows 0-199 and every MATH-500 problem are excluded, AIME 2025 was used before (we use 2024), MBPP+ / MMLU-Pro /
synthetic tasks were never used. `python build_tasks.py OUT --check-disjoint` re-checks by content.
GPQA is not used (its licence asks that it not be republished). usage: build_tasks.py OUT.jsonl
"""
import ast
import glob
import gzip
import json
import random
import re
import sys
import urllib.request
from datetime import date, datetime, timedelta
from fractions import Fraction
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download, snapshot_download

SEED = 20261017
HE_PLUS_GZ = "https://github.com/evalplus/humanevalplus_release/releases/download/v0.1.10/HumanEvalPlus.jsonl.gz"
MAX_TEST_BYTES = 60_000
#: the dataset's own test is broken here (its reference solution fails it), found by running the reference solutions
BROKEN_TESTS = {"HumanEval/32"}


def pq(repo, pattern):
    d = snapshot_download(repo, repo_type="dataset", allow_patterns=[pattern])
    return pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{d}/{pattern}"))], ignore_index=True)


def terciles(rows, key):
    rows = sorted(rows, key=lambda r: (key(r), r["id"]))
    n = len(rows)
    return rows[: n // 3], rows[n // 3: 2 * n // 3], rows[2 * n // 3:]


def pick(rng, rows, key, counts):
    """counts = (easy, medium, hard) picked from terciles of `key`."""
    out = []
    for level, group, k in zip(("easy", "medium", "hard"), terciles(rows, key), counts):
        for r in rng.sample(group, k):
            out.append({**r, "difficulty": level})
    return out


# --------------------------------------------------------------------------- coding
def prior_humaneval_ids():
    """HumanEval+ ids the earlier calibration build could have used (seed 17 shuffle, generous prefix)."""
    rows = [json.loads(l) for l in gzip.open(urllib.request.urlretrieve(HE_PLUS_GZ)[0], "rt")]
    random.Random(17).shuffle(rows)
    return {r["task_id"] for r in rows[:60]}


def coding(rng):
    out = []
    he = pq("evalplus/humanevalplus", "data/*.parquet")
    used = prior_humaneval_ids()
    rows = [{"id": f"coding-{r.task_id.replace('/', '-')}", "src": r.task_id, "prompt": r.prompt, "test": r.test,
             "entry_point": r.entry_point, "sol": r.canonical_solution}
            for r in he.itertuples() if r.task_id not in used and r.task_id not in BROKEN_TESTS and len(r.test) <= MAX_TEST_BYTES]
    for r in pick(rng, rows, lambda r: len([l for l in r["sol"].splitlines() if l.strip()]), (2, 3, 2)):
        out.append({"id": r["id"], "category": "coding", "difficulty": r["difficulty"],
                    "source": "HumanEval+ " + r["src"], "licence": "Apache-2.0", "grader": "code",
                    "prompt": "Complete this Python function. Reply with the full function (including any imports) "
                              "in one Python code block.\n\n" + r["prompt"],
                    "check": r["test"] + f"\ncheck({r['entry_point']})\n"})
    mb = pq("evalplus/mbppplus", "data/*.parquet")
    rows = [{"id": f"coding-mbpp-{r.task_id}", "src": r.task_id, "text": r.prompt, "test": r.test,
             "example": list(r.test_list)[0], "sol": r.code}
            for r in mb.itertuples() if len(r.test) <= MAX_TEST_BYTES]
    for r in pick(rng, rows, lambda r: len([l for l in r["sol"].splitlines() if l.strip()]), (3, 3, 2)):
        out.append({"id": r["id"], "category": "coding", "difficulty": r["difficulty"],
                    "source": "MBPP+ " + str(r["src"]), "licence": "Apache-2.0", "grader": "code",
                    "prompt": f"{r['text']}\nYour code must define the function used in this example test:\n"
                              f"{r['example']}\nReply with the full solution in one Python code block.",
                    "check": r["test"]})
    return out


# --------------------------------------------------------------------------- math
BOXED = re.compile(r"\\boxed\{")


def boxed(solution):
    i = [m.end() for m in BOXED.finditer(solution)]
    if not i:
        return None
    s, depth, j = i[-1], 1, i[-1]
    while j < len(solution) and depth:
        depth += {"{": 1, "}": -1}.get(solution[j], 0)
        j += 1
    return solution[s:j - 1].strip() if depth == 0 else None


SIMPLE = re.compile(r"^-?\d+(\.\d+)?$|^-?\\d?frac\{\d+\}\{\d+\}$")
SUFFIX = ("\n\nWork it out, then finish with one line of the form `Final answer: <answer>` "
          "(a number, or a fraction like 3/4).")


def math_tasks(rng):
    out = []
    gsm = pq("openai/gsm8k", "main/test-*.parquet")
    rows = [{"id": f"math-gsm8k-{i}", "q": r.question, "a": r.answer.split("####")[-1].strip().replace(",", "")}
            for i, r in enumerate(gsm.itertuples()) if i >= 200]
    for r in rng.sample(rows, 5):
        out.append({"id": r["id"], "category": "math", "difficulty": "easy", "source": "GSM8K test row " + r["id"].split("-")[-1],
                    "licence": "MIT", "grader": "number", "prompt": r["q"] + SUFFIX, "expected": r["a"]})
    d = snapshot_download("HuggingFaceH4/MATH-500", repo_type="dataset")
    m500 = set(pd.concat([pd.read_json(f, lines=True) for f in glob.glob(f"{d}/**/*.jsonl", recursive=True)])["problem"])
    subjects = ["algebra", "counting_and_probability", "geometry", "intermediate_algebra", "number_theory",
                "prealgebra", "precalculus"]
    rows = []
    for s in subjects:
        df = pq("EleutherAI/hendrycks_math", f"{s}/test-*.parquet")
        for i, r in enumerate(df.itertuples()):
            a = boxed(r.solution)
            if r.level in ("Level 4", "Level 5") and r.problem not in m500 and a and SIMPLE.match(a):
                rows.append({"id": f"math-hendrycks-{s}-{i}", "q": r.problem, "a": a.replace("\\dfrac", "\\frac"),
                             "lvl": r.level})
    chosen = rng.sample([r for r in rows if r["lvl"] == "Level 4"], 2) + rng.sample([r for r in rows if r["lvl"] == "Level 5"], 3)
    for r in chosen:
        out.append({"id": r["id"], "category": "math", "difficulty": "medium", "source": f"MATH test {r['id'][len('math-hendrycks-'):]} ({r['lvl']})",
                    "licence": "MIT", "grader": "number", "prompt": r["q"] + SUFFIX, "expected": r["a"]})
    aime = pq("Maxwell-Jia/AIME_2024", "*.parquet")
    for r in rng.sample(list(aime.itertuples()), 5):
        out.append({"id": f"math-aime2024-{r.ID}", "category": "math", "difficulty": "hard", "source": f"AIME {r.ID}",
                    "licence": "MIT (dataset card)", "grader": "number",
                    "prompt": r.Problem + SUFFIX.replace(" (a number, or a fraction like 3/4)", " (an integer from 0 to 999)"),
                    "expected": str(r.Answer)})
    return out


# --------------------------------------------------------------------------- knowledge
BUCKETS = {"easy": ["psychology", "history", "health", "economics", "business"],
           "medium": ["biology", "philosophy", "other", "computer science", "law"],
           "hard": ["math", "physics", "chemistry", "engineering"]}


def knowledge(rng):
    df = pq("TIGER-Lab/MMLU-Pro", "data/test-*.parquet")
    out = []
    for level, cats in BUCKETS.items():
        pool = [r for r in df.itertuples() if r.category in cats and len(r.options) == 10 and len(r.question) < 2000]
        for r in rng.sample(pool, 5):
            opts = "\n".join(f"{chr(65 + i)}. {o}" for i, o in enumerate(r.options))
            out.append({"id": f"knowledge-mmlupro-{r.question_id}", "category": "knowledge", "difficulty": level,
                        "source": f"MMLU-Pro question {r.question_id} ({r.category})", "licence": "MIT", "grader": "letter",
                        "prompt": f"{r.question}\n\n{opts}\n\nThink it through, then finish with one line of the form "
                                  f"`Answer: <letter>` (A-J).", "expected": r.answer})
    return out


# --------------------------------------------------------------------------- everyday (synthetic)
NAMES = ["Marta Okafor", "Jonas Lindqvist", "Priya Raman", "Tomasz Wrona", "Aiko Tanabe", "Lucía Ferrer", "Omar Haddad"]


def everyday(rng):
    R, out = rng, []

    def add(n, level, kind, prompt, expected, grader, **extra):
        out.append({"id": f"everyday-{n:02d}-{kind}", "category": "everyday", "difficulty": level,
                    "source": "synthetic generator in build_tasks.py", "licence": "CC0-1.0", "grader": grader,
                    "prompt": prompt, "expected": expected, **extra})

    FA = "\n\nFinish with one line of the form `Final answer: <value>`."
    # --- easy
    mi = R.choice([13.1, 26.2, 42.7, 8.4, 17.9])
    add(1, "easy", "miles-km", f"Convert {mi} miles to kilometres (1 mile = 1.609344 km). Round to 2 decimals." + FA,
        f"{mi * 1.609344:.2f}", "number", tol=0.011)
    f = R.choice([-4, 23, 59, 77, 104, 98.6]); add(2, "easy", "f-to-c", f"What is {f} degrees Fahrenheit in degrees Celsius? Round to 1 decimal." + FA,
                                                   f"{(f - 32) * 5 / 9:.1f}", "number", tol=0.051)
    p, d, t = R.choice([48, 79, 129, 215]), R.choice([15, 20, 30]), R.choice([7, 8, 19])
    add(3, "easy", "discount", f"A jacket costs {p}.00 euro. A {d}% discount is applied first, then {t}% sales tax is added to the "
                               f"discounted price. What is the final price in euro, rounded to 2 decimals?" + FA,
        f"{p * (1 - d / 100) * (1 + t / 100):.2f}", "number", tol=0.011)
    a, b, c = R.choice([150, 200, 250]), R.choice([3, 4, 5]), R.choice([1, 2])
    s_from, s_to = 4, R.choice([6, 10, 14])
    add(4, "easy", "recipe-json", f"A recipe for {s_from} servings needs {a} g flour, {b} eggs and {c * 100} ml milk. Scale it to "
                                  f"{s_to} servings. Reply with JSON only, with the integer keys \"flour_g\", \"eggs\" and \"milk_ml\".",
        {"flour_g": a * s_to // s_from, "eggs": b * s_to // s_from, "milk_ml": c * 100 * s_to // s_from}
        if (a * s_to) % s_from == 0 and (b * s_to) % s_from == 0 and (c * 100 * s_to) % s_from == 0 else None, "json")
    dep_h, dep_m = R.choice([21, 22, 23]), R.choice([5, 20, 35, 50]); dur_h, dur_m = R.choice([7, 8, 9]), R.choice([10, 25, 40])
    arr = (dep_h * 60 + dep_m + dur_h * 60 + dur_m) % 1440
    add(5, "easy", "arrival", f"A night train leaves at {dep_h:02d}:{dep_m:02d} and the trip takes {dur_h} hours {dur_m} minutes. "
                              f"At what local time (24-hour HH:MM) does it arrive, on the following day? All times are in the same time zone." + FA,
        f"{arr // 60:02d}:{arr % 60:02d}", "string")
    # --- medium
    d0 = date(2027, 1, 1) + timedelta(days=R.randrange(0, 700)); n = R.randrange(220, 900)
    add(6, "medium", "date-plus", f"What is the date {n} days after {d0.isoformat()}? Use the real Gregorian calendar (leap years count). "
                                  f"Give it as YYYY-MM-DD." + FA, (d0 + timedelta(days=n)).isoformat(), "string")
    d1 = date(2027, 1, 1) + timedelta(days=R.randrange(0, 1500))
    add(7, "medium", "weekday", f"Which day of the week is {d1.strftime('%d %B %Y')}? Answer with the English weekday name." + FA,
        d1.strftime("%A"), "string")
    who, amt, cur = R.choice(NAMES), R.choice([1249.5, 87.2, 3015.75, 640.0]), R.choice(["EUR", "USD", "GBP"])
    dd = date(2027, 1, 1) + timedelta(days=R.randrange(0, 360))
    add(8, "medium", "email-json",
        f"Extract the data from this message and reply with JSON only, with the keys \"customer\" (string), \"due_date\" (ISO YYYY-MM-DD) "
        f"and \"amount\" (number) and \"currency\" (3-letter code).\n\n---\nHi team, quick one: {who} called about the open invoice. "
        f"They will transfer {cur} {amt:,.2f} on {dd.strftime('%d.%m.%Y')} at the latest - after that they said they'd dispute it. "
        f"Please do not chase before then. Thanks!\n---",
        {"customer": who, "due_date": dd.isoformat(), "amount": amt, "currency": cur}, "json")
    s = date(2027, 3, 1) + timedelta(days=R.randrange(0, 200)); e = s + timedelta(days=R.randrange(20, 60)); hol = s + timedelta(days=R.randrange(3, 15))
    while hol.weekday() >= 5:
        hol += timedelta(days=1)
    cnt = sum(1 for k in range((e - s).days + 1) if (s + timedelta(days=k)).weekday() < 5 and s + timedelta(days=k) != hol)
    add(9, "medium", "business-days", f"How many working days (Monday to Friday) are there from {s.isoformat()} to {e.isoformat()}, both dates "
                                      f"inclusive, if {hol.isoformat()} is a public holiday and does not count? Answer with an integer." + FA,
        str(cnt), "number", tol=0.01)
    h, m_ = R.choice([9, 11, 14, 16]), R.choice([0, 15, 30, 45]); base = datetime(2027, 6, R.randrange(2, 27), h, m_)
    tk = base + timedelta(hours=9 + 4)  # New York UTC-4 -> Tokyo UTC+9
    add(10, "medium", "timezone", f"A meeting starts on {base.strftime('%Y-%m-%d')} at {base.strftime('%H:%M')} in New York (UTC-4). What is the "
                                  f"date and time in Tokyo (UTC+9)? Format: YYYY-MM-DD HH:MM." + FA, tk.strftime("%Y-%m-%d %H:%M"), "string")
    # --- hard
    items = [(R.choice(["USB-C cable", "Desk lamp", "Notebook A5", "Headset", "Monitor stand", "Mouse pad"]), R.randrange(1, 8),
              R.choice([4.5, 12.9, 19.99, 34.0, 59.5]), R.choice([0, 0, 10, 15, 25])) for _ in range(4)]
    seen, uniq = set(), []
    for it in items:
        if it[0] not in seen:
            seen.add(it[0]); uniq.append(it)
    lines = "\n".join(f"- {nm}: {q} x {pr:.2f} EUR" + (f", line discount {dc}%" if dc else "") for nm, q, pr, dc in uniq)
    tot = round(sum(q * pr * (1 - dc / 100) for nm, q, pr, dc in uniq), 2)
    add(11, "hard", "invoice-json", "Compute the invoice from these lines (a line discount applies to that line only). Reply with JSON only: "
                                    "{\"lines\": {<item name>: <line total in EUR rounded to 2 decimals>, ...}, \"total\": <sum of the line totals>}.\n\n" + lines,
        {"lines": {nm: round(q * pr * (1 - dc / 100), 2) for nm, q, pr, dc in uniq}, "total": tot}, "json")
    d2 = date(2027, 1, 1) + timedelta(days=R.randrange(0, 900)); iso = d2.isocalendar()
    add(12, "hard", "iso-week", f"Give the ISO 8601 week date of {d2.isoformat()} in the form YYYY-Www-D (ISO week-year, week number zero-padded to "
                                f"2 digits, weekday 1=Monday..7=Sunday)." + FA, f"{iso[0]}-W{iso[1]:02d}-{iso[2]}", "string")
    P, rate, months = R.choice([12000, 18500, 9000]), R.choice([4.8, 6.6, 7.2]), R.choice([24, 36, 48]); i_ = rate / 1200
    pay = P * i_ / (1 - (1 + i_) ** -months)
    add(13, "hard", "loan", f"A loan of {P} EUR has a nominal annual interest rate of {rate}% compounded monthly (monthly rate = rate/12) and is repaid in "
                            f"{months} equal monthly payments at the end of each month. What is the monthly payment in EUR, rounded to 2 decimals?" + FA,
        f"{pay:.2f}", "number", tol=0.02)
    mt = []
    for _ in range(5):
        st = R.randrange(9 * 60, 16 * 60, 15); mt.append((st, st + R.choice([30, 45, 60, 90, 120])))
    mt.sort(); merged = []
    for a_, b_ in mt:
        a_, b_ = max(a_, 9 * 60), min(b_, 17 * 60)
        if merged and a_ <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b_)
        else:
            merged.append([a_, b_])
    free = 8 * 60 - sum(b_ - a_ for a_, b_ in merged)
    fm = lambda x: f"{x // 60:02d}:{x % 60:02d}"
    add(14, "hard", "free-minutes", "Between 09:00 and 17:00 these meetings are in the calendar (some overlap): "
                                    + ", ".join(f"{fm(a_)}-{fm(b_)}" for a_, b_ in mt) + ". How many minutes between 09:00 and 17:00 are free (covered by no meeting)? "
                                    "Answer with an integer." + FA, str(free), "number", tol=0.01)
    codes = [200, 200, 301, 404, 500, 502, 503, 200]
    ips = [f"10.{R.randrange(0, 4)}.{R.randrange(0, 4)}.{R.randrange(1, 9)}" for _ in range(12)]
    st_ = [R.choice(codes) for _ in range(12)]
    log = "\n".join(f"{ip} - - [12/Oct/2027:10:{k * 4:02d}:00 +0000] \"GET /api/v{R.randrange(1, 3)}/items HTTP/1.1\" {c} {R.randrange(120, 9000)}"
                    for k, (ip, c) in enumerate(zip(ips, st_)))
    bad = sorted({ip for ip, c in zip(ips, st_) if 500 <= c < 600})
    add(15, "hard", "log-json", "From this access log, find the distinct client IPs that received at least one 5xx status. Reply with JSON only: "
                                "{\"count\": <number of distinct IPs>, \"ips\": [<those IPs sorted as plain strings>]}.\n\n" + log,
        {"count": len(bad), "ips": bad}, "json")
    return out


def main(out_path):
    rng = random.Random(SEED)
    tasks = coding(rng) + math_tasks(rng) + knowledge(rng) + everyday(rng)
    tasks = [t for t in tasks]
    assert len(tasks) == 60 and len({t["id"] for t in tasks}) == 60, len(tasks)
    assert all(t["expected"] is not None for t in tasks if t["category"] != "coding")
    with open(out_path, "w") as f:
        for t in tasks:
            f.write(json.dumps(t, sort_keys=True, ensure_ascii=False) + "\n")
    print(len(tasks), "tasks")


if __name__ == "__main__":
    main(sys.argv[1])
