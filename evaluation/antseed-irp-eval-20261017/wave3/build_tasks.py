#!/usr/bin/env python3
"""Build wave 3: 60 fresh tasks, same composition and rules as wave 2 (../wave2/build_tasks.py), seeded, fixed order, disjoint from waves 1 and 2
(problems, MMLU-Pro items and generator outputs already used are excluded). Needs network + `pandas huggingface_hub`.

  python build_tasks.py OUT_FULL.jsonl OUT_PUBLIC.jsonl

  coding    14  LiveCodeBench v6 (Jan-Apr 2025 AtCoder / LeetCode problems), hidden tests run in the sandbox
  math      16  AIME 2026, HMMT Nov 2025, HMMT Feb 2026, CMIMC 2025, BRUMO 2025 (MathArena copies), exact numeric answer
  knowledge 10  MMLU-Pro (MIT): 6 from the SciBench / TheoremQA sourced items (college-level), 4 easy-category items
  everyday  20  synthetic, generated here with a seeded RNG (CC0): date / table / graph / schedule / constrained-JSON tasks

Difficulty is chosen ONLY from public per-item metadata, never from running our models: LiveCodeBench's own
easy/medium/hard label, the position of a competition problem in its paper (later = harder), and the MMLU-Pro source tag
(SciBench / TheoremQA are the college-level sources). No model of this study was run on any of these items before
registration.

Licences: MMLU-Pro is MIT, the synthetic tasks are CC0. The MathArena problem sets are CC BY-NC-SA 4.0 (the problems are
(c) their competitions) and LiveCodeBench problem statements are (c) AtCoder / LeetCode, so for those the PUBLIC file
carries only id, source link, answer-free metadata and the SHA-256 of the prompt, not the text. OUT_FULL.jsonl (with
every prompt, key and hidden test) is rebuilt byte-identically by this script and is not committed.
"""
import glob
import base64
import hashlib
import json
import pickle
import random
import re
import sys
import zlib
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download, snapshot_download

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import grade as g1  # wave-1 helpers (_num)


def used_before():
    """ids of every task of waves 1 and 2 (public files); wave 3 must not repeat a problem, an MMLU-Pro item or a (contest, position)."""
    ids = set()
    for p in (HERE.parent / "tasks-public.jsonl", HERE.parent / "wave2" / "tasks-public.jsonl"):
        ids |= {json.loads(l)["id"] for l in open(p)}
    return ids


USED = used_before()

SEED = 20261019
MAX_TEST_BYTES = 1_500_000
MAX_TESTS = 30
PROBLEM_NOTE = "Source text is not redistributed (see README 'Licences'); rebuild it with build_tasks.py."


def sha(s):
    return hashlib.sha256(s.encode()).hexdigest()


# --------------------------------------------------------------------------- coding (LiveCodeBench v6)
IMPORTS = ("from typing import *\nfrom collections import *\nfrom itertools import *\nfrom functools import *\nfrom heapq import *\n"
           "from bisect import *\nfrom math import *\nimport math, sys, re, string, heapq, bisect, itertools, functools, collections\n")


def lcb_rows():
    f = hf_hub_download("livecodebench/code_generation_lite", "test6.jsonl", repo_type="dataset")
    rows = []
    for l in open(f):
        r = json.loads(l)
        pub = json.loads(r["public_test_cases"])
        priv = json.loads(pickle.loads(zlib.decompress(base64.b64decode(r["private_test_cases"].encode()))))
        tests = pub + priv
        if not tests or len({t["testtype"] for t in tests}) != 1:
            continue
        if sum(len(t["input"]) + len(t["output"]) for t in tests) > MAX_TEST_BYTES:
            continue
        if len(tests) > MAX_TESTS:  # keep the public ones, then evenly spaced private ones
            keep = list(range(len(pub))) + [len(pub) + k * len(priv) // (MAX_TESTS - len(pub)) for k in range(MAX_TESTS - len(pub))]
            tests = [tests[k] for k in sorted(set(keep))]
        r["tests"] = [{"input": t["input"], "output": t["output"]} for t in tests]
        r["testtype"] = tests[0]["testtype"]
        rows.append(r)
    return rows


def coding(rng):
    rows = lcb_rows()
    out = []
    plan = [("easy", [("leetcode", 1), ("atcoder", 2)]), ("medium", [("leetcode", 2), ("atcoder", 2)]),
            ("hard", [("leetcode", 4), ("atcoder", 3)])]
    for level, parts in plan:
        for platform, k in parts:
            pool = sorted([r for r in rows if r["difficulty"] == level and r["platform"] == platform
                           and f"coding-lcb6-{r['platform']}-{r['question_id']}" not in USED], key=lambda r: r["question_id"])
            for r in rng.sample(pool, k):
                tid = f"coding-lcb6-{r['platform']}-{r['question_id']}"
                if r["testtype"] == "stdin":
                    prompt = (r["question_content"].rstrip() + "\n\nWrite a complete Python 3 program that reads from standard input and writes to "
                              "standard output. Reply with the full program in one Python code block.")
                    extra = {"mode": "stdin", "tests": r["tests"]}
                else:
                    func = json.loads(r["metadata"])["func_name"]
                    prompt = (r["question_content"].rstrip() + "\n\nStarter code:\n```python\n" + r["starter_code"].rstrip() +
                              "\n```\n\nComplete the class `Solution`. Reply with the full class in one Python code block "
                              "(add any imports you need).")
                    extra = {"mode": "functional", "func_name": func, "tests": r["tests"]}
                out.append({"id": tid, "category": "coding", "difficulty": level, "grader": "lcb",
                            "source": f"LiveCodeBench v6 {r['platform']} {r['question_id']} ({r['contest_date'][:10]}, label {level})",
                            "source_url": "https://huggingface.co/datasets/livecodebench/code_generation_lite",
                            "licence": "problem text (c) AtCoder / LeetCode; not redistributed", "text_published": False,
                            "prompt": prompt, "n_tests": len(r["tests"]), **extra})
    return out


# --------------------------------------------------------------------------- math (MathArena copies of public contests)
SETS = {"aime_2026": ("AIME 2026", "an integer from 0 to 999"), "hmmt_nov_2025": ("HMMT November 2025", "a number or a fraction like 3/4"),
        "hmmt_feb_2026": ("HMMT February 2026", "a number or a fraction like 3/4"),
        "cmimc_2025": ("CMIMC 2025", "a number or a fraction like 3/4"), "brumo_2025": ("BRUMO 2025", "a number or a fraction like 3/4")}
#: (set, lo, hi) position ranges: position inside the contest paper is the public difficulty signal
MATH_PLAN = [("easy", [("aime_2026", 1, 3), ("hmmt_nov_2025", 1, 3), ("brumo_2025", 1, 10)]),
             ("medium", [("aime_2026", 6, 10), ("aime_2026", 6, 10), ("hmmt_nov_2025", 5, 10), ("brumo_2025", 11, 20)]),
             ("hard", [("aime_2026", 11, 15), ("aime_2026", 11, 15), ("hmmt_nov_2025", 11, 15), ("hmmt_nov_2025", 11, 15),
                       ("hmmt_feb_2026", 9, 20), ("hmmt_feb_2026", 9, 20), ("hmmt_feb_2026", 21, 33), ("cmimc_2025", 7, 20),
                       ("cmimc_2025", 21, 40)])]
# wave 2 used 3 of the 5 AIME 2026 positions 11-15, so wave 3 takes 2 there and one more HMMT Feb 2026 item (positions 21-33) instead


def math_tasks(rng):
    frames = {}
    for s in SETS:
        d = snapshot_download(f"MathArena/{s}", repo_type="dataset")
        frames[s] = pd.read_parquet(glob.glob(f"{d}/data/*.parquet")[0])
    out, used = [], set()
    for level, parts in MATH_PLAN:
        for s, lo, hi in parts:
            df = frames[s]
            pool = [r for r in df.itertuples() if lo <= r.problem_idx <= hi and (s, r.problem_idx) not in used
                    and f"math-{s}-{r.problem_idx}" not in USED
                    and g1._num(str(r.answer).replace("\\dfrac", "\\frac")) is not None
                    and re.fullmatch(r"-?[\d./\\{}a-z]+", str(r.answer).replace("\\dfrac", "\\frac"))]
            r = rng.choice(pool)
            used.add((s, r.problem_idx))
            name, fmt = SETS[s]
            out.append({"id": f"math-{s}-{r.problem_idx}", "category": "math", "difficulty": level, "grader": "number",
                        "source": f"{name} problem {r.problem_idx}",
                        "source_url": f"https://huggingface.co/datasets/MathArena/{s}", "licence": "CC BY-NC-SA 4.0 (MathArena); problems (c) the competition",
                        "text_published": False,
                        "prompt": r.problem + f"\n\nWork it out, then finish with one line of the form `Final answer: <answer>` ({fmt}).",
                        "expected": str(r.answer).replace("\\dfrac", "\\frac")})
    return out


# --------------------------------------------------------------------------- knowledge (MMLU-Pro)
def knowledge(rng, used_ids):
    df = pd.read_parquet(glob.glob(snapshot_download("TIGER-Lab/MMLU-Pro", repo_type="dataset", allow_patterns=["data/test-*.parquet"]) + "/data/test-*.parquet")[0])
    out = []
    for level, pick in (("hard", lambda r: r.src.startswith(("scibench", "theoremQA")) and r.category in ("physics", "chemistry", "math")),
                        ("easy", lambda r: r.src.startswith("ori_mmlu") and r.category in ("psychology", "history", "health", "economics"))):
        pool = [r for r in df.itertuples() if pick(r) and len(r.options) == 10 and len(r.question) < 2000 and r.question_id not in used_ids and f"knowledge-mmlupro-{r.question_id}" not in USED]
        for r in rng.sample(pool, 6 if level == "hard" else 4):
            opts = "\n".join(f"{chr(65 + i)}. {o}" for i, o in enumerate(r.options))
            out.append({"id": f"knowledge-mmlupro-{r.question_id}", "category": "knowledge", "difficulty": level, "grader": "letter",
                        "source": f"MMLU-Pro question {r.question_id} ({r.category}, src {r.src})", "licence": "MIT", "text_published": True,
                        "prompt": f"{r.question}\n\n{opts}\n\nThink it through, then finish with one line of the form `Answer: <letter>` (A-J).",
                        "expected": r.answer})
    return out


# --------------------------------------------------------------------------- everyday (synthetic, multi-step, exact checks)
FA = "\n\nFinish with one line of the form `Final answer: <value>`."
NAMES = ["Marta Okafor", "Jonas Lindqvist", "Priya Raman", "Tomasz Wrona", "Aiko Tanabe", "Lucia Ferrer", "Omar Haddad", "Ines Duarte", "Kofi Mensah",
         "Hana Novak", "Liam Byrne", "Sofia Rossi"]
DEPTS = ["Sales", "Support", "Research", "Finance", "Ops"]


def gen_unit_chain(R):
    km = R.choice([12.5, 42.195, 7.3, 21.1])
    mph = R.choice([6.5, 7.2, 9.1])
    secs = km / 1.609344 / mph * 3600
    return (f"Someone runs {km} km at a constant {mph} miles per hour (1 mile = 1.609344 km). How many minutes does that take? Round to 1 decimal." + FA,
            f"{secs / 60:.1f}", "number", {"tol": 0.051})


def gen_string_ops(R, n_ops):
    words = R.sample(["harbor", "violet", "granite", "thunder", "pocket", "lantern", "meadow", "copper", "sketch", "orbit", "velvet", "river"], 4)
    s = " ".join(words)
    ops_all = [("reverse the order of the words", lambda x: " ".join(reversed(x.split()))),
               ("remove every vowel (a, e, i, o, u)", lambda x: re.sub("[aeiou]", "", x)),
               ("replace every letter 'r' by '#'", lambda x: x.replace("r", "#")),
               ("swap the first and the last character of every word", lambda x: " ".join((w[-1] + w[1:-1] + w[0]) if len(w) > 1 else w for w in x.split())),
               ("delete every word that is shorter than 4 characters", lambda x: " ".join(w for w in x.split() if len(w) >= 4)),
               ("move the first character of every word to the end of that word", lambda x: " ".join(w[1:] + w[0] for w in x.split())),
               ("replace each space by a hyphen", lambda x: x.replace(" ", "-"))]
    ops = R.sample(ops_all, n_ops)
    x = s
    for _, f in ops:
        x = f(x)
    steps = "\n".join(f"{i + 1}. {d}" for i, (d, _) in enumerate(ops))
    return (f"Start with the text `{s}` and apply these steps in order, each to the result of the previous one:\n{steps}\n"
            f"What is the final text? (Write it exactly, without quotes.)" + FA, x, "string", {})


def gen_nth_weekday(R):
    y, mth = R.choice([2027, 2028, 2029]), R.randrange(1, 13)
    wd, nth = R.randrange(0, 7), R.choice([2, 3, 4])
    d = date(y, mth, 1)
    d += timedelta(days=(wd - d.weekday()) % 7 + 7 * (nth - 1))
    add = R.randrange(40, 400)
    names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    return (f"Take the {nth}{'nd' if nth == 2 else 'rd' if nth == 3 else 'th'} {names[wd]} of {d.strftime('%B')} {y}. What is the date {add} days later? "
            f"Use the real Gregorian calendar. Give it as YYYY-MM-DD." + FA, (d + timedelta(days=add)).isoformat(), "string", {})


def gen_rooms(R, n):
    ms = []
    for _ in range(n):
        st = R.randrange(8 * 60, 16 * 60, 15)
        ms.append((st, st + R.choice([30, 45, 60, 75, 90, 120])))
    ev = sorted([(a, 1) for a, b in ms] + [(b, -1) for a, b in ms], key=lambda e: (e[0], e[1]))  # an end at t frees the room for a start at t
    cur = best = 0
    for _, d in ev:
        cur += d
        best = max(best, cur)
    fm = lambda x: f"{x // 60:02d}:{x % 60:02d}"
    return (f"A company has these {n} meetings today: " + ", ".join(f"{fm(a)}-{fm(b)}" for a, b in ms) +
            ". A meeting that ends at exactly the time another one starts can use the same room. What is the smallest number of rooms needed so "
            "that every meeting has a room for its whole duration? Answer with an integer." + FA, str(best), "number", {"tol": 0.01})


def gen_deps(R):
    names = "ABCDEFGHI"
    dur = {n: R.randrange(2, 9) for n in names}
    deps = {}
    for i, n in enumerate(names):
        deps[n] = sorted(R.sample(names[:i], min(i, R.choice([0, 1, 2])))) if i else []
    fin = {}
    for n in names:
        fin[n] = dur[n] + max([fin[d] for d in deps[n]] or [0])
    lines = "\n".join(f"- {n}: {dur[n]} days" + (f", can start only after {', '.join(deps[n])} finished" if deps[n] else ", no prerequisites") for n in names)
    return ("A project has these tasks. Any number of tasks may run in parallel as long as their prerequisites are finished.\n" + lines +
            "\nWhat is the shortest possible total project duration in days? Answer with an integer." + FA, str(max(fin.values())), "number", {"tol": 0.01})


def gen_table(R, hard):
    rows = []
    for nm in R.sample(NAMES, 10):
        rows.append((nm, R.choice(DEPTS), R.randrange(38, 120) * 1000 + R.randrange(0, 10) * 500, date(2018, 1, 1) + timedelta(days=R.randrange(0, 2400))))
    tbl = "| Name | Dept | Salary (EUR) | Start date |\n|---|---|---|---|\n" + "\n".join(f"| {n} | {d} | {s} | {sd.isoformat()} |" for n, d, s, sd in rows)
    cut = date(2021, 1, 1) + timedelta(days=R.randrange(0, 700))
    sel = [r for r in rows if r[3] < cut]
    if hard:
        by = {}
        for n, d, s, _ in sel:
            by.setdefault(d, []).append(s)
        if len(by) < 2:
            return None
        means = {d: sum(v) / len(v) for d, v in by.items()}
        top = max(means.values())
        if sorted(means.values())[-2] == top:
            return None
        dep = [d for d, m in means.items() if m == top][0]
        q = (f"Consider only employees who started strictly before {cut.isoformat()}. Among the departments of those employees, which one has the highest "
             f"average salary (average over those employees only)? Answer with the department name.")
        return tbl + "\n\n" + q + FA, dep, "string", {}
    if len(sel) < 3:
        return None
    tot = sum(s for _, _, s, _ in sel)
    q = f"What is the total salary in EUR of all employees who started strictly before {cut.isoformat()}? Answer with an integer."
    return tbl + "\n\n" + q + FA, str(tot), "number", {"tol": 0.01}


def gen_bizdays(R):
    start = date(2027, 1, 1) + timedelta(days=R.randrange(0, 600))
    n = R.randrange(25, 70)
    hol = sorted({start + timedelta(days=R.randrange(1, n + 20)) for _ in range(4)})
    d, c = start, 0
    while c < n:
        d += timedelta(days=1)
        if d.weekday() < 5 and d not in hol:
            c += 1
    return (f"Count working days as Monday to Friday except these public holidays: {', '.join(h.isoformat() for h in hol)}. Starting on {start.isoformat()} "
            f"(day 0, not counted), on what date does working day number {n} fall? Give it as YYYY-MM-DD." + FA, d.isoformat(), "string", {})


def gen_json_group(R):
    custs = R.sample(["Acme GmbH", "Birch & Co", "Cobalt Ltd", "Delta SA", "Eon AG"], 4)
    orders = []
    for k in range(11):
        orders.append({"id": 100 + k, "customer": R.choice(custs), "amount": round(R.randrange(500, 40000) / 100, 2),
                       "status": R.choice(["paid", "paid", "paid", "cancelled", "refunded"])})
    agg = {}
    for o in orders:
        if o["status"] == "cancelled":
            continue
        v = o["amount"] if o["status"] == "paid" else -o["amount"]
        a = agg.setdefault(o["customer"], [0.0, 0])
        a[0] += v
        a[1] += 1
    res = sorted(({"customer": c, "orders": n, "net": round(t, 2)} for c, (t, n) in agg.items()), key=lambda e: (-e["net"], e["customer"]))
    if not res:
        return None
    lines = "\n".join(f"{o['id']},{o['customer']},{o['amount']:.2f},{o['status']}" for o in orders)
    return ("Orders (id,customer,amount in EUR,status):\n" + lines + "\n\nIgnore cancelled orders. A refunded order counts as a negative amount (the "
            "amount is subtracted). For each remaining customer compute the number of orders (refunded ones included) and the net sum, rounded to 2 decimals. "
            "Reply with JSON only: {\"result\": [{\"customer\": <name>, \"orders\": <int>, \"net\": <number>}, ...]} sorted by net descending (ties by "
            "customer name ascending).", {"result": res}, "json", {})


def gen_tz_flight(R):
    from zoneinfo import ZoneInfo
    a, b = R.choice([("America/New_York", "Europe/London"), ("America/New_York", "Europe/Berlin"), ("America/Chicago", "Europe/Paris"),
                     ("America/Los_Angeles", "Europe/London")])
    dep = datetime(2027, 3, R.randrange(15, 27), R.randrange(6, 23), R.choice([0, 10, 25, 40, 55]), tzinfo=ZoneInfo(a))
    dur = R.randrange(6 * 60, 11 * 60 + 30)
    arr = (dep.astimezone(ZoneInfo("UTC")) + timedelta(minutes=dur)).astimezone(ZoneInfo(b))
    city = lambda z: z.split("/")[1].replace("_", " ")
    return (f"A flight leaves {city(a)} on {dep.strftime('%Y-%m-%d')} at {dep.strftime('%H:%M')} local time and lands in {city(b)} on "
            f"{arr.strftime('%Y-%m-%d')} at {arr.strftime('%H:%M')} local time. Both cities follow their normal civil time zones including daylight saving time as "
            f"it applies on those dates. How long was the flight in minutes? Answer with an integer." + FA, str(dur), "number", {"tol": 0.01})


def gen_graph(R):
    nodes = "ABCDEFGH"
    edges = {}
    for i, u in enumerate(nodes):
        for v in nodes[i + 1:]:
            if R.random() < 0.42:
                edges[(u, v)] = R.randrange(2, 15)
    for u, v in zip(nodes, nodes[1:]):
        edges.setdefault((u, v), R.randrange(8, 20))
    import heapq

    def dij(skip):
        adj = {n: [] for n in nodes}
        for (u, v), w in edges.items():
            if skip in (u, v):
                continue
            adj[u].append((v, w))
            adj[v].append((u, w))
        dist, pq = {"A": 0}, [(0, "A")]
        while pq:
            d, u = heapq.heappop(pq)
            if d > dist.get(u, 1e9):
                continue
            for v, w in adj[u]:
                if d + w < dist.get(v, 1e9):
                    dist[v] = d + w
                    heapq.heappush(pq, (d + w, v))
        return dist.get("H")

    skip = R.choice("BCDEFG")
    ans = dij(skip)
    if ans is None:
        return None
    lst = ", ".join(f"{u}-{v}:{w}" for (u, v), w in sorted(edges.items()))
    return (f"An undirected road network has these roads (road: length in km): {lst}. Road works make town {skip} impassable (you cannot enter it). "
            f"What is the length of the shortest route from A to H in km? Answer with an integer." + FA, str(ans), "number", {"tol": 0.01})


def gen_amort(R):
    P, rate, months = R.choice([15000, 24000, 9500]), R.choice([4.8, 6.0, 7.5]), R.choice([36, 48, 60])
    k = R.randrange(10, months - 6)
    i = rate / 1200
    pay = P * i / (1 - (1 + i) ** -months)
    bal = P
    for _ in range(k):
        bal = bal * (1 + i) - pay
    return (f"A loan of {P} EUR has a nominal annual rate of {rate}% compounded monthly (monthly rate = rate/12) and is repaid with {months} equal monthly "
            f"payments at the end of each month. The payment is the exact annuity payment (do not round it). What is the outstanding balance in EUR right after "
            f"payment number {k}, rounded to 2 decimals?" + FA, f"{bal:.2f}", "number", {"tol": 0.02})


def gen_ledger(R):
    bal = R.randrange(500, 2000)
    start = bal
    tx = []
    wd = 0
    for _ in range(9):
        if R.random() < 0.4:
            tx.append(("deposit", R.randrange(50, 500)))
            bal += tx[-1][1]
        else:
            tx.append(("withdrawal", R.randrange(20, 300)))
            wd += 1
            bal -= tx[-1][1] + (2.5 if wd % 3 == 0 else 0)
    bal = round(bal * 1.01, 2)
    lines = "\n".join(f"{i + 1}. {k} {a} EUR" for i, (k, a) in enumerate(tx))
    return (f"An account starts with {start} EUR. Transactions in this order:\n{lines}\nRules: every third withdrawal (the 3rd, 6th, 9th, ... withdrawal, "
            f"counting withdrawals only) also costs a fee of 2.50 EUR on top of its amount. After all transactions the bank adds 1% interest on the final "
            f"balance. What is the final balance in EUR, rounded to 2 decimals?" + FA, f"{bal:.2f}", "number", {"tol": 0.011})


def gen_dedupe(R):
    people = R.sample(NAMES, 4)
    recs = []
    for k in range(7):
        p = R.choice(people)
        mail = p.split()[0].lower() + "@example.org"
        recs.append({"name": p, "email": R.choice([mail, mail.upper(), mail.capitalize()]), "created": (date(2026, 1, 1) + timedelta(days=R.randrange(0, 300))).isoformat()})
    first = {}
    for r in recs:
        key = r["email"].lower()
        if key not in first or r["created"] < first[key]["created"]:
            first[key] = r
    cnt = {}
    for r in recs:
        cnt[r["email"].lower()] = cnt.get(r["email"].lower(), 0) + 1
    res = [{"email": k, "name": v["name"], "first_created": v["created"], "records": cnt[k]} for k, v in sorted(first.items())]
    lines = "\n".join(f"{r['name']} | {r['email']} | {r['created']}" for r in recs)
    return ("Contact records (name | email | created):\n" + lines + "\n\nEmail addresses are case-insensitive. Merge records with the same email. For each distinct "
            "email output its lowercase address, the name on its earliest record, the earliest created date and the number of merged records. Reply with JSON only: "
            "{\"contacts\": [{\"email\": ..., \"name\": ..., \"first_created\": ..., \"records\": <int>}, ...]} sorted by email ascending.", {"contacts": res}, "json", {})


def everyday(rng):
    R = rng
    plan = [("easy", gen_unit_chain, ()), ("easy", gen_string_ops, (2,)),
            ("medium", gen_nth_weekday, ()), ("medium", gen_nth_weekday, ()), ("medium", gen_rooms, (6,)), ("medium", gen_rooms, (8,)),
            ("medium", gen_string_ops, (4,)), ("medium", gen_string_ops, (5,)), ("medium", gen_deps, ()), ("medium", gen_table, (False,)),
            ("hard", gen_table, (True,)), ("hard", gen_table, (True,)), ("hard", gen_bizdays, ()), ("hard", gen_bizdays, ()),
            ("hard", gen_json_group, ()), ("hard", gen_json_group, ()), ("hard", gen_tz_flight, ()), ("hard", gen_tz_flight, ()),
            ("hard", gen_graph, ()), ("hard", gen_dedupe, ())]
    out = []
    for n, (level, fn, args) in enumerate(plan, 1):
        for _ in range(50):
            got = fn(R, *args)
            if got is not None:
                break
        prompt, expected, grader, extra = got
        out.append({"id": f"everyday3-{n:02d}-{fn.__name__[4:]}", "category": "everyday", "difficulty": level, "grader": grader,
                    "source": "synthetic generator in build_tasks.py", "licence": "CC0-1.0", "text_published": True,
                    "prompt": prompt, "expected": expected, **extra})
    return out


def wave1_ids():
    return {i.split("-")[-1] for i in USED if i.startswith("knowledge-")}


def public_view(t):
    p = {k: v for k, v in t.items() if k not in ("tests", "check")}
    p["prompt_sha256"] = sha(t["prompt"])
    if not t["text_published"]:
        p.pop("prompt")
        p["note"] = PROBLEM_NOTE
    if t["grader"] == "lcb":
        p["tests_sha256"] = sha(json.dumps(t["tests"], sort_keys=True))
    return p


def main(out_full, out_public):
    rng = random.Random(SEED)
    tasks = coding(rng) + math_tasks(rng) + knowledge(rng, wave1_ids()) + everyday(rng)
    assert len(tasks) == 60 and len({t["id"] for t in tasks}) == 60, len(tasks)
    assert all(t.get("expected") is not None for t in tasks if t["category"] != "coding")
    with open(out_full, "w") as f:
        for t in tasks:
            f.write(json.dumps(t, sort_keys=True, ensure_ascii=False) + "\n")
    with open(out_public, "w") as f:
        for t in tasks:
            f.write(json.dumps(public_view(t), sort_keys=True, ensure_ascii=False) + "\n")
    print(len(tasks), "tasks", {k: sum(1 for t in tasks if t["difficulty"] == k) for k in ("easy", "medium", "hard")})


if __name__ == "__main__":
    main(*sys.argv[1:3])
