"""Valid-pair accounting for a paired held-out evaluation.

A category's sample size is not the number of rows in the ledger. The
evaluation is *paired* - the same task is sent to the routing policy and to a
comparator with an identical prompt - so the unit of evidence is the **pair**,
and a pair is only usable when **both** of its rows were actually graded.

Three things make a pair unusable, and all three are counted separately rather
than quietly dropped:

``invalid_truncated``   one side hit the harness's output budget. The provider
                        never finished the answer, so it says nothing about the
                        route's capability - and it is not a free pass either:
                        it costs the pair.
``invalid_sandbox``     the coding grader could not run isolated. A harness
                        failure, never a model failure.
``invalid_incomplete``  one side of the pair has no row at all, because the run
                        stopped, the budget guard fired, or the task was never
                        attempted on that arm.

A failed or refused **provider call** is none of these. It is a property of the
route, it is graded ``False``, and its pair stays valid - otherwise a flaky
provider could drop its own failures out of its own denominator.

Two things an independent reviewer was right to insist on
---------------------------------------------------------
**A truncated pair *is* removed from the quality denominator.** Saying it
"costs the pair" is true but soft: 22 valid design pairs out of 26 attempted
means four attempts are not in the pass rate. The registered rule says to
exclude them, and that rule was written *after* a truncation had been seen on
the original run, so every category also carries
``if_truncation_counted_as_failure`` - the same numbers under the opposite,
harsher rule. If a conclusion only survives under one of the two, it is not a
conclusion.

**The attempted universe is the registered task list, not the rows that exist.**
Deriving it from the ledger lets a task that was never run vanish instead of
being counted incomplete. Pass ``manifest=`` and it cannot.
"""

from __future__ import annotations

CATEGORIES = ("design", "coding", "math", "research", "summarisation", "cache_repeat")

#: Substrings the harness writes into ``detail`` for the two documented
#: exclusions. Nothing else is ever treated as an exclusion.
TRUNCATED = "TRUNCATED"
SANDBOX_UNAVAILABLE = "SANDBOX UNAVAILABLE"


def _graded(row: dict | None) -> bool:
    return row is not None and row.get("passed") is not None


def _exclusion(row: dict | None) -> str | None:
    if row is None:
        return "incomplete"
    if row.get("passed") is not None:
        return None
    detail = row.get("detail") or ""
    if TRUNCATED in detail:
        return "truncated"
    if SANDBOX_UNAVAILABLE in detail:
        return "sandbox"
    return "ungraded"


#: Whether the tasks in a category are independent samples of one another.
#: The cache-eligible repeats deliberately are not: they are one long shared
#: prefix asked eight different questions, run in order so the later turns find
#: a warm cache. Twelve such pairs are twelve *observations* but nothing like
#: twelve independent tasks, and a category that reaches the sample floor purely
#: on repeats has not shown breadth. Reported, not silently averaged in.
INDEPENDENT_SAMPLES = {"design": True, "coding": True, "math": True,
                       "research": True, "summarisation": True, "cache_repeat": False}


def index_rows(rows: list[dict]) -> dict[tuple[str, str], dict]:
    """``(task_id, arm) -> row``, last write wins.

    De-duplicating on the key is what makes it safe to concatenate two ledgers:
    a row that appears in both is one observation, not two. It is also the one
    place a rerun could quietly replace an inconvenient outcome, so every
    collapsed duplicate is counted and reported - see ``duplicates``.
    """
    return {(row["task_id"], row["arm"]): row for row in rows}


def duplicates(rows: list[dict], arms: tuple[str, ...]) -> dict[str, dict]:
    """Collapsed observations per category, with the (task, arm) key of each.

    Counting task ids understated it: three rows for one key are *two* collapsed
    observations, and two arms of one task are two keys, not one task. Both were
    reported as "1" until an independent reviewer said so.
    """
    seen: dict[tuple[str, str], int] = {}
    category: dict[str, str] = {}
    for row in rows:
        if row["arm"] not in arms:
            continue
        key = (row["task_id"], row["arm"])
        seen[key] = seen.get(key, 0) + 1
        category[row["task_id"]] = row["category"]
    out: dict[str, dict] = {}
    for (task_id, arm), count in sorted(seen.items()):
        if count <= 1:
            continue
        entry = out.setdefault(category[task_id], {"collapsed": 0, "keys": [], "task_ids": []})
        entry["collapsed"] += count - 1
        entry["keys"].append([task_id, arm])
        if task_id not in entry["task_ids"]:
            entry["task_ids"].append(task_id)
    return out


def pair_accounting(rows: list[dict], router_arm: str, comparator_arm: str,
                    *, manifest: list[dict] | None = None) -> dict[str, dict]:
    """Per-category pair counts for one (routing policy, comparator) comparison.

    ``manifest`` is the registered task list. When it is given it - not the
    ledger - defines which tasks were attempted, so a task with no rows at all
    is reported as ``invalid_incomplete`` instead of disappearing.
    """
    indexed = index_rows(rows)
    categories = {row["category"] for row in rows}
    task_category = {row["task_id"]: row["category"] for row in rows}
    task_ids: dict[str, list[str]] = {}
    for entry in manifest or []:
        task_category.setdefault(entry["id"], entry["category"])
        bucket = task_ids.setdefault(entry["category"], [])
        if entry["id"] not in bucket:
            bucket.append(entry["id"])
        categories.add(entry["category"])
    registered = {t["id"] for t in manifest} if manifest else None
    unregistered: dict[str, list[str]] = {}
    for (task_id, arm), row in indexed.items():
        if arm not in (router_arm, comparator_arm):
            continue
        if registered is not None and task_id not in registered:
            # A row for a task nobody registered is reported, never analysed.
            # Adding it to the universe was exactly the hole the manifest was
            # supposed to close.
            bucket = unregistered.setdefault(row["category"], [])
            if task_id not in bucket:
                bucket.append(task_id)
            continue
        bucket = task_ids.setdefault(row["category"], [])
        if task_id not in bucket:
            bucket.append(task_id)
    dup = duplicates(rows, (router_arm, comparator_arm))

    out: dict[str, dict] = {}
    for category in sorted(set(CATEGORIES) | categories):
        entry = {
            "attempted_pairs": 0, "valid_pairs": 0,
            "invalid_truncated": 0, "invalid_sandbox": 0,
            "invalid_incomplete": 0, "invalid_other": 0,
            "router_passed": 0, "comparator_passed": 0,
            "both_passed": 0, "router_only_passed": 0, "comparator_only_passed": 0,
            "neither_passed": 0,
            "invalid_task_ids": [],
            "independent_samples": INDEPENDENT_SAMPLES.get(category, True),
            "duplicate_observations": dup.get(category, {}).get("collapsed", 0),
            "duplicate_keys": dup.get(category, {}).get("keys", []),
            "duplicate_task_ids": dup.get(category, {}).get("task_ids", []),
            "unregistered_task_ids": sorted(unregistered.get(category, [])),
        }
        strict = {"valid_pairs": 0, "router_passed": 0, "comparator_passed": 0,
                  "router_only_passed": 0, "comparator_only_passed": 0}
        for task_id in sorted(task_ids.get(category, [])):
            assert task_category[task_id] == category
            left = indexed.get((task_id, router_arm))
            right = indexed.get((task_id, comparator_arm))
            entry["attempted_pairs"] += 1
            reasons = {_exclusion(left), _exclusion(right)} - {None}
            # Sensitivity: the same pair under the harsher rule, where a
            # provider that never finished its answer has failed the task.
            if reasons <= {"truncated"} and left is not None and right is not None:
                strict["valid_pairs"] += 1
                sl, sr = bool(left["passed"]), bool(right["passed"])
                strict["router_passed"] += sl
                strict["comparator_passed"] += sr
                strict["router_only_passed"] += sl and not sr
                strict["comparator_only_passed"] += sr and not sl
            if not reasons:
                entry["valid_pairs"] += 1
                lp, rp = bool(left["passed"]), bool(right["passed"])
                entry["router_passed"] += lp
                entry["comparator_passed"] += rp
                entry["both_passed"] += lp and rp
                entry["router_only_passed"] += lp and not rp
                entry["comparator_only_passed"] += rp and not lp
                entry["neither_passed"] += (not lp) and (not rp)
                continue
            entry["invalid_task_ids"].append(task_id)
            # One pair, one reason: truncation is reported first because it is
            # the failure mode this evaluation is known to lose rows to.
            if "truncated" in reasons:
                entry["invalid_truncated"] += 1
            elif "sandbox" in reasons:
                entry["invalid_sandbox"] += 1
            elif "incomplete" in reasons:
                entry["invalid_incomplete"] += 1
            else:
                entry["invalid_other"] += 1
        entry["if_truncation_counted_as_failure"] = strict
        out[category] = entry
    return out


def mcnemar_discordant(entry: dict) -> tuple[int, int]:
    """The two discordant cells of a paired binary comparison."""
    return entry["router_only_passed"], entry["comparator_only_passed"]


def paired_difference_ci(entry: dict, conf: float = 0.95) -> tuple[float, float]:
    """A 95% interval for the paired difference in pass rate, router - comparator.

    Two separate Wilson intervals do not say anything about the *within-task*
    difference, which is the quantity the paired design exists to measure: the
    concordant pairs carry no information about it and the discordant ones are
    not independent of them. This is the standard conditional construction -
    an exact Clopper-Pearson interval on the split of the discordant pairs,
    scaled by the share of pairs that are discordant. It is conditional on the
    number of discordant pairs, which is stated wherever it is printed.
    """
    n = entry.get("valid_pairs") or 0
    b = entry.get("router_only_passed", 0)
    c = entry.get("comparator_only_passed", 0)
    m = b + c
    if n == 0:
        return (-1.0, 1.0)
    # The difference is (share of pairs that disagree) x (2 x router's share of
    # those - 1). Both factors are estimated, so both get an exact interval and
    # the product is taken over the corners. Conditioning on the OBSERVED
    # discordant count - which the first version did - returned [0, 0] for
    # twelve concordant pairs, i.e. "proven equal" from no disagreement at all.
    # An independent reviewer caught it. This is conservative rather than exact,
    # and is labelled that way wherever it is printed.
    d_low, d_high = _clopper_pearson(m, n, conf)
    if m == 0:
        return (round(-d_high, 4), round(d_high, 4))
    s_low, s_high = _clopper_pearson(b, m, conf)
    corners = [d * (2 * split - 1)
               for d in (d_low, d_high) for split in (s_low, s_high)]
    point = (b - c) / n
    return (round(min(min(corners), point), 4), round(max(max(corners), point), 4))


def _clopper_pearson(k: int, n: int, conf: float) -> tuple[float, float]:
    """Exact binomial interval, computed by bisection on the binomial tail.

    No SciPy in this project's dependencies, and a normal approximation at
    n = 3 would be exactly the kind of number this evaluation refuses to print.
    """
    import math
    alpha = 1.0 - conf

    def tail_ge(p: float) -> float:    # P[X >= k]
        return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1))

    def tail_le(p: float) -> float:    # P[X <= k]
        return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(0, k + 1))

    low = 0.0 if k == 0 else _bisect(lambda p: tail_ge(p) - alpha / 2)
    high = 1.0 if k == n else _bisect(lambda p: tail_le(p) - alpha / 2)
    return low, high


def _bisect(f, lo: float = 0.0, hi: float = 1.0, steps: int = 60) -> float:
    for _ in range(steps):
        mid = (lo + hi) / 2
        if f(lo) * f(mid) <= 0:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def sign_test_p(b: int, c: int) -> float | None:
    """Exact two-sided binomial sign test on the discordant pairs.

    ``None`` when there are no discordant pairs at all: with nothing to
    disagree about there is no test, and reporting ``p = 1.0`` would look like
    a result rather than the absence of one.
    """
    import math
    n = b + c
    if n == 0:
        return None
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)
