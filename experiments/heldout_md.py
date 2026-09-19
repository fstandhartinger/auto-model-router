"""The markdown report for a held-out run, including the separation question.

``heldout.report`` produces the machine-readable ``report.json``: per-category
pass rates, Wilson 95 % intervals and measured spend, all derived from the
ledger. This module turns that into the document a reader reads, and adds the
one thing a table of per-arm intervals cannot state for itself:

    does any category show a significant separation between the arms?

Two answers are given because they answer different questions, and reporting
only one of them would be a choice made after seeing the numbers:

**Overlap of the per-arm Wilson intervals.** This is the comparison the
analysis plan pre-registers ("Wilson 95 % interval per category"). It is
*conservative* for this design: it throws away the pairing, treating two arms
that answered the identical prompt as independent samples. Two overlapping
intervals therefore do not establish equality, and non-overlap is a strong
signal rather than a borderline one.

**The paired view.** The run is paired by construction, so the unit of evidence
is the pair and the informative cells are the discordant ones: tasks where
exactly one arm passed. ``experiments.pairing`` counts them, gives the
conservative interval on the paired difference and the exact sign-test p. A
category can sit far from significance on both and still not be evidence of
equality - that is a separate claim, needing a margin nobody registered.

Nothing here computes a pass rate of its own. Every figure is read back from
``report.json`` and from the ledger.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments import heldout, pairing                       # noqa: E402


def intervals_overlap(a: list[float], b: list[float]) -> bool:
    return not (a[1] < b[0] or b[1] < a[0])


def separation(report: dict, router_arm: str, comparator_arm: str) -> dict:
    """Per category: do the two arms' Wilson intervals overlap, and by how much?"""
    out: dict[str, dict] = {}
    for category, arms in report["per_category"].items():
        left, right = arms.get(router_arm), arms.get(comparator_arm)
        if not left or not right:
            continue
        overlap = intervals_overlap(left["wilson_95"], right["wilson_95"])
        out[category] = {
            "router": {"n": left["n"], "passed": left["passed"],
                       "pass_rate": left["pass_rate"], "wilson_95": left["wilson_95"]},
            "comparator": {"n": right["n"], "passed": right["passed"],
                           "pass_rate": right["pass_rate"], "wilson_95": right["wilson_95"]},
            "intervals_overlap": overlap,
            "separated": not overlap,
            "meets_ten_task_floor": min(left["n"], right["n"]) >= 10,
        }
    return out


def _paired(run_dir: Path, router_arm: str, comparator_arm: str) -> dict:
    rows = [json.loads(line) for line in (run_dir / "ledger.jsonl").read_text().splitlines()
            if line.strip()]
    manifest = heldout.load_tasks(run_dir)
    return pairing.pair_accounting(rows, router_arm, comparator_arm, manifest=manifest)


#: The pre-registered floor, applied to the unit this design actually analyses.
PAIR_FLOOR = 10


def build(run_dir: Path, router_arm: str = "router",
          comparator_arm: str = "control-metered") -> dict:
    report = heldout.report(run_dir)
    pairs = _paired(run_dir, router_arm, comparator_arm)
    sep = separation(report, router_arm, comparator_arm)
    # Ten graded rows on each arm is not ten pairs: a truncation on one arm and
    # a different truncation on the other leave both arms at ten graded rows
    # and the category at nine valid pairs. The floor is judged on pairs.
    for category, entry in sep.items():
        entry["valid_pairs"] = (pairs.get(category) or {}).get("valid_pairs", 0)
        entry["meets_ten_task_floor"] = entry["valid_pairs"] >= PAIR_FLOOR
    return {
        "run_dir": str(run_dir),
        "router_arm": router_arm,
        "comparator_arm": comparator_arm,
        "report": report,
        "separation": sep,
        "pairs": pairs,
    }


def _spend_by_arm(report: dict) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for arms in report["per_category"].values():
        for arm, entry in arms.items():
            cost = entry["metered_cost_usd"]
            if cost is None:
                out.setdefault(arm, None)
            else:
                out[arm] = round((out.get(arm) or 0.0) + cost, 6)
    return out


def format_markdown(data: dict) -> str:
    report = data["report"]
    router_arm, comparator_arm = data["router_arm"], data["comparator_arm"]
    prereg = json.loads((Path(data["run_dir"]) / "preregistration.json").read_text())
    lines: list[str] = []
    add = lines.append

    add(f"# Held-out evaluation — task set `{report['preregistration_sha256'][:16]}`")
    add("")
    add(f"Registered **{report['registered_at']}**, seed `{prereg['seed']}`, "
        f"{prereg['task_count']} tasks. Report generated {report['generated_at']}.")
    add("")
    add("| category | tasks registered | grader |")
    add("|---|---:|---|")
    for category in heldout.CATEGORIES:
        add(f"| {category} | {prereg['tasks_by_category'].get(category, 0)} | "
            f"{prereg['grader_by_category'].get(category, '-')} |")
    add("")

    # -- the headline table ------------------------------------------------
    add("## Per-category result")
    add("")
    add("| category | arm | n | passed | pass rate | Wilson 95 % | measured USD | grader |")
    add("|---|---|---:|---:|---:|---|---:|---|")
    for category, arms in report["per_category"].items():
        for arm, entry in arms.items():
            low, high = entry["wilson_95"]
            cost = ("$0.0000 (free route)" if entry["metered_cost_usd"] is None
                    else f"${entry['metered_cost_usd']:.4f}")
            add(f"| {category} | {arm} | {entry['n']} | {entry['passed']} | "
                f"{entry['pass_rate']:.2f} | {low:.2f}–{high:.2f} | {cost} | "
                f"{entry['grader_kind']} |")
    add("")
    spend = _spend_by_arm(report)
    add("**Measured spend by arm:** " + " · ".join(
        f"{arm} **{'$0.0000 (every route free)' if usd is None else f'${usd:.4f}'}**"
        for arm, usd in spend.items()))
    add("")
    add(f"Graded rows {report['graded_rows']}, excluded {report['excluded_rows']}"
        + (f". Exclusions: {'; '.join(report['exclusion_reasons'])}"
           if report["exclusion_reasons"] else "."))
    add("")

    # -- the separation question ------------------------------------------
    add(f"## Does any category separate `{router_arm}` from `{comparator_arm}`?")
    add("")
    add("| category | valid pairs | router | comparator | Wilson intervals | "
        "discordant b/c | paired difference (95 %) | sign test p |")
    add("|---|---:|---|---|---|---|---|---:|")
    separated: list[str] = []
    # The floor is judged over every registered category, not only over the
    # ones that made it into the table: a category with one arm missing
    # entirely has no separation row at all, and must not vanish from this list.
    registered = set(prereg.get("tasks_by_category") or {}) or set(heldout.CATEGORIES)
    under_floor: list[str] = sorted(
        c for c in registered
        if (data["pairs"].get(c) or {}).get("valid_pairs", 0) < PAIR_FLOOR)
    for category in heldout.CATEGORIES:
        sep = data["separation"].get(category)
        if not sep:
            continue
        entry = data["pairs"].get(category, {})
        b, c = pairing.mcnemar_discordant(entry) if entry else (0, 0)
        low, high = pairing.paired_difference_ci(entry) if entry else (0.0, 0.0)
        p = pairing.sign_test_p(b, c) if entry else None
        overlap = "overlap" if sep["intervals_overlap"] else "**DO NOT overlap**"
        if sep["separated"]:
            separated.append(category)
        add(f"| {category} | {entry.get('valid_pairs', 0)} | "
            f"{sep['router']['passed']}/{sep['router']['n']} "
            f"({sep['router']['wilson_95'][0]:.2f}–{sep['router']['wilson_95'][1]:.2f}) | "
            f"{sep['comparator']['passed']}/{sep['comparator']['n']} "
            f"({sep['comparator']['wilson_95'][0]:.2f}–{sep['comparator']['wilson_95'][1]:.2f}) | "
            f"{overlap} | [{b}, {c}] | {low:+.3f} to {high:+.3f} | "
            f"{'n/a' if p is None else f'{p:.3f}'} |")
    add("")
    below = [c for c in separated if c in under_floor]
    separated = [c for c in separated if c not in under_floor]
    if below:
        add("Non-overlapping intervals below the ten-pair floor, reported but NOT "
            "counted as a separation: " + ", ".join(below) + ".")
        add("")
    if separated:
        add("**Answer: yes, in " + ", ".join(separated) + ".** In "
            + ("that category" if len(separated) == 1 else "those categories")
            + " the two arms' Wilson 95 % intervals do not overlap. Read the paired "
              "columns beside it before treating that as the size of the effect: the "
              "interval comparison discards the pairing and the paired difference is "
              "the quantity this design actually estimates.")
    else:
        add("**Answer: no.** In every category at or above the floor the two arms' "
            "Wilson 95 % intervals overlap, so no category separates the routing policy from the metered "
            "comparator at this sample size. That is not evidence that they are "
            "equal: an overlap is the absence of a demonstrated difference, and the "
            "non-inferiority claim that *would* say they are equivalent needs a "
            "margin, which was never pre-registered and cannot honestly be chosen now "
            "that the results are in.")
    add("")
    if under_floor:
        add("Below the pre-registered floor of ten valid pairs, so no quality claim is "
            "made for: " + ", ".join(sorted(under_floor)) + ".")
    else:
        add("Every category reaches the pre-registered floor of ten valid pairs (both "
            "arms graded on the same task) inside this single registration — no pooling "
            "across registrations, and no amendment made after an outcome was seen.")
    add("")
    duplicated = {c: e["duplicate_keys"] for c, e in data["pairs"].items()
                  if e.get("duplicate_observations")}
    unregistered = {c: e["unregistered_task_ids"] for c, e in data["pairs"].items()
                    if e.get("unregistered_task_ids")}
    add("Ledger integrity: "
        + ("**duplicate task/arm rows** " + json.dumps(duplicated) if duplicated
           else "every task/arm pair appears at most once")
        + "; " + ("**rows for unregistered tasks** " + json.dumps(unregistered)
                  if unregistered else "no row for an unregistered task") + ".")
    add("")
    incomplete = {c: e["invalid_incomplete"] for c, e in data["pairs"].items()
                  if e.get("invalid_incomplete")}
    if incomplete:
        add("Registered tasks with a missing arm (not run, e.g. the spend cap was reached): "
            + "; ".join(f"{c}: {n}" for c, n in incomplete.items()) + ".")
        add("")

    # -- sensitivity -------------------------------------------------------
    strict_lines = []
    for category in heldout.CATEGORIES:
        entry = data["pairs"].get(category) or {}
        strict = entry.get("if_truncation_counted_as_failure") or {}
        if entry.get("invalid_truncated"):
            strict_lines.append(
                f"| {category} | {entry['router_passed']}/{entry['valid_pairs']} vs "
                f"{entry['comparator_passed']}/{entry['valid_pairs']} | "
                f"{strict.get('router_passed')}/{strict.get('valid_pairs')} vs "
                f"{strict.get('comparator_passed')}/{strict.get('valid_pairs')} |")
    if strict_lines:
        add("### Sensitivity: truncation scored as a failure instead of excluded")
        add("")
        add("The registered rule excludes a truncated answer. That rule was written "
            "after a truncation had already been seen, so the opposite rule is "
            "reported beside it; a conclusion that survives only one of the two is "
            "not a conclusion.")
        add("")
        add("| category | registered rule | truncation-as-failure |")
        add("|---|---|---|")
        lines.extend(strict_lines)
        add("")

    add("## Not claimed")
    add("")
    for claim in report["claims_not_made"]:
        add(f"- {claim}")
    if report.get("amendments"):
        add("")
        add("## Amendments after registration")
        add("")
        for amendment in report["amendments"]:
            add(f"- {amendment['at']}: {amendment['reason']}")
    if report.get("harness_drift_since_registration"):
        add("")
        add("**UNRECORDED harness drift since registration:** "
            + ", ".join(report["harness_drift_since_registration"]))
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dir", type=Path, required=True)
    parser.add_argument("--router-arm", default="router")
    parser.add_argument("--comparator-arm", default="control-metered")
    args = parser.parse_args(argv)
    data = build(args.dir, args.router_arm, args.comparator_arm)
    text = format_markdown(data)
    (args.dir / "report.md").write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
