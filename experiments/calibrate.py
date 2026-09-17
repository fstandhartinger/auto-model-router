"""Turn a graded matrix into the router's measured success table.

    python experiments/calibrate.py --matrix runs/matrix.jsonl --out success.json \
        --alias gpt-5.6-sol-aiml=gpt-5.6-sol --alias claude-opus-5-aiml=claude-opus-5

Rates are smoothed towards the mean of all models for the same category and
difficulty (``--prior`` pseudo-observations), because cells hold only a few tasks.
Categories without their own tasks (knowledge, general) use the pooled rate
across all categories.
"""

import argparse
import json
from collections import defaultdict

LEVELS = ("easy", "medium", "hard")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--alias", action="append", default=[])
    ap.add_argument("--prior", type=float, default=3.0)
    args = ap.parse_args()
    alias = dict(a.split("=", 1) for a in args.alias)
    cells = defaultdict(lambda: [0, 0])
    for line in open(args.matrix):
        r = json.loads(line)
        for key in ((r["model"], r["category"], r["difficulty"]), (r["model"], "*", r["difficulty"])):
            cells[key][0] += int(r["ok"])
            cells[key][1] += 1
    models = sorted({k[0] for k in cells})
    categories = sorted({k[1] for k in cells})
    rows = []
    for cat in categories:
        for level in LEVELS:
            s = sum(cells[(m, cat, level)][0] for m in models)
            n = sum(cells[(m, cat, level)][1] for m in models)
            mean = s / n if n else 0.5
            for m in models:
                ms, mn = cells[(m, cat, level)]
                p = (ms + args.prior * mean) / (mn + args.prior)
                name = alias.get(m, m)
                if cat == "*":
                    rows += [[name, c, level, round(p, 4)] for c in ("knowledge", "general")]
                else:
                    rows.append([name, cat, level, round(p, 4)])
    with open(args.out, "w") as fh:
        fh.write("[\n" + ",\n".join(json.dumps(r) for r in rows) + "\n]\n")
    print(f"{len(rows)} rates for {len(models)} models")


if __name__ == "__main__":
    main()
