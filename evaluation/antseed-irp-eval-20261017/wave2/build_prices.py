#!/usr/bin/env python3
"""Wave-2 prices.json = wave-1 prices.json (same AntSeed snapshot of 2026-10-17, sha256 inside) plus a third basis "list":
the OpenRouter list price per model (USD per 1M tokens, read from /api/v1/models on 2026-10-17; cache read = input price,
prompts are unique so it does not matter). Bases: median seller (PRIMARY for headlines), cheapest seller, OpenRouter list.
usage: build_prices.py ../prices.json OUT.json"""
import json
import sys

OR_LIST = {"claude-opus-5": (5.0, 25.0), "gpt-5.6-sol": (4.0, 20.0), "gpt-5.6-luna": (0.2, 1.2), "kimi-k3": (0.304, 11.936),
           "glm-5.3-flash": (0.0238, 0.3966), "deepseek-v4.1-flash": (0.0153, 0.594)}
p = json.load(open(sys.argv[1]))
for m, (i, o) in OR_LIST.items():
    p["models"][m]["list"] = {"input": i, "output": o, "cache_read": i}
p["bases_note"] = "median = primary headline basis; cheapest = lowest-priced seller (promotional outliers); list = OpenRouter list price"
json.dump(p, open(sys.argv[2], "w"), indent=1, sort_keys=True)
open(sys.argv[2], "a").write("\n")
