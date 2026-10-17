#!/usr/bin/env python3
"""Summarise the AntSeed catalog snapshot into prices.json (USD per million tokens).

Primary price per model = the cheapest seller (lowest input+output price, both > 0;
zero-priced offers are excluded because they are promotions, not a price a buyer can
plan on). The median over sellers with a positive price is recorded as a sensitivity.
usage: build_prices.py SNAPSHOT.json OUT.json
"""
import hashlib
import json
import statistics
import sys

MODELS = {  # key -> (AntSeed id, OpenRouter id used to generate answers)
    "claude-opus-5": ("claude-opus-5", "anthropic/claude-opus-5"),
    "gpt-5.6-sol": ("gpt-5.6-sol", "openai/gpt-5.6-sol"),
    "gpt-5.6-luna": ("gpt-5.6-luna", "openai/gpt-5.6-luna"),
    "kimi-k3": ("kimi-k3", "moonshotai/kimi-k3"),
    "glm-5.3-flash": ("glm-5.3-flash", "z-ai/glm-5.3-flash"),
    "deepseek-v4.1-flash": ("deepseek-v4.1-flash", "deepseek/deepseek-v4.1-flash"),
}


def offer(p):
    i, o = p["inputUsdPerMillion"], p["outputUsdPerMillion"]
    c = p.get("cachedInputUsdPerMillion")
    return {"seller": p["displayName"], "input": i, "output": o, "cache_read": i if c is None else c}


def main(snapshot, out):
    raw = open(snapshot, "rb").read()
    data = {m["id"]: m for m in json.loads(raw)["data"]}
    result = {"snapshot_sha256": hashlib.sha256(raw).hexdigest(), "unit": "USD per 1M tokens", "models": {}}
    for key, (ant, orid) in MODELS.items():
        sellers = [offer(p) for p in data[ant]["peers"]]
        paid = [s for s in sellers if s["input"] > 0 and s["output"] > 0]
        cheapest = min(paid, key=lambda s: (s["input"] + s["output"], s["seller"]))
        median = {k: statistics.median(s[k] for s in paid) for k in ("input", "output", "cache_read")}
        result["models"][key] = {"antseed_id": ant, "openrouter_id": orid, "sellers_total": len(sellers),
                                 "sellers_priced": len(paid), "cheapest": cheapest, "median": median}
    with open(out, "w") as f:
        json.dump(result, f, indent=1, sort_keys=True)
        f.write("\n")


if __name__ == "__main__":
    main(*sys.argv[1:3])
