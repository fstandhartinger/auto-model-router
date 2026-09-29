#!/usr/bin/env python3
"""Verify reported point estimates directly from the public matched outcomes."""
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

root = Path(__file__).resolve().parent
manifest = json.loads((root / 'manifest.json').read_text())
for name, expected in manifest['files_sha256'].items():
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected, name
rows = list(csv.DictReader((root / 'paired-results.csv').open()))
assert len(rows) == len({r['task_id'] for r in rows}) == 40
analysis = json.loads((root / 'analysis.json').read_text())
for metric, field in [('pass_rate', 'passed'), ('budget_cost_usd', 'budget_cost_usd'),
                      ('e2e_latency_ms', 'latency_ms')]:
    means = {}
    for arm in ['control', 'router']:
        vals = [float(r[f'{arm}_{field}'] == 'True') if field == 'passed'
                else float(r[f'{arm}_{field}']) for r in rows]
        means[arm] = statistics.mean(vals)
    means['delta_router_minus_control'] = means['router'] - means['control']
    for key, value in means.items():
        assert math.isclose(analysis['metrics'][metric][key]['mean'], value,
                            abs_tol=1e-11), (metric, key, value)
ratio = statistics.mean(float(r['router_budget_cost_usd']) for r in rows) / statistics.mean(
    float(r['control_budget_cost_usd']) for r in rows)
assert math.isclose(analysis['metrics']['cost_ratio_router_over_control']['mean'], ratio,
                    abs_tol=1e-11)
print('PASS: 40 observed pairs, all point estimates and artifact hashes agree.')
