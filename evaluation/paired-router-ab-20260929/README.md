# Paired router A/B, 29 September 2026

This directory publishes the public-safe prompt set, paired per-task outcomes, aggregate analysis, a public protocol summary, selected router configuration, live model-price snapshot and catalog snapshot. `manifest.json` records the frozen registration hash and SHA-256 for each published artifact.

The paired CSV reports grader pass/fail, model, accounted cost and latency. Grader keys, answer text, credentials and historical budget ledgers are excluded. See `protocol.md` for scope and grading details.

## Reporting correction, 29 September 2026

Point estimates use the observed 40 task pairs. An earlier summary used the average of the bootstrap draws for means and the cost ratio. Those values remain under `bootstrap_distribution_mean` in `analysis.json`; the raw task outcomes, registration and confidence intervals are unchanged. Run `python3 check_observed.py` to verify the corrected estimates and artifact hashes.
