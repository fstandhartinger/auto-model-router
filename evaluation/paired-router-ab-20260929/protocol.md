# Public protocol summary

Study: `paired-router-ab-20260929`. The pre-inference registration SHA-256 is recorded in `manifest.json`; its task, policy, model, source and analysis-plan hashes were checked before generation, blind grading and analysis.

- **Arms:** one Claude Opus 5.5 completion through OpenRouter per task; one `F_expected` first-choice completion from the five-model API catalog through the same endpoint. Weiche-395M fp16 on CPU supplies category and difficulty.
- **Tasks:** 40 fixed public-safe tasks, ten each in coding, arithmetic, fictional-source extraction and structural HTML. The system/user text and output cap match within each pair. Task order and arm order were randomized before calls.
- **Blinding:** each response has an opaque submission ID. The deterministic grader reads the task, hidden key and anonymous answer; provider, route and arm labels are joined only after grades are frozen.
- **Graders:** code executes inside Bubblewrap; arithmetic checks the final numeric answer; extraction checks required source facts; design checks HTML structure. Provider errors and `finish_reason=length` do not pass. No retries.
- **Quality:** pass-rate difference (router minus control); report Wilson intervals per arm/category and a 10,000-draw, category-stratified paired bootstrap 95% interval, seed `20260929`.
- **Cost:** per-task accounted cost is the higher of OpenRouter-reported upstream cost and configured list-price cost. The newly collected 80 calls used `$0.269939`.
- **Latency:** end-to-end task time includes warm local classification, selection and provider response. One-time Weiche initialization is reported separately (4.595 s).

The study measures first choices only; answer verification, retries, escalation, subscription routes and local generation are outside scope. The design grader is structural rather than visual, and the synthetic task set does not estimate generation-to-generation variance or long-running agent work. The route-head's raw success-probability output is not consumed by the current `F_expected` policy. The public playground adds a value-of-time term, which this study did not exercise.
