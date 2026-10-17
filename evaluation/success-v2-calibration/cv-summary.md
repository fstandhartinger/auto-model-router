### 10-fold, grouped by task

Log-loss (lower is better), weighted by fold size; in brackets the best and worst fold.

| model | log-loss | Brier | worst fold | best fold |
|---|---|---|---|---|
| const | 0.4207 | 0.1259 | 0.638 | 0.198 |
| v1 | 0.5004 | 0.1304 | 0.820 | 0.132 |
| v1-curve | 0.5299 | 0.1480 | 0.913 | 0.140 |
| v2-nocap | 0.3944 | 0.1220 | 0.734 | 0.188 |
| v2-nolength | 0.3994 | 0.1241 | 0.712 | 0.195 |
| v2 | 0.3951 | 0.1222 | 0.732 | 0.188 |
| v2+measured | 0.3921 | 0.1212 | 0.704 | 0.183 |

Fold-level log-loss:

| fold | n | const | v1 | v1-curve | v2-nocap | v2-nolength | v2 | v2+measured |
|---|---|---|---|---|---|---|---|---|
| task-fold-0 | 72 | 0.428 | 0.642 | 0.552 | 0.414 | 0.414 | 0.415 | 0.421 |
| task-fold-1 | 72 | 0.556 | 0.738 | 0.743 | 0.734 | 0.712 | 0.732 | 0.704 |
| task-fold-2 | 72 | 0.478 | 0.710 | 0.672 | 0.441 | 0.449 | 0.439 | 0.448 |
| task-fold-3 | 72 | 0.453 | 0.545 | 0.595 | 0.370 | 0.392 | 0.373 | 0.372 |
| task-fold-4 | 72 | 0.403 | 0.461 | 0.461 | 0.316 | 0.326 | 0.314 | 0.314 |
| task-fold-5 | 72 | 0.198 | 0.132 | 0.140 | 0.188 | 0.195 | 0.188 | 0.183 |
| task-fold-6 | 72 | 0.638 | 0.820 | 0.913 | 0.562 | 0.573 | 0.572 | 0.577 |
| task-fold-7 | 72 | 0.309 | 0.222 | 0.380 | 0.271 | 0.269 | 0.272 | 0.264 |
| task-fold-8 | 72 | 0.504 | 0.564 | 0.579 | 0.421 | 0.427 | 0.419 | 0.419 |
| task-fold-9 | 72 | 0.241 | 0.171 | 0.264 | 0.227 | 0.237 | 0.227 | 0.218 |

### leave-one-wave-out (2 folds)

Log-loss (lower is better), weighted by fold size; in brackets the best and worst fold.

| model | log-loss | Brier | worst fold | best fold |
|---|---|---|---|---|
| const | 0.4302 | 0.1282 | 0.493 | 0.367 |
| v1 | 0.5004 | 0.1304 | 0.548 | 0.452 |
| v1-curve | 0.5299 | 0.1480 | 0.584 | 0.476 |
| v2-nocap | 0.4550 | 0.1395 | 0.480 | 0.429 |
| v2-nolength | 0.4231 | 0.1314 | 0.484 | 0.363 |
| v2 | 0.4364 | 0.1357 | 0.480 | 0.392 |
| v2+measured | 0.4324 | 0.1341 | 0.480 | 0.385 |

Fold-level log-loss:

| fold | n | const | v1 | v1-curve | v2-nocap | v2-nolength | v2 | v2+measured |
|---|---|---|---|---|---|---|---|---|
| train-wave1-test-wave2 | 360 | 0.493 | 0.548 | 0.584 | 0.480 | 0.484 | 0.480 | 0.480 |
| train-wave2-test-wave1 | 360 | 0.367 | 0.452 | 0.476 | 0.429 | 0.363 | 0.392 | 0.385 |

### leave-one-model-out (6 folds; the held-out model is unseen in training)

Log-loss (lower is better), weighted by fold size; in brackets the best and worst fold.

| model | log-loss | Brier | worst fold | best fold |
|---|---|---|---|---|
| const | 0.4221 | 0.1262 | 0.620 | 0.312 |
| v1 | 0.5004 | 0.1304 | 0.848 | 0.341 |
| v1-curve | 0.5299 | 0.1480 | 0.835 | 0.334 |
| v2-nocap | 0.3766 | 0.1163 | 0.550 | 0.290 |
| v2-nolength | 0.3939 | 0.1225 | 0.575 | 0.295 |
| v2 | 0.3805 | 0.1179 | 0.552 | 0.288 |
| v2+measured | 0.3800 | 0.1176 | 0.560 | 0.287 |

Fold-level log-loss:

| fold | n | const | v1 | v1-curve | v2-nocap | v2-nolength | v2 | v2+measured |
|---|---|---|---|---|---|---|---|---|
| leave-out-claude-opus-5 | 120 | 0.312 | 0.344 | 0.334 | 0.294 | 0.295 | 0.294 | 0.287 |
| leave-out-deepseek-v4.1-flash | 120 | 0.500 | 0.535 | 0.535 | 0.428 | 0.436 | 0.455 | 0.455 |
| leave-out-glm-5.3-flash | 120 | 0.620 | 0.848 | 0.835 | 0.550 | 0.575 | 0.552 | 0.560 |
| leave-out-gpt-5.6-luna | 120 | 0.338 | 0.341 | 0.600 | 0.322 | 0.375 | 0.318 | 0.315 |
| leave-out-gpt-5.6-sol | 120 | 0.325 | 0.432 | 0.356 | 0.290 | 0.309 | 0.288 | 0.289 |
| leave-out-kimi-k3 | 120 | 0.438 | 0.503 | 0.519 | 0.375 | 0.374 | 0.375 | 0.375 |

### Offline router replay on held-out tasks (the picks of irp.rank() made by a model trained without that task; median AntSeed seller prices; outcomes as recorded)

**10-fold grouped by task, 120 tasks** (cost shown as USD total and % of always-Opus)

| arm | passed | cost USD | % of Opus cost |
|---|---|---|---|
| always-claude-opus-5 | 110 | 1.011 | 100.0 |
| always-gpt-5.6-sol | 109 | 0.318 | 31.5 |
| always-gpt-5.6-luna | 108 | 0.046 | 4.5 |
| always-kimi-k3 | 101 | 0.572 | 56.5 |
| always-deepseek-v4.1-flash | 97 | 0.083 | 8.2 |
| always-glm-5.3-flash | 90 | 0.078 | 7.7 |
| oracle (cheapest passing model per task) | 114 | 0.072 | 7.2 |
| router v1 t0 | 100 | 0.195 | 19.3 |
| router v1 t1 | 100 | 0.218 | 21.5 |
| router v1 t2 | 98 | 0.096 | 9.5 |
| router v1 t3 | 95 | 0.070 | 6.9 |
| router v1 t5 | 95 | 0.068 | 6.7 |
| router v1 t7 | 95 | 0.068 | 6.7 |
| router v1 t10 | 90 | 0.078 | 7.7 |
| router v2 t0 | 107 | 0.252 | 24.9 |
| router v2 t1 | 108 | 0.046 | 4.5 |
| router v2 t2 | 108 | 0.046 | 4.5 |
| router v2 t3 | 108 | 0.046 | 4.6 |
| router v2 t5 | 106 | 0.047 | 4.7 |
| router v2 t7 | 95 | 0.066 | 6.5 |
| router v2 t10 | 90 | 0.078 | 7.7 |
| router v2+measured t0 | 108 | 0.286 | 28.3 |
| router v2+measured t1 | 108 | 0.075 | 7.4 |
| router v2+measured t2 | 108 | 0.047 | 4.6 |
| router v2+measured t3 | 107 | 0.047 | 4.6 |
| router v2+measured t5 | 104 | 0.049 | 4.8 |
| router v2+measured t7 | 95 | 0.069 | 6.8 |
| router v2+measured t10 | 90 | 0.078 | 7.7 |

Per fold (passed / cost USD, t0 and t2):

| fold | tasks | v1 t0 | v2 t0 | v2+m t0 | v1 t2 | v2 t2 | v2+m t2 | always-Opus | always-Sol | always-Luna |
|---|---|---|---|---|---|---|---|---|---|---|
| task-fold-0 | 12 | 8 / 0.010 | 11 / 0.060 | 11 / 0.060 | 8 / 0.010 | 11 / 0.008 | 11 / 0.008 | 11 / 0.170 | 11 / 0.060 | 11 / 0.008 |
| task-fold-1 | 12 | 11 / 0.047 | 9 / 0.006 | 10 / 0.012 | 10 / 0.008 | 9 / 0.006 | 9 / 0.005 | 11 / 0.138 | 11 / 0.032 | 9 / 0.006 |
| task-fold-2 | 12 | 10 / 0.002 | 10 / 0.004 | 10 / 0.007 | 10 / 0.002 | 10 / 0.002 | 10 / 0.002 | 10 / 0.037 | 10 / 0.009 | 10 / 0.002 |
| task-fold-3 | 12 | 8 / 0.038 | 10 / 0.053 | 10 / 0.053 | 9 / 0.012 | 12 / 0.008 | 12 / 0.008 | 12 / 0.113 | 10 / 0.053 | 12 / 0.008 |
| task-fold-4 | 12 | 10 / 0.007 | 11 / 0.006 | 11 / 0.009 | 10 / 0.007 | 11 / 0.004 | 11 / 0.004 | 11 / 0.066 | 11 / 0.020 | 11 / 0.004 |
| task-fold-5 | 12 | 12 / 0.005 | 12 / 0.013 | 12 / 0.013 | 12 / 0.005 | 12 / 0.002 | 12 / 0.002 | 12 / 0.045 | 12 / 0.013 | 12 / 0.002 |
| task-fold-6 | 12 | 8 / 0.036 | 10 / 0.075 | 10 / 0.075 | 8 / 0.035 | 10 / 0.007 | 10 / 0.007 | 9 / 0.251 | 10 / 0.075 | 10 / 0.007 |
| task-fold-7 | 12 | 11 / 0.003 | 11 / 0.009 | 11 / 0.009 | 11 / 0.003 | 11 / 0.001 | 11 / 0.002 | 11 / 0.032 | 11 / 0.009 | 11 / 0.001 |
| task-fold-8 | 12 | 10 / 0.042 | 11 / 0.015 | 11 / 0.035 | 9 / 0.010 | 10 / 0.006 | 10 / 0.006 | 11 / 0.117 | 11 / 0.036 | 10 / 0.006 |
| task-fold-9 | 12 | 12 / 0.006 | 12 / 0.012 | 12 / 0.012 | 11 / 0.005 | 12 / 0.003 | 12 / 0.003 | 12 / 0.041 | 12 / 0.012 | 12 / 0.003 |

**leave-one-wave-out, 120 tasks** (cost shown as USD total and % of always-Opus)

| arm | passed | cost USD | % of Opus cost |
|---|---|---|---|
| always-claude-opus-5 | 110 | 1.011 | 100.0 |
| always-gpt-5.6-sol | 109 | 0.318 | 31.5 |
| always-gpt-5.6-luna | 108 | 0.046 | 4.5 |
| always-kimi-k3 | 101 | 0.572 | 56.5 |
| always-deepseek-v4.1-flash | 97 | 0.083 | 8.2 |
| always-glm-5.3-flash | 90 | 0.078 | 7.7 |
| oracle (cheapest passing model per task) | 114 | 0.072 | 7.2 |
| router v1 t0 | 100 | 0.195 | 19.3 |
| router v1 t1 | 100 | 0.218 | 21.5 |
| router v1 t2 | 98 | 0.096 | 9.5 |
| router v1 t3 | 95 | 0.070 | 6.9 |
| router v1 t5 | 95 | 0.068 | 6.7 |
| router v1 t7 | 95 | 0.068 | 6.7 |
| router v1 t10 | 90 | 0.078 | 7.7 |
| router v2 t0 | 107 | 0.113 | 11.2 |
| router v2 t1 | 108 | 0.046 | 4.6 |
| router v2 t2 | 108 | 0.046 | 4.6 |
| router v2 t3 | 106 | 0.047 | 4.6 |
| router v2 t5 | 97 | 0.060 | 6.0 |
| router v2 t7 | 93 | 0.075 | 7.4 |
| router v2 t10 | 90 | 0.078 | 7.7 |
| router v2+measured t0 | 110 | 0.236 | 23.3 |
| router v2+measured t1 | 106 | 0.050 | 4.9 |
| router v2+measured t2 | 106 | 0.047 | 4.6 |
| router v2+measured t3 | 106 | 0.048 | 4.7 |
| router v2+measured t5 | 95 | 0.076 | 7.5 |
| router v2+measured t7 | 93 | 0.076 | 7.5 |
| router v2+measured t10 | 90 | 0.078 | 7.7 |

Per fold (passed / cost USD, t0 and t2):

| fold | tasks | v1 t0 | v2 t0 | v2+m t0 | v1 t2 | v2 t2 | v2+m t2 | always-Opus | always-Sol | always-Luna |
|---|---|---|---|---|---|---|---|---|---|---|
| train-wave1-test-wave2 | 60 | 48 / 0.170 | 53 / 0.034 | 56 / 0.157 | 45 / 0.081 | 53 / 0.034 | 51 / 0.035 | 56 / 0.839 | 55 / 0.239 | 53 / 0.034 |
| train-wave2-test-wave1 | 60 | 52 / 0.025 | 54 / 0.079 | 54 / 0.079 | 53 / 0.015 | 55 / 0.012 | 55 / 0.012 | 54 / 0.172 | 54 / 0.080 | 55 / 0.012 |
