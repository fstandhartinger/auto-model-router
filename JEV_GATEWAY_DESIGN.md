# Neutral multi-provider Jev gateway — design only

**Decision:** grow `jev-cascade` into a client-side gateway. Do not launch a
paid, ranking-driven hosted endpoint. Benchmark Heaven publishes the ranking;
selling an endpoint that chooses winners would damage the neutrality that
makes JevBench valuable.

## Client-side product (recommended)

Ship a library plus `jev-gateway` CLI, with an OpenAI/TypeSafe-compatible local
HTTP endpoint. Users bring provider keys through environment-variable names;
keys, prompts and payments never pass through us. A configuration pins a
primary backend and an ordered fallback list. Fallback occurs only for timeout,
429, quota exhaustion, 5xx, or an invalid response—not because our ranking
quietly prefers another entrant.

The existing `jev-cascade` repository already has most primitives: local Laya,
Jev, djev and generic TypeSafe-compatible backends, ordered failover, policies,
budgets and prompt-free audit logs. Work remaining:

- normalize all current hosted variants and preserve native probabilities;
- add health probes, retry budgets and circuit breakers;
- expose `/v1/systemone`, a local CLI, and a small SDK;
- add encrypted-at-rest optional key storage, while keeping environment keys as
  the default;
- package signed installers and build a provider contract test suite.

Estimate: **5–8 engineer-days** for a solid CLI/library beta from `jev-cascade`,
then **2–3 weeks** for signed cross-platform installers, provider conformance,
documentation and update handling.

## Hosted form

A stateless BYOK pass-through needs no inference fleet. At low volume, one or
two small gateway instances, logs/metrics and egress are roughly **$20–100 per
month**, before support, security review and on-call time. A reliable public
service needs at least two regions/instances, secret handling, abuse controls
and incident response; engineering dominates the compute bill.

If we pay providers and rebill users, provider inference becomes a variable
cost plus payment fees, fraud, credit risk and margin. Prior self-hosting work
also found that a redundant GPU service needs hundreds of thousands of daily
decisions before it clearly beats existing Jev tariffs. There is no measured
demand at that scale.

More importantly, we publish JevBench. If our hosted endpoint uses that ranking
to allocate traffic, we rank vendors, select winners and earn from the flow we
created. Disclosure does not remove the conflict. **Recommendation: do not
offer that product.** If a hosted convenience layer is ever built, make it
BYOK pass-through only: user-pinned primary and fallbacks, no ranking-derived
default, no paid placement, and no money flowing through us to model vendors.

## One-click local runner

The current Python/PyTorch Laya integration proves correctness but is too large
and slow for a consumer installer: about **1.7 GB** of weights, **2.9 GB** peak
RAM here, and **15.3/21.4 s** p50/p95 for the router's seven-question decision.

The product path is a **Rust or Go launcher embedding ONNX Runtime**, using the
151M openJev Verdict encoder exported to ONNX and dynamically quantized to
INT8. Fetch weights on first run with a pinned revision and SHA-256, store them
in the OS model cache, and expose both a tray/CLI control and localhost HTTP.
Expected payload: roughly **150–220 MB** quantized weights plus **30–80 MB** for
the signed runner/runtime; target working memory **0.5–1.0 GB**. These are
engineering estimates until an export is validated against the model's native
probabilities.

Platform plan:

- **Windows 10/11 x64:** signed MSIX/winget package, DirectML optional but CPU
  is the baseline. Handle SmartScreen reputation, long cache paths, corporate
  proxy certificates and AV scanning of first-run downloads.
- **macOS 13+ Intel and Apple Silicon:** notarized universal app/pkg. Ship both
  ONNX Runtime architectures, use the Application Support cache, and account
  for Gatekeeper quarantine and Rosetta only as a last resort.
- **Linux x64/arm64:** AppImage plus tarball; build against an old glibc, avoid
  system Python, respect XDG cache paths, and provide a systemd-user unit
  without requiring root. Musl distributions may need a separate static build.

Prototype sequence: export and compare 534 JevBench probabilities; quantize and
repeat the accuracy/calibration check; benchmark p50/p95 on one low-end x64 and
one Apple Silicon machine; then package auto-update and rollback. Estimate
**8–12 engineer-days** for a validated two-platform prototype and **3–5 weeks**
for signed Windows/macOS/Linux releases. Do not advertise “every machine” until
x64 and arm64 CI plus those low-end hardware tests pass.
