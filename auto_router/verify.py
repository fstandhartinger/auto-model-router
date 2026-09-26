"""Check a cheap route's answer with Jev, and escalate when it falls short.

The router chooses a model; until now it never looked at what came back. For a
frontier route that is the only honest option - the judge is not smarter than
the thing it would be grading, and a judge that is weaker than the model
produces false alarms, not quality. For the cheap tier the relation is the
other way round: Jev is a typed-question model of roughly frontier-level
reasoning answering one narrow question about an answer a much smaller model
produced, and the 17 September evaluation measured it catching 85 % of wrong
coding answers and 67 % of wrong maths answers with no false alarms
(EXPERIMENTS.md section 4).

So this module verifies exactly the cases where that evidence exists:

* the answering route is in the **cheap tier** (free, ``:free``, or a list
  price at or below ``max_price_per_mtok``) and its capability evidence stays
  below ``max_capability``;
* the request is **self-contained** - the judge sees the request and the
  answer and nothing else, so a question about a document it cannot read is
  the one measured case where it produces false alarms, and is skipped;
* a judge is configured at all.

Everything else is recorded as "not verified", with the reason, and the turn
behaves exactly as it did before this module existed.

What a failed check does
------------------------
``Verdict.escalate`` is true when ``p_adequate`` is below the threshold for
that category. The router then re-runs the turn on the next candidate by its
own expected-cost ranking that is *more capable* than the one that failed
(``Router.escalate`` -> ``Policy.on_failure``), which also raises the
conversation's difficulty floor so the next turn in the same conversation does
not start below the level this one just proved it needs.

Thresholds and the measured rates that go with them live in the config under
``policy.verify``; the defaults here are the ones the calibration in
``experiments/verify_calibrate.py`` chose on the 78-task set.

The intelligence-threshold rule
-------------------------------
A second, independent reason to check an answer: the route's model is *less
intelligent than a named reference model* - by default GPT-5.6 Terra - on the
benchmark data (``policy.verify.intelligence_threshold``). The comparison is
per category when both models have a directly measured score for the request's
category, and on the headline intelligence index otherwise. A route whose
intelligence is unknown is checked. The reference's number comes from the
benchmark data (network, disk cache or bundled snapshot); a configured
``value`` is used when the reference is not in the data. This rule was set by
the operator, not calibrated: the catch and false-flag rates above were
measured on the cheap tier and are assumed, not measured, for these routes.
Answers served through a plan's own login pass through unchanged and are never
graded by either rule.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

from .catalog import ModelInfo

#: Categories whose answers the judge cannot see the evidence for. A question
#: about a pasted document is graded without the document, which the 17 Sep
#: evaluation measured as a false alarm on every single adequate answer
#: (24/24). Not a threshold problem: no threshold fixes a judge that is blind.
SKIP_CATEGORIES: tuple[str, ...] = ("long_context",)

#: Per-category adequacy thresholds. Below this, escalate. Chosen on the 18 Sep
#: calibration (EXPERIMENTS.md section 14, 192 cheap answers): the two
#: categories that were measured get their own number, and every other category
#: keeps the generic 0.30 the 17 Sep evaluation used - an operator who cares
#: about a third category should measure it rather than inherit a guess.
#:
#: The two differ because the score distributions differ. A maths answer the
#: judge approves never scored below 0.77, so a high bar there costs nothing;
#: a coding answer it approves can score anywhere, so the bar has to sit low.
DEFAULT_THRESHOLDS: dict[str, float] = {"default": 0.30, "coding": 0.25, "math": 0.60}

#: P(an inadequate answer is flagged), by category, at those thresholds. Used
#: by the cost model: a cheap route whose failures are caught cheaply is worth
#: more than one whose failures reach the user.
DEFAULT_CATCH_RATE: dict[str, float] = {"default": 0.75, "coding": 0.85, "math": 0.73}

#: P(an adequate answer is flagged anyway) - the cost side of the same trade.
#: Four of the six false flags in the coding measurement were the same task,
#: whose stated specification contradicts its own hidden tests; the rate is
#: kept as measured rather than adjusted for that.
DEFAULT_FALSE_FLAG_RATE: dict[str, float] = {"default": 0.10, "coding": 0.10, "math": 0.0}

#: P(the escalation target actually gets it right | the judge rejected the
#: first answer). Measured by re-running every flagged answer and grading it:
#: 18 of 35 overall, 16 of 28 for coding, 2 of 7 for maths (a small number,
#: kept as measured and not rounded up). This is *not* the target route's
#: success rate on the category: the turns that reach it are the ones a cheaper
#: route already failed, which is what makes them harder than average.
DEFAULT_FIX_RATE: dict[str, float] = {"default": 0.51, "coding": 0.57, "math": 0.29}

#: What one judge call costs and how long it takes. The seconds are measured
#: (192 calls, median 0.69 s, p90 0.78 s). The dollars are 1,395 input and 66
#: output tokens - both measured - priced at a small model's public rate, which
#: is an assumption about the judge's price list and is why it is a parameter.
#: Both belong in the expected-cost model: verifying is not free, it is just
#: much cheaper than being wrong.
DEFAULT_JUDGE_USD = 0.0004
DEFAULT_JUDGE_SECONDS = 0.7


def blended_price(model: ModelInfo) -> float:
    """List price in $/Mtok at the 80/20 input:output mix the router prices with."""
    return model.prices.input * 0.8 + model.prices.output * 0.2


@dataclass(frozen=True)
class VerifyPolicy:
    """Which answers get checked, against which threshold, and at what cost.

    Every field is configurable because every field is an operator's judgement
    about their own catalog: which of their routes are "cheap", how much a
    wrong answer costs them, how often they are willing to pay for a second
    attempt that turns out not to have been needed.
    """

    enabled: bool = True
    #: Blended list price at or below which a route counts as the cheap tier.
    #: Free and ``:free`` routes price at 0 and always clear it.
    max_price_per_mtok: float = 1.0
    #: Capability ceiling for the category being answered, after the evidence
    #: discount. A route the evidence already rates near the judge's own level
    #: is not one the judge can grade, whatever it costs - which is the case a
    #: free frontier-class route would otherwise walk straight through.
    max_capability: float = 72.0
    thresholds: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_THRESHOLDS))
    catch_rate: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_CATCH_RATE))
    false_flag_rate: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_FALSE_FLAG_RATE))
    fix_rate: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_FIX_RATE))
    skip_categories: tuple[str, ...] = SKIP_CATEGORIES
    #: Routes that are always / never checked, whatever the price says.
    always: tuple[str, ...] = ()
    never: tuple[str, ...] = ()
    #: Longest request the judge is asked about. Beyond this the request is a
    #: document, and the judge is being asked to grade something it cannot see.
    max_request_chars: int = 6000
    #: Give the second attempt the first answer to react to. Off by default:
    #: the calibration measured it (EXPERIMENTS.md section 14) and a failed
    #: cheap answer anchors the stronger model more often than it helps.
    carry_failed_attempt: bool = False
    #: Minimum difficulty floor written into the conversation memory after an
    #: escalation, on top of whatever the policy's own failure rule sets.
    difficulty_floor: float = 0.55
    #: Capability points the second route must clear the first one by. One
    #: point - the ordinary escalation bar - lets a catalog of similar free
    #: routes answer a rejected answer with a route of the same tier.
    min_capability_gain: float = 4.0
    judge_usd: float = DEFAULT_JUDGE_USD
    judge_seconds: float = DEFAULT_JUDGE_SECONDS
    #: ``{enabled, reference_model, value, per_category, unknown}``; None = off.
    #: See the module docstring.
    intelligence_threshold: dict | None = None
    #: The resolved reference (``config.resolve_reference``): its intelligence
    #: index and per-category capability, and where they came from.
    reference: dict | None = None
    #: Verified escalations allowed per turn. Above 1 a second route that is
    #: still below the intelligence threshold is graded again.
    max_escalations: int = 1
    #: Hold back a streamed answer from a below-threshold route until it has
    #: been graded, so an escalated answer can replace it instead of being
    #: appended. Costs the whole generation time as time-to-first-token on
    #: those routes only; off means stream-then-check as before.
    buffer_streams: bool = True

    @classmethod
    def from_config(cls, policy: dict[str, Any] | None) -> "VerifyPolicy":
        conf = dict((policy or {}).get("verify") or {})
        if not conf:
            return cls()
        known = {f for f in cls.__dataclass_fields__}
        kwargs: dict[str, Any] = {k: v for k, v in conf.items() if k in known}
        for name in ("skip_categories", "always", "never"):
            if name in kwargs:
                kwargs[name] = tuple(kwargs[name] or ())
        for name in ("thresholds", "catch_rate", "false_flag_rate", "fix_rate"):
            if name in kwargs:
                default = dict(getattr(cls(), name))
                default.update({str(k): float(v) for k, v in (kwargs[name] or {}).items()})
                kwargs[name] = default
        if "max_escalations" in kwargs:
            kwargs["max_escalations"] = max(0, int(kwargs["max_escalations"]))
        if kwargs.get("intelligence_threshold") is not None and not isinstance(
                kwargs["intelligence_threshold"], dict):
            raise ValueError("verify.intelligence_threshold must be a mapping")
        return cls(**kwargs)

    # -- the intelligence threshold -----------------------------------------
    @property
    def threshold_active(self) -> bool:
        conf = self.intelligence_threshold
        return isinstance(conf, dict) and conf.get("enabled", True) is not False

    def intelligence_gate(self, model: ModelInfo, category: str) -> tuple[bool | None, dict]:
        """Is ``model`` below the reference? ``(None, info)`` when the rule cannot decide.

        ``info`` is what the decision record logs: the model's number, the
        threshold, which scale they were compared on and where the threshold
        came from. It never contains request or answer text.
        """
        if not self.threshold_active:
            return None, {}
        conf = self.intelligence_threshold or {}
        ref = self.reference or {}
        info: dict[str, Any] = {
            "reference_model": ref.get("resolved_id") or conf.get("reference_model"),
            "threshold_source": ref.get("source") or "none",
            "intelligence": model.intelligence_index,
        }
        if model.capability_assumed_from:
            info["intelligence_assumed_from"] = model.capability_assumed_from
        ref_cat = (ref.get("capability") or {}).get(category) or {}
        if (conf.get("per_category", True) and ref_cat.get("strength") == "direct"
                and model.capability_strength.get(category) == "direct"
                and category in model.capability):
            info.update(intelligence=round(model.capability[category], 2),
                        threshold=ref_cat.get("value"), threshold_basis=f"category:{category}")
            return model.capability[category] < float(ref_cat["value"]), info
        threshold = ref.get("intelligence_index")
        if threshold is None and conf.get("value") is not None:
            threshold = float(conf["value"])
            info["threshold_source"] = "configured-value"
        info.update(threshold=threshold, threshold_basis="intelligence_index")
        if threshold is None:
            info["note"] = "no threshold: the reference is not in the benchmark data and no value is configured"
            return None, info
        if model.intelligence_index is None:
            info["note"] = "the route's intelligence is unknown"
            return (str(conf.get("unknown", "check")).lower() != "skip"), info
        return model.intelligence_index < float(threshold), info

    # -- per-category numbers ------------------------------------------------
    def threshold(self, category: str) -> float:
        return float(self.thresholds.get(category, self.thresholds.get("default", 0.30)))

    def catch(self, category: str) -> float:
        return float(self.catch_rate.get(category, self.catch_rate.get("default", 0.0)))

    def false_flag(self, category: str) -> float:
        return float(self.false_flag_rate.get(category, self.false_flag_rate.get("default", 0.0)))

    def fix(self, category: str) -> float:
        return float(self.fix_rate.get(category, self.fix_rate.get("default", 0.5)))

    # -- the gate ------------------------------------------------------------
    def tier_ok(self, model: ModelInfo, category: str, evidence_discount: float = 0.0) -> bool:
        """True when ``model`` is in the cheap tier for this category."""
        if model.name in self.never:
            return False
        if model.name in self.always:
            return True
        if model.subscription:
            # A flat-rate plan is a frontier route billed differently, not a
            # cheap one: no marginal dollar does not mean low capability.
            return False
        if blended_price(model) > self.max_price_per_mtok:
            return False
        return model.cap(category, evidence_discount=evidence_discount) < self.max_capability

    def applies(self, model: ModelInfo, category: str, *, request_chars: int = 0,
                needs_long_context: bool = False, evidence_discount: float = 0.0,
                judge_available: bool = True) -> tuple[bool, str]:
        """Should this answer be checked? Returns (yes, reason-when-not)."""
        ok, why, _info = self.gate(model, category, request_chars=request_chars,
                                   needs_long_context=needs_long_context,
                                   evidence_discount=evidence_discount,
                                   judge_available=judge_available)
        return ok, why

    def gate(self, model: ModelInfo, category: str, *, request_chars: int = 0,
             needs_long_context: bool = False, evidence_discount: float = 0.0,
             judge_available: bool = True) -> tuple[bool, str, dict]:
        """``applies`` plus which rule fired and the intelligence numbers behind it."""
        below, info = (self.intelligence_gate(model, category)
                       if model.name not in self.never and not model.subscription
                       else (None, {}))
        if not self.enabled:
            return False, "verification disabled", info
        if not judge_available:
            return False, "no judge configured", info
        # The request's own reasons come first. Whether the judge can see what
        # the answer depends on is a property of the *question*, true of every
        # route; reporting "this route is too strong to grade" instead would
        # hide the more interesting half of the gate behind an accident of
        # which route happened to answer.
        if category in self.skip_categories or needs_long_context:
            return False, ("the judge cannot see the document this answer depends on "
                           f"(category {category})"), info
        if request_chars > self.max_request_chars:
            return False, (f"request is {request_chars} characters; beyond "
                           f"{self.max_request_chars} the judge would grade what it cannot read"), info
        # The threshold rule is named first when both fire: it is the one that
        # holds a stream back for grading (``Router.buffers_stream``), and a
        # cheap route below the reference is exactly what it exists for.
        if below:
            return True, "", {**info, "rule": "intelligence-threshold"}
        if self.tier_ok(model, category, evidence_discount):
            return True, "", {**info, "rule": "cheap-tier"}
        why = f"{model.name} is above the verified cheap tier"
        if below is False:
            why += (f" and not below the intelligence threshold "
                    f"({info.get('intelligence')} >= {info.get('threshold')}, "
                    f"{info.get('threshold_basis')})")
        return False, why, info


@dataclass(frozen=True)
class Verdict:
    """What the judge said about one answer, and what the router does with it.

    Separate from ``ObservedOutcome`` on purpose: this is a *judgement about an
    answer*, exactly as ``TaskClassification`` is a judgement about a request.
    It never carries the answer, only the numbers.
    """

    verified: bool
    p_adequate: float | None = None
    threshold: float | None = None
    escalate: bool = False
    failure: str = "unknown"
    latency_ms: float = 0.0
    #: Set when the judge itself was unreachable or unusable.
    judge_failed: bool = False
    judge_model: str = ""
    #: Why the answer was not checked, when it was not.
    reason: str = ""
    category: str = ""
    model: str = ""
    escalated_to: str | None = None
    #: Which rule asked for the check: ``cheap-tier`` or ``intelligence-threshold``.
    rule: str | None = None
    #: The intelligence-threshold numbers (model, threshold, basis, source);
    #: empty when the rule is off. Numbers and ids only.
    intelligence: dict = field(default_factory=dict)
    #: Which judge answered: ``hosted-jev`` or ``local-jev-class``.
    judge_backend: str | None = None
    #: Verified escalations that already happened in this turn before this check.
    escalations: int = 0

    def to_dict(self) -> dict:
        return {
            "verified": self.verified,
            "model": self.model or None,
            "category": self.category or None,
            "p_adequate": None if self.p_adequate is None else round(self.p_adequate, 3),
            "threshold": self.threshold,
            "escalate": self.escalate,
            "failure": self.failure,
            "latency_ms": round(self.latency_ms, 1),
            "judge_failed": self.judge_failed,
            "judge_model": self.judge_model or None,
            "escalated_to": self.escalated_to,
            "reason": self.reason or None,
            "rule": self.rule,
            "intelligence": dict(self.intelligence) or None,
            "judge_backend": self.judge_backend,
            "judge_latency_ms": round(self.latency_ms, 1) if self.verified else None,
            "escalations": self.escalations,
        }

    def chip(self, label: str | None = None) -> str:
        """One line for a log or a user interface. Never contains the answer."""
        if not self.verified:
            return "not checked"
        if self.judge_failed:
            return "check unavailable"
        name = label or self.model
        if self.escalate:
            return (f"Jev flagged it as {self.failure.replace('_', '-')} ({self.p_adequate:.2f})"
                    + (f" -> escalated to {self.escalated_to}" if self.escalated_to else ""))
        return f"Checked by Jev: adequate ({self.p_adequate:.2f}){'' if name is None else ''}"


def not_verified(reason: str, *, model: str = "", category: str = "") -> Verdict:
    return Verdict(verified=False, reason=reason, model=model, category=category)


def verdict_from(judgement, policy: VerifyPolicy, category: str, model: str) -> Verdict:
    """Turn a ``jev.Judgement`` into a decision against the category threshold.

    A judge that failed never escalates. An outage in the checker must not
    become an escalation storm that costs more than the failures it was meant
    to catch; the turn is recorded as unverified and the answer stands.
    """
    threshold = policy.threshold(category)
    if judgement.failed:
        return Verdict(verified=True, p_adequate=None, threshold=threshold, escalate=False,
                       failure="unknown", latency_ms=judgement.latency_s * 1000,
                       judge_failed=True, reason="the judge was unreachable",
                       category=category, model=model)
    return Verdict(
        verified=True,
        p_adequate=judgement.p_adequate,
        threshold=threshold,
        escalate=judgement.p_adequate < threshold,
        failure=judgement.failure,
        latency_ms=judgement.latency_s * 1000,
        judge_model=judgement.model,
        category=category,
        model=model,
    )


def retry_messages(messages: list[dict], answer: str, verdict: Verdict,
                   policy: VerifyPolicy) -> list[dict]:
    """The message list for the second attempt.

    With ``carry_failed_attempt`` off this is the original request, unchanged:
    the stronger model gets a clean shot. With it on, the failed answer is
    appended as context together with what the judge objected to - which helps
    when the failure is *incomplete* (continue the work) and hurts when it is
    *wrong* (anchoring on a wrong answer), so the calibration decides, not a
    preference.
    """
    if not policy.carry_failed_attempt or not answer.strip():
        return list(messages)
    kind = {"wrong": "contains an error", "incomplete": "is incomplete",
            "off_topic": "does not answer the question"}.get(verdict.failure, "was judged inadequate")
    note = ("A previous attempt by a smaller model " + kind +
            ". Do not assume it is correct; answer the original request yourself. "
            "Previous attempt:\n\n" + answer.strip())
    return [*messages[:-1], messages[-1], {"role": "user", "content": note}]


def escalation_choice(routing_policy, conversation, request, ctx, failed: str, tried: set[str],
                      policy: VerifyPolicy, rule: str | None = None):
    """The cheapest *meaningfully* stronger route by the policy's own ranking.

    ``Policy.on_failure`` accepts any route rated more than one capability point
    above the one that failed, and ranks by cost per success. In a catalog with
    several free routes that reduces to "the first free model that is nominally
    better", which after a judge has rejected an answer is not an escalation at
    all - it is the same tier again, with the same failure modes, for the same
    zero dollars. So a verified failure asks for a real step up
    (``min_capability_gain``) and takes the cheapest route that clears it by the
    expected-cost ranking the router already uses. Returns ``None`` when no
    route clears the bar, and the caller falls back to the ordinary rule.
    """
    failed_model = ctx.catalog.get(failed)
    if failed_model is None:
        return None
    bar = failed_model.cap(request.category) + policy.min_capability_gain
    scored = [(value, m) for m, _call, _p, value in routing_policy.evaluate(conversation, request, ctx)
              if m.name not in tried and m.cap(request.category) >= bar and value == value
              and value != float("inf")]
    if not scored:
        return None
    note = ""
    if rule == "intelligence-threshold":
        # The rule is "below the reference is checked", so the step up prefers
        # a route that is not below it; only when none exists does it settle
        # for the strongest step it can get.
        above = [(v, m) for v, m in scored if policy.intelligence_gate(m, request.category)[0] is False]
        if above:
            scored, note = above, " at or above the intelligence threshold"
    value, model = min(scored, key=lambda pair: pair[0])
    return model.name, (f"the judge rejected {failed}: escalating to the cheapest route at least "
                        f"{policy.min_capability_gain:g} capability points stronger{note} "
                        f"(expected ${value:.4f})")


def bump_floor(conversation, verdict: Verdict, policy: VerifyPolicy, now: float) -> None:
    """Remember that this conversation proved harder than it was routed for.

    ``Policy.on_failure`` already raises the floor to the difficulty at which
    the failed route succeeds half the time. This adds a lower bound, so a
    route the success table rates optimistically still leaves a mark, and it
    re-stamps the timestamp the memory decays from.
    """
    if not verdict.escalate:
        return
    conversation.floor = max(conversation.floor, policy.difficulty_floor)
    conversation.floor_set_at = now


def with_measured(policy: VerifyPolicy, *, thresholds: dict[str, float] | None = None,
                  catch_rate: dict[str, float] | None = None,
                  false_flag_rate: dict[str, float] | None = None,
                  fix_rate: dict[str, float] | None = None) -> VerifyPolicy:
    """A copy carrying freshly measured rates. Used by the calibration script."""
    return replace(
        policy,
        thresholds={**policy.thresholds, **(thresholds or {})},
        catch_rate={**policy.catch_rate, **(catch_rate or {})},
        false_flag_rate={**policy.false_flag_rate, **(false_flag_rate or {})},
        fix_rate={**policy.fix_rate, **(fix_rate or {})},
    )
