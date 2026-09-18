"""The routing decision record.

Requirement being tested: classification, route selection, observed outcome and
estimated outcome are separately represented, the explanation is compact and
carries no prompt text, and a specialised route is selected from evidence
rather than from a hard-coded model name.
"""

import json

import pytest

from auto_router.catalog import CacheRules, Catalog, ModelInfo, Prices
from auto_router.config import RouterConfig
from auto_router.decision import ObservedOutcome
from auto_router.jev import Classification
from auto_router.router import Router

CACHE = CacheRules(ttl_seconds=300, min_tokens=1024, hit_rate=0.9)

SECRET = "sk-" + "ant-api03-" + "Z" * 24   # built, never a literal: see the gate
PROMPT = "Build a responsive pricing page with a dark theme and a comparison table. "


def _model(name, inp, out, read, design, coding, strength="direct", stale=False, **kw):
    return ModelInfo(
        name, "host", name, Prices(inp, out, read), CACHE,
        capability={"design": design, "coding": coding, "general": (design + coding) / 2},
        capability_basis={"design": f"designarena elo {1200 + (design - 50) * 4:.0f}",
                          "coding": "aa_coding_index", "general": "aa_intelligence_index x 1.5"},
        capability_strength={"design": strength, "coding": "direct", "general": "derived"},
        evidence_stale=stale, evidence={"source": "stale-cache" if stale else "network"}, **kw)


#: Design rank and coding rank deliberately disagree: the cheaper route is the
#: better designer, the dearer route is the better coder.
DESIGNER = _model("designer", 3.0, 15.0, 0.3, design=80, coding=76)
CODER = _model("coder", 5.0, 25.0, 0.5, design=46, coding=90)
CHEAP = _model("cheap", 0.1, 0.4, 0.01, design=45, coding=50, strength="derived")


def _router(*models, classifier=None, discount=0.5, **policy):
    config = RouterConfig(providers={}, catalog=Catalog(list(models)),
                          policy={"success": {"evidence_discount": discount}, **policy})
    return Router(config, classifier=classifier or (lambda text, ctx: _cls("design", 0.5)))


def _cls(category, difficulty, **kw):
    return Classification(category=category, category_probs={category: 0.9}, difficulty=difficulty,
                          difficulty_confidence=kw.pop("dconf", 0.8), needs_tools=0.0,
                          needs_vision=0.0, needs_long_context=0.0, follow_up=0.1, stakes=0.5,
                          category_confidence=kw.pop("cconf", 0.9), model="jev-1.13.0", **kw)


def _messages(text=PROMPT):
    return [{"role": "user", "content": text * 40}]


# -- separation -------------------------------------------------------------
def test_the_four_kinds_of_statement_are_separate_objects():
    result = _router(DESIGNER, CODER, CHEAP).route(_messages())
    record = result.explanation.to_dict()
    assert set(record) >= {"classification", "selection", "cache", "estimated_outcome",
                           "observed_outcome", "candidates"}
    assert record["estimated_outcome"]["kind"] == "estimate"
    assert record["observed_outcome"] is None, "nothing is observed before the call"
    # A classification never names a model; a selection always does.
    assert "model" not in record["classification"]
    assert record["selection"]["selected"] in {"designer", "coder", "cheap"}


def test_the_estimate_is_not_overwritten_by_the_observation():
    router = _router(DESIGNER, CODER, CHEAP)
    result = router.route(_messages())
    estimate = result.explanation.estimated.cost_usd
    router.observe(result, ObservedOutcome(model=result.model.name, status="ok",
                                           cost_usd=99.0, cost_basis="test"))
    record = result.explanation.to_dict()
    assert record["estimated_outcome"]["cost_usd"] == pytest.approx(estimate)
    assert record["observed_outcome"]["cost_usd"] == 99.0
    assert record["observed_outcome"]["kind"] == "observed"


def test_an_observation_without_a_price_basis_is_none_not_zero():
    router = _router(DESIGNER, CODER, CHEAP)
    result = router.route(_messages())
    router.observe(result, ObservedOutcome(model=result.model.name, status="ok",
                                           cost_usd=None,
                                           cost_basis="subscription route: no marginal cash cost"))
    observed = result.explanation.to_dict()["observed_outcome"]
    assert observed["cost_usd"] is None
    assert "no marginal cash cost" in observed["cost_basis"]


def test_observed_cache_hit_rate_is_computed_from_reported_tokens():
    outcome = ObservedOutcome(model="m", uncached_input_tokens=100, cached_read_tokens=900,
                              cache_write_tokens=0, output_tokens=50)
    assert outcome.cache_hit_rate == pytest.approx(0.9)
    assert ObservedOutcome(model="m").cache_hit_rate is None


# -- no prompt text ---------------------------------------------------------
def test_the_explanation_carries_no_prompt_or_secret_text():
    text = f"{PROMPT} my api_key={SECRET} do not leak this. "
    result = _router(DESIGNER, CODER, CHEAP).route(_messages(text))
    blob = json.dumps(result.explanation.to_dict())
    assert SECRET not in blob
    assert "pricing page" not in blob
    assert "do not leak" not in blob
    # Only counts, identifiers, categories and numbers leave the process.
    assert result.explanation.cache.key_scope and PROMPT[:20] not in blob


def test_the_one_line_summary_carries_no_prompt_text():
    result = _router(DESIGNER, CODER, CHEAP).route(_messages())
    summary = result.explanation.summary()
    assert "pricing" not in summary and "design" in summary


def test_response_headers_expose_the_decision_id_not_the_content():
    result = _router(DESIGNER, CODER, CHEAP).route(_messages())
    headers = result.headers
    assert headers["X-Router-Decision"] == result.explanation.id
    assert headers["X-Router-Cache"] in {"warm", "cold", "too-short", "no-cache"}
    assert all("pricing" not in v for v in headers.values())


# -- evidence-driven specialisation -----------------------------------------
def test_a_design_task_follows_the_design_evidence_not_the_coding_evidence():
    router = _router(DESIGNER, CODER, CHEAP)
    result = router.route(_messages())
    assert result.model.name == "designer", "coder has higher coding capability and loses anyway"
    chosen = next(c for c in result.explanation.candidates if c.model == "designer")
    assert chosen.evidence_strength == "direct"
    assert chosen.capability_basis.startswith("designarena")


def test_the_same_catalog_sends_a_coding_task_to_the_coding_route():
    """Same models, same prices: only the category changes, and so does the route."""
    hard_coding = lambda text, ctx: _cls("coding", 0.9)      # noqa: E731
    hard_design = lambda text, ctx: _cls("design", 0.9)      # noqa: E731
    assert _router(DESIGNER, CODER, classifier=hard_coding).route(_messages()).model.name == "coder"
    assert _router(DESIGNER, CODER, classifier=hard_design).route(_messages()).model.name == "designer"


def test_nothing_in_the_router_hard_codes_a_specialised_model_name():
    """Rename every model and the design task still lands on the design evidence."""
    renamed = [DESIGNER.with_(name="route-alpha"), CODER.with_(name="route-beta"),
               CHEAP.with_(name="route-gamma")]
    assert _router(*renamed).route(_messages()).model.name == "route-alpha"


def test_a_candidate_list_records_what_was_compared():
    result = _router(DESIGNER, CODER, CHEAP).route(_messages())
    record = result.explanation.to_dict()
    assert record["selection"]["candidates_considered"] == 3
    assert len(record["candidates"]) == 3
    for candidate in record["candidates"]:
        assert candidate["capability_basis"]
        assert candidate["evidence_strength"] in {"direct", "derived", "weak", "none"}
        assert candidate["estimated_p_success"] is not None
    # Ranked cheapest expected cost first.
    costs = [c["estimated_expected_usd"] for c in record["candidates"]]
    assert costs == sorted(costs)


def test_a_fallback_route_is_always_named_when_one_exists():
    result = _router(DESIGNER, CODER, CHEAP).route(_messages())
    fallback = result.explanation.selection.fallback
    assert fallback and fallback != result.model.name


# -- safe fallback ----------------------------------------------------------
def test_a_classifier_outage_is_recorded_as_a_safe_fallback():
    from auto_router.jev import FALLBACK
    result = _router(DESIGNER, CODER, CHEAP, classifier=lambda t, c: FALLBACK).route(_messages())
    selection = result.explanation.selection
    assert selection.safe_fallback == "classifier-unavailable"
    assert result.explanation.classification.source == "fallback"
    assert selection.evidence_confidence < 1.0
    assert any("Jev was unreachable" in n for n in result.explanation.notes)


def test_stale_evidence_is_flagged_and_lowers_confidence():
    stale = _model("stale-designer", 3.0, 15.0, 0.3, design=80, coding=76, stale=True)
    result = _router(stale).route(_messages())
    selection = result.explanation.selection
    assert selection.safe_fallback == "stale-evidence"
    assert any("stale" in n for n in result.explanation.notes)
    fresh = _router(DESIGNER).route(_messages())
    assert selection.evidence_confidence < fresh.explanation.selection.evidence_confidence


def test_a_route_with_no_category_evidence_is_flagged():
    blank = ModelInfo("blank", "host", "blank", Prices(1.0, 2.0), CACHE,
                      capability={"general": 60.0}, capability_basis={"general": "x"},
                      capability_strength={"general": "derived"})
    result = _router(blank).route(_messages())
    assert result.explanation.selection.safe_fallback == "weak-category-evidence"


def test_the_evidence_discount_makes_a_thin_cheap_route_less_attractive():
    """With the discount on, a cheap route with only derived design evidence
    must not out-rank a route with measured design evidence."""
    thin = _model("thin", 0.05, 0.2, 0.005, design=82, coding=50, strength="weak")
    hard = lambda text, ctx: _cls("design", 0.9)             # noqa: E731
    with_discount = _router(DESIGNER, thin, classifier=hard, discount=0.6).route(_messages())
    without = _router(DESIGNER, thin, classifier=hard, discount=0.0).route(_messages())
    assert without.model.name == "thin", "undiscounted, the cheap claim wins"
    assert with_discount.model.name == "designer", "discounted, measured evidence wins"


# -- cache decision ---------------------------------------------------------
def test_cache_status_reflects_the_prefix_state():
    router = _router(DESIGNER, CODER, CHEAP)
    first = router.route(_messages())
    assert first.explanation.cache.status in {"cold", "too-short"}
    assert first.explanation.cache.estimated_usd_avoided is None

    router.commit(first, prompt_tokens=40_000, output_tokens=500)
    second = router.route(_messages() + [{"role": "assistant", "content": "ok"},
                                         {"role": "user", "content": "now make it lighter " * 400}])
    cache = second.explanation.cache
    if cache.status == "warm":
        assert cache.warm_tokens > 0
        assert cache.estimated_usd_avoided is not None
        assert "not a measured saving" in cache.basis


def test_no_saving_is_claimed_without_a_cache_read_price():
    no_price = ModelInfo("nocacheprice", "host", "x", Prices(1.0, 4.0, cache_read=None), CACHE,
                         capability={"design": 80, "general": 70},
                         capability_strength={"design": "direct"},
                         capability_basis={"design": "designarena"})
    router = _router(no_price)
    first = router.route(_messages())
    router.commit(first, prompt_tokens=40_000, output_tokens=500)
    second = router.route(_messages(PROMPT * 3))
    assert second.explanation.cache.estimated_usd_avoided is None
    if second.explanation.cache.status == "warm":
        assert "no cache-read price" in second.explanation.cache.basis


def test_a_route_without_caching_reports_no_cache():
    plain = ModelInfo("plain", "host", "x", Prices(1.0, 4.0), CacheRules(ttl_seconds=0),
                      capability={"design": 70, "general": 70},
                      capability_strength={"design": "direct"}, capability_basis={"design": "b"})
    result = _router(plain).route(_messages())
    assert result.explanation.cache.status == "no-cache"


# -- history and robustness -------------------------------------------------
def test_decisions_are_retained_and_bounded():
    router = _router(DESIGNER, CODER, CHEAP, decision_history=3)
    for i in range(6):
        router.route([{"role": "user", "content": f"{PROMPT}{i}" * 40}])
    assert len(router.decisions) == 3


def test_a_broken_explanation_never_breaks_routing(monkeypatch):
    router = _router(DESIGNER, CODER, CHEAP)
    monkeypatch.setattr(router, "explain",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    result = router.route(_messages())
    assert result.model.name and result.explanation is None


def test_a_tool_loop_turn_is_labelled_as_such():
    router = _router(DESIGNER, CODER, CHEAP)
    first = router.route(_messages())
    router.commit(first, prompt_tokens=2000, output_tokens=100)
    loop = router.route(_messages() + [
        {"role": "assistant", "content": "calling a tool"},
        {"role": "user", "content": [{"type": "tool_result", "content": "done"}]}])
    assert loop.explanation.classification.source == "tool-loop"
    assert loop.explanation.selection.turn_start is False
