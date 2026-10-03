"""Replay sets for the operator-rule regression test (``test_paid_rule.py``).

Uses only APIs that existed before the rule (e6d4f28), so ``replay_records()``
runs unchanged on that tree; its output there is ``paid_rule_regression.json``.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from auto_router.catalog import Catalog, ModelInfo, Prices
from auto_router.config import Provider, RouterConfig, load_config
from auto_router.jev import Classification
from auto_router.policies import Conversation
from auto_router.router import Router
from experiments.heldout_run import output_budget
from experiments.tasks_heldout import build_tasks

ROOT = Path(__file__).resolve().parent.parent
PAIRED_AB = ROOT / "evaluation" / "paired-router-ab-20260929"
CATEGORIES = ("general", "coding", "math", "knowledge", "summarisation", "design", "long_context",
              "tool_use", "agentic")
BUDGETS = (None, 4000, 8000, 9999, 12000)


def _route(name, provider, upstream, free, caps):
    prices = Prices.free() if free else Prices(2.0, 10.0, 0.2, 2.5)
    return ModelInfo(name, provider, upstream, prices, capability=dict(zip(CATEGORIES, caps)))


#: The round-2 registered catalog (names, providers, prices, capabilities).
ROUND2 = [
    _route("dsv4-flash", "host", "deepseek-ai/DeepSeek-V4-Flash-0731-TEE", True,
           (51.45, 69.1, 51.45, 51.45, 60.86, 69.1, 79.67, 51.45, 49.8)),
    _route("gpt-5.6-sol", "openrouter", "openai/gpt-5.6-sol", False,
           (58.8, 76.3, 58.8, 58.8, 65.98, 44.0, 80.33, 80.99, 61.6)),
    _route("kimi-k3", "host", "moonshotai/Kimi-K3-TEE", True,
           (63.5946, 74.3946, 63.5946, 63.5946, 71.3546, 74.4446, 86.8646, 63.5946, 59.4046)),
    _route("qwen3.8-27b", "host", "Qwen/Qwen3.8-27B-TEE", True,
           (49.8469, 67.3969, 49.8469, 49.8469, 60.3269, 67.3969, 81.2969, 49.8469, 56.7369)),
]
ROUND2_POLICY = {"escalate_after_tool_errors": 3, "jev_difficulty_calibration": [0.27, 0.51]}
PROVIDERS = {"host": Provider("host", "https://example.invalid/v1"),
             "openrouter": Provider("openrouter", "https://example.invalid/v1")}


def round2_config(**policy) -> RouterConfig:
    return RouterConfig(providers=PROVIDERS, catalog=Catalog(list(ROUND2)),
                        policy={**ROUND2_POLICY, **policy})


def stub_classifier(task: dict):
    category = {"research": "knowledge", "cache_repeat": "knowledge"}.get(task["category"], task["category"])
    difficulty = {"easy": 0.2, "medium": 0.5, "hard": 0.8}.get(task.get("difficulty"), 0.5)

    def classify(_text, _context):
        return Classification(category=category, category_probs={}, difficulty=difficulty,
                              difficulty_confidence=0.9, needs_tools=0.0, needs_vision=0.0,
                              needs_long_context=0.0, follow_up=0.2, stakes=0.5, latency_s=0.01)
    return classify


def _router(config: RouterConfig, classify) -> Router:
    router = Router(config, classifier=classify, quota_reader=lambda: {})
    if classify is None:
        router.classifier = None
    return router


def _records(label: str, config_for, tasks: list[dict], own_budget, with_stub: bool) -> list[dict]:
    out = []
    for task in tasks:
        for budget in (own_budget(task), *BUDGETS):
            for kind in ("none", "stub") if with_stub else ("none",):
                router = _router(config_for(), stub_classifier(task) if kind == "stub" else None)
                messages = [{"role": "user", "content": task["prompt"]}]
                res = router.route(messages, system=task.get("system"), max_tokens=budget, now=1000.0)
                choice = router.policy.choose(Conversation(), res.request, router.context())
                out.append({"set": label, "task": task["id"], "max_tokens": budget, "classifier": kind,
                            "routed": res.model.name, "routed_reason": res.reason,
                            "model": choice.model, "reason": choice.reason,
                            "expected_cost": repr(choice.expected_cost),
                            "p_success": repr(choice.p_success)})
    return out


def paired_ab_config() -> RouterConfig:
    loaded = load_config(PAIRED_AB / "router.yaml", use_bench=False)
    return RouterConfig(providers=loaded.providers, catalog=loaded.catalog,
                        policy={"name": "F_expected", "classifier": {"backend": "none"}})


def replay_records() -> list[dict]:
    """Every replayed choice, in a fixed order. Runs unchanged on the pre-rule tree."""
    heldout = build_tasks(random.Random(20260925))
    paired = [json.loads(line) for line in (PAIRED_AB / "tasks-public.jsonl").read_text().splitlines()
              if line.strip()]
    return (_records("heldout-round2", round2_config, heldout, output_budget, True)
            + _records("paired-ab-20260929", paired_ab_config, paired, lambda t: t.get("max_tokens"), True))
