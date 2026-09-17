"""Routing metrics exposed at /v1/router/metrics."""

from __future__ import annotations

import threading
import time
from collections import Counter, deque
from dataclasses import dataclass, field

from .catalog import ModelInfo
from .pricing import Usage, cost_usd


@dataclass
class Metrics:
    started_at: float = field(default_factory=time.time)
    total_requests: int = 0
    classifier_failures: int = 0
    switches: int = 0
    stickiness_saves: int = 0
    escalations: int = 0
    by_category: Counter = field(default_factory=Counter)
    by_model: Counter = field(default_factory=Counter)
    errors_by_model: Counter = field(default_factory=Counter)
    metered_cost_usd: float = 0.0
    subscription_calls: int = 0
    tokens: Usage = field(default_factory=Usage)
    classification_ms: deque = field(default_factory=lambda: deque(maxlen=500))
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record(
        self,
        *,
        category: str,
        model: ModelInfo,
        classification_ms: float,
        usage: Usage | None = None,
        switched: bool = False,
        stickiness_save: bool = False,
        escalated: bool = False,
        classifier_failed: bool = False,
    ) -> None:
        with self._lock:
            self.total_requests += 1
            self.by_category[category] += 1
            self.by_model[model.name] += 1
            self.classification_ms.append(classification_ms)
            self.switches += int(switched)
            self.stickiness_saves += int(stickiness_save)
            self.escalations += int(escalated)
            self.classifier_failures += int(classifier_failed)
            if model.subscription:
                self.subscription_calls += 1
            if usage:
                self.tokens.uncached_input += usage.uncached_input
                self.tokens.cached_read += usage.cached_read
                self.tokens.cache_write += usage.cache_write
                self.tokens.output += usage.output
                self.metered_cost_usd += cost_usd(model, usage)

    def record_error(self, model_name: str) -> None:
        with self._lock:
            self.errors_by_model[model_name] += 1

    def to_dict(self) -> dict:
        with self._lock:
            times = list(self.classification_ms)
            total_in = self.tokens.total_input
            return {
                "uptime_seconds": round(time.time() - self.started_at, 1),
                "total_requests": self.total_requests,
                "classifier_failures": self.classifier_failures,
                "model_switches": self.switches,
                "stickiness_saves": self.stickiness_saves,
                "escalations": self.escalations,
                "subscription_calls": self.subscription_calls,
                "requests_by_category": dict(self.by_category),
                "requests_by_model": dict(self.by_model),
                "errors_by_model": dict(self.errors_by_model),
                "avg_classification_ms": round(sum(times) / len(times), 1) if times else 0.0,
                "tokens": self.tokens.to_dict(),
                "observed_cache_hit_rate": (
                    round(self.tokens.cached_read / total_in, 4) if total_in else 0.0
                ),
                # Excludes subscription traffic, which has zero marginal cost and
                # would otherwise make the average look better than it is.
                "metered_cost_usd": round(self.metered_cost_usd, 6),
            }


metrics = Metrics()
