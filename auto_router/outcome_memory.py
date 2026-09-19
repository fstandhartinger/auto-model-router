"""What earlier attempts on comparable requests actually did, and nothing else.

``truncation.py`` decides whether *one* answer ran out of output budget, and the
server retries that turn on another route. Until now the lesson stopped there:
the next comparable request went straight back to the route that had just run
out of budget, because nothing on the selection side ever saw the observation.
The 18 September design run showed why that matters - the route that burned a
12,000-token budget did it on both of the hardest tasks, i.e. repeatedly, and
no capability score predicted it.

This module is that missing memory, kept deliberately small and conservative:

* **Observations only.** It is fed from ``ObservedOutcome.status`` - the
  provider's own stop flag, via ``truncation.py`` - and never from an estimate,
  a success probability or a judge's verdict. ``ok`` and ``truncated`` are the
  only statuses it counts: a 5xx says the route was unavailable, not that it
  cannot finish, and ``not_taken`` says nothing about the route at all.
* **Comparable means the same category and the same output-budget bucket.**
  Running out of budget is a property of a route *and* a budget; a route that
  cannot finish a page in 12,000 tokens says nothing about a 500-token answer.
* **A stated evidence basis, or no effect.** A route is avoided only after
  ``min_truncations`` observed length stops that are at least ``min_rate`` of
  its recent observed outcomes for that key, within ``ttl_s``. One truncation is
  a retry (already handled by the server), not a pattern.
* **Only when a usable fallback exists.** Avoidance never removes the last
  route; the router keeps the policy's choice and says why.
* **Nothing identifying is stored.** A key is a route name, a category from a
  fixed vocabulary and a bucket label; an entry is a status word and a time.
  No prompt, answer, provider body or error text can enter it.
* **Cold start is inert.** With no observations every query answers "no
  evidence", so a fresh router routes exactly as it did before this existed.

Nothing here edits a capability score or a success estimate.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass

#: The only observed statuses that say anything about whether a route finishes.
COUNTED_STATUSES = frozenset({"ok", "truncated"})

#: Upper bounds of the output-budget buckets, in tokens. A request without an
#: explicit ``max_tokens`` gets its own bucket: its real budget is the
#: provider's default, which the router does not know.
BUDGET_BUCKETS = (1024, 4096, 16384, 65536)


def budget_bucket(max_tokens: int | None) -> str:
    """A coarse, fixed label for the caller's output budget."""
    if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens <= 0:
        return "default"
    for upper in BUDGET_BUCKETS:
        if max_tokens <= upper:
            return f"<={upper}"
    return f">{BUDGET_BUCKETS[-1]}"


@dataclass(frozen=True)
class OutcomeKey:
    """Which requests count as comparable. Every part comes from a fixed set."""

    category: str
    budget: str

    def label(self) -> str:
        return f"{self.category}/{self.budget}"


@dataclass(frozen=True)
class AvoidanceRule:
    """When a route's observed truncations are enough to route around it."""

    enabled: bool = True
    min_truncations: int = 2
    min_rate: float = 0.5
    window: int = 8
    ttl_s: float = 24 * 3600.0

    @classmethod
    def from_config(cls, policy: dict) -> "AvoidanceRule":
        conf = (policy or {}).get("truncation_memory") or {}
        rule = cls(**{k: conf[k] for k in cls.__dataclass_fields__ if k in conf})
        return cls(enabled=bool(rule.enabled),
                   min_truncations=max(1, int(rule.min_truncations)),
                   min_rate=min(1.0, max(0.0, float(rule.min_rate))),
                   window=max(1, int(rule.window)),
                   ttl_s=max(0.0, float(rule.ttl_s)))


@dataclass(frozen=True)
class RouteRecord:
    """Observed evidence about one route for one key. Counts, never content."""

    model: str
    key: str
    observed: int
    truncated: int
    window: int
    ttl_s: float

    @property
    def rate(self) -> float:
        return self.truncated / self.observed if self.observed else 0.0

    def basis(self) -> str:
        return (f"{self.truncated} of the last {self.observed} observed outcomes of {self.model} "
                f"on {self.key} were provider-flagged length stops "
                f"(window {self.window}, within {self.ttl_s / 3600:.0f} h)")


class OutcomeMemory:
    """Recent observed outcomes per (route, comparable key), bounded and in memory."""

    def __init__(self, rule: AvoidanceRule | None = None):
        self.rule = rule or AvoidanceRule()
        self._lock = threading.Lock()
        self._seen: dict[tuple[str, OutcomeKey], deque[tuple[float, str]]] = {}

    def record(self, model: str, key: OutcomeKey | None, status: str,
               now: float | None = None) -> bool:
        """Remember one observed outcome. Returns whether it was counted."""
        if key is None or status not in COUNTED_STATUSES:
            return False
        with self._lock:
            entries = self._seen.setdefault((model, key), deque(maxlen=self.rule.window))
            entries.append((now if now is not None else time.time(), status))
        return True

    def evidence(self, model: str, key: OutcomeKey | None,
                 now: float | None = None) -> RouteRecord | None:
        """What was observed for this route and key, within the window and TTL."""
        if key is None:
            return None
        now = now if now is not None else time.time()
        with self._lock:
            entries = list(self._seen.get((model, key), ()))
        recent = [status for at, status in entries if now - at <= self.rule.ttl_s]
        if not recent:
            return None
        return RouteRecord(model=model, key=key.label(), observed=len(recent),
                           truncated=sum(1 for s in recent if s == "truncated"),
                           window=self.rule.window, ttl_s=self.rule.ttl_s)

    def should_avoid(self, model: str, key: OutcomeKey | None,
                     now: float | None = None) -> RouteRecord | None:
        """The evidence record when it meets the rule, else ``None``."""
        if not self.rule.enabled:
            return None
        record = self.evidence(model, key, now)
        if record is None:
            return None
        if record.truncated >= self.rule.min_truncations and record.rate >= self.rule.min_rate:
            return record
        return None

    @property
    def stats(self) -> dict:
        with self._lock:
            keys = len(self._seen)
            counted = sum(len(v) for v in self._seen.values())
        return {"enabled": self.rule.enabled, "keys": keys, "observations": counted,
                "min_truncations": self.rule.min_truncations, "min_rate": self.rule.min_rate,
                "window": self.rule.window, "ttl_s": self.rule.ttl_s}
