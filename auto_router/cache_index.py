"""Prefix-cache index.

Tracks which model holds a warm provider-side prefix cache for which
conversation prefix, so the router can price "stay" against "switch".

Keying
------
We hash the request prefix the way the providers do: tools, then system, then
messages in order (Anthropic states this hierarchy explicitly). A rolling hash
is recorded after every message boundary, which lets a later request find the
longest prefix it shares with an earlier one even when the tail has changed.

Expiry
------
Entries are stamped with the time the writing request STARTED, not when it
finished. Anthropic measures the 5-minute lifetime from the start of the
request, so generation time counts against it; stamping on completion would
systematically overestimate how warm a prefix still is.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Iterable

from .catalog import ModelInfo


def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def prefix_hashes(
    messages: list[dict],
    system: Any = None,
    tools: Any = None,
) -> list[str]:
    """Rolling hashes, one per message boundary, cheapest-to-longest.

    Index i covers tools + system + messages[0..i].
    """
    digest = hashlib.sha256()
    digest.update(_canonical(tools or []).encode())
    digest.update(b"\x00system\x00")
    digest.update(_canonical(system or "").encode())

    out: list[str] = []
    for message in messages:
        digest.update(b"\x00message\x00")
        digest.update(_canonical(message).encode())
        out.append(digest.hexdigest())
    return out


def estimate_tokens(messages: list[dict], system: Any = None, tools: Any = None) -> int:
    """Character-based token estimate.

    Deliberately crude: it is only used to compare two costs, and both sides of
    the comparison use the same estimate, so a constant bias mostly cancels.
    CalibratedEstimator below removes the residual bias using observed usage.
    """
    chars = len(_canonical(tools or [])) + len(_canonical(system or "")) + len(_canonical(messages))
    return max(1, chars // 4)


class CalibratedEstimator:
    """Corrects the char/4 heuristic using real prompt_tokens from responses."""

    def __init__(self, initial: float = 1.0, alpha: float = 0.2) -> None:
        self._ratio = initial
        self._alpha = alpha
        self._samples = 0
        self._lock = threading.Lock()

    def estimate(self, raw: int) -> int:
        with self._lock:
            return max(1, int(raw * self._ratio))

    def observe(self, raw_estimate: int, actual_tokens: int) -> None:
        if raw_estimate <= 0 or actual_tokens <= 0:
            return
        with self._lock:
            observed = actual_tokens / raw_estimate
            self._ratio = (1 - self._alpha) * self._ratio + self._alpha * observed
            self._samples += 1

    @property
    def stats(self) -> dict:
        with self._lock:
            return {"ratio": round(self._ratio, 4), "samples": self._samples}


@dataclass
class Warm:
    """A prefix known to be cached on a specific model."""

    model: str
    prefix_tokens: int
    started_at: float
    message_index: int

    def age(self, now: float | None = None) -> float:
        return (now or time.time()) - self.started_at


@dataclass
class _Entry:
    model: str
    prefix_tokens: int
    started_at: float
    message_index: int
    ttl_seconds: int


class PrefixCacheIndex:
    """(model, prefix_hash) -> warm entry, with provider-specific TTLs."""

    def __init__(self, safety_seconds: int = 30, max_entries: int = 20_000) -> None:
        self._entries: dict[tuple[str, str], _Entry] = {}
        self._safety = safety_seconds
        self._max_entries = max_entries
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def record(
        self,
        model: ModelInfo,
        hashes: list[str],
        token_counts: list[int],
        started_at: float,
    ) -> None:
        """Record that `model` processed a request with these prefix hashes.

        Providers cache the whole prefix up to the final breakpoint, so every
        rolling hash of this request is now warm on this model.
        """
        if not hashes:
            return
        total = token_counts[-1] if token_counts else 0
        if total < model.cache.min_tokens:
            # Below the provider minimum nothing is cached and no error is
            # returned, so recording a warm entry here would be a lie.
            return
        with self._lock:
            for idx, (h, tokens) in enumerate(zip(hashes, token_counts)):
                self._entries[(model.name, h)] = _Entry(
                    model=model.name,
                    prefix_tokens=tokens,
                    started_at=started_at,
                    message_index=idx,
                    ttl_seconds=model.cache.ttl_seconds,
                )
            if len(self._entries) > self._max_entries:
                self._evict_locked()

    def _evict_locked(self) -> None:
        now = time.time()
        stale = [k for k, e in self._entries.items()
                 if now - e.started_at > e.ttl_seconds]
        for k in stale:
            del self._entries[k]
        if len(self._entries) > self._max_entries:
            ordered = sorted(self._entries.items(), key=lambda kv: kv[1].started_at)
            for k, _ in ordered[: len(self._entries) - self._max_entries]:
                del self._entries[k]

    def lookup(self, model_name: str, hashes: list[str], now: float | None = None) -> Warm | None:
        """Longest still-warm prefix this model holds for this conversation."""
        now = now or time.time()
        with self._lock:
            for idx in range(len(hashes) - 1, -1, -1):
                entry = self._entries.get((model_name, hashes[idx]))
                if entry is None:
                    continue
                if now - entry.started_at > max(0, entry.ttl_seconds - self._safety):
                    continue
                self.hits += 1
                return Warm(entry.model, entry.prefix_tokens, entry.started_at, idx)
        self.misses += 1
        return None

    def warm_models(self, hashes: list[str], now: float | None = None) -> dict[str, Warm]:
        """Every model currently holding a warm prefix for this conversation."""
        now = now or time.time()
        with self._lock:
            names = {k[0] for k in self._entries}
        out: dict[str, Warm] = {}
        for name in names:
            warm = self.lookup(name, hashes, now)
            if warm is not None:
                out[name] = warm
        return out

    @property
    def stats(self) -> dict:
        with self._lock:
            live = len(self._entries)
        total = self.hits + self.misses
        return {
            "entries": live,
            "lookup_hits": self.hits,
            "lookup_misses": self.misses,
            "hit_rate": round(self.hits / total, 4) if total else 0.0,
        }
