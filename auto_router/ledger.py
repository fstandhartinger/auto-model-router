"""Append-only JSONL ledger of routing decisions.

One line per decision, written after the outcome is known so that estimate and
observation sit side by side in the same record and can be compared later
without re-running anything.

Only ``RoutingExplanation.to_dict()`` is written, which by construction holds
no prompt text, no response text and no credentials. Writing is best effort: a
full disk or a read-only path must never take routing down, so failures are
logged once and then counted.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path

log = logging.getLogger("auto_router.ledger")


class RoutingLedger:
    """Writes decision records to a JSONL file, or nowhere when disabled."""

    def __init__(self, path: str | Path | None):
        self.path = Path(path).expanduser() if path else None
        self.written = 0
        self.failed = 0
        self._lock = threading.Lock()
        self._warned = False

    @classmethod
    def from_env(cls) -> "RoutingLedger":
        return cls(os.environ.get("AUTO_ROUTER_LEDGER") or None)

    @property
    def enabled(self) -> bool:
        return self.path is not None

    def write(self, explanation, *, event: str | None = None) -> bool:
        """Append one decision record.

        ``event`` marks a later, complete copy of a decision that was already
        written - the answer check happens after the outcome line - with the
        same ``id``. A reader keeps the last line per id.
        """
        record = explanation.to_dict()
        if event:
            record["event"] = event
        return self._append(record)

    def _append(self, record: dict) -> bool:
        if self.path is None:
            return False
        line = json.dumps(record, separators=(",", ":"), default=str)
        with self._lock:
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as fh:
                    fh.write(line + "\n")
                self.written += 1
                return True
            except OSError as exc:
                self.failed += 1
                if not self._warned:
                    self._warned = True
                    log.warning("routing ledger %s is not writable (%s); continuing without it",
                                self.path, type(exc).__name__)
                return False

    @property
    def stats(self) -> dict:
        return {"enabled": self.enabled, "path": str(self.path) if self.path else None,
                "written": self.written, "failed": self.failed}
