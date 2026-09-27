"""Optional local routing classifier: a small open encoder with typed routing heads, run via ONNX.

One encoder pass per user turn returns the same fields the router's policies read from Jev or
Laya (category, difficulty, needs_*, follow_up, stakes) plus measured-outcome heads that other
classifiers do not have: P(success) for a small, mid and strong model tier.

The weights are downloaded once from Hugging Face into the normal model cache. Inference is
CPU-only through ``onnxruntime`` with the ``tokenizers`` fast tokenizer. Neither PyTorch nor
``transformers`` is needed. Every failure (a missing optional dependency, a failed download, a
bad input) degrades to ``jev.FALLBACK``, exactly like the other classifier backends.

Config::

    classifier: {backend: local-route-head, model: benchmarkheaven/weiche-395m, variant: fp16, threads: 2}
"""

from __future__ import annotations

import json
import os
import threading
import time

from . import jev

DEFAULT_MODEL = "benchmarkheaven/weiche-395m"
VARIANTS = {"fp32": "model.onnx", "fp16": "model_fp16.onnx", "int8": "model_int8.onnx", "int4": "model_int4.onnx"}
CATS = ["coding", "agentic", "math", "knowledge", "long_context", "tool_use", "design", "summarisation", "general"]
NOULS = ["needs_tools", "needs_vision", "needs_long_context", "follow_up"]
TIERS = ["small", "mid", "strong"]
N_STATE = 14
HEAD_CHARS, TAIL_CHARS, CONTEXT_CHARS = 1400, 1000, 600
SOURCE = "local-route-head"


def render(request: str, context: str = "") -> str:
    """Exactly the training-time input format (head + tail of a long request)."""
    req = request or ""
    if len(req) > HEAD_CHARS + TAIL_CHARS:
        req = req[:HEAD_CHARS] + "\n[...]\n" + req[-TAIL_CHARS:]
    ctx = (context or "").strip()[:CONTEXT_CHARS] or "(new conversation)"
    return f"Context: {ctx}\nRequest: {req}"


class LocalRouteHeadClassifier:
    """Lazy ONNX route-head classifier with the same result shape as hosted Jev."""

    def __init__(self, model: str = DEFAULT_MODEL, variant: str = "fp16", threads: int = 2):
        if variant not in VARIANTS:
            raise ValueError(f"unknown route-head variant {variant!r}; use one of {sorted(VARIANTS)}")
        self.model = model
        self.variant = variant
        self.threads = max(1, int(threads))
        self._session = None
        self._tokenizer = None
        self._max_len = 512
        self._lock = threading.Lock()

    def _files(self) -> str:
        if os.path.isdir(self.model):
            return self.model
        from huggingface_hub import snapshot_download
        name = VARIANTS[self.variant]
        return snapshot_download(self.model, allow_patterns=[name, name + ".data", "tokenizer.json",
                                                             "router_head_config.json"])

    def _load(self):
        if self._session is None:
            with self._lock:
                if self._session is None:
                    import onnxruntime as ort
                    from tokenizers import Tokenizer
                    root = self._files()
                    cfg_path = os.path.join(root, "router_head_config.json")
                    if os.path.exists(cfg_path):
                        self._max_len = int(json.load(open(cfg_path)).get("max_len") or 512)
                    tok = Tokenizer.from_file(os.path.join(root, "tokenizer.json"))
                    tok.enable_truncation(self._max_len)
                    tok.no_padding()
                    opts = ort.SessionOptions()
                    opts.intra_op_num_threads = self.threads
                    opts.inter_op_num_threads = 1
                    self._session = ort.InferenceSession(os.path.join(root, VARIANTS[self.variant]), opts,
                                                         providers=["CPUExecutionProvider"])
                    self._tokenizer = tok
        return self._session, self._tokenizer

    def predict(self, request: str, context: str = "", state: list[float] | None = None) -> dict:
        """Raw head outputs for one turn. ``state`` is the 14-float router-state vector (see model card)."""
        import numpy as np
        session, tok = self._load()
        enc = tok.encode(render(jev.scrub(request, jev.REQUEST_CHARS), jev.scrub(context, jev.CONTEXT_CHARS)))
        ids = np.asarray([enc.ids], dtype=np.int64)
        feeds = {"input_ids": ids, "attention_mask": np.ones_like(ids),
                 "state": np.asarray([state or [0.0] * N_STATE], dtype=np.float32)}
        category, scalars, noul, succ, route = session.run(None, feeds)
        return {"category": dict(zip(CATS, category[0].tolist())),
                "difficulty_outcome": float(scalars[0, 0]), "difficulty": float(scalars[0, 1]),
                "stakes": float(scalars[0, 2]), "noul": dict(zip(NOULS, noul[0].tolist())),
                "succ": dict(zip(TIERS, succ[0].tolist())), "route": dict(zip(TIERS, route[0].tolist())),
                "input_tokens": len(enc.ids)}

    def __call__(self, request: str, context: str = "") -> jev.Classification:
        started = time.perf_counter()
        try:
            out = self.predict(request, context)
            probs = out["category"]
            category = max(probs, key=probs.get)
            # A regression head has no score distribution; its confidence is how far the two
            # independent difficulty signals (rubric head, measured-outcome head) agree.
            agreement = 1.0 - min(1.0, abs(out["difficulty"] - out["difficulty_outcome"]) * 2)
            return jev.Classification(
                category=category, category_probs=probs, category_confidence=float(probs[category]),
                difficulty=out["difficulty"], difficulty_confidence=round(agreement, 3),
                needs_tools=out["noul"]["needs_tools"], needs_vision=out["noul"]["needs_vision"],
                needs_long_context=out["noul"]["needs_long_context"], follow_up=out["noul"]["follow_up"],
                stakes=out["stakes"], latency_s=time.perf_counter() - started,
                model=f"{self.model}:{self.variant}", input_tokens=out["input_tokens"], output_tokens=0,
                raw={"succ": out["succ"], "difficulty_outcome": out["difficulty_outcome"]},
                source_name=SOURCE)
        except Exception:  # a local model failure must never take down the routed LLM call
            return jev.FALLBACK
