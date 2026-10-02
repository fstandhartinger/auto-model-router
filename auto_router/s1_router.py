"""Optional local routing classifier: s1-llm-auto-router, a small multilingual encoder run via ONNX.

The model answers exactly the router's seven classifier questions (category, difficulty, stakes,
needs_tools, needs_vision, needs_long_context, follow_up) in one encoder pass, about 20-30 ms on
four CPU cores. It is a fixed-schema routing model, not a general Jev-class model.

Inference is CPU-only through ``onnxruntime`` with the ``tokenizers`` fast tokenizer; neither
PyTorch nor ``transformers`` is needed. ``model`` is a local directory (``model.gq.onnx``,
``meta.json``, ``tokenizer/tokenizer.json``) or a Hugging Face repo id. Every failure degrades to
``jev.FALLBACK``, exactly like the other classifier backends.

Config::

    classifier: {backend: s1-llm-auto-router, model: /path/to/model-dir, threads: 2}
"""

from __future__ import annotations

import json
import os
import threading
import time

from . import jev

DEFAULT_MODEL = "system1models/s1-llm-auto-router"
MODEL_FILE = "model.gq.onnx"
CATS = ["coding", "agentic", "math", "knowledge", "long_context", "tool_use", "design", "summarisation", "general"]
FLAGS = ["needs_tools", "needs_vision", "needs_long_context", "follow_up"]
CONTEXT_CHARS = 600
SOURCE = "s1-llm-auto-router"


def _softmax(x):
    import numpy as np
    e = np.exp(x - x.max())
    return e / e.sum()


class LocalS1RouterClassifier:
    """Lazy ONNX s1-llm-auto-router classifier with the same result shape as hosted Jev."""

    def __init__(self, model: str = DEFAULT_MODEL, threads: int = 2):
        self.model = model
        self.threads = max(1, int(threads))
        self._session = None
        self._tokenizer = None
        self._temps: dict = {}
        self._lock = threading.Lock()

    def _files(self) -> str:
        if os.path.isdir(self.model):
            return self.model
        from huggingface_hub import snapshot_download
        return snapshot_download(self.model, allow_patterns=[MODEL_FILE, "meta.json", "tokenizer/*"])

    def _load(self):
        if self._session is None:
            with self._lock:
                if self._session is None:
                    import onnxruntime as ort
                    from tokenizers import Tokenizer
                    root = self._files()
                    meta = json.load(open(os.path.join(root, "meta.json")))
                    tok = Tokenizer.from_file(os.path.realpath(os.path.join(root, "tokenizer", "tokenizer.json")))
                    tok.enable_truncation(min(int(meta.get("max_len") or 512), 512), strategy="only_second")
                    tok.no_padding()
                    opts = ort.SessionOptions()
                    opts.intra_op_num_threads = self.threads
                    opts.inter_op_num_threads = 1
                    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                    self._session = ort.InferenceSession(os.path.realpath(os.path.join(root, MODEL_FILE)), opts,
                                                         providers=["CPUExecutionProvider"])
                    self._temps = meta.get("temperatures") or {}
                    self._tokenizer = tok
        return self._session, self._tokenizer

    def predict(self, request: str, context: str = "") -> dict:
        """Calibrated head outputs for one turn: probabilities per category / level / flag."""
        import numpy as np
        session, tok = self._load()
        ctx = jev.scrub(context, jev.CONTEXT_CHARS)[:CONTEXT_CHARS]
        enc = tok.encode(ctx, jev.scrub(request, jev.REQUEST_CHARS))
        ids = np.asarray([enc.ids], dtype=np.int64)
        cat, dif, stk, flg = session.run(None, {"input_ids": ids, "attention_mask": np.ones_like(ids)})
        t = self._temps
        pc = _softmax(cat[0] / t.get("category", 1.0))
        pd = _softmax(dif[0] / t.get("difficulty", 1.0))
        ps = _softmax(stk[0] / t.get("stakes", 1.0))
        pf = 1 / (1 + np.exp(-flg[0] / t.get("flags", 1.0)))
        return {"category": dict(zip(CATS, pc.tolist())), "difficulty": pd.tolist(), "stakes": ps.tolist(),
                "flags": dict(zip(FLAGS, pf.tolist())), "input_tokens": len(enc.ids)}

    def __call__(self, request: str, context: str = "") -> jev.Classification:
        started = time.perf_counter()
        try:
            out = self.predict(request, context)
            probs = out["category"]
            category = max(probs, key=probs.get)
            dif, stk = out["difficulty"], out["stakes"]
            return jev.Classification(
                category=category, category_probs=probs, category_confidence=float(probs[category]),
                difficulty=sum(i * p for i, p in enumerate(dif)) / (len(dif) - 1),
                difficulty_confidence=float(max(dif)),
                needs_tools=out["flags"]["needs_tools"], needs_vision=out["flags"]["needs_vision"],
                needs_long_context=out["flags"]["needs_long_context"], follow_up=out["flags"]["follow_up"],
                stakes=sum(i * p for i, p in enumerate(stk)) / (len(stk) - 1),
                latency_s=time.perf_counter() - started, model=SOURCE,
                input_tokens=out["input_tokens"], output_tokens=0, source_name=SOURCE)
        except Exception:  # a local model failure must never take down the routed LLM call
            return jev.FALLBACK


#: Hosted endpoint of the same model (system1models.ai), billed per input token.
API_URL = os.environ.get("AUTO_ROUTER_S1_URL", "https://api.system1models.ai/v1/systemone")
API_KEY_ENV = ("S1M_API_KEY", "SYSTEM1_API_KEY")
#: The model reads at most 512 tokens and 600 context characters; trimming keeps the
#: request well inside the API's 16 KiB state limit even for 3-byte scripts.
API_REQUEST_CHARS = 3000


def deps_available() -> bool:
    """True when the optional CPU runtime for the local model is installed ([s1-router] extra)."""
    import importlib.util
    return all(importlib.util.find_spec(m) is not None
               for m in ("onnxruntime", "tokenizers", "huggingface_hub", "numpy"))


class HostedS1RouterClassifier:
    """s1-llm-auto-router through the system1models.ai API (one request answers all seven questions)."""

    def __init__(self, api_key_env: str | None = None, url: str = API_URL, timeout: float = 10.0):
        self.api_key_env = api_key_env
        self.url = url
        self.timeout = timeout

    def _key(self) -> str | None:
        names = (self.api_key_env,) if self.api_key_env else API_KEY_ENV
        return next((os.environ[n] for n in names if os.environ.get(n)), None)

    def __call__(self, request: str, context: str = "") -> jev.Classification:
        import urllib.request
        started = time.perf_counter()
        try:
            key = self._key()
            if not key:
                raise RuntimeError("no system1models.ai API key (S1M_API_KEY) is set")
            state = {"request": jev.scrub(request, API_REQUEST_CHARS),
                     "context": jev.scrub(context, CONTEXT_CHARS) or "(new conversation)"}
            body = json.dumps({"model": SOURCE, "state": state, "questions": jev.QUESTIONS}).encode()
            req = urllib.request.Request(self.url, data=body, headers={
                "Authorization": f"Bearer {key}", "Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                payload = json.loads(resp.read())
            a = payload["answers"]
            usage = payload.get("usage") or {}
            probs = a["category"].get("probabilities") or {}
            return jev.Classification(
                category=a["category"]["choice"], category_probs=probs,
                category_confidence=float(a["category"].get("confidence") or 0.0),
                difficulty=jev._score01(a["difficulty"], len(jev.QUESTIONS["difficulty"]["criteria"])),
                difficulty_confidence=float(a["difficulty"].get("confidence") or 0.0),
                needs_tools=float(a["needs_tools"]["noul"]), needs_vision=float(a["needs_vision"]["noul"]),
                needs_long_context=float(a["needs_long_context"]["noul"]), follow_up=float(a["follow_up"]["noul"]),
                stakes=jev._score01(a["stakes"], len(jev.QUESTIONS["stakes"]["criteria"])),
                latency_s=time.perf_counter() - started, model=SOURCE,
                input_tokens=int(usage.get("input_tokens") or 0), output_tokens=0,
                raw=a, source_name=SOURCE + "-api")
        except Exception:  # an API outage must never take down the routed LLM call
            return jev.FALLBACK
