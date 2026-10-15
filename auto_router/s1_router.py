"""s1-llm-auto-router as the routing classifier: a 183 MB open model that answers the router's seven questions.

One encoder pass returns the same fields the policies read from Jev (category, difficulty, stakes,
needs_tools, needs_vision, needs_long_context, follow_up). Model card, weights and licence (MIT):
https://huggingface.co/decisionmodels/s1-llm-auto-router

Two backends share one result shape (``jev.Classification``):

* ``local-s1``: ONNX on CPU through ``onnxruntime`` and the ``tokenizers`` fast tokenizer; no PyTorch,
  no ``transformers``, no model server code. The three files are fetched from a pinned Hugging Face
  revision and checked against pinned SHA-256 digests before they are loaded; a mismatch is a
  failure, not a warning.
* ``decisionmodels``: the same model through ``api.decisionmodels.io`` with your own key
  (``DECISIONMODELS_API_KEY``). The key is read from the environment and never logged or returned.

Difficulty and stakes are the expected level of the model's 5- and 4-level heads scaled to 0..1, the
same mapping the published routing-agreement evaluation used (``_score01`` for the hosted answer).
Every failure degrades to ``jev.FALLBACK`` like the other backends, so a classifier problem never
takes down the routed LLM call.

Config::

    classifier: {backend: local-s1, threads: 2}
    classifier: {backend: decisionmodels}          # DECISIONMODELS_API_KEY
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import urllib.error
import urllib.request

from . import jev

REPO = "decisionmodels/s1-llm-auto-router"
#: The Hugging Face commit the digests below belong to. Bump both together.
REVISION = "572c5b6cdb53b77fbb26ed33568ebbc5b71c69c0"
MODEL_FILE = "model.gq.onnx"
FILES = {
    MODEL_FILE: "53a02107e46be7913f148037b63b4f4735a400f91221789b0aaefcde0ad8da80",
    "meta.json": "b6689ba21a44e0910cc2ae831c6ecf2bcc6f542b3fc5cd29d570e012865649c2",
    "tokenizer/tokenizer.json": "c01f8492531b2e569bd8d28ef1de1c66779275753c78cb497f7118762ae3230d",
}
CATS = ["coding", "agentic", "math", "knowledge", "long_context", "tool_use", "design", "summarisation", "general"]
FLAGS = ["needs_tools", "needs_vision", "needs_long_context", "follow_up"]
DIFFICULTY_LEVELS = ["trivial", "easy", "moderate", "hard", "frontier"]
STAKES_LEVELS = ["negligible", "low", "medium", "high"]
#: The model reads at most 512 tokens and the first 600 characters of the conversation summary.
CONTEXT_CHARS = 600
LOCAL_SOURCE = "local-s1"
HOSTED_SOURCE = "decisionmodels"
MODEL_ID = "s1-llm-auto-router"

API_URL = os.environ.get("AUTO_ROUTER_DECISIONMODELS_URL", "https://api.decisionmodels.io/v1/systemone")
API_KEY_ENV = "DECISIONMODELS_API_KEY"
#: Keeps the state well inside the API's size limit even for 3-byte scripts; the model truncates at 512 tokens anyway.
API_REQUEST_CHARS = 3000

#: The model answers exactly these seven questions with exactly these option sets (see the model card).
QUESTIONS = {
    "category": {"type": "choice", "criteria": {c: "" for c in CATS}},
    "difficulty": {"type": "score", "criteria": list(DIFFICULTY_LEVELS)},
    "stakes": {"type": "score", "criteria": list(STAKES_LEVELS)},
    **{f: {"type": "noul"} for f in FLAGS},
}


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def verify_files(root: str) -> None:
    """Raise unless every model file under ``root`` matches its pinned digest."""
    for name, want in FILES.items():
        path = os.path.realpath(os.path.join(root, name))
        if not os.path.isfile(path):
            raise FileNotFoundError(f"s1-llm-auto-router file missing: {name}")
        if sha256_file(path) != want:
            raise ValueError(f"s1-llm-auto-router file {name} does not match the pinned SHA-256")


def deps_available() -> bool:
    """True when the optional CPU runtime for the local model is installed (extra ``[s1-router]``)."""
    import importlib.util
    return all(importlib.util.find_spec(m) is not None
               for m in ("onnxruntime", "tokenizers", "huggingface_hub", "numpy"))


def _softmax(x):
    import numpy as np
    e = np.exp(x - x.max())
    return e / e.sum()


def _expected_level(probs: list[float]) -> float:
    """Expected level of an ordinal head, scaled to 0..1 (the router's difficulty / stakes scale)."""
    return sum(i * p for i, p in enumerate(probs)) / (len(probs) - 1)


class LocalS1Classifier:
    """Lazy, pinned, CPU-only ONNX classifier with the same result shape as hosted Jev."""

    def __init__(self, model: str = REPO, threads: int = 2):
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
        return snapshot_download(self.model, revision=REVISION, allow_patterns=[MODEL_FILE, "meta.json", "tokenizer/*"])

    def _load(self):
        if self._session is None:
            with self._lock:
                if self._session is None:
                    import onnxruntime as ort
                    from tokenizers import Tokenizer
                    root = self._files()
                    verify_files(root)
                    meta = json.load(open(os.path.join(root, "meta.json")))
                    tok = Tokenizer.from_file(os.path.realpath(os.path.join(root, "tokenizer", "tokenizer.json")))
                    tok.enable_truncation(min(int(meta.get("max_len") or 512), 512), strategy="only_second")
                    tok.no_padding()
                    opts = ort.SessionOptions()
                    opts.intra_op_num_threads = self.threads
                    opts.inter_op_num_threads = 1
                    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                    self._temps = meta.get("temperatures") or {}
                    self._tokenizer = tok
                    self._session = ort.InferenceSession(os.path.realpath(os.path.join(root, MODEL_FILE)), opts,
                                                         providers=["CPUExecutionProvider"])
        return self._session, self._tokenizer

    def warm(self) -> bool:
        """Load the model now (and run one request) so the first real turn is not the slow one."""
        return not self("warm-up request").failed

    def predict(self, request: str, context: str = "") -> dict:
        """Calibrated head outputs for one turn: probability per category, level and flag."""
        import numpy as np
        session, tok = self._load()
        ctx = jev.scrub(context, jev.CONTEXT_CHARS)[:CONTEXT_CHARS] or "(new conversation)"
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
            return jev.Classification(
                category=category, category_probs=probs, category_confidence=float(probs[category]),
                difficulty=_expected_level(out["difficulty"]), difficulty_confidence=float(max(out["difficulty"])),
                needs_tools=out["flags"]["needs_tools"], needs_vision=out["flags"]["needs_vision"],
                needs_long_context=out["flags"]["needs_long_context"], follow_up=out["flags"]["follow_up"],
                stakes=_expected_level(out["stakes"]), latency_s=time.perf_counter() - started, model=MODEL_ID,
                input_tokens=out["input_tokens"], output_tokens=0, source_name=LOCAL_SOURCE)
        except Exception:  # a local model failure must never take down the routed LLM call
            return jev.FALLBACK


def classification_from_answers(a: dict, *, latency_s: float, model: str, usage: dict | None = None,
                                source_name: str = HOSTED_SOURCE) -> jev.Classification:
    """Turn a ``/v1/systemone`` answer set for the seven questions into a ``jev.Classification``."""
    usage = usage or {}
    # Validate shapes up front so malformed answers raise TypeError (caught by the caller), never AttributeError.
    if not isinstance(a, dict) or not all(isinstance(a.get(q), dict) for q in QUESTIONS) \
            or not isinstance(a["category"].get("choice"), str) or not isinstance(usage, dict):
        raise TypeError("malformed /v1/systemone answers")
    return jev.Classification(
        category=a["category"]["choice"], category_probs=a["category"].get("probabilities") or {},
        category_confidence=jev._confidence(a["category"]),
        difficulty=jev._score01(a["difficulty"], len(DIFFICULTY_LEVELS)),
        difficulty_confidence=jev._confidence(a["difficulty"]),
        needs_tools=float(a["needs_tools"]["noul"]), needs_vision=float(a["needs_vision"]["noul"]),
        needs_long_context=float(a["needs_long_context"]["noul"]), follow_up=float(a["follow_up"]["noul"]),
        stakes=jev._score01(a["stakes"], len(STAKES_LEVELS)), latency_s=latency_s, model=model,
        input_tokens=int(usage.get("input_tokens") or 0), output_tokens=int(usage.get("output_tokens") or 0),
        raw=a, source_name=source_name)


class DecisionModelsClassifier:
    """s1-llm-auto-router through api.decisionmodels.io: one request answers all seven questions."""

    def __init__(self, api_key_env: str | None = None, url: str = API_URL, timeout: float = 10.0):
        self.api_key_env = api_key_env or API_KEY_ENV
        self.url = url
        self.timeout = timeout

    def __call__(self, request: str, context: str = "") -> jev.Classification:
        started = time.perf_counter()
        try:
            key = os.environ.get(self.api_key_env)
            if not key:
                raise RuntimeError(f"{self.api_key_env} is not set")
            state = {"request": jev.scrub(request, API_REQUEST_CHARS),
                     "context": jev.scrub(context, CONTEXT_CHARS)[:CONTEXT_CHARS] or "(new conversation)"}
            body = json.dumps({"model": MODEL_ID, "state": state, "questions": QUESTIONS}).encode()
            req = urllib.request.Request(self.url, data=body, headers={
                "Authorization": f"Bearer {key}", "Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                payload = json.loads(resp.read())
            return classification_from_answers(
                payload["answers"], latency_s=time.perf_counter() - started,
                model=str(payload.get("model") or MODEL_ID), usage=payload.get("usage"))
        except (urllib.error.URLError, KeyError, TypeError, ValueError, RuntimeError, OSError,
                AttributeError, OverflowError):
            return jev.FALLBACK  # an API outage must never take down the routed LLM call
