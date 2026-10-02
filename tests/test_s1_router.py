import json
import os
import sys
import types

import pytest

from auto_router import jev, route_head, s1_router


class FakeSession:
    def __init__(self):
        self.feeds = None

    def run(self, _outputs, feeds):
        import numpy as np
        self.feeds = feeds
        category = np.zeros((1, 9), dtype=np.float32); category[0, 0] = 8.0
        difficulty = np.asarray([[0.0, 0.0, 9.0, 0.0, 0.0]], dtype=np.float32)
        stakes = np.asarray([[0.0, 0.0, 0.0, 9.0]], dtype=np.float32)
        flags = np.asarray([[4.0, -6.0, -6.0, 0.0]], dtype=np.float32)
        return [category, difficulty, stakes, flags]


class FakeTokenizer:
    def __init__(self):
        self.pairs = []

    def encode(self, context, request):
        self.pairs.append((context, request))
        return types.SimpleNamespace(ids=[1, 2, 3, 4])


def loaded(monkeypatch):
    pytest.importorskip("numpy")
    clf = s1_router.LocalS1RouterClassifier("some/dir", threads=1)
    session, tok = FakeSession(), FakeTokenizer()
    monkeypatch.setattr(clf, "_load", lambda: (session, tok))
    return clf, session, tok


def test_s1_router_maps_to_router_classification(monkeypatch):
    clf, session, _ = loaded(monkeypatch)
    result = clf("fix the failing test in utils.py", "3 messages, 2 from the user.")
    assert result.source == "s1-llm-auto-router" and not result.failed
    assert result.category == "coding" and result.category_confidence > 0.95
    assert sum(result.category_probs.values()) == pytest.approx(1.0)
    assert result.difficulty == pytest.approx(0.5, abs=1e-3)       # level 2 of 0..4
    assert result.stakes == pytest.approx(1.0, abs=1e-3)           # level 3 of 0..3
    assert result.difficulty_confidence > 0.99
    assert result.needs_tools == pytest.approx(0.982, abs=1e-3) and result.follow_up == pytest.approx(0.5)
    assert result.input_tokens == 4 and result.output_tokens == 0
    assert session.feeds["input_ids"].tolist() == [[1, 2, 3, 4]]


def test_s1_router_scrubs_and_truncates_context(monkeypatch):
    clf, _, tok = loaded(monkeypatch)
    clf("deploy with API_KEY=sk-abcdefghijklmnopqrstuvwxyz0123456789", "x" * 5000)
    ctx, req = tok.pairs[0]
    assert len(ctx) == s1_router.CONTEXT_CHARS
    assert "sk-abcdefghijklmnopqrstuvwxyz0123456789" not in req


def test_s1_router_failure_is_fallback(monkeypatch):
    clf = s1_router.LocalS1RouterClassifier("/does/not/exist", threads=1)
    monkeypatch.setattr(clf, "_files", lambda: (_ for _ in ()).throw(OSError("no model")))
    assert clf("hello") is jev.FALLBACK


@pytest.mark.parametrize("name", ["s1-llm-auto-router", "s1-router"])
def test_s1_router_config_wiring(name):
    clf = jev.classifier_from_config({"classifier": {"backend": name, "model": "/m", "threads": 3}})
    assert isinstance(clf, s1_router.LocalS1RouterClassifier)
    assert clf.model == "/m" and clf.threads == 3
    default = jev.classifier_from_config({"classifier": {"backend": name}})
    assert default.model == s1_router.DEFAULT_MODEL and default.threads == 2


def test_route_head_materialises_symlinked_snapshot(tmp_path, monkeypatch):
    blobs, snap = tmp_path / "blobs", tmp_path / "snap"
    blobs.mkdir(); snap.mkdir()
    for f in ["model_fp16.onnx", "model_fp16.onnx.data", "tokenizer.json"]:
        (blobs / f).write_text(f)
        os.symlink(blobs / f, snap / f)
    monkeypatch.setenv("AUTO_ROUTER_CACHE", str(tmp_path / "cache"))
    out = route_head._materialise(str(snap), "model_fp16.onnx")
    assert out != str(snap)
    for f in ["model_fp16.onnx", "model_fp16.onnx.data", "tokenizer.json"]:
        p = os.path.join(out, f)
        assert os.path.isfile(p) and not os.path.islink(p) and open(p).read() == f
    assert route_head._materialise(str(snap), "model_fp16.onnx") == out     # cached on the second call
    plain = tmp_path / "plain"; plain.mkdir(); (plain / "model_fp16.onnx").write_text("x")
    assert route_head._materialise(str(plain), "model_fp16.onnx") == str(plain)


S1R_REF = "/home/flori/jobs/s1-llm-auto-router-20261001"


@pytest.mark.skipif(not os.environ.get("S1R_MODEL_DIR"), reason="set S1R_MODEL_DIR to run the parity check")
def test_s1_router_parity_with_reference():
    pytest.importorskip("onnxruntime"); pytest.importorskip("transformers")
    sys.path.insert(0, os.path.join(S1R_REF, "serve"))
    from router_onnx import RouterONNX
    model_dir = os.environ["S1R_MODEL_DIR"]
    ref = RouterONNX(model_dir, "gq", threads=2)
    clf = s1_router.LocalS1RouterClassifier(model_dir, threads=2)
    rows = [json.loads(line) for line in open(os.path.join(S1R_REF, "data/final/test.jsonl"))][:50]
    for r in rows:
        want, _ = ref.answers(r["request"], r["context"])
        got = clf.predict(r["request"], r["context"])
        assert max(got["category"], key=got["category"].get) == want["category"]["choice"]
        for c, p in want["category"]["probabilities"].items():
            assert got["category"][c] == pytest.approx(p, abs=1e-3)
        for f in s1_router.FLAGS:
            assert got["flags"][f] == pytest.approx(want[f]["noul"], abs=1e-3)
