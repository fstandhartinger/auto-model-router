import hashlib
import json
import os
import types
import urllib.error

import pytest

from auto_router import jev, s1_router


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
    clf = s1_router.LocalS1Classifier("some/dir", threads=1)
    session, tok = FakeSession(), FakeTokenizer()
    monkeypatch.setattr(clf, "_load", lambda: (session, tok))
    return clf, session, tok


def test_local_s1_maps_to_router_classification(monkeypatch):
    clf, session, _ = loaded(monkeypatch)
    result = clf("fix the failing test in utils.py", "3 messages, 2 from the user.")
    assert result.source == "local-s1" and not result.failed
    assert result.category == "coding" and result.category_confidence > 0.95
    assert sum(result.category_probs.values()) == pytest.approx(1.0)
    assert result.difficulty == pytest.approx(0.5, abs=1e-3)       # level 2 of 0..4
    assert result.stakes == pytest.approx(1.0, abs=1e-3)           # level 3 of 0..3
    assert result.difficulty_confidence > 0.99
    assert result.needs_tools == pytest.approx(0.982, abs=1e-3) and result.follow_up == pytest.approx(0.5)
    assert result.input_tokens == 4 and result.output_tokens == 0
    assert session.feeds["input_ids"].tolist() == [[1, 2, 3, 4]]


def test_local_s1_scrubs_and_truncates_context(monkeypatch):
    clf, _, tok = loaded(monkeypatch)
    clf("deploy with API_KEY=sk-abcdefghijklmnopqrstuvwxyz0123456789", "x" * 5000)
    ctx, req = tok.pairs[0]
    assert len(ctx) == s1_router.CONTEXT_CHARS
    assert "sk-abcdefghijklmnopqrstuvwxyz0123456789" not in req
    clf("hello")
    assert tok.pairs[1][0] == "(new conversation)"


def test_local_s1_failure_is_fallback(monkeypatch):
    clf = s1_router.LocalS1Classifier("/does/not/exist", threads=1)
    monkeypatch.setattr(clf, "_files", lambda: (_ for _ in ()).throw(OSError("no model")))
    assert clf("hello") is jev.FALLBACK


def test_pinned_digests_are_checked(tmp_path):
    names = list(s1_router.FILES)
    for n in names:
        (tmp_path / n).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / n).write_bytes(b"not the model")
    with pytest.raises(ValueError, match="pinned SHA-256"):
        s1_router.verify_files(str(tmp_path))
    (tmp_path / names[0]).unlink()
    with pytest.raises(FileNotFoundError):
        s1_router.verify_files(str(tmp_path))


def test_tampered_local_directory_degrades_to_fallback(tmp_path):
    pytest.importorskip("onnxruntime"); pytest.importorskip("tokenizers")
    for n in s1_router.FILES:
        (tmp_path / n).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / n).write_bytes(b"{}")
    assert s1_router.LocalS1Classifier(str(tmp_path))("hello") is jev.FALLBACK


def test_sha256_file(tmp_path):
    p = tmp_path / "f"
    p.write_bytes(b"abc")
    assert s1_router.sha256_file(str(p)) == hashlib.sha256(b"abc").hexdigest()


@pytest.mark.parametrize("name", ["local-s1", "s1-llm-auto-router", "s1-router"])
def test_local_backend_config_wiring(name):
    clf = jev.classifier_from_config({"classifier": {"backend": name, "model": "/m", "threads": 3}})
    assert isinstance(clf, s1_router.LocalS1Classifier)
    assert clf.model == "/m" and clf.threads == 3
    default = jev.classifier_from_config({"classifier": {"backend": name}})
    assert default.model == s1_router.REPO and default.threads == 2


def test_decisionmodels_backend_config_wiring():
    clf = jev.classifier_from_config({"classifier": {"backend": "decisionmodels", "api_key_env": "MY_KEY"}})
    assert isinstance(clf, s1_router.DecisionModelsClassifier) and clf.api_key_env == "MY_KEY"
    assert jev.classifier_from_config({"classifier": {"backend": "decisionmodels"}}).api_key_env == "DECISIONMODELS_API_KEY"


def test_unknown_backend_message_lists_the_new_ones():
    with pytest.raises(ValueError, match="local-s1, decisionmodels"):
        jev.classifier_from_config({"classifier": {"backend": "nope"}})


def test_questions_are_exactly_the_models_seven():
    assert list(s1_router.QUESTIONS) == ["category", "difficulty", "stakes", *s1_router.FLAGS]
    assert set(s1_router.QUESTIONS["category"]["criteria"]) == set(jev.CATEGORY_OPTIONS)
    assert len(s1_router.QUESTIONS["difficulty"]["criteria"]) == len(jev.QUESTIONS["difficulty"]["criteria"]) == 5
    assert len(s1_router.QUESTIONS["stakes"]["criteria"]) == len(jev.QUESTIONS["stakes"]["criteria"]) == 4


def _api_answers():
    return {
        "category": {"type": "choice", "choice": "math", "confidence": 0.9,
                     "probabilities": {c: (0.9 if c == "math" else 0.0125) for c in s1_router.CATS}},
        "difficulty": {"type": "score", "score": 3.0, "confidence": 0.8,
                       "probabilities": {"0": 0, "1": 0, "2": 0, "3": 1, "4": 0}},
        "stakes": {"type": "score", "score": 0.0, "confidence": 0.7,
                   "probabilities": {"0": 1, "1": 0, "2": 0, "3": 0}},
        "needs_tools": {"type": "noul", "noul": 0.2}, "needs_vision": {"type": "noul", "noul": 0.01},
        "needs_long_context": {"type": "noul", "noul": 0.03}, "follow_up": {"type": "noul", "noul": 0.6},
    }


class Resp:
    def __init__(self, body):
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self):
        return json.dumps(self.body).encode()


def test_decisionmodels_sends_the_seven_questions_in_one_request(monkeypatch):
    sent = {}

    def fake_urlopen(req, timeout):
        sent["url"], sent["body"], sent["auth"] = req.full_url, json.loads(req.data), req.headers["Authorization"]
        return Resp({"model": "s1-llm-auto-router", "answers": _api_answers(),
                     "usage": {"input_tokens": 42, "output_tokens": 0}})

    monkeypatch.setenv("DECISIONMODELS_API_KEY", "dm_test_key")
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    out = s1_router.DecisionModelsClassifier()("Prove that sqrt(2) is irrational", "")
    assert sent["url"] == "https://api.decisionmodels.io/v1/systemone"
    assert sent["body"]["model"] == "s1-llm-auto-router"
    assert sent["body"]["questions"] == s1_router.QUESTIONS
    assert sent["body"]["state"]["context"] == "(new conversation)"
    assert sent["auth"] == "Bearer dm_test_key"
    assert out.category == "math" and out.difficulty == pytest.approx(0.75) and out.stakes == 0.0
    assert out.follow_up == pytest.approx(0.6) and out.input_tokens == 42
    assert out.source == "decisionmodels" and not out.failed


def test_decisionmodels_never_leaks_the_key(monkeypatch):
    monkeypatch.setenv("DECISIONMODELS_API_KEY", "dm_secret_key_value")

    def boom(req, timeout):
        raise urllib.error.HTTPError(req.full_url, 401, "unauthorized", {}, None)

    monkeypatch.setattr("urllib.request.urlopen", boom)
    out = s1_router.DecisionModelsClassifier()("hi")
    assert out is jev.FALLBACK and "dm_secret_key_value" not in repr(out)


def test_decisionmodels_without_key_falls_back_without_a_request(monkeypatch):
    monkeypatch.delenv("DECISIONMODELS_API_KEY", raising=False)

    def forbidden(*a, **k):
        raise AssertionError("no request may be sent without a key")

    monkeypatch.setattr("urllib.request.urlopen", forbidden)
    assert s1_router.DecisionModelsClassifier()("hi") is jev.FALLBACK


def test_decisionmodels_malformed_answer_falls_back(monkeypatch):
    monkeypatch.setenv("DECISIONMODELS_API_KEY", "k")
    monkeypatch.setattr("urllib.request.urlopen", lambda req, timeout: Resp({"answers": {"category": {}}}))
    assert s1_router.DecisionModelsClassifier()("hi") is jev.FALLBACK


@pytest.mark.skipif(not os.environ.get("S1_LIVE_MODEL"), reason="set S1_LIVE_MODEL=1 (pinned model already in the HF cache; the suite blocks network)")
def test_pinned_revision_loads_and_routes_real_requests():
    pytest.importorskip("onnxruntime"); pytest.importorskip("tokenizers"); pytest.importorskip("huggingface_hub")
    clf = s1_router.LocalS1Classifier(threads=2)
    code = clf("Write a Python function that returns the square of a number.")
    assert not code.failed and code.category == "coding" and code.difficulty < 0.5
    chat = clf("hi, how are you?")
    assert chat.category == "general" and chat.difficulty < code.difficulty + 0.25
    hard = clf("Prove that every bounded monotone sequence of reals converges, then generalise to nets.")
    assert hard.category == "math"
