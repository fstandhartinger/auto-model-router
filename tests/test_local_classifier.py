import sys
import types

from auto_router import jev


PAYLOAD = {
    "model": "laya-test",
    "answers": {
        "category": {"choice": "coding", "probabilities": {"coding": 0.9}, "confidence": 0.8},
        "difficulty": {"score": 2, "confidence": 0.7},
        "needs_tools": {"noul": 0.9},
        "needs_vision": {"noul": 0.1},
        "needs_long_context": {"noul": 0.2},
        "follow_up": {"noul": 0.3},
        "stakes": {"score": 2, "confidence": 0.6},
    },
    "usage": {"input_tokens": 42, "output_tokens": 0},
}


def test_local_laya_maps_to_router_classification(monkeypatch):
    class Agent:
        def predict(self, state, questions):
            assert "request" in state and questions is jev.QUESTIONS
            return PAYLOAD

    monkeypatch.setitem(sys.modules, "laya", types.SimpleNamespace(load=lambda model: Agent()))
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(set_num_threads=lambda n: None))
    result = jev.LocalLayaClassifier(threads=2)("write code")
    assert result.category == "coding" and result.difficulty == 0.5
    assert result.source == "local-laya" and result.model == "laya-test"
    assert result.input_tokens == 42 and result.output_tokens == 0


def test_classifier_backend_selection(monkeypatch):
    assert isinstance(jev.classifier_from_config({"classifier": {"backend": "local"}}),
                      jev.LocalLayaClassifier)
    assert jev.classifier_from_config({"classifier": {"backend": "hosted"}}) is jev.classify
    assert jev.classifier_from_config({"classifier": {"backend": "heuristic"}}) is None
    try:
        jev.classifier_from_config({"classifier": {"backend": "mystery"}})
    except ValueError as exc:
        assert "unknown classifier backend" in str(exc)
    else:
        raise AssertionError("unknown backend was accepted")


def test_local_import_failure_degrades_safely(monkeypatch):
    monkeypatch.setitem(sys.modules, "laya", None)
    result = jev.LocalLayaClassifier()("request")
    assert result.failed and result.source == "fallback"
