import sys
import types

import pytest

from auto_router import jev, route_head


class FakeSession:
    def __init__(self):
        self.feeds = None

    def run(self, _outputs, feeds):
        import numpy as np
        self.feeds = feeds
        category = np.zeros((1, 9), dtype=np.float32); category[0, 0] = 0.9; category[0, 8] = 0.1
        scalars = np.asarray([[0.7, 0.6, 0.4]], dtype=np.float32)
        noul = np.asarray([[0.8, 0.0, 0.1, 0.3]], dtype=np.float32)
        succ = np.asarray([[0.2, 0.6, 0.9]], dtype=np.float32)
        route = np.asarray([[0.1, 0.7, 0.2]], dtype=np.float32)
        return [category, scalars, noul, succ, route]


class FakeTokenizer:
    def encode(self, text):
        assert text.startswith("Context: ") and "\nRequest: " in text
        return types.SimpleNamespace(ids=[1, 2, 3])


def loaded(monkeypatch):
    pytest.importorskip("numpy")
    clf = route_head.LocalRouteHeadClassifier("some/repo", "int8", threads=1)
    session = FakeSession()
    monkeypatch.setattr(clf, "_load", lambda: (session, FakeTokenizer()))
    return clf, session


def test_route_head_maps_to_router_classification(monkeypatch):
    clf, session = loaded(monkeypatch)
    result = clf("fix the failing test in utils.py", "3 messages, 2 from the user.")
    assert result.source == "local-route-head" and not result.failed
    assert result.category == "coding" and result.category_confidence == pytest.approx(0.9)
    assert result.difficulty == pytest.approx(0.6)
    assert result.difficulty_confidence == pytest.approx(0.8)      # |0.6 - 0.7| * 2 = 0.2 apart
    assert result.needs_tools == pytest.approx(0.8) and result.stakes == pytest.approx(0.4)
    assert result.raw["succ"]["strong"] == pytest.approx(0.9)
    assert session.feeds["state"].shape == (1, route_head.N_STATE)
    assert session.feeds["input_ids"].tolist() == [[1, 2, 3]]


def test_route_head_scrubs_credentials(monkeypatch):
    clf, _ = loaded(monkeypatch)
    seen = {}

    class Tok(FakeTokenizer):
        def encode(self, text):
            seen["text"] = text
            return super().encode(text)

    monkeypatch.setattr(clf, "_load", lambda: (FakeSession(), Tok()))
    clf("use key sk-ant-abcdefghijklmnopqrstuvwxyz0123 to call the API")
    assert "sk-ant-" not in seen["text"]


def test_route_head_long_request_keeps_head_and_tail():
    text = "A" * 3000 + "QUESTION AT THE END"
    rendered = route_head.render(text)
    assert rendered.endswith("QUESTION AT THE END") and "[...]" in rendered


def test_route_head_failure_degrades_safely(monkeypatch):
    monkeypatch.setitem(sys.modules, "onnxruntime", None)
    result = route_head.LocalRouteHeadClassifier("/nonexistent/dir")("request")
    assert result.failed and result.source == "fallback"


def test_route_head_backend_selection():
    clf = jev.classifier_from_config({"classifier": {"backend": "local-route-head", "variant": "int4"}})
    assert isinstance(clf, route_head.LocalRouteHeadClassifier)
    assert clf.model == route_head.DEFAULT_MODEL and clf.variant == "int4"
    with pytest.raises(ValueError):
        route_head.LocalRouteHeadClassifier(variant="int3")
