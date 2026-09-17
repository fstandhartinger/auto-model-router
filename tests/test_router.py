from auto_router.catalog import Catalog
from auto_router.config import Provider, RouterConfig
from auto_router.jev import Classification
from auto_router.policies import EscalatePolicy
from auto_router.router import Router, is_tool_continuation, recent_tool_errors
from tests.helpers import CHEAP, FRONTIER, MID


def stub(category="coding", difficulty=0.1):
    def classify(text, context):
        return Classification(category, {}, difficulty, 1.0, 0.9, 0.0, 0.0, 0.2, 0.3, 0.01)
    return classify


def make_router(**kw):
    cfg = RouterConfig(providers={"p": Provider("p", "https://example.invalid/v1")},
                       catalog=Catalog([CHEAP, MID, FRONTIER]))
    return Router(cfg, policy=EscalatePolicy(), quota_reader=lambda: {}, **kw)


SYSTEM = "You are a coding agent. " * 200
TOOLS = [{"name": "bash", "input_schema": {"type": "object"}}]


def test_tool_loop_detection_both_formats():
    assert is_tool_continuation([{"role": "tool", "content": "ok", "tool_call_id": "1"}])
    assert is_tool_continuation([{"role": "user", "content": [{"type": "tool_result", "tool_use_id": "1"}]}])
    assert not is_tool_continuation([{"role": "user", "content": "next task"}])


def test_counts_consecutive_failing_tool_results():
    convo = [{"role": "user", "content": "fix it"}]
    for _ in range(3):
        convo += [{"role": "assistant", "content": [{"type": "tool_use", "id": "t", "name": "bash", "input": {}}]},
                 {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t",
                                               "content": "1 FAILED, 3 passed"}]}]
    assert recent_tool_errors(convo) == 3


def test_turn_start_classifies_then_tool_loop_sticks():
    r = make_router(classifier=stub(difficulty=0.1))
    convo = [{"role": "user", "content": "rename foo to bar"}]
    first = r.route(convo, SYSTEM, TOOLS, now=1000.0)
    assert first.turn_start and first.model.name == "cheap"
    r.commit(first, 5000, 200)
    convo += [{"role": "assistant", "content": [{"type": "tool_use", "id": "t", "name": "bash", "input": {}}]},
             {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t", "content": "done"}]}]
    second = r.route(convo, SYSTEM, TOOLS, now=1005.0)
    assert not second.turn_start and second.model.name == "cheap"
    assert second.conversation_id == first.conversation_id


def test_repeated_tool_failures_escalate_mid_loop():
    r = make_router(classifier=stub(difficulty=0.1))
    convo = [{"role": "user", "content": "make the tests pass"}]
    first = r.route(convo, SYSTEM, TOOLS, now=1000.0)
    r.commit(first, 5000, 200)
    for _ in range(3):
        convo += [{"role": "assistant", "content": [{"type": "tool_use", "id": "t", "name": "bash", "input": {}}]},
                 {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t",
                                               "content": "Traceback: AssertionError"}]}]
    nxt = r.route(convo, SYSTEM, TOOLS, now=1010.0)
    assert nxt.model.cap("coding") > first.model.cap("coding")


def test_without_classifier_falls_back_to_a_cautious_middle():
    r = make_router(classifier=None)
    r.classifier = None
    res = r.route([{"role": "user", "content": "hello"}], None, None, now=1.0)
    assert res.request.difficulty == 0.5
