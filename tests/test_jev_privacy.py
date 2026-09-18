"""Privacy and transport behaviour of the Jev client.

The rule the router has to keep: nothing that leaves this process toward an
external judgement service may carry a credential, and a secret must not be
able to survive by sitting past the truncation point.
"""

import json
import urllib.error

import pytest

from auto_router import jev


SYNTHETIC = "synthetic-sensitive-value"


def test_scrub_removes_a_matching_environment_value(monkeypatch):
    monkeypatch.setenv("EXAMPLE_API_KEY", SYNTHETIC)
    assert SYNTHETIC not in jev.scrub("Please inspect " + SYNTHETIC, 6000)


def test_scrub_runs_before_truncation(monkeypatch):
    """A secret past the cut is still removed, not merely cut off."""
    monkeypatch.setenv("EXAMPLE_API_KEY", SYNTHETIC)
    prefix = "x" * 5995
    assert jev.scrub(prefix + SYNTHETIC, 6000) == prefix + "[REDA"
    # Even when the secret starts beyond the limit it is removed first.
    assert jev.scrub("x" * 6001 + SYNTHETIC, 6000) == "x" * 6000
    assert "synthetic" not in jev.scrub(SYNTHETIC, 9)


def test_scrub_leaves_ordinary_text_alone():
    assert jev.scrub("Implement stable sorting", 9) == "Implement"
    assert jev.scrub("Refactor the pricing module", 6000) == "Refactor the pricing module"


def test_short_environment_values_are_not_redacted(monkeypatch):
    """A one-word env value would otherwise redact ordinary prose."""
    monkeypatch.setenv("REGION_KEY", "eu")
    assert jev.scrub("the eu region", 6000) == "the eu region"


@pytest.mark.parametrize("text", [
    f"password={SYNTHETIC}",
    f'{{"api_key": "{SYNTHETIC}"}}',
    f"CLIENT_SECRET: '{SYNTHETIC}'",
    f"Authorization: Bearer {SYNTHETIC}",
    f"proxy-credential = {SYNTHETIC}",
    f"https://user:{SYNTHETIC}@example.invalid/path",
    f"postgres://admin:{SYNTHETIC}@db.example.invalid:5432/app",
])
def test_scrub_credential_assignments_and_urls(text):
    assert SYNTHETIC not in jev.scrub(text, 6000)


#: Built by concatenation on purpose. A literal of a real credential shape in a
#: source file trips every secret scanner - including this repository's own
#: publication gate - and a repository that cries wolf teaches people to ignore it.
SHAPES = [
    ("anthropic-key", "sk-" + "ant-api03-" + "A" * 24),
    ("openai-key", "sk-" + "proj-" + "B" * 32),
    ("openai-key", "sk-" + "C" * 32),
    ("github-token", "gh" + "p_" + "D" * 36),
    ("github-fine-grained", "github" + "_pat_" + "E" * 30),
    ("slack-token", "xo" + "xb-1234567890-abcdefghij"),
    ("google-api-key", "AI" + "za" + "F" * 35),
    ("aws-access-key-id", "AK" + "IA" + "IOSFODNN7EXAMPLE"),
    ("stripe-key", "sk" + "_live_" + "G" * 24),
    ("hugging-face-token", "h" + "f_" + "H" * 34),
    ("jwt", "ey" + "JhbGciOiJIUzI1NiJ9." + "eyJzdWIiOiIxMjM0NTY3ODkwIn0." + "dBjftJeZ4CVPmB92K27u"),
]


@pytest.mark.parametrize("shape,sample", SHAPES)
def test_known_secret_shapes_are_detected_and_removed(shape, sample):
    """Known secret-shaped cases, as the privacy requirement asks for."""
    assert shape in jev.scrub_report(sample)
    scrubbed = jev.scrub(f"here is the value {sample} in a sentence", 6000)
    assert sample not in scrubbed
    assert "[REDACTED]" in scrubbed


#: Assembled at run time for the same reason as SHAPES above: a literal PEM
#: header in a source file is reported as a private key by every scanner.
DASHES = "-" * 5


def _pem(kind: str, body: str, terminated: bool = True) -> str:
    header = f"{DASHES}BEGIN {kind}{DASHES}"
    footer = f"{DASHES}END {kind}{DASHES}"
    return f"{header}\n{body}" + (f"\n{footer}" if terminated else "")


def test_scrub_multiline_private_key_block():
    text = _pem("PRIVATE KEY", "synthetic-material")
    assert jev.scrub(text, 6000) == "[REDACTED]"
    assert "private-key-block" in jev.scrub_report(text)


def test_unterminated_private_key_block_is_still_removed():
    text = _pem("RSA PRIVATE KEY", "synthetic-material-with-no-end-marker", terminated=False)
    assert "synthetic-material" not in jev.scrub(text, 6000)


def test_a_certificate_body_is_removed_too():
    text = _pem("CERTIFICATE", "synthetic-certificate-body")
    assert "synthetic-certificate-body" not in jev.scrub(text, 6000)


def test_classifier_and_judge_scrub_every_field(monkeypatch):
    """Both public entry points scrub all of their state fields."""
    states = []

    def capture(state, questions, api_key, timeout, **kw):
        states.append(state)
        raise RuntimeError("offline")

    monkeypatch.setattr(jev, "_post", capture)
    text = f"password={SYNTHETIC} and " + "sk-" + "ant-api03-" + "A" * 24
    assert jev.classify(text, text).failed
    assert jev.judge(text, text).failed
    blob = json.dumps(states)
    assert SYNTHETIC not in blob
    assert "ant-api03" not in blob
    assert {"request", "context"} <= set(states[0])
    assert {"request", "response"} <= set(states[1])


def test_field_caps_bound_what_is_sent(monkeypatch):
    states = []

    def capture(state, questions, api_key, timeout, **kw):
        states.append(state)
        raise RuntimeError("offline")

    monkeypatch.setattr(jev, "_post", capture)
    jev.classify("y" * 50_000, "z" * 50_000)
    assert len(states[0]["request"]) == jev.REQUEST_CHARS
    assert len(states[0]["context"]) == jev.CONTEXT_CHARS


# -- transport --------------------------------------------------------------
class _FakeHTTPError(urllib.error.HTTPError):
    def __init__(self, code, retry_after=None):
        headers = {"retry-after": retry_after} if retry_after else {}
        super().__init__("https://example.invalid", code, "boom", headers, None)


def test_retry_after_header_is_honoured():
    assert jev._retry_after_seconds(_FakeHTTPError(429, "2"), 0) == 2.0
    assert jev._retry_after_seconds(_FakeHTTPError(429, "9999"), 0) == jev.MAX_BACKOFF_S


def test_backoff_is_bounded_without_a_header():
    delays = [jev._retry_after_seconds(_FakeHTTPError(429), i) for i in range(8)]
    assert delays[0] == jev.BASE_BACKOFF_S
    assert delays == sorted(delays)
    assert max(delays) <= jev.MAX_BACKOFF_S


def test_post_retries_rate_limits_then_succeeds(monkeypatch):
    calls, slept = [], []
    payload = json.dumps({"model": "jev-1.13.0", "answers": {}}).encode()

    class _Resp:
        def read(self):
            return payload

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def urlopen(req, timeout):
        calls.append(req)
        if len(calls) < 3:
            raise _FakeHTTPError(429, "0")
        return _Resp()

    monkeypatch.setattr(jev.urllib.request, "urlopen", urlopen)
    data, _ = jev._post({}, {}, "test-key", 5.0, sleep=slept.append)
    assert data["model"] == "jev-1.13.0"
    assert len(calls) == 3 and len(slept) == 2


def test_post_does_not_retry_an_authentication_error(monkeypatch):
    calls = []

    def urlopen(req, timeout):
        calls.append(req)
        raise _FakeHTTPError(401)

    monkeypatch.setattr(jev.urllib.request, "urlopen", urlopen)
    with pytest.raises(urllib.error.HTTPError):
        jev._post({}, {}, "test-key", 5.0, sleep=lambda _s: None)
    assert len(calls) == 1


def test_classifier_failure_is_a_cautious_fallback(monkeypatch):
    monkeypatch.setattr(jev, "_post", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("down")))
    cls = jev.classify("anything")
    assert cls.failed and cls.source == "fallback"
    assert cls.category == "general" and cls.difficulty == 0.5
    assert cls.category_confidence == 0.0 and cls.difficulty_confidence == 0.0


def test_missing_api_key_never_raises_out_of_classify(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.setattr(jev.urllib.request, "urlopen",
                        lambda *a, **k: pytest.fail("no request may be made without a key"))
    assert jev.classify("hello").failed
    assert jev.judge("hello", "world").failed


def test_classification_parses_confidence_and_model(monkeypatch):
    answers = {
        "category": {"type": "choice", "choice": "design",
                     "probabilities": {"design": 0.8, "coding": 0.2}, "confidence": 0.77},
        "difficulty": {"type": "score", "score": 2.0, "confidence": 0.61},
        "needs_tools": {"type": "noul", "noul": 0.1},
        "needs_vision": {"type": "noul", "noul": 0.0},
        "needs_long_context": {"type": "noul", "noul": 0.05},
        "follow_up": {"type": "noul", "noul": 0.2},
        "stakes": {"type": "score", "score": 1.0},
    }
    monkeypatch.setattr(jev, "_post", lambda *a, **k: (
        {"model": "jev-1.13.0", "answers": answers,
         "usage": {"input_tokens": 312, "output_tokens": 48}}, 0.4))
    cls = jev.classify("Build a landing page")
    assert cls.category == "design" and cls.model == "jev-1.13.0"
    assert cls.category_confidence == 0.77 and cls.difficulty_confidence == 0.61
    # score 2 of 5 ordered levels -> 0.5 on the 0..1 axis
    assert cls.difficulty == pytest.approx(0.5)
    assert cls.stakes == pytest.approx(1 / 3)
    assert cls.input_tokens == 312 and not cls.failed


def test_noul_answers_carry_no_confidence_field():
    """The API documents confidence on Choice and Score only."""
    assert jev._confidence({"type": "noul", "noul": 0.9}) == 0.0


def test_design_and_summarisation_are_offered_as_categories():
    from auto_router.catalog import CATEGORIES
    assert "design" in jev.CATEGORY_OPTIONS and "summarisation" in jev.CATEGORY_OPTIONS
    assert set(jev.CATEGORY_OPTIONS) <= set(CATEGORIES)


def test_scrub_stays_linear_on_a_large_prompt():
    """A long unbroken word run must not trigger quadratic regex backtracking.

    The earlier pattern anchored an unbounded ``[\\w.-]*`` in front of a
    literal, which took over a second on a 200k-character paste - long enough
    to dominate a routing hop.
    """
    import time

    def elapsed(n):
        text = "y" * n
        start = time.perf_counter()
        jev.scrub(text, 6000)
        return time.perf_counter() - start

    small, large = elapsed(20_000), elapsed(200_000)
    assert large < 2.0, f"scrub took {large:.2f}s on 200k characters"
    # 10x the input must not cost anything like 100x the time.
    assert large < max(small, 0.005) * 30
