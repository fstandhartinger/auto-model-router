"""The held-out task set: six categories, deterministic, fully offline.

Nothing here downloads a dataset. Every task, its grader and its expected
answer live in this file, so the set can be rebuilt byte for byte from the seed
and checked against the pre-registered digest.

The tasks are deliberately small. The point of the set is to separate the
*categories* from each other - a route that is strong at coding and weak at
web/UI design has to show that difference - not to be a frontier benchmark.

Each task carries ``difficulty`` (easy/medium/hard) so per-category results can
be read at a difficulty, and ``repeat_of`` on cache-eligible repeats so the
runner knows which earlier task's prefix should still be warm.
"""

from __future__ import annotations

import random

SYSTEM_CODE = ("You are a careful engineer. Reply with one fenced Python code block and "
               "nothing else. Do not include tests or example calls.")
SYSTEM_MATH = "Solve the problem. End your reply with a line: Final answer: <value>"
SYSTEM_FACT = "Answer in at most three sentences. Be precise."
SYSTEM_DESIGN = ("You are a front-end engineer. Reply with one fenced html code block "
                 "containing a complete, self-contained page. No external CSS or JS files.")
SYSTEM_SUM = "Summarise faithfully. Do not add facts or numbers that are not in the source."


# ---------------------------------------------------------------------------
# coding
# ---------------------------------------------------------------------------
def _coding() -> list[dict]:
    return [
        {
            "id": "coding-easy-runlength",
            "difficulty": "easy",
            "prompt": ("Write a function `encode(s: str) -> str` that run-length encodes a string: "
                       "each maximal run of one character becomes the character followed by the run "
                       "length, but a run of length 1 stays a bare character. "
                       "encode('aaabbc') == 'a3b2c'. Empty input returns ''."),
            "hidden_tests": (
                "cases = [('aaabbc','a3b2c'), ('', ''), ('a','a'), ('aa','a2'), "
                "('abc','abc'), ('zzzzzzzzzzz','z11'), ('aabbaa','a2b2a2')]\n"
                "for src, want in cases:\n"
                "    got = encode(src)\n"
                "    assert got == want, (src, got, want)\n"
                "print('ALL_TESTS_PASSED')\n"),
        },
        {
            "id": "coding-medium-intervals",
            "difficulty": "medium",
            "prompt": ("Write `merge(intervals: list[tuple[int,int]]) -> list[tuple[int,int]]` that "
                       "merges overlapping or touching closed integer intervals and returns them "
                       "sorted by start. Touching means (1,2) and (2,5) merge into (1,5). "
                       "The input may be unsorted and may be empty."),
            "hidden_tests": (
                "assert merge([]) == []\n"
                "assert merge([(1,3),(2,6),(8,10),(15,18)]) == [(1,6),(8,10),(15,18)]\n"
                "assert merge([(5,6),(1,2)]) == [(1,2),(5,6)]\n"
                "assert merge([(1,2),(2,5)]) == [(1,5)]\n"
                "assert merge([(1,10),(2,3)]) == [(1,10)]\n"
                "assert merge([(1,1)]) == [(1,1)]\n"
                "print('ALL_TESTS_PASSED')\n"),
        },
        {
            "id": "coding-medium-parser",
            "difficulty": "medium",
            "prompt": ("Write `parse_config(text: str) -> dict` for an INI-like format: lines of "
                       "`key = value`, `#` starts a comment to end of line, blank lines are ignored, "
                       "keys and values are stripped, a value of `true`/`false` (any case) becomes a "
                       "bool, a value that is all digits becomes an int, everything else stays a "
                       "string. A line with no `=` raises ValueError."),
            "hidden_tests": (
                "got = parse_config('a = 1\\n# comment\\n\\nb= true \\nc =hello world # trailing\\n')\n"
                "assert got == {'a': 1, 'b': True, 'c': 'hello world'}, got\n"
                "assert parse_config('x=FALSE') == {'x': False}\n"
                "assert parse_config('') == {}\n"
                "try:\n"
                "    parse_config('nonsense')\n"
                "    raise AssertionError('should have raised')\n"
                "except ValueError:\n"
                "    pass\n"
                "print('ALL_TESTS_PASSED')\n"),
        },
        {
            "id": "coding-hard-scheduler",
            "difficulty": "hard",
            "prompt": ("Write `order(tasks: dict[str, list[str]]) -> list[str]`. `tasks` maps a task "
                       "name to the list of task names it depends on. Return a topological order in "
                       "which every dependency comes before its dependent; among tasks that become "
                       "available at the same time, pick the alphabetically smallest first. Raise "
                       "ValueError('cycle') if the graph has a cycle. A dependency that is not itself "
                       "a key is treated as an already-satisfied external input and does not appear "
                       "in the output."),
            "hidden_tests": (
                "assert order({'a': [], 'b': ['a'], 'c': ['a'], 'd': ['b','c']}) == "
                "['a','b','c','d']\n"
                "assert order({'z': [], 'a': []}) == ['a','z']\n"
                "assert order({'build': ['external'], 'test': ['build']}) == ['build','test']\n"
                "assert order({}) == []\n"
                "try:\n"
                "    order({'a': ['b'], 'b': ['a']})\n"
                "    raise AssertionError('should have raised')\n"
                "except ValueError:\n"
                "    pass\n"
                "print('ALL_TESTS_PASSED')\n"),
        },
    ]


# ---------------------------------------------------------------------------
# maths / reasoning
# ---------------------------------------------------------------------------
def _math() -> list[dict]:
    return [
        {"id": "math-easy-discount", "difficulty": "easy", "expected": "68",
         "prompt": ("A jacket costs 100 euro. It is discounted by 15 percent, then a further 20 "
                    "percent is taken off the discounted price. What is the final price in euro?")},
        {"id": "math-easy-rate", "difficulty": "easy", "expected": "48",
         "prompt": ("Three machines fill 1200 bottles in 50 minutes. How many minutes do five "
                    "machines of the same kind need to fill 1920 bottles?")},
        {"id": "math-medium-digits", "difficulty": "medium", "expected": "271",
         "prompt": ("How many positive integers below 1000 have at least one digit equal to 7?")},
        {"id": "math-medium-probability", "difficulty": "medium", "expected": "0.222",
         "atol": 0.0015,
         "prompt": ("An urn holds 5 red and 5 blue balls. Two are drawn without replacement. "
                    "What is the probability that both are red? Give a decimal rounded to three "
                    "places.")},
        {"id": "math-hard-modular", "difficulty": "hard", "expected": "36",
         "prompt": ("What is the remainder when 7^2026 is divided by 43? "
                    "Give the integer remainder.")},
        {"id": "math-hard-geometry", "difficulty": "hard", "expected": "7",
         "prompt": ("A right triangle has legs of integer length and a perimeter of 56. "
                    "Its hypotenuse is 25. What is the length of the shorter leg?")},
    ]


# ---------------------------------------------------------------------------
# factual research
# ---------------------------------------------------------------------------
def _research() -> list[dict]:
    return [
        {"id": "research-easy-http-status", "difficulty": "easy",
         "must_contain": ["409"], "must_not_contain": ["404 conflict"],
         "prompt": ("Which HTTP status code does RFC 9110 define for a request that conflicts with "
                    "the current state of the target resource? Give the numeric code.")},
        {"id": "research-easy-unicode", "difficulty": "easy",
         "must_contain": ["utf-8"], "must_not_contain": [],
         "prompt": ("Which Unicode encoding form is backwards compatible with ASCII for all code "
                    "points below 128 and is the default for JSON on the wire? Name it exactly.")},
        {"id": "research-medium-sql", "difficulty": "medium",
         "must_contain": ["phantom"], "must_not_contain": ["dirty read is possible"],
         "prompt": ("In the SQL standard isolation levels, which read phenomenon is still permitted "
                    "at REPEATABLE READ but forbidden at SERIALIZABLE? Name the phenomenon.")},
        {"id": "research-medium-tls", "difficulty": "medium",
         "must_contain": ["1.3"], "must_not_contain": [],
         "prompt": ("Which TLS version first made the full handshake complete in one round trip and "
                    "removed renegotiation and static RSA key exchange? Give the version number.")},
        {"id": "research-hard-posix", "difficulty": "hard",
         "must_contain": ["eintr"], "must_not_contain": [],
         "prompt": ("On POSIX, which errno does a slow system call return when it is interrupted by "
                    "a signal handler installed without SA_RESTART? Give the errno name.")},
    ]


# ---------------------------------------------------------------------------
# web / UI design
# ---------------------------------------------------------------------------
def _design() -> list[dict]:
    return [
        {
            "id": "design-easy-pricing-card",
            "difficulty": "easy",
            "prompt": ("Build a single self-contained HTML page with an inline <style> block showing "
                       "one pricing card. It must use a <main> landmark, an <h1>, a <button> with "
                       "visible text, and a CSS custom property for the accent colour."),
            "rules": [
                {"label": "has a <main> landmark", "pattern": r"<main[\s>]"},
                {"label": "has an <h1>", "pattern": r"<h1[\s>]"},
                {"label": "has a <button>", "pattern": r"<button[\s>]"},
                {"label": "declares a CSS custom property", "pattern": r"--[a-z-]+\s*:"},
                {"label": "has an inline <style> block", "pattern": r"<style[\s>]"},
            ],
        },
        {
            "id": "design-medium-responsive-table",
            "difficulty": "medium",
            "prompt": ("Build a single self-contained HTML page comparing three plans in a <table> "
                       "with a <caption> and a <thead>. Below 600px the layout must switch to a "
                       "stacked card view using a CSS media query. Include a skip link to the main "
                       "content and a visible focus style."),
            "rules": [
                {"label": "has a <table> with a <caption>", "pattern": r"<table[\s>].*?<caption[\s>]"},
                {"label": "has a <thead>", "pattern": r"<thead[\s>]"},
                {"label": "has a max-width media query", "pattern": r"@media[^{]*max-width"},
                {"label": "has a skip link", "pattern": r'href="#[a-z-]*(?:main|content)'},
                {"label": "styles :focus or :focus-visible", "pattern": r":focus(?:-visible)?\s*[,{]"},
            ],
        },
        {
            "id": "design-medium-form",
            "difficulty": "medium",
            "prompt": ("Build a single self-contained HTML page with a newsletter sign-up form. "
                       "Every input must have an associated <label for=...>, the email field must be "
                       "type=email and required, errors must be announced with aria-live, and the "
                       "form must be usable at 320px width."),
            "rules": [
                {"label": "labels are associated with for=", "pattern": r"<label[^>]+for="},
                {"label": "email input is typed and required",
                 "pattern": r'<input[^>]+type="?email"?[^>]*required|<input[^>]+required[^>]*type="?email'},
                {"label": "has an aria-live region", "pattern": r"aria-live"},
                {"label": "has a viewport meta tag", "pattern": r'name="viewport"'},
                {"label": "has a media query or fluid width", "pattern": r"@media|max-width\s*:\s*\d|width\s*:\s*100%"},
            ],
        },
        {
            "id": "design-hard-dashboard",
            "difficulty": "hard",
            "prompt": ("Build a single self-contained HTML page: an analytics dashboard with a "
                       "sidebar navigation and a responsive grid of four stat tiles. It must use CSS "
                       "grid, collapse the sidebar below 768px with a media query, respect "
                       "prefers-reduced-motion, provide a dark colour scheme via "
                       "prefers-color-scheme, and mark the navigation with a <nav> landmark and "
                       "aria-current on the active item."),
            "rules": [
                {"label": "uses CSS grid", "pattern": r"display\s*:\s*grid"},
                {"label": "has a <nav> landmark", "pattern": r"<nav[\s>]"},
                {"label": "marks the active item with aria-current", "pattern": r"aria-current"},
                {"label": "collapses below 768px", "pattern": r"@media[^{]*768"},
                {"label": "respects prefers-reduced-motion", "pattern": r"prefers-reduced-motion"},
                {"label": "supports a dark colour scheme", "pattern": r"prefers-color-scheme"},
            ],
        },
    ]


# ---------------------------------------------------------------------------
# summarisation
# ---------------------------------------------------------------------------
_SOURCE_CACHE = (
    "A prompt cache stores the tokens a provider has already processed for a prefix of a request. "
    "When the next request begins with the same 40000 tokens, those tokens are billed at the cache "
    "read price instead of the input price. On the providers measured here the read price was "
    "about one tenth of the input price, and a cache entry stayed valid for 300 seconds, refreshed "
    "on every hit. Below a minimum prefix of 1024 tokens nothing is cached at all. Switching a "
    "conversation to a different model abandons the cached prefix, because the cache belongs to "
    "the route and not to the conversation. In a week of traffic covering 57696 calls, 96 percent "
    "of all input tokens were served as cache reads.")

_SOURCE_QUOTA = (
    "A flat-rate subscription has no marginal price per call, which makes it look free to a cost "
    "model and causes a naive router to send everything through it until the plan is exhausted. "
    "Quota pacing gives the plan a shadow price instead. While projected weekly use stays below a "
    "reserve line of 65 percent the shadow price is zero. As projected use approaches the line the "
    "multiplier rises toward the list price of a comparable metered model. Above the line, or when "
    "the rolling 5 hour session window is nearly full, the plan is closed and the router must pick "
    "a metered route. The pacing decision is recomputed at most once every 60 seconds.")


def _summarisation() -> list[dict]:
    return [
        {"id": "summary-easy-cache", "difficulty": "easy", "source": _SOURCE_CACHE,
         "min_words": 25, "max_words": 70,
         "must_retain": ["cache", "prefix"],
         "prompt": ("Summarise the following text in 25 to 70 words for an engineer who has not "
                    "read it. Do not invent numbers.\n\n" + _SOURCE_CACHE)},
        {"id": "summary-medium-cache-constrained", "difficulty": "medium", "source": _SOURCE_CACHE,
         "min_words": 20, "max_words": 45,
         "must_retain": ["1024", "300"],
         "prompt": ("In 20 to 45 words, state the two hard thresholds in this text and what each "
                    "one does. Use only numbers that appear in the text.\n\n" + _SOURCE_CACHE)},
        {"id": "summary-medium-quota", "difficulty": "medium", "source": _SOURCE_QUOTA,
         "min_words": 25, "max_words": 60,
         "must_retain": ["shadow price", "reserve"],
         "prompt": ("Summarise the following text in 25 to 60 words. Keep the mechanism, drop the "
                    "motivation. Do not invent numbers.\n\n" + _SOURCE_QUOTA)},
        {"id": "summary-hard-quota-exact", "difficulty": "hard", "source": _SOURCE_QUOTA,
         "min_words": 15, "max_words": 40,
         "must_retain": ["65", "60"],
         "prompt": ("In 15 to 40 words, state exactly when the plan is closed and how often the "
                    "decision is recomputed. Use only numbers that appear in the text.\n\n"
                    + _SOURCE_QUOTA)},
    ]


# ---------------------------------------------------------------------------
# cache-eligible repeats
# ---------------------------------------------------------------------------
_LEDGER_CONTEXT = (
    "SERVICE CATALOGUE (internal reference, revision 41)\n"
    "The billing service exposes three endpoints. POST /v1/invoices creates an invoice and returns "
    "409 when an invoice with the same idempotency key already exists. GET /v1/invoices/{id} "
    "returns the invoice, or 404 when the caller is not the owner, deliberately hiding existence. "
    "POST /v1/invoices/{id}/void voids an unpaid invoice and returns 422 when the invoice is "
    "already paid. Every endpoint requires the X-Tenant header. Rate limits are 100 requests per "
    "minute per tenant, and a request over the limit returns 429 with a retry-after header in "
    "seconds. The service stores amounts in minor units as integers and never as floats. "
    "Timestamps are RFC 3339 in UTC with a trailing Z. Pagination uses an opaque cursor in the "
    "`after` query parameter and returns at most 200 items per page.\n") * 6


def _cache_repeats() -> list[dict]:
    """Four questions over one long shared prefix: the second and later turns should hit the cache."""
    questions = [
        ("cache-repeat-1", "Which status code does creating a duplicate invoice return?", ["409"]),
        ("cache-repeat-2", "Which header must every endpoint receive?", ["x-tenant"]),
        ("cache-repeat-3", "What does voiding an already paid invoice return?", ["422"]),
        ("cache-repeat-4", "How are monetary amounts stored?", ["integer"]),
    ]
    out = []
    for index, (task_id, question, must) in enumerate(questions):
        out.append({
            "id": task_id,
            "difficulty": "easy",
            "prompt": _LEDGER_CONTEXT + "\nQuestion: " + question + " Answer in one short sentence.",
            "must_contain": must,
            "must_not_contain": [],
            "shared_prefix": "service-catalogue-41",
            "repeat_of": None if index == 0 else questions[0][0],
        })
    return out


# ---------------------------------------------------------------------------
def build_tasks(rng: random.Random | None = None) -> list[dict]:
    """The full held-out set. Deterministic: the seed only fixes the ordering."""
    groups = [
        ("design", "design", SYSTEM_DESIGN, _design()),
        ("coding", "coding", SYSTEM_CODE, _coding()),
        ("math", "math", SYSTEM_MATH, _math()),
        ("research", "research", SYSTEM_FACT, _research()),
        ("summarisation", "summarisation", SYSTEM_SUM, _summarisation()),
        ("cache_repeat", "cache_repeat", SYSTEM_FACT, _cache_repeats()),
    ]
    tasks: list[dict] = []
    for category, grader, system, items in groups:
        for item in items:
            tasks.append({"category": category, "grader": grader, "system": system, **item})
    if rng is not None:
        # Cache repeats must keep their order so the shared prefix is warmed first.
        shuffled = [t for t in tasks if t["category"] != "cache_repeat"]
        rng.shuffle(shuffled)
        tasks = shuffled + [t for t in tasks if t["category"] == "cache_repeat"]
    return tasks
