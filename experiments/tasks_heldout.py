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
    """The full held-out set. Deterministic: the seed only fixes the ordering.

    Each category is the original list registered on 2026-09-18 followed by the
    2026-09-19 extension, in that order. The originals are never re-ordered
    relative to each other and never edited, so a task id that has already been
    run still names exactly the same prompt.
    """
    repeats = _cache_repeats()
    groups = [
        ("design", "design", SYSTEM_DESIGN, _design() + _design_extra()),
        ("coding", "coding", SYSTEM_CODE, _coding() + _coding_extra()),
        ("math", "math", SYSTEM_MATH, _math() + _math_extra()),
        ("research", "research", SYSTEM_FACT, _research() + _research_extra()),
        ("summarisation", "summarisation", SYSTEM_SUM,
         _summarisation() + _summarisation_extra()),
        ("cache_repeat", "cache_repeat", SYSTEM_FACT,
         repeats + _cache_repeats_extra(repeats[0]["id"])),
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


# ===========================================================================
# 2026-09-19 extension: enough tasks per category to reach the pre-registered
# floor of ten *valid pairs*.
#
# Appended, never edited. The four functions above return exactly the bodies
# that were registered as ``84731010531b266f…`` on 2026-09-18 and run against
# both arms; rewriting any of them would silently invalidate that finished
# ledger, so the new tasks live in their own ``_extra`` functions and
# ``build_tasks`` concatenates them.
#
# The counts are deliberately above ten, not at it. A pair needs *both* rows
# graded, and the 18 September run lost four design rows and one summarisation
# row to the harness output budget - at exactly ten tasks a single truncation
# drops the category back under the floor it was extended to clear. Design and
# summarisation therefore get twelve, the rest eleven.
#
# The new design tasks are compact components rather than full pages, for the
# same reason: every truncation observed so far was a full-page task, where a
# free route spends its whole output budget before the closing tag.
# ===========================================================================


def _coding_extra() -> list[dict]:
    return [
        {
            "id": "coding-easy-balanced",
            "difficulty": "easy",
            "prompt": ("Write `is_balanced(s: str) -> bool` that returns True when every round, "
                       "square and curly bracket in `s` is closed in the right order. Characters "
                       "that are not one of those six brackets are ignored. The empty string is "
                       "balanced."),
            "hidden_tests": (
                "assert is_balanced('') is True\n"
                "assert is_balanced('([]{})') is True\n"
                "assert is_balanced('a(b)c') is True\n"
                "assert is_balanced('(') is False\n"
                "assert is_balanced(')(') is False\n"
                "assert is_balanced('([)]') is False\n"
                "assert is_balanced('{[()()]}') is True\n"
                "print('ALL_TESTS_PASSED')\n"),
        },
        {
            "id": "coding-easy-rotate",
            "difficulty": "easy",
            "prompt": ("Write `rotate(items: list, k: int) -> list` that returns a new list rotated "
                       "`k` places to the right. A negative `k` rotates left, a `k` larger than the "
                       "list wraps around, and the input list must not be modified. An empty list "
                       "returns an empty list for any `k`."),
            "hidden_tests": (
                "assert rotate([1,2,3,4,5], 2) == [4,5,1,2,3]\n"
                "assert rotate([1,2,3,4,5], -1) == [2,3,4,5,1]\n"
                "assert rotate([1,2,3], 7) == [3,1,2]\n"
                "assert rotate([], 3) == []\n"
                "assert rotate([1], 0) == [1]\n"
                "src = [1,2,3]\n"
                "rotate(src, 1)\n"
                "assert src == [1,2,3], 'the input was modified'\n"
                "print('ALL_TESTS_PASSED')\n"),
        },
        {
            "id": "coding-medium-roman",
            "difficulty": "medium",
            "prompt": ("Write `to_roman(n: int) -> str` converting an integer from 1 to 3999 "
                       "inclusive into an uppercase Roman numeral using the usual subtractive "
                       "forms (4 is IV, 9 is IX, 40 is XL, 900 is CM). Anything outside that range "
                       "raises ValueError."),
            "hidden_tests": (
                "assert to_roman(1) == 'I'\n"
                "assert to_roman(4) == 'IV'\n"
                "assert to_roman(40) == 'XL'\n"
                "assert to_roman(1994) == 'MCMXCIV'\n"
                "assert to_roman(3999) == 'MMMCMXCIX'\n"
                "for bad in (0, -1, 4000):\n"
                "    try:\n"
                "        to_roman(bad)\n"
                "        raise AssertionError('should have raised for %r' % (bad,))\n"
                "    except ValueError:\n"
                "        pass\n"
                "print('ALL_TESTS_PASSED')\n"),
        },
        {
            "id": "coding-medium-flatten",
            "difficulty": "medium",
            "prompt": ("Write `flatten(items: list) -> list` that flattens arbitrarily nested lists "
                       "into a single list, keeping the original left-to-right order. Only `list` "
                       "is descended into: tuples, strings and every other value are kept as they "
                       "are, so a string is never split into characters."),
            "hidden_tests": (
                "assert flatten([]) == []\n"
                "assert flatten([1, [2, [3, [4]]], 5]) == [1,2,3,4,5]\n"
                "assert flatten(['ab', ['cd']]) == ['ab','cd']\n"
                "assert flatten([[], [[]], 1]) == [1]\n"
                "assert flatten([(1,2), [3]]) == [(1,2), 3]\n"
                "assert flatten([None, [False, [0]]]) == [None, False, 0]\n"
                "print('ALL_TESTS_PASSED')\n"),
        },
        {
            "id": "coding-medium-wordfreq",
            "difficulty": "medium",
            "prompt": ("Write `top_words(text: str, n: int) -> list[tuple[str, int]]`. A word is a "
                       "maximal run of ASCII letters and apostrophes, compared case-insensitively "
                       "and returned lower case. Return the `n` most frequent words as "
                       "(word, count) pairs, most frequent first, ties broken alphabetically. "
                       "Return fewer than `n` pairs when there are fewer distinct words."),
            "hidden_tests": (
                "assert top_words('', 3) == []\n"
                "assert top_words('The cat the CAT a dog', 2) == [('cat',2), ('the',2)]\n"
                "assert top_words(\"it's it's fine\", 1) == [(\"it's\",2)]\n"
                "assert top_words('a b c', 10) == [('a',1), ('b',1), ('c',1)]\n"
                "assert top_words('one, two; two!', 2) == [('two',2), ('one',1)]\n"
                "print('ALL_TESTS_PASSED')\n"),
        },
        {
            "id": "coding-hard-lru",
            "difficulty": "hard",
            "prompt": ("Write a class `LRUCache` constructed as `LRUCache(capacity)` with methods "
                       "`get(key)` and `put(key, value)`. `get` returns the stored value or -1 when "
                       "the key is absent. Both `get` and `put` count as a use. When a `put` would "
                       "exceed the capacity, the least recently used key is evicted first. "
                       "Re-putting an existing key updates its value and counts as a use without "
                       "evicting anything."),
            "hidden_tests": (
                "c = LRUCache(2)\n"
                "c.put(1,1); c.put(2,2)\n"
                "assert c.get(1) == 1\n"
                "c.put(3,3)\n"
                "assert c.get(2) == -1\n"
                "assert c.get(3) == 3\n"
                "c.put(4,4)\n"
                "assert c.get(1) == -1\n"
                "assert c.get(3) == 3 and c.get(4) == 4\n"
                "d = LRUCache(1)\n"
                "d.put('a',1); d.put('b',2)\n"
                "assert d.get('a') == -1 and d.get('b') == 2\n"
                "e = LRUCache(2)\n"
                "e.put(1,1); e.put(2,2); e.put(1,10); e.put(3,3)\n"
                "assert e.get(2) == -1\n"
                "assert e.get(1) == 10 and e.get(3) == 3\n"
                "print('ALL_TESTS_PASSED')\n"),
        },
        {
            "id": "coding-hard-lcs",
            "difficulty": "hard",
            "prompt": ("Write `lcs_length(a: str, b: str) -> int` returning the length of the "
                       "longest common subsequence of the two strings. A subsequence keeps order "
                       "but need not be contiguous. It must handle strings of a few hundred "
                       "characters without recursing once per character."),
            "hidden_tests": (
                "assert lcs_length('', 'abc') == 0\n"
                "assert lcs_length('abc', '') == 0\n"
                "assert lcs_length('abc', 'abc') == 3\n"
                "assert lcs_length('abc', 'def') == 0\n"
                "assert lcs_length('abcde', 'ace') == 3\n"
                "assert lcs_length('AGGTAB', 'GXTXAYB') == 4\n"
                "assert lcs_length('a'*400, 'a'*300) == 300\n"
                "print('ALL_TESTS_PASSED')\n"),
        },
    ]


def _math_extra() -> list[dict]:
    return [
        {"id": "math-easy-average", "difficulty": "easy", "expected": "70",
         "prompt": ("A class of 20 students has a mean score of 72. Two students, who scored 95 "
                    "and 85, leave the class. What is the mean score of the remaining students?")},
        {"id": "math-medium-multiples", "difficulty": "medium", "expected": "2318",
         "prompt": ("What is the sum of all positive integers below 100 that are a multiple of 3 "
                    "or a multiple of 5?")},
        {"id": "math-medium-arrangements", "difficulty": "medium", "expected": "1260",
         "prompt": ("How many distinct arrangements are there of all seven letters of the word "
                    "BALLOON?")},
        {"id": "math-hard-pipes", "difficulty": "hard", "expected": "9",
         "prompt": ("Pipe A alone fills a tank in 12 hours and pipe B alone fills it in 18 hours. "
                    "An open drain empties a full tank in 36 hours. With both pipes and the drain "
                    "open on an empty tank, how many hours does filling take?")},
        {"id": "math-hard-factorial-zeros", "difficulty": "hard", "expected": "25",
         "prompt": ("What is the smallest positive integer n for which n factorial is divisible "
                    "by 10 to the power 5?")},
    ]


def _research_extra() -> list[dict]:
    return [
        {"id": "research-easy-redirect", "difficulty": "easy",
         "must_contain": ["308"], "must_not_contain": [],
         "prompt": ("Which HTTP status code marks a permanent redirect that, unlike the older "
                    "permanent redirect, forbids the client from changing the request method? "
                    "Give the numeric code.")},
        {"id": "research-easy-git-revert", "difficulty": "easy",
         "must_contain": ["revert"], "must_not_contain": [],
         "prompt": ("Which Git subcommand records a new commit undoing an earlier one, without "
                    "rewriting any existing history? Name the subcommand.")},
        {"id": "research-medium-dns-cname", "difficulty": "medium",
         "must_contain": ["cname"], "must_not_contain": [],
         "prompt": ("Which DNS record type aliases one name to another name, and for that reason "
                    "may not coexist with any other record at the same owner name? Give the record "
                    "type.")},
        {"id": "research-medium-posix-execute", "difficulty": "medium",
         "must_contain": ["execute"], "must_not_contain": [],
         "prompt": ("On a POSIX filesystem, which permission bit on a directory lets a process "
                    "reach a file inside it by name while still being unable to list the "
                    "directory's contents? Name the bit.")},
        {"id": "research-medium-float-nan", "difficulty": "medium",
         "must_contain": ["nan"], "must_not_contain": [],
         "prompt": ("Which IEEE 754 floating-point value compares unequal to itself under the "
                    "standard's ordinary comparison rules? Name it.")},
        {"id": "research-hard-elf-data", "difficulty": "hard",
         "must_contain": [".data"], "must_not_contain": [],
         "prompt": ("In an ELF binary on Linux, which section holds initialised global variables "
                    "that the program may modify at run time? Give the section name including its "
                    "leading dot.")},
    ]


def _design_extra() -> list[dict]:
    return [
        {
            "id": "design-easy-alert-banner",
            "difficulty": "easy",
            "prompt": ("Build a single self-contained HTML page with one dismissible alert banner. "
                       "It must sit inside a <main> landmark, announce itself to assistive "
                       "technology, carry a dismiss <button> with an accessible name, and take its "
                       "colour from a CSS custom property in an inline <style> block."),
            "rules": [
                {"label": "has an inline <style> block", "pattern": r"<style[\s>]"},
                {"label": "has a <main> landmark", "pattern": r"<main[\s>]"},
                {"label": "announces the alert", "pattern": r"role\s*=\s*[\"']?alert|aria-live"},
                {"label": "has a dismiss <button>", "pattern": r"<button[\s>]"},
                {"label": "the dismiss control has an accessible name",
                 "pattern": r"aria-label|<button[^>]*>\s*[a-z0-9]"},
                {"label": "declares a CSS custom property", "pattern": r"--[a-z-]+\s*:"},
            ],
        },
        {
            "id": "design-easy-site-header",
            "difficulty": "easy",
            "prompt": ("Build a single self-contained HTML page with a site header: a word mark on "
                       "the left and a primary navigation of three links on the right. Use a "
                       "<header> and a labelled <nav> containing a list, and style it with an "
                       "inline <style> block."),
            "rules": [
                {"label": "has a <header>", "pattern": r"<header[\s>]"},
                {"label": "has a <nav> landmark", "pattern": r"<nav[\s>]"},
                {"label": "the navigation is a list", "pattern": r"<ul[\s>]|<ol[\s>]"},
                {"label": "the navigation is labelled", "pattern": r"aria-label"},
                {"label": "has an inline <style> block", "pattern": r"<style[\s>]"},
            ],
        },
        {
            "id": "design-easy-stat-tile",
            "difficulty": "easy",
            "prompt": ("Build a single self-contained HTML page showing one statistic tile: a large "
                       "number, a short caption below it, and a trend indicator whose meaning is "
                       "available to a screen reader rather than by colour alone. Use a <section>, "
                       "an <h2> and an inline <style> block."),
            "rules": [
                {"label": "has a <section>", "pattern": r"<section[\s>]"},
                {"label": "has an <h2>", "pattern": r"<h2[\s>]"},
                {"label": "has an inline <style> block", "pattern": r"<style[\s>]"},
                {"label": "the trend is not conveyed by colour alone",
                 "pattern": r"aria-label|<span[^>]*class=[\"']?[^\"'>]*sr-only|visually-hidden"},
                {"label": "declares a CSS custom property", "pattern": r"--[a-z-]+\s*:"},
            ],
        },
        {
            "id": "design-medium-card-grid",
            "difficulty": "medium",
            "prompt": ("Build a single self-contained HTML page with a responsive grid of six "
                       "article cards. Use CSS grid with a minmax() track so the columns reflow "
                       "without a fixed breakpoint, make each card an <article> with an <h2>, and "
                       "draw the thumbnail with a CSS gradient rather than an image file. Add one "
                       "media query for the narrowest layout."),
            "rules": [
                {"label": "uses CSS grid", "pattern": r"display\s*:\s*grid"},
                {"label": "uses a minmax() track", "pattern": r"minmax\s*\("},
                {"label": "cards are <article> elements", "pattern": r"<article[\s>]"},
                {"label": "each card has an <h2>", "pattern": r"<h2[\s>]"},
                {"label": "has a media query", "pattern": r"@media"},
                {"label": "the thumbnail is a CSS gradient", "pattern": r"gradient\s*\("},
            ],
        },
        {
            "id": "design-medium-modal",
            "difficulty": "medium",
            "prompt": ("Build a single self-contained HTML page with a confirmation modal dialog "
                       "over a short page body. The dialog must be marked as a modal dialog, "
                       "labelled by its own heading, contain a confirm and a cancel <button>, and "
                       "define a visible focus style."),
            "rules": [
                {"label": "is a dialog", "pattern": r"<dialog[\s>]|role\s*=\s*[\"']?dialog"},
                {"label": "is marked modal", "pattern": r"aria-modal|<dialog[^>]*\bopen\b"},
                {"label": "is labelled by its heading", "pattern": r"aria-labelledby|aria-label"},
                {"label": "has buttons", "pattern": r"<button[\s>]"},
                {"label": "styles :focus or :focus-visible", "pattern": r":focus(?:-visible)?\s*[,{]"},
            ],
        },
        {
            "id": "design-medium-stepper",
            "difficulty": "medium",
            "prompt": ("Build a single self-contained HTML page with a three-step checkout "
                       "progress indicator above a form for step two. The steps must be an ordered "
                       "list with the current step marked for assistive technology, every field "
                       "must have an associated <label for=...>, and the layout must stack on a "
                       "narrow screen."),
            "rules": [
                {"label": "the steps are an ordered list", "pattern": r"<ol[\s>]"},
                {"label": "marks the current step", "pattern": r"aria-current"},
                {"label": "has a <form>", "pattern": r"<form[\s>]"},
                {"label": "labels are associated with for=", "pattern": r"<label[^>]+for="},
                {"label": "has a media query", "pattern": r"@media"},
            ],
        },
        {
            "id": "design-medium-breadcrumb",
            "difficulty": "medium",
            "prompt": ("Build a single self-contained HTML page with a breadcrumb trail of four "
                       "levels above a short article. The trail must be a labelled <nav> holding "
                       "an ordered list, the last item must be marked as the current page, the "
                       "separators must be drawn in CSS rather than typed as text, and the trail "
                       "must wrap on a narrow screen."),
            "rules": [
                {"label": "has a <nav> landmark", "pattern": r"<nav[\s>]"},
                {"label": "the navigation is labelled", "pattern": r"aria-label"},
                {"label": "the trail is an ordered list", "pattern": r"<ol[\s>]"},
                {"label": "marks the current page", "pattern": r"aria-current"},
                {"label": "separators come from CSS", "pattern": r"::(?:after|before)"},
            ],
        },
        {
            "id": "design-easy-toggle",
            "difficulty": "easy",
            "prompt": ("Build a single self-contained HTML page with one labelled on/off switch "
                       "for a 'Weekly digest' setting. Build it from a real checkbox so the "
                       "keyboard reaches it, associate a <label> with it, give it a visible focus "
                       "style, and style the track and knob in an inline <style> block."),
            "rules": [
                {"label": "is built from a real checkbox",
                 "pattern": r"<input[^>]+type\s*=\s*[\"']?checkbox"},
                {"label": "the label is associated with for=", "pattern": r"<label[^>]+for="},
                {"label": "styles :focus or :focus-visible", "pattern": r":focus(?:-visible)?\s*[,{]"},
                {"label": "has an inline <style> block", "pattern": r"<style[\s>]"},
                {"label": "styles the checked state", "pattern": r":checked"},
            ],
        },
        {
            "id": "design-medium-empty-state",
            "difficulty": "medium",
            "prompt": ("Build a single self-contained HTML page showing an empty-state panel for a "
                       "project list that has no projects yet: a heading, one sentence of "
                       "explanation, a primary action button, and an illustration drawn with CSS "
                       "or inline SVG rather than an image file. Centre it and keep it readable at "
                       "320px."),
            "rules": [
                {"label": "has a <section>", "pattern": r"<section[\s>]"},
                {"label": "has a heading", "pattern": r"<h1[\s>]|<h2[\s>]"},
                {"label": "has a primary action", "pattern": r"<button[\s>]"},
                {"label": "the illustration is drawn, not linked",
                 "pattern": r"<svg[\s>]|gradient\s*\(|border-radius"},
                {"label": "stays readable on a narrow screen",
                 "pattern": r"@media|max-width\s*:\s*\d|width\s*:\s*100%"},
            ],
        },
        {
            "id": "design-hard-settings-tabs",
            "difficulty": "hard",
            "prompt": ("Build a single self-contained HTML page: a settings panel with three tabs "
                       "(Profile, Notifications, Billing) and the first panel visible. Implement "
                       "the ARIA tabs pattern with a tablist, tabs, panels and a selected state, "
                       "respect prefers-color-scheme for a dark theme, respect "
                       "prefers-reduced-motion, and collapse the tabs to a stacked list below "
                       "600px."),
            "rules": [
                {"label": "has a tablist", "pattern": r"role\s*=\s*[\"']?tablist"},
                {"label": "has tabs", "pattern": r"role\s*=\s*[\"']?tab[\"'\s>]"},
                {"label": "has tab panels", "pattern": r"role\s*=\s*[\"']?tabpanel"},
                {"label": "marks the selected tab", "pattern": r"aria-selected"},
                {"label": "supports a dark colour scheme", "pattern": r"prefers-color-scheme"},
                {"label": "respects prefers-reduced-motion", "pattern": r"prefers-reduced-motion"},
                {"label": "collapses below 600px", "pattern": r"@media[^{]*600"},
            ],
        },
    ]


_SOURCE_ROUTING = (
    "A router picks a route by combining three things for every candidate: a capability score per "
    "task category taken from published benchmark evidence, a price per million tokens, and a "
    "record of how that route behaved on recent calls. A capability score older than 45 days is "
    "marked stale, and a stale route is chosen only when no fresher route can do the work. For "
    "each candidate the router forecasts a success probability and a cost, picks the candidate "
    "with the best expected value, and writes both the forecast and the observed result to a "
    "ledger so the two can be compared afterwards. A route that fails 3 calls in a row is taken "
    "out of the pool for 600 seconds.")

_SOURCE_SANDBOX = (
    "Model-produced code is executed only inside a sandbox built with Bubblewrap. The sandbox has "
    "no network namespace, so a socket call fails before it reaches the wire, and it can read "
    "neither the home directory nor any environment variable holding a credential. The working "
    "directory is a private temporary filesystem that is discarded when the process exits. Every "
    "run is bounded by a wall clock limit of 20 seconds and a memory limit of 512 megabytes, and a "
    "run that exceeds either is killed and reported as a timeout rather than as a wrong answer. "
    "When the sandbox is unavailable the harness refuses to execute the code at all and records "
    "the answer as ungraded, because running unverified code on the host is the failure this "
    "design exists to prevent.")

_SOURCE_LEDGER = (
    "Every call the harness makes is appended to a ledger as one JSON line holding the route, the "
    "tag of the task, the token counts split into 3 parts of uncached input, cache reads and "
    "output, the latency, and 2 cost figures. The first figure is what the gateway said it "
    "charged and the second is what the same call would cost at the route's published list price. "
    "The higher of the 2 is the figure counted against the spend cap, because a gateway that "
    "reports 0 for a call made with the caller's own key is not evidence that nobody was charged. "
    "The ledger is the only source the report reads, and no number in the report is typed in by "
    "hand.")


def _summarisation_extra() -> list[dict]:
    return [
        {"id": "summary-medium-cache-switch", "difficulty": "medium", "source": _SOURCE_CACHE,
         "min_words": 20, "max_words": 45,
         "must_retain": ["route"],
         "prompt": ("In 20 to 45 words, state what happens to a cached prefix when a conversation "
                    "is moved to a different model, and why. Use only numbers that appear in the "
                    "text.\n\n" + _SOURCE_CACHE)},
        {"id": "summary-medium-quota-shadow", "difficulty": "medium", "source": _SOURCE_QUOTA,
         "min_words": 25, "max_words": 55,
         "must_retain": ["shadow price", "zero"],
         "prompt": ("In 25 to 55 words, describe the shadow price and how it changes as projected "
                    "use rises toward the reserve line. Use only numbers that appear in the "
                    "text.\n\n" + _SOURCE_QUOTA)},
        {"id": "summary-easy-routing", "difficulty": "easy", "source": _SOURCE_ROUTING,
         "min_words": 25, "max_words": 70,
         "must_retain": ["capability", "ledger"],
         "prompt": ("Summarise the following text in 25 to 70 words for an engineer who has not "
                    "read it. Do not invent numbers.\n\n" + _SOURCE_ROUTING)},
        {"id": "summary-medium-routing-exact", "difficulty": "medium", "source": _SOURCE_ROUTING,
         "min_words": 15, "max_words": 40,
         "must_retain": ["45", "600"],
         "prompt": ("In 15 to 40 words, state when a capability score becomes stale and what "
                    "happens to a route that keeps failing. Use only numbers that appear in the "
                    "text.\n\n" + _SOURCE_ROUTING)},
        {"id": "summary-easy-sandbox", "difficulty": "easy", "source": _SOURCE_SANDBOX,
         "min_words": 25, "max_words": 70,
         "must_retain": ["network", "home"],
         "prompt": ("Summarise the following text in 25 to 70 words for an engineer who has not "
                    "read it. Do not invent numbers.\n\n" + _SOURCE_SANDBOX)},
        {"id": "summary-hard-sandbox-exact", "difficulty": "hard", "source": _SOURCE_SANDBOX,
         "min_words": 15, "max_words": 40,
         "must_retain": ["20", "512"],
         "prompt": ("In 15 to 40 words, state the two resource limits on a sandboxed run and what "
                    "happens when a run exceeds one of them. Use only numbers that appear in the "
                    "text.\n\n" + _SOURCE_SANDBOX)},
        {"id": "summary-medium-ledger", "difficulty": "medium", "source": _SOURCE_LEDGER,
         "min_words": 25, "max_words": 60,
         "must_retain": ["ledger", "list price"],
         "prompt": ("Summarise the following text in 25 to 60 words. Keep the mechanism, drop the "
                    "justification. Do not invent numbers.\n\n" + _SOURCE_LEDGER)},
        {"id": "summary-hard-ledger-exact", "difficulty": "hard", "source": _SOURCE_LEDGER,
         "min_words": 15, "max_words": 40,
         "must_retain": ["higher"],
         "prompt": ("In 15 to 40 words, state which of the two cost figures is counted against the "
                    "spend cap and why. Use only numbers that appear in the text.\n\n"
                    + _SOURCE_LEDGER)},
    ]


def _cache_repeats_extra(first_id: str) -> list[dict]:
    """Seven more questions over the *same* warm prefix as ``_cache_repeats``.

    They deliberately do not open a second prefix family. One family is what
    the cache measurement is about, and a second one would halve the number of
    hits per family while doubling the cold writes.
    """
    questions = [
        ("cache-repeat-5", "What does a GET on an invoice return when the caller is not the owner?",
         ["404"]),
        ("cache-repeat-6", "How many requests per minute per tenant are allowed?", ["100"]),
        ("cache-repeat-7", "Which status code is returned to a caller over the rate limit?", ["429"]),
        ("cache-repeat-8", "Which specification and time zone do the timestamps follow?",
         ["3339", "utc"]),
        ("cache-repeat-9", "Is the pagination cursor opaque or structured, and which query "
                           "parameter carries it?", ["opaque", "after"]),
        ("cache-repeat-10", "At most how many items does one page return?", ["200"]),
        ("cache-repeat-11", "In which units are monetary amounts stored?", ["minor"]),
    ]
    return [{
        "id": task_id,
        "difficulty": "easy",
        "prompt": _LEDGER_CONTEXT + "\nQuestion: " + question + " Answer in one short sentence.",
        "must_contain": must,
        "must_not_contain": [],
        "shared_prefix": "service-catalogue-41",
        "repeat_of": first_id,
    } for task_id, question, must in questions]
