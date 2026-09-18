"""The held-out **supplement** task set: 60 new tasks over the same six categories.

Why a second set exists
-----------------------
The original 27-task set is immutable evidence. It cannot reach ten valid
*paired* tasks per category, mostly because four of its twelve design rows are
harness truncations rather than grades. Rather than edit or re-run it, this
file adds a separately pre-registered supplement whose rows are *combined* with
it for reporting and never merged into it. No id here collides with an id
there.

Same rules as the original
--------------------------
Nothing is downloaded. Every task, its grader and its expected answer live in
this file, so the set rebuilds byte for byte from the seed and is checked
against the digest recorded before any model is called.

One declared difference, stated before the run
----------------------------------------------
The design tasks here are mostly **compact single components** rather than full
pages. That is a deliberate, pre-registered choice, and it narrows what a design
result means: it measures whether a route can produce a small, correct,
self-contained component, not whether it can hold a whole page together. The
reason is on the record - on the original set the free routes spent a
12,000-token output budget without finishing the larger pages, so a set made of
large pages cannot produce valid pairs at all on those routes. Three full-page
"hard" tasks are kept so that the failure mode stays visible instead of being
designed away.

``REFERENCE_ANSWERS`` gives one answer per task that the task's own grader must
accept. It is a test fixture, never sent to a model: a task whose own grader
cannot be satisfied is an un-passable task, and the suite refuses to register
one.
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

COMPACT = (" Keep the page small: one component, no more than about sixty lines of markup "
           "and style together.")


# ---------------------------------------------------------------------------
# coding
# ---------------------------------------------------------------------------
def _coding() -> list[dict]:
    return [
        {
            "id": "s-coding-easy-chunk", "difficulty": "easy",
            "prompt": ("Write `chunk(items: list, size: int) -> list[list]` that splits a list into "
                       "consecutive pieces of at most `size` elements, preserving order. The last "
                       "piece may be shorter. An empty list returns []. A size below 1 raises "
                       "ValueError."),
            "hidden_tests": (
                "assert chunk([1,2,3,4,5], 2) == [[1,2],[3,4],[5]]\n"
                "assert chunk([], 3) == []\n"
                "assert chunk([1], 5) == [[1]]\n"
                "assert chunk([1,2,3,4], 4) == [[1,2,3,4]]\n"
                "assert chunk(list(range(7)), 3) == [[0,1,2],[3,4,5],[6]]\n"
                "try:\n"
                "    chunk([1], 0)\n"
                "    raise AssertionError('should have raised')\n"
                "except ValueError:\n"
                "    pass\n"),
        },
        {
            "id": "s-coding-easy-roman", "difficulty": "easy",
            "prompt": ("Write `to_roman(n: int) -> str` converting an integer from 1 to 3999 into "
                       "an upper-case Roman numeral using the subtractive forms IV, IX, XL, XC, CD "
                       "and CM. Anything outside 1..3999 raises ValueError."),
            "hidden_tests": (
                "assert to_roman(1) == 'I'\n"
                "assert to_roman(4) == 'IV'\n"
                "assert to_roman(9) == 'IX'\n"
                "assert to_roman(1994) == 'MCMXCIV'\n"
                "assert to_roman(3999) == 'MMMCMXCIX'\n"
                "assert to_roman(40) == 'XL'\n"
                "for bad in (0, 4000, -3):\n"
                "    try:\n"
                "        to_roman(bad)\n"
                "        raise AssertionError('should have raised')\n"
                "    except ValueError:\n"
                "        pass\n"),
        },
        {
            "id": "s-coding-medium-balanced", "difficulty": "medium",
            "prompt": ("Write `is_balanced(s: str) -> bool` returning True when every round, square "
                       "and curly bracket in the string is closed by the matching kind in the right "
                       "order. Characters that are not brackets are ignored. The empty string is "
                       "balanced."),
            "hidden_tests": (
                "assert is_balanced('') is True\n"
                "assert is_balanced('a(b)[c]{d}') is True\n"
                "assert is_balanced('([{}])') is True\n"
                "assert is_balanced('(]') is False\n"
                "assert is_balanced('(()') is False\n"
                "assert is_balanced('())(') is False\n"
                "assert is_balanced('{[}]') is False\n"),
        },
        {
            "id": "s-coding-medium-flatten", "difficulty": "medium",
            "prompt": ("Write `flatten(data: dict, sep: str = '.') -> dict` that turns a nested "
                       "dictionary into a flat one whose names are the path joined by `sep`. Only "
                       "dictionaries are descended into; lists and every other value are left as "
                       "they are. An empty nested dictionary disappears entirely."),
            "hidden_tests": (
                "assert flatten({'a': 1, 'b': {'c': 2, 'd': {'e': 3}}}) == "
                "{'a': 1, 'b.c': 2, 'b.d.e': 3}\n"
                "assert flatten({}) == {}\n"
                "assert flatten({'a': {}}) == {}\n"
                "assert flatten({'a': [1, {'b': 2}]}) == {'a': [1, {'b': 2}]}\n"
                "assert flatten({'a': {'b': 1}}, sep='/') == {'a/b': 1}\n"
                "assert flatten({'a': None}) == {'a': None}\n"),
        },
        {
            "id": "s-coding-medium-versions", "difficulty": "medium",
            "prompt": ("Write `compare(a: str, b: str) -> int` comparing two dotted numeric version "
                       "strings. Return -1, 0 or 1. Versions may have different numbers of parts; a "
                       "missing part counts as 0, so '1.2' equals '1.2.0'. Leading zeros are "
                       "numeric, so '1.02' equals '1.2'. A part that is not a non-negative integer "
                       "raises ValueError."),
            "hidden_tests": (
                "assert compare('1.2', '1.10') == -1\n"
                "assert compare('1.2', '1.2.0') == 0\n"
                "assert compare('1.02', '1.2') == 0\n"
                "assert compare('2.0', '1.9.9') == 1\n"
                "assert compare('1.0.0.1', '1.0.0') == 1\n"
                "assert compare('0', '0.0.0') == 0\n"
                "try:\n"
                "    compare('1.x', '1.0')\n"
                "    raise AssertionError('should have raised')\n"
                "except ValueError:\n"
                "    pass\n"),
        },
        {
            "id": "s-coding-medium-csvrow", "difficulty": "medium",
            "prompt": ("Write `split_row(line: str) -> list[str]` splitting one CSV line on commas. "
                       "A field wrapped in double quotes may contain commas and may contain a "
                       "doubled double quote meaning one literal double quote. Unquoted fields are "
                       "taken as they are, without stripping. An empty line returns ['']."),
            "hidden_tests": (
                "assert split_row('a,b,c') == ['a','b','c']\n"
                "assert split_row('') == ['']\n"
                "assert split_row('a,,c') == ['a','','c']\n"
                "assert split_row('\"a,b\",c') == ['a,b','c']\n"
                "assert split_row('\"he said \"\"hi\"\"\",x') == ['he said \"hi\"','x']\n"
                "assert split_row('\" spaced \",y') == [' spaced ','y']\n"),
        },
        {
            "id": "s-coding-hard-lru", "difficulty": "hard",
            "prompt": ("Write `class LRU` with `__init__(self, capacity: int)`, `get(self, k)` "
                       "returning the value or None, and `put(self, k, v)`. Both reading and "
                       "writing mark an entry as most recently used. When the cache is over "
                       "capacity the least recently used entry is dropped. A capacity below 1 "
                       "raises ValueError. Also provide `__len__`."),
            "hidden_tests": (
                "c = LRU(2)\n"
                "c.put('a', 1); c.put('b', 2)\n"
                "assert c.get('a') == 1\n"
                "c.put('c', 3)\n"
                "assert c.get('b') is None\n"
                "assert c.get('a') == 1 and c.get('c') == 3\n"
                "assert len(c) == 2\n"
                "c.put('a', 9)\n"
                "assert c.get('a') == 9 and len(c) == 2\n"
                "try:\n"
                "    LRU(0)\n"
                "    raise AssertionError('should have raised')\n"
                "except ValueError:\n"
                "    pass\n"),
        },
        {
            "id": "s-coding-hard-wildcard", "difficulty": "hard",
            "prompt": ("Write `matches(pattern: str, text: str) -> bool` for glob-style matching "
                       "over the whole string, where `*` matches any run of characters including "
                       "none and `?` matches exactly one character. No other character is special. "
                       "Matching is case sensitive."),
            "hidden_tests": (
                "assert matches('', '') is True\n"
                "assert matches('*', '') is True\n"
                "assert matches('?', '') is False\n"
                "assert matches('a*b', 'ab') is True\n"
                "assert matches('a*b', 'axxxb') is True\n"
                "assert matches('a*b', 'axxx') is False\n"
                "assert matches('a?c', 'abc') is True\n"
                "assert matches('a?c', 'ac') is False\n"
                "assert matches('*a*b*', 'xaybz') is True\n"
                "assert matches('A', 'a') is False\n"
                "assert matches('*'*8 + 'z', 'a'*40) is False\n"),
        },
    ]


CODING_REFERENCE = {
    "s-coding-easy-chunk": (
        "def chunk(items, size):\n"
        "    if size < 1:\n"
        "        raise ValueError('size')\n"
        "    return [items[i:i + size] for i in range(0, len(items), size)]\n"),
    "s-coding-easy-roman": (
        "def to_roman(n):\n"
        "    if not isinstance(n, int) or not 1 <= n <= 3999:\n"
        "        raise ValueError('out of range')\n"
        "    table = [(1000,'M'),(900,'CM'),(500,'D'),(400,'CD'),(100,'C'),(90,'XC'),\n"
        "             (50,'L'),(40,'XL'),(10,'X'),(9,'IX'),(5,'V'),(4,'IV'),(1,'I')]\n"
        "    out = []\n"
        "    for value, sign in table:\n"
        "        while n >= value:\n"
        "            out.append(sign)\n"
        "            n -= value\n"
        "    return ''.join(out)\n"),
    "s-coding-medium-balanced": (
        "def is_balanced(s):\n"
        "    pairs = {')': '(', ']': '[', '}': '{'}\n"
        "    stack = []\n"
        "    for ch in s:\n"
        "        if ch in '([{':\n"
        "            stack.append(ch)\n"
        "        elif ch in pairs:\n"
        "            if not stack or stack.pop() != pairs[ch]:\n"
        "                return False\n"
        "    return not stack\n"),
    "s-coding-medium-flatten": (
        "def flatten(data, sep='.'):\n"
        "    out = {}\n"
        "    for name, value in data.items():\n"
        "        if isinstance(value, dict):\n"
        "            for inner, deep in flatten(value, sep).items():\n"
        "                out[f'{name}{sep}{inner}'] = deep\n"
        "        else:\n"
        "            out[name] = value\n"
        "    return out\n"),
    "s-coding-medium-versions": (
        "def compare(a, b):\n"
        "    def parts(v):\n"
        "        out = []\n"
        "        for piece in v.split('.'):\n"
        "            if not piece.isdigit():\n"
        "                raise ValueError(piece)\n"
        "            out.append(int(piece))\n"
        "        return out\n"
        "    left, right = parts(a), parts(b)\n"
        "    width = max(len(left), len(right))\n"
        "    left += [0] * (width - len(left))\n"
        "    right += [0] * (width - len(right))\n"
        "    return (left > right) - (left < right)\n"),
    "s-coding-medium-csvrow": (
        "def split_row(line):\n"
        "    out, field, i, quoted = [], [], 0, False\n"
        "    while i < len(line):\n"
        "        ch = line[i]\n"
        "        if quoted:\n"
        "            if ch == '\"':\n"
        "                if i + 1 < len(line) and line[i + 1] == '\"':\n"
        "                    field.append('\"')\n"
        "                    i += 1\n"
        "                else:\n"
        "                    quoted = False\n"
        "            else:\n"
        "                field.append(ch)\n"
        "        elif ch == '\"' and not field:\n"
        "            quoted = True\n"
        "        elif ch == ',':\n"
        "            out.append(''.join(field))\n"
        "            field = []\n"
        "        else:\n"
        "            field.append(ch)\n"
        "        i += 1\n"
        "    out.append(''.join(field))\n"
        "    return out\n"),
    "s-coding-hard-lru": (
        "from collections import OrderedDict\n"
        "class LRU:\n"
        "    def __init__(self, capacity):\n"
        "        if capacity < 1:\n"
        "            raise ValueError('capacity')\n"
        "        self.capacity = capacity\n"
        "        self.data = OrderedDict()\n"
        "    def get(self, k):\n"
        "        if k not in self.data:\n"
        "            return None\n"
        "        self.data.move_to_end(k)\n"
        "        return self.data[k]\n"
        "    def put(self, k, v):\n"
        "        self.data[k] = v\n"
        "        self.data.move_to_end(k)\n"
        "        while len(self.data) > self.capacity:\n"
        "            self.data.popitem(last=False)\n"
        "    def __len__(self):\n"
        "        return len(self.data)\n"),
    "s-coding-hard-wildcard": (
        "def matches(pattern, text):\n"
        "    memo = {}\n"
        "    def go(i, j):\n"
        "        if (i, j) in memo:\n"
        "            return memo[(i, j)]\n"
        "        if i == len(pattern):\n"
        "            ok = j == len(text)\n"
        "        elif pattern[i] == '*':\n"
        "            ok = go(i + 1, j) or (j < len(text) and go(i, j + 1))\n"
        "        elif j < len(text) and pattern[i] in ('?', text[j]):\n"
        "            ok = go(i + 1, j + 1)\n"
        "        else:\n"
        "            ok = False\n"
        "        memo[(i, j)] = ok\n"
        "        return ok\n"
        "    return go(0, 0)\n"),
}


# ---------------------------------------------------------------------------
# maths / reasoning
# ---------------------------------------------------------------------------
def _math() -> list[dict]:
    return [
        {"id": "s-math-easy-interest", "difficulty": "easy", "expected": "360",
         "prompt": ("A deposit of 2400 euro earns simple interest at 5 percent per year for three "
                    "years. How much interest is earned in total, in euro?")},
        {"id": "s-math-easy-average-speed", "difficulty": "easy", "expected": "84",
         "prompt": ("A train covers 210 kilometres at 70 kilometres per hour and returns along the "
                    "same route at 105 kilometres per hour. What is its average speed for the whole "
                    "journey, in kilometres per hour?")},
        {"id": "s-math-medium-distinct-digits", "difficulty": "medium", "expected": "4536",
         "prompt": ("How many four-digit whole numbers have four different digits? A four-digit "
                    "number may not start with zero.")},
        {"id": "s-math-medium-inclusion", "difficulty": "medium", "expected": "9168",
         "prompt": ("What is the sum of all positive whole numbers below 200 that are divisible by "
                    "3 or by 5?")},
        {"id": "s-math-medium-dice-increasing", "difficulty": "medium", "expected": "0.093",
         "atol": 0.0015,
         "prompt": ("A fair six-sided die is rolled three times. What is the probability that the "
                    "three results are strictly increasing? Give a decimal rounded to three "
                    "places.")},
        {"id": "s-math-hard-modular-97", "difficulty": "hard", "expected": "54",
         "prompt": ("What is the remainder when 2 raised to the power 2026 is divided by 97? Give "
                    "the whole-number remainder.")},
        {"id": "s-math-hard-incircle", "difficulty": "hard", "expected": "6",
         "prompt": ("A right triangle has legs of length 20 and 21. What is the radius of its "
                    "inscribed circle?")},
    ]


# ---------------------------------------------------------------------------
# factual research
# ---------------------------------------------------------------------------
def _research() -> list[dict]:
    return [
        {"id": "s-research-easy-http-429", "difficulty": "easy",
         "must_contain": ["429"], "must_not_contain": ["the code is 503"],
         "prompt": ("Which HTTP status code does RFC 6585 define for a client that has sent too "
                    "many requests in a given amount of time? Give the numeric code.")},
        {"id": "s-research-easy-json-absent", "difficulty": "easy",
         "must_contain": ["null"], "must_not_contain": ["undefined is the literal"],
         "prompt": ("Which JSON literal does RFC 8259 define for an empty or absent value? Name the "
                    "literal exactly as it is written on the wire.")},
        {"id": "s-research-medium-acid-durability", "difficulty": "medium",
         "must_contain": ["durab"], "must_not_contain": ["isolation guarantees survival"],
         "prompt": ("In the ACID properties of a database transaction, which property guarantees "
                    "that a committed transaction survives a crash of the system? Name it.")},
        {"id": "s-research-medium-dns-ipv6", "difficulty": "medium",
         "must_contain": ["aaaa"], "must_not_contain": ["the a record holds an ipv6"],
         "prompt": ("Which DNS resource record type maps a host name to an IPv6 address? Give the "
                    "record type.")},
        {"id": "s-research-medium-errno-refused", "difficulty": "medium",
         "must_contain": ["econnrefused"], "must_not_contain": [],
         "prompt": ("On POSIX, which errno does connect() report when the peer host actively "
                    "refuses the connection, for example because nothing is listening on the port? "
                    "Give the errno name.")},
        {"id": "s-research-hard-git-tree", "difficulty": "hard",
         "must_contain": ["tree"], "must_not_contain": ["a blob stores a directory"],
         "prompt": ("Which Git object type stores a directory listing - the names, modes and "
                    "object ids of the entries in one directory? Name the object type.")},
        {"id": "s-research-hard-ieee-nan", "difficulty": "hard",
         "must_contain": ["nan"], "must_not_contain": [],
         "prompt": ("In IEEE 754 binary64 arithmetic, which special value compares as not equal to "
                    "itself? Name it.")},
    ]


# ---------------------------------------------------------------------------
# web / UI design
# ---------------------------------------------------------------------------
_R_STYLE = {"label": "has an inline <style> block", "pattern": r"<style[\s>]"}
_R_CUSTOM = {"label": "declares a CSS custom property", "pattern": r"--[a-z-]+\s*:"}
_R_FOCUS = {"label": "styles :focus or :focus-visible", "pattern": r":focus(?:-visible)?\s*[,{:]"}


def _design() -> list[dict]:
    compact = [
        ("s-design-easy-status-badges",
         "a row of three status badges reading Active, Paused and Failed",
         ["Each badge must be a list item inside a <ul>.",
          "The colour of each badge must come from a CSS custom property.",
          "The row must sit inside a <main> landmark with an <h1> above it."],
         [{"label": "has a <ul>", "pattern": r"<ul[\s>]"},
          {"label": "has a <main> landmark", "pattern": r"<main[\s>]"},
          {"label": "has an <h1>", "pattern": r"<h1[\s>]"},
          _R_CUSTOM, _R_STYLE],
         '<main><h1>Status</h1><ul><li class="b">Active</li><li>Paused</li><li>Failed</li></ul></main>',
         ":root{--badge-active:#0a0;}"),
        ("s-design-easy-alert",
         "one dismissible warning alert",
         ["The alert must carry role=\"alert\".",
          "It must contain a <button> whose accessible name is given by aria-label.",
          "The accent colour must come from a CSS custom property."],
         [{"label": "has role=alert", "pattern": r'role\s*=\s*"?alert"?'},
          {"label": "has a <button>", "pattern": r"<button[\s>]"},
          {"label": "names the button with aria-label", "pattern": r"aria-label"},
          _R_CUSTOM, _R_STYLE],
         '<main><div role="alert"><p>Disk is nearly full.</p>'
         '<button aria-label="Dismiss this alert">x</button></div></main>',
         ":root{--warn:#c60;}"),
        ("s-design-easy-breadcrumb",
         "a breadcrumb trail of three levels ending on the current page",
         ["The trail must be a <nav> landmark with an aria-label.",
          "The items must be an ordered list.",
          "The last item must be marked with aria-current=\"page\"."],
         [{"label": "has a <nav> landmark", "pattern": r"<nav[\s>]"},
          {"label": "names the nav with aria-label", "pattern": r"aria-label"},
          {"label": "uses an ordered list", "pattern": r"<ol[\s>]"},
          {"label": "marks the current page", "pattern": r'aria-current\s*=\s*"?page'},
          _R_STYLE],
         '<nav aria-label="Breadcrumb"><ol><li><a href="#a">Home</a></li>'
         '<li><a href="#b">Billing</a></li><li><a href="#c" aria-current="page">Invoice</a></li></ol></nav>',
         "nav ol{display:flex;}"),
        ("s-design-easy-progress",
         "a labelled upload progress indicator standing at 40 percent",
         ["The bar must expose its value with aria-valuenow and role=\"progressbar\".",
          "It must have a visible text label associated with it.",
          "The fill colour must come from a CSS custom property."],
         [{"label": "has role=progressbar", "pattern": r'role\s*=\s*"?progressbar'},
          {"label": "exposes aria-valuenow", "pattern": r"aria-valuenow"},
          {"label": "has a visible label element", "pattern": r"<label[\s>]|aria-labelledby"},
          _R_CUSTOM, _R_STYLE],
         '<main><label id="l" for="bar">Uploading</label>'
         '<div id="bar" role="progressbar" aria-valuenow="40" aria-valuemin="0" aria-valuemax="100">'
         '<span class="fill"></span></div></main>',
         ":root{--fill:#268;}"),
        ("s-design-easy-toggle",
         "a single on/off switch for e-mail notifications",
         ["The control must be a checkbox input with a <label for=...> pointing at it.",
          "The switch must change appearance when the checkbox is :checked.",
          "There must be a visible focus style."],
         [{"label": "labels the control with for=", "pattern": r"<label[^>]+for="},
          {"label": "uses a checkbox input", "pattern": r'<input[^>]+type\s*=\s*"?checkbox'},
          {"label": "styles the :checked state", "pattern": r":checked"},
          _R_FOCUS, _R_STYLE],
         '<main><input type="checkbox" id="n"><label for="n">E-mail notifications</label></main>',
         'input:checked + label{font-weight:700;} input:focus-visible{outline:2px solid;}'),
        ("s-design-easy-stat-tile",
         "one statistic tile showing a metric name, its value and its change since last week",
         ["The three parts must be marked up as a description list.",
          "The tile must sit inside a <section> with an accessible name.",
          "The accent colour must come from a CSS custom property."],
         [{"label": "uses a description list", "pattern": r"<dl[\s>]"},
          {"label": "has a <dt>", "pattern": r"<dt[\s>]"},
          {"label": "has a <dd>", "pattern": r"<dd[\s>]"},
          {"label": "names the section", "pattern": r"aria-label|aria-labelledby"},
          _R_CUSTOM, _R_STYLE],
         '<section aria-label="Weekly signups"><dl><dt>Signups</dt><dd>1284</dd>'
         '<dt>Change</dt><dd>+12%</dd></dl></section>',
         ":root{--accent:#147;}"),
        ("s-design-easy-segmented",
         "a segmented control of three buttons where one is selected",
         ["Each segment must be a <button>.",
          "The selected segment must be marked with aria-pressed.",
          "The buttons must be laid out with a CSS gap.",
          "There must be a visible focus style."],
         [{"label": "has a <button>", "pattern": r"<button[\s>]"},
          {"label": "marks the selection with aria-pressed", "pattern": r"aria-pressed"},
          {"label": "uses a CSS gap", "pattern": r"\bgap\s*:"},
          _R_FOCUS, _R_STYLE],
         '<main><div class="seg"><button aria-pressed="true">Day</button>'
         '<button aria-pressed="false">Week</button><button aria-pressed="false">Month</button></div></main>',
         ".seg{display:flex;gap:4px;} button:focus-visible{outline:2px solid;}"),
        ("s-design-easy-empty-state",
         "an empty-state panel for a project list that has no projects yet",
         ["It must have a heading, one explanatory paragraph and one <button>.",
          "It must contain an inline <svg> illustration hidden from assistive technology.",
          "The spacing scale must come from a CSS custom property."],
         [{"label": "has a heading", "pattern": r"<h[12][\s>]"},
          {"label": "has a paragraph", "pattern": r"<p[\s>]"},
          {"label": "has a <button>", "pattern": r"<button[\s>]"},
          {"label": "has an inline <svg>", "pattern": r"<svg[\s>]"},
          {"label": "hides the illustration from assistive technology", "pattern": r"aria-hidden"},
          _R_CUSTOM, _R_STYLE],
         '<main><h1>No projects yet</h1><svg aria-hidden="true" viewBox="0 0 8 8"></svg>'
         '<p>Create one to get started.</p><button>New project</button></main>',
         ":root{--space:8px;}"),
        ("s-design-easy-search-field",
         "a site search field with a submit button",
         ["The input must be type=\"search\" and must have a <label for=...>.",
          "The form must have an accessible name via aria-label or aria-labelledby.",
          "There must be a visible focus style."],
         [{"label": "labels the input with for=", "pattern": r"<label[^>]+for="},
          {"label": "uses a search input", "pattern": r'<input[^>]+type\s*=\s*"?search'},
          {"label": "has a submit button", "pattern": r'<button[^>]*type\s*=\s*"?submit|<button[\s>]'},
          {"label": "names the form", "pattern": r"aria-label|aria-labelledby"},
          _R_FOCUS, _R_STYLE],
         '<main><form role="search" aria-label="Search the site"><label for="q">Search</label>'
         '<input id="q" type="search"><button type="submit">Go</button></form></main>',
         'input:focus{outline:2px solid;}'),
        ("s-design-easy-skeleton",
         "a loading skeleton placeholder for a list of three rows",
         ["The animation must be declared with @keyframes.",
          "The animation must be switched off under prefers-reduced-motion.",
          "The loading region must be marked with aria-busy."],
         [{"label": "declares @keyframes", "pattern": r"@keyframes"},
          {"label": "respects prefers-reduced-motion", "pattern": r"prefers-reduced-motion"},
          {"label": "marks the region aria-busy", "pattern": r"aria-busy"},
          _R_STYLE],
         '<main aria-busy="true"><div class="row"></div><div class="row"></div>'
         '<div class="row"></div></main>',
         "@keyframes pulse{to{opacity:.4}} .row{animation:pulse 1s infinite}"
         "@media (prefers-reduced-motion: reduce){.row{animation:none}}"),
        ("s-design-easy-avatar",
         "a user avatar showing two initials next to a name",
         ["The circle must be made with border-radius.",
          "The initials must be hidden from assistive technology, with the full name visible.",
          "The background colour must come from a CSS custom property."],
         [{"label": "uses border-radius", "pattern": r"border-radius\s*:"},
          {"label": "hides the initials from assistive technology", "pattern": r"aria-hidden"},
          _R_CUSTOM, _R_STYLE],
         '<main><span class="av" aria-hidden="true">FS</span><span>Florian S.</span></main>',
         ":root{--av-bg:#eee;} .av{border-radius:50%;}"),
        ("s-design-easy-quote",
         "a pull quote with its attribution",
         ["The quotation must be a <blockquote> and the attribution a <cite>.",
          "The pair must be wrapped in a <figure> with a <figcaption>.",
          "The rule colour must come from a CSS custom property."],
         [{"label": "has a <blockquote>", "pattern": r"<blockquote[\s>]"},
          {"label": "has a <cite>", "pattern": r"<cite[\s>]"},
          {"label": "has a <figure>", "pattern": r"<figure[\s>]"},
          {"label": "has a <figcaption>", "pattern": r"<figcaption[\s>]"},
          _R_CUSTOM, _R_STYLE],
         '<figure><blockquote><p>Measure it or do not claim it.</p></blockquote>'
         '<figcaption><cite>A reviewer</cite></figcaption></figure>',
         ":root{--rule:#ccc;}"),
    ]
    medium = [
        ("s-design-medium-card-grid",
         "a responsive grid of three feature cards",
         ["The grid must use CSS grid with repeat(auto-fit, minmax(...)) so it reflows without a "
          "media query.",
          "Each card must be an <article> with its own heading.",
          "There must be a media query that increases the padding on wide screens."],
         [{"label": "uses CSS grid", "pattern": r"display\s*:\s*grid"},
          {"label": "reflows with auto-fit and minmax", "pattern": r"repeat\(\s*auto-fit[^)]*minmax\("},
          {"label": "uses <article> for each card", "pattern": r"<article[\s>]"},
          {"label": "has a media query", "pattern": r"@media"},
          _R_STYLE],
         '<main><div class="g"><article><h2>Fast</h2><p>a</p></article>'
         '<article><h2>Cheap</h2><p>b</p></article><article><h2>Good</h2><p>c</p></article></div></main>',
         ".g{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));}"
         "@media (min-width:800px){.g{padding:2rem}}"),
        ("s-design-medium-tabs",
         "a three-tab interface with one panel visible",
         ["The tabs must use role=\"tablist\", role=\"tab\" and role=\"tabpanel\".",
          "The selected tab must carry aria-selected and point at its panel with aria-controls.",
          "There must be a visible focus style."],
         [{"label": "has a tablist", "pattern": r'role\s*=\s*"?tablist'},
          {"label": "has tabs", "pattern": r'role\s*=\s*"?tab"'},
          {"label": "has a tabpanel", "pattern": r'role\s*=\s*"?tabpanel'},
          {"label": "marks the selected tab", "pattern": r"aria-selected"},
          {"label": "links tab to panel", "pattern": r"aria-controls"},
          _R_FOCUS, _R_STYLE],
         '<main><div role="tablist"><button role="tab" aria-selected="true" aria-controls="p1" id="t1">One</button>'
         '<button role="tab" aria-selected="false" aria-controls="p2" id="t2">Two</button>'
         '<button role="tab" aria-selected="false" aria-controls="p3" id="t3">Three</button></div>'
         '<div role="tabpanel" id="p1" aria-labelledby="t1">first</div></main>',
         '[role="tab"]:focus-visible{outline:2px solid;}'),
        ("s-design-medium-accordion",
         "an accordion of four frequently asked questions",
         ["Each item must be a native <details> with a <summary>.",
          "The open/close transition must be disabled under prefers-reduced-motion.",
          "There must be a visible focus style on the summary."],
         [{"label": "uses <details>", "pattern": r"<details[\s>]"},
          {"label": "uses <summary>", "pattern": r"<summary[\s>]"},
          {"label": "respects prefers-reduced-motion", "pattern": r"prefers-reduced-motion"},
          _R_FOCUS, _R_STYLE],
         '<main><details><summary>Why?</summary><p>Because.</p></details>'
         '<details><summary>How?</summary><p>Like this.</p></details>'
         '<details><summary>When?</summary><p>Now.</p></details>'
         '<details><summary>Where?</summary><p>Here.</p></details></main>',
         'summary:focus-visible{outline:2px solid;} details{transition:all .2s}'
         '@media (prefers-reduced-motion: reduce){details{transition:none}}'),
        ("s-design-medium-login-form",
         "a sign-in form with an e-mail field and a passphrase field",
         ["Every input must have a <label for=...>.",
          "The e-mail field must be type=email and required.",
          "Each field must point at its hint text with aria-describedby.",
          "The page must declare a viewport meta tag."],
         [{"label": "labels every input with for=", "pattern": r"<label[^>]+for="},
          {"label": "email input is typed and required",
           "pattern": r'<input[^>]+type\s*=\s*"?email"?[^>]*required|<input[^>]+required[^>]*type\s*=\s*"?email'},
          {"label": "describes fields with aria-describedby", "pattern": r"aria-describedby"},
          {"label": "has a viewport meta tag", "pattern": r'name\s*=\s*"?viewport'},
          _R_STYLE],
         '<main><form><label for="e">E-mail</label>'
         '<input id="e" type="email" required aria-describedby="eh"><p id="eh">We never share it.</p>'
         '<label for="p">Passphrase</label><input id="p" type="password" aria-describedby="ph">'
         '<p id="ph">At least twelve characters.</p><button>Sign in</button></form></main>',
         "form{max-width:20rem}",
         '<meta name="viewport" content="width=device-width, initial-scale=1">'),
        ("s-design-medium-notifications",
         "a notification list of three items that announces new arrivals",
         ["The list must be an aria-live=\"polite\" region.",
          "Each timestamp must be a <time> element with a machine-readable datetime attribute.",
          "There must be a visible focus style on each item's link."],
         [{"label": "has a polite live region", "pattern": r'aria-live\s*=\s*"?polite'},
          {"label": "uses <time datetime=...>", "pattern": r"<time[^>]+datetime\s*="},
          {"label": "uses a list", "pattern": r"<ul[\s>]|<ol[\s>]"},
          _R_FOCUS, _R_STYLE],
         '<main><ul aria-live="polite"><li><a href="#1">Build finished</a> '
         '<time datetime="2026-09-18T09:00Z">09:00</time></li>'
         '<li><a href="#2">Deploy queued</a> <time datetime="2026-09-18T09:05Z">09:05</time></li>'
         '<li><a href="#3">Tests green</a> <time datetime="2026-09-18T09:10Z">09:10</time></li></ul></main>',
         "a:focus-visible{outline:2px solid;}"),
        ("s-design-medium-sortable-table",
         "a table of four invoices whose date column is currently sorted",
         ["The table must have a <caption> and a <thead>.",
          "Every header cell must carry scope=\"col\".",
          "The sorted column must be marked with aria-sort.",
          "Rows must alternate background using :nth-child."],
         [{"label": "has a <table> with a <caption>", "pattern": r"<table[\s>].*?<caption[\s>]"},
          {"label": "has a <thead>", "pattern": r"<thead[\s>]"},
          {"label": "scopes the header cells", "pattern": r'scope\s*=\s*"?col'},
          {"label": "marks the sorted column", "pattern": r"aria-sort"},
          {"label": "zebra-stripes with :nth-child", "pattern": r":nth-child"},
          _R_STYLE],
         '<main><table><caption>Invoices</caption><thead><tr>'
         '<th scope="col" aria-sort="ascending">Date</th><th scope="col">Amount</th></tr></thead>'
         '<tbody><tr><td>01</td><td>10</td></tr><tr><td>02</td><td>20</td></tr>'
         '<tr><td>03</td><td>30</td></tr><tr><td>04</td><td>40</td></tr></tbody></table></main>',
         "tbody tr:nth-child(even){background:#f4f4f4}"),
        ("s-design-medium-timeline",
         "a vertical timeline of four deployment events",
         ["The events must be an ordered list.",
          "The connecting line must be drawn with a ::before pseudo-element.",
          "The line colour must come from a CSS custom property.",
          "Below 480px the timeline must stack without indentation, via a media query."],
         [{"label": "uses an ordered list", "pattern": r"<ol[\s>]"},
          {"label": "draws the line with ::before", "pattern": r"::before"},
          {"label": "has a max-width media query", "pattern": r"@media[^{]*max-width"},
          _R_CUSTOM, _R_STYLE],
         '<main><ol class="tl"><li>build</li><li>test</li><li>stage</li><li>ship</li></ol></main>',
         ":root{--line:#bbb} .tl li::before{content:'';background:var(--line)}"
         "@media (max-width:480px){.tl{padding-left:0}}"),
    ]
    full_page = [
        ("s-design-hard-settings",
         "a full account settings page with a sidebar and a settings form",
         ["The sidebar must be a <nav> landmark and the active entry must carry aria-current.",
          "The layout must use CSS grid and collapse below 768px with a media query.",
          "The form controls must be grouped in a <fieldset> with a <legend>.",
          "The page must support a dark colour scheme via prefers-color-scheme."],
         [{"label": "uses CSS grid", "pattern": r"display\s*:\s*grid"},
          {"label": "has a <nav> landmark", "pattern": r"<nav[\s>]"},
          {"label": "marks the active entry", "pattern": r"aria-current"},
          {"label": "collapses below 768px", "pattern": r"@media[^{]*768"},
          {"label": "groups controls in a fieldset", "pattern": r"<fieldset[\s>]"},
          {"label": "has a <legend>", "pattern": r"<legend[\s>]"},
          {"label": "supports a dark colour scheme", "pattern": r"prefers-color-scheme"},
          _R_STYLE],
         '<div class="l"><nav><a href="#a" aria-current="page">Profile</a></nav>'
         '<main><h1>Settings</h1><form><fieldset><legend>Notifications</legend>'
         '<label for="x">E-mail</label><input id="x" type="checkbox"></fieldset></form></main></div>',
         ".l{display:grid;grid-template-columns:200px 1fr}"
         "@media (max-width:768px){.l{grid-template-columns:1fr}}"
         "@media (prefers-color-scheme: dark){body{background:#111;color:#eee}}"),
        ("s-design-hard-checkout",
         "a full three-step checkout page showing step two of three",
         ["The step indicator must be an ordered list inside a <nav> landmark, with the current "
          "step marked by aria-current.",
          "The address fields must be grouped in a <fieldset> with a <legend>.",
          "The summary and the form must sit in a CSS grid that becomes one column below 700px.",
          "Animations must be disabled under prefers-reduced-motion."],
         [{"label": "has a <nav> landmark", "pattern": r"<nav[\s>]"},
          {"label": "uses an ordered list for the steps", "pattern": r"<ol[\s>]"},
          {"label": "marks the current step", "pattern": r"aria-current"},
          {"label": "groups address fields in a fieldset", "pattern": r"<fieldset[\s>]"},
          {"label": "has a <legend>", "pattern": r"<legend[\s>]"},
          {"label": "uses CSS grid", "pattern": r"display\s*:\s*grid"},
          {"label": "collapses below 700px", "pattern": r"@media[^{]*700"},
          {"label": "respects prefers-reduced-motion", "pattern": r"prefers-reduced-motion"},
          _R_STYLE],
         '<nav aria-label="Checkout steps"><ol><li>Cart</li>'
         '<li aria-current="step">Address</li><li>Pay</li></ol></nav>'
         '<main class="g"><form><fieldset><legend>Address</legend>'
         '<label for="s">Street</label><input id="s"></fieldset></form><aside>Total 42</aside></main>',
         ".g{display:grid;grid-template-columns:2fr 1fr}"
         "@media (max-width:700px){.g{grid-template-columns:1fr}}"
         "@media (prefers-reduced-motion: reduce){*{animation:none;transition:none}}"),
        ("s-design-hard-docs",
         "a full documentation page with a left navigation, an article and a right table of contents",
         ["There must be a skip link to the main content as the first focusable element.",
          "The three columns must use CSS grid and drop to one column below 900px.",
          "The table of contents must be an <aside> that stays visible with position: sticky.",
          "The navigation must be a <nav> landmark whose current page carries aria-current.",
          "The page must support a dark colour scheme via prefers-color-scheme."],
         [{"label": "has a skip link", "pattern": r'href\s*=\s*"#[a-z-]*(?:main|content)'},
          {"label": "uses CSS grid", "pattern": r"display\s*:\s*grid"},
          {"label": "drops to one column below 900px", "pattern": r"@media[^{]*900"},
          {"label": "has a sticky <aside>", "pattern": r"<aside[\s>]"},
          {"label": "uses position: sticky", "pattern": r"position\s*:\s*sticky"},
          {"label": "has a <nav> landmark", "pattern": r"<nav[\s>]"},
          {"label": "marks the current page", "pattern": r"aria-current"},
          {"label": "supports a dark colour scheme", "pattern": r"prefers-color-scheme"},
          _R_STYLE],
         '<a class="skip" href="#main">Skip to content</a>'
         '<div class="g"><nav><a href="#a" aria-current="page">Routing</a></nav>'
         '<main id="main"><h1>Routing</h1><p>text</p></main><aside><ol><li>Intro</li></ol></aside></div>',
         ".g{display:grid;grid-template-columns:200px 1fr 200px}aside{position:sticky;top:0}"
         "@media (max-width:900px){.g{grid-template-columns:1fr}}"
         "@media (prefers-color-scheme: dark){body{background:#111}}"),
    ]

    out: list[dict] = []
    for group, difficulty, is_compact in ((compact, "easy", True), (medium, "medium", True),
                                          (full_page, "hard", False)):
        for entry in group:
            task_id, subject, requirements, rules, body, style = entry[:6]
            head_extra = entry[6] if len(entry) > 6 else ""
            prompt = ("Build a single self-contained HTML page showing " + subject + ". "
                      + " ".join(requirements) + (COMPACT if is_compact else ""))
            out.append({"id": task_id, "difficulty": difficulty, "prompt": prompt, "rules": rules,
                        "_reference": _page(body, style, head_extra)})
    return out


def _page(body: str, style: str, head_extra: str = "") -> str:
    return ("```html\n<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
            + (head_extra + "\n" if head_extra else "")
            + "<style>\n" + style + "\n</style>\n</head>\n<body>\n" + body
            + "\n</body>\n</html>\n```\n")


# ---------------------------------------------------------------------------
# summarisation
# ---------------------------------------------------------------------------
_SOURCE_TRUNCATION = (
    "A completion that the provider itself flags as a length stop has not finished. On the first "
    "held-out run one free route spent an output budget of 12000 tokens on two larger pages and "
    "returned after 305 and 377 seconds without closing the markup, while a metered route answered "
    "the same 2 prompts in about 2200 tokens and 18 seconds. The router recorded both of the "
    "unfinished answers as successes, because the gateway had returned status 200 with a choices "
    "array. The fix records the length stop as an observed failed attempt and lets the existing "
    "safe fallback try one other route. Only the provider's own machine-readable flag is read; the "
    "text of the answer is never inspected, and a stream is recorded but never retried because its "
    "bytes are already on the wire.")

_SOURCE_EVIDENCE = (
    "Routing decisions are driven by measured capability per category rather than by a single "
    "overall score, because a route that is strong at coding can be weak at page layout. Each "
    "benchmark document carries its own provenance and its own freshness, and a category whose "
    "evidence is older than the staleness window is discounted rather than trusted. When no usable "
    "evidence exists at all the router falls back to a safe default instead of guessing. The "
    "benchmark input is internal only and carries a warning that it may not be used commercially. "
    "Estimates and observations are kept in separate fields of the decision record so that a "
    "forecast can never be added to a measurement by accident.")

_SOURCE_ISOLATION = (
    "Answers produced by a model are executed only inside a Bubblewrap sandbox on an ephemeral "
    "copy of the files. The sandbox has no network, cannot read the home directory, receives no "
    "environment variables that look like credentials, and works in a private temporary directory "
    "that is discarded afterwards. The verdict of a coding task is a random value generated "
    "outside the sandbox and printed only after every assertion has passed, so an answer cannot "
    "reach it by printing a fixed marker or by exiting early. If the sandbox is unavailable the "
    "task is reported as unavailable and excluded, and is never run on the host instead.")


def _summarisation() -> list[dict]:
    return [
        {"id": "s-summary-easy-truncation", "difficulty": "easy", "source": _SOURCE_TRUNCATION,
         "min_words": 25, "max_words": 70, "must_retain": ["length", "fallback"],
         "prompt": ("Summarise the following text in 25 to 70 words for an engineer who has not "
                    "read it. Do not invent numbers.\n\n" + _SOURCE_TRUNCATION)},
        {"id": "s-summary-medium-truncation-exact", "difficulty": "medium",
         "source": _SOURCE_TRUNCATION, "min_words": 20, "max_words": 45,
         "must_retain": ["12000", "305"],
         "prompt": ("In 20 to 45 words, state the output budget the free route spent and the two "
                    "response times reported for it. Use only numbers that appear in the text."
                    "\n\n" + _SOURCE_TRUNCATION)},
        {"id": "s-summary-medium-truncation-limits", "difficulty": "medium",
         "source": _SOURCE_TRUNCATION, "min_words": 20, "max_words": 50,
         "must_retain": ["stream", "flag"],
         "prompt": ("In 20 to 50 words, state which signal the fix is allowed to read and why a "
                    "stream is not retried. Do not invent numbers.\n\n" + _SOURCE_TRUNCATION)},
        {"id": "s-summary-easy-evidence", "difficulty": "easy", "source": _SOURCE_EVIDENCE,
         "min_words": 25, "max_words": 70, "must_retain": ["category", "provenance"],
         "prompt": ("Summarise the following text in 25 to 70 words for an engineer who has not "
                    "read it. Do not invent numbers.\n\n" + _SOURCE_EVIDENCE)},
        {"id": "s-summary-medium-evidence-mechanism", "difficulty": "medium",
         "source": _SOURCE_EVIDENCE, "min_words": 20, "max_words": 50,
         "must_retain": ["stale", "fallback"],
         "prompt": ("In 20 to 50 words, state what happens to evidence that is too old and what "
                    "happens when there is no usable evidence. Keep the mechanism, drop the "
                    "motivation.\n\n" + _SOURCE_EVIDENCE)},
        {"id": "s-summary-hard-evidence-separation", "difficulty": "hard",
         "source": _SOURCE_EVIDENCE, "min_words": 15, "max_words": 40,
         "must_retain": ["estimate", "measurement"],
         "prompt": ("In 15 to 40 words, state exactly why forecasts and measurements are kept in "
                    "separate fields. Use the words estimate and measurement.\n\n"
                    + _SOURCE_EVIDENCE)},
        {"id": "s-summary-easy-isolation", "difficulty": "easy", "source": _SOURCE_ISOLATION,
         "min_words": 25, "max_words": 70, "must_retain": ["sandbox", "network"],
         "prompt": ("Summarise the following text in 25 to 70 words for an engineer who has not "
                    "read it. Do not invent numbers.\n\n" + _SOURCE_ISOLATION)},
        {"id": "s-summary-hard-isolation-verdict", "difficulty": "hard",
         "source": _SOURCE_ISOLATION, "min_words": 15, "max_words": 40,
         "must_retain": ["verdict", "unavailable"],
         "prompt": ("In 15 to 40 words, state how the verdict of a coding task is protected and "
                    "what happens when the sandbox cannot run. Do not invent numbers.\n\n"
                    + _SOURCE_ISOLATION)},
    ]


# ---------------------------------------------------------------------------
# cache-eligible repeats
# ---------------------------------------------------------------------------
_RUNBOOK = (
    "DEPLOYMENT RUNBOOK (internal reference, revision 12)\n"
    "The ingest service runs behind a load balancer with a readiness probe on /healthz that must "
    "answer within 2 seconds. A rolling release replaces at most 25 percent of the pods at a time "
    "and waits for two consecutive successful probes before continuing. If the error rate measured "
    "over a 5 minute window rises above 1 percent the release halts automatically and the previous "
    "revision is restored. Database migrations always run before the new pods start and must be "
    "backwards compatible for exactly one release, because the old and the new revision serve "
    "traffic at the same time. Configuration is read once at start-up and never re-read, so a "
    "configuration change requires a restart. Logs are shipped as structured JSON on standard "
    "output and are retained for 30 days. Any release outside the window from 08:00 to 18:00 UTC "
    "needs a second approver recorded in the change ticket.\n") * 6


def _cache_repeats() -> list[dict]:
    questions = [
        ("s-cache-repeat-1", "How much of the fleet may a rolling release replace at a time?", ["25"]),
        ("s-cache-repeat-2", "Which path does the readiness probe use?", ["/healthz"]),
        ("s-cache-repeat-3", "What halts a release automatically?", ["error rate"]),
        ("s-cache-repeat-4", "For how long are the logs retained?", ["30"]),
        ("s-cache-repeat-5", "When do database migrations run relative to the new pods?", ["before"]),
        ("s-cache-repeat-6", "What does a configuration change require?", ["restart"]),
        ("s-cache-repeat-7", "What does a release outside the stated window need?",
         ["second approver"]),
        ("s-cache-repeat-8", "In which format are logs shipped?", ["json"]),
    ]
    out = []
    for index, (task_id, question, must) in enumerate(questions):
        out.append({
            "id": task_id, "difficulty": "easy",
            "prompt": _RUNBOOK + "\nQuestion: " + question + " Answer in one short sentence.",
            "must_contain": must, "must_not_contain": [],
            "shared_prefix": "deployment-runbook-12",
            "repeat_of": None if index == 0 else questions[0][0],
        })
    return out


# ---------------------------------------------------------------------------
# reference answers (test fixture only - never sent to a model)
# ---------------------------------------------------------------------------
def _reference_answers() -> dict[str, str]:
    out: dict[str, str] = dict(CODING_REFERENCE)
    for task in _math():
        out[task["id"]] = f"Working omitted.\nFinal answer: {task['expected']}"
    for task in _research():
        out[task["id"]] = ("The answer is " + " and ".join(task["must_contain"])
                           + ", which is what the question asks for.")
    for task in _design():
        out[task["id"]] = task["_reference"]
    out.update({
        "s-summary-easy-truncation":
            "A length stop means the provider never finished the answer. The router used to "
            "record such an unfinished reply as a success because the gateway returned a normal "
            "response. It now records a failed attempt and lets the safe fallback try another "
            "route, reading only the provider flag.",
        "s-summary-medium-truncation-exact":
            "The free route spent an output budget of 12000 tokens and returned after 305 and 377 "
            "seconds without finishing either of the larger pages it had been asked for.",
        "s-summary-medium-truncation-limits":
            "Only the provider's own machine-readable flag may be read, never the answer text. A "
            "stream is recorded but never retried, because its bytes have already gone out on the "
            "wire and cannot be taken back.",
        "s-summary-easy-evidence":
            "Routing uses measured capability per category rather than one overall score, since a "
            "route strong at code can be weak at layout. Every benchmark document carries "
            "provenance and freshness; stale evidence is discounted and missing evidence falls "
            "back to a safe default. Forecasts stay separate from measurements.",
        "s-summary-medium-evidence-mechanism":
            "Evidence older than the staleness window is discounted rather than trusted, and stale "
            "input never decides a route on its own. When no usable evidence exists the router "
            "takes a safe default fallback instead of guessing.",
        "s-summary-hard-evidence-separation":
            "An estimate and a measurement live in separate fields of the decision record, so a "
            "forecast can never be added to a measurement by accident.",
        "s-summary-easy-isolation":
            "Model-produced answers run only inside a Bubblewrap sandbox on a throwaway copy, with "
            "no network, no home directory and no credential-shaped environment. The verdict comes "
            "from a random value made outside the sandbox and printed only after every assertion "
            "passes. If the sandbox is missing the task is excluded.",
        "s-summary-hard-isolation-verdict":
            "The verdict is a random value generated outside and printed only after every "
            "assertion passes. If the sandbox cannot run, the task is reported unavailable and "
            "excluded rather than run on the host.",
    })
    for task in _cache_repeats():
        out[task["id"]] = ("In the runbook this is stated as "
                           + " and ".join(task["must_contain"]) + ".")
    return out


# ---------------------------------------------------------------------------
def build_tasks(rng: random.Random | None = None) -> list[dict]:
    """The full supplement set. Deterministic: the seed only fixes the ordering."""
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
            tasks.append({"category": category, "grader": grader, "system": system,
                          **{k: v for k, v in item.items() if not k.startswith("_")}})
    if rng is not None:
        shuffled = [t for t in tasks if t["category"] != "cache_repeat"]
        rng.shuffle(shuffled)
        tasks = shuffled + [t for t in tasks if t["category"] == "cache_repeat"]
    return tasks


REFERENCE_ANSWERS = _reference_answers()
