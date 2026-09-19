"""The two design tasks of the second design-floor supplement (2026-09-19, 14:45Z slice).

The first supplement (``runs/design-supplement-20260919T1046Z``) produced no
valid pair: both metered control calls were rejected with HTTP 401 before any
model ran, because the runner's environment had no ``OPEN_ROUTER_API_KEY``.
That run, like the extended held-out run, is immutable evidence. This file adds
two *new* design tasks for a third, separately registered set; it replaces
nothing and reuses none of the earlier task ids or prompts.

The task shape follows ``tasks_design_supplement``: every rule the grader checks
carries the exact ``demand`` phrase that must appear in its prompt (tests
enforce it, including the two checks ``graders.grade_design`` always applies),
and the pages are compact so a complete answer fits the 12,000-token budget.

The builder is a pure function with no randomness.
"""

from __future__ import annotations

from experiments.tasks_design_supplement import (IMPLICIT_CHECK_DEMANDS, SYSTEM,  # noqa: F401
                                                 _COMMON_HEAD, _COMMON_TAIL, _DOCTYPE, _STYLE)

_SPECS = [
    {
        "id": "ds3-design-easy-opening-hours",
        "lead": "Build a single self-contained HTML page showing a small shop's opening hours "
                "for the seven days of the week.",
        "steps": [
            "3. Put the hours in a <table> whose <caption> reads 'Opening hours'.",
            "4. Start every row with a day header written as <th scope=\"row\">.",
            "5. Write every opening and closing time inside a <time> element, "
            "for example <time>09:00</time>.",
            "6. Add one @media (prefers-color-scheme: dark) rule that switches the table "
            "to light text on a dark background.",
        ],
        "rules": [
            _DOCTYPE, _STYLE,
            {"label": "uses a <table>", "pattern": r"<table[\s>]", "demand": "<table>"},
            {"label": "the table has a <caption>", "pattern": r"<caption[\s>]",
             "demand": "<caption>"},
            {"label": "row headers use scope=row",
             "pattern": r"<th[^>]+scope\s*=\s*[\"']?row", "demand": "<th scope=\"row\">"},
            {"label": "times are marked with <time>", "pattern": r"<time[\s>]",
             "demand": "<time> element"},
            {"label": "has a dark-mode media query", "pattern": r"prefers-color-scheme\s*:\s*dark",
             "demand": "@media (prefers-color-scheme: dark) rule"},
        ],
    },
    {
        "id": "ds3-design-easy-recipe-card",
        "lead": "Build a single self-contained HTML page with a compact recipe card for "
                "'Tomato soup'.",
        "steps": [
            "3. Wrap the card in an <article> element with an <h2> title.",
            "4. List the ingredients in an unordered list, a <ul>.",
            "5. List the method steps in an ordered list, an <ol>.",
            "6. Declare the accent colour once as a CSS custom property and apply it at least "
            "once with var(--...), for example color: var(--accent).",
        ],
        "rules": [
            _DOCTYPE, _STYLE,
            {"label": "wrapped in an <article>", "pattern": r"<article[\s>]",
             "demand": "<article> element"},
            {"label": "has an <h2> title", "pattern": r"<h2[\s>]", "demand": "<h2> title"},
            {"label": "ingredients in a <ul>", "pattern": r"<ul[\s>]", "demand": "a <ul>"},
            {"label": "steps in an <ol>", "pattern": r"<ol[\s>]", "demand": "an <ol>"},
            {"label": "uses a CSS custom property via var()", "pattern": r"var\(\s*--",
             "demand": "var(--...)"},
        ],
    },
]


def build_tasks() -> list[dict]:
    """The two tasks of the second design supplement, in registered run order."""
    tasks = []
    for spec in _SPECS:
        prompt = "\n".join([spec["lead"], "", _COMMON_HEAD.rstrip("\n"),
                            *spec["steps"], "", _COMMON_TAIL])
        tasks.append({
            "id": spec["id"],
            "category": "design",
            "difficulty": "easy",
            "grader": "design",
            "system": SYSTEM,
            "prompt": prompt,
            "rules": [dict(rule) for rule in spec["rules"]],
            "implicit_check_demands": dict(IMPLICIT_CHECK_DEMANDS),
        })
    return tasks
