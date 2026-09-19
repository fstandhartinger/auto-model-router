"""The two design tasks of the 2026-09-19 design-floor supplement.

The extended held-out run (``runs/heldout-extended-20260919T062948Z``) finished
with nine valid design pairs, one short of its floor of ten, because five router
design answers hit the 12,000-token output budget. That run is immutable
evidence. This file adds two *new* design tasks for a separately registered
supplement; it replaces nothing.

Two lessons from the earlier sets are built into the task shape:

* **Every rule the grader checks is demanded, in so many words, by the
  prompt.** Two registered design tasks graded a CSS custom property and a
  ``<section>`` their prompts never asked for, and one of those decided the
  only discordant design pair. Here each rule carries the exact ``demand``
  phrase that must appear in its prompt, and the tests enforce it, including
  for the two checks ``graders.grade_design`` always applies (markup, not prose;
  nothing loaded from a URL).
* **The pages are compact.** A complete answer is a few dozen lines, so it fits
  the registered 12,000-token budget with a wide margin. Whether a reasoning
  route spends that budget thinking is not something a prompt can control; a
  truncation is recorded as one.

The builder is a pure function with no randomness: the same call always
returns the same tasks in the same order.
"""

from __future__ import annotations

SYSTEM = ("You are a front-end engineer. Reply with exactly one fenced html code block "
          "containing a complete, self-contained page, and nothing else.")

#: The checks ``graders.grade_design`` applies to every design answer, whatever
#: the task says, and the prompt phrase that demands each of them here.
IMPLICIT_CHECK_DEMANDS = {
    "is markup, not prose": "Start the page with <!DOCTYPE html>",
    "self-contained (no external stylesheet, script, font, image or frame)":
        "load nothing from a URL (no external stylesheet, script, font, image or frame)",
}

_COMMON_HEAD = ("Requirements:\n"
                "1. Start the page with <!DOCTYPE html>.\n"
                "2. Put all CSS in one inline <style> block and load nothing from a URL "
                "(no external stylesheet, script, font, image or frame).\n")
_COMMON_TAIL = ("Keep it short: no JavaScript, under 60 lines of code, and no commentary "
                "outside the code block.")

_DOCTYPE = {"label": "starts with a doctype", "pattern": r"<!doctype html",
            "demand": "<!DOCTYPE html>"}
_STYLE = {"label": "has an inline <style> block", "pattern": r"<style[\s>]",
          "demand": "one inline <style> block"}

_SPECS = [
    {
        "id": "ds2-design-easy-shortcut-card",
        "lead": "Build a single self-contained HTML page showing a compact keyboard-shortcut "
                "reference card with four shortcuts.",
        "steps": [
            "3. Wrap the card in a <main> element.",
            "4. Give the card an <h1> heading.",
            "5. List the shortcuts in a description list: a <dl> with one <dt> per key "
            "combination and one <dd> per action.",
            "6. Mark up every key with a <kbd> element.",
            "7. Add one @media rule that stacks each term above its description on narrow "
            "screens.",
        ],
        "rules": [
            _DOCTYPE, _STYLE,
            {"label": "has a <main> landmark", "pattern": r"<main[\s>]",
             "demand": "<main> element"},
            {"label": "has an <h1>", "pattern": r"<h1[\s>]", "demand": "<h1> heading"},
            {"label": "uses a description list", "pattern": r"<dl[\s>]",
             "demand": "a <dl>"},
            {"label": "has terms", "pattern": r"<dt[\s>]", "demand": "one <dt>"},
            {"label": "has descriptions", "pattern": r"<dd[\s>]", "demand": "one <dd>"},
            {"label": "marks keys with <kbd>", "pattern": r"<kbd[\s>]",
             "demand": "<kbd> element"},
            {"label": "has a media query", "pattern": r"@media", "demand": "@media rule"},
        ],
    },
    {
        "id": "ds2-design-easy-star-rating",
        "lead": "Build a single self-contained HTML page with a five-star rating input for "
                "'Rate this article'.",
        "steps": [
            "3. Group the stars in a <fieldset> with a <legend> that reads 'Rate this "
            "article'.",
            "4. Build each star from a real <input type=\"radio\">, all sharing one name, so "
            "the keyboard reaches them.",
            "5. Give each radio an accessible name with a <label for=...> such as '3 stars'.",
            "6. Show a visible keyboard focus with a :focus-visible rule.",
        ],
        "rules": [
            _DOCTYPE, _STYLE,
            {"label": "groups the stars in a <fieldset>", "pattern": r"<fieldset[\s>]",
             "demand": "<fieldset>"},
            {"label": "the group has a <legend>", "pattern": r"<legend[\s>]",
             "demand": "<legend>"},
            {"label": "is built from radio inputs",
             "pattern": r"<input[^>]+type\s*=\s*[\"']?radio",
             "demand": "<input type=\"radio\">"},
            {"label": "labels are associated with for=", "pattern": r"<label[^>]+for\s*=",
             "demand": "<label for=...>"},
            {"label": "styles :focus-visible", "pattern": r":focus-visible",
             "demand": ":focus-visible rule"},
        ],
    },
]


def build_tasks() -> list[dict]:
    """The two supplement tasks, in registered run order."""
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
