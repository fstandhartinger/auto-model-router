"""Publication gate: no organisation-specific names may enter the repository.

The only permitted match is Anthropic's own message-id prefix in the API translation,
which the Anthropic API translation must reproduce.
"""
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Built from pieces so this file does not match itself.
FORBIDDEN = re.compile("|".join(["m" + "sg", "ch" + "utes"]), re.IGNORECASE)
ALLOWED_LINE = re.compile(r'f"' + "m" + r'sg_\{uuid\.uuid4\(\)\.hex\[:24\]\}"')
SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", ".venv", "node_modules"}


def _files():
    try:
        out = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard"],
                             cwd=ROOT, capture_output=True, text=True, check=True).stdout.split()
        paths = [ROOT / p for p in out]
    except (subprocess.CalledProcessError, FileNotFoundError):
        paths = [p for p in ROOT.rglob("*") if p.is_file()]
    for path in paths:
        if any(part in SKIP_DIRS for part in path.relative_to(ROOT).parts) or not path.is_file():
            continue
        yield path


def test_no_org_mentions():
    offenders = []
    for path in _files():
        rel = str(path.relative_to(ROOT))
        if FORBIDDEN.search(rel):
            offenders.append(f"path: {rel}")
        try:
            text = path.read_text(errors="replace")
        except OSError:
            continue
        for number, line in enumerate(text.splitlines(), 1):
            if FORBIDDEN.search(line) and not ALLOWED_LINE.search(line):
                offenders.append(f"{rel}:{number}: {line.strip()[:120]}")
    assert not offenders, "organisation mentions found:\n" + "\n".join(offenders)
