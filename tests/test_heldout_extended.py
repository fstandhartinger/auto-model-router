"""The 2026-09-19 extension of the held-out task set.

The extension exists for one arithmetic reason: a paired comparison counts
PAIRS, a pair needs both rows graded, and the 18 September registration held
too few tasks per category to reach the pre-registered floor of ten valid pairs
once truncated rows were excluded.

Extending a task set after results have been seen is exactly the move a
pre-registration exists to prevent, so the three properties that make it
legitimate are pinned here rather than asserted in prose:

1. every category now carries at least ten tasks;
2. the 27 tasks that were already run are byte-identical - same id, same
   prompt, same system prompt, same category and grader - so the finished
   ledger keyed on those ids still describes what was actually asked;
3. the builder is still deterministic for a fixed seed.

The frozen digests in ``ORIGINAL_27`` were taken from the builder *before* the
extension was written. They are the anchor: a later edit to an original task
body changes a digest here, and this file has to be edited in the same commit
for the suite to stay green.
"""

import hashlib
import random
import re
import sys
from collections import Counter
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments import graders, heldout, sandbox              # noqa: E402
from experiments.tasks_heldout import build_tasks              # noqa: E402


def _identity(task: dict) -> str:
    """Everything about a task that decides what was asked of the model."""
    return hashlib.sha256("\x00".join(
        (task["id"], task["prompt"], task["system"], task["category"],
         task["grader"])).encode()).hexdigest()


#: id -> identity digest, computed from the builder as it stood at the
#: 2026-09-18 registration (task file sha256 84731010531b266f...).
ORIGINAL_27 = {
    "design-easy-pricing-card": "f790dd0a08e32d8327d1de42f1a3ea4e4fd8d80b78d444ffedd7b8b0fe217e59",
    "design-medium-responsive-table": "aaa09c50de59d37bad94b105ccdfce109785b9b622358acd9da802340920fad5",
    "design-medium-form": "47f0ba0331a82140cb41dc3da151d5bc973183f083842061483936c81431dce0",
    "design-hard-dashboard": "aacd24bc7ee944eefe612460ceae54466cdc0d3f2200d665f320575ffa0fb6f2",
    "coding-easy-runlength": "2b6f7bc2f790de3f9ebf394e93116b9f1b67a325db2d322d0ce07c270f567982",
    "coding-medium-intervals": "5c68c30e601d71d0bdb7b12c48693f2ae5a3d5ab9c51901c3ef34e3ea53cf119",
    "coding-medium-parser": "b718d4835fe6a7f218de75ae6abd25bb3284fe960fd161b69070a40963a5bdd7",
    "coding-hard-scheduler": "5687ae936c98023138fc39e9526fe3d1178a1ec1b6eb0d9ab6246942632c0d52",
    "math-easy-discount": "01e0f06fcff73b5cfec89ad45466cb30c7ed45f786829b4f6ca67d6f6807c3e6",
    "math-easy-rate": "21abf4c7b4af95c1579efb25215542dad6062b2f7b101a720b596d49084ce677",
    "math-medium-digits": "c57b67ff357942e19b7caf2c905efcd8e09a5d534bca2dea519ed9051e4cea05",
    "math-medium-probability": "394654b2cd84cf9835f578d7f6f7fe9c303b97e36f4c6192e4b9f223c61654c2",
    "math-hard-modular": "a873dde029d95a09920eb9a8874348da7b2fdaf75e33ec70df48b95dfab4c0c0",
    "math-hard-geometry": "3f7420bc9966c3550f54865777bca164c443c86e8c8b9f72947ea1713991520a",
    "research-easy-http-status": "5163678e4e04ba73b0c678fb90ad49e87d8c8e4d034a29053fad8fd4bb10fc45",
    "research-easy-unicode": "a9597da92ae7ea943ab3c2fe98582270107527b871f1e934f7b58dc5eb28b427",
    "research-medium-sql": "f365555fb5798e7ab46f9b156f8555e73e77b3e7ca638f90c003a959cf663831",
    "research-medium-tls": "da5476773119b7b56223bf18d5bbafd3bbdd6909cfdd9ec48a22c44d8e0796ec",
    "research-hard-posix": "c1050de124530a501d2100c40aa463ef474f471d0f46a83c3fbaeb0a08d416cb",
    "summary-easy-cache": "808bbccbf686335ecf69715d59f856c7f986ea97508e2af95c349c777eaeaea2",
    "summary-medium-cache-constrained": "7b0373c10e1ae4c7be9ac9a287a20e8d4d174eac4b5fc9af0b9b4d672fe86180",
    "summary-medium-quota": "75907dae05c3f0fbeec14bb23456af87ba7592dfdade1dce085474fa41f6c275",
    "summary-hard-quota-exact": "7031bf6de35772ad262131d88324c1a13f1d13a64730ed9284fd7d6c28ef3a44",
    "cache-repeat-1": "923ced205ea9b8ad6bdace493a292edc31e0bca45ab4966e2615c4587ae51268",
    "cache-repeat-2": "452ef50b1562c3b120ccfa5c046cc3fd0b30bb3ebac0357c4778f1cf93039b20",
    "cache-repeat-3": "1e9351ec7fbd6ccab1d31971eab86f6402f65f3b17bc6e2159364549e03b1165",
    "cache-repeat-4": "8cb63738b2770daa8ae319e3f497cb2b59f843d9e15e09446a91fb3f679c01d5",
}


#: id -> sha256 of the WHOLE task object (json, sorted keys), read from the
#: frozen 2026-09-18 task file ``runs/heldout/tasks.jsonl`` (sha256
#: 84731010531b266f...) - not from the builder. Unlike ``ORIGINAL_27`` this also
#: pins what decides a grade (expected answers, hidden tests, design rules,
#: must-retain facts), so an original task cannot keep its prompt and quietly
#: change how it is scored. Added 2026-09-19 after an independent review.
ORIGINAL_27_FULL = {
    "cache-repeat-1": "fd0b81ac0b1c5368f57ba6d2b702f740c21fb5865cc40a4b58a171cb1e2bf7ea",
    "cache-repeat-2": "c6441193df2e15fbd90182ce5e07c483fdc9f6e604ede901c8c8709e4c3b9253",
    "cache-repeat-3": "2696585b4d92f1a69607ca6db0a3e43e126dca524928e7dcd36cd79d52d94a5b",
    "cache-repeat-4": "5330771ef5fbd2a8b6ade26622d30fab624f97b2e3b399c9ec2849da48bb45f8",
    "coding-easy-runlength": "22a98be73b09a433aed55fe02c36a5102c255f3944e98b597626b18ad190c56a",
    "coding-hard-scheduler": "74413ccdfaf1c5614e41e453b2016864a1ca4c0dde3b96a20b85b388f3a04561",
    "coding-medium-intervals": "d74435f841da1ec1742f6b7e1429b30591c79f8bf86d16f212d956b8325dc6b3",
    "coding-medium-parser": "90c4ffef0209ccd6605dc950840e159fb8f28f721fe639161438514347fe8903",
    "design-easy-pricing-card": "167a1fbe2d9379990b6b105932d10aaaf1eec87782f5455058089ecca1300d65",
    "design-hard-dashboard": "72658aa98b8a0307bc0120fd4dd13d1663e1958df8683b5054a76304e811d797",
    "design-medium-form": "59b72d87ff32b1432d8b92540f754cfb0d63655352412c95342ac8415d00fe19",
    "design-medium-responsive-table": "61b2039a9c7e34261418599052c4af51ac6e8581870d9d4716b01ebd8d1b4648",
    "math-easy-discount": "725d7b118d745426e73a204e710a824eb77fc517c4d677f3e43bba0e6b0bcf8b",
    "math-easy-rate": "1c32cf2a9a2ac55c2dffdc4da3348bd90632c13557fea3a0a1abe047d7d6d871",
    "math-hard-geometry": "f44a6aab0464fd6cf91f27a3a9387b241b8936a4ec7450866541fa65e13c20ea",
    "math-hard-modular": "f49b85008461cfde20da9a9d992cdfce71916fdf9b21bf4b7a7e1af1a69e5d47",
    "math-medium-digits": "0c6afa6c5b3120baed5301a104abfe361f246d9ef28e35d530658d93a0f00ff2",
    "math-medium-probability": "1b4ed585c10ea8cee4a7572a08ab99940224766fa2e7f395ae056fc2e1de66ad",
    "research-easy-http-status": "b0ee6a9347b70377363ab279b17d5c1de6a4b066ad5cec0f505eb75685d79c20",
    "research-easy-unicode": "f60485e7d43180bd2c2bcbc885d4c139e26408cca83d9bb73aa718b6461462f8",
    "research-hard-posix": "96b8971a7ede0d0371d4d959dac28d09fa1782761ff2b59cb6323979be4e10a0",
    "research-medium-sql": "d76dea9685b5f73f935bd7b76392410d05b575d38d91a6fe4dca5d2eb127be99",
    "research-medium-tls": "0f9b9b0e1be8349085f13825b870610164a7d6830fac01a357230f007a605784",
    "summary-easy-cache": "06ed3a2b7b6c3da656e87a43c004594a5b94bd54e1e6f597854ae938c3a3ebdd",
    "summary-hard-quota-exact": "566a0cdfab745f9a0ac7dd48eba5b663861279c34dac662e2bb69775f07f1620",
    "summary-medium-cache-constrained": "15eff41d35c2537d81d3eab80eb32466c1a88caa2d1da85728218eedbd0e910f",
    "summary-medium-quota": "b2e81e2cb254264ded4deb8b35709c846cf03e4689a2e18bd0709dad6674cd2c",
}


# -- the three properties the extension has to have -------------------------
def test_every_category_now_has_at_least_ten_tasks():
    counts = Counter(t["category"] for t in build_tasks())
    assert set(counts) == set(heldout.CATEGORIES)
    under = {c: n for c, n in counts.items() if n < 10}
    assert not under, f"still under the pre-registered floor of ten: {under}"


def test_the_twenty_seven_already_run_tasks_are_byte_identical():
    """A task id that has already been answered must still mean the same prompt."""
    current = {t["id"]: t for t in build_tasks()}
    missing = sorted(set(ORIGINAL_27) - set(current))
    assert not missing, f"registered task ids have disappeared: {missing}"
    changed = {task_id: _identity(current[task_id])
               for task_id, frozen in ORIGINAL_27.items()
               if _identity(current[task_id]) != frozen}
    assert not changed, ("these tasks were already run on 2026-09-18 and their bodies "
                         f"have since changed: {sorted(changed)}")


def test_the_twenty_seven_are_identical_including_how_they_are_graded():
    import json
    current = {t["id"]: t for t in build_tasks()}
    changed = sorted(
        task_id for task_id, frozen in ORIGINAL_27_FULL.items()
        if hashlib.sha256(json.dumps(current[task_id], sort_keys=True).encode()).hexdigest()
        != frozen)
    assert not changed, f"grading inputs of already-run tasks have changed: {changed}"


def test_build_tasks_is_deterministic_for_a_fixed_seed():
    a = build_tasks(random.Random(20260919))
    b = build_tasks(random.Random(20260919))
    assert [_identity(t) for t in a] == [_identity(t) for t in b]
    # and a different seed only reorders, never changes the membership
    c = build_tasks(random.Random(7))
    assert sorted(_identity(t) for t in a) == sorted(_identity(t) for t in c)


# -- the extension introduces no new grader and no malformed task -----------
def test_the_extension_introduces_no_new_grader_kind():
    kinds = {graders.GRADER_KIND[t["grader"]] for t in build_tasks()}
    assert kinds == {"structural-proxy", "executed", "exact", "rubric"}


def test_every_task_carries_the_fields_its_grader_reads():
    for task in build_tasks():
        grader = task["grader"]
        if grader == "coding":
            assert task["hidden_tests"].strip()
        elif grader == "math":
            assert str(task["expected"]).strip()
        elif grader in ("research", "cache_repeat"):
            assert task["must_contain"] and isinstance(task["must_not_contain"], list)
        elif grader == "design":
            assert len(task["rules"]) >= 5
            for rule in task["rules"]:
                re.compile(rule["pattern"])          # a broken rule fails every answer
        elif grader == "summarisation":
            assert task["min_words"] < task["max_words"]
            assert task["must_retain"]


def test_a_summarisation_task_never_requires_a_fact_its_source_lacks():
    """``must_retain`` is checked against the answer; the source has to contain it.

    Otherwise the task is unpassable and the category quietly loses a pair.
    """
    for task in build_tasks():
        if task["grader"] != "summarisation":
            continue
        for fact in task["must_retain"]:
            assert fact.lower() in task["source"].lower(), (task["id"], fact)


def test_cache_repeats_still_share_exactly_one_warm_prefix():
    repeats = [t for t in build_tasks() if t["category"] == "cache_repeat"]
    assert len(repeats) >= 10
    assert len({t["shared_prefix"] for t in repeats}) == 1
    assert repeats[0]["repeat_of"] is None
    assert all(t["repeat_of"] == repeats[0]["id"] for t in repeats[1:])
    # the shuffle must leave the warming task first
    ordered = [t for t in build_tasks(random.Random(20260919))
               if t["category"] == "cache_repeat"]
    assert ordered[0]["id"] == repeats[0]["id"]


# -- the new coding tasks are solvable, and their hidden tests agree --------
#: One reference solution per task added on 2026-09-19. These are not model
#: answers: they exist so a spec that disagrees with its own hidden tests is
#: caught here rather than by a paid run producing a category of zeros.
REFERENCE_SOLUTIONS = {
    "coding-easy-balanced": '''
def is_balanced(s):
    pairs = {')': '(', ']': '[', '}': '{'}
    stack = []
    for ch in s:
        if ch in '([{':
            stack.append(ch)
        elif ch in pairs:
            if not stack or stack.pop() != pairs[ch]:
                return False
    return not stack
''',
    "coding-easy-rotate": '''
def rotate(items, k):
    if not items:
        return []
    k %= len(items)
    return items[-k:] + items[:-k] if k else list(items)
''',
    "coding-medium-roman": '''
def to_roman(n):
    if not isinstance(n, int) or not 1 <= n <= 3999:
        raise ValueError(n)
    table = [(1000,'M'),(900,'CM'),(500,'D'),(400,'CD'),(100,'C'),(90,'XC'),
             (50,'L'),(40,'XL'),(10,'X'),(9,'IX'),(5,'V'),(4,'IV'),(1,'I')]
    out = []
    for value, sign in table:
        while n >= value:
            out.append(sign)
            n -= value
    return "".join(out)
''',
    "coding-medium-flatten": '''
def flatten(items):
    out = []
    for item in items:
        if isinstance(item, list):
            out.extend(flatten(item))
        else:
            out.append(item)
    return out
''',
    "coding-medium-wordfreq": '''
import re
from collections import Counter

def top_words(text, n):
    words = re.findall(r"[A-Za-z']+", text or "")
    counts = Counter(w.lower() for w in words)
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return ranked[:n]
''',
    "coding-hard-lru": '''
from collections import OrderedDict

class LRUCache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.data = OrderedDict()

    def get(self, key):
        if key not in self.data:
            return -1
        self.data.move_to_end(key)
        return self.data[key]

    def put(self, key, value):
        if key in self.data:
            self.data.move_to_end(key)
        self.data[key] = value
        while len(self.data) > self.capacity:
            self.data.popitem(last=False)
''',
    "coding-hard-lcs": '''
def lcs_length(a, b):
    previous = [0] * (len(b) + 1)
    for ch in a:
        current = [0]
        for j, other in enumerate(b):
            current.append(previous[j] + 1 if ch == other
                           else max(previous[j + 1], current[j]))
        previous = current
    return previous[-1]
''',
}


@pytest.mark.parametrize("task_id", sorted(REFERENCE_SOLUTIONS))
def test_each_new_coding_task_passes_its_own_hidden_tests(task_id):
    if not sandbox.preflight()["available"]:
        pytest.skip("no Bubblewrap sandbox; coding answers are never run unisolated")
    task = next(t for t in build_tasks() if t["id"] == task_id)
    passed, detail = graders.grade_coding(
        task, "```python\n" + REFERENCE_SOLUTIONS[task_id] + "\n```")
    assert passed, detail


def test_every_new_coding_task_has_a_reference_solution():
    """A coding task with no reference solution has never been shown to be solvable."""
    ids = {t["id"] for t in build_tasks() if t["category"] == "coding"}
    original = ids & set(ORIGINAL_27)
    assert (ids - original) == set(REFERENCE_SOLUTIONS)


# -- the registration now freezes the arm, not only the questions -----------
def _write_config(tmp_path: Path) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(
        "providers:\n"
        "  p:\n"
        "    base_url: https://example.invalid/v1\n"
        "    api_key_env: DOES_NOT_EXIST\n"
        "models:\n"
        "- name: free-one\n"
        "  provider: p\n"
        "  upstream_id: vendor/free-one\n"
        "  free: true\n"
        "- name: metered-one\n"
        "  provider: p\n"
        "  upstream_id: vendor/metered-one\n"
        "  prices:\n"
        "    input: 1.0\n"
        "    output: 2.0\n")
    return path


def test_preregistering_with_a_config_freezes_the_policy_and_catalog(tmp_path):
    record = heldout.preregister(tmp_path / "run", seed=20260919,
                                 config_path=_write_config(tmp_path))
    assert record["tasks_by_category"] == Counter(
        t["category"] for t in build_tasks(random.Random(20260919)))
    identity = record["policy_identity"]
    assert record["policy_identity_sha256"] == identity["sha256"]
    assert {r["name"] for r in identity["catalog"]} == {"free-one", "metered-one"}
    assert "auto_router/router.py" in record["product_sha256"]
    assert record["config_sha256"]


def test_preregistering_without_a_config_still_works_and_pins_nothing_about_the_arm(tmp_path):
    record = heldout.preregister(tmp_path / "run", seed=20260919)
    assert "policy_identity" not in record
    assert record["task_count"] == len(build_tasks())


def test_the_note_records_what_this_registration_supersedes(tmp_path):
    record = heldout.preregister(tmp_path / "run", seed=20260919,
                                 note="supersedes only the sample size")
    assert "supersedes only the sample size" in record["note"]
    # and never at the cost of the standing warning about a self-certifying file
    assert "self-certifying" in record["note"]


def test_a_run_refuses_to_start_when_the_registered_catalog_has_moved(tmp_path):
    """The arm under test is the policy over the catalog, so it has to be pinned."""
    from auto_router.config import load_config
    from auto_router.router import Router
    from experiments import heldout_run

    config_path = _write_config(tmp_path)
    record = heldout.preregister(tmp_path / "run", seed=20260919, config_path=config_path)

    config = load_config(config_path)
    heldout_run._assert_registered_policy(record, config, Router(config))   # unchanged: fine

    config_path.write_text(config_path.read_text().replace("output: 2.0", "output: 9.0"))
    moved = load_config(config_path)
    with pytest.raises(SystemExit) as excinfo:
        heldout_run._assert_registered_policy(record, moved, Router(moved))
    assert "metered-one" in str(excinfo.value)


def test_a_registration_without_a_policy_identity_runs_and_says_it_is_unpinned(tmp_path, capsys):
    from auto_router.config import load_config
    from auto_router.router import Router
    from experiments import heldout_run

    config = load_config(_write_config(tmp_path))
    heldout_run._assert_registered_policy({}, config, Router(config))
    assert "not pinned" in capsys.readouterr().err


# -- the new design rules are satisfiable by a page a model could write ------
#: One minimal reference page per design task added on 2026-09-19. Like the
#: coding reference solutions, these are not model answers: a structural rule
#: with a typo in its regular expression fails every answer and turns a whole
#: category into zeros that look like a finding. This catches that before a
#: paid run does.
REFERENCE_PAGES = {
    "design-easy-alert-banner": '''<!doctype html><html><head><style>
      :root { --accent: #b00; } .a { color: var(--accent); }
    </style></head><body><main>
      <div class="a" role="alert">Saved.</div>
      <button aria-label="Dismiss this message">x</button>
    </main></body></html>''',
    "design-easy-site-header": '''<!doctype html><html><head><style>
      header { display: flex; }
    </style></head><body><header><span>Acme</span>
      <nav aria-label="Primary"><ul><li><a href="#a">A</a></li><li><a href="#b">B</a></li>
      <li><a href="#c">C</a></li></ul></nav></header></body></html>''',
    "design-easy-stat-tile": '''<!doctype html><html><head><style>
      :root { --up: green; } .sr-only { position: absolute; }
    </style></head><body><section><h2>Revenue</h2><p>1.2M</p>
      <span aria-label="up 4 percent since last month">&#9650;</span>
    </section></body></html>''',
    "design-easy-toggle": '''<!doctype html><html><head><style>
      .t:focus-visible { outline: 2px solid; } .t:checked + .track { background: green; }
    </style></head><body><main>
      <input class="t" id="digest" type="checkbox"><label for="digest">Weekly digest</label>
      <span class="track"></span></main></body></html>''',
    "design-medium-card-grid": '''<!doctype html><html><head><style>
      .g { display: grid; grid-template-columns: repeat(auto-fit, minmax(16rem, 1fr)); }
      .thumb { background: linear-gradient(90deg, #135, #468); }
      @media (max-width: 40rem) { .g { gap: 0; } }
    </style></head><body><main class="g">
      <article><div class="thumb"></div><h2>One</h2></article>
      <article><div class="thumb"></div><h2>Two</h2></article>
    </main></body></html>''',
    "design-medium-modal": '''<!doctype html><html><head><style>
      button:focus-visible { outline: 2px solid; }
    </style></head><body><main><p>Body</p>
      <dialog open aria-modal="true" aria-labelledby="t"><h2 id="t">Delete?</h2>
      <button>Confirm</button><button>Cancel</button></dialog></main></body></html>''',
    "design-medium-stepper": '''<!doctype html><html><head><style>
      @media (max-width: 40rem) { ol { display: block; } }
    </style></head><body><main>
      <ol><li>Cart</li><li aria-current="step">Address</li><li>Pay</li></ol>
      <form><label for="street">Street</label><input id="street"></form>
    </main></body></html>''',
    "design-medium-breadcrumb": '''<!doctype html><html><head><style>
      li + li::before { content: "/"; } ol { flex-wrap: wrap; }
    </style></head><body><nav aria-label="Breadcrumb"><ol>
      <li><a href="#a">Home</a></li><li><a href="#b">Docs</a></li><li><a href="#c">API</a></li>
      <li><a href="#d" aria-current="page">Routes</a></li></ol></nav></body></html>''',
    "design-medium-empty-state": '''<!doctype html><html><head><style>
      section { max-width: 100%; text-align: center; }
    </style></head><body><section><svg width="64" height="64"></svg>
      <h2>No projects yet</h2><p>Create one to get started.</p>
      <button>New project</button></section></body></html>''',
    "design-hard-settings-tabs": '''<!doctype html><html><head><style>
      @media (prefers-color-scheme: dark) { body { background: #111; } }
      @media (prefers-reduced-motion: reduce) { * { transition: none; } }
      @media (max-width: 600px) { [role="tablist"] { display: block; } }
    </style></head><body><main>
      <div role="tablist"><button role="tab" aria-selected="true">Profile</button>
      <button role="tab" aria-selected="false">Notifications</button>
      <button role="tab" aria-selected="false">Billing</button></div>
      <div role="tabpanel"><h2>Profile</h2></div></main></body></html>''',
}


@pytest.mark.parametrize("task_id", sorted(REFERENCE_PAGES))
def test_each_new_design_task_can_be_satisfied(task_id):
    task = next(t for t in build_tasks() if t["id"] == task_id)
    passed, detail = graders.grade_design(
        task, "```html\n" + REFERENCE_PAGES[task_id] + "\n```")
    assert passed, detail


def test_every_new_design_task_has_a_reference_page():
    ids = {t["id"] for t in build_tasks() if t["category"] == "design"}
    assert (ids - set(ORIGINAL_27)) == set(REFERENCE_PAGES)
