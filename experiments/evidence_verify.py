#!/usr/bin/env python3
"""Offline verification of a frozen run registration — and its exact limit.

``supplement.py verify`` is the **live-run** verifier. It re-reads the runtime
config, re-resolves the policy and catalog from it, and refuses to run when that
file is missing or has drifted. That is the right behaviour before spending
money, and nothing here weakens it.

It is also, by construction, **not reproducible by a reader**. The runtime
config carries provider credentials, is not in the repository and cannot be
reconstructed from the registration, so anyone who does not already hold that
exact file gets a refusal rather than a verification.

This module is the part a reader *can* run. It re-derives the **public
identity** of a finished run from the registration and the published checkout,
and it never opens a config at all:

* the task file, against its registered digest, and the registered task ids and
  per-category counts against the task file itself;
* the analysis plan, against the plan in this checkout;
* the frozen policy/catalog identity, against the digest it carries — the
  resolved catalog is *inside* the registration, so editing a route's price,
  capability or staleness after the fact is detectable without the config;
* every registered experiment-file digest and every registered product-file
  digest, against this checkout, in both directions for the product package so
  that a file added after registration is a failure too.

**What it cannot do, and says so on every run.** A digest of a secret-bearing
file is the only thing in the registration that binds it to the configuration
that was actually used. Recomputing that digest needs the file. So:

    offline verification of a frozen redacted identity is not proof that the
    original secret-bearing runtime config existed, that it hashed to the
    recorded digest, or that the frozen identity was derived from it.

That gap is not repairable after the fact for a run that did not freeze
anything credential-free. It *is* repairable for the next one:
``freeze_public_config`` writes a credential-free projection of the config next
to the registration at registration time, and ``verify_offline`` checks it when
a run has one. A rotated credential does not change its digest; a changed route,
endpoint or policy setting does.

Usage::

    python experiments/evidence_verify.py --dir runs/heldout-supplement-<ts>
    python experiments/evidence_verify.py --dir <new-run> --freeze-public-config <config>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments import heldout, supplement                   # noqa: E402

#: The filename a run uses for its credential-free configuration identity.
PUBLIC_CONFIG_FILE = "config-public-identity.json"

#: Mapping keys whose *value* is a credential. Matched by shape, not by a list
#: of vendors, so a config naming a provider this repository has never heard of
#: is redacted just the same. It over-matches on purpose - ``token_budget`` goes
#: too - because the artifact this feeds is meant to be published, and a dropped
#: key is still listed by name while a leaked one cannot be taken back.
CREDENTIAL_KEY = re.compile(
    r"(?i)(api[_-]?key|auth[_-]?token|access[_-]?token|bearer|"
    r"secret|password|passwd|credential|(^|[_-])token($|[_-])|(^|[_-])key$)")

#: ``api_key_env`` and friends hold the *name* of an environment variable, never
#: the value, and the name is part of the public identity worth keeping.
NAMES_A_VARIABLE = re.compile(r"(?i)_(env|env_var|var|file|path)$")

#: A second net, over values rather than keys: anything key-shaped is dropped
#: wherever it appears, including under an innocent-looking key.
CREDENTIAL_VALUE = re.compile(
    r"(sk-[A-Za-z0-9_-]{16,}|ghp_[A-Za-z0-9]{16,}|cpk_[A-Za-z0-9.]{16,}|"
    r"AKIA[0-9A-Z]{16}|Bearer\s+[A-Za-z0-9._-]{20,}|[A-Za-z0-9_-]{48,})")


def _sha256_of(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


# ---------------------------------------------------------------------------
# a credential-free projection of a config
# ---------------------------------------------------------------------------
def _redact(node):
    """Return ``node`` with every credential value removed, recording the keys."""
    if isinstance(node, dict):
        kept, dropped = {}, []
        for key, value in node.items():
            name = str(key)
            is_credential = bool(CREDENTIAL_KEY.search(name)) and not NAMES_A_VARIABLE.search(name)
            if not is_credential and isinstance(value, str) and CREDENTIAL_VALUE.search(value):
                is_credential = True
            if is_credential:
                dropped.append(name)
                continue
            kept[name] = _redact(value)
        if dropped:
            kept["credential_keys"] = sorted(dropped)
        return kept
    if isinstance(node, list):
        return [_redact(item) for item in node]
    return node


def public_config_identity(config_path: str | Path) -> dict:
    """The part of a runtime config that can be published, plus its own digest.

    Everything credential-shaped is dropped and only its *key name* is kept, so
    a provider that is configured cannot be mistaken for one that is not. The
    digest is over the projection, which means rotating a key leaves it
    unchanged while changing a route, an endpoint or a policy setting does not.
    """
    path = Path(config_path)
    text = path.read_text()
    if path.suffix in (".yaml", ".yml"):
        import yaml
        raw = yaml.safe_load(text) or {}
    else:
        raw = json.loads(text)
    identity = _redact(raw)
    identity["sha256"] = _sha256_of(identity)
    return identity


def freeze_public_config(run_dir: str | Path, config_path: str | Path) -> Path:
    """Write the credential-free configuration identity next to a registration.

    Meant to run **at registration time** for a new run. It refuses to overwrite
    an existing artifact: rewriting a frozen identity later is the thing
    freezing exists to prevent.
    """
    run_dir = Path(run_dir)
    out = run_dir / PUBLIC_CONFIG_FILE
    if out.exists():
        raise SystemExit(f"{out} already exists; a frozen identity is not rewritten")
    registration = run_dir / "preregistration.json"
    if not registration.exists():
        raise SystemExit(f"no registration in {run_dir}; register the run first")
    record = json.loads(registration.read_text())
    actual = heldout.digest(Path(config_path))
    if record.get("config_sha256") not in (None, actual):
        raise SystemExit(
            f"{config_path} is not the config this run registered\n"
            f"  registered: {record['config_sha256']}\n  actual:     {actual}")
    identity = public_config_identity(config_path)
    payload = {
        "config_path": record.get("config_path") or str(Path(config_path).resolve()),
        "config_sha256": record.get("config_sha256") or actual,
        "config_sha256_note": ("the digest of the secret-bearing file itself. It is recorded "
                               "here and in the registration; it cannot be recomputed by anyone "
                               "who does not hold that file."),
        "public_identity": identity,
        "note": ("Every credential value was dropped and only its key name kept. Rotating a "
                 "credential does not change public_identity.sha256; changing a route, an "
                 "endpoint or a policy setting does."),
    }
    out.write_text(json.dumps(payload, indent=1, sort_keys=True))
    return out


# ---------------------------------------------------------------------------
# offline verification
# ---------------------------------------------------------------------------
def _plan_digest_for(record: dict) -> str:
    return (supplement.plan_digest() if record.get("role") == "supplement"
            else heldout.plan_digest())


def _current_experiment_digests(names) -> dict[str, str]:
    root = Path(__file__).resolve().parent
    return {name: (heldout.digest(root / name) if (root / name).exists() else None)
            for name in names}


def _current_product_digests() -> dict[str, str]:
    root = Path(__file__).resolve().parent.parent / supplement.PRODUCT_PACKAGE
    return {f"{supplement.PRODUCT_PACKAGE}/{path.name}": heldout.digest(path)
            for path in sorted(root.glob("*.py"))}


def verify_offline(run_dir: str | Path) -> dict:
    """Re-derive a run's public identity. Never opens a runtime config."""
    run_dir = Path(run_dir)
    checks: list[tuple[bool, str, str]] = []
    limits: list[str] = []

    def check(name: str, ok: bool, detail: str = "") -> bool:
        checks.append((bool(ok), name, detail))
        return bool(ok)

    path = run_dir / "preregistration.json"
    try:
        record = json.loads(path.read_text())
    except Exception as exc:                                   # noqa: BLE001
        check("the registration parses", False, f"{type(exc).__name__}: {exc}")
        return {"ok": False, "checks": checks, "limits": limits, "record": {}, "dir": run_dir}
    check("the registration parses", True,
          f"{record.get('role', 'original')}, registered {record.get('registered_at')}")

    # --- the task file --------------------------------------------------
    task_file = run_dir / record.get("task_file", "tasks.jsonl")
    actual = heldout.digest(task_file) if task_file.exists() else None
    check("the task file matches its registered digest",
          actual is not None and actual == record.get("task_file_sha256"),
          f"registered {str(record.get('task_file_sha256'))[:16]}, actual {str(actual)[:16]}")

    lines = [json.loads(line) for line in task_file.read_text().splitlines()
             if line.strip().startswith("{")] if task_file.exists() else []
    ids = [task.get("id") for task in lines]
    counts: dict[str, int] = {}
    for task in lines:
        counts[task.get("category")] = counts.get(task.get("category"), 0) + 1
    consistent = (ids == record.get("task_ids", ids)
                  and len(lines) == record.get("task_count", len(lines))
                  and counts == (record.get("tasks_by_category") or counts))
    check("the registered task ids and counts match the task file", consistent,
          f"{len(lines)} tasks, {len(counts)} categories")

    # --- the analysis plan ----------------------------------------------
    # Two separate questions, because `amend` refreshes the digest and leaves
    # the human-readable plan in the file at the version first registered. That
    # is defensible - the original wording is the thing a pre-registration is
    # for - but it makes the record self-inconsistent unless the amendment
    # trail accounts for the text that is actually sitting there.
    expected_plan = record.get("analysis_plan_sha256")
    check("the analysis plan matches this checkout",
          expected_plan == _plan_digest_for(record),
          f"registered {str(expected_plan)[:16]}, checkout {_plan_digest_for(record)[:16]}")

    embedded = record.get("analysis_plan")
    if embedded is not None:
        embedded_sha = _sha256_of(embedded)
        superseded = {a.get("previous_analysis_plan_sha256")
                      for a in (record.get("amendments") or [])}
        where = ("the registered digest" if embedded_sha == expected_plan
                 else "a digest the amendment trail records as superseded"
                 if embedded_sha in superseded else "NOTHING in this registration")
        check("the analysis plan text in the registration is accounted for",
              embedded_sha == expected_plan or embedded_sha in superseded,
              f"text hashes to {embedded_sha[:16]}, which is {where}")
        if embedded_sha != expected_plan and embedded_sha in superseded:
            limits.append(
                "The human-readable analysis plan in this registration is the version first "
                f"registered ({embedded_sha[:16]}); the digest the runner enforces is the "
                f"amended one ({str(expected_plan)[:16]}). Both are on the record and the "
                "amendment trail links them, but the text and the digest in this file are "
                "not the same plan. Read the amendment reasons before reading the plan.")

    # --- the frozen policy/catalog identity -----------------------------
    identity = record.get("policy_identity")
    if identity is None:
        limits.append("This registration froze no policy/catalog identity, so there is "
                      "nothing here to re-derive about the routes it was measured against.")
    else:
        body = {k: v for k, v in identity.items() if k != "sha256"}
        recomputed = _sha256_of(body)
        check("the frozen policy/catalog identity is internally consistent",
              recomputed == identity.get("sha256") == record.get("policy_identity_sha256"),
              f"recomputed {recomputed[:16]}, "
              f"registered {str(record.get('policy_identity_sha256'))[:16]}, "
              f"{len(identity.get('catalog') or [])} routes, "
              f"policy {identity.get('policy_name')}")

    # --- the code -------------------------------------------------------
    registered_code = record.get("code_sha256") or {}
    current_code = _current_experiment_digests(registered_code)
    bad = sorted(name for name, expected in registered_code.items()
                 if current_code.get(name) != expected)
    check("the registered experiment code matches this checkout", not bad,
          f"{len(registered_code)} files" + (f"; mismatched: {', '.join(bad)}" if bad else ""))

    registered_product = record.get("product_sha256") or {}
    if not registered_product:
        limits.append("This registration froze no product-code digests, so the routing code "
                      "the run used is not pinned by it.")
    else:
        current_product = _current_product_digests()
        bad = sorted(
            [name for name, expected in registered_product.items()
             if current_product.get(name) != expected]
            + [f"{name} (present in the checkout, not registered)"
               for name in set(current_product) - set(registered_product)])
        check("the registered product code matches this checkout", not bad,
              f"{len(registered_product)} files"
              + (f"; mismatched: {', '.join(bad)}" if bad else ""))

    # --- the credential-free configuration identity, if the run froze one -
    frozen = run_dir / PUBLIC_CONFIG_FILE
    if not frozen.exists():
        limits.append(
            "This run froze no frozen public config identity, so nothing offline binds it to "
            "the configuration it was measured against. A future run should freeze one "
            "(evidence_verify.py --freeze-public-config) at registration time.")
    else:
        payload = json.loads(frozen.read_text())
        body = payload.get("public_identity") or {}
        recomputed = _sha256_of({k: v for k, v in body.items() if k != "sha256"})
        check("the frozen public config identity is internally consistent and matches "
              "the registration",
              recomputed == body.get("sha256")
              and payload.get("config_sha256") == record.get("config_sha256"),
              f"recomputed {recomputed[:16]}, frozen {str(body.get('sha256'))[:16]}")
        limits.append(
            "The frozen public config identity is a credential-free projection. It shows the "
            "routes, endpoints and policy settings the run was configured with; it is still "
            "not proof that the secret-bearing file itself existed or hashed as recorded.")

    return {"ok": all(ok for ok, _, _ in checks), "checks": checks, "limits": limits,
            "record": record, "dir": run_dir}


GUARANTEE = """\
LIMITED GUARANTEE — what a passing run above does and does not establish.

  Established, by anyone, from this checkout alone:
    the task file, the registered task ids and counts, the analysis plan, the
    frozen policy/catalog identity, and every registered experiment- and
    product-code digest are exactly what the registration says they are.

  Not verified here, and not verifiable here:
    that the runtime config recorded as
      {path}
      sha256 {digest}
    ever existed, that it hashed to that digest, that the frozen policy/catalog
    identity was derived from it, or that the run was executed against it. That
    digest is recorded, not verified. The file carries provider credentials, is
    not published, and cannot be reconstructed from this registration.

    Offline verification of a frozen redacted identity is not proof that the
    original secret-bearing runtime config existed.

  Only `supplement.py verify --dir <run> --config <that exact file>`, run by
  whoever holds it, establishes the configuration half - and it refuses when the
  file is missing or has drifted. A reader who does not hold it cannot run it,
  and no claim of independent verification should be made on its behalf."""


def format_offline(result: dict) -> str:
    lines = [f"offline evidence verification — {result['dir']}", ""]
    for ok, name, detail in result["checks"]:
        lines.append(f"[{'ok  ' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))
    failed = [name for ok, name, _ in result["checks"] if not ok]
    lines += ["", f"RESULT: {'PASS' if result['ok'] else 'FAIL'} — "
                  f"{len(result['checks']) - len(failed)}/{len(result['checks'])} checks", ""]
    for limit in result["limits"]:
        lines.append(f"  note: {limit}")
    if result["limits"]:
        lines.append("")
    record = result.get("record") or {}
    lines.append(GUARANTEE.format(path=record.get("config_path", "<none recorded>"),
                                  digest=record.get("config_sha256", "<none recorded>")))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dir", type=Path, required=True, help="a finished run directory")
    parser.add_argument("--freeze-public-config", type=Path, metavar="CONFIG",
                        help="write this run's credential-free configuration identity; "
                             "for a run being registered now, not for a finished one")
    args = parser.parse_args(argv)
    if args.freeze_public_config:
        out = freeze_public_config(args.dir, args.freeze_public_config)
        print(f"wrote {out}")
        return 0
    result = verify_offline(args.dir)
    print(format_offline(result))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
