"""Offline verification of a frozen registration, and the line it must not cross.

A pre-registration freezes two different kinds of thing:

* a **public identity** — the task file, the analysis plan, the experiment and
  product code, and the resolved policy/catalog the run was measured against.
  All of that is either in the registration or in the published checkout, so
  anyone can re-derive it.
* a **configuration identity** — a path and a digest of the runtime config.
  That file carries provider credentials. It is not published, it cannot be
  reconstructed from the registration, and so a third party cannot check it.

The offline verifier exists for the first kind and must be loud about not
covering the second. These tests pin both halves: it fails on any change to the
frozen public identity, and it never reports the configuration as verified —
not even when the config file happens to be sitting next to it.

The strict *live-run* verifier is a separate thing and keeps refusing when the
registered config is missing or drifted; that is pinned here too, because the
temptation when an offline path exists is to soften the strict one.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments import evidence_verify, supplement                 # noqa: E402

FAKE_KEY = "sk-" + "z" * 40


def _write_config(tmp_path: Path, *, name: str = "config.yaml") -> Path:
    """A config in the shape of the real one: credentials and all."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / name
    path.write_text(
        "providers:\n"
        "  p:\n"
        "    base_url: https://example.invalid/v1\n"
        f"    api_key: {FAKE_KEY}\n"
        "    cache: generic\n"
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
        "    output: 2.0\n"
        "policy:\n"
        "  escalate_after_tool_errors: 3\n")
    return path


def _registered(tmp_path: Path, *, keep_config: bool = False) -> Path:
    """A registered run whose config has been removed again - the published case."""
    config = _write_config(tmp_path)
    out = tmp_path / "supp"
    supplement.preregister(out, config)
    if not keep_config:
        config.unlink()
    return out


def _rewrite(out: Path, mutate) -> None:
    record = json.loads((out / "preregistration.json").read_text())
    mutate(record)
    (out / "preregistration.json").write_text(json.dumps(record, indent=1))


# ---------------------------------------------------------------------------
# what it verifies
# ---------------------------------------------------------------------------
def test_the_public_identity_verifies_with_no_config_anywhere(tmp_path):
    out = _registered(tmp_path)
    result = evidence_verify.verify_offline(out)
    assert result["ok"] is True, result["checks"]
    assert [name for ok, name, _ in result["checks"] if not ok] == []


def test_it_checks_the_task_file_the_plan_the_identity_and_both_code_sets(tmp_path):
    result = evidence_verify.verify_offline(_registered(tmp_path))
    names = " | ".join(name for _, name, _ in result["checks"])
    for expected in ("task file", "analysis plan", "policy/catalog identity",
                     "experiment code", "product code"):
        assert expected in names, (expected, names)


def test_a_tampered_task_file_fails(tmp_path):
    out = _registered(tmp_path)
    (out / "tasks.jsonl").write_text('{"id": "tampered"}\n')
    result = evidence_verify.verify_offline(out)
    assert result["ok"] is False
    assert any("task file" in name for ok, name, _ in result["checks"] if not ok)


def test_a_task_id_list_that_no_longer_matches_the_task_file_fails(tmp_path):
    out = _registered(tmp_path)
    _rewrite(out, lambda r: r["task_ids"].append("smuggled-in-later"))
    result = evidence_verify.verify_offline(out)
    assert result["ok"] is False


def test_an_edited_frozen_policy_identity_fails(tmp_path):
    """The identity carries its own digest, so editing the catalog is detectable."""
    out = _registered(tmp_path)
    _rewrite(out, lambda r: r["policy_identity"]["catalog"][0].update({"free": False}))
    result = evidence_verify.verify_offline(out)
    assert result["ok"] is False
    assert any("policy/catalog identity" in name
               for ok, name, _ in result["checks"] if not ok)


def test_an_edited_identity_digest_fails_too(tmp_path):
    out = _registered(tmp_path)
    _rewrite(out, lambda r: r.update({"policy_identity_sha256": "0" * 64}))
    assert evidence_verify.verify_offline(out)["ok"] is False


def test_a_changed_analysis_plan_digest_fails(tmp_path):
    out = _registered(tmp_path)
    _rewrite(out, lambda r: r.update({"analysis_plan_sha256": "0" * 64}))
    result = evidence_verify.verify_offline(out)
    assert result["ok"] is False
    assert any("analysis plan" in name for ok, name, _ in result["checks"] if not ok)


def test_an_edited_analysis_plan_text_that_nothing_accounts_for_fails(tmp_path):
    """`amend` refreshes the digest and leaves the plan text alone, so the text
    has to be checked against the whole trail rather than only the current digest."""
    out = _registered(tmp_path)
    _rewrite(out, lambda r: r["analysis_plan"].update({"question": "something else entirely"}))
    result = evidence_verify.verify_offline(out)
    assert result["ok"] is False
    assert any("accounted for" in name for ok, name, _ in result["checks"] if not ok)


def test_plan_text_left_behind_by_an_amendment_passes_and_is_called_out(tmp_path):
    out = _registered(tmp_path)
    superseded = json.loads((out / "preregistration.json").read_text())["analysis_plan_sha256"]

    def stale(record):
        record["analysis_plan_sha256"] = "1" * 64
        record["amendments"] = [{"at": "2026-09-18T11:24:42Z", "reason": "wording",
                                 "previous_analysis_plan_sha256": superseded}]

    _rewrite(out, stale)
    result = evidence_verify.verify_offline(out)
    accounted = [ok for ok, name, _ in result["checks"] if "accounted for" in name]
    assert accounted == [True]
    assert any("not the same plan" in limit for limit in result["limits"])


def test_experiment_code_that_no_longer_matches_the_checkout_fails(tmp_path):
    out = _registered(tmp_path)
    _rewrite(out, lambda r: r["code_sha256"].update({"graders.py": "0" * 64}))
    result = evidence_verify.verify_offline(out)
    assert result["ok"] is False
    assert any("graders.py" in detail for ok, _, detail in result["checks"] if not ok)


def test_product_code_that_no_longer_matches_the_checkout_fails(tmp_path):
    out = _registered(tmp_path)
    _rewrite(out, lambda r: r["product_sha256"].update({"auto_router/router.py": "0" * 64}))
    result = evidence_verify.verify_offline(out)
    assert result["ok"] is False
    assert any("auto_router/router.py" in detail
               for ok, _, detail in result["checks"] if not ok)


def test_a_product_file_added_after_registration_fails(tmp_path):
    out = _registered(tmp_path)
    _rewrite(out, lambda r: r["product_sha256"].pop("auto_router/router.py"))
    result = evidence_verify.verify_offline(out)
    assert result["ok"] is False


# ---------------------------------------------------------------------------
# what it must refuse to claim
# ---------------------------------------------------------------------------
def test_the_limited_guarantee_is_always_emitted(tmp_path):
    for out in (_registered(tmp_path / "a"), _registered(tmp_path / "b")):
        text = evidence_verify.format_offline(evidence_verify.verify_offline(out))
        assert "LIMITED GUARANTEE" in text
        assert "not verified" in text.lower()


def test_the_guarantee_names_the_config_digest_as_recorded_but_unverified(tmp_path):
    out = _registered(tmp_path)
    record = json.loads((out / "preregistration.json").read_text())
    text = evidence_verify.format_offline(evidence_verify.verify_offline(out))
    assert record["config_sha256"] in text
    assert "recorded, not verified" in text


def test_it_never_reports_the_configuration_as_verified_even_when_present(tmp_path):
    """The config sitting next to the registration proves nothing to a reader
    who cannot see it. Passing offline must mean the same thing on every machine."""
    with_config = _registered(tmp_path / "with", keep_config=True)
    without = _registered(tmp_path / "without")
    a, b = (evidence_verify.verify_offline(with_config),
            evidence_verify.verify_offline(without))
    assert [n for _, n, _ in a["checks"]] == [n for _, n, _ in b["checks"]]
    assert not any("config" in name.lower() and ok for ok, name, _ in a["checks"])


def test_a_failure_still_prints_the_guarantee_and_does_not_read_as_a_pass(tmp_path):
    out = _registered(tmp_path)
    (out / "tasks.jsonl").write_text("tampered\n")
    text = evidence_verify.format_offline(evidence_verify.verify_offline(out))
    assert "LIMITED GUARANTEE" in text
    assert "FAIL" in text


def test_the_guarantee_says_a_frozen_identity_is_not_proof_the_config_existed(tmp_path):
    text = evidence_verify.format_offline(
        evidence_verify.verify_offline(_registered(tmp_path)))
    lowered = text.lower()
    assert "is not proof" in lowered
    assert "credential" in lowered


# ---------------------------------------------------------------------------
# the secret-free config identity a future run should freeze
# ---------------------------------------------------------------------------
def test_a_public_config_identity_keeps_no_credential_value(tmp_path):
    identity = evidence_verify.public_config_identity(_write_config(tmp_path))
    assert FAKE_KEY not in json.dumps(identity)
    assert "zzzz" not in json.dumps(identity)


def test_a_public_config_identity_records_that_a_credential_was_there(tmp_path):
    """Dropping the value silently would make an unconfigured provider look the
    same as a configured one."""
    identity = evidence_verify.public_config_identity(_write_config(tmp_path))
    assert identity["providers"]["p"]["credential_keys"] == ["api_key"]
    assert identity["providers"]["p"]["base_url"] == "https://example.invalid/v1"


def test_a_public_config_identity_is_stable_and_carries_its_own_digest(tmp_path):
    path = _write_config(tmp_path)
    first = evidence_verify.public_config_identity(path)
    second = evidence_verify.public_config_identity(path)
    assert first == second
    assert len(first["sha256"]) == 64


def test_rotating_only_the_credential_does_not_change_the_public_digest(tmp_path):
    """That is the point: it can be published and it still identifies the config."""
    path = _write_config(tmp_path)
    before = evidence_verify.public_config_identity(path)["sha256"]
    path.write_text(path.read_text().replace(FAKE_KEY, "sk-" + "q" * 40))
    assert evidence_verify.public_config_identity(path)["sha256"] == before


def test_changing_a_route_does_change_the_public_digest(tmp_path):
    path = _write_config(tmp_path)
    before = evidence_verify.public_config_identity(path)["sha256"]
    path.write_text(path.read_text().replace("vendor/free-one", "vendor/other"))
    assert evidence_verify.public_config_identity(path)["sha256"] != before


def test_a_frozen_public_config_identity_is_verified_when_the_run_has_one(tmp_path):
    out = _registered(tmp_path, keep_config=True)
    evidence_verify.freeze_public_config(out, tmp_path / "config.yaml")
    result = evidence_verify.verify_offline(out)
    assert result["ok"] is True
    assert any("public config identity" in name and ok
               for ok, name, _ in result["checks"])


def test_a_tampered_frozen_public_config_identity_fails(tmp_path):
    out = _registered(tmp_path, keep_config=True)
    path = evidence_verify.freeze_public_config(out, tmp_path / "config.yaml")
    frozen = json.loads(path.read_text())
    frozen["public_identity"]["providers"]["p"]["base_url"] = "https://elsewhere.invalid/v1"
    path.write_text(json.dumps(frozen))
    assert evidence_verify.verify_offline(out)["ok"] is False


def test_freezing_refuses_to_overwrite_an_existing_artifact(tmp_path):
    """Rewriting it later is exactly the thing freezing is supposed to prevent."""
    out = _registered(tmp_path, keep_config=True)
    evidence_verify.freeze_public_config(out, tmp_path / "config.yaml")
    with pytest.raises(SystemExit, match="already"):
        evidence_verify.freeze_public_config(out, tmp_path / "config.yaml")


def test_freezing_refuses_a_config_that_is_not_the_one_the_run_registered(tmp_path):
    out = _registered(tmp_path, keep_config=True)
    other = _write_config(tmp_path / "other")
    other.write_text(other.read_text().replace("vendor/free-one", "vendor/someone-else"))
    with pytest.raises(SystemExit, match="not the config this run registered"):
        evidence_verify.freeze_public_config(out, other)


def test_freezing_refuses_a_directory_with_no_registration(tmp_path):
    config = _write_config(tmp_path)
    (tmp_path / "empty").mkdir()
    with pytest.raises(SystemExit, match="no registration"):
        evidence_verify.freeze_public_config(tmp_path / "empty", config)


def test_redaction_errs_towards_dropping_rather_than_keeping(tmp_path):
    """`credential_keys` is itself credential-shaped, so a config that happens to
    use that name has it dropped too. Over-redacting a public artifact is the
    safe direction; the key name is still listed, so nothing vanishes silently."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "odd.yaml"
    path.write_text("providers:\n  p:\n    credential_keys: [written-by-hand]\n"
                    f"    api_key: {FAKE_KEY}\n    base_url: https://kept.invalid/v1\n")
    provider = evidence_verify.public_config_identity(path)["providers"]["p"]
    assert provider["credential_keys"] == ["api_key", "credential_keys"]
    assert provider["base_url"] == "https://kept.invalid/v1"
    assert "written-by-hand" not in json.dumps(provider)


def test_a_run_without_one_says_so_instead_of_passing_quietly(tmp_path):
    result = evidence_verify.verify_offline(_registered(tmp_path))
    assert result["ok"] is True
    assert any("no frozen public config identity" in limit.lower()
               for limit in result["limits"])


# ---------------------------------------------------------------------------
# the CLI
# ---------------------------------------------------------------------------
def test_the_cli_exits_zero_on_an_intact_run_and_one_on_a_tampered_one(tmp_path, capsys):
    out = _registered(tmp_path)
    assert evidence_verify.main(["--dir", str(out)]) == 0
    assert "LIMITED GUARANTEE" in capsys.readouterr().out
    (out / "tasks.jsonl").write_text("tampered\n")
    assert evidence_verify.main(["--dir", str(out)]) == 1


def test_the_cli_never_takes_a_config_argument(tmp_path):
    """An offline verifier that accepts a config invites being run with one and
    reported as if the offline guarantee were the stronger one."""
    with pytest.raises(SystemExit):
        evidence_verify.main(["--dir", str(tmp_path), "--config", "x.yaml"])


# ---------------------------------------------------------------------------
# the strict live verifier is not softened by any of the above
# ---------------------------------------------------------------------------
def test_the_live_verifier_still_refuses_when_the_registered_config_is_absent(tmp_path):
    out = _registered(tmp_path)
    with pytest.raises(SystemExit, match="missing"):
        supplement.load_preregistration(out)


def test_the_live_verifier_still_refuses_a_config_path_that_does_not_exist(tmp_path):
    out = _registered(tmp_path)
    with pytest.raises(SystemExit, match="refusing to run"):
        supplement.load_preregistration(out, config_path=tmp_path / "nowhere.yaml")


def test_the_live_verifier_still_refuses_a_drifted_config(tmp_path):
    config = _write_config(tmp_path)
    out = tmp_path / "supp"
    supplement.preregister(out, config)
    config.write_text(config.read_text() + "  jev_difficulty_calibration: [0.1, 0.2]\n")
    with pytest.raises(SystemExit, match="config file"):
        supplement.load_preregistration(out, config_path=config)


def test_the_offline_verifier_is_not_a_substitute_for_the_live_one(tmp_path):
    """Both run against the same directory; only one of them can see the config."""
    out = _registered(tmp_path)
    assert evidence_verify.verify_offline(out)["ok"] is True
    with pytest.raises(SystemExit):
        supplement.load_preregistration(out)
