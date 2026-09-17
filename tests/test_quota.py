import json

from auto_router.quota import PacingRule, QuotaState, decide, from_budget_file, from_codex_rollouts

WEEK = 7 * 24 * 3600
NOW = 1_800_000_000.0


def state(used, hours_left, session=None):
    return QuotaState(week_used=used, week_resets_at=NOW + hours_left * 3600, session_used=session,
                      measured_at=NOW)


def test_plenty_of_slack_is_free():
    d = decide(state(0.10, 84), now=NOW)       # half-way, 10 % used -> projected 20 %
    assert d.open and d.multiplier == 0.0


def test_projection_above_reserve_closes():
    d = decide(state(0.40, 84), now=NOW)       # projected 80 % > 65 %
    assert not d.open


def test_shadow_price_ramps_near_reserve():
    d = decide(state(0.28, 84), now=NOW)       # projected 56 %: 9 points of slack
    assert d.open and 0 < d.multiplier < 1


def test_hard_stop_and_session_window():
    assert not decide(state(0.81, 10), now=NOW).open
    assert not decide(state(0.05, 100, session=0.75), now=NOW).open


def test_late_week_use_it_or_lose_it():
    # 70 % used, resets in 20 h: projected ~78 % < late reserve 85 %
    d = decide(state(0.70, 20), now=NOW)
    assert d.open


def test_missing_or_stale_measurement_closes():
    assert not decide(None, now=NOW).open
    old = QuotaState(0.1, NOW + WEEK / 2, measured_at=NOW - 10 * 3600)
    assert not decide(old, now=NOW).open


def test_budget_file_reader(tmp_path):
    p = tmp_path / "budget.json"
    p.write_text(json.dumps({"generated_at": "2026-09-17 12:07",
                             "claude": {"week_percent": 68.0, "session_percent": 3.0,
                                        "week_resets_at": "2026-09-19T04:59:59+00:00"}}))
    s = from_budget_file(p, "claude")
    assert s.week_used == 0.68 and s.session_used == 0.03 and s.week_resets_at


def test_codex_rollout_reader(tmp_path):
    d = tmp_path / "2026" / "09"
    d.mkdir(parents=True)
    (d / "rollout-x.jsonl").write_text("\n".join([
        json.dumps({"timestamp": "2026-09-17T10:00:00Z", "payload": {"type": "token_count",
                    "rate_limits": {"primary": {"used_percent": 40.0, "resets_at": 1789805411}}}}),
        json.dumps({"timestamp": "2026-09-17T11:00:00Z", "payload": {"type": "token_count",
                    "rate_limits": {"primary": {"used_percent": 42.0, "resets_at": 1789805411}}}}),
    ]))
    s = from_codex_rollouts(tmp_path)
    assert s.week_used == 0.42
