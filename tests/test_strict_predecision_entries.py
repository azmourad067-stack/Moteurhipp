import pandas as pd
import pytest

from horse_intelligence.intelligence.feature_engineering import (
    _entries_at_decision,
    _market,
)


DECISION = "2026-11-13T12:00:00Z"
START = "2026-11-13T12:10:00Z"


def _races(*race_ids):
    return pd.DataFrame({
        "race_id": race_ids,
        "decision_time": [DECISION] * len(race_ids),
        "scheduled_start_utc": [START] * len(race_ids),
    })


def _entries(rows):
    return pd.DataFrame(
        rows,
        columns=["race_id", "num_pmu", "observed_at", "status", "value"],
    )


def test_keeps_latest_entry_snapshot_at_or_before_decision():
    entries = _entries([
        (1, 7, "2026-11-13T11:00:00Z", "PARTANT", "old"),
        (1, 7, DECISION, "NON_PARTANT", "at-decision"),
        (1, 7, "2026-11-13T12:01:00Z", "PARTANT", "future"),
        (1, 8, "2026-11-13T11:30:00Z", "PARTANT", "other"),
    ])

    result = _entries_at_decision(entries, _races(1))

    assert result.set_index("num_pmu")["value"].to_dict() == {
        7: "at-decision",
        8: "other",
    }
    assert result["entry_seen_before_decision"].all()


def test_excludes_whole_race_if_one_entry_has_only_future_observations():
    entries = _entries([
        (1, 7, "2026-11-13T11:00:00Z", "PARTANT", "known"),
        (1, 8, "2026-11-13T12:01:00Z", "PARTANT", "future-only"),
        (2, 9, "2026-11-13T11:45:00Z", "PARTANT", "other-race"),
    ])

    result = _entries_at_decision(entries, _races(1, 2))

    assert result[["race_id", "num_pmu"]].to_records(index=False).tolist() == [(2, 9)]


def test_missing_observation_time_does_not_trigger_future_fallback():
    entries = _entries([(1, 7, None, "PARTANT", "unknown-time")])

    assert _entries_at_decision(entries, _races(1)).empty


def test_empty_entries_preserve_columns_needed_by_build_dataset():
    result = _entries_at_decision(_entries([]), _races(1))

    assert "decision_time" in result.columns
    assert "entry_seen_before_decision" in result.columns


def test_rejects_duplicate_decision_times_for_a_race():
    entries = _entries([(1, 7, "2026-11-13T11:00:00Z", "PARTANT", "known")])

    with pytest.raises(ValueError, match="Plusieurs decision_time"):
        _entries_at_decision(entries, _races(1, 1))


def test_market_rejects_odds_observed_after_decision_even_if_published_early():
    runners = pd.DataFrame({"race_id": [1, 1], "num_pmu": [7, 8]})
    odds = pd.DataFrame([
        {
            "race_id": 1, "num_pmu": 7, "pool": "INTERNET", "odds": 2.0,
            "published_at": "2026-11-13T11:50:00Z",
            "observed_at": "2026-11-13T11:51:00Z",
        },
        {
            "race_id": 1, "num_pmu": 8, "pool": "INTERNET", "odds": 3.0,
            "published_at": "2026-11-13T11:50:00Z",
            "observed_at": "2026-11-13T12:01:00Z",
        },
    ])
    odds[["published_at", "observed_at"]] = odds[
        ["published_at", "observed_at"]
    ].apply(pd.to_datetime, utc=True)

    assert _market(odds, _races(1), runners).empty


def test_market_accepts_complete_odds_observed_by_decision():
    runners = pd.DataFrame({"race_id": [1, 1], "num_pmu": [7, 8]})
    odds = pd.DataFrame([
        {
            "race_id": 1, "num_pmu": 7, "pool": "INTERNET", "odds": 2.0,
            "published_at": "2026-11-13T11:50:00Z",
            "observed_at": "2026-11-13T11:51:00Z",
        },
        {
            "race_id": 1, "num_pmu": 8, "pool": "INTERNET", "odds": 3.0,
            "published_at": "2026-11-13T11:50:00Z",
            "observed_at": DECISION,
        },
    ])
    odds[["published_at", "observed_at"]] = odds[
        ["published_at", "observed_at"]
    ].apply(pd.to_datetime, utc=True)

    result = _market(odds, _races(1), runners)

    assert len(result) == 2
    assert result["mkt_prob"].sum() == pytest.approx(1.0)
