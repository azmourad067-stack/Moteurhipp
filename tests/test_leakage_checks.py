import pandas as pd

from horse_intelligence.evaluation.leakage_checks import _entry_predecision_counts


def test_entry_coverage_counts_only_snapshots_available_by_decision():
    raw = {
        "races": pd.DataFrame({
            "race_id": [1, 2, 3],
            "scheduled_start_utc": [
                "2026-11-13T12:10:00Z",
                "2026-11-13T13:10:00Z",
                None,  # no decision time: outside build_dataset's eligible races
            ],
        }),
        "entries": pd.DataFrame({
            "race_id": [1, 1, 1, 2, 2, 3],
            "num_pmu": [7, 7, 8, 9, 10, 11],
            "observed_at": [
                "2026-11-13T11:30:00Z",  # horse 7 has a valid earlier snapshot
                "2026-11-13T12:05:00Z",  # the later snapshot is after decision
                "2026-11-13T12:05:00Z",  # after decision, although before start
                "2026-11-13T13:11:00Z",  # after start
                None,  # unknown timestamp cannot count as pre-decision
                None,  # no decision time, so it is not in the coverage denominator
            ],
        }),
    }

    assert _entry_predecision_counts(raw) == (3, 4)


def test_entry_coverage_handles_no_identifiable_entries():
    raw = {
        "races": pd.DataFrame({
            "race_id": [1],
            "scheduled_start_utc": ["2026-11-13T12:10:00Z"],
        }),
        "entries": pd.DataFrame(columns=["race_id", "num_pmu", "observed_at"]),
    }

    assert _entry_predecision_counts(raw) == (0, 0)
