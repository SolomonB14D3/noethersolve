"""Tests for the transaction isolation anomaly checker."""

import pytest
from noethersolve.isolation import (
    check_isolation, analyze_schedule, list_anomalies,
    ANOMALIES, ISOLATION_LEVELS,
)


class TestCheckIsolation:
    def test_read_uncommitted_prevents_nothing(self):
        r = check_isolation("READ_UNCOMMITTED")
        assert len(r.prevented_anomalies) == 0
        assert len(r.possible_anomalies) == len(ANOMALIES)

    def test_read_committed_prevents_dirty_read(self):
        r = check_isolation("READ_COMMITTED")
        assert "dirty_read" in r.prevented_anomalies
        assert "non_repeatable_read" in r.possible_anomalies
        assert "phantom_read" in r.possible_anomalies

    def test_repeatable_read_prevents_fuzzy_read(self):
        r = check_isolation("REPEATABLE_READ")
        assert "dirty_read" in r.prevented_anomalies
        assert "non_repeatable_read" in r.prevented_anomalies
        assert "phantom_read" in r.possible_anomalies  # common misconception!

    def test_snapshot_prevents_phantoms(self):
        r = check_isolation("SNAPSHOT")
        assert "phantom_read" in r.prevented_anomalies
        assert "write_skew" in r.possible_anomalies  # snapshot doesn't prevent this

    def test_serializable_prevents_everything(self):
        r = check_isolation("SERIALIZABLE")
        assert len(r.possible_anomalies) == 0
        assert len(r.prevented_anomalies) == len(ANOMALIES)

    def test_specific_anomaly_query(self):
        r = check_isolation("READ_COMMITTED", anomaly="phantom_read")
        assert r.queried_anomaly == "phantom_read"
        s = str(r)
        assert "phantom_read" in s

    def test_fuzzy_level_matching(self):
        r = check_isolation("read committed")
        assert r.isolation_level == "READ_COMMITTED"

    def test_unknown_level_raises(self):
        with pytest.raises(ValueError):
            check_isolation("TOTALLY_FAKE_LEVEL")

    def test_misconceptions_included(self):
        r = check_isolation("REPEATABLE_READ")
        assert len(r.misconceptions) > 0
        # Should warn about phantom reads
        assert any("phantom" in m.lower() for m in r.misconceptions)

    def test_report_string(self):
        r = check_isolation("READ_COMMITTED")
        s = str(r)
        assert "Isolation Level Analysis" in s
        assert "READ_COMMITTED" in s
        assert "Prevented" in s
        assert "STILL POSSIBLE" in s


class TestAnalyzeSchedule:
    def test_read_write_conflict(self):
        """T1 reads x, T2 writes x — RW conflict."""
        r = analyze_schedule(
            transactions=[[("R", "x")], [("W", "x")]],
            isolation="READ_COMMITTED",
        )
        assert len(r.conflicts) > 0

    def test_lost_update_pattern(self):
        """T1: R(x), W(x) and T2: R(x), W(x) — lost update."""
        r = analyze_schedule(
            transactions=[
                [("R", "x"), ("W", "x")],
                [("R", "x"), ("W", "x")],
            ],
            isolation="READ_COMMITTED",
        )
        assert any("lost update" in a.lower() for a in r.possible_anomalies)

    def test_write_skew_pattern(self):
        """T1: R(x), W(y) and T2: R(y), W(x) — write skew."""
        r = analyze_schedule(
            transactions=[
                [("R", "x"), ("W", "y")],
                [("R", "y"), ("W", "x")],
            ],
            isolation="SNAPSHOT",
        )
        assert any("write skew" in a.lower() for a in r.possible_anomalies)

    def test_serializable_no_anomalies(self):
        """Under SERIALIZABLE, no anomalies should be reported."""
        r = analyze_schedule(
            transactions=[
                [("R", "x"), ("W", "x")],
                [("R", "x"), ("W", "x")],
            ],
            isolation="SERIALIZABLE",
        )
        assert len(r.possible_anomalies) == 0

    def test_no_conflict_disjoint_transactions(self):
        """Transactions on different items have no conflicts."""
        r = analyze_schedule(
            transactions=[
                [("R", "x"), ("W", "x")],
                [("R", "y"), ("W", "y")],
            ],
            isolation="READ_COMMITTED",
        )
        assert r.serializable  # no cycle possible

    def test_report_string(self):
        r = analyze_schedule(
            transactions=[[("R", "x")], [("W", "x")]],
        )
        s = str(r)
        assert "Schedule Analysis" in s


class TestListAnomalies:
    def test_returns_all_anomalies(self):
        anomalies = list_anomalies()
        assert len(anomalies) == len(ANOMALIES)

    def test_descriptions_present(self):
        anomalies = list_anomalies()
        for a in anomalies:
            assert len(a) > 10  # not empty


class TestIsolationLevelCompleteness:
    def test_all_levels_are_subsets(self):
        """Each higher isolation level prevents at least what the lower one does."""
        levels = ["READ_UNCOMMITTED", "READ_COMMITTED", "REPEATABLE_READ", "SERIALIZABLE"]
        for i in range(len(levels) - 1):
            lower = ISOLATION_LEVELS[levels[i]]
            higher = ISOLATION_LEVELS[levels[i + 1]]
            assert lower.issubset(higher), (
                f"{levels[i+1]} should prevent everything {levels[i]} prevents"
            )

    def test_snapshot_between_rr_and_serializable(self):
        """Snapshot prevents more than RR but less than SERIALIZABLE."""
        rr = ISOLATION_LEVELS["REPEATABLE_READ"]
        snap = ISOLATION_LEVELS["SNAPSHOT"]
        ser = ISOLATION_LEVELS["SERIALIZABLE"]
        assert rr.issubset(snap)
        assert snap.issubset(ser)
        assert snap != ser  # snapshot doesn't prevent write_skew
