"""Tests for the distributed systems calculator module."""

import pytest
from noethersolve.distributed_calc import (
    quorum_calc,
    byzantine_threshold,
    vector_clock_compare,
    consistency_model,
    gossip_convergence,
)


class TestQuorumCalc:
    def test_majority_5_nodes(self):
        """Majority quorum of 5: R=3, W=3, strong consistency."""
        r = quorum_calc(total_nodes=5, strategy="majority")
        assert r.read_quorum == 3
        assert r.write_quorum == 3
        assert r.strong_consistency is True  # 3+3=6 > 5

    def test_majority_3_nodes(self):
        r = quorum_calc(total_nodes=3, strategy="majority")
        assert r.read_quorum == 2
        assert r.write_quorum == 2
        assert r.strong_consistency is True  # 2+2=4 > 3

    def test_read_heavy(self):
        """Read-heavy: R=1, W=N."""
        r = quorum_calc(total_nodes=5, strategy="read_heavy")
        assert r.read_quorum == 1
        assert r.write_quorum == 5
        assert r.strong_consistency is True  # 1+5=6 > 5

    def test_write_heavy(self):
        """Write-heavy: R=N, W=1."""
        r = quorum_calc(total_nodes=5, strategy="write_heavy")
        assert r.read_quorum == 5
        assert r.write_quorum == 1
        assert r.strong_consistency is True  # 5+1=6 > 5

    def test_custom_quorum_weak_consistency(self):
        """R+W <= N means no strong consistency guarantee."""
        r = quorum_calc(total_nodes=5, read_quorum=1, write_quorum=1)
        assert r.strong_consistency is False

    def test_fault_tolerance(self):
        """Majority of 5: tolerates min(5-3, 5-3) = 2 failures."""
        r = quorum_calc(total_nodes=5, strategy="majority")
        assert r.fault_tolerance == 2

    def test_single_node(self):
        r = quorum_calc(total_nodes=1)
        assert r.read_quorum == 1
        assert r.write_quorum == 1

    def test_error_zero_nodes(self):
        with pytest.raises(ValueError, match="at least 1"):
            quorum_calc(total_nodes=0)

    def test_error_quorum_exceeds_nodes(self):
        with pytest.raises(ValueError, match="cannot exceed"):
            quorum_calc(total_nodes=3, read_quorum=4, write_quorum=2)

    def test_report_string(self):
        r = quorum_calc(total_nodes=5, strategy="majority")
        s = str(r)
        assert "Quorum" in s
        assert "Strong consistency" in s


class TestByzantineThreshold:
    def test_7_nodes_tolerates_2(self):
        """7 nodes: f = (7-1)//3 = 2 Byzantine faults."""
        r = byzantine_threshold(total_nodes=7)
        assert r.max_byzantine_faults == 2
        assert r.safe is True

    def test_3f_plus_1_from_faults(self):
        """Given f=2 faults, need 3*2+1 = 7 nodes minimum."""
        r = byzantine_threshold(max_faults=2)
        assert r.total_nodes == 7
        assert r.min_nodes_for_f_faults == 7

    def test_4_nodes_tolerates_1(self):
        r = byzantine_threshold(total_nodes=4)
        assert r.max_byzantine_faults == 1

    def test_unsafe_configuration(self):
        """3 nodes with 1 fault: 3 < 3*1+1=4, not safe."""
        r = byzantine_threshold(total_nodes=3, max_faults=1)
        assert r.safe is False

    def test_safe_configuration(self):
        r = byzantine_threshold(total_nodes=4, max_faults=1)
        assert r.safe is True

    def test_pbft_algorithm(self):
        r = byzantine_threshold(total_nodes=7, algorithm="PBFT")
        assert r.rounds_needed == r.max_byzantine_faults + 1

    def test_error_neither_specified(self):
        with pytest.raises(ValueError, match="Must specify"):
            byzantine_threshold()

    def test_report_string(self):
        r = byzantine_threshold(total_nodes=7)
        s = str(r)
        assert "Byzantine" in s
        assert "3f+1" in s


class TestVectorClockCompare:
    def test_concurrent_events(self):
        """[2,0] vs [1,1]: neither dominates, so concurrent."""
        r = vector_clock_compare([2, 0], [1, 1])
        assert r.relationship == "concurrent"

    def test_a_before_b(self):
        """[1,0] vs [2,1]: A <= B in all components with at least one strict."""
        r = vector_clock_compare([1, 0], [2, 1])
        assert r.relationship == "a_before_b"

    def test_b_before_a(self):
        """[2,1] vs [1,0]: B happened first."""
        r = vector_clock_compare([2, 1], [1, 0])
        assert r.relationship == "b_before_a"

    def test_equal_clocks(self):
        r = vector_clock_compare([3, 3], [3, 3])
        assert r.relationship == "equal"

    def test_merge_is_element_wise_max(self):
        r = vector_clock_compare([2, 0], [1, 1])
        assert r.merged == [2, 1]

    def test_three_node_concurrent(self):
        r = vector_clock_compare([3, 1, 0], [1, 0, 2])
        assert r.relationship == "concurrent"
        assert r.merged == [3, 1, 2]

    def test_three_node_causal(self):
        r = vector_clock_compare([1, 2, 3], [2, 3, 4])
        assert r.relationship == "a_before_b"

    def test_error_different_dimensions(self):
        with pytest.raises(ValueError, match="same dimension"):
            vector_clock_compare([1, 0], [1, 0, 0])

    def test_report_string(self):
        r = vector_clock_compare([2, 0], [1, 1])
        s = str(r)
        assert "Vector Clock" in s
        assert "concurrent" in s


class TestConsistencyModel:
    def test_linearizable_no_stale_reads(self):
        """Linearizable is strongest: no stale reads, no reordering."""
        r = consistency_model("linearizable")
        assert r.allows_stale_reads is False
        assert r.allows_reordering is False
        assert r.requires_coordination is True
        assert r.monotonic_reads is True
        assert r.read_your_writes is True

    def test_eventual_allows_stale(self):
        r = consistency_model("eventual")
        assert r.allows_stale_reads is True
        assert r.monotonic_reads is False
        assert r.read_your_writes is False

    def test_causal_available_under_partition(self):
        """Causal consistency is available during partitions (AP)."""
        r = consistency_model("causal")
        assert "available" in r.partition_behavior.lower()
        assert r.requires_coordination is False

    def test_sequential_consistency(self):
        r = consistency_model("sequential")
        assert r.allows_stale_reads is False
        assert r.monotonic_reads is True

    def test_read_your_writes(self):
        r = consistency_model("read_your_writes")
        assert r.read_your_writes is True
        assert r.allows_stale_reads is True  # from other writers

    def test_monotonic_reads_model(self):
        r = consistency_model("monotonic_reads")
        assert r.monotonic_reads is True
        assert r.read_your_writes is False

    def test_error_unknown_model(self):
        with pytest.raises(ValueError, match="Unknown model"):
            consistency_model("quantum_consistency")

    def test_report_string(self):
        r = consistency_model("linearizable")
        s = str(r)
        assert "Consistency Model" in s
        assert "stale reads" in s.lower()


class TestGossipConvergence:
    def test_broadcast_fanout(self):
        """If fanout >= N-1, all nodes reached in 1 round."""
        r = gossip_convergence(total_nodes=5, fanout=10)
        assert r.rounds_to_all == 1

    def test_rounds_increase_with_nodes(self):
        """More nodes requires more rounds."""
        r_small = gossip_convergence(total_nodes=10, fanout=3)
        r_large = gossip_convergence(total_nodes=1000, fanout=3)
        assert r_large.rounds_to_all > r_small.rounds_to_all

    def test_higher_fanout_fewer_rounds(self):
        r_low = gossip_convergence(total_nodes=100, fanout=2)
        r_high = gossip_convergence(total_nodes=100, fanout=10)
        assert r_high.rounds_to_all <= r_low.rounds_to_all

    def test_many_rounds_low_uninformed(self):
        """After many rounds, probability of uninformed should be low."""
        r = gossip_convergence(total_nodes=10, fanout=3, rounds=20)
        assert r.probability_uninformed < 0.01

    def test_error_single_node(self):
        with pytest.raises(ValueError, match="at least 2"):
            gossip_convergence(total_nodes=1)

    def test_error_zero_fanout(self):
        with pytest.raises(ValueError, match="at least 1"):
            gossip_convergence(total_nodes=10, fanout=0)

    def test_report_string(self):
        r = gossip_convergence(total_nodes=10, fanout=3)
        s = str(r)
        assert "Gossip" in s
        assert "rounds" in s.lower()
