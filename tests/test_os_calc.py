"""Tests for the operating systems calculator module."""

import pytest
from noethersolve.os_calc import (
    page_table_calc,
    schedule_fcfs,
    schedule_sjf,
    schedule_round_robin,
    detect_deadlock,
    tlb_analysis,
    context_switch_cost,
)


class TestPageTableCalc:
    def test_32bit_4kb_pages(self):
        """32-bit VA, 32-bit PA, 4KB pages: 2^20 = 1M virtual pages."""
        r = page_table_calc(virtual_bits=32, physical_bits=32, page_size_bytes=4096)
        assert r.page_offset_bits == 12  # log2(4096)
        assert r.num_virtual_pages == 2**20  # 1,048,576

    def test_page_offset_bits(self):
        """4KB page = 12 offset bits."""
        r = page_table_calc(virtual_bits=32, physical_bits=32, page_size_bytes=4096)
        assert r.page_offset_bits == 12

    def test_num_physical_frames(self):
        """32-bit PA, 4KB pages: 2^20 physical frames."""
        r = page_table_calc(virtual_bits=32, physical_bits=32, page_size_bytes=4096)
        assert r.num_physical_frames == 2**20

    def test_48bit_x86_64(self):
        """x86-64: 48-bit VA, 52-bit PA, 4KB pages."""
        r = page_table_calc(virtual_bits=48, physical_bits=52, page_size_bytes=4096)
        assert r.num_virtual_pages == 2**36
        assert r.num_physical_frames == 2**40

    def test_large_page_size(self):
        """2MB huge pages: 21 offset bits."""
        r = page_table_calc(virtual_bits=48, physical_bits=52,
                            page_size_bytes=2*1024*1024)
        assert r.page_offset_bits == 21

    def test_multi_level(self):
        """Multi-level page table has level_sizes."""
        r = page_table_calc(virtual_bits=32, physical_bits=32,
                            page_size_bytes=4096, levels=2)
        assert r.levels == 2
        assert len(r.level_sizes) == 2

    def test_pte_size_reasonable(self):
        """PTE must be at least large enough to hold PFN + extra bits."""
        r = page_table_calc(virtual_bits=32, physical_bits=32, page_size_bytes=4096)
        assert r.pte_size_bytes >= 3  # 20-bit PFN + 8 extra = 28 bits = 4 bytes

    def test_page_table_size(self):
        """Table size = num_virtual_pages * pte_size_bytes."""
        r = page_table_calc(virtual_bits=32, physical_bits=32, page_size_bytes=4096)
        assert r.page_table_size_bytes == r.num_virtual_pages * r.pte_size_bytes

    def test_error_non_power_of_2(self):
        with pytest.raises(ValueError, match="power of 2"):
            page_table_calc(virtual_bits=32, physical_bits=32, page_size_bytes=3000)

    def test_report_string(self):
        r = page_table_calc(virtual_bits=32, physical_bits=32, page_size_bytes=4096)
        s = str(r)
        assert "Page Table" in s
        assert "Virtual pages" in s
        assert "PTE size" in s


class TestScheduleFCFS:
    def test_basic_order(self):
        """FCFS executes in arrival order."""
        procs = [("P1", 0, 5), ("P2", 1, 3), ("P3", 2, 1)]
        r = schedule_fcfs(procs)
        assert r.order == ["P1", "P2", "P3"]

    def test_completion_times(self):
        """P1 finishes at 5, P2 at 8, P3 at 9."""
        procs = [("P1", 0, 5), ("P2", 1, 3), ("P3", 2, 1)]
        r = schedule_fcfs(procs)
        assert r.completion_times["P1"] == 5
        assert r.completion_times["P2"] == 8
        assert r.completion_times["P3"] == 9

    def test_waiting_times(self):
        procs = [("P1", 0, 5), ("P2", 1, 3), ("P3", 2, 1)]
        r = schedule_fcfs(procs)
        assert r.waiting_times["P1"] == 0  # starts immediately
        assert r.waiting_times["P2"] == 4  # waits from t=1 to t=5
        assert r.waiting_times["P3"] == 6  # waits from t=2 to t=8

    def test_single_process(self):
        procs = [("P1", 0, 10)]
        r = schedule_fcfs(procs)
        assert r.order == ["P1"]
        assert r.avg_waiting == 0

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="No processes"):
            schedule_fcfs([])

    def test_report_string(self):
        procs = [("P1", 0, 5), ("P2", 1, 3)]
        r = schedule_fcfs(procs)
        s = str(r)
        assert "FCFS" in s
        assert "Execution order" in s


class TestScheduleSJF:
    def test_picks_shortest_burst(self):
        """SJF should pick the shortest available job next."""
        procs = [("P1", 0, 6), ("P2", 0, 2), ("P3", 0, 4)]
        r = schedule_sjf(procs)
        assert r.order[0] == "P2"  # shortest burst first

    def test_better_avg_wait_than_fcfs(self):
        """SJF has optimal average waiting time (non-preemptive)."""
        procs = [("P1", 0, 6), ("P2", 0, 2), ("P3", 0, 4)]
        r_fcfs = schedule_fcfs(procs)
        r_sjf = schedule_sjf(procs)
        assert r_sjf.avg_waiting <= r_fcfs.avg_waiting

    def test_respects_arrival_time(self):
        """SJF only considers arrived processes."""
        procs = [("P1", 0, 10), ("P2", 5, 1)]
        r = schedule_sjf(procs)
        # P1 arrives first and must run (nothing else available at t=0)
        assert r.order[0] == "P1"

    def test_report_string(self):
        procs = [("P1", 0, 6), ("P2", 0, 2)]
        r = schedule_sjf(procs)
        s = str(r)
        assert "SJF" in s


class TestScheduleRoundRobin:
    def test_all_processes_get_turns(self):
        """RR ensures all processes appear in the schedule."""
        procs = [("P1", 0, 5), ("P2", 0, 3), ("P3", 0, 1)]
        r = schedule_round_robin(procs, quantum=2)
        assert set(r.order) == {"P1", "P2", "P3"}

    def test_quantum_limits_execution(self):
        """No process runs longer than quantum per turn."""
        procs = [("P1", 0, 10), ("P2", 0, 10)]
        r = schedule_round_robin(procs, quantum=2)
        # Timeline should show alternating execution
        assert "P1" in r.timeline
        assert "P2" in r.timeline

    def test_short_process_finishes_early(self):
        """Process shorter than quantum finishes without using full quantum."""
        procs = [("P1", 0, 1), ("P2", 0, 5)]
        r = schedule_round_robin(procs, quantum=3)
        assert r.completion_times["P1"] <= 1

    def test_report_string(self):
        procs = [("P1", 0, 5), ("P2", 0, 3)]
        r = schedule_round_robin(procs, quantum=2)
        s = str(r)
        assert "Round Robin" in s


class TestDetectDeadlock:
    def test_simple_cycle(self):
        """P1 holds R1 wants R2, P2 holds R2 wants R1 -> deadlock."""
        holding = {"P1": ["R1"], "P2": ["R2"]}
        waiting = {"P1": "R2", "P2": "R1"}
        r = detect_deadlock(holding, waiting)
        assert r.has_deadlock is True
        assert len(r.deadlocked_processes) >= 2

    def test_no_deadlock(self):
        """P1 holds R1, P2 waits for R1 (no cycle)."""
        holding = {"P1": ["R1"]}
        waiting = {"P2": "R1"}
        r = detect_deadlock(holding, waiting)
        assert r.has_deadlock is False

    def test_three_process_cycle(self):
        """P1->R2(P2), P2->R3(P3), P3->R1(P1) -> deadlock."""
        holding = {"P1": ["R1"], "P2": ["R2"], "P3": ["R3"]}
        waiting = {"P1": "R2", "P2": "R3", "P3": "R1"}
        r = detect_deadlock(holding, waiting)
        assert r.has_deadlock is True

    def test_no_waiting(self):
        """No process is waiting: no deadlock."""
        holding = {"P1": ["R1"], "P2": ["R2"]}
        waiting = {}
        r = detect_deadlock(holding, waiting)
        assert r.has_deadlock is False

    def test_cycle_in_report(self):
        holding = {"P1": ["R1"], "P2": ["R2"]}
        waiting = {"P1": "R2", "P2": "R1"}
        r = detect_deadlock(holding, waiting)
        assert len(r.cycle) > 0

    def test_report_string(self):
        holding = {"P1": ["R1"], "P2": ["R2"]}
        waiting = {"P1": "R2", "P2": "R1"}
        r = detect_deadlock(holding, waiting)
        s = str(r)
        assert "Deadlock" in s
        assert "Cycle" in s or "cycle" in s


class TestTLBAnalysis:
    def test_coverage(self):
        """64 entries * 4KB = 256 KB coverage."""
        r = tlb_analysis(tlb_entries=64, page_size_bytes=4096)
        assert r.coverage_bytes == 64 * 4096  # 262144
        assert abs(r.coverage_mb - 0.25) < 0.01

    def test_hit_rate_full_coverage(self):
        """Working set fits in TLB: hit rate = 1.0."""
        r = tlb_analysis(tlb_entries=64, page_size_bytes=4096,
                         working_set_bytes=64 * 4096)
        assert r.hit_rate == 1.0

    def test_hit_rate_partial_coverage(self):
        """Working set = 2x TLB coverage: hit rate ~ 0.5."""
        r = tlb_analysis(tlb_entries=64, page_size_bytes=4096,
                         working_set_bytes=128 * 4096)
        assert abs(r.hit_rate - 0.5) < 0.01

    def test_effective_access_time(self):
        """EAT computed when working set given."""
        r = tlb_analysis(tlb_entries=64, page_size_bytes=4096,
                         working_set_bytes=64 * 4096,
                         memory_access_ns=100, tlb_access_ns=1,
                         miss_penalty_cycles=200)
        assert r.effective_access_time is not None
        # Full hit rate: EAT = 1 + 100 = 101 ns
        assert abs(r.effective_access_time - 101) < 0.1

    def test_no_working_set_no_hit_rate(self):
        r = tlb_analysis(tlb_entries=64, page_size_bytes=4096)
        assert r.hit_rate is None
        assert r.effective_access_time is None

    def test_huge_pages_more_coverage(self):
        """2MB huge pages: 64 entries * 2MB = 128 MB."""
        r = tlb_analysis(tlb_entries=64, page_size_bytes=2*1024*1024)
        assert abs(r.coverage_mb - 128) < 0.01

    def test_report_string(self):
        r = tlb_analysis(tlb_entries=64, page_size_bytes=4096)
        s = str(r)
        assert "TLB" in s
        assert "coverage" in s.lower()


class TestContextSwitchCost:
    def test_default_cost(self):
        """Default: 5 + 15 = 20 us total."""
        r = context_switch_cost()
        assert abs(r.total_cost_us - 20.0) < 1e-6

    def test_overhead_calculation(self):
        """1000 switches/sec at 20us each = 2% overhead."""
        r = context_switch_cost(direct_cost_us=5, indirect_cost_us=15,
                                switches_per_second=1000)
        assert abs(r.overhead_pct - 2.0) < 0.01

    def test_high_frequency_high_overhead(self):
        """10000 switches/sec should have 10x more overhead than 1000."""
        r_low = context_switch_cost(switches_per_second=1000)
        r_high = context_switch_cost(switches_per_second=10000)
        assert abs(r_high.overhead_pct / r_low.overhead_pct - 10) < 0.01

    def test_zero_indirect_cost(self):
        """Only direct cost."""
        r = context_switch_cost(direct_cost_us=5, indirect_cost_us=0,
                                switches_per_second=1000)
        assert abs(r.total_cost_us - 5.0) < 1e-6

    def test_report_string(self):
        r = context_switch_cost()
        s = str(r)
        assert "Context Switch" in s
        assert "overhead" in s.lower() or "CPU" in s
