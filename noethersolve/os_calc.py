"""Operating systems calculator — derives answers from first principles.

Covers page table sizing, virtual address translation, scheduling algorithms,
and deadlock detection via resource allocation graphs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple


@dataclass
class PageTableReport:
    """Result of page table sizing calculation."""
    virtual_bits: int
    physical_bits: int
    page_size_bytes: int
    page_offset_bits: int
    num_virtual_pages: int
    num_physical_frames: int
    pte_size_bytes: int
    page_table_size_bytes: int
    levels: int  # for multi-level
    level_sizes: List[int]  # entries per level

    def __str__(self) -> str:
        lines = [
            "Page Table Analysis:",
            f"  Virtual address: {self.virtual_bits} bits",
            f"  Physical address: {self.physical_bits} bits",
            f"  Page size: {self.page_size_bytes} bytes ({self.page_size_bytes//1024} KB)",
            f"  Page offset: {self.page_offset_bits} bits",
            f"  Virtual pages: {self.num_virtual_pages} ({self.num_virtual_pages:.2e})",
            f"  Physical frames: {self.num_physical_frames}",
            f"  PTE size: {self.pte_size_bytes} bytes",
            f"  Single-level table: {self.page_table_size_bytes} bytes ({self.page_table_size_bytes/(1024*1024):.1f} MB)",
        ]
        if self.levels > 1:
            lines.append(f"  Multi-level ({self.levels} levels): entries per level = {self.level_sizes}")
        return "\n".join(lines)


@dataclass
class SchedulingReport:
    """Result of scheduling algorithm simulation."""
    algorithm: str
    order: List[str]  # process execution order
    completion_times: Dict[str, float]
    turnaround_times: Dict[str, float]
    waiting_times: Dict[str, float]
    avg_turnaround: float
    avg_waiting: float
    timeline: str  # Gantt chart string

    def __str__(self) -> str:
        lines = [
            f"Scheduling — {self.algorithm}:",
            f"  Execution order: {' → '.join(self.order)}",
            f"  Avg turnaround time: {self.avg_turnaround:.2f}",
            f"  Avg waiting time: {self.avg_waiting:.2f}",
            f"  Timeline: {self.timeline}",
            "  Details:",
        ]
        for p in self.order:
            if p in self.completion_times:
                lines.append(
                    f"    {p}: complete={self.completion_times[p]:.1f}, "
                    f"turnaround={self.turnaround_times[p]:.1f}, "
                    f"wait={self.waiting_times[p]:.1f}"
                )
        return "\n".join(lines)


@dataclass
class DeadlockReport:
    """Result of deadlock detection."""
    has_deadlock: bool
    deadlocked_processes: List[str]
    cycle: List[str]  # the cycle in the wait-for graph
    safe_sequence: Optional[List[str]]  # if using Banker's algorithm
    explanation: str

    def __str__(self) -> str:
        lines = [
            "Deadlock Detection:",
            f"  Deadlock detected: {self.has_deadlock}",
        ]
        if self.has_deadlock:
            lines.append(f"  Deadlocked processes: {self.deadlocked_processes}")
            lines.append(f"  Cycle: {' → '.join(self.cycle)}")
        if self.safe_sequence:
            lines.append(f"  Safe sequence: {' → '.join(self.safe_sequence)}")
        lines.append(f"  {self.explanation}")
        return "\n".join(lines)


@dataclass
class TLBReport:
    """Result of TLB analysis."""
    tlb_entries: int
    page_size_bytes: int
    coverage_bytes: int
    coverage_mb: float
    hit_rate: Optional[float]  # estimated given working set
    miss_penalty_cycles: int
    effective_access_time: Optional[float]  # nanoseconds

    def __str__(self) -> str:
        lines = [
            "TLB Analysis:",
            f"  Entries: {self.tlb_entries}",
            f"  Page size: {self.page_size_bytes // 1024} KB",
            f"  TLB coverage: {self.coverage_bytes} bytes ({self.coverage_mb:.2f} MB)",
        ]
        if self.hit_rate is not None:
            lines.append(f"  Estimated hit rate: {self.hit_rate*100:.1f}%")
        if self.effective_access_time is not None:
            lines.append(f"  Effective access time: {self.effective_access_time:.1f} ns")
        return "\n".join(lines)


@dataclass
class ContextSwitchReport:
    """Result of context switch cost analysis."""
    direct_cost_us: float  # microseconds
    indirect_cost_us: float  # cache/TLB cold start
    total_cost_us: float
    switches_per_second: int
    overhead_pct: float  # percent of CPU time lost to switching

    def __str__(self) -> str:
        lines = [
            "Context Switch Cost Analysis:",
            f"  Direct cost: {self.direct_cost_us:.1f} µs (register save/restore + TLB flush)",
            f"  Indirect cost: {self.indirect_cost_us:.1f} µs (cache/TLB warmup)",
            f"  Total per switch: {self.total_cost_us:.1f} µs",
            f"  At {self.switches_per_second} switches/sec: {self.overhead_pct:.2f}% CPU overhead",
        ]
        return "\n".join(lines)


def page_table_calc(
    virtual_bits: int,
    physical_bits: int,
    page_size_bytes: int,
    pte_extra_bits: int = 8,
    levels: int = 1,
) -> PageTableReport:
    """Calculate page table dimensions from address space parameters.

    Derives page offset bits, number of pages/frames, PTE sizes,
    and page table memory requirements.

    Args:
        virtual_bits: Virtual address width (e.g., 48 for x86-64)
        physical_bits: Physical address width (e.g., 52 for x86-64)
        page_size_bytes: Page size (must be power of 2)
        pte_extra_bits: Extra bits per PTE (present, dirty, accessed, etc.)
        levels: Number of page table levels (1-5)

    Returns:
        PageTableReport.
    """
    if page_size_bytes <= 0 or (page_size_bytes & (page_size_bytes - 1)) != 0:
        raise ValueError("Page size must be a positive power of 2")

    offset_bits = int(math.log2(page_size_bytes))
    num_virtual_pages = 2 ** (virtual_bits - offset_bits)
    num_physical_frames = 2 ** (physical_bits - offset_bits)

    # PTE must hold physical frame number + extra bits
    pfn_bits = physical_bits - offset_bits
    pte_bits = pfn_bits + pte_extra_bits
    pte_bytes = math.ceil(pte_bits / 8)
    # Round up to power of 2 for alignment
    pte_bytes = 1 << math.ceil(math.log2(max(1, pte_bytes)))

    # Single-level page table size
    pt_size = num_virtual_pages * pte_bytes

    # Multi-level breakdown
    vpn_bits = virtual_bits - offset_bits
    level_sizes = []
    if levels > 1:
        bits_per_level = vpn_bits // levels
        remainder = vpn_bits % levels
        for i in range(levels):
            extra = 1 if i < remainder else 0
            level_sizes.append(2 ** (bits_per_level + extra))
    else:
        level_sizes = [num_virtual_pages]

    return PageTableReport(
        virtual_bits=virtual_bits,
        physical_bits=physical_bits,
        page_size_bytes=page_size_bytes,
        page_offset_bits=offset_bits,
        num_virtual_pages=num_virtual_pages,
        num_physical_frames=num_physical_frames,
        pte_size_bytes=pte_bytes,
        page_table_size_bytes=pt_size,
        levels=levels,
        level_sizes=level_sizes,
    )


def schedule_fcfs(
    processes: List[Tuple[str, float, float]],
) -> SchedulingReport:
    """Simulate First-Come-First-Served scheduling.

    Args:
        processes: List of (name, arrival_time, burst_time)

    Returns:
        SchedulingReport.
    """
    return _schedule(processes, "FCFS")


def schedule_sjf(
    processes: List[Tuple[str, float, float]],
    preemptive: bool = False,
) -> SchedulingReport:
    """Simulate Shortest Job First scheduling.

    Args:
        processes: List of (name, arrival_time, burst_time)
        preemptive: If True, use SRTF (preemptive SJF)

    Returns:
        SchedulingReport.
    """
    algo = "SRTF" if preemptive else "SJF"
    return _schedule(processes, algo)


def schedule_round_robin(
    processes: List[Tuple[str, float, float]],
    quantum: float = 2.0,
) -> SchedulingReport:
    """Simulate Round Robin scheduling.

    Args:
        processes: List of (name, arrival_time, burst_time)
        quantum: Time quantum

    Returns:
        SchedulingReport.
    """
    return _schedule(processes, "RR", quantum=quantum)


def detect_deadlock(
    holding: Dict[str, List[str]],
    waiting: Dict[str, str],
) -> DeadlockReport:
    """Detect deadlock using wait-for graph cycle detection.

    Deadlock requires all four Coffman conditions: mutual exclusion,
    hold and wait, no preemption, circular wait.

    Args:
        holding: {process: [resources it holds]}
        waiting: {process: resource it's waiting for}

    Returns:
        DeadlockReport.
    """
    # Build wait-for graph: process -> set of processes it waits for
    # Process A waits for process B if A is waiting for a resource held by B
    resource_owner: Dict[str, str] = {}
    for proc, resources in holding.items():
        for r in resources:
            resource_owner[r] = proc

    wait_for: Dict[str, Set[str]] = {}
    for proc, resource in waiting.items():
        if resource in resource_owner:
            owner = resource_owner[resource]
            if owner != proc:
                if proc not in wait_for:
                    wait_for[proc] = set()
                wait_for[proc].add(owner)

    # DFS cycle detection
    visited: Set[str] = set()
    rec_stack: Set[str] = set()
    cycle: List[str] = []

    def dfs(node: str, path: List[str]) -> bool:
        visited.add(node)
        rec_stack.add(node)
        path.append(node)

        for neighbor in wait_for.get(node, set()):
            if neighbor not in visited:
                if dfs(neighbor, path):
                    return True
            elif neighbor in rec_stack:
                # Found cycle
                idx = path.index(neighbor)
                cycle.extend(path[idx:])
                cycle.append(neighbor)
                return True

        path.pop()
        rec_stack.remove(node)
        return False

    has_deadlock = False
    for proc in wait_for:
        if proc not in visited:
            if dfs(proc, []):
                has_deadlock = True
                break

    deadlocked = list(set(cycle[:-1])) if cycle else []

    if has_deadlock:
        explanation = (f"Circular wait detected: {' → '.join(cycle)}. "
                       "All four Coffman conditions are present.")
    else:
        explanation = "No circular wait found. System is deadlock-free."

    return DeadlockReport(
        has_deadlock=has_deadlock,
        deadlocked_processes=deadlocked,
        cycle=cycle,
        safe_sequence=None,
        explanation=explanation,
    )


def tlb_analysis(
    tlb_entries: int,
    page_size_bytes: int,
    working_set_bytes: Optional[int] = None,
    memory_access_ns: float = 100.0,
    tlb_access_ns: float = 1.0,
    miss_penalty_cycles: int = 200,
) -> TLBReport:
    """Analyze TLB coverage and performance.

    The TLB caches virtual-to-physical translations (not data, not code).

    Args:
        tlb_entries: Number of TLB entries
        page_size_bytes: Page size
        working_set_bytes: Optional working set size for hit rate estimation
        memory_access_ns: Memory access time in nanoseconds
        tlb_access_ns: TLB lookup time in nanoseconds
        miss_penalty_cycles: Penalty for TLB miss (page table walk)

    Returns:
        TLBReport.
    """
    coverage = tlb_entries * page_size_bytes
    coverage_mb = coverage / (1024 * 1024)

    hit_rate = None
    eat = None
    if working_set_bytes is not None:
        pages_needed = math.ceil(working_set_bytes / page_size_bytes)
        hit_rate = min(1.0, tlb_entries / pages_needed) if pages_needed > 0 else 1.0
        # Effective access time = hit_rate * (TLB + mem) + (1-hit_rate) * (TLB + miss_penalty + mem)
        eat = hit_rate * (tlb_access_ns + memory_access_ns) + \
              (1 - hit_rate) * (tlb_access_ns + miss_penalty_cycles + memory_access_ns)

    return TLBReport(
        tlb_entries=tlb_entries,
        page_size_bytes=page_size_bytes,
        coverage_bytes=coverage,
        coverage_mb=coverage_mb,
        hit_rate=hit_rate,
        miss_penalty_cycles=miss_penalty_cycles,
        effective_access_time=eat,
    )


def context_switch_cost(
    direct_cost_us: float = 5.0,
    indirect_cost_us: float = 15.0,
    switches_per_second: int = 1000,
) -> ContextSwitchReport:
    """Calculate context switch overhead.

    Context switches include saving/restoring registers AND TLB flush
    (a common LLM weak spot — models often forget the TLB flush cost).

    Args:
        direct_cost_us: Direct cost in microseconds (register save/restore + TLB flush)
        indirect_cost_us: Indirect cost (cache/TLB warmup after switch)
        switches_per_second: Frequency of context switches

    Returns:
        ContextSwitchReport.
    """
    total = direct_cost_us + indirect_cost_us
    # Overhead = switches * cost / 1_000_000 seconds * 100%
    overhead_pct = (switches_per_second * total / 1_000_000) * 100

    return ContextSwitchReport(
        direct_cost_us=direct_cost_us,
        indirect_cost_us=indirect_cost_us,
        total_cost_us=total,
        switches_per_second=switches_per_second,
        overhead_pct=overhead_pct,
    )


def _schedule(
    processes: List[Tuple[str, float, float]],
    algo: str,
    quantum: float = 2.0,
) -> SchedulingReport:
    """Generic scheduling simulator."""
    if not processes:
        raise ValueError("No processes to schedule")

    procs = [(name, arrival, burst) for name, arrival, burst in processes]
    procs.sort(key=lambda x: x[1])  # sort by arrival

    if algo == "FCFS":
        return _sim_fcfs(procs)
    elif algo == "SJF":
        return _sim_sjf(procs)
    elif algo == "SRTF":
        return _sim_srtf(procs)
    elif algo == "RR":
        return _sim_rr(procs, quantum)
    else:
        raise ValueError(f"Unknown algorithm: {algo}")


def _sim_fcfs(procs):
    time = 0.0
    order = []
    completion = {}
    turnaround = {}
    waiting = {}
    timeline_parts = []

    for name, arrival, burst in procs:
        if time < arrival:
            timeline_parts.append(f"[idle:{arrival-time:.0f}]")
            time = arrival
        completion[name] = time + burst
        turnaround[name] = completion[name] - arrival
        waiting[name] = time - arrival
        timeline_parts.append(f"[{name}:{burst:.0f}]")
        time += burst
        order.append(name)

    # Deduplicate order for display
    seen = set()
    unique_order = []
    for o in order:
        if o not in seen:
            unique_order.append(o)
            seen.add(o)

    avg_ta = sum(turnaround.values()) / len(turnaround)
    avg_w = sum(waiting.values()) / len(waiting)

    return SchedulingReport(
        algorithm="FCFS",
        order=unique_order,
        completion_times=completion,
        turnaround_times=turnaround,
        waiting_times=waiting,
        avg_turnaround=avg_ta,
        avg_waiting=avg_w,
        timeline="".join(timeline_parts),
    )


def _sim_sjf(procs):
    """Non-preemptive SJF."""
    remaining = [(name, arrival, burst) for name, arrival, burst in procs]
    time = 0.0
    order = []
    completion = {}
    turnaround = {}
    waiting = {}
    timeline_parts = []
    done = set()

    while len(done) < len(procs):
        available = [(n, a, b) for n, a, b in remaining if a <= time and n not in done]
        if not available:
            # Jump to next arrival
            next_arrivals = [(n, a, b) for n, a, b in remaining if n not in done]
            next_arrival = min(a for _, a, _ in next_arrivals)
            timeline_parts.append(f"[idle:{next_arrival-time:.0f}]")
            time = next_arrival
            continue

        # Pick shortest burst
        available.sort(key=lambda x: x[2])
        name, arrival, burst = available[0]
        completion[name] = time + burst
        turnaround[name] = completion[name] - arrival
        waiting[name] = time - arrival
        timeline_parts.append(f"[{name}:{burst:.0f}]")
        time += burst
        order.append(name)
        done.add(name)

    avg_ta = sum(turnaround.values()) / len(turnaround)
    avg_w = sum(waiting.values()) / len(waiting)

    return SchedulingReport(
        algorithm="SJF (non-preemptive)",
        order=order,
        completion_times=completion,
        turnaround_times=turnaround,
        waiting_times=waiting,
        avg_turnaround=avg_ta,
        avg_waiting=avg_w,
        timeline="".join(timeline_parts),
    )


def _sim_srtf(procs):
    """Preemptive SJF (Shortest Remaining Time First)."""
    remaining_burst = {name: burst for name, _, burst in procs}
    arrival_map = {name: arrival for name, arrival, _ in procs}
    time = 0.0
    order = []
    completion = {}
    done = set()
    timeline_parts = []
    last_proc = None

    events = sorted(set(a for _, a, _ in procs))
    time = events[0] if events else 0

    while len(done) < len(procs):
        available = [n for n in remaining_burst if arrival_map[n] <= time and n not in done]
        if not available:
            undone = [n for n in remaining_burst if n not in done]
            if not undone:
                break
            time = min(arrival_map[n] for n in undone)
            continue

        # Pick shortest remaining
        current = min(available, key=lambda n: remaining_burst[n])
        if current != last_proc:
            order.append(current)
            last_proc = current

        # Run until next arrival or completion
        next_arrivals = [arrival_map[n] for n in remaining_burst
                         if n not in done and arrival_map[n] > time]
        run_until = remaining_burst[current] + time
        if next_arrivals:
            run_until = min(run_until, min(next_arrivals))

        duration = run_until - time
        remaining_burst[current] -= duration
        timeline_parts.append(f"[{current}:{duration:.0f}]")
        time = run_until

        if remaining_burst[current] <= 1e-9:
            completion[current] = time
            done.add(current)

    turnaround = {n: completion[n] - arrival_map[n] for n in completion}
    waiting_times = {n: turnaround[n] - (procs[i][2]) for i, (n, _, _) in enumerate(procs) if n in completion}
    avg_ta = sum(turnaround.values()) / max(1, len(turnaround))
    avg_w = sum(waiting_times.values()) / max(1, len(waiting_times))

    # Deduplicate
    seen = set()
    unique_order = []
    for o in order:
        if o not in seen:
            unique_order.append(o)
            seen.add(o)

    return SchedulingReport(
        algorithm="SRTF (preemptive SJF)",
        order=unique_order,
        completion_times=completion,
        turnaround_times=turnaround,
        waiting_times=waiting_times,
        avg_turnaround=avg_ta,
        avg_waiting=avg_w,
        timeline="".join(timeline_parts),
    )


def _sim_rr(procs, quantum):
    """Round Robin scheduling."""
    queue = []
    remaining_burst = {}
    arrival_map = {}
    original_burst = {}
    for name, arrival, burst in procs:
        remaining_burst[name] = burst
        arrival_map[name] = arrival
        original_burst[name] = burst

    time = procs[0][1]
    # Add initially available
    for name, arrival, burst in procs:
        if arrival <= time:
            queue.append(name)

    order = []
    completion = {}
    done = set()
    timeline_parts = []
    added = set(queue)

    while queue or len(done) < len(procs):
        if not queue:
            undone = [n for n in remaining_burst if n not in done]
            if not undone:
                break
            time = min(arrival_map[n] for n in undone)
            for name, arrival, _ in procs:
                if arrival <= time and name not in done and name not in added:
                    queue.append(name)
                    added.add(name)
            continue

        current = queue.pop(0)
        run_time = min(quantum, remaining_burst[current])
        timeline_parts.append(f"[{current}:{run_time:.0f}]")
        if current not in order:
            order.append(current)
        remaining_burst[current] -= run_time
        time += run_time

        # Add newly arrived processes
        for name, arrival, _ in procs:
            if arrival <= time and name not in done and name not in added:
                queue.append(name)
                added.add(name)

        if remaining_burst[current] <= 1e-9:
            completion[current] = time
            done.add(current)
        else:
            queue.append(current)

    turnaround = {n: completion[n] - arrival_map[n] for n in completion}
    waiting_times = {n: turnaround[n] - original_burst[n] for n in completion}
    avg_ta = sum(turnaround.values()) / max(1, len(turnaround))
    avg_w = sum(waiting_times.values()) / max(1, len(waiting_times))

    return SchedulingReport(
        algorithm=f"Round Robin (q={quantum})",
        order=order,
        completion_times=completion,
        turnaround_times=turnaround,
        waiting_times=waiting_times,
        avg_turnaround=avg_ta,
        avg_waiting=avg_w,
        timeline="".join(timeline_parts),
    )
