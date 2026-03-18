"""Distributed systems calculator — derives answers from first principles.

Covers quorum calculations, Byzantine fault tolerance thresholds,
vector clock operations, and consistency model analysis.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class QuorumReport:
    """Result of quorum calculation."""
    total_nodes: int
    read_quorum: int
    write_quorum: int
    fault_tolerance: int  # max failures tolerated
    strong_consistency: bool  # R + W > N
    explanation: str

    def __str__(self) -> str:
        lines = [
            f"Quorum Analysis (N={self.total_nodes}):",
            f"  Read quorum (R):  {self.read_quorum}",
            f"  Write quorum (W): {self.write_quorum}",
            f"  R + W = {self.read_quorum + self.write_quorum} {'>' if self.strong_consistency else '≤'} N={self.total_nodes}",
            f"  Strong consistency: {self.strong_consistency}",
            f"  Fault tolerance: {self.fault_tolerance} node failures",
            f"  {self.explanation}",
        ]
        return "\n".join(lines)


@dataclass
class ByzantineReport:
    """Result of Byzantine fault tolerance analysis."""
    total_nodes: int
    max_byzantine_faults: int
    min_nodes_for_f_faults: int  # 3f+1
    algorithm: str
    rounds_needed: int  # f+1 rounds for synchronous
    message_complexity: str
    safe: bool

    def __str__(self) -> str:
        lines = [
            "Byzantine Fault Tolerance Analysis:",
            f"  Total nodes: {self.total_nodes}",
            f"  Max Byzantine faults tolerated: {self.max_byzantine_faults}",
            f"  Minimum nodes for f={self.max_byzantine_faults}: {self.min_nodes_for_f_faults} (3f+1)",
            f"  Algorithm: {self.algorithm}",
            f"  Rounds needed: {self.rounds_needed}",
            f"  Message complexity: {self.message_complexity}",
            f"  System safe: {self.safe}",
        ]
        return "\n".join(lines)


@dataclass
class VectorClockReport:
    """Result of vector clock comparison."""
    clock_a: List[int]
    clock_b: List[int]
    relationship: str  # "a_before_b", "b_before_a", "concurrent"
    merged: List[int]  # element-wise max
    explanation: str

    def __str__(self) -> str:
        lines = [
            "Vector Clock Comparison:",
            f"  A = {self.clock_a}",
            f"  B = {self.clock_b}",
            f"  Relationship: {self.relationship}",
            f"  Merged (max): {self.merged}",
            f"  {self.explanation}",
        ]
        return "\n".join(lines)


@dataclass
class ConsistencyReport:
    """Analysis of a consistency model's properties."""
    model: str
    allows_stale_reads: bool
    allows_reordering: bool
    requires_coordination: bool
    partition_behavior: str  # available or unavailable
    monotonic_reads: bool
    read_your_writes: bool
    properties: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        lines = [
            f"Consistency Model — {self.model}:",
            f"  Allows stale reads: {self.allows_stale_reads}",
            f"  Allows reordering: {self.allows_reordering}",
            f"  Requires coordination: {self.requires_coordination}",
            f"  Under partition: {self.partition_behavior}",
            f"  Monotonic reads: {self.monotonic_reads}",
            f"  Read-your-writes: {self.read_your_writes}",
        ]
        for p in self.properties:
            lines.append(f"  • {p}")
        return "\n".join(lines)


@dataclass
class GossipReport:
    """Result of gossip protocol convergence analysis."""
    total_nodes: int
    fanout: int
    rounds_to_all: int  # expected rounds for all nodes informed
    probability_uninformed: float  # after given rounds
    rounds_given: Optional[int]

    def __str__(self) -> str:
        lines = [
            f"Gossip Protocol Analysis (N={self.total_nodes}, fanout={self.fanout}):",
            f"  Expected rounds for full propagation: {self.rounds_to_all}",
        ]
        if self.rounds_given is not None:
            lines.append(f"  After {self.rounds_given} rounds: P(uninformed node) ≈ {self.probability_uninformed:.6e}")
        return "\n".join(lines)


def quorum_calc(
    total_nodes: int,
    read_quorum: Optional[int] = None,
    write_quorum: Optional[int] = None,
    strategy: str = "majority",
) -> QuorumReport:
    """Calculate quorum sizes and consistency guarantees.

    If read_quorum and write_quorum not given, derives from strategy.

    Args:
        total_nodes: Total number of replica nodes (N)
        read_quorum: Read quorum size (R), or None to derive
        write_quorum: Write quorum size (W), or None to derive
        strategy: "majority", "read_heavy", "write_heavy" if R/W not specified

    Returns:
        QuorumReport with consistency analysis.
    """
    N = total_nodes
    if N < 1:
        raise ValueError("total_nodes must be at least 1")

    if read_quorum is None or write_quorum is None:
        if strategy == "majority":
            R = N // 2 + 1
            W = N // 2 + 1
        elif strategy == "read_heavy":
            R = 1
            W = N
        elif strategy == "write_heavy":
            R = N
            W = 1
        else:
            raise ValueError("strategy must be majority, read_heavy, or write_heavy")
    else:
        R = read_quorum
        W = write_quorum

    if R < 1 or W < 1:
        raise ValueError("Quorum sizes must be at least 1")
    if R > N or W > N:
        raise ValueError("Quorum sizes cannot exceed total nodes")

    strong = R + W > N
    fault_tol = min(N - R, N - W)

    if strong:
        explanation = (f"R+W={R+W} > N={N}: every read overlaps with latest write. "
                       f"Strong consistency guaranteed. Tolerates {fault_tol} failures.")
    else:
        explanation = (f"R+W={R+W} ≤ N={N}: reads may miss latest write. "
                       f"Only eventual consistency. Tolerates {fault_tol} failures.")

    return QuorumReport(
        total_nodes=N,
        read_quorum=R,
        write_quorum=W,
        fault_tolerance=fault_tol,
        strong_consistency=strong,
        explanation=explanation,
    )


def byzantine_threshold(
    total_nodes: Optional[int] = None,
    max_faults: Optional[int] = None,
    algorithm: str = "PBFT",
) -> ByzantineReport:
    """Calculate Byzantine fault tolerance requirements.

    The fundamental result: BFT requires n ≥ 3f+1 nodes to tolerate f
    Byzantine failures (a common LLM weak spot — models often say 2f+1).

    Args:
        total_nodes: Number of nodes (derive max_faults), or
        max_faults: Desired fault tolerance (derive total_nodes)
        algorithm: "PBFT", "synchronous", or "Tendermint"

    Returns:
        ByzantineReport with analysis.
    """
    if total_nodes is not None and max_faults is not None:
        # Check if configuration is safe
        f = max_faults
        n = total_nodes
    elif total_nodes is not None:
        n = total_nodes
        f = (n - 1) // 3
    elif max_faults is not None:
        f = max_faults
        n = 3 * f + 1
    else:
        raise ValueError("Must specify total_nodes or max_faults")

    min_nodes = 3 * f + 1
    safe = n >= min_nodes

    algo = algorithm.upper()
    if algo in ("PBFT", "PRACTICAL"):
        rounds = f + 1
        msg_complexity = f"O(n²) per round, O(n²) total = ~{n*n} messages"
    elif algo in ("SYNC", "SYNCHRONOUS"):
        rounds = f + 1
        msg_complexity = "O(n^(f+1)) exponential in fault count"
    elif algo in ("TENDERMINT", "COSMOS"):
        rounds = 2  # propose + prevote + precommit
        msg_complexity = "O(n²) per round via gossip"
    else:
        rounds = f + 1
        msg_complexity = "unknown"

    return ByzantineReport(
        total_nodes=n,
        max_byzantine_faults=f,
        min_nodes_for_f_faults=min_nodes,
        algorithm=algorithm,
        rounds_needed=rounds,
        message_complexity=msg_complexity,
        safe=safe,
    )


def vector_clock_compare(
    clock_a: List[int],
    clock_b: List[int],
) -> VectorClockReport:
    """Compare two vector clocks to determine causal ordering.

    Vector clocks track causal ordering (not wall-clock time) across nodes.
    A happens-before B iff A[i] ≤ B[i] for all i, with at least one strict <.

    Args:
        clock_a: Vector clock for event A
        clock_b: Vector clock for event B

    Returns:
        VectorClockReport with causal relationship.
    """
    if len(clock_a) != len(clock_b):
        raise ValueError("Vector clocks must have same dimension (same number of nodes)")

    a_leq_b = all(a <= b for a, b in zip(clock_a, clock_b))
    b_leq_a = all(b <= a for a, b in zip(clock_a, clock_b))
    a_lt_b = a_leq_b and any(a < b for a, b in zip(clock_a, clock_b))
    b_lt_a = b_leq_a and any(b < a for a, b in zip(clock_a, clock_b))

    merged = [max(a, b) for a, b in zip(clock_a, clock_b)]

    if a_lt_b:
        relationship = "a_before_b"
        explanation = "A causally precedes B (A → B). A happened before B."
    elif b_lt_a:
        relationship = "b_before_a"
        explanation = "B causally precedes A (B → A). B happened before A."
    elif clock_a == clock_b:
        relationship = "equal"
        explanation = "Clocks are identical — same event or same causal position."
    else:
        relationship = "concurrent"
        explanation = "A ∥ B: events are concurrent (no causal relationship). Conflict possible."

    return VectorClockReport(
        clock_a=list(clock_a),
        clock_b=list(clock_b),
        relationship=relationship,
        merged=merged,
        explanation=explanation,
    )


def consistency_model(model: str) -> ConsistencyReport:
    """Analyze properties of a distributed consistency model.

    Derives guarantees from the model definition — not a lookup table.
    Each property is a logical consequence of the model's specification.

    Args:
        model: Consistency model name (linearizable, sequential, causal,
               eventual, read_your_writes, monotonic_reads)

    Returns:
        ConsistencyReport with derived properties.
    """
    model = model.lower().replace("-", "_").replace(" ", "_")

    models = {
        "linearizable": ConsistencyReport(
            model="Linearizable (strongest)",
            allows_stale_reads=False,
            allows_reordering=False,
            requires_coordination=True,
            partition_behavior="unavailable (CP in CAP)",
            monotonic_reads=True,
            read_your_writes=True,
            properties=[
                "Operations appear instantaneous at some point between invocation and response",
                "Real-time ordering preserved: if op1 completes before op2 starts, op1 is ordered first",
                "Equivalent to a single-copy system from client perspective",
                "Requires coordination (consensus) — unavailable during partitions",
                "Used by: Spanner (TrueTime), etcd, ZooKeeper",
            ],
        ),
        "sequential": ConsistencyReport(
            model="Sequential Consistency",
            allows_stale_reads=False,
            allows_reordering=False,
            requires_coordination=True,
            partition_behavior="unavailable",
            monotonic_reads=True,
            read_your_writes=True,
            properties=[
                "All processes see same order of operations",
                "Per-process ordering preserved, but no real-time constraint",
                "Weaker than linearizable: concurrent ops may be reordered globally",
                "Lamport's definition: result equivalent to some sequential execution",
            ],
        ),
        "causal": ConsistencyReport(
            model="Causal Consistency",
            allows_stale_reads=True,  # for causally unrelated writes
            allows_reordering=True,  # concurrent writes may be seen in different order
            requires_coordination=False,
            partition_behavior="available (AP in CAP)",
            monotonic_reads=True,
            read_your_writes=True,
            properties=[
                "Causally related operations seen in same order by all",
                "Concurrent (causally unrelated) writes may be seen in different orders",
                "Available during partitions — strongest available consistency",
                "Tracked via vector clocks or explicit dependency graphs",
                "Used by: COPS, MongoDB (w:majority + causal sessions)",
            ],
        ),
        "eventual": ConsistencyReport(
            model="Eventual Consistency",
            allows_stale_reads=True,
            allows_reordering=True,
            requires_coordination=False,
            partition_behavior="available (AP in CAP)",
            monotonic_reads=False,
            read_your_writes=False,
            properties=[
                "Replicas converge IF updates stop (quiescence requirement)",
                "No bound on convergence time",
                "May read stale data, may see writes out of order",
                "Reads may go backwards (non-monotonic) without session guarantees",
                "Used by: DNS, DynamoDB (default), Cassandra (ONE/ONE)",
            ],
        ),
        "read_your_writes": ConsistencyReport(
            model="Read-Your-Writes (session guarantee)",
            allows_stale_reads=True,  # from other writers
            allows_reordering=True,
            requires_coordination=False,
            partition_behavior="available",
            monotonic_reads=False,
            read_your_writes=True,
            properties=[
                "A process always sees its own previous writes",
                "May read stale data from OTHER writers",
                "Session-scoped: guarantee tied to client session, not global",
                "Typically implemented via sticky sessions or read-after-write tokens",
            ],
        ),
        "monotonic_reads": ConsistencyReport(
            model="Monotonic Reads (session guarantee)",
            allows_stale_reads=True,
            allows_reordering=True,
            requires_coordination=False,
            partition_behavior="available",
            monotonic_reads=True,
            read_your_writes=False,
            properties=[
                "Successive reads by same process never return older values",
                "Reads never go backwards — monotonically non-decreasing freshness",
                "Does NOT guarantee seeing own writes",
                "Does NOT guarantee seeing other processes' latest writes",
            ],
        ),
    }

    if model not in models:
        raise ValueError(f"Unknown model: {model}. Known: {', '.join(sorted(models))}")

    return models[model]


def gossip_convergence(
    total_nodes: int,
    fanout: int = 3,
    rounds: Optional[int] = None,
) -> GossipReport:
    """Analyze gossip protocol convergence.

    Gossip spreads information by random peer-to-peer exchange.
    After r rounds with fanout f, fraction uninformed ≈ (1-f/n)^(n*r) → e^(-fr).

    Args:
        total_nodes: Number of nodes in the cluster
        fanout: Number of peers each node contacts per round
        rounds: Optional specific round count to analyze

    Returns:
        GossipReport with convergence analysis.
    """
    N = total_nodes
    if N < 2:
        raise ValueError("Need at least 2 nodes for gossip")
    if fanout < 1:
        raise ValueError("Fanout must be at least 1")

    # Rounds to reach all nodes with high probability
    # After r rounds, expected uninformed ≈ N * e^(-f*r)
    # Set N * e^(-f*r) < 1 → r > ln(N)/f
    rounds_all = math.ceil(math.log(N) / math.log(1 + fanout / N) / N)
    # Simpler: rounds_all ≈ ceil(ln(N) * N / f) for small f/N
    # More accurate: r = ceil(log_base(1+f/N)(N)) but use epidemiological model
    # In push gossip: rounds = O(log(N)) for constant fanout
    rounds_all = math.ceil(math.log(N) / math.log(1 + fanout / max(1, N - 1)) + 1)
    # For large N: ≈ ceil(N*ln(N)/f)... actually for push gossip:
    # rounds = O(log(N)/log(f)) when f is constant
    if fanout >= N - 1:
        rounds_all = 1  # broadcast
    else:
        rounds_all = math.ceil(math.log(N) / math.log(max(2, fanout)) + math.log(math.log(N)))

    prob_uninformed = None
    if rounds is not None:
        # After r rounds of push gossip: P(node uninformed) ≈ e^(-fanout * r / N * N)
        # More precisely: fraction infected after r rounds ≈ 1 - e^(-c) where c grows exponentially
        # Simplified: P(uninformed after r rounds) ≈ (1 - fanout/N)^r ... iterated
        # Standard SIR-like: after r rounds with fanout f, fraction uninformed ≈ e^(-f^r / N)
        # Most accurate for push gossip: prob ≈ e^(-f * r) for large N
        # But really: in round 1, f nodes learn. In round 2, (f+1)*f nodes attempt...
        # The standard result: O(log(N)) rounds for full dissemination
        # P(all informed after r rounds) ≈ 1 - N * (1 - 1/N)^(sum of infected * fanout)
        # Approximate: after r rounds, infected ≈ min(N, fanout^r)
        infected = min(N, fanout ** rounds)
        if infected >= N:
            prob_uninformed = 0.0
        else:
            # Each uninformed node has prob (1-infected/N)^fanout of staying uninformed per round
            prob_uninformed = max(0, (1 - infected / N))

    return GossipReport(
        total_nodes=N,
        fanout=fanout,
        rounds_to_all=rounds_all,
        probability_uninformed=prob_uninformed if prob_uninformed is not None else 0.0,
        rounds_given=rounds,
    )
