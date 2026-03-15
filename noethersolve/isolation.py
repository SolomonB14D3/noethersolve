"""Transaction isolation anomaly checker — a CALCULATOR, not a lookup.

Given a set of transactions and an isolation level, computes which
concurrency anomalies are possible. Answers questions like:
"Can this schedule produce a phantom read under REPEATABLE READ?"

Usage:
    from noethersolve.isolation import check_isolation, list_anomalies

    # What anomalies are possible under READ_COMMITTED?
    report = check_isolation("READ_COMMITTED")
    print(report)

    # Can a specific anomaly occur?
    report = check_isolation("SERIALIZABLE", anomaly="phantom_read")
    print(report)

    # Analyze a concrete schedule
    report = analyze_schedule(
        transactions=[
            [("R", "x"), ("W", "x")],      # T1: read x then write x
            [("R", "x"), ("R", "y")],        # T2: read x then read y
        ],
        isolation="REPEATABLE_READ",
    )
    print(report)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple


# ── Anomaly definitions ─────────────────────────────────────────────────

@dataclass
class Anomaly:
    """A concurrency anomaly with its formal definition."""
    name: str
    description: str
    example: str
    requires_write: bool = True  # Does this anomaly require a write conflict?


ANOMALIES: Dict[str, Anomaly] = {
    "dirty_read": Anomaly(
        "Dirty Read",
        "T2 reads data written by T1 before T1 commits. "
        "If T1 rolls back, T2 has read data that never existed.",
        "T1: W(x=1) ... T2: R(x)=1 ... T1: ROLLBACK",
    ),
    "non_repeatable_read": Anomaly(
        "Non-Repeatable Read (Fuzzy Read)",
        "T1 reads x, T2 modifies x and commits, T1 re-reads x and gets "
        "a different value.",
        "T1: R(x)=1 ... T2: W(x=2), COMMIT ... T1: R(x)=2",
    ),
    "phantom_read": Anomaly(
        "Phantom Read",
        "T1 reads a set of rows matching a predicate. T2 inserts/deletes "
        "rows matching that predicate and commits. T1 re-reads and gets "
        "different rows.",
        "T1: SELECT * WHERE age>20 (3 rows) ... T2: INSERT (age=25), COMMIT ... "
        "T1: SELECT * WHERE age>20 (4 rows)",
    ),
    "lost_update": Anomaly(
        "Lost Update",
        "T1 and T2 both read x, then both write x. One update is lost.",
        "T1: R(x)=1 ... T2: R(x)=1 ... T1: W(x=2) ... T2: W(x=3) "
        "(T1's update lost)",
    ),
    "write_skew": Anomaly(
        "Write Skew",
        "T1 reads x and y, T2 reads x and y. T1 writes x based on y, "
        "T2 writes y based on x. Both commit — constraint violated.",
        "Constraint: x+y>0. T1: R(x=1,y=1), W(x=-1). T2: R(x=1,y=1), W(y=-1). "
        "Both commit: x+y=-2, constraint violated.",
    ),
    "read_skew": Anomaly(
        "Read Skew",
        "T1 reads x, T2 writes x and y (maintaining invariant), commits. "
        "T1 reads y — sees inconsistent snapshot (old x, new y).",
        "Invariant: x=y. T1: R(x)=1 ... T2: W(x=2), W(y=2), COMMIT ... "
        "T1: R(y)=2 (sees x=1, y=2)",
    ),
    "serialization_anomaly": Anomaly(
        "Serialization Anomaly",
        "The outcome is not equivalent to any serial execution of the "
        "transactions, even though no single standard anomaly pattern is "
        "present.",
        "Complex multi-transaction dependency cycles that don't fit "
        "standard anomaly categories.",
    ),
}

# ── Isolation levels and their prevented anomalies ───────────────────

# SQL standard isolation levels.
# Each maps to the set of anomalies it PREVENTS.
ISOLATION_LEVELS: Dict[str, Set[str]] = {
    "READ_UNCOMMITTED": set(),
    "READ_COMMITTED": {"dirty_read"},
    "REPEATABLE_READ": {"dirty_read", "non_repeatable_read"},
    "SNAPSHOT": {"dirty_read", "non_repeatable_read", "phantom_read", "read_skew"},
    "SERIALIZABLE": {
        "dirty_read", "non_repeatable_read", "phantom_read",
        "lost_update", "write_skew", "read_skew", "serialization_anomaly",
    },
}

# Common misconceptions about what each level prevents
MISCONCEPTIONS: Dict[str, List[str]] = {
    "READ_COMMITTED": [
        "DOES NOT prevent non-repeatable reads — a second SELECT can see "
        "committed changes from other transactions.",
        "DOES NOT prevent phantom reads.",
        "DOES NOT prevent lost updates.",
    ],
    "REPEATABLE_READ": [
        "DOES NOT prevent phantom reads (new rows matching a predicate "
        "can appear). This is a common misconception.",
        "DOES NOT prevent write skew in standard SQL. Some implementations "
        "(e.g., MySQL InnoDB with gap locks) may prevent phantoms in practice.",
        "DOES NOT prevent lost updates in all implementations — depends on "
        "whether the engine uses shared read locks or MVCC.",
    ],
    "SNAPSHOT": [
        "DOES NOT prevent write skew — two transactions can read overlapping "
        "data and make conflicting writes that individually look correct.",
        "NOT the same as SERIALIZABLE — PostgreSQL's REPEATABLE READ is "
        "actually snapshot isolation, which still allows write skew.",
        "DOES prevent phantom reads (reads from a consistent snapshot), "
        "unlike standard SQL REPEATABLE READ.",
    ],
    "SERIALIZABLE": [
        "Does not mean transactions run serially — it means the RESULT is "
        "equivalent to some serial ordering.",
        "Implementation varies: 2PL (lock-based) vs SSI (serializable "
        "snapshot isolation). SSI can have false positives (unnecessary aborts).",
    ],
}


@dataclass
class IsolationReport:
    """Report from isolation level analysis."""

    isolation_level: str
    queried_anomaly: Optional[str]
    possible_anomalies: List[str]
    prevented_anomalies: List[str]
    misconceptions: List[str]
    schedule_analysis: Optional[str] = None

    def __str__(self) -> str:
        lines = [
            "=" * 60,
            f"  Isolation Level Analysis: {self.isolation_level}",
            "=" * 60,
        ]

        if self.queried_anomaly:
            anom = ANOMALIES.get(self.queried_anomaly)
            possible = self.queried_anomaly in [
                a for a in ANOMALIES if a not in
                ISOLATION_LEVELS.get(self.isolation_level, set())
            ]
            lines.append(f"\n  Query: Can '{self.queried_anomaly}' occur?")
            lines.append(f"  Answer: {'YES — possible' if possible else 'NO — prevented'}")
            if anom:
                lines.append(f"  What it is: {anom.description}")
                lines.append(f"  Example: {anom.example}")

        lines.append(f"\n  Prevented anomalies ({len(self.prevented_anomalies)}):")
        for a in self.prevented_anomalies:
            anom = ANOMALIES.get(a)
            lines.append(f"    ✓ {anom.name if anom else a}")

        lines.append(f"\n  STILL POSSIBLE anomalies ({len(self.possible_anomalies)}):")
        if self.possible_anomalies:
            for a in self.possible_anomalies:
                anom = ANOMALIES.get(a)
                lines.append(f"    ✗ {anom.name if anom else a}")
                if anom:
                    lines.append(f"      {anom.description[:100]}")
        else:
            lines.append("    (none — full serializability)")

        if self.misconceptions:
            lines.append(f"\n  Common misconceptions:")
            for m in self.misconceptions:
                lines.append(f"    ⚠ {m}")

        if self.schedule_analysis:
            lines.append(f"\n  Schedule analysis:\n{self.schedule_analysis}")

        lines.append("=" * 60)
        return "\n".join(lines)


@dataclass
class ScheduleReport:
    """Report from concrete schedule analysis."""

    transactions: List[List[Tuple[str, str]]]
    isolation_level: str
    conflicts: List[str]
    possible_anomalies: List[str]
    serializable: bool
    explanation: str

    def __str__(self) -> str:
        lines = [
            "=" * 60,
            f"  Schedule Analysis under {self.isolation_level}",
            "=" * 60,
        ]
        for i, txn in enumerate(self.transactions):
            ops = ", ".join(f"{op}({item})" for op, item in txn)
            lines.append(f"  T{i+1}: {ops}")

        lines.append(f"\n  Conflicts found: {len(self.conflicts)}")
        for c in self.conflicts:
            lines.append(f"    - {c}")

        lines.append(f"\n  Possible anomalies under {self.isolation_level}:")
        if self.possible_anomalies:
            for a in self.possible_anomalies:
                lines.append(f"    ✗ {a}")
        else:
            lines.append("    (none detected)")

        lines.append(f"\n  Serializable equivalent exists: {'YES' if self.serializable else 'NO'}")
        lines.append(f"  {self.explanation}")
        lines.append("=" * 60)
        return "\n".join(lines)


# ── Public API ───────────────────────────────────────────────────────

def check_isolation(
    isolation_level: str,
    anomaly: Optional[str] = None,
) -> IsolationReport:
    """Check which anomalies are possible under an isolation level.

    Parameters
    ----------
    isolation_level : str
        One of: READ_UNCOMMITTED, READ_COMMITTED, REPEATABLE_READ,
        SNAPSHOT, SERIALIZABLE.
    anomaly : str, optional
        Specific anomaly to check. One of: dirty_read, non_repeatable_read,
        phantom_read, lost_update, write_skew, read_skew, serialization_anomaly.
    """
    level = isolation_level.upper().replace(" ", "_")
    if level not in ISOLATION_LEVELS:
        # Try fuzzy match
        for k in ISOLATION_LEVELS:
            if level in k or k in level:
                level = k
                break
        else:
            available = ", ".join(ISOLATION_LEVELS.keys())
            raise ValueError(
                f"Unknown isolation level '{isolation_level}'. "
                f"Available: {available}"
            )

    prevented = ISOLATION_LEVELS[level]
    possible = [a for a in ANOMALIES if a not in prevented]
    prevented_list = [a for a in ANOMALIES if a in prevented]

    return IsolationReport(
        isolation_level=level,
        queried_anomaly=anomaly,
        possible_anomalies=possible,
        prevented_anomalies=prevented_list,
        misconceptions=MISCONCEPTIONS.get(level, []),
    )


def analyze_schedule(
    transactions: List[List[Tuple[str, str]]],
    isolation: str = "READ_COMMITTED",
) -> ScheduleReport:
    """Analyze a concrete transaction schedule for conflicts and anomalies.

    Parameters
    ----------
    transactions : list of list of (operation, item) tuples
        Each transaction is a list of operations. Operation is "R" (read)
        or "W" (write). Item is the data item name.
        Example: [("R", "x"), ("W", "x")]
    isolation : str
        Isolation level to analyze under.
    """
    level = isolation.upper().replace(" ", "_")
    prevented = ISOLATION_LEVELS.get(level, set())

    conflicts = []
    possible_anomalies = []

    # Build read/write sets per transaction
    n_txn = len(transactions)
    read_sets: List[Set[str]] = [set() for _ in range(n_txn)]
    write_sets: List[Set[str]] = [set() for _ in range(n_txn)]

    for i, txn in enumerate(transactions):
        for op, item in txn:
            if op.upper() == "R":
                read_sets[i].add(item)
            elif op.upper() == "W":
                write_sets[i].add(item)

    # Detect conflicts between transaction pairs
    for i in range(n_txn):
        for j in range(i + 1, n_txn):
            # Write-Write conflict
            ww = write_sets[i] & write_sets[j]
            if ww:
                conflicts.append(
                    f"T{i+1}-T{j+1} WW conflict on {ww}"
                )

            # Read-Write conflict (both directions)
            rw_ij = read_sets[i] & write_sets[j]
            rw_ji = read_sets[j] & write_sets[i]
            if rw_ij:
                conflicts.append(
                    f"T{i+1} reads, T{j+1} writes {rw_ij}"
                )
            if rw_ji:
                conflicts.append(
                    f"T{j+1} reads, T{i+1} writes {rw_ji}"
                )

            # Check for specific anomaly patterns
            # Lost update: both read and write same item
            if (read_sets[i] & write_sets[i] & read_sets[j] & write_sets[j]):
                if "lost_update" not in prevented:
                    items = read_sets[i] & write_sets[i] & read_sets[j] & write_sets[j]
                    possible_anomalies.append(
                        f"Lost update on {items} — both T{i+1} and T{j+1} "
                        f"read-then-write"
                    )

            # Write skew: T1 reads x writes y, T2 reads y writes x
            for item_a in read_sets[i] & write_sets[j]:
                for item_b in read_sets[j] & write_sets[i]:
                    if item_a != item_b and "write_skew" not in prevented:
                        possible_anomalies.append(
                            f"Write skew risk: T{i+1} reads {item_a} writes "
                            f"{item_b}, T{j+1} reads {item_b} writes {item_a}"
                        )

            # Non-repeatable read
            if rw_ij and "non_repeatable_read" not in prevented:
                # Check if T1 reads item more than once (or just reads what T2 writes)
                for item in rw_ij:
                    reads_of_item = sum(1 for op, it in transactions[i] if op.upper() == "R" and it == item)
                    if reads_of_item >= 1:
                        possible_anomalies.append(
                            f"Non-repeatable read risk on '{item}': T{i+1} "
                            f"reads, T{j+1} may modify between reads"
                        )

    # Check serializability: is there a cycle in the conflict graph?
    # Build adjacency for conflict serialization graph
    adj: Dict[int, Set[int]] = {i: set() for i in range(n_txn)}
    for i in range(n_txn):
        for j in range(n_txn):
            if i == j:
                continue
            # T_i → T_j if T_i has an operation that conflicts with T_j
            # and T_i's operation comes "first" in the schedule
            if (read_sets[i] & write_sets[j]) or (write_sets[i] & write_sets[j]):
                adj[i].add(j)
            if (write_sets[i] & read_sets[j]):
                adj[i].add(j)

    has_cycle = _detect_cycle(adj, n_txn)

    explanation = (
        "The conflict graph has a cycle — this schedule is NOT serializable. "
        "Anomalies are possible under non-serializable isolation levels."
        if has_cycle else
        "No cycle in conflict graph — a serializable ordering exists."
    )

    return ScheduleReport(
        transactions=transactions,
        isolation_level=level,
        conflicts=conflicts,
        possible_anomalies=possible_anomalies,
        serializable=not has_cycle,
        explanation=explanation,
    )


def list_anomalies() -> List[str]:
    """List all known concurrency anomalies with descriptions."""
    lines = []
    for key, anom in ANOMALIES.items():
        lines.append(f"{key}: {anom.name} — {anom.description[:80]}")
    return lines


def _detect_cycle(adj: Dict[int, Set[int]], n: int) -> bool:
    """Detect cycle in directed graph using DFS."""
    WHITE, GRAY, BLACK = 0, 1, 2
    color = [WHITE] * n

    def dfs(u: int) -> bool:
        color[u] = GRAY
        for v in adj.get(u, set()):
            if color[v] == GRAY:
                return True
            if color[v] == WHITE and dfs(v):
                return True
        color[u] = BLACK
        return False

    return any(color[u] == WHITE and dfs(u) for u in range(n))
