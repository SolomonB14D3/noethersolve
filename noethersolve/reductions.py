"""
noethersolve.reductions — Computational reduction chain validator.

Validates chains of computational reductions for type consistency, transitivity,
and correctness. Catches common errors in complexity theory arguments where
people mix incompatible reduction types or make circular arguments.

Checks:
  - Transitivity: does the chain compose correctly, and what type results?
  - Type consistency: does the claimed reduction type match what the chain supports?
  - Circularity: does the chain contain a cycle (only valid if all nodes equivalent)?
  - Hardness preservation: does the reduction preserve the claimed complexity class?
  - Known reduction validation: cross-reference against established reductions.

Usage:
    from noethersolve.reductions import (
        validate_chain, check_reduction, strongest_reduction,
        list_known_reductions, get_reduction_info,
        ChainReport, ChainIssue, ReductionResult,
    )

    # Validate a reduction chain
    chain = [
        ("3-SAT", "many-one", "CLIQUE"),
        ("CLIQUE", "many-one", "VERTEX-COVER"),
    ]
    report = validate_chain(chain)
    print(report)
    # PASS — chain is transitive, type-consistent, acyclic

    # Check a single reduction step
    result = check_reduction("3-SAT", "many-one", "CLIQUE")
    print(result)
    # KNOWN — Karp 1972

    # What's the strongest type a chain supports?
    chain = [("SAT", "many-one", "3-SAT"), ("3-SAT", "Turing", "CLIQUE")]
    print(strongest_reduction(chain))
    # "Turing" (weakened from many-one by the second step)

    # List known reductions
    for r in list_known_reductions("3-SAT"):
        print(r)

    # Get properties of a reduction type
    info = get_reduction_info("many-one")
    print(info["preserves"])  # ['NP-hardness', 'NP-completeness']
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ─── Reduction type hierarchy and properties ─────────────────────────────────

# Power ordering: index 0 = most restrictive, higher = more powerful.
# If A ≤_x B and B ≤_y C, the composed reduction is max(x, y) in power.
_POWER_ORDER = [
    "first-order",
    "log-space",
    "many-one",
    "truth-table",
    "Turing",
]

_POWER_RANK = {name: i for i, name in enumerate(_POWER_ORDER)}

# Types outside the main hierarchy (not directly comparable)
_SPECIAL_TYPES = {"polynomial-time counting", "randomized", "Levin"}

REDUCTION_TYPES = {
    "many-one": {
        "aliases": ["Karp", "polynomial-time many-one", "≤_m", "≤_p"],
        "description": (
            "A ≤_m B. Polynomial-time computable function f such that "
            "x in A iff f(x) in B."
        ),
        "transitive": True,
        "preserves": ["NP-hardness", "NP-completeness"],
        "resource": "polynomial-time",
        "power_rank": _POWER_RANK["many-one"],
    },
    "Turing": {
        "aliases": ["Cook", "oracle", "≤_T"],
        "description": (
            "A ≤_T B. Polynomial-time oracle machine that decides A "
            "using oracle for B."
        ),
        "transitive": True,
        "preserves": ["NP-hardness", "PSPACE-hardness"],
        "does_not_preserve": ["NP-completeness"],
        "resource": "polynomial-time",
        "power_rank": _POWER_RANK["Turing"],
    },
    "truth-table": {
        "aliases": ["≤_tt", "non-adaptive"],
        "description": (
            "A ≤_tt B. Like Turing but queries are non-adaptive "
            "(all asked before any answered)."
        ),
        "transitive": True,
        "preserves": ["NP-hardness"],
        "resource": "polynomial-time",
        "power_rank": _POWER_RANK["truth-table"],
    },
    "log-space": {
        "aliases": ["≤_L", "logspace"],
        "description": (
            "A ≤_L B. Log-space computable reduction. Used for "
            "NL-completeness and P-completeness."
        ),
        "transitive": True,
        "preserves": ["NL-completeness", "P-completeness"],
        "resource": "log-space",
        "power_rank": _POWER_RANK["log-space"],
    },
    "first-order": {
        "aliases": ["AC0", "≤_FO", "FO"],
        "description": (
            "A ≤_{FO} B. Computable by AC0 circuits. Used for "
            "problems within NC1. Very restrictive."
        ),
        "transitive": True,
        "preserves": ["AC0-hardness"],
        "resource": "AC0",
        "power_rank": _POWER_RANK["first-order"],
    },
    "polynomial-time counting": {
        "aliases": ["#P", "parsimonious", "counting"],
        "description": (
            "#A ≤ #B. For counting problems (#P). Parsimonious "
            "reductions preserve solution counts exactly."
        ),
        "transitive": True,
        "preserves": ["#P-hardness", "#P-completeness"],
        "resource": "polynomial-time",
        "power_rank": None,
    },
    "randomized": {
        "aliases": ["≤_R", "probabilistic"],
        "description": (
            "A ≤_R B. Reduction uses randomness. Used in some "
            "BPP/RP arguments. Not always composable cleanly."
        ),
        "transitive": False,
        "preserves": ["BPP-hardness"],
        "resource": "polynomial-time (randomized)",
        "power_rank": None,
    },
    "Levin": {
        "aliases": ["≤_Levin", "witness-preserving"],
        "description": (
            "A ≤_L B with polynomial overhead on witnesses. "
            "Preserves NP search problem structure."
        ),
        "transitive": True,
        "preserves": ["NP-search-hardness"],
        "resource": "polynomial-time",
        "power_rank": None,
    },
}

# Build alias lookup
_ALIAS_MAP = {}
for rtype, info in REDUCTION_TYPES.items():
    _ALIAS_MAP[rtype.lower()] = rtype
    for alias in info["aliases"]:
        _ALIAS_MAP[alias.lower()] = rtype


# ─── Known reductions ────────────────────────────────────────────────────────

KNOWN_REDUCTIONS = [
    {
        "from": "SAT",
        "to": "3-SAT",
        "type": "many-one",
        "reference": "Cook-Levin + gadgets",
        "note": "Polynomial-time many-one reduction via clause splitting.",
    },
    {
        "from": "3-SAT",
        "to": "CLIQUE",
        "type": "many-one",
        "reference": "Karp 1972",
        "note": "One of Karp's 21 NP-complete problems.",
    },
    {
        "from": "3-SAT",
        "to": "VERTEX-COVER",
        "type": "many-one",
        "reference": "Karp 1972",
        "note": "Via INDEPENDENT-SET complement.",
    },
    {
        "from": "3-SAT",
        "to": "HAMILTONIAN-PATH",
        "type": "many-one",
        "reference": "Karp 1972",
        "note": "Gadget construction mapping clauses to graph paths.",
    },
    {
        "from": "3-SAT",
        "to": "SUBSET-SUM",
        "type": "many-one",
        "reference": "Karp 1972",
        "note": "Encoding satisfying assignments as subset sum targets.",
    },
    {
        "from": "CIRCUIT-SAT",
        "to": "SAT",
        "type": "many-one",
        "reference": "Tseitin transformation",
        "note": "Each gate becomes a clause; linear blowup.",
    },
    {
        "from": "3-SAT",
        "to": "INDEPENDENT-SET",
        "type": "many-one",
        "reference": "Karp 1972",
        "note": "Clause gadgets with conflict edges.",
    },
    {
        "from": "HAMILTONIAN-PATH",
        "to": "HAMILTONIAN-CYCLE",
        "type": "many-one",
        "reference": "Standard reduction",
        "note": "Add vertex connected to start and end.",
    },
    {
        "from": "CLIQUE",
        "to": "INDEPENDENT-SET",
        "type": "many-one",
        "reference": "Complement graph",
        "note": "Clique in G iff independent set in complement(G).",
    },
    {
        "from": "s-t-CONNECTIVITY",
        "to": "REACHABILITY",
        "type": "log-space",
        "reference": "NL-completeness",
        "note": "Log-space reduction; s-t-CONNECTIVITY is NL-complete.",
    },
    {
        "from": "SAT",
        "to": "#SAT",
        "type": "polynomial-time counting",
        "reference": "Counting is at least as hard as decision",
        "note": "If you can count solutions, you can decide if count > 0.",
    },
]

# CVP is P-complete under log-space reductions (stored as metadata, not a
# from->to pair since P-completeness is a property, not a specific reduction)
KNOWN_COMPLETENESS = {
    "SAT": {"class": "NP-complete", "reduction": "many-one"},
    "3-SAT": {"class": "NP-complete", "reduction": "many-one"},
    "CLIQUE": {"class": "NP-complete", "reduction": "many-one"},
    "VERTEX-COVER": {"class": "NP-complete", "reduction": "many-one"},
    "HAMILTONIAN-PATH": {"class": "NP-complete", "reduction": "many-one"},
    "HAMILTONIAN-CYCLE": {"class": "NP-complete", "reduction": "many-one"},
    "SUBSET-SUM": {"class": "NP-complete", "reduction": "many-one"},
    "INDEPENDENT-SET": {"class": "NP-complete", "reduction": "many-one"},
    "CIRCUIT-SAT": {"class": "NP-complete", "reduction": "many-one"},
    "s-t-CONNECTIVITY": {"class": "NL-complete", "reduction": "log-space"},
    "REACHABILITY": {"class": "NL-complete", "reduction": "log-space"},
    "CVP": {"class": "P-complete", "reduction": "log-space"},
    "#SAT": {"class": "#P-complete", "reduction": "polynomial-time counting"},
    "PERMANENT": {"class": "#P-complete", "reduction": "polynomial-time counting"},
}


# ─── Normalize helper ────────────────────────────────────────────────────────

def _normalize_type(reduction_type: str) -> str:
    """Resolve aliases to canonical reduction type names."""
    key = reduction_type.strip().lower()
    if key in _ALIAS_MAP:
        return _ALIAS_MAP[key]
    raise ValueError(
        f"Unknown reduction type: '{reduction_type}'. "
        f"Known types: {', '.join(sorted(REDUCTION_TYPES.keys()))}"
    )


def _normalize_problem(name: str) -> str:
    """Normalize problem name for comparison (uppercase, strip whitespace)."""
    return name.strip().upper().replace(" ", "-")


# ─── Dataclasses ──────────────────────────────────────────────────────────────

@dataclass
class ChainIssue:
    """A single issue found in a reduction chain."""
    check: str              # TRANSITIVITY, TYPE_CONSISTENCY, CIRCULARITY,
                            # HARDNESS_PRESERVATION, KNOWN_REDUCTION
    severity: str           # HIGH, MODERATE, LOW, INFO
    description: str
    step: Optional[int] = None      # which step in the chain (0-indexed), if applicable
    details: Dict = field(default_factory=dict)

    def __str__(self):
        step_str = f" (step {self.step})" if self.step is not None else ""
        return f"  [{self.severity}] {self.check}{step_str}: {self.description}"


@dataclass
class ReductionResult:
    """Result of checking a single reduction step."""
    problem_a: str
    reduction_type: str
    problem_b: str
    known: bool                     # whether this is a known valid reduction
    reference: Optional[str] = None
    note: Optional[str] = None
    issues: List[ChainIssue] = field(default_factory=list)

    def __str__(self):
        status = "KNOWN" if self.known else "UNKNOWN"
        lines = [f"{self.problem_a} <={self.reduction_type} {self.problem_b} [{status}]"]
        if self.reference:
            lines.append(f"  Reference: {self.reference}")
        if self.note:
            lines.append(f"  Note: {self.note}")
        for issue in self.issues:
            lines.append(str(issue))
        return "\n".join(lines)


@dataclass
class ChainReport:
    """Result of validate_chain()."""
    verdict: str                    # PASS, WARN, or FAIL
    chain_length: int
    effective_type: str             # strongest reduction the chain supports
    issues: List[ChainIssue]
    steps: List[ReductionResult]
    warnings: List[str]

    def __str__(self):
        lines = []
        lines.append(f"{'=' * 60}")
        lines.append(f"  Reduction Chain Validation: {self.verdict}")
        lines.append(f"{'=' * 60}")
        lines.append(f"  Steps: {self.chain_length}")
        lines.append(f"  Effective type: {self.effective_type}")
        lines.append("")

        # Chain visualization
        if self.steps:
            lines.append("  Chain:")
            for i, step in enumerate(self.steps):
                status = "KNOWN" if step.known else "?"
                lines.append(
                    f"    {i}: {step.problem_a} <={step.reduction_type} "
                    f"{step.problem_b} [{status}]"
                )
            lines.append("")

        # Issues sorted by severity
        high = [i for i in self.issues if i.severity == "HIGH"]
        moderate = [i for i in self.issues if i.severity == "MODERATE"]
        low = [i for i in self.issues if i.severity == "LOW"]
        info = [i for i in self.issues if i.severity == "INFO"]

        if high:
            lines.append(f"  ERRORS ({len(high)}):")
            for issue in high:
                lines.append(str(issue))
        if moderate:
            lines.append(f"  WARNINGS ({len(moderate)}):")
            for issue in moderate:
                lines.append(str(issue))
        if low:
            lines.append(f"  NOTES ({len(low)}):")
            for issue in low:
                lines.append(str(issue))
        if info:
            lines.append(f"  INFO ({len(info)}):")
            for issue in info:
                lines.append(str(issue))

        if self.warnings:
            lines.append("")
            lines.append("  Suggestions:")
            for w in self.warnings:
                lines.append(f"    - {w}")

        lines.append(f"{'=' * 60}")
        return "\n".join(lines)

    @property
    def passed(self) -> bool:
        return self.verdict == "PASS"


# ─── Core functions ──────────────────────────────────────────────────────────

def _find_known_reduction(
    problem_a: str, reduction_type: str, problem_b: str
) -> Optional[dict]:
    """Look up a reduction in the known reductions table."""
    a_norm = _normalize_problem(problem_a)
    b_norm = _normalize_problem(problem_b)
    for kr in KNOWN_REDUCTIONS:
        if (_normalize_problem(kr["from"]) == a_norm
                and _normalize_problem(kr["to"]) == b_norm
                and _normalize_type(kr["type"]) == reduction_type):
            return kr
    return None


def _compose_types(type_a: str, type_b: str) -> str:
    """Determine the resulting reduction type when composing two steps.

    If A <=_x B and B <=_y C, then A <=_z C where z is the weakest
    (most powerful) of x and y within the hierarchy. For types outside
    the hierarchy, composition may not be well-defined.

    Returns the canonical name of the composed type.
    """
    rank_a = REDUCTION_TYPES[type_a].get("power_rank")
    rank_b = REDUCTION_TYPES[type_b].get("power_rank")

    # Both in the main hierarchy: take the weaker (higher rank)
    if rank_a is not None and rank_b is not None:
        if rank_a >= rank_b:
            return type_a
        return type_b

    # One in hierarchy, one special: if one is Turing-level or below,
    # and the other is polynomial-time based, Turing subsumes
    if rank_a is not None and rank_b is None:
        if type_b in ("polynomial-time counting", "Levin"):
            # These are polynomial-time reductions; Turing subsumes
            return "Turing" if rank_a >= _POWER_RANK["Turing"] else "Turing"
        return type_b  # randomized or other special — keep as-is
    if rank_b is not None and rank_a is None:
        if type_a in ("polynomial-time counting", "Levin"):
            return "Turing" if rank_b >= _POWER_RANK["Turing"] else "Turing"
        return type_a

    # Both special — not generally composable
    if type_a == type_b:
        return type_a
    return "Turing"  # safest fallback


def check_reduction(
    problem_a: str, reduction_type: str, problem_b: str
) -> ReductionResult:
    """Check a single reduction step for validity.

    Args:
        problem_a: the problem being reduced from.
        reduction_type: the type of reduction (e.g., "many-one", "Turing").
        problem_b: the problem being reduced to.

    Returns:
        ReductionResult with known status, reference, and any issues.
    """
    rtype = _normalize_type(reduction_type)
    issues = []

    # Look up in known reductions
    kr = _find_known_reduction(problem_a, rtype, problem_b)
    if kr is not None:
        return ReductionResult(
            problem_a=problem_a,
            reduction_type=rtype,
            problem_b=problem_b,
            known=True,
            reference=kr.get("reference"),
            note=kr.get("note"),
            issues=[],
        )

    # Check if the reverse is known (common mistake: direction wrong)
    kr_rev = _find_known_reduction(problem_b, rtype, problem_a)
    if kr_rev is not None:
        issues.append(ChainIssue(
            check="KNOWN_REDUCTION",
            severity="HIGH",
            description=(
                f"The reduction {problem_b} <={rtype} {problem_a} is known "
                f"({kr_rev.get('reference', '?')}), but you have it backwards. "
                f"Reductions are not symmetric."
            ),
        ))

    # Check if a different type is known for this pair
    a_norm = _normalize_problem(problem_a)
    b_norm = _normalize_problem(problem_b)
    for kr_other in KNOWN_REDUCTIONS:
        if (_normalize_problem(kr_other["from"]) == a_norm
                and _normalize_problem(kr_other["to"]) == b_norm
                and _normalize_type(kr_other["type"]) != rtype):
            actual_type = _normalize_type(kr_other["type"])
            # Is the claimed type weaker or stronger?
            claimed_rank = REDUCTION_TYPES[rtype].get("power_rank")
            actual_rank = REDUCTION_TYPES[actual_type].get("power_rank")
            if claimed_rank is not None and actual_rank is not None:
                if claimed_rank < actual_rank:
                    # Claiming stronger than known
                    issues.append(ChainIssue(
                        check="TYPE_CONSISTENCY",
                        severity="HIGH",
                        description=(
                            f"Known reduction is {actual_type} "
                            f"({kr_other.get('reference', '?')}), but you claim "
                            f"{rtype} which is strictly stronger. Not established."
                        ),
                    ))
                elif claimed_rank > actual_rank:
                    # Claiming weaker than known — valid but wasteful
                    issues.append(ChainIssue(
                        check="TYPE_CONSISTENCY",
                        severity="INFO",
                        description=(
                            f"Known reduction is {actual_type} "
                            f"({kr_other.get('reference', '?')}), which is "
                            f"stronger than your claimed {rtype}. Valid but "
                            f"you could use the stronger type."
                        ),
                    ))

    return ReductionResult(
        problem_a=problem_a,
        reduction_type=rtype,
        problem_b=problem_b,
        known=False,
        reference=None,
        note=None,
        issues=issues,
    )


def strongest_reduction(chain: List[Tuple[str, str, str]]) -> str:
    """Determine the strongest (most specific) reduction type the chain supports.

    The chain weakens to the most powerful type present, because composition
    through a Turing step means the whole chain is at best Turing, even if
    other steps are many-one.

    Args:
        chain: list of (problem_a, reduction_type, problem_b) tuples.

    Returns:
        Canonical name of the strongest reduction type the chain supports.

    Raises:
        ValueError: if the chain is empty or contains unknown types.
    """
    if not chain:
        raise ValueError("Chain is empty.")

    _, first_type, _ = chain[0]
    result_type = _normalize_type(first_type)

    for _, rtype_raw, _ in chain[1:]:
        rtype = _normalize_type(rtype_raw)
        result_type = _compose_types(result_type, rtype)

    return result_type


def validate_chain(chain: List[Tuple[str, str, str]]) -> ChainReport:
    """Validate a chain of computational reductions.

    Checks transitivity, type consistency, circularity, hardness preservation,
    and cross-references against known reductions.

    Args:
        chain: list of (problem_a, reduction_type, problem_b) tuples.
            Example: [("3-SAT", "many-one", "CLIQUE"),
                      ("CLIQUE", "many-one", "VERTEX-COVER")]

    Returns:
        ChainReport with verdict (PASS/WARN/FAIL), per-step results, issues.
    """
    issues: List[ChainIssue] = []
    warnings: List[str] = []
    steps: List[ReductionResult] = []

    if not chain:
        return ChainReport(
            verdict="PASS",
            chain_length=0,
            effective_type="",
            issues=[],
            steps=[],
            warnings=["Empty chain — nothing to validate."],
        )

    # ── Normalize all types ──────────────────────────────────────────────
    normalized_chain = []
    for i, (a, rtype_raw, b) in enumerate(chain):
        try:
            rtype = _normalize_type(rtype_raw)
        except ValueError as e:
            issues.append(ChainIssue(
                check="TYPE_CONSISTENCY",
                severity="HIGH",
                step=i,
                description=str(e),
            ))
            rtype = rtype_raw  # keep going with raw for reporting
        normalized_chain.append((a, rtype, b))

    # ── Check each step individually ─────────────────────────────────────
    for i, (a, rtype, b) in enumerate(normalized_chain):
        result = check_reduction(a, rtype, b)
        steps.append(result)
        # Propagate step-level issues with step index
        for issue in result.issues:
            issue.step = i
            issues.append(issue)

    # ── TRANSITIVITY CHECK ───────────────────────────────────────────────
    # Verify chain links: problem_b of step i == problem_a of step i+1
    for i in range(len(normalized_chain) - 1):
        _, _, b_curr = normalized_chain[i]
        a_next, _, _ = normalized_chain[i + 1]
        if _normalize_problem(b_curr) != _normalize_problem(a_next):
            issues.append(ChainIssue(
                check="TRANSITIVITY",
                severity="HIGH",
                step=i,
                description=(
                    f"Chain breaks at step {i}->{i+1}: "
                    f"'{b_curr}' != '{a_next}'. "
                    f"For transitivity, step {i} must reduce to the same "
                    f"problem that step {i+1} reduces from."
                ),
            ))

    # Check for non-transitive types in chain
    for i, (a, rtype, b) in enumerate(normalized_chain):
        if rtype in REDUCTION_TYPES and not REDUCTION_TYPES[rtype]["transitive"]:
            issues.append(ChainIssue(
                check="TRANSITIVITY",
                severity="MODERATE",
                step=i,
                description=(
                    f"Reduction type '{rtype}' is not guaranteed to be "
                    f"transitive. Chain composition through this step may "
                    f"not preserve the reduction relationship."
                ),
            ))

    # ── TYPE CONSISTENCY CHECK ───────────────────────────────────────────
    # Compute effective type of the whole chain
    try:
        effective_type = strongest_reduction(chain)
    except ValueError:
        effective_type = "unknown"

    # Check for type weakening (common mistake: claiming many-one from a
    # chain that includes a Turing step)
    types_in_chain = set()
    for _, rtype, _ in normalized_chain:
        if rtype in REDUCTION_TYPES:
            types_in_chain.add(rtype)

    # If someone has both many-one and Turing steps, note the weakening
    if len(types_in_chain) > 1:
        ranks = [
            (t, REDUCTION_TYPES[t].get("power_rank"))
            for t in types_in_chain
            if REDUCTION_TYPES.get(t, {}).get("power_rank") is not None
        ]
        if ranks:
            ranks.sort(key=lambda x: x[1])
            strongest_name, _ = ranks[0]
            weakest_name, _ = ranks[-1]
            if strongest_name != weakest_name:
                issues.append(ChainIssue(
                    check="TYPE_CONSISTENCY",
                    severity="LOW",
                    description=(
                        f"Chain mixes reduction types: "
                        f"{', '.join(t for t, _ in ranks)}. "
                        f"The chain is only as strong as its weakest link: "
                        f"{effective_type}."
                    ),
                ))

    # ── CIRCULARITY CHECK ────────────────────────────────────────────────
    # Build the set of problems appearing in the chain
    problems_in_order = []
    for a, _, b in normalized_chain:
        problems_in_order.append(_normalize_problem(a))
    if normalized_chain:
        problems_in_order.append(_normalize_problem(normalized_chain[-1][2]))

    # Check if any problem appears more than once (cycle)
    seen = {}
    for idx, p in enumerate(problems_in_order):
        if p in seen:
            cycle_start = seen[p]
            cycle_problems = problems_in_order[cycle_start:idx + 1]
            unique_in_cycle = set(cycle_problems)

            if len(unique_in_cycle) == 1:
                # Self-loop: A <= A (trivially true but suspicious)
                issues.append(ChainIssue(
                    check="CIRCULARITY",
                    severity="MODERATE",
                    description=(
                        f"Self-loop detected: {p} reduces to itself. "
                        f"Trivially true but likely an error."
                    ),
                ))
            else:
                # Real cycle: A <= B <= ... <= A
                # Only valid if all problems in cycle are equivalent
                issues.append(ChainIssue(
                    check="CIRCULARITY",
                    severity="HIGH",
                    description=(
                        f"Cycle detected: {' -> '.join(cycle_problems)}. "
                        f"This is only valid if all problems in the cycle are "
                        f"equivalent under {effective_type} reductions. "
                        f"If you intend to prove equivalence, both directions "
                        f"must be independently established."
                    ),
                    details={"cycle": cycle_problems},
                ))
        else:
            seen[p] = idx

    # ── HARDNESS PRESERVATION CHECK ──────────────────────────────────────
    # Check if the reduction chain actually preserves claimed hardness
    if effective_type in REDUCTION_TYPES:
        set(REDUCTION_TYPES[effective_type].get("preserves", []))
        does_not_preserve = set(
            REDUCTION_TYPES[effective_type].get("does_not_preserve", [])
        )

        # Check source problem's known completeness
        if normalized_chain:
            source = _normalize_problem(normalized_chain[0][0])
            target = _normalize_problem(normalized_chain[-1][2])

            source_info = KNOWN_COMPLETENESS.get(source)
            target_info = KNOWN_COMPLETENESS.get(target)

            if source_info and target_info:
                source_class = source_info["class"]
                target_class = target_info["class"]

                # NP-completeness via Turing reductions is a common error
                if ("NP-completeness" in does_not_preserve
                        and "NP-complete" in source_class
                        and "NP-complete" in target_class):
                    issues.append(ChainIssue(
                        check="HARDNESS_PRESERVATION",
                        severity="MODERATE",
                        description=(
                            f"Both {source} and {target} are NP-complete, "
                            f"but Turing reductions do not preserve "
                            f"NP-completeness (only NP-hardness). "
                            f"NP-completeness proofs require many-one reductions "
                            f"by convention."
                        ),
                    ))

            # Check: reducing from harder class to easier class is suspicious
            # (e.g., claiming #P problem reduces to NP problem via many-one)
            if source_info and target_info:
                _class_order = {
                    "NL-complete": 0,
                    "P-complete": 1,
                    "NP-complete": 2,
                    "#P-complete": 3,
                }
                src_rank = _class_order.get(source_info["class"])
                tgt_rank = _class_order.get(target_info["class"])
                if src_rank is not None and tgt_rank is not None:
                    if src_rank > tgt_rank:
                        issues.append(ChainIssue(
                            check="HARDNESS_PRESERVATION",
                            severity="HIGH",
                            description=(
                                f"Reducing {source} ({source_info['class']}) to "
                                f"{target} ({target_info['class']}) would imply "
                                f"the source class collapses to the target class. "
                                f"This is almost certainly an error — check the "
                                f"direction of your reduction."
                            ),
                        ))

    # ── KNOWN REDUCTION CHECK (chain-level) ──────────────────────────────
    # If the overall chain connects two problems with a known direct reduction,
    # note it as informational
    if len(normalized_chain) > 1:
        first_a = normalized_chain[0][0]
        last_b = normalized_chain[-1][2]
        for kr in KNOWN_REDUCTIONS:
            if (_normalize_problem(kr["from"]) == _normalize_problem(first_a)
                    and _normalize_problem(kr["to"]) == _normalize_problem(last_b)):
                known_type = _normalize_type(kr["type"])
                issues.append(ChainIssue(
                    check="KNOWN_REDUCTION",
                    severity="INFO",
                    description=(
                        f"A direct {known_type} reduction from {first_a} to "
                        f"{last_b} is already known ({kr.get('reference', '?')}). "
                        f"Your chain is valid but the direct reduction may be simpler."
                    ),
                ))
                break

    # ── Verdict ──────────────────────────────────────────────────────────
    has_high = any(i.severity == "HIGH" for i in issues)
    has_moderate = any(i.severity == "MODERATE" for i in issues)

    if has_high:
        verdict = "FAIL"
    elif has_moderate:
        verdict = "WARN"
    else:
        verdict = "PASS"

    # ── Suggestions ──────────────────────────────────────────────────────
    if has_high:
        circularity_issues = [i for i in issues if i.check == "CIRCULARITY" and i.severity == "HIGH"]
        if circularity_issues:
            warnings.append(
                "Circular reductions only establish equivalence, not hardness. "
                "To prove hardness, you need a reduction FROM a known-hard "
                "problem, not a cycle."
            )
        transitivity_issues = [i for i in issues if i.check == "TRANSITIVITY" and i.severity == "HIGH"]
        if transitivity_issues:
            warnings.append(
                "Chain is broken — ensure each step's target matches the "
                "next step's source."
            )
        direction_issues = [
            i for i in issues
            if i.check == "KNOWN_REDUCTION" and i.severity == "HIGH"
        ]
        if direction_issues:
            warnings.append(
                "Reduction direction appears backwards. A <=_m B means "
                "'A is no harder than B', equivalently 'B is at least as "
                "hard as A'."
            )

    return ChainReport(
        verdict=verdict,
        chain_length=len(normalized_chain),
        effective_type=effective_type,
        issues=issues,
        steps=steps,
        warnings=warnings,
    )


def list_known_reductions(problem: str = None) -> List[dict]:
    """List all known reductions, optionally filtered by problem name.

    Args:
        problem: if provided, only return reductions involving this problem
            (as source or target). Case-insensitive.

    Returns:
        List of dicts with keys: from, to, type, reference, note.
    """
    if problem is None:
        return [dict(kr) for kr in KNOWN_REDUCTIONS]

    p_norm = _normalize_problem(problem)
    results = []
    for kr in KNOWN_REDUCTIONS:
        if (_normalize_problem(kr["from"]) == p_norm
                or _normalize_problem(kr["to"]) == p_norm):
            results.append(dict(kr))
    return results


def get_reduction_info(reduction_type: str) -> dict:
    """Get properties of a reduction type.

    Args:
        reduction_type: canonical name or alias (e.g., "many-one", "Karp").

    Returns:
        Dict with keys: aliases, description, transitive, preserves,
        resource, power_rank, and optionally does_not_preserve.

    Raises:
        ValueError: if the reduction type is unknown.
    """
    rtype = _normalize_type(reduction_type)
    return dict(REDUCTION_TYPES[rtype])
