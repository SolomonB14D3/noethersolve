"""
noethersolve.complexity — Computational complexity class relationship auditor.

Validates claims about complexity class relationships against the known
hierarchy. Catches invalid inclusions, false separations, incorrect
completeness claims, and claims that would imply collapses of established
separations.

Hardcoded data covers ~20 complexity classes with known inclusions, proven
separations, oracle separations, completeness results, and open questions
drawn from the standard references (Arora & Barak, Sipser, Goldreich).

Usage:
    from noethersolve.complexity import audit_complexity, ComplexityReport

    report = audit_complexity([
        "P = NP",
        "SAT in P",
        "GI is NP-complete",
        "P ⊆ PSPACE",
    ])
    print(report)
    # Shows per-claim diagnostics, severity levels, and overall verdict

    # Individual checks
    from noethersolve.complexity import check_inclusion, check_completeness

    result = check_inclusion("P", "NP")
    print(result)  # ESTABLISHED: P ⊆ NP

    result = check_completeness("SAT", "P")
    print(result)  # Would imply P = NP (open)

    # Class info
    from noethersolve.complexity import get_class_info
    info = get_class_info("NP")
    print(info["contained_in"])  # classes that NP is known to be inside
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple


# ─── Complexity class hierarchy data ─────────────────────────────────────────

# Known inclusions: (A, B) means A ⊆ B is established.
# These are unconditional, proven containments.
KNOWN_INCLUSIONS: Set[Tuple[str, str]] = {
    # Deterministic time hierarchy
    ("L", "NL"),
    ("NL", "P"),
    ("P", "NP"),
    ("P", "coNP"),
    ("NP", "PSPACE"),
    ("coNP", "PSPACE"),
    ("PSPACE", "EXP"),
    ("EXP", "NEXP"),
    # Space
    ("L", "P"),
    ("NL", "P"),
    ("L", "NL"),
    ("L", "PSPACE"),
    ("NL", "PSPACE"),
    ("PSPACE", "EXPSPACE"),
    # PSPACE = NPSPACE (Savitch's theorem)
    ("PSPACE", "NPSPACE"),
    ("NPSPACE", "PSPACE"),
    # Probabilistic
    ("P", "BPP"),
    ("P", "RP"),
    ("P", "coRP"),
    ("RP", "BPP"),
    ("coRP", "BPP"),
    ("BPP", "PSPACE"),
    ("RP", "NP"),
    ("coRP", "coNP"),
    # Quantum
    ("P", "BQP"),
    ("BPP", "BQP"),
    ("BQP", "PSPACE"),
    # Circuit complexity
    ("AC0", "NC1"),
    ("NC1", "NC"),
    ("NC", "P"),
    ("AC0", "NC"),
    ("AC0", "P"),
    # Counting
    ("NP", "PP"),
    ("coNP", "PP"),
    ("PP", "PSPACE"),
    # Polynomial hierarchy
    ("P", "PH"),
    ("NP", "PH"),
    ("coNP", "PH"),
    ("PH", "PSPACE"),
    # Advice
    ("P", "P/poly"),
    ("BPP", "P/poly"),
    # Nondeterministic exponential
    ("NP", "NEXP"),
    ("EXP", "NEXP"),
    # Graph Isomorphism (Babai 2015: quasipolynomial time; in NP ∩ coAM)
    ("GI", "NP"),
    ("GI", "coAM"),
    ("GI", "PSPACE"),
    ("GI", "EXP"),
    ("GI", "NEXP"),
}

# Known equalities
KNOWN_EQUALITIES: Set[Tuple[str, str]] = {
    ("PSPACE", "NPSPACE"),   # Savitch's theorem
    ("PSPACE", "APTIME"),    # PSPACE = APTIME (alternation)
    ("NL", "coNL"),          # Immerman-Szelepcsenyi theorem
    ("IP", "PSPACE"),        # Shamir's theorem
}

# Known strict separations: (A, B) means A ⊊ B is proven.
KNOWN_SEPARATIONS: Set[Tuple[str, str]] = {
    ("AC0", "NC1"),          # Parity not in AC0 (Furst-Saxe-Sipser / Hastad)
    ("P", "EXP"),            # Time hierarchy theorem
    ("NP", "NEXP"),          # Nondeterministic time hierarchy theorem
    ("L", "PSPACE"),         # Space hierarchy theorem
    ("NL", "PSPACE"),        # Space hierarchy theorem
    ("EXP", "EXPSPACE"),     # Space hierarchy theorem
}

# Oracle separations: relationships where relativizing techniques cannot resolve.
# (A, B) means there exist oracles O1, O2 such that A^O1 = B^O1 and A^O2 ≠ B^O2.
ORACLE_SEPARATIONS: Set[Tuple[str, str]] = {
    ("P", "NP"),             # Baker-Gill-Solovay 1975
    ("P", "BPP"),
    ("NP", "coNP"),
    ("BQP", "NP"),
    ("NP", "BQP"),
}

# Open questions: pairs (A, B) where A ⊆ B is not known to be strict or equal.
# These are the famous unresolved containments.
OPEN_QUESTIONS: Set[Tuple[str, str]] = {
    ("P", "NP"),             # P vs NP
    ("P", "BPP"),            # Is BPP = P? (conjectured yes)
    ("P", "PSPACE"),         # P vs PSPACE
    ("NP", "coNP"),          # NP vs coNP
    ("NP", "BQP"),           # NP vs BQP — incomparable?
    ("BQP", "NP"),           # BQP vs NP — incomparable?
    ("L", "P"),              # L vs P
    ("L", "NL"),             # L vs NL
    ("NP", "PP"),            # Is the inclusion strict?
    ("PH", "PSPACE"),        # PH vs PSPACE
    ("P", "NC"),             # P vs NC (is everything in P parallelizable?)
    ("P", "GI"),             # Is GI in P? (Babai 2015 gives quasipoly, not poly)
}

# Completeness results: {problem: {class: status}}
# status is "complete", "in" (member but not known complete), or "not_known_complete"
COMPLETENESS: Dict[str, Dict[str, str]] = {
    "SAT": {
        "NP": "complete",       # Cook-Levin theorem
        "P": "unknown",         # Would imply P = NP
    },
    "3SAT": {
        "NP": "complete",
    },
    "CLIQUE": {
        "NP": "complete",
    },
    "VERTEX_COVER": {
        "NP": "complete",
    },
    "HAM_CYCLE": {
        "NP": "complete",       # Hamiltonian cycle
    },
    "TSP": {
        "NP": "complete",       # Decision version
    },
    "TQBF": {
        "PSPACE": "complete",   # True quantified Boolean formula
    },
    "QBF": {
        "PSPACE": "complete",   # Same as TQBF
    },
    "HALTING": {
        "RE": "complete",       # Halting problem
    },
    "GI": {
        # Graph Isomorphism: in NP ∩ coAM, quasi-polynomial time (Babai 2015)
        # NOT known to be NP-complete
        "NP": "in",
        "coAM": "in",
        "NP-complete": "not_known",
    },
    "FACTORING": {
        # Integer factoring: in NP ∩ coNP ∩ BQP
        # NOT known to be NP-complete (would collapse NP = coNP)
        "NP": "in",
        "coNP": "in",
        "BQP": "in",
        "NP-complete": "not_known",
    },
    "DISCRETE_LOG": {
        "NP": "in",
        "coNP": "in",
        "BQP": "in",
        "NP-complete": "not_known",
    },
    "PRIMALITY": {
        "P": "in",              # AKS algorithm
    },
    "LP": {
        "P": "in",              # Linear programming (Khachiyan / Karmarkar)
    },
    "INTEGER_LP": {
        "NP": "complete",       # Integer linear programming
    },
}

# Collapse implications: if A ⊆ B, what known-separate hierarchy collapses?
# Format: (A, B) -> list of (description, severity)
COLLAPSE_IMPLICATIONS: Dict[Tuple[str, str], List[Tuple[str, str]]] = {
    ("NP", "P"): [
        ("P = NP would collapse the polynomial hierarchy (PH = P)", "HIGH"),
        ("All NP-complete problems would be solvable in polynomial time", "HIGH"),
        ("Would imply P = coNP", "HIGH"),
    ],
    ("NP", "BPP"): [
        ("Would imply NP = RP (derandomization of NP)", "HIGH"),
    ],
    ("NP", "BQP"): [
        ("Would imply quantum computers can solve all NP problems efficiently", "MODERATE"),
    ],
    ("PSPACE", "NP"): [
        ("Would collapse PSPACE into NP, implying PH = PSPACE = NP", "HIGH"),
    ],
    ("PSPACE", "P"): [
        ("Would collapse the entire known hierarchy: P = NP = PSPACE", "HIGH"),
    ],
    ("EXP", "P"): [
        ("Contradicts the time hierarchy theorem (P ⊊ EXP is proven)", "HIGH"),
    ],
    ("EXP", "NP"): [
        ("Would imply NP = EXP, contradicting time hierarchy separations", "HIGH"),
    ],
    ("NEXP", "NP"): [
        ("Contradicts the nondeterministic time hierarchy theorem (NP ⊊ NEXP)", "HIGH"),
    ],
    ("NEXP", "P"): [
        ("Contradicts both time hierarchy theorems", "HIGH"),
    ],
    ("NP", "P/poly"): [
        ("NP ⊆ P/poly would collapse PH to the second level (Karp-Lipton)", "HIGH"),
    ],
    ("PSPACE", "BPP"): [
        ("Would collapse PSPACE = BPP, implying PH = BPP", "HIGH"),
    ],
    ("NC1", "AC0"): [
        ("Contradicts the proven separation AC0 ⊊ NC1 (parity)", "HIGH"),
    ],
    ("EXP", "PSPACE"): [
        ("Would imply PSPACE = EXP (not known, but contradicts no proven separation)", "MODERATE"),
    ],
}

# All known class names for validation
ALL_CLASSES: Set[str] = {
    "AC0", "NC1", "NC", "L", "NL", "coNL", "P", "NP", "coNP", "RP", "coRP",
    "BPP", "BQP", "PP", "PH", "PSPACE", "NPSPACE", "APTIME", "IP",
    "EXP", "NEXP", "EXPSPACE", "P/poly", "RE",
    # Less common but referenced
    "coAM", "AM", "MA", "SZK", "ZPP",
    # Problem-defined classes (used in inclusion queries like "GI in NP")
    "GI",
}

# Problem name aliases for fuzzy matching
PROBLEM_ALIASES: Dict[str, str] = {
    "sat": "SAT",
    "3sat": "3SAT",
    "3-sat": "3SAT",
    "clique": "CLIQUE",
    "vertex cover": "VERTEX_COVER",
    "vertex_cover": "VERTEX_COVER",
    "hamiltonian cycle": "HAM_CYCLE",
    "ham_cycle": "HAM_CYCLE",
    "ham cycle": "HAM_CYCLE",
    "tsp": "TSP",
    "travelling salesman": "TSP",
    "traveling salesman": "TSP",
    "tqbf": "TQBF",
    "qbf": "QBF",
    "halting": "HALTING",
    "halting problem": "HALTING",
    "gi": "GI",
    "graph isomorphism": "GI",
    "graph_isomorphism": "GI",
    "factoring": "FACTORING",
    "integer factoring": "FACTORING",
    "discrete log": "DISCRETE_LOG",
    "discrete_log": "DISCRETE_LOG",
    "primality": "PRIMALITY",
    "lp": "LP",
    "linear programming": "LP",
    "integer lp": "INTEGER_LP",
    "integer_lp": "INTEGER_LP",
    "ilp": "INTEGER_LP",
}


# ─── Dataclasses ──────────────────────────────────────────────────────────────

@dataclass
class ComplexityIssue:
    """A single issue found when checking a complexity claim."""
    claim: str
    check_type: str          # INCLUSION_CHECK, SEPARATION_CHECK, COMPLETENESS_CHECK, COLLAPSE_CHECK
    severity: str            # HIGH, MODERATE, LOW, INFO
    description: str
    details: Dict[str, str] = field(default_factory=dict)

    def __str__(self):
        return f"  [{self.severity}] {self.check_type}: {self.description}"


@dataclass
class InclusionResult:
    """Result of check_inclusion(A, B) — is A ⊆ B?"""
    class_a: str
    class_b: str
    status: str              # ESTABLISHED, OPEN, CONTRADICTS_SEPARATION, TRIVIAL, UNKNOWN_CLASS
    description: str
    issues: List[ComplexityIssue] = field(default_factory=list)

    def __str__(self):
        return f"  {self.class_a} ⊆ {self.class_b}: {self.status} — {self.description}"


@dataclass
class CompletenessResult:
    """Result of check_completeness(problem, class)."""
    problem: str
    complexity_class: str
    status: str              # CORRECT, INCORRECT, OPEN, UNKNOWN_PROBLEM
    description: str
    issues: List[ComplexityIssue] = field(default_factory=list)

    def __str__(self):
        return f"  {self.problem} is {self.complexity_class}-complete: {self.status} — {self.description}"


@dataclass
class ComplexityReport:
    """Result of audit_complexity()."""
    verdict: str                          # PASS, WARN, or FAIL
    n_claims: int
    n_issues: int
    n_high: int
    n_moderate: int
    n_low: int
    n_info: int
    claims_checked: List[str]
    issues: List[ComplexityIssue]
    warnings: List[str]

    def __str__(self):
        lines = []
        lines.append(f"{'=' * 60}")
        lines.append(f"  Complexity Class Audit: {self.verdict}")
        lines.append(f"{'=' * 60}")
        lines.append(f"  Claims: {self.n_claims} checked, "
                     f"{self.n_high} HIGH, "
                     f"{self.n_moderate} MODERATE, "
                     f"{self.n_low} LOW, "
                     f"{self.n_info} INFO")
        lines.append("")

        # Issues sorted by severity
        if self.issues:
            lines.append("  Issues found:")
            severity_order = {"HIGH": 0, "MODERATE": 1, "LOW": 2, "INFO": 3}
            for issue in sorted(self.issues, key=lambda i: severity_order.get(i.severity, 4)):
                lines.append(str(issue))
            lines.append("")

        # Warnings
        if self.warnings:
            lines.append("  Warnings:")
            for w in self.warnings:
                lines.append(f"    - {w}")

        lines.append(f"{'=' * 60}")
        return "\n".join(lines)

    @property
    def passed(self) -> bool:
        return self.verdict == "PASS"


# ─── Helper functions ─────────────────────────────────────────────────────────

def _normalize_class(name: str) -> str:
    """Normalize a complexity class name to canonical form."""
    s = name.strip().upper().replace(" ", "")
    # Handle special cases
    mapping = {
        "P/POLY": "P/poly",
        "CONP": "coNP",
        "CORP": "coRP",
        "CONL": "coNL",
        "COAM": "coAM",
        "BQPP": "BQP",
    }
    return mapping.get(s, s)


def _normalize_problem(name: str) -> str:
    """Normalize a problem name via aliases."""
    lower = name.strip().lower().replace("-", "_")
    if lower in PROBLEM_ALIASES:
        return PROBLEM_ALIASES[lower]
    # Try exact upper
    upper = name.strip().upper().replace(" ", "_").replace("-", "_")
    if upper in COMPLETENESS:
        return upper
    return name.strip()


def _is_known_inclusion(a: str, b: str) -> bool:
    """Check if A ⊆ B is in the known inclusions (direct or transitive)."""
    if a == b:
        return True
    # Check equalities
    for x, y in KNOWN_EQUALITIES:
        if (a == x and b == y) or (a == y and b == x):
            return True
    # BFS for transitive closure
    visited = {a}
    frontier = [a]
    while frontier:
        current = frontier.pop(0)
        if current == b:
            return True
        for src, dst in KNOWN_INCLUSIONS:
            if src == current and dst not in visited:
                visited.add(dst)
                frontier.append(dst)
        # Also follow equalities
        for x, y in KNOWN_EQUALITIES:
            if x == current and y not in visited:
                visited.add(y)
                frontier.append(y)
            elif y == current and x not in visited:
                visited.add(x)
                frontier.append(x)
    return False


def _is_known_separation(a: str, b: str) -> bool:
    """Check if A ⊊ B is proven (A strictly contained in B)."""
    return (a, b) in KNOWN_SEPARATIONS


def _is_open(a: str, b: str) -> bool:
    """Check if the relationship between A and B is an open question."""
    return (a, b) in OPEN_QUESTIONS or (b, a) in OPEN_QUESTIONS


def _has_oracle_separation(a: str, b: str) -> bool:
    """Check if A vs B has an oracle separation (relativization barrier)."""
    return (a, b) in ORACLE_SEPARATIONS or (b, a) in ORACLE_SEPARATIONS


def _get_collapse_implications(a: str, b: str) -> List[Tuple[str, str]]:
    """Get collapse implications if A ⊆ B were true."""
    results = []
    # Direct lookup
    if (a, b) in COLLAPSE_IMPLICATIONS:
        results.extend(COLLAPSE_IMPLICATIONS[(a, b)])
    # If claiming B ⊆ A (reverse), check if that contradicts a proven separation
    if _is_known_separation(b, a):
        # This is backwards — a strict separation B ⊊ A means B is strictly inside A,
        # which is compatible with B ⊆ A. We need to check if A ⊆ B contradicts A ⊊ B.
        pass
    # Check if the inclusion contradicts any proven separation
    # If we know X ⊊ Y (strict), and the claim implies Y ⊆ X, that's a contradiction.
    for small, big in KNOWN_SEPARATIONS:
        # If claim is big ⊆ small, contradicts small ⊊ big
        if a == big and b == small:
            results.append((
                f"Contradicts proven separation {small} ⊊ {big}",
                "HIGH",
            ))
    return results


# ─── Public API ───────────────────────────────────────────────────────────────

def check_inclusion(class_a: str, class_b: str) -> InclusionResult:
    """Check if class_a ⊆ class_b is consistent with the known hierarchy.

    Args:
        class_a: complexity class name (e.g., "P", "NP", "BQP")
        class_b: complexity class name

    Returns:
        InclusionResult with status and description.
    """
    a = _normalize_class(class_a)
    b = _normalize_class(class_b)
    issues: List[ComplexityIssue] = []

    # Unknown classes
    if a not in ALL_CLASSES:
        return InclusionResult(a, b, "UNKNOWN_CLASS",
                               f"Unrecognized complexity class: {a}")
    if b not in ALL_CLASSES:
        return InclusionResult(a, b, "UNKNOWN_CLASS",
                               f"Unrecognized complexity class: {b}")

    # Trivial: A ⊆ A
    if a == b:
        return InclusionResult(a, b, "TRIVIAL", f"{a} ⊆ {a} is trivially true")

    # Check equalities
    for x, y in KNOWN_EQUALITIES:
        if (a == x and b == y) or (a == y and b == x):
            return InclusionResult(a, b, "ESTABLISHED",
                                   f"{a} = {b} (proven equality)")

    # Established inclusion via transitive closure
    if _is_known_inclusion(a, b):
        return InclusionResult(a, b, "ESTABLISHED",
                               f"{a} ⊆ {b} is established")

    # Check if it contradicts a proven separation
    # If we know B ⊊ A (B strictly inside A), then A ⊆ B would mean A = B,
    # contradicting strictness.
    if _is_known_separation(b, a):
        # Known: B ⊊ A. Claim: A ⊆ B. Together: A = B. Contradicts strictness.
        issues.append(ComplexityIssue(
            claim=f"{a} ⊆ {b}",
            check_type="INCLUSION_CHECK",
            severity="HIGH",
            description=f"Contradicts proven separation {b} ⊊ {a}",
        ))
        return InclusionResult(a, b, "CONTRADICTS_SEPARATION",
                               f"Contradicts proven {b} ⊊ {a}", issues)

    # Check if the reverse inclusion would contradict a proven separation
    # If we know A ⊊ B, then A ⊆ B is already established (handled above).
    # But if we're claiming something like EXP ⊆ P, and we know P ⊊ EXP...
    if _is_known_separation(a, b):
        # Known: A ⊊ B. Claim: A ⊆ B. This is just the known inclusion.
        return InclusionResult(a, b, "ESTABLISHED",
                               f"{a} ⊊ {b} is proven (strict containment)")

    # Check for collapse implications
    collapses = _get_collapse_implications(a, b)
    if collapses:
        for desc, sev in collapses:
            issues.append(ComplexityIssue(
                claim=f"{a} ⊆ {b}",
                check_type="COLLAPSE_CHECK",
                severity=sev,
                description=desc,
            ))

    # Open question
    if _is_open(a, b):
        oracle = _has_oracle_separation(a, b)
        desc = f"{a} ⊆ {b} is an open question"
        if oracle:
            desc += " (oracle separation exists — cannot resolve by relativization)"
        if issues:
            return InclusionResult(a, b, "OPEN", desc, issues)
        issues.append(ComplexityIssue(
            claim=f"{a} ⊆ {b}",
            check_type="INCLUSION_CHECK",
            severity="MODERATE",
            description=desc,
        ))
        return InclusionResult(a, b, "OPEN", desc, issues)

    # Not in our data — might be true, might not
    # Check if the reverse is known (B ⊆ A established but A ⊆ B not)
    if _is_known_inclusion(b, a):
        desc = f"{b} ⊆ {a} is established, but {a} ⊆ {b} is not known"
        issues.append(ComplexityIssue(
            claim=f"{a} ⊆ {b}",
            check_type="INCLUSION_CHECK",
            severity="MODERATE",
            description=desc,
        ))
        return InclusionResult(a, b, "OPEN", desc, issues)

    # Unknown relationship
    return InclusionResult(a, b, "OPEN",
                           f"Relationship between {a} and {b} is not established",
                           issues)


def check_completeness(problem: str, complexity_class: str) -> CompletenessResult:
    """Check if a problem is complete for a complexity class.

    Args:
        problem: problem name (e.g., "SAT", "GI", "Graph Isomorphism")
        complexity_class: class name (e.g., "NP", "PSPACE")

    Returns:
        CompletenessResult with status and description.
    """
    prob = _normalize_problem(problem)
    cls = _normalize_class(complexity_class)
    issues: List[ComplexityIssue] = []

    # Unknown problem
    if prob not in COMPLETENESS:
        return CompletenessResult(prob, cls, "UNKNOWN_PROBLEM",
                                  f"Problem '{prob}' not in database")

    info = COMPLETENESS[prob]

    # Check if completeness is claimed for this class
    if cls in info:
        status = info[cls]
        if status == "complete":
            return CompletenessResult(prob, cls, "CORRECT",
                                      f"{prob} is {cls}-complete (established)")
        elif status == "in":
            issues.append(ComplexityIssue(
                claim=f"{prob} is {cls}-complete",
                check_type="COMPLETENESS_CHECK",
                severity="HIGH",
                description=f"{prob} is in {cls} but NOT known to be {cls}-complete",
            ))
            return CompletenessResult(prob, cls, "INCORRECT",
                                      f"{prob} is in {cls} but not known to be {cls}-complete",
                                      issues)
        elif status == "not_known":
            issues.append(ComplexityIssue(
                claim=f"{prob} is {cls}-complete",
                check_type="COMPLETENESS_CHECK",
                severity="HIGH",
                description=f"{prob} is NOT known to be {cls}-complete",
            ))
            return CompletenessResult(prob, cls, "INCORRECT",
                                      f"{prob} is not known to be {cls}-complete",
                                      issues)
        elif status == "unknown":
            # E.g., SAT in P — this is an open question
            issues.append(ComplexityIssue(
                claim=f"{prob} is {cls}-complete",
                check_type="COMPLETENESS_CHECK",
                severity="MODERATE",
                description=f"Whether {prob} is in {cls} is an open question",
            ))
            return CompletenessResult(prob, cls, "OPEN",
                                      f"Whether {prob} is in {cls} is open",
                                      issues)

    # Check "NP-complete" as a pseudo-class
    if cls == "NP" and "NP-complete" in info and info["NP-complete"] == "not_known":
        issues.append(ComplexityIssue(
            claim=f"{prob} is NP-complete",
            check_type="COMPLETENESS_CHECK",
            severity="HIGH",
            description=f"{prob} is NOT known to be NP-complete",
        ))
        return CompletenessResult(prob, cls, "INCORRECT",
                                  f"{prob} is in NP but not known to be NP-complete",
                                  issues)

    # Not in our data for this class
    return CompletenessResult(prob, cls, "UNKNOWN_PROBLEM",
                              f"No data for {prob} vs {cls}")


def get_class_info(class_name: str) -> dict:
    """Return known inclusions, separations, and open questions for a class.

    Args:
        class_name: complexity class name (e.g., "NP", "P", "BQP")

    Returns:
        dict with keys:
            - "name": normalized class name
            - "contains": classes known to be inside this class
            - "contained_in": classes this class is known to be inside
            - "equals": classes known to be equal
            - "strict_subsets": classes proven strictly smaller
            - "strict_supersets": classes proven strictly larger
            - "open_with": classes with unresolved relationship
            - "oracle_separations": classes with oracle separation barriers
            - "complete_problems": problems known to be complete for this class
    """
    c = _normalize_class(class_name)

    contains: List[str] = []
    contained_in: List[str] = []
    equals: List[str] = []
    strict_subsets: List[str] = []
    strict_supersets: List[str] = []
    open_with: List[str] = []
    oracle_seps: List[str] = []
    complete_probs: List[str] = []

    # Inclusions (direct only, not transitive — keep it readable)
    for a, b in KNOWN_INCLUSIONS:
        if b == c and a != c:
            contains.append(a)
        if a == c and b != c:
            contained_in.append(b)

    # Equalities
    for a, b in KNOWN_EQUALITIES:
        if a == c:
            equals.append(b)
        elif b == c:
            equals.append(a)

    # Strict separations
    for a, b in KNOWN_SEPARATIONS:
        if b == c:
            strict_subsets.append(a)
        if a == c:
            strict_supersets.append(b)

    # Open questions
    for a, b in OPEN_QUESTIONS:
        if a == c:
            open_with.append(b)
        elif b == c:
            open_with.append(a)

    # Oracle separations
    for a, b in ORACLE_SEPARATIONS:
        if a == c:
            oracle_seps.append(b)
        elif b == c:
            oracle_seps.append(a)

    # Complete problems
    for prob, info in COMPLETENESS.items():
        if c in info and info[c] == "complete":
            complete_probs.append(prob)

    return {
        "name": c,
        "contains": sorted(set(contains)),
        "contained_in": sorted(set(contained_in)),
        "equals": sorted(set(equals)),
        "strict_subsets": sorted(set(strict_subsets)),
        "strict_supersets": sorted(set(strict_supersets)),
        "open_with": sorted(set(open_with)),
        "oracle_separations": sorted(set(oracle_seps)),
        "complete_problems": sorted(set(complete_probs)),
    }


# ─── Claim parser ─────────────────────────────────────────────────────────────

def _parse_claim(claim: str) -> Optional[Tuple[str, ...]]:
    """Parse a natural-language claim into a structured form.

    Returns a tuple: (claim_type, *args)
        - ("equality", class_a, class_b)       — "P = NP"
        - ("inclusion", class_a, class_b)       — "P ⊆ NP", "BQP in PSPACE"
        - ("separation", class_a, class_b)      — "P ⊊ NP", "P ≠ NP", "P != NP"
        - ("completeness", problem, class)      — "SAT is NP-complete", "SAT in P"
        - ("membership", problem, class)        — "GI in NP"
        - None if unparseable
    """
    s = claim.strip()

    # "X = Y" (equality)
    m = re.match(r'^(\S+)\s*=\s*(\S+)$', s)
    if m:
        return ("equality", m.group(1), m.group(2))

    # "X ⊆ Y" or "X <= Y" or "X ⊂ Y" or "X subset Y" or "X in Y" (inclusion)
    m = re.match(r'^(\S+)\s*(?:⊆|<=|\\subseteq|⊂|\\subset)\s*(\S+)$', s)
    if m:
        return ("inclusion", m.group(1), m.group(2))

    # "X in Y" for classes
    m = re.match(r'^(\S+)\s+in\s+(\S+)$', s, re.IGNORECASE)
    if m:
        a, b = m.group(1).strip(), m.group(2).strip()
        # If 'a' looks like a problem name, treat as membership
        if _normalize_problem(a) in COMPLETENESS:
            return ("membership", a, b)
        return ("inclusion", a, b)

    # "X ⊊ Y" or "X < Y" or "X ⊂ Y" strict or "X != Y" or "X ≠ Y"
    m = re.match(r'^(\S+)\s*(?:⊊|⊂|<|\\subsetneq)\s*(\S+)$', s)
    if m:
        return ("separation", m.group(1), m.group(2))

    m = re.match(r'^(\S+)\s*(?:!=|≠|\\neq)\s*(\S+)$', s)
    if m:
        return ("separation", m.group(1), m.group(2))

    # "X is Y-complete" or "X is Y complete"
    m = re.match(r'^(.+?)\s+is\s+(\S+?)[-\s]complete$', s, re.IGNORECASE)
    if m:
        return ("completeness", m.group(1).strip(), m.group(2).strip())

    # "X is in Y"
    m = re.match(r'^(.+?)\s+is\s+in\s+(\S+)$', s, re.IGNORECASE)
    if m:
        a = m.group(1).strip()
        b = m.group(2).strip()
        if _normalize_problem(a) in COMPLETENESS:
            return ("membership", a, b)
        return ("inclusion", a, b)

    # "X not in Y" or "X is not in Y"
    m = re.match(r'^(.+?)\s+(?:is\s+)?not\s+in\s+(\S+)$', s, re.IGNORECASE)
    if m:
        a = m.group(1).strip()
        b = m.group(2).strip()
        return ("separation", a, b)

    return None


# ─── Main entry point ─────────────────────────────────────────────────────────

def audit_complexity(claims: List[str]) -> ComplexityReport:
    """Audit a list of complexity class relationship claims.

    Each claim is a string like:
        "P = NP"
        "SAT in P"
        "GI is NP-complete"
        "BQP ⊆ P"
        "P ⊊ EXP"
        "P ⊆ PSPACE"

    Args:
        claims: list of claim strings

    Returns:
        ComplexityReport with per-claim issues and overall verdict.
    """
    all_issues: List[ComplexityIssue] = []
    warnings: List[str] = []

    for claim in claims:
        parsed = _parse_claim(claim)
        if parsed is None:
            warnings.append(f"Could not parse claim: '{claim}'")
            continue

        ctype = parsed[0]

        if ctype == "equality":
            a, b = _normalize_class(parsed[1]), _normalize_class(parsed[2])
            # A = B means A ⊆ B and B ⊆ A
            r_ab = check_inclusion(a, b)
            r_ba = check_inclusion(b, a)

            # Check if equality is already known
            is_equal = False
            for x, y in KNOWN_EQUALITIES:
                if (a == x and b == y) or (a == y and b == x):
                    is_equal = True
                    break

            if is_equal:
                all_issues.append(ComplexityIssue(
                    claim=claim,
                    check_type="INCLUSION_CHECK",
                    severity="INFO",
                    description=f"{a} = {b} is established",
                ))
            elif a == b:
                all_issues.append(ComplexityIssue(
                    claim=claim,
                    check_type="INCLUSION_CHECK",
                    severity="INFO",
                    description=f"{a} = {a} is trivially true",
                ))
            else:
                # Collect issues from both directions
                for r in (r_ab, r_ba):
                    all_issues.extend(r.issues)

                # If either direction contradicts a proven separation
                if r_ab.status == "CONTRADICTS_SEPARATION" or r_ba.status == "CONTRADICTS_SEPARATION":
                    all_issues.append(ComplexityIssue(
                        claim=claim,
                        check_type="INCLUSION_CHECK",
                        severity="HIGH",
                        description=f"{a} = {b} contradicts a proven separation",
                    ))
                elif _is_known_separation(a, b) or _is_known_separation(b, a):
                    small, big = (a, b) if _is_known_separation(a, b) else (b, a)
                    all_issues.append(ComplexityIssue(
                        claim=claim,
                        check_type="SEPARATION_CHECK",
                        severity="HIGH",
                        description=f"{a} = {b} contradicts proven separation {small} ⊊ {big}",
                    ))
                elif _is_open(a, b) or _is_open(b, a):
                    oracle = _has_oracle_separation(a, b)
                    desc = f"{a} = {b} is an open question"
                    if oracle:
                        desc += " (oracle separation exists)"
                    all_issues.append(ComplexityIssue(
                        claim=claim,
                        check_type="INCLUSION_CHECK",
                        severity="MODERATE",
                        description=desc,
                    ))
                else:
                    # Check collapse implications for both directions
                    collapses = _get_collapse_implications(a, b) + _get_collapse_implications(b, a)
                    if collapses:
                        for desc, sev in collapses:
                            all_issues.append(ComplexityIssue(
                                claim=claim,
                                check_type="COLLAPSE_CHECK",
                                severity=sev,
                                description=f"{a} = {b} would imply: {desc}",
                            ))
                    else:
                        all_issues.append(ComplexityIssue(
                            claim=claim,
                            check_type="INCLUSION_CHECK",
                            severity="MODERATE",
                            description=f"{a} = {b} is not established",
                        ))

        elif ctype == "inclusion":
            a, b = _normalize_class(parsed[1]), _normalize_class(parsed[2])
            result = check_inclusion(a, b)
            if result.status == "ESTABLISHED" or result.status == "TRIVIAL":
                all_issues.append(ComplexityIssue(
                    claim=claim,
                    check_type="INCLUSION_CHECK",
                    severity="INFO",
                    description=result.description,
                ))
            else:
                all_issues.extend(result.issues)
                if not result.issues:
                    all_issues.append(ComplexityIssue(
                        claim=claim,
                        check_type="INCLUSION_CHECK",
                        severity="MODERATE",
                        description=result.description,
                    ))

        elif ctype == "separation":
            a, b = _normalize_class(parsed[1]), _normalize_class(parsed[2])

            # "A ⊊ B" means A is strictly inside B
            # "A ≠ B" means they're not equal
            if _is_known_separation(a, b):
                all_issues.append(ComplexityIssue(
                    claim=claim,
                    check_type="SEPARATION_CHECK",
                    severity="INFO",
                    description=f"{a} ⊊ {b} is proven",
                ))
            elif _is_known_separation(b, a):
                all_issues.append(ComplexityIssue(
                    claim=claim,
                    check_type="SEPARATION_CHECK",
                    severity="INFO",
                    description=f"{b} ⊊ {a} is proven (note: direction differs from claim)",
                ))
            elif _is_open(a, b):
                oracle = _has_oracle_separation(a, b)
                desc = f"{a} ≠ {b} is an open question"
                if oracle:
                    desc += " (oracle separation exists — relativization barrier)"
                all_issues.append(ComplexityIssue(
                    claim=claim,
                    check_type="SEPARATION_CHECK",
                    severity="MODERATE",
                    description=desc,
                ))
            else:
                # Check if they're known to be equal
                is_equal = False
                for x, y in KNOWN_EQUALITIES:
                    if (a == x and b == y) or (a == y and b == x):
                        is_equal = True
                        break
                if is_equal:
                    all_issues.append(ComplexityIssue(
                        claim=claim,
                        check_type="SEPARATION_CHECK",
                        severity="HIGH",
                        description=f"{a} ≠ {b} contradicts proven equality {a} = {b}",
                    ))
                else:
                    all_issues.append(ComplexityIssue(
                        claim=claim,
                        check_type="SEPARATION_CHECK",
                        severity="MODERATE",
                        description=f"Separation {a} ≠ {b} is not established",
                    ))

        elif ctype == "completeness":
            prob = _normalize_problem(parsed[1])
            cls = _normalize_class(parsed[2])
            result = check_completeness(prob, cls)
            if result.status == "CORRECT":
                all_issues.append(ComplexityIssue(
                    claim=claim,
                    check_type="COMPLETENESS_CHECK",
                    severity="INFO",
                    description=result.description,
                ))
            else:
                all_issues.extend(result.issues)
                if not result.issues:
                    all_issues.append(ComplexityIssue(
                        claim=claim,
                        check_type="COMPLETENESS_CHECK",
                        severity="MODERATE",
                        description=result.description,
                    ))

        elif ctype == "membership":
            prob = _normalize_problem(parsed[1])
            cls = _normalize_class(parsed[2])
            # Check if problem is known to be in this class
            if prob in COMPLETENESS:
                info = COMPLETENESS[prob]
                if cls in info and info[cls] in ("complete", "in"):
                    all_issues.append(ComplexityIssue(
                        claim=claim,
                        check_type="COMPLETENESS_CHECK",
                        severity="INFO",
                        description=f"{prob} is in {cls} (established)",
                    ))
                elif cls in info and info[cls] == "unknown":
                    all_issues.append(ComplexityIssue(
                        claim=claim,
                        check_type="COMPLETENESS_CHECK",
                        severity="MODERATE",
                        description=f"Whether {prob} is in {cls} is an open question",
                    ))
                else:
                    all_issues.append(ComplexityIssue(
                        claim=claim,
                        check_type="COMPLETENESS_CHECK",
                        severity="LOW",
                        description=f"No data on {prob} membership in {cls}",
                    ))
            else:
                all_issues.append(ComplexityIssue(
                    claim=claim,
                    check_type="COMPLETENESS_CHECK",
                    severity="LOW",
                    description=f"Problem '{prob}' not in database",
                ))

        # ── COLLAPSE_CHECK for all claim types ──
        # (Already handled inline for equality/inclusion above)

    # ── Overall verdict ───────────────────────────────────────────────────
    n_high = sum(1 for i in all_issues if i.severity == "HIGH")
    n_moderate = sum(1 for i in all_issues if i.severity == "MODERATE")
    n_low = sum(1 for i in all_issues if i.severity == "LOW")
    n_info = sum(1 for i in all_issues if i.severity == "INFO")

    if n_high > 0:
        verdict = "FAIL"
    elif n_moderate > 0:
        verdict = "WARN"
    else:
        verdict = "PASS"

    return ComplexityReport(
        verdict=verdict,
        n_claims=len(claims),
        n_issues=len(all_issues),
        n_high=n_high,
        n_moderate=n_moderate,
        n_low=n_low,
        n_info=n_info,
        claims_checked=claims,
        issues=all_issues,
        warnings=warnings,
    )
