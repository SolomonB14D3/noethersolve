"""
noethersolve.proof_barriers — Proof technique barrier checker.

Checks whether known barriers in complexity theory and mathematical logic
prevent a given proof technique from resolving a target theorem or problem.
This is the #1 mistake in amateur proof attempts: using a technique that
provably cannot work.

Barriers are results like Baker-Gill-Solovay (relativization), Razborov-Rudich
(natural proofs), and Aaronson-Wigderson (algebrization) that rule out entire
classes of techniques for certain problems. This module encodes ~10 major
barriers, the proof techniques they block, and the problems they apply to.

Usage:
    from noethersolve.proof_barriers import (
        check_barriers, list_barriers, list_techniques,
        get_barrier, what_works_for,
        BarrierReport, BarrierIssue, BarrierInfo,
    )

    # Check if diagonalization can resolve P vs NP
    report = check_barriers("diagonalization", "P vs NP")
    print(report)
    # FAIL — relativization barrier blocks diagonalization for P vs NP

    # What techniques are NOT blocked for P vs NP?
    safe = what_works_for("P vs NP")
    print(safe)
    # ['algebraic geometry (GCT)', 'interactive proofs / PCPs', ...]

    # List all known barriers
    for b in list_barriers():
        print(f"{b.name}: {b.year} — {b.summary}")

    # Get full info on one barrier
    info = get_barrier("natural proofs")
    print(info.formal_statement)
"""

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Set, Tuple


# ─── Barrier database ────────────────────────────────────────────────────────

@dataclass(frozen=True)
class BarrierInfo:
    """Full information about a known proof barrier."""
    name: str
    authors: str
    year: int
    summary: str
    formal_statement: str
    blocked_problems: FrozenSet[str]
    blocked_techniques: FrozenSet[str]
    reference: str

    def __str__(self):
        probs = ", ".join(sorted(self.blocked_problems))
        techs = ", ".join(sorted(self.blocked_techniques))
        return (
            f"{self.name} ({self.authors}, {self.year})\n"
            f"  {self.summary}\n"
            f"  Blocks techniques: {techs}\n"
            f"  Blocks problems:   {probs}\n"
            f"  Reference: {self.reference}"
        )


# ── The ~10 major barriers ──────────────────────────────────────────────────

_BARRIERS: List[BarrierInfo] = [
    BarrierInfo(
        name="relativization",
        authors="Baker, Gill, Solovay",
        year=1975,
        summary=(
            "There exist oracles A, B where P^A = NP^A and P^B != NP^B. "
            "Any proof resolving P vs NP must use non-relativizing techniques."
        ),
        formal_statement=(
            "For any proof technique T that relativizes (i.e., T applies uniformly "
            "in the presence of any oracle), T cannot resolve P vs NP, P vs PSPACE, "
            "or NP vs coNP, because contradictory oracle worlds exist for each."
        ),
        blocked_problems=frozenset({
            "P vs NP", "P vs PSPACE", "NP vs coNP",
        }),
        blocked_techniques=frozenset({
            "diagonalization", "simulation", "counting",
        }),
        reference="Baker, Gill, Solovay. Relativizations of the P =? NP question. SIAM J. Comput. 4(4), 1975.",
    ),
    BarrierInfo(
        name="natural proofs",
        authors="Razborov, Rudich",
        year=1997,
        summary=(
            "If one-way functions exist, no 'natural' proof (constructive + largeness) "
            "can prove super-polynomial circuit lower bounds against P/poly."
        ),
        formal_statement=(
            "A natural proof against P/poly has two properties: constructivity "
            "(can efficiently recognize hard functions) and largeness (a random "
            "function satisfies the combinatorial property with non-negligible "
            "probability). Under the assumption that one-way functions exist, "
            "no natural proof can establish super-polynomial circuit lower bounds."
        ),
        blocked_problems=frozenset({
            "P vs NP", "circuit lower bounds",
        }),
        blocked_techniques=frozenset({
            "combinatorial", "natural proof", "boolean function analysis",
        }),
        reference="Razborov, Rudich. Natural proofs. JCSS 55(1), 1997.",
    ),
    BarrierInfo(
        name="algebrization",
        authors="Aaronson, Wigderson",
        year=2009,
        summary=(
            "Strengthens relativization. There exist algebrizing oracles for both "
            "P=NP and P!=NP. Blocks algebraic techniques for major separations."
        ),
        formal_statement=(
            "A proof technique algebrizes if it still applies when the oracle is "
            "replaced by a low-degree algebraic extension. There exist algebrizing "
            "oracles witnessing P=NP, P!=NP, NP in BPP, NP not in BPP, P=BQP, "
            "and P!=BQP. Any technique that algebrizes cannot resolve these questions."
        ),
        blocked_problems=frozenset({
            "P vs NP", "NP vs BPP", "P vs BQP", "NP vs coNP",
            "derandomization",
        }),
        blocked_techniques=frozenset({
            "arithmetization", "polynomial method", "algebraic",
        }),
        reference="Aaronson, Wigderson. Algebrization: a new barrier in complexity theory. TOCT 1(1), 2009.",
    ),
    BarrierInfo(
        name="black-box derandomization",
        authors="(folklore / Impagliazzo-Wigderson framework)",
        year=1997,
        summary=(
            "Black-box techniques alone cannot prove BPP = P. Must exploit "
            "specific structure of pseudorandom generators, not just their existence."
        ),
        formal_statement=(
            "A black-box reduction from derandomization to circuit lower bounds "
            "treats the PRG as an oracle. Such reductions cannot unconditionally "
            "establish BPP = P because there exist oracles relative to which "
            "BPP != P. The PRG construction must use non-black-box structure."
        ),
        blocked_problems=frozenset({
            "derandomization",
        }),
        blocked_techniques=frozenset({
            "black-box PRG", "generic derandomization",
        }),
        reference="Impagliazzo, Wigderson. P = BPP if E requires exponential circuits. STOC 1997.",
    ),
    BarrierInfo(
        name="GCT barrier",
        authors="Mulmuley",
        year=2009,
        summary=(
            "Geometric Complexity Theory approach to P vs NP requires resolving "
            "hard algebraic geometry conjectures (positivity hypotheses) that are "
            "currently beyond reach."
        ),
        formal_statement=(
            "GCT reduces circuit lower bounds to representation-theoretic "
            "questions about multiplicities in coordinate rings of orbit closures. "
            "The required positivity conjectures (that certain multiplicities are "
            "nonzero or obey specific inequalities) are themselves major open "
            "problems in algebraic combinatorics, blocking the GCT program."
        ),
        blocked_problems=frozenset({
            "P vs NP", "circuit lower bounds",
        }),
        blocked_techniques=frozenset({
            "GCT", "geometric complexity theory",
        }),
        reference="Mulmuley. On P vs NP and Geometric Complexity Theory. JACM 58(2), 2011.",
    ),
    BarrierInfo(
        name="independence",
        authors="Godel, Cohen",
        year=1963,
        summary=(
            "Some statements are independent of ZFC. The continuum hypothesis "
            "can neither be proved nor disproved in standard set theory."
        ),
        formal_statement=(
            "Godel (1940) showed Con(ZFC) => Con(ZFC + CH) via constructible "
            "universe L. Cohen (1963) showed Con(ZFC) => Con(ZFC + not-CH) via "
            "forcing. Therefore CH is independent of ZFC. Similarly, some "
            "combinatorial statements (Paris-Harrington) are independent of "
            "Peano arithmetic."
        ),
        blocked_problems=frozenset({
            "continuum hypothesis", "set theory independence",
        }),
        blocked_techniques=frozenset({
            "ZFC proof", "standard set theory", "first-order set theory",
        }),
        reference="Cohen. The independence of the continuum hypothesis. PNAS 50(6), 1963.",
    ),
    BarrierInfo(
        name="incompleteness",
        authors="Godel",
        year=1931,
        summary=(
            "Any consistent formal system capable of expressing arithmetic "
            "contains true statements that cannot be proved within it."
        ),
        formal_statement=(
            "First incompleteness theorem: For any consistent, recursively "
            "enumerable formal system F capable of expressing basic arithmetic, "
            "there exists a sentence G_F that is true in N but not provable in F. "
            "Second: F cannot prove its own consistency. This blocks complete "
            "axiomatization of arithmetic and decision procedures for all "
            "mathematical truth."
        ),
        blocked_problems=frozenset({
            "complete axiomatization of arithmetic",
            "decision procedure for mathematical truth",
            "consistency proof within system",
        }),
        blocked_techniques=frozenset({
            "complete axiomatization", "decision procedure",
            "internal consistency proof",
        }),
        reference="Godel. Uber formal unentscheidbare Satze. Monatshefte fur Mathematik 38, 1931.",
    ),
    BarrierInfo(
        name="monotone circuit",
        authors="Razborov",
        year=1985,
        summary=(
            "Exponential lower bounds for monotone circuits computing clique "
            "do NOT extend to general circuits. Cannot prove general circuit "
            "lower bounds from monotone ones alone."
        ),
        formal_statement=(
            "Razborov proved exponential monotone circuit lower bounds for "
            "k-CLIQUE. However, general (non-monotone) circuits can compute "
            "CLIQUE with polynomial-size circuits using negation. The monotone "
            "lower bound technique fundamentally cannot transfer to the general "
            "setting because negation provides exponential power."
        ),
        blocked_problems=frozenset({
            "circuit lower bounds",
        }),
        blocked_techniques=frozenset({
            "monotone arguments", "monotone circuit lower bounds",
        }),
        reference="Razborov. Lower bounds on monotone complexity of the logical permanent. Math. Notes 37(6), 1985.",
    ),
    BarrierInfo(
        name="SOS degree",
        authors="(Grigoriev, Schoenebeck, others)",
        year=2001,
        summary=(
            "Degree-d SOS proofs cannot refute random 3-SAT for d < n^{1/2}. "
            "SOS/Lasserre hierarchy has known degree limitations for specific problems."
        ),
        formal_statement=(
            "The Sum-of-Squares (SOS) / Lasserre hierarchy at degree d requires "
            "d >= Omega(n^{1/2}) to refute random 3-SAT instances. For planted "
            "clique, degree-4 SOS fails to distinguish planted vs random for "
            "clique size < n^{1/2}. These are unconditional lower bounds on the "
            "SOS proof system."
        ),
        blocked_problems=frozenset({
            "circuit lower bounds", "proof complexity",
        }),
        blocked_techniques=frozenset({
            "SOS", "sum-of-squares", "convex relaxation", "Lasserre hierarchy",
        }),
        reference="Grigoriev. Linear lower bound on degrees of Positivstellensatz calculus proofs. STACS 2001.",
    ),
    BarrierInfo(
        name="descriptive complexity",
        authors="Cai, Furer, Immerman",
        year=1992,
        summary=(
            "Certain logics capture certain complexity classes (FO = AC^0, "
            "SO-exists = NP). Extensions face CFI-type barriers for graph "
            "isomorphism and related problems."
        ),
        formal_statement=(
            "The Cai-Furer-Immerman construction provides pairs of non-isomorphic "
            "graphs that cannot be distinguished by k-variable counting logics "
            "(equivalently, by the k-dimensional Weisfeiler-Leman algorithm) for "
            "any fixed k. This blocks the descriptive complexity approach to "
            "separating P from NP via fixed-variable logics and limits the "
            "Weisfeiler-Leman method for graph isomorphism."
        ),
        blocked_problems=frozenset({
            "graph isomorphism", "P vs NP",
        }),
        blocked_techniques=frozenset({
            "fixed-variable logic", "Weisfeiler-Leman",
            "descriptive complexity",
        }),
        reference="Cai, Furer, Immerman. An optimal lower bound on the number of variables for graph identification. Combinatorica 12(4), 1992.",
    ),
]

# Build lookup indices
_BARRIER_BY_NAME: Dict[str, BarrierInfo] = {b.name: b for b in _BARRIERS}

# ── Normalize aliases ────────────────────────────────────────────────────────

_TECHNIQUE_ALIASES: Dict[str, str] = {
    # diagonalization
    "diagonal argument": "diagonalization",
    "diagonal": "diagonalization",
    # simulation
    "simulation argument": "simulation",
    # counting
    "counting argument": "counting",
    "simple counting": "counting",
    "combinatorial counting": "combinatorial",
    "combinatorial argument": "combinatorial",
    # natural proof
    "natural property": "natural proof",
    "natural": "natural proof",
    # boolean function analysis
    "boolean function": "boolean function analysis",
    # arithmetization
    "arithmetic": "arithmetization",
    "algebraic extension": "arithmetization",
    # polynomial method
    "polynomial": "polynomial method",
    "polynomial identity testing": "polynomial method",
    # SOS variants
    "sum of squares": "SOS",
    "sos": "SOS",
    "lasserre": "Lasserre hierarchy",
    # monotone
    "monotone": "monotone arguments",
    "monotone circuit": "monotone circuit lower bounds",
    # GCT
    "gct": "GCT",
    # descriptive
    "weisfeiler-leman": "Weisfeiler-Leman",
    "wl": "Weisfeiler-Leman",
    "fixed variable logic": "fixed-variable logic",
    # set theory / logic
    "zfc": "ZFC proof",
    "set theory": "standard set theory",
    "axiomatization": "complete axiomatization",
    # black-box
    "black box": "black-box PRG",
    "black-box": "black-box PRG",
    "generic prg": "generic derandomization",
    # consistency
    "consistency proof": "internal consistency proof",
    # decision
    "decision": "decision procedure",
    # convex
    "convex": "convex relaxation",
    "sdp": "convex relaxation",
    "semidefinite programming": "convex relaxation",
}

_PROBLEM_ALIASES: Dict[str, str] = {
    "p vs np": "P vs NP",
    "p != np": "P vs NP",
    "p = np": "P vs NP",
    "p versus np": "P vs NP",
    "p≠np": "P vs NP",
    "p=np": "P vs NP",
    "p vs pspace": "P vs PSPACE",
    "np vs conp": "NP vs coNP",
    "np vs bpp": "NP vs BPP",
    "p vs bqp": "P vs BQP",
    "bpp = p": "derandomization",
    "bpp vs p": "derandomization",
    "bpp=p": "derandomization",
    "circuit lower bound": "circuit lower bounds",
    "circuit complexity": "circuit lower bounds",
    "ch": "continuum hypothesis",
    "graph isomorphism": "graph isomorphism",
    "gi": "graph isomorphism",
    "arithmetic truth": "decision procedure for mathematical truth",
    "complete arithmetic": "complete axiomatization of arithmetic",
    "consistency": "consistency proof within system",
    "sos lower bounds": "proof complexity",
    "proof complexity lower bounds": "proof complexity",
    "independence": "set theory independence",
}


def _normalize_technique(raw: str) -> str:
    """Normalize a technique name to canonical form."""
    key = raw.strip().lower()
    if key in _TECHNIQUE_ALIASES:
        return _TECHNIQUE_ALIASES[key]
    # Check if it matches a canonical technique directly (case-insensitive)
    all_techniques = set()
    for b in _BARRIERS:
        all_techniques.update(b.blocked_techniques)
    for t in all_techniques:
        if t.lower() == key:
            return t
    return raw.strip()


def _normalize_problem(raw: str) -> str:
    """Normalize a problem name to canonical form."""
    key = raw.strip().lower()
    if key in _PROBLEM_ALIASES:
        return _PROBLEM_ALIASES[key]
    all_problems = set()
    for b in _BARRIERS:
        all_problems.update(b.blocked_problems)
    for p in all_problems:
        if p.lower() == key:
            return p
    return raw.strip()


# ─── Report dataclasses ──────────────────────────────────────────────────────

@dataclass
class BarrierIssue:
    """A single barrier that blocks the proposed technique for the target."""
    barrier_name: str
    severity: str            # HIGH, MODERATE, LOW, INFO
    description: str
    reference: str
    suggestion: str          # what to do instead

    def __str__(self):
        return f"  [{self.severity}] {self.barrier_name}: {self.description}"


@dataclass
class BarrierReport:
    """Result of check_barriers()."""
    verdict: str                # PASS, WARN, or FAIL
    technique: str              # normalized technique name
    target: str                 # normalized target problem
    issues: List[BarrierIssue]
    n_high: int
    n_moderate: int
    n_low: int
    n_info: int
    suggestions: List[str]

    def __str__(self):
        lines = []
        lines.append(f"{'=' * 60}")
        lines.append(f"  Proof Barrier Check: {self.verdict}")
        lines.append(f"{'=' * 60}")
        lines.append(f"  Technique: {self.technique}")
        lines.append(f"  Target:    {self.target}")
        lines.append(f"  Issues:    {self.n_high} HIGH, {self.n_moderate} MODERATE, "
                     f"{self.n_low} LOW, {self.n_info} INFO")
        lines.append("")

        if self.issues:
            # Group by severity
            for severity in ("HIGH", "MODERATE", "LOW", "INFO"):
                group = [i for i in self.issues if i.severity == severity]
                if group:
                    lines.append(f"  {severity}:")
                    for issue in group:
                        lines.append(f"    {issue.barrier_name} ({issue.reference})")
                        lines.append(f"      {issue.description}")
                        if issue.suggestion:
                            lines.append(f"      -> {issue.suggestion}")
                    lines.append("")

        if self.suggestions:
            lines.append("  Overall suggestions:")
            for s in self.suggestions:
                lines.append(f"    - {s}")
            lines.append("")

        lines.append(f"{'=' * 60}")
        return "\n".join(lines)

    @property
    def passed(self) -> bool:
        return self.verdict == "PASS"


# ─── Technique-to-barrier mapping ────────────────────────────────────────────
# For each (technique, problem) pair, which barriers apply and at what severity.
# Severity rules:
#   HIGH     — the barrier provably blocks this technique for this problem
#   MODERATE — the barrier partially applies or applies under standard assumptions
#   LOW      — the barrier is tangentially relevant; caution advised
#   INFO     — informational; the barrier is related but does not directly block

def _find_blocking_barriers(technique: str, target: str) -> List[Tuple[BarrierInfo, str]]:
    """Find all barriers that block (technique, target).

    Returns list of (BarrierInfo, severity) tuples.
    """
    results = []
    for barrier in _BARRIERS:
        tech_match = technique in barrier.blocked_techniques
        prob_match = target in barrier.blocked_problems

        if tech_match and prob_match:
            # Direct hit: technique is blocked AND problem is covered
            results.append((barrier, "HIGH"))
        elif tech_match and not prob_match:
            # Technique is blocked by this barrier but for OTHER problems.
            # If the target is closely related, it may still be relevant.
            # We include as INFO so users know the technique has issues elsewhere.
            results.append((barrier, "INFO"))
        elif not tech_match and prob_match:
            # Problem is covered by this barrier but via OTHER techniques.
            # Include as LOW — the barrier landscape is relevant.
            results.append((barrier, "LOW"))

    return results


# ── Suggestions for what DOES work ───────────────────────────────────────────

# Techniques known to potentially work for each problem (not blocked by any barrier)
_POSITIVE_SUGGESTIONS: Dict[str, List[str]] = {
    "P vs NP": [
        "algebraic geometry (GCT) — blocked by its own positivity conjectures "
        "but not by relativization/natural proofs/algebrization",
        "interactive proofs / PCPs — IP=PSPACE is non-relativizing",
        "circuit complexity via non-natural properties",
        "proof complexity / meta-mathematical approaches",
    ],
    "circuit lower bounds": [
        "non-natural combinatorial properties (properties that are not 'large')",
        "algebraic circuit lower bounds (Shub-Smale / tau conjecture)",
        "communication complexity reductions",
        "proof complexity lower bounds (indirect route)",
    ],
    "derandomization": [
        "non-black-box PRG constructions exploiting specific hardness",
        "hardness-randomness tradeoffs with explicit constructions",
        "algebraic derandomization (Kabanets-Impagliazzo framework)",
    ],
    "continuum hypothesis": [
        "forcing axioms (Martin's Axiom, PFA, etc.) — resolve CH in extended set theories",
        "large cardinal axioms — may settle CH in stronger systems",
        "inner model theory / ultimate L program",
    ],
    "P vs BQP": [
        "query complexity separations (already known: Simon's, BQP vs BPP oracles)",
        "communication complexity approaches",
    ],
    "NP vs coNP": [
        "proof complexity lower bounds (e.g., resolution, cutting planes)",
        "non-relativizing / non-algebrizing techniques",
    ],
    "graph isomorphism": [
        "group-theoretic algorithms (Babai's quasipolynomial result)",
        "individualization-refinement beyond WL",
    ],
}


# ─── Public API ──────────────────────────────────────────────────────────────

def check_barriers(technique: str, target: str) -> BarrierReport:
    """Check whether known barriers block a proof technique for a target problem.

    Args:
        technique: proof technique, e.g. "diagonalization", "natural proof",
            "arithmetization", "monotone arguments", "SOS".
        target: target problem, e.g. "P vs NP", "circuit lower bounds",
            "derandomization", "continuum hypothesis".

    Returns:
        BarrierReport with verdict (PASS/WARN/FAIL), per-barrier issues,
        and suggestions for alternative techniques.

    Examples:
        >>> report = check_barriers("diagonalization", "P vs NP")
        >>> report.verdict
        'FAIL'

        >>> report = check_barriers("interactive proofs", "P vs NP")
        >>> report.verdict
        'PASS'
    """
    norm_tech = _normalize_technique(technique)
    norm_target = _normalize_problem(target)

    hits = _find_blocking_barriers(norm_tech, norm_target)

    issues = []
    for barrier, severity in hits:
        suggestion = ""
        if severity == "HIGH":
            # Get positive suggestions for this problem
            pos = _POSITIVE_SUGGESTIONS.get(norm_target, [])
            if pos:
                suggestion = f"Consider instead: {pos[0]}"
            else:
                suggestion = "No known unblocked technique catalogued for this problem."
        elif severity == "MODERATE":
            suggestion = "This technique may work with significant modifications to avoid the barrier."
        elif severity == "LOW":
            suggestion = (
                f"Your technique is not directly blocked, but {barrier.name} "
                f"constrains other approaches to this problem — be aware of the landscape."
            )

        issues.append(BarrierIssue(
            barrier_name=barrier.name,
            severity=severity,
            description=barrier.summary,
            reference=barrier.reference,
            suggestion=suggestion,
        ))

    n_high = sum(1 for i in issues if i.severity == "HIGH")
    n_moderate = sum(1 for i in issues if i.severity == "MODERATE")
    n_low = sum(1 for i in issues if i.severity == "LOW")
    n_info = sum(1 for i in issues if i.severity == "INFO")

    # Verdict: FAIL if any HIGH, WARN if any MODERATE, PASS otherwise
    if n_high > 0:
        verdict = "FAIL"
    elif n_moderate > 0:
        verdict = "WARN"
    else:
        verdict = "PASS"

    # Overall suggestions
    suggestions = []
    if n_high > 0:
        barrier_names = [i.barrier_name for i in issues if i.severity == "HIGH"]
        suggestions.append(
            f"Technique '{norm_tech}' is provably blocked for '{norm_target}' by: "
            f"{', '.join(barrier_names)}. This approach cannot work."
        )
        pos = _POSITIVE_SUGGESTIONS.get(norm_target, [])
        if pos:
            suggestions.append("Potentially viable alternatives:")
            for p in pos:
                suggestions.append(f"  {p}")
    elif n_moderate > 0:
        suggestions.append(
            f"Technique '{norm_tech}' faces partial barriers for '{norm_target}'. "
            f"Proceed with caution and verify the technique avoids the barrier conditions."
        )
    elif issues:
        suggestions.append(
            f"No direct barriers found for '{norm_tech}' on '{norm_target}'. "
            f"Some related barriers exist — review the INFO/LOW items."
        )

    return BarrierReport(
        verdict=verdict,
        technique=norm_tech,
        target=norm_target,
        issues=issues,
        n_high=n_high,
        n_moderate=n_moderate,
        n_low=n_low,
        n_info=n_info,
        suggestions=suggestions,
    )


def list_barriers() -> List[BarrierInfo]:
    """Return all known proof barriers.

    Returns:
        List of BarrierInfo objects, sorted by year.

    Example:
        >>> barriers = list_barriers()
        >>> len(barriers)
        10
        >>> barriers[0].name
        'incompleteness'
    """
    return sorted(_BARRIERS, key=lambda b: b.year)


def list_techniques() -> List[str]:
    """Return all known proof techniques that are tracked by barriers.

    Returns:
        Sorted list of canonical technique names.

    Example:
        >>> techs = list_techniques()
        >>> "diagonalization" in techs
        True
    """
    techniques: Set[str] = set()
    for barrier in _BARRIERS:
        techniques.update(barrier.blocked_techniques)
    return sorted(techniques)


def get_barrier(name: str) -> BarrierInfo:
    """Get full information about a single barrier by name.

    Args:
        name: barrier name (case-insensitive), e.g. "relativization",
            "natural proofs", "algebrization".

    Returns:
        BarrierInfo with full details.

    Raises:
        KeyError: if the barrier name is not found.

    Example:
        >>> info = get_barrier("relativization")
        >>> info.year
        1975
        >>> info.authors
        'Baker, Gill, Solovay'
    """
    key = name.strip().lower()
    # Try exact match
    if key in _BARRIER_BY_NAME:
        return _BARRIER_BY_NAME[key]
    # Try fuzzy: strip trailing 's', common variants
    variants = [key, key.rstrip("s"), key.replace(" ", "_"), key.replace("_", " ")]
    # Also handle "natural proofs" -> "natural proofs" barrier name
    for variant in variants:
        for bname, binfo in _BARRIER_BY_NAME.items():
            if bname.lower() == variant:
                return binfo
    # Substring match as last resort
    matches = [b for b in _BARRIERS if key in b.name.lower()]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        names = ", ".join(m.name for m in matches)
        raise KeyError(f"Ambiguous barrier name '{name}'. Matches: {names}")
    raise KeyError(
        f"Unknown barrier '{name}'. Known barriers: "
        f"{', '.join(sorted(_BARRIER_BY_NAME.keys()))}"
    )


def what_works_for(target: str) -> List[str]:
    """Suggest proof techniques that are NOT blocked for a given target problem.

    This returns techniques from the positive suggestions database — approaches
    that are known to avoid the major barriers. Note: "not blocked" does not
    mean "will succeed", only that no known barrier rules it out.

    Args:
        target: target problem, e.g. "P vs NP", "circuit lower bounds".

    Returns:
        List of technique descriptions that are not blocked by known barriers.
        Empty list if no suggestions are catalogued for this problem.

    Example:
        >>> suggestions = what_works_for("P vs NP")
        >>> len(suggestions) > 0
        True
    """
    norm_target = _normalize_problem(target)
    return list(_POSITIVE_SUGGESTIONS.get(norm_target, []))
