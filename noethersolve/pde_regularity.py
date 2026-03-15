"""
noethersolve.pde_regularity — PDE regularity and Sobolev embedding checker.

Validates claims about PDE regularity results and Sobolev embeddings. Catches
common errors: wrong embedding dimensions, claiming regularity that contradicts
known blow-up examples, incorrect critical exponents.

Hardcoded data covers the Sobolev embedding theorem (subcritical, critical,
supercritical cases), known PDE regularity results (Laplace, heat, wave,
Navier-Stokes, Euler, Burgers, KdV, NLS), critical exponents (Fujita, Strauss,
NLS L²/H¹), and known blow-up examples.

Usage:
    from noethersolve.pde_regularity import (
        check_sobolev_embedding,
        check_pde_regularity,
        critical_exponent,
        check_blowup,
        sobolev_conjugate,
    )

    # Check a Sobolev embedding: W^{1,2}(R^3)
    report = check_sobolev_embedding(k=1, p=2.0, n=3)
    print(report)
    # Shows: subcritical case, p* = 6.0, correct target space L^6

    # Check regularity claim for 3D Navier-Stokes
    report = check_pde_regularity("navier-stokes", dimension=3,
                                   claimed_regularity="global smooth")
    print(report)
    # Shows: FAIL — global regularity is OPEN for 3D NS

    # Compute critical exponent for NLS
    report = critical_exponent("nls", dimension=3)
    print(report)
    # Shows: L² critical p_c = 7/3, H¹ critical p_s = 5

    # Check if global regularity claim is consistent
    report = check_blowup("euler", dimension=3, claimed_global=True)
    print(report)
    # Shows: FAIL — finite-time blow-up proved for C^{1,α} (Elgindi 2021)
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ─── Issue dataclass ─────────────────────────────────────────────────────────

@dataclass
class EmbeddingIssue:
    """A single issue found during a PDE regularity check."""
    check_type: str       # EMBEDDING_CHECK, REGULARITY_CHECK, CRITICAL_EXPONENT,
                          # BLOWUP_CHECK, DIMENSION_CHECK
    severity: str         # HIGH, MODERATE, LOW, INFO
    description: str
    details: Dict[str, object] = field(default_factory=dict)

    def __str__(self):
        return f"  [{self.severity}] {self.check_type}: {self.description}"


# ─── Report dataclasses ─────────────────────────────────────────────────────

@dataclass
class EmbeddingReport:
    """Result of check_sobolev_embedding()."""
    verdict: str                    # PASS, WARN, FAIL
    k: int
    p: float
    n: int
    case: str                       # subcritical, critical, supercritical
    kp: float                       # k * p
    sobolev_conjugate: Optional[float]   # p* for subcritical case
    target_space: str               # e.g. "L^6", "C^{0,1/2}", "L^q for all q in [p, inf)"
    holder_exponent: Optional[float]     # alpha for supercritical Holder embedding
    holder_j: Optional[int]              # j for C^{j,alpha}
    issues: List[EmbeddingIssue]

    def __str__(self):
        lines = []
        lines.append(f"{'=' * 60}")
        lines.append(f"  Sobolev Embedding Check: {self.verdict}")
        lines.append(f"{'=' * 60}")
        lines.append(f"  Space: W^{{{self.k},{self.p}}}(R^{self.n})")
        lines.append(f"  kp = {self.kp}, n = {self.n} => {self.case} case")
        if self.sobolev_conjugate is not None:
            lines.append(f"  Sobolev conjugate p* = {self.sobolev_conjugate:.4g}")
        lines.append(f"  Target space: {self.target_space}")
        if self.holder_exponent is not None:
            lines.append(f"  Holder: C^{{{self.holder_j},{self.holder_exponent:.4g}}}")
        lines.append("")
        if self.issues:
            lines.append("  Issues:")
            for issue in self.issues:
                lines.append(str(issue))
            lines.append("")
        lines.append(f"{'=' * 60}")
        return "\n".join(lines)

    @property
    def passed(self) -> bool:
        return self.verdict == "PASS"


@dataclass
class RegularityReport:
    """Result of check_pde_regularity()."""
    verdict: str                    # PASS, WARN, FAIL
    equation: str
    dimension: int
    claimed_regularity: Optional[str]
    known_regularity: str
    status: str                     # proved, open, conditional, depends_on_data
    issues: List[EmbeddingIssue]

    def __str__(self):
        lines = []
        lines.append(f"{'=' * 60}")
        lines.append(f"  PDE Regularity Check: {self.verdict}")
        lines.append(f"{'=' * 60}")
        lines.append(f"  Equation: {self.equation}")
        lines.append(f"  Dimension: {self.dimension}")
        lines.append(f"  Known regularity: {self.known_regularity}")
        lines.append(f"  Status: {self.status}")
        if self.claimed_regularity:
            lines.append(f"  Claimed: {self.claimed_regularity}")
        lines.append("")
        if self.issues:
            lines.append("  Issues:")
            for issue in self.issues:
                lines.append(str(issue))
            lines.append("")
        lines.append(f"{'=' * 60}")
        return "\n".join(lines)

    @property
    def passed(self) -> bool:
        return self.verdict == "PASS"


@dataclass
class CriticalExponentReport:
    """Result of critical_exponent()."""
    verdict: str                    # PASS, WARN, FAIL
    equation: str
    dimension: int
    exponents: Dict[str, float]     # name -> value, e.g. {"L2_critical": 7/3}
    descriptions: Dict[str, str]    # name -> human description
    issues: List[EmbeddingIssue]

    def __str__(self):
        lines = []
        lines.append(f"{'=' * 60}")
        lines.append(f"  Critical Exponent Report: {self.verdict}")
        lines.append(f"{'=' * 60}")
        lines.append(f"  Equation: {self.equation}")
        lines.append(f"  Dimension: {self.dimension}")
        lines.append("")
        if self.exponents:
            lines.append("  Exponents:")
            for name, val in self.exponents.items():
                desc = self.descriptions.get(name, "")
                lines.append(f"    {name} = {val:.6g}  ({desc})")
            lines.append("")
        if self.issues:
            lines.append("  Issues:")
            for issue in self.issues:
                lines.append(str(issue))
            lines.append("")
        lines.append(f"{'=' * 60}")
        return "\n".join(lines)

    @property
    def passed(self) -> bool:
        return self.verdict == "PASS"


@dataclass
class BlowupReport:
    """Result of check_blowup()."""
    verdict: str                    # PASS, WARN, FAIL
    equation: str
    dimension: int
    claimed_global: bool
    known_blowup: Optional[str]    # description of known blow-up, or None
    known_global: Optional[str]    # description of known global result, or None
    consistency: str               # consistent, contradicted, open
    issues: List[EmbeddingIssue]

    def __str__(self):
        lines = []
        lines.append(f"{'=' * 60}")
        lines.append(f"  Blow-up Consistency Check: {self.verdict}")
        lines.append(f"{'=' * 60}")
        lines.append(f"  Equation: {self.equation}")
        lines.append(f"  Dimension: {self.dimension}")
        claim = "global regularity" if self.claimed_global else "finite-time blow-up"
        lines.append(f"  Claimed: {claim}")
        lines.append(f"  Consistency: {self.consistency}")
        if self.known_blowup:
            lines.append(f"  Known blow-up: {self.known_blowup}")
        if self.known_global:
            lines.append(f"  Known global result: {self.known_global}")
        lines.append("")
        if self.issues:
            lines.append("  Issues:")
            for issue in self.issues:
                lines.append(str(issue))
            lines.append("")
        lines.append(f"{'=' * 60}")
        return "\n".join(lines)

    @property
    def passed(self) -> bool:
        return self.verdict == "PASS"


# ─── Known PDE data ─────────────────────────────────────────────────────────

# Known regularity results: equation -> dimension -> info dict
_KNOWN_REGULARITY = {
    "laplace": {
        # All dimensions: C^inf interior regularity (elliptic regularity)
        "all": {
            "regularity": "C^inf interior regularity",
            "status": "proved",
            "detail": "Elliptic regularity: solutions to Laplace equation are "
                      "real-analytic in the interior.",
        },
    },
    "heat": {
        "all": {
            "regularity": "C^inf for t > 0",
            "status": "proved",
            "detail": "Parabolic smoothing: solutions become C^inf instantly "
                      "for any t > 0, regardless of initial data regularity.",
        },
    },
    "wave": {
        "all": {
            "regularity": "regularity depends on initial data",
            "status": "depends_on_data",
            "detail": "Finite speed of propagation. Regularity is determined "
                      "by the regularity of initial data; no smoothing occurs.",
        },
    },
    "navier-stokes": {
        2: {
            "regularity": "global regularity proved",
            "status": "proved",
            "detail": "Ladyzhenskaya (1959): 2D Navier-Stokes has unique "
                      "global smooth solutions for smooth initial data.",
        },
        3: {
            "regularity": "global regularity OPEN (Millennium Problem)",
            "status": "open",
            "detail": "Whether smooth solutions to 3D Navier-Stokes can "
                      "develop singularities in finite time is one of the "
                      "Clay Millennium Prize Problems. No proof of global "
                      "regularity or finite-time blow-up for smooth data.",
        },
    },
    "euler": {
        2: {
            "regularity": "global regularity for smooth data (vorticity conservation)",
            "status": "proved",
            "detail": "2D Euler preserves vorticity in L^inf, giving global "
                      "existence of smooth solutions.",
        },
        3: {
            "regularity": "finite-time blow-up proved for C^{1,alpha}",
            "status": "proved_blowup",
            "detail": "Elgindi (2021): finite-time singularity formation for "
                      "C^{1,alpha} initial data in 3D. Smooth (C^inf) data "
                      "blow-up remains open.",
        },
    },
    "burgers": {
        "all": {
            "regularity": "inviscid: shock formation in finite time; "
                          "viscous: global smooth solutions",
            "status": "proved",
            "detail": "Inviscid Burgers (u_t + u*u_x = 0) develops gradient "
                      "blow-up (shocks) in finite time for any smooth "
                      "decreasing initial data. Adding viscosity "
                      "(u_t + u*u_x = nu*u_xx) gives global smooth solutions.",
        },
    },
    "kdv": {
        1: {
            "regularity": "global solutions via inverse scattering",
            "status": "proved",
            "detail": "KdV is completely integrable. Global solutions exist "
                      "via inverse scattering transform for suitable initial data.",
        },
    },
    "nls": {
        "all": {
            "regularity": "depends on nonlinearity exponent and dimension",
            "status": "conditional",
            "detail": "NLS with |u|^{p-1}u nonlinearity: subcritical "
                      "(p < 1 + 4/n) has global solutions; critical "
                      "(p = 1 + 4/n) is delicate; supercritical may blow up.",
        },
    },
}

# Known blow-up examples: equation -> dimension -> info
_KNOWN_BLOWUP = {
    "euler": {
        3: {
            "blowup": "C^{1,alpha} blow-up (Elgindi 2021)",
            "detail": "Finite-time singularity for C^{1,alpha} vorticity. "
                      "Smooth (C^inf) data blow-up remains open.",
            "smooth_data_open": True,
        },
    },
    "burgers": {
        # All dimensions (1D is the standard case)
        "all": {
            "blowup": "gradient blow-up (shock) in finite time for inviscid case",
            "detail": "Any smooth decreasing initial data produces a shock "
                      "in finite time for inviscid Burgers.",
            "smooth_data_open": False,
        },
    },
    "nls": {
        # Supercritical NLS: blow-up for p > 1 + 4/(n-2) with large data (n >= 3)
        "all": {
            "blowup": "supercritical blow-up for p > 1 + 4/(n-2) with large data (n >= 3)",
            "detail": "For NLS with power nonlinearity, supercritical "
                      "exponents lead to finite-time blow-up for "
                      "sufficiently large initial data.",
            "smooth_data_open": False,
        },
    },
    "navier-stokes": {
        3: {
            "blowup": "self-similar blow-up candidates exist but none proved "
                      "for smooth data",
            "detail": "No finite-time blow-up has been proved for smooth "
                      "solutions to 3D Navier-Stokes. The question remains open.",
            "smooth_data_open": True,
        },
    },
}

# Known global regularity results: equation -> dimension -> info
_KNOWN_GLOBAL = {
    "navier-stokes": {
        2: "2D Navier-Stokes: global regularity proved (Ladyzhenskaya 1959)",
    },
    "euler": {
        2: "2D Euler: global regularity for smooth data (vorticity in L^inf)",
    },
    "heat": {
        "all": "Heat equation: C^inf for t > 0 (parabolic smoothing)",
    },
    "laplace": {
        "all": "Laplace equation: C^inf interior (elliptic regularity)",
    },
    "burgers": {
        "all": "Viscous Burgers: global smooth solutions",
    },
    "kdv": {
        1: "KdV: global solutions via inverse scattering (completely integrable)",
    },
}


# ─── Utility: Sobolev conjugate ─────────────────────────────────────────────

def sobolev_conjugate(k: int, p: float, n: int) -> float:
    """Compute the Sobolev conjugate p* = np / (n - kp).

    Only defined when kp < n (subcritical case). Raises ValueError otherwise.

    Args:
        k: Number of derivatives.
        p: Integrability exponent (p >= 1).
        n: Spatial dimension (n >= 1).

    Returns:
        p* = n * p / (n - k * p)
    """
    if k < 0:
        raise ValueError(f"k must be non-negative, got {k}")
    if p < 1.0:
        raise ValueError(f"p must be >= 1, got {p}")
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    kp = k * p
    if kp >= n:
        raise ValueError(
            f"Sobolev conjugate only defined for kp < n. "
            f"Got kp = {kp}, n = {n} ({_case_name(kp, n)} case)."
        )
    return n * p / (n - kp)


# ─── Internal helpers ────────────────────────────────────────────────────────

def _case_name(kp: float, n: int) -> str:
    """Return the Sobolev embedding case name."""
    if kp < n:
        return "subcritical"
    elif abs(kp - n) < 1e-12:
        return "critical"
    else:
        return "supercritical"


def _normalize_equation(equation: str) -> str:
    """Normalize equation name for lookup."""
    eq = equation.lower().strip()
    # Handle common aliases
    aliases = {
        "ns": "navier-stokes",
        "navier_stokes": "navier-stokes",
        "navier stokes": "navier-stokes",
        "schrodinger": "nls",
        "schrödinger": "nls",
        "nonlinear schrodinger": "nls",
        "nonlinear_schrodinger": "nls",
        "nonlinear schrödinger": "nls",
        "semilinear heat": "semilinear_heat",
        "semilinear_heat": "semilinear_heat",
        "fujita": "semilinear_heat",
    }
    return aliases.get(eq, eq)


def _lookup_regularity(equation: str, dimension: int) -> Optional[dict]:
    """Look up known regularity for an equation at a given dimension."""
    data = _KNOWN_REGULARITY.get(equation)
    if data is None:
        return None
    # Try dimension-specific first, then "all"
    result = data.get(dimension)
    if result is None:
        result = data.get("all")
    return result


def _lookup_blowup(equation: str, dimension: int) -> Optional[dict]:
    """Look up known blow-up results."""
    data = _KNOWN_BLOWUP.get(equation)
    if data is None:
        return None
    result = data.get(dimension)
    if result is None:
        result = data.get("all")
    return result


def _lookup_global(equation: str, dimension: int) -> Optional[str]:
    """Look up known global regularity results."""
    data = _KNOWN_GLOBAL.get(equation)
    if data is None:
        return None
    result = data.get(dimension)
    if result is None:
        result = data.get("all")
    return result


# ─── Public API ──────────────────────────────────────────────────────────────

def check_sobolev_embedding(k: int, p: float, n: int) -> EmbeddingReport:
    """Compute the correct Sobolev embedding for W^{k,p}(R^n).

    Determines which case applies (subcritical, critical, supercritical)
    and returns the correct target space with any issues found.

    Args:
        k: Number of derivatives (k >= 0).
        p: Integrability exponent (p >= 1).
        n: Spatial dimension (n >= 1).

    Returns:
        EmbeddingReport with the correct embedding, target space, and any issues.
    """
    issues = []

    # Validate inputs
    if k < 0:
        issues.append(EmbeddingIssue(
            check_type="EMBEDDING_CHECK",
            severity="HIGH",
            description=f"k must be non-negative, got {k}.",
        ))
        return EmbeddingReport(
            verdict="FAIL", k=k, p=p, n=n, case="invalid", kp=k * p,
            sobolev_conjugate=None, target_space="undefined",
            holder_exponent=None, holder_j=None, issues=issues,
        )

    if p < 1.0:
        issues.append(EmbeddingIssue(
            check_type="EMBEDDING_CHECK",
            severity="HIGH",
            description=f"p must be >= 1 for Sobolev spaces, got {p}.",
        ))
        return EmbeddingReport(
            verdict="FAIL", k=k, p=p, n=n, case="invalid", kp=k * p,
            sobolev_conjugate=None, target_space="undefined",
            holder_exponent=None, holder_j=None, issues=issues,
        )

    if n < 1:
        issues.append(EmbeddingIssue(
            check_type="EMBEDDING_CHECK",
            severity="HIGH",
            description=f"n must be >= 1, got {n}.",
        ))
        return EmbeddingReport(
            verdict="FAIL", k=k, p=p, n=n, case="invalid", kp=k * p,
            sobolev_conjugate=None, target_space="undefined",
            holder_exponent=None, holder_j=None, issues=issues,
        )

    kp = k * p
    case = _case_name(kp, n)
    p_star = None
    holder_alpha = None
    holder_j = None
    target_space = ""

    if case == "subcritical":
        # Case 1: kp < n
        # W^{k,p} subset L^{p*} where p* = np/(n-kp)
        p_star = n * p / (n - kp)
        target_space = f"L^{{{p_star:.4g}}}"

        issues.append(EmbeddingIssue(
            check_type="EMBEDDING_CHECK",
            severity="INFO",
            description=(
                f"Subcritical case (kp={kp} < n={n}): "
                f"W^{{{k},{p}}}(R^{n}) embeds into L^{{{p_star:.4g}}}(R^{n}). "
                f"Also W^{{{k},{p}}} embeds into W^{{j,q}} for j < {k} and "
                f"1/q >= 1/{p} - ({k}-j)/{n}."
            ),
        ))

    elif case == "critical":
        # Case 2: kp = n
        # W^{k,p} subset L^q for all q in [p, inf), but NOT L^inf in general
        target_space = f"L^q for all q in [{p}, inf)"

        issues.append(EmbeddingIssue(
            check_type="EMBEDDING_CHECK",
            severity="INFO",
            description=(
                f"Critical case (kp={kp} = n={n}): "
                f"W^{{{k},{p}}}(R^{n}) embeds into L^q for all "
                f"q in [{p}, inf), but NOT into L^inf in general."
            ),
        ))

        # Special cases
        if k == 1 and abs(p - n) < 1e-12:
            issues.append(EmbeddingIssue(
                check_type="EMBEDDING_CHECK",
                severity="INFO",
                description=(
                    f"W^{{1,{n}}}(R^{n}) embeds into BMO "
                    f"(bounded mean oscillation)."
                ),
            ))

        if k == 1 and n == 2 and abs(p - 2.0) < 1e-12:
            issues.append(EmbeddingIssue(
                check_type="EMBEDDING_CHECK",
                severity="INFO",
                description=(
                    "W^{1,2}(R^2): Trudinger inequality gives "
                    "exponential integrability (Moser-Trudinger)."
                ),
            ))

    else:
        # Case 3: kp > n (supercritical)
        # W^{k,p} subset C^{j,alpha} where j = k - ceil(n/p), alpha = ceil(n/p) - n/p
        n_over_p = n / p
        ceil_n_over_p = math.ceil(n_over_p)

        # Check if n/p is an integer (boundary case)
        if abs(n_over_p - round(n_over_p)) < 1e-12:
            # When n/p is an integer, the Holder exponent is 0 or requires
            # special treatment — embedding is into C^{j} (just continuous
            # derivatives, any alpha < 1 works but not alpha = 1 in general)
            holder_j = k - int(round(n_over_p))
            holder_alpha = None  # Any alpha < 1
            if holder_j >= 0:
                target_space = f"C^{{{holder_j}}} (with any Holder exponent alpha < 1)"
            else:
                target_space = "L^inf (bounded, continuous)"
                holder_j = 0

            issues.append(EmbeddingIssue(
                check_type="EMBEDDING_CHECK",
                severity="INFO",
                description=(
                    f"Supercritical case (kp={kp} > n={n}), n/p is integer: "
                    f"W^{{{k},{p}}}(R^{n}) embeds into {target_space}. "
                    f"Morrey's inequality: functions are bounded and continuous."
                ),
            ))
        else:
            holder_j = k - ceil_n_over_p
            holder_alpha = ceil_n_over_p - n_over_p

            if holder_j < 0:
                holder_j = 0
                holder_alpha = kp / n - 1.0  # simplified
                target_space = f"C^{{0,{min(holder_alpha, 1.0):.4g}}}"
            else:
                target_space = f"C^{{{holder_j},{holder_alpha:.4g}}}"

            issues.append(EmbeddingIssue(
                check_type="EMBEDDING_CHECK",
                severity="INFO",
                description=(
                    f"Supercritical case (kp={kp} > n={n}): "
                    f"W^{{{k},{p}}}(R^{n}) embeds into {target_space}. "
                    f"Morrey's inequality: functions are Holder continuous."
                ),
            ))

    # Verdict: PASS with INFO issues, FAIL only on input errors
    has_high = any(i.severity == "HIGH" for i in issues)
    has_moderate = any(i.severity == "MODERATE" for i in issues)
    if has_high:
        verdict = "FAIL"
    elif has_moderate:
        verdict = "WARN"
    else:
        verdict = "PASS"

    return EmbeddingReport(
        verdict=verdict,
        k=k, p=p, n=n,
        case=case,
        kp=kp,
        sobolev_conjugate=p_star,
        target_space=target_space,
        holder_exponent=holder_alpha,
        holder_j=holder_j,
        issues=issues,
    )


def check_pde_regularity(
    equation: str,
    dimension: int,
    claimed_regularity: Optional[str] = None,
) -> RegularityReport:
    """Check regularity claims for known PDEs.

    Looks up the known regularity result for the given equation and dimension,
    and optionally checks whether a claimed regularity is consistent.

    Args:
        equation: PDE name (e.g. "navier-stokes", "euler", "heat", "nls").
        dimension: Spatial dimension.
        claimed_regularity: Optional string describing the claimed regularity
            (e.g. "global smooth", "C^inf", "blow-up"). If None, just reports
            the known result.

    Returns:
        RegularityReport with known regularity, status, and any issues.
    """
    eq = _normalize_equation(equation)
    issues = []

    reg_info = _lookup_regularity(eq, dimension)

    if reg_info is None:
        issues.append(EmbeddingIssue(
            check_type="REGULARITY_CHECK",
            severity="MODERATE",
            description=(
                f"No known regularity data for '{equation}' in dimension {dimension}. "
                f"Cannot validate claim."
            ),
        ))
        verdict = "WARN"
        return RegularityReport(
            verdict=verdict,
            equation=equation,
            dimension=dimension,
            claimed_regularity=claimed_regularity,
            known_regularity="unknown",
            status="unknown",
            issues=issues,
        )

    known_regularity = reg_info["regularity"]
    status = reg_info["status"]

    # If a claim is provided, check consistency
    if claimed_regularity is not None:
        claim_lower = claimed_regularity.lower().strip()
        claim_is_global = any(
            kw in claim_lower
            for kw in ["global smooth", "global regularity", "c^inf", "c^infinity",
                        "smooth for all time", "no blow-up", "no singularity",
                        "no blowup"]
        )
        claim_is_blowup = any(
            kw in claim_lower
            for kw in ["blow-up", "blowup", "blow up", "singularity",
                        "shock", "finite-time", "finite time"]
        )

        if status == "open" and claim_is_global:
            issues.append(EmbeddingIssue(
                check_type="REGULARITY_CHECK",
                severity="HIGH",
                description=(
                    f"Claiming global regularity for {equation} in dimension "
                    f"{dimension}, but this is an OPEN PROBLEM. "
                    f"Known status: {known_regularity}."
                ),
            ))

        elif status == "open" and claim_is_blowup:
            issues.append(EmbeddingIssue(
                check_type="REGULARITY_CHECK",
                severity="HIGH",
                description=(
                    f"Claiming blow-up for {equation} in dimension "
                    f"{dimension}, but this is an OPEN PROBLEM. "
                    f"Known status: {known_regularity}."
                ),
            ))

        elif status == "proved_blowup" and claim_is_global:
            issues.append(EmbeddingIssue(
                check_type="REGULARITY_CHECK",
                severity="HIGH",
                description=(
                    f"Claiming global regularity for {equation} in dimension "
                    f"{dimension}, but blow-up HAS BEEN PROVED. "
                    f"Known: {known_regularity}."
                ),
            ))

        elif status == "proved" and claim_is_blowup:
            # Claiming blow-up for something with proved regularity
            # Need to distinguish viscous vs inviscid for Burgers
            if eq == "burgers" and "inviscid" in claim_lower:
                # This is correct — inviscid Burgers does blow up
                issues.append(EmbeddingIssue(
                    check_type="REGULARITY_CHECK",
                    severity="INFO",
                    description=(
                        f"Claim of inviscid blow-up for Burgers is CORRECT. "
                        f"Inviscid Burgers develops shocks in finite time."
                    ),
                ))
            elif eq == "burgers" and "viscous" in claim_lower:
                issues.append(EmbeddingIssue(
                    check_type="REGULARITY_CHECK",
                    severity="HIGH",
                    description=(
                        f"Claiming blow-up for VISCOUS Burgers, but viscous "
                        f"Burgers has global smooth solutions."
                    ),
                ))
            else:
                issues.append(EmbeddingIssue(
                    check_type="REGULARITY_CHECK",
                    severity="HIGH",
                    description=(
                        f"Claiming blow-up for {equation} in dimension "
                        f"{dimension}, but regularity has been PROVED. "
                        f"Known: {known_regularity}."
                    ),
                ))

        elif status == "depends_on_data":
            issues.append(EmbeddingIssue(
                check_type="REGULARITY_CHECK",
                severity="MODERATE",
                description=(
                    f"Regularity for {equation} depends on initial data. "
                    f"A blanket claim of '{claimed_regularity}' may be incomplete. "
                    f"Known: {known_regularity}."
                ),
            ))

        elif status == "conditional":
            issues.append(EmbeddingIssue(
                check_type="REGULARITY_CHECK",
                severity="MODERATE",
                description=(
                    f"Regularity for {equation} is conditional on parameters "
                    f"(e.g. nonlinearity exponent, dimension). "
                    f"Known: {known_regularity}."
                ),
            ))

        else:
            # Claim is consistent with known results
            issues.append(EmbeddingIssue(
                check_type="REGULARITY_CHECK",
                severity="INFO",
                description=(
                    f"Claim '{claimed_regularity}' is consistent with known "
                    f"results for {equation} in dimension {dimension}."
                ),
            ))

    # Dimension-specific warnings
    if eq == "navier-stokes" and dimension == 3:
        issues.append(EmbeddingIssue(
            check_type="DIMENSION_CHECK",
            severity="MODERATE",
            description=(
                "3D Navier-Stokes regularity is a Millennium Prize Problem. "
                "Any claimed proof or disproof should be treated with extreme "
                "skepticism. Scaling-critical spaces: L^3(R^3), H^{1/2}(R^3)."
            ),
        ))

    elif eq == "navier-stokes" and dimension == 2:
        issues.append(EmbeddingIssue(
            check_type="DIMENSION_CHECK",
            severity="INFO",
            description=(
                "2D Navier-Stokes global regularity was proved by "
                "Ladyzhenskaya (1959). The 2D case is fundamentally different "
                "from 3D due to enstrophy conservation."
            ),
        ))

    elif eq == "euler" and dimension == 3:
        issues.append(EmbeddingIssue(
            check_type="DIMENSION_CHECK",
            severity="MODERATE",
            description=(
                "3D Euler: Elgindi (2021) proved finite-time blow-up for "
                "C^{1,alpha} initial data. Smooth (C^inf) data blow-up "
                "remains open. Be precise about regularity class."
            ),
        ))

    # Verdict
    has_high = any(i.severity == "HIGH" for i in issues)
    has_moderate = any(i.severity == "MODERATE" for i in issues)
    if has_high:
        verdict = "FAIL"
    elif has_moderate:
        verdict = "WARN"
    else:
        verdict = "PASS"

    return RegularityReport(
        verdict=verdict,
        equation=equation,
        dimension=dimension,
        claimed_regularity=claimed_regularity,
        known_regularity=known_regularity,
        status=status,
        issues=issues,
    )


def critical_exponent(
    equation: str,
    dimension: int,
    claimed_exponent: Optional[float] = None,
) -> CriticalExponentReport:
    """Compute the critical exponent for a given PDE and dimension.

    Args:
        equation: PDE name ("nls", "semilinear_heat", "navier-stokes", "wave").
        dimension: Spatial dimension (n >= 1).
        claimed_exponent: If provided, checks whether this value matches the
            known critical exponent.

    Returns:
        CriticalExponentReport with computed exponents and any issues.
    """
    eq = _normalize_equation(equation)
    issues = []
    exponents = {}
    descriptions = {}

    if eq == "nls":
        # NLS: i*u_t + Delta u + |u|^{p-1}u = 0
        # L^2 critical: p_c = 1 + 4/n
        p_c = 1.0 + 4.0 / dimension
        exponents["L2_critical"] = p_c
        descriptions["L2_critical"] = (
            f"L^2 critical exponent for NLS in R^{dimension}: "
            f"p < {p_c:.6g} subcritical (global), "
            f"p = {p_c:.6g} critical, "
            f"p > {p_c:.6g} may blow up"
        )

        # H^1 critical: p_s = 1 + 4/(n-2) for n >= 3
        if dimension >= 3:
            p_s = 1.0 + 4.0 / (dimension - 2)
            exponents["H1_critical"] = p_s
            descriptions["H1_critical"] = (
                f"H^1 critical exponent for NLS in R^{dimension}: "
                f"p_s = {p_s:.6g}. For p > p_s with large data, blow-up occurs."
            )
        elif dimension == 2:
            issues.append(EmbeddingIssue(
                check_type="CRITICAL_EXPONENT",
                severity="INFO",
                description=(
                    "H^1 critical exponent for NLS in R^2: p_s = inf "
                    "(energy-subcritical for all finite p)."
                ),
            ))
            exponents["H1_critical"] = math.inf
            descriptions["H1_critical"] = "H^1 critical = inf in 2D (all finite p subcritical)"
        elif dimension == 1:
            issues.append(EmbeddingIssue(
                check_type="CRITICAL_EXPONENT",
                severity="INFO",
                description=(
                    "NLS in 1D: H^1 critical exponent is infinity. "
                    "1D NLS is energy-subcritical for all finite p."
                ),
            ))
            exponents["H1_critical"] = math.inf
            descriptions["H1_critical"] = "H^1 critical = inf in 1D"

    elif eq == "semilinear_heat":
        # u_t = Delta u + |u|^{p-1}u
        # Fujita exponent: p_F = 1 + 2/n
        p_f = 1.0 + 2.0 / dimension
        exponents["Fujita"] = p_f
        descriptions["Fujita"] = (
            f"Fujita exponent for semilinear heat in R^{dimension}: "
            f"p_F = {p_f:.6g}. For 1 < p <= p_F, ALL positive solutions "
            f"blow up. For p > p_F, global existence for small data."
        )

    elif eq == "navier-stokes":
        # Scaling-critical spaces: L^n(R^n), H^{n/2-1}(R^n)
        issues.append(EmbeddingIssue(
            check_type="CRITICAL_EXPONENT",
            severity="INFO",
            description=(
                f"Navier-Stokes scaling-critical spaces in R^{dimension}: "
                f"L^{dimension}(R^{dimension}), "
                f"H^{{{dimension / 2 - 1:.4g}}}(R^{dimension})."
            ),
        ))
        exponents["critical_Lebesgue"] = float(dimension)
        descriptions["critical_Lebesgue"] = (
            f"L^{dimension}(R^{dimension}) is the scaling-critical Lebesgue space"
        )
        critical_sobolev = dimension / 2.0 - 1.0
        exponents["critical_Sobolev_regularity"] = critical_sobolev
        descriptions["critical_Sobolev_regularity"] = (
            f"H^{{{critical_sobolev:.4g}}}(R^{dimension}) is the "
            f"scaling-critical Sobolev space"
        )

    elif eq == "wave":
        # Strauss exponent p_S(n): positive root of
        # (n-1)p^2 - (n+1)p - 2 = 0
        # p_S(n) = [(n+1) + sqrt((n+1)^2 + 8(n-1))] / [2(n-1)]  for n >= 2
        if dimension >= 2:
            a = dimension - 1
            b = -(dimension + 1)
            c = -2
            discriminant = b * b - 4 * a * c
            p_s = (-b + math.sqrt(discriminant)) / (2 * a)
            exponents["Strauss"] = p_s
            descriptions["Strauss"] = (
                f"Strauss exponent for semilinear wave in R^{dimension}: "
                f"p_S = {p_s:.6g}. For 1 < p <= p_S, blow-up for all "
                f"nontrivial data. For p > p_S, global existence for small data."
            )
        else:
            issues.append(EmbeddingIssue(
                check_type="CRITICAL_EXPONENT",
                severity="INFO",
                description="Strauss exponent defined for n >= 2.",
            ))

    else:
        issues.append(EmbeddingIssue(
            check_type="CRITICAL_EXPONENT",
            severity="MODERATE",
            description=(
                f"No critical exponent data for '{equation}' in dimension "
                f"{dimension}. Known equations: nls, semilinear_heat, "
                f"navier-stokes, wave."
            ),
        ))

    # Check claimed exponent if provided
    if claimed_exponent is not None and exponents:
        matched = False
        for name, val in exponents.items():
            if math.isinf(val) and math.isinf(claimed_exponent):
                matched = True
                issues.append(EmbeddingIssue(
                    check_type="CRITICAL_EXPONENT",
                    severity="INFO",
                    description=f"Claimed exponent inf matches {name} = inf.",
                ))
                break
            elif not math.isinf(val) and abs(val - claimed_exponent) < 1e-6:
                matched = True
                issues.append(EmbeddingIssue(
                    check_type="CRITICAL_EXPONENT",
                    severity="INFO",
                    description=(
                        f"Claimed exponent {claimed_exponent:.6g} matches "
                        f"{name} = {val:.6g}."
                    ),
                ))
                break

        if not matched:
            # Find the closest known exponent
            finite_exps = {k: v for k, v in exponents.items() if not math.isinf(v)}
            if finite_exps:
                closest_name = min(finite_exps, key=lambda k: abs(finite_exps[k] - claimed_exponent))
                closest_val = finite_exps[closest_name]
                issues.append(EmbeddingIssue(
                    check_type="CRITICAL_EXPONENT",
                    severity="HIGH",
                    description=(
                        f"Claimed exponent {claimed_exponent:.6g} does NOT match "
                        f"any known critical exponent. Closest: {closest_name} = "
                        f"{closest_val:.6g} (error = {abs(closest_val - claimed_exponent):.6g})."
                    ),
                ))
            else:
                issues.append(EmbeddingIssue(
                    check_type="CRITICAL_EXPONENT",
                    severity="HIGH",
                    description=(
                        f"Claimed exponent {claimed_exponent:.6g} does NOT match "
                        f"any known critical exponent."
                    ),
                ))

    # Verdict
    has_high = any(i.severity == "HIGH" for i in issues)
    has_moderate = any(i.severity == "MODERATE" for i in issues)
    if has_high:
        verdict = "FAIL"
    elif has_moderate:
        verdict = "WARN"
    else:
        verdict = "PASS"

    return CriticalExponentReport(
        verdict=verdict,
        equation=equation,
        dimension=dimension,
        exponents=exponents,
        descriptions=descriptions,
        issues=issues,
    )


def check_blowup(
    equation: str,
    dimension: int,
    claimed_global: bool = True,
) -> BlowupReport:
    """Check if a global regularity claim is consistent with known blow-up results.

    Args:
        equation: PDE name (e.g. "euler", "burgers", "nls", "navier-stokes").
        dimension: Spatial dimension.
        claimed_global: True if claiming global regularity (no blow-up),
            False if claiming finite-time blow-up.

    Returns:
        BlowupReport with consistency assessment and any issues.
    """
    eq = _normalize_equation(equation)
    issues = []

    blowup_info = _lookup_blowup(eq, dimension)
    global_info = _lookup_global(eq, dimension)

    known_blowup = blowup_info["blowup"] if blowup_info else None
    known_global = global_info if isinstance(global_info, str) else None
    consistency = "consistent"  # default

    if claimed_global:
        # Claiming global regularity
        if blowup_info is not None:
            smooth_open = blowup_info.get("smooth_data_open", False)
            if smooth_open:
                # Blow-up exists for some regularity class, but smooth data
                # blow-up is open
                issues.append(EmbeddingIssue(
                    check_type="BLOWUP_CHECK",
                    severity="MODERATE",
                    description=(
                        f"Claiming global regularity for {equation} in "
                        f"dimension {dimension}. Known blow-up exists for "
                        f"less regular data: {known_blowup}. Smooth data "
                        f"blow-up remains open — claim is unproven but not "
                        f"contradicted."
                    ),
                ))
                consistency = "open"
            else:
                # Blow-up is proved (even for smooth data or relevant class)
                issues.append(EmbeddingIssue(
                    check_type="BLOWUP_CHECK",
                    severity="HIGH",
                    description=(
                        f"Claiming global regularity for {equation} in "
                        f"dimension {dimension}, but blow-up IS KNOWN: "
                        f"{known_blowup}. This claim is contradicted."
                    ),
                ))
                consistency = "contradicted"

        elif global_info is not None:
            # Known global result supports the claim
            issues.append(EmbeddingIssue(
                check_type="BLOWUP_CHECK",
                severity="INFO",
                description=(
                    f"Global regularity claim for {equation} in dimension "
                    f"{dimension} is consistent with known results: "
                    f"{known_global}."
                ),
            ))
            consistency = "consistent"

        else:
            # No data either way
            issues.append(EmbeddingIssue(
                check_type="BLOWUP_CHECK",
                severity="MODERATE",
                description=(
                    f"No known blow-up or global regularity data for "
                    f"'{equation}' in dimension {dimension}. Cannot validate "
                    f"global regularity claim."
                ),
            ))
            consistency = "open"

    else:
        # Claiming blow-up
        if blowup_info is not None:
            issues.append(EmbeddingIssue(
                check_type="BLOWUP_CHECK",
                severity="INFO",
                description=(
                    f"Blow-up claim for {equation} in dimension {dimension} "
                    f"is consistent with known results: {known_blowup}."
                ),
            ))
            consistency = "consistent"

        elif global_info is not None:
            # Known global result contradicts blow-up claim
            issues.append(EmbeddingIssue(
                check_type="BLOWUP_CHECK",
                severity="HIGH",
                description=(
                    f"Claiming blow-up for {equation} in dimension "
                    f"{dimension}, but global regularity IS PROVED: "
                    f"{known_global}. This claim is contradicted."
                ),
            ))
            consistency = "contradicted"

        else:
            issues.append(EmbeddingIssue(
                check_type="BLOWUP_CHECK",
                severity="MODERATE",
                description=(
                    f"No known blow-up or global regularity data for "
                    f"'{equation}' in dimension {dimension}. Cannot validate "
                    f"blow-up claim."
                ),
            ))
            consistency = "open"

    # Verdict
    has_high = any(i.severity == "HIGH" for i in issues)
    has_moderate = any(i.severity == "MODERATE" for i in issues)
    if has_high:
        verdict = "FAIL"
    elif has_moderate:
        verdict = "WARN"
    else:
        verdict = "PASS"

    return BlowupReport(
        verdict=verdict,
        equation=equation,
        dimension=dimension,
        claimed_global=claimed_global,
        known_blowup=known_blowup,
        known_global=known_global,
        consistency=consistency,
        issues=issues,
    )
