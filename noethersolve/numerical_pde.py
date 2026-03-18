"""Numerical PDE scheme analysis.

Verified stability and accuracy analysis for numerical PDE solvers:
- CFL condition and exact thresholds
- Von Neumann stability analysis
- Lax equivalence theorem conditions
- Scheme accuracy (order of convergence)

KEY POINTS LLMs GET WRONG:
1. CFL condition is NECESSARY for stability, not sufficient (explicit schemes)
2. Leapfrog is UNSTABLE for diffusion (parabolic) — only works for wave (hyperbolic)
3. Implicit schemes (Crank-Nicolson, backward Euler) are unconditionally stable
4. Lax equivalence: consistency + stability ⟺ convergence (for LINEAR problems ONLY)
5. Order of accuracy is not the same as stability — you can have 4th-order unstable schemes

CRITICAL DISTINCTIONS:
- Hyperbolic (wave): information propagates at finite speed, CFL = c×dt/dx ≤ 1
- Parabolic (diffusion): smoothing, CFL = D×dt/dx² ≤ 1/2 for FTCS
- Elliptic (Laplace): boundary value, no time-stepping
"""

from dataclasses import dataclass
from typing import Optional
import math


# ─── Report Dataclasses ─────────────────────────────────────────────────────

@dataclass
class CFLReport:
    """Report on CFL condition analysis."""
    pde_type: str  # hyperbolic, parabolic
    scheme: str
    cfl_number: float
    cfl_limit: float
    is_stable: bool
    stability_type: str  # conditionally stable, unconditionally stable, unstable
    notes: list[str]

    def __str__(self) -> str:
        lines = [
            "=" * 60,
            f"  CFL Condition Analysis: {self.scheme}",
            "=" * 60,
            f"  PDE type: {self.pde_type}",
            f"  CFL number: {self.cfl_number:.4f}",
            f"  CFL limit: {self.cfl_limit:.4f}",
            "-" * 60,
        ]
        if self.is_stable:
            lines.append(f"  ✓ STABLE (CFL ≤ {self.cfl_limit})")
        else:
            lines.append(f"  ✗ UNSTABLE (CFL > {self.cfl_limit})")
        lines.append(f"  Stability type: {self.stability_type}")
        if self.notes:
            lines.append("-" * 60)
            for note in self.notes:
                lines.append(f"  • {note}")
        lines.append("=" * 60)
        return "\n".join(lines)


@dataclass
class VonNeumannReport:
    """Report on Von Neumann stability analysis."""
    scheme: str
    amplification_factor: complex
    amplitude: float  # |G|
    is_stable: bool  # |G| ≤ 1
    is_dissipative: bool  # |G| < 1
    phase_error: Optional[float]  # Dispersion
    notes: list[str]

    def __str__(self) -> str:
        lines = [
            "=" * 60,
            f"  Von Neumann Stability: {self.scheme}",
            "=" * 60,
            f"  Amplification factor G = {self.amplification_factor}",
            f"  |G| = {self.amplitude:.6f}",
            "-" * 60,
        ]
        if self.is_stable:
            lines.append("  ✓ STABLE (|G| ≤ 1)")
        else:
            lines.append("  ✗ UNSTABLE (|G| > 1)")
        if self.is_dissipative:
            lines.append("  Dissipative: scheme damps oscillations")
        else:
            lines.append("  Non-dissipative: amplitude preserved")
        if self.phase_error is not None:
            lines.append(f"  Phase error: {self.phase_error:.4f} rad")
        if self.notes:
            lines.append("-" * 60)
            for note in self.notes:
                lines.append(f"  • {note}")
        lines.append("=" * 60)
        return "\n".join(lines)


@dataclass
class SchemeReport:
    """Report on a numerical scheme's properties."""
    name: str
    pde_type: str  # hyperbolic, parabolic, elliptic
    order_space: int
    order_time: int
    stability_condition: str  # e.g., "CFL ≤ 1", "unconditionally stable"
    is_explicit: bool
    is_conservative: bool
    stencil: str  # e.g., "3-point", "5-point"
    cfl_limit: Optional[float]
    notes: list[str]

    def __str__(self) -> str:
        lines = [
            "=" * 60,
            f"  Numerical Scheme: {self.name}",
            "=" * 60,
            f"  PDE type: {self.pde_type}",
            f"  Order: O(Δx^{self.order_space}, Δt^{self.order_time})",
            f"  Type: {'Explicit' if self.is_explicit else 'Implicit'}",
            f"  Stencil: {self.stencil}",
            "-" * 60,
            f"  Stability: {self.stability_condition}",
        ]
        if self.cfl_limit is not None:
            lines.append(f"  CFL limit: {self.cfl_limit}")
        if self.is_conservative:
            lines.append("  ✓ Conservative (preserves integral)")
        else:
            lines.append("  ✗ Non-conservative")
        if self.notes:
            lines.append("-" * 60)
            for note in self.notes:
                lines.append(f"  • {note}")
        lines.append("=" * 60)
        return "\n".join(lines)


@dataclass
class LaxEquivalenceReport:
    """Report on Lax equivalence theorem check."""
    is_consistent: bool
    is_stable: bool
    is_convergent: bool
    consistency_order: int
    stability_type: str
    problem_type: str  # "linear" or "nonlinear"
    theorem_applies: bool  # Only for linear problems
    notes: list[str]

    def __str__(self) -> str:
        lines = [
            "=" * 60,
            "  Lax Equivalence Theorem Check",
            "=" * 60,
            f"  Problem type: {self.problem_type}",
            "-" * 60,
        ]
        if self.is_consistent:
            lines.append(f"  ✓ Consistent (order {self.consistency_order})")
        else:
            lines.append("  ✗ NOT Consistent")
        if self.is_stable:
            lines.append(f"  ✓ Stable ({self.stability_type})")
        else:
            lines.append("  ✗ NOT Stable")
        lines.append("-" * 60)
        if self.theorem_applies:
            if self.is_consistent and self.is_stable:
                lines.append("  ⟹ CONVERGENT (Lax equivalence)")
            else:
                lines.append("  ⟹ NOT CONVERGENT")
        else:
            lines.append("  ⚠ Lax theorem does NOT apply (nonlinear)")
            lines.append(f"    Convergence: {self.is_convergent} (must verify separately)")
        if self.notes:
            lines.append("-" * 60)
            for note in self.notes:
                lines.append(f"  • {note}")
        lines.append("=" * 60)
        return "\n".join(lines)


@dataclass
class AccuracyReport:
    """Report on scheme accuracy analysis."""
    scheme: str
    truncation_error: str  # Leading term
    order_space: int
    order_time: int
    leading_error_type: str  # dissipative, dispersive
    richardson_possible: bool  # Can use Richardson extrapolation
    notes: list[str]

    def __str__(self) -> str:
        lines = [
            "=" * 60,
            f"  Accuracy Analysis: {self.scheme}",
            "=" * 60,
            f"  Truncation error: {self.truncation_error}",
            f"  Order: O(Δx^{self.order_space}) in space, O(Δt^{self.order_time}) in time",
            f"  Leading error: {self.leading_error_type}",
            "-" * 60,
        ]
        if self.richardson_possible:
            lines.append("  ✓ Richardson extrapolation applicable")
        else:
            lines.append("  ✗ Richardson extrapolation not straightforward")
        if self.notes:
            lines.append("-" * 60)
            for note in self.notes:
                lines.append(f"  • {note}")
        lines.append("=" * 60)
        return "\n".join(lines)


# ─── Scheme Database ────────────────────────────────────────────────────────

SCHEMES = {
    # Hyperbolic (wave equation, advection)
    "upwind": {
        "name": "Upwind (first-order)",
        "pde_type": "hyperbolic",
        "order_space": 1,
        "order_time": 1,
        "stability": "CFL ≤ 1",
        "cfl_limit": 1.0,
        "is_explicit": True,
        "is_conservative": True,
        "stencil": "2-point",
        "notes": [
            "Diffusive (numerical viscosity ~ Δx)",
            "Direction-aware (follows characteristic)",
        ],
    },
    "lax_friedrichs": {
        "name": "Lax-Friedrichs",
        "pde_type": "hyperbolic",
        "order_space": 1,
        "order_time": 1,
        "stability": "CFL ≤ 1",
        "cfl_limit": 1.0,
        "is_explicit": True,
        "is_conservative": True,
        "stencil": "3-point",
        "notes": [
            "Very diffusive (numerical viscosity ~ Δx²/Δt)",
            "Simple but smears discontinuities",
        ],
    },
    "lax_wendroff": {
        "name": "Lax-Wendroff",
        "pde_type": "hyperbolic",
        "order_space": 2,
        "order_time": 2,
        "stability": "CFL ≤ 1",
        "cfl_limit": 1.0,
        "is_explicit": True,
        "is_conservative": True,
        "stencil": "3-point",
        "notes": [
            "Second-order accurate",
            "Dispersive (oscillations near shocks)",
            "Richtmyer two-step variant is equivalent",
        ],
    },
    "leapfrog_hyperbolic": {
        "name": "Leapfrog (hyperbolic)",
        "pde_type": "hyperbolic",
        "order_space": 2,
        "order_time": 2,
        "stability": "CFL ≤ 1",
        "cfl_limit": 1.0,
        "is_explicit": True,
        "is_conservative": True,
        "stencil": "3-point space, 3-point time",
        "notes": [
            "Non-dissipative (preserves amplitude)",
            "Has computational mode (odd-even decoupling)",
            "NOT stable for diffusion — only use for wave/advection",
        ],
    },
    "beam_warming": {
        "name": "Beam-Warming (implicit)",
        "pde_type": "hyperbolic",
        "order_space": 2,
        "order_time": 2,
        "stability": "Unconditionally stable",
        "cfl_limit": None,
        "is_explicit": False,
        "is_conservative": True,
        "stencil": "3-point",
        "notes": [
            "Implicit — requires solving linear system",
            "No CFL restriction but accuracy degrades for large CFL",
        ],
    },

    # Parabolic (heat/diffusion equation)
    "ftcs": {
        "name": "FTCS (Forward Time Central Space)",
        "pde_type": "parabolic",
        "order_space": 2,
        "order_time": 1,
        "stability": "CFL ≤ 0.5 (1D)",
        "cfl_limit": 0.5,
        "is_explicit": True,
        "is_conservative": True,
        "stencil": "3-point",
        "notes": [
            "CFL = D×Δt/Δx² ≤ 1/2 (1D), ≤ 1/4 (2D), ≤ 1/6 (3D)",
            "Simple but restrictive time step",
        ],
    },
    "btcs": {
        "name": "BTCS (Backward Time Central Space)",
        "pde_type": "parabolic",
        "order_space": 2,
        "order_time": 1,
        "stability": "Unconditionally stable",
        "cfl_limit": None,
        "is_explicit": False,
        "is_conservative": True,
        "stencil": "3-point",
        "notes": [
            "Implicit — solve tridiagonal system",
            "Dissipative (over-smooths for large Δt)",
            "First-order accurate in time",
        ],
    },
    "crank_nicolson": {
        "name": "Crank-Nicolson",
        "pde_type": "parabolic",
        "order_space": 2,
        "order_time": 2,
        "stability": "Unconditionally stable",
        "cfl_limit": None,
        "is_explicit": False,
        "is_conservative": True,
        "stencil": "3-point",
        "notes": [
            "Second-order accurate in time",
            "Implicit — solve tridiagonal system",
            "Gold standard for diffusion problems",
        ],
    },
    "dufort_frankel": {
        "name": "DuFort-Frankel",
        "pde_type": "parabolic",
        "order_space": 2,
        "order_time": 2,
        "stability": "Unconditionally stable",
        "cfl_limit": None,
        "is_explicit": True,  # Explicit but unconditionally stable!
        "is_conservative": True,
        "stencil": "3-point space, 3-point time",
        "notes": [
            "Explicit AND unconditionally stable (rare!)",
            "Consistency requires Δt/Δx² → 0",
            "Watch out: inconsistent if Δt/Δx → const",
        ],
    },
    "leapfrog_parabolic": {
        "name": "Leapfrog (parabolic)",
        "pde_type": "parabolic",
        "order_space": 2,
        "order_time": 2,
        "stability": "UNCONDITIONALLY UNSTABLE",
        "cfl_limit": 0.0,  # Never stable
        "is_explicit": True,
        "is_conservative": False,
        "stencil": "3-point space, 3-point time",
        "notes": [
            "NEVER use for diffusion — always unstable",
            "Common error: trying leapfrog on parabolic PDE",
            "Use FTCS, Crank-Nicolson, or BTCS instead",
        ],
    },
    "adi": {
        "name": "ADI (Alternating Direction Implicit)",
        "pde_type": "parabolic",
        "order_space": 2,
        "order_time": 2,
        "stability": "Unconditionally stable (2D)",
        "cfl_limit": None,
        "is_explicit": False,
        "is_conservative": True,
        "stencil": "5-point (2D)",
        "notes": [
            "Splits 2D problem into 1D solves",
            "Peaceman-Rachford variant is second-order",
            "Efficient: O(N) per dimension",
        ],
    },
}


# ─── CFL Analysis ───────────────────────────────────────────────────────────

def check_cfl(
    scheme: str,
    cfl_number: float,
) -> CFLReport:
    """Check CFL condition for a numerical scheme.

    The CFL (Courant-Friedrichs-Lewy) condition is a NECESSARY condition
    for stability of explicit schemes. For hyperbolic PDEs: CFL = c×Δt/Δx.
    For parabolic PDEs: CFL = D×Δt/Δx².

    CRITICAL: CFL ≤ limit is necessary but NOT always sufficient!
    Some schemes have additional stability requirements.

    Args:
        scheme: Name of numerical scheme (e.g., "upwind", "ftcs", "crank_nicolson")
        cfl_number: Computed CFL number for your discretization

    Returns:
        CFLReport with stability analysis
    """
    scheme_lower = scheme.lower().replace("-", "_").replace(" ", "_")

    if scheme_lower not in SCHEMES:
        available = list(SCHEMES.keys())
        raise ValueError(f"Unknown scheme '{scheme}'. Available: {available}")

    info = SCHEMES[scheme_lower]
    cfl_limit = info.get("cfl_limit")
    pde_type = info["pde_type"]
    name = info["name"]

    notes = []

    # Determine stability
    if cfl_limit is None:
        # Unconditionally stable
        is_stable = True
        stability_type = "unconditionally stable"
        notes.append("Implicit scheme — no CFL restriction")
        notes.append("But accuracy may degrade for very large CFL")
        actual_limit = float('inf')
    elif cfl_limit == 0.0:
        # Always unstable (e.g., leapfrog for diffusion)
        is_stable = False
        stability_type = "unconditionally unstable"
        notes.append("This scheme is NEVER stable for this PDE type")
        notes.append("Use a different scheme!")
        actual_limit = 0.0
    else:
        # Conditionally stable
        is_stable = cfl_number <= cfl_limit
        stability_type = "conditionally stable"
        actual_limit = cfl_limit
        if is_stable:
            notes.append(f"CFL = {cfl_number:.4f} ≤ {cfl_limit} ✓")
        else:
            notes.append(f"CFL = {cfl_number:.4f} > {cfl_limit} — REDUCE Δt or increase Δx")

    # Add scheme-specific notes
    for note in info.get("notes", []):
        notes.append(note)

    return CFLReport(
        pde_type=pde_type,
        scheme=name,
        cfl_number=cfl_number,
        cfl_limit=actual_limit,
        is_stable=is_stable,
        stability_type=stability_type,
        notes=notes,
    )


def cfl_hyperbolic(
    c: float,
    dt: float,
    dx: float,
) -> float:
    """Compute CFL number for hyperbolic PDE (wave/advection).

    CFL = c × Δt / Δx

    Args:
        c: Wave speed or advection velocity
        dt: Time step
        dx: Grid spacing

    Returns:
        CFL number
    """
    if dx <= 0:
        raise ValueError("Grid spacing dx must be positive")
    return abs(c) * dt / dx


def cfl_parabolic(
    D: float,
    dt: float,
    dx: float,
) -> float:
    """Compute CFL number for parabolic PDE (diffusion).

    CFL = D × Δt / Δx²

    Args:
        D: Diffusion coefficient
        dt: Time step
        dx: Grid spacing

    Returns:
        CFL number
    """
    if dx <= 0:
        raise ValueError("Grid spacing dx must be positive")
    return abs(D) * dt / (dx * dx)


def max_timestep(
    scheme: str,
    c_or_D: float,
    dx: float,
    pde_type: str = "hyperbolic",
    safety_factor: float = 0.9,
) -> float:
    """Compute maximum stable timestep for a scheme.

    Args:
        scheme: Name of numerical scheme
        c_or_D: Wave speed (hyperbolic) or diffusion coefficient (parabolic)
        dx: Grid spacing
        pde_type: "hyperbolic" or "parabolic"
        safety_factor: Multiply limit by this (default 0.9)

    Returns:
        Maximum stable Δt
    """
    scheme_lower = scheme.lower().replace("-", "_").replace(" ", "_")

    if scheme_lower not in SCHEMES:
        raise ValueError(f"Unknown scheme '{scheme}'")

    info = SCHEMES[scheme_lower]
    cfl_limit = info.get("cfl_limit")

    if cfl_limit is None:
        return float('inf')  # Unconditionally stable
    if cfl_limit == 0.0:
        return 0.0  # Never stable

    pde = pde_type.lower()
    if pde == "hyperbolic":
        # CFL = c*dt/dx ≤ limit  =>  dt ≤ limit * dx / c
        dt_max = cfl_limit * dx / abs(c_or_D)
    elif pde == "parabolic":
        # CFL = D*dt/dx² ≤ limit  =>  dt ≤ limit * dx² / D
        dt_max = cfl_limit * dx * dx / abs(c_or_D)
    else:
        raise ValueError(f"Unknown PDE type '{pde_type}'")

    return safety_factor * dt_max


# ─── Von Neumann Stability ──────────────────────────────────────────────────

def von_neumann_analysis(
    scheme: str,
    cfl: float,
    wavenumber_dx: float = math.pi / 2,
) -> VonNeumannReport:
    """Perform Von Neumann stability analysis.

    The Von Neumann method checks stability by examining the amplification
    factor G for Fourier modes: u^{n+1}_j = G × u^n_j.
    Stability requires |G| ≤ 1 for all wavenumbers k.

    Args:
        scheme: Name of numerical scheme
        cfl: CFL number
        wavenumber_dx: k×Δx value to check (default π/2, range [0, π])

    Returns:
        VonNeumannReport with amplification factor analysis
    """
    scheme_lower = scheme.lower().replace("-", "_").replace(" ", "_")
    kdx = wavenumber_dx

    notes = []

    # Compute amplification factor for specific schemes
    if scheme_lower == "upwind":
        # G = 1 - CFL × (1 - e^{-ikΔx}) = 1 - CFL × (1 - cos(kΔx) + i sin(kΔx))
        G = complex(1 - cfl * (1 - math.cos(kdx)), -cfl * math.sin(kdx))
        notes.append("Upwind: G = 1 - CFL × (1 - e^{-ikΔx})")
    elif scheme_lower == "lax_friedrichs":
        # G = cos(kΔx) - i × CFL × sin(kΔx)
        G = complex(math.cos(kdx), -cfl * math.sin(kdx))
        notes.append("Lax-Friedrichs: G = cos(kΔx) - i×CFL×sin(kΔx)")
    elif scheme_lower == "lax_wendroff":
        # G = 1 - CFL²×(1 - cos(kΔx)) - i×CFL×sin(kΔx)
        G = complex(1 - cfl * cfl * (1 - math.cos(kdx)), -cfl * math.sin(kdx))
        notes.append("Lax-Wendroff: G = 1 - CFL²×(1-cos) - i×CFL×sin")
    elif scheme_lower in ["leapfrog_hyperbolic", "leapfrog"]:
        # G² - 2i×CFL×sin(kΔx)×G - 1 = 0
        # For |CFL| ≤ 1: |G| = 1 (exactly)
        a = cfl * math.sin(kdx)
        if abs(a) <= 1:
            G = complex(-a, math.sqrt(1 - a * a))
        else:
            G = complex(-a, math.sqrt(a * a - 1))  # Unstable root
        notes.append("Leapfrog: |G| = 1 if CFL ≤ 1 (non-dissipative)")
    elif scheme_lower == "ftcs":
        # Heat equation: G = 1 - 4×CFL×sin²(kΔx/2)
        G = complex(1 - 4 * cfl * (math.sin(kdx / 2) ** 2), 0)
        notes.append("FTCS: G = 1 - 4×CFL×sin²(kΔx/2)")
    elif scheme_lower in ["btcs", "backward_euler"]:
        # G = 1 / (1 + 4×CFL×sin²(kΔx/2))
        denom = 1 + 4 * cfl * (math.sin(kdx / 2) ** 2)
        G = complex(1 / denom, 0)
        notes.append("BTCS: G = 1/(1 + 4×CFL×sin²) — always |G| < 1")
    elif scheme_lower == "crank_nicolson":
        # G = (1 - 2×CFL×sin²(kΔx/2)) / (1 + 2×CFL×sin²(kΔx/2))
        s2 = math.sin(kdx / 2) ** 2
        num = 1 - 2 * cfl * s2
        denom = 1 + 2 * cfl * s2
        G = complex(num / denom, 0)
        notes.append("Crank-Nicolson: G = (1-2α)/(1+2α), always |G| ≤ 1")
    elif scheme_lower == "leapfrog_parabolic":
        # Always unstable for diffusion
        G = complex(float('inf'), 0)
        notes.append("Leapfrog for diffusion: ALWAYS UNSTABLE")
    else:
        raise ValueError(f"Von Neumann analysis not implemented for '{scheme}'")

    amplitude = abs(G)
    is_stable = amplitude <= 1.0 + 1e-10  # Small tolerance for numerical error
    is_dissipative = amplitude < 1.0 - 1e-10

    # Phase error (for hyperbolic schemes)
    if G.imag != 0 or G.real != 0:
        exact_phase = kdx * cfl  # For advection at CFL=1
        numerical_phase = math.atan2(-G.imag, G.real)
        phase_error = numerical_phase - exact_phase
    else:
        phase_error = None

    return VonNeumannReport(
        scheme=scheme,
        amplification_factor=G,
        amplitude=amplitude,
        is_stable=is_stable,
        is_dissipative=is_dissipative,
        phase_error=phase_error,
        notes=notes,
    )


# ─── Scheme Information ─────────────────────────────────────────────────────

def get_scheme_info(scheme: str) -> SchemeReport:
    """Get detailed information about a numerical scheme.

    Args:
        scheme: Name of numerical scheme

    Returns:
        SchemeReport with full scheme properties
    """
    scheme_lower = scheme.lower().replace("-", "_").replace(" ", "_")

    if scheme_lower not in SCHEMES:
        available = list(SCHEMES.keys())
        raise ValueError(f"Unknown scheme '{scheme}'. Available: {available}")

    info = SCHEMES[scheme_lower]

    return SchemeReport(
        name=info["name"],
        pde_type=info["pde_type"],
        order_space=info["order_space"],
        order_time=info["order_time"],
        stability_condition=info["stability"],
        is_explicit=info["is_explicit"],
        is_conservative=info["is_conservative"],
        stencil=info["stencil"],
        cfl_limit=info.get("cfl_limit"),
        notes=info.get("notes", []),
    )


def list_schemes(pde_type: str = "") -> list[str]:
    """List available numerical schemes.

    Args:
        pde_type: Filter by PDE type ("hyperbolic", "parabolic", or "" for all)

    Returns:
        List of scheme names
    """
    if not pde_type:
        return list(SCHEMES.keys())

    pde = pde_type.lower()
    return [name for name, info in SCHEMES.items() if info["pde_type"] == pde]


# ─── Lax Equivalence Theorem ────────────────────────────────────────────────

def check_lax_equivalence(
    is_consistent: bool,
    consistency_order: int,
    is_stable: bool,
    stability_type: str = "conditionally stable",
    is_linear: bool = True,
) -> LaxEquivalenceReport:
    """Check Lax equivalence theorem conditions.

    THEOREM (Lax, 1956): For a consistent finite difference approximation
    to a well-posed LINEAR initial value problem, stability is the
    NECESSARY AND SUFFICIENT condition for convergence.

    Consistency + Stability ⟺ Convergence (LINEAR problems only!)

    CRITICAL POINT LLMs miss: This theorem ONLY applies to LINEAR problems.
    For nonlinear PDEs, additional conditions are needed.

    Args:
        is_consistent: Does the scheme converge to the PDE as Δx, Δt → 0?
        consistency_order: Order of consistency (truncation error order)
        is_stable: Is the scheme stable?
        stability_type: Type of stability
        is_linear: Is the problem linear?

    Returns:
        LaxEquivalenceReport with analysis
    """
    notes = []

    problem_type = "linear" if is_linear else "nonlinear"
    theorem_applies = is_linear

    if is_linear:
        # Lax theorem gives us convergence
        is_convergent = is_consistent and is_stable
        if is_convergent:
            notes.append("Lax equivalence: consistency + stability ⟹ convergence")
            notes.append(f"Convergence rate: O(Δx^{consistency_order})")
        else:
            if not is_consistent:
                notes.append("NOT consistent — scheme does not approximate PDE")
            if not is_stable:
                notes.append("NOT stable — errors grow unboundedly")
    else:
        # Nonlinear — need to check separately
        is_convergent = None  # Unknown from Lax theorem alone
        notes.append("Lax theorem does NOT apply to nonlinear problems")
        notes.append("Must verify convergence by other means")
        notes.append("Options: entropy conditions, TVD analysis, numerical experiments")

    return LaxEquivalenceReport(
        is_consistent=is_consistent,
        is_stable=is_stable,
        is_convergent=is_convergent if is_convergent is not None else False,
        consistency_order=consistency_order,
        stability_type=stability_type,
        problem_type=problem_type,
        theorem_applies=theorem_applies,
        notes=notes,
    )


# ─── Accuracy Analysis ──────────────────────────────────────────────────────

def analyze_accuracy(scheme: str) -> AccuracyReport:
    """Analyze truncation error and accuracy of a scheme.

    IMPORTANT DISTINCTION:
    - Order of accuracy ≠ stability
    - High-order schemes can be unstable
    - Low-order schemes can be stable and useful

    Args:
        scheme: Name of numerical scheme

    Returns:
        AccuracyReport with truncation error analysis
    """
    scheme_lower = scheme.lower().replace("-", "_").replace(" ", "_")

    if scheme_lower not in SCHEMES:
        raise ValueError(f"Unknown scheme '{scheme}'")

    info = SCHEMES[scheme_lower]
    order_space = info["order_space"]
    order_time = info["order_time"]

    notes = []

    # Determine leading error type
    if scheme_lower in ["upwind", "lax_friedrichs"]:
        leading_error = "dissipative (numerical viscosity)"
        truncation = "O(Δx) + O(Δt)"
        notes.append("First-order schemes have numerical diffusion")
        notes.append("Smears sharp gradients and discontinuities")
        richardson = True
    elif scheme_lower in ["lax_wendroff", "leapfrog_hyperbolic"]:
        leading_error = "dispersive (phase errors)"
        truncation = "O(Δx²) + O(Δt²)"
        notes.append("Second-order schemes have dispersion errors")
        notes.append("Causes oscillations near discontinuities")
        richardson = True
    elif scheme_lower == "crank_nicolson":
        leading_error = "neither dominant (balanced)"
        truncation = "O(Δx²) + O(Δt²)"
        notes.append("Crank-Nicolson is optimally balanced")
        notes.append("Gold standard for parabolic PDEs")
        richardson = True
    elif scheme_lower == "ftcs":
        leading_error = "dissipative"
        truncation = "O(Δx²) + O(Δt)"
        notes.append("First-order in time, second in space")
        notes.append("Time step dominates error")
        richardson = True
    else:
        leading_error = "scheme-dependent"
        truncation = f"O(Δx^{order_space}) + O(Δt^{order_time})"
        richardson = True

    notes.append(f"Space order {order_space}: halving Δx reduces error by 2^{order_space}")
    notes.append(f"Time order {order_time}: halving Δt reduces error by 2^{order_time}")

    return AccuracyReport(
        scheme=info["name"],
        truncation_error=truncation,
        order_space=order_space,
        order_time=order_time,
        leading_error_type=leading_error,
        richardson_possible=richardson,
        notes=notes,
    )


# ─── Common Errors ──────────────────────────────────────────────────────────

def check_common_error(claim: str) -> dict:
    """Check if a claim about numerical PDEs is a common error.

    Args:
        claim: A statement about numerical PDEs to check

    Returns:
        Dict with is_error, correct_statement, explanation
    """
    claim_lower = claim.lower()

    errors = {
        "cfl is sufficient": {
            "is_error": True,
            "correct": "CFL is NECESSARY but not always SUFFICIENT for stability",
            "explanation": "Some schemes need additional conditions beyond CFL",
        },
        "leapfrog for diffusion": {
            "is_error": True,
            "correct": "Leapfrog is UNCONDITIONALLY UNSTABLE for diffusion (parabolic PDEs)",
            "explanation": "Leapfrog only works for hyperbolic PDEs (wave, advection)",
        },
        "implicit always better": {
            "is_error": True,
            "correct": "Implicit schemes are unconditionally stable but may have accuracy issues",
            "explanation": "Large timesteps can cause phase errors and over-smoothing",
        },
        "higher order always better": {
            "is_error": True,
            "correct": "Higher order ≠ better. Stability, efficiency, and oscillations matter",
            "explanation": "Second-order schemes can oscillate near shocks; sometimes first-order preferred",
        },
        "lax applies to nonlinear": {
            "is_error": True,
            "correct": "Lax equivalence theorem is for LINEAR problems only",
            "explanation": "Nonlinear PDEs require additional analysis (entropy conditions, etc.)",
        },
        "consistency implies convergence": {
            "is_error": True,
            "correct": "Consistency alone does NOT imply convergence — need stability too",
            "explanation": "Lax: consistency + stability ⟺ convergence (for linear problems)",
        },
    }

    for key, info in errors.items():
        if key in claim_lower:
            return info

    return {
        "is_error": None,
        "correct": "Claim not in common error database",
        "explanation": "Verify claim against literature",
    }
