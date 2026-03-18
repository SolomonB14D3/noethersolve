"""Dimension-dependent physics formulas.

Models are systematically blind to how physics changes with spatial dimension.
This tool provides verified dimension-aware formulas.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
from enum import Enum


class Dimension(Enum):
    D1 = 1
    D2 = 2
    D3 = 3
    DN = "n"  # general n dimensions


@dataclass
class DimensionalFormula:
    """A physics formula that depends on spatial dimension."""
    concept: str
    dimension: int
    formula: str
    latex: str
    notes: str
    common_error: str  # What models typically get wrong


# Verified dimensional physics database
DIMENSIONAL_PHYSICS: Dict[str, Dict[int, DimensionalFormula]] = {
    "laplacian_greens_function": {
        1: DimensionalFormula(
            concept="Laplacian Green's function",
            dimension=1,
            formula="|x|/2",
            latex=r"G_1(x) = \frac{|x|}{2}",
            notes="Linear in 1D, satisfies G''(x) = δ(x)",
            common_error="Models often guess exp(-|x|) or 1/|x|"
        ),
        2: DimensionalFormula(
            concept="Laplacian Green's function",
            dimension=2,
            formula="-ln(r)/(2π)",
            latex=r"G_2(r) = -\frac{\ln r}{2\pi}",
            notes="Logarithmic in 2D, unique to planar geometry",
            common_error="Models default to 1/r (the 3D form)"
        ),
        3: DimensionalFormula(
            concept="Laplacian Green's function",
            dimension=3,
            formula="1/(4πr)",
            latex=r"G_3(r) = \frac{1}{4\pi r}",
            notes="The familiar Coulomb potential form",
            common_error="This is actually what models default to for ALL dimensions"
        ),
    },
    "coulomb_potential": {
        2: DimensionalFormula(
            concept="Coulomb/electrostatic potential",
            dimension=2,
            formula="-q·ln(r)/(2πε₀)",
            latex=r"V_2(r) = -\frac{q \ln r}{2\pi\varepsilon_0}",
            notes="Logarithmic in 2D (e.g., charged wire cross-section)",
            common_error="Models apply 3D formula 1/r even in 2D contexts"
        ),
        3: DimensionalFormula(
            concept="Coulomb/electrostatic potential",
            dimension=3,
            formula="q/(4πε₀r)",
            latex=r"V_3(r) = \frac{q}{4\pi\varepsilon_0 r}",
            notes="The standard 1/r Coulomb law",
            common_error="Models correctly know this but wrongly apply it to 2D"
        ),
    },
    "vortex_topology": {
        2: DimensionalFormula(
            concept="Vortex structure",
            dimension=2,
            formula="point singularities",
            latex=r"\omega = \Gamma\delta(\mathbf{x})",
            notes="Vorticity is scalar; vortices are 0D points in 2D flow",
            common_error="Models say 'vortex lines' even in 2D"
        ),
        3: DimensionalFormula(
            concept="Vortex structure",
            dimension=3,
            formula="line singularities (tubes)",
            latex=r"\boldsymbol{\omega} = \Gamma\delta^{(2)}",
            notes="Vorticity is a vector; vortices are 1D curves in 3D flow",
            common_error="Models correctly know this but apply it universally"
        ),
    },
    "turbulence_energy_cascade": {
        2: DimensionalFormula(
            concept="Energy cascade direction",
            dimension=2,
            formula="inverse (to large scales)",
            latex=r"\text{Energy} \to k^{-1}",
            notes="2D: enstrophy cascades forward, energy inverse",
            common_error="Models say energy cascades to small scales (3D behavior)"
        ),
        3: DimensionalFormula(
            concept="Energy cascade direction",
            dimension=3,
            formula="forward (to small scales)",
            latex=r"\text{Energy} \to k^{+\infty}",
            notes="3D: energy cascades to small scales, dissipates",
            common_error="Models correctly know this, it's the default"
        ),
    },
    "enstrophy_cascade": {
        2: DimensionalFormula(
            concept="Enstrophy cascade direction",
            dimension=2,
            formula="forward (to small scales)",
            latex=r"\text{Enstrophy} \to k^{+\infty}",
            notes="In 2D, enstrophy (not energy) cascades forward",
            common_error="Models confuse enstrophy with vorticity"
        ),
        3: DimensionalFormula(
            concept="Enstrophy cascade direction",
            dimension=3,
            formula="not independently conserved",
            latex=r"\text{Enstrophy not special in 3D}",
            notes="3D enstrophy is stretched/folded, not conserved",
            common_error="Models don't distinguish 2D/3D enstrophy"
        ),
    },
    "ns_regularity": {
        2: DimensionalFormula(
            concept="Navier-Stokes global regularity",
            dimension=2,
            formula="proven (no finite-time blowup)",
            latex=r"\text{Global existence proven}",
            notes="Ladyzhenskaya (1969): 2D NS has unique global solutions",
            common_error="Models apply 3D Millennium Problem uncertainty to 2D"
        ),
        3: DimensionalFormula(
            concept="Navier-Stokes global regularity",
            dimension=3,
            formula="open (Millennium Problem)",
            latex=r"\text{Open problem: }\$1M",
            notes="Whether smooth initial data stays smooth is unknown",
            common_error="Models know this but don't realize 2D is solved"
        ),
    },
    "point_vortex_stream": {
        2: DimensionalFormula(
            concept="Point vortex stream function",
            dimension=2,
            formula="ψ = -Γ·ln(r)/(2π)",
            latex=r"\psi = -\frac{\Gamma \ln r}{2\pi}",
            notes="Logarithmic stream function for 2D point vortex",
            common_error="Models guess 1/r or r² forms"
        ),
        3: DimensionalFormula(
            concept="Vortex filament induced velocity",
            dimension=3,
            formula="Biot-Savart: v ∝ 1/r²",
            latex=r"v = \frac{\Gamma}{4\pi}\int \frac{d\ell \times \hat{r}}{r^2}",
            notes="3D vortex filaments induce 1/r² velocity",
            common_error="Models may confuse with 2D point vortex"
        ),
    },
    "wave_equation_propagation": {
        1: DimensionalFormula(
            concept="Wave equation propagation",
            dimension=1,
            formula="exact d'Alembert (no tail)",
            latex=r"u(x,t) = f(x-ct) + g(x+ct)",
            notes="Clean propagation, no dispersion or tails",
            common_error="Models don't distinguish 1D uniqueness"
        ),
        2: DimensionalFormula(
            concept="Wave equation propagation",
            dimension=2,
            formula="logarithmic tail (Huygens fails)",
            latex=r"\text{Tail } \propto t^{-1}\ln t",
            notes="2D waves have tails; Huygens' principle fails",
            common_error="Models assume sharp wavefronts like 3D"
        ),
        3: DimensionalFormula(
            concept="Wave equation propagation",
            dimension=3,
            formula="sharp Huygens (no tail)",
            latex=r"u(x,t) = \text{sharp wavefront}",
            notes="Huygens' principle holds; waves have sharp fronts",
            common_error="Models correctly know this, apply universally"
        ),
    },
}


@dataclass
class DimensionCheckResult:
    """Result of checking a physics concept for dimension dependence."""
    concept: str
    dimension_dependent: bool
    formulas: Dict[int, DimensionalFormula]
    model_common_error: str
    recommendation: str

    def __str__(self) -> str:
        lines = [
            f"Concept: {self.concept}",
            f"Dimension-dependent: {'YES' if self.dimension_dependent else 'NO'}",
            "",
        ]

        for dim, formula in sorted(self.formulas.items()):
            lines.append(f"  {dim}D: {formula.formula}")
            lines.append(f"      LaTeX: {formula.latex}")
            lines.append(f"      Note: {formula.notes}")
            lines.append("")

        lines.append(f"Common model error: {self.model_common_error}")
        lines.append(f"Recommendation: {self.recommendation}")

        return "\n".join(lines)


def check_dimension_dependence(concept: str) -> DimensionCheckResult:
    """Check if a physics concept depends on spatial dimension.

    Args:
        concept: Physics concept to check. Examples:
            - "greens_function" or "laplacian_greens_function"
            - "coulomb" or "coulomb_potential"
            - "vortex" or "vortex_topology"
            - "turbulence" or "energy_cascade"
            - "navier_stokes" or "ns_regularity"
            - "wave" or "wave_equation_propagation"

    Returns:
        DimensionCheckResult with formulas for each dimension.
    """
    # Normalize concept name
    concept_lower = concept.lower().replace(" ", "_").replace("-", "_")

    # Map common names to database keys
    name_map = {
        "greens_function": "laplacian_greens_function",
        "green": "laplacian_greens_function",
        "laplacian": "laplacian_greens_function",
        "coulomb": "coulomb_potential",
        "electrostatic": "coulomb_potential",
        "vortex": "vortex_topology",
        "vortices": "vortex_topology",
        "turbulence": "turbulence_energy_cascade",
        "energy_cascade": "turbulence_energy_cascade",
        "cascade": "turbulence_energy_cascade",
        "enstrophy": "enstrophy_cascade",
        "ns": "ns_regularity",
        "navier_stokes": "ns_regularity",
        "blowup": "ns_regularity",
        "regularity": "ns_regularity",
        "stream": "point_vortex_stream",
        "stream_function": "point_vortex_stream",
        "wave": "wave_equation_propagation",
        "huygens": "wave_equation_propagation",
    }

    # Find matching concept
    db_key = name_map.get(concept_lower, concept_lower)

    if db_key not in DIMENSIONAL_PHYSICS:
        # Try partial match
        for key in DIMENSIONAL_PHYSICS:
            if concept_lower in key or key in concept_lower:
                db_key = key
                break

    if db_key not in DIMENSIONAL_PHYSICS:
        available = list(DIMENSIONAL_PHYSICS.keys())
        return DimensionCheckResult(
            concept=concept,
            dimension_dependent=True,  # Assume yes to be safe
            formulas={},
            model_common_error=f"Unknown concept. Available: {available}",
            recommendation="Verify dimension dependence manually."
        )

    formulas = DIMENSIONAL_PHYSICS[db_key]

    # Get common error from any formula
    sample_formula = next(iter(formulas.values()))
    common_error = sample_formula.common_error

    # Check if 2D differs from 3D
    f2 = formulas.get(2)
    f3 = formulas.get(3)
    differs = f2 and f3 and f2.formula != f3.formula

    recommendation = (
        "CAUTION: 2D and 3D formulas DIFFER. Always specify dimension explicitly."
        if differs else
        "Formulas may be similar across dimensions, but verify."
    )

    return DimensionCheckResult(
        concept=sample_formula.concept,
        dimension_dependent=differs,
        formulas=formulas,
        model_common_error=common_error,
        recommendation=recommendation
    )


def get_formula(concept: str, dimension: int) -> Optional[DimensionalFormula]:
    """Get the correct formula for a concept in a specific dimension.

    Args:
        concept: Physics concept (see check_dimension_dependence for options)
        dimension: Spatial dimension (1, 2, or 3)

    Returns:
        DimensionalFormula or None if not found.
    """
    result = check_dimension_dependence(concept)
    return result.formulas.get(dimension)


def list_dimension_dependent_concepts() -> List[str]:
    """List all concepts that depend on spatial dimension."""
    return list(DIMENSIONAL_PHYSICS.keys())


# Quick test
if __name__ == "__main__":
    print("=== Dimension-Dependent Physics Tool ===\n")

    for concept in ["greens_function", "vortex", "turbulence", "ns"]:
        result = check_dimension_dependence(concept)
        print(result)
        print("-" * 60)
