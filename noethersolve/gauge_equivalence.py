"""Gauge equivalence and constraint solving.

This module reveals the deep parallel between:
- Type inference in programming languages (finding most general unifiers)
- Gauge fixing in physics (removing redundant degrees of freedom)

Both domains solve the SAME mathematical problem: given constraints with
equivalence classes, find a canonical representative.

Models are blind to this connection because training data keeps these
domains separate.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum


class Domain(Enum):
    TYPE_SYSTEM = "type_system"
    GAUGE_THEORY = "gauge_theory"
    BOTH = "both"


@dataclass
class RedundantDOF:
    """A redundant degree of freedom identified in a constraint system."""
    name: str
    domain: Domain
    description: str
    resolution: str  # How it's typically resolved
    type_analog: Optional[str] = None  # Analog in type theory
    gauge_analog: Optional[str] = None  # Analog in gauge theory


# Known parallels between type systems and gauge theories
KNOWN_PARALLELS: Dict[str, Dict[str, str]] = {
    "most_general_unifier": {
        "type_concept": "Most General Unifier (MGU)",
        "type_description": "Smallest substitution that makes two types equal",
        "gauge_concept": "Gauge orbit representative",
        "gauge_description": "Canonical field configuration from equivalence class",
        "shared_structure": "Both find canonical representatives of equivalence classes",
        "example_type": "unify(List[α], List[Int]) → {α ↦ Int}",
        "example_gauge": "Coulomb gauge: ∇·A = 0 picks one A from [A] = {A + ∇χ}",
    },
    "type_variable": {
        "type_concept": "Type variable (α, β)",
        "type_description": "Placeholder to be unified later",
        "gauge_concept": "Gauge parameter (χ)",
        "gauge_description": "Free function parameterizing gauge transformations",
        "shared_structure": "Both represent unresolved DOF to be fixed",
        "example_type": "α in id: α → α",
        "example_gauge": "χ in A → A + ∇χ",
    },
    "unification_constraint": {
        "type_concept": "Type constraint (τ₁ = τ₂)",
        "type_description": "Requirement that two types be equal",
        "gauge_concept": "Gauge constraint (G = 0)",
        "gauge_description": "Condition removing gauge freedom",
        "shared_structure": "Both reduce redundancy by imposing equations",
        "example_type": "α = Int in function application",
        "example_gauge": "∇·A = 0 (Coulomb), ∂μAμ = 0 (Lorenz)",
    },
    "substitution": {
        "type_concept": "Type substitution σ",
        "type_description": "Mapping from type variables to types",
        "gauge_concept": "Gauge transformation",
        "gauge_description": "Mapping from fields to gauge-equivalent fields",
        "shared_structure": "Both are operations that move within equivalence classes",
        "example_type": "σ = {α ↦ Int, β ↦ α}",
        "example_gauge": "A → A + ∇χ, ψ → e^{iχ}ψ",
    },
    "occurs_check": {
        "type_concept": "Occurs check failure",
        "type_description": "α = List[α] is unsolvable (infinite type)",
        "gauge_concept": "Gribov ambiguity",
        "gauge_description": "Gauge condition doesn't uniquely fix configuration",
        "shared_structure": "Both detect when constraints can't fully determine solution",
        "example_type": "Cannot unify α with Tree[α]",
        "example_gauge": "Multiple A satisfy ∇·A = 0 in large gauge transforms",
    },
    "principal_type": {
        "type_concept": "Principal type",
        "type_description": "Most general type assignable to an expression",
        "gauge_concept": "Residual gauge freedom",
        "gauge_description": "Remaining symmetry after gauge fixing",
        "shared_structure": "Both capture 'leftover' generality after constraints",
        "example_type": "id has principal type ∀α. α → α",
        "example_gauge": "Coulomb gauge leaves global U(1) unfixed",
    },
}


@dataclass
class ConstraintSystem:
    """A system of constraints that may have redundant degrees of freedom."""
    name: str
    domain: Domain
    constraints: List[str]
    free_variables: List[str]
    redundant_dofs: List[RedundantDOF] = field(default_factory=list)

    def add_redundancy(self, name: str, description: str, resolution: str):
        """Identify a redundant DOF in the system."""
        self.redundant_dofs.append(RedundantDOF(
            name=name,
            domain=self.domain,
            description=description,
            resolution=resolution,
        ))


@dataclass
class GaugeEquivalenceReport:
    """Report on gauge equivalence / redundant DOF analysis."""
    input_system: str
    domain: Domain
    has_redundancy: bool
    redundant_dofs: List[RedundantDOF]
    fixing_conditions: List[str]
    residual_freedom: Optional[str]
    cross_domain_analogy: Optional[Dict[str, str]]

    def __str__(self) -> str:
        lines = [
            "=" * 60,
            "GAUGE EQUIVALENCE / REDUNDANT DOF ANALYSIS",
            "=" * 60,
            f"System: {self.input_system}",
            f"Domain: {self.domain.value}",
            f"Has redundancy: {'YES' if self.has_redundancy else 'NO'}",
            "",
        ]

        if self.redundant_dofs:
            lines.append("Redundant degrees of freedom:")
            for dof in self.redundant_dofs:
                lines.append(f"  • {dof.name}: {dof.description}")
                lines.append(f"    Resolution: {dof.resolution}")
                if dof.type_analog:
                    lines.append(f"    Type theory analog: {dof.type_analog}")
                if dof.gauge_analog:
                    lines.append(f"    Gauge theory analog: {dof.gauge_analog}")
            lines.append("")

        if self.fixing_conditions:
            lines.append("Fixing conditions (remove redundancy):")
            for cond in self.fixing_conditions:
                lines.append(f"  • {cond}")
            lines.append("")

        if self.residual_freedom:
            lines.append(f"Residual freedom: {self.residual_freedom}")
            lines.append("")

        if self.cross_domain_analogy:
            lines.append("Cross-domain analogy:")
            for key, val in self.cross_domain_analogy.items():
                lines.append(f"  {key}: {val}")
            lines.append("")

        lines.append("=" * 60)
        return "\n".join(lines)


def check_gauge_equivalence(
    system_description: str,
    constraints: Optional[List[str]] = None,
    domain: str = "auto",
) -> GaugeEquivalenceReport:
    """Analyze a constraint system for gauge equivalence / redundant DOF.

    This tool identifies redundant degrees of freedom in constraint systems
    and shows the parallel between type inference and gauge fixing.

    Args:
        system_description: Description of the system (e.g., "U(1) gauge theory",
            "Hindley-Milner type inference", "Maxwell equations")
        constraints: Optional explicit list of constraints
        domain: "type_system", "gauge_theory", or "auto" to detect

    Returns:
        GaugeEquivalenceReport with redundant DOF analysis.

    Examples:
        >>> check_gauge_equivalence("U(1) gauge theory")
        >>> check_gauge_equivalence("Hindley-Milner type inference")
        >>> check_gauge_equivalence("Yang-Mills SU(2)")
    """
    desc_lower = system_description.lower()

    # Auto-detect domain
    if domain == "auto":
        type_keywords = ["type", "unifier", "hindley", "milner", "polymorphism",
                        "inference", "substitution", "mgu"]
        gauge_keywords = ["gauge", "u(1)", "su(", "maxwell", "yang-mills",
                         "qed", "qcd", "electromagnetism", "field"]

        type_score = sum(1 for kw in type_keywords if kw in desc_lower)
        gauge_score = sum(1 for kw in gauge_keywords if kw in desc_lower)

        if type_score > gauge_score:
            detected_domain = Domain.TYPE_SYSTEM
        elif gauge_score > type_score:
            detected_domain = Domain.GAUGE_THEORY
        else:
            detected_domain = Domain.BOTH
    else:
        detected_domain = Domain(domain)

    redundant_dofs = []
    fixing_conditions = []
    residual_freedom = None
    cross_analogy = None

    # Analyze specific systems
    if "u(1)" in desc_lower or "electromagnetism" in desc_lower or "maxwell" in desc_lower:
        redundant_dofs.append(RedundantDOF(
            name="U(1) gauge freedom",
            domain=Domain.GAUGE_THEORY,
            description="A_μ → A_μ + ∂_μχ leaves physics unchanged",
            resolution="Impose gauge condition: Coulomb (∇·A = 0) or Lorenz (∂_μA^μ = 0)",
            type_analog="Type variable that appears in multiple constraints",
        ))
        fixing_conditions = [
            "Coulomb gauge: ∇·A = 0 (breaks Lorentz invariance, physical)",
            "Lorenz gauge: ∂_μA^μ = 0 (Lorentz covariant)",
            "Axial gauge: A_3 = 0 (simple but axis-dependent)",
            "Temporal gauge: A_0 = 0 (convenient for Hamiltonian formulation)",
        ]
        residual_freedom = "Global U(1) remains (constant χ)"
        cross_analogy = KNOWN_PARALLELS["most_general_unifier"]

    elif "yang-mills" in desc_lower or "su(" in desc_lower:
        redundant_dofs.append(RedundantDOF(
            name="Non-abelian gauge freedom",
            domain=Domain.GAUGE_THEORY,
            description="A_μ → U(A_μ + i∂_μ)U† with U ∈ SU(N)",
            resolution="Gauge condition + Faddeev-Popov ghosts in path integral",
            type_analog="Higher-kinded type unification",
        ))
        fixing_conditions = [
            "Lorenz gauge: ∂_μA^μ = 0 (standard for perturbation theory)",
            "Coulomb gauge: ∇·A = 0 (for lattice QCD)",
            "Maximal Abelian gauge (for monopole physics)",
        ]
        residual_freedom = "Gribov copies: multiple A satisfy gauge condition"
        cross_analogy = {
            "type_concept": "Higher-kinded type inference",
            "gauge_concept": "Non-abelian gauge fixing",
            "shared_structure": "Constraints interact non-linearly; multiple solutions possible",
        }

    elif "hindley" in desc_lower or "milner" in desc_lower or "type inference" in desc_lower:
        redundant_dofs.append(RedundantDOF(
            name="Type variable freedom",
            domain=Domain.TYPE_SYSTEM,
            description="α can be any type until unified",
            resolution="Unification algorithm assigns concrete types",
            gauge_analog="Gauge parameter χ before gauge fixing",
        ))
        fixing_conditions = [
            "Unification: τ₁ = τ₂ determines substitutions",
            "Occurs check: prevents infinite types",
            "Generalization: ∀α quantifies remaining freedom",
        ]
        residual_freedom = "Principal type captures minimal constraints (like residual gauge)"
        cross_analogy = KNOWN_PARALLELS["most_general_unifier"]

    elif "polymorphism" in desc_lower:
        redundant_dofs.append(RedundantDOF(
            name="Parametric polymorphism",
            domain=Domain.TYPE_SYSTEM,
            description="∀α allows α to range over all types",
            resolution="Instantiation: ∀α.τ becomes τ[α/τ']",
            gauge_analog="Global gauge transformations (constant χ)",
        ))
        fixing_conditions = [
            "Instantiation at call site",
            "Type application (explicit types)",
        ]
        residual_freedom = "Identity function id: ∀α. α → α keeps full freedom"
        cross_analogy = KNOWN_PARALLELS["principal_type"]

    else:
        # Generic analysis
        redundant_dofs.append(RedundantDOF(
            name="Unspecified DOF",
            domain=detected_domain,
            description="System may contain redundant degrees of freedom",
            resolution="Add constraints until DOF count matches physical observables",
        ))
        fixing_conditions = ["Add gauge-fixing conditions or unification constraints"]
        residual_freedom = "May have residual symmetry (global transformations)"
        cross_analogy = KNOWN_PARALLELS["unification_constraint"]

    return GaugeEquivalenceReport(
        input_system=system_description,
        domain=detected_domain,
        has_redundancy=len(redundant_dofs) > 0,
        redundant_dofs=redundant_dofs,
        fixing_conditions=fixing_conditions,
        residual_freedom=residual_freedom,
        cross_domain_analogy=cross_analogy,
    )


def explain_parallel(concept: str) -> Optional[Dict[str, str]]:
    """Explain the parallel between type theory and gauge theory for a concept.

    Args:
        concept: Concept name, e.g., "most_general_unifier", "type_variable",
            "occurs_check", "principal_type"

    Returns:
        Dictionary explaining the parallel, or None if unknown.
    """
    concept_lower = concept.lower().replace(" ", "_").replace("-", "_")

    # Try direct match
    if concept_lower in KNOWN_PARALLELS:
        return KNOWN_PARALLELS[concept_lower]

    # Try partial match
    for key, value in KNOWN_PARALLELS.items():
        if concept_lower in key or key in concept_lower:
            return value

    return None


def list_parallels() -> List[str]:
    """List all known type/gauge parallels."""
    return list(KNOWN_PARALLELS.keys())


@dataclass
class UnificationResult:
    """Result of type unification."""
    success: bool
    substitution: Dict[str, str]
    occurs_check_failure: bool = False
    error_message: Optional[str] = None

    def __str__(self) -> str:
        if not self.success:
            if self.occurs_check_failure:
                return f"Unification failed: occurs check ({self.error_message})"
            return f"Unification failed: {self.error_message}"

        if not self.substitution:
            return "Types are already equal (empty substitution)"

        subs = ", ".join(f"{k} ↦ {v}" for k, v in self.substitution.items())
        return f"MGU: {{{subs}}}"


def simple_unify(type1: str, type2: str) -> UnificationResult:
    """Simple first-order type unification (educational, not full HM).

    Demonstrates the parallel with gauge fixing: finding canonical representative.

    Args:
        type1: First type, e.g., "List[α]"
        type2: Second type, e.g., "List[Int]"

    Returns:
        UnificationResult with substitution (= gauge-fixing conditions).

    Examples:
        >>> simple_unify("α", "Int")
        MGU: {α ↦ Int}
        >>> simple_unify("List[α]", "List[Int]")
        MGU: {α ↦ Int}
        >>> simple_unify("α", "List[α]")  # occurs check failure
        Unification failed: occurs check (α occurs in List[α])
    """
    # This is a simplified unification for demonstration
    # Real HM inference is more complex


    # Check if types are identical
    if type1 == type2:
        return UnificationResult(success=True, substitution={})

    # Check for type variables (single Greek letters or single lowercase)
    is_var1 = len(type1) == 1 and (type1.isalpha() and type1.islower() or type1 in "αβγδεζηθικλμνξοπρστυφχψω")
    is_var2 = len(type2) == 1 and (type2.isalpha() and type2.islower() or type2 in "αβγδεζηθικλμνξοπρστυφχψω")

    if is_var1:
        # Occurs check: α cannot unify with something containing α
        if type1 in type2:
            return UnificationResult(
                success=False,
                substitution={},
                occurs_check_failure=True,
                error_message=f"{type1} occurs in {type2}",
            )
        return UnificationResult(success=True, substitution={type1: type2})

    if is_var2:
        if type2 in type1:
            return UnificationResult(
                success=False,
                substitution={},
                occurs_check_failure=True,
                error_message=f"{type2} occurs in {type1}",
            )
        return UnificationResult(success=True, substitution={type2: type1})

    # Check for type constructors: Foo[Bar] pattern
    def parse_constructor(t: str) -> Optional[Tuple[str, str]]:
        if '[' in t and t.endswith(']'):
            idx = t.index('[')
            return (t[:idx], t[idx+1:-1])
        return None

    parsed1 = parse_constructor(type1)
    parsed2 = parse_constructor(type2)

    if parsed1 and parsed2:
        ctor1, arg1 = parsed1
        ctor2, arg2 = parsed2

        if ctor1 != ctor2:
            return UnificationResult(
                success=False,
                substitution={},
                error_message=f"Constructor mismatch: {ctor1} vs {ctor2}",
            )

        # Recurse on arguments
        result = simple_unify(arg1, arg2)
        return result

    # No match
    return UnificationResult(
        success=False,
        substitution={},
        error_message=f"Cannot unify {type1} with {type2}",
    )


# Quick test
if __name__ == "__main__":
    print("=== Gauge Equivalence Tool ===\n")

    # Test gauge theory analysis
    print("--- U(1) Gauge Theory ---")
    report = check_gauge_equivalence("U(1) electromagnetism")
    print(report)

    # Test type inference analysis
    print("--- Hindley-Milner Type Inference ---")
    report = check_gauge_equivalence("Hindley-Milner type inference")
    print(report)

    # Test unification
    print("--- Simple Unification Examples ---")
    print(simple_unify("α", "Int"))
    print(simple_unify("List[α]", "List[Int]"))
    print(simple_unify("α", "List[α]"))
    print(simple_unify("Pair[α,β]", "Pair[Int,String]"))
