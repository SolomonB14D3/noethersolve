"""noethersolve.origin_of_life — Prebiotic chemistry and origin-of-life calculators.

Computes autocatalytic network detection (RAF theory), prebiotic plausibility
scoring, Eigen's error threshold for replication fidelity, simple RNA folding
free energy (nearest-neighbor model), and Miller-Urey amino acid yield estimates.

Conservation law philosophy: the origin of life IS a thermodynamic phase
transition — from dissipative chemistry to self-replicating chemistry.
Eigen's error threshold is a conservation law for information: below the
threshold, the master sequence is maintained (information conserved); above it,
the quasi-species distribution collapses (information lost). Autocatalytic sets
are the chemical analog of fixed points — the network sustains itself through
mass-action closure.

Usage:
    from noethersolve.origin_of_life import (
        check_autocatalytic_set, prebiotic_plausibility,
        eigen_error_threshold, rna_folding_energy, miller_urey_yield,
    )

    # Check if a reaction network contains an autocatalytic set
    r = check_autocatalytic_set(
        reactions=[
            {"reactants": ["A", "B"], "products": ["C", "A"]},
            {"reactants": ["C"], "products": ["B"]},
        ],
        food_set=["A", "B"],
    )
    print(r)

    # Eigen's error threshold
    r = eigen_error_threshold(genome_length=100, error_rate=0.01, selective_advantage=10)
    print(r)  # mu*L = 1.0, ln(a) = 2.30 -> information survives

    # RNA folding free energy
    r = rna_folding_energy("GCAUGC")
    print(r)  # free energy from nearest-neighbor base pair stacking

    # Prebiotic plausibility
    r = prebiotic_plausibility("C2H5NO2")  # glycine
    print(r)

    # Miller-Urey yield
    r = miller_urey_yield(energy_kJ=1000, atmosphere="reducing")
    print(r)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple


# ─── Constants ────────────────────────────────────────────────────────────────

# RNA nearest-neighbor stacking free energies (kcal/mol) at 37C, 1M NaCl
# From Turner & Mathews (2010) — simplified set for short sequences.
# Key format: (5'XY pair-step, 3'WZ pair-step) -> dG
_NN_PARAMS: Dict[Tuple[str, str], float] = {
    # GC/GC stacks
    ("GC", "GC"): -3.42,
    ("GC", "CG"): -2.36,
    ("CG", "GC"): -2.36,
    ("CG", "CG"): -3.42,
    # AU/AU stacks
    ("AU", "AU"): -0.93,
    ("AU", "UA"): -1.10,
    ("UA", "AU"): -1.10,
    ("UA", "UA"): -0.93,
    # GC/AU mixed stacks
    ("GC", "AU"): -2.08,
    ("GC", "UA"): -2.24,
    ("AU", "GC"): -2.24,
    ("AU", "CG"): -2.08,
    ("CG", "AU"): -1.44,
    ("CG", "UA"): -2.11,
    ("UA", "GC"): -2.11,
    ("UA", "CG"): -1.44,
    # GU wobble pairs (simplified)
    ("GU", "GU"): -0.50,
    ("GU", "UG"): -0.30,
    ("UG", "GU"): -0.30,
    ("UG", "UG"): -0.50,
    ("GC", "GU"): -1.50,
    ("GC", "UG"): -1.27,
    ("GU", "GC"): -1.27,
    ("UG", "GC"): -1.50,
    ("GU", "CG"): -1.50,
    ("CG", "GU"): -1.27,
    ("CG", "UG"): -1.50,
    ("UG", "CG"): -1.27,
    ("AU", "GU"): -0.55,
    ("AU", "UG"): -0.68,
    ("GU", "AU"): -0.68,
    ("UG", "AU"): -0.55,
    ("GU", "UA"): -0.55,
    ("UA", "GU"): -0.68,
    ("UA", "UG"): -0.55,
    ("UG", "UA"): -0.68,
}

# Helix initiation free energy penalty (kcal/mol)
_INIT_PENALTY = 4.09

# Watson-Crick and wobble pair complements
_COMPLEMENTS = {"A": "U", "U": "A", "G": "C", "C": "G"}
_WOBBLE_PAIRS = {("G", "U"), ("U", "G")}

# Known prebiotic molecules and their synthesis pathways
_PREBIOTIC_MOLECULES: Dict[str, Dict] = {
    # Amino acids (Miller-Urey confirmed)
    "C2H5NO2": {"name": "glycine", "score": 0.95, "pathway": "Strecker synthesis from HCN + formaldehyde + NH3"},
    "C3H7NO2": {"name": "alanine", "score": 0.90, "pathway": "Strecker synthesis from acetaldehyde + HCN + NH3"},
    "C4H7NO4": {"name": "aspartic acid", "score": 0.75, "pathway": "Miller-Urey spark discharge"},
    "C5H9NO4": {"name": "glutamic acid", "score": 0.70, "pathway": "Miller-Urey spark discharge"},
    "C3H7NO3": {"name": "serine", "score": 0.65, "pathway": "Miller-Urey spark discharge"},
    "C4H9NO3": {"name": "threonine", "score": 0.45, "pathway": "Miller-Urey (low yield)"},
    "C5H11NO2": {"name": "valine", "score": 0.50, "pathway": "Miller-Urey (moderate yield)"},
    "C6H13NO2": {"name": "leucine", "score": 0.45, "pathway": "Miller-Urey (low yield)"},
    "C9H11NO2": {"name": "phenylalanine", "score": 0.30, "pathway": "Requires aromatic precursors"},
    "C11H12N2O2": {"name": "tryptophan", "score": 0.10, "pathway": "Complex indole synthesis, unlikely prebiotic"},
    "C6H14N4O2": {"name": "arginine", "score": 0.15, "pathway": "Complex guanidinium group, unlikely prebiotic"},
    # Nucleobases
    "C5H5N5": {"name": "adenine", "score": 0.85, "pathway": "HCN pentamerization (Oro 1961)"},
    "C4H5N3O": {"name": "cytosine", "score": 0.60, "pathway": "Cyanoacetylene + urea (Ferris 1968)"},
    "C5H6N2O2": {"name": "uracil", "score": 0.65, "pathway": "Cytosine hydrolysis or direct synthesis"},
    "C5H5N5O": {"name": "guanine", "score": 0.70, "pathway": "HCN polymerization + hydrolysis"},
    # Key prebiotic building blocks
    "HCN": {"name": "hydrogen cyanide", "score": 1.00, "pathway": "Atmospheric photochemistry (CH4 + N2 + UV)"},
    "CH2O": {"name": "formaldehyde", "score": 1.00, "pathway": "CO + H2O photochemistry; meteoritic delivery"},
    "CH5N": {"name": "methylamine", "score": 0.90, "pathway": "Miller-Urey; meteoritic delivery"},
    "C3HN": {"name": "cyanoacetylene", "score": 0.80, "pathway": "Spark discharge from CH4/N2"},
    # Sugars
    "C5H10O5": {"name": "ribose", "score": 0.40, "pathway": "Formose reaction (borate-stabilized, Ricardo 2004)"},
    "C6H12O6": {"name": "glucose", "score": 0.35, "pathway": "Formose reaction (low selectivity)"},
    "C3H6O3": {"name": "glyceraldehyde", "score": 0.55, "pathway": "Formose reaction intermediate"},
}

# Miller-Urey experimental yields
# Based on Miller (1953), Cleaves et al. (2008) reanalysis
_MU_YIELDS: Dict[str, Dict] = {
    "reducing": {
        "description": "CH4 + NH3 + H2O + H2 (original Miller atmosphere)",
        "glycine": 2.1,
        "alanine": 1.7,
        "beta_alanine": 0.76,
        "aspartic_acid": 0.024,
        "glutamic_acid": 0.051,
        "total_amino_acid_pct": 5.9,
        "energy_efficiency": 1.5e-4,  # mol amino acid / kJ
    },
    "weakly_reducing": {
        "description": "CO2 + N2 + H2O + traces H2 (more realistic early Earth)",
        "glycine": 0.25,
        "alanine": 0.10,
        "beta_alanine": 0.03,
        "aspartic_acid": 0.005,
        "glutamic_acid": 0.008,
        "total_amino_acid_pct": 0.5,
        "energy_efficiency": 1.3e-5,
    },
    "neutral": {
        "description": "CO2 + N2 + H2O (no reducing agents)",
        "glycine": 0.01,
        "alanine": 0.003,
        "beta_alanine": 0.001,
        "aspartic_acid": 0.0,
        "glutamic_acid": 0.0,
        "total_amino_acid_pct": 0.02,
        "energy_efficiency": 5e-7,
    },
}


# ─── Report Dataclasses ──────────────────────────────────────────────────────

@dataclass
class AutocatalyticSetReport:
    """Report from autocatalytic set (RAF) analysis."""
    has_raf: bool
    raf_reactions: List[int]
    raf_species: Set[str]
    food_set: Set[str]
    all_species: Set[str]
    closure_iterations: int
    network_size: int
    notes: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        lines = ["=" * 60, "  Autocatalytic Set (RAF) Analysis", "=" * 60]
        lines.append(f"  Network: {self.network_size} reactions, "
                     f"{len(self.all_species)} species")
        lines.append(f"  Food set: {{{', '.join(sorted(self.food_set))}}}")
        lines.append("-" * 60)
        if self.has_raf:
            lines.append(f"  RAF FOUND: {len(self.raf_reactions)} reactions, "
                         f"{len(self.raf_species)} species")
            lines.append(f"  RAF reactions: {self.raf_reactions}")
            lines.append(f"  RAF species: {{{', '.join(sorted(self.raf_species))}}}")
        else:
            lines.append("  No RAF set found in this network.")
        lines.append(f"  Closure iterations: {self.closure_iterations}")
        lines.append("-" * 60)
        lines.append("  RAF = Reflexively Autocatalytic and Food-generated set")
        lines.append("  (Hordijk & Steel, 2004; Kauffman, 1971)")
        if self.notes:
            for n in self.notes:
                lines.append(f"  Note: {n}")
        lines.append("=" * 60)
        return "\n".join(lines)


@dataclass
class PrebioticPlausibilityReport:
    """Report on prebiotic plausibility of a molecule."""
    formula: str
    name: Optional[str]
    score: float
    pathway: Optional[str]
    element_penalties: Dict[str, float]
    complexity_penalty: float
    plausibility_class: str
    notes: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        lines = ["=" * 60, "  Prebiotic Plausibility Assessment", "=" * 60]
        lines.append(f"  Formula: {self.formula}")
        if self.name:
            lines.append(f"  Identified as: {self.name}")
        lines.append(f"  Plausibility score: {self.score:.2f} / 1.00")
        lines.append(f"  Class: {self.plausibility_class}")
        if self.pathway:
            lines.append(f"  Known pathway: {self.pathway}")
        lines.append("-" * 60)
        if self.element_penalties:
            lines.append("  Element penalties:")
            for elem, pen in sorted(self.element_penalties.items()):
                lines.append(f"    {elem}: {pen:+.2f}")
        lines.append(f"  Complexity penalty: {self.complexity_penalty:+.2f}")
        lines.append("-" * 60)
        lines.append("  Score based on: elemental availability, molecular complexity,")
        lines.append("  and match to known prebiotic chemistry pathways.")
        if self.notes:
            for n in self.notes:
                lines.append(f"  Note: {n}")
        lines.append("=" * 60)
        return "\n".join(lines)


@dataclass
class EigenThresholdReport:
    """Report on Eigen's error threshold for replication fidelity."""
    genome_length: int
    error_rate: float
    selective_advantage: float
    mu_L: float
    ln_a: float
    q_per_base: float
    Q_total: float
    Q_min: float
    survives: bool
    safety_margin: float
    max_genome_length: int
    notes: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        lines = ["=" * 60, "  Eigen's Error Threshold", "=" * 60]
        lines.append(f"  Genome length L = {self.genome_length} nt")
        lines.append(f"  Error rate mu = {self.error_rate:.2e} per base per replication")
        lines.append(f"  Selective advantage a = {self.selective_advantage:.4g}")
        lines.append("-" * 60)
        lines.append(f"  Mutation load: mu * L = {self.mu_L:.4g}")
        lines.append(f"  Information capacity: ln(a) = {self.ln_a:.4g}")
        lines.append(f"  Per-base fidelity: q = 1 - mu = {self.q_per_base:.6f}")
        lines.append(f"  Total fidelity: Q = q^L = {self.Q_total:.4e}")
        lines.append(f"  Minimum fidelity: Q_min = 1/a = {self.Q_min:.4e}")
        lines.append("-" * 60)
        if self.survives:
            lines.append(f"  INFORMATION SURVIVES "
                         f"(mu*L = {self.mu_L:.4g} < ln(a) = {self.ln_a:.4g})")
            lines.append(f"  Safety margin: {self.safety_margin:.4g}")
        else:
            lines.append(f"  ERROR CATASTROPHE "
                         f"(mu*L = {self.mu_L:.4g} > ln(a) = {self.ln_a:.4g})")
            lines.append(f"  Deficit: {self.safety_margin:.4g}")
        lines.append(f"  Maximum genome for this error rate: "
                     f"L_max = {self.max_genome_length} nt")
        lines.append("-" * 60)
        lines.append("  Eigen (1971): quasi-species collapses when mu*L > ln(a).")
        lines.append("  RNA world error rates ~0.01/base -> L_max ~ 100 nt.")
        lines.append("  Modern DNA polymerase ~1e-8/base -> L_max ~ 10^9 nt.")
        if self.notes:
            for n in self.notes:
                lines.append(f"  Note: {n}")
        lines.append("=" * 60)
        return "\n".join(lines)


@dataclass
class RNAFoldingReport:
    """Report on RNA folding free energy estimate."""
    sequence: str
    length: int
    structure: str
    base_pairs: List[Tuple[int, int]]
    stacking_energies: List[Tuple[str, float]]
    dG_stacking: float
    dG_init: float
    dG_total: float
    n_gc: int
    n_au: int
    n_gu: int
    notes: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        lines = ["=" * 60, "  RNA Folding Free Energy (Nearest-Neighbor)", "=" * 60]
        lines.append(f"  Sequence: {self.sequence}")
        lines.append(f"  Length: {self.length} nt")
        lines.append(f"  Structure: {self.structure}")
        lines.append(f"  Base pairs: {len(self.base_pairs)} "
                     f"({self.n_gc} GC, {self.n_au} AU, {self.n_gu} GU)")
        lines.append("-" * 60)
        if self.stacking_energies:
            lines.append("  Stacking contributions:")
            for desc, dg in self.stacking_energies:
                lines.append(f"    {desc}: {dg:+.2f} kcal/mol")
        lines.append(f"  Total stacking: {self.dG_stacking:+.2f} kcal/mol")
        lines.append(f"  Initiation penalty: {self.dG_init:+.2f} kcal/mol")
        lines.append(f"  Total dG: {self.dG_total:+.2f} kcal/mol")
        lines.append("-" * 60)
        if self.dG_total < -5.0:
            lines.append("  Strongly stable fold (dG < -5 kcal/mol)")
        elif self.dG_total < -1.0:
            lines.append("  Moderately stable fold (-5 < dG < -1 kcal/mol)")
        elif self.dG_total < 0:
            lines.append("  Marginally stable fold (-1 < dG < 0 kcal/mol)")
        else:
            lines.append("  Unstable (dG > 0): duplex does not form spontaneously")
        lines.append("  Model: Turner nearest-neighbor parameters (37C, 1M NaCl)")
        if self.notes:
            for n in self.notes:
                lines.append(f"  Note: {n}")
        lines.append("=" * 60)
        return "\n".join(lines)


@dataclass
class MillerUreyReport:
    """Report on estimated Miller-Urey amino acid yield."""
    energy_kJ: float
    atmosphere: str
    atmosphere_description: str
    total_yield_pct: float
    amino_acid_yields: Dict[str, float]
    estimated_mass_mg: float
    energy_efficiency: float
    notes: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        lines = ["=" * 60, "  Miller-Urey Amino Acid Yield Estimate", "=" * 60]
        lines.append(f"  Energy input: {self.energy_kJ:.4g} kJ")
        lines.append(f"  Atmosphere: {self.atmosphere}")
        lines.append(f"  ({self.atmosphere_description})")
        lines.append("-" * 60)
        lines.append(f"  Total yield: {self.total_yield_pct:.3g}% of carbon as amino acids")
        lines.append(f"  Energy efficiency: {self.energy_efficiency:.2e} mol AA/kJ")
        lines.append(f"  Estimated total AA mass: {self.estimated_mass_mg:.4g} mg")
        lines.append("-" * 60)
        lines.append("  Individual amino acid yields (% of carbon):")
        for aa, yld in sorted(self.amino_acid_yields.items(), key=lambda x: -x[1]):
            lines.append(f"    {aa}: {yld:.3g}%")
        lines.append("-" * 60)
        lines.append("  Based on Miller (1953) and Cleaves et al. (2008) reanalysis.")
        lines.append("  Reducing atmosphere yields ~10x more than weakly reducing.")
        if self.notes:
            for n in self.notes:
                lines.append(f"  Note: {n}")
        lines.append("=" * 60)
        return "\n".join(lines)


# ─── Helper Functions ─────────────────────────────────────────────────────────

def _parse_formula(formula: str) -> Dict[str, int]:
    """Parse a molecular formula like 'C2H5NO2' into element counts."""
    counts: Dict[str, int] = {}
    i = 0
    while i < len(formula):
        if formula[i].isupper():
            elem = formula[i]
            i += 1
            if i < len(formula) and formula[i].islower():
                elem += formula[i]
                i += 1
            num_str = ""
            while i < len(formula) and formula[i].isdigit():
                num_str += formula[i]
                i += 1
            count = int(num_str) if num_str else 1
            counts[elem] = counts.get(elem, 0) + count
        else:
            i += 1
    return counts


def _can_pair(b1: str, b2: str) -> bool:
    """Check if two RNA bases can form a Watson-Crick or wobble pair."""
    wc = _COMPLEMENTS.get(b1)
    if wc == b2:
        return True
    return (b1, b2) in _WOBBLE_PAIRS


def _pair_type(b1: str, b2: str) -> str:
    """Return pair type: 'GC', 'AU', or 'GU'."""
    pair = tuple(sorted([b1, b2]))
    if pair == ("C", "G"):
        return "GC"
    elif pair == ("A", "U"):
        return "AU"
    elif pair == ("G", "U"):
        return "GU"
    return "unknown"


# ─── Public API ───────────────────────────────────────────────────────────────

def check_autocatalytic_set(
    reactions: List[Dict],
    food_set: List[str],
) -> AutocatalyticSetReport:
    """Check if a reaction network contains a reflexively autocatalytic
    food-generated (RAF) set.

    A RAF set is a subset R of reactions such that:
    1. Every reactant of every reaction in R is either in the food set or
       produced by some reaction in R (food-generated / closure).
    2. Every reaction in R is catalyzed by a molecule that is either in
       the food set or produced by some reaction in R (autocatalytic).

    If no explicit catalyst is specified for a reaction, it is treated as
    uncatalyzed (only the food-generated closure condition applies).

    Args:
        reactions: list of dicts with keys:
            - "reactants": list of species names
            - "products": list of species names
            - "catalyst" (optional): species that catalyzes this reaction
        food_set: list of species available from the environment

    Returns:
        AutocatalyticSetReport with RAF detection results.

    Example:
        check_autocatalytic_set(
            reactions=[
                {"reactants": ["A", "B"], "products": ["C", "A"], "catalyst": "C"},
                {"reactants": ["C"], "products": ["B"], "catalyst": "A"},
            ],
            food_set=["A", "B"],
        )
    """
    if not reactions:
        raise ValueError("Reaction list cannot be empty")
    if not food_set:
        raise ValueError("Food set cannot be empty")

    food = set(food_set)
    n = len(reactions)

    # Collect all species
    all_species: Set[str] = set(food)
    for rxn in reactions:
        all_species.update(rxn.get("reactants", []))
        all_species.update(rxn.get("products", []))
        cat = rxn.get("catalyst")
        if cat:
            all_species.add(cat)

    # Hordijk-Steel iterative pruning algorithm
    active = set(range(n))
    iterations = 0
    max_iter = n + 1

    while iterations < max_iter:
        iterations += 1

        # Compute closure: species available from food + active products
        available = set(food)
        changed = True
        while changed:
            changed = False
            for i in active:
                rxn = reactions[i]
                reactants = set(rxn.get("reactants", []))
                if reactants.issubset(available):
                    products = set(rxn.get("products", []))
                    new = products - available
                    if new:
                        available.update(new)
                        changed = True

        # Prune reactions whose requirements are not met
        to_remove = set()
        for i in active:
            rxn = reactions[i]
            reactants = set(rxn.get("reactants", []))
            if not reactants.issubset(available):
                to_remove.add(i)
                continue
            catalyst = rxn.get("catalyst")
            if catalyst is not None and catalyst not in available:
                to_remove.add(i)

        if not to_remove:
            break
        active -= to_remove

    has_raf = len(active) > 0

    # Collect RAF species
    raf_species: Set[str] = set()
    if has_raf:
        for i in active:
            rxn = reactions[i]
            raf_species.update(rxn.get("reactants", []))
            raf_species.update(rxn.get("products", []))
            cat = rxn.get("catalyst")
            if cat:
                raf_species.add(cat)

    notes: List[str] = []
    if has_raf:
        if len(active) == n:
            notes.append("The entire network is a RAF set")
        else:
            notes.append(f"RAF subset: {len(active)}/{n} reactions")
        # Check for autocatalytic species
        products_in_raf: Set[str] = set()
        reactants_in_raf: Set[str] = set()
        for i in active:
            products_in_raf.update(reactions[i].get("products", []))
            reactants_in_raf.update(reactions[i].get("reactants", []))
        overlap = products_in_raf & reactants_in_raf
        if overlap:
            notes.append(f"Autocatalytic species (product AND reactant): "
                         f"{{{', '.join(sorted(overlap))}}}")
    else:
        notes.append("No food-generated autocatalytic subset exists")
        notes.append("Consider: expanding the food set, adding catalytic "
                     "annotations, or adding intermediate reactions")

    return AutocatalyticSetReport(
        has_raf=has_raf,
        raf_reactions=sorted(active),
        raf_species=raf_species,
        food_set=food,
        all_species=all_species,
        closure_iterations=iterations,
        network_size=n,
        notes=notes,
    )


def prebiotic_plausibility(
    formula: str,
) -> PrebioticPlausibilityReport:
    """Score how plausible a molecule is under prebiotic conditions.

    Scoring based on:
    1. Match to known prebiotic molecules database (~30 molecules)
    2. Elemental composition (C, H, N, O highly available; S, P less so)
    3. Molecular complexity (atom count as proxy)

    Args:
        formula: molecular formula (e.g., "C2H5NO2" for glycine)

    Returns:
        PrebioticPlausibilityReport with score and analysis.

    Example:
        prebiotic_plausibility("C2H5NO2")  # glycine -> score 0.95
        prebiotic_plausibility("C11H12N2O2")  # tryptophan -> score 0.10
    """
    # Check database first
    known = _PREBIOTIC_MOLECULES.get(formula)

    counts = _parse_formula(formula)
    if not counts:
        raise ValueError(f"Could not parse formula: {formula}")

    total_atoms = sum(counts.values())

    # Element availability scores (higher = more available prebiotically)
    element_availability = {
        "H": 1.0, "C": 0.9, "N": 0.8, "O": 0.9,
        "S": 0.3, "P": 0.4, "Fe": 0.3, "Mg": 0.3,
        "Ca": 0.2, "Na": 0.3, "K": 0.2, "Cl": 0.4,
        "Zn": 0.1, "Cu": 0.1, "Se": 0.05, "Mo": 0.05,
        "Si": 0.5, "B": 0.3, "F": 0.15, "Br": 0.1,
    }

    element_penalties: Dict[str, float] = {}
    weighted_score = 0.0
    total_weight = 0.0

    for elem, count in counts.items():
        avail = element_availability.get(elem, 0.01)
        penalty = -(1.0 - avail) * count / max(total_atoms, 1)
        element_penalties[elem] = round(penalty, 4)
        weighted_score += avail * count
        total_weight += count

    base_score = weighted_score / total_weight if total_weight > 0 else 0.0

    # Complexity penalty: larger molecules are less likely to form
    if total_atoms <= 5:
        complexity_penalty = 0.0
    elif total_atoms <= 15:
        complexity_penalty = -0.05 * (total_atoms - 5) / 10
    elif total_atoms <= 30:
        complexity_penalty = -0.05 - 0.15 * (total_atoms - 15) / 15
    else:
        complexity_penalty = -0.20 - 0.30 * min((total_atoms - 30) / 30, 1.0)

    if known:
        score = known["score"]
        name = known["name"]
        pathway = known["pathway"]
    else:
        score = max(0.0, min(1.0, base_score + complexity_penalty))
        name = None
        pathway = None

    if score >= 0.80:
        plausibility_class = "very plausible (confirmed prebiotic synthesis)"
    elif score >= 0.60:
        plausibility_class = "plausible (likely prebiotic pathways exist)"
    elif score >= 0.40:
        plausibility_class = "possible (some prebiotic routes, low yield)"
    elif score >= 0.20:
        plausibility_class = "unlikely (requires specialized conditions)"
    else:
        plausibility_class = "implausible (no known prebiotic pathway)"

    notes: List[str] = []
    if "S" in counts:
        notes.append("Sulfur compounds: limited prebiotic sources "
                     "(volcanic H2S, meteoritic)")
    if "P" in counts:
        notes.append("Phosphorus: prebiotic phosphorylation is a major "
                     "unsolved problem (Schreiber et al. 2017)")
    if total_atoms > 30:
        notes.append("Large molecule: spontaneous assembly unlikely "
                     "without template/catalyst")
    if not known and score > 0.5:
        notes.append("Not in prebiotic database -- score is estimate "
                     "from elemental composition")

    return PrebioticPlausibilityReport(
        formula=formula,
        name=name,
        score=round(score, 3),
        pathway=pathway,
        element_penalties=element_penalties,
        complexity_penalty=round(complexity_penalty, 4),
        plausibility_class=plausibility_class,
        notes=notes,
    )


def eigen_error_threshold(
    genome_length: int,
    error_rate: float,
    selective_advantage: float,
) -> EigenThresholdReport:
    """Check Eigen's error threshold for replication fidelity.

    The error threshold condition: mu * L < ln(a)

    Where:
        mu = per-base error rate per replication
        L = genome length (nucleotides)
        a = selective advantage of master sequence = f_master / f_average

    If mu * L > ln(a), the quasi-species distribution collapses and
    the master sequence is lost (error catastrophe).

    Args:
        genome_length: genome length in nucleotides (L)
        error_rate: per-base error rate per replication (mu), in (0, 1)
        selective_advantage: fitness ratio f_master / f_average (a > 1)

    Returns:
        EigenThresholdReport with threshold analysis.

    Example:
        eigen_error_threshold(100, 0.01, 10)
        # mu*L = 1.0, ln(a) = 2.30 -> information survives
    """
    if genome_length <= 0:
        raise ValueError(f"Genome length must be positive, got {genome_length}")
    if error_rate <= 0 or error_rate >= 1:
        raise ValueError(f"Error rate must be in (0, 1), got {error_rate}")
    if selective_advantage <= 1:
        raise ValueError(
            f"Selective advantage must be > 1, got {selective_advantage}")

    L = genome_length
    mu = error_rate
    a = selective_advantage

    mu_L = mu * L
    ln_a = math.log(a)
    q = 1 - mu
    Q_total = q ** L
    Q_min = 1.0 / a
    survives = mu_L < ln_a
    safety_margin = ln_a - mu_L
    max_genome = int(ln_a / mu)

    notes: List[str] = []
    if mu > 0.01:
        notes.append(f"High error rate ({mu:.2e}): typical of RNA replicases (~0.01)")
    elif mu < 1e-6:
        notes.append(f"Very low error rate ({mu:.2e}): typical of DNA polymerase "
                     "+ proofreading")
    if L < 100:
        notes.append(f"Short genome ({L} nt): compatible with RNA world "
                     "ribozyme sizes")
    elif L > 10000:
        notes.append(f"Large genome ({L} nt): requires high-fidelity replication")
    if survives and safety_margin < 0.5:
        notes.append("Near threshold: small increase in error rate could "
                     "cause information loss")
    if not survives:
        notes.append(f"To rescue: reduce mu below {ln_a / L:.2e}/base, "
                     f"or shorten genome below {max_genome} nt")

    return EigenThresholdReport(
        genome_length=L,
        error_rate=mu,
        selective_advantage=a,
        mu_L=mu_L,
        ln_a=ln_a,
        q_per_base=q,
        Q_total=Q_total,
        Q_min=Q_min,
        survives=survives,
        safety_margin=safety_margin,
        max_genome_length=max_genome,
        notes=notes,
    )


def rna_folding_energy(
    sequence: str,
    structure: Optional[str] = None,
) -> RNAFoldingReport:
    """Estimate RNA folding free energy using nearest-neighbor stacking model.

    For short RNA sequences, computes the free energy of a given secondary
    structure. Uses Turner nearest-neighbor stacking parameters (37C, 1M NaCl).

    If no structure is given, assumes the sequence folds as a simple
    self-complementary duplex (pairs from outside in).

    For explicit structure, provide dot-bracket notation:
        '(' = paired (5' side), ')' = paired (3' side), '.' = unpaired

    Args:
        sequence: RNA sequence (A, U, G, C only), uppercase. T converted to U.
        structure: optional dot-bracket secondary structure

    Returns:
        RNAFoldingReport with free energy breakdown.

    Example:
        rna_folding_energy("GCAUGC")
        rna_folding_energy("GGGAAACCC", "(((...)))")
    """
    seq = sequence.upper().replace("T", "U")
    for c in seq:
        if c not in "AUGC":
            raise ValueError(
                f"Invalid RNA base '{c}' in sequence. Use A, U, G, C only.")
    if len(seq) < 2:
        raise ValueError("Sequence must be at least 2 nucleotides")

    base_pairs: List[Tuple[int, int]] = []

    if structure is not None:
        if len(structure) != len(seq):
            raise ValueError(
                f"Structure length ({len(structure)}) != "
                f"sequence length ({len(seq)})")
        stack: List[int] = []
        for i, c in enumerate(structure):
            if c == "(":
                stack.append(i)
            elif c == ")":
                if not stack:
                    raise ValueError(f"Unmatched ')' at position {i}")
                j = stack.pop()
                if _can_pair(seq[j], seq[i]):
                    base_pairs.append((j, i))
                else:
                    raise ValueError(
                        f"Cannot pair {seq[j]}({j}) with {seq[i]}({i})")
            elif c == ".":
                pass
            else:
                raise ValueError(
                    f"Invalid structure character '{c}' at position {i}")
        if stack:
            raise ValueError(f"Unmatched '(' at positions {stack}")
    else:
        # Auto-detect: try simple self-complementary duplex
        n = len(seq)
        i, j = 0, n - 1
        while i < j:
            if _can_pair(seq[i], seq[j]):
                base_pairs.append((i, j))
                i += 1
                j -= 1
            else:
                break
        # Build dot-bracket
        struct_chars = ["."] * n
        for pi, pj in base_pairs:
            struct_chars[pi] = "("
            struct_chars[pj] = ")"
        structure = "".join(struct_chars)

    base_pairs.sort()

    # Count pair types
    n_gc = 0
    n_au = 0
    n_gu = 0
    for i, j in base_pairs:
        pt = _pair_type(seq[i], seq[j])
        if pt == "GC":
            n_gc += 1
        elif pt == "AU":
            n_au += 1
        elif pt == "GU":
            n_gu += 1

    # Compute stacking energies for consecutive base pairs
    stacking_energies: List[Tuple[str, float]] = []
    dG_stacking = 0.0

    for idx in range(len(base_pairs) - 1):
        i1, j1 = base_pairs[idx]
        i2, j2 = base_pairs[idx + 1]
        # Check if consecutive stacking pairs
        if i2 == i1 + 1 and j2 == j1 - 1:
            top_pair = seq[i1] + seq[j1]
            bot_pair = seq[i2] + seq[j2]
            key = (top_pair, bot_pair)
            dG = _NN_PARAMS.get(key, -1.0)
            desc = (f"5'{seq[i1]}{seq[i2]}/3'{seq[j1]}{seq[j2]} "
                    f"({top_pair}/{bot_pair})")
            stacking_energies.append((desc, dG))
            dG_stacking += dG

    # Initiation penalty (per helix)
    n_helices = 1 if base_pairs else 0
    dG_init = _INIT_PENALTY * n_helices
    dG_total = dG_stacking + dG_init

    notes: List[str] = []
    if not base_pairs:
        notes.append("No base pairs found -- sequence may not fold")
        dG_total = 0.0
        dG_init = 0.0
    if len(seq) > 30:
        notes.append("Nearest-neighbor model is approximate for sequences "
                     "> 30 nt; use ViennaRNA or mfold for accurate predictions")
    if n_gu > 0:
        notes.append(f"{n_gu} GU wobble pair(s): weaker than Watson-Crick "
                     "but common in RNA structure")

    return RNAFoldingReport(
        sequence=seq,
        length=len(seq),
        structure=structure,
        base_pairs=base_pairs,
        stacking_energies=stacking_energies,
        dG_stacking=round(dG_stacking, 2),
        dG_init=round(dG_init, 2),
        dG_total=round(dG_total, 2),
        n_gc=n_gc,
        n_au=n_au,
        n_gu=n_gu,
        notes=notes,
    )


def miller_urey_yield(
    energy_kJ: float,
    atmosphere: str = "reducing",
    carbon_mass_g: float = 1.0,
) -> MillerUreyReport:
    """Estimate amino acid yield from Miller-Urey-type spark discharge.

    Based on published experimental data from Miller (1953) and the Cleaves
    et al. (2008) reanalysis using modern analytical techniques.

    Args:
        energy_kJ: total energy input in kilojoules
        atmosphere: "reducing" (CH4+NH3+H2+H2O, original Miller),
                    "weakly_reducing" (CO2+N2+H2+H2O, more realistic),
                    or "neutral" (CO2+N2+H2O, no reducing agents)
        carbon_mass_g: total carbon mass in system (grams), default 1.0

    Returns:
        MillerUreyReport with estimated yields by amino acid.

    Example:
        miller_urey_yield(1000, "reducing")
        miller_urey_yield(500, "weakly_reducing", carbon_mass_g=2.0)
    """
    if energy_kJ <= 0:
        raise ValueError(f"Energy must be positive, got {energy_kJ}")
    if carbon_mass_g <= 0:
        raise ValueError(f"Carbon mass must be positive, got {carbon_mass_g}")

    atmosphere = atmosphere.lower().strip()
    if atmosphere not in _MU_YIELDS:
        raise ValueError(
            f"Unknown atmosphere '{atmosphere}'. "
            f"Available: {list(_MU_YIELDS.keys())}")

    data = _MU_YIELDS[atmosphere]

    # Energy scaling: reference is ~340 kJ for week-long spark discharge
    ref_energy = 340.0
    energy_scale = min(energy_kJ / ref_energy, 5.0)  # diminishing returns

    amino_acid_yields = {}
    for key, value in data.items():
        if key not in ("description", "total_amino_acid_pct", "energy_efficiency"):
            amino_acid_yields[key] = round(value * energy_scale, 6)

    total_yield = data["total_amino_acid_pct"] * energy_scale
    efficiency = data["energy_efficiency"]

    # Estimate total mass: yield% of carbon * carbon_mass
    estimated_mass_mg = (total_yield / 100.0) * carbon_mass_g * 1000.0

    notes: List[str] = []
    if atmosphere == "reducing":
        notes.append("Original Miller atmosphere: highest yields but now "
                     "considered unlikely for early Earth")
    elif atmosphere == "weakly_reducing":
        notes.append("Weakly reducing: consensus best estimate for early "
                     "Earth atmosphere (Kasting 1993, Catling & Zahnle 2020)")
    elif atmosphere == "neutral":
        notes.append("Neutral atmosphere: minimal amino acid production; "
                     "would require other energy sources (UV, impacts)")
    if energy_kJ > 5 * ref_energy:
        notes.append("Energy well above reference: yields capped at ~5x "
                     "(diminishing returns)")
    if energy_kJ < 10:
        notes.append("Low energy input: may be below threshold for "
                     "significant synthesis")

    return MillerUreyReport(
        energy_kJ=energy_kJ,
        atmosphere=atmosphere,
        atmosphere_description=data["description"],
        total_yield_pct=round(total_yield, 4),
        amino_acid_yields=amino_acid_yields,
        estimated_mass_mg=round(estimated_mass_mg, 4),
        energy_efficiency=efficiency,
        notes=notes,
    )
