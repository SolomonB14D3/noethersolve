#!/usr/bin/env python3
"""origin_of_life_lab.py -- Prebiotic chemistry analysis prototype.

Chains NoetherSolve origin-of-life tools (autocatalytic sets, prebiotic
plausibility, Eigen error threshold, RNA folding, Miller-Urey yields) to
analyze abiogenesis scenarios and evaluate molecular origins.

Usage:
    python labs/origin_of_life_lab.py
    python labs/origin_of_life_lab.py --verbose
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

# Ensure noethersolve is importable
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from noethersolve.origin_of_life import (
    check_autocatalytic_set,
    prebiotic_plausibility,
    eigen_error_threshold,
    rna_folding_energy,
    miller_urey_yield,
)

RESULTS_DIR = _ROOT / "results" / "labs" / "origin_of_life"


# ---------------------------------------------------------------------------
# Scenario definitions
# ---------------------------------------------------------------------------

@dataclass
class AutocatalyticScenario:
    """A reaction network scenario to test for autocatalysis."""
    name: str
    description: str
    reactions: List[Dict[str, List[str]]]  # list of {reactants: [...], products: [...]}
    food_set: List[str]  # available from environment


@dataclass
class MoleculeScenario:
    """A molecule to evaluate for prebiotic plausibility."""
    name: str
    formula: str
    description: str


@dataclass
class ReplicatorScenario:
    """A replicator scenario for Eigen threshold analysis."""
    name: str
    description: str
    genome_length: int
    error_rate: float  # per nucleotide per replication
    selective_advantage: float


@dataclass
class RNAScenario:
    """An RNA sequence for folding energy analysis."""
    name: str
    sequence: str
    description: str


@dataclass
class MillerUreyScenario:
    """A Miller-Urey atmosphere/energy scenario."""
    name: str
    description: str
    energy_kJ: float
    atmosphere: str  # "reducing", "neutral", etc.


# Sample scenarios
AUTOCATALYTIC_SCENARIOS: List[AutocatalyticScenario] = [
    AutocatalyticScenario(
        name="simple_cycle",
        description="Simple autocatalytic cycle: A catalyzes B formation, B catalyzes A",
        reactions=[
            {"reactants": ["A", "X"], "products": ["A", "B"]},
            {"reactants": ["B", "Y"], "products": ["A", "B"]},
        ],
        food_set=["X", "Y"],
    ),
    AutocatalyticScenario(
        name="formose_like",
        description="Formose-like reaction: formaldehyde -> sugars autocatalytically",
        reactions=[
            {"reactants": ["HCHO", "HCHO"], "products": ["C2"]},
            {"reactants": ["C2", "HCHO"], "products": ["C3"]},
            {"reactants": ["C3", "HCHO"], "products": ["C4"]},
            {"reactants": ["C4"], "products": ["C2", "C2"]},  # retroaldol
        ],
        food_set=["HCHO"],
    ),
    AutocatalyticScenario(
        name="replicase_emergence",
        description="Emergence of ribozyme replicase",
        reactions=[
            {"reactants": ["R", "M"], "products": ["R", "R"]},  # R catalyzes own copy
            {"reactants": ["N1", "N2", "N3"], "products": ["M"]},  # monomer synthesis
        ],
        food_set=["N1", "N2", "N3"],
    ),
]

MOLECULE_SCENARIOS: List[MoleculeScenario] = [
    MoleculeScenario("glycine", "C2H5NO2", "Simplest amino acid - found in meteorites"),
    MoleculeScenario("alanine", "C3H7NO2", "Common prebiotic amino acid"),
    MoleculeScenario("adenine", "C5H5N5", "Nucleobase - can form from HCN polymerization"),
    MoleculeScenario("ribose", "C5H10O5", "RNA sugar - formose reaction product"),
    MoleculeScenario("urea", "CH4N2O", "Prebiotic nitrogen carrier - Miller-Urey product"),
]

REPLICATOR_SCENARIOS: List[ReplicatorScenario] = [
    ReplicatorScenario(
        name="minimal_ribozyme",
        description="Minimal ribozyme (~50 nt) with moderate fidelity",
        genome_length=50,
        error_rate=0.01,
        selective_advantage=10.0,
    ),
    ReplicatorScenario(
        name="hypercycle_member",
        description="Hypercycle member (~100 nt), higher fidelity needed",
        genome_length=100,
        error_rate=0.005,
        selective_advantage=5.0,
    ),
    ReplicatorScenario(
        name="protocell_genome",
        description="Early protocell genome (~1000 nt)",
        genome_length=1000,
        error_rate=0.001,
        selective_advantage=2.0,
    ),
    ReplicatorScenario(
        name="rna_world_limit",
        description="RNA world complexity limit - maximum viable genome",
        genome_length=5000,
        error_rate=0.001,
        selective_advantage=1.5,
    ),
]

RNA_SCENARIOS: List[RNAScenario] = [
    RNAScenario("hammerhead_core", "GCGAUGCGGCUGAUG", "Hammerhead ribozyme core motif"),
    RNAScenario("simple_hairpin", "GCAUGC", "Simple GC-rich hairpin"),
    RNAScenario("tetraloop", "GGAACC", "GNRA tetraloop sequence"),
    RNAScenario("aptamer_start", "GGCAUGGCAUGGCC", "ATP aptamer-like sequence"),
]

MILLER_UREY_SCENARIOS: List[MillerUreyScenario] = [
    MillerUreyScenario(
        name="classic_reducing",
        description="Classic Miller-Urey: reducing atmosphere (CH4/NH3/H2)",
        energy_kJ=1000,
        atmosphere="reducing",
    ),
    MillerUreyScenario(
        name="neutral_co2",
        description="Neutral CO2-rich atmosphere - more realistic early Earth",
        energy_kJ=1000,
        atmosphere="neutral",
    ),
    MillerUreyScenario(
        name="volcanic_spark",
        description="Volcanic lightning in ash plumes - localized reducing",
        energy_kJ=500,
        atmosphere="reducing",
    ),
]


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------

@dataclass
class AutocatalyticResult:
    """Result from autocatalytic set analysis."""
    name: str
    is_autocatalytic: bool
    raf_size: int
    molecules_in_raf: List[str]
    closure_iterations: int


def analyze_autocatalytic(scenario: AutocatalyticScenario, verbose: bool = False) -> AutocatalyticResult:
    """Analyze a reaction network for autocatalytic sets."""
    result = check_autocatalytic_set(
        reactions=scenario.reactions,
        food_set=scenario.food_set,
    )
    if verbose:
        print(result)

    return AutocatalyticResult(
        name=scenario.name,
        is_autocatalytic=result.has_raf,
        raf_size=len(result.raf_species),
        molecules_in_raf=list(result.raf_species),
        closure_iterations=result.closure_iterations,
    )


@dataclass
class MoleculeResult:
    """Result from prebiotic plausibility analysis."""
    name: str
    formula: str
    plausibility_score: float
    pathway: Optional[str]
    plausibility_class: str


def analyze_molecule(scenario: MoleculeScenario, verbose: bool = False) -> MoleculeResult:
    """Analyze a molecule for prebiotic plausibility."""
    result = prebiotic_plausibility(scenario.formula)
    if verbose:
        print(result)

    return MoleculeResult(
        name=scenario.name,
        formula=scenario.formula,
        plausibility_score=result.score,
        pathway=result.pathway,
        plausibility_class=result.plausibility_class,
    )


@dataclass
class ReplicatorResult:
    """Result from Eigen threshold analysis."""
    name: str
    genome_length: int
    error_rate: float
    mu_L: float  # error rate * genome length
    ln_a: float  # ln(selective_advantage)
    information_survives: bool
    max_viable_length: int


def analyze_replicator(scenario: ReplicatorScenario, verbose: bool = False) -> ReplicatorResult:
    """Analyze a replicator for information maintenance."""
    result = eigen_error_threshold(
        genome_length=scenario.genome_length,
        error_rate=scenario.error_rate,
        selective_advantage=scenario.selective_advantage,
    )
    if verbose:
        print(result)

    return ReplicatorResult(
        name=scenario.name,
        genome_length=scenario.genome_length,
        error_rate=scenario.error_rate,
        mu_L=result.mu_L,
        ln_a=result.ln_a,
        information_survives=result.survives,
        max_viable_length=result.max_genome_length,
    )


@dataclass
class RNAResult:
    """Result from RNA folding analysis."""
    name: str
    sequence: str
    length: int
    folding_energy: float  # kcal/mol
    stable: bool


def analyze_rna(scenario: RNAScenario, verbose: bool = False) -> RNAResult:
    """Analyze RNA folding energy."""
    result = rna_folding_energy(scenario.sequence)
    if verbose:
        print(result)

    # Consider stable if dG < -1 kcal/mol
    stable = result.dG_total < -1.0

    return RNAResult(
        name=scenario.name,
        sequence=scenario.sequence,
        length=len(scenario.sequence),
        folding_energy=result.dG_total,
        stable=stable,
    )


@dataclass
class MillerUreyResult:
    """Result from Miller-Urey yield analysis."""
    name: str
    atmosphere: str
    energy_kJ: float
    total_yield_mg: float
    glycine_yield_pct: float
    n_amino_acids: int


def analyze_miller_urey(scenario: MillerUreyScenario, verbose: bool = False) -> MillerUreyResult:
    """Analyze Miller-Urey yields."""
    result = miller_urey_yield(
        energy_kJ=scenario.energy_kJ,
        atmosphere=scenario.atmosphere,
    )
    if verbose:
        print(result)

    # Get glycine yield from amino_acid_yields dict if available
    glycine_pct = result.amino_acid_yields.get("glycine", 0.0)

    return MillerUreyResult(
        name=scenario.name,
        atmosphere=scenario.atmosphere,
        energy_kJ=scenario.energy_kJ,
        total_yield_mg=result.estimated_mass_mg,
        glycine_yield_pct=glycine_pct,
        n_amino_acids=len(result.amino_acid_yields),
    )


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def print_report(
    auto_results: List[AutocatalyticResult],
    mol_results: List[MoleculeResult],
    rep_results: List[ReplicatorResult],
    rna_results: List[RNAResult],
    mu_results: List[MillerUreyResult],
):
    """Print comprehensive origin of life analysis report."""
    print("\n" + "=" * 78)
    print("  ORIGIN OF LIFE LAB -- Prebiotic Chemistry Analysis")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 78)

    # Autocatalytic Networks
    print("\n  1. AUTOCATALYTIC NETWORKS (RAF Theory)")
    print(f"  {'Scenario':20s} {'Is RAF':>8s} {'RAF Size':>10s} {'Iterations':>10s}")
    print(f"  {'-'*20} {'-'*8} {'-'*10} {'-'*10}")
    for r in auto_results:
        raf_str = "YES" if r.is_autocatalytic else "NO"
        print(f"  {r.name:20s} {raf_str:>8s} {r.raf_size:>10d} {r.closure_iterations:>10d}")

    # Prebiotic Molecules
    print("\n  2. PREBIOTIC PLAUSIBILITY")
    print(f"  {'Molecule':12s} {'Formula':>12s} {'Score':>8s} {'Class':>14s}")
    print(f"  {'-'*12} {'-'*12} {'-'*8} {'-'*14}")
    for r in mol_results:
        print(f"  {r.name:12s} {r.formula:>12s} {r.plausibility_score:8.2f} "
              f"{r.plausibility_class:>14s}")

    # Eigen Error Threshold
    print("\n  3. EIGEN ERROR THRESHOLD (Information Survival)")
    print(f"  {'Scenario':20s} {'Length':>8s} {'mu*L':>8s} {'ln(a)':>10s} {'Survives':>10s}")
    print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*10} {'-'*10}")
    for r in rep_results:
        surv_str = "YES" if r.information_survives else "NO"
        print(f"  {r.name:20s} {r.genome_length:>8d} {r.mu_L:8.3f} "
              f"{r.ln_a:10.3f} {surv_str:>10s}")

    # RNA Folding
    print("\n  4. RNA FOLDING STABILITY")
    print(f"  {'Name':16s} {'Sequence':>16s} {'dG (kcal/mol)':>14s} {'Stable':>8s}")
    print(f"  {'-'*16} {'-'*16} {'-'*14} {'-'*8}")
    for r in rna_results:
        stable_str = "YES" if r.stable else "NO"
        print(f"  {r.name:16s} {r.sequence:>16s} {r.folding_energy:14.2f} {stable_str:>8s}")

    # Miller-Urey Yields
    print("\n  5. MILLER-UREY YIELDS")
    print(f"  {'Scenario':20s} {'Atmosphere':>12s} {'Total (mg)':>12s} {'Gly %':>8s} {'# AAs':>8s}")
    print(f"  {'-'*20} {'-'*12} {'-'*12} {'-'*8} {'-'*8}")
    for r in mu_results:
        print(f"  {r.name:20s} {r.atmosphere:>12s} {r.total_yield_mg:12.2f} "
              f"{r.glycine_yield_pct:7.1f}% {r.n_amino_acids:>8d}")

    # Summary
    print("\n" + "=" * 78)
    print("  Summary:")
    n_raf = sum(1 for r in auto_results if r.is_autocatalytic)
    n_plausible = sum(1 for r in mol_results if r.plausibility_score > 0.5)
    n_survives = sum(1 for r in rep_results if r.information_survives)
    n_stable = sum(1 for r in rna_results if r.stable)
    print(f"    Autocatalytic networks: {n_raf}/{len(auto_results)} are RAF")
    print(f"    Prebiotic molecules: {n_plausible}/{len(mol_results)} plausible (score > 0.5)")
    print(f"    Replicators: {n_survives}/{len(rep_results)} survive error threshold")
    print(f"    RNA sequences: {n_stable}/{len(rna_results)} thermodynamically stable")
    print("=" * 78 + "\n")


def save_results(
    auto_results: List[AutocatalyticResult],
    mol_results: List[MoleculeResult],
    rep_results: List[ReplicatorResult],
    rna_results: List[RNAResult],
    mu_results: List[MillerUreyResult],
    outpath: Path,
):
    """Save results to JSON."""
    data = {
        "timestamp": datetime.now().isoformat(),
        "pipeline": "origin_of_life_lab v0.1",
        "autocatalytic": [asdict(r) for r in auto_results],
        "molecules": [asdict(r) for r in mol_results],
        "replicators": [asdict(r) for r in rep_results],
        "rna_folding": [asdict(r) for r in rna_results],
        "miller_urey": [asdict(r) for r in mu_results],
        "summary": {
            "n_raf": sum(1 for r in auto_results if r.is_autocatalytic),
            "n_plausible": sum(1 for r in mol_results if r.plausibility_score > 0.5),
            "n_survives": sum(1 for r in rep_results if r.information_survives),
            "n_stable": sum(1 for r in rna_results if r.stable),
        },
    }
    outpath.parent.mkdir(parents=True, exist_ok=True)
    with open(outpath, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Results saved to {outpath}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Origin of Life Lab")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print detailed tool output")
    args = parser.parse_args()

    print(f"\n  Analyzing abiogenesis scenarios...")
    print(f"    {len(AUTOCATALYTIC_SCENARIOS)} autocatalytic networks")
    print(f"    {len(MOLECULE_SCENARIOS)} prebiotic molecules")
    print(f"    {len(REPLICATOR_SCENARIOS)} replicator scenarios")
    print(f"    {len(RNA_SCENARIOS)} RNA sequences")
    print(f"    {len(MILLER_UREY_SCENARIOS)} Miller-Urey scenarios")

    auto_results = []
    for scenario in AUTOCATALYTIC_SCENARIOS:
        try:
            result = analyze_autocatalytic(scenario, verbose=args.verbose)
            auto_results.append(result)
        except Exception as e:
            print(f"  ERROR (autocatalytic): {scenario.name}: {e}")

    mol_results = []
    for scenario in MOLECULE_SCENARIOS:
        try:
            result = analyze_molecule(scenario, verbose=args.verbose)
            mol_results.append(result)
        except Exception as e:
            print(f"  ERROR (molecule): {scenario.name}: {e}")

    rep_results = []
    for scenario in REPLICATOR_SCENARIOS:
        try:
            result = analyze_replicator(scenario, verbose=args.verbose)
            rep_results.append(result)
        except Exception as e:
            print(f"  ERROR (replicator): {scenario.name}: {e}")

    rna_results = []
    for scenario in RNA_SCENARIOS:
        try:
            result = analyze_rna(scenario, verbose=args.verbose)
            rna_results.append(result)
        except Exception as e:
            print(f"  ERROR (rna): {scenario.name}: {e}")

    mu_results = []
    for scenario in MILLER_UREY_SCENARIOS:
        try:
            result = analyze_miller_urey(scenario, verbose=args.verbose)
            mu_results.append(result)
        except Exception as e:
            print(f"  ERROR (miller_urey): {scenario.name}: {e}")

    print_report(auto_results, mol_results, rep_results, rna_results, mu_results)

    outpath = RESULTS_DIR / "abiogenesis_results.json"
    save_results(auto_results, mol_results, rep_results, rna_results, mu_results, outpath)


if __name__ == "__main__":
    main()
