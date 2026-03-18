"""
noethersolve.audit_chem — Chemical reaction network thermodynamic auditor.

Checks whether a reaction network is thermodynamically consistent WITHOUT
running a simulation. Pure algebraic checks on the network structure and
rate constants.

Catches:
  - Wegscheider cyclicity violations (rate constants inconsistent around cycles)
  - Missing conservation laws (stoichiometry matrix rank deficiency)
  - Detailed balance violations at a given concentration
  - Negative entropy production (second law violation)
  - Non-physical rate constants (negative or zero)

Usage:
    from noethersolve.audit_chem import audit_network, AuditReport

    report = audit_network(
        species=["A", "B", "C"],
        stoichiometry=[[-1, 1, 0, 0], [1, -1, -1, 1], [0, 0, 1, -1]],
        rate_constants=[0.5, 0.3, 0.4, 0.2],
        reactant_matrix=[[1, 0, 0, 0], [0, 1, 1, 0], [0, 0, 0, 1]],
        reverse_pairs=[(0, 1), (2, 3)],
    )
    print(report)
    # Shows conservation laws, Wegscheider products, warnings
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from scipy.linalg import null_space


@dataclass
class ConservationLaw:
    """A linear conservation law: sum of coefficients * species = const."""
    coefficients: Dict[str, float]
    description: str

    def __str__(self):
        terms = []
        for sp, coef in sorted(self.coefficients.items()):
            if abs(coef) < 1e-10:
                continue
            if abs(coef - 1.0) < 1e-10:
                terms.append(sp)
            elif abs(coef - round(coef)) < 1e-10:
                terms.append(f"{int(round(coef))}*{sp}")
            else:
                terms.append(f"{coef:.3g}*{sp}")
        return " + ".join(terms) + " = const"


@dataclass
class CycleCheck:
    """Result of checking one reaction cycle for Wegscheider consistency."""
    cycle: List[Tuple[int, int]]  # list of (forward, reverse) reaction indices
    product: float                 # product of k_fwd/k_rev around cycle
    log_product: float             # ln of product (should be 0 for thermodynamic consistency)
    consistent: bool               # |log_product| < threshold


@dataclass
class AuditReport:
    """Result of audit_network()."""
    verdict: str                            # PASS, WARN, or FAIL
    n_species: int
    n_reactions: int
    n_reversible_pairs: int
    conservation_laws: List[ConservationLaw]
    rank_deficiency: int                    # dim of null space = number of conservation laws
    rate_warnings: List[str]                # non-physical rate constants
    cycle_checks: List[CycleCheck]
    wegscheider_consistent: bool
    detailed_balance_at_ref: Optional[Dict[str, float]]  # ratio of forward/reverse at ref conc
    entropy_production_at_ref: Optional[float]
    warnings: List[str]

    def __str__(self):
        lines = []
        lines.append(f"{'=' * 60}")
        lines.append(f"  Chemical Network Audit: {self.verdict}")
        lines.append(f"{'=' * 60}")
        lines.append(f"  Species: {self.n_species}, Reactions: {self.n_reactions}, "
                     f"Reversible pairs: {self.n_reversible_pairs}")
        lines.append("")

        # Conservation laws
        lines.append(f"  Conservation laws ({len(self.conservation_laws)}):")
        for law in self.conservation_laws:
            lines.append(f"    {law}")

        # Wegscheider
        if self.cycle_checks:
            lines.append("")
            lines.append("  Wegscheider cyclicity:")
            for cc in self.cycle_checks:
                status = "OK" if cc.consistent else "VIOLATED"
                lines.append(f"    Cycle {cc.cycle}: product={cc.product:.6f}, "
                           f"ln={cc.log_product:.4f} [{status}]")

        # Detailed balance
        if self.detailed_balance_at_ref is not None:
            lines.append("")
            lines.append("  Detailed balance ratios (at reference concentration):")
            for pair, ratio in self.detailed_balance_at_ref.items():
                status = "near equilibrium" if abs(ratio - 1.0) < 0.1 else "away from eq."
                lines.append(f"    {pair}: {ratio:.4f} ({status})")

        # Entropy production
        if self.entropy_production_at_ref is not None:
            status = "OK (non-negative)" if self.entropy_production_at_ref >= 0 else "VIOLATED"
            lines.append("")
            lines.append(f"  Entropy production at ref: {self.entropy_production_at_ref:.6f} [{status}]")

        # Warnings
        if self.warnings:
            lines.append("")
            lines.append("  Warnings:")
            for w in self.warnings:
                lines.append(f"    - {w}")

        lines.append(f"{'=' * 60}")
        return "\n".join(lines)

    @property
    def passed(self) -> bool:
        return self.verdict == "PASS"


def _find_cycles(reverse_pairs: List[Tuple[int, int]], n_reactions: int) -> List[List[Tuple[int, int]]]:
    """Find reaction cycles from reversible pairs.

    For a simple linear chain A<->B<->C, the full cycle is [(0,1), (2,3)].
    For a triangle A<->B<->C<->A, cycle is [(0,1), (2,3), (4,5)].

    Currently returns the full set of reversible pairs as a single cycle.
    More sophisticated cycle detection (e.g., from graph theory) would be
    needed for complex networks with multiple independent cycles.
    """
    if len(reverse_pairs) < 2:
        return []
    # For now, treat all reversible pairs as one cycle
    # This is correct for simple networks; complex networks need proper cycle basis
    return [reverse_pairs]


def audit_network(
    species: List[str],
    stoichiometry,
    rate_constants=None,
    reactant_matrix=None,
    reverse_pairs: Optional[List[Tuple[int, int]]] = None,
    reference_concentration=None,
    wegscheider_threshold: float = 0.01,
) -> AuditReport:
    """Audit a chemical reaction network for thermodynamic consistency.

    Args:
        species: list of species names (e.g., ["A", "B", "C"])
        stoichiometry: (n_species, n_reactions) stoichiometry matrix
        rate_constants: array of rate constants (one per reaction)
        reactant_matrix: (n_species, n_reactions) reactant stoichiometry
            (non-negative integers: how many of each species consumed per reaction)
        reverse_pairs: list of (forward_idx, reverse_idx) for reversible reactions
        reference_concentration: concentration vector for detailed balance check
            (default: uniform 1.0 for all species)
        wegscheider_threshold: |ln(cycle_product)| above this = inconsistent

    Returns:
        AuditReport with conservation laws, Wegscheider checks, and warnings.
    """
    S = np.asarray(stoichiometry, dtype=np.float64)
    n_species, n_reactions = S.shape
    reverse_pairs = reverse_pairs or []
    warnings = []
    rate_warnings = []

    # ── Validate inputs ──────────────────────────────────────────────────
    if len(species) != n_species:
        raise ValueError(f"species list ({len(species)}) doesn't match "
                        f"stoichiometry rows ({n_species})")

    k = None
    if rate_constants is not None:
        k = np.asarray(rate_constants, dtype=np.float64)
        if len(k) != n_reactions:
            raise ValueError(f"rate_constants ({len(k)}) doesn't match "
                           f"stoichiometry columns ({n_reactions})")
        # Check for non-physical rates
        if np.any(k <= 0):
            bad = [i for i in range(len(k)) if k[i] <= 0]
            rate_warnings.append(f"Non-positive rate constants at indices {bad}")
        if np.any(np.isinf(k)):
            rate_warnings.append("Infinite rate constants detected")

    R = None
    if reactant_matrix is not None:
        R = np.asarray(reactant_matrix, dtype=np.float64)
        if R.shape != S.shape:
            raise ValueError(f"reactant_matrix shape {R.shape} doesn't match "
                           f"stoichiometry shape {S.shape}")

    # ── Conservation laws ────────────────────────────────────────────────
    ns = null_space(S.T)
    conservation_laws = []
    rank_deficiency = ns.shape[1] if ns.size > 0 else 0

    if ns.size > 0:
        for i in range(ns.shape[1]):
            w = ns[:, i].copy()
            w[np.abs(w) < 1e-10] = 0
            if np.any(w != 0):
                w = w / np.max(np.abs(w))
                coefficients = {species[j]: float(w[j]) for j in range(n_species) if abs(w[j]) > 1e-10}
                # Generate description
                desc_parts = []
                for sp, coef in coefficients.items():
                    if abs(coef - 1.0) < 1e-10:
                        desc_parts.append(f"[{sp}]")
                    else:
                        desc_parts.append(f"{coef:.3g}[{sp}]")
                desc = " + ".join(desc_parts)
                conservation_laws.append(ConservationLaw(coefficients, desc))

    if rank_deficiency == 0:
        warnings.append("No conservation laws found. All species can be independently produced/consumed.")

    # ── Wegscheider cyclicity ────────────────────────────────────────────
    cycle_checks = []
    wegscheider_consistent = True

    if k is not None and len(reverse_pairs) >= 2:
        cycles = _find_cycles(reverse_pairs, n_reactions)
        for cycle in cycles:
            product = 1.0
            for fwd, rev in cycle:
                if k[rev] > 1e-30:
                    product *= k[fwd] / k[rev]
                else:
                    product = float("inf")

            log_product = np.log(product) if product > 0 and np.isfinite(product) else float("inf")
            consistent = abs(log_product) < wegscheider_threshold
            if not consistent:
                wegscheider_consistent = False

            cycle_checks.append(CycleCheck(
                cycle=cycle,
                product=product,
                log_product=log_product,
                consistent=consistent,
            ))

    # ── Detailed balance at reference concentration ──────────────────────
    detailed_balance_at_ref = None
    entropy_at_ref = None

    if k is not None and R is not None and reverse_pairs:
        if reference_concentration is None:
            c_ref = np.ones(n_species)
        else:
            c_ref = np.asarray(reference_concentration, dtype=np.float64)

        detailed_balance_at_ref = {}
        entropy_at_ref = 0.0

        for idx, (fwd, rev) in enumerate(reverse_pairs):
            # Forward rate
            rate_fwd = k[fwd]
            for i in range(n_species):
                if R[i, fwd] > 0:
                    rate_fwd *= c_ref[i] ** R[i, fwd]
            # Reverse rate
            rate_rev = k[rev]
            for i in range(n_species):
                if R[i, rev] > 0:
                    rate_rev *= c_ref[i] ** R[i, rev]

            if rate_rev > 1e-30:
                ratio = rate_fwd / rate_rev
                detailed_balance_at_ref[f"pair_{idx} (rxn {fwd}<->{rev})"] = ratio
            else:
                detailed_balance_at_ref[f"pair_{idx} (rxn {fwd}<->{rev})"] = float("inf")

            # Entropy production contribution
            if rate_fwd > 1e-30 and rate_rev > 1e-30:
                entropy_at_ref += (rate_fwd - rate_rev) * np.log(rate_fwd / rate_rev)

        if entropy_at_ref < -1e-10:
            warnings.append(f"Negative entropy production ({entropy_at_ref:.6f}) "
                          f"at reference concentration — second law violated")

    # ── Overall verdict ──────────────────────────────────────────────────
    if rate_warnings:
        verdict = "FAIL"
    elif not wegscheider_consistent:
        verdict = "WARN"
        warnings.append("Wegscheider cyclicity violated — rate constants may be "
                       "thermodynamically inconsistent. Check that k_fwd/k_rev "
                       "ratios satisfy the equilibrium constant constraint.")
    elif entropy_at_ref is not None and entropy_at_ref < -1e-10:
        verdict = "FAIL"
    else:
        verdict = "PASS"

    return AuditReport(
        verdict=verdict,
        n_species=n_species,
        n_reactions=n_reactions,
        n_reversible_pairs=len(reverse_pairs),
        conservation_laws=conservation_laws,
        rank_deficiency=rank_deficiency,
        rate_warnings=rate_warnings,
        cycle_checks=cycle_checks,
        wegscheider_consistent=wegscheider_consistent,
        detailed_balance_at_ref=detailed_balance_at_ref,
        entropy_production_at_ref=entropy_at_ref,
        warnings=warnings,
    )
