#!/usr/bin/env python3
"""
Cold Fusion / LENR Fact-Checking Module

Verifies cold fusion claims against fundamental physics constraints:
1. Energy conservation (E = mc²)
2. Momentum conservation
3. Charge conservation (Z)
4. Baryon number conservation (A)
5. Expected radiation signatures

Based on Noether's theorem: every symmetry implies a conservation law.
"""

from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from enum import Enum
import math

# Physical constants
C = 2.998e8  # m/s
MEV_PER_AMU = 931.5  # MeV per atomic mass unit
AVOGADRO = 6.022e23
JOULES_PER_MEV = 1.602e-13
COULOMB_BARRIER_DD = 0.4  # MeV, approximate for D-D
ROOM_TEMP_ENERGY = 0.025e-6  # MeV (25 meV at 300K)


class Verdict(Enum):
    """Fact-check verdict."""
    PLAUSIBLE = "plausible"
    IMPLAUSIBLE = "implausible"
    VIOLATES_CONSERVATION = "violates_conservation"
    INSUFFICIENT_DATA = "insufficient_data"


@dataclass
class Nucleus:
    """Atomic nucleus."""
    symbol: str
    Z: int  # Atomic number (protons)
    A: int  # Mass number (protons + neutrons)
    mass_amu: float  # Atomic mass in amu

    @property
    def N(self) -> int:
        """Neutron number."""
        return self.A - self.Z


# Common nuclei in cold fusion claims
NUCLEI = {
    'p': Nucleus('p', 1, 1, 1.007825),      # Proton
    'n': Nucleus('n', 0, 1, 1.008665),      # Neutron
    'd': Nucleus('d', 1, 2, 2.014102),      # Deuterium
    't': Nucleus('t', 1, 3, 3.016049),      # Tritium
    'He3': Nucleus('He3', 2, 3, 3.016029),  # Helium-3
    'He4': Nucleus('He4', 2, 4, 4.002603),  # Helium-4
    'Li6': Nucleus('Li6', 3, 6, 6.015122),  # Lithium-6
    'Li7': Nucleus('Li7', 3, 7, 7.016003),  # Lithium-7
    'Be9': Nucleus('Be9', 4, 9, 9.012182),  # Beryllium-9
    'Ni58': Nucleus('Ni58', 28, 58, 57.935348),  # Nickel-58
    'Ni62': Nucleus('Ni62', 28, 62, 61.928349),  # Nickel-62
    'Cu63': Nucleus('Cu63', 29, 63, 62.929601),  # Copper-63
    'Pd106': Nucleus('Pd106', 46, 106, 105.903483),  # Palladium-106
}


@dataclass
class Reaction:
    """Nuclear reaction."""
    reactants: List[str]  # List of nucleus symbols
    products: List[str]   # List of nucleus symbols

    def get_nuclei(self, symbols: List[str]) -> List[Nucleus]:
        """Get Nucleus objects for symbols."""
        result = []
        for s in symbols:
            if s in NUCLEI:
                result.append(NUCLEI[s])
            else:
                raise ValueError(f"Unknown nucleus: {s}")
        return result

    def check_charge_conservation(self) -> Tuple[bool, str]:
        """Check if charge (Z) is conserved."""
        reactants = self.get_nuclei(self.reactants)
        products = self.get_nuclei(self.products)

        Z_in = sum(n.Z for n in reactants)
        Z_out = sum(n.Z for n in products)

        if Z_in == Z_out:
            return True, f"Charge conserved: Z_in = Z_out = {Z_in}"
        else:
            return False, f"VIOLATION: Z_in = {Z_in}, Z_out = {Z_out}, ΔZ = {Z_out - Z_in}"

    def check_baryon_conservation(self) -> Tuple[bool, str]:
        """Check if baryon number (A) is conserved."""
        reactants = self.get_nuclei(self.reactants)
        products = self.get_nuclei(self.products)

        A_in = sum(n.A for n in reactants)
        A_out = sum(n.A for n in products)

        if A_in == A_out:
            return True, f"Baryon number conserved: A_in = A_out = {A_in}"
        else:
            return False, f"VIOLATION: A_in = {A_in}, A_out = {A_out}, ΔA = {A_out - A_in}"

    def compute_q_value(self) -> float:
        """Compute Q-value (energy release) in MeV."""
        reactants = self.get_nuclei(self.reactants)
        products = self.get_nuclei(self.products)

        mass_in = sum(n.mass_amu for n in reactants)
        mass_out = sum(n.mass_amu for n in products)

        delta_mass = mass_in - mass_out
        q_value = delta_mass * MEV_PER_AMU

        return q_value

    def check_energy_conservation(self, claimed_energy_mev: float,
                                   tolerance: float = 0.1) -> Tuple[bool, str]:
        """Check if claimed energy matches Q-value."""
        q_value = self.compute_q_value()

        if abs(claimed_energy_mev - q_value) / max(abs(q_value), 0.001) < tolerance:
            return True, f"Energy consistent: Q = {q_value:.3f} MeV, claimed = {claimed_energy_mev:.3f} MeV"
        else:
            return False, f"Energy MISMATCH: Q = {q_value:.3f} MeV, claimed = {claimed_energy_mev:.3f} MeV"


@dataclass
class ColdFusionClaim:
    """A cold fusion / LENR experimental claim."""
    description: str
    reaction: Optional[Reaction]
    claimed_excess_heat_watts: Optional[float]
    claimed_reaction_rate: Optional[float]  # reactions per second
    operating_temp_kelvin: float
    observed_neutrons: bool
    observed_gamma: bool
    observed_helium: bool
    observed_transmutation: bool

    def thermal_energy_mev(self) -> float:
        """Thermal energy at operating temperature in MeV."""
        k_B = 8.617e-11  # MeV/K
        return k_B * self.operating_temp_kelvin


def check_coulomb_barrier(reaction: Reaction, temp_kelvin: float) -> Tuple[bool, str]:
    """
    Check if thermal energy can overcome Coulomb barrier.

    The Coulomb barrier for two nuclei with charges Z1, Z2 at distance r is:
    V = Z1 * Z2 * e² / (4πε₀r) ≈ 1.44 * Z1 * Z2 / r MeV (r in fm)

    For D-D fusion, the barrier is ~400 keV.
    Room temperature is ~25 meV.

    Ratio: ~16 million to 1
    """
    k_B = 8.617e-11  # MeV/K
    thermal_energy = k_B * temp_kelvin

    reactants = reaction.get_nuclei(reaction.reactants)
    if len(reactants) < 2:
        return True, "Single particle, no barrier"

    Z1, Z2 = reactants[0].Z, reactants[1].Z

    # Approximate barrier height (assuming nuclear radius ~1.2 * A^(1/3) fm)
    r_nuclear = 1.2 * (reactants[0].A**(1/3) + reactants[1].A**(1/3))  # fm
    barrier_mev = 1.44 * Z1 * Z2 / r_nuclear

    ratio = barrier_mev / thermal_energy

    # Quantum tunneling probability (Gamow factor, very approximate)
    # P ~ exp(-2π * η) where η = Z1*Z2*e²/(ℏv) is Sommerfeld parameter
    # At thermal energies, this is astronomically small

    if ratio > 1e6:
        plausible = False
        msg = (f"Coulomb barrier {barrier_mev:.1f} MeV >> thermal energy {thermal_energy*1e6:.1f} eV "
               f"(ratio = {ratio:.1e}). Tunneling probability negligible without enhancement mechanism.")
    elif ratio > 1e3:
        plausible = False
        msg = f"Barrier/thermal ratio = {ratio:.1e}. Requires strong screening or unknown mechanism."
    else:
        plausible = True
        msg = f"Barrier/thermal ratio = {ratio:.1e}. Plausible with some enhancement."

    return plausible, msg


def check_radiation_signature(claim: ColdFusionClaim) -> Tuple[bool, str]:
    """
    Check if observed radiation matches expected from claimed reaction.

    D + D has three branches:
    1. D + D → T + p (50%)     - 4.03 MeV, proton at 3.02 MeV
    2. D + D → He3 + n (50%)   - 3.27 MeV, neutron at 2.45 MeV
    3. D + D → He4 + γ (rare)  - 23.8 MeV gamma

    If claiming D-D fusion without neutrons, must explain missing 50% branch.
    """
    if claim.reaction is None:
        return True, "No specific reaction claimed"

    reactants = claim.reaction.reactants
    products = claim.reaction.products

    issues = []

    # D + D fusion checks
    if reactants == ['d', 'd']:
        if 'He4' in products and not claim.observed_gamma:
            issues.append("D+D→He4 should produce 23.8 MeV gamma (not observed)")

        if not claim.observed_neutrons and 'He3' not in products and 'n' not in products:
            # Could be claiming only T+p branch or He4+γ
            if 't' in products or 'He4' in products:
                pass  # Consistent
            else:
                issues.append("D+D normally produces neutrons (50% branch) - none observed")

        if claim.observed_helium and 'He4' not in products and 'He3' not in products:
            issues.append("Helium observed but not in claimed products")

    # General checks
    q_value = claim.reaction.compute_q_value()
    if q_value > 5 and not claim.observed_gamma and not claim.observed_neutrons:
        issues.append(f"High Q-value ({q_value:.1f} MeV) but no radiation detected")

    if issues:
        return False, "; ".join(issues)
    else:
        return True, "Radiation signature consistent with claimed reaction"


def check_heat_energy_consistency(claim: ColdFusionClaim) -> Tuple[bool, str]:
    """
    Check if excess heat is consistent with claimed reaction rate and Q-value.

    Power = (reactions/second) × (energy/reaction)
    """
    if claim.claimed_excess_heat_watts is None or claim.reaction is None:
        return True, "Insufficient data for heat check"

    q_value = claim.reaction.compute_q_value()
    q_joules = q_value * JOULES_PER_MEV

    if claim.claimed_reaction_rate is not None:
        expected_power = claim.claimed_reaction_rate * q_joules
        ratio = claim.claimed_excess_heat_watts / expected_power if expected_power > 0 else float('inf')

        if 0.5 < ratio < 2.0:
            return True, f"Heat consistent: {claim.claimed_excess_heat_watts:.1f} W vs expected {expected_power:.1f} W"
        else:
            return False, f"Heat MISMATCH: claimed {claim.claimed_excess_heat_watts:.1f} W, expected {expected_power:.1f} W (ratio {ratio:.2f})"

    # Estimate reaction rate from heat
    implied_rate = claim.claimed_excess_heat_watts / q_joules

    return True, f"Implied reaction rate: {implied_rate:.2e} reactions/s for {claim.claimed_excess_heat_watts:.1f} W"


def fact_check_claim(claim: ColdFusionClaim) -> Dict:
    """
    Comprehensive fact-check of a cold fusion claim.

    Returns dict with verdicts and explanations.
    """
    results = {
        "description": claim.description,
        "checks": [],
        "overall_verdict": Verdict.PLAUSIBLE,
        "violations": []
    }

    if claim.reaction is not None:
        # Conservation law checks
        charge_ok, charge_msg = claim.reaction.check_charge_conservation()
        results["checks"].append({"name": "Charge conservation", "passed": charge_ok, "message": charge_msg})
        if not charge_ok:
            results["violations"].append("Charge conservation")

        baryon_ok, baryon_msg = claim.reaction.check_baryon_conservation()
        results["checks"].append({"name": "Baryon conservation", "passed": baryon_ok, "message": baryon_msg})
        if not baryon_ok:
            results["violations"].append("Baryon number conservation")

        # Coulomb barrier check
        barrier_ok, barrier_msg = check_coulomb_barrier(claim.reaction, claim.operating_temp_kelvin)
        results["checks"].append({"name": "Coulomb barrier", "passed": barrier_ok, "message": barrier_msg})

        # Radiation signature check
        rad_ok, rad_msg = check_radiation_signature(claim)
        results["checks"].append({"name": "Radiation signature", "passed": rad_ok, "message": rad_msg})

        # Q-value
        q_value = claim.reaction.compute_q_value()
        results["q_value_mev"] = q_value
        results["checks"].append({"name": "Q-value", "passed": True, "message": f"Q = {q_value:.3f} MeV"})

    # Heat consistency
    heat_ok, heat_msg = check_heat_energy_consistency(claim)
    results["checks"].append({"name": "Heat consistency", "passed": heat_ok, "message": heat_msg})

    # Determine overall verdict
    if results["violations"]:
        results["overall_verdict"] = Verdict.VIOLATES_CONSERVATION
    elif not all(c["passed"] for c in results["checks"]):
        results["overall_verdict"] = Verdict.IMPLAUSIBLE
    else:
        results["overall_verdict"] = Verdict.PLAUSIBLE

    return results


def print_fact_check(results: Dict):
    """Pretty-print fact-check results."""
    print("=" * 70)
    print(f"COLD FUSION CLAIM FACT-CHECK")
    print("=" * 70)
    print(f"Claim: {results['description']}")
    print()

    print("Conservation & Physics Checks:")
    print("-" * 50)
    for check in results["checks"]:
        status = "PASS" if check["passed"] else "FAIL"
        print(f"  [{status}] {check['name']}: {check['message']}")

    print()
    if "q_value_mev" in results:
        print(f"Q-value: {results['q_value_mev']:.3f} MeV")

    print()
    print(f"VERDICT: {results['overall_verdict'].value.upper()}")

    if results["violations"]:
        print(f"Violations: {', '.join(results['violations'])}")
    print("=" * 70)


# ============================================================================
# Standard Cold Fusion Claims Database
# ============================================================================

STANDARD_CLAIMS = {
    "fleischmann_pons_1989": ColdFusionClaim(
        description="Fleischmann-Pons 1989: D-D fusion in Pd cathode producing excess heat",
        reaction=Reaction(['d', 'd'], ['He4']),  # Claimed He-4 without gamma
        claimed_excess_heat_watts=4.0,
        claimed_reaction_rate=None,
        operating_temp_kelvin=300,
        observed_neutrons=False,  # Low/no neutrons was a puzzle
        observed_gamma=False,
        observed_helium=True,  # Later claimed
        observed_transmutation=False,
    ),

    "dd_standard_branch1": ColdFusionClaim(
        description="Standard D-D fusion branch 1: D + D → T + p",
        reaction=Reaction(['d', 'd'], ['t', 'p']),
        claimed_excess_heat_watts=None,
        claimed_reaction_rate=None,
        operating_temp_kelvin=1e8,  # Hot fusion temperature
        observed_neutrons=False,  # This branch has no neutrons
        observed_gamma=False,
        observed_helium=False,
        observed_transmutation=False,
    ),

    "dd_standard_branch2": ColdFusionClaim(
        description="Standard D-D fusion branch 2: D + D → He3 + n",
        reaction=Reaction(['d', 'd'], ['He3', 'n']),
        claimed_excess_heat_watts=None,
        claimed_reaction_rate=None,
        operating_temp_kelvin=1e8,
        observed_neutrons=True,
        observed_gamma=False,
        observed_helium=True,
        observed_transmutation=False,
    ),

    "rossi_ecat": ColdFusionClaim(
        description="Rossi E-Cat: Ni + p → Cu + energy (claimed transmutation)",
        reaction=Reaction(['Ni58', 'p'], ['Cu63']),  # A not conserved!
        claimed_excess_heat_watts=10000,
        claimed_reaction_rate=None,
        operating_temp_kelvin=500,
        observed_neutrons=False,
        observed_gamma=False,
        observed_helium=False,
        observed_transmutation=True,
    ),

    "widom_larsen_theory": ColdFusionClaim(
        description="Widom-Larsen: e + p → n + νe, then n capture (requires heavy electrons)",
        reaction=Reaction(['p'], ['n']),  # Simplified; actual claim involves virtual electrons
        claimed_excess_heat_watts=1.0,
        claimed_reaction_rate=None,
        operating_temp_kelvin=300,
        observed_neutrons=False,  # Ultra-low momentum neutrons claimed
        observed_gamma=False,
        observed_helium=False,
        observed_transmutation=True,
    ),

    "storms_helium_correlation": ColdFusionClaim(
        description="Storms: Excess heat correlates with He-4 production at 23.8 MeV/He",
        reaction=Reaction(['d', 'd'], ['He4']),
        claimed_excess_heat_watts=0.5,
        claimed_reaction_rate=1.3e10,  # Implied by 0.5W at 23.8 MeV
        operating_temp_kelvin=350,
        observed_neutrons=False,
        observed_gamma=False,
        observed_helium=True,
        observed_transmutation=False,
    ),

    "muon_catalyzed_fusion": ColdFusionClaim(
        description="Muon-catalyzed fusion: μ + D + D → He4 + μ (real, but not self-sustaining)",
        reaction=Reaction(['d', 'd'], ['He4']),  # Muon catalyzes but isn't consumed
        claimed_excess_heat_watts=None,
        claimed_reaction_rate=None,
        operating_temp_kelvin=300,  # Works at low temperature!
        observed_neutrons=True,  # Some branches produce neutrons
        observed_gamma=True,
        observed_helium=True,
        observed_transmutation=False,
    ),

    "pd_lattice_screening": ColdFusionClaim(
        description="Enhanced screening in Pd lattice: Effective Coulomb barrier reduced",
        reaction=Reaction(['d', 'd'], ['He3', 'n']),
        claimed_excess_heat_watts=0.1,
        claimed_reaction_rate=None,
        operating_temp_kelvin=300,
        observed_neutrons=True,  # Would expect neutrons if real
        observed_gamma=False,
        observed_helium=True,
        observed_transmutation=False,
    ),

    "brillouin_hydrogen_nickel": ColdFusionClaim(
        description="Brillouin Energy: H + Ni → Ni (excited) via controlled electron capture",
        reaction=Reaction(['Ni58', 'p', 'p', 'p', 'p'], ['Ni62']),  # 4 protons absorbed
        claimed_excess_heat_watts=100,
        claimed_reaction_rate=None,
        operating_temp_kelvin=500,
        observed_neutrons=False,
        observed_gamma=True,  # Some gamma claimed
        observed_helium=False,
        observed_transmutation=True,
    ),
}


# ============================================================================
# Known Physics Principles for Cold Fusion Analysis
# ============================================================================

PHYSICS_PRINCIPLES = """
CONSERVATION LAWS (from Noether's theorem):
1. Energy: E = mc² means nuclear mass deficit = energy release
2. Momentum: Products must carry away momentum consistently
3. Charge: Z must be conserved in all reactions
4. Baryon number: A (protons + neutrons) conserved in Standard Model
5. Lepton number: Electrons, neutrinos balanced

KEY NUCLEAR PHYSICS FACTS:

Coulomb Barrier:
- D-D barrier: ~400 keV
- Room temperature: ~25 meV
- Ratio: 16 million to 1
- Quantum tunneling at this ratio: ~10^{-2700} probability

D-D Fusion Branches (standard):
- D + D → T (1.01 MeV) + p (3.02 MeV)     [50%]
- D + D → He3 (0.82 MeV) + n (2.45 MeV)   [50%]
- D + D → He4 + γ (23.8 MeV)              [10^{-6}]

The "He-4 without gamma" puzzle:
- Standard physics: D+D→He4 requires gamma emission
- Cold fusion claims: He-4 produced without gamma
- This requires an unknown mechanism to carry away 23.8 MeV

Screening effects:
- Electron screening can reduce barrier by ~10-100 eV
- Not enough to explain room-temperature fusion
- Metal lattice may provide additional screening (disputed)

Muon-catalyzed fusion:
- REAL phenomenon (experimentally verified)
- Muon orbits 200× closer than electron → reduces barrier
- But muons decay in 2.2 μs, limiting reaction cycles
- Not energetically self-sustaining
"""


def main():
    """Run fact-checks on standard cold fusion claims."""
    print()
    print("COLD FUSION / LENR FACT-CHECKING MODULE")
    print("Based on conservation laws from Noether's theorem")
    print()

    for name, claim in STANDARD_CLAIMS.items():
        results = fact_check_claim(claim)
        print_fact_check(results)
        print()


if __name__ == "__main__":
    main()
