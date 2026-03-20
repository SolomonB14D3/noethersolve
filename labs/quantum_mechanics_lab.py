#!/usr/bin/env python3
"""quantum_mechanics_lab.py -- Autonomous quantum mechanics calculator lab.

Chains NoetherSolve QM tools to calculate energy levels, tunneling probabilities,
uncertainty relations, and angular momentum coupling for various quantum systems.

Usage:
    python labs/quantum_mechanics_lab.py
    python labs/quantum_mechanics_lab.py --verbose
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional

# Ensure noethersolve is importable
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from noethersolve.qm_calculator import (
    particle_in_box,
    hydrogen_energy,
    uncertainty_check,
    tunneling_probability,
    harmonic_oscillator,
    angular_momentum_addition,
)


# ---------------------------------------------------------------------------
# Quantum system candidate definitions
# ---------------------------------------------------------------------------

@dataclass
class QuantumSystemCandidate:
    """A quantum system configuration to analyze."""
    name: str
    system_type: str  # "box", "hydrogen", "tunneling", "oscillator", "spin"
    # Parameters vary by system_type
    params: dict


CANDIDATES: List[QuantumSystemCandidate] = [
    # Particle in a box systems
    QuantumSystemCandidate(
        name="electron_in_1nm_box",
        system_type="box",
        params={"n": 1, "L": 1e-9, "m": 9.109e-31},
    ),
    QuantumSystemCandidate(
        name="electron_in_1nm_box_n3",
        system_type="box",
        params={"n": 3, "L": 1e-9, "m": 9.109e-31},
    ),
    QuantumSystemCandidate(
        name="proton_in_nuclear_well",
        system_type="box",
        params={"n": 1, "L": 1e-14, "m": 1.67e-27},  # Nuclear scale
    ),

    # Hydrogen atom systems
    QuantumSystemCandidate(
        name="hydrogen_ground_state",
        system_type="hydrogen",
        params={"n": 1, "Z": 1},
    ),
    QuantumSystemCandidate(
        name="hydrogen_n2",
        system_type="hydrogen",
        params={"n": 2, "Z": 1},
    ),
    QuantumSystemCandidate(
        name="helium_ion",
        system_type="hydrogen",
        params={"n": 1, "Z": 2},  # He+
    ),

    # Quantum tunneling scenarios
    QuantumSystemCandidate(
        name="alpha_decay_analog",
        system_type="tunneling",
        params={"E": 5.0, "V": 10.0, "L": 1e-14, "m": 6.64e-27},  # Alpha particle
    ),
    QuantumSystemCandidate(
        name="electron_tunneling_barrier",
        system_type="tunneling",
        params={"E": 2.0, "V": 5.0, "L": 1e-9, "m": 9.109e-31},
    ),
    QuantumSystemCandidate(
        name="above_barrier_transmission",
        system_type="tunneling",
        params={"E": 10.0, "V": 5.0, "L": 1e-10, "m": 9.109e-31},  # E > V
    ),

    # Harmonic oscillator systems
    QuantumSystemCandidate(
        name="molecular_vibration",
        system_type="oscillator",
        params={"n": 0, "omega": 1e14, "m": 1.67e-27},  # Proton vibration
    ),
    QuantumSystemCandidate(
        name="molecular_vibration_excited",
        system_type="oscillator",
        params={"n": 5, "omega": 1e14, "m": 1.67e-27},
    ),

    # Angular momentum coupling
    QuantumSystemCandidate(
        name="two_spin_half",
        system_type="spin",
        params={"j1": 0.5, "j2": 0.5},  # Singlet + triplet
    ),
    QuantumSystemCandidate(
        name="spin_orbit_coupling",
        system_type="spin",
        params={"j1": 1.0, "j2": 0.5},  # l=1 + s=1/2
    ),
    QuantumSystemCandidate(
        name="multi_electron_coupling",
        system_type="spin",
        params={"j1": 2.0, "j2": 1.5},  # d-orbital + 3 electrons
    ),
]


# ---------------------------------------------------------------------------
# Screening pipeline
# ---------------------------------------------------------------------------

@dataclass
class QMScreeningResult:
    """Result of screening a quantum system."""
    name: str
    system_type: str
    # Common fields
    energy_eV: Optional[float] = None
    energy_J: Optional[float] = None
    # Box-specific
    wavelength_m: Optional[float] = None
    nodes: Optional[int] = None
    # Hydrogen-specific
    radius_A: Optional[float] = None
    ionization_eV: Optional[float] = None
    degeneracy: Optional[int] = None
    transition_nm: Optional[float] = None
    # Tunneling-specific
    transmission: Optional[float] = None
    reflection: Optional[float] = None
    regime: Optional[str] = None
    # Oscillator-specific
    zero_point_eV: Optional[float] = None
    classical_amplitude: Optional[float] = None
    # Spin-specific
    j_min: Optional[float] = None
    j_max: Optional[float] = None
    allowed_j: Optional[List[float]] = None
    total_states: Optional[int] = None
    # Overall
    physics_check: str = "UNKNOWN"  # PASS, CAUTION, FAIL
    notes: List[str] = field(default_factory=list)


def screen_candidate(system: QuantumSystemCandidate, verbose: bool = False) -> QMScreeningResult:
    """Run the quantum mechanics analysis pipeline on one system."""

    result = QMScreeningResult(
        name=system.name,
        system_type=system.system_type,
        notes=[],
    )

    try:
        if system.system_type == "box":
            report = particle_in_box(**system.params)
            if verbose:
                print(report)
            result.energy_eV = report.E_n_eV
            result.energy_J = report.E_n_J
            result.wavelength_m = report.wavelength
            result.nodes = report.nodes
            result.notes.extend(report.notes)

            # Physics check: energy should be positive and finite
            if 0 < result.energy_eV < 1e12:
                result.physics_check = "PASS"
            else:
                result.physics_check = "CAUTION"
                result.notes.append("Energy outside typical range")

        elif system.system_type == "hydrogen":
            report = hydrogen_energy(**system.params)
            if verbose:
                print(report)
            result.energy_eV = report.E_n_eV
            result.energy_J = report.E_n_J
            result.radius_A = report.radius_A
            result.ionization_eV = report.ionization_eV
            result.degeneracy = report.degeneracy
            result.transition_nm = report.wavelength_nm
            result.notes.extend(report.notes)

            # Physics check: energy should be negative (bound state)
            if result.energy_eV < 0:
                result.physics_check = "PASS"
            else:
                result.physics_check = "FAIL"
                result.notes.append("Positive energy indicates unbound state")

        elif system.system_type == "tunneling":
            report = tunneling_probability(**system.params)
            if verbose:
                print(report)
            result.transmission = report.T
            result.reflection = report.R
            result.regime = report.regime
            result.notes.extend(report.notes)

            # Physics check: T + R should equal 1
            if abs(report.T + report.R - 1.0) < 1e-10:
                result.physics_check = "PASS"
            else:
                result.physics_check = "FAIL"
                result.notes.append("T + R != 1 violates unitarity")

        elif system.system_type == "oscillator":
            report = harmonic_oscillator(**system.params)
            if verbose:
                print(report)
            result.energy_eV = report.E_n_eV
            result.energy_J = report.E_n_J
            result.zero_point_eV = report.zero_point_eV
            result.classical_amplitude = report.classical_amplitude
            result.notes.extend(report.notes)

            # Physics check: energy should be positive
            if result.energy_eV > 0:
                result.physics_check = "PASS"
            else:
                result.physics_check = "FAIL"

        elif system.system_type == "spin":
            report = angular_momentum_addition(**system.params)
            if verbose:
                print(report)
            result.j_min = report.j_min
            result.j_max = report.j_max
            result.allowed_j = report.allowed_j
            result.total_states = report.total_states
            result.notes.extend(report.notes)

            # Physics check: state counting should be consistent
            computed_states = sum(int(2 * j + 1) for j in report.allowed_j)
            if computed_states == report.total_states:
                result.physics_check = "PASS"
            else:
                result.physics_check = "FAIL"
                result.notes.append(f"State counting mismatch: {computed_states} != {report.total_states}")

    except Exception as e:
        result.physics_check = "ERROR"
        result.notes.append(f"Exception: {str(e)}")

    return result


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def print_report(results: List[QMScreeningResult]):
    """Print a human-readable QM screening report."""
    print("\n" + "=" * 72)
    print("  QUANTUM MECHANICS LAB -- Autonomous Calculation Report")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 72)

    # Group by system type
    by_type = {}
    for r in results:
        if r.system_type not in by_type:
            by_type[r.system_type] = []
        by_type[r.system_type].append(r)

    for sys_type, sys_results in by_type.items():
        print(f"\n  --- {sys_type.upper()} SYSTEMS ---")
        for r in sys_results:
            tag = {"PASS": "[PASS]", "CAUTION": "[CAUTION]", "FAIL": "[FAIL]",
                   "ERROR": "[ERROR]", "UNKNOWN": "[?]"}[r.physics_check]
            print(f"\n  {r.name:35s}  {tag}")

            if sys_type == "box":
                print(f"       E = {r.energy_eV:.4g} eV    λ = {r.wavelength_m:.4g} m    nodes = {r.nodes}")
            elif sys_type == "hydrogen":
                print(f"       E = {r.energy_eV:.4g} eV    r = {r.radius_A:.4g} Å    deg = {r.degeneracy}")
                if r.transition_nm:
                    print(f"       Lyman transition: {r.transition_nm:.2f} nm")
            elif sys_type == "tunneling":
                print(f"       T = {r.transmission:.6g}    R = {r.reflection:.6g}    regime = {r.regime}")
            elif sys_type == "oscillator":
                print(f"       E = {r.energy_eV:.4g} eV    ZPE = {r.zero_point_eV:.4g} eV")
                print(f"       Classical amplitude = {r.classical_amplitude:.4g} m")
            elif sys_type == "spin":
                print(f"       J: {r.j_min} to {r.j_max}    states = {r.total_states}")
                print(f"       Allowed J: {r.allowed_j}")

            if r.notes:
                for note in r.notes[:2]:
                    print(f"       Note: {note}")

    # Summary
    n_pass = sum(1 for r in results if r.physics_check == "PASS")
    n_caution = sum(1 for r in results if r.physics_check == "CAUTION")
    n_fail = sum(1 for r in results if r.physics_check in ("FAIL", "ERROR"))
    print(f"\n  {'='*72}")
    print(f"  Summary: {n_pass} PASS / {n_caution} CAUTION / {n_fail} FAIL "
          f"out of {len(results)} systems")
    print(f"  {'='*72}\n")


def save_results(results: List[QMScreeningResult], outpath: Path):
    """Save results to JSON."""
    data = {
        "timestamp": datetime.now().isoformat(),
        "pipeline": "quantum_mechanics_lab v0.1",
        "n_systems": len(results),
        "n_pass": sum(1 for r in results if r.physics_check == "PASS"),
        "n_caution": sum(1 for r in results if r.physics_check == "CAUTION"),
        "n_fail": sum(1 for r in results if r.physics_check in ("FAIL", "ERROR")),
        "results": [asdict(r) for r in results],
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
    parser = argparse.ArgumentParser(description="Quantum Mechanics Lab -- calculation pipeline")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print detailed reports for each system")
    args = parser.parse_args()

    print("\n  Analyzing %d quantum systems..." % len(CANDIDATES))

    results = []
    for system in CANDIDATES:
        try:
            result = screen_candidate(system, verbose=args.verbose)
            results.append(result)
        except Exception as e:
            print(f"  ERROR analyzing {system.name}: {e}")

    if not results:
        print("  No results generated.")
        return

    print_report(results)

    outpath = _ROOT / "results" / "labs" / "quantum_mechanics" / "calculation_results.json"
    save_results(results, outpath)


if __name__ == "__main__":
    main()
