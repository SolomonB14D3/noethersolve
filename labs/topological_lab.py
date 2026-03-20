#!/usr/bin/env python3
"""topological_lab.py -- Topological materials classification prototype.

Chains NoetherSolve topological invariant tools to classify candidate
materials/systems: Chern number, Z2 invariant, Berry phase, quantum Hall
conductance, and symmetry class lookup from the periodic table.

Usage:
    python labs/topological_lab.py
    python labs/topological_lab.py --verbose
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional

# Ensure noethersolve is importable
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from noethersolve.topological_invariants import (
    chern_number,
    z2_invariant,
    berry_phase,
    quantum_hall,
    topological_classification,
    bulk_boundary_correspondence,
)

RESULTS_DIR = _ROOT / "results" / "labs" / "topological_materials"


# ---------------------------------------------------------------------------
# Test system definitions
# ---------------------------------------------------------------------------

@dataclass
class TestSystem:
    """A candidate material or model system for topological classification."""
    name: str
    description: str
    # Chern number parameters
    chern_system: Optional[str]       # system kwarg for chern_number()
    chern_band: int                   # band index
    # Z2 parameters
    z2_nu: Optional[int]              # Z2 invariant (0 or 1), None to skip
    z2_dim: int                       # 2 or 3
    # Berry phase
    berry_value: Optional[float]      # phase in radians, None to skip
    berry_symmetry: Optional[str]     # protecting symmetry
    # Quantum Hall
    qh_filling: Optional[float]       # filling factor, None to skip
    qh_integer: bool                  # integer vs fractional QHE
    # Symmetry class lookup
    az_class: str                     # Altland-Zirnbauer class
    az_dim: int                       # spatial dimension


SYSTEMS: List[TestSystem] = [
    TestSystem(
        name="2D Quantum Hall (nu=1)",
        description="Integer quantum Hall state at filling factor 1. "
                    "Prototype for Chern insulators.",
        chern_system="quantum_hall", chern_band=1,
        z2_nu=None, z2_dim=2,
        berry_value=None, berry_symmetry=None,
        qh_filling=1.0, qh_integer=True,
        az_class="A", az_dim=2,
    ),
    TestSystem(
        name="FQHE (nu=1/3, Laughlin)",
        description="Fractional QHE at filling 1/3 — Laughlin state with "
                    "fractional charge e/3 quasiparticles.",
        chern_system="quantum_hall", chern_band=1,
        z2_nu=None, z2_dim=2,
        berry_value=None, berry_symmetry=None,
        qh_filling=1/3, qh_integer=False,
        az_class="A", az_dim=2,
    ),
    TestSystem(
        name="FQHE (nu=2/5, Jain sequence)",
        description="Fractional QHE at filling 2/5 — composite fermion state.",
        chern_system="quantum_hall", chern_band=1,
        z2_nu=None, z2_dim=2,
        berry_value=None, berry_symmetry=None,
        qh_filling=2/5, qh_integer=False,
        az_class="A", az_dim=2,
    ),
    TestSystem(
        name="Graphene (trivial)",
        description="Honeycomb lattice with inversion symmetry. Berry phase "
                    "is pi at each Dirac point but Chern number is zero.",
        chern_system="quantum_hall", chern_band=1,
        z2_nu=0, z2_dim=2,
        berry_value=3.14159265, berry_symmetry="inversion",
        qh_filling=None, qh_integer=True,
        az_class="AII", az_dim=2,
    ),
    TestSystem(
        name="Bi2Se3 (3D TI)",
        description="Bismuth selenide — canonical 3D strong topological "
                    "insulator with a single Dirac cone surface state.",
        chern_system=None, chern_band=1,
        z2_nu=1, z2_dim=3,
        berry_value=3.14159265, berry_symmetry="time_reversal",
        qh_filling=None, qh_integer=True,
        az_class="AII", az_dim=3,
    ),
    TestSystem(
        name="Weyl Semimetal",
        description="Broken time-reversal or inversion yields Weyl nodes "
                    "with Chern number +/-1 on enclosing surfaces. Class A in 3D.",
        chern_system="chern_insulator", chern_band=1,
        z2_nu=None, z2_dim=3,
        berry_value=0.0, berry_symmetry=None,
        qh_filling=None, qh_integer=True,
        az_class="A", az_dim=3,
    ),
    TestSystem(
        name="Kitaev Chain (1D TSC)",
        description="1D p-wave superconductor with Majorana zero modes. "
                    "Class BDI in 1D with Z invariant.",
        chern_system=None, chern_band=1,
        z2_nu=None, z2_dim=2,
        berry_value=3.14159265, berry_symmetry="inversion",
        qh_filling=None, qh_integer=True,
        az_class="BDI", az_dim=1,
    ),
]


# ---------------------------------------------------------------------------
# Classification result
# ---------------------------------------------------------------------------

@dataclass
class ClassificationResult:
    """Full topological classification of one system."""
    name: str
    description: str
    # Chern
    chern_number: Optional[int]
    chern_quantized: Optional[bool]
    # Z2
    z2_nu: Optional[int]
    z2_classification: Optional[str]
    # Berry phase
    berry_phase_rad: Optional[float]
    berry_phase_pi: Optional[float]
    berry_quantized: Optional[bool]
    # Quantum Hall
    qh_conductance: Optional[float]
    qh_resistance: Optional[float]
    qh_plateau: Optional[str]
    # Periodic table
    az_class: str
    az_dim: int
    invariant_type: str
    has_T: bool
    has_C: bool
    has_S: bool
    # Bulk-boundary
    bulk_boundary_satisfied: Optional[bool]
    edge_modes: Optional[int]
    # Overall
    phase: str  # "topological" / "trivial" / "semimetal"


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def classify_system(sys: TestSystem, verbose: bool = False) -> ClassificationResult:
    """Run the full classification pipeline on one test system."""

    # -- Step 1: Symmetry class lookup --
    topo_class = topological_classification(
        symmetry_class=sys.az_class, dimension=sys.az_dim,
    )
    if verbose:
        print(topo_class)

    # -- Step 2: Chern number --
    c_num = None
    c_quant = None
    if sys.chern_system is not None:
        report = chern_number(band_index=sys.chern_band, system=sys.chern_system)
        c_num = report.chern_number
        c_quant = report.is_exactly_quantized
        if verbose:
            print(report)

    # -- Step 3: Z2 invariant --
    z2_class = None
    if sys.z2_nu is not None:
        report = z2_invariant(nu=sys.z2_nu, dimension=sys.z2_dim)
        z2_class = report.classification
        if verbose:
            print(report)

    # -- Step 4: Berry phase --
    b_rad = None
    b_pi = None
    b_quant = None
    if sys.berry_value is not None:
        report = berry_phase(
            phase_value=sys.berry_value, symmetry=sys.berry_symmetry,
        )
        b_rad = report.berry_phase
        b_pi = report.berry_phase_pi
        b_quant = report.is_quantized
        if verbose:
            print(report)

    # -- Step 5: Quantum Hall conductance --
    qh_cond = None
    qh_res = None
    qh_plat = None
    if sys.qh_filling is not None:
        report = quantum_hall(
            filling_factor=sys.qh_filling, is_integer=sys.qh_integer,
        )
        qh_cond = report.hall_conductance
        qh_res = report.hall_resistance
        qh_plat = report.plateau_type
        if verbose:
            print(report)

    # -- Step 6: Bulk-boundary correspondence --
    bb_ok = None
    edge = None
    bulk_inv = c_num if c_num is not None else (sys.z2_nu if sys.z2_nu is not None else None)
    if bulk_inv is not None:
        bb_type = "quantum_hall" if sys.qh_filling else (
            "z2_insulator" if sys.z2_nu is not None else "chern_insulator"
        )
        report = bulk_boundary_correspondence(
            bulk_invariant=bulk_inv, system_type=bb_type,
        )
        bb_ok = report.correspondence_satisfied
        edge = report.edge_modes
        if verbose:
            print(report)

    # -- Determine overall phase --
    inv = topo_class.invariant_type
    if inv == "0":
        phase = "trivial"
    elif sys.z2_nu == 0 and c_num in (None, 0):
        phase = "trivial"
    elif "A" == sys.az_class and sys.az_dim == 3:
        phase = "semimetal"
    else:
        phase = "topological"

    return ClassificationResult(
        name=sys.name,
        description=sys.description,
        chern_number=c_num,
        chern_quantized=c_quant,
        z2_nu=sys.z2_nu,
        z2_classification=z2_class,
        berry_phase_rad=b_rad,
        berry_phase_pi=b_pi,
        berry_quantized=b_quant,
        qh_conductance=qh_cond,
        qh_resistance=qh_res,
        qh_plateau=qh_plat,
        az_class=sys.az_class,
        az_dim=sys.az_dim,
        invariant_type=inv,
        has_T=topo_class.has_time_reversal,
        has_C=topo_class.has_particle_hole,
        has_S=topo_class.has_chiral,
        bulk_boundary_satisfied=bb_ok,
        edge_modes=edge,
        phase=phase,
    )


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_report(results: List[ClassificationResult]):
    """Print a human-readable classification table."""
    print("\n" + "=" * 78)
    print("  TOPOLOGICAL MATERIALS LAB -- Classification Report")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 78)

    # Table header
    hdr = (f"  {'System':28s}  {'AZ':4s}  {'d':1s}  {'Inv':4s}  "
           f"{'C':>3s}  {'Z2':>3s}  {'Berry':>6s}  {'QH':>8s}  {'Phase':12s}")
    print(f"\n{hdr}")
    print(f"  {'─'*28}  {'─'*4}  {'─'*1}  {'─'*4}  "
          f"{'─'*3}  {'─'*3}  {'─'*6}  {'─'*8}  {'─'*12}")

    for r in results:
        c_str = str(r.chern_number) if r.chern_number is not None else " --"
        z_str = str(r.z2_nu) if r.z2_nu is not None else " --"
        b_str = f"{r.berry_phase_pi:.2f}p" if r.berry_phase_pi is not None else "   --"
        q_str = f"{r.qh_conductance:.2f}e/h" if r.qh_conductance is not None else "     --"
        tag = r.phase.upper()
        print(f"  {r.name:28s}  {r.az_class:4s}  {r.az_dim:1d}  {r.invariant_type:4s}  "
              f"{c_str:>3s}  {z_str:>3s}  {b_str:>6s}  {q_str:>8s}  {tag:12s}")

    # Detail blocks
    for r in results:
        print(f"\n  --- {r.name} ---")
        print(f"  {r.description}")
        print(f"  Symmetry class {r.az_class} in {r.az_dim}D -> invariant type {r.invariant_type}")
        print(f"  T={r.has_T}  C={r.has_C}  S={r.has_S}")
        if r.chern_number is not None:
            print(f"  Chern number C = {r.chern_number} (exactly quantized: {r.chern_quantized})")
        if r.z2_nu is not None:
            print(f"  Z2 invariant nu = {r.z2_nu} -> {r.z2_classification}")
        if r.berry_phase_pi is not None:
            q = " (QUANTIZED)" if r.berry_quantized else ""
            print(f"  Berry phase = {r.berry_phase_pi:.4f}*pi{q}")
        if r.qh_conductance is not None:
            print(f"  Hall conductance = {r.qh_conductance:.4f} e^2/h  "
                  f"R_H = {r.qh_resistance:.2f} Ohm  ({r.qh_plateau} QHE)")
        if r.bulk_boundary_satisfied is not None:
            ok = "SATISFIED" if r.bulk_boundary_satisfied else "VIOLATED"
            print(f"  Bulk-boundary correspondence: {ok} ({r.edge_modes} edge modes)")

    n_topo = sum(1 for r in results if r.phase == "topological")
    n_triv = sum(1 for r in results if r.phase == "trivial")
    n_semi = sum(1 for r in results if r.phase == "semimetal")
    print(f"\n{'=' * 78}")
    print(f"  Summary: {n_topo} topological / {n_triv} trivial / "
          f"{n_semi} semimetal out of {len(results)} systems")
    print(f"{'=' * 78}\n")


def save_results(results: List[ClassificationResult], outpath: Path):
    """Save results to JSON."""
    data = {
        "timestamp": datetime.now().isoformat(),
        "pipeline": "topological_lab v0.1",
        "n_systems": len(results),
        "n_topological": sum(1 for r in results if r.phase == "topological"),
        "n_trivial": sum(1 for r in results if r.phase == "trivial"),
        "n_semimetal": sum(1 for r in results if r.phase == "semimetal"),
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
    parser = argparse.ArgumentParser(
        description="Topological Materials Lab -- classification pipeline")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print full tool reports for each system")
    args = parser.parse_args()

    print("\n  Classifying %d candidate systems..." % len(SYSTEMS))

    results: List[ClassificationResult] = []
    for sys_spec in SYSTEMS:
        try:
            result = classify_system(sys_spec, verbose=args.verbose)
            results.append(result)
        except Exception as e:
            print(f"  ERROR classifying {sys_spec.name}: {e}")

    if not results:
        print("  No results generated.")
        return

    print_report(results)

    outpath = RESULTS_DIR / "classification_results.json"
    save_results(results, outpath)


if __name__ == "__main__":
    main()
