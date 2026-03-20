#!/usr/bin/env python3
"""drug_therapy_lab.py -- Autonomous drug candidate screening prototype.

Chains NoetherSolve PK, enzyme kinetics, and drug interaction tools to
screen a panel of drugs, score them on therapeutic index,
interaction risk, and metabolic stability, and output a ranked report.

Now pulls real PK data from ChEMBL + FDA labels instead of hardcoded values.

Usage:
    python labs/drug_therapy_lab.py                    # Screen default 100 drugs
    python labs/drug_therapy_lab.py --drugs metformin warfarin  # Specific drugs
    python labs/drug_therapy_lab.py --top 50           # Top 50 most prescribed
    python labs/drug_therapy_lab.py --verbose

Data sources:
    - ChEMBL (https://www.ebi.ac.uk/chembl/) -- ADMET data, pharmacokinetics
    - PubChem (https://pubchem.ncbi.nlm.nih.gov/) -- molecular properties
    - FDA package inserts (curated subset) -- therapeutic windows

⚠️  DISCLAIMER: FOR EDUCATIONAL AND RESEARCH USE ONLY
    This tool does NOT provide medical advice. Drug dosing, interactions,
    and therapeutic decisions must be made by licensed healthcare providers
    using validated clinical resources (e.g., Lexicomp, Micromedex, UpToDate).
    Never use this tool to make treatment decisions for patients.
"""

from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Ensure noethersolve is importable
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from noethersolve.pk_model import (
    one_compartment_oral,
    half_life,
    steady_state,
)
from noethersolve.enzyme_kinetics import michaelis_menten
from noethersolve.drug_interactions import (
    check_interaction,
    get_drug_profile,
    predict_auc_change,
    check_pharmacogenomics,
)
from noethersolve.drug_data_source import (
    DrugDataSource,
    DrugPKProfile,
    merge_with_fda_data,
)


# ---------------------------------------------------------------------------
# Drug candidate definitions
# ---------------------------------------------------------------------------

@dataclass
class DrugCandidate:
    """A drug candidate with its known PK and metabolic properties."""
    name: str
    dose_mg: float           # typical single oral dose
    bioavailability: float   # F (0-1)
    Vd_L: float              # volume of distribution (L)
    ka_per_h: float          # absorption rate constant (h^-1)
    ke_per_h: float          # elimination rate constant (h^-1)
    dosing_interval_h: float # tau (h)
    therapeutic_min: float   # mg/L -- minimum effective concentration
    therapeutic_max: float   # mg/L -- toxic threshold
    primary_cyp: str         # main CYP enzyme
    # Enzyme kinetics of primary metabolic pathway
    Vmax_metabolism: float   # umol/min (hepatic Vmax for the drug)
    Km_metabolism: float     # umol/L (Km of the CYP for this drug)


# ---------------------------------------------------------------------------
# Drug candidate loading from database
# ---------------------------------------------------------------------------

def profile_to_candidate(profile: DrugPKProfile) -> Optional[DrugCandidate]:
    """Convert a DrugPKProfile from database to DrugCandidate for screening."""
    # Merge with FDA data for best accuracy
    profile = merge_with_fda_data(profile)

    # Check we have minimum required fields
    if not profile.half_life_h or not profile.Vd_L:
        return None

    # Calculate ke from half-life
    ke = 0.693 / profile.half_life_h

    # Estimate ka from Tmax if available, else use typical oral absorption
    if profile.Tmax_h and profile.Tmax_h > 0:
        # Rough estimate: ka ~ 2-4x ke for most oral drugs
        ka = max(2.0 * ke, 0.693 / profile.Tmax_h)
    else:
        ka = 1.5  # default moderate absorption

    # Estimate Vmax/Km for metabolism (simplified)
    # These are approximations when not available
    Vmax = 200.0  # default
    Km = 100.0    # default

    return DrugCandidate(
        name=profile.name,
        dose_mg=profile.typical_dose_mg or 100,
        bioavailability=profile.bioavailability or 0.5,
        Vd_L=profile.Vd_L,
        ka_per_h=ka,
        ke_per_h=ke,
        dosing_interval_h=profile.dosing_interval_h or 24,
        therapeutic_min=profile.therapeutic_min or 0.01,
        therapeutic_max=profile.therapeutic_max or 10.0,
        primary_cyp=profile.primary_cyp or "unknown",
        Vmax_metabolism=Vmax,
        Km_metabolism=Km,
    )


def load_candidates_from_database(drug_names: List[str], verbose: bool = False) -> List[DrugCandidate]:
    """Load drug candidates from ChEMBL/FDA database."""
    ds = DrugDataSource()
    candidates = []

    for i, name in enumerate(drug_names):
        if verbose:
            print(f"  [{i+1:3d}/{len(drug_names)}] Fetching {name}...")

        try:
            profile = ds.get_drug(name)
            if profile:
                candidate = profile_to_candidate(profile)
                if candidate:
                    candidates.append(candidate)
                elif verbose:
                    print(f"         ⚠ Incomplete PK data for {name}")
            elif verbose:
                print(f"         ⚠ Not found: {name}")
        except Exception as e:
            if verbose:
                print(f"         ⚠ Error: {e}")

    return candidates


# ---------------------------------------------------------------------------
# Fallback hardcoded candidates (for offline use / testing)
# ---------------------------------------------------------------------------

FALLBACK_CANDIDATES: List[DrugCandidate] = [
    # =========================================================================
    # ANALGESICS / NSAIDs (7)
    # =========================================================================
    DrugCandidate(
        name="acetaminophen",
        dose_mg=1000, bioavailability=0.85, Vd_L=67, ka_per_h=2.0,
        ke_per_h=0.28, dosing_interval_h=6,
        therapeutic_min=10.0, therapeutic_max=150.0,
        primary_cyp="CYP2E1",
        Vmax_metabolism=500.0, Km_metabolism=4000.0,
    ),
    DrugCandidate(
        name="ibuprofen",
        dose_mg=400, bioavailability=0.80, Vd_L=10, ka_per_h=3.0,
        ke_per_h=0.35, dosing_interval_h=6,
        therapeutic_min=10.0, therapeutic_max=50.0,
        primary_cyp="CYP2C9",
        Vmax_metabolism=200.0, Km_metabolism=100.0,
    ),
    DrugCandidate(
        name="naproxen",
        dose_mg=500, bioavailability=0.95, Vd_L=10, ka_per_h=1.5,
        ke_per_h=0.05, dosing_interval_h=12,
        therapeutic_min=25.0, therapeutic_max=100.0,
        primary_cyp="CYP2C9",
        Vmax_metabolism=150.0, Km_metabolism=80.0,
    ),
    DrugCandidate(
        name="celecoxib",
        dose_mg=200, bioavailability=0.40, Vd_L=400, ka_per_h=1.2,
        ke_per_h=0.06, dosing_interval_h=12,
        therapeutic_min=0.1, therapeutic_max=1.0,
        primary_cyp="CYP2C9",
        Vmax_metabolism=100.0, Km_metabolism=15.0,
    ),
    DrugCandidate(
        name="diclofenac",
        dose_mg=50, bioavailability=0.55, Vd_L=13, ka_per_h=2.5,
        ke_per_h=0.58, dosing_interval_h=8,
        therapeutic_min=0.5, therapeutic_max=3.0,
        primary_cyp="CYP2C9",
        Vmax_metabolism=180.0, Km_metabolism=50.0,
    ),
    DrugCandidate(
        name="meloxicam",
        dose_mg=15, bioavailability=0.89, Vd_L=10, ka_per_h=0.5,
        ke_per_h=0.03, dosing_interval_h=24,
        therapeutic_min=0.3, therapeutic_max=2.0,
        primary_cyp="CYP2C9",
        Vmax_metabolism=50.0, Km_metabolism=20.0,
    ),
    DrugCandidate(
        name="tramadol",
        dose_mg=50, bioavailability=0.75, Vd_L=200, ka_per_h=1.5,
        ke_per_h=0.12, dosing_interval_h=6,
        therapeutic_min=0.1, therapeutic_max=0.8,
        primary_cyp="CYP2D6",
        Vmax_metabolism=200.0, Km_metabolism=100.0,
    ),

    # =========================================================================
    # CARDIOVASCULAR - Statins (5)
    # =========================================================================
    DrugCandidate(
        name="simvastatin",
        dose_mg=40, bioavailability=0.05, Vd_L=250, ka_per_h=2.5,
        ke_per_h=0.35, dosing_interval_h=24,
        therapeutic_min=0.001, therapeutic_max=0.015,
        primary_cyp="CYP3A4",
        Vmax_metabolism=400.0, Km_metabolism=2.0,
    ),
    DrugCandidate(
        name="atorvastatin",
        dose_mg=20, bioavailability=0.14, Vd_L=381, ka_per_h=1.5,
        ke_per_h=0.05, dosing_interval_h=24,
        therapeutic_min=0.002, therapeutic_max=0.03,
        primary_cyp="CYP3A4",
        Vmax_metabolism=350.0, Km_metabolism=5.0,
    ),
    DrugCandidate(
        name="rosuvastatin",
        dose_mg=10, bioavailability=0.20, Vd_L=134, ka_per_h=2.0,
        ke_per_h=0.04, dosing_interval_h=24,
        therapeutic_min=0.001, therapeutic_max=0.02,
        primary_cyp="CYP2C9",  # minimal CYP metabolism
        Vmax_metabolism=50.0, Km_metabolism=100.0,
    ),
    DrugCandidate(
        name="pravastatin",
        dose_mg=40, bioavailability=0.17, Vd_L=34, ka_per_h=1.8,
        ke_per_h=0.38, dosing_interval_h=24,
        therapeutic_min=0.01, therapeutic_max=0.1,
        primary_cyp="none",  # minimal CYP
        Vmax_metabolism=30.0, Km_metabolism=200.0,
    ),
    DrugCandidate(
        name="lovastatin",
        dose_mg=40, bioavailability=0.05, Vd_L=400, ka_per_h=2.0,
        ke_per_h=0.25, dosing_interval_h=24,
        therapeutic_min=0.002, therapeutic_max=0.02,
        primary_cyp="CYP3A4",
        Vmax_metabolism=300.0, Km_metabolism=3.0,
    ),

    # =========================================================================
    # CARDIOVASCULAR - Beta Blockers (4)
    # =========================================================================
    DrugCandidate(
        name="metoprolol",
        dose_mg=50, bioavailability=0.50, Vd_L=290, ka_per_h=2.0,
        ke_per_h=0.19, dosing_interval_h=12,
        therapeutic_min=0.02, therapeutic_max=0.3,
        primary_cyp="CYP2D6",
        Vmax_metabolism=150.0, Km_metabolism=50.0,
    ),
    DrugCandidate(
        name="atenolol",
        dose_mg=50, bioavailability=0.50, Vd_L=63, ka_per_h=1.0,
        ke_per_h=0.11, dosing_interval_h=24,
        therapeutic_min=0.2, therapeutic_max=1.0,
        primary_cyp="none",  # renally eliminated
        Vmax_metabolism=20.0, Km_metabolism=500.0,
    ),
    DrugCandidate(
        name="propranolol",
        dose_mg=40, bioavailability=0.26, Vd_L=270, ka_per_h=2.5,
        ke_per_h=0.23, dosing_interval_h=8,
        therapeutic_min=0.02, therapeutic_max=0.15,
        primary_cyp="CYP2D6",
        Vmax_metabolism=200.0, Km_metabolism=40.0,
    ),
    DrugCandidate(
        name="carvedilol",
        dose_mg=25, bioavailability=0.25, Vd_L=115, ka_per_h=1.5,
        ke_per_h=0.10, dosing_interval_h=12,
        therapeutic_min=0.01, therapeutic_max=0.1,
        primary_cyp="CYP2D6",
        Vmax_metabolism=180.0, Km_metabolism=30.0,
    ),

    # =========================================================================
    # CARDIOVASCULAR - ACE Inhibitors / ARBs (4)
    # =========================================================================
    DrugCandidate(
        name="lisinopril",
        dose_mg=10, bioavailability=0.25, Vd_L=124, ka_per_h=1.0,
        ke_per_h=0.06, dosing_interval_h=24,
        therapeutic_min=0.01, therapeutic_max=0.1,
        primary_cyp="none",  # renally eliminated
        Vmax_metabolism=10.0, Km_metabolism=1000.0,
    ),
    DrugCandidate(
        name="enalapril",
        dose_mg=10, bioavailability=0.60, Vd_L=50, ka_per_h=1.5,
        ke_per_h=0.06, dosing_interval_h=24,
        therapeutic_min=0.01, therapeutic_max=0.15,
        primary_cyp="none",  # esterase hydrolysis
        Vmax_metabolism=20.0, Km_metabolism=500.0,
    ),
    DrugCandidate(
        name="losartan",
        dose_mg=50, bioavailability=0.33, Vd_L=34, ka_per_h=2.0,
        ke_per_h=0.35, dosing_interval_h=24,
        therapeutic_min=0.05, therapeutic_max=0.5,
        primary_cyp="CYP2C9",
        Vmax_metabolism=100.0, Km_metabolism=25.0,
    ),
    DrugCandidate(
        name="valsartan",
        dose_mg=80, bioavailability=0.25, Vd_L=17, ka_per_h=1.5,
        ke_per_h=0.12, dosing_interval_h=24,
        therapeutic_min=0.5, therapeutic_max=5.0,
        primary_cyp="CYP2C9",
        Vmax_metabolism=80.0, Km_metabolism=50.0,
    ),

    # =========================================================================
    # ANTICOAGULANTS (3)
    # =========================================================================
    DrugCandidate(
        name="warfarin",
        dose_mg=5, bioavailability=0.99, Vd_L=10, ka_per_h=1.0,
        ke_per_h=0.018, dosing_interval_h=24,
        therapeutic_min=1.0, therapeutic_max=4.0,
        primary_cyp="CYP2C9",
        Vmax_metabolism=10.0, Km_metabolism=4.0,
    ),
    DrugCandidate(
        name="rivaroxaban",
        dose_mg=20, bioavailability=0.80, Vd_L=50, ka_per_h=1.5,
        ke_per_h=0.07, dosing_interval_h=24,
        therapeutic_min=0.02, therapeutic_max=0.4,
        primary_cyp="CYP3A4",
        Vmax_metabolism=150.0, Km_metabolism=20.0,
    ),
    DrugCandidate(
        name="apixaban",
        dose_mg=5, bioavailability=0.50, Vd_L=21, ka_per_h=2.0,
        ke_per_h=0.06, dosing_interval_h=12,
        therapeutic_min=0.05, therapeutic_max=0.4,
        primary_cyp="CYP3A4",
        Vmax_metabolism=100.0, Km_metabolism=15.0,
    ),

    # =========================================================================
    # DIABETES (4)
    # =========================================================================
    DrugCandidate(
        name="metformin",
        dose_mg=500, bioavailability=0.55, Vd_L=654, ka_per_h=1.5,
        ke_per_h=0.12, dosing_interval_h=12,
        therapeutic_min=0.5, therapeutic_max=5.0,
        primary_cyp="none",  # renally eliminated
        Vmax_metabolism=50.0, Km_metabolism=5000.0,
    ),
    DrugCandidate(
        name="glipizide",
        dose_mg=10, bioavailability=0.95, Vd_L=10, ka_per_h=2.5,
        ke_per_h=0.20, dosing_interval_h=12,
        therapeutic_min=0.1, therapeutic_max=1.0,
        primary_cyp="CYP2C9",
        Vmax_metabolism=80.0, Km_metabolism=20.0,
    ),
    DrugCandidate(
        name="pioglitazone",
        dose_mg=30, bioavailability=0.83, Vd_L=63, ka_per_h=1.0,
        ke_per_h=0.15, dosing_interval_h=24,
        therapeutic_min=0.3, therapeutic_max=2.0,
        primary_cyp="CYP2C8",
        Vmax_metabolism=120.0, Km_metabolism=30.0,
    ),
    DrugCandidate(
        name="sitagliptin",
        dose_mg=100, bioavailability=0.87, Vd_L=198, ka_per_h=1.5,
        ke_per_h=0.06, dosing_interval_h=24,
        therapeutic_min=0.01, therapeutic_max=0.2,
        primary_cyp="CYP3A4",
        Vmax_metabolism=60.0, Km_metabolism=100.0,
    ),

    # =========================================================================
    # GI - PPIs (3)
    # =========================================================================
    DrugCandidate(
        name="omeprazole",
        dose_mg=20, bioavailability=0.50, Vd_L=15, ka_per_h=3.5,
        ke_per_h=0.7, dosing_interval_h=24,
        therapeutic_min=0.02, therapeutic_max=2.0,
        primary_cyp="CYP2C19",
        Vmax_metabolism=300.0, Km_metabolism=20.0,
    ),
    DrugCandidate(
        name="pantoprazole",
        dose_mg=40, bioavailability=0.77, Vd_L=11, ka_per_h=3.0,
        ke_per_h=0.58, dosing_interval_h=24,
        therapeutic_min=0.02, therapeutic_max=3.0,
        primary_cyp="CYP2C19",
        Vmax_metabolism=250.0, Km_metabolism=25.0,
    ),
    DrugCandidate(
        name="esomeprazole",
        dose_mg=40, bioavailability=0.64, Vd_L=16, ka_per_h=3.0,
        ke_per_h=0.5, dosing_interval_h=24,
        therapeutic_min=0.03, therapeutic_max=3.0,
        primary_cyp="CYP2C19",
        Vmax_metabolism=280.0, Km_metabolism=22.0,
    ),

    # =========================================================================
    # ANTIDEPRESSANTS / ANXIOLYTICS (5)
    # =========================================================================
    DrugCandidate(
        name="sertraline",
        dose_mg=50, bioavailability=0.44, Vd_L=1400, ka_per_h=1.0,
        ke_per_h=0.03, dosing_interval_h=24,
        therapeutic_min=0.01, therapeutic_max=0.2,
        primary_cyp="CYP2D6",
        Vmax_metabolism=200.0, Km_metabolism=40.0,
    ),
    DrugCandidate(
        name="fluoxetine",
        dose_mg=20, bioavailability=0.72, Vd_L=2500, ka_per_h=0.8,
        ke_per_h=0.01, dosing_interval_h=24,  # very long half-life
        therapeutic_min=0.05, therapeutic_max=0.5,
        primary_cyp="CYP2D6",
        Vmax_metabolism=150.0, Km_metabolism=30.0,
    ),
    DrugCandidate(
        name="escitalopram",
        dose_mg=10, bioavailability=0.80, Vd_L=1100, ka_per_h=1.2,
        ke_per_h=0.02, dosing_interval_h=24,
        therapeutic_min=0.01, therapeutic_max=0.13,
        primary_cyp="CYP2C19",
        Vmax_metabolism=100.0, Km_metabolism=50.0,
    ),
    DrugCandidate(
        name="alprazolam",
        dose_mg=0.5, bioavailability=0.90, Vd_L=70, ka_per_h=1.5,
        ke_per_h=0.06, dosing_interval_h=8,
        therapeutic_min=0.01, therapeutic_max=0.06,
        primary_cyp="CYP3A4",
        Vmax_metabolism=80.0, Km_metabolism=10.0,
    ),
    DrugCandidate(
        name="bupropion",
        dose_mg=150, bioavailability=0.87, Vd_L=2000, ka_per_h=1.0,
        ke_per_h=0.03, dosing_interval_h=12,
        therapeutic_min=0.025, therapeutic_max=0.1,
        primary_cyp="CYP2B6",
        Vmax_metabolism=250.0, Km_metabolism=60.0,
    ),

    # =========================================================================
    # ANTIBIOTICS (4)
    # =========================================================================
    DrugCandidate(
        name="amoxicillin",
        dose_mg=500, bioavailability=0.90, Vd_L=20, ka_per_h=2.0,
        ke_per_h=0.58, dosing_interval_h=8,
        therapeutic_min=2.0, therapeutic_max=20.0,
        primary_cyp="none",  # renally eliminated
        Vmax_metabolism=20.0, Km_metabolism=1000.0,
    ),
    DrugCandidate(
        name="azithromycin",
        dose_mg=500, bioavailability=0.37, Vd_L=3150, ka_per_h=0.5,
        ke_per_h=0.01, dosing_interval_h=24,
        therapeutic_min=0.01, therapeutic_max=0.5,
        primary_cyp="CYP3A4",
        Vmax_metabolism=100.0, Km_metabolism=200.0,
    ),
    DrugCandidate(
        name="ciprofloxacin",
        dose_mg=500, bioavailability=0.70, Vd_L=170, ka_per_h=1.5,
        ke_per_h=0.17, dosing_interval_h=12,
        therapeutic_min=0.5, therapeutic_max=5.0,
        primary_cyp="CYP1A2",
        Vmax_metabolism=150.0, Km_metabolism=100.0,
    ),
    DrugCandidate(
        name="metronidazole",
        dose_mg=500, bioavailability=0.99, Vd_L=52, ka_per_h=2.0,
        ke_per_h=0.09, dosing_interval_h=8,
        therapeutic_min=5.0, therapeutic_max=25.0,
        primary_cyp="CYP3A4",
        Vmax_metabolism=120.0, Km_metabolism=80.0,
    ),

    # =========================================================================
    # THYROID / MISC (3)
    # =========================================================================
    DrugCandidate(
        name="levothyroxine",
        dose_mg=0.1, bioavailability=0.80, Vd_L=10, ka_per_h=0.5,
        ke_per_h=0.004, dosing_interval_h=24,  # very long t1/2
        therapeutic_min=0.005, therapeutic_max=0.02,
        primary_cyp="none",  # deiodination
        Vmax_metabolism=5.0, Km_metabolism=100.0,
    ),
    DrugCandidate(
        name="gabapentin",
        dose_mg=300, bioavailability=0.60, Vd_L=58, ka_per_h=1.5,
        ke_per_h=0.12, dosing_interval_h=8,
        therapeutic_min=2.0, therapeutic_max=20.0,
        primary_cyp="none",  # renally eliminated
        Vmax_metabolism=10.0, Km_metabolism=1000.0,
    ),
    DrugCandidate(
        name="pregabalin",
        dose_mg=150, bioavailability=0.90, Vd_L=45, ka_per_h=1.8,
        ke_per_h=0.12, dosing_interval_h=12,
        therapeutic_min=1.0, therapeutic_max=10.0,
        primary_cyp="none",  # renally eliminated
        Vmax_metabolism=10.0, Km_metabolism=1000.0,
    ),
]

# Default to fallback for backward compatibility
CANDIDATES = FALLBACK_CANDIDATES


# ---------------------------------------------------------------------------
# Screening pipeline
# ---------------------------------------------------------------------------

@dataclass
class ScreeningResult:
    """Result of screening a single drug candidate."""
    name: str
    # PK profile
    Cmax: float
    Tmax_h: float
    half_life_h: float
    clearance_L_h: float
    # Steady state
    Css_avg: float
    Css_peak: float
    Css_trough: float
    time_to_ss_h: float
    # Therapeutic window
    therapeutic_index: float   # therapeutic_max / Css_avg
    within_window: bool        # Css_trough >= min AND Css_peak <= max
    # Metabolic stability
    mm_velocity: float         # V0 at typical hepatic [S]
    mm_fraction_Vmax: float    # how saturated the enzyme is
    metabolic_stability: str   # "stable", "moderate", "saturating"
    # Interaction risk
    n_interactions: int
    worst_interaction_severity: str
    interaction_details: List[str]
    # Overall
    score: float               # composite 0-100
    verdict: str               # PASS / CAUTION / FAIL


def screen_candidate(drug: DrugCandidate, all_candidates: List[DrugCandidate] = None,
                     verbose: bool = False) -> ScreeningResult:
    """Run the full screening pipeline on one drug candidate."""

    # -- Step 1: PK profile via oral dosing model --
    pk = one_compartment_oral(
        dose=drug.dose_mg, F=drug.bioavailability, Vd=drug.Vd_L,
        ka=drug.ka_per_h, ke=drug.ke_per_h, t=drug.dosing_interval_h / 2,
    )
    if verbose:
        print(pk)

    # -- Step 2: Half-life --
    hl = half_life(ke=drug.ke_per_h, Vd=drug.Vd_L)
    if verbose:
        print(hl)

    # -- Step 3: Steady state --
    CL = drug.ke_per_h * drug.Vd_L
    ss = steady_state(
        dose=drug.dose_mg, F=drug.bioavailability,
        CL=CL, tau=drug.dosing_interval_h, Vd=drug.Vd_L,
    )
    if verbose:
        print(ss)

    # -- Step 4: Therapeutic window check --
    within = (ss.Css_trough >= drug.therapeutic_min and
              ss.Css_peak <= drug.therapeutic_max)
    ti = drug.therapeutic_max / ss.Css_avg if ss.Css_avg > 0 else float("inf")

    # -- Step 5: Enzyme kinetics of metabolism --
    # Estimate hepatic concentration ~ Css_avg (simplification)
    hepatic_S = ss.Css_avg  # mg/L as proxy for umol/L (rough)
    mm = michaelis_menten(
        Vmax=drug.Vmax_metabolism, Km=drug.Km_metabolism, S=hepatic_S,
    )
    if verbose:
        print(mm)

    if mm.fraction_Vmax < 0.3:
        met_stab = "stable"
    elif mm.fraction_Vmax < 0.7:
        met_stab = "moderate"
    else:
        met_stab = "saturating"

    # -- Step 6: Drug interactions (pairwise with other candidates) --
    interaction_details: List[str] = []
    worst_severity = "none"
    severity_rank = {"none": 0, "minor": 1, "moderate": 2, "major": 3,
                     "contraindicated": 4}

    # Use provided candidates list or fallback
    check_against = all_candidates if all_candidates else FALLBACK_CANDIDATES

    for other in check_against:
        if other.name == drug.name:
            continue
        try:
            report = check_interaction(drug.name, other.name)
            report_str = str(report)
            # Extract severity from the report string
            for sev in ["contraindicated", "major", "moderate", "minor"]:
                if sev.upper() in report_str.upper():
                    if severity_rank.get(sev, 0) > severity_rank.get(worst_severity, 0):
                        worst_severity = sev
                    if sev in ("contraindicated", "major", "moderate"):
                        interaction_details.append(
                            f"{drug.name} + {other.name}: {sev}"
                        )
                    break
        except Exception:
            # Drug not in interaction database -- no known interaction
            pass

    n_interactions = len(interaction_details)

    # -- Step 7: Composite score (0-100) --
    # Therapeutic index score (0-40): higher TI = better
    ti_score = min(40, ti * 4) if ti < float("inf") else 40

    # Window compliance (0-20)
    window_score = 20.0 if within else 0.0

    # Metabolic stability (0-20): less saturated = better
    met_score = 20.0 * (1.0 - mm.fraction_Vmax)

    # Interaction safety (0-20): fewer/less severe = better
    interaction_penalty = severity_rank.get(worst_severity, 0) * 5
    int_score = max(0, 20.0 - interaction_penalty - n_interactions * 2)

    total = ti_score + window_score + met_score + int_score

    if total >= 70 and within and worst_severity not in ("contraindicated", "major"):
        verdict = "PASS"
    elif total >= 40:
        verdict = "CAUTION"
    else:
        verdict = "FAIL"

    return ScreeningResult(
        name=drug.name,
        Cmax=pk.Cmax, Tmax_h=pk.Tmax, half_life_h=hl.half_life,
        clearance_L_h=CL,
        Css_avg=ss.Css_avg, Css_peak=ss.Css_peak, Css_trough=ss.Css_trough,
        time_to_ss_h=ss.time_to_ss,
        therapeutic_index=ti, within_window=within,
        mm_velocity=mm.V0, mm_fraction_Vmax=mm.fraction_Vmax,
        metabolic_stability=met_stab,
        n_interactions=n_interactions,
        worst_interaction_severity=worst_severity,
        interaction_details=interaction_details,
        score=total, verdict=verdict,
    )


# ---------------------------------------------------------------------------
# Enhanced DDI Analysis (uses predict_auc_change for quantitative predictions)
# ---------------------------------------------------------------------------

@dataclass
class DDIAnalysisResult:
    """Result from DDI AUC prediction analysis."""
    drug_pair: Tuple[str, str]
    perpetrator: str
    victim: str
    auc_fold_low: float
    auc_fold_high: float
    interaction_type: Optional[str]
    enzyme: Optional[str]
    clinically_significant: bool  # AUC change > 2x or < 0.5x
    recommendation: str


def analyze_ddi_pairs(candidates: List[DrugCandidate], verbose: bool = False) -> List[DDIAnalysisResult]:
    """Analyze all drug pairs for DDI AUC changes.

    Uses predict_auc_change for quantitative predictions of AUC fold-changes.
    """
    results: List[DDIAnalysisResult] = []

    for i, drug_a in enumerate(candidates):
        for drug_b in candidates[i+1:]:
            # Check both directions (A→B and B→A)
            for perp, vic in [(drug_a.name, drug_b.name), (drug_b.name, drug_a.name)]:
                try:
                    pred = predict_auc_change(perp, vic)
                    auc_lo = pred.get("auc_low", 1.0)
                    auc_hi = pred.get("auc_high", 1.0)

                    # Skip if no interaction
                    if auc_lo == 1.0 and auc_hi == 1.0:
                        continue

                    clinically_significant = auc_hi >= 2.0 or auc_lo <= 0.5

                    result = DDIAnalysisResult(
                        drug_pair=(perp, vic),
                        perpetrator=perp,
                        victim=vic,
                        auc_fold_low=auc_lo,
                        auc_fold_high=auc_hi,
                        interaction_type=pred.get("interaction_type"),
                        enzyme=pred.get("enzyme"),
                        clinically_significant=clinically_significant,
                        recommendation=pred.get("recommendation", "Monitor clinically"),
                    )
                    results.append(result)

                    if verbose:
                        print(f"  {perp} → {vic}: AUC {auc_lo:.2f}-{auc_hi:.2f}×")
                except Exception:
                    pass

    return results


@dataclass
class CYPProfileSummary:
    """Summary of CYP profile for a drug candidate."""
    drug: str
    metabolizing_enzymes: List[str]
    is_sensitive_substrate: bool
    is_prodrug: bool
    inhibitor_of: List[str]
    inducer_of: List[str]
    pgx_relevant: bool  # has CYP2D6/2C19/2C9 involvement


def analyze_cyp_profiles(candidates: List[DrugCandidate], verbose: bool = False) -> List[CYPProfileSummary]:
    """Analyze CYP profiles for all candidates."""
    results: List[CYPProfileSummary] = []

    pgx_enzymes = {"CYP2D6", "CYP2C19", "CYP2C9"}

    for drug in candidates:
        try:
            profile = get_drug_profile(drug.name)

            pgx_relevant = bool(pgx_enzymes & set(profile.metabolizing_enzymes))

            result = CYPProfileSummary(
                drug=drug.name,
                metabolizing_enzymes=profile.metabolizing_enzymes,
                is_sensitive_substrate=profile.is_sensitive_substrate,
                is_prodrug=profile.is_prodrug,
                inhibitor_of=list(profile.known_inhibitor_of.keys()),
                inducer_of=list(profile.known_inducer_of.keys()),
                pgx_relevant=pgx_relevant,
            )
            results.append(result)

            if verbose:
                print(f"  {drug.name}: {', '.join(profile.metabolizing_enzymes) or 'none'}")
        except Exception:
            results.append(CYPProfileSummary(
                drug=drug.name,
                metabolizing_enzymes=[],
                is_sensitive_substrate=False,
                is_prodrug=False,
                inhibitor_of=[],
                inducer_of=[],
                pgx_relevant=False,
            ))

    return results


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def print_report(results: List[ScreeningResult]):
    """Print a human-readable screening report."""
    print("\n" + "=" * 72)
    print("  DRUG THERAPY LAB -- Autonomous Screening Report")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 72)

    # Sort by score descending
    ranked = sorted(results, key=lambda r: r.score, reverse=True)

    for rank, r in enumerate(ranked, 1):
        tag = {"PASS": "[PASS]", "CAUTION": "[CAUTION]", "FAIL": "[FAIL]"}[r.verdict]
        print(f"\n  #{rank}  {r.name.upper():20s}  Score: {r.score:5.1f}/100  {tag}")
        print(f"       Cmax={r.Cmax:.3g} mg/L  Tmax={r.Tmax_h:.1f}h  "
              f"t1/2={r.half_life_h:.1f}h  CL={r.clearance_L_h:.1f} L/h")
        print(f"       Css: avg={r.Css_avg:.3g}  peak={r.Css_peak:.3g}  "
              f"trough={r.Css_trough:.3g} mg/L")
        print(f"       Therapeutic index: {r.therapeutic_index:.1f}  "
              f"In window: {'YES' if r.within_window else 'NO'}")
        print(f"       Metabolism: {r.metabolic_stability} "
              f"({r.mm_fraction_Vmax:.0%} Vmax saturation)")
        if r.interaction_details:
            print(f"       Interactions ({r.n_interactions}): "
                  f"worst={r.worst_interaction_severity}")
            for d in r.interaction_details[:3]:
                print(f"         - {d}")
        else:
            print(f"       Interactions: none flagged")

    # Summary
    n_pass = sum(1 for r in ranked if r.verdict == "PASS")
    n_caution = sum(1 for r in ranked if r.verdict == "CAUTION")
    n_fail = sum(1 for r in ranked if r.verdict == "FAIL")
    print(f"\n  {'='*72}")
    print(f"  Summary: {n_pass} PASS / {n_caution} CAUTION / {n_fail} FAIL "
          f"out of {len(ranked)} candidates")
    print(f"  {'='*72}\n")


def save_results(results: List[ScreeningResult], outpath: Path):
    """Save results to JSON."""
    ranked = sorted(results, key=lambda r: r.score, reverse=True)
    data = {
        "timestamp": datetime.now().isoformat(),
        "pipeline": "drug_therapy_lab v0.1",
        "n_candidates": len(ranked),
        "n_pass": sum(1 for r in ranked if r.verdict == "PASS"),
        "n_caution": sum(1 for r in ranked if r.verdict == "CAUTION"),
        "n_fail": sum(1 for r in ranked if r.verdict == "FAIL"),
        "results": [asdict(r) for r in ranked],
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
    parser = argparse.ArgumentParser(description="Drug Therapy Lab -- screening pipeline")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print detailed PK/kinetics reports for each drug")
    parser.add_argument("--drugs", "-d", nargs="+",
                        help="Specific drug names to screen (fetched from database)")
    parser.add_argument("--top", "-t", type=int, default=0,
                        help="Screen top N most prescribed drugs from database")
    parser.add_argument("--offline", action="store_true",
                        help="Use hardcoded fallback data (no API calls)")
    args = parser.parse_args()

    # Determine which candidates to screen
    if args.offline:
        candidates = FALLBACK_CANDIDATES
        print(f"\n  [OFFLINE MODE] Using {len(candidates)} hardcoded candidates...")
    elif args.drugs:
        print(f"\n  Fetching {len(args.drugs)} drugs from database...")
        candidates = load_candidates_from_database(args.drugs, verbose=args.verbose)
        if not candidates:
            print("  No valid drug profiles found. Falling back to hardcoded data.")
            candidates = FALLBACK_CANDIDATES
    elif args.top > 0:
        ds = DrugDataSource()
        drug_names = ds.get_common_drugs()[:args.top]
        print(f"\n  Fetching top {len(drug_names)} drugs from database...")
        candidates = load_candidates_from_database(drug_names, verbose=args.verbose)
        if len(candidates) < 5:
            print("  Too few valid profiles. Adding fallback data.")
            candidates.extend(FALLBACK_CANDIDATES)
    else:
        # Default: use fallback (fast, no API calls)
        candidates = FALLBACK_CANDIDATES
        print(f"\n  Using {len(candidates)} pre-configured candidates...")
        print("  (Use --top N or --drugs NAME to fetch from ChEMBL database)")

    print(f"\n  Screening {len(candidates)} drug candidates...")

    results = []
    for drug in candidates:
        try:
            result = screen_candidate(drug, all_candidates=candidates, verbose=args.verbose)
            results.append(result)
        except Exception as e:
            print(f"  ERROR screening {drug.name}: {e}")

    if not results:
        print("  No results generated.")
        return

    print_report(results)

    outpath = _ROOT / "results" / "labs" / "drug_therapy" / "screening_results.json"
    save_results(results, outpath)

    # Enhanced DDI Analysis Section
    print("\n" + "=" * 72)
    print("  ENHANCED DDI ANALYSIS -- Quantitative AUC Predictions")
    print("=" * 72)

    # CYP Profile Summary
    print("\n  CYP450 Profiles for screened candidates:")
    print(f"  {'Drug':16s} {'Enzymes':20s} {'PGx':>5s} {'Sensitive':>10s}")
    print(f"  {'-'*16} {'-'*20} {'-'*5} {'-'*10}")

    cyp_profiles = analyze_cyp_profiles(candidates, verbose=args.verbose)
    for cp in cyp_profiles:
        enzymes_str = ", ".join(cp.metabolizing_enzymes[:3]) if cp.metabolizing_enzymes else "none"
        if len(cp.metabolizing_enzymes) > 3:
            enzymes_str += "..."
        pgx_str = "YES" if cp.pgx_relevant else ""
        sens_str = "YES" if cp.is_sensitive_substrate else ""
        print(f"  {cp.drug:16s} {enzymes_str:20s} {pgx_str:>5s} {sens_str:>10s}")

    n_pgx = sum(1 for cp in cyp_profiles if cp.pgx_relevant)
    n_sens = sum(1 for cp in cyp_profiles if cp.is_sensitive_substrate)
    print(f"\n  Summary: {n_pgx}/{len(cyp_profiles)} have PGx-relevant enzymes, "
          f"{n_sens}/{len(cyp_profiles)} are sensitive substrates")

    # DDI AUC Predictions
    print("\n  Drug-Drug Interaction AUC Predictions (clinically significant only):")
    print(f"  {'Perpetrator':14s} → {'Victim':14s} {'AUC Change':>15s} {'Enzyme':>10s}")
    print(f"  {'-'*14}   {'-'*14} {'-'*15} {'-'*10}")

    ddi_results = analyze_ddi_pairs(candidates, verbose=args.verbose)
    significant_ddis = [d for d in ddi_results if d.clinically_significant]

    for ddi in significant_ddis[:10]:  # top 10
        auc_str = f"{ddi.auc_fold_low:.1f}-{ddi.auc_fold_high:.1f}×"
        if ddi.auc_fold_low < 1.0:
            auc_str = f"↓ {(1-ddi.auc_fold_high)*100:.0f}-{(1-ddi.auc_fold_low)*100:.0f}%"
        else:
            auc_str = f"↑ {ddi.auc_fold_low:.1f}-{ddi.auc_fold_high:.1f}×"
        enzyme_str = ddi.enzyme or "?"
        print(f"  {ddi.perpetrator:14s} → {ddi.victim:14s} {auc_str:>15s} {enzyme_str:>10s}")

    print(f"\n  Total DDI pairs analyzed: {len(ddi_results)}")
    print(f"  Clinically significant (AUC ≥2× or ≤0.5×): {len(significant_ddis)}")
    print("=" * 72)

    # Save enhanced results
    enhanced_data = {
        "timestamp": datetime.now().isoformat(),
        "cyp_profiles": [asdict(cp) for cp in cyp_profiles],
        "ddi_predictions": [
            {
                "perpetrator": d.perpetrator,
                "victim": d.victim,
                "auc_fold_low": d.auc_fold_low,
                "auc_fold_high": d.auc_fold_high,
                "interaction_type": d.interaction_type,
                "enzyme": d.enzyme,
                "clinically_significant": d.clinically_significant,
            }
            for d in ddi_results
        ],
        "n_significant": len(significant_ddis),
    }
    ddi_path = _ROOT / "results" / "labs" / "drug_therapy" / "ddi_analysis.json"
    with open(ddi_path, "w") as f:
        json.dump(enhanced_data, f, indent=2)
    print(f"  DDI analysis saved to {ddi_path}\n")


if __name__ == "__main__":
    main()
