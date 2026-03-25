"""Shared domain name normalization — single source of truth.

Every part of the pipeline that matches domain names to adapter/facts files
MUST use normalize_domain_name() from this module. This prevents the recurring
bug where special characters (parentheses, hyphens) in domain display names
cause adapter lookup failures.

History: "Optimal f(r) Combination" → "optimal_f(r)_combination" broke matching
because adapter files use "optimal_fr_combination". Bio-AI, Stretch-Resistant,
and other hyphenated names had similar issues.
"""

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Project root — resolved from this file's location
_PROJECT = Path(__file__).resolve().parent.parent

# Default model for oracle evaluation. Upgraded from 4B to 14B on 2026-03-25.
# 4B adapters still compatible (same vocab 151936).
DEFAULT_MODEL = "Qwen/Qwen3-14B-Base"

# Canonical mapping for domain names that can't be derived by normalization alone.
# Key: normalized display name. Value: file prefix used on disk.
# This is the SINGLE SOURCE OF TRUTH — adapter_trainer.py, the hook, oracle_wrapper,
# and all other callers import from here.
DOMAIN_CANONICAL: Dict[str, str] = {
    "q_f_ratio_invariant": "qf_ratio",
    "qf_ratio_invariant": "qf_ratio",
    "electromagnetic_zilch_and_optical_chirality": "em_zilch",
    "continuous_q_f_and_euler_conservation_laws": "continuous_qf",
    "continuous_qf_and_euler_conservation_laws": "continuous_qf",
    "bioai_computational_parallels": "bio_ai_parallels",
    "bio_ai_computational_parallels": "bio_ai_parallels",
    "reduced_navier_stokes_vortex_conservation": "vortex_pair",
    "reduced_navier_stokes_vortex_conservation_unsolved": "vortex_pair",
    "optimal_fr_combination": "optimal_f",
    "optimal_f_r_combination": "optimal_f",
    "kinetic_invariant_k": "kinetic_k",
    "elliptic_curve_theory": "elliptic_curve",
    "ns_regularity_and_stretchresistant_q_f": "ns_regularity",
    "ns_regularity_and_stretch_resistant_q_f": "ns_regularity",
    "hamiltonian_mechanics_invariants": "hamiltonian",
    "chemical_reaction_network_conservation": "chemical_conservation",
    "information_theory": "information_theory",
    "3body_conservation": "3body_conservation",
    "llm_hallucination_grounded": "llm_hallucination_balanced",
    "physics_fundamentals": "physics_fundamentals_2d_turbulence",
    "organic_chemistry": "chemistry",
    "clinical_biochemistry": "biochemistry",
    "biology": "aging_biology",
    "geophysics_seismic": "geophysics_seismic",
    "pathophysiology": "pathophysiology",
}


def model_short(model: str) -> str:
    """Convert a model name/path to its short directory form.

    Matches the convention used by steering vectors and adapters:
    "Qwen/Qwen3-4B-Base" → "qwen3_4b_base"
    """
    return model.split("/")[-1].lower().replace("-", "_")


def get_adapters_dir(model: Optional[str] = None, project: Optional[Path] = None) -> Path:
    """Return the adapter directory for a given model.

    Args:
        model: Model name (e.g., "Qwen/Qwen3-4B-Base"). Defaults to DEFAULT_MODEL.
        project: Project root. Auto-detected if None.

    Returns:
        Path like /project/adapters/qwen3_4b_base/
    """
    proj = project or _PROJECT
    m = model_short(model or DEFAULT_MODEL)
    d = proj / "adapters" / m
    d.mkdir(parents=True, exist_ok=True)
    return d


def normalize_domain_name(name: str) -> str:
    """Normalize a domain display name to snake_case for file matching.

    Handles: spaces, hyphens, parentheses, version suffixes, mixed case.
    Deterministic and idempotent (normalize(normalize(x)) == normalize(x)).

    Examples:
        >>> normalize_domain_name("Optimal f(r) Combination")
        'optimal_fr_combination'
        >>> normalize_domain_name("Bio-AI Computational Parallels V2")
        'bioai_computational_parallels'
        >>> normalize_domain_name("NS Regularity and Stretch-Resistant Q_f")
        'ns_regularity_and_stretchresistant_q_f'
    """
    base = name.replace(" V2", "").replace("_v2", "").replace(" v2", "")
    base = base.replace(" ", "_").lower()
    base = "".join(c for c in base if c.isalnum() or c == "_")
    # Collapse multiple underscores
    base = re.sub(r"_+", "_", base).strip("_")
    return base


def domain_to_file_prefix(name: str) -> str:
    """Map a domain display name to its adapter/facts file prefix.

    First normalizes, then checks the canonical mapping for special cases.
    Returns the file prefix used on disk (e.g., "optimal_f", "em_zilch").

    Examples:
        >>> domain_to_file_prefix("Optimal f(r) Combination")
        'optimal_f'
        >>> domain_to_file_prefix("EM Zilch and Optical Chirality")
        'em_zilch_and_optical_chirality'  # no mapping, returns normalized
    """
    norm = normalize_domain_name(name)
    return DOMAIN_CANONICAL.get(norm, norm)


def has_adapter(name: str, adapters_dir: Path) -> bool:
    """Check if any adapter .npz exists for a domain.

    Uses both normalized name and canonical mapping. Falls back to
    prefix matching for partial hits.
    """
    norm = normalize_domain_name(name)
    prefix = domain_to_file_prefix(name)

    # Check both normalized and canonical prefix
    for p in [norm, prefix]:
        if list(adapters_dir.glob(f"{p}*.npz")):
            return True

    # Substring fallback (catches e.g. "bio_ai" in "bioai_computational_parallels_adapter.npz")
    if any(norm in a.stem for a in adapters_dir.glob("*.npz")):
        return True

    return False


def find_adapters(name: str, adapters_dir: Path) -> List[Path]:
    """Find all adapter .npz files for a domain."""
    norm = normalize_domain_name(name)
    prefix = domain_to_file_prefix(name)

    found = set()
    for p in [norm, prefix]:
        found.update(adapters_dir.glob(f"{p}*.npz"))

    return sorted(found)


def find_facts_file(
    name: str,
    problems_dir: Path,
    prefer_v2: bool = True,
) -> Optional[Path]:
    """Find the facts JSON file for a domain.

    Prefers V2 over V1 when both exist.
    """
    norm = normalize_domain_name(name)
    prefix = domain_to_file_prefix(name)

    for p in [prefix, norm]:
        if prefer_v2:
            v2 = problems_dir / f"{p}_facts_v2.json"
            if v2.exists():
                return v2
        v1 = problems_dir / f"{p}_facts.json"
        if v1.exists():
            return v1

    # Glob fallback
    for p in [prefix, norm]:
        candidates = sorted(problems_dir.glob(f"{p}*_facts*.json"), reverse=True)
        if candidates:
            return candidates[0]

    return None
