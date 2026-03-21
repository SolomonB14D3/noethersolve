"""noethersolve.chembl_api — ChEMBL REST API client for drug bioactivity data.

Connects to the ChEMBL REST API (https://www.ebi.ac.uk/chembl/api/data/)
to fetch drug bioactivity data, CYP inhibition Ki values, and target
information. Uses only urllib (no external dependencies).

Complements the static drug_interactions module by providing live lookup
of bioactivity data from ChEMBL's curated database of 2M+ compounds.

Usage:
    from noethersolve.chembl_api import (
        fetch_drug_info, fetch_cyp_inhibition,
        fetch_drug_targets, predict_ddi_from_chembl,
    )

    # Look up a drug
    info = fetch_drug_info("imatinib")
    print(info)  # DrugRecord(name='IMATINIB', chembl_id='CHEMBL941', ...)

    # Get CYP inhibition data
    cyp_data = fetch_cyp_inhibition("ketoconazole", cyp_enzyme="CYP3A4")
    for rec in cyp_data:
        print(f"{rec.cyp_enzyme}: Ki={rec.ki_nm} nM")

    # Predict DDI risk from ChEMBL bioactivity
    risk = predict_ddi_from_chembl("ketoconazole", "midazolam")
    if risk:
        print(f"Risk: {risk.risk_level}, AUC ratio: {risk.predicted_auc_ratio:.1f}x")
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ─── Configuration ────────────────────────────────────────────────────────────

CHEMBL_BASE = "https://www.ebi.ac.uk/chembl/api/data"
REQUEST_TIMEOUT = 15  # seconds
MAX_RESULTS = 100     # max results per API page

# CYP enzyme target ChEMBL IDs (most common drug-metabolizing CYPs)
CYP_TARGET_MAP: Dict[str, str] = {
    "CYP3A4":  "CHEMBL340",
    "CYP2D6":  "CHEMBL289",
    "CYP2C9":  "CHEMBL3397",
    "CYP2C19": "CHEMBL3356",
    "CYP1A2":  "CHEMBL3544",
    "CYP2B6":  "CHEMBL4706",
    "CYP2E1":  "CHEMBL4462",
    "CYP2C8":  "CHEMBL3522",
}

# Reverse map: ChEMBL ID -> CYP name
CYP_ID_TO_NAME: Dict[str, str] = {v: k for k, v in CYP_TARGET_MAP.items()}

# Known CYP substrates for DDI prediction (drug_name_lower -> list of CYP enzymes)
KNOWN_CYP_SUBSTRATES: Dict[str, List[str]] = {
    "midazolam":     ["CYP3A4"],
    "triazolam":     ["CYP3A4"],
    "simvastatin":   ["CYP3A4"],
    "lovastatin":    ["CYP3A4"],
    "atorvastatin":  ["CYP3A4"],
    "cyclosporine":  ["CYP3A4"],
    "tacrolimus":    ["CYP3A4"],
    "sildenafil":    ["CYP3A4"],
    "nifedipine":    ["CYP3A4"],
    "codeine":       ["CYP2D6"],
    "tramadol":      ["CYP2D6"],
    "dextromethorphan": ["CYP2D6"],
    "metoprolol":    ["CYP2D6"],
    "warfarin":      ["CYP2C9"],
    "phenytoin":     ["CYP2C9", "CYP2C19"],
    "celecoxib":     ["CYP2C9"],
    "omeprazole":    ["CYP2C19"],
    "clopidogrel":   ["CYP2C19"],
    "theophylline":  ["CYP1A2"],
    "caffeine":      ["CYP1A2"],
    "tizanidine":    ["CYP1A2"],
}


# ─── Session Cache ────────────────────────────────────────────────────────────

_cache: Dict[str, Any] = {}


def _cache_key(prefix: str, *args: str) -> str:
    """Build a deterministic cache key."""
    return f"{prefix}:{'|'.join(a.lower().strip() for a in args)}"


# ─── HTTP Helper ──────────────────────────────────────────────────────────────

def _api_get(url: str) -> Optional[Dict]:
    """Fetch JSON from a URL, returning None on any failure."""
    if url in _cache:
        return _cache[url]

    try:
        req = urllib.request.Request(
            url,
            headers={
                "Accept": "application/json",
                "User-Agent": "NoetherSolve/1.0 (research tool)",
            },
        )
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            _cache[url] = data
            return data
    except urllib.error.HTTPError as e:
        logger.warning("ChEMBL HTTP error %d for %s", e.code, url)
        return None
    except urllib.error.URLError as e:
        logger.warning("ChEMBL network error for %s: %s", url, e.reason)
        return None
    except (json.JSONDecodeError, OSError, TimeoutError) as e:
        logger.warning("ChEMBL request failed for %s: %s", url, e)
        return None


# ─── Dataclasses ──────────────────────────────────────────────────────────────

@dataclass
class DrugRecord:
    """Core molecular properties from ChEMBL."""
    name: str
    chembl_id: str
    molecular_weight: Optional[float] = None
    logp: Optional[float] = None
    psa: Optional[float] = None        # polar surface area (A^2)
    max_phase: Optional[int] = None     # 4 = approved, 3 = phase III, etc.

    def __str__(self) -> str:
        parts = [f"{self.name} ({self.chembl_id})"]
        if self.molecular_weight is not None:
            parts.append(f"MW={self.molecular_weight:.1f}")
        if self.logp is not None:
            parts.append(f"logP={self.logp:.2f}")
        if self.psa is not None:
            parts.append(f"PSA={self.psa:.1f} A^2")
        if self.max_phase is not None:
            phase_labels = {4: "Approved", 3: "Phase III", 2: "Phase II",
                            1: "Phase I", 0: "Preclinical"}
            parts.append(phase_labels.get(self.max_phase, f"Phase {self.max_phase}"))
        return " | ".join(parts)


@dataclass
class CYPInhibitionRecord:
    """CYP enzyme inhibition data from ChEMBL bioactivity."""
    drug_name: str
    cyp_enzyme: str
    ki_nm: Optional[float] = None       # Ki in nanomolar
    ic50_nm: Optional[float] = None     # IC50 in nanomolar
    source_assay: Optional[str] = None  # ChEMBL assay ID

    def __str__(self) -> str:
        parts = [f"{self.drug_name} -> {self.cyp_enzyme}"]
        if self.ki_nm is not None:
            parts.append(f"Ki={self.ki_nm:.1f} nM")
        if self.ic50_nm is not None:
            parts.append(f"IC50={self.ic50_nm:.1f} nM")
        if self.source_assay:
            parts.append(f"assay={self.source_assay}")
        return " | ".join(parts)


@dataclass
class TargetRecord:
    """Drug-target bioactivity from ChEMBL."""
    drug_name: str
    target_name: str
    target_chembl_id: str
    activity_type: str                  # Ki, IC50, EC50, Kd, etc.
    activity_value: Optional[float] = None
    activity_units: Optional[str] = None

    def __str__(self) -> str:
        parts = [f"{self.drug_name} -> {self.target_name} ({self.target_chembl_id})"]
        if self.activity_value is not None:
            unit_str = f" {self.activity_units}" if self.activity_units else ""
            parts.append(f"{self.activity_type}={self.activity_value:.2g}{unit_str}")
        return " | ".join(parts)


@dataclass
class DDIRiskRecord:
    """Predicted drug-drug interaction risk from ChEMBL Ki data."""
    drug1: str
    drug2: str
    shared_cyp: str
    inhibitor_ki_nm: Optional[float] = None
    substrate_name: str = ""
    predicted_auc_ratio: Optional[float] = None  # fold-change in AUC
    risk_level: str = "unknown"  # low / moderate / high / contraindicated

    def __str__(self) -> str:
        parts = [f"DDI: {self.drug1} + {self.drug2} via {self.shared_cyp}"]
        if self.inhibitor_ki_nm is not None:
            parts.append(f"Ki={self.inhibitor_ki_nm:.1f} nM")
        if self.predicted_auc_ratio is not None:
            parts.append(f"AUC ratio={self.predicted_auc_ratio:.2f}x")
        parts.append(f"risk={self.risk_level}")
        return " | ".join(parts)


# ─── Core API Functions ───────────────────────────────────────────────────────

def fetch_drug_info(drug_name: str) -> Optional[DrugRecord]:
    """Search ChEMBL for a drug by name and return molecular properties.

    Args:
        drug_name: Drug name (generic or trade name).

    Returns:
        DrugRecord with molecular properties, or None if not found / network error.
    """
    ck = _cache_key("drug_info", drug_name)
    if ck in _cache:
        return _cache[ck]

    encoded = urllib.parse.quote(drug_name.strip())
    url = f"{CHEMBL_BASE}/molecule/search?q={encoded}&format=json&limit=5"
    data = _api_get(url)
    if data is None or not data.get("molecules"):
        return None

    # Pick the best match: prefer highest max_phase, then first result
    molecules = data["molecules"]
    best = max(molecules, key=lambda m: (m.get("max_phase") or 0))

    props = best.get("molecule_properties") or {}
    record = DrugRecord(
        name=(best.get("pref_name") or drug_name).upper(),
        chembl_id=best.get("molecule_chembl_id", ""),
        molecular_weight=_safe_float(props.get("full_mwt")),
        logp=_safe_float(props.get("alogp")),
        psa=_safe_float(props.get("psa")),
        max_phase=best.get("max_phase"),
    )
    _cache[ck] = record
    return record


def fetch_cyp_inhibition(
    drug_name: str,
    cyp_enzyme: Optional[str] = None,
) -> List[CYPInhibitionRecord]:
    """Fetch CYP inhibition Ki/IC50 values for a drug from ChEMBL.

    Args:
        drug_name: Drug name to look up.
        cyp_enzyme: Optional CYP enzyme filter (e.g., "CYP3A4"). If None,
                    returns inhibition data for all CYP enzymes.

    Returns:
        List of CYPInhibitionRecord. Empty list if not found or on error.
    """
    ck = _cache_key("cyp_inhib", drug_name, cyp_enzyme or "all")
    if ck in _cache:
        return _cache[ck]

    # First, resolve drug name -> ChEMBL ID
    drug = fetch_drug_info(drug_name)
    if drug is None:
        return []

    # Determine which CYP targets to query
    if cyp_enzyme:
        cyp_upper = cyp_enzyme.upper().replace(" ", "")
        target_ids = {cyp_upper: CYP_TARGET_MAP.get(cyp_upper)}
        if target_ids[cyp_upper] is None:
            logger.warning("Unknown CYP enzyme: %s", cyp_enzyme)
            return []
    else:
        target_ids = dict(CYP_TARGET_MAP)

    results: List[CYPInhibitionRecord] = []

    for cyp_name, target_chembl_id in target_ids.items():
        if target_chembl_id is None:
            continue
        url = (
            f"{CHEMBL_BASE}/activity?"
            f"molecule_chembl_id={drug.chembl_id}"
            f"&target_chembl_id={target_chembl_id}"
            f"&format=json&limit={MAX_RESULTS}"
        )
        data = _api_get(url)
        if data is None or not data.get("activities"):
            continue

        for act in data["activities"]:
            activity_type = (act.get("standard_type") or "").upper()
            if activity_type not in ("KI", "IC50", "KB"):
                continue

            value = _safe_float(act.get("standard_value"))
            units = act.get("standard_units") or ""
            assay_id = act.get("assay_chembl_id") or ""

            # Convert to nM if needed
            value_nm = _convert_to_nm(value, units)

            rec = CYPInhibitionRecord(
                drug_name=drug.name,
                cyp_enzyme=cyp_name,
                ki_nm=value_nm if activity_type == "KI" else None,
                ic50_nm=value_nm if activity_type in ("IC50", "KB") else None,
                source_assay=assay_id,
            )
            results.append(rec)

    _cache[ck] = results
    return results


def fetch_drug_targets(drug_name: str) -> List[TargetRecord]:
    """Fetch known drug targets with activity values from ChEMBL.

    Args:
        drug_name: Drug name to look up.

    Returns:
        List of TargetRecord with bioactivity data. Empty list on error.
    """
    ck = _cache_key("targets", drug_name)
    if ck in _cache:
        return _cache[ck]

    drug = fetch_drug_info(drug_name)
    if drug is None:
        return []

    url = (
        f"{CHEMBL_BASE}/activity?"
        f"molecule_chembl_id={drug.chembl_id}"
        f"&target_type=SINGLE_PROTEIN"
        f"&format=json&limit={MAX_RESULTS}"
    )
    data = _api_get(url)
    if data is None or not data.get("activities"):
        return []

    # Deduplicate by (target_id, activity_type) keeping best (lowest) value
    best: Dict[Tuple[str, str], TargetRecord] = {}

    for act in data["activities"]:
        target_id = act.get("target_chembl_id") or ""
        target_name = act.get("target_pref_name") or "Unknown"
        act_type = (act.get("standard_type") or "").upper()
        value = _safe_float(act.get("standard_value"))
        units = act.get("standard_units") or ""

        if not target_id or not act_type:
            continue

        key = (target_id, act_type)
        rec = TargetRecord(
            drug_name=drug.name,
            target_name=target_name,
            target_chembl_id=target_id,
            activity_type=act_type,
            activity_value=value,
            activity_units=units,
        )

        # Keep the record with the lowest (most potent) activity value
        if key not in best:
            best[key] = rec
        elif value is not None and (best[key].activity_value is None
                                     or value < best[key].activity_value):
            best[key] = rec

    results = sorted(best.values(), key=lambda r: (r.activity_value or 1e12))
    _cache[ck] = results
    return results


def predict_ddi_from_chembl(
    drug1: str,
    drug2: str,
) -> Optional[DDIRiskRecord]:
    """Predict drug-drug interaction risk using ChEMBL Ki data.

    Uses the basic DDI equation from FDA guidance:
        AUC_ratio = 1 + [I] / Ki

    where [I] is the estimated inhibitor concentration (approximated from
    typical clinical Cmax) and Ki is from ChEMBL bioactivity data.

    For drugs without known Cmax, uses a default estimate based on
    typical oral dosing (Cmax ~ 1000 nM for most oral drugs).

    Args:
        drug1: First drug name.
        drug2: Second drug name.

    Returns:
        DDIRiskRecord with predicted AUC ratio and risk level, or None
        if insufficient data exists for prediction.
    """
    ck = _cache_key("ddi", drug1, drug2)
    if ck in _cache:
        return _cache[ck]

    # Try both orderings: drug1 as inhibitor of drug2's metabolism, and vice versa
    result = _try_ddi_direction(drug1, drug2)
    if result is None:
        result = _try_ddi_direction(drug2, drug1)

    if result is not None:
        _cache[ck] = result
    return result


# ─── Internal Helpers ─────────────────────────────────────────────────────────

# Typical Cmax estimates (nM) for common inhibitors at standard clinical doses
# Source: FDA DDI guidance tables, clinical pharmacology literature
_CMAX_ESTIMATES_NM: Dict[str, float] = {
    "ketoconazole":   7800.0,   # 400 mg oral
    "itraconazole":   1500.0,   # 200 mg oral
    "fluconazole":    25000.0,  # 200 mg oral
    "ritonavir":      15000.0,  # 100 mg oral (boosting dose)
    "clarithromycin": 5000.0,   # 500 mg oral
    "erythromycin":   5000.0,   # 500 mg oral
    "diltiazem":      600.0,    # 120 mg oral
    "verapamil":      500.0,    # 120 mg oral
    "fluvoxamine":    600.0,    # 100 mg oral
    "fluoxetine":     1000.0,   # 20 mg oral
    "paroxetine":     200.0,    # 20 mg oral
    "quinidine":      9000.0,   # 200 mg oral
    "cimetidine":     10000.0,  # 400 mg oral
    "ciprofloxacin":  8000.0,   # 500 mg oral
}

DEFAULT_CMAX_NM = 1000.0  # Conservative default for unknown drugs


def _try_ddi_direction(
    inhibitor: str,
    substrate: str,
) -> Optional[DDIRiskRecord]:
    """Try to predict DDI with inhibitor -> substrate direction.

    Returns DDIRiskRecord if sufficient data, None otherwise.
    """
    sub_lower = substrate.strip().lower()

    # Check if substrate has known CYP metabolism
    substrate_cyps = KNOWN_CYP_SUBSTRATES.get(sub_lower)
    if not substrate_cyps:
        # Try to infer from ChEMBL target data
        substrate_cyps = _infer_cyp_substrates(substrate)
        if not substrate_cyps:
            return None

    # Get inhibitor's CYP inhibition data for relevant enzymes
    best_ki: Optional[float] = None
    best_cyp: Optional[str] = None

    for cyp in substrate_cyps:
        records = fetch_cyp_inhibition(inhibitor, cyp_enzyme=cyp)
        for rec in records:
            ki = rec.ki_nm or rec.ic50_nm
            if ki is not None and ki > 0:
                if best_ki is None or ki < best_ki:
                    best_ki = ki
                    best_cyp = cyp

    if best_ki is None or best_cyp is None:
        return None

    # Estimate AUC ratio: AUC_ratio = 1 + [I]/Ki (FDA basic model)
    cmax = _CMAX_ESTIMATES_NM.get(inhibitor.strip().lower(), DEFAULT_CMAX_NM)
    auc_ratio = 1.0 + cmax / best_ki

    # Classify risk
    risk_level = _classify_risk(auc_ratio)

    return DDIRiskRecord(
        drug1=inhibitor.strip(),
        drug2=substrate.strip(),
        shared_cyp=best_cyp,
        inhibitor_ki_nm=best_ki,
        substrate_name=substrate.strip(),
        predicted_auc_ratio=auc_ratio,
        risk_level=risk_level,
    )


def _infer_cyp_substrates(drug_name: str) -> List[str]:
    """Try to infer CYP substrates from ChEMBL target data."""
    targets = fetch_drug_targets(drug_name)
    cyps = []
    for t in targets:
        cyp_name = CYP_ID_TO_NAME.get(t.target_chembl_id)
        if cyp_name and cyp_name not in cyps:
            cyps.append(cyp_name)
    return cyps


def _classify_risk(auc_ratio: float) -> str:
    """Classify DDI risk based on predicted AUC fold-change.

    Based on FDA DDI guidance classification:
    - Strong: AUC >= 5x
    - Moderate: 2x <= AUC < 5x
    - Weak: 1.25x <= AUC < 2x
    - Minimal: AUC < 1.25x
    """
    if auc_ratio >= 5.0:
        return "high"
    elif auc_ratio >= 2.0:
        return "moderate"
    elif auc_ratio >= 1.25:
        return "low"
    else:
        return "minimal"


def _safe_float(val: Any) -> Optional[float]:
    """Safely convert a value to float, returning None on failure."""
    if val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _convert_to_nm(value: Optional[float], units: str) -> Optional[float]:
    """Convert an activity value to nanomolar."""
    if value is None:
        return None

    units_upper = units.upper().strip()
    if units_upper in ("NM", "NMOL/L"):
        return value
    elif units_upper in ("UM", "UMOL/L"):
        return value * 1000.0
    elif units_upper in ("MM", "MMOL/L"):
        return value * 1e6
    elif units_upper in ("PM", "PMOL/L"):
        return value / 1000.0
    elif units_upper in ("M", "MOL/L"):
        return value * 1e9
    else:
        # Assume nM if units unrecognized
        return value


def clear_cache() -> int:
    """Clear the session cache. Returns number of entries cleared."""
    n = len(_cache)
    _cache.clear()
    return n
