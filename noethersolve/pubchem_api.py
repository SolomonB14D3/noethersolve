"""noethersolve.pubchem_api — PubChem REST API client for compound data.

Connects to the PubChem PUG REST API (https://pubchem.ncbi.nlm.nih.gov/rest/pug)
to fetch compound properties, safety data, and Lipinski Rule of 5 analysis.
Uses only urllib (no external dependencies).

Complements the ChEMBL API module by providing compound-level property data,
GHS safety classification, and drug-likeness screening from PubChem's
110M+ compound database.

Usage:
    from noethersolve.pubchem_api import (
        fetch_compound_info, fetch_compound_properties,
        check_lipinski, fetch_compound_safety,
    )

    # Look up a compound
    info = fetch_compound_info("aspirin")
    print(info)  # CompoundRecord(cid=2244, iupac_name='2-acetoxybenzoic acid', ...)

    # Check drug-likeness
    lip = check_lipinski("ibuprofen")
    print(lip)  # passes=True, n_violations=0

    # Get safety data
    safety = fetch_compound_safety("methanol")
    print(safety)  # signal_word='Danger', hazard_statements=[...]
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ─── Configuration ────────────────────────────────────────────────────────────

PUBCHEM_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
PUBCHEM_VIEW = "https://pubchem.ncbi.nlm.nih.gov/rest/pug_view"
REQUEST_TIMEOUT = 15  # seconds

# ─── Local Fallback Cache ────────────────────────────────────────────────────
# 10 common drugs with verified properties for offline / rate-limited use.

_FALLBACK_COMPOUNDS: Dict[str, Dict[str, Any]] = {
    "aspirin": {
        "cid": 2244,
        "iupac_name": "2-acetoxybenzoic acid",
        "molecular_formula": "C9H8O4",
        "molecular_weight": 180.16,
        "canonical_smiles": "CC(=O)OC1=CC=CC=C1C(=O)O",
        "inchi_key": "BSYNRYMUTXBXSQ-UHFFFAOYSA-N",
        "xlogp": 1.2,
        "tpsa": 63.6,
        "hbd": 1,
        "hba": 4,
        "rotatable_bonds": 3,
    },
    "ibuprofen": {
        "cid": 3672,
        "iupac_name": "2-[4-(2-methylpropyl)phenyl]propanoic acid",
        "molecular_formula": "C13H18O2",
        "molecular_weight": 206.28,
        "canonical_smiles": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
        "inchi_key": "HEFNNWSXXWATRW-UHFFFAOYSA-N",
        "xlogp": 3.5,
        "tpsa": 37.3,
        "hbd": 1,
        "hba": 2,
        "rotatable_bonds": 4,
    },
    "acetaminophen": {
        "cid": 1983,
        "iupac_name": "N-(4-hydroxyphenyl)acetamide",
        "molecular_formula": "C8H9NO2",
        "molecular_weight": 151.16,
        "canonical_smiles": "CC(=O)NC1=CC=C(O)C=C1",
        "inchi_key": "RZVAJINKPMORJF-UHFFFAOYSA-N",
        "xlogp": 0.5,
        "tpsa": 49.3,
        "hbd": 2,
        "hba": 2,
        "rotatable_bonds": 1,
    },
    "metformin": {
        "cid": 4091,
        "iupac_name": "3-(diaminomethylidene)-1,1-dimethylguanidine",
        "molecular_formula": "C4H11N5",
        "molecular_weight": 129.16,
        "canonical_smiles": "CN(C)C(=N)N=C(N)N",
        "inchi_key": "BPCAIDRSOTFPOS-UHFFFAOYSA-N",
        "xlogp": -1.4,
        "tpsa": 91.5,
        "hbd": 3,
        "hba": 5,
        "rotatable_bonds": 2,
    },
    "atorvastatin": {
        "cid": 60823,
        "iupac_name": "(3R,5R)-7-[2-(4-fluorophenyl)-3-phenyl-4-(phenylcarbamoyl)-5-propan-2-ylpyrrol-1-yl]-3,5-dihydroxyheptanoic acid",
        "molecular_formula": "C33H35FN2O5",
        "molecular_weight": 558.64,
        "canonical_smiles": "CC(C)C1=C(C(=C(N1CCC(CC(CC(=O)O)O)O)C2=CC=CC=C2)C3=CC=C(C=C3)F)C(=O)NC4=CC=CC=C4",
        "inchi_key": "XUKUURHRXDUEBC-UHFFFAOYSA-N",
        "xlogp": 6.4,
        "tpsa": 111.8,
        "hbd": 4,
        "hba": 7,
        "rotatable_bonds": 12,
    },
    "omeprazole": {
        "cid": 4594,
        "iupac_name": "5-methoxy-2-[(4-methoxy-3,5-dimethylpyridin-2-yl)methylsulfinyl]-1H-benzimidazole",
        "molecular_formula": "C17H19N3O3S",
        "molecular_weight": 345.42,
        "canonical_smiles": "CC1=CN=C(C(=C1OC)C)CS(=O)C2=NC3=C(N2)C=C(C=C3)OC",
        "inchi_key": "SUBDBMMJDZJVOS-UHFFFAOYSA-N",
        "xlogp": 2.2,
        "tpsa": 77.1,
        "hbd": 1,
        "hba": 6,
        "rotatable_bonds": 5,
    },
    "lisinopril": {
        "cid": 5362119,
        "iupac_name": "(2S)-1-[(2S)-6-amino-2-[[(1S)-1-carboxy-3-phenylpropyl]amino]hexanoyl]pyrrolidine-2-carboxylic acid",
        "molecular_formula": "C21H31N3O5",
        "molecular_weight": 405.49,
        "canonical_smiles": "C(CCN)CC(C(=O)N1CCCC1C(=O)O)NC(CCC2=CC=CC=C2)C(=O)O",
        "inchi_key": "RLAWWYSOJDYHDC-BZSNNMDCSA-N",
        "xlogp": -0.8,
        "tpsa": 132.6,
        "hbd": 4,
        "hba": 7,
        "rotatable_bonds": 12,
    },
    "metoprolol": {
        "cid": 4171,
        "iupac_name": "1-[4-(2-methoxyethyl)phenoxy]-3-(propan-2-ylamino)propan-2-ol",
        "molecular_formula": "C15H25NO3",
        "molecular_weight": 267.36,
        "canonical_smiles": "CC(C)NCC(COC1=CC=C(C=C1)CCOC)O",
        "inchi_key": "IUBSYMUCCVLNPP-UHFFFAOYSA-N",
        "xlogp": 1.9,
        "tpsa": 50.7,
        "hbd": 2,
        "hba": 4,
        "rotatable_bonds": 9,
    },
    "amlodipine": {
        "cid": 2162,
        "iupac_name": "3-O-ethyl 5-O-methyl 2-(2-aminoethoxymethyl)-4-(2-chlorophenyl)-6-methyl-1,4-dihydropyridine-3,5-dicarboxylate",
        "molecular_formula": "C20H25ClN2O5",
        "molecular_weight": 408.88,
        "canonical_smiles": "CCOC(=O)C1=C(NC(=C(C1C2=CC=CC=C2Cl)C(=O)OC)C)COCC N",
        "inchi_key": "HTIQEAQVCQDNRM-UHFFFAOYSA-N",
        "xlogp": 3.0,
        "tpsa": 99.9,
        "hbd": 3,
        "hba": 7,
        "rotatable_bonds": 10,
    },
    "levothyroxine": {
        "cid": 5819,
        "iupac_name": "(2S)-2-amino-3-[4-(4-hydroxy-3,5-diiodophenoxy)-3,5-diiodophenyl]propanoic acid",
        "molecular_formula": "C15H11I4NO4",
        "molecular_weight": 776.87,
        "canonical_smiles": "C1=CC(=C(C(=C1OC2=CC(=C(C(=C2)I)O)I)I)I)CC(C(=O)O)N",
        "inchi_key": "XUIIKFGFIJCVMT-LBPRGKRZSA-N",
        "xlogp": 4.1,
        "tpsa": 92.8,
        "hbd": 4,
        "hba": 5,
        "rotatable_bonds": 5,
    },
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
        logger.warning("PubChem HTTP error %d for %s", e.code, url)
        return None
    except urllib.error.URLError as e:
        logger.warning("PubChem network error for %s: %s", url, e.reason)
        return None
    except (json.JSONDecodeError, OSError, TimeoutError) as e:
        logger.warning("PubChem request failed for %s: %s", url, e)
        return None


# ─── Dataclasses ──────────────────────────────────────────────────────────────

@dataclass
class CompoundRecord:
    """Core compound properties from PubChem."""
    cid: int
    iupac_name: Optional[str] = None
    molecular_formula: Optional[str] = None
    molecular_weight: Optional[float] = None
    canonical_smiles: Optional[str] = None
    inchi_key: Optional[str] = None
    xlogp: Optional[float] = None
    tpsa: Optional[float] = None
    hbd: Optional[int] = None   # hydrogen bond donors
    hba: Optional[int] = None   # hydrogen bond acceptors
    rotatable_bonds: Optional[int] = None

    def __str__(self) -> str:
        parts = [f"CID {self.cid}"]
        if self.iupac_name:
            parts.append(self.iupac_name)
        if self.molecular_formula:
            parts.append(f"Formula={self.molecular_formula}")
        if self.molecular_weight is not None:
            parts.append(f"MW={self.molecular_weight:.2f}")
        if self.canonical_smiles:
            parts.append(f"SMILES={self.canonical_smiles}")
        if self.inchi_key:
            parts.append(f"InChIKey={self.inchi_key}")
        if self.xlogp is not None:
            parts.append(f"XLogP={self.xlogp:.1f}")
        if self.tpsa is not None:
            parts.append(f"TPSA={self.tpsa:.1f} A^2")
        if self.hbd is not None:
            parts.append(f"HBD={self.hbd}")
        if self.hba is not None:
            parts.append(f"HBA={self.hba}")
        if self.rotatable_bonds is not None:
            parts.append(f"RotBonds={self.rotatable_bonds}")
        return " | ".join(parts)


@dataclass
class PropertyRecord:
    """Compound physicochemical properties with Lipinski assessment."""
    name: str
    molecular_formula: Optional[str] = None
    molecular_weight: Optional[float] = None
    xlogp: Optional[float] = None
    tpsa: Optional[float] = None
    hbd: Optional[int] = None
    hba: Optional[int] = None
    rotatable_bonds: Optional[int] = None
    complexity: Optional[float] = None
    lipinski_passes: bool = True
    lipinski_violations: int = 0

    def __str__(self) -> str:
        parts = [self.name]
        if self.molecular_formula:
            parts.append(f"Formula={self.molecular_formula}")
        if self.molecular_weight is not None:
            parts.append(f"MW={self.molecular_weight:.2f}")
        if self.xlogp is not None:
            parts.append(f"XLogP={self.xlogp:.1f}")
        if self.tpsa is not None:
            parts.append(f"TPSA={self.tpsa:.1f} A^2")
        if self.hbd is not None:
            parts.append(f"HBD={self.hbd}")
        if self.hba is not None:
            parts.append(f"HBA={self.hba}")
        if self.rotatable_bonds is not None:
            parts.append(f"RotBonds={self.rotatable_bonds}")
        if self.complexity is not None:
            parts.append(f"Complexity={self.complexity:.1f}")
        parts.append(f"Lipinski={'PASS' if self.lipinski_passes else 'FAIL'} ({self.lipinski_violations} violations)")
        return " | ".join(parts)


@dataclass
class LipinskiResult:
    """Lipinski Rule of 5 drug-likeness assessment."""
    name: str
    molecular_weight: Optional[float] = None
    hbd: Optional[int] = None
    hba: Optional[int] = None
    xlogp: Optional[float] = None
    n_violations: int = 0
    passes: bool = True
    violations_list: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        parts = [f"{self.name} — Lipinski Ro5: {'PASS' if self.passes else 'FAIL'}"]
        if self.molecular_weight is not None:
            flag = " *" if self.molecular_weight >= 500 else ""
            parts.append(f"MW={self.molecular_weight:.2f}{flag}")
        if self.hbd is not None:
            flag = " *" if self.hbd > 5 else ""
            parts.append(f"HBD={self.hbd}{flag}")
        if self.hba is not None:
            flag = " *" if self.hba > 10 else ""
            parts.append(f"HBA={self.hba}{flag}")
        if self.xlogp is not None:
            flag = " *" if self.xlogp > 5 else ""
            parts.append(f"XLogP={self.xlogp:.1f}{flag}")
        parts.append(f"Violations: {self.n_violations}/4")
        if self.violations_list:
            parts.append("Failed: " + ", ".join(self.violations_list))
        return " | ".join(parts)


@dataclass
class SafetyRecord:
    """GHS safety classification from PubChem."""
    name: str
    cid: int
    signal_word: Optional[str] = None
    hazard_statements: List[str] = field(default_factory=list)
    pictograms: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        parts = [f"{self.name} (CID {self.cid})"]
        if self.signal_word:
            parts.append(f"Signal: {self.signal_word}")
        if self.hazard_statements:
            parts.append(f"Hazards: {'; '.join(self.hazard_statements)}")
        if self.pictograms:
            parts.append(f"Pictograms: {', '.join(self.pictograms)}")
        if not self.signal_word and not self.hazard_statements:
            parts.append("No GHS classification data available")
        return " | ".join(parts)


# ─── Internal Helpers ─────────────────────────────────────────────────────────

def _safe_float(val: Any) -> Optional[float]:
    """Safely convert a value to float, returning None on failure."""
    if val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _safe_int(val: Any) -> Optional[int]:
    """Safely convert a value to int, returning None on failure."""
    if val is None:
        return None
    try:
        return int(val)
    except (ValueError, TypeError):
        return None


def _is_cid(name_or_cid: str) -> bool:
    """Check if the input looks like a numeric CID."""
    return name_or_cid.strip().isdigit()


def _resolve_cid(name_or_cid: str) -> Optional[int]:
    """Resolve a compound name or CID string to a numeric CID.

    Returns the CID as int, or None if resolution fails.
    """
    cleaned = name_or_cid.strip()
    if _is_cid(cleaned):
        return int(cleaned)

    # Try the local fallback first
    lower = cleaned.lower()
    if lower in _FALLBACK_COMPOUNDS:
        return _FALLBACK_COMPOUNDS[lower]["cid"]

    # Query PubChem by name to get CID
    encoded = urllib.parse.quote(cleaned)
    url = f"{PUBCHEM_BASE}/compound/name/{encoded}/cids/JSON"
    data = _api_get(url)
    if data is None:
        return None

    cids = data.get("IdentifierList", {}).get("CID", [])
    if cids:
        return cids[0]
    return None


def _get_fallback(name_or_cid: str) -> Optional[Dict[str, Any]]:
    """Check the local fallback cache for a compound."""
    cleaned = name_or_cid.strip().lower()
    if cleaned in _FALLBACK_COMPOUNDS:
        return _FALLBACK_COMPOUNDS[cleaned]

    # Check by CID
    if _is_cid(name_or_cid.strip()):
        cid = int(name_or_cid.strip())
        for entry in _FALLBACK_COMPOUNDS.values():
            if entry["cid"] == cid:
                return entry
    return None


def _compute_lipinski(
    mw: Optional[float],
    hbd: Optional[int],
    hba: Optional[int],
    xlogp: Optional[float],
) -> tuple:
    """Compute Lipinski Rule of 5 violations.

    Returns (n_violations, passes, violations_list).
    """
    violations: List[str] = []

    if mw is not None and mw >= 500:
        violations.append(f"MW={mw:.1f} >= 500")
    if hbd is not None and hbd > 5:
        violations.append(f"HBD={hbd} > 5")
    if hba is not None and hba > 10:
        violations.append(f"HBA={hba} > 10")
    if xlogp is not None and xlogp > 5:
        violations.append(f"XLogP={xlogp:.1f} > 5")

    n_violations = len(violations)
    passes = n_violations <= 1  # Ro5 allows 1 violation
    return n_violations, passes, violations


# ─── Core API Functions ───────────────────────────────────────────────────────

def fetch_compound_info(name_or_cid: str) -> Optional[CompoundRecord]:
    """Fetch compound information from PubChem by name or CID.

    Args:
        name_or_cid: Compound name (e.g., 'aspirin') or PubChem CID (e.g., '2244').

    Returns:
        CompoundRecord with full property data, or None if not found / network error.
    """
    ck = _cache_key("compound_info", name_or_cid)
    if ck in _cache:
        return _cache[ck]

    cleaned = name_or_cid.strip()

    # Build the API URL based on input type
    if _is_cid(cleaned):
        url = f"{PUBCHEM_BASE}/compound/cid/{cleaned}/JSON"
    else:
        encoded = urllib.parse.quote(cleaned)
        url = f"{PUBCHEM_BASE}/compound/name/{encoded}/JSON"

    data = _api_get(url)

    # Try fallback if API fails
    if data is None:
        fb = _get_fallback(cleaned)
        if fb is not None:
            record = CompoundRecord(
                cid=fb["cid"],
                iupac_name=fb.get("iupac_name"),
                molecular_formula=fb.get("molecular_formula"),
                molecular_weight=fb.get("molecular_weight"),
                canonical_smiles=fb.get("canonical_smiles"),
                inchi_key=fb.get("inchi_key"),
                xlogp=fb.get("xlogp"),
                tpsa=fb.get("tpsa"),
                hbd=fb.get("hbd"),
                hba=fb.get("hba"),
                rotatable_bonds=fb.get("rotatable_bonds"),
            )
            _cache[ck] = record
            return record
        return None

    # Parse the PubChem compound JSON
    compounds = data.get("PC_Compounds", [])
    if not compounds:
        return None

    compound = compounds[0]
    cid = compound.get("id", {}).get("id", {}).get("cid", 0)

    # Extract properties from the props array
    props: Dict[str, Any] = {}
    for prop in compound.get("props", []):
        urn = prop.get("urn", {})
        label = urn.get("label", "")
        name_field = urn.get("name", "")
        value = prop.get("value", {})

        # Extract the actual value (can be sval, ival, fval)
        val = value.get("sval") or value.get("fval") or value.get("ival")

        if label == "IUPAC Name" and name_field == "Preferred":
            props["iupac_name"] = val
        elif label == "Molecular Formula":
            props["molecular_formula"] = val
        elif label == "Molecular Weight":
            props["molecular_weight"] = _safe_float(val)
        elif label == "SMILES" and name_field == "Canonical":
            props["canonical_smiles"] = val
        elif label == "InChIKey":
            props["inchi_key"] = val
        elif label == "Log P" and name_field == "XLogP3":
            props["xlogp"] = _safe_float(val)
        elif label == "Topological" and name_field == "Polar Surface Area":
            props["tpsa"] = _safe_float(val)

    # Extract counts (reserved for future property API integration)
    # counts = compound.get("count", {})

    record = CompoundRecord(
        cid=cid,
        iupac_name=props.get("iupac_name"),
        molecular_formula=props.get("molecular_formula"),
        molecular_weight=props.get("molecular_weight"),
        canonical_smiles=props.get("canonical_smiles"),
        inchi_key=props.get("inchi_key"),
        xlogp=props.get("xlogp"),
        tpsa=props.get("tpsa"),
    )

    # Fetch HBD, HBA, rotatable bonds from the property endpoint (more reliable)
    prop_record = _fetch_property_data(name_or_cid)
    if prop_record:
        record.hbd = prop_record.get("hbd")
        record.hba = prop_record.get("hba")
        record.rotatable_bonds = prop_record.get("rotatable_bonds")
        # Fill in any missing top-level props
        if record.xlogp is None:
            record.xlogp = prop_record.get("xlogp")
        if record.tpsa is None:
            record.tpsa = prop_record.get("tpsa")
        if record.molecular_weight is None:
            record.molecular_weight = prop_record.get("molecular_weight")
        if record.molecular_formula is None:
            record.molecular_formula = prop_record.get("molecular_formula")

    _cache[ck] = record
    return record


def _fetch_property_data(name_or_cid: str) -> Optional[Dict[str, Any]]:
    """Fetch structured property data via the PubChem property endpoint.

    Returns a dict with keys: molecular_formula, molecular_weight, xlogp,
    tpsa, hbd, hba, rotatable_bonds, complexity.
    """
    cleaned = name_or_cid.strip()
    props_list = "MolecularFormula,MolecularWeight,XLogP,TPSA,HBondDonorCount,HBondAcceptorCount,RotatableBondCount,Complexity"

    if _is_cid(cleaned):
        url = f"{PUBCHEM_BASE}/compound/cid/{cleaned}/property/{props_list}/JSON"
    else:
        encoded = urllib.parse.quote(cleaned)
        url = f"{PUBCHEM_BASE}/compound/name/{encoded}/property/{props_list}/JSON"

    data = _api_get(url)
    if data is None:
        return None

    table = data.get("PropertyTable", {}).get("Properties", [])
    if not table:
        return None

    row = table[0]
    return {
        "molecular_formula": row.get("MolecularFormula"),
        "molecular_weight": _safe_float(row.get("MolecularWeight")),
        "xlogp": _safe_float(row.get("XLogP")),
        "tpsa": _safe_float(row.get("TPSA")),
        "hbd": _safe_int(row.get("HBondDonorCount")),
        "hba": _safe_int(row.get("HBondAcceptorCount")),
        "rotatable_bonds": _safe_int(row.get("RotatableBondCount")),
        "complexity": _safe_float(row.get("Complexity")),
    }


def fetch_compound_properties(name_or_cid: str) -> Optional[PropertyRecord]:
    """Fetch physicochemical properties with Lipinski Rule of 5 assessment.

    Args:
        name_or_cid: Compound name or PubChem CID.

    Returns:
        PropertyRecord with properties and Lipinski assessment, or None on failure.
    """
    ck = _cache_key("compound_props", name_or_cid)
    if ck in _cache:
        return _cache[ck]

    cleaned = name_or_cid.strip()
    prop_data = _fetch_property_data(cleaned)

    # Try fallback if API fails
    if prop_data is None:
        fb = _get_fallback(cleaned)
        if fb is not None:
            prop_data = {
                "molecular_formula": fb.get("molecular_formula"),
                "molecular_weight": fb.get("molecular_weight"),
                "xlogp": fb.get("xlogp"),
                "tpsa": fb.get("tpsa"),
                "hbd": fb.get("hbd"),
                "hba": fb.get("hba"),
                "rotatable_bonds": fb.get("rotatable_bonds"),
                "complexity": None,
            }
        else:
            return None

    mw = prop_data.get("molecular_weight")
    hbd = prop_data.get("hbd")
    hba = prop_data.get("hba")
    xlogp = prop_data.get("xlogp")

    n_viol, passes, _ = _compute_lipinski(mw, hbd, hba, xlogp)

    record = PropertyRecord(
        name=cleaned,
        molecular_formula=prop_data.get("molecular_formula"),
        molecular_weight=mw,
        xlogp=xlogp,
        tpsa=prop_data.get("tpsa"),
        hbd=hbd,
        hba=hba,
        rotatable_bonds=prop_data.get("rotatable_bonds"),
        complexity=prop_data.get("complexity"),
        lipinski_passes=passes,
        lipinski_violations=n_viol,
    )

    _cache[ck] = record
    return record


def check_lipinski(name_or_cid: str) -> Optional[LipinskiResult]:
    """Check Lipinski Rule of 5 drug-likeness for a compound.

    The Rule of 5 predicts poor absorption/permeation when:
    - Molecular weight >= 500 Da
    - H-bond donors > 5
    - H-bond acceptors > 10
    - XLogP > 5

    A compound with <= 1 violation is considered drug-like.

    Args:
        name_or_cid: Compound name or PubChem CID.

    Returns:
        LipinskiResult with pass/fail and violation details, or None on failure.
    """
    ck = _cache_key("lipinski", name_or_cid)
    if ck in _cache:
        return _cache[ck]

    cleaned = name_or_cid.strip()
    prop_data = _fetch_property_data(cleaned)

    # Try fallback if API fails
    if prop_data is None:
        fb = _get_fallback(cleaned)
        if fb is not None:
            prop_data = {
                "molecular_weight": fb.get("molecular_weight"),
                "xlogp": fb.get("xlogp"),
                "hbd": fb.get("hbd"),
                "hba": fb.get("hba"),
            }
        else:
            return None

    mw = prop_data.get("molecular_weight")
    hbd = prop_data.get("hbd")
    hba = prop_data.get("hba")
    xlogp = prop_data.get("xlogp")

    n_viol, passes, violations = _compute_lipinski(mw, hbd, hba, xlogp)

    result = LipinskiResult(
        name=cleaned,
        molecular_weight=mw,
        hbd=hbd,
        hba=hba,
        xlogp=xlogp,
        n_violations=n_viol,
        passes=passes,
        violations_list=violations,
    )

    _cache[ck] = result
    return result


def fetch_compound_safety(name_or_cid: str) -> Optional[SafetyRecord]:
    """Fetch GHS safety classification (hazard statements, signal word, pictograms).

    Args:
        name_or_cid: Compound name or PubChem CID.

    Returns:
        SafetyRecord with GHS data, or None if compound not found.
    """
    ck = _cache_key("safety", name_or_cid)
    if ck in _cache:
        return _cache[ck]

    cleaned = name_or_cid.strip()

    # Resolve to CID first (safety endpoint requires CID)
    cid = _resolve_cid(cleaned)
    if cid is None:
        return None

    url = f"{PUBCHEM_VIEW}/data/compound/{cid}/JSON?heading=GHS+Classification"
    data = _api_get(url)

    display_name = cleaned if not _is_cid(cleaned) else f"CID {cleaned}"

    if data is None:
        # Return a record with no safety data rather than None
        record = SafetyRecord(name=display_name, cid=cid)
        _cache[ck] = record
        return record

    # Parse GHS classification from the PUG View response
    signal_word = None
    hazard_statements: List[str] = []
    pictograms: List[str] = []

    def _walk_sections(sections: List[Dict]) -> None:
        nonlocal signal_word
        for section in sections:
            heading = section.get("TOCHeading", "")

            # Recurse into subsections
            if "Section" in section:
                _walk_sections(section["Section"])

            # Extract information blocks
            for info in section.get("Information", []):
                info_name = info.get("Name", "")
                value = info.get("Value", {})

                if "Signal" in heading or "Signal" in info_name:
                    strings = value.get("StringWithMarkup", [])
                    if strings:
                        signal_word = strings[0].get("String", "")

                elif "Hazard" in heading or "Hazard" in info_name:
                    strings = value.get("StringWithMarkup", [])
                    for s in strings:
                        stmt = s.get("String", "").strip()
                        if stmt and stmt not in hazard_statements:
                            hazard_statements.append(stmt)

                elif "Pictogram" in heading or "Pictogram" in info_name:
                    strings = value.get("StringWithMarkup", [])
                    for s in strings:
                        for markup in s.get("Markup", []):
                            extra = markup.get("Extra", "")
                            if extra and extra not in pictograms:
                                pictograms.append(extra)
                        # Also check the string itself
                        pict = s.get("String", "").strip()
                        if pict and pict not in pictograms:
                            pictograms.append(pict)

    record_data = data.get("Record", {})
    top_sections = record_data.get("Section", [])
    _walk_sections(top_sections)

    record = SafetyRecord(
        name=display_name,
        cid=cid,
        signal_word=signal_word,
        hazard_statements=hazard_statements,
        pictograms=pictograms,
    )

    _cache[ck] = record
    return record


def clear_cache() -> int:
    """Clear the session cache. Returns number of entries cleared."""
    n = len(_cache)
    _cache.clear()
    return n


# ─── CLI Demo ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    compounds = sys.argv[1:] if len(sys.argv) > 1 else ["aspirin", "ibuprofen", "atorvastatin"]

    for name in compounds:
        print(f"\n{'='*60}")
        print(f"  {name.upper()}")
        print(f"{'='*60}")

        info = fetch_compound_info(name)
        if info:
            print(f"\n  Compound Info:\n    {info}")
        else:
            print("\n  Compound Info: not found")

        props = fetch_compound_properties(name)
        if props:
            print(f"\n  Properties:\n    {props}")

        lip = check_lipinski(name)
        if lip:
            print(f"\n  Lipinski Ro5:\n    {lip}")

        safety = fetch_compound_safety(name)
        if safety:
            print(f"\n  Safety:\n    {safety}")
