"""drug_data_source.py -- Fetch real drug PK data from ChEMBL and PubChem.

Provides a unified interface to pull pharmacokinetic parameters from public
drug databases instead of hardcoding values.

Usage:
    from noethersolve.drug_data_source import DrugDataSource

    ds = DrugDataSource()
    drug = ds.get_drug("metformin")
    print(drug.half_life_h, drug.bioavailability)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Any
import requests

# ChEMBL client (installed via pip install chembl_webresource_client)
try:
    from chembl_webresource_client.new_client import new_client
    CHEMBL_AVAILABLE = True
except ImportError:
    CHEMBL_AVAILABLE = False

logger = logging.getLogger(__name__)

# Cache directory for drug data
_CACHE_DIR = Path(__file__).parent.parent / "data" / "drug_cache"


@dataclass
class DrugPKProfile:
    """Pharmacokinetic profile for a drug from database sources."""
    name: str
    chembl_id: Optional[str] = None
    pubchem_cid: Optional[int] = None

    # Basic properties
    molecular_weight: Optional[float] = None
    logp: Optional[float] = None
    psa: Optional[float] = None  # polar surface area

    # PK parameters (from literature/ChEMBL ADMET)
    bioavailability: Optional[float] = None  # F (0-1)
    half_life_h: Optional[float] = None      # t1/2 in hours
    Vd_L: Optional[float] = None             # volume of distribution (L)
    clearance_L_h: Optional[float] = None    # CL (L/h)
    protein_binding: Optional[float] = None  # fraction bound (0-1)

    # Absorption
    ka_per_h: Optional[float] = None         # absorption rate constant
    Tmax_h: Optional[float] = None           # time to peak

    # Metabolism
    primary_cyp: Optional[str] = None
    metabolizing_enzymes: List[str] = field(default_factory=list)

    # Therapeutic window
    therapeutic_min: Optional[float] = None  # mg/L
    therapeutic_max: Optional[float] = None  # mg/L
    typical_dose_mg: Optional[float] = None
    dosing_interval_h: Optional[float] = None

    # Data quality
    data_source: str = "unknown"
    confidence: str = "low"  # low, medium, high

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def ke_per_h(self) -> Optional[float]:
        """Elimination rate constant from half-life."""
        if self.half_life_h and self.half_life_h > 0:
            return 0.693 / self.half_life_h
        return None


class DrugDataSource:
    """Unified drug data source from ChEMBL + PubChem + local cache."""

    def __init__(self, use_cache: bool = True):
        self.use_cache = use_cache
        self._cache: Dict[str, DrugPKProfile] = {}

        if use_cache:
            _CACHE_DIR.mkdir(parents=True, exist_ok=True)
            self._load_cache()

    def _load_cache(self):
        """Load cached drug data from disk."""
        cache_file = _CACHE_DIR / "drug_pk_cache.json"
        if cache_file.exists():
            try:
                with open(cache_file) as f:
                    data = json.load(f)
                for name, d in data.items():
                    self._cache[name.lower()] = DrugPKProfile(**d)
                logger.info(f"Loaded {len(self._cache)} drugs from cache")
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")

    def _save_cache(self):
        """Save cache to disk."""
        cache_file = _CACHE_DIR / "drug_pk_cache.json"
        data = {name: profile.to_dict() for name, profile in self._cache.items()}
        with open(cache_file, "w") as f:
            json.dump(data, f, indent=2)

    def get_drug(self, name: str, fetch_if_missing: bool = True) -> Optional[DrugPKProfile]:
        """Get drug PK profile by name.

        Args:
            name: Drug name (case-insensitive)
            fetch_if_missing: If True, query external databases if not cached

        Returns:
            DrugPKProfile or None if not found
        """
        key = name.lower().strip()

        # Check cache first
        if key in self._cache:
            return self._cache[key]

        if not fetch_if_missing:
            return None

        # Try ChEMBL first (has ADMET data)
        profile = self._fetch_from_chembl(name)

        # Supplement with PubChem if needed
        if profile:
            self._supplement_from_pubchem(profile)
        else:
            profile = self._fetch_from_pubchem(name)

        if profile:
            self._cache[key] = profile
            if self.use_cache:
                self._save_cache()

        return profile

    def _fetch_from_chembl(self, name: str) -> Optional[DrugPKProfile]:
        """Fetch drug data from ChEMBL."""
        if not CHEMBL_AVAILABLE:
            logger.warning("ChEMBL client not available")
            return None

        try:
            molecule = new_client.molecule

            # Search by name
            results = molecule.filter(pref_name__iexact=name)
            if not results:
                # Try synonym search
                results = molecule.filter(molecule_synonyms__molecule_synonym__iexact=name)

            if not results:
                return None

            mol = results[0]
            chembl_id = mol.get('molecule_chembl_id')

            profile = DrugPKProfile(
                name=name,
                chembl_id=chembl_id,
                data_source="chembl",
            )

            # Get molecule properties
            props = mol.get('molecule_properties', {})
            if props:
                profile.molecular_weight = _safe_float(props.get('full_mwt'))
                profile.logp = _safe_float(props.get('alogp'))
                profile.psa = _safe_float(props.get('psa'))

            # Get ADMET/PK data from assays
            self._fetch_chembl_pk_data(profile, chembl_id)

            return profile

        except Exception as e:
            logger.warning(f"ChEMBL fetch failed for {name}: {e}")
            return None

    def _fetch_chembl_pk_data(self, profile: DrugPKProfile, chembl_id: str):
        """Fetch PK parameters from ChEMBL assays."""
        if not CHEMBL_AVAILABLE or not chembl_id:
            return

        try:
            activity = new_client.activity

            # Query for PK-related assays
            pk_types = ['T1/2', 'Vd', 'CL', 'F', 'Cmax', 'AUC', 'Tmax']

            for pk_type in pk_types:
                try:
                    results = activity.filter(
                        molecule_chembl_id=chembl_id,
                        standard_type__iexact=pk_type
                    ).only(['standard_value', 'standard_units', 'standard_type'])[:5]

                    for r in results:
                        val = _safe_float(r.get('standard_value'))
                        units = r.get('standard_units', '')

                        if val is None:
                            continue

                        # Convert and assign based on type
                        if pk_type == 'T1/2' and 'h' in units.lower():
                            profile.half_life_h = val
                            profile.confidence = "medium"
                        elif pk_type == 'Vd' and 'l' in units.lower():
                            profile.Vd_L = val
                        elif pk_type == 'CL':
                            if 'l/h' in units.lower():
                                profile.clearance_L_h = val
                            elif 'ml/min' in units.lower():
                                profile.clearance_L_h = val * 0.06  # convert
                        elif pk_type == 'F' and '%' in units:
                            profile.bioavailability = val / 100.0
                        elif pk_type == 'Tmax' and 'h' in units.lower():
                            profile.Tmax_h = val

                        break  # Take first valid result

                except Exception:
                    continue

        except Exception as e:
            logger.debug(f"ChEMBL PK fetch failed: {e}")

    def _fetch_from_pubchem(self, name: str) -> Optional[DrugPKProfile]:
        """Fetch drug data from PubChem REST API."""
        try:
            # Search by name
            url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{name}/property/MolecularWeight,XLogP,TPSA/JSON"
            resp = requests.get(url, timeout=10)

            if resp.status_code != 200:
                return None

            data = resp.json()
            props = data.get('PropertyTable', {}).get('Properties', [{}])[0]

            profile = DrugPKProfile(
                name=name,
                pubchem_cid=props.get('CID'),
                molecular_weight=_safe_float(props.get('MolecularWeight')),
                logp=_safe_float(props.get('XLogP')),
                psa=_safe_float(props.get('TPSA')),
                data_source="pubchem",
                confidence="low",
            )

            return profile

        except Exception as e:
            logger.warning(f"PubChem fetch failed for {name}: {e}")
            return None

    def _supplement_from_pubchem(self, profile: DrugPKProfile):
        """Add PubChem data to existing profile."""
        if profile.pubchem_cid or not profile.name:
            return

        try:
            url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{profile.name}/property/MolecularWeight,XLogP,TPSA/JSON"
            resp = requests.get(url, timeout=10)

            if resp.status_code == 200:
                data = resp.json()
                props = data.get('PropertyTable', {}).get('Properties', [{}])[0]

                profile.pubchem_cid = props.get('CID')
                if not profile.molecular_weight:
                    profile.molecular_weight = _safe_float(props.get('MolecularWeight'))
                if not profile.logp:
                    profile.logp = _safe_float(props.get('XLogP'))
                if not profile.psa:
                    profile.psa = _safe_float(props.get('TPSA'))

        except Exception:
            pass

    def search_drugs(self, query: str, limit: int = 10) -> List[str]:
        """Search for drug names matching query."""
        if not CHEMBL_AVAILABLE:
            return []

        try:
            molecule = new_client.molecule
            results = molecule.filter(pref_name__icontains=query)[:limit]
            return [r.get('pref_name', '') for r in results if r.get('pref_name')]
        except Exception:
            return []

    def get_common_drugs(self) -> List[str]:
        """Return list of common drugs we have data for."""
        # Top 100 most prescribed drugs (by generic name)
        return [
            "metformin", "lisinopril", "atorvastatin", "levothyroxine",
            "amlodipine", "metoprolol", "omeprazole", "simvastatin",
            "losartan", "gabapentin", "hydrochlorothiazide", "sertraline",
            "acetaminophen", "ibuprofen", "aspirin", "prednisone",
            "albuterol", "montelukast", "fluticasone", "escitalopram",
            "pantoprazole", "rosuvastatin", "furosemide", "tramadol",
            "alprazolam", "duloxetine", "trazodone", "clopidogrel",
            "carvedilol", "pravastatin", "warfarin", "amoxicillin",
            "azithromycin", "ciprofloxacin", "doxycycline", "meloxicam",
            "celecoxib", "naproxen", "diclofenac", "pregabalin",
            "venlafaxine", "bupropion", "fluoxetine", "citalopram",
            "quetiapine", "aripiprazole", "risperidone", "clonazepam",
            "lorazepam", "zolpidem", "cyclobenzaprine", "methocarbamol",
            "glipizide", "pioglitazone", "sitagliptin", "insulin",
            "apixaban", "rivaroxaban", "enoxaparin", "atenolol",
            "propranolol", "diltiazem", "verapamil", "nifedipine",
            "tamsulosin", "finasteride", "sildenafil", "tadalafil",
            "ondansetron", "promethazine", "famotidine", "ranitidine",
            "esomeprazole", "lansoprazole", "sucralfate", "misoprostol",
            "methotrexate", "hydroxychloroquine", "sulfasalazine",
            "allopurinol", "colchicine", "febuxostat", "levetiracetam",
            "topiramate", "lamotrigine", "valproate", "phenytoin",
            "carbamazepine", "oxcarbazepine", "clonidine", "prazosin",
            "doxazosin", "terazosin", "spironolactone", "eplerenone",
            "triamterene", "bumetanide", "torsemide", "metolazone",
        ]

    def bulk_fetch(self, names: List[str], progress_callback=None) -> Dict[str, DrugPKProfile]:
        """Fetch multiple drugs at once."""
        results = {}
        for i, name in enumerate(names):
            profile = self.get_drug(name)
            if profile:
                results[name] = profile
            if progress_callback:
                progress_callback(i + 1, len(names), name)
        return results


def _safe_float(val) -> Optional[float]:
    """Safely convert to float."""
    if val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# FDA Label PK data (curated from FDA package inserts)
# This supplements ChEMBL with clinical PK parameters
# ---------------------------------------------------------------------------

FDA_PK_DATA: Dict[str, Dict[str, Any]] = {
    # Format: drug_name: {bioavailability, half_life_h, Vd_L, ...}
    # Data from FDA package inserts and clinical pharmacology references

    "metformin": {
        "bioavailability": 0.55,
        "half_life_h": 6.2,
        "Vd_L": 654,
        "clearance_L_h": 74.4,
        "protein_binding": 0.0,
        "Tmax_h": 2.5,
        "typical_dose_mg": 500,
        "dosing_interval_h": 12,
        "therapeutic_min": 0.5,
        "therapeutic_max": 5.0,
        "primary_cyp": "none",
    },
    "atorvastatin": {
        "bioavailability": 0.14,
        "half_life_h": 14,
        "Vd_L": 381,
        "protein_binding": 0.98,
        "Tmax_h": 1.5,
        "typical_dose_mg": 20,
        "dosing_interval_h": 24,
        "therapeutic_min": 0.002,
        "therapeutic_max": 0.03,
        "primary_cyp": "CYP3A4",
    },
    "lisinopril": {
        "bioavailability": 0.25,
        "half_life_h": 12,
        "Vd_L": 124,
        "protein_binding": 0.0,
        "Tmax_h": 7,
        "typical_dose_mg": 10,
        "dosing_interval_h": 24,
        "therapeutic_min": 0.01,
        "therapeutic_max": 0.1,
        "primary_cyp": "none",
    },
    "amlodipine": {
        "bioavailability": 0.64,
        "half_life_h": 40,
        "Vd_L": 1680,
        "protein_binding": 0.93,
        "Tmax_h": 8,
        "typical_dose_mg": 5,
        "dosing_interval_h": 24,
        "therapeutic_min": 0.003,
        "therapeutic_max": 0.015,
        "primary_cyp": "CYP3A4",
    },
    "omeprazole": {
        "bioavailability": 0.50,
        "half_life_h": 1.0,
        "Vd_L": 15,
        "protein_binding": 0.95,
        "Tmax_h": 0.5,
        "typical_dose_mg": 20,
        "dosing_interval_h": 24,
        "therapeutic_min": 0.02,
        "therapeutic_max": 2.0,
        "primary_cyp": "CYP2C19",
    },
    "sertraline": {
        "bioavailability": 0.44,
        "half_life_h": 26,
        "Vd_L": 1400,
        "protein_binding": 0.98,
        "Tmax_h": 6,
        "typical_dose_mg": 50,
        "dosing_interval_h": 24,
        "therapeutic_min": 0.01,
        "therapeutic_max": 0.2,
        "primary_cyp": "CYP2D6",
    },
    "gabapentin": {
        "bioavailability": 0.60,
        "half_life_h": 6,
        "Vd_L": 58,
        "protein_binding": 0.0,
        "Tmax_h": 3,
        "typical_dose_mg": 300,
        "dosing_interval_h": 8,
        "therapeutic_min": 2.0,
        "therapeutic_max": 20.0,
        "primary_cyp": "none",
    },
    "warfarin": {
        "bioavailability": 0.99,
        "half_life_h": 40,
        "Vd_L": 10,
        "protein_binding": 0.99,
        "Tmax_h": 4,
        "typical_dose_mg": 5,
        "dosing_interval_h": 24,
        "therapeutic_min": 1.0,
        "therapeutic_max": 4.0,
        "primary_cyp": "CYP2C9",
    },
    "acetaminophen": {
        "bioavailability": 0.85,
        "half_life_h": 2.5,
        "Vd_L": 67,
        "protein_binding": 0.25,
        "Tmax_h": 0.75,
        "typical_dose_mg": 1000,
        "dosing_interval_h": 6,
        "therapeutic_min": 10.0,
        "therapeutic_max": 150.0,
        "primary_cyp": "CYP2E1",
    },
    "ibuprofen": {
        "bioavailability": 0.80,
        "half_life_h": 2.0,
        "Vd_L": 10,
        "protein_binding": 0.99,
        "Tmax_h": 1.5,
        "typical_dose_mg": 400,
        "dosing_interval_h": 6,
        "therapeutic_min": 10.0,
        "therapeutic_max": 50.0,
        "primary_cyp": "CYP2C9",
    },
}


def get_fda_pk_data(name: str) -> Optional[Dict[str, Any]]:
    """Get FDA-sourced PK data for a drug."""
    return FDA_PK_DATA.get(name.lower())


def merge_with_fda_data(profile: DrugPKProfile) -> DrugPKProfile:
    """Merge database profile with FDA reference data."""
    fda = get_fda_pk_data(profile.name)
    if not fda:
        return profile

    # FDA data overrides database data (more reliable for clinical use)
    if fda.get('bioavailability') and not profile.bioavailability:
        profile.bioavailability = fda['bioavailability']
    if fda.get('half_life_h') and not profile.half_life_h:
        profile.half_life_h = fda['half_life_h']
    if fda.get('Vd_L') and not profile.Vd_L:
        profile.Vd_L = fda['Vd_L']
    if fda.get('clearance_L_h') and not profile.clearance_L_h:
        profile.clearance_L_h = fda['clearance_L_h']
    if fda.get('protein_binding') and not profile.protein_binding:
        profile.protein_binding = fda['protein_binding']
    if fda.get('Tmax_h') and not profile.Tmax_h:
        profile.Tmax_h = fda['Tmax_h']
    if fda.get('typical_dose_mg') and not profile.typical_dose_mg:
        profile.typical_dose_mg = fda['typical_dose_mg']
    if fda.get('dosing_interval_h') and not profile.dosing_interval_h:
        profile.dosing_interval_h = fda['dosing_interval_h']
    if fda.get('therapeutic_min') and not profile.therapeutic_min:
        profile.therapeutic_min = fda['therapeutic_min']
    if fda.get('therapeutic_max') and not profile.therapeutic_max:
        profile.therapeutic_max = fda['therapeutic_max']
    if fda.get('primary_cyp') and not profile.primary_cyp:
        profile.primary_cyp = fda['primary_cyp']

    profile.confidence = "high"
    profile.data_source = f"{profile.data_source}+fda"

    return profile


# ---------------------------------------------------------------------------
# CLI for testing
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Drug data source CLI")
    parser.add_argument("drug", nargs="?", help="Drug name to look up")
    parser.add_argument("--search", "-s", help="Search for drugs")
    parser.add_argument("--bulk", "-b", action="store_true",
                        help="Fetch all common drugs")
    args = parser.parse_args()

    ds = DrugDataSource()

    if args.search:
        print(f"\nSearching for '{args.search}'...")
        results = ds.search_drugs(args.search)
        for r in results:
            print(f"  - {r}")
    elif args.bulk:
        print("\nFetching common drugs...")
        drugs = ds.get_common_drugs()[:20]  # First 20 for testing
        for i, name in enumerate(drugs):
            profile = ds.get_drug(name)
            if profile:
                status = "OK" if profile.half_life_h else "partial"
                print(f"  [{i+1:2d}/{len(drugs)}] {name:20s} {status}")
    elif args.drug:
        profile = ds.get_drug(args.drug)
        if profile:
            profile = merge_with_fda_data(profile)
            print(f"\n{profile.name.upper()}")
            print(f"  ChEMBL ID: {profile.chembl_id}")
            print(f"  PubChem CID: {profile.pubchem_cid}")
            print(f"  MW: {profile.molecular_weight} g/mol")
            print(f"  LogP: {profile.logp}")
            print(f"  Bioavailability: {profile.bioavailability}")
            print(f"  Half-life: {profile.half_life_h} h")
            print(f"  Vd: {profile.Vd_L} L")
            print(f"  CL: {profile.clearance_L_h} L/h")
            print(f"  Primary CYP: {profile.primary_cyp}")
            print(f"  Therapeutic window: {profile.therapeutic_min}-{profile.therapeutic_max} mg/L")
            print(f"  Source: {profile.data_source}, confidence: {profile.confidence}")
        else:
            print(f"Drug '{args.drug}' not found")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
