"""noethersolve.uniprot_api — UniProt REST API client for protein data.

Connects to the UniProt REST API (https://rest.uniprot.org) to fetch
protein information, function annotations, variant data, and interaction
partners. Uses only urllib (no external dependencies).

Complements the static aggregation/pipeline modules by providing live
lookup of protein annotations from UniProt's curated database of 250M+
sequences.

Usage:
    from noethersolve.uniprot_api import (
        fetch_protein_info, fetch_protein_function,
        fetch_protein_variants, fetch_protein_interactions,
    )

    # Look up a protein
    info = fetch_protein_info("TP53")
    print(info)  # ProteinRecord(gene='TP53', accession='P04637', ...)

    # Get function annotation
    func = fetch_protein_function("BRCA1")
    print(func)  # FunctionRecord(gene='BRCA1', function_text='...')

    # Get known variants
    variants = fetch_protein_variants("CFTR")
    for v in variants:
        print(f"{v.position} {v.original_aa}->{v.variant_aa}: {v.description}")

    # Get interaction partners
    partners = fetch_protein_interactions("EGFR")
    for p in partners:
        print(f"{p.partner_gene}: {p.interaction_type}")
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

UNIPROT_BASE = "https://rest.uniprot.org"
REQUEST_TIMEOUT = 15  # seconds

# ─── Session Cache ────────────────────────────────────────────────────────────

_cache: Dict[str, Any] = {}


def _cache_key(prefix: str, *args: str) -> str:
    """Build a deterministic cache key."""
    return f"{prefix}:{'|'.join(a.lower().strip() for a in args)}"


# ─── Local Fallback Cache ────────────────────────────────────────────────────

# Minimal cached data for common proteins, used when the API is unreachable.
# Each entry: accession, protein_name, sequence_length, function_text, subcellular_location
_FALLBACK_CACHE: Dict[str, Dict[str, Any]] = {
    "TP53": {
        "accession": "P04637",
        "protein_name": "Cellular tumor antigen p53",
        "sequence_length": 393,
        "function_text": (
            "Acts as a tumor suppressor in many tumor types; induces growth "
            "arrest or apoptosis depending on the physiological circumstances "
            "and cell type. Involved in cell cycle regulation as a "
            "trans-activator that acts to negatively regulate cell division."
        ),
        "subcellular_location": "Nucleus, Cytoplasm",
        "go_terms": [
            "GO:0005634 nucleus",
            "GO:0006915 apoptotic process",
            "GO:0006355 regulation of transcription",
            "GO:0003700 DNA-binding transcription factor activity",
            "GO:0042802 identical protein binding",
        ],
        "diseases": [
            "Li-Fraumeni syndrome (LFS)",
            "Hepatocellular carcinoma (HCC)",
            "Choroid plexus papilloma (CPP)",
        ],
    },
    "BRCA1": {
        "accession": "P38398",
        "protein_name": "Breast cancer type 1 susceptibility protein",
        "sequence_length": 1863,
        "function_text": (
            "E3 ubiquitin-protein ligase that plays a central role in DNA "
            "repair by facilitating cellular responses to DNA damage. Required "
            "for appropriate cell cycle arrests after ionizing irradiation in "
            "both the S-phase and the G2 phase of the cell cycle."
        ),
        "subcellular_location": "Nucleus",
        "go_terms": [
            "GO:0005634 nucleus",
            "GO:0006281 DNA repair",
            "GO:0006974 cellular response to DNA damage stimulus",
            "GO:0004842 ubiquitin-protein transferase activity",
            "GO:0005515 protein binding",
        ],
        "diseases": [
            "Breast-ovarian cancer, familial, 1 (BROVCA1)",
            "Fanconi anemia complementation group S (FANCS)",
        ],
    },
    "EGFR": {
        "accession": "P00533",
        "protein_name": "Epidermal growth factor receptor",
        "sequence_length": 1210,
        "function_text": (
            "Receptor tyrosine kinase binding ligands of the EGF family and "
            "activating several signaling cascades to convert extracellular "
            "cues into appropriate cellular responses. Involved in cell "
            "proliferation, differentiation, and survival."
        ),
        "subcellular_location": "Cell membrane, Nucleus",
        "go_terms": [
            "GO:0005886 plasma membrane",
            "GO:0004716 receptor signaling protein tyrosine kinase activity",
            "GO:0008283 cell population proliferation",
            "GO:0007169 transmembrane receptor protein tyrosine kinase signaling pathway",
            "GO:0005524 ATP binding",
        ],
        "diseases": [
            "Lung cancer (LC)",
            "Inflammatory skin and bowel disease, neonatal, 2 (NISBD2)",
        ],
    },
    "INS": {
        "accession": "P01308",
        "protein_name": "Insulin",
        "sequence_length": 110,
        "function_text": (
            "Insulin decreases blood glucose concentration. It increases cell "
            "permeability to monosaccharides, amino acids and fatty acids. It "
            "accelerates glycolysis, the pentose phosphate cycle, and glycogen "
            "synthesis in liver."
        ),
        "subcellular_location": "Secreted",
        "go_terms": [
            "GO:0005576 extracellular region",
            "GO:0005159 insulin-like growth factor receptor binding",
            "GO:0046326 positive regulation of glucose import",
            "GO:0045721 negative regulation of gluconeogenesis",
            "GO:0005102 signaling receptor binding",
        ],
        "diseases": [
            "Diabetes mellitus, permanent neonatal (PNDM)",
            "Hyperproinsulinemia",
            "Maturity-onset diabetes of the young 10 (MODY10)",
        ],
    },
    "HBB": {
        "accession": "P68871",
        "protein_name": "Hemoglobin subunit beta",
        "sequence_length": 147,
        "function_text": (
            "Involved in oxygen transport from the lung to the various "
            "peripheral tissues. Hemoglobin binds oxygen in a cooperative "
            "manner with the transition between deoxy and oxy conformations "
            "described by the two-state MWC model."
        ),
        "subcellular_location": "Cytoplasm",
        "go_terms": [
            "GO:0005833 hemoglobin complex",
            "GO:0005344 oxygen carrier activity",
            "GO:0015671 oxygen transport",
            "GO:0019825 oxygen binding",
            "GO:0020037 heme binding",
        ],
        "diseases": [
            "Sickle cell disease",
            "Beta-thalassemia",
            "Heinz body anemia",
        ],
    },
    "ACE2": {
        "accession": "Q9BYF1",
        "protein_name": "Angiotensin-converting enzyme 2",
        "sequence_length": 805,
        "function_text": (
            "Carboxypeptidase which converts angiotensin II to angiotensin 1-7 "
            "and angiotensin I to angiotensin 1-9. Also functions as the "
            "receptor for SARS-CoV and SARS-CoV-2 spike glycoproteins, "
            "mediating viral entry into cells."
        ),
        "subcellular_location": "Cell membrane, Secreted",
        "go_terms": [
            "GO:0005886 plasma membrane",
            "GO:0004180 carboxypeptidase activity",
            "GO:0008241 peptidyl-dipeptidase activity",
            "GO:0003823 antigen binding",
            "GO:0046872 metal ion binding",
        ],
        "diseases": [
            "COVID-19 susceptibility (SARS-CoV-2 receptor)",
        ],
    },
    "CFTR": {
        "accession": "P13569",
        "protein_name": "Cystic fibrosis transmembrane conductance regulator",
        "sequence_length": 1480,
        "function_text": (
            "Chloride channel involved in regulating epithelial ion and water "
            "transport and fluid homeostasis. Conducts chloride ions across "
            "the cell membrane following activation by cAMP-dependent "
            "phosphorylation."
        ),
        "subcellular_location": "Cell membrane, Apical cell membrane",
        "go_terms": [
            "GO:0005886 plasma membrane",
            "GO:0005254 chloride channel activity",
            "GO:0006821 chloride transport",
            "GO:0005524 ATP binding",
            "GO:0043855 cyclic nucleotide-gated ion channel activity",
        ],
        "diseases": [
            "Cystic fibrosis (CF)",
            "Congenital bilateral absence of vas deferens (CBAVD)",
        ],
    },
    "HTT": {
        "accession": "P42858",
        "protein_name": "Huntingtin",
        "sequence_length": 3144,
        "function_text": (
            "May play a role in microtubule-mediated transport or vesicle "
            "function. Required for normal development. Involved in the "
            "regulation of BDNF transcription and axonal transport of BDNF."
        ),
        "subcellular_location": "Cytoplasm, Nucleus, Cell projection",
        "go_terms": [
            "GO:0005737 cytoplasm",
            "GO:0005634 nucleus",
            "GO:0005515 protein binding",
            "GO:0007420 brain development",
            "GO:0006888 endoplasmic reticulum to Golgi vesicle-mediated transport",
        ],
        "diseases": [
            "Huntington disease (HD)",
        ],
    },
}


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
                "User-Agent": "NoetherSolve/1.0 (research tool; https://github.com/SolomonB14D3/noethersolve)",
            },
        )
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            _cache[url] = data
            return data
    except urllib.error.HTTPError as e:
        logger.warning("UniProt HTTP error %d for %s", e.code, url)
        return None
    except urllib.error.URLError as e:
        logger.warning("UniProt network error for %s: %s", url, e.reason)
        return None
    except (json.JSONDecodeError, OSError, TimeoutError) as e:
        logger.warning("UniProt request failed for %s: %s", url, e)
        return None


# ─── Dataclasses ──────────────────────────────────────────────────────────────

@dataclass
class ProteinRecord:
    """Core protein information from UniProt."""
    accession: str
    gene: str
    protein_name: str
    sequence_length: int = 0
    subcellular_location: str = ""
    function_text: str = ""
    go_terms: List[str] = field(default_factory=list)
    diseases: List[str] = field(default_factory=list)
    from_cache: bool = False

    def __str__(self) -> str:
        parts = [f"{self.gene} ({self.accession})"]
        parts.append(f"Name: {self.protein_name}")
        if self.sequence_length > 0:
            parts.append(f"Length: {self.sequence_length} aa")
        if self.subcellular_location:
            parts.append(f"Location: {self.subcellular_location}")
        if self.function_text:
            func_short = self.function_text[:200]
            if len(self.function_text) > 200:
                func_short += "..."
            parts.append(f"Function: {func_short}")
        if self.go_terms:
            parts.append(f"GO terms: {', '.join(self.go_terms[:5])}")
        if self.diseases:
            parts.append(f"Diseases: {', '.join(self.diseases[:5])}")
        if self.from_cache:
            parts.append("[from local cache - API unreachable]")
        return "\n  ".join(parts)


@dataclass
class FunctionRecord:
    """Detailed function annotation from UniProt."""
    gene: str
    function_text: str = ""
    catalytic_activity: List[str] = field(default_factory=list)
    pathway: str = ""
    tissue_specificity: str = ""
    from_cache: bool = False

    def __str__(self) -> str:
        parts = [f"{self.gene} — Function"]
        if self.function_text:
            parts.append(f"Function: {self.function_text}")
        if self.catalytic_activity:
            parts.append(f"Catalytic activity: {'; '.join(self.catalytic_activity)}")
        if self.pathway:
            parts.append(f"Pathway: {self.pathway}")
        if self.tissue_specificity:
            parts.append(f"Tissue specificity: {self.tissue_specificity}")
        if self.from_cache:
            parts.append("[from local cache - API unreachable]")
        return "\n  ".join(parts)


@dataclass
class VariantRecord:
    """Protein variant/mutation annotation from UniProt."""
    gene: str
    position: int
    original_aa: str = ""
    variant_aa: str = ""
    description: str = ""
    clinical_significance: str = ""

    def __str__(self) -> str:
        mutation = f"{self.original_aa}{self.position}{self.variant_aa}"
        parts = [f"{self.gene} {mutation}"]
        if self.description:
            parts.append(self.description)
        if self.clinical_significance:
            parts.append(f"[{self.clinical_significance}]")
        return " | ".join(parts)


@dataclass
class InteractionRecord:
    """Protein-protein interaction from UniProt."""
    gene: str
    partner_gene: str
    interaction_type: str = "physical"
    experiments_count: int = 0

    def __str__(self) -> str:
        parts = [f"{self.gene} <-> {self.partner_gene}"]
        parts.append(f"type={self.interaction_type}")
        if self.experiments_count > 0:
            parts.append(f"experiments={self.experiments_count}")
        return " | ".join(parts)


# ─── Internal Helpers ─────────────────────────────────────────────────────────

def _safe_str(val: Any, default: str = "") -> str:
    """Safely convert a value to string, returning default on failure."""
    if val is None:
        return default
    try:
        return str(val)
    except (ValueError, TypeError):
        return default


def _safe_int(val: Any, default: int = 0) -> int:
    """Safely convert a value to int, returning default on failure."""
    if val is None:
        return default
    try:
        return int(val)
    except (ValueError, TypeError):
        return default


def _search_uniprot(query: str, fields: str = "") -> Optional[Dict]:
    """Search UniProt for a human protein by gene name or accession.

    Args:
        query: Gene name (e.g., 'TP53') or UniProt accession (e.g., 'P04637').
        fields: Comma-separated field list, or empty for default fields.

    Returns:
        First matching entry dict, or None.
    """
    encoded = urllib.parse.quote(query.strip())
    url = (
        f"{UNIPROT_BASE}/uniprotkb/search?"
        f"query={encoded}+AND+organism_id:9606"
        f"&format=json&size=1"
    )
    if fields:
        url += f"&fields={fields}"

    data = _api_get(url)
    if data is None or not data.get("results"):
        return None
    return data["results"][0]


def _extract_comments(entry: Dict, comment_type: str) -> List[Dict]:
    """Extract comment annotations of a specific type from a UniProt entry."""
    comments = entry.get("comments", [])
    return [c for c in comments if c.get("commentType", "").upper() == comment_type.upper()]


def _extract_comment_text(entry: Dict, comment_type: str) -> str:
    """Extract the text value from comments of a given type."""
    matches = _extract_comments(entry, comment_type)
    texts = []
    for c in matches:
        # Some comments have texts as list of dicts with 'value' key
        for t in c.get("texts", []):
            val = t.get("value", "")
            if val:
                texts.append(val)
    return " ".join(texts)


def _extract_go_terms(entry: Dict, limit: int = 5) -> List[str]:
    """Extract GO term annotations from cross-references."""
    refs = entry.get("uniProtKBCrossReferences", [])
    go_terms = []
    for ref in refs:
        if ref.get("database") == "GO":
            go_id = ref.get("id", "")
            # Extract the term name from properties
            props = ref.get("properties", [])
            term_name = ""
            for p in props:
                if p.get("key") == "GoTerm":
                    term_name = p.get("value", "")
                    break
            if go_id:
                label = f"{go_id} {term_name}" if term_name else go_id
                go_terms.append(label)
            if len(go_terms) >= limit:
                break
    return go_terms


def _extract_diseases(entry: Dict) -> List[str]:
    """Extract disease associations from comment annotations."""
    disease_comments = _extract_comments(entry, "DISEASE")
    diseases = []
    for c in disease_comments:
        disease = c.get("disease", {})
        name = disease.get("diseaseId", "")
        if name:
            diseases.append(name)
    return diseases


def _extract_subcellular_location(entry: Dict) -> str:
    """Extract subcellular location from comment annotations."""
    loc_comments = _extract_comments(entry, "SUBCELLULAR LOCATION")
    locations = []
    for c in loc_comments:
        for sl in c.get("subcellularLocations", []):
            loc = sl.get("location", {})
            val = loc.get("value", "")
            if val and val not in locations:
                locations.append(val)
    return ", ".join(locations) if locations else ""


def _extract_gene_name(entry: Dict) -> str:
    """Extract the primary gene name from a UniProt entry."""
    genes = entry.get("genes", [])
    if genes:
        primary = genes[0].get("geneName", {})
        return primary.get("value", "")
    return ""


def _extract_protein_name(entry: Dict) -> str:
    """Extract the recommended protein name from a UniProt entry."""
    prot_desc = entry.get("proteinDescription", {})
    rec_name = prot_desc.get("recommendedName", {})
    full_name = rec_name.get("fullName", {})
    name = full_name.get("value", "")
    if not name:
        # Fall back to submittedName
        sub_names = prot_desc.get("submissionNames", [])
        if sub_names:
            name = sub_names[0].get("fullName", {}).get("value", "")
    return name


def _extract_features(entry: Dict, feature_type: str) -> List[Dict]:
    """Extract features of a specific type from a UniProt entry."""
    features = entry.get("features", [])
    return [f for f in features if f.get("type", "").upper() == feature_type.upper()]


def _get_fallback(gene: str) -> Optional[Dict]:
    """Look up a gene in the local fallback cache."""
    gene_upper = gene.strip().upper()
    return _FALLBACK_CACHE.get(gene_upper)


# ─── Core API Functions ───────────────────────────────────────────────────────

def fetch_protein_info(gene_or_accession: str) -> Optional[ProteinRecord]:
    """Search UniProt for a human protein and return core information.

    Args:
        gene_or_accession: Gene name (e.g., 'TP53') or UniProt accession
                          (e.g., 'P04637').

    Returns:
        ProteinRecord with protein info, or None if not found.
        Falls back to local cache for common proteins when API is unreachable.
    """
    ck = _cache_key("protein_info", gene_or_accession)
    if ck in _cache:
        return _cache[ck]

    entry = _search_uniprot(gene_or_accession)

    if entry is None:
        # Try local fallback
        fb = _get_fallback(gene_or_accession)
        if fb is None:
            return None
        record = ProteinRecord(
            accession=fb["accession"],
            gene=gene_or_accession.strip().upper(),
            protein_name=fb["protein_name"],
            sequence_length=fb["sequence_length"],
            subcellular_location=fb.get("subcellular_location", ""),
            function_text=fb.get("function_text", ""),
            go_terms=fb.get("go_terms", []),
            diseases=fb.get("diseases", []),
            from_cache=True,
        )
        _cache[ck] = record
        return record

    # Extract fields from the live API response
    accession = entry.get("primaryAccession", "")
    gene = _extract_gene_name(entry) or gene_or_accession.strip().upper()
    protein_name = _extract_protein_name(entry)
    seq_length = entry.get("sequence", {}).get("length", 0)
    subcellular = _extract_subcellular_location(entry)
    function_text = _extract_comment_text(entry, "FUNCTION")
    go_terms = _extract_go_terms(entry, limit=5)
    diseases = _extract_diseases(entry)

    record = ProteinRecord(
        accession=accession,
        gene=gene,
        protein_name=protein_name,
        sequence_length=_safe_int(seq_length),
        subcellular_location=subcellular,
        function_text=function_text,
        go_terms=go_terms,
        diseases=diseases,
    )
    _cache[ck] = record
    return record


def fetch_protein_function(gene_or_accession: str) -> Optional[FunctionRecord]:
    """Fetch detailed function annotation for a human protein from UniProt.

    Args:
        gene_or_accession: Gene name or UniProt accession.

    Returns:
        FunctionRecord with function details, or None if not found.
    """
    ck = _cache_key("protein_function", gene_or_accession)
    if ck in _cache:
        return _cache[ck]

    entry = _search_uniprot(gene_or_accession)

    if entry is None:
        # Try local fallback (limited data)
        fb = _get_fallback(gene_or_accession)
        if fb is None:
            return None
        record = FunctionRecord(
            gene=gene_or_accession.strip().upper(),
            function_text=fb.get("function_text", ""),
            from_cache=True,
        )
        _cache[ck] = record
        return record

    gene = _extract_gene_name(entry) or gene_or_accession.strip().upper()
    function_text = _extract_comment_text(entry, "FUNCTION")

    # Extract catalytic activity
    cat_comments = _extract_comments(entry, "CATALYTIC ACTIVITY")
    catalytic_activities = []
    for c in cat_comments:
        reaction = c.get("reaction", {})
        name = reaction.get("name", "")
        if name:
            catalytic_activities.append(name)

    # Extract pathway
    pathway_text = _extract_comment_text(entry, "PATHWAY")

    # Extract tissue specificity
    tissue_text = _extract_comment_text(entry, "TISSUE SPECIFICITY")

    record = FunctionRecord(
        gene=gene,
        function_text=function_text,
        catalytic_activity=catalytic_activities,
        pathway=pathway_text,
        tissue_specificity=tissue_text,
    )
    _cache[ck] = record
    return record


def fetch_protein_variants(gene_or_accession: str) -> List[VariantRecord]:
    """Fetch variant/mutation annotations for a human protein from UniProt.

    Args:
        gene_or_accession: Gene name or UniProt accession.

    Returns:
        List of VariantRecord. Empty list if not found or on error.
    """
    ck = _cache_key("protein_variants", gene_or_accession)
    if ck in _cache:
        return _cache[ck]

    entry = _search_uniprot(gene_or_accession)
    if entry is None:
        return []

    gene = _extract_gene_name(entry) or gene_or_accession.strip().upper()
    variant_features = _extract_features(entry, "Natural variant")

    results: List[VariantRecord] = []
    for feat in variant_features:
        location = feat.get("location", {})
        start = location.get("start", {}).get("value")
        position = _safe_int(start)

        # Extract amino acid change from alternativeSequence
        alt_seq = feat.get("alternativeSequence", {})
        original = alt_seq.get("originalSequence", "")
        variants = alt_seq.get("alternativeSequences", [])
        variant_aa = variants[0] if variants else ""

        description = feat.get("description", "")

        # Try to extract clinical significance from evidences or description
        clinical_sig = ""
        if "pathogenic" in description.lower():
            clinical_sig = "Pathogenic"
        elif "benign" in description.lower():
            clinical_sig = "Benign"
        elif "polymorphism" in description.lower():
            clinical_sig = "Polymorphism"
        elif "disease" in description.lower() or "dbSNP" in description:
            clinical_sig = "Disease-associated"

        rec = VariantRecord(
            gene=gene,
            position=position,
            original_aa=original,
            variant_aa=variant_aa,
            description=description,
            clinical_significance=clinical_sig,
        )
        results.append(rec)

    _cache[ck] = results
    return results


def fetch_protein_interactions(gene_or_accession: str) -> List[InteractionRecord]:
    """Fetch protein-protein interaction annotations from UniProt.

    Args:
        gene_or_accession: Gene name or UniProt accession.

    Returns:
        List of InteractionRecord. Empty list if not found or on error.
    """
    ck = _cache_key("protein_interactions", gene_or_accession)
    if ck in _cache:
        return _cache[ck]

    entry = _search_uniprot(gene_or_accession)
    if entry is None:
        return []

    gene = _extract_gene_name(entry) or gene_or_accession.strip().upper()

    # Extract interaction comments
    interaction_comments = _extract_comments(entry, "INTERACTION")
    results: List[InteractionRecord] = []

    for c in interaction_comments:
        interactions = c.get("interactions", [])
        for inter in interactions:
            # interactant_one = inter.get("interactantOne", {})  # Self - not needed
            interactant_two = inter.get("interactantTwo", {})

            # The partner is interactantTwo
            partner_gene = interactant_two.get("geneName", "")
            if not partner_gene:
                partner_id = interactant_two.get("uniProtKBAccession", "")
                partner_gene = partner_id if partner_id else "Unknown"

            experiments = _safe_int(inter.get("numberOfExperiments", 0))

            # Determine interaction type from organism
            org_two = interactant_two.get("organismDiffer", False)
            interaction_type = "physical (cross-species)" if org_two else "physical"

            rec = InteractionRecord(
                gene=gene,
                partner_gene=partner_gene,
                interaction_type=interaction_type,
                experiments_count=experiments,
            )
            results.append(rec)

    _cache[ck] = results
    return results


def clear_cache() -> int:
    """Clear the session cache. Returns number of entries cleared."""
    n = len(_cache)
    _cache.clear()
    return n


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== UniProt API Demo ===\n")

    # Fetch protein info
    print("--- Protein Info: TP53 ---")
    info = fetch_protein_info("TP53")
    if info:
        print(f"  {info}")
    else:
        print("  Not found")

    print()

    # Fetch function annotation
    print("--- Function: BRCA1 ---")
    func = fetch_protein_function("BRCA1")
    if func:
        print(f"  {func}")
    else:
        print("  Not found")

    print()

    # Fetch variants
    print("--- Variants: CFTR (first 5) ---")
    variants = fetch_protein_variants("CFTR")
    if variants:
        for v in variants[:5]:
            print(f"  {v}")
    else:
        print("  No variants found")

    print()

    # Fetch interactions
    print("--- Interactions: EGFR (first 5) ---")
    partners = fetch_protein_interactions("EGFR")
    if partners:
        for p in partners[:5]:
            print(f"  {p}")
    else:
        print("  No interactions found")

    print()
    print(f"Cache entries: {len(_cache)}")
    cleared = clear_cache()
    print(f"Cleared {cleared} cache entries")
