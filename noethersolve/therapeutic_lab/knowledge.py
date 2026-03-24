"""
Built-in knowledge base for therapeutic design.

Contains common targets, mechanisms, and disease associations.
"""

from typing import Dict, List

# ── Gene/Protein Recognition ─────────────────────────────────────────

# Well-known therapeutic targets with metadata
KNOWN_TARGETS: Dict[str, Dict] = {
    # Kinases
    "EGFR": {"type": "kinase", "druggable": True, "drugs": ["erlotinib", "gefitinib", "osimertinib"]},
    "HER2": {"type": "receptor", "druggable": True, "drugs": ["trastuzumab", "pertuzumab", "lapatinib"]},
    "BRAF": {"type": "kinase", "druggable": True, "drugs": ["vemurafenib", "dabrafenib", "encorafenib"]},
    "ALK": {"type": "kinase", "druggable": True, "drugs": ["crizotinib", "alectinib", "ceritinib"]},
    "BCR-ABL": {"type": "kinase", "druggable": True, "drugs": ["imatinib", "dasatinib", "nilotinib"]},
    "JAK2": {"type": "kinase", "druggable": True, "drugs": ["ruxolitinib", "fedratinib"]},
    "BTK": {"type": "kinase", "druggable": True, "drugs": ["ibrutinib", "acalabrutinib"]},
    "CDK4": {"type": "kinase", "druggable": True, "drugs": ["palbociclib", "ribociclib"]},
    "CDK6": {"type": "kinase", "druggable": True, "drugs": ["palbociclib", "ribociclib"]},
    "KRAS": {"type": "enzyme", "druggable": True, "drugs": ["sotorasib", "adagrasib"]},  # G12C
    "NRAS": {"type": "enzyme", "druggable": False, "drugs": []},
    "HRAS": {"type": "enzyme", "druggable": False, "drugs": []},
    "MET": {"type": "kinase", "druggable": True, "drugs": ["capmatinib", "tepotinib"]},
    "RET": {"type": "kinase", "druggable": True, "drugs": ["selpercatinib", "pralsetinib"]},
    "FGFR": {"type": "kinase", "druggable": True, "drugs": ["erdafitinib", "pemigatinib"]},
    "PI3K": {"type": "kinase", "druggable": True, "drugs": ["alpelisib", "idelalisib"]},
    "AKT": {"type": "kinase", "druggable": True, "drugs": ["capivasertib"]},
    "MTOR": {"type": "kinase", "druggable": True, "drugs": ["everolimus", "temsirolimus"]},

    # Tumor suppressors (often loss-of-function)
    "TP53": {"type": "transcription_factor", "druggable": False, "drugs": []},
    "RB1": {"type": "transcription_factor", "druggable": False, "drugs": []},
    "PTEN": {"type": "enzyme", "druggable": False, "drugs": []},
    "BRCA1": {"type": "structural", "druggable": False, "drugs": ["olaparib"]},  # Synthetic lethality
    "BRCA2": {"type": "structural", "druggable": False, "drugs": ["olaparib"]},
    "APC": {"type": "structural", "druggable": False, "drugs": []},
    "VHL": {"type": "enzyme", "druggable": False, "drugs": []},

    # Ion channels
    "CFTR": {"type": "ion_channel", "druggable": True, "drugs": ["ivacaftor", "lumacaftor", "tezacaftor"]},
    "SCN1A": {"type": "ion_channel", "druggable": True, "drugs": []},
    "KCNQ1": {"type": "ion_channel", "druggable": True, "drugs": []},
    "CACNA1A": {"type": "ion_channel", "druggable": True, "drugs": []},

    # Receptors
    "PD1": {"type": "receptor", "druggable": True, "drugs": ["pembrolizumab", "nivolumab"]},
    "PDL1": {"type": "receptor", "druggable": True, "drugs": ["atezolizumab", "durvalumab"]},
    "CTLA4": {"type": "receptor", "druggable": True, "drugs": ["ipilimumab"]},
    "TNF": {"type": "secreted", "druggable": True, "drugs": ["adalimumab", "infliximab", "etanercept"]},
    "IL6": {"type": "secreted", "druggable": True, "drugs": ["tocilizumab", "siltuximab"]},
    "IL17": {"type": "secreted", "druggable": True, "drugs": ["secukinumab", "ixekizumab"]},
    "VEGF": {"type": "secreted", "druggable": True, "drugs": ["bevacizumab", "aflibercept"]},
    "VEGFR": {"type": "receptor", "druggable": True, "drugs": ["sunitinib", "sorafenib"]},

    # Enzymes
    "ACE": {"type": "enzyme", "druggable": True, "drugs": ["lisinopril", "enalapril"]},
    "ACE2": {"type": "enzyme", "druggable": True, "drugs": []},
    "HMGCR": {"type": "enzyme", "druggable": True, "drugs": ["atorvastatin", "simvastatin"]},
    "COX1": {"type": "enzyme", "druggable": True, "drugs": ["aspirin", "ibuprofen"]},
    "COX2": {"type": "enzyme", "druggable": True, "drugs": ["celecoxib"]},
    "PCSK9": {"type": "enzyme", "druggable": True, "drugs": ["evolocumab", "alirocumab"]},
    "DPP4": {"type": "enzyme", "druggable": True, "drugs": ["sitagliptin", "saxagliptin"]},
    "SGLT2": {"type": "enzyme", "druggable": True, "drugs": ["empagliflozin", "dapagliflozin"]},

    # Secreted proteins
    "EPO": {"type": "secreted", "druggable": True, "drugs": ["epoetin alfa"]},
    "GH": {"type": "secreted", "druggable": True, "drugs": ["somatropin"]},
    "INSULIN": {"type": "secreted", "druggable": True, "drugs": ["insulin"]},
    "FSH": {"type": "secreted", "druggable": True, "drugs": ["follitropin"]},
    "GLP1": {"type": "secreted", "druggable": True, "drugs": ["semaglutide", "liraglutide"]},
    "F8": {"type": "secreted", "druggable": True, "drugs": ["factor VIII"]},
    "F9": {"type": "secreted", "druggable": True, "drugs": ["factor IX"]},

    # Transcription factors (generally hard to drug)
    "MYC": {"type": "transcription_factor", "druggable": False, "drugs": []},
    "MYCN": {"type": "transcription_factor", "druggable": False, "drugs": []},
    "HIF1A": {"type": "transcription_factor", "druggable": True, "drugs": ["belzutifan"]},
    "STAT3": {"type": "transcription_factor", "druggable": False, "drugs": []},
    "NFkB": {"type": "transcription_factor", "druggable": False, "drugs": []},
    "AR": {"type": "transcription_factor", "druggable": True, "drugs": ["enzalutamide", "abiraterone"]},
    "ER": {"type": "transcription_factor", "druggable": True, "drugs": ["tamoxifen", "fulvestrant"]},

    # DNA repair
    "PARP1": {"type": "enzyme", "druggable": True, "drugs": ["olaparib", "niraparib", "rucaparib"]},
    "ATM": {"type": "kinase", "druggable": True, "drugs": []},
    "ATR": {"type": "kinase", "druggable": True, "drugs": []},

    # Proteases
    "HIV_PROTEASE": {"type": "protease", "druggable": True, "drugs": ["ritonavir", "lopinavir"]},
    "HCV_NS3": {"type": "protease", "druggable": True, "drugs": ["telaprevir", "boceprevir"]},
    "TMPRSS2": {"type": "protease", "druggable": True, "drugs": []},

    # Neurodegenerative targets
    "APP": {"type": "structural", "druggable": True, "drugs": ["lecanemab", "aducanumab"]},
    "TAU": {"type": "structural", "druggable": False, "drugs": []},
    "SNCA": {"type": "structural", "druggable": False, "drugs": []},  # Alpha-synuclein
    "HTT": {"type": "structural", "druggable": True, "drugs": []},  # Huntingtin - ASO target
    "SOD1": {"type": "enzyme", "druggable": True, "drugs": ["tofersen"]},
    "SMN1": {"type": "structural", "druggable": True, "drugs": ["nusinersen", "onasemnogene"]},
    "DMD": {"type": "structural", "druggable": True, "drugs": ["eteplirsen", "golodirsen"]},
}

# Gene name aliases
GENE_ALIASES: Dict[str, str] = {
    "ERBB2": "HER2",
    "NEU": "HER2",
    "P53": "TP53",
    "RAS": "KRAS",
    "PDCD1": "PD1",
    "CD274": "PDL1",
    "B7H1": "PDL1",
    "TNFA": "TNF",
    "TNFSF1A": "TNF",
    "IL6R": "IL6",
    "ABCC7": "CFTR",
    "EPO": "EPO",
    "ERYTHROPOIETIN": "EPO",
    "GLP1R": "GLP1",
    "FVIII": "F8",
    "FIX": "F9",
    "DYSTROPHIN": "DMD",
    "HUNTINGTON": "HTT",
    "HUNTINGTIN": "HTT",
}


# ── Mechanism Keywords ─────────────────────────────────────────────

MECHANISM_KEYWORDS: Dict[str, List[str]] = {
    "loss_of_function": [
        "loss of function", "loss-of-function", "lof",
        "deficiency", "deficient", "absent", "missing",
        "inactivating", "null", "knockout", "ko",
        "truncating", "nonsense", "frameshift",
        "hypomorphic", "reduced activity", "impaired",
    ],
    "gain_of_function": [
        "gain of function", "gain-of-function", "gof",
        "activating", "constitutively active", "hyperactive",
        "oncogenic", "driver mutation", "hotspot mutation",
    ],
    "overexpression": [
        "overexpression", "overexpressed", "amplification", "amplified",
        "upregulation", "upregulated", "elevated", "increased expression",
        "high expression", "overproduction",
    ],
    "underexpression": [
        "underexpression", "downregulation", "downregulated",
        "reduced expression", "low expression", "silenced",
        "epigenetically silenced", "methylated",
    ],
    "mutation": [
        "mutation", "mutant", "mutated", "variant",
        "polymorphism", "snp", "missense", "point mutation",
    ],
    "misfolding": [
        "misfolding", "misfolded", "aggregation", "aggregate",
        "protein aggregates", "amyloid", "inclusion bodies",
        "unfolded protein", "er stress",
    ],
}


# ── Tissue Keywords ─────────────────────────────────────────────────

TISSUE_KEYWORDS: Dict[str, List[str]] = {
    "lung": ["lung", "pulmonary", "respiratory", "bronchial", "alveolar", "airway"],
    "liver": ["liver", "hepatic", "hepatocyte", "hepato-"],
    "brain": ["brain", "cerebral", "neuronal", "cns", "central nervous", "cortical"],
    "heart": ["heart", "cardiac", "cardiovascular", "myocardial", "cardiomyocyte"],
    "kidney": ["kidney", "renal", "nephron", "glomerular"],
    "pancreas": ["pancreas", "pancreatic", "islet", "beta cell"],
    "blood": ["blood", "hematopoietic", "leukemia", "lymphoma", "bone marrow"],
    "skin": ["skin", "dermal", "epidermal", "melanoma", "keratinocyte"],
    "muscle": ["muscle", "skeletal muscle", "muscular", "myopathy"],
    "bone": ["bone", "skeletal", "osteoblast", "osteoclast"],
    "eye": ["eye", "retinal", "ocular", "ophthalmic"],
    "gut": ["intestine", "intestinal", "colon", "colonic", "gi tract", "gastrointestinal"],
    "breast": ["breast", "mammary"],
    "prostate": ["prostate", "prostatic"],
    "ovary": ["ovary", "ovarian"],
    "thyroid": ["thyroid"],
    "immune": ["immune", "lymphocyte", "t cell", "b cell", "macrophage", "dendritic"],
}


# ── Disease-Target Associations ─────────────────────────────────────

DISEASE_TARGETS: Dict[str, List[str]] = {
    "cystic fibrosis": ["CFTR"],
    "chronic myeloid leukemia": ["BCR-ABL"],
    "cml": ["BCR-ABL"],
    "melanoma": ["BRAF", "NRAS", "CDKN2A", "PD1"],
    "non-small cell lung cancer": ["EGFR", "KRAS", "ALK", "ROS1", "MET", "RET"],
    "nsclc": ["EGFR", "KRAS", "ALK", "ROS1", "MET", "RET"],
    "breast cancer": ["HER2", "ER", "BRCA1", "BRCA2", "CDK4", "CDK6"],
    "colorectal cancer": ["KRAS", "BRAF", "APC", "TP53"],
    "pancreatic cancer": ["KRAS", "TP53", "CDKN2A", "SMAD4"],
    "prostate cancer": ["AR", "PTEN", "BRCA2"],
    "ovarian cancer": ["BRCA1", "BRCA2", "TP53"],
    "leukemia": ["BCR-ABL", "FLT3", "JAK2", "BTK"],
    "lymphoma": ["BTK", "BCL2", "MYC", "PD1"],
    "multiple myeloma": ["BTK", "BCMA"],

    "rheumatoid arthritis": ["TNF", "IL6", "JAK"],
    "psoriasis": ["IL17", "IL23", "TNF"],
    "crohn's disease": ["TNF", "IL23"],
    "ulcerative colitis": ["TNF", "JAK"],

    "alzheimer's disease": ["APP", "TAU", "APOE"],
    "parkinson's disease": ["SNCA", "LRRK2", "GBA"],
    "huntington's disease": ["HTT"],
    "als": ["SOD1", "C9orf72", "FUS"],
    "sma": ["SMN1"],
    "duchenne muscular dystrophy": ["DMD"],
    "dmd": ["DMD"],

    "hemophilia a": ["F8"],
    "hemophilia b": ["F9"],
    "sickle cell disease": ["HBB", "BCL11A"],

    "diabetes": ["GLP1", "SGLT2", "DPP4", "INSULIN"],
    "obesity": ["GLP1", "MC4R"],
    "hypercholesterolemia": ["PCSK9", "HMGCR", "LDLR"],
    "hypertension": ["ACE", "AGT"],

    "covid-19": ["ACE2", "TMPRSS2", "SPIKE"],
    "hiv": ["HIV_PROTEASE", "CCR5", "CD4"],
    "hepatitis c": ["HCV_NS3", "HCV_NS5A", "HCV_NS5B"],
}


# ── Target Type → Modality Preferences ─────────────────────────────

MODALITY_PREFERENCES: Dict[str, List[str]] = {
    "enzyme": ["small_molecule", "antibody"],
    "kinase": ["small_molecule"],
    "receptor": ["antibody", "small_molecule"],
    "ion_channel": ["small_molecule"],
    "transcription_factor": ["aso", "crispr"],
    "secreted": ["antibody", "mrna"],
    "structural": ["aso", "crispr", "gene_therapy"],
    "protease": ["small_molecule"],
}

MECHANISM_MODALITY_PREFERENCES: Dict[str, List[str]] = {
    "loss_of_function": ["mrna", "gene_therapy", "small_molecule"],  # Replace or potentiate
    "gain_of_function": ["small_molecule", "antibody", "crispr"],  # Inhibit
    "overexpression": ["small_molecule", "antibody", "aso"],  # Inhibit or knockdown
    "underexpression": ["mrna", "gene_therapy"],  # Replace
    "mutation": ["crispr", "small_molecule"],  # Correct or target mutant
    "misfolding": ["small_molecule", "mrna"],  # Chaperone or replace
}


def resolve_gene_alias(name: str) -> str:
    """Resolve gene alias to canonical name."""
    name_upper = name.upper()
    return GENE_ALIASES.get(name_upper, name_upper)


def get_known_target_info(name: str) -> Dict:
    """Get information about a known target."""
    canonical = resolve_gene_alias(name)
    return KNOWN_TARGETS.get(canonical, {})


def get_disease_targets(disease: str) -> List[str]:
    """Get known targets for a disease."""
    disease_lower = disease.lower()
    for key, targets in DISEASE_TARGETS.items():
        if key in disease_lower or disease_lower in key:
            return targets
    return []


def get_modality_preferences(target_type: str, mechanism: str) -> List[str]:
    """Get preferred modalities based on target type and mechanism."""
    type_prefs = MODALITY_PREFERENCES.get(target_type, [])
    mech_prefs = MECHANISM_MODALITY_PREFERENCES.get(mechanism, [])

    # Combine, prioritizing mechanism-specific preferences
    combined = []
    for m in mech_prefs:
        if m not in combined:
            combined.append(m)
    for m in type_prefs:
        if m not in combined:
            combined.append(m)

    return combined if combined else ["small_molecule", "antibody"]
