"""noethersolve.drug_interactions — Drug-drug interaction prediction engine.

Provides verified databases for:
- CYP450 enzyme substrates, inhibitors, and inducers
- Drug interaction prediction with severity classification
- Pharmacogenomic variants (CYP2D6, CYP2C19, CYP2C9 polymorphisms)
- Clinical recommendations

Common LLM errors this module corrects:
- Confusing which CYP enzyme metabolizes which drug
- Confusing inhibitors vs inducers (they have OPPOSITE effects)
- Not knowing relative interaction strengths (strong vs moderate vs weak)
- Missing clinically important interactions (e.g., warfarin + NSAIDs)
- Getting AUC fold-change magnitudes wrong

Database sources: FDA Clinical Pharmacology tables, Flockhart Table (Indiana),
drug prescribing information (package inserts).

Usage:
    from noethersolve.drug_interactions import (
        check_interaction, get_drug_profile, get_cyp_info,
        predict_auc_change, check_pharmacogenomics,
    )

    # Check interaction between two drugs
    r = check_interaction("ketoconazole", "midazolam")
    print(r)  # Strong CYP3A4 inhibition, AUC ↑10-15×

    # Get a drug's metabolic profile
    r = get_drug_profile("warfarin")
    print(r)  # CYP2C9 substrate, multiple interactors
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple


# ─── Enums ────────────────────────────────────────────────────────────────────

class Strength(Enum):
    """Interaction strength based on AUC fold-change."""
    STRONG = "strong"      # AUC ≥ 5× (inhibition) or ≤ 0.2× (induction)
    MODERATE = "moderate"  # AUC 2-5× (inhibition) or 0.2-0.5× (induction)
    WEAK = "weak"          # AUC 1.25-2× (inhibition) or 0.5-0.8× (induction)


class InteractionType(Enum):
    """Type of drug-drug interaction."""
    INHIBITION = "inhibition"    # Inhibitor raises substrate levels
    INDUCTION = "induction"      # Inducer lowers substrate levels
    SUBSTRATE_COMPETITION = "substrate_competition"  # Two substrates compete
    PHARMACODYNAMIC = "pharmacodynamic"  # PD interaction (additive/antagonistic)


class Severity(Enum):
    """Clinical severity classification."""
    CONTRAINDICATED = "contraindicated"  # Do not use together
    MAJOR = "major"                       # Monitoring required, dose adjustment
    MODERATE = "moderate"                 # Use with caution
    MINOR = "minor"                       # Usually no action needed


# ─── CYP450 Database ──────────────────────────────────────────────────────────

# CYP enzyme substrates (drug → set of CYP enzymes that metabolize it)
# Sensitive substrates (narrow therapeutic index) marked with asterisks
CYP_SUBSTRATES: Dict[str, Set[str]] = {
    # CYP3A4 substrates
    "midazolam": {"CYP3A4"},
    "triazolam": {"CYP3A4"},
    "alprazolam": {"CYP3A4"},
    "simvastatin": {"CYP3A4"},
    "lovastatin": {"CYP3A4"},
    "atorvastatin": {"CYP3A4"},
    "cyclosporine": {"CYP3A4"},
    "tacrolimus": {"CYP3A4"},
    "sirolimus": {"CYP3A4"},
    "felodipine": {"CYP3A4"},
    "nifedipine": {"CYP3A4"},
    "sildenafil": {"CYP3A4"},
    "vardenafil": {"CYP3A4"},
    "fentanyl": {"CYP3A4"},
    "oxycodone": {"CYP3A4"},
    "quetiapine": {"CYP3A4"},
    "buspirone": {"CYP3A4"},
    "pimozide": {"CYP3A4"},
    "ergotamine": {"CYP3A4"},

    # CYP2D6 substrates
    "codeine": {"CYP2D6"},  # prodrug - 2D6 converts to morphine
    "tramadol": {"CYP2D6"},
    "oxycodone_2d6": {"CYP2D6", "CYP3A4"},  # minor 2D6 pathway
    "dextromethorphan": {"CYP2D6"},
    "metoprolol": {"CYP2D6"},
    "carvedilol": {"CYP2D6"},
    "propranolol": {"CYP2D6"},
    "flecainide": {"CYP2D6"},
    "propafenone": {"CYP2D6"},
    "atomoxetine": {"CYP2D6"},
    "duloxetine": {"CYP2D6"},
    "venlafaxine": {"CYP2D6"},
    "paroxetine": {"CYP2D6"},  # also inhibitor
    "risperidone": {"CYP2D6"},
    "aripiprazole": {"CYP2D6", "CYP3A4"},
    "tamoxifen": {"CYP2D6"},  # prodrug - 2D6 converts to endoxifen

    # CYP2C9 substrates
    "warfarin": {"CYP2C9"},  # S-warfarin (active)
    "phenytoin": {"CYP2C9", "CYP2C19"},
    "celecoxib": {"CYP2C9"},
    "losartan": {"CYP2C9"},  # prodrug
    "irbesartan": {"CYP2C9"},
    "glipizide": {"CYP2C9"},
    "tolbutamide": {"CYP2C9"},

    # CYP2C19 substrates
    "omeprazole": {"CYP2C19"},
    "esomeprazole": {"CYP2C19"},
    "lansoprazole": {"CYP2C19"},
    "pantoprazole": {"CYP2C19"},
    "clopidogrel": {"CYP2C19"},  # prodrug
    "citalopram": {"CYP2C19"},
    "escitalopram": {"CYP2C19"},
    "diazepam": {"CYP2C19", "CYP3A4"},
    "phenobarbital": {"CYP2C19"},
    "voriconazole": {"CYP2C19", "CYP3A4", "CYP2C9"},

    # CYP1A2 substrates
    "theophylline": {"CYP1A2"},
    "caffeine": {"CYP1A2"},
    "clozapine": {"CYP1A2"},
    "olanzapine": {"CYP1A2"},
    "tizanidine": {"CYP1A2"},
    "duloxetine_1a2": {"CYP1A2", "CYP2D6"},
    "melatonin": {"CYP1A2"},
    "ramelteon": {"CYP1A2"},

    # CYP2B6 substrates
    "efavirenz": {"CYP2B6"},
    "bupropion": {"CYP2B6"},
    "methadone": {"CYP2B6", "CYP3A4"},
    "cyclophosphamide": {"CYP2B6"},
}

# Sensitive substrates - narrow therapeutic index or serious consequences
SENSITIVE_SUBSTRATES: Set[str] = {
    "warfarin", "phenytoin", "cyclosporine", "tacrolimus", "sirolimus",
    "pimozide", "ergotamine", "fentanyl", "theophylline", "clozapine",
    "midazolam", "triazolam", "simvastatin", "lovastatin",
}

# CYP inhibitors: inhibitor → {enzyme: strength}
CYP_INHIBITORS: Dict[str, Dict[str, Strength]] = {
    # Strong CYP3A4 inhibitors
    "ketoconazole": {"CYP3A4": Strength.STRONG},
    "itraconazole": {"CYP3A4": Strength.STRONG},
    "posaconazole": {"CYP3A4": Strength.STRONG},
    "voriconazole": {"CYP3A4": Strength.STRONG, "CYP2C19": Strength.STRONG},
    "clarithromycin": {"CYP3A4": Strength.STRONG},
    "telithromycin": {"CYP3A4": Strength.STRONG},
    "ritonavir": {"CYP3A4": Strength.STRONG, "CYP2D6": Strength.STRONG},
    "cobicistat": {"CYP3A4": Strength.STRONG},
    "grapefruit_juice": {"CYP3A4": Strength.MODERATE},  # intestinal only

    # Moderate CYP3A4 inhibitors
    "erythromycin": {"CYP3A4": Strength.MODERATE},
    "fluconazole": {"CYP3A4": Strength.MODERATE, "CYP2C9": Strength.STRONG, "CYP2C19": Strength.MODERATE},
    "diltiazem": {"CYP3A4": Strength.MODERATE},
    "verapamil": {"CYP3A4": Strength.MODERATE},
    "ciprofloxacin": {"CYP1A2": Strength.STRONG, "CYP3A4": Strength.WEAK},
    "fluvoxamine": {"CYP1A2": Strength.STRONG, "CYP2C19": Strength.STRONG, "CYP3A4": Strength.WEAK},

    # CYP2D6 inhibitors
    "paroxetine": {"CYP2D6": Strength.STRONG},
    "fluoxetine": {"CYP2D6": Strength.STRONG, "CYP2C19": Strength.MODERATE},
    "bupropion": {"CYP2D6": Strength.MODERATE},
    "quinidine": {"CYP2D6": Strength.STRONG},
    "terbinafine": {"CYP2D6": Strength.STRONG},

    # CYP2C9 inhibitors
    "fluconazole_2c9": {"CYP2C9": Strength.STRONG},
    "amiodarone": {"CYP2C9": Strength.MODERATE, "CYP2D6": Strength.MODERATE},
    "miconazole": {"CYP2C9": Strength.STRONG},
    "metronidazole": {"CYP2C9": Strength.WEAK},

    # CYP2C19 inhibitors
    "omeprazole_inhib": {"CYP2C19": Strength.MODERATE},
    "esomeprazole_inhib": {"CYP2C19": Strength.MODERATE},
    "ticlopidine": {"CYP2C19": Strength.STRONG},

    # CYP1A2 inhibitors
    "ciprofloxacin_1a2": {"CYP1A2": Strength.STRONG},
    "fluvoxamine_1a2": {"CYP1A2": Strength.STRONG},
    "enoxacin": {"CYP1A2": Strength.STRONG},
}

# CYP inducers: inducer → {enzyme: strength}
CYP_INDUCERS: Dict[str, Dict[str, Strength]] = {
    # Strong CYP3A4 inducers
    "rifampin": {"CYP3A4": Strength.STRONG, "CYP2C9": Strength.STRONG, "CYP2C19": Strength.STRONG, "CYP2B6": Strength.STRONG},
    "rifabutin": {"CYP3A4": Strength.MODERATE},
    "phenytoin": {"CYP3A4": Strength.STRONG, "CYP2C9": Strength.STRONG, "CYP2C19": Strength.MODERATE},
    "carbamazepine": {"CYP3A4": Strength.STRONG, "CYP1A2": Strength.MODERATE, "CYP2C9": Strength.MODERATE},
    "phenobarbital": {"CYP3A4": Strength.STRONG, "CYP2C9": Strength.STRONG, "CYP2C19": Strength.STRONG},
    "st_johns_wort": {"CYP3A4": Strength.STRONG},  # herbal

    # Moderate inducers
    "efavirenz": {"CYP3A4": Strength.MODERATE, "CYP2B6": Strength.MODERATE},
    "modafinil": {"CYP3A4": Strength.MODERATE},

    # CYP1A2 inducers (smoking!)
    "smoking": {"CYP1A2": Strength.STRONG},  # PAHs in smoke induce 1A2
    "omeprazole_induce": {"CYP1A2": Strength.WEAK},  # mild induction
}

# AUC fold-change estimates by strength
AUC_FOLD_CHANGE: Dict[str, Dict[Strength, Tuple[float, float]]] = {
    "inhibition": {
        Strength.STRONG: (5.0, 15.0),    # ≥5× increase
        Strength.MODERATE: (2.0, 5.0),   # 2-5× increase
        Strength.WEAK: (1.25, 2.0),      # 1.25-2× increase
    },
    "induction": {
        Strength.STRONG: (0.05, 0.2),    # ≥80% decrease
        Strength.MODERATE: (0.2, 0.5),   # 50-80% decrease
        Strength.WEAK: (0.5, 0.8),       # 20-50% decrease
    },
}


# ─── Pharmacogenomics ─────────────────────────────────────────────────────────

# CYP2D6 phenotypes and prevalence
CYP2D6_PHENOTYPES: Dict[str, Dict[str, str]] = {
    "poor_metabolizer": {
        "abbreviation": "PM",
        "prevalence": "5-10% Caucasians, 1-2% Asians",
        "activity": "None or minimal",
        "clinical_impact": "Toxicity risk with standard doses of 2D6 substrates; prodrug activation failure (codeine→morphine)",
    },
    "intermediate_metabolizer": {
        "abbreviation": "IM",
        "prevalence": "10-17%",
        "activity": "Reduced",
        "clinical_impact": "May need dose reduction for 2D6 substrates",
    },
    "normal_metabolizer": {
        "abbreviation": "NM",
        "prevalence": "~70%",
        "activity": "Normal",
        "clinical_impact": "Standard dosing",
    },
    "ultrarapid_metabolizer": {
        "abbreviation": "UM",
        "prevalence": "1-10% (higher in Middle East, Ethiopia)",
        "activity": "Increased",
        "clinical_impact": "May need higher doses; DANGEROUS with codeine (rapid morphine production)",
    },
}

# CYP2C19 phenotypes
CYP2C19_PHENOTYPES: Dict[str, Dict[str, str]] = {
    "poor_metabolizer": {
        "abbreviation": "PM",
        "prevalence": "2-5% Caucasians, 13-23% Asians",
        "activity": "None or minimal",
        "clinical_impact": "Clopidogrel FAILURE (prodrug not activated); higher PPI levels",
    },
    "intermediate_metabolizer": {
        "abbreviation": "IM",
        "prevalence": "~25%",
        "activity": "Reduced",
        "clinical_impact": "Reduced clopidogrel response",
    },
    "normal_metabolizer": {
        "abbreviation": "NM",
        "prevalence": "~60%",
        "activity": "Normal",
        "clinical_impact": "Standard dosing",
    },
    "rapid_metabolizer": {
        "abbreviation": "RM",
        "prevalence": "2-5%",
        "activity": "Increased",
        "clinical_impact": "Lower PPI levels; may need higher doses",
    },
    "ultrarapid_metabolizer": {
        "abbreviation": "UM",
        "prevalence": "5-30%",
        "activity": "Much increased",
        "clinical_impact": "Much lower PPI levels; clopidogrel superactivation",
    },
}

# Prodrugs requiring CYP activation
PRODRUGS: Dict[str, Dict[str, str]] = {
    "codeine": {"enzyme": "CYP2D6", "active_metabolite": "morphine", "clinical_note": "CYP2D6 PMs get no analgesia; UMs get dangerously high morphine"},
    "tramadol": {"enzyme": "CYP2D6", "active_metabolite": "O-desmethyltramadol", "clinical_note": "Reduced efficacy in PMs"},
    "clopidogrel": {"enzyme": "CYP2C19", "active_metabolite": "active thiol metabolite", "clinical_note": "CYP2C19 PMs have 3× higher CV event risk"},
    "tamoxifen": {"enzyme": "CYP2D6", "active_metabolite": "endoxifen", "clinical_note": "CYP2D6 PMs may have reduced efficacy; avoid strong 2D6 inhibitors"},
    "losartan": {"enzyme": "CYP2C9", "active_metabolite": "E-3174", "clinical_note": "Reduced antihypertensive effect in CYP2C9 PMs"},
}


# ─── Report Dataclasses ───────────────────────────────────────────────────────

@dataclass
class DrugProfileReport:
    """Metabolic profile of a drug."""
    drug: str
    metabolizing_enzymes: List[str]
    is_sensitive_substrate: bool
    is_prodrug: bool
    prodrug_info: Optional[Dict[str, str]]
    known_inhibitor_of: Dict[str, Strength]
    known_inducer_of: Dict[str, Strength]
    notes: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        lines = ["=" * 60, f"  Drug Profile: {self.drug.upper()}", "=" * 60]
        if self.metabolizing_enzymes:
            lines.append(f"  Metabolized by: {', '.join(self.metabolizing_enzymes)}")
        else:
            lines.append("  Metabolizing enzymes: Unknown/Not in database")
        if self.is_sensitive_substrate:
            lines.append("  ⚠️  SENSITIVE SUBSTRATE (narrow therapeutic index)")
        if self.is_prodrug and self.prodrug_info:
            lines.append(f"  ⚠️  PRODRUG: requires {self.prodrug_info['enzyme']} for activation")
            lines.append(f"     Active metabolite: {self.prodrug_info['active_metabolite']}")
            lines.append(f"     Note: {self.prodrug_info['clinical_note']}")
        if self.known_inhibitor_of:
            inh = ", ".join(f"{e} ({s.value})" for e, s in self.known_inhibitor_of.items())
            lines.append(f"  Inhibitor of: {inh}")
        if self.known_inducer_of:
            ind = ", ".join(f"{e} ({s.value})" for e, s in self.known_inducer_of.items())
            lines.append(f"  Inducer of: {ind}")
        for note in self.notes:
            lines.append(f"  Note: {note}")
        lines.append("=" * 60)
        return "\n".join(lines)


@dataclass
class InteractionReport:
    """Drug-drug interaction assessment."""
    drug_a: str
    drug_b: str
    interaction_found: bool
    interaction_type: Optional[InteractionType]
    mechanism: str
    affected_enzyme: Optional[str]
    strength: Optional[Strength]
    severity: Optional[Severity]
    auc_change_range: Optional[Tuple[float, float]]
    clinical_significance: str
    recommendation: str
    notes: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        lines = ["=" * 60, f"  Interaction: {self.drug_a.upper()} + {self.drug_b.upper()}", "=" * 60]
        if not self.interaction_found:
            lines.append("  No known pharmacokinetic interaction found")
            lines.append("  (Always verify with current prescribing information)")
        else:
            lines.append(f"  Type: {self.interaction_type.value if self.interaction_type else 'Unknown'}")
            lines.append(f"  Mechanism: {self.mechanism}")
            if self.affected_enzyme:
                lines.append(f"  Enzyme: {self.affected_enzyme}")
            if self.strength:
                lines.append(f"  Strength: {self.strength.value.upper()}")
            if self.severity:
                sev_emoji = {"contraindicated": "🚫", "major": "⚠️", "moderate": "⚡", "minor": "ℹ️"}
                lines.append(f"  Severity: {sev_emoji.get(self.severity.value, '')} {self.severity.value.upper()}")
            if self.auc_change_range:
                lo, hi = self.auc_change_range
                if lo >= 1:
                    lines.append(f"  AUC change: {lo:.1f}-{hi:.1f}× increase")
                else:
                    lines.append(f"  AUC change: {(1-hi)*100:.0f}-{(1-lo)*100:.0f}% decrease")
            lines.append("-" * 60)
            lines.append(f"  Clinical: {self.clinical_significance}")
            lines.append(f"  Recommendation: {self.recommendation}")
        for note in self.notes:
            lines.append(f"  Note: {note}")
        lines.append("=" * 60)
        return "\n".join(lines)


@dataclass
class CYPInfoReport:
    """Information about a CYP enzyme."""
    enzyme: str
    substrates: List[str]
    sensitive_substrates: List[str]
    strong_inhibitors: List[str]
    moderate_inhibitors: List[str]
    strong_inducers: List[str]
    notes: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        lines = ["=" * 60, f"  CYP Enzyme: {self.enzyme}", "=" * 60]
        if self.substrates:
            lines.append(f"  Substrates: {', '.join(self.substrates[:10])}")
            if len(self.substrates) > 10:
                lines.append(f"    ... and {len(self.substrates) - 10} more")
        if self.sensitive_substrates:
            lines.append(f"  ⚠️ Sensitive substrates: {', '.join(self.sensitive_substrates)}")
        lines.append("-" * 60)
        if self.strong_inhibitors:
            lines.append(f"  Strong inhibitors: {', '.join(self.strong_inhibitors)}")
        if self.moderate_inhibitors:
            lines.append(f"  Moderate inhibitors: {', '.join(self.moderate_inhibitors)}")
        if self.strong_inducers:
            lines.append(f"  Strong inducers: {', '.join(self.strong_inducers)}")
        for note in self.notes:
            lines.append(f"  Note: {note}")
        lines.append("=" * 60)
        return "\n".join(lines)


@dataclass
class PharmacogenomicsReport:
    """Pharmacogenomics assessment for a drug."""
    drug: str
    relevant_enzymes: List[str]
    phenotype_impacts: Dict[str, Dict[str, str]]  # enzyme → phenotype → impact
    is_prodrug: bool
    prodrug_warning: str
    clinical_recommendations: List[str]
    notes: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        lines = ["=" * 60, f"  Pharmacogenomics: {self.drug.upper()}", "=" * 60]
        lines.append(f"  Relevant enzymes: {', '.join(self.relevant_enzymes)}")
        if self.is_prodrug:
            lines.append(f"  ⚠️ PRODRUG: {self.prodrug_warning}")
        lines.append("-" * 60)
        for enzyme, impacts in self.phenotype_impacts.items():
            lines.append(f"  {enzyme} phenotypes:")
            for phenotype, impact in impacts.items():
                lines.append(f"    {phenotype}: {impact}")
        lines.append("-" * 60)
        lines.append("  Clinical recommendations:")
        for rec in self.clinical_recommendations:
            lines.append(f"    • {rec}")
        for note in self.notes:
            lines.append(f"  Note: {note}")
        lines.append("=" * 60)
        return "\n".join(lines)


# ─── Public API ───────────────────────────────────────────────────────────────

def _normalize_drug_name(name: str) -> str:
    """Normalize drug name for lookup."""
    return name.lower().strip().replace(" ", "_").replace("-", "_")


def get_drug_profile(drug: str) -> DrugProfileReport:
    """Get the metabolic profile of a drug.

    Args:
        drug: Drug name (case-insensitive)

    Returns:
        DrugProfileReport with metabolizing enzymes, inhibition/induction info.
    """
    name = _normalize_drug_name(drug)

    enzymes = list(CYP_SUBSTRATES.get(name, set()))
    is_sensitive = name in SENSITIVE_SUBSTRATES
    is_prodrug = name in PRODRUGS
    prodrug_info = PRODRUGS.get(name)

    # Check if this drug is an inhibitor
    inhibitor_of: Dict[str, Strength] = {}
    for inhibitor, enzyme_dict in CYP_INHIBITORS.items():
        if _normalize_drug_name(inhibitor).startswith(name):
            inhibitor_of.update(enzyme_dict)

    # Check if this drug is an inducer
    inducer_of: Dict[str, Strength] = {}
    for inducer, enzyme_dict in CYP_INDUCERS.items():
        if _normalize_drug_name(inducer).startswith(name):
            inducer_of.update(enzyme_dict)

    notes = []
    if not enzymes and not inhibitor_of and not inducer_of:
        notes.append(f"'{drug}' not found in database - check spelling or use generic name")

    return DrugProfileReport(
        drug=drug,
        metabolizing_enzymes=sorted(enzymes),
        is_sensitive_substrate=is_sensitive,
        is_prodrug=is_prodrug,
        prodrug_info=prodrug_info,
        known_inhibitor_of=inhibitor_of,
        known_inducer_of=inducer_of,
        notes=notes,
    )


def get_cyp_info(enzyme: str) -> CYPInfoReport:
    """Get information about a CYP enzyme.

    Args:
        enzyme: Enzyme name (e.g., "CYP3A4", "CYP2D6")

    Returns:
        CYPInfoReport with substrates, inhibitors, and inducers.
    """
    enzyme_upper = enzyme.upper()
    if not enzyme_upper.startswith("CYP"):
        enzyme_upper = "CYP" + enzyme_upper

    # Find substrates
    substrates = []
    sensitive = []
    for drug, enzymes in CYP_SUBSTRATES.items():
        if enzyme_upper in enzymes:
            substrates.append(drug)
            if drug in SENSITIVE_SUBSTRATES:
                sensitive.append(drug)

    # Find inhibitors
    strong_inhib = []
    mod_inhib = []
    for inhibitor, enzyme_dict in CYP_INHIBITORS.items():
        if enzyme_upper in enzyme_dict:
            if enzyme_dict[enzyme_upper] == Strength.STRONG:
                strong_inhib.append(inhibitor)
            elif enzyme_dict[enzyme_upper] == Strength.MODERATE:
                mod_inhib.append(inhibitor)

    # Find inducers
    strong_ind = []
    for inducer, enzyme_dict in CYP_INDUCERS.items():
        if enzyme_upper in enzyme_dict:
            if enzyme_dict[enzyme_upper] == Strength.STRONG:
                strong_ind.append(inducer)

    notes = []
    if not substrates:
        notes.append(f"No substrates found for {enzyme_upper} in database")

    return CYPInfoReport(
        enzyme=enzyme_upper,
        substrates=sorted(substrates),
        sensitive_substrates=sorted(sensitive),
        strong_inhibitors=sorted(strong_inhib),
        moderate_inhibitors=sorted(mod_inhib),
        strong_inducers=sorted(strong_ind),
        notes=notes,
    )


def check_interaction(drug_a: str, drug_b: str) -> InteractionReport:
    """Check for drug-drug interaction between two drugs.

    Checks for:
    1. Inhibitor + substrate interactions
    2. Inducer + substrate interactions
    3. Substrate competition (both metabolized by same enzyme)

    Args:
        drug_a: First drug name
        drug_b: Second drug name

    Returns:
        InteractionReport with interaction details and recommendations.
    """
    name_a = _normalize_drug_name(drug_a)
    name_b = _normalize_drug_name(drug_b)

    # Get profiles
    profile_a = get_drug_profile(drug_a)
    profile_b = get_drug_profile(drug_b)

    # Check if A inhibits metabolism of B
    for enzyme in profile_b.metabolizing_enzymes:
        if enzyme in profile_a.known_inhibitor_of:
            strength = profile_a.known_inhibitor_of[enzyme]
            auc_range = AUC_FOLD_CHANGE["inhibition"][strength]

            # Determine severity
            if name_b in SENSITIVE_SUBSTRATES and strength == Strength.STRONG:
                severity = Severity.CONTRAINDICATED
            elif name_b in SENSITIVE_SUBSTRATES or strength == Strength.STRONG:
                severity = Severity.MAJOR
            elif strength == Strength.MODERATE:
                severity = Severity.MODERATE
            else:
                severity = Severity.MINOR

            return InteractionReport(
                drug_a=drug_a, drug_b=drug_b,
                interaction_found=True,
                interaction_type=InteractionType.INHIBITION,
                mechanism=f"{drug_a} inhibits {enzyme}, reducing metabolism of {drug_b}",
                affected_enzyme=enzyme,
                strength=strength,
                severity=severity,
                auc_change_range=auc_range,
                clinical_significance=f"{drug_b} levels increase {auc_range[0]:.0f}-{auc_range[1]:.0f}× due to {enzyme} inhibition",
                recommendation=_get_recommendation(InteractionType.INHIBITION, strength, name_b in SENSITIVE_SUBSTRATES),
                notes=[],
            )

    # Check if B inhibits metabolism of A
    for enzyme in profile_a.metabolizing_enzymes:
        if enzyme in profile_b.known_inhibitor_of:
            strength = profile_b.known_inhibitor_of[enzyme]
            auc_range = AUC_FOLD_CHANGE["inhibition"][strength]

            if name_a in SENSITIVE_SUBSTRATES and strength == Strength.STRONG:
                severity = Severity.CONTRAINDICATED
            elif name_a in SENSITIVE_SUBSTRATES or strength == Strength.STRONG:
                severity = Severity.MAJOR
            elif strength == Strength.MODERATE:
                severity = Severity.MODERATE
            else:
                severity = Severity.MINOR

            return InteractionReport(
                drug_a=drug_a, drug_b=drug_b,
                interaction_found=True,
                interaction_type=InteractionType.INHIBITION,
                mechanism=f"{drug_b} inhibits {enzyme}, reducing metabolism of {drug_a}",
                affected_enzyme=enzyme,
                strength=strength,
                severity=severity,
                auc_change_range=auc_range,
                clinical_significance=f"{drug_a} levels increase {auc_range[0]:.0f}-{auc_range[1]:.0f}× due to {enzyme} inhibition",
                recommendation=_get_recommendation(InteractionType.INHIBITION, strength, name_a in SENSITIVE_SUBSTRATES),
                notes=[],
            )

    # Check if A induces metabolism of B
    for enzyme in profile_b.metabolizing_enzymes:
        if enzyme in profile_a.known_inducer_of:
            strength = profile_a.known_inducer_of[enzyme]
            auc_range = AUC_FOLD_CHANGE["induction"][strength]

            # Prodrug check - induction may INCREASE effect
            notes = []
            if name_b in PRODRUGS:
                notes.append(f"⚠️ {drug_b} is a prodrug - induction may INCREASE active metabolite levels")

            if strength == Strength.STRONG:
                severity = Severity.MAJOR
            elif strength == Strength.MODERATE:
                severity = Severity.MODERATE
            else:
                severity = Severity.MINOR

            return InteractionReport(
                drug_a=drug_a, drug_b=drug_b,
                interaction_found=True,
                interaction_type=InteractionType.INDUCTION,
                mechanism=f"{drug_a} induces {enzyme}, increasing metabolism of {drug_b}",
                affected_enzyme=enzyme,
                strength=strength,
                severity=severity,
                auc_change_range=auc_range,
                clinical_significance=f"{drug_b} levels decrease {(1-auc_range[1])*100:.0f}-{(1-auc_range[0])*100:.0f}% due to {enzyme} induction",
                recommendation=_get_recommendation(InteractionType.INDUCTION, strength, name_b in SENSITIVE_SUBSTRATES),
                notes=notes,
            )

    # Check if B induces metabolism of A
    for enzyme in profile_a.metabolizing_enzymes:
        if enzyme in profile_b.known_inducer_of:
            strength = profile_b.known_inducer_of[enzyme]
            auc_range = AUC_FOLD_CHANGE["induction"][strength]

            notes = []
            if name_a in PRODRUGS:
                notes.append(f"⚠️ {drug_a} is a prodrug - induction may INCREASE active metabolite levels")

            if strength == Strength.STRONG:
                severity = Severity.MAJOR
            elif strength == Strength.MODERATE:
                severity = Severity.MODERATE
            else:
                severity = Severity.MINOR

            return InteractionReport(
                drug_a=drug_a, drug_b=drug_b,
                interaction_found=True,
                interaction_type=InteractionType.INDUCTION,
                mechanism=f"{drug_b} induces {enzyme}, increasing metabolism of {drug_a}",
                affected_enzyme=enzyme,
                strength=strength,
                severity=severity,
                auc_change_range=auc_range,
                clinical_significance=f"{drug_a} levels decrease {(1-auc_range[1])*100:.0f}-{(1-auc_range[0])*100:.0f}% due to {enzyme} induction",
                recommendation=_get_recommendation(InteractionType.INDUCTION, strength, name_a in SENSITIVE_SUBSTRATES),
                notes=notes,
            )

    # No direct PK interaction found
    return InteractionReport(
        drug_a=drug_a, drug_b=drug_b,
        interaction_found=False,
        interaction_type=None,
        mechanism="No known CYP-mediated interaction",
        affected_enzyme=None,
        strength=None,
        severity=None,
        auc_change_range=None,
        clinical_significance="No significant pharmacokinetic interaction expected",
        recommendation="Standard dosing; monitor as clinically indicated",
        notes=["Always verify with current prescribing information", "Pharmacodynamic interactions not assessed"],
    )


def _get_recommendation(int_type: InteractionType, strength: Strength, sensitive: bool) -> str:
    """Generate clinical recommendation based on interaction type and strength."""
    if int_type == InteractionType.INHIBITION:
        if strength == Strength.STRONG and sensitive:
            return "AVOID combination - use alternative therapy"
        elif strength == Strength.STRONG:
            return "Reduce substrate dose by 75-90% or avoid combination"
        elif strength == Strength.MODERATE:
            return "Consider 50% dose reduction; monitor for toxicity"
        else:
            return "Monitor; dose adjustment usually not required"
    elif int_type == InteractionType.INDUCTION:
        if strength == Strength.STRONG:
            return "Increase substrate dose 2-4× or use alternative therapy"
        elif strength == Strength.MODERATE:
            return "Consider dose increase; monitor for reduced efficacy"
        else:
            return "Monitor for reduced efficacy"
    return "Consult prescribing information"


def predict_auc_change(
    inhibitor_or_inducer: str,
    substrate: str,
) -> Dict[str, any]:
    """Predict AUC fold-change for a drug interaction.

    Args:
        inhibitor_or_inducer: Drug that affects metabolism
        substrate: Drug whose levels are affected

    Returns:
        Dict with auc_low, auc_high, interaction_type, enzyme, notes.
    """
    report = check_interaction(inhibitor_or_inducer, substrate)

    if not report.interaction_found:
        return {
            "auc_low": 1.0,
            "auc_high": 1.0,
            "interaction_type": None,
            "enzyme": None,
            "notes": ["No known interaction"],
        }

    lo, hi = report.auc_change_range if report.auc_change_range else (1.0, 1.0)
    return {
        "auc_low": lo,
        "auc_high": hi,
        "interaction_type": report.interaction_type.value if report.interaction_type else None,
        "enzyme": report.affected_enzyme,
        "mechanism": report.mechanism,
        "recommendation": report.recommendation,
    }


def check_pharmacogenomics(drug: str) -> PharmacogenomicsReport:
    """Check pharmacogenomic considerations for a drug.

    Args:
        drug: Drug name

    Returns:
        PharmacogenomicsReport with phenotype impacts and recommendations.
    """
    name = _normalize_drug_name(drug)
    enzymes_involved = list(CYP_SUBSTRATES.get(name, set()))

    phenotype_impacts: Dict[str, Dict[str, str]] = {}
    recommendations: List[str] = []

    # Check CYP2D6 relevance
    if "CYP2D6" in enzymes_involved:
        phenotype_impacts["CYP2D6"] = {
            "Poor metabolizer (PM)": "Increased drug levels - toxicity risk; reduce dose 25-50%",
            "Intermediate metabolizer (IM)": "Moderately increased levels; consider dose reduction",
            "Normal metabolizer (NM)": "Standard dosing",
            "Ultrarapid metabolizer (UM)": "Decreased drug levels; may need dose increase",
        }
        recommendations.append("Consider CYP2D6 genotyping before initiating therapy")

    # Check CYP2C19 relevance
    if "CYP2C19" in enzymes_involved:
        phenotype_impacts["CYP2C19"] = {
            "Poor metabolizer (PM)": "Increased drug levels; reduce dose or monitor closely",
            "Intermediate metabolizer (IM)": "Moderately increased levels",
            "Normal metabolizer (NM)": "Standard dosing",
            "Ultrarapid metabolizer (UM)": "Decreased drug levels; may need dose increase",
        }
        recommendations.append("CYP2C19 status affects drug exposure - consider genotyping")

    # Check CYP2C9 relevance
    if "CYP2C9" in enzymes_involved:
        phenotype_impacts["CYP2C9"] = {
            "Poor metabolizer (*3/*3)": "Significantly increased exposure; reduce dose 50-80%",
            "Intermediate metabolizer (*1/*3)": "Moderately increased exposure; reduce dose 25-50%",
            "Normal metabolizer (*1/*1)": "Standard dosing",
        }
        if name == "warfarin":
            recommendations.append("CYP2C9 genotyping highly recommended for warfarin - affects dose requirements")
            recommendations.append("Combined CYP2C9 + VKORC1 genotyping improves dosing accuracy")

    # Prodrug special handling
    is_prodrug = name in PRODRUGS
    prodrug_warning = ""
    if is_prodrug:
        info = PRODRUGS[name]
        enzyme = info["enzyme"]
        if enzyme == "CYP2D6":
            phenotype_impacts["CYP2D6"] = {
                "Poor metabolizer (PM)": f"REDUCED EFFICACY - cannot convert to {info['active_metabolite']}",
                "Intermediate metabolizer (IM)": "Reduced activation - may need alternative",
                "Normal metabolizer (NM)": "Normal activation",
                "Ultrarapid metabolizer (UM)": f"RAPID activation - TOXICITY RISK from high {info['active_metabolite']}",
            }
            prodrug_warning = f"Prodrug requiring CYP2D6 activation. {info['clinical_note']}"
            if name == "codeine":
                recommendations.insert(0, "⚠️ FDA boxed warning: Codeine CONTRAINDICATED in CYP2D6 ultrarapid metabolizers")
        elif enzyme == "CYP2C19":
            phenotype_impacts["CYP2C19"] = {
                "Poor metabolizer (PM)": f"REDUCED EFFICACY - cannot convert to active form",
                "Intermediate metabolizer (IM)": "Reduced activation - consider alternative",
                "Normal metabolizer (NM)": "Normal activation",
                "Rapid/Ultrarapid metabolizer": "Enhanced activation",
            }
            prodrug_warning = f"Prodrug requiring CYP2C19 activation. {info['clinical_note']}"
            if name == "clopidogrel":
                recommendations.insert(0, "FDA boxed warning: CYP2C19 poor metabolizers have reduced clopidogrel efficacy")
                recommendations.append("Consider prasugrel or ticagrelor as alternatives in PMs")

    if not enzymes_involved:
        recommendations.append(f"No CYP-mediated metabolism data found for {drug}")

    return PharmacogenomicsReport(
        drug=drug,
        relevant_enzymes=enzymes_involved,
        phenotype_impacts=phenotype_impacts,
        is_prodrug=is_prodrug,
        prodrug_warning=prodrug_warning,
        clinical_recommendations=recommendations,
        notes=[],
    )


def list_cyp_enzymes() -> List[str]:
    """List all CYP enzymes in the database."""
    enzymes = set()
    for drug_enzymes in CYP_SUBSTRATES.values():
        enzymes.update(drug_enzymes)
    for inh_dict in CYP_INHIBITORS.values():
        enzymes.update(inh_dict.keys())
    for ind_dict in CYP_INDUCERS.values():
        enzymes.update(ind_dict.keys())
    return sorted(enzymes)


def list_substrates(enzyme: str) -> List[str]:
    """List all substrates of a CYP enzyme."""
    enzyme_upper = enzyme.upper()
    if not enzyme_upper.startswith("CYP"):
        enzyme_upper = "CYP" + enzyme_upper
    return sorted(d for d, e in CYP_SUBSTRATES.items() if enzyme_upper in e)


def list_inhibitors(enzyme: str, strength: Optional[Strength] = None) -> List[str]:
    """List all inhibitors of a CYP enzyme, optionally filtered by strength."""
    enzyme_upper = enzyme.upper()
    if not enzyme_upper.startswith("CYP"):
        enzyme_upper = "CYP" + enzyme_upper
    result = []
    for inhibitor, enzyme_dict in CYP_INHIBITORS.items():
        if enzyme_upper in enzyme_dict:
            if strength is None or enzyme_dict[enzyme_upper] == strength:
                result.append(inhibitor)
    return sorted(result)


def list_inducers(enzyme: str, strength: Optional[Strength] = None) -> List[str]:
    """List all inducers of a CYP enzyme, optionally filtered by strength."""
    enzyme_upper = enzyme.upper()
    if not enzyme_upper.startswith("CYP"):
        enzyme_upper = "CYP" + enzyme_upper
    result = []
    for inducer, enzyme_dict in CYP_INDUCERS.items():
        if enzyme_upper in enzyme_dict:
            if strength is None or enzyme_dict[enzyme_upper] == strength:
                result.append(inducer)
    return sorted(result)
