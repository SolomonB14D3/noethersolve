"""
noethersolve.pharmacokinetics — Pharmacogenomic CYP interaction checker.

Checks drug lists for CYP enzyme-mediated interactions, pharmacogenomic
phenotype risks, and HLA-associated adverse reactions. Knowledge tables
are derived from the FDA Table of Pharmacogenomic Biomarkers.

Catches:
  - Strong inhibitor + substrate on same CYP enzyme (HIGH)
  - Moderate inhibitor + substrate on same CYP enzyme (MODERATE)
  - Strong inducer + substrate on same CYP enzyme (MODERATE)
  - Poor/ultrarapid metabolizer phenotype affecting prescribed drugs (HIGH/MODERATE)
  - HLA allele + drug combinations requiring mandatory pre-screening (HIGH)

Usage:
    from noethersolve.pharmacokinetics import audit_drug_list

    report = audit_drug_list(
        drugs=["codeine", "paroxetine", "simvastatin", "clarithromycin"],
        hla_alleles=["HLA-B*57:01"],
        phenotypes={"CYP2D6": "poor_metabolizer"},
    )
    print(report)
    # Shows interactions, phenotype warnings, HLA warnings, and verdict

    # Individual checks:
    from noethersolve.pharmacokinetics import (
        check_drug_interactions,
        check_phenotype,
        check_hla,
        get_enzyme_for_drug,
        get_interactions,
    )

    interactions = check_drug_interactions(["codeine", "paroxetine"])
    pheno = check_phenotype("CYP2D6", "poor_metabolizer", ["codeine", "tamoxifen"])
    hla = check_hla(["HLA-B*57:01"], ["abacavir", "metformin"])
    enzymes = get_enzyme_for_drug("codeine")        # ["CYP2D6"]
    info = get_interactions("codeine")              # substrates/inhibitors/inducers
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set


# ─── Knowledge tables ────────────────────────────────────────────────────────

# Major CYP enzymes and their key substrates
_CYP_SUBSTRATES: Dict[str, Set[str]] = {
    "CYP2D6": {"codeine", "tramadol", "tamoxifen", "metoprolol", "dextromethorphan",
                "fluoxetine", "paroxetine", "venlafaxine", "atomoxetine", "risperidone",
                "aripiprazole", "nortriptyline", "oxycodone", "hydrocodone"},
    "CYP3A4": {"midazolam", "cyclosporine", "tacrolimus", "simvastatin", "atorvastatin",
                "nifedipine", "erythromycin", "fentanyl", "carbamazepine", "apixaban",
                "rivaroxaban", "ibrutinib", "paxlovid", "olaparib"},
    "CYP2C19": {"clopidogrel", "omeprazole", "esomeprazole", "voriconazole",
                 "citalopram", "escitalopram", "diazepam", "phenytoin"},
    "CYP2C9": {"warfarin", "phenytoin", "losartan", "celecoxib", "fluvastatin",
                "tolbutamide", "glipizide"},
    "CYP1A2": {"theophylline", "caffeine", "clozapine", "tizanidine", "duloxetine",
                "melatonin", "olanzapine"},
}

# Strong inhibitors (cause >5x increase in AUC of substrate)
_STRONG_INHIBITORS: Dict[str, Set[str]] = {
    "CYP2D6": {"paroxetine", "fluoxetine", "bupropion", "quinidine"},
    "CYP3A4": {"ketoconazole", "itraconazole", "clarithromycin", "ritonavir",
                "cobicistat", "posaconazole"},
    "CYP2C19": {"fluconazole", "fluvoxamine", "ticlopidine"},
    "CYP2C9": {"fluconazole", "amiodarone"},
    "CYP1A2": {"fluvoxamine", "ciprofloxacin"},
}

# Moderate inhibitors (cause 2-5x increase in AUC)
_MODERATE_INHIBITORS: Dict[str, Set[str]] = {
    "CYP2D6": {"duloxetine", "sertraline", "terbinafine"},
    "CYP3A4": {"erythromycin", "fluconazole", "diltiazem", "verapamil",
                "grapefruit_juice", "aprepitant"},
    "CYP2C19": {"omeprazole", "esomeprazole"},
    "CYP2C9": {"miconazole"},
    "CYP1A2": set(),
}

# Strong inducers (cause >80% decrease in AUC)
_STRONG_INDUCERS: Dict[str, Set[str]] = {
    "CYP2D6": set(),  # CYP2D6 is not significantly inducible
    "CYP3A4": {"rifampin", "phenytoin", "carbamazepine", "st_johns_wort",
                "phenobarbital", "enzalutamide"},
    "CYP2C19": {"rifampin", "st_johns_wort"},
    "CYP2C9": {"rifampin"},
    "CYP1A2": {"smoking", "rifampin", "phenytoin", "carbamazepine"},
}

# Pharmacogenomic phenotypes and their clinical impact
_PHENOTYPE_IMPACT: Dict[str, Dict[str, str]] = {
    "CYP2D6": {
        "poor_metabolizer": "Accumulates active drug; reduce dose or avoid. Codeine: NO analgesic effect (can't convert to morphine). Tamoxifen: reduced efficacy.",
        "ultrarapid_metabolizer": "Rapid conversion; risk of toxicity with prodrugs (codeine -> morphine excess). May need higher doses of active drugs.",
        "intermediate_metabolizer": "Reduced metabolism; consider dose reduction for drugs with narrow therapeutic index.",
    },
    "CYP2C19": {
        "poor_metabolizer": "Clopidogrel: NO antiplatelet effect (prodrug requires activation). PPIs: increased exposure, may benefit.",
        "ultrarapid_metabolizer": "PPIs: reduced efficacy. Clopidogrel: enhanced activation, increased bleeding risk.",
        "intermediate_metabolizer": "Clopidogrel: reduced activation, consider alternative antiplatelet.",
    },
    "CYP2C9": {
        "poor_metabolizer": "Warfarin: requires 50-80% dose reduction. NSAIDs: increased GI bleeding risk.",
        "intermediate_metabolizer": "Warfarin: requires 20-40% dose reduction.",
    },
}

# HLA associations (critical pharmacogenomic safety tests)
_HLA_ASSOCIATIONS: Dict[str, Dict[str, str]] = {
    "HLA-B*57:01": {"abacavir": "Hypersensitivity reaction -- MANDATORY pre-screening"},
    "HLA-B*15:02": {"carbamazepine": "Stevens-Johnson syndrome / toxic epidermal necrolysis in SE Asian ancestry",
                     "phenytoin": "SJS/TEN risk"},
    "HLA-B*58:01": {"allopurinol": "Severe cutaneous adverse reactions"},
    "HLA-A*31:01": {"carbamazepine": "Drug reaction with eosinophilia and systemic symptoms (DRESS)"},
}


# ─── Dataclasses ──────────────────────────────────────────────────────────────

@dataclass
class DrugInteraction:
    """A single CYP-mediated drug-drug interaction."""
    drug_a: str              # the inhibitor or inducer
    drug_b: str              # the substrate affected
    enzyme: str              # CYP enzyme involved
    interaction_type: str    # "inhibition" or "induction"
    severity: str            # HIGH, MODERATE
    mechanism: str           # human-readable explanation

    def __str__(self):
        return (f"  [{self.severity}] {self.drug_a} {self.interaction_type} of "
                f"{self.enzyme} affects {self.drug_b}: {self.mechanism}")


@dataclass
class PharmIssue:
    """A single pharmacogenomic issue (phenotype or HLA warning)."""
    check_name: str          # PHENOTYPE, HLA
    severity: str            # HIGH, MODERATE, LOW, INFO
    message: str
    suggestion: str

    def __str__(self):
        return f"  [{self.severity}] {self.check_name}: {self.message}"


@dataclass
class InteractionReport:
    """Result of check_drug_interactions()."""
    drugs: List[str]
    interactions: List[DrugInteraction]
    verdict: str

    def __str__(self):
        lines = []
        lines.append(f"{'=' * 60}")
        lines.append(f"  Drug Interaction Check: {self.verdict}")
        lines.append(f"{'=' * 60}")
        lines.append(f"  Drugs: {', '.join(self.drugs)}")
        lines.append(f"")
        if self.interactions:
            lines.append(f"  Interactions ({len(self.interactions)}):")
            severity_order = {"HIGH": 0, "MODERATE": 1, "LOW": 2, "INFO": 3}
            for ix in sorted(self.interactions, key=lambda i: severity_order.get(i.severity, 4)):
                lines.append(str(ix))
            lines.append(f"")
        else:
            lines.append(f"  No CYP-mediated interactions detected.")
            lines.append(f"")
        n_high = sum(1 for i in self.interactions if i.severity == "HIGH")
        n_moderate = sum(1 for i in self.interactions if i.severity == "MODERATE")
        lines.append(f"  Summary: {n_high} HIGH, {n_moderate} MODERATE")
        lines.append(f"{'=' * 60}")
        return "\n".join(lines)

    @property
    def passed(self) -> bool:
        return self.verdict == "PASS"


@dataclass
class PhenotypeReport:
    """Result of check_phenotype()."""
    enzyme: str
    phenotype: str
    drugs: List[str]
    affected_drugs: List[str]
    warnings: List[PharmIssue]
    clinical_impact: str
    verdict: str

    def __str__(self):
        lines = []
        lines.append(f"{'=' * 60}")
        lines.append(f"  Phenotype Check: {self.verdict}")
        lines.append(f"{'=' * 60}")
        lines.append(f"  Enzyme: {self.enzyme}, Phenotype: {self.phenotype}")
        lines.append(f"  Drugs checked: {', '.join(self.drugs)}")
        lines.append(f"")
        if self.affected_drugs:
            lines.append(f"  Affected drugs: {', '.join(self.affected_drugs)}")
            lines.append(f"  Clinical impact: {self.clinical_impact}")
            lines.append(f"")
        if self.warnings:
            for w in self.warnings:
                lines.append(str(w))
                if w.suggestion:
                    lines.append(f"    -> {w.suggestion}")
            lines.append(f"")
        else:
            lines.append(f"  No phenotype-related concerns for these drugs.")
            lines.append(f"")
        lines.append(f"{'=' * 60}")
        return "\n".join(lines)

    @property
    def passed(self) -> bool:
        return self.verdict == "PASS"


@dataclass
class HLAReport:
    """Result of check_hla()."""
    hla_alleles: List[str]
    drugs: List[str]
    warnings: List[PharmIssue]
    required_testing: List[str]
    verdict: str

    def __str__(self):
        lines = []
        lines.append(f"{'=' * 60}")
        lines.append(f"  HLA Safety Check: {self.verdict}")
        lines.append(f"{'=' * 60}")
        lines.append(f"  HLA alleles: {', '.join(self.hla_alleles)}")
        lines.append(f"  Drugs checked: {', '.join(self.drugs)}")
        lines.append(f"")
        if self.warnings:
            for w in self.warnings:
                lines.append(str(w))
                if w.suggestion:
                    lines.append(f"    -> {w.suggestion}")
            lines.append(f"")
        if self.required_testing:
            lines.append(f"  Required pre-screening tests:")
            for t in self.required_testing:
                lines.append(f"    - {t}")
            lines.append(f"")
        if not self.warnings and not self.required_testing:
            lines.append(f"  No HLA-drug concerns detected.")
            lines.append(f"")
        lines.append(f"{'=' * 60}")
        return "\n".join(lines)

    @property
    def passed(self) -> bool:
        return self.verdict == "PASS"


@dataclass
class PharmReport:
    """Comprehensive pharmacogenomic audit result."""
    drugs: List[str]
    interactions: List[DrugInteraction]
    phenotype_warnings: List[PharmIssue]
    hla_warnings: List[PharmIssue]
    required_testing: List[str]
    verdict: str
    issues: List = field(default_factory=list)  # all combined PharmIssue + DrugInteraction

    def __str__(self):
        lines = []
        lines.append(f"{'=' * 60}")
        lines.append(f"  Pharmacogenomic Audit: {self.verdict}")
        lines.append(f"{'=' * 60}")
        lines.append(f"  Drugs: {', '.join(self.drugs)}")
        lines.append(f"")

        # Drug interactions
        if self.interactions:
            lines.append(f"  Drug-Drug Interactions ({len(self.interactions)}):")
            severity_order = {"HIGH": 0, "MODERATE": 1, "LOW": 2, "INFO": 3}
            for ix in sorted(self.interactions, key=lambda i: severity_order.get(i.severity, 4)):
                lines.append(str(ix))
            lines.append(f"")

        # Phenotype warnings
        if self.phenotype_warnings:
            lines.append(f"  Phenotype Warnings ({len(self.phenotype_warnings)}):")
            for w in self.phenotype_warnings:
                lines.append(str(w))
                if w.suggestion:
                    lines.append(f"    -> {w.suggestion}")
            lines.append(f"")

        # HLA warnings
        if self.hla_warnings:
            lines.append(f"  HLA Warnings ({len(self.hla_warnings)}):")
            for w in self.hla_warnings:
                lines.append(str(w))
                if w.suggestion:
                    lines.append(f"    -> {w.suggestion}")
            lines.append(f"")

        # Required testing
        if self.required_testing:
            lines.append(f"  Required Pre-Screening Tests:")
            for t in self.required_testing:
                lines.append(f"    - {t}")
            lines.append(f"")

        # No issues
        if not self.interactions and not self.phenotype_warnings and not self.hla_warnings:
            lines.append(f"  No pharmacogenomic concerns detected.")
            lines.append(f"")

        # Summary counts
        all_severities = (
            [i.severity for i in self.interactions]
            + [w.severity for w in self.phenotype_warnings]
            + [w.severity for w in self.hla_warnings]
        )
        n_high = sum(1 for s in all_severities if s == "HIGH")
        n_moderate = sum(1 for s in all_severities if s == "MODERATE")
        n_info = sum(1 for s in all_severities if s == "INFO")
        lines.append(f"  Summary: {n_high} HIGH, {n_moderate} MODERATE, "
                     f"{n_info} INFO")
        lines.append(f"{'=' * 60}")
        return "\n".join(lines)

    @property
    def passed(self) -> bool:
        return self.verdict == "PASS"


# ─── Helper functions ─────────────────────────────────────────────────────────

def _normalize_drug(name: str) -> str:
    """Normalize a drug name to lowercase with underscores."""
    return name.strip().lower().replace(" ", "_").replace("-", "_")


def _build_substrate_index() -> Dict[str, List[str]]:
    """Build reverse index: drug -> list of CYP enzymes that metabolize it."""
    index: Dict[str, List[str]] = {}
    for enzyme, substrates in _CYP_SUBSTRATES.items():
        for drug in substrates:
            index.setdefault(drug, []).append(enzyme)
    return index


_SUBSTRATE_INDEX = _build_substrate_index()


def _build_inhibitor_index() -> Dict[str, List[tuple]]:
    """Build reverse index: drug -> list of (enzyme, strength) tuples."""
    index: Dict[str, List[tuple]] = {}
    for enzyme, drugs in _STRONG_INHIBITORS.items():
        for drug in drugs:
            index.setdefault(drug, []).append((enzyme, "strong"))
    for enzyme, drugs in _MODERATE_INHIBITORS.items():
        for drug in drugs:
            index.setdefault(drug, []).append((enzyme, "moderate"))
    return index


_INHIBITOR_INDEX = _build_inhibitor_index()


def _build_inducer_index() -> Dict[str, List[tuple]]:
    """Build reverse index: drug -> list of (enzyme, strength) tuples."""
    index: Dict[str, List[tuple]] = {}
    for enzyme, drugs in _STRONG_INDUCERS.items():
        for drug in drugs:
            index.setdefault(drug, []).append((enzyme, "strong"))
    return index


_INDUCER_INDEX = _build_inducer_index()


# ─── Public utility functions ─────────────────────────────────────────────────

def get_enzyme_for_drug(drug: str) -> List[str]:
    """Return which CYP enzymes metabolize this drug.

    Args:
        drug: Drug name (case-insensitive, spaces/hyphens OK).

    Returns:
        List of CYP enzyme names (e.g. ["CYP2D6", "CYP3A4"]).
        Empty list if the drug is not in the substrate tables.
    """
    name = _normalize_drug(drug)
    return list(_SUBSTRATE_INDEX.get(name, []))


def get_interactions(drug: str) -> Dict[str, object]:
    """Return inhibitors, inducers, and substrates for this drug's enzymes.

    Args:
        drug: Drug name (case-insensitive, spaces/hyphens OK).

    Returns:
        Dict with keys:
          - "metabolized_by": list of CYP enzymes
          - "inhibits": list of (enzyme, strength) tuples
          - "induces": list of (enzyme, strength) tuples
          - "co_substrates": dict of enzyme -> set of other substrates
    """
    name = _normalize_drug(drug)
    enzymes = _SUBSTRATE_INDEX.get(name, [])
    co_substrates = {}
    for enz in enzymes:
        others = _CYP_SUBSTRATES.get(enz, set()) - {name}
        if others:
            co_substrates[enz] = others

    return {
        "metabolized_by": list(enzymes),
        "inhibits": list(_INHIBITOR_INDEX.get(name, [])),
        "induces": list(_INDUCER_INDEX.get(name, [])),
        "co_substrates": co_substrates,
    }


# ─── Core check functions ────────────────────────────────────────────────────

def check_drug_interactions(drugs: List[str]) -> InteractionReport:
    """Check a drug list for CYP enzyme-mediated drug-drug interactions.

    For each pair of drugs, checks whether one inhibits or induces the CYP
    enzyme that metabolizes the other. A drug that is both a substrate and
    an inhibitor of the same enzyme (e.g. paroxetine is a CYP2D6 substrate
    AND strong CYP2D6 inhibitor) will be flagged when paired with another
    substrate of that enzyme.

    Args:
        drugs: List of drug names (lowercase, underscores for spaces).

    Returns:
        InteractionReport with all detected interactions and verdict.
        Verdict: FAIL if any HIGH, WARN if any MODERATE, PASS otherwise.
    """
    normalized = [_normalize_drug(d) for d in drugs]
    interactions: List[DrugInteraction] = []
    seen = set()

    for i, drug_a in enumerate(normalized):
        for j, drug_b in enumerate(normalized):
            if i == j:
                continue

            # Check if drug_a inhibits an enzyme that metabolizes drug_b
            for enzyme, strength in _INHIBITOR_INDEX.get(drug_a, []):
                if drug_b in _CYP_SUBSTRATES.get(enzyme, set()):
                    key = (drug_a, drug_b, enzyme, "inhibition")
                    if key not in seen:
                        seen.add(key)
                        severity = "HIGH" if strength == "strong" else "MODERATE"
                        mechanism = (
                            f"{drug_a} is a {strength} {enzyme} inhibitor; "
                            f"{drug_b} is a {enzyme} substrate. "
                            f"Expected >{('5' if strength == 'strong' else '2')}x "
                            f"increase in {drug_b} exposure (AUC)."
                        )
                        interactions.append(DrugInteraction(
                            drug_a=drug_a,
                            drug_b=drug_b,
                            enzyme=enzyme,
                            interaction_type="inhibition",
                            severity=severity,
                            mechanism=mechanism,
                        ))

            # Check if drug_a induces an enzyme that metabolizes drug_b
            for enzyme, strength in _INDUCER_INDEX.get(drug_a, []):
                if drug_b in _CYP_SUBSTRATES.get(enzyme, set()):
                    key = (drug_a, drug_b, enzyme, "induction")
                    if key not in seen:
                        seen.add(key)
                        mechanism = (
                            f"{drug_a} is a {strength} {enzyme} inducer; "
                            f"{drug_b} is a {enzyme} substrate. "
                            f"Expected >80% decrease in {drug_b} exposure (AUC)."
                        )
                        interactions.append(DrugInteraction(
                            drug_a=drug_a,
                            drug_b=drug_b,
                            enzyme=enzyme,
                            interaction_type="induction",
                            severity="MODERATE",
                            mechanism=mechanism,
                        ))

    # Verdict
    has_high = any(ix.severity == "HIGH" for ix in interactions)
    has_moderate = any(ix.severity == "MODERATE" for ix in interactions)
    if has_high:
        verdict = "FAIL"
    elif has_moderate:
        verdict = "WARN"
    else:
        verdict = "PASS"

    return InteractionReport(
        drugs=normalized,
        interactions=interactions,
        verdict=verdict,
    )


def check_phenotype(enzyme: str, phenotype: str, drugs: List[str]) -> PhenotypeReport:
    """Check which drugs are affected by a CYP metabolizer phenotype.

    Args:
        enzyme: CYP enzyme name (e.g. "CYP2D6").
        phenotype: Metabolizer phenotype (e.g. "poor_metabolizer",
                   "intermediate_metabolizer", "ultrarapid_metabolizer").
        drugs: List of drug names to check.

    Returns:
        PhenotypeReport with affected drugs, clinical impact, and verdict.
        Verdict: FAIL if poor/ultrarapid + affected drugs, WARN if
        intermediate + affected drugs, PASS otherwise.
    """
    normalized = [_normalize_drug(d) for d in drugs]
    enzyme = enzyme.upper() if not enzyme.startswith("CYP") else enzyme
    phenotype = phenotype.lower().replace(" ", "_")

    substrates = _CYP_SUBSTRATES.get(enzyme, set())
    affected = [d for d in normalized if d in substrates]

    # Get clinical impact text
    enzyme_impacts = _PHENOTYPE_IMPACT.get(enzyme, {})
    clinical_impact = enzyme_impacts.get(phenotype, "")

    warnings: List[PharmIssue] = []
    if affected and clinical_impact:
        # Determine severity based on phenotype
        if phenotype in ("poor_metabolizer", "ultrarapid_metabolizer"):
            severity = "HIGH"
        else:
            severity = "MODERATE"

        for drug in affected:
            warnings.append(PharmIssue(
                check_name="PHENOTYPE",
                severity=severity,
                message=(f"{drug} is a {enzyme} substrate; patient is "
                         f"{phenotype.replace('_', ' ')}. {clinical_impact}"),
                suggestion=(f"Consider dose adjustment or alternative drug "
                            f"not metabolized by {enzyme}."),
            ))
    elif affected and not clinical_impact:
        # Enzyme/phenotype combo not in our impact table
        for drug in affected:
            warnings.append(PharmIssue(
                check_name="PHENOTYPE",
                severity="LOW",
                message=(f"{drug} is a {enzyme} substrate; patient is "
                         f"{phenotype.replace('_', ' ')}. Clinical impact "
                         f"data not available in table."),
                suggestion="Consult pharmacogenomic reference for guidance.",
            ))

    # Verdict
    has_high = any(w.severity == "HIGH" for w in warnings)
    has_moderate = any(w.severity == "MODERATE" for w in warnings)
    if has_high:
        verdict = "FAIL"
    elif has_moderate:
        verdict = "WARN"
    else:
        verdict = "PASS"

    return PhenotypeReport(
        enzyme=enzyme,
        phenotype=phenotype,
        drugs=normalized,
        affected_drugs=affected,
        warnings=warnings,
        clinical_impact=clinical_impact,
        verdict=verdict,
    )


def check_hla(hla_alleles: List[str], drugs: List[str]) -> HLAReport:
    """Check for HLA allele-drug combinations requiring pre-screening.

    Any match between a patient's HLA alleles and a prescribed drug that has
    a known HLA-associated adverse reaction triggers a HIGH severity warning.

    Args:
        hla_alleles: List of HLA allele designations (e.g. ["HLA-B*57:01"]).
        drugs: List of drug names to check.

    Returns:
        HLAReport with warnings, required testing, and verdict.
        Verdict: FAIL if any HLA-drug match found, PASS otherwise.
    """
    normalized_drugs = [_normalize_drug(d) for d in drugs]
    warnings: List[PharmIssue] = []
    required_testing: List[str] = []

    for allele in hla_alleles:
        drug_risks = _HLA_ASSOCIATIONS.get(allele, {})
        for drug in normalized_drugs:
            if drug in drug_risks:
                risk_desc = drug_risks[drug]
                warnings.append(PharmIssue(
                    check_name="HLA",
                    severity="HIGH",
                    message=(f"{allele} + {drug}: {risk_desc}"),
                    suggestion=f"Do NOT prescribe {drug} without {allele} pre-screening.",
                ))
                test_str = f"{allele} screening before {drug} administration"
                if test_str not in required_testing:
                    required_testing.append(test_str)

    # Also flag drugs that have ANY known HLA association, even if the
    # patient's alleles weren't provided for that specific one
    all_hla_drugs: Dict[str, List[str]] = {}
    for allele, drug_map in _HLA_ASSOCIATIONS.items():
        for drug in drug_map:
            all_hla_drugs.setdefault(drug, []).append(allele)

    patient_allele_set = set(hla_alleles)
    for drug in normalized_drugs:
        if drug in all_hla_drugs:
            missing_alleles = [a for a in all_hla_drugs[drug]
                               if a not in patient_allele_set]
            for allele in missing_alleles:
                risk_desc = _HLA_ASSOCIATIONS[allele][drug]
                test_str = f"{allele} screening before {drug} administration"
                if test_str not in required_testing:
                    required_testing.append(test_str)
                    warnings.append(PharmIssue(
                        check_name="HLA",
                        severity="INFO",
                        message=(f"{drug} has known {allele} association "
                                 f"({risk_desc}). Allele status not provided."),
                        suggestion=f"Consider {allele} testing if not already done.",
                    ))

    # Verdict: FAIL only on confirmed allele+drug HIGH matches
    has_high = any(w.severity == "HIGH" for w in warnings)
    verdict = "FAIL" if has_high else "PASS"

    return HLAReport(
        hla_alleles=list(hla_alleles),
        drugs=normalized_drugs,
        warnings=warnings,
        required_testing=required_testing,
        verdict=verdict,
    )


# ─── Comprehensive audit ─────────────────────────────────────────────────────

def audit_drug_list(
    drugs: List[str],
    hla_alleles: Optional[List[str]] = None,
    phenotypes: Optional[Dict[str, str]] = None,
) -> PharmReport:
    """Comprehensive pharmacogenomic audit of a drug list.

    Runs all three checks (drug interactions, phenotype, HLA) and combines
    the results into a single report.

    Args:
        drugs: List of drug names (case-insensitive, spaces/hyphens OK).
        hla_alleles: Optional list of patient HLA alleles for safety check.
        phenotypes: Optional dict mapping CYP enzyme to phenotype, e.g.
                    {"CYP2D6": "poor_metabolizer", "CYP2C19": "intermediate_metabolizer"}.

    Returns:
        PharmReport with all interactions, warnings, required testing, and
        overall verdict. FAIL if any HIGH, WARN if any MODERATE, PASS otherwise.
    """
    if hla_alleles is None:
        hla_alleles = []
    if phenotypes is None:
        phenotypes = {}

    # 1. Drug-drug interactions
    interaction_report = check_drug_interactions(drugs)
    all_interactions = interaction_report.interactions

    # 2. Phenotype checks
    all_phenotype_warnings: List[PharmIssue] = []
    normalized_drugs = [_normalize_drug(d) for d in drugs]
    for enzyme, phenotype in phenotypes.items():
        pheno_report = check_phenotype(enzyme, phenotype, drugs)
        all_phenotype_warnings.extend(pheno_report.warnings)

    # 3. HLA checks
    all_hla_warnings: List[PharmIssue] = []
    all_required_testing: List[str] = []
    if hla_alleles:
        hla_report = check_hla(hla_alleles, drugs)
        all_hla_warnings = hla_report.warnings
        all_required_testing = hla_report.required_testing
    else:
        # Still check if any drugs have known HLA associations
        all_hla_drugs: Dict[str, List[str]] = {}
        for allele, drug_map in _HLA_ASSOCIATIONS.items():
            for drug in drug_map:
                all_hla_drugs.setdefault(drug, []).append(allele)
        for drug in normalized_drugs:
            if drug in all_hla_drugs:
                for allele in all_hla_drugs[drug]:
                    risk_desc = _HLA_ASSOCIATIONS[allele][drug]
                    test_str = f"{allele} screening before {drug} administration"
                    if test_str not in all_required_testing:
                        all_required_testing.append(test_str)
                        all_hla_warnings.append(PharmIssue(
                            check_name="HLA",
                            severity="INFO",
                            message=(f"{drug} has known {allele} association "
                                     f"({risk_desc}). No HLA alleles provided."),
                            suggestion=f"Consider {allele} testing before prescribing {drug}.",
                        ))

    # Combine all issues
    all_issues: List = []
    all_issues.extend(all_interactions)
    all_issues.extend(all_phenotype_warnings)
    all_issues.extend(all_hla_warnings)

    # Overall verdict
    all_severities = (
        [i.severity for i in all_interactions]
        + [w.severity for w in all_phenotype_warnings]
        + [w.severity for w in all_hla_warnings]
    )
    has_high = any(s == "HIGH" for s in all_severities)
    has_moderate = any(s == "MODERATE" for s in all_severities)
    if has_high:
        verdict = "FAIL"
    elif has_moderate:
        verdict = "WARN"
    else:
        verdict = "PASS"

    return PharmReport(
        drugs=normalized_drugs,
        interactions=all_interactions,
        phenotype_warnings=all_phenotype_warnings,
        hla_warnings=all_hla_warnings,
        required_testing=all_required_testing,
        verdict=verdict,
        issues=all_issues,
    )
