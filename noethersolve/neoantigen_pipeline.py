"""Neoantigen prediction pipeline for cancer immunotherapy.

Models consistently overweight MHC binding affinity and miss the other
critical steps in the antigen presentation pathway:

1. PROTEASOMAL CLEAVAGE — Which peptides are generated?
2. TAP TRANSPORT — Which peptides enter the ER?
3. MHC BINDING — Which peptides bind MHC? (models focus here)
4. TCR RECOGNITION — Which peptides elicit T-cell response?

A peptide must pass ALL steps to be immunogenic. MHC binding alone
is insufficient — this is a common LLM error.

This module provides scoring for each step and a combined pipeline score.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
import math


class MHCClass(Enum):
    CLASS_I = "MHC-I"   # CD8+ T cells (cytotoxic)
    CLASS_II = "MHC-II"  # CD4+ T cells (helper)


class RiskLevel(Enum):
    HIGH = "high"       # Good neoantigen candidate
    MODERATE = "moderate"
    LOW = "low"
    VERY_LOW = "very_low"


# Amino acid properties for scoring
HYDROPHOBICITY = {
    "I": 4.5, "V": 4.2, "L": 3.8, "F": 2.8, "C": 2.5, "M": 1.9, "A": 1.8,
    "G": -0.4, "T": -0.7, "S": -0.8, "W": -0.9, "Y": -1.3, "P": -1.6,
    "H": -3.2, "E": -3.5, "Q": -3.5, "D": -3.5, "N": -3.5, "K": -3.9, "R": -4.5,
}

# Proteasome cleavage preferences (C-terminal)
# Based on immunoproteasome preferences
CLEAVAGE_C_TERM = {
    "L": 1.2, "Y": 1.1, "F": 1.0, "K": 0.9, "R": 0.8, "I": 0.7, "V": 0.6,
    "A": 0.5, "M": 0.5, "W": 0.4, "H": 0.3, "T": 0.2, "S": 0.1, "N": 0.0,
    "Q": -0.1, "C": -0.2, "G": -0.3, "E": -0.4, "D": -0.5, "P": -1.0,
}

# TAP binding preferences (positions 1-3 and C-terminal matter most)
TAP_PREFERENCES = {
    "pos1": {"R": 1.0, "K": 0.8, "Y": 0.6, "F": 0.5, "L": 0.3, "I": 0.2},
    "pos2": {"L": 1.0, "I": 0.8, "V": 0.6, "F": 0.5, "M": 0.4},
    "pos3": {"K": 0.8, "R": 0.7, "Y": 0.5, "F": 0.4},
    "c_term": {"L": 1.0, "Y": 0.9, "F": 0.8, "K": 0.7, "R": 0.6, "I": 0.5, "V": 0.4},
}

# MHC-I anchor positions (typically positions 2 and C-terminus)
MHC_I_ANCHORS = {
    "HLA-A*02:01": {
        "pos2": {"L": 1.0, "M": 0.9, "V": 0.7, "I": 0.6, "A": 0.4, "T": 0.3},
        "pos9": {"V": 1.0, "L": 0.9, "I": 0.8, "A": 0.5, "T": 0.4},
    },
    "HLA-A*01:01": {
        "pos2": {"T": 1.0, "S": 0.9, "D": 0.6, "E": 0.5},
        "pos9": {"Y": 1.0, "F": 0.8, "W": 0.6},
    },
    "HLA-B*07:02": {
        "pos2": {"P": 1.0, "A": 0.5, "S": 0.4},
        "pos9": {"L": 1.0, "F": 0.8, "M": 0.6},
    },
}

# TCR-facing positions (typically positions 4-7 for 9-mers)
TCR_POSITIONS = [3, 4, 5, 6]  # 0-indexed

# Amino acids that favor TCR recognition (charged, aromatic)
TCR_FAVORABLE = {"R", "K", "D", "E", "Y", "W", "F", "H"}


@dataclass
class CleavageReport:
    """Report on proteasomal cleavage prediction."""
    peptide: str
    c_term_score: float
    n_term_score: float
    overall_cleavage_score: float
    cleavage_probability: float
    notes: str

    def __str__(self) -> str:
        lines = [
            "Proteasomal Cleavage Analysis",
            "=" * 40,
            f"Peptide: {self.peptide}",
            f"C-terminal cleavage score: {self.c_term_score:.2f}",
            f"N-terminal context score: {self.n_term_score:.2f}",
            f"Overall cleavage score: {self.overall_cleavage_score:.2f}",
            f"Cleavage probability: {self.cleavage_probability:.1%}",
            "",
            f"Note: {self.notes}",
        ]
        return "\n".join(lines)


def score_cleavage(peptide: str, n_flank: str = "", c_flank: str = "") -> CleavageReport:
    """Score proteasomal cleavage probability.

    The proteasome cleaves proteins into peptides. MHC-I peptides
    are typically 8-11 amino acids with specific C-terminal residues.

    CRITICAL: This is step 1 of 4. A peptide with poor cleavage
    will never be presented, regardless of MHC binding affinity.

    Args:
        peptide: The peptide sequence to evaluate
        n_flank: N-terminal flanking sequence (context)
        c_flank: C-terminal flanking sequence (context)

    Returns:
        CleavageReport with cleavage prediction.
    """
    peptide = peptide.upper()

    # C-terminal residue is most important
    c_term_aa = peptide[-1] if peptide else "X"
    c_term_score = CLEAVAGE_C_TERM.get(c_term_aa, 0.0)

    # N-terminal context (P1' position)
    n_term_score = 0.0
    if n_flank:
        n_flank_aa = n_flank[-1] if n_flank else "X"
        # Proline at P1' strongly inhibits cleavage
        if n_flank_aa == "P":
            n_term_score = -1.0
        else:
            n_term_score = 0.2  # Neutral contribution

    # Combined score
    overall = c_term_score + n_term_score

    # Convert to probability (sigmoid)
    probability = 1 / (1 + math.exp(-overall))

    # Notes
    if c_term_aa in ["L", "Y", "F", "K", "R"]:
        notes = f"Good C-terminal residue ({c_term_aa}) for immunoproteasome"
    elif c_term_aa == "P":
        notes = "Proline at C-terminus strongly disfavored"
    else:
        notes = f"C-terminal {c_term_aa} has moderate cleavage preference"

    return CleavageReport(
        peptide=peptide,
        c_term_score=c_term_score,
        n_term_score=n_term_score,
        overall_cleavage_score=overall,
        cleavage_probability=probability,
        notes=notes,
    )


@dataclass
class TAPReport:
    """Report on TAP transport prediction."""
    peptide: str
    tap_score: float
    transport_probability: float
    limiting_positions: List[str]
    notes: str

    def __str__(self) -> str:
        lines = [
            "TAP Transport Analysis",
            "=" * 40,
            f"Peptide: {self.peptide}",
            f"TAP binding score: {self.tap_score:.2f}",
            f"Transport probability: {self.transport_probability:.1%}",
        ]
        if self.limiting_positions:
            lines.append(f"Limiting positions: {', '.join(self.limiting_positions)}")
        lines.extend(["", f"Note: {self.notes}"])
        return "\n".join(lines)


def score_tap(peptide: str) -> TAPReport:
    """Score TAP (Transporter associated with Antigen Processing) transport.

    TAP transports peptides from cytosol to ER for MHC loading.
    TAP has strong preferences for positions 1-3 and the C-terminus.

    CRITICAL: This is step 2 of 4. Poor TAP binders never reach
    MHC molecules, regardless of theoretical MHC affinity.

    Args:
        peptide: The peptide sequence to evaluate

    Returns:
        TAPReport with transport prediction.
    """
    peptide = peptide.upper()
    n = len(peptide)

    if n < 8:
        return TAPReport(
            peptide=peptide,
            tap_score=-2.0,
            transport_probability=0.05,
            limiting_positions=["length"],
            notes="Peptide too short for TAP transport (need ≥8aa)",
        )

    scores = []
    limiting = []

    # Position 1
    aa1 = peptide[0]
    s1 = TAP_PREFERENCES["pos1"].get(aa1, 0.0)
    scores.append(s1)
    if s1 < 0.3:
        limiting.append(f"P1:{aa1}")

    # Position 2
    aa2 = peptide[1]
    s2 = TAP_PREFERENCES["pos2"].get(aa2, 0.0)
    scores.append(s2)
    if s2 < 0.3:
        limiting.append(f"P2:{aa2}")

    # Position 3
    aa3 = peptide[2]
    s3 = TAP_PREFERENCES["pos3"].get(aa3, 0.0)
    scores.append(s3)
    if s3 < 0.3:
        limiting.append(f"P3:{aa3}")

    # C-terminus (most important)
    aa_c = peptide[-1]
    s_c = TAP_PREFERENCES["c_term"].get(aa_c, 0.0)
    scores.append(s_c * 1.5)  # Weight C-term more heavily
    if s_c < 0.4:
        limiting.append(f"C-term:{aa_c}")

    tap_score = sum(scores) / len(scores)

    # Convert to probability
    transport_prob = 1 / (1 + math.exp(-2 * tap_score))

    # Notes
    if tap_score > 0.7:
        notes = "Excellent TAP binder"
    elif tap_score > 0.4:
        notes = "Good TAP binder"
    elif tap_score > 0.2:
        notes = "Moderate TAP binder — may limit presentation"
    else:
        notes = "Poor TAP binder — unlikely to be transported"

    return TAPReport(
        peptide=peptide,
        tap_score=tap_score,
        transport_probability=transport_prob,
        limiting_positions=limiting,
        notes=notes,
    )


@dataclass
class MHCBindingReport:
    """Report on MHC binding prediction."""
    peptide: str
    allele: str
    binding_score: float
    binding_affinity_nm: float  # IC50 in nM
    binding_level: str  # "strong", "weak", "non"
    anchor_scores: Dict[str, float]
    notes: str

    def __str__(self) -> str:
        lines = [
            "MHC Binding Analysis",
            "=" * 40,
            f"Peptide: {self.peptide}",
            f"Allele: {self.allele}",
            f"Binding score: {self.binding_score:.2f}",
            f"Predicted IC50: {self.binding_affinity_nm:.0f} nM",
            f"Binding level: {self.binding_level.upper()}",
            "",
            "Anchor residue scores:",
        ]
        for pos, score in self.anchor_scores.items():
            lines.append(f"  {pos}: {score:.2f}")
        lines.extend(["", f"Note: {self.notes}"])
        return "\n".join(lines)


def score_mhc_binding(
    peptide: str,
    allele: str = "HLA-A*02:01",
) -> MHCBindingReport:
    """Score MHC-I binding affinity.

    MHC molecules present peptides to T cells. Each allele has
    specific anchor preferences (typically positions 2 and C-terminus).

    CRITICAL: This is step 3 of 4. Models often ONLY consider this
    step, but a peptide must pass ALL four steps to be immunogenic.

    Args:
        peptide: The peptide sequence (typically 9-mer for MHC-I)
        allele: HLA allele (default HLA-A*02:01, most common)

    Returns:
        MHCBindingReport with binding prediction.
    """
    peptide = peptide.upper()
    n = len(peptide)

    # Use HLA-A*02:01 as default if unknown allele
    anchors = MHC_I_ANCHORS.get(allele, MHC_I_ANCHORS["HLA-A*02:01"])

    anchor_scores = {}

    # Position 2 anchor
    if n >= 2:
        aa2 = peptide[1]
        pos2_prefs = anchors.get("pos2", {})
        anchor_scores["pos2"] = pos2_prefs.get(aa2, 0.1)

    # C-terminal anchor (position 9 for 9-mer, adjust for length)
    aa_c = peptide[-1]
    pos_c_prefs = anchors.get("pos9", {})
    anchor_scores["C-term"] = pos_c_prefs.get(aa_c, 0.1)

    # Overall binding score
    binding_score = sum(anchor_scores.values()) / len(anchor_scores) if anchor_scores else 0.0

    # Convert to IC50 (nM) — rough approximation
    # Strong binders: <50 nM, Weak: 50-500 nM, Non: >500 nM
    if binding_score > 0.8:
        ic50 = 20
        level = "strong"
    elif binding_score > 0.5:
        ic50 = 100
        level = "weak"
    elif binding_score > 0.3:
        ic50 = 300
        level = "weak"
    else:
        ic50 = 1000
        level = "non"

    # Notes
    if level == "strong":
        notes = f"Strong binder to {allele} — but check other pipeline steps!"
    elif level == "weak":
        notes = f"Weak binder to {allele} — may still be presented"
    else:
        notes = f"Non-binder to {allele} — unlikely to be presented via this allele"

    return MHCBindingReport(
        peptide=peptide,
        allele=allele,
        binding_score=binding_score,
        binding_affinity_nm=ic50,
        binding_level=level,
        anchor_scores=anchor_scores,
        notes=notes,
    )


@dataclass
class TCRReport:
    """Report on TCR recognition potential."""
    peptide: str
    tcr_score: float
    immunogenicity: str
    exposed_residues: str
    foreign_score: float
    notes: str

    def __str__(self) -> str:
        lines = [
            "TCR Recognition Analysis",
            "=" * 40,
            f"Peptide: {self.peptide}",
            f"TCR-facing residues: {self.exposed_residues}",
            f"TCR recognition score: {self.tcr_score:.2f}",
            f"Foreignness score: {self.foreign_score:.2f}",
            f"Predicted immunogenicity: {self.immunogenicity.upper()}",
            "",
            f"Note: {self.notes}",
        ]
        return "\n".join(lines)


def score_tcr_recognition(
    peptide: str,
    wildtype: Optional[str] = None,
) -> TCRReport:
    """Score TCR recognition potential.

    The TCR contacts the central residues of MHC-bound peptides
    (typically positions 4-7 for 9-mers). Charged and aromatic
    residues enhance recognition.

    For neoantigens, compare to wildtype to assess "foreignness".

    CRITICAL: This is step 4 of 4. Even strong MHC binders may
    not elicit T-cell responses if they're not recognized by TCRs.

    Args:
        peptide: The peptide sequence
        wildtype: Optional wildtype sequence to compute foreignness

    Returns:
        TCRReport with TCR recognition prediction.
    """
    peptide = peptide.upper()
    n = len(peptide)

    # TCR-facing positions (adjust for length)
    if n >= 9:
        tcr_pos = [3, 4, 5, 6]  # positions 4-7 (0-indexed)
    elif n >= 8:
        tcr_pos = [2, 3, 4, 5]
    else:
        tcr_pos = list(range(1, n - 1))

    # Extract TCR-facing residues
    exposed = "".join(peptide[i] for i in tcr_pos if i < n)

    # Score based on favorable residues
    favorable_count = sum(1 for aa in exposed if aa in TCR_FAVORABLE)
    tcr_score = favorable_count / len(exposed) if exposed else 0.0

    # Foreignness score (if wildtype provided)
    foreign_score = 0.0
    if wildtype and len(wildtype) == len(peptide):
        wildtype = wildtype.upper()
        mismatches = sum(1 for a, b in zip(peptide, wildtype) if a != b)
        # Weight mutations in TCR-facing positions higher
        tcr_mismatches = sum(1 for i in tcr_pos if i < n and peptide[i] != wildtype[i])
        foreign_score = (mismatches + tcr_mismatches) / (n + len(tcr_pos))
    else:
        # Assume novel peptide
        foreign_score = 0.5

    # Combined immunogenicity
    combined = tcr_score * 0.5 + foreign_score * 0.5

    if combined > 0.6:
        immunogenicity = "high"
    elif combined > 0.4:
        immunogenicity = "moderate"
    elif combined > 0.2:
        immunogenicity = "low"
    else:
        immunogenicity = "very_low"

    # Notes
    if immunogenicity == "high":
        notes = "Good TCR recognition potential — promising neoantigen"
    elif immunogenicity == "moderate":
        notes = "Moderate TCR recognition — may need experimental validation"
    else:
        notes = "Poor TCR recognition — unlikely to elicit strong T-cell response"

    return TCRReport(
        peptide=peptide,
        tcr_score=tcr_score,
        immunogenicity=immunogenicity,
        exposed_residues=exposed,
        foreign_score=foreign_score,
        notes=notes,
    )


@dataclass
class PipelineReport:
    """Complete neoantigen pipeline report."""
    peptide: str
    allele: str
    cleavage: CleavageReport
    tap: TAPReport
    mhc: MHCBindingReport
    tcr: TCRReport
    combined_score: float
    pipeline_pass: bool
    limiting_step: str
    recommendation: str

    def __str__(self) -> str:
        lines = [
            "=" * 60,
            "NEOANTIGEN PIPELINE ASSESSMENT",
            "=" * 60,
            f"Peptide: {self.peptide}",
            f"Allele: {self.allele}",
            "",
            f"COMBINED SCORE: {self.combined_score:.2f}",
            f"Pipeline: {'PASS' if self.pipeline_pass else 'FAIL'}",
            f"Limiting step: {self.limiting_step}",
            "",
            f"Recommendation: {self.recommendation}",
            "",
            "─" * 60,
            "",
            "Step 1: PROTEASOMAL CLEAVAGE",
            f"  Score: {self.cleavage.cleavage_probability:.1%}",
            f"  Status: {'✓' if self.cleavage.cleavage_probability > 0.3 else '✗'}",
            "",
            "Step 2: TAP TRANSPORT",
            f"  Score: {self.tap.transport_probability:.1%}",
            f"  Status: {'✓' if self.tap.transport_probability > 0.3 else '✗'}",
            "",
            "Step 3: MHC BINDING",
            f"  Score: {self.mhc.binding_score:.2f} ({self.mhc.binding_level})",
            f"  Status: {'✓' if self.mhc.binding_level != 'non' else '✗'}",
            "",
            "Step 4: TCR RECOGNITION",
            f"  Score: {self.tcr.immunogenicity}",
            f"  Status: {'✓' if self.tcr.immunogenicity in ['high', 'moderate'] else '✗'}",
            "",
            "─" * 60,
            "",
            "CRITICAL: A neoantigen must pass ALL 4 steps.",
            "Models often focus only on MHC binding (Step 3) — this is WRONG!",
        ]
        return "\n".join(lines)


def evaluate_neoantigen(
    peptide: str,
    allele: str = "HLA-A*02:01",
    wildtype: Optional[str] = None,
    n_flank: str = "",
    c_flank: str = "",
) -> PipelineReport:
    """Complete neoantigen pipeline evaluation.

    Evaluates ALL 4 STEPS of antigen presentation:
    1. Proteasomal cleavage — Is this peptide generated?
    2. TAP transport — Does it reach the ER?
    3. MHC binding — Does it bind MHC?
    4. TCR recognition — Will T cells respond?

    CRITICAL: Models typically only check MHC binding (step 3).
    This is a major error — all 4 steps must pass!

    Args:
        peptide: Candidate neoantigen sequence
        allele: HLA allele for MHC binding prediction
        wildtype: Optional wildtype sequence for foreignness
        n_flank: N-terminal flanking sequence for cleavage context
        c_flank: C-terminal flanking sequence for cleavage context

    Returns:
        PipelineReport with all 4 steps and combined assessment.
    """
    # Run all 4 steps
    cleavage = score_cleavage(peptide, n_flank, c_flank)
    tap = score_tap(peptide)
    mhc = score_mhc_binding(peptide, allele)
    tcr = score_tcr_recognition(peptide, wildtype)

    # Combined score (geometric mean to penalize any weak link)
    scores = [
        cleavage.cleavage_probability,
        tap.transport_probability,
        mhc.binding_score,
        0.8 if tcr.immunogenicity in ["high", "moderate"] else 0.3,
    ]
    combined = (scores[0] * scores[1] * scores[2] * scores[3]) ** 0.25

    # Find limiting step
    step_scores = {
        "Cleavage": cleavage.cleavage_probability,
        "TAP": tap.transport_probability,
        "MHC": mhc.binding_score,
        "TCR": 0.8 if tcr.immunogenicity == "high" else 0.5 if tcr.immunogenicity == "moderate" else 0.2,
    }
    limiting = min(step_scores, key=step_scores.get)

    # Pipeline pass/fail
    pipeline_pass = all([
        cleavage.cleavage_probability > 0.3,
        tap.transport_probability > 0.3,
        mhc.binding_level != "non",
        tcr.immunogenicity in ["high", "moderate"],
    ])

    # Recommendation
    if pipeline_pass and combined > 0.6:
        rec = "STRONG candidate — proceed to experimental validation"
    elif pipeline_pass:
        rec = "Moderate candidate — consider if alternatives are limited"
    elif mhc.binding_level == "strong" and not pipeline_pass:
        rec = f"Strong MHC binder but FAILS at {limiting} — NOT recommended"
    else:
        rec = "Poor candidate — not recommended for development"

    return PipelineReport(
        peptide=peptide,
        allele=allele,
        cleavage=cleavage,
        tap=tap,
        mhc=mhc,
        tcr=tcr,
        combined_score=combined,
        pipeline_pass=pipeline_pass,
        limiting_step=limiting,
        recommendation=rec,
    )


def compare_candidates(
    peptides: List[str],
    allele: str = "HLA-A*02:01",
) -> str:
    """Compare multiple neoantigen candidates.

    Args:
        peptides: List of candidate peptide sequences
        allele: HLA allele for comparison

    Returns:
        Formatted comparison table.
    """
    results = []
    for pep in peptides:
        report = evaluate_neoantigen(pep, allele)
        results.append((pep, report.combined_score, report.pipeline_pass, report.limiting_step))

    # Sort by combined score
    results.sort(key=lambda x: -x[1])

    lines = [
        "NEOANTIGEN CANDIDATE COMPARISON",
        "=" * 60,
        f"Allele: {allele}",
        "",
        f"{'Rank':<6}{'Peptide':<15}{'Score':<8}{'Pass':<6}{'Limiting Step'}",
        "-" * 60,
    ]

    for i, (pep, score, passed, limiting) in enumerate(results, 1):
        status = "✓" if passed else "✗"
        lines.append(f"{i:<6}{pep:<15}{score:<8.2f}{status:<6}{limiting}")

    lines.extend([
        "",
        "Note: All 4 pipeline steps must pass for a good candidate.",
        "High MHC binding alone is NOT sufficient!",
    ])

    return "\n".join(lines)


# Quick test
if __name__ == "__main__":
    print("=== Neoantigen Pipeline Test ===\n")

    # Test with example peptide
    test_peptide = "YLQLVFGIEV"  # Example neoantigen sequence
    report = evaluate_neoantigen(test_peptide, "HLA-A*02:01")
    print(report)

    print("\n" + "=" * 60 + "\n")

    # Compare candidates
    candidates = ["YLQLVFGIEV", "KLLPKLDGI", "FLYNRPLSV"]
    print(compare_candidates(candidates))
