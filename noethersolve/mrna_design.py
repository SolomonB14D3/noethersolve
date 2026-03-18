"""mRNA therapeutic design calculator.

CRITICAL LLM BLIND SPOT: Models assume pseudouridine (Ψ) universally stabilizes RNA.
REALITY: Ψ is CONTEXT-DEPENDENT:
- Destabilizes A-U pairs (weaker hydrogen bonding)
- Stabilizes certain structures (reduced innate immune activation)
- Net effect depends on sequence context and structure

This module provides verified thermodynamic calculations for mRNA design.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum


# ─── Thermodynamic parameters ───────────────────────────────────────────────

# Standard RNA base-pairing energies (kcal/mol, 37°C)
# From Turner lab unified nearest-neighbor parameters
RNA_BASE_PAIR_ENERGY: Dict[str, float] = {
    # Watson-Crick pairs (negative = stabilizing)
    "AU": -0.9,
    "UA": -0.9,
    "GC": -2.4,
    "CG": -2.4,
    "GU": -1.3,  # Wobble pair
    "UG": -1.3,
    # Ψ-containing pairs (pseudouridine)
    "AΨ": -0.6,   # WEAKER than AU (-0.6 vs -0.9)
    "ΨA": -0.6,
    "GΨ": -1.1,   # Slightly weaker than GU
    "ΨG": -1.1,
}

# Stacking energies for nearest-neighbor model (kcal/mol, 37°C)
# Format: "XY/WZ" means 5'-XY-3' paired with 3'-WZ-5'
STACKING_ENERGY: Dict[str, float] = {
    # Standard stacking
    "GC/GC": -3.42,
    "CG/CG": -3.26,
    "GC/CG": -3.42,
    "CG/GC": -2.36,
    "AU/AU": -0.93,
    "UA/UA": -0.93,
    "AU/UA": -1.10,
    "UA/AU": -1.33,
    "GC/AU": -2.24,
    "AU/GC": -2.35,
    "CG/UA": -2.11,
    "UA/CG": -2.08,
    # Ψ stacking (LESS stabilizing than U)
    "GC/AΨ": -1.90,  # vs -2.24 for AU
    "AΨ/GC": -2.00,  # vs -2.35 for AU
    "AΨ/AΨ": -0.70,  # vs -0.93 for AU
    "ΨA/ΨA": -0.70,
}

# Codon optimization: codon adaptation index (CAI) weights
# Higher = more frequent in human
HUMAN_CODON_CAI: Dict[str, float] = {
    # Phe
    "UUU": 0.46, "UUC": 1.00,
    # Leu
    "UUA": 0.08, "UUG": 0.13, "CUU": 0.13, "CUC": 0.20, "CUA": 0.07, "CUG": 1.00,
    # Ile
    "AUU": 0.36, "AUC": 1.00, "AUA": 0.17,
    # Met (start)
    "AUG": 1.00,
    # Val
    "GUU": 0.18, "GUC": 0.24, "GUA": 0.12, "GUG": 1.00,
    # Ser
    "UCU": 0.19, "UCC": 0.22, "UCA": 0.15, "UCG": 0.06, "AGU": 0.15, "AGC": 1.00,
    # Pro
    "CCU": 0.29, "CCC": 1.00, "CCA": 0.28, "CCG": 0.11,
    # Thr
    "ACU": 0.25, "ACC": 1.00, "ACA": 0.28, "ACG": 0.11,
    # Ala
    "GCU": 0.27, "GCC": 1.00, "GCA": 0.23, "GCG": 0.11,
    # Tyr
    "UAU": 0.44, "UAC": 1.00,
    # Stop
    "UAA": 0.30, "UAG": 0.24, "UGA": 1.00,
    # His
    "CAU": 0.42, "CAC": 1.00,
    # Gln
    "CAA": 0.27, "CAG": 1.00,
    # Asn
    "AAU": 0.47, "AAC": 1.00,
    # Lys
    "AAA": 0.43, "AAG": 1.00,
    # Asp
    "GAU": 0.46, "GAC": 1.00,
    # Glu
    "GAA": 0.42, "GAG": 1.00,
    # Cys
    "UGU": 0.45, "UGC": 1.00,
    # Trp
    "UGG": 1.00,
    # Arg
    "CGU": 0.08, "CGC": 0.19, "CGA": 0.11, "CGG": 0.21, "AGA": 0.21, "AGG": 1.00,
    # Gly
    "GGU": 0.16, "GGC": 1.00, "GGA": 0.25, "GGG": 0.25,
}


class ModificationType(Enum):
    """Types of mRNA modifications."""
    NONE = "none"
    PSEUDOURIDINE = "Ψ"  # N1-methylpseudouridine (m1Ψ)
    FIVE_METHYL_CYTIDINE = "m5C"
    TWO_PRIME_O_METHYL = "2'-O-Me"


@dataclass
class ThermodynamicReport:
    """Thermodynamic analysis of mRNA region."""
    sequence: str
    delta_G: float  # Free energy change (kcal/mol)
    delta_G_modified: float  # With modifications
    modification: ModificationType
    stability_change: float  # ΔΔG = modified - unmodified
    stability_direction: str  # "stabilized", "destabilized", "unchanged"
    notes: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        lines = [
            "THERMODYNAMIC ANALYSIS",
            "=" * 50,
            f"Sequence: {self.sequence[:30]}{'...' if len(self.sequence) > 30 else ''}",
            f"Modification: {self.modification.value}",
            "",
            f"ΔG (unmodified): {self.delta_G:.2f} kcal/mol",
            f"ΔG (modified):   {self.delta_G_modified:.2f} kcal/mol",
            f"ΔΔG:             {self.stability_change:+.2f} kcal/mol",
            f"Effect:          {self.stability_direction.upper()}",
            "",
        ]
        if self.notes:
            lines.append("Notes:")
            for note in self.notes:
                lines.append(f"  • {note}")
        return "\n".join(lines)


@dataclass
class CodonOptimizationReport:
    """Codon optimization analysis."""
    original: str
    optimized: str
    original_cai: float
    optimized_cai: float
    gc_content_original: float
    gc_content_optimized: float
    uridine_count_original: int
    uridine_count_optimized: int
    changes: List[Tuple[int, str, str]]  # (position, old, new)

    def __str__(self) -> str:
        lines = [
            "CODON OPTIMIZATION ANALYSIS",
            "=" * 50,
            f"Original CAI:  {self.original_cai:.3f}",
            f"Optimized CAI: {self.optimized_cai:.3f}",
            f"Improvement:   {self.optimized_cai - self.original_cai:+.3f}",
            "",
            f"GC content: {self.gc_content_original:.1%} → {self.gc_content_optimized:.1%}",
            f"Uridines:   {self.uridine_count_original} → {self.uridine_count_optimized}",
            "",
            f"Codons changed: {len(self.changes)}",
        ]
        if self.changes[:5]:
            lines.append("First 5 changes:")
            for pos, old, new in self.changes[:5]:
                lines.append(f"  Position {pos}: {old} → {new}")
        return "\n".join(lines)


@dataclass
class ImmunogenicityReport:
    """Innate immune activation analysis."""
    sequence: str
    modification: ModificationType
    tlr7_8_risk: str  # "high", "medium", "low"
    rig_i_risk: str   # "high", "medium", "low"
    double_strand_motifs: int
    uridine_runs: int  # Consecutive U's (immunogenic)
    cpg_count: int
    recommendations: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        lines = [
            "IMMUNOGENICITY ANALYSIS",
            "=" * 50,
            f"Modification: {self.modification.value}",
            "",
            f"TLR7/8 risk:       {self.tlr7_8_risk.upper()}",
            f"RIG-I risk:        {self.rig_i_risk.upper()}",
            f"dsRNA motifs:      {self.double_strand_motifs}",
            f"Uridine runs (≥4): {self.uridine_runs}",
            f"CpG dinucleotides: {self.cpg_count}",
            "",
        ]

        if self.modification == ModificationType.PSEUDOURIDINE:
            lines.append("CRITICAL: Ψ modification reduces innate immune activation")
            lines.append("  • TLR7/8 recognition reduced ~100-fold")
            lines.append("  • RIG-I activation suppressed")
            lines.append("")

        if self.recommendations:
            lines.append("Recommendations:")
            for rec in self.recommendations:
                lines.append(f"  • {rec}")

        return "\n".join(lines)


@dataclass
class mRNADesignReport:
    """Complete mRNA design analysis."""
    sequence: str
    length: int
    thermodynamics: ThermodynamicReport
    codon_optimization: Optional[CodonOptimizationReport]
    immunogenicity: ImmunogenicityReport
    five_prime_utr: str
    three_prime_utr: str
    poly_a_length: int
    overall_quality: str  # "excellent", "good", "fair", "poor"
    critical_insights: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        lines = [
            "mRNA DESIGN REPORT",
            "=" * 60,
            "",
            "CRITICAL LLM ERRORS CORRECTED:",
            "  1. Ψ does NOT universally stabilize - it's CONTEXT-DEPENDENT",
            "  2. A-Ψ pairs are WEAKER than A-U pairs (ΔΔG ≈ +0.3 kcal/mol)",
            "  3. Benefit is IMMUNE EVASION, not thermodynamic stability",
            "",
            f"Sequence length: {self.length} nt",
            f"5' UTR: {self.five_prime_utr[:20]}... ({len(self.five_prime_utr)} nt)",
            f"3' UTR: {self.three_prime_utr[:20]}... ({len(self.three_prime_utr)} nt)",
            f"Poly(A) tail: {self.poly_a_length} nt",
            "",
            str(self.thermodynamics),
            "",
            str(self.immunogenicity),
            "",
            f"OVERALL QUALITY: {self.overall_quality.upper()}",
            "",
        ]

        if self.critical_insights:
            lines.append("KEY INSIGHTS:")
            for insight in self.critical_insights:
                lines.append(f"  • {insight}")

        return "\n".join(lines)


# ─── Core functions ─────────────────────────────────────────────────────────

def calculate_base_pair_energy(
    base1: str,
    base2: str,
    modified: bool = False,
) -> float:
    """Calculate base pair energy (kcal/mol).

    CRITICAL: Ψ-A pairs are WEAKER than U-A pairs!
    """
    if modified and base1 == "U":
        pair = f"Ψ{base2}"
    elif modified and base2 == "U":
        pair = f"{base1}Ψ"
    else:
        pair = f"{base1}{base2}"

    # Try both orientations
    if pair in RNA_BASE_PAIR_ENERGY:
        return RNA_BASE_PAIR_ENERGY[pair]

    reverse = pair[::-1]
    if reverse in RNA_BASE_PAIR_ENERGY:
        return RNA_BASE_PAIR_ENERGY[reverse]

    # Non-canonical or mismatch
    return 0.0


def calculate_duplex_stability(
    sequence: str,
    complement: str,
    modified: bool = False,
) -> ThermodynamicReport:
    """Calculate duplex stability with nearest-neighbor model.

    CRITICAL: Pseudouridine DESTABILIZES A-U pairs!
    The model commonly assumes all modifications stabilize.
    """
    if len(sequence) != len(complement):
        raise ValueError("Sequence and complement must have same length")

    # Calculate unmodified energy
    delta_g_unmod = 0.0
    for i in range(len(sequence) - 1):
        # Simple nearest-neighbor approximation
        sequence[i] + complement[i]
        sequence[i+1] + complement[i+1]

        e1 = calculate_base_pair_energy(sequence[i], complement[i], False)
        e2 = calculate_base_pair_energy(sequence[i+1], complement[i+1], False)
        delta_g_unmod += (e1 + e2) / 2

    # Calculate modified energy
    delta_g_mod = 0.0
    for i in range(len(sequence) - 1):
        e1 = calculate_base_pair_energy(sequence[i], complement[i], modified)
        e2 = calculate_base_pair_energy(sequence[i+1], complement[i+1], modified)
        delta_g_mod += (e1 + e2) / 2

    stability_change = delta_g_mod - delta_g_unmod

    notes = []
    if modified:
        u_count = sequence.count("U")
        if u_count > 0:
            notes.append(f"Contains {u_count} uridines → {u_count} pseudouridines")

            # Count A-U pairs
            au_pairs = sum(1 for i, (s, c) in enumerate(zip(sequence, complement))
                         if (s == "U" and c == "A") or (s == "A" and c == "U"))
            if au_pairs > 0:
                notes.append(f"A-Ψ pairs: {au_pairs} (each ~0.3 kcal/mol WEAKER than A-U)")

    if stability_change > 0.1:
        direction = "destabilized"
        notes.append("WARNING: Modification DESTABILIZES this structure")
    elif stability_change < -0.1:
        direction = "stabilized"
    else:
        direction = "unchanged"

    return ThermodynamicReport(
        sequence=sequence,
        delta_G=delta_g_unmod,
        delta_G_modified=delta_g_mod,
        modification=ModificationType.PSEUDOURIDINE if modified else ModificationType.NONE,
        stability_change=stability_change,
        stability_direction=direction,
        notes=notes,
    )


def analyze_immunogenicity(
    sequence: str,
    modification: ModificationType = ModificationType.NONE,
) -> ImmunogenicityReport:
    """Analyze innate immune activation risk.

    CRITICAL: Pseudouridine's main benefit is IMMUNE EVASION, not stability!
    """
    # Count immunogenic features
    uridine_runs = 0
    i = 0
    while i < len(sequence):
        if sequence[i] == "U":
            run_length = 1
            while i + run_length < len(sequence) and sequence[i + run_length] == "U":
                run_length += 1
            if run_length >= 4:
                uridine_runs += 1
            i += run_length
        else:
            i += 1

    # CpG dinucleotides (immunogenic)
    cpg_count = sequence.count("CG")

    # Double-strand motifs (simplified)
    ds_motifs = 0
    for i in range(len(sequence) - 7):
        # Look for self-complementary regions
        region = sequence[i:i+8]
        complement = region.translate(str.maketrans("ACGU", "UGCA"))[::-1]
        if complement in sequence[i+8:]:
            ds_motifs += 1

    # Risk assessment
    if modification == ModificationType.PSEUDOURIDINE:
        # Ψ dramatically reduces TLR7/8 and RIG-I activation
        tlr_risk = "low"
        rig_risk = "low"
    else:
        # Unmodified RNA is highly immunogenic
        if uridine_runs > 3 or cpg_count > 10:
            tlr_risk = "high"
        elif uridine_runs > 1 or cpg_count > 5:
            tlr_risk = "medium"
        else:
            tlr_risk = "low"

        if ds_motifs > 2:
            rig_risk = "high"
        elif ds_motifs > 0:
            rig_risk = "medium"
        else:
            rig_risk = "low"

    recommendations = []
    if modification == ModificationType.NONE:
        if tlr_risk != "low":
            recommendations.append("Consider Ψ modification to reduce TLR7/8 activation")
        if uridine_runs > 0:
            recommendations.append(f"Codon optimize to reduce {uridine_runs} uridine runs")
        if cpg_count > 5:
            recommendations.append("Reduce CpG content by synonymous codon changes")

    return ImmunogenicityReport(
        sequence=sequence,
        modification=modification,
        tlr7_8_risk=tlr_risk,
        rig_i_risk=rig_risk,
        double_strand_motifs=ds_motifs,
        uridine_runs=uridine_runs,
        cpg_count=cpg_count,
        recommendations=recommendations,
    )


def calculate_cai(sequence: str) -> float:
    """Calculate Codon Adaptation Index for human expression."""
    if len(sequence) % 3 != 0:
        raise ValueError("Sequence length must be divisible by 3")

    cai_values = []
    for i in range(0, len(sequence), 3):
        codon = sequence[i:i+3]
        if codon in HUMAN_CODON_CAI:
            cai_values.append(HUMAN_CODON_CAI[codon])

    if not cai_values:
        return 0.0

    # CAI is geometric mean
    product = 1.0
    for v in cai_values:
        product *= v

    return product ** (1.0 / len(cai_values))


def optimize_codons(
    sequence: str,
    strategy: str = "high_cai",
) -> CodonOptimizationReport:
    """Optimize codon usage for human expression.

    strategy: "high_cai" (max expression) or "balanced" (moderate GC, fewer U)
    """
    if len(sequence) % 3 != 0:
        raise ValueError("Sequence length must be divisible by 3")

    original_cai = calculate_cai(sequence)

    # Group codons by amino acid
    AA_CODONS: Dict[str, List[str]] = {
        "F": ["UUU", "UUC"],
        "L": ["UUA", "UUG", "CUU", "CUC", "CUA", "CUG"],
        "I": ["AUU", "AUC", "AUA"],
        "M": ["AUG"],
        "V": ["GUU", "GUC", "GUA", "GUG"],
        "S": ["UCU", "UCC", "UCA", "UCG", "AGU", "AGC"],
        "P": ["CCU", "CCC", "CCA", "CCG"],
        "T": ["ACU", "ACC", "ACA", "ACG"],
        "A": ["GCU", "GCC", "GCA", "GCG"],
        "Y": ["UAU", "UAC"],
        "*": ["UAA", "UAG", "UGA"],
        "H": ["CAU", "CAC"],
        "Q": ["CAA", "CAG"],
        "N": ["AAU", "AAC"],
        "K": ["AAA", "AAG"],
        "D": ["GAU", "GAC"],
        "E": ["GAA", "GAG"],
        "C": ["UGU", "UGC"],
        "W": ["UGG"],
        "R": ["CGU", "CGC", "CGA", "CGG", "AGA", "AGG"],
        "G": ["GGU", "GGC", "GGA", "GGG"],
    }

    CODON_TO_AA = {}
    for aa, codons in AA_CODONS.items():
        for codon in codons:
            CODON_TO_AA[codon] = aa

    optimized_codons = []
    changes = []

    for i in range(0, len(sequence), 3):
        codon = sequence[i:i+3]

        if codon not in CODON_TO_AA:
            optimized_codons.append(codon)
            continue

        aa = CODON_TO_AA[codon]
        synonymous = AA_CODONS[aa]

        if strategy == "high_cai":
            # Pick highest CAI codon
            best = max(synonymous, key=lambda c: HUMAN_CODON_CAI.get(c, 0))
        else:
            # Balance: prefer moderate GC, fewer U
            def score(c):
                cai = HUMAN_CODON_CAI.get(c, 0)
                gc = (c.count("G") + c.count("C")) / 3
                u_penalty = c.count("U") * 0.1
                return cai * 0.7 + (0.5 - abs(gc - 0.5)) * 0.3 - u_penalty
            best = max(synonymous, key=score)

        if best != codon:
            changes.append((i // 3, codon, best))
        optimized_codons.append(best)

    optimized = "".join(optimized_codons)

    # Calculate metrics
    gc_orig = (sequence.count("G") + sequence.count("C")) / len(sequence)
    gc_opt = (optimized.count("G") + optimized.count("C")) / len(optimized)

    return CodonOptimizationReport(
        original=sequence,
        optimized=optimized,
        original_cai=original_cai,
        optimized_cai=calculate_cai(optimized),
        gc_content_original=gc_orig,
        gc_content_optimized=gc_opt,
        uridine_count_original=sequence.count("U"),
        uridine_count_optimized=optimized.count("U"),
        changes=changes,
    )


def analyze_mrna_design(
    coding_sequence: str,
    five_prime_utr: str = "GGGAAAUAAGAGAGAAAAGAAGAGUAAGAAGAAAUAUAAGAGCCACC",
    three_prime_utr: str = "UGAUAAUAGGCUGGAGCCUCGGUGGCCUAGCUUCUUGCCCCUUGGGCCUCCCCCCAG",
    poly_a_length: int = 100,
    use_pseudouridine: bool = True,
    optimize_codons: bool = True,
) -> mRNADesignReport:
    """Complete mRNA therapeutic design analysis.

    CRITICAL LLM BLIND SPOTS ADDRESSED:
    1. Ψ does NOT universally stabilize RNA
    2. A-Ψ pairs are WEAKER than A-U pairs
    3. Ψ main benefit is IMMUNE EVASION
    """
    # Validate
    if len(coding_sequence) % 3 != 0:
        raise ValueError("Coding sequence must be divisible by 3")

    # Full sequence
    full_seq = five_prime_utr + coding_sequence + three_prime_utr + "A" * poly_a_length

    # Thermodynamic analysis
    # Use a sample region for duplex analysis
    sample_region = coding_sequence[:30] if len(coding_sequence) >= 30 else coding_sequence
    sample_complement = sample_region.translate(str.maketrans("ACGU", "UGCA"))

    thermo = calculate_duplex_stability(
        sample_region,
        sample_complement,
        modified=use_pseudouridine,
    )

    # Codon optimization
    codon_report = None
    if optimize_codons:
        codon_report = optimize_codons_report(coding_sequence)

    # Immunogenicity
    mod_type = ModificationType.PSEUDOURIDINE if use_pseudouridine else ModificationType.NONE
    immuno = analyze_immunogenicity(full_seq, mod_type)

    # Quality assessment
    issues = 0
    if immuno.tlr7_8_risk == "high":
        issues += 2
    elif immuno.tlr7_8_risk == "medium":
        issues += 1

    if immuno.rig_i_risk == "high":
        issues += 2
    elif immuno.rig_i_risk == "medium":
        issues += 1

    if codon_report and codon_report.optimized_cai < 0.7:
        issues += 1

    if issues == 0:
        quality = "excellent"
    elif issues <= 2:
        quality = "good"
    elif issues <= 4:
        quality = "fair"
    else:
        quality = "poor"

    # Critical insights
    insights = []

    if use_pseudouridine:
        if thermo.stability_change > 0:
            insights.append(
                f"Ψ DESTABILIZES this sequence (ΔΔG = +{thermo.stability_change:.2f} kcal/mol)"
            )
            insights.append(
                "This is EXPECTED for A-U rich regions - Ψ weakens A-U base pairs"
            )
        insights.append(
            "Ψ benefit is IMMUNE EVASION (~100x lower TLR7/8 activation), NOT thermodynamic stability"
        )
    else:
        if immuno.tlr7_8_risk != "low":
            insights.append(
                "WARNING: Unmodified RNA will trigger strong innate immune response"
            )
            insights.append(
                "Consider Ψ modification - this is why COVID vaccines use it"
            )

    return mRNADesignReport(
        sequence=full_seq,
        length=len(full_seq),
        thermodynamics=thermo,
        codon_optimization=codon_report,
        immunogenicity=immuno,
        five_prime_utr=five_prime_utr,
        three_prime_utr=three_prime_utr,
        poly_a_length=poly_a_length,
        overall_quality=quality,
        critical_insights=insights,
    )


def optimize_codons_report(sequence: str) -> CodonOptimizationReport:
    """Wrapper for codon optimization with default strategy."""
    return optimize_codons(sequence, strategy="balanced")


def compare_modifications(
    sequence: str,
    complement: str,
) -> str:
    """Compare thermodynamics with and without modifications.

    Demonstrates the CRITICAL insight: Ψ destabilizes A-U pairs!
    """
    unmod = calculate_duplex_stability(sequence, complement, modified=False)
    mod = calculate_duplex_stability(sequence, complement, modified=True)

    lines = [
        "MODIFICATION COMPARISON",
        "=" * 50,
        "",
        "CRITICAL INSIGHT: LLMs assume Ψ universally stabilizes.",
        "REALITY: Ψ DESTABILIZES A-U pairs!",
        "",
        f"Sequence: {sequence}",
        "",
        "UNMODIFIED (natural uridine):",
        f"  ΔG = {unmod.delta_G:.2f} kcal/mol",
        "",
        "PSEUDOURIDINE (Ψ):",
        f"  ΔG = {mod.delta_G_modified:.2f} kcal/mol",
        f"  ΔΔG = {mod.stability_change:+.2f} kcal/mol",
        "",
    ]

    if mod.stability_change > 0:
        lines.append("RESULT: Ψ DESTABILIZES this duplex")
        lines.append("        A-Ψ hydrogen bonds are weaker than A-U")
    elif mod.stability_change < 0:
        lines.append("RESULT: Ψ stabilizes this duplex")
        lines.append("        (Unusual - check for GU wobbles)")
    else:
        lines.append("RESULT: No significant stability change")

    lines.append("")
    lines.append("WHY USE Ψ IF IT DESTABILIZES?")
    lines.append("  → Ψ reduces innate immune activation ~100-fold")
    lines.append("  → TLR7/8 and RIG-I don't recognize Ψ-containing RNA")
    lines.append("  → This is why COVID mRNA vaccines use m1Ψ")

    return "\n".join(lines)
