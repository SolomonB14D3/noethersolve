"""Antibody developability assessment.

Models often get antibody developability wrong because they conflate:
- Net charge vs hydrophobicity (net charge predicts viscosity, not hydrophobicity!)
- Aggregation propensity vs solubility
- Polyreactivity vs specificity

This module provides verified calculators for key developability metrics.
"""

from dataclasses import dataclass
from typing import List, Tuple
from enum import Enum


# Physical constants
K_B = 1.380649e-23  # Boltzmann constant (J/K)


class RiskLevel(Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"


# pKa values for amino acid side chains
PKA_VALUES = {
    "D": 3.9,   # Asp (acidic)
    "E": 4.1,   # Glu (acidic)
    "H": 6.0,   # His (basic)
    "K": 10.5,  # Lys (basic)
    "R": 12.5,  # Arg (basic)
    "Y": 10.1,  # Tyr (hydroxyl)
    "C": 8.3,   # Cys (thiol)
    "N_term": 9.0,   # N-terminus
    "C_term": 2.1,   # C-terminus
}


# Hydropathy index (Kyte-Doolittle scale)
HYDROPATHY = {
    "I": 4.5, "V": 4.2, "L": 3.8, "F": 2.8, "C": 2.5,
    "M": 1.9, "A": 1.8, "G": -0.4, "T": -0.7, "S": -0.8,
    "W": -0.9, "Y": -1.3, "P": -1.6, "H": -3.2, "E": -3.5,
    "Q": -3.5, "D": -3.5, "N": -3.5, "K": -3.9, "R": -4.5,
}


# Aggregation propensity (Tango-derived scale)
AGGREGATION_PROPENSITY = {
    "I": 1.0, "V": 0.9, "L": 0.8, "F": 0.7, "Y": 0.4,
    "W": 0.3, "M": 0.3, "A": 0.2, "C": 0.2, "T": 0.1,
    "S": 0.0, "G": -0.1, "N": -0.2, "Q": -0.2, "H": -0.3,
    "P": -0.5, "D": -0.7, "E": -0.7, "K": -1.0, "R": -1.0,
}


# Deamidation liability: NG, NS, NT, DG, DS motifs
DEAMIDATION_MOTIFS = ["NG", "NS", "NT", "DG", "DS"]
DEAMIDATION_WEIGHTS = {"NG": 1.0, "NS": 0.5, "NT": 0.3, "DG": 0.4, "DS": 0.2}


# Oxidation-sensitive residues
OXIDATION_RESIDUES = {"M", "W", "C"}


# Glycosylation motifs (N-linked: N-X-S/T where X ≠ P)
def find_glycosylation_sites(seq: str) -> List[int]:
    """Find N-linked glycosylation motifs (N-X-S/T, X ≠ P)."""
    sites = []
    for i in range(len(seq) - 2):
        if seq[i] == "N" and seq[i+1] != "P" and seq[i+2] in ("S", "T"):
            sites.append(i)
    return sites


@dataclass
class ChargeReport:
    """Report on antibody charge properties."""
    sequence: str
    net_charge_pH7: float
    net_charge_pH5: float  # Formulation pH
    positive_patches: int
    negative_patches: int
    asymmetry_index: float  # Spatial charge asymmetry
    viscosity_risk: RiskLevel
    isoelectric_point: float

    def __str__(self) -> str:
        lines = [
            "Antibody Charge Analysis",
            "=" * 50,
            "",
            f"Net charge at pH 7.4: {self.net_charge_pH7:+.1f}",
            f"Net charge at pH 5.0: {self.net_charge_pH5:+.1f}",
            f"Isoelectric point (pI): {self.isoelectric_point:.1f}",
            "",
            f"Positive patches (K/R clusters): {self.positive_patches}",
            f"Negative patches (D/E clusters): {self.negative_patches}",
            f"Charge asymmetry index: {self.asymmetry_index:.2f}",
            "",
            f"Viscosity risk: {self.viscosity_risk.value.upper()}",
            "",
            "CRITICAL INSIGHT: Net charge (not hydrophobicity!) predicts viscosity.",
            "  High positive charge → electrostatic repulsion → low viscosity",
            "  Near-neutral charge → aggregation → HIGH viscosity",
        ]
        return "\n".join(lines)


def calc_charge_at_ph(seq: str, ph: float) -> float:
    """Calculate net charge of a sequence at given pH.

    Uses Henderson-Hasselbalch equation:
    For acidic groups: charge = -1 / (1 + 10^(pKa - pH))
    For basic groups: charge = +1 / (1 + 10^(pH - pKa))
    """
    charge = 0.0

    # N-terminus (basic)
    pKa = PKA_VALUES["N_term"]
    charge += 1 / (1 + 10**(ph - pKa))

    # C-terminus (acidic)
    pKa = PKA_VALUES["C_term"]
    charge -= 1 / (1 + 10**(pKa - ph))

    # Side chains
    for aa in seq:
        if aa == "D":
            charge -= 1 / (1 + 10**(PKA_VALUES["D"] - ph))
        elif aa == "E":
            charge -= 1 / (1 + 10**(PKA_VALUES["E"] - ph))
        elif aa == "K":
            charge += 1 / (1 + 10**(ph - PKA_VALUES["K"]))
        elif aa == "R":
            charge += 1 / (1 + 10**(ph - PKA_VALUES["R"]))
        elif aa == "H":
            charge += 1 / (1 + 10**(ph - PKA_VALUES["H"]))
        elif aa == "Y":
            charge -= 1 / (1 + 10**(PKA_VALUES["Y"] - ph))
        elif aa == "C":
            charge -= 1 / (1 + 10**(PKA_VALUES["C"] - ph))

    return charge


def estimate_pI(seq: str) -> float:
    """Estimate isoelectric point by bisection."""
    low, high = 0.0, 14.0

    for _ in range(50):
        mid = (low + high) / 2
        charge = calc_charge_at_ph(seq, mid)
        if abs(charge) < 0.01:
            return mid
        if charge > 0:
            low = mid
        else:
            high = mid

    return (low + high) / 2


def count_patches(seq: str, residues: str, window: int = 5, threshold: int = 3) -> int:
    """Count charged patches (clusters of charged residues)."""
    patches = 0
    i = 0
    while i < len(seq) - window:
        window_seq = seq[i:i+window]
        count = sum(1 for aa in window_seq if aa in residues)
        if count >= threshold:
            patches += 1
            i += window  # Skip past this patch
        else:
            i += 1
    return patches


def analyze_charge(seq: str) -> ChargeReport:
    """Analyze antibody charge properties.

    CRITICAL: Net charge (not hydrophobicity) predicts viscosity!
    This is a common LLM error — models often cite hydrophobicity
    as the key determinant, but electrostatic interactions dominate.

    Args:
        seq: Amino acid sequence (single letter code)

    Returns:
        ChargeReport with viscosity risk assessment.
    """
    seq = seq.upper().replace(" ", "")

    net_charge_pH7 = calc_charge_at_ph(seq, 7.4)
    net_charge_pH5 = calc_charge_at_ph(seq, 5.0)
    pI = estimate_pI(seq)

    positive_patches = count_patches(seq, "KR")
    negative_patches = count_patches(seq, "DE")

    # Charge asymmetry: difference between N-terminal and C-terminal halves
    mid = len(seq) // 2
    n_term_charge = calc_charge_at_ph(seq[:mid], 7.4)
    c_term_charge = calc_charge_at_ph(seq[mid:], 7.4)
    asymmetry = abs(n_term_charge - c_term_charge) / max(abs(net_charge_pH7), 1)

    # Viscosity risk based on net charge at formulation pH (~5.0)
    # High positive charge → low viscosity (repulsion)
    # Near-neutral → high viscosity (aggregation)
    abs_charge = abs(net_charge_pH5)
    if abs_charge > 10:
        risk = RiskLevel.LOW
    elif abs_charge > 5:
        risk = RiskLevel.MODERATE
    elif abs_charge > 2:
        risk = RiskLevel.HIGH
    else:
        risk = RiskLevel.VERY_HIGH

    return ChargeReport(
        sequence=seq[:20] + "..." if len(seq) > 20 else seq,
        net_charge_pH7=net_charge_pH7,
        net_charge_pH5=net_charge_pH5,
        positive_patches=positive_patches,
        negative_patches=negative_patches,
        asymmetry_index=asymmetry,
        viscosity_risk=risk,
        isoelectric_point=pI,
    )


@dataclass
class AggregationReport:
    """Report on aggregation propensity."""
    sequence: str
    aggregation_score: float  # Higher = more aggregation-prone
    hydrophobicity_score: float
    hotspot_regions: List[Tuple[int, int]]  # (start, end)
    aggregation_risk: RiskLevel

    def __str__(self) -> str:
        lines = [
            "Antibody Aggregation Analysis",
            "=" * 50,
            "",
            f"Aggregation score: {self.aggregation_score:.2f}",
            f"Hydrophobicity (mean Kyte-Doolittle): {self.hydrophobicity_score:.2f}",
            f"Risk level: {self.aggregation_risk.value.upper()}",
            "",
        ]

        if self.hotspot_regions:
            lines.append(f"Hotspot regions ({len(self.hotspot_regions)}):")
            for start, end in self.hotspot_regions[:5]:
                lines.append(f"  Residues {start+1}-{end+1}")
        else:
            lines.append("No aggregation hotspots detected.")

        lines.extend([
            "",
            "Note: Aggregation is driven by exposed hydrophobic patches,",
            "not overall hydrophobicity. CDR loops often contain hotspots.",
        ])

        return "\n".join(lines)


def analyze_aggregation(seq: str, window: int = 7) -> AggregationReport:
    """Analyze aggregation propensity.

    Uses sliding window to identify aggregation-prone regions (APRs).

    Args:
        seq: Amino acid sequence
        window: Window size for scanning (default 7)

    Returns:
        AggregationReport with hotspot identification.
    """
    seq = seq.upper().replace(" ", "")

    # Calculate overall scores
    agg_scores = [AGGREGATION_PROPENSITY.get(aa, 0) for aa in seq]
    hydro_scores = [HYDROPATHY.get(aa, 0) for aa in seq]

    agg_score = sum(agg_scores) / len(seq) if seq else 0
    hydro_score = sum(hydro_scores) / len(seq) if seq else 0

    # Find hotspot regions using sliding window
    hotspots = []
    threshold = 0.4  # Average propensity threshold

    i = 0
    while i < len(seq) - window:
        window_score = sum(agg_scores[i:i+window]) / window
        if window_score > threshold:
            # Extend hotspot
            start = i
            while i < len(seq) - window:
                window_score = sum(agg_scores[i:i+window]) / window
                if window_score <= threshold:
                    break
                i += 1
            hotspots.append((start, min(i + window - 1, len(seq) - 1)))
        else:
            i += 1

    # Risk assessment
    if agg_score > 0.3 or len(hotspots) > 3:
        risk = RiskLevel.VERY_HIGH
    elif agg_score > 0.2 or len(hotspots) > 2:
        risk = RiskLevel.HIGH
    elif agg_score > 0.1 or len(hotspots) > 1:
        risk = RiskLevel.MODERATE
    else:
        risk = RiskLevel.LOW

    return AggregationReport(
        sequence=seq[:20] + "..." if len(seq) > 20 else seq,
        aggregation_score=agg_score,
        hydrophobicity_score=hydro_score,
        hotspot_regions=hotspots,
        aggregation_risk=risk,
    )


@dataclass
class PolyreactivityReport:
    """Report on polyreactivity risk."""
    sequence: str
    positive_charge_density: float  # KR per 100 residues
    aromatic_density: float  # FWY per 100 residues
    hydrophobic_clusters: int
    polyreactivity_risk: RiskLevel
    suggestions: List[str]

    def __str__(self) -> str:
        lines = [
            "Antibody Polyreactivity Analysis",
            "=" * 50,
            "",
            f"Positive charge density: {self.positive_charge_density:.1f} (K+R per 100 aa)",
            f"Aromatic density: {self.aromatic_density:.1f} (F+W+Y per 100 aa)",
            f"Hydrophobic clusters: {self.hydrophobic_clusters}",
            "",
            f"Polyreactivity risk: {self.polyreactivity_risk.value.upper()}",
            "",
        ]

        if self.suggestions:
            lines.append("Suggestions to reduce polyreactivity:")
            for s in self.suggestions:
                lines.append(f"  • {s}")

        lines.extend([
            "",
            "Note: Polyreactivity ≠ poor specificity. It measures binding to",
            "multiple unrelated targets, often driven by charged/aromatic surfaces.",
        ])

        return "\n".join(lines)


def analyze_polyreactivity(seq: str) -> PolyreactivityReport:
    """Analyze polyreactivity (promiscuous binding) risk.

    Polyreactivity correlates with:
    - High positive charge density (K, R)
    - High aromatic content (F, W, Y)
    - Surface-exposed hydrophobic patches

    Args:
        seq: Amino acid sequence

    Returns:
        PolyreactivityReport with risk assessment.
    """
    seq = seq.upper().replace(" ", "")
    n = len(seq)

    positive_count = sum(1 for aa in seq if aa in "KR")
    aromatic_count = sum(1 for aa in seq if aa in "FWY")

    positive_density = (positive_count / n) * 100 if n else 0
    aromatic_density = (aromatic_count / n) * 100 if n else 0

    # Count hydrophobic clusters (ILVF runs)
    hydrophobic_clusters = 0
    in_cluster = False
    cluster_len = 0
    for aa in seq:
        if aa in "ILVFM":
            cluster_len += 1
            if cluster_len >= 3 and not in_cluster:
                hydrophobic_clusters += 1
                in_cluster = True
        else:
            cluster_len = 0
            in_cluster = False

    # Risk assessment
    risk_score = 0
    if positive_density > 15:
        risk_score += 2
    elif positive_density > 10:
        risk_score += 1

    if aromatic_density > 8:
        risk_score += 2
    elif aromatic_density > 5:
        risk_score += 1

    if hydrophobic_clusters > 2:
        risk_score += 1

    if risk_score >= 4:
        risk = RiskLevel.VERY_HIGH
    elif risk_score >= 3:
        risk = RiskLevel.HIGH
    elif risk_score >= 2:
        risk = RiskLevel.MODERATE
    else:
        risk = RiskLevel.LOW

    # Suggestions
    suggestions = []
    if positive_density > 12:
        suggestions.append("Consider K→Q or R→Q mutations to reduce charge")
    if aromatic_density > 6:
        suggestions.append("Consider W→L or F→L mutations in non-CDR regions")
    if hydrophobic_clusters > 1:
        suggestions.append("Break up hydrophobic clusters with polar insertions")

    return PolyreactivityReport(
        sequence=seq[:20] + "..." if len(seq) > 20 else seq,
        positive_charge_density=positive_density,
        aromatic_density=aromatic_density,
        hydrophobic_clusters=hydrophobic_clusters,
        polyreactivity_risk=risk,
        suggestions=suggestions,
    )


@dataclass
class LiabilityReport:
    """Report on chemical liabilities."""
    sequence: str
    deamidation_sites: List[Tuple[int, str]]  # (position, motif)
    oxidation_sites: List[Tuple[int, str]]  # (position, residue)
    glycosylation_sites: List[int]
    isomerization_sites: List[int]  # Asp isomerization
    total_liabilities: int
    risk_summary: str

    def __str__(self) -> str:
        lines = [
            "Antibody Chemical Liability Analysis",
            "=" * 50,
            "",
        ]

        if self.deamidation_sites:
            lines.append(f"Deamidation hotspots ({len(self.deamidation_sites)}):")
            for pos, motif in self.deamidation_sites[:5]:
                lines.append(f"  Position {pos+1}: {motif}")
            lines.append("")

        if self.oxidation_sites:
            lines.append(f"Oxidation-sensitive residues ({len(self.oxidation_sites)}):")
            for pos, res in self.oxidation_sites[:5]:
                lines.append(f"  Position {pos+1}: {res}")
            lines.append("")

        if self.glycosylation_sites:
            lines.append(f"Potential N-glycosylation sites ({len(self.glycosylation_sites)}):")
            for pos in self.glycosylation_sites[:5]:
                lines.append(f"  Position {pos+1}: N-X-S/T")
            lines.append("")

        if self.isomerization_sites:
            lines.append(f"Asp isomerization sites ({len(self.isomerization_sites)}):")
            for pos in self.isomerization_sites[:5]:
                lines.append(f"  Position {pos+1}: DG")
            lines.append("")

        lines.extend([
            f"Total liabilities: {self.total_liabilities}",
            "",
            f"Risk summary: {self.risk_summary}",
        ])

        return "\n".join(lines)


def analyze_liabilities(seq: str) -> LiabilityReport:
    """Analyze chemical liabilities (deamidation, oxidation, glycosylation).

    Critical for stability during manufacturing and storage.

    Args:
        seq: Amino acid sequence

    Returns:
        LiabilityReport with detailed liability inventory.
    """
    seq = seq.upper().replace(" ", "")

    # Deamidation hotspots
    deamidation = []
    for motif in DEAMIDATION_MOTIFS:
        idx = 0
        while True:
            pos = seq.find(motif, idx)
            if pos == -1:
                break
            deamidation.append((pos, motif))
            idx = pos + 1

    # Sort by position
    deamidation.sort(key=lambda x: x[0])

    # Oxidation-sensitive
    oxidation = [(i, aa) for i, aa in enumerate(seq) if aa in OXIDATION_RESIDUES]

    # Glycosylation
    glyco = find_glycosylation_sites(seq)

    # Asp isomerization (DG motif)
    isomerization = []
    idx = 0
    while True:
        pos = seq.find("DG", idx)
        if pos == -1:
            break
        isomerization.append(pos)
        idx = pos + 1

    total = len(deamidation) + len(oxidation) + len(glyco) + len(isomerization)

    # Risk summary
    if total > 10:
        risk = "HIGH - multiple liabilities require engineering"
    elif total > 5:
        risk = "MODERATE - consider liability reduction"
    else:
        risk = "LOW - acceptable for development"

    return LiabilityReport(
        sequence=seq[:20] + "..." if len(seq) > 20 else seq,
        deamidation_sites=deamidation,
        oxidation_sites=oxidation,
        glycosylation_sites=glyco,
        isomerization_sites=isomerization,
        total_liabilities=total,
        risk_summary=risk,
    )


@dataclass
class DevelopabilityReport:
    """Comprehensive developability assessment."""
    sequence: str
    charge: ChargeReport
    aggregation: AggregationReport
    polyreactivity: PolyreactivityReport
    liabilities: LiabilityReport
    overall_risk: RiskLevel
    recommendation: str

    def __str__(self) -> str:
        lines = [
            "=" * 60,
            "COMPREHENSIVE ANTIBODY DEVELOPABILITY ASSESSMENT",
            "=" * 60,
            "",
            f"Overall risk: {self.overall_risk.value.upper()}",
            f"Recommendation: {self.recommendation}",
            "",
            "-" * 60,
            "",
        ]

        lines.append(str(self.charge))
        lines.append("")
        lines.append("-" * 60)
        lines.append("")
        lines.append(str(self.aggregation))
        lines.append("")
        lines.append("-" * 60)
        lines.append("")
        lines.append(str(self.polyreactivity))
        lines.append("")
        lines.append("-" * 60)
        lines.append("")
        lines.append(str(self.liabilities))

        return "\n".join(lines)


def assess_developability(seq: str) -> DevelopabilityReport:
    """Comprehensive developability assessment.

    Evaluates:
    - Viscosity risk (via net charge)
    - Aggregation propensity
    - Polyreactivity
    - Chemical liabilities

    Args:
        seq: Amino acid sequence (Fv or full antibody)

    Returns:
        DevelopabilityReport with comprehensive assessment.
    """
    charge = analyze_charge(seq)
    agg = analyze_aggregation(seq)
    poly = analyze_polyreactivity(seq)
    liab = analyze_liabilities(seq)

    # Overall risk is the worst individual risk
    risks = [charge.viscosity_risk, agg.aggregation_risk, poly.polyreactivity_risk]
    risk_order = [RiskLevel.LOW, RiskLevel.MODERATE, RiskLevel.HIGH, RiskLevel.VERY_HIGH]
    overall = max(risks, key=lambda r: risk_order.index(r))

    # Recommendation
    if overall == RiskLevel.LOW:
        rec = "Proceed to development with standard monitoring"
    elif overall == RiskLevel.MODERATE:
        rec = "Consider engineering to address moderate risks before scale-up"
    elif overall == RiskLevel.HIGH:
        rec = "Engineering required before CMC development"
    else:
        rec = "Significant re-engineering likely needed; consider backup candidates"

    return DevelopabilityReport(
        sequence=seq[:20] + "..." if len(seq) > 20 else seq,
        charge=charge,
        aggregation=agg,
        polyreactivity=poly,
        liabilities=liab,
        overall_risk=overall,
        recommendation=rec,
    )


# Quick test
if __name__ == "__main__":
    # Test with a mock antibody sequence (trastuzumab light chain fragment)
    test_seq = "DIQMTQSPSSLSASVGDRVTITCRASQDVNTAVAWYQQKPGKAPKLLIYSASFLYSGVPSRFSGSRSGTDFTLTISSLQPEDFATYYCQQHYTTPPTFGQGTKVEIK"

    print("=== Antibody Developability Tool Test ===\n")

    report = assess_developability(test_seq)
    print(report)
