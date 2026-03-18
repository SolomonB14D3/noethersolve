"""
noethersolve.aggregation — Protein aggregation propensity predictor.

Predicts aggregation risk from amino acid sequence using hydrophobicity,
AGGRESCAN propensity scores, charge analysis, and low-complexity detection.

Checks:
  - Aggregation-prone regions (APRs) via AGGRESCAN sliding window
  - Overall hydrophobicity (Kyte-Doolittle mean)
  - Hydrophobic patches (consecutive hydrophobic residues)
  - Net charge at pH 7 (near-neutral = aggregation risk)
  - Low-complexity regions (limited residue diversity)

Usage:
    from noethersolve.aggregation import predict_aggregation, AggregationReport

    report = predict_aggregation("MILVFAILVILMFAILVM")
    print(report)
    # Shows per-check diagnostics, severity levels, and overall verdict

    if not report.passed:
        for issue in report.issues:
            print(issue)

    # Access raw scales
    from noethersolve.aggregation import KYTE_DOOLITTLE, AGGRESCAN
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# ─── Hydrophobicity and aggregation scales ───────────────────────────────────

_KYTE_DOOLITTLE = {
    "I": 4.5, "V": 4.2, "L": 3.8, "F": 2.8, "C": 2.5,
    "M": 1.9, "A": 1.8, "G": -0.4, "T": -0.7, "S": -0.8,
    "W": -0.9, "Y": -1.3, "P": -1.6, "H": -3.2, "D": -3.5,
    "E": -3.5, "N": -3.5, "Q": -3.5, "K": -3.9, "R": -4.5,
}

_AGGRESCAN = {
    "I": 0.478, "V": 0.293, "L": 0.393, "F": 0.427, "C": -0.032,
    "M": 0.144, "A": 0.038, "G": -0.180, "T": -0.174, "S": -0.264,
    "W": 0.278, "Y": 0.145, "P": -0.334, "H": -0.372, "D": -0.274,
    "E": -0.147, "N": -0.354, "Q": -0.179, "K": -0.339, "R": -0.176,
}

# Public API aliases (without underscore prefix)
KYTE_DOOLITTLE = dict(_KYTE_DOOLITTLE)
AGGRESCAN = dict(_AGGRESCAN)

_VALID_RESIDUES = set(_KYTE_DOOLITTLE.keys())


# ─── Dataclasses ─────────────────────────────────────────────────────────────

@dataclass
class AggregationIssue:
    """A single aggregation risk issue found in a protein sequence."""
    check_name: str           # APR, HYDROPHOBICITY, HYDROPHOBIC_PATCH, etc.
    severity: str             # HIGH, MODERATE, LOW
    message: str
    position: Optional[int] = None   # 0-based position in sequence, if applicable
    value: float = 0.0

    def __str__(self):
        pos_str = f" at pos {self.position}" if self.position is not None else ""
        return f"  [{self.severity}] {self.check_name}{pos_str}: {self.message}"


@dataclass
class AggregationReport:
    """Result of predict_aggregation()."""
    sequence_length: int
    mean_hydrophobicity: float
    mean_aggrescan: float
    n_aprs: int
    longest_hydrophobic_patch: int
    net_charge: int
    charge_density: float
    issues: List[AggregationIssue]
    verdict: str                      # PASS, WARN, or FAIL
    apr_positions: List[Tuple[int, int, float]] = field(default_factory=list)

    def __str__(self):
        lines = []
        lines.append(f"{'=' * 60}")
        lines.append(f"  Aggregation Propensity Report: {self.verdict}")
        lines.append(f"{'=' * 60}")
        lines.append(f"  Length: {self.sequence_length} residues")
        lines.append(f"  Mean hydrophobicity (KD): {self.mean_hydrophobicity:+.3f}")
        lines.append(f"  Mean AGGRESCAN score: {self.mean_aggrescan:+.4f}")
        lines.append(f"  Aggregation-prone regions: {self.n_aprs}")
        lines.append(f"  Longest hydrophobic patch: {self.longest_hydrophobic_patch} residues")
        lines.append(f"  Net charge (pH 7): {self.net_charge:+d} "
                     f"(density: {self.charge_density:.4f})")
        lines.append("")

        if self.apr_positions:
            lines.append("  APR locations:")
            for start, end, score in self.apr_positions:
                lines.append(f"    pos {start}-{end}: mean a3v = {score:+.4f}")
            lines.append("")

        # Issues sorted by severity
        if self.issues:
            lines.append("  Issues found:")
            for issue in sorted(self.issues,
                                key=lambda i: {"HIGH": 0, "MODERATE": 1, "LOW": 2}.get(i.severity, 3)):
                lines.append(str(issue))
            lines.append("")
        else:
            lines.append("  No issues found.")
            lines.append("")

        lines.append(f"{'=' * 60}")
        return "\n".join(lines)

    @property
    def passed(self) -> bool:
        """True if no HIGH or MODERATE issues were found."""
        return self.verdict == "PASS"


# ─── Internal check functions ────────────────────────────────────────────────

def _check_aprs(seq: str, window_size: int) -> Tuple[List[AggregationIssue], List[Tuple[int, int, float]]]:
    """Check for aggregation-prone regions via AGGRESCAN sliding window.

    A window with mean AGGRESCAN score > 0.0 is an APR.

    Thresholds:
        >=3 APRs = HIGH
        1-2 APRs = MODERATE
        0 = OK
    """
    issues = []
    apr_positions = []

    if len(seq) < window_size:
        return issues, apr_positions

    for i in range(len(seq) - window_size + 1):
        window = seq[i:i + window_size]
        mean_score = sum(_AGGRESCAN[r] for r in window) / window_size
        if mean_score > 0.0:
            apr_positions.append((i, i + window_size - 1, mean_score))

    n_aprs = len(apr_positions)

    if n_aprs >= 3:
        issues.append(AggregationIssue(
            check_name="APR",
            severity="HIGH",
            message=(f"{n_aprs} aggregation-prone regions detected "
                     f"(AGGRESCAN window mean > 0.0) — high aggregation risk"),
            value=float(n_aprs),
        ))
    elif n_aprs >= 1:
        issues.append(AggregationIssue(
            check_name="APR",
            severity="MODERATE",
            message=(f"{n_aprs} aggregation-prone region{'s' if n_aprs > 1 else ''} detected "
                     f"(AGGRESCAN window mean > 0.0)"),
            value=float(n_aprs),
        ))

    return issues, apr_positions


def _check_hydrophobicity(seq: str) -> List[AggregationIssue]:
    """Check overall sequence hydrophobicity via Kyte-Doolittle mean.

    Thresholds:
        mean > 0.4 = HIGH (very hydrophobic, aggregation-prone)
        mean > 0.0 = MODERATE (somewhat hydrophobic)
        mean <= 0.0 = OK (hydrophilic)
    """
    issues = []
    if len(seq) == 0:
        return issues

    mean_kd = sum(_KYTE_DOOLITTLE[r] for r in seq) / len(seq)

    if mean_kd > 0.4:
        issues.append(AggregationIssue(
            check_name="HYDROPHOBICITY",
            severity="HIGH",
            message=(f"Mean Kyte-Doolittle score {mean_kd:+.3f} exceeds +0.4 — "
                     f"very hydrophobic, high aggregation risk"),
            value=mean_kd,
        ))
    elif mean_kd > 0.0:
        issues.append(AggregationIssue(
            check_name="HYDROPHOBICITY",
            severity="MODERATE",
            message=(f"Mean Kyte-Doolittle score {mean_kd:+.3f} is positive — "
                     f"somewhat hydrophobic"),
            value=mean_kd,
        ))

    return issues


def _check_hydrophobic_patches(seq: str) -> List[AggregationIssue]:
    """Check for consecutive stretches of hydrophobic residues (KD > 0).

    Thresholds:
        >=10 consecutive = HIGH
        >=7 consecutive = MODERATE
        <7 = OK
    """
    issues = []
    if len(seq) == 0:
        return issues

    longest = 0
    longest_pos = 0
    current = 0
    current_start = 0

    for i, r in enumerate(seq):
        if _KYTE_DOOLITTLE[r] > 0:
            if current == 0:
                current_start = i
            current += 1
            if current > longest:
                longest = current
                longest_pos = current_start
        else:
            current = 0

    if longest >= 10:
        issues.append(AggregationIssue(
            check_name="HYDROPHOBIC_PATCH",
            severity="HIGH",
            message=(f"{longest} consecutive hydrophobic residues (KD > 0) "
                     f"— long hydrophobic stretch promotes aggregation"),
            position=longest_pos,
            value=float(longest),
        ))
    elif longest >= 7:
        issues.append(AggregationIssue(
            check_name="HYDROPHOBIC_PATCH",
            severity="MODERATE",
            message=(f"{longest} consecutive hydrophobic residues (KD > 0) "
                     f"— moderate hydrophobic stretch"),
            position=longest_pos,
            value=float(longest),
        ))

    return issues


def _check_net_charge(seq: str) -> List[AggregationIssue]:
    """Check net charge at pH 7.

    K and R carry +1 charge; D and E carry -1 charge at physiological pH.
    Proteins near zero net charge aggregate more easily due to reduced
    electrostatic repulsion.

    Threshold:
        |net_charge| / length < 0.02 = MODERATE (near-neutral, aggregation risk)
    """
    issues = []
    if len(seq) == 0:
        return issues

    positive = seq.count("K") + seq.count("R")
    negative = seq.count("D") + seq.count("E")
    net = positive - negative
    density = abs(net) / len(seq)

    if density < 0.02:
        issues.append(AggregationIssue(
            check_name="NET_CHARGE",
            severity="MODERATE",
            message=(f"Net charge {net:+d} (density {density:.4f}) — "
                     f"near-neutral charge increases aggregation risk "
                     f"(K+R={positive}, D+E={negative})"),
            value=float(net),
        ))

    return issues


def _check_low_complexity(seq: str) -> List[AggregationIssue]:
    """Check for low-complexity regions using a sliding window of 20 residues.

    If any window has <=5 unique amino acid types, it is flagged as
    low-complexity. Low-complexity regions are associated with intrinsic
    disorder and can promote aggregation.

    Any such region = MODERATE.
    """
    issues = []
    window = 20

    if len(seq) < window:
        return issues

    found_positions = []
    for i in range(len(seq) - window + 1):
        region = seq[i:i + window]
        n_unique = len(set(region))
        if n_unique <= 5:
            found_positions.append((i, n_unique))

    if found_positions:
        # Report the worst (fewest unique)
        worst_pos, worst_unique = min(found_positions, key=lambda x: x[1])
        issues.append(AggregationIssue(
            check_name="LOW_COMPLEXITY",
            severity="MODERATE",
            message=(f"{len(found_positions)} low-complexity window{'s' if len(found_positions) > 1 else ''} "
                     f"detected (<=5 unique residues in 20-residue window) — "
                     f"worst at pos {worst_pos} ({worst_unique} unique)"),
            position=worst_pos,
            value=float(worst_unique),
        ))

    return issues


# ─── Core prediction function ───────────────────────────────────────────────

def predict_aggregation(sequence: str, window_size: int = 7) -> AggregationReport:
    """Predict protein aggregation propensity from amino acid sequence.

    Runs five checks: aggregation-prone regions (APRs), overall
    hydrophobicity, hydrophobic patches, net charge, and low-complexity
    regions.

    Args:
        sequence: protein sequence in single-letter amino acid codes
            (20 standard residues). Case-insensitive. Whitespace is stripped.
        window_size: sliding window size for APR detection (default 7).

    Returns:
        AggregationReport with per-check results and overall verdict.

    Raises:
        ValueError: if the sequence contains invalid characters or is empty.
    """
    # ── Clean and validate sequence ─────────────────────────────────
    s = sequence.upper().replace(" ", "").replace("\n", "").replace("\r", "").replace("\t", "")

    if len(s) == 0:
        raise ValueError("Sequence is empty after stripping whitespace.")

    invalid = set(s) - _VALID_RESIDUES
    if invalid:
        raise ValueError(
            f"Invalid characters in sequence: {sorted(invalid)}. "
            f"Only standard amino acid single-letter codes are allowed: "
            f"{sorted(_VALID_RESIDUES)}"
        )

    # ── Run all checks ──────────────────────────────────────────────
    all_issues = []

    apr_issues, apr_positions = _check_aprs(s, window_size)
    all_issues.extend(apr_issues)
    all_issues.extend(_check_hydrophobicity(s))
    all_issues.extend(_check_hydrophobic_patches(s))
    all_issues.extend(_check_net_charge(s))
    all_issues.extend(_check_low_complexity(s))

    # ── Compute summary metrics ─────────────────────────────────────
    mean_kd = sum(_KYTE_DOOLITTLE[r] for r in s) / len(s)
    mean_agg = sum(_AGGRESCAN[r] for r in s) / len(s)

    # Longest hydrophobic patch
    longest_patch = 0
    current = 0
    for r in s:
        if _KYTE_DOOLITTLE[r] > 0:
            current += 1
            if current > longest_patch:
                longest_patch = current
        else:
            current = 0

    # Net charge
    positive = s.count("K") + s.count("R")
    negative = s.count("D") + s.count("E")
    net_charge = positive - negative
    charge_density = abs(net_charge) / len(s)

    # ── Verdict ─────────────────────────────────────────────────────
    has_high = any(iss.severity == "HIGH" for iss in all_issues)
    has_moderate = any(iss.severity == "MODERATE" for iss in all_issues)

    if has_high:
        verdict = "FAIL"
    elif has_moderate:
        verdict = "WARN"
    else:
        verdict = "PASS"

    return AggregationReport(
        sequence_length=len(s),
        mean_hydrophobicity=mean_kd,
        mean_aggrescan=mean_agg,
        n_aprs=len(apr_positions),
        longest_hydrophobic_patch=longest_patch,
        net_charge=net_charge,
        charge_density=charge_density,
        issues=all_issues,
        verdict=verdict,
        apr_positions=apr_positions,
    )
