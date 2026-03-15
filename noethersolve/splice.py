"""
noethersolve.splice -- Splice site strength scorer using position weight matrices.

Scores donor (5' GT) and acceptor (3' AG) splice sites against simplified
Shapiro & Senapathy (1987) mammalian consensus position weight matrices.
Can score individual sites or scan full sequences for all potential splice
sites ranked by strength.

Usage:
    from noethersolve.splice import score_donor, score_acceptor, scan_splice_sites

    # Score a known donor site (9nt: 3 exonic + GT + 4 intronic)
    report = score_donor("CAGGTAAGT")
    print(report)
    # Shows position-by-position PWM scores, strength, and canonical GT check

    # Score a known acceptor site (16nt: 14 intronic + AG + 2 exonic)
    report = score_acceptor("TTTTTTTTTTTTAG" + "GA")
    print(report)

    # Scan a full sequence for all splice sites
    sites = scan_splice_sites("ATGCAGGTAAGTCCC...TTTTTCCCCCAGGAT", site_type="both")
    for site in sites:
        print(f"{site.site_type} at {site.position}: {site.strength} ({site.score:.2f})")

    # Evaluate polypyrimidine tract quality
    from noethersolve.splice import pyrimidine_tract_score
    py_frac = pyrimidine_tract_score("TCTTTCCTTCTTT")  # 1.0 = all pyrimidines
"""

from dataclasses import dataclass, field
from typing import List


# --- Position weight matrices (log-odds) -------------------------------------------
#
# Donor site: 9 positions -- 3 exonic (-3, -2, -1) + invariant GT (+1, +2)
#             + 4 intronic (+3, +4, +5, +6)
# Simplified from Shapiro & Senapathy (1987) mammalian consensus.

_DONOR_PWM = {
    #           -3     -2     -1     +1     +2     +3     +4     +5     +6
    "A": [ 0.27,  0.59, -0.89,  0.00, -3.00,  0.63,  0.74, -0.13, -0.63],
    "C": [ 0.35, -0.07, -0.47, -3.00, -3.00, -0.43, -0.35, -0.58, -0.85],
    "G": [-0.19, -0.28,  0.80,  1.26, -3.00,  0.40, -0.51,  0.72, -0.42],
    "T": [-0.62, -0.55, -0.57, -3.00,  1.16, -0.78, -0.25, -0.35,  0.76],
}

# Acceptor site: 16 positions -- 12 intronic (-14 to -3, includes polypyrimidine
#                tract + invariant AG at -2, -1) + 2 exonic (+1, +2)

_ACCEPTOR_PWM = {
    #         -14   -13   -12   -11   -10    -9    -8    -7    -6    -5    -4    -3    -2    -1    +1    +2
    "A": [-0.20, -0.20, -0.20, -0.20, -0.20,  0.00,  0.00,  0.00, -0.20, -0.50,  0.30, -3.00,  1.16, -3.00,  0.10,  0.10],
    "C": [ 0.50,  0.50,  0.50,  0.40,  0.40,  0.30,  0.30,  0.30,  0.50,  0.60,  0.20, -3.00, -3.00, -3.00,  0.10,  0.00],
    "G": [-1.00, -1.00, -1.00, -1.00, -1.00, -1.00, -1.00, -1.00, -1.00, -1.00, -1.00,  1.00, -3.00,  1.16, -0.10,  0.30],
    "T": [ 0.40,  0.40,  0.40,  0.40,  0.40,  0.40,  0.40,  0.40,  0.40,  0.50,  0.20, -3.00, -3.00, -3.00, -0.10, -0.30],
}

_DONOR_LEN = 9
_ACCEPTOR_LEN = 16

# Donor: invariant GT at 0-indexed positions 3, 4
_DONOR_GT_POS = (3, 4)
# Acceptor: invariant AG at 0-indexed positions 11, 12
_ACCEPTOR_AG_POS = (11, 12)

# Strength thresholds
_STRONG_THRESHOLD = 8.0
_MODERATE_THRESHOLD = 4.0


# --- Dataclass --------------------------------------------------------------------

@dataclass
class SpliceSiteReport:
    """Result of scoring a single splice site."""
    position: int                          # 0-based position in the scanned sequence (or 0 for direct scoring)
    site_type: str                         # "donor" or "acceptor"
    sequence: str                          # the scored window
    score: float                           # PWM sum score
    strength: str                          # "STRONG", "MODERATE", or "WEAK"
    has_canonical_dinucleotide: bool        # GT for donor, AG for acceptor
    per_position_scores: List[float] = field(default_factory=list)

    def __str__(self):
        lines = []
        lines.append(f"{'=' * 50}")
        tag = "Donor (5' GT)" if self.site_type == "donor" else "Acceptor (3' AG)"
        lines.append(f"  Splice Site: {tag}")
        lines.append(f"{'=' * 50}")
        lines.append(f"  Position: {self.position}")
        lines.append(f"  Sequence: {self.sequence}")
        lines.append(f"  Score: {self.score:.3f}")
        lines.append(f"  Strength: {self.strength}")
        canonical = "GT" if self.site_type == "donor" else "AG"
        status = "YES" if self.has_canonical_dinucleotide else "NO (non-canonical)"
        lines.append(f"  Canonical {canonical}: {status}")
        lines.append(f"")
        # Per-position breakdown
        if self.per_position_scores:
            lines.append(f"  Per-position scores:")
            for i, (base, sc) in enumerate(zip(self.sequence, self.per_position_scores)):
                lines.append(f"    pos {i:>2d}  {base}  {sc:+.3f}")
            lines.append(f"")
        lines.append(f"{'=' * 50}")
        return "\n".join(lines)

    @property
    def passed(self) -> bool:
        """True if strength is STRONG or MODERATE."""
        return self.strength in ("STRONG", "MODERATE")


# --- Helper -----------------------------------------------------------------------

def pyrimidine_tract_score(sequence: str) -> float:
    """Count fraction of pyrimidines (C/T) in a sequence.

    Useful for evaluating the polypyrimidine tract upstream of an acceptor
    splice site. A strong tract is >80% pyrimidine.

    Args:
        sequence: nucleotide string (case-insensitive). U is treated as T.

    Returns:
        Fraction of C+T bases (0.0 to 1.0). Returns 0.0 for empty input.
    """
    s = sequence.upper().replace("U", "T")
    if len(s) == 0:
        return 0.0
    ct = s.count("C") + s.count("T")
    return ct / len(s)


# --- Internal scoring -------------------------------------------------------------

def _score_pwm(sequence: str, pwm: dict, expected_len: int) -> List[float]:
    """Score each position of *sequence* against *pwm*, returning per-position log-odds."""
    seq = sequence.upper().replace("U", "T")
    scores = []
    for i, base in enumerate(seq):
        if base in pwm and i < len(pwm["A"]):
            scores.append(pwm[base][i])
        else:
            scores.append(0.0)  # unknown base gets neutral score
    return scores


def _classify_strength(score: float) -> str:
    """Map a PWM sum score to a strength label."""
    if score > _STRONG_THRESHOLD:
        return "STRONG"
    elif score >= _MODERATE_THRESHOLD:
        return "MODERATE"
    else:
        return "WEAK"


# --- Public scoring functions -----------------------------------------------------

def score_donor(sequence: str) -> SpliceSiteReport:
    """Score a 9-nt donor (5') splice site against the mammalian consensus PWM.

    The input should be 9 nucleotides: 3 exonic bases, then the GT
    dinucleotide, then 4 intronic bases.  Example: ``CAGGTAAGT``.

    Args:
        sequence: 9-character nucleotide string (case-insensitive, U accepted).

    Returns:
        SpliceSiteReport with PWM score, strength classification, and
        canonical GT check.

    Raises:
        ValueError: if *sequence* is not exactly 9 characters or contains
            invalid nucleotides.
    """
    seq = sequence.upper().replace("U", "T").replace(" ", "")
    if len(seq) != _DONOR_LEN:
        raise ValueError(
            f"Donor site sequence must be exactly {_DONOR_LEN} nt, "
            f"got {len(seq)} ('{sequence}')"
        )
    _validate_bases(seq)

    per_pos = _score_pwm(seq, _DONOR_PWM, _DONOR_LEN)
    total = sum(per_pos)
    has_gt = (seq[_DONOR_GT_POS[0]] == "G" and seq[_DONOR_GT_POS[1]] == "T")

    return SpliceSiteReport(
        position=0,
        site_type="donor",
        sequence=seq,
        score=total,
        strength=_classify_strength(total),
        has_canonical_dinucleotide=has_gt,
        per_position_scores=per_pos,
    )


def score_acceptor(sequence: str) -> SpliceSiteReport:
    """Score a 16-nt acceptor (3') splice site against the mammalian consensus PWM.

    The input should be 16 nucleotides: 11 intronic bases (including the
    polypyrimidine tract), the AG dinucleotide, then 2 exonic bases.

    Args:
        sequence: 16-character nucleotide string (case-insensitive, U accepted).

    Returns:
        SpliceSiteReport with PWM score, strength classification, and
        canonical AG check.

    Raises:
        ValueError: if *sequence* is not exactly 16 characters or contains
            invalid nucleotides.
    """
    seq = sequence.upper().replace("U", "T").replace(" ", "")
    if len(seq) != _ACCEPTOR_LEN:
        raise ValueError(
            f"Acceptor site sequence must be exactly {_ACCEPTOR_LEN} nt, "
            f"got {len(seq)} ('{sequence}')"
        )
    _validate_bases(seq)

    per_pos = _score_pwm(seq, _ACCEPTOR_PWM, _ACCEPTOR_LEN)
    total = sum(per_pos)
    has_ag = (seq[_ACCEPTOR_AG_POS[0]] == "A" and seq[_ACCEPTOR_AG_POS[1]] == "G")

    return SpliceSiteReport(
        position=0,
        site_type="acceptor",
        sequence=seq,
        score=total,
        strength=_classify_strength(total),
        has_canonical_dinucleotide=has_ag,
        per_position_scores=per_pos,
    )


def scan_splice_sites(
    sequence: str,
    site_type: str = "both",
) -> List[SpliceSiteReport]:
    """Scan a full sequence for all potential donor and/or acceptor splice sites.

    For each GT dinucleotide found, extracts a 9-nt window and scores it as
    a donor site. For each AG dinucleotide found, extracts a 16-nt window
    and scores it as an acceptor site. Results are sorted by score
    (strongest first).

    Args:
        sequence: nucleotide string of any length (case-insensitive, U accepted).
        site_type: ``"donor"``, ``"acceptor"``, or ``"both"`` (default).

    Returns:
        List of SpliceSiteReport objects sorted by descending score.

    Raises:
        ValueError: if *site_type* is not one of the accepted values, or
            if *sequence* contains invalid nucleotides.
    """
    site_type = site_type.lower()
    if site_type not in ("donor", "acceptor", "both"):
        raise ValueError(
            f"site_type must be 'donor', 'acceptor', or 'both', got '{site_type}'"
        )

    seq = sequence.upper().replace("U", "T").replace(" ", "").replace("\n", "")
    _validate_bases(seq)

    results: List[SpliceSiteReport] = []
    n = len(seq)

    # Scan for donor sites (GT dinucleotides)
    if site_type in ("donor", "both"):
        for i in range(n - 1):
            if seq[i] == "G" and seq[i + 1] == "T":
                # GT should be at 0-indexed positions 3-4 within the 9-nt window
                window_start = i - 3
                window_end = window_start + _DONOR_LEN
                if window_start < 0 or window_end > n:
                    continue  # not enough context
                window = seq[window_start:window_end]
                per_pos = _score_pwm(window, _DONOR_PWM, _DONOR_LEN)
                total = sum(per_pos)
                results.append(SpliceSiteReport(
                    position=window_start,
                    site_type="donor",
                    sequence=window,
                    score=total,
                    strength=_classify_strength(total),
                    has_canonical_dinucleotide=True,  # found via GT scan
                    per_position_scores=per_pos,
                ))

    # Scan for acceptor sites (AG dinucleotides)
    if site_type in ("acceptor", "both"):
        for i in range(n - 1):
            if seq[i] == "A" and seq[i + 1] == "G":
                # AG should be at 0-indexed positions 11-12 within the 16-nt window
                window_start = i - 11
                window_end = window_start + _ACCEPTOR_LEN
                if window_start < 0 or window_end > n:
                    continue  # not enough context
                window = seq[window_start:window_end]
                per_pos = _score_pwm(window, _ACCEPTOR_PWM, _ACCEPTOR_LEN)
                total = sum(per_pos)
                results.append(SpliceSiteReport(
                    position=window_start,
                    site_type="acceptor",
                    sequence=window,
                    score=total,
                    strength=_classify_strength(total),
                    has_canonical_dinucleotide=True,  # found via AG scan
                    per_position_scores=per_pos,
                ))

    # Sort by score descending (strongest first)
    results.sort(key=lambda r: r.score, reverse=True)
    return results


# --- Validation helper ------------------------------------------------------------

def _validate_bases(seq: str) -> None:
    """Raise ValueError if seq contains non-ACGT characters."""
    valid = set("ACGT")
    invalid = set(seq) - valid
    if invalid:
        raise ValueError(
            f"Invalid characters in sequence: {sorted(invalid)}. "
            f"Only A, C, G, T (and U, converted internally) are allowed."
        )
