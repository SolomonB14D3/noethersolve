"""
noethersolve.crispr — CRISPR guide RNA scorer for therapeutic design.

Scores a 20nt guide RNA sequence for on-target activity predictors and
off-target risk factors. Pure sequence analysis — no alignment or genome
lookup required.

On-target checks:
  - GC content (optimal 40-70%)
  - Homopolymer runs (>=4T = Pol III termination signal)
  - Terminal GC (positions 1 and 20 for duplex stability)
  - Position 20 identity (G at PAM-proximal position boosts activity)

Off-target checks:
  - Seed region GC (positions 1-12, PAM-proximal)
  - Self-complementarity (hairpin formation reduces RISC loading)

Usage:
    from noethersolve.crispr import score_guide, score_guides, check_offtarget_pair

    report = score_guide("ATCGATCGATCGATCGATCG")
    print(report)
    # Shows GC content, seed GC, homopolymer runs, issues, verdict

    reports = score_guides(["ATCGATCGATCGATCGATCG", "GGGGTTTTCCCCAAAATTTT"])
    for r in reports:
        print(r)

    pair = check_offtarget_pair("ATCGATCGATCGATCGATCG", "ATCGATCGATCGATCGATCC")
    print(pair)
    # {'n_mismatches': 1, 'seed_mismatches': 0, 'non_seed_mismatches': 1,
    #  'risk_level': 'HIGH'}
"""

from dataclasses import dataclass
from typing import Dict, List


# ─── Constants ───────────────────────────────────────────────────────────────

_VALID_BASES = set("ACGT")
_COMPLEMENT = {"A": "T", "T": "A", "C": "G", "G": "C"}
_KNOWN_PAMS = {"NGG", "NAG", "NGA", "NNGRRT", "NNNNGATT", "TTTN", "TTTV"}

_MIN_GUIDE_LEN = 17
_MAX_GUIDE_LEN = 25
_SEED_END = 12  # positions 1-12 (0-indexed: 0-11)


# ─── Dataclasses ─────────────────────────────────────────────────────────────

@dataclass
class GuideIssue:
    """A single quality issue found in a guide RNA sequence."""
    check_name: str       # GC_CONTENT, HOMOPOLYMER, TERMINAL_GC, POS20, SEED_GC, SELF_COMP
    severity: str         # HIGH, MODERATE, LOW
    message: str
    score_penalty: float  # 0-1 scale mapped to the 0-100 activity score

    def __str__(self):
        return f"  [{self.severity}] {self.check_name}: {self.message} (penalty: -{self.score_penalty:.0f})"


@dataclass
class GuideReport:
    """Result of score_guide()."""
    sequence: str
    length: int
    gc_content: float
    seed_gc: float
    longest_homopolymer: int
    longest_self_comp: int
    issues: List[GuideIssue]
    activity_score: float       # 0-100, higher is better
    offtarget_risk: str         # LOW, MODERATE, HIGH
    verdict: str                # PASS, WARN, FAIL

    def __str__(self):
        lines = []
        lines.append(f"{'=' * 60}")
        lines.append(f"  CRISPR Guide Scorer: {self.verdict}")
        lines.append(f"{'=' * 60}")
        lines.append(f"  Sequence:  {self.sequence}")
        lines.append(f"  Length:    {self.length}nt")
        lines.append(f"  GC:       {self.gc_content:.1%}")
        lines.append(f"  Seed GC:  {self.seed_gc:.1%} (positions 1-{_SEED_END})")
        lines.append(f"  Homopoly: {self.longest_homopolymer}nt longest run")
        lines.append(f"  Self-comp: {self.longest_self_comp}bp longest match")
        lines.append("")
        lines.append(f"  Activity score: {self.activity_score:.0f}/100")
        lines.append(f"  Off-target risk: {self.offtarget_risk}")
        lines.append("")

        if self.issues:
            lines.append("  Issues found:")
            for issue in sorted(self.issues,
                                key=lambda i: {"HIGH": 0, "MODERATE": 1, "LOW": 2}.get(i.severity, 3)):
                lines.append(str(issue))
        else:
            lines.append("  No issues found.")

        lines.append(f"{'=' * 60}")
        return "\n".join(lines)

    @property
    def passed(self) -> bool:
        return self.verdict == "PASS"


# ─── Helper functions ────────────────────────────────────────────────────────

def _gc_content(seq: str) -> float:
    """Fraction of G+C in the sequence."""
    if not seq:
        return 0.0
    gc = sum(1 for b in seq if b in "GC")
    return gc / len(seq)


def _longest_homopolymer(seq: str) -> int:
    """Length of the longest single-base run in the sequence."""
    if not seq:
        return 0
    max_run = 1
    current_run = 1
    for i in range(1, len(seq)):
        if seq[i] == seq[i - 1]:
            current_run += 1
            if current_run > max_run:
                max_run = current_run
        else:
            current_run = 1
    return max_run


def _homopolymer_base_and_length(seq: str) -> List[tuple]:
    """Return all homopolymer runs as (base, length, start_pos) tuples."""
    if not seq:
        return []
    runs = []
    start = 0
    for i in range(1, len(seq) + 1):
        if i == len(seq) or seq[i] != seq[start]:
            if i - start >= 4:
                runs.append((seq[start], i - start, start))
            start = i
    return runs


def _longest_self_complementary(seq: str) -> int:
    """Find the longest self-complementary region (potential hairpin).

    Checks if any substring of length k has its reverse complement
    elsewhere in the sequence. Returns the longest such match length.
    """
    n = len(seq)
    if n < 6:
        return 0

    best = 0
    # Check from longest possible down to 6
    for k in range(min(n // 2, n - 1), 5, -1):
        if best >= k:
            break
        for i in range(n - k + 1):
            substr = seq[i:i + k]
            rev_comp = "".join(_COMPLEMENT[b] for b in reversed(substr))
            # Look for rev_comp elsewhere (non-overlapping)
            for j in range(n - k + 1):
                if abs(j - i) < k:
                    continue  # overlapping
                if seq[j:j + k] == rev_comp:
                    best = k
                    break
            if best >= k:
                break

    return best


# ─── On-target activity checks ──────────────────────────────────────────────

def _check_gc_content(seq: str) -> List[GuideIssue]:
    """Check GC content is in the optimal 40-70% range."""
    gc = _gc_content(seq)
    issues = []

    if gc < 0.30 or gc > 0.70:
        if gc < 0.30:
            msg = f"GC content {gc:.1%} is below 30% — poor activity expected"
        else:
            msg = f"GC content {gc:.1%} is above 70% — poor activity expected"
        issues.append(GuideIssue(
            check_name="GC_CONTENT",
            severity="HIGH",
            message=msg,
            score_penalty=20,
        ))
    elif gc < 0.40 or gc > 0.70:
        if gc < 0.40:
            msg = f"GC content {gc:.1%} is below 40% — reduced activity"
        else:
            msg = f"GC content {gc:.1%} is above 70% — reduced activity"
        issues.append(GuideIssue(
            check_name="GC_CONTENT",
            severity="MODERATE",
            message=msg,
            score_penalty=20,
        ))

    return issues


def _check_homopolymer(seq: str) -> List[GuideIssue]:
    """Check for homopolymer runs that reduce activity."""
    issues = []
    runs = _homopolymer_base_and_length(seq)

    for base, length, pos in runs:
        if base == "T" and length >= 4:
            issues.append(GuideIssue(
                check_name="HOMOPOLYMER",
                severity="HIGH",
                message=f"{length}xT run at position {pos + 1} — Pol III termination signal",
                score_penalty=30,
            ))
        elif length >= 4:
            issues.append(GuideIssue(
                check_name="HOMOPOLYMER",
                severity="MODERATE",
                message=f"{length}x{base} run at position {pos + 1} — reduced activity",
                score_penalty=10,
            ))

    return issues


def _check_terminal_gc(seq: str) -> List[GuideIssue]:
    """Check for G or C at terminal positions (1 and last) for stability."""
    issues = []
    has_gc_first = seq[0] in "GC"
    has_gc_last = seq[-1] in "GC"

    if not has_gc_first and not has_gc_last:
        issues.append(GuideIssue(
            check_name="TERMINAL_GC",
            severity="MODERATE",
            message=f"No G/C at position 1 ({seq[0]}) or position {len(seq)} ({seq[-1]}) — reduced duplex stability",
            score_penalty=10,
        ))

    return issues


def _check_pos20(seq: str) -> List[GuideIssue]:
    """Check if position 20 (last nt, PAM-proximal) is G for activity boost."""
    issues = []
    last = seq[-1]

    if last != "G":
        issues.append(GuideIssue(
            check_name="POS20",
            severity="LOW",
            message=f"Position {len(seq)} is {last}, not G — slightly reduced activity",
            score_penalty=5,
        ))

    return issues


# ─── Off-target risk checks ─────────────────────────────────────────────────

def _check_seed_gc(seq: str) -> List[GuideIssue]:
    """Check GC content in the seed region (positions 1-12, PAM-proximal)."""
    issues = []
    seed = seq[:_SEED_END]
    seed_gc = _gc_content(seed)

    if seed_gc > 0.75:
        issues.append(GuideIssue(
            check_name="SEED_GC",
            severity="MODERATE",
            message=f"Seed GC {seed_gc:.1%} above 75% — increased off-target binding",
            score_penalty=10,
        ))
    elif seed_gc < 0.25:
        issues.append(GuideIssue(
            check_name="SEED_GC",
            severity="HIGH",
            message=f"Seed GC {seed_gc:.1%} below 25% — poor on-target binding",
            score_penalty=10,
        ))

    return issues


def _check_self_complementarity(seq: str) -> List[GuideIssue]:
    """Check for self-complementary regions that could form hairpins."""
    issues = []
    longest = _longest_self_complementary(seq)

    if longest >= 8:
        issues.append(GuideIssue(
            check_name="SELF_COMP",
            severity="HIGH",
            message=f"{longest}bp self-complementary region — hairpin formation reduces RISC loading",
            score_penalty=20,
        ))
    elif longest >= 6:
        issues.append(GuideIssue(
            check_name="SELF_COMP",
            severity="MODERATE",
            message=f"{longest}bp self-complementary region — potential hairpin",
            score_penalty=10,
        ))

    return issues


# ─── Core scoring function ──────────────────────────────────────────────────

def score_guide(sequence: str, pam: str = "NGG") -> GuideReport:
    """Score a CRISPR guide RNA sequence for on-target activity and off-target risk.

    Args:
        sequence: 17-25nt DNA guide sequence (ACGT only, auto-uppercased).
        pam: PAM sequence for the target nuclease (default NGG for SpCas9).
            Currently used for validation only — future versions will adjust
            scoring by PAM type.

    Returns:
        GuideReport with activity score, off-target risk, and detailed issues.

    Raises:
        ValueError: if sequence contains non-DNA characters, is wrong length,
            or PAM is unrecognized.
    """
    # ── Validate ──────────────────────────────────────────────────────────
    seq = sequence.strip().upper()

    if not seq:
        raise ValueError("Empty guide sequence")

    invalid = set(seq) - _VALID_BASES
    if invalid:
        raise ValueError(f"Invalid bases in guide: {sorted(invalid)}. "
                        f"Only A, C, G, T are accepted.")

    if len(seq) < _MIN_GUIDE_LEN or len(seq) > _MAX_GUIDE_LEN:
        raise ValueError(f"Guide length {len(seq)}nt outside accepted range "
                        f"({_MIN_GUIDE_LEN}-{_MAX_GUIDE_LEN}nt)")

    pam_upper = pam.strip().upper()
    if pam_upper not in _KNOWN_PAMS:
        raise ValueError(f"Unrecognized PAM '{pam}'. Known PAMs: {sorted(_KNOWN_PAMS)}")

    # ── Run all checks ────────────────────────────────────────────────────
    all_issues = []
    all_issues.extend(_check_gc_content(seq))
    all_issues.extend(_check_homopolymer(seq))
    all_issues.extend(_check_terminal_gc(seq))
    all_issues.extend(_check_pos20(seq))
    all_issues.extend(_check_seed_gc(seq))
    all_issues.extend(_check_self_complementarity(seq))

    # ── Compute metrics ───────────────────────────────────────────────────
    gc = _gc_content(seq)
    seed = seq[:_SEED_END]
    seed_gc = _gc_content(seed)
    homo_len = _longest_homopolymer(seq)
    self_comp_len = _longest_self_complementary(seq)

    # ── Activity score ────────────────────────────────────────────────────
    activity_score = 100.0
    for issue in all_issues:
        activity_score -= issue.score_penalty
    activity_score = max(0.0, activity_score)

    # ── Off-target risk ───────────────────────────────────────────────────
    if seed_gc > 0.75:
        offtarget_risk = "HIGH"
    elif seed_gc > 0.60 or self_comp_len >= 6:
        offtarget_risk = "MODERATE"
    else:
        offtarget_risk = "LOW"

    # ── Verdict ───────────────────────────────────────────────────────────
    if activity_score < 50:
        verdict = "FAIL"
    elif activity_score < 70:
        verdict = "WARN"
    else:
        verdict = "PASS"

    return GuideReport(
        sequence=seq,
        length=len(seq),
        gc_content=gc,
        seed_gc=seed_gc,
        longest_homopolymer=homo_len,
        longest_self_comp=self_comp_len,
        issues=all_issues,
        activity_score=activity_score,
        offtarget_risk=offtarget_risk,
        verdict=verdict,
    )


def score_guides(sequences: List[str], pam: str = "NGG") -> List[GuideReport]:
    """Score multiple guide RNA sequences in batch.

    Args:
        sequences: list of guide RNA sequences (17-25nt DNA each).
        pam: PAM sequence (default NGG).

    Returns:
        List of GuideReport, one per input sequence.
    """
    return [score_guide(seq, pam=pam) for seq in sequences]


# ─── Off-target pair comparison ──────────────────────────────────────────────

def check_offtarget_pair(guide: str, offtarget: str) -> Dict:
    """Compare a guide to a potential off-target site and assess risk.

    Counts mismatches in the seed region (positions 1-12, PAM-proximal)
    vs non-seed region (positions 13-end). Fewer seed mismatches = higher
    off-target cleavage risk.

    Args:
        guide: the designed guide RNA sequence (DNA, ACGT).
        offtarget: the potential off-target genomic site (DNA, ACGT).
            Must be the same length as guide.

    Returns:
        Dict with keys: n_mismatches, seed_mismatches, non_seed_mismatches,
        mismatch_positions (1-indexed), risk_level (HIGH/MODERATE/LOW).

    Raises:
        ValueError: if sequences differ in length or contain non-DNA characters.
    """
    g = guide.strip().upper()
    o = offtarget.strip().upper()

    if not g or not o:
        raise ValueError("Empty sequence provided")

    for label, seq in [("guide", g), ("offtarget", o)]:
        invalid = set(seq) - _VALID_BASES
        if invalid:
            raise ValueError(f"Invalid bases in {label}: {sorted(invalid)}")

    if len(g) != len(o):
        raise ValueError(f"Length mismatch: guide={len(g)}nt, offtarget={len(o)}nt")

    # ── Count mismatches by region ────────────────────────────────────────
    seed_mm = 0
    non_seed_mm = 0
    mismatch_positions = []

    for i in range(len(g)):
        if g[i] != o[i]:
            pos = i + 1  # 1-indexed
            mismatch_positions.append(pos)
            if i < _SEED_END:
                seed_mm += 1
            else:
                non_seed_mm += 1

    total_mm = seed_mm + non_seed_mm

    # ── Risk level ────────────────────────────────────────────────────────
    if seed_mm <= 2:
        risk_level = "HIGH"
    elif seed_mm <= 4:
        risk_level = "MODERATE"
    else:
        risk_level = "LOW"

    return {
        "n_mismatches": total_mm,
        "seed_mismatches": seed_mm,
        "non_seed_mismatches": non_seed_mm,
        "mismatch_positions": mismatch_positions,
        "risk_level": risk_level,
    }
