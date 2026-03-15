"""
noethersolve.audit_sequence — DNA/RNA therapeutic sequence design auditor.

Checks nucleotide sequences for properties that cause problems in therapeutic
contexts: immune activation (CpG), expression efficiency (GC content),
transcription termination (homopolymers), mis-splicing (cryptic splice sites),
premature truncation (poly-A signals), and secondary structure formation
(self-complementarity).

Catches:
  - CpG dinucleotide density (TLR9 immune response trigger)
  - GC content outside optimal 40-60% range
  - Homopolymer runs (Pol III termination, sequencing errors)
  - Cryptic splice sites (GT...AG pairs at splicing distances)
  - Premature poly-A signals (AATAAA / ATTAAA hexamers)
  - Self-complementary palindromes (hairpin formation)

Usage:
    from noethersolve.audit_sequence import audit_sequence, SequenceReport

    report = audit_sequence("ATGCGATCGATCGAATAAAAATTTTTCG")
    print(report)
    # Shows per-check diagnostics, severity levels, and overall verdict

    # RNA input (U converted to T internally)
    report = audit_sequence("AUGCGAUCGAUCG", seq_type="rna")

    # Helpers
    from noethersolve.audit_sequence import gc_content, cpg_observed_expected
    gc = gc_content("ATGCGATCGATCG")
    oe = cpg_observed_expected("ATGCGATCGATCG")
"""

import re
from dataclasses import dataclass
from typing import List, Optional


# ─── Dataclasses ──────────────────────────────────────────────────────────────

@dataclass
class SequenceIssue:
    """A single quality issue found in a sequence."""
    check_name: str           # CpG_DENSITY, GC_CONTENT, HOMOPOLYMER, etc.
    severity: str             # HIGH, MODERATE, LOW
    message: str
    position: Optional[int] = None   # 0-based position in sequence, if applicable
    value: float = 0.0

    def __str__(self):
        pos_str = f" at pos {self.position}" if self.position is not None else ""
        return f"  [{self.severity}] {self.check_name}{pos_str}: {self.message}"


@dataclass
class SequenceReport:
    """Result of audit_sequence()."""
    sequence_length: int
    gc_content: float
    cpg_density: float
    cpg_oe_ratio: float
    longest_homopolymer: int
    n_cryptic_splice: int
    n_polya_signals: int
    longest_palindrome: int
    issues: List[SequenceIssue]
    verdict: str                      # PASS, WARN, or FAIL

    def __str__(self):
        lines = []
        lines.append(f"{'=' * 60}")
        lines.append(f"  Sequence Design Audit: {self.verdict}")
        lines.append(f"{'=' * 60}")
        lines.append(f"  Length: {self.sequence_length} nt")
        lines.append(f"  GC content: {self.gc_content:.1%}")
        lines.append(f"  CpG density: {self.cpg_density:.4f} "
                     f"(O/E ratio: {self.cpg_oe_ratio:.3f})")
        lines.append(f"  Longest homopolymer: {self.longest_homopolymer} nt")
        lines.append(f"  Cryptic splice sites: {self.n_cryptic_splice}")
        lines.append(f"  Poly-A signals: {self.n_polya_signals}")
        lines.append(f"  Longest palindrome: {self.longest_palindrome} bp")
        lines.append(f"")

        # Issues sorted by severity
        if self.issues:
            lines.append(f"  Issues found:")
            for issue in sorted(self.issues,
                                key=lambda i: {"HIGH": 0, "MODERATE": 1, "LOW": 2}.get(i.severity, 3)):
                lines.append(str(issue))
            lines.append(f"")
        else:
            lines.append(f"  No issues found.")
            lines.append(f"")

        lines.append(f"{'=' * 60}")
        return "\n".join(lines)

    @property
    def passed(self) -> bool:
        return self.verdict == "PASS"


# ─── Helper functions ─────────────────────────────────────────────────────────

def gc_content(seq: str) -> float:
    """Calculate GC content as a fraction of total bases.

    Args:
        seq: nucleotide sequence (ACGT/U, case-insensitive).
            Whitespace is stripped. U is treated as T.

    Returns:
        Fraction of G+C bases (0.0 to 1.0).
    """
    s = seq.upper().replace("U", "T").replace(" ", "").replace("\n", "")
    if len(s) == 0:
        return 0.0
    gc = s.count("G") + s.count("C")
    return gc / len(s)


def cpg_observed_expected(seq: str) -> float:
    """Calculate CpG observed/expected ratio.

    O/E = (count_CpG * N) / (count_C * count_G)

    where N is the sequence length. Values below 0.6 indicate CpG
    suppression (typical of vertebrate genomes). Values above 0.6
    suggest CpG islands. Unmethylated bacterial DNA typically has
    O/E near 1.0.

    Args:
        seq: nucleotide sequence (ACGT/U, case-insensitive).

    Returns:
        CpG observed/expected ratio. Returns 0.0 if C or G count is zero.
    """
    s = seq.upper().replace("U", "T").replace(" ", "").replace("\n", "")
    n = len(s)
    if n < 2:
        return 0.0
    count_c = s.count("C")
    count_g = s.count("G")
    if count_c == 0 or count_g == 0:
        return 0.0
    count_cpg = sum(1 for i in range(n - 1) if s[i] == "C" and s[i + 1] == "G")
    return (count_cpg * n) / (count_c * count_g)


# ─── Internal check functions ─────────────────────────────────────────────────

def _check_cpg_density(seq: str) -> List[SequenceIssue]:
    """Check CpG dinucleotide density.

    CpG dinucleotides in unmethylated DNA trigger TLR9-mediated innate
    immune responses. High CpG density is characteristic of bacterial
    DNA and is immunostimulatory in therapeutic contexts.

    Thresholds:
        >0.04 = HIGH (typical unmethylated bacterial DNA)
        >0.02 = MODERATE
        <=0.02 = OK
    """
    issues = []
    n = len(seq)
    if n < 2:
        return issues

    n_dinucleotides = n - 1
    cpg_count = sum(1 for i in range(n - 1) if seq[i] == "C" and seq[i + 1] == "G")
    density = cpg_count / n_dinucleotides

    if density > 0.04:
        issues.append(SequenceIssue(
            check_name="CpG_DENSITY",
            severity="HIGH",
            message=(f"CpG density {density:.4f} ({cpg_count}/{n_dinucleotides} dinucleotides) "
                     f"exceeds 0.04 — typical of unmethylated bacterial DNA, "
                     f"high TLR9 activation risk"),
            value=density,
        ))
    elif density > 0.02:
        issues.append(SequenceIssue(
            check_name="CpG_DENSITY",
            severity="MODERATE",
            message=(f"CpG density {density:.4f} ({cpg_count}/{n_dinucleotides} dinucleotides) "
                     f"exceeds 0.02 — moderate immune activation risk"),
            value=density,
        ))

    return issues


def _check_gc_content(seq: str) -> List[SequenceIssue]:
    """Check GC content is in the optimal 40-60% therapeutic range.

    Extreme GC content affects mRNA stability, secondary structure
    formation, and codon optimization efficiency.

    Thresholds:
        <30% or >70% = HIGH
        30-40% or 60-70% = MODERATE
        40-60% = OK
    """
    issues = []
    if len(seq) == 0:
        return issues

    gc = (seq.count("G") + seq.count("C")) / len(seq)

    if gc < 0.30 or gc > 0.70:
        issues.append(SequenceIssue(
            check_name="GC_CONTENT",
            severity="HIGH",
            message=(f"GC content {gc:.1%} is outside safe range — "
                     f"{'low GC reduces mRNA stability' if gc < 0.30 else 'high GC promotes secondary structures'}"),
            value=gc,
        ))
    elif gc < 0.40 or gc > 0.60:
        issues.append(SequenceIssue(
            check_name="GC_CONTENT",
            severity="MODERATE",
            message=(f"GC content {gc:.1%} is outside optimal 40-60% range"),
            value=gc,
        ))

    return issues


def _check_homopolymers(seq: str) -> List[SequenceIssue]:
    """Check for homopolymer runs of identical nucleotides.

    Runs of >=4 identical bases cause problems:
    - TTTT (or UUUU in RNA) terminates Pol III transcription (critical for
      guide RNA expression)
    - Long runs of any base cause polymerase slippage and sequencing errors

    Thresholds:
        >=6 of any base = HIGH
        >=4 T = MODERATE (Pol III termination signal)
        >=4 other = LOW
    """
    issues = []
    if len(seq) == 0:
        return issues

    # Find all homopolymer runs
    i = 0
    while i < len(seq):
        base = seq[i]
        run_start = i
        while i < len(seq) and seq[i] == base:
            i += 1
        run_len = i - run_start

        if run_len >= 6:
            issues.append(SequenceIssue(
                check_name="HOMOPOLYMER",
                severity="HIGH",
                message=(f"{base * run_len} ({run_len}-mer) — polymerase slippage "
                         f"and sequencing error risk"),
                position=run_start,
                value=float(run_len),
            ))
        elif run_len >= 4 and base == "T":
            issues.append(SequenceIssue(
                check_name="HOMOPOLYMER",
                severity="MODERATE",
                message=(f"{base * run_len} ({run_len}-mer) — Pol III transcription "
                         f"termination signal"),
                position=run_start,
                value=float(run_len),
            ))
        elif run_len >= 4:
            issues.append(SequenceIssue(
                check_name="HOMOPOLYMER",
                severity="LOW",
                message=f"{base * run_len} ({run_len}-mer)",
                position=run_start,
                value=float(run_len),
            ))

    return issues


def _check_cryptic_splice_sites(seq: str) -> List[SequenceIssue]:
    """Check for cryptic splice site pairs (GT donor...AG acceptor).

    In coding sequences, GT dinucleotides followed by AG within 50-500bp
    can act as cryptic splice sites, causing mRNA mis-splicing and
    loss of the intended protein product.

    Thresholds:
        >3 pairs = HIGH
        1-3 pairs = MODERATE
        0 = OK
    """
    issues = []
    if len(seq) < 4:
        return issues

    # Find all GT (donor) and AG (acceptor) positions
    gt_positions = [i for i in range(len(seq) - 1) if seq[i] == "G" and seq[i + 1] == "T"]
    ag_positions = [i for i in range(len(seq) - 1) if seq[i] == "A" and seq[i + 1] == "G"]

    pairs = []
    for gt_pos in gt_positions:
        for ag_pos in ag_positions:
            distance = ag_pos - gt_pos
            if 50 <= distance <= 500:
                pairs.append((gt_pos, ag_pos))

    if len(pairs) > 3:
        issues.append(SequenceIssue(
            check_name="CRYPTIC_SPLICE",
            severity="HIGH",
            message=(f"{len(pairs)} GT...AG pairs within splicing distance (50-500bp) — "
                     f"high mis-splicing risk"),
            value=float(len(pairs)),
        ))
    elif len(pairs) >= 1:
        issues.append(SequenceIssue(
            check_name="CRYPTIC_SPLICE",
            severity="MODERATE",
            message=(f"{len(pairs)} GT...AG pair{'s' if len(pairs) > 1 else ''} "
                     f"within splicing distance (50-500bp)"),
            value=float(len(pairs)),
        ))

    return issues


def _check_polya_signals(seq: str) -> List[SequenceIssue]:
    """Check for premature polyadenylation signals.

    AATAAA and ATTAAA hexamers in coding regions cause premature
    cleavage and polyadenylation, truncating the mRNA transcript
    before the intended stop codon.

    Any occurrence = HIGH (causes truncation).
    """
    issues = []
    signals = ["AATAAA", "ATTAAA"]

    for signal in signals:
        start = 0
        while True:
            pos = seq.find(signal, start)
            if pos == -1:
                break
            issues.append(SequenceIssue(
                check_name="POLYA_SIGNAL",
                severity="HIGH",
                message=(f"{signal} hexamer — premature polyadenylation signal "
                         f"causes transcript truncation"),
                position=pos,
                value=1.0,
            ))
            start = pos + 1

    return issues


def _check_self_complementarity(seq: str) -> List[SequenceIssue]:
    """Check for internal palindromic sequences that form hairpin structures.

    DNA/RNA sequences with internal complementary regions fold back on
    themselves, forming hairpins that reduce translation efficiency,
    interfere with PCR amplification, and cause probe binding issues.

    Checks for reverse-complement palindromes of length >= 8bp.

    Thresholds:
        >=12bp palindrome = HIGH
        >=8bp palindrome = MODERATE
        <8bp = OK
    """
    issues = []
    n = len(seq)
    if n < 8:
        return issues

    complement = {"A": "T", "T": "A", "C": "G", "G": "C"}

    longest = 0
    longest_pos = 0

    # Check all substrings of length >= 8 for being palindromic
    # (equal to own reverse complement)
    # Start from longest possible and work down for efficiency
    for length in range(min(n, 30), 7, -1):  # cap at 30 for performance
        if longest >= length:
            break  # already found one this long or longer
        for start in range(n - length + 1):
            subseq = seq[start:start + length]
            # Build reverse complement
            try:
                rev_comp = "".join(complement[b] for b in reversed(subseq))
            except KeyError:
                continue  # skip if non-standard base
            if subseq == rev_comp:
                if length > longest:
                    longest = length
                    longest_pos = start
                break  # found one at this length, move to next

    if longest >= 12:
        issues.append(SequenceIssue(
            check_name="SELF_COMPLEMENT",
            severity="HIGH",
            message=(f"{longest}bp palindrome (self-complementary region) — "
                     f"strong hairpin formation risk"),
            position=longest_pos,
            value=float(longest),
        ))
    elif longest >= 8:
        issues.append(SequenceIssue(
            check_name="SELF_COMPLEMENT",
            severity="MODERATE",
            message=(f"{longest}bp palindrome (self-complementary region) — "
                     f"potential hairpin formation"),
            position=longest_pos,
            value=float(longest),
        ))

    return issues


# ─── Core audit function ─────────────────────────────────────────────────────

def audit_sequence(seq: str, seq_type: str = "dna") -> SequenceReport:
    """Audit a nucleotide sequence for therapeutic design issues.

    Runs six checks: CpG density, GC content, homopolymer runs, cryptic
    splice sites, premature poly-A signals, and self-complementarity.

    Args:
        seq: nucleotide sequence (A, C, G, T/U characters + whitespace).
            Case-insensitive. Whitespace is stripped.
        seq_type: "dna" or "rna". RNA sequences have U converted to T
            internally for analysis.

    Returns:
        SequenceReport with per-check results and overall verdict.

    Raises:
        ValueError: if seq_type is not "dna" or "rna", or if the sequence
            contains invalid characters.
    """
    # ── Validate seq_type ─────────────────────────────────────────────
    seq_type = seq_type.lower()
    if seq_type not in ("dna", "rna"):
        raise ValueError(f"seq_type must be 'dna' or 'rna', got '{seq_type}'")

    # ── Clean and validate sequence ───────────────────────────────────
    s = seq.upper().replace(" ", "").replace("\n", "").replace("\r", "").replace("\t", "")

    if seq_type == "rna":
        s = s.replace("U", "T")

    # Validate characters
    valid = set("ACGT")
    invalid = set(s) - valid
    if invalid:
        raise ValueError(f"Invalid characters in sequence: {sorted(invalid)}. "
                         f"Only A, C, G, T{'/U' if seq_type == 'rna' else ''} "
                         f"and whitespace are allowed.")

    if len(s) == 0:
        return SequenceReport(
            sequence_length=0,
            gc_content=0.0,
            cpg_density=0.0,
            cpg_oe_ratio=0.0,
            longest_homopolymer=0,
            n_cryptic_splice=0,
            n_polya_signals=0,
            longest_palindrome=0,
            issues=[],
            verdict="PASS",
        )

    # ── Run all checks ────────────────────────────────────────────────
    all_issues = []

    all_issues.extend(_check_cpg_density(s))
    all_issues.extend(_check_gc_content(s))
    all_issues.extend(_check_homopolymers(s))
    all_issues.extend(_check_cryptic_splice_sites(s))
    all_issues.extend(_check_polya_signals(s))
    all_issues.extend(_check_self_complementarity(s))

    # ── Compute summary metrics ───────────────────────────────────────
    seq_gc = gc_content(s)
    seq_cpg_oe = cpg_observed_expected(s)

    n_dinucleotides = len(s) - 1
    cpg_count = sum(1 for i in range(n_dinucleotides) if s[i] == "C" and s[i + 1] == "G")
    cpg_dens = cpg_count / n_dinucleotides if n_dinucleotides > 0 else 0.0

    # Longest homopolymer
    longest_homo = 0
    i = 0
    while i < len(s):
        run_start = i
        while i < len(s) and s[i] == s[run_start]:
            i += 1
        longest_homo = max(longest_homo, i - run_start)

    # Cryptic splice site count
    gt_positions = [i for i in range(len(s) - 1) if s[i] == "G" and s[i + 1] == "T"]
    ag_positions = [i for i in range(len(s) - 1) if s[i] == "A" and s[i + 1] == "G"]
    n_splice = sum(1 for gt in gt_positions for ag in ag_positions if 50 <= ag - gt <= 500)

    # Poly-A signal count
    n_polya = sum(len(re.findall(sig, s)) for sig in ["AATAAA", "ATTAAA"])

    # Longest palindrome (from issues, or 0)
    palindrome_issues = [iss for iss in all_issues if iss.check_name == "SELF_COMPLEMENT"]
    longest_pal = int(max((iss.value for iss in palindrome_issues), default=0))

    # ── Verdict ───────────────────────────────────────────────────────
    has_high = any(iss.severity == "HIGH" for iss in all_issues)
    has_moderate = any(iss.severity == "MODERATE" for iss in all_issues)

    if has_high:
        verdict = "FAIL"
    elif has_moderate:
        verdict = "WARN"
    else:
        verdict = "PASS"

    return SequenceReport(
        sequence_length=len(s),
        gc_content=seq_gc,
        cpg_density=cpg_dens,
        cpg_oe_ratio=seq_cpg_oe,
        longest_homopolymer=longest_homo,
        n_cryptic_splice=n_splice,
        n_polya_signals=n_polya,
        longest_palindrome=longest_pal,
        issues=all_issues,
        verdict=verdict,
    )
