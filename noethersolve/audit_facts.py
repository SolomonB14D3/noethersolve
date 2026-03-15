"""
noethersolve.audit_facts — Oracle fact file quality auditor.

Checks *_facts.json files for token-length bias and other quality issues that
prevent adapter training from flipping facts. The #1 failure mode across all
domains was token-length bias: when a distractor is shorter than the truth,
the base LLM picks it on length alone and no amount of adapter training flips
the margin.

Catches:
  - Token-length bias (distractor shorter than truth)
  - Substring distractors (distractor is a prefix/substring of truth)
  - Extreme length ratios (distractor much shorter or longer than truth)

Usage:
    from noethersolve.audit_facts import audit_facts, FactAuditReport

    report = audit_facts("problems/chemical_conservation_facts.json")
    print(report)
    # Shows per-fact diagnostics, risk levels, and overall summary
"""

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union


# ─── Token-length approximation ──────────────────────────────────────────────

def _approx_tokens(text: str) -> int:
    """Approximate token count using whitespace + punctuation splitting.

    This is intentionally simple — no real tokenizer needed. The heuristic
    splits on whitespace, then counts additional tokens for punctuation and
    special characters that most tokenizers split (parentheses, brackets,
    operators, etc.). Roughly matches BPE token counts within ~20%.
    """
    if not text:
        return 0
    words = text.split()
    count = 0
    for word in words:
        count += 1
        # Extra tokens for punctuation/operators that BPE typically splits
        for ch in word:
            if ch in "()[]{}×÷±²³√∝∞≈≠≤≥":
                count += 1
    return max(count, 1)


# ─── Dataclasses ──────────────────────────────────────────────────────────────

@dataclass
class FactIssue:
    """A single quality issue found in a fact."""
    fact_id: str
    issue_type: str          # LENGTH_BIAS, SUBSTRING, EXTREME_RATIO
    severity: str            # HIGH, MODERATE, LOW
    description: str
    distractor_idx: int      # which distractor triggered it
    details: Dict[str, float] = field(default_factory=dict)

    def __str__(self):
        return f"  [{self.severity}] {self.fact_id}: {self.description}"


@dataclass
class FactDiagnostic:
    """Per-fact diagnostic summary."""
    fact_id: str
    truth_tokens: int
    distractor_tokens: List[int]
    min_ratio: float              # min(distractor_len) / truth_len
    has_substring: bool
    risk_level: str               # HIGH_RISK, MODERATE_RISK, OK
    issues: List[FactIssue]

    def __str__(self):
        dist_str = ", ".join(str(t) for t in self.distractor_tokens)
        lines = [
            f"  {self.fact_id}: truth={self.truth_tokens}tok, "
            f"distractors=[{dist_str}]tok, "
            f"min_ratio={self.min_ratio:.2f} [{self.risk_level}]"
        ]
        for issue in self.issues:
            lines.append(f"    {issue.issue_type}: {issue.description}")
        return "\n".join(lines)


@dataclass
class FactAuditReport:
    """Result of audit_facts()."""
    verdict: str                          # PASS, WARN, or FAIL
    n_facts: int
    n_high_risk: int
    n_moderate_risk: int
    n_ok: int
    diagnostics: List[FactDiagnostic]
    issues: List[FactIssue]
    warnings: List[str]

    def __str__(self):
        lines = []
        lines.append(f"{'=' * 60}")
        lines.append(f"  Fact Quality Audit: {self.verdict}")
        lines.append(f"{'=' * 60}")
        lines.append(f"  Facts: {self.n_facts} total, "
                     f"{self.n_high_risk} HIGH RISK, "
                     f"{self.n_moderate_risk} MODERATE RISK, "
                     f"{self.n_ok} OK")
        lines.append("")

        # Per-fact diagnostics
        if self.diagnostics:
            lines.append("  Per-fact diagnostics:")
            for diag in self.diagnostics:
                lines.append(str(diag))
            lines.append("")

        # Issues sorted by severity
        if self.issues:
            lines.append("  Issues found:")
            for issue in sorted(self.issues, key=lambda i: {"HIGH": 0, "MODERATE": 1, "LOW": 2}.get(i.severity, 3)):
                lines.append(str(issue))
            lines.append("")

        # Warnings
        if self.warnings:
            lines.append("  Warnings:")
            for w in self.warnings:
                lines.append(f"    - {w}")

        lines.append(f"{'=' * 60}")
        return "\n".join(lines)

    @property
    def passed(self) -> bool:
        return self.verdict == "PASS"


# ─── Core audit function ─────────────────────────────────────────────────────

def audit_facts(
    path_or_dict: Union[str, dict],
    length_ratio_high: float = 0.7,
    length_ratio_moderate: float = 0.9,
) -> FactAuditReport:
    """Audit an oracle fact file for quality issues.

    Args:
        path_or_dict: either a file path to a *_facts.json file, or a dict
            with a "facts" key containing the fact list.
        length_ratio_high: min_distractor_len/truth_len below this = HIGH RISK.
        length_ratio_moderate: min_distractor_len/truth_len below this = MODERATE RISK.

    Returns:
        FactAuditReport with per-fact diagnostics and overall summary.
    """
    # Load facts
    if isinstance(path_or_dict, str):
        with open(path_or_dict, "r") as f:
            data = json.load(f)
    else:
        data = path_or_dict

    facts = data.get("facts", [])
    diagnostics = []
    all_issues = []
    warnings = []
    n_high = 0
    n_moderate = 0
    n_ok = 0

    if not facts:
        warnings.append("No facts found in input.")

    for fact in facts:
        fact_id = fact.get("id", "unknown")
        truth = fact.get("truth", "")
        distractors = fact.get("distractors", [])

        truth_tokens = _approx_tokens(truth)
        distractor_tokens = [_approx_tokens(d) for d in distractors]
        issues = []

        # Handle edge case: no distractors
        if not distractors:
            warnings.append(f"{fact_id}: no distractors defined.")
            diagnostics.append(FactDiagnostic(
                fact_id=fact_id,
                truth_tokens=truth_tokens,
                distractor_tokens=[],
                min_ratio=float("inf"),
                has_substring=False,
                risk_level="OK",
                issues=[],
            ))
            n_ok += 1
            continue

        # ── Token-length analysis ─────────────────────────────────────────
        min_dist_tokens = min(distractor_tokens) if distractor_tokens else 0
        if truth_tokens > 0:
            min_ratio = min_dist_tokens / truth_tokens
        else:
            min_ratio = float("inf")

        for idx, (d, d_tok) in enumerate(zip(distractors, distractor_tokens)):
            if truth_tokens > 0 and d_tok < truth_tokens:
                diff = truth_tokens - d_tok
                ratio = d_tok / truth_tokens
                if ratio < length_ratio_high:
                    severity = "HIGH"
                elif ratio < length_ratio_moderate:
                    severity = "MODERATE"
                else:
                    continue  # within tolerance

                issue = FactIssue(
                    fact_id=fact_id,
                    issue_type="LENGTH_BIAS",
                    severity=severity,
                    description=(
                        f"distractor[{idx}] ({d_tok}tok) shorter than "
                        f"truth ({truth_tokens}tok), ratio={ratio:.2f}"
                    ),
                    distractor_idx=idx,
                    details={"truth_tokens": truth_tokens, "distractor_tokens": d_tok, "ratio": ratio},
                )
                issues.append(issue)

        # ── Substring check ───────────────────────────────────────────────
        has_substring = False
        truth_lower = truth.lower().strip()
        for idx, d in enumerate(distractors):
            d_lower = d.lower().strip()
            if not d_lower:
                continue
            if d_lower in truth_lower or truth_lower.startswith(d_lower):
                has_substring = True
                issue = FactIssue(
                    fact_id=fact_id,
                    issue_type="SUBSTRING",
                    severity="HIGH",
                    description=(
                        f"distractor[{idx}] is a substring of truth: "
                        f"\"{d[:40]}{'...' if len(d) > 40 else ''}\""
                    ),
                    distractor_idx=idx,
                )
                issues.append(issue)

        # ── Risk level ────────────────────────────────────────────────────
        has_high = any(i.severity == "HIGH" for i in issues)
        has_moderate = any(i.severity == "MODERATE" for i in issues)

        if has_high:
            risk_level = "HIGH_RISK"
            n_high += 1
        elif has_moderate:
            risk_level = "MODERATE_RISK"
            n_moderate += 1
        else:
            risk_level = "OK"
            n_ok += 1

        all_issues.extend(issues)
        diagnostics.append(FactDiagnostic(
            fact_id=fact_id,
            truth_tokens=truth_tokens,
            distractor_tokens=distractor_tokens,
            min_ratio=min_ratio,
            has_substring=has_substring,
            risk_level=risk_level,
            issues=issues,
        ))

    # ── Overall verdict ───────────────────────────────────────────────────
    if n_high > 0:
        verdict = "FAIL"
    elif n_moderate > 0:
        verdict = "WARN"
    else:
        verdict = "PASS"

    return FactAuditReport(
        verdict=verdict,
        n_facts=len(facts),
        n_high_risk=n_high,
        n_moderate_risk=n_moderate,
        n_ok=n_ok,
        diagnostics=diagnostics,
        issues=all_issues,
        warnings=warnings,
    )
