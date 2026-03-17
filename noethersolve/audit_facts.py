"""
noethersolve.audit_facts — Oracle fact file quality auditor.

Checks *_facts.json files for token-length bias and other quality issues that
prevent adapter training from flipping facts. The #1 failure mode across all
domains was token-length bias: when a distractor is shorter than the truth,
the base LLM picks it on length alone and no amount of adapter training flips
the margin.

Key finding (Length Ratio Discovery, 2026-03-16):
  - Length ratio = truth_len / min(distractor_lens)
  - Correlation with baseline accuracy: r = -0.742 (strong negative)
  - Ratio < 1.2: 64% baseline (easy)
  - Ratio 1.2-2.5: 13% baseline (hard)
  - Ratio > 2.5: 7% baseline (very hard)

Key finding (Linguistic Hedge Predictor, 2026-03-16):
  - Hedge words (may, might, possibly, etc.) predict mean-scoring failure
  - Confidence words (exactly, proven, always, etc.) predict mean-scoring success
  - Zero-shot accuracy: 72% (no model inference needed)
  - Perfect domain-level correlation: r = -1.000 between hedged% and mean pass%

Catches:
  - Token-length bias (distractor shorter than truth)
  - Length ratio issues (truth too long relative to distractors)
  - Substring distractors (distractor is a prefix/substring of truth)

Usage:
    from noethersolve.audit_facts import audit_facts, FactAuditReport
    from noethersolve.audit_facts import predict_difficulty, predict_mean_pass

    report = audit_facts("problems/chemical_conservation_facts.json")
    print(report)
    # Shows per-fact diagnostics, risk levels, and overall summary

    prediction = predict_difficulty("problems/my_facts.json")
    print(f"Predicted sum pass: {prediction['sum_pct']:.0f}%")
    print(f"Predicted mean pass: {prediction['mean_pct']:.0f}%")

CLI:
    python -m noethersolve.audit_facts --all
    python -m noethersolve.audit_facts --file problems/my_facts.json
    python -m noethersolve.audit_facts --all --predict-difficulty
    python -m noethersolve.audit_facts --file problems/my_facts.json --predict-difficulty
"""

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union


# ─── Technical Simplification Bias markers ───────────────────────────────────
# Discovery: LLMs prefer simple/familiar terms over precise technical language.
# t = -3.73, p = 0.0004 (highly significant)
# See: results/discoveries/novel_findings/technical_simplification_bias.md

TECHNICAL_MARKERS = [
    # Math notation
    'ln(r)', 'log', 'sqrt', 'π', 'exp', 'integral', '∫', '∂', '∇',
    # Fluid mechanics
    'enstrophy', 'vorticity', 'advection', 'dissipation', 'helicity',
    # Physics jargon
    'quasi-normal', 'supertranslation', 'holographic', 'geodesic',
    # Subtle distinctions
    'deficit', 'asymmetry', 'hierarchy', 'ordering', 'degeneracy',
    # Hedging/nuance
    'tension', 'disagree', 'uncertain', 'pending', 'anomaly',
    # Nuanced terms
    'model-dependent', 'viable', 'consistent', 'favored', 'excluded',
]

SIMPLE_MARKERS = [
    # Basic physics
    'energy', 'momentum', 'mass', 'force', 'velocity', 'acceleration',
    # Definitive outcomes
    'confirmed', 'proven', 'discovered', 'detected', 'found', 'shows',
    # Absolutism
    'perfect', 'exact', 'precisely', 'always', 'all', 'every', 'completely',
    # Closure
    'explained', 'resolved', 'determined', 'established', 'known',
    # Basic concepts
    'particle', 'wave', 'field', 'direct', 'simple',
]


def _count_technical(text: str) -> int:
    """Count technical markers in text."""
    text_lower = text.lower()
    return sum(1 for m in TECHNICAL_MARKERS if m.lower() in text_lower)


def _count_simple(text: str) -> int:
    """Count simple/familiar markers in text."""
    text_lower = text.lower()
    return sum(1 for m in SIMPLE_MARKERS if m.lower() in text_lower)


def technical_ratio(truth: str, distractor: str) -> float:
    """Compute technical complexity ratio (truth / distractor).

    Ratio > 1.5 indicates high failure risk due to technical simplification bias.
    """
    truth_tech = _count_technical(truth)
    dist_tech = _count_technical(distractor)
    return (truth_tech + 0.5) / (dist_tech + 0.5)


# ─── Certainty Contamination Bias markers ────────────────────────────────────
# Discovery: LLMs prefer definitive claims over hedged scientific language.
# Correlation r = -0.402 between certainty gap and oracle margin.
# See: results/discoveries/novel_findings/certainty_contamination_bias.md

CERTAINTY_MARKERS = [
    'definitively', 'completely', 'proven', 'ruled out',
    'impossible', 'always', 'never', 'guaranteed',
    'certain', 'absolutely', 'all ', 'none', 'every',
    'must', 'cannot', 'does not exist', 'no ', 'zero',
    'perfect', 'exactly', 'precise', 'fundamentally',
    'whatsoever', 'entirely', 'permanently', 'universal'
]

HEDGING_MARKERS = [
    'may', 'might', 'could', 'uncertain', 'varies',
    'approximately', 'suggests', 'indicates', 'possible',
    'likely', 'probably', 'tentative', 'preliminary',
    'not ruled out', 'remains open', 'still debated',
    'large uncertainties', 'significance varies',
    'consistent', 'some', 'limited', 'current', 'hints',
    'awaits confirmation', 'inconclusive', 'not precisely'
]


def _count_markers(text: str, markers: List[str]) -> int:
    """Count how many markers appear in text."""
    text_lower = text.lower()
    return sum(1 for m in markers if m in text_lower)


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

        # ── Certainty bias check ─────────────────────────────────────────
        # Discovery: distractors with more certainty markers than truth
        # cause oracle failures (r = -0.402, t = 3.57).
        truth_certainty = _count_markers(truth, CERTAINTY_MARKERS)
        max_dist_certainty = 0
        worst_dist_idx = 0
        for idx, d in enumerate(distractors):
            d_certainty = _count_markers(d, CERTAINTY_MARKERS)
            if d_certainty > max_dist_certainty:
                max_dist_certainty = d_certainty
                worst_dist_idx = idx

        certainty_gap = max_dist_certainty - truth_certainty

        if certainty_gap >= 4:
            issue = FactIssue(
                fact_id=fact_id,
                issue_type="CERTAINTY_BIAS",
                severity="HIGH",
                description=(
                    f"CRITICAL certainty gap={certainty_gap}: distractor has {max_dist_certainty} "
                    f"certainty markers vs truth's {truth_certainty}. "
                    f"Rewrite distractor with hedged language."
                ),
                distractor_idx=worst_dist_idx,
                details={"certainty_gap": certainty_gap, "dist_certainty": max_dist_certainty},
            )
            issues.append(issue)
        elif certainty_gap >= 3:
            issue = FactIssue(
                fact_id=fact_id,
                issue_type="CERTAINTY_BIAS",
                severity="MODERATE",
                description=(
                    f"High certainty gap={certainty_gap}: distractor has {max_dist_certainty} "
                    f"certainty markers vs truth's {truth_certainty}."
                ),
                distractor_idx=worst_dist_idx,
                details={"certainty_gap": certainty_gap, "dist_certainty": max_dist_certainty},
            )
            issues.append(issue)

        # ── Technical simplification bias check ───────────────────────────
        # Discovery: when truth uses technical jargon and distractors use
        # simple/familiar terms, the model picks the simpler answer.
        # t = -3.73, p = 0.0004 (highly significant)
        truth_tech = _count_technical(truth)
        truth_simple = _count_simple(truth)
        max_tech_ratio = 0.0
        worst_tech_idx = 0

        for idx, d in enumerate(distractors):
            dist_tech = _count_technical(d)
            dist_simple = _count_simple(d)

            # Technical gap: truth is technical, distractor is simple
            # Positive gap = high failure risk
            tech_gap = (truth_tech - dist_tech) + (dist_simple - truth_simple)
            ratio = technical_ratio(truth, d)

            if ratio > max_tech_ratio:
                max_tech_ratio = ratio
                worst_tech_idx = idx

        # Flag if truth is significantly more technical
        if truth_tech >= 2 and max_tech_ratio >= 2.0:
            issue = FactIssue(
                fact_id=fact_id,
                issue_type="TECHNICAL_BIAS",
                severity="HIGH",
                description=(
                    f"CRITICAL technical gap: truth has {truth_tech} technical markers, "
                    f"distractor[{worst_tech_idx}] is simpler (ratio={max_tech_ratio:.1f}). "
                    f"Use equally technical distractor."
                ),
                distractor_idx=worst_tech_idx,
                details={"truth_technical": truth_tech, "tech_ratio": max_tech_ratio},
            )
            issues.append(issue)
        elif truth_tech >= 1 and max_tech_ratio >= 1.5:
            # Check if distractor has simple markers making it attractive
            worst_dist = distractors[worst_tech_idx] if distractors else ""
            dist_simple = _count_simple(worst_dist)
            if dist_simple >= 2:
                issue = FactIssue(
                    fact_id=fact_id,
                    issue_type="TECHNICAL_BIAS",
                    severity="MODERATE",
                    description=(
                        f"Technical imbalance: truth uses technical language ({truth_tech} markers), "
                        f"distractor[{worst_tech_idx}] uses simple terms ({dist_simple} simple markers)."
                    ),
                    distractor_idx=worst_tech_idx,
                    details={"truth_technical": truth_tech, "dist_simple": dist_simple},
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


# ─── Length ratio analysis (from Length Ratio Discovery) ────────────────────

def analyze_length_ratio(path_or_dict) -> dict:
    """Analyze length ratio statistics for a fact file.

    Returns:
        dict with avg_ratio, max_ratio, predicted_baseline, and per-fact details.
    """
    if isinstance(path_or_dict, str):
        with open(path_or_dict, "r") as f:
            data = json.load(f)
    else:
        data = path_or_dict

    facts = data.get("facts", data.get("examples", []))
    results = []

    for fact in facts:
        fact_id = fact.get("id", "unknown")
        truth = fact.get("truth", "")
        distractors = fact.get("distractors", [])

        if not distractors:
            continue

        truth_len = len(truth)
        min_distractor_len = min(len(d) for d in distractors)

        if min_distractor_len == 0:
            ratio = 999.0
        else:
            ratio = truth_len / min_distractor_len

        results.append({
            "fact_id": fact_id,
            "truth_len": truth_len,
            "min_distractor_len": min_distractor_len,
            "length_ratio": ratio,
        })

    if not results:
        return {"avg_ratio": 0, "max_ratio": 0, "predicted_baseline": "N/A", "facts": []}

    avg_ratio = sum(r["length_ratio"] for r in results) / len(results)
    max_ratio = max(r["length_ratio"] for r in results)

    # Predict baseline based on correlation analysis
    if avg_ratio < 1.2:
        predicted = "HIGH (>50%)"
    elif avg_ratio < 2.5:
        predicted = "MEDIUM (15-50%)"
    else:
        predicted = "LOW (<15%)"

    return {
        "avg_ratio": avg_ratio,
        "max_ratio": max_ratio,
        "predicted_baseline": predicted,
        "facts": results,
    }


# ─── Linguistic hedge predictor (from Linguistic Hedge Predictor Discovery) ──

HEDGE_WORDS = {
    "may", "might", "could", "possibly", "potentially", "likely", "unlikely",
    "suggests", "suggesting", "uncertain", "unknown", "unclear",
    "remains", "pending", "viable", "possible", "perhaps", "probably",
    "estimated", "approximately", "roughly", "about", "some",
    "appears", "seems", "indicates", "implies", "but",
}

CONFIDENCE_WORDS = {
    "exactly", "precisely", "always", "never", "must", "definitely", "certainly",
    "proven", "confirmed", "established", "demonstrated", "known", "guaranteed",
    "is", "are", "has", "have", "will", "does",
}


def _hedge_score(text: str) -> int:
    """Compute linguistic hedge score. Positive = hedged, negative = confident."""
    import re
    words = re.findall(r'\b\w+\b', text.lower())
    word_set = set(words)

    hedge_count = len(word_set & HEDGE_WORDS)
    confidence_count = len(word_set & CONFIDENCE_WORDS)

    score = hedge_count - confidence_count

    if '(' in text:  # Parentheticals add hedging
        score += 1
    if len(words) > 15:  # Long explanations add hedging
        score += 1

    return score


def predict_mean_pass(fact: dict) -> bool:
    """Predict if fact will pass mean-normalized scoring (no model needed).

    Based on Linguistic Hedge Predictor Discovery: 72% accuracy.
    """
    truth = fact.get("truth", "")
    distractors = fact.get("distractors", [])

    if not distractors:
        return True

    truth_score = _hedge_score(truth)
    min_dist_score = min(_hedge_score(d) for d in distractors)

    return truth_score <= min_dist_score


def predict_sum_pass(fact: dict) -> bool:
    """Predict if fact will pass sum scoring (no model needed)."""
    truth = fact.get("truth", "")
    distractors = fact.get("distractors", [])

    if not distractors:
        return True

    truth_len = len(truth)
    min_dist_len = min(len(d) for d in distractors)

    if min_dist_len == 0:
        return True

    length_ratio = truth_len / min_dist_len
    truth_hedge = _hedge_score(truth)
    min_dist_hedge = min(_hedge_score(d) for d in distractors)

    # Sum passes if: shorter AND not too much more hedged
    if length_ratio < 1.2:
        return True
    if truth_hedge < min_dist_hedge - 1:
        return True
    if length_ratio < 1.5 and truth_hedge <= min_dist_hedge:
        return True
    return False


def predict_difficulty(path_or_dict) -> dict:
    """Zero-shot oracle difficulty prediction using linguistic features.

    No model inference needed. ~72% accuracy on mean scoring predictions.

    Returns:
        dict with predicted sum/mean pass rates and per-fact predictions.
    """
    if isinstance(path_or_dict, str):
        with open(path_or_dict, "r") as f:
            data = json.load(f)
    else:
        data = path_or_dict

    facts = data.get("facts", data.get("examples", []))
    results = []

    for fact in facts:
        fact_id = fact.get("id", "unknown")
        sum_pred = predict_sum_pass(fact)
        mean_pred = predict_mean_pass(fact)
        truth_hedge = _hedge_score(fact.get("truth", ""))

        results.append({
            "fact_id": fact_id,
            "sum_pred": sum_pred,
            "mean_pred": mean_pred,
            "truth_hedge_score": truth_hedge,
        })

    n = len(results)
    if n == 0:
        return {"sum_pct": 0, "mean_pct": 0, "facts": []}

    sum_pct = 100 * sum(r["sum_pred"] for r in results) / n
    mean_pct = 100 * sum(r["mean_pred"] for r in results) / n

    return {
        "sum_pct": sum_pct,
        "mean_pct": mean_pct,
        "facts": results,
    }


# ─── CLI ────────────────────────────────────────────────────────────────────

def main():
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description="Audit oracle fact files for length ratio and quality issues"
    )
    parser.add_argument(
        "--file", type=Path,
        help="Audit a specific fact file"
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Audit all fact files in problems/"
    )
    parser.add_argument(
        "--check-lengths", action="store_true",
        help="Show length ratio analysis (from Length Ratio Discovery)"
    )
    parser.add_argument(
        "--predict-difficulty", action="store_true",
        help="Zero-shot difficulty prediction using linguistic features (no model needed)"
    )
    args = parser.parse_args()

    if args.file:
        if args.predict_difficulty:
            prediction = predict_difficulty(str(args.file))
            print(f"File: {args.file}")
            print(f"  Zero-shot difficulty prediction (72% accurate on mean scoring)")
            print()
            print(f"  Predicted sum pass rate:  {prediction['sum_pct']:.0f}%")
            print(f"  Predicted mean pass rate: {prediction['mean_pct']:.0f}%")
            print()
            print("  Per-fact predictions:")
            print(f"  {'ID':<15} {'Sum':>6} {'Mean':>6} {'Hedge':>6}")
            print("  " + "-" * 35)
            for f in prediction["facts"]:
                sum_str = "PASS" if f["sum_pred"] else "FAIL"
                mean_str = "PASS" if f["mean_pred"] else "FAIL"
                print(f"  {f['fact_id']:<15} {sum_str:>6} {mean_str:>6} {f['truth_hedge_score']:>6}")
        elif args.check_lengths:
            analysis = analyze_length_ratio(str(args.file))
            print(f"File: {args.file}")
            print(f"  Avg length ratio: {analysis['avg_ratio']:.2f}")
            print(f"  Max length ratio: {analysis['max_ratio']:.2f}")
            print(f"  Predicted baseline: {analysis['predicted_baseline']}")
            print()
            print("  Facts with ratio > 2.0:")
            for f in analysis["facts"]:
                if f["length_ratio"] > 2.0:
                    print(f"    {f['fact_id']}: {f['length_ratio']:.1f} "
                          f"(truth={f['truth_len']}, dist={f['min_distractor_len']})")
        else:
            report = audit_facts(str(args.file))
            print(report)

    elif args.all:
        # Find problems directory relative to this file
        script_dir = Path(__file__).parent.parent
        problems_dir = script_dir / "problems"

        if not problems_dir.exists():
            print(f"Problems directory not found: {problems_dir}")
            return

        facts_files = sorted(problems_dir.glob("*_facts.json"))

        if args.predict_difficulty:
            print("=" * 75)
            print("ZERO-SHOT DIFFICULTY PREDICTION (see linguistic_hedge_predictor.md)")
            print("72% accurate on mean scoring, no model inference needed")
            print("=" * 75)
            print()
            print(f"{'Domain':<35} {'N':>3} {'Sum%':>6} {'Mean%':>6}")
            print("-" * 55)

            results = []
            for f in facts_files:
                try:
                    prediction = predict_difficulty(str(f))
                    domain = f.stem.replace("_facts", "")
                    n = len(prediction["facts"])
                    results.append((domain, n, prediction["sum_pct"], prediction["mean_pct"]))
                except Exception as e:
                    print(f"Error: {f}: {e}")

            # Sort by mean pass rate (hardest first)
            results.sort(key=lambda x: x[3])

            for domain, n, sum_pct, mean_pct in results:
                print(f"{domain:<35} {n:>3} {sum_pct:>5.0f}% {mean_pct:>5.0f}%")

            print()
            print("Interpretation:")
            print("  - Low mean% + high sum% = hedged truths → use SUM scoring")
            print("  - High mean% + low sum% = verbose truths → use MEAN scoring")
            print("  - Low both = hard domain → needs incoherent distractors")

        elif args.check_lengths:
            print("=" * 75)
            print("LENGTH RATIO ANALYSIS (see length_ratio_discovery.md)")
            print("=" * 75)
            print()
            print(f"{'Domain':<35} {'N':>3} {'AvgRatio':>8} {'MaxRatio':>8} {'Predicted':<15}")
            print("-" * 75)

            results = []
            for f in facts_files:
                try:
                    analysis = analyze_length_ratio(str(f))
                    domain = f.stem.replace("_facts", "")
                    results.append((domain, len(analysis["facts"]),
                                   analysis["avg_ratio"], analysis["max_ratio"],
                                   analysis["predicted_baseline"]))
                except Exception as e:
                    print(f"Error: {f}: {e}")

            # Sort by avg ratio (worst first)
            results.sort(key=lambda x: -x[2])

            for domain, n, avg, maxr, pred in results:
                print(f"{domain:<35} {n:>3} {avg:>8.2f} {maxr:>8.1f} {pred:<15}")

            print()
            print("Correlation with baseline: r = -0.742")
            print("Recommendation: Keep ratio < 1.5 for best results")

        else:
            for f in facts_files:
                report = audit_facts(str(f))
                if report.verdict != "PASS":
                    print(f"\n{f.name}: {report.verdict}")
                    for issue in report.issues[:3]:
                        print(f"  {issue}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
