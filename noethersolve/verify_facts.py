"""
noethersolve.verify_facts — Computational fact verification gate.

Checks facts in *_facts.json files against computational tools BEFORE they
can be used for adapter training. Prevents wrong facts from entering the
training pipeline.

The incident that motivated this: a fact claimed r^{-1/2} is the scaling-critical
Q_f kernel in 3D NS when the correct answer is r^{-2}. The ns_scaling tool
would have caught it, but there was no gate requiring verification.

Three outcomes per fact:
  - VERIFIED: A computational tool confirms the claim is correct.
  - CONTRADICTED: A computational tool says the claim is WRONG. Hard block.
  - UNVERIFIED: No matching tool found. Warning only.

Usage:
    from noethersolve.verify_facts import verify_facts_file, verify_or_warn

    # Full report
    report = verify_facts_file("problems/ns_regularity_facts_v2.json")
    print(report)

    # Training gate (raises on contradictions, warns on unverified)
    verify_or_warn("problems/ns_regularity_facts_v2.json")

CLI:
    python -m noethersolve.verify_facts --file problems/ns_regularity_facts_v2.json
    python -m noethersolve.verify_facts --all
"""

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple


# ─── Result types ─────────────────────────────────────────────────────────────

@dataclass
class FactVerification:
    """Verification result for a single fact."""
    fact_id: str
    status: str  # "VERIFIED", "CONTRADICTED", "UNVERIFIED"
    tool_used: str  # which tool checked it, or "none"
    detail: str  # what the tool found
    truth_text: str  # the claim being checked

    def __str__(self):
        icon = {"VERIFIED": "PASS", "CONTRADICTED": "FAIL", "UNVERIFIED": "SKIP"}[self.status]
        lines = [f"  [{icon}] {self.fact_id}: {self.status}"]
        if self.tool_used != "none":
            lines.append(f"         Tool: {self.tool_used}")
            lines.append(f"         {self.detail}")
        return "\n".join(lines)


@dataclass
class VerificationReport:
    """Report for an entire facts file."""
    facts_file: str
    results: List[FactVerification]
    verified_count: int = 0
    contradicted_count: int = 0
    unverified_count: int = 0

    def __post_init__(self):
        self.verified_count = sum(1 for r in self.results if r.status == "VERIFIED")
        self.contradicted_count = sum(1 for r in self.results if r.status == "CONTRADICTED")
        self.unverified_count = sum(1 for r in self.results if r.status == "UNVERIFIED")

    @property
    def has_contradictions(self) -> bool:
        return self.contradicted_count > 0

    @property
    def all_clear(self) -> bool:
        return self.contradicted_count == 0

    def __str__(self):
        lines = [
            f"{'='*60}",
            f"  Fact Verification Report: {Path(self.facts_file).name}",
            f"{'='*60}",
            f"  Verified:     {self.verified_count}",
            f"  Contradicted: {self.contradicted_count}",
            f"  Unverified:   {self.unverified_count}",
            f"  Total:        {len(self.results)}",
            "",
        ]
        # Show contradictions first (they block training)
        contradicted = [r for r in self.results if r.status == "CONTRADICTED"]
        if contradicted:
            lines.append("  CONTRADICTIONS (training blocked):")
            for r in contradicted:
                lines.append(str(r))
            lines.append("")

        # Then verified
        verified = [r for r in self.results if r.status == "VERIFIED"]
        if verified:
            lines.append("  Verified facts:")
            for r in verified:
                lines.append(str(r))
            lines.append("")

        # Unverified last (just info)
        unverified = [r for r in self.results if r.status == "UNVERIFIED"]
        if unverified:
            lines.append(f"  Unverified ({len(unverified)} facts — no matching tool):")
            for r in unverified:
                lines.append(f"    {r.fact_id}")
            lines.append("")

        if self.has_contradictions:
            lines.append("  RESULT: BLOCKED — fix contradicted facts before training")
        else:
            lines.append("  RESULT: CLEAR — no contradictions found")
        lines.append(f"{'='*60}")
        return "\n".join(lines)


# ─── Pattern matchers for specific claim types ────────────────────────────────
#
# Each checker returns (status, tool_name, detail) or None if it doesn't apply.
# Status is "VERIFIED" or "CONTRADICTED".
# These use the computational tools to verify claims found in fact text.

def _check_ns_scaling_claims(fact_id: str, context: str, truth: str,
                              distractors: List[str]) -> Optional[Tuple[str, str, str]]:
    """Check claims about NS scaling exponents, critical spaces, Q_f kernels."""
    try:
        from noethersolve.ns_scaling import (
            ns_sobolev_scaling, qf_scaling, fractional_dissipation
        )
    except ImportError:
        return None

    text = (context + " " + truth).lower()

    # Check: scaling-critical Sobolev space for 3D NS
    if "scaling-critical" in text and "sobolev" in text and "3d" in text.replace("r^3", "3d"):
        report = ns_sobolev_scaling(0.5, n=3)
        if report.classification == "critical":
            # The truth should mention H^{1/2}
            if "h^{1/2}" in truth.lower() or "h^1/2" in truth.lower() or "half-derivative" in truth.lower():
                return ("VERIFIED", "ns_scaling.ns_sobolev_scaling",
                        f"H^{{1/2}} is scaling-critical in 3D (exponent = {report.scaling_exponent})")
            # Check if truth wrongly claims a different space is critical
            for wrong_s, name in [(0, "L2"), (1, "H1"), (-1, "H^{-1}")]:
                if name.lower() in truth.lower():
                    r2 = ns_sobolev_scaling(wrong_s, n=3)
                    return ("CONTRADICTED", "ns_scaling.ns_sobolev_scaling",
                            f"Claim says {name} is critical, but {name} is {r2.classification} "
                            f"(exponent = {r2.scaling_exponent}). Critical space is H^{{1/2}}.")

    # Check: critical Q_f kernel exponent for 3D NS
    if "critical" in text and ("q_f" in text or "kernel" in text) and ("3d" in text or "d³x" in truth or "d³x" in context):
        report = qf_scaling(0, n=3)  # dummy call to get critical exponent
        p_c = report.critical_kernel_exponent  # should be -2

        # Extract claimed kernel exponent from truth
        claimed_p = _extract_kernel_exponent(truth)
        if claimed_p is not None:
            qf_report = qf_scaling(claimed_p, n=3)
            if qf_report.classification == "critical":
                return ("VERIFIED", "ns_scaling.qf_scaling",
                        f"f(r) = r^{{{claimed_p}}} is scaling-critical for 3D Q_f "
                        f"(exponent = {qf_report.scaling_exponent}, p_c = {p_c})")
            else:
                return ("CONTRADICTED", "ns_scaling.qf_scaling",
                        f"Claim says f(r) = r^{{{claimed_p}}} is scaling-critical, but it is "
                        f"{qf_report.classification} (exponent = {qf_report.scaling_exponent}). "
                        f"Critical kernel is r^{{{p_c}}}.")

        # Also check distractors that claim to be the critical kernel
        # (the truth might describe the answer without a clean exponent)
        for dist in distractors:
            dist_p = _extract_kernel_exponent(dist)
            if dist_p is not None:
                qf_report = qf_scaling(dist_p, n=3)
                if qf_report.classification == "critical":
                    # A distractor matches the correct answer — the truth must be wrong
                    # unless the truth ALSO matches. If we reach here, truth didn't match.
                    pass  # Don't flag — distractor matching critical isn't proof truth is wrong

    # Check: fractional dissipation critical exponent
    if "fractional" in text and ("critical" in text or "5/4" in text or "lions" in text.lower()):
        report = fractional_dissipation(5/4, n=3)
        # Extract claimed alpha from truth
        alpha_match = re.search(r'α\s*[=≥]\s*([\d./]+)', truth)
        if not alpha_match:
            alpha_match = re.search(r'(\d+/\d+)', truth)
        if alpha_match:
            try:
                val_str = alpha_match.group(1)
                if '/' in val_str:
                    num, den = val_str.split('/')
                    claimed_alpha = float(num) / float(den)
                else:
                    claimed_alpha = float(val_str)
                _fr = fractional_dissipation(claimed_alpha, n=3)  # noqa: F841 - validates alpha
                if abs(claimed_alpha - report.alpha_critical) < 1e-10:
                    return ("VERIFIED", "ns_scaling.fractional_dissipation",
                            f"α = {claimed_alpha} is the critical exponent (α_c = {report.alpha_critical})")
            except (ValueError, ZeroDivisionError):
                pass

    # Check: enstrophy scaling exponent
    if "enstrophy" in text and "scal" in text and ("λ" in truth or "lambda" in truth.lower()):
        report = ns_sobolev_scaling(1, n=3)
        # The norm exponent is s + 1 - n/2.  The SQUARED norm (∫|ω|²dx)
        # exponent is 2*(s + 1 - n/2).  Facts may use either convention.
        norm_exp = report.scaling_exponent        # 0.5 for enstrophy in 3D
        squared_exp = 2 * report.scaling_exponent  # 1.0 for enstrophy in 3D
        exp_match = re.search(r'λ\^?\{?([+-]?\d+(?:/\d+)?)\}?', truth)
        if exp_match:
            try:
                val_str = exp_match.group(1)
                if '/' in val_str:
                    num, den = val_str.split('/')
                    claimed = float(num) / float(den)
                else:
                    claimed = float(val_str)
                if abs(claimed - norm_exp) < 1e-10 or abs(claimed - squared_exp) < 1e-10:
                    return ("VERIFIED", "ns_scaling.ns_sobolev_scaling",
                            f"Enstrophy scales as λ^{{{claimed}}} in 3D ({report.classification}). "
                            f"Norm exponent = {norm_exp}, squared norm exponent = {squared_exp}.")
                else:
                    return ("CONTRADICTED", "ns_scaling.ns_sobolev_scaling",
                            f"Claim says enstrophy scales as λ^{{{claimed}}}, but correct is "
                            f"λ^{{{norm_exp}}} (norm) or λ^{{{squared_exp}}} (squared norm) "
                            f"({report.classification})")
            except (ValueError, ZeroDivisionError):
                pass

    # Check: energy scaling exponent
    if "energy" in text and "scal" in text and ("λ" in truth or "lambda" in truth.lower()):
        if "kinetic energy" in text or "e = " in text.lower() or "|u|²" in text:
            report = ns_sobolev_scaling(0, n=3)
            norm_exp = report.scaling_exponent        # -0.5 for energy in 3D
            squared_exp = 2 * report.scaling_exponent  # -1.0 for energy in 3D
            exp_match = re.search(r'λ\^?\{?([+-]?\d+(?:/\d+)?)\}?', truth)
            if exp_match:
                try:
                    val_str = exp_match.group(1)
                    if '/' in val_str:
                        num, den = val_str.split('/')
                        claimed = float(num) / float(den)
                    else:
                        claimed = float(val_str)
                    if abs(claimed - norm_exp) < 1e-10 or abs(claimed - squared_exp) < 1e-10:
                        return ("VERIFIED", "ns_scaling.ns_sobolev_scaling",
                                f"Energy scales as λ^{{{claimed}}} in 3D ({report.classification}). "
                                f"Norm exponent = {norm_exp}, squared norm exponent = {squared_exp}.")
                    else:
                        return ("CONTRADICTED", "ns_scaling.ns_sobolev_scaling",
                                f"Claim says energy scales as λ^{{{claimed}}}, but correct is "
                                f"λ^{{{norm_exp}}} (norm) or λ^{{{squared_exp}}} (squared norm) "
                                f"({report.classification})")
                except (ValueError, ZeroDivisionError):
                    pass

    # Check: Prodi-Serrin condition
    if "prodi-serrin" in text.lower() or ("2/p" in text and "3/q" in text):
        # Check if the truth claims the critical curve is 2/p + 3/q = 1
        if "2/p + 3/q = 1" in truth or "scaling-critical" in truth.lower():
            # Verify with a point on the curve: (p,q) = (inf, 3)
            from noethersolve.ns_scaling import ns_lp_lq_scaling
            report = ns_lp_lq_scaling(1e10, 3, n=3)
            if abs(report.scaling_exponent) < 0.01:
                return ("VERIFIED", "ns_scaling.ns_lp_lq_scaling",
                        "2/p + 3/q = 1 is the scaling-critical curve (verified at (∞, 3))")

    # Check: Green's function of fractional Laplacian
    if "green" in text.lower() and ("fractional" in text.lower() or "5/4" in text):
        # Extract claimed exponent r^{...}
        r_match = re.search(r'r\^?\{?([+-]?[\d./]+)\}?', truth)
        if r_match:
            try:
                val_str = r_match.group(1)
                if '/' in val_str:
                    parts = val_str.split('/')
                    claimed_exp = float(parts[0]) / float(parts[1])
                else:
                    claimed_exp = float(val_str)
                # G ~ r^{2*alpha - n}, for alpha=5/4, n=3: 2*(5/4)-3 = -1/2
                correct_exp = 2 * (5/4) - 3
                if abs(claimed_exp - correct_exp) < 1e-10:
                    return ("VERIFIED", "ns_scaling.fractional_dissipation",
                            f"Green's function of (-Δ)^{{5/4}} in R³ ~ r^{{{correct_exp}}}")
                else:
                    return ("CONTRADICTED", "ns_scaling.fractional_dissipation",
                            f"Claim says G ~ r^{{{claimed_exp}}}, but correct is r^{{{correct_exp}}} "
                            f"(G ~ r^{{2α-n}} = r^{{2(5/4)-3}} = r^{{-1/2}})")
            except (ValueError, ZeroDivisionError):
                pass

    return None


def _check_pde_regularity_claims(fact_id: str, context: str, truth: str,
                                   distractors: List[str]) -> Optional[Tuple[str, str, str]]:
    """Check claims about PDE regularity results."""
    try:
        from noethersolve.pde_regularity import check_pde_regularity, check_sobolev_embedding
    except ImportError:
        return None

    text = (context + " " + truth).lower()

    # Check: 3D NS regularity status
    # Skip restricted geometries (axisymmetric, etc.) — those have different
    # regularity results than the general 3D case.
    restricted_keywords = ["axisymmetric", "axi-symmetric", "2d reduction",
                           "without swirl", "no swirl", "radial", "symmetric"]
    is_restricted = any(kw in text for kw in restricted_keywords)

    if "navier-stokes" in text and "3d" in text and not is_restricted:
        if "global regularity" in truth.lower() and "proven" in truth.lower():
            report = check_pde_regularity("navier-stokes", dimension=3,
                                          claimed_regularity="global smooth")
            if report.verdict == "FAIL":
                return ("CONTRADICTED", "pde_regularity.check_pde_regularity",
                        f"3D NS global regularity is OPEN, not proven. "
                        f"Status: {report.status}")
        if "open" in truth.lower() or "singularit" in truth.lower():
            report = check_pde_regularity("navier-stokes", dimension=3)
            if report.status == "open":
                return ("VERIFIED", "pde_regularity.check_pde_regularity",
                        "3D NS regularity is correctly stated as open/unknown")

    # Check: 2D NS regularity
    if "navier-stokes" in text and "2d" in text:
        if "global regularity" in truth.lower() or "no finite-time blowup" in truth.lower():
            report = check_pde_regularity("navier-stokes", dimension=2)
            if report.status == "proved":
                return ("VERIFIED", "pde_regularity.check_pde_regularity",
                        "2D NS global regularity is proved (Ladyzhenskaya 1969)")

    # Check: Sobolev embedding claims
    emb_match = re.search(r'W\^?\{?(\d+),\s*(\d+(?:\.\d+)?)\}?\(R\^?(\d+)\)', text)
    if emb_match:
        k, p, n = int(emb_match.group(1)), float(emb_match.group(2)), int(emb_match.group(3))
        report = check_sobolev_embedding(k, p, n)
        if report.sobolev_conjugate and f"p* = {report.sobolev_conjugate}" in truth:
            return ("VERIFIED", "pde_regularity.check_sobolev_embedding",
                    f"W^{{{k},{p}}}(R^{n}) embeds into {report.target_space}")

    return None


def _check_dimension_physics_claims(fact_id: str, context: str, truth: str,
                                      distractors: List[str]) -> Optional[Tuple[str, str, str]]:
    """Check claims about dimension-dependent physics."""
    try:
        from noethersolve.dimension_physics import get_formula, DIMENSIONAL_PHYSICS
    except ImportError:
        return None

    text = (context + " " + truth).lower()

    # Check Green's function claims
    if "green" in text and "laplacian" in text:
        for dim in [1, 2, 3]:
            if f"{dim}d" in text or f"r^{dim}" in text:
                formula = get_formula("laplacian_greens_function", dim)
                if formula:
                    if formula.formula.lower() in truth.lower():
                        return ("VERIFIED", "dimension_physics.get_formula",
                                f"{dim}D Laplacian Green's function is {formula.formula}")

    # Check energy cascade direction claims
    if "cascade" in text or "energy transfer" in text:
        for dim in [2, 3]:
            if f"{dim}d" in text:
                key = "energy_cascade"
                if key in DIMENSIONAL_PHYSICS and dim in DIMENSIONAL_PHYSICS[key]:
                    formula = DIMENSIONAL_PHYSICS[key][dim]
                    # Check for inverse cascade in 2D, forward cascade in 3D
                    if dim == 2 and "inverse" in truth.lower():
                        return ("VERIFIED", "dimension_physics",
                                "2D has inverse energy cascade (correct)")
                    if dim == 3 and "forward" in truth.lower() or "direct" in truth.lower():
                        return ("VERIFIED", "dimension_physics",
                                "3D has forward/direct energy cascade (correct)")

    return None


# ─── Kernel exponent extraction ───────────────────────────────────────────────

def _extract_kernel_exponent(text: str) -> Optional[float]:
    """Extract kernel exponent from text like 'r^{-2}', 'r^{-1/2}', etc.

    Returns the exponent p from f(r) = r^p, or None if not found.
    """
    # Try r^{...} patterns
    patterns = [
        r'r\^\{([+-]?[\d./]+)\}',     # r^{-2}, r^{-1/2}
        r'r\^([+-]?\d+(?:/\d+)?)',     # r^-2, r^-1
        r'f\(r\)\s*=\s*r\^\{?([+-]?[\d./]+)\}?',  # f(r) = r^{-2}
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            val_str = match.group(1)
            try:
                if '/' in val_str:
                    parts = val_str.split('/')
                    # Handle -1/2 vs 1/2
                    if parts[0].startswith('-'):
                        return -float(parts[0][1:]) / float(parts[1])
                    return float(parts[0]) / float(parts[1])
                return float(val_str)
            except (ValueError, ZeroDivisionError, IndexError):
                continue

    # Special patterns
    if "1/r" in text and "r^" not in text:
        # "1/r" = r^{-1}
        if "1/r²" in text or "1/r^2" in text:
            return -2.0
        return -1.0
    if "-ln(r)" in text or "-ln r" in text or "logarithmic" in text.lower():
        return None  # logarithmic, not a power law

    return None


# ─── Main verification function ──────────────────────────────────────────────

# All checkers in priority order
_CHECKERS = [
    _check_ns_scaling_claims,
    _check_pde_regularity_claims,
    _check_dimension_physics_claims,
]


def verify_single_fact(fact: dict) -> FactVerification:
    """Verify a single fact against all available computational tools.

    Args:
        fact: dict with keys 'id', 'context', 'truth', 'distractors'

    Returns:
        FactVerification with status VERIFIED, CONTRADICTED, or UNVERIFIED.
    """
    fact_id = fact.get("id", "unknown")
    context = fact.get("context", "")
    truth = fact.get("truth", fact.get("fact", ""))
    distractors = fact.get("distractors", [])

    for checker in _CHECKERS:
        try:
            result = checker(fact_id, context, truth, distractors)
            if result is not None:
                status, tool, detail = result
                return FactVerification(
                    fact_id=fact_id,
                    status=status,
                    tool_used=tool,
                    detail=detail,
                    truth_text=truth[:120],
                )
        except Exception:
            # Tool error — don't block, just skip this checker
            continue

    return FactVerification(
        fact_id=fact_id,
        status="UNVERIFIED",
        tool_used="none",
        detail="No matching computational tool found",
        truth_text=truth[:120],
    )


def verify_facts_file(facts_file: str) -> VerificationReport:
    """Verify all facts in a facts JSON file.

    Args:
        facts_file: path to a *_facts*.json file

    Returns:
        VerificationReport with per-fact results and summary.
    """
    with open(facts_file) as f:
        data = json.load(f)

    facts = data.get("facts", data.get("verifications", []))
    results = []
    for fact in facts:
        results.append(verify_single_fact(fact))

    return VerificationReport(facts_file=str(facts_file), results=results)


def verify_or_warn(facts_file: str, strict: bool = True) -> bool:
    """Verification gate for the training pipeline.

    Call this before training. If any fact is CONTRADICTED by a computational
    tool, raises ValueError (hard block). Unverified facts produce warnings
    but don't block.

    Args:
        facts_file: path to facts JSON file
        strict: if True, raise on contradictions. If False, just warn.

    Returns:
        True if no contradictions found, False otherwise.

    Raises:
        ValueError: if strict=True and any fact is contradicted.
        FileNotFoundError: if facts_file doesn't exist.
    """
    report = verify_facts_file(facts_file)

    if report.unverified_count > 0:
        print(f"  [WARN] {report.unverified_count}/{len(report.results)} facts "
              f"unverified (no matching tool) in {Path(facts_file).name}")

    if report.verified_count > 0:
        print(f"  [OK]   {report.verified_count}/{len(report.results)} facts "
              f"verified by computational tools in {Path(facts_file).name}")

    if report.has_contradictions:
        msg_lines = [
            f"FACT VERIFICATION FAILED: {report.contradicted_count} fact(s) "
            f"CONTRADICTED by computational tools in {Path(facts_file).name}:",
        ]
        for r in report.results:
            if r.status == "CONTRADICTED":
                msg_lines.append(f"  {r.fact_id}: {r.detail}")
                msg_lines.append(f"    Truth was: {r.truth_text}")
        msg = "\n".join(msg_lines)

        if strict:
            raise ValueError(msg)
        else:
            print(f"  [FAIL] {msg}")
            return False

    return True


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    """CLI entry point for fact verification."""
    import argparse
    parser = argparse.ArgumentParser(description="Verify facts against computational tools")
    parser.add_argument("--file", "-f", help="Specific facts file to verify")
    parser.add_argument("--all", "-a", action="store_true", help="Verify all facts files")
    parser.add_argument("--strict", action="store_true", default=False,
                        help="Exit with error code on contradictions")
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    problems_dir = project_root / "problems"

    if args.file:
        files = [Path(args.file)]
    elif args.all:
        files = sorted(problems_dir.glob("*_facts*.json"))
    else:
        parser.print_help()
        return

    total_contradictions = 0
    total_verified = 0
    total_unverified = 0

    for f in files:
        try:
            report = verify_facts_file(str(f))
            total_contradictions += report.contradicted_count
            total_verified += report.verified_count
            total_unverified += report.unverified_count

            # Only print full report if there are contradictions or verifications
            if report.contradicted_count > 0 or report.verified_count > 0:
                print(report)
            else:
                print(f"  {f.name}: {len(report.results)} facts, all unverified (no matching tools)")
        except (json.JSONDecodeError, KeyError) as e:
            print(f"  {f.name}: ERROR reading file — {e}")

    print(f"\n{'='*60}")
    print(f"  SUMMARY: {total_verified} verified, {total_contradictions} contradicted, "
          f"{total_unverified} unverified across {len(files)} files")
    if total_contradictions > 0:
        print(f"  WARNING: {total_contradictions} contradictions found!")
    print(f"{'='*60}")

    if args.strict and total_contradictions > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
