"""
noethersolve.pipeline — Therapeutic pipeline consistency validator.

Checks that design choices across a gene therapy pipeline are internally
consistent. Each check enforces a cross-domain consistency rule derived from
genetics therapeutics knowledge: vector capacity, serotype-tissue pairing,
promoter-tissue pairing, route-tissue pairing, modality-payload compatibility,
redosing immunogenicity, and safety monitoring requirements.

Catches:
  - AAV transgene exceeding packaging capacity (>4.7 kb = HIGH)
  - Serotype-tissue mismatches (AAV8 for CNS = HIGH)
  - Promoter-tissue mismatches (liver promoter for muscle = HIGH)
  - Route-tissue mismatches (inhaled for CNS = HIGH)
  - Modality-payload incompatibility (lnp_sirna for gene_replacement = HIGH)
  - Redosing with AAV (anti-capsid NAbs block second dose = HIGH)
  - Missing safety monitoring (informational, not a failure)

Usage:
    from noethersolve.pipeline import validate_pipeline, TherapyDesign

    design = TherapyDesign(
        modality="aav",
        target_tissue="liver",
        transgene_size_kb=4.5,
        vector_serotype="AAV8",
        promoter="TBG",
        route="iv",
        payload_type="gene_replacement",
        redosing_planned=False,
    )
    report = validate_pipeline(design)
    print(report)
    # Shows per-rule diagnostics, required monitoring, and overall verdict

    # Or from a dict:
    from noethersolve.pipeline import validate_pipeline_dict
    report = validate_pipeline_dict({"modality": "aav", "target_tissue": "cns", ...})
"""

from dataclasses import dataclass
from typing import Dict, List, Set, Tuple


# ─── Dataclasses ──────────────────────────────────────────────────────────────

@dataclass
class TherapyDesign:
    """Specification for a gene therapy product."""
    modality: str           # "aav", "lnp_mrna", "lnp_sirna", "aso", "base_edit", "prime_edit"
    target_tissue: str      # "liver", "cns", "muscle", "eye", "lung", "heart", "blood", "tumor"
    transgene_size_kb: float = 0.0    # for AAV: must be <= 4.7
    vector_serotype: str = ""          # for AAV: "AAV8", "AAV9", "AAVrh10", etc.
    promoter: str = ""                 # "CMV", "CAG", "TBG", "hAAT", "ApoE_hAAT", "MCK", "desmin", "CBA"
    route: str = ""                    # "iv", "intrathecal", "subretinal", "intramuscular", "inhaled", "subcutaneous"
    payload_type: str = ""             # "gene_replacement", "gene_silencing", "gene_editing", "gene_addition"
    redosing_planned: bool = False     # relevant for immune considerations


@dataclass
class PipelineIssue:
    """A single consistency issue found in the therapy design."""
    rule_name: str           # VECTOR_CAPACITY, SEROTYPE_TISSUE, etc.
    severity: str            # HIGH, MODERATE, LOW, INFO
    message: str
    suggestion: str

    def __str__(self):
        return f"  [{self.severity}] {self.rule_name}: {self.message}"


@dataclass
class PipelineReport:
    """Result of validate_pipeline()."""
    design: TherapyDesign
    issues: List[PipelineIssue]
    required_monitoring: List[str]
    verdict: str                      # PASS, WARN, or FAIL

    def __str__(self):
        lines = []
        lines.append(f"{'=' * 60}")
        lines.append(f"  Pipeline Consistency Audit: {self.verdict}")
        lines.append(f"{'=' * 60}")
        lines.append(f"  Modality: {self.design.modality}, "
                     f"Target: {self.design.target_tissue}, "
                     f"Route: {self.design.route or 'unspecified'}")
        lines.append(f"")

        # Issues sorted by severity
        if self.issues:
            lines.append(f"  Issues ({len(self.issues)}):")
            severity_order = {"HIGH": 0, "MODERATE": 1, "LOW": 2, "INFO": 3}
            for issue in sorted(self.issues, key=lambda i: severity_order.get(i.severity, 4)):
                lines.append(str(issue))
                if issue.suggestion:
                    lines.append(f"    -> {issue.suggestion}")
            lines.append(f"")

        # Required monitoring
        if self.required_monitoring:
            lines.append(f"  Required monitoring:")
            for m in self.required_monitoring:
                lines.append(f"    - {m}")
            lines.append(f"")

        # Summary counts
        n_high = sum(1 for i in self.issues if i.severity == "HIGH")
        n_moderate = sum(1 for i in self.issues if i.severity == "MODERATE")
        n_info = sum(1 for i in self.issues if i.severity == "INFO")
        lines.append(f"  Summary: {n_high} HIGH, {n_moderate} MODERATE, "
                     f"{n_info} INFO")
        lines.append(f"{'=' * 60}")
        return "\n".join(lines)

    @property
    def passed(self) -> bool:
        return self.verdict == "PASS"


# ─── Knowledge tables ────────────────────────────────────────────────────────

# Serotype -> set of tissues it targets well
_SEROTYPE_TISSUE: Dict[str, Set[str]] = {
    "AAV8":    {"liver", "muscle"},
    "AAV9":    {"cns", "heart", "muscle"},
    "AAVrh10": {"cns"},
    "AAV2":    {"eye", "liver"},
    "AAV5":    {"lung", "eye"},
    "AAV1":    {"muscle"},
}

# Promoter -> set of tissues it is appropriate for, plus flags
_PROMOTER_TISSUE: Dict[str, Set[str]] = {
    "TBG":       {"liver"},
    "hAAT":      {"liver"},
    "ApoE_hAAT": {"liver"},
    "MCK":       {"muscle"},
    "desmin":    {"muscle"},
}
_UBIQUITOUS_PROMOTERS: Set[str] = {"CBA", "CAG", "CMV"}

# Route -> set of tissues it reaches
_ROUTE_TISSUE: Dict[str, Set[str]] = {
    "iv":             {"liver", "muscle"},
    "intrathecal":    {"cns"},
    "subretinal":     {"eye"},
    "intramuscular":  {"muscle"},
    "inhaled":        {"lung"},
    "subcutaneous":   {"liver"},
}

# Modality -> set of compatible payload types
_MODALITY_PAYLOAD: Dict[str, Set[str]] = {
    "aav":        {"gene_replacement", "gene_addition", "gene_editing"},
    "lnp_mrna":   {"gene_replacement", "gene_editing"},
    "lnp_sirna":  {"gene_silencing"},
    "aso":        {"gene_silencing", "splice_modulation"},
    "base_edit":  {"gene_editing"},
    "prime_edit": {"gene_editing"},
}


# ─── Individual check functions ───────────────────────────────────────────────

def _check_vector_capacity(design: TherapyDesign) -> List[PipelineIssue]:
    """Check AAV transgene packaging capacity."""
    issues = []
    if design.modality != "aav":
        return issues
    if design.transgene_size_kb <= 0:
        return issues

    if design.transgene_size_kb > 4.7:
        issues.append(PipelineIssue(
            rule_name="VECTOR_CAPACITY",
            severity="HIGH",
            message=(f"Transgene {design.transgene_size_kb:.1f} kb exceeds AAV "
                     f"packaging limit (4.7 kb)"),
            suggestion=("Consider dual-AAV strategy, truncated transgene, or "
                        "switch to LNP-mRNA for larger payloads"),
        ))
    elif design.transgene_size_kb > 4.2:
        issues.append(PipelineIssue(
            rule_name="VECTOR_CAPACITY",
            severity="MODERATE",
            message=(f"Transgene {design.transgene_size_kb:.1f} kb is near AAV "
                     f"limit (4.7 kb) — packaging efficiency drops above 4.2 kb"),
            suggestion=("Monitor vector titer carefully; consider codon "
                        "optimization to reduce insert size"),
        ))
    return issues


def _check_serotype_tissue(design: TherapyDesign) -> List[PipelineIssue]:
    """Check AAV serotype-tissue pairing."""
    issues = []
    if design.modality != "aav":
        return issues
    if not design.vector_serotype:
        return issues

    serotype = design.vector_serotype.upper()
    tissue = design.target_tissue.lower()
    optimal_tissues = _SEROTYPE_TISSUE.get(serotype)

    if optimal_tissues is None:
        # Unknown serotype — informational only
        issues.append(PipelineIssue(
            rule_name="SEROTYPE_TISSUE",
            severity="LOW",
            message=f"Serotype {design.vector_serotype} not in known pairing table",
            suggestion="Verify tropism data for this serotype",
        ))
        return issues

    if tissue not in optimal_tissues:
        # Check if it's a complete mismatch vs suboptimal
        # Complete mismatch: no known overlap at all
        all_known_tissues = set()
        for t_set in _SEROTYPE_TISSUE.values():
            all_known_tissues.update(t_set)

        issues.append(PipelineIssue(
            rule_name="SEROTYPE_TISSUE",
            severity="HIGH",
            message=(f"{design.vector_serotype} is not optimal for "
                     f"{design.target_tissue} (optimal: "
                     f"{', '.join(sorted(optimal_tissues))})"),
            suggestion=(f"Consider serotypes known to target {design.target_tissue}: "
                        f"{', '.join(s for s, t in _SEROTYPE_TISSUE.items() if tissue in t) or 'none in table'}"),
        ))
    return issues


def _check_promoter_tissue(design: TherapyDesign) -> List[PipelineIssue]:
    """Check promoter-tissue pairing."""
    issues = []
    if not design.promoter:
        return issues

    promoter = design.promoter
    tissue = design.target_tissue.lower()

    # Ubiquitous promoters
    if promoter in _UBIQUITOUS_PROMOTERS:
        if promoter == "CMV":
            issues.append(PipelineIssue(
                rule_name="PROMOTER_TISSUE",
                severity="MODERATE",
                message="CMV promoter is ubiquitous but silenced in vivo",
                suggestion="Consider tissue-specific promoter for durable expression",
            ))
        else:
            issues.append(PipelineIssue(
                rule_name="PROMOTER_TISSUE",
                severity="MODERATE",
                message=(f"{promoter} is ubiquitous — OK for {tissue} but may "
                         f"cause off-target expression"),
                suggestion=("Consider tissue-specific promoter to reduce off-target "
                            "expression risk"),
            ))
        return issues

    # Tissue-specific promoters
    target_tissues = _PROMOTER_TISSUE.get(promoter)
    if target_tissues is None:
        issues.append(PipelineIssue(
            rule_name="PROMOTER_TISSUE",
            severity="LOW",
            message=f"Promoter {promoter} not in known pairing table",
            suggestion="Verify tissue specificity data for this promoter",
        ))
        return issues

    if tissue not in target_tissues:
        issues.append(PipelineIssue(
            rule_name="PROMOTER_TISSUE",
            severity="HIGH",
            message=(f"{promoter} is a {'/'.join(sorted(target_tissues))}-specific "
                     f"promoter, used for {tissue} target"),
            suggestion=(f"Use a promoter specific to {tissue} or a ubiquitous "
                        f"promoter (CAG, CBA)"),
        ))
    return issues


def _check_route_tissue(design: TherapyDesign) -> List[PipelineIssue]:
    """Check administration route-tissue consistency."""
    issues = []
    if not design.route:
        return issues

    route = design.route.lower()
    tissue = design.target_tissue.lower()
    reachable = _ROUTE_TISSUE.get(route)

    if reachable is None:
        issues.append(PipelineIssue(
            rule_name="ROUTE_TISSUE",
            severity="LOW",
            message=f"Route {design.route} not in known pairing table",
            suggestion="Verify biodistribution for this route",
        ))
        return issues

    if tissue not in reachable:
        issues.append(PipelineIssue(
            rule_name="ROUTE_TISSUE",
            severity="HIGH",
            message=(f"{design.route} administration does not reach "
                     f"{design.target_tissue} (reaches: "
                     f"{', '.join(sorted(reachable))})"),
            suggestion=(f"Consider routes that target {design.target_tissue}: "
                        f"{', '.join(r for r, t in _ROUTE_TISSUE.items() if tissue in t) or 'none in table'}"),
        ))
    return issues


def _check_modality_payload(design: TherapyDesign) -> List[PipelineIssue]:
    """Check modality-payload type compatibility."""
    issues = []
    if not design.payload_type:
        return issues

    modality = design.modality.lower()
    payload = design.payload_type.lower()
    compatible = _MODALITY_PAYLOAD.get(modality)

    if compatible is None:
        issues.append(PipelineIssue(
            rule_name="MODALITY_PAYLOAD",
            severity="LOW",
            message=f"Modality {design.modality} not in compatibility table",
            suggestion="Verify payload compatibility for this modality",
        ))
        return issues

    if payload not in compatible:
        issues.append(PipelineIssue(
            rule_name="MODALITY_PAYLOAD",
            severity="HIGH",
            message=(f"{design.modality} is not compatible with "
                     f"{design.payload_type} (compatible: "
                     f"{', '.join(sorted(compatible))})"),
            suggestion=(f"Consider modalities that support {design.payload_type}: "
                        f"{', '.join(m for m, p in _MODALITY_PAYLOAD.items() if payload in p) or 'none in table'}"),
        ))
    return issues


def _check_redosing_immunity(design: TherapyDesign) -> List[PipelineIssue]:
    """Check redosing feasibility given immune considerations."""
    issues = []
    if not design.redosing_planned:
        return issues

    if design.modality == "aav":
        issues.append(PipelineIssue(
            rule_name="REDOSING_IMMUNITY",
            severity="HIGH",
            message=("AAV generates neutralizing antibodies after first dose — "
                     "redosing is blocked by anti-capsid immunity"),
            suggestion=("Consider immunosuppression protocol, capsid switching, "
                        "or switch to LNP (redosable)"),
        ))
    return issues


def _check_safety_monitoring(design: TherapyDesign) -> Tuple[List[PipelineIssue], List[str]]:
    """Determine required safety monitoring based on modality and payload.

    Returns:
        Tuple of (list of INFO-level PipelineIssues, list of monitoring strings).
    """
    issues = []
    monitoring: List[str] = []
    modality = design.modality.lower()

    # Universal requirements
    monitoring.append("Biodistribution study")
    monitoring.append("Germline exclusion testing")

    # Modality-specific
    if modality == "aav":
        monitoring.append("ALT/AST monitoring (hepatotoxicity)")
        monitoring.append("Complement monitoring (thrombocytopenia risk)")
        monitoring.append("Long-term follow-up 15 years (LTFU)")
    elif modality in ("lnp_mrna", "lnp_sirna"):
        monitoring.append("Cytokine panel (CRS risk)")
        monitoring.append("Complement monitoring")
    elif modality == "aso":
        monitoring.append("Platelet count monitoring (thrombocytopenia)")
        monitoring.append("Renal function monitoring")

    # Payload-specific
    payload = design.payload_type.lower() if design.payload_type else ""
    if payload == "gene_editing" or modality in ("base_edit", "prime_edit"):
        monitoring.append("Off-target editing analysis (GUIDE-seq or CIRCLE-seq)")

    for m in monitoring:
        issues.append(PipelineIssue(
            rule_name="SAFETY_MONITORING",
            severity="INFO",
            message=m,
            suggestion="",
        ))

    return issues, monitoring


# ─── Core validation function ────────────────────────────────────────────────

def validate_pipeline(design: TherapyDesign) -> PipelineReport:
    """Validate a gene therapy design for cross-domain consistency.

    Runs all 7 consistency checks:
      1. Vector capacity (AAV packaging limit)
      2. Serotype-tissue match
      3. Promoter-tissue match
      4. Route-tissue consistency
      5. Modality-payload compatibility
      6. Redosing + AAV immune problem
      7. Safety monitoring requirements (informational)

    Args:
        design: TherapyDesign specification to validate.

    Returns:
        PipelineReport with issues, required monitoring, and verdict.
        Verdict: FAIL if any HIGH, WARN if any MODERATE, PASS if only LOW/INFO.
    """
    all_issues = []

    # Run checks 1-6 (may produce HIGH/MODERATE/LOW)
    all_issues.extend(_check_vector_capacity(design))
    all_issues.extend(_check_serotype_tissue(design))
    all_issues.extend(_check_promoter_tissue(design))
    all_issues.extend(_check_route_tissue(design))
    all_issues.extend(_check_modality_payload(design))
    all_issues.extend(_check_redosing_immunity(design))

    # Check 7: safety monitoring (returns issues + monitoring list)
    safety_issues, required_monitoring = _check_safety_monitoring(design)
    all_issues.extend(safety_issues)

    # ── Overall verdict ──────────────────────────────────────────────────
    has_high = any(i.severity == "HIGH" for i in all_issues)
    has_moderate = any(i.severity == "MODERATE" for i in all_issues)

    if has_high:
        verdict = "FAIL"
    elif has_moderate:
        verdict = "WARN"
    else:
        verdict = "PASS"

    return PipelineReport(
        design=design,
        issues=all_issues,
        required_monitoring=required_monitoring,
        verdict=verdict,
    )


def validate_pipeline_dict(d: dict) -> PipelineReport:
    """Convenience wrapper: construct TherapyDesign from a dict and validate.

    Args:
        d: dict with keys matching TherapyDesign fields. Unknown keys are
           silently ignored. Missing optional fields use defaults.

    Returns:
        PipelineReport from validate_pipeline().
    """
    known_fields = {f.name for f in TherapyDesign.__dataclass_fields__.values()}
    filtered = {k: v for k, v in d.items() if k in known_fields}
    design = TherapyDesign(**filtered)
    return validate_pipeline(design)
