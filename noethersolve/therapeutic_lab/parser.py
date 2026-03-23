"""
Pathophysiology Parser — Extract molecular targets from disease descriptions.

Uses regex patterns and keyword matching to identify genes, mechanisms, and tissues.
Optionally verifies extractions with the 4B oracle.
"""

import re
from typing import List, Optional, Tuple

from .types import (
    MolecularTarget,
    PathophysiologyExtraction,
    OracleVerification,
    TargetType,
    Mechanism,
    ValidationLevel,
)
from .knowledge import (
    KNOWN_TARGETS,
    GENE_ALIASES,
    MECHANISM_KEYWORDS,
    TISSUE_KEYWORDS,
    DISEASE_TARGETS,
    resolve_gene_alias,
    get_known_target_info,
    get_disease_targets,
)


class PathophysiologyParser:
    """Parse disease pathophysiology to extract molecular targets."""

    # Gene name pattern: 2-6 uppercase letters, optionally followed by numbers
    GENE_PATTERN = re.compile(r'\b([A-Z][A-Z0-9]{1,7})\b')

    # Common false positives to filter out
    FALSE_POSITIVES = {
        "DNA", "RNA", "ATP", "ADP", "GTP", "GDP", "NAD", "NADH", "NADP",
        "THE", "AND", "FOR", "WITH", "FROM", "THIS", "THAT", "THESE",
        "ARE", "HAS", "WAS", "BEEN", "HAVE", "WHICH", "WHEN", "WHERE",
        "FDA", "NIH", "WHO", "CDC", "USA", "MRI", "CT", "PET",
        "ICU", "ER", "OR", "IV", "IM", "SC", "PO",
        "QD", "BID", "TID", "QID", "PRN",
    }

    def __init__(self, oracle=None, verify_claims: bool = True):
        """
        Initialize parser.

        Args:
            oracle: Optional oracle verifier for claim verification
            verify_claims: Whether to verify extracted claims with oracle
        """
        self.oracle = oracle
        self.verify_claims = verify_claims and oracle is not None

    def parse(self, description: str, disease_name: str = "") -> PathophysiologyExtraction:
        """
        Parse pathophysiology description to extract molecular targets.

        Args:
            description: Natural language description of disease mechanism
            disease_name: Optional disease name for context

        Returns:
            PathophysiologyExtraction with targets, mechanisms, and tissues
        """
        # Extract components
        targets = self._extract_targets(description, disease_name)
        mechanisms = self._extract_mechanisms(description)
        tissues = self._extract_tissues(description)

        # Assign mechanisms to targets
        self._assign_mechanisms(targets, description)

        # Verify with oracle if available
        verifications = []
        if self.verify_claims:
            verifications = self._verify_extractions(targets, description, disease_name)

        return PathophysiologyExtraction(
            disease_name=disease_name,
            description=description,
            molecular_targets=targets,
            mechanisms=mechanisms,
            affected_tissues=tissues,
            oracle_verifications=verifications,
        )

    def _extract_targets(self, description: str, disease_name: str) -> List[MolecularTarget]:
        """Extract molecular targets from text."""
        targets = []
        seen = set()

        # First, check for known disease-target associations
        disease_targets = get_disease_targets(disease_name) if disease_name else []
        for gene in disease_targets:
            if gene not in seen:
                target = self._create_target(gene, description)
                target.confidence = 0.9  # High confidence for known associations
                targets.append(target)
                seen.add(gene)

        # Then look for gene names in text
        for match in self.GENE_PATTERN.finditer(description):
            gene = match.group(1)

            # Filter false positives
            if gene in self.FALSE_POSITIVES:
                continue

            # Resolve aliases
            canonical = resolve_gene_alias(gene)

            if canonical not in seen:
                # Only add if it's a known target or appears multiple times
                known_info = get_known_target_info(canonical)
                if known_info or description.lower().count(gene.lower()) >= 2:
                    target = self._create_target(canonical, description)
                    targets.append(target)
                    seen.add(canonical)

        # Also check for full protein names mentioned
        targets.extend(self._extract_protein_names(description, seen))

        return targets

    def _create_target(self, gene: str, description: str) -> MolecularTarget:
        """Create a MolecularTarget from a gene name."""
        known_info = get_known_target_info(gene)

        target_type = TargetType.UNKNOWN
        if known_info:
            type_str = known_info.get("type", "unknown")
            try:
                target_type = TargetType(type_str)
            except ValueError:
                target_type = TargetType.UNKNOWN

        validation = ValidationLevel.THEORETICAL
        known_drugs = []
        if known_info:
            known_drugs = known_info.get("drugs", [])
            if known_drugs:
                validation = ValidationLevel.VALIDATED
            elif known_info.get("druggable", False):
                validation = ValidationLevel.CLINICAL

        return MolecularTarget(
            name=gene,
            target_type=target_type,
            gene_symbol=gene,
            validation_level=validation,
            known_drugs=known_drugs,
            confidence=0.7,
        )

    def _extract_protein_names(self, description: str, seen: set) -> List[MolecularTarget]:
        """Extract targets from full protein names."""
        targets = []

        # Common protein name patterns
        patterns = [
            (r"(?i)\b(CFTR)\s+(?:chloride\s+)?channel", "CFTR"),
            (r"(?i)\b(?:p53|tp53)\s+(?:tumor\s+)?(?:suppressor)?", "TP53"),
            (r"(?i)\b(?:bcr-abl|bcrabl)\s+(?:fusion)?", "BCR-ABL"),
            (r"(?i)\b(?:her2|erbb2)\s+(?:receptor)?", "HER2"),
            (r"(?i)\bepidermal\s+growth\s+factor\s+receptor", "EGFR"),
            (r"(?i)\bvascular\s+endothelial\s+growth\s+factor", "VEGF"),
            (r"(?i)\btumor\s+necrosis\s+factor", "TNF"),
            (r"(?i)\binterleukin[- ]?(\d+)", "IL\\1"),
            (r"(?i)\berythropoietin", "EPO"),
            (r"(?i)\binsulin(?!\s+receptor)", "INSULIN"),
            (r"(?i)\bhuntingtin", "HTT"),
            (r"(?i)\bdystrophin", "DMD"),
            (r"(?i)\bfactor\s+(viii|8)", "F8"),
            (r"(?i)\bfactor\s+(ix|9)", "F9"),
        ]

        for pattern, gene in patterns:
            if re.search(pattern, description):
                canonical = resolve_gene_alias(gene)
                if canonical not in seen:
                    target = self._create_target(canonical, description)
                    target.confidence = 0.85
                    targets.append(target)
                    seen.add(canonical)

        return targets

    def _extract_mechanisms(self, description: str) -> List[str]:
        """Extract disease mechanisms from text."""
        mechanisms = []
        desc_lower = description.lower()

        for mechanism, keywords in MECHANISM_KEYWORDS.items():
            for keyword in keywords:
                if keyword in desc_lower:
                    if mechanism not in mechanisms:
                        mechanisms.append(mechanism)
                    break

        return mechanisms

    def _extract_tissues(self, description: str) -> List[str]:
        """Extract affected tissues from text."""
        tissues = []
        desc_lower = description.lower()

        for tissue, keywords in TISSUE_KEYWORDS.items():
            for keyword in keywords:
                if keyword in desc_lower:
                    if tissue not in tissues:
                        tissues.append(tissue)
                    break

        return tissues

    def _assign_mechanisms(self, targets: List[MolecularTarget], description: str):
        """Assign mechanisms to targets based on context."""
        desc_lower = description.lower()

        for target in targets:
            target_name_lower = target.name.lower()

            # Look for mechanism keywords near the target name
            for mechanism, keywords in MECHANISM_KEYWORDS.items():
                for keyword in keywords:
                    # Check if keyword appears near target name
                    pattern = rf"(?i)(?:{target.name}|{target_name_lower}).{{0,50}}{re.escape(keyword)}|{re.escape(keyword)}.{{0,50}}(?:{target.name}|{target_name_lower})"
                    if re.search(pattern, description):
                        try:
                            target.mechanism = Mechanism(mechanism)
                            break
                        except ValueError:
                            pass
                if target.mechanism != Mechanism.UNKNOWN:
                    break

            # Infer from target type if mechanism still unknown
            if target.mechanism == Mechanism.UNKNOWN:
                known_info = get_known_target_info(target.name)
                if known_info:
                    # Tumor suppressors typically have loss of function
                    if target.name in {"TP53", "RB1", "PTEN", "APC", "BRCA1", "BRCA2", "VHL"}:
                        target.mechanism = Mechanism.LOSS_OF_FUNCTION
                    # Oncogenes typically have gain of function or overexpression
                    elif target.name in {"KRAS", "BRAF", "EGFR", "MYC", "HER2"}:
                        if "overexpress" in desc_lower or "amplif" in desc_lower:
                            target.mechanism = Mechanism.OVEREXPRESSION
                        else:
                            target.mechanism = Mechanism.GAIN_OF_FUNCTION

    def _verify_extractions(
        self,
        targets: List[MolecularTarget],
        description: str,
        disease_name: str,
    ) -> List[OracleVerification]:
        """Verify extracted targets with the oracle."""
        verifications = []

        for target in targets:
            claim = self._generate_claim(target, disease_name)
            distractors = self._generate_distractors(target, disease_name)

            try:
                result = self.oracle.verify_claim(
                    claim=claim,
                    domain="pathophysiology",
                    distractors=distractors,
                    context=f"Disease pathophysiology: {description[:200]}",
                )
                verification = OracleVerification(
                    claim=claim,
                    verdict=result.verdict,
                    confidence=result.confidence,
                    margin=result.margin,
                    domain="pathophysiology",
                )
                verifications.append(verification)

                # Update target confidence based on oracle
                if result.verdict == "TRUE":
                    target.confidence = min(0.95, target.confidence + 0.1)
                elif result.verdict == "FALSE":
                    target.confidence = max(0.2, target.confidence - 0.3)

            except Exception:
                # Oracle unavailable, skip verification
                pass

        return verifications

    def _generate_claim(self, target: MolecularTarget, disease_name: str) -> str:
        """Generate a claim for oracle verification."""
        if target.mechanism != Mechanism.UNKNOWN:
            return (
                f"{target.name} is involved in {disease_name or 'this disease'} "
                f"through {target.mechanism.value.replace('_', ' ')}"
            )
        return f"{target.name} is a therapeutic target for {disease_name or 'this disease'}"

    def _generate_distractors(self, target: MolecularTarget, disease_name: str) -> List[str]:
        """Generate distractor claims for oracle."""
        return [
            f"{target.name} has no role in {disease_name or 'this disease'}",
            f"{target.name} is only a biomarker, not a therapeutic target",
            f"Targeting {target.name} would worsen the disease",
        ]
