"""
Candidate Generator — Generate therapeutic candidates using NoetherSolve tools.

Routes to existing CRISPR, mRNA, antibody, neoantigen, and PK tools.
"""

from typing import List

from .types import (
    MolecularTarget,
    ModalityRecommendation,
    TherapeuticCandidate,
    Modality,
    Mechanism,
)


class CandidateGenerator:
    """Generate therapeutic candidates based on modality."""

    def __init__(self, oracle=None):
        """Initialize generator."""
        self.oracle = oracle

    def generate(
        self,
        target: MolecularTarget,
        modality: Modality,
        max_candidates: int = 3,
    ) -> List[TherapeuticCandidate]:
        """
        Generate candidates for a target using the specified modality.

        Args:
            target: Molecular target
            modality: Therapeutic modality
            max_candidates: Maximum candidates to generate

        Returns:
            List of therapeutic candidates
        """
        if modality == Modality.CRISPR:
            return self._generate_crispr(target, max_candidates)
        elif modality == Modality.MRNA:
            return self._generate_mrna(target, max_candidates)
        elif modality == Modality.ANTIBODY:
            return self._generate_antibody(target, max_candidates)
        elif modality == Modality.NEOANTIGEN:
            return self._generate_neoantigen(target, max_candidates)
        elif modality == Modality.ASO:
            return self._generate_aso(target, max_candidates)
        elif modality == Modality.SMALL_MOLECULE:
            return self._generate_small_molecule(target, max_candidates)
        elif modality == Modality.GENE_THERAPY:
            return self._generate_gene_therapy(target, max_candidates)
        else:
            return self._generate_generic(target, modality, max_candidates)

    def generate_all(
        self,
        target: MolecularTarget,
        recommendations: List[ModalityRecommendation],
        max_per_modality: int = 2,
    ) -> List[TherapeuticCandidate]:
        """Generate candidates across all recommended modalities."""
        all_candidates = []

        for rec in recommendations:
            candidates = self.generate(target, rec.modality, max_per_modality)
            all_candidates.extend(candidates)

        return all_candidates

    def _generate_crispr(
        self, target: MolecularTarget, max_candidates: int
    ) -> List[TherapeuticCandidate]:
        """Generate CRISPR guide candidates."""
        try:
            from noethersolve.crispr import score_guide
        except ImportError:
            return self._generate_placeholder(target, Modality.CRISPR, max_candidates)

        candidates = []

        # Generate example guides targeting the gene
        # In a real implementation, we would design guides based on gene sequence
        example_guides = self._design_example_guides(target.name, max_candidates)

        for i, (spacer, note) in enumerate(example_guides):
            try:
                report = score_guide(spacer, pam="NGG")

                # Calculate scores
                efficacy = report.activity_score / 100
                safety = 1.0 if report.offtarget_risk == "LOW" else 0.6 if report.offtarget_risk == "MODERATE" else 0.3
                developability = 0.8 if report.verdict == "PASS" else 0.5

                candidate = TherapeuticCandidate(
                    candidate_id=f"CRISPR-{target.name}-{i+1}",
                    modality=Modality.CRISPR,
                    target=target,
                    description=f"CRISPR guide targeting {target.name}: {note}",
                    efficacy_score=efficacy * 100,
                    safety_score=safety * 100,
                    developability_score=developability * 100,
                    modality_data={
                        "spacer": spacer,
                        "pam": "NGG",
                        "activity_score": report.activity_score,
                        "offtarget_risk": report.offtarget_risk,
                        "gc_content": report.gc_content,
                        "verdict": report.verdict,
                    },
                    development_path="Preclinical guide validation → Off-target profiling → IND-enabling studies",
                    key_experiments=[
                        "GUIDE-seq or CIRCLE-seq for off-target analysis",
                        "Editing efficiency in relevant cell types",
                        "In vivo delivery optimization",
                    ],
                )
                candidate.combined_score = (
                    0.4 * candidate.efficacy_score +
                    0.35 * candidate.safety_score +
                    0.25 * candidate.developability_score
                )
                candidates.append(candidate)

            except Exception:
                pass

        return candidates[:max_candidates]

    def _generate_mrna(
        self, target: MolecularTarget, max_candidates: int
    ) -> List[TherapeuticCandidate]:
        """Generate mRNA therapeutic candidates."""
        try:
            from noethersolve.mrna_design import analyze_mrna_design, optimize_codons
        except ImportError:
            return self._generate_placeholder(target, Modality.MRNA, max_candidates)

        candidates = []

        # Generate example mRNA constructs
        example_sequences = self._design_example_mrna(target.name, max_candidates)

        for i, (coding_seq, note) in enumerate(example_sequences):
            try:
                # Analyze mRNA design
                design_report = analyze_mrna_design(
                    coding_sequence=coding_seq,
                    use_pseudouridine=True,
                )

                # Optimize codons
                codon_report = optimize_codons(coding_seq, strategy="balanced")

                # Calculate scores
                quality_map = {"excellent": 95, "good": 75, "fair": 55, "poor": 35}
                efficacy = quality_map.get(design_report.overall_quality, 50)

                risk_map = {"low": 90, "medium": 60, "high": 30}
                safety = risk_map.get(design_report.immunogenicity.tlr7_8_risk.lower(), 50)

                developability = 70 + 20 * codon_report.optimized_cai

                candidate = TherapeuticCandidate(
                    candidate_id=f"mRNA-{target.name}-{i+1}",
                    modality=Modality.MRNA,
                    target=target,
                    description=f"mRNA encoding {target.name}: {note}",
                    efficacy_score=efficacy,
                    safety_score=safety,
                    developability_score=developability,
                    modality_data={
                        "coding_length": len(coding_seq),
                        "overall_quality": design_report.overall_quality,
                        "tlr7_8_risk": design_report.immunogenicity.tlr7_8_risk,
                        "original_cai": codon_report.original_cai,
                        "optimized_cai": codon_report.optimized_cai,
                    },
                    development_path="LNP formulation → In vitro expression → Biodistribution studies → IND",
                    key_experiments=[
                        "Protein expression in target cells",
                        "LNP delivery optimization",
                        "Immunogenicity and reactogenicity studies",
                    ],
                )
                candidate.combined_score = (
                    0.4 * candidate.efficacy_score +
                    0.35 * candidate.safety_score +
                    0.25 * candidate.developability_score
                )
                candidates.append(candidate)

            except Exception:
                pass

        return candidates[:max_candidates]

    def _generate_antibody(
        self, target: MolecularTarget, max_candidates: int
    ) -> List[TherapeuticCandidate]:
        """Generate antibody candidates."""
        try:
            from noethersolve.antibody_developability import assess_developability
        except ImportError:
            return self._generate_placeholder(target, Modality.ANTIBODY, max_candidates)

        candidates = []

        # Generate example antibody sequences
        example_sequences = self._design_example_antibodies(target.name, max_candidates)

        for i, (vh_seq, note) in enumerate(example_sequences):
            try:
                dev_report = assess_developability(vh_seq)

                # Calculate scores based on risk levels
                risk_score = {"LOW": 90, "MODERATE": 65, "HIGH": 35, "VERY_HIGH": 15}
                developability = risk_score.get(dev_report.overall_risk.value, 50)

                # Assume efficacy and safety for demonstration
                efficacy = 70  # Would need binding assay data
                safety = developability * 0.8  # Correlated with developability

                candidate = TherapeuticCandidate(
                    candidate_id=f"Ab-{target.name}-{i+1}",
                    modality=Modality.ANTIBODY,
                    target=target,
                    description=f"Antibody targeting {target.name}: {note}",
                    efficacy_score=efficacy,
                    safety_score=safety,
                    developability_score=developability,
                    modality_data={
                        "vh_length": len(vh_seq),
                        "overall_risk": dev_report.overall_risk.value,
                        "recommendation": dev_report.recommendation,
                    },
                    development_path="Lead optimization → Cell line development → CMC → IND",
                    key_experiments=[
                        "Binding affinity and specificity",
                        "Fc effector function assessment",
                        "Manufacturability and stability testing",
                    ],
                )
                candidate.combined_score = (
                    0.4 * candidate.efficacy_score +
                    0.35 * candidate.safety_score +
                    0.25 * candidate.developability_score
                )
                candidates.append(candidate)

            except Exception:
                pass

        return candidates[:max_candidates]

    def _generate_neoantigen(
        self, target: MolecularTarget, max_candidates: int
    ) -> List[TherapeuticCandidate]:
        """Generate neoantigen candidates."""
        try:
            from noethersolve.neoantigen_pipeline import evaluate_neoantigen
        except ImportError:
            return self._generate_placeholder(target, Modality.NEOANTIGEN, max_candidates)

        candidates = []

        # Generate example neoantigen peptides
        example_peptides = self._design_example_neoantigens(target.name, max_candidates)

        for i, (peptide, allele, note) in enumerate(example_peptides):
            try:
                report = evaluate_neoantigen(peptide, allele=allele)

                efficacy = report.combined_score * 100 if report.pipeline_pass else 30
                safety = 70  # Neoantigens are generally tumor-specific
                developability = 60 if report.pipeline_pass else 30

                candidate = TherapeuticCandidate(
                    candidate_id=f"Neo-{target.name}-{i+1}",
                    modality=Modality.NEOANTIGEN,
                    target=target,
                    description=f"Neoantigen vaccine for {target.name}: {note}",
                    efficacy_score=efficacy,
                    safety_score=safety,
                    developability_score=developability,
                    modality_data={
                        "peptide": peptide,
                        "allele": allele,
                        "combined_score": report.combined_score,
                        "pipeline_pass": report.pipeline_pass,
                        "limiting_step": report.limiting_step,
                    },
                    development_path="Peptide synthesis → T cell assays → Vaccine formulation → Clinical trial",
                    key_experiments=[
                        "T cell activation assays",
                        "HLA restriction confirmation",
                        "Immunogenicity in relevant models",
                    ],
                )
                candidate.combined_score = (
                    0.4 * candidate.efficacy_score +
                    0.35 * candidate.safety_score +
                    0.25 * candidate.developability_score
                )
                candidates.append(candidate)

            except Exception:
                pass

        return candidates[:max_candidates]

    def _generate_aso(
        self, target: MolecularTarget, max_candidates: int
    ) -> List[TherapeuticCandidate]:
        """Generate ASO candidates."""
        # ASO design would require gene sequence; using placeholder
        return self._generate_placeholder(target, Modality.ASO, max_candidates)

    def _generate_small_molecule(
        self, target: MolecularTarget, max_candidates: int
    ) -> List[TherapeuticCandidate]:
        """Generate small molecule concepts."""
        # Would require docking/virtual screening; using placeholder with PK
        candidates = []

        for i in range(max_candidates):
            # Conceptual small molecule approaches
            if target.mechanism == Mechanism.GAIN_OF_FUNCTION:
                approach = "competitive inhibitor"
            elif target.mechanism == Mechanism.OVEREXPRESSION:
                approach = "allosteric inhibitor"
            else:
                approach = "modulator"

            candidate = TherapeuticCandidate(
                candidate_id=f"SM-{target.name}-{i+1}",
                modality=Modality.SMALL_MOLECULE,
                target=target,
                description=f"Small molecule {approach} for {target.name}",
                efficacy_score=70,
                safety_score=65,
                developability_score=80,
                modality_data={
                    "approach": approach,
                    "oral_bioavailability": "likely" if i == 0 else "to be optimized",
                },
                development_path="HTS → Lead optimization → ADMET → IND-enabling → Phase I",
                key_experiments=[
                    "Biochemical IC50 determination",
                    "Cellular potency assays",
                    "ADMET and PK profiling",
                ],
            )
            candidate.combined_score = (
                0.4 * candidate.efficacy_score +
                0.35 * candidate.safety_score +
                0.25 * candidate.developability_score
            )
            candidates.append(candidate)

        return candidates

    def _generate_gene_therapy(
        self, target: MolecularTarget, max_candidates: int
    ) -> List[TherapeuticCandidate]:
        """Generate gene therapy concepts."""
        candidates = []

        vectors = ["AAV9", "AAV8", "Lentivirus"]

        for i, vector in enumerate(vectors[:max_candidates]):
            candidate = TherapeuticCandidate(
                candidate_id=f"GT-{target.name}-{i+1}",
                modality=Modality.GENE_THERAPY,
                target=target,
                description=f"{vector}-based gene therapy for {target.name}",
                efficacy_score=75,
                safety_score=55,
                developability_score=50,
                modality_data={
                    "vector": vector,
                    "payload": f"{target.name} cDNA",
                    "promoter": "tissue-specific" if i == 0 else "constitutive",
                },
                development_path="Vector optimization → Biodistribution → Toxicology → IND",
                key_experiments=[
                    "Transgene expression in target tissue",
                    "Vector biodistribution",
                    "Immune response to vector",
                ],
            )
            candidate.combined_score = (
                0.4 * candidate.efficacy_score +
                0.35 * candidate.safety_score +
                0.25 * candidate.developability_score
            )
            candidates.append(candidate)

        return candidates

    def _generate_generic(
        self, target: MolecularTarget, modality: Modality, max_candidates: int
    ) -> List[TherapeuticCandidate]:
        """Generate generic placeholder candidates."""
        return self._generate_placeholder(target, modality, max_candidates)

    def _generate_placeholder(
        self, target: MolecularTarget, modality: Modality, max_candidates: int
    ) -> List[TherapeuticCandidate]:
        """Generate placeholder candidates when tools unavailable."""
        candidates = []

        for i in range(max_candidates):
            candidate = TherapeuticCandidate(
                candidate_id=f"{modality.value[:3].upper()}-{target.name}-{i+1}",
                modality=modality,
                target=target,
                description=f"{modality.value} approach for {target.name}",
                efficacy_score=50,
                safety_score=50,
                developability_score=50,
                development_path="To be determined based on modality requirements",
                key_experiments=["Detailed experimental plan pending"],
            )
            candidate.combined_score = 50
            candidates.append(candidate)

        return candidates

    # ── Example sequence generators (simplified) ──────────────────────

    def _design_example_guides(self, gene: str, n: int) -> List[tuple]:
        """Design example CRISPR guides for a gene."""
        # In practice, this would use actual gene sequence
        guides = [
            ("GTGGATCCAGACTGCCTTCC", f"Exon 1 region of {gene}"),
            ("CCATTGTTCAATATCGTCCG", f"Exon 2 region of {gene}"),
            ("GGGTGGGTGTGTCTACAGGA", f"Exon 3 region of {gene}"),
        ]
        return guides[:n]

    def _design_example_mrna(self, gene: str, n: int) -> List[tuple]:
        """Design example mRNA sequences."""
        # Placeholder sequences
        sequences = [
            ("AUGGCUAAAGCUGCUGGACUGGCUCCUGGUUUUACUGGUACCUGCCAUGGCAGAAGGCAG", f"Full-length {gene}"),
            ("AUGGCUAAAGCUGCUGGACUGGCUCCUGGUUUUACUGGUACCUGCCAUGGCAGAAGGCAGUAG", f"Optimized {gene}"),
        ]
        return sequences[:n]

    def _design_example_antibodies(self, gene: str, n: int) -> List[tuple]:
        """Design example antibody VH sequences."""
        # Placeholder VH sequences based on known formats
        sequences = [
            (
                "EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIYPTNGYTRYAD"
                "SVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYCSRWGGDGFYAMDYWGQGTLVTVSS",
                f"Human VH targeting {gene}"
            ),
            (
                "QVQLVQSGAEVKKPGASVKVSCKASGYTFTSYGISWVRQAPGQGLEWMGWISAYNG"
                "NTNYAQKLQGRVTMTTDTSTSTAYMELRSLRSDDTAVYYCARDVPLGYSMDVWGQGTTVTVSS",
                f"Humanized VH targeting {gene}"
            ),
        ]
        return sequences[:n]

    def _design_example_neoantigens(self, gene: str, n: int) -> List[tuple]:
        """Design example neoantigen peptides."""
        # Common mutation-derived peptides
        peptides = [
            ("VVVGADGVGK", "HLA-A*02:01", f"Mutant {gene} peptide"),
            ("HMTEVVRRC", "HLA-A*02:01", f"Alternative mutant {gene} peptide"),
        ]
        return peptides[:n]
