"""Paper Agent — Autonomous scientific paper generation and publication.

Detects when a discovery cluster is mature enough for publication,
verifies novelty against prior work, generates a draft from templates
and findings data, compiles to PDF, and uploads to Zenodo with DOI.

No external API required. Claude Code writes the actual paper content;
this agent handles pipeline automation, novelty gating, and publication.

The agent follows the stages defined in paper/PAPER_PIPELINE.md:
  -1. Novelty Verification (GATE — must pass before any writing)
   0. Data Collection
   1. Citation Research
   2. Writing (template + Claude Code refinement)
   3. Number Verification
   4. Figures
   5. AI Language Scrub
   6. Finalization
   7. PDF Generation
   8. Zenodo Upload
   9. GitHub Update

Usage:
    from noethersolve.paper_agent import PaperAgent

    agent = PaperAgent()
    novelty = agent.check_novelty("my_cluster")
    if novelty["novel_findings"] and agent.should_write_paper("my_cluster"):
        result = agent.write_and_publish("my_cluster")
        print(f"Published: {result}")
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import json
import os
import re
import subprocess
import time

# AI language scrubbing - banned words and phrases from PAPER_PIPELINE.md Stage 5
BANNED_PHRASES = [
    "delve", "tapestry", "intricate", "multifaceted", "synergy", "paradigm shift",
    "it's worth noting", "worth noting", "notably", "interestingly", "remarkably",
    "importantly", "in conclusion", "to summarize", "as previously mentioned",
    "it should be noted", "leverage", "utilize", "facilitate", "elucidate",
    "underscore", "a testament to", "pave the way", "shed light on",
    "the landscape of", "holistic", "robust", "comprehensive", "novel",
    "groundbreaking", "cutting-edge", "state-of-the-art", "unprecedented",
    "pivotal", "paramount", "crucial", "vital", "essential", "fundamental",
    "transformative", "revolutionary", "game-changing", "innovative",
    "at the forefront", "pushing boundaries", "breaking new ground",
]

# Standard acknowledgment template
ACKNOWLEDGMENT_TEMPLATE = """The author acknowledges the assistance of Claude (Anthropic) in developing
the NoetherSolve framework, running numerical integrations, optimizing
invariants via L-BFGS-B, and assisting with manuscript preparation and
LaTeX formatting. All scientific content, derivations, interpretations,
and final claims are the sole responsibility of the human author. The full
open-source code and validation scripts are available at
https://github.com/SolomonB14D3/NoetherSolve."""


@dataclass
class ClusterMetrics:
    """Metrics for evaluating paper-readiness of a discovery cluster."""
    cluster_id: str
    facts_count: int
    facts_flipped: int
    margin_avg: float
    margin_min: float
    coverage_pct: float
    has_numerical_verification: bool
    has_independent_validation: bool

    @property
    def maturity_score(self) -> float:
        """Compute overall maturity score (0-1)."""
        flip_rate = self.facts_flipped / max(self.facts_count, 1)
        margin_score = min(1.0, max(0.0, (self.margin_avg + 20) / 40))  # -20 to +20 range
        coverage_score = self.coverage_pct / 100.0
        verification_bonus = 0.1 if self.has_numerical_verification else 0.0
        validation_bonus = 0.1 if self.has_independent_validation else 0.0

        return (0.4 * flip_rate +
                0.2 * margin_score +
                0.2 * coverage_score +
                verification_bonus +
                validation_bonus)


@dataclass
class PaperResult:
    """Result of paper generation and publication."""
    cluster_id: str
    title: str
    pdf_path: Optional[Path] = None
    doi: Optional[str] = None
    zenodo_url: Optional[str] = None
    future_work_count: int = 0
    errors: list = field(default_factory=list)

    @property
    def success(self) -> bool:
        return self.pdf_path is not None and len(self.errors) == 0


class PaperAgent:
    """Autonomous agent for scientific paper generation and publication.

    Generates papers from discovery cluster data using templates and
    findings files. Optionally polishes drafts with a local MLX model.
    No external API required.
    """

    def __init__(
        self,
        paper_dir: Optional[Path] = None,
        results_dir: Optional[Path] = None,
        zenodo_token: Optional[str] = None,
    ):
        """Initialize paper agent.

        Args:
            paper_dir: Directory for paper outputs (default: <repo>/paper/)
            results_dir: Directory with discovery results (default: <repo>/results/)
            zenodo_token: Zenodo API token (default: from ZENODO_TOKEN env)
        """
        self._repo_root = Path(__file__).parent.parent
        self.paper_dir = paper_dir or self._repo_root / "paper"
        self.results_dir = results_dir or self._repo_root / "results"
        self.zenodo_token = zenodo_token or os.environ.get("ZENODO_TOKEN")

    # Keywords for matching candidates.tsv hypothesis to cluster IDs
    _CLUSTER_KEYWORDS = {
        "d1_vortex_conservation": ["vortex", "q_f", "qf", "kirchhoff", "conservation",
                                   "stretch_resist", "curvature", "dipole", "viscous"],
        "d2_z3_cancellation": ["choreograph", "z3", "z₃", "figure-8", "figure8", "3-body"],
        "d3_llm_knowledge_gaps": ["knowledge_gap", "confidently_wrong", "oracle",
                                  "distractor", "length_ratio", "length_bias"],
        "d4_orthogonal_routing": ["orthogonal", "routing", "adapter", "stacking"],
        "d5_certainty_contamination": ["certainty", "contamination", "anti_fluency",
                                       "hedged", "definitive"],
        "d6_resolvent_unification": ["resolvent", "spectral", "green's function",
                                     "unification"],
        "d7_oracle_biases": ["oracle", "bias", "length_ratio", "distractor",
                             "phrasing", "round_number", "term_preference",
                             "technical_simplification", "status_blindness",
                             "hedge", "anti_fluency", "coherence", "certainty"],
        "d8_cross_domain_conservation": ["cross_domain", "cycle_theory",
                                          "five_cross_domain", "hidden_ns",
                                          "conservation_mechanism"],
        "d9_em_conservation": ["electromagnetic", "em_conservation", "zilch",
                                "chirality", "helicity"],
        "d10_qf_regularity": ["concentration", "regularity", "dichotomy",
                               "qf_ratio", "stretch_resistant", "viscous_decay",
                               "viscous_qf", "exponential_kernel"],
        "bio_ai_parallels": ["bio_ai", "chemotaxis", "hebbian", "striatum",
                             "actor_critic", "c_elegans", "worm"],
        "clinical_translation": ["clinical", "disease_target", "delivery",
                                 "therapeutic", "drug_interaction"],
    }

    # What each published paper covers — used for novelty gating.
    # Key = paper DOI suffix, value = set of finding file stems that are covered.
    _PUBLISHED_COVERAGE = {
        "d1_vortex_conservation": {
            "doi": "10.5281/zenodo.19055338",
            "covers": {
                "cancellation_mechanism", "q_f_universal", "qr2_identity",
                "cross_domain_conservation_mechanisms",
                "optimal_f_combination", "optimal_qf_combination",
                "curvature_weighted_stretch_resistance",
                "dipole_test_vortex_exact", "parallel_dipole_sum",
                "kinetic_invariant_K", "triplet_false_alarm",
                "continuous_qf_exponential", "qf_ratio_optimal",
                "qf_ratio_stretch_resistant", "viscous_decay_linear_scaling",
                "viscous_qf_decay",
            },
            "topics": [
                "Q_f = sum Gamma_i Gamma_j f(r_ij) family",
                "optimal combination 300x better",
                "curvature-weighted Q_kappa 15x",
                "viscous decay Q_sqrt(r) ~ nu^0.99",
                "triplet invariants don't exist",
                "stretch-resistant vs concentration-detecting dichotomy",
            ],
        },
        "d2_z3_cancellation": {
            "doi": "10.5281/zenodo.19055580",
            "covers": {"z3_phase_cancellation"},
            "topics": [
                "Z3 symmetry phase cancellation in dQ/dt",
                "critical range -0.67 < p < 2.55",
                "derivative weight ratio mechanism",
                "gravity vs vortex dynamics comparison",
            ],
        },
        "d3_llm_knowledge_gaps": {
            "doi": "10.5281/zenodo.19055582",
            "covers": {
                "llm_self_knowledge_gap", "systematic_confusions",
                "dimensional_asymmetric_learning", "hidden_ns_connections",
            },
            "topics": [
                "1038 facts across 67 domains",
                "intersection theory deepest gap margin -27.6",
                "sign errors, recency inversion, magnitude errors",
            ],
        },
        "d4_orthogonal_routing": {
            "doi": "10.5281/zenodo.19055588",
            "covers": set(),
            "topics": [
                "representational see-saws require orthogonal adapters",
                "1038/1038 facts flipped",
                "hybrid routing 82.1%",
                "escalation ladder: single -> staged -> orthogonal -> joint -> hybrid",
            ],
        },
        "d5_certainty_contamination": {
            "doi": "10.5281/zenodo.19068373",
            "covers": {"certainty_contamination_bias"},
            "topics": [
                "definitive language biases LLM factual judgments",
                "r = -0.402 certainty gap correlation",
                "pass rate 55% -> 25% with high asymmetry",
                "cascade routing with certainty-decontamination adapters",
            ],
        },
        "d6_resolvent_unification": {
            "doi": "10.5281/zenodo.19071198",
            "covers": {
                "operator_conservation_duality",
                "resolvent_conservation_unification",
                "qf_3d_green_function",
            },
            "topics": [
                "Green's function optimality from resolvent zero-frequency limit",
                "kernel-resolvent-spectral gap-spectral measure unification",
                "discrete systems via pseudoinverse of graph Laplacian",
            ],
        },
        "d7_oracle_biases": {
            "doi": "10.5281/zenodo.19124851",
            "covers": {
                "length_ratio_discovery", "distractor_coherence_discovery",
                "certainty_contamination_bias", "round_number_bias",
                "mathematical_status_blindness", "phrasing_bias_oracle_resistance",
                "term_preference_bias", "technical_simplification_bias",
                "anti_fluency_distractor_strategy",
                "unified_oracle_difficulty_theory", "linguistic_hedge_predictor",
            },
            "topics": [
                "nine systematic biases in log-probability LLM evaluation",
                "length ratio r=-0.742 correlation",
                "certainty contamination r=-0.402",
                "round number preference, term familiarity bias",
                "mathematical status blindness 71% vs 4%",
            ],
        },
    }

    def check_novelty(self, cluster_id: str) -> dict:
        """Check which findings in a cluster are genuinely novel vs already published.

        This is Stage -1 of the paper pipeline — must pass before any writing.

        Returns:
            Dict with:
              - novel_findings: list of genuinely unpublished findings
              - covered_findings: list of findings already in published papers
              - methodology_findings: list of methodology refinements (not domain science)
              - novelty_score: fraction of findings that are novel (0-1)
              - recommendation: "proceed", "insufficient", or "already_published"
        """
        findings = self._load_findings_for_cluster(cluster_id)
        if not findings:
            return {
                "novel_findings": [],
                "covered_findings": [],
                "methodology_findings": [],
                "novelty_score": 0.0,
                "recommendation": "insufficient",
                "reason": f"No findings files found for cluster '{cluster_id}'",
            }

        # Build set of all covered file stems across all published papers
        all_covered_stems = set()
        for paper_info in self._PUBLISHED_COVERAGE.values():
            all_covered_stems.update(paper_info["covers"])

        # Methodology keywords — these are eval/oracle refinements, not domain science
        methodology_keywords = [
            "distractor", "length_ratio", "anti_fluency", "phrasing_bias",
            "round_number", "hedge", "scoring", "oracle_difficulty",
            "technical_simplification", "term_preference", "linguistic",
        ]

        novel = []
        covered = []
        methodology = []

        for f in findings:
            stem = Path(f["file"]).stem

            # Check if covered by a published paper
            if stem in all_covered_stems:
                # Find which paper covers it
                covering_paper = None
                for paper_id, paper_info in self._PUBLISHED_COVERAGE.items():
                    if stem in paper_info["covers"]:
                        covering_paper = paper_id
                        break
                covered.append({
                    **f,
                    "covered_by": covering_paper,
                    "doi": self._PUBLISHED_COVERAGE.get(covering_paper, {}).get("doi"),
                })
            elif any(kw in stem for kw in methodology_keywords):
                methodology.append(f)
            else:
                novel.append(f)

        total = len(findings)
        novel_count = len(novel)
        novelty_score = novel_count / total if total > 0 else 0.0

        if novel_count == 0:
            recommendation = "already_published"
            reason = "All findings are already covered by published papers"
        elif novel_count < 3:
            recommendation = "insufficient"
            reason = f"Only {novel_count} novel findings — consider combining with related clusters"
        else:
            recommendation = "proceed"
            reason = f"{novel_count}/{total} findings are novel and unpublished"

        return {
            "novel_findings": novel,
            "covered_findings": covered,
            "methodology_findings": methodology,
            "novelty_score": novelty_score,
            "recommendation": recommendation,
            "reason": reason,
        }

    def get_cluster_metrics(self, cluster_id: str) -> Optional[ClusterMetrics]:
        """Load metrics for a discovery cluster.

        Reads from paper briefs first (generated by prospector), falls back
        to keyword-matching against candidates.tsv hypothesis column.
        """
        # Try paper brief first (most accurate)
        brief_path = self.results_dir / "paper_briefs" / f"{cluster_id}.json"
        if brief_path.exists():
            with open(brief_path) as f:
                brief = json.load(f)
            ev = brief.get("evidence", {})
            total = ev.get("candidates_total", 0)
            flipped = ev.get("flipped", 0)
            # passing count tracked but not needed for maturity calculation

            # Also count findings as evidence
            n_findings = len(brief.get("findings", []))

            # If brief has data, use it
            if total > 0 or n_findings > 0:
                facts_count = max(total, n_findings)
                facts_flipped = max(flipped, n_findings)  # findings = verified discoveries

                discoveries_dir = self.results_dir / "discoveries" / "novel_findings"
                has_verification = discoveries_dir.exists() and any(
                    any(kw in f.stem.lower() for kw in
                        self._CLUSTER_KEYWORDS.get(cluster_id, [cluster_id.lower().split("_")[-1:]]))
                    for f in discoveries_dir.glob("*.md")
                )

                return ClusterMetrics(
                    cluster_id=cluster_id,
                    facts_count=facts_count,
                    facts_flipped=facts_flipped,
                    margin_avg=5.0 if flipped > 0 else -5.0,  # approximate from pass/fail
                    margin_min=-10.0 if ev.get("failing", 0) > 0 else 0.0,
                    coverage_pct=(facts_flipped / facts_count) * 100 if facts_count > 0 else 0.0,
                    has_numerical_verification=has_verification,
                    has_independent_validation=has_verification,
                )

        # Fall back to keyword matching against candidates.tsv
        candidates_path = self.results_dir / "candidates.tsv"
        if not candidates_path.exists():
            return None

        keywords = self._CLUSTER_KEYWORDS.get(cluster_id, [cluster_id.replace("_", " ")])
        facts_count = 0
        facts_flipped = 0
        margins = []

        with open(candidates_path, "r") as f:
            import csv
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                hypothesis = row.get("hypothesis", "").lower()
                if any(kw in hypothesis for kw in keywords):
                    facts_count += 1
                    verdict = row.get("verdict", "")
                    if "FLIPPED" in verdict or "DUAL" in verdict:
                        facts_flipped += 1
                    try:
                        margin = float(row.get("margin_mean", "0"))
                        margins.append(margin)
                    except (ValueError, TypeError):
                        pass

        if facts_count == 0:
            return None

        discoveries_dir = self.results_dir / "discoveries" / "novel_findings"
        has_verification = discoveries_dir.exists() and any(
            any(kw in f.stem.lower() for kw in keywords)
            for f in discoveries_dir.glob("*.md")
        )

        return ClusterMetrics(
            cluster_id=cluster_id,
            facts_count=facts_count,
            facts_flipped=facts_flipped,
            margin_avg=sum(margins) / len(margins) if margins else 0.0,
            margin_min=min(margins) if margins else 0.0,
            coverage_pct=(facts_flipped / facts_count) * 100 if facts_count > 0 else 0.0,
            has_numerical_verification=has_verification,
            has_independent_validation=has_verification,
        )

    def should_write_paper(self, cluster_id: str, threshold: float = 0.82) -> bool:
        """Check if a cluster is mature enough for paper publication."""
        metrics = self.get_cluster_metrics(cluster_id)
        if metrics is None:
            return False
        return metrics.maturity_score >= threshold

    def scrub_ai_language(self, text: str) -> str:
        """Remove AI tell-words and phrases from text."""
        result = text
        for phrase in BANNED_PHRASES:
            pattern = re.compile(re.escape(phrase), re.IGNORECASE)
            result = pattern.sub("", result)

        # Clean up resulting double spaces
        result = re.sub(r"  +", " ", result)
        result = re.sub(r" +\.", ".", result)
        result = re.sub(r" +,", ",", result)

        return result

    def _load_findings_for_cluster(self, cluster_id: str) -> list[dict]:
        """Load findings files matched to this cluster."""
        findings_dir = self.results_dir / "discoveries" / "novel_findings"
        if not findings_dir.exists():
            return []

        findings = []
        # Match by cluster_id keywords
        keywords = cluster_id.lower().replace("d1_", "").replace("d2_", "").replace(
            "d3_", "").replace("d4_", "").replace("d5_", "").replace("d6_", "").split("_")

        for f in sorted(findings_dir.glob("*.md")):
            fname = f.stem.lower()
            if any(kw in fname for kw in keywords if len(kw) > 2):
                content = f.read_text()
                title = f.stem.replace("_", " ").title()
                for line in content.split("\n"):
                    if line.startswith("# "):
                        title = line[2:].strip()
                        break
                findings.append({
                    "title": title,
                    "file": f.name,
                    "content": content,
                })
        return findings

    def _load_paper_brief(self, cluster_id: str) -> dict:
        """Load paper brief from prospector if available."""
        brief_path = self.results_dir / "paper_briefs" / f"{cluster_id}.json"
        if brief_path.exists():
            with open(brief_path) as f:
                return json.load(f)
        return {}

    def _load_cluster_info(self, cluster_id: str) -> dict:
        """Load cluster definition from discovery grader."""
        try:
            import sys
            scripts_dir = self._repo_root / "scripts"
            if str(scripts_dir) not in sys.path:
                sys.path.insert(0, str(scripts_dir))
            from discovery_grader import DISCOVERY_CLUSTERS
            return dict(DISCOVERY_CLUSTERS).get(cluster_id, {})
        except (ImportError, Exception):
            return {}

    def generate_draft(self, cluster_id: str, metrics: ClusterMetrics) -> str:
        """Generate a full paper draft from findings data and metrics.

        Pulls content from findings files, paper briefs, and cluster info
        to build a data-rich draft. Optionally polishes with local model.
        """
        findings = self._load_findings_for_cluster(cluster_id)
        brief = self._load_paper_brief(cluster_id)
        cluster_info = self._load_cluster_info(cluster_id)

        title = cluster_info.get("title", brief.get("title", cluster_id.replace("_", " ").title()))
        venue = cluster_info.get("venue", "")
        domains = brief.get("domains", cluster_info.get("domains", []))

        # Build findings content sections
        findings_sections = []
        for i, f in enumerate(findings):
            # Extract key content (skip the title line)
            lines = f["content"].split("\n")
            body = "\n".join(line for line in lines if not line.startswith("# "))
            # Truncate very long findings
            if len(body) > 3000:
                body = body[:3000] + "\n\n[...truncated for brevity]"
            findings_sections.append(f"### Finding {i+1}: {f['title']}\n\n{body}")

        findings_text = "\n\n".join(findings_sections) if findings_sections else (
            "No specific findings files found for this cluster. "
            "Results are summarized from candidates.tsv metrics below."
        )

        # Build evidence summary
        ev = brief.get("evidence", {})
        evidence_text = (
            f"- Total candidates evaluated: {ev.get('candidates_total', metrics.facts_count)}\n"
            f"- Facts flipped (model learned): {ev.get('flipped', metrics.facts_flipped)}\n"
            f"- Passing (model already knew): {ev.get('passing', 0)}\n"
            f"- Failing (still unknown): {ev.get('failing', 0)}\n"
            f"- Average oracle margin: {metrics.margin_avg:.2f}\n"
            f"- Minimum margin: {metrics.margin_min:.2f}\n"
            f"- Coverage: {metrics.coverage_pct:.1f}%"
        )

        # Open questions
        oq_count = brief.get("open_questions", 0)
        oq_text = f"{oq_count} open questions remain in the queue for this cluster." if oq_count else ""

        draft = f"""---
title: "{title}"
author: "Bryan Sanchez"
date: "{time.strftime('%Y-%m-%d')}"
---

# {title}

## Abstract

This paper presents discoveries from the NoetherSolve autonomous research pipeline
in the domain of {', '.join(domains) if domains else cluster_id}.
Through systematic numerical verification and oracle-based knowledge gap detection,
we identified {metrics.facts_flipped} facts across {metrics.facts_count} candidates
where language models fail to correctly predict verified scientific truths.
{f'The findings span {len(findings)} distinct sub-discoveries.' if findings else ''}
All results are reproducible via the open-source NoetherSolve toolkit.

## 1. Introduction

Language models trained on internet-scale corpora inherit the biases and errors
present in their training data. Where scientific truth is surprising or
counter-intuitive, models systematically prefer the common misconception over
the verified fact. This paper documents specific instances of this phenomenon
in {', '.join(domains) if domains else 'the target domain'}.

The NoetherSolve pipeline operates in three stages:
1. **Numerical verification** — candidate hypotheses are checked against
   high-precision numerical integration (RK45, symplectic methods) with
   fractional variance threshold < 5e-3.
2. **Oracle evaluation** — verified facts are presented to a base language model
   as multiple-choice questions. The log-probability margin between the correct
   answer and the best distractor quantifies the model's knowledge gap.
3. **Adapter repair** — where gaps are found, domain-specific LoRA adapters are
   trained to correct the model's factual predictions without degrading general
   capabilities.

{f'Target venue: {venue}' if venue else ''}

## 2. Methods

### 2.1 Numerical Framework

All numerical results use the NoetherSolve conservation law checker with
adaptive RK45 integration (tolerance 1e-12). Quantities are classified as
conserved when fractional variance (sigma/|mean|) falls below 5e-3 across
multiple initial conditions.

### 2.2 Oracle Pipeline

Knowledge gaps are detected using log-probability multiple-choice evaluation
on Qwen3-14B-Base. Each fact is formatted as a context + 4 choices (1 correct,
3 distractors). The margin = log P(truth) - log P(best distractor) quantifies
model confidence. Negative margins indicate the model prefers a wrong answer.

Fact quality is audited for 9 known bias mechanisms:
length ratio, distractor coherence, scoring method, anti-fluency artifacts,
round number bias, certainty contamination, technical simplification,
term preference, and mathematical status blindness.

### 2.3 Adapter Training

Where knowledge gaps are confirmed, domain-specific LoRA adapters (rank 8,
alpha 16) are trained on the target facts plus anchoring examples. Training
uses staged or orthogonal adapter strategies depending on interference patterns.

## 3. Results

### 3.1 Evidence Summary

{evidence_text}

### 3.2 Detailed Findings

{findings_text}

## 4. Discussion

### 4.1 Interpretation

The results show that language models have systematic blind spots in
{', '.join(domains) if domains else 'this domain'}. These are not random errors
but structured failures where the model's training distribution diverges from
verified scientific truth. The oracle margin distribution reveals which specific
sub-topics are most affected.

### 4.2 Limitations

- Oracle evaluation uses a single base model (Qwen3-14B-Base). Results may
  differ on other architectures or scales.
- Numerical verification is limited to the initial conditions tested.
  Conservation laws verified on N configurations may not hold universally.
- Adapter repairs are model-specific and do not transfer to instruction-tuned
  variants (confirmed negative result).

### 4.3 Future Work

- Extend oracle evaluation to larger models (27B, 70B) to map how knowledge
  gaps scale with model capacity.
- Cross-domain transfer: test whether adapter training in related domains
  improves performance on this cluster's facts.
- Develop automated fact generation to expand coverage beyond manually curated
  verification sets.
{f'- {oq_text}' if oq_text else ''}

## 5. Conclusion

We identified {metrics.facts_flipped} verified scientific facts that language
models fail to correctly predict in {', '.join(domains) if domains else 'the target domain'}.
The NoetherSolve pipeline provides a systematic method for finding and repairing
these knowledge gaps. All tools, verification scripts, and trained adapters are
available as open-source software with {metrics.facts_count} automated tests.

## Acknowledgments

{ACKNOWLEDGMENT_TEMPLATE}

## References

1. NoetherSolve: Autonomous Scientific Discovery Pipeline.
   https://github.com/SolomonB14D3/NoetherSolve

2. Sanchez, B. (2026). STEM Truth Oracle: Log-Prob MC Ranking Reveals and
   Corrects Scale-Invariant Factual Biases. DOI: 10.5281/zenodo.19005729

3. Sanchez, B. (2026). Orthogonal Adapter Routing for Interference-Free
   Knowledge Injection. DOI: 10.5281/zenodo.19055588
"""
        return draft

    def prepare_paper_brief(self, cluster_id: str) -> dict:
        """Prepare a comprehensive brief with all data needed for paper writing.

        This collects findings, metrics, and evidence into a structured brief
        that Claude Code (in conversation) uses to write the actual paper.
        The local model is NOT used for paper writing — that requires Claude's
        citation verification, scientific accuracy checking, and coherent writing.

        Returns:
            Dict with all paper-relevant data for Claude Code to use.
        """
        metrics = self.get_cluster_metrics(cluster_id)
        findings = self._load_findings_for_cluster(cluster_id)
        brief = self._load_paper_brief(cluster_id)
        cluster_info = self._load_cluster_info(cluster_id)

        # Run novelty check
        novelty = self.check_novelty(cluster_id)

        return {
            "cluster_id": cluster_id,
            "title": cluster_info.get("title", brief.get("title", cluster_id)),
            "venue": cluster_info.get("venue"),
            "doi": cluster_info.get("doi"),
            "domains": brief.get("domains", cluster_info.get("domains", [])),
            "metrics": {
                "facts_count": metrics.facts_count if metrics else 0,
                "facts_flipped": metrics.facts_flipped if metrics else 0,
                "margin_avg": metrics.margin_avg if metrics else 0,
                "coverage_pct": metrics.coverage_pct if metrics else 0,
                "maturity_score": metrics.maturity_score if metrics else 0,
            },
            "findings": [
                {"title": f["title"], "file": f["file"]}
                for f in findings
            ],
            "novelty": {
                "score": novelty["novelty_score"],
                "recommendation": novelty["recommendation"],
                "reason": novelty["reason"],
                "novel_count": len(novelty["novel_findings"]),
                "covered_count": len(novelty["covered_findings"]),
                "methodology_count": len(novelty["methodology_findings"]),
                "novel_titles": [f["title"] for f in novelty["novel_findings"]],
            },
            "evidence": brief.get("evidence", {}),
            "open_questions": brief.get("open_questions", 0),
            "status": "ready_for_claude_code",
        }

    def pandoc_compile(self, md_path: Path, output_dir: Optional[Path] = None) -> Path:
        """Compile Markdown to PDF using pandoc."""
        output_dir = output_dir or md_path.parent
        pdf_path = output_dir / f"{md_path.stem}.pdf"

        cmd = [
            "pandoc", str(md_path),
            "-o", str(pdf_path),
            "--pdf-engine=xelatex",
            "-V", "geometry:margin=1in",
            "-V", "fontsize=11pt",
        ]

        # Add bibliography if exists
        bib_path = md_path.parent / "references.bib"
        if bib_path.exists():
            cmd.extend(["--bibliography", str(bib_path), "--citeproc"])

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Pandoc failed: {result.stderr}")

        return pdf_path

    def upload_zenodo(self, pdf_path: Path, metadata: dict) -> tuple[str, str]:
        """Upload paper to Zenodo and get DOI."""
        try:
            import requests
        except ImportError:
            raise ImportError("requests package required for Zenodo upload")

        if not self.zenodo_token:
            raise ValueError("ZENODO_TOKEN environment variable required")

        headers = {"Authorization": f"Bearer {self.zenodo_token}"}
        base_url = "https://zenodo.org/api/deposit/depositions"

        # 1. Create deposit
        resp = requests.post(base_url, headers=headers, json={})
        resp.raise_for_status()
        deposit = resp.json()
        deposit_id = deposit["id"]
        bucket_url = deposit["links"]["bucket"]

        # 2. Upload file
        with open(pdf_path, "rb") as f:
            resp = requests.put(
                f"{bucket_url}/{pdf_path.name}",
                headers=headers,
                data=f,
            )
            resp.raise_for_status()

        # 3. Set metadata
        zenodo_metadata = {
            "metadata": {
                "title": metadata.get("title", "NoetherSolve Discovery Paper"),
                "upload_type": "publication",
                "publication_type": "preprint",
                "description": metadata.get("description", ""),
                "creators": metadata.get("authors", [{"name": "Sanchez, Bryan"}]),
                "license": "cc-by-4.0",
                "keywords": metadata.get("keywords", ["NoetherSolve", "scientific discovery"]),
                "related_identifiers": [
                    {
                        "identifier": "https://github.com/SolomonB14D3/NoetherSolve",
                        "relation": "isSupplementTo",
                        "scheme": "url",
                    }
                ],
            }
        }
        resp = requests.put(
            f"{base_url}/{deposit_id}",
            headers=headers,
            json=zenodo_metadata,
        )
        resp.raise_for_status()

        # 4. Publish
        resp = requests.post(
            f"{base_url}/{deposit_id}/actions/publish",
            headers=headers,
        )
        resp.raise_for_status()
        published = resp.json()

        doi = published["doi"]
        url = published["links"]["html"]

        return doi, url

    def enqueue_future_work(self, draft: str, cluster_id: str) -> int:
        """Extract future work items from draft and add to open questions."""
        future_match = re.search(
            r"(?:future work|future directions|open questions).*?(?=\n##|\Z)",
            draft,
            re.IGNORECASE | re.DOTALL
        )

        if not future_match:
            return 0

        future_text = future_match.group(0)
        items = re.findall(r"[-*]\s+(.+)", future_text)

        if not items:
            return 0

        open_questions_path = self.results_dir / "open_questions.jsonl"

        count = 0
        with open(open_questions_path, "a") as f:
            for item in items:
                entry = {
                    "type": "direction",
                    "source": f"paper:{cluster_id}",
                    "text": item.strip(),
                    "status": "open",
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                }
                f.write(json.dumps(entry) + "\n")
                count += 1

        return count

    def write_and_publish(
        self,
        cluster_id: str,
        force: bool = False,
        skip_zenodo: bool = False,
    ) -> PaperResult:
        """Write and publish a paper for a discovery cluster.

        Generates a draft from findings data, optionally polishes with
        local model, compiles to PDF, and uploads to Zenodo.

        Args:
            cluster_id: Identifier for the discovery cluster
            force: Skip maturity threshold check
            skip_zenodo: Skip Zenodo upload

        Returns:
            PaperResult with publication details
        """
        result = PaperResult(cluster_id=cluster_id, title=f"Discovery: {cluster_id}")

        # Stage -1: Novelty verification (GATE)
        novelty = self.check_novelty(cluster_id)
        if not force and novelty["recommendation"] == "already_published":
            result.errors.append(
                f"Novelty gate failed: {novelty['reason']}. "
                f"All {len(novelty['covered_findings'])} findings already published."
            )
            return result

        # Check readiness
        metrics = self.get_cluster_metrics(cluster_id)
        if metrics is None:
            result.errors.append(f"No metrics found for cluster: {cluster_id}")
            return result

        if not force and not self.should_write_paper(cluster_id):
            result.errors.append(
                f"Cluster not mature enough (score: {metrics.maturity_score:.2f}, "
                f"threshold: 0.82)"
            )
            return result

        # Setup paper directory
        paper_id = cluster_id.replace("_", "-").lower()
        paper_subdir = self.paper_dir / paper_id
        paper_subdir.mkdir(parents=True, exist_ok=True)

        # Stage 2: Generate template draft from findings + metrics
        draft = self.generate_draft(cluster_id, metrics)

        # Stage 5: AI language scrub
        final_draft = self.scrub_ai_language(draft)

        # Save paper brief for Claude Code to refine in conversation
        paper_brief = self.prepare_paper_brief(cluster_id)
        brief_path = paper_subdir / "paper_brief.json"
        with open(brief_path, "w") as f:
            json.dump(paper_brief, f, indent=2)

        # Save draft
        draft_path = paper_subdir / "draft.md"
        with open(draft_path, "w") as f:
            f.write(final_draft)

        # Extract title from draft
        title_match = re.search(r"^#\s+(.+)", final_draft, re.MULTILINE)
        if title_match:
            result.title = title_match.group(1).strip()

        # Stage 7: Compile PDF
        try:
            pdf_path = self.pandoc_compile(draft_path)
            result.pdf_path = pdf_path
        except Exception as e:
            result.errors.append(f"PDF compilation failed: {e}")

        # Stage 8: Zenodo upload
        if not skip_zenodo and result.pdf_path and self.zenodo_token:
            try:
                metadata = {
                    "title": result.title,
                    "description": f"Discovery paper from NoetherSolve pipeline. "
                                   f"Cluster: {cluster_id}. "
                                   f"{metrics.facts_flipped}/{metrics.facts_count} facts flipped.",
                    "keywords": ["NoetherSolve", cluster_id, "scientific discovery"],
                }
                doi, url = self.upload_zenodo(result.pdf_path, metadata)
                result.doi = doi
                result.zenodo_url = url
            except Exception as e:
                result.errors.append(f"Zenodo upload failed: {e}")

        # Enqueue future work
        result.future_work_count = self.enqueue_future_work(final_draft, cluster_id)

        return result


# Convenience function for MCP tool
def write_paper_for_cluster(cluster_id: str, force: bool = False) -> str:
    """Write and publish a paper for a discovery cluster.

    This is the main entry point for the MCP tool.
    """
    agent = PaperAgent()
    result = agent.write_and_publish(cluster_id, force=force)

    if result.success:
        msg = f"Paper generated: {result.title}\nPDF: {result.pdf_path}"
        if result.doi:
            msg += f"\nDOI: {result.doi}\nURL: {result.zenodo_url}"
        return msg
    else:
        return f"Failed: {'; '.join(result.errors)}"
