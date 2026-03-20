#!/usr/bin/env python3
"""
discovery_grader.py — Grade discoveries by publication impact.

We're mining papers. Every discovery gets graded on how close it is
to a high-impact publication, and the orchestrator prioritizes
accordingly.

Grades (highest to lowest):

    NOBEL       — Paradigm shift. Cross-field implications. Would reshape
                  how people think about a fundamental question.
                  (e.g., proving a Millennium Problem, new conservation law
                  family with experimental predictions)

    EUREKA      — Major surprise. Nobody expected this result. Clear,
                  clean, and undeniable. Single-paper Nature/Science material.
                  (e.g., universal 41% constant, contrastive decoding rescuing
                  a phase transition)

    GEM         — Beautiful result. Elegant, surprising, and self-contained.
                  Top venue (ICML/NeurIPS/EMNLP oral, JFM, Phys Rev Letters).
                  (e.g., certainty contamination bias, deconcentration score,
                  resolvent-conservation unification)

    SOLID       — Good science. Publishable at a strong venue. Adds real
                  knowledge but builds on known frameworks.
                  (e.g., orthogonal adapter routing, length ratio discovery,
                  staged training methodology)

    NUGGET      — Interesting finding. Worth documenting in a paper's
                  supplementary or as part of a larger story. Not standalone.
                  (e.g., individual domain oracle results, single adapter
                  flip, negative results like stacking failure)

    ORE         — Raw material. Data collected, pattern noticed, but not
                  yet refined into a clear finding. Needs more work.
                  (e.g., fresh oracle sweep results, untrained domains,
                  borderline margins)

Usage:
    python scripts/discovery_grader.py                    # Grade all discoveries
    python scripts/discovery_grader.py --domain X         # Grade specific domain
    python scripts/discovery_grader.py --ready             # Show paper-ready clusters
    python scripts/discovery_grader.py --mine              # Suggest highest-ROI work
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path

_HERE = Path(__file__).parent.parent
sys.path.insert(0, str(_HERE))

RESULTS = _HERE / "results"
STATUS_FILE = RESULTS / "research_status.json"
GRADES_FILE = RESULTS / "discovery_grades.json"
ESCALATION_FILE = RESULTS / "escalations.jsonl"
CANDIDATES_TSV = RESULTS / "candidates.tsv"

# ---------------------------------------------------------------------------
# Grade definitions
# ---------------------------------------------------------------------------

GRADES = {
    "NOBEL": {
        "rank": 6,
        "color": "#FFD700",
        "emoji": "👑",
        "description": "Paradigm shift — reshapes a field",
        "min_domains": 3,
        "min_facts_flipped": 50,
        "requires_cross_field": True,
        "paper_priority": 100,
    },
    "EUREKA": {
        "rank": 5,
        "color": "#FF6B35",
        "emoji": "⚡",
        "description": "Major surprise — nobody expected this",
        "min_domains": 2,
        "min_facts_flipped": 20,
        "requires_cross_field": False,
        "paper_priority": 80,
    },
    "GEM": {
        "rank": 4,
        "color": "#00D4FF",
        "emoji": "💎",
        "description": "Beautiful result — elegant and self-contained",
        "min_domains": 1,
        "min_facts_flipped": 10,
        "requires_cross_field": False,
        "paper_priority": 60,
    },
    "SOLID": {
        "rank": 3,
        "color": "#4CAF50",
        "emoji": "🪨",
        "description": "Good science — publishable at a strong venue",
        "min_domains": 1,
        "min_facts_flipped": 5,
        "requires_cross_field": False,
        "paper_priority": 40,
    },
    "NUGGET": {
        "rank": 2,
        "color": "#9C27B0",
        "emoji": "🔮",
        "description": "Interesting finding — part of a larger story",
        "min_domains": 1,
        "min_facts_flipped": 1,
        "requires_cross_field": False,
        "paper_priority": 20,
    },
    "ORE": {
        "rank": 1,
        "color": "#6E7681",
        "emoji": "⛏️",
        "description": "Raw material — needs refinement",
        "min_domains": 0,
        "min_facts_flipped": 0,
        "requires_cross_field": False,
        "paper_priority": 5,
    },
}


# ---------------------------------------------------------------------------
# Discovery clusters — group related domains into paper candidates
# ---------------------------------------------------------------------------

# Known discovery clusters from the research so far
# Includes all papers on Zenodo (Papers 1-9 from rho-eval, D1-D6 from NoetherSolve)
DISCOVERY_CLUSTERS = {
    # === rho-eval Papers (ML/interpretability) ===
    # === rho-eval Papers (ML/interpretability) ===
    # stages_complete tracks the paper lifecycle:
    #   discovery, evidence, writing, zenodo, submitted, in_review, revised, accepted, published
    "p1_rho_guided_sft": {
        "title": "Rho-Guided SFT: Post-Training Calibration Repair",
        "domains": [],
        "doi": "10.5281/zenodo.18854943",
        "venue": "ML venue TBD",
        "status": "on_zenodo",
        "grade_override": "GEM",
        "date_written": "2026-03-04",
        "stages_complete": ["discovery", "evidence", "writing", "zenodo"],
    },
    "p2_grassmann_geometry": {
        "title": "Grassmann Geometry of Behavioral Subspaces",
        "domains": [],
        "doi": "10.5281/zenodo.18865861",
        "venue": "ML venue TBD",
        "status": "on_zenodo",
        "grade_override": "GEM",
        "date_written": "2026-03-04",
        "stages_complete": ["discovery", "evidence", "writing", "zenodo"],
    },
    "p3_scale_ladder": {
        "title": "Scale Ladder Phase Transitions: Geometry Precedes Emergence",
        "domains": [],
        "doi": "10.5281/zenodo.18865198",
        "venue": "ML venue TBD",
        "status": "on_zenodo",
        "grade_override": "EUREKA",
        "date_written": "2026-03-04",
        "stages_complete": ["discovery", "evidence", "writing", "zenodo"],
    },
    "p4_confidence_cartography": {
        "title": "Confidence Cartography: Teacher-Forced Probability as False-Belief Sensor",
        "domains": [],
        "doi": "10.5281/zenodo.18703505",
        "venue": "ML venue TBD",
        "status": "on_zenodo",
        "grade_override": "GEM",
        "date_written": "2026-03-04",
        "stages_complete": ["discovery", "evidence", "writing", "zenodo"],
    },
    "p5_cf90": {
        "title": "CF90: Knowledge-Preserving SVD Compression",
        "domains": [],
        "doi": "10.5281/zenodo.18718545",
        "venue": "ML venue TBD",
        "status": "on_zenodo",
        "grade_override": "SOLID",
        "date_written": "2026-03-04",
        "stages_complete": ["discovery", "evidence", "writing", "zenodo"],
    },
    "p6_contrastive_pretraining": {
        "title": "Contrastive Pretraining: 5% Injection Breaks the Behavioral Wall",
        "domains": [],
        "doi": "10.5281/zenodo.18870555",
        "venue": "ML venue TBD",
        "status": "on_zenodo",
        "grade_override": "EUREKA",
        "date_written": "2026-03-04",
        "stages_complete": ["discovery", "evidence", "writing", "zenodo"],
    },
    "p7_expression_bottleneck": {
        "title": "The Expression Bottleneck: 41% Universal Constant",
        "domains": [],
        "doi": "10.5281/zenodo.18895248",
        "venue": "ML venue TBD",
        "status": "on_zenodo",
        "grade_override": "EUREKA",
        "date_written": "2026-03-06",
        "stages_complete": ["discovery", "evidence", "writing", "zenodo"],
    },
    "p8_snap_on": {
        "title": "Snap-On Communication Modules: Frozen Logit-Space Adapters",
        "domains": [],
        "doi": "10.5281/zenodo.18902616",
        "venue": "ML venue TBD",
        "status": "on_zenodo",
        "grade_override": "GEM",
        "date_written": "2026-03-07",
        "stages_complete": ["discovery", "evidence", "writing", "zenodo"],
    },
    "p9_stem_oracle": {
        "title": "STEM Truth Oracle: Log-Prob MC Ranking Reveals Scale-Invariant Biases",
        "domains": [],
        "doi": "10.5281/zenodo.19005729",
        "venue": "ML venue TBD",
        "status": "on_zenodo",
        "grade_override": "GEM",
        "date_written": "2026-03-13",
        "stages_complete": ["discovery", "evidence", "writing", "zenodo"],
    },

    # === NoetherSolve Discovery Papers (D1-D6) ===
    "d1_vortex_conservation": {
        "title": "Approximate Conservation Laws in Point Vortex Dynamics",
        "domains": [
            "reduced_navier_stokes_vortex_conservation",
            "reduced_navier_stokes_vortex_conservation_unsolved",
            "reduced_navier_stokes_vortex_conservation_unsolved_v2",
            "Continuous Q_f and Euler Conservation Laws",
            "Q_f Ratio Invariant",
        ],
        "doi": "10.5281/zenodo.19055338",
        "venue": "Journal of Fluid Mechanics",
        "status": "submitted",
        "grade_override": "EUREKA",
        "date_written": "2026-03-14",
        "stages_complete": ["discovery", "evidence", "writing", "zenodo", "submitted"],
    },
    "d2_z3_cancellation": {
        "title": "Z₃ Phase Cancellation in Choreographic Orbits",
        "domains": ["3body_conservation"],
        "doi": "10.5281/zenodo.19055580",
        "venue": "Celestial Mechanics and Dynamical Astronomy",
        "status": "on_zenodo",
        "grade_override": "GEM",
        "date_written": "2026-03-14",
        "stages_complete": ["discovery", "evidence", "writing", "zenodo"],
    },
    "d3_llm_knowledge_gaps": {
        "title": "Where LLMs Are Confidently Wrong: 1038 Facts Across 67 Domains",
        "domains": [
            "knot_invariants", "analysis_pde_conjectures",
            "computational_conjectures", "Intersection Theory",
            "Hamiltonian Mechanics Invariants",
            "NS Regularity and Stretch-Resistant Q_f",
        ],
        "doi": "10.5281/zenodo.19055582",
        "venue": "Nature Machine Intelligence",
        "status": "on_zenodo",
        "grade_override": "GEM",
        "date_written": "2026-03-16",
        "stages_complete": ["discovery", "evidence", "writing", "zenodo"],
    },
    "d4_orthogonal_routing": {
        "title": "Orthogonal Adapter Routing for Interference-Free Knowledge Injection",
        "domains": ["Kinetic Invariant K"],
        "doi": "10.5281/zenodo.19055588",
        "venue": "EMNLP",
        "status": "on_zenodo",
        "grade_override": "SOLID",
        "date_written": "2026-03-16",
        "stages_complete": ["discovery", "evidence", "writing", "zenodo"],
    },
    "d5_certainty_contamination": {
        "title": "Certainty Contamination: How Definitive Language Biases LLM Factual Judgments",
        "domains": ["climate_science_frontiers", "black_hole_frontiers"],
        "doi": "10.5281/zenodo.19068373",
        "venue": "EMNLP",
        "status": "on_zenodo",
        "grade_override": "GEM",
        "date_written": "2026-03-17",
        "stages_complete": ["discovery", "evidence", "writing", "zenodo"],
    },
    "d6_resolvent_unification": {
        "title": "Resolvent-Conservation Unification: Spectral Theory of Approximate Invariants",
        "domains": ["Continuous Q_f and Euler Conservation Laws"],
        "doi": "10.5281/zenodo.19071198",
        "venue": "Journal of Mathematical Physics",
        "status": "on_zenodo",
        "grade_override": "EUREKA",
        "date_written": "2026-03-17",
        "stages_complete": ["discovery", "evidence", "writing", "zenodo"],
    },

    "d7_oracle_biases": {
        "title": "Nine Systematic Biases in Log-Probability LLM Evaluation",
        "domains": ["oracle_methodology"],
        "doi": "10.5281/zenodo.19124851",
        "venue": "EMNLP",
        "status": "on_zenodo",
        "grade_override": "GEM",
        "date_written": "2026-03-19",
        "stages_complete": ["discovery", "evidence", "writing", "zenodo"],
    },

    "d8_cross_domain_conservation": {
        "title": "Unified Cycle Theory: Conservation Laws Across Physical Domains",
        "domains": ["cross_domain"],
        "doi": "10.5281/zenodo.19124858",
        "venue": "Journal of Mathematical Physics",
        "status": "on_zenodo",
        "grade_override": "GEM",
        "date_written": "2026-03-20",
        "stages_complete": ["discovery", "evidence", "writing", "zenodo"],
    },

    # === New potential papers from the 27B sweep ===
    "bio_ai_parallels": {
        "title": "Computational Parallels Between Biological and Artificial Intelligence",
        "domains": ["Bio-AI Computational Parallels"],
        "doi": None,
        "venue": None,
        "status": "ore",
    },
    "clinical_translation": {
        "title": "Clinical Translation Knowledge Gaps in Language Models",
        "domains": ["clinical_translation", "disease_targets",
                     "delivery_optimization", "genetics_therapeutics"],
        "doi": None,
        "venue": None,
        "status": "ore",
    },
    "27b_knowledge_map": {
        "title": "Knowledge Cartography: What a 27B Model Knows and Doesn't",
        "domains": [],  # Will be populated from sweep results
        "doi": None,
        "venue": None,
        "status": "ore",
    },

    # === D9: Cross-domain equivalences ===
    "d9_cross_domain_equivalences": {
        "title": "Five Cross-Domain Mathematical Equivalences as LLM Blind Spots",
        "domains": ["cross_domain"],
        "doi": None,
        "venue": None,
        "status": "ore",
        "grade_override": "GEM",
        "date_written": "2026-03-17",
        "stages_complete": ["discovery", "evidence", "writing"],
    },

    # === D10: Bio-AI convergence ===
    "d10_bio_ai_convergence": {
        "title": "Convergent Computation: Bio-AI Parallels",
        "domains": ["Bio-AI Computational Parallels"],
        "doi": None,
        "venue": "PLOS Computational Biology",
        "status": "ore",
        "grade_override": "SOLID",
        "date_written": "2026-03-20",
        "stages_complete": ["discovery", "evidence", "writing"],
    },

    # === Applied science papers ===
    "catalyst_discovery_her": {
        "title": "Automated Catalyst Prescreening for HER",
        "domains": ["catalyst_discovery"],
        "doi": None,
        "venue": None,
        "status": "ore",
        "grade_override": "SOLID",
        "date_written": "2026-03-20",
        "stages_complete": ["discovery", "evidence", "writing"],
    },
    "epidemiology_dynamics": {
        "title": "Verified Epidemic Dynamics Calculations",
        "domains": ["epidemiology"],
        "doi": None,
        "venue": None,
        "status": "ore",
        "grade_override": "SOLID",
        "date_written": "2026-03-20",
        "stages_complete": ["discovery", "evidence", "writing"],
    },
    "origin_of_life": {
        "title": "Quantitative Abiogenesis: Prebiotic Pathway Plausibility",
        "domains": ["origin_of_life"],
        "doi": None,
        "venue": None,
        "status": "ore",
        "grade_override": "SOLID",
        "date_written": "2026-03-20",
        "stages_complete": ["discovery", "evidence", "writing"],
    },
    "topological_materials": {
        "title": "Topological Materials Classification via Berry Phase and Z₂ Invariants",
        "domains": ["topological_phases"],
        "doi": None,
        "venue": None,
        "status": "ore",
        "grade_override": "SOLID",
        "date_written": "2026-03-20",
        "stages_complete": ["discovery", "evidence", "writing"],
    },
    "climate_sensitivity": {
        "title": "Climate Sensitivity Across Emission Scenarios and Feedback Profiles",
        "domains": ["climate_science"],
        "doi": None,
        "venue": None,
        "status": "ore",
        "grade_override": "SOLID",
        "date_written": "2026-03-20",
        "stages_complete": ["discovery", "evidence", "writing"],
    },
    "genetic_therapeutics": {
        "title": "Genetic Therapeutic Design Pipeline: CRISPR, mRNA, Neoantigen, Antibody",
        "domains": ["genetics_therapeutics"],
        "doi": None,
        "venue": None,
        "status": "ore",
        "grade_override": "SOLID",
        "date_written": "2026-03-20",
        "stages_complete": ["discovery", "evidence", "writing"],
    },
    "drug_therapy_screening": {
        "title": "Pharmacokinetic Screening and DDI Assessment",
        "domains": ["pharmacokinetics", "drug_interactions"],
        "doi": None,
        "venue": None,
        "status": "ore",
        "grade_override": "SOLID",
        "date_written": "2026-03-20",
        "stages_complete": ["discovery", "evidence", "writing"],
    },
    "battery_materials": {
        "title": "Battery Materials Screening: LFP vs NCA vs NMC",
        "domains": ["battery_technology"],
        "doi": None,
        "venue": None,
        "status": "ore",
        "grade_override": "SOLID",
        "date_written": "2026-03-20",
        "stages_complete": ["discovery", "evidence", "writing"],
    },
}


# Paper lifecycle stages — the full development pathway
# Each stage has a weight toward the 100% readiness score
PAPER_STAGES = {
    "discovery":   {"weight": 10, "label": "Discovery",       "desc": "Raw results exist"},
    "evidence":    {"weight": 15, "label": "Evidence",         "desc": "Strong experimental backing"},
    "writing":     {"weight": 15, "label": "Writing",          "desc": "Draft written and reviewed"},
    "zenodo":      {"weight": 10, "label": "On Zenodo",        "desc": "Preprint deposited, DOI minted"},
    "submitted":   {"weight": 15, "label": "Submitted",        "desc": "Submitted to journal/venue"},
    "in_review":   {"weight": 10, "label": "In Review",        "desc": "Under peer review"},
    "revised":     {"weight": 10, "label": "Revised",          "desc": "Revisions addressed"},
    "accepted":    {"weight": 10, "label": "Accepted",         "desc": "Accepted for publication"},
    "published":   {"weight": 5,  "label": "Published",        "desc": "Appears in venue"},
}
# Total weights = 100


@dataclass
class GradedDiscovery:
    cluster_id: str
    title: str
    grade: str
    grade_rank: int
    paper_priority: float
    domains: list
    n_domains_passing: int = 0
    n_domains_failing: int = 0
    n_facts_flipped: int = 0
    mean_margin: float = 0.0
    has_doi: bool = False
    status: str = "ore"
    paper_readiness: float = 0.0  # 0-100%
    current_stage: str = ""       # Current lifecycle stage
    stages_complete: list = field(default_factory=list)  # Which stages are done
    bottleneck: str = ""
    next_action: str = ""


def load_research_status() -> dict:
    if STATUS_FILE.exists():
        with open(STATUS_FILE) as f:
            return json.load(f)
    return {}


def count_flipped_facts(domains: list) -> int:
    """Count facts flipped across domains from candidates.tsv."""
    if not CANDIDATES_TSV.exists():
        return 0
    count = 0
    with open(CANDIDATES_TSV) as f:
        for line in f:
            upper = line.upper()
            if "FLIPPED" in upper or "DUAL-PASS" in upper:
                # Check if any domain keyword appears in the line
                for d in domains:
                    if d.lower().replace(" ", "_") in line.lower():
                        count += 1
                        break
    return count


def grade_cluster(cluster_id: str, cluster: dict, research: dict) -> GradedDiscovery:
    """Grade a discovery cluster based on its current state."""
    results = research.get("domain_results", {})

    domains = cluster["domains"]
    passing = 0
    failing = 0
    margins = []

    for d in domains:
        r = results.get(d, {})
        if r.get("verdict") == "PASS":
            passing += 1
        elif r:
            failing += 1
        if "mean_margin" in r:
            margins.append(r["mean_margin"])

    mean_margin = sum(margins) / len(margins) if margins else -999
    n_flipped = count_flipped_facts(domains)
    has_doi = cluster.get("doi") is not None
    status = cluster.get("status", "ore")

    # Use manual grade override if set (for published papers with known impact)
    grade_override = cluster.get("grade_override")

    # Auto-grade based on metrics
    if grade_override:
        grade = grade_override
    elif has_doi and status == "published":
        if len(domains) >= 3 and n_flipped >= 50:
            grade = "EUREKA"
        elif n_flipped >= 10:
            grade = "GEM"
        else:
            grade = "SOLID"
    elif passing >= 3 and mean_margin > 5:
        grade = "GEM"
    elif passing >= 1 and mean_margin > 0:
        grade = "SOLID"
    elif n_flipped >= 1:
        grade = "NUGGET"
    else:
        grade = "ORE"

    # Paper readiness score — tracks full lifecycle
    # Each completed stage contributes its weight toward 100%
    stages_complete = cluster.get("stages_complete", [])

    # Auto-detect early stages from experimental data if not manually set
    if not stages_complete:
        if passing > 0 or n_flipped > 0:
            stages_complete.append("discovery")
        if passing >= len(domains) * 0.5 and mean_margin > 0:
            stages_complete.append("evidence")
        if has_doi:
            stages_complete.extend(["writing", "zenodo"])

    readiness = sum(
        PAPER_STAGES[s]["weight"] for s in stages_complete if s in PAPER_STAGES
    )

    # Identify current stage and next action
    stage_order = list(PAPER_STAGES.keys())
    current_stage = stages_complete[-1] if stages_complete else ""
    next_stage_idx = stage_order.index(current_stage) + 1 if current_stage in stage_order else 0
    next_stage = stage_order[next_stage_idx] if next_stage_idx < len(stage_order) else ""

    bottleneck = PAPER_STAGES[next_stage]["desc"] if next_stage else "complete"
    next_action_map = {
        "discovery": "run experiments, identify novel findings",
        "evidence": "strengthen results, run ablations, add baselines",
        "writing": "write draft, run self-critique, scrub AI language",
        "zenodo": "deposit preprint on Zenodo",
        "submitted": "submit to target venue",
        "in_review": "waiting for reviewer feedback",
        "revised": "address reviewer comments, resubmit",
        "accepted": "waiting for publication",
        "published": "paper is published in venue",
    }
    next_action = next_action_map.get(next_stage, "done")

    grade_info = GRADES[grade]
    return GradedDiscovery(
        cluster_id=cluster_id,
        title=cluster["title"],
        grade=grade,
        grade_rank=grade_info["rank"],
        paper_priority=grade_info["paper_priority"] * (readiness / 100),
        domains=domains,
        n_domains_passing=passing,
        n_domains_failing=failing,
        n_facts_flipped=n_flipped,
        mean_margin=mean_margin,
        has_doi=has_doi,
        status=status,
        paper_readiness=readiness,
        current_stage=current_stage,
        stages_complete=stages_complete,
        bottleneck=bottleneck,
        next_action=next_action,
    )


def grade_all() -> list[GradedDiscovery]:
    """Grade all discovery clusters."""
    research = load_research_status()

    # Populate the 27B knowledge map cluster with all domains
    results = research.get("domain_results", {})
    if results:
        all_domain_names = list(results.keys())
        passing_names = [n for n, r in results.items() if r.get("verdict") == "PASS"]
        failing_names = [n for n, r in results.items() if r.get("verdict") != "PASS"]
        DISCOVERY_CLUSTERS["27b_knowledge_map"]["domains"] = all_domain_names

    # Also find unclustered domains that could form new papers
    clustered = set()
    for c in DISCOVERY_CLUSTERS.values():
        clustered.update(c["domains"])

    unclustered_failing = []
    for name, r in results.items():
        if name not in clustered and r.get("verdict") != "PASS":
            unclustered_failing.append(name)

    if unclustered_failing:
        # Group unclustered failures as potential new paper material
        DISCOVERY_CLUSTERS["unclustered_gaps"] = {
            "title": f"Unclustered Knowledge Gaps ({len(unclustered_failing)} domains)",
            "domains": unclustered_failing,
            "doi": None,
            "venue": None,
            "status": "ore",
        }

    graded = []
    for cluster_id, cluster in DISCOVERY_CLUSTERS.items():
        g = grade_cluster(cluster_id, cluster, research)
        graded.append(g)

    # Sort by paper priority (highest first)
    graded.sort(key=lambda g: g.paper_priority, reverse=True)
    return graded


def save_grades(graded: list[GradedDiscovery]):
    """Save grades to JSON for the dashboard."""
    data = {
        "timestamp": datetime.now().isoformat(),
        "grades": [asdict(g) for g in graded],
        "summary": {
            "total_clusters": len(graded),
            "by_grade": {},
            "paper_ready": sum(1 for g in graded if g.paper_readiness >= 80),
            "in_progress": sum(1 for g in graded if 20 < g.paper_readiness < 80),
            "raw_ore": sum(1 for g in graded if g.paper_readiness <= 20),
        },
    }
    for grade_name in GRADES:
        data["summary"]["by_grade"][grade_name] = sum(
            1 for g in graded if g.grade == grade_name
        )
    with open(GRADES_FILE, "w") as f:
        json.dump(data, f, indent=2)


def get_mining_recommendations(graded: list[GradedDiscovery]) -> list[dict]:
    """
    Recommend where to spend GPU time for maximum paper output.

    Like Amelia's portfolio allocation: invest in positions with
    the best risk-adjusted return. Here, "return" = paper output
    and "risk" = GPU time that might not yield results.
    """
    recommendations = []

    for g in graded:
        if g.status == "published":
            continue  # Already mined

        # ROI estimate: paper_priority / estimated_gpu_hours
        if g.paper_readiness >= 60:
            # Close to paper — high ROI, low cost
            est_hours = 2
            roi = g.paper_priority / est_hours
            action = f"WRITE PAPER: {g.next_action}"
        elif g.paper_readiness >= 30:
            # Midway — moderate ROI
            est_hours = 8
            roi = g.paper_priority / est_hours
            action = f"CONTINUE: {g.next_action}"
        else:
            # Raw ore — uncertain ROI
            est_hours = 20
            roi = g.paper_priority / est_hours
            action = f"EXPLORE: {g.next_action}"

        recommendations.append({
            "cluster": g.cluster_id,
            "title": g.title,
            "grade": g.grade,
            "readiness": g.paper_readiness,
            "roi": roi,
            "est_hours": est_hours,
            "action": action,
        })

    # Sort by ROI
    recommendations.sort(key=lambda r: r["roi"], reverse=True)
    return recommendations


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def print_grades(graded: list[GradedDiscovery]):
    print(f"\n{'='*75}")
    print(f"  NoetherSolve Discovery Grades")
    print(f"{'='*75}\n")

    for g in graded:
        info = GRADES[g.grade]
        bar_len = int(g.paper_readiness / 5)
        bar = "█" * bar_len + "░" * (20 - bar_len)

        status_str = f"[{g.status}]" if g.status != "ore" else ""
        doi_str = " DOI" if g.has_doi else ""

        print(f"  {info['emoji']}  {g.grade:<8} {g.title[:50]:<50} {status_str}{doi_str}")
        print(f"     Readiness: [{bar}] {g.paper_readiness:.0f}%")
        print(f"     Domains: {g.n_domains_passing} pass / {g.n_domains_failing} fail"
              f" | Facts flipped: {g.n_facts_flipped}"
              f" | Margin: {g.mean_margin:+.1f}")
        if g.bottleneck and g.status != "published":
            print(f"     Bottleneck: {g.bottleneck}")
            print(f"     Next: {g.next_action}")
        print()


def print_mining_plan(recommendations: list[dict]):
    print(f"\n{'='*75}")
    print(f"  Mining Plan — Highest ROI First")
    print(f"{'='*75}\n")

    print(f"  {'#':<3} {'Grade':<8} {'ROI':<6} {'Hours':<6} {'Ready':<7} Action")
    print(f"  {'-'*3} {'-'*8} {'-'*6} {'-'*6} {'-'*7} {'-'*40}")

    for i, r in enumerate(recommendations[:10], 1):
        print(f"  {i:<3} {r['grade']:<8} {r['roi']:>5.1f} {r['est_hours']:>4}h"
              f"   {r['readiness']:>4.0f}%  {r['action'][:50]}")

    print()


def main():
    parser = argparse.ArgumentParser(description="Grade NoetherSolve discoveries")
    parser.add_argument("--domain", default=None, help="Grade specific domain")
    parser.add_argument("--ready", action="store_true", help="Show paper-ready clusters")
    parser.add_argument("--mine", action="store_true", help="Show highest-ROI work")
    parser.add_argument("--json", action="store_true", help="JSON output")
    args = parser.parse_args()

    graded = grade_all()
    save_grades(graded)

    if args.json:
        print(json.dumps([asdict(g) for g in graded], indent=2))
        return

    if args.ready:
        ready = [g for g in graded if g.paper_readiness >= 60]
        if ready:
            print(f"\n  Paper-ready clusters ({len(ready)}):\n")
            for g in ready:
                info = GRADES[g.grade]
                print(f"  {info['emoji']}  {g.grade:<8} {g.title}")
                print(f"     Readiness: {g.paper_readiness:.0f}% | Next: {g.next_action}")
                print()
        else:
            print("\n  No clusters at ≥60% readiness yet.\n")
        return

    if args.mine:
        recommendations = get_mining_recommendations(graded)
        print_mining_plan(recommendations)
        return

    print_grades(graded)
    recommendations = get_mining_recommendations(graded)
    print_mining_plan(recommendations)


if __name__ == "__main__":
    main()
