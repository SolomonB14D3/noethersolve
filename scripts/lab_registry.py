#!/usr/bin/env python3
"""
lab_registry.py — Registry of all active labs and their status.

Each lab is a computational pipeline that uses NoetherSolve MCP tools
to screen candidates, run experiments, and produce results.

The dashboard reads this to populate the Labs section.
"""
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path

_HERE = Path(__file__).parent.parent
RESULTS = _HERE / "results"
LABS_DIR = RESULTS / "labs"
REGISTRY_FILE = RESULTS / "lab_registry.json"


@dataclass
class LabProject:
    lab_id: str
    title: str
    status: str  # idea, design, prototyping, running, producing, paper_feed
    n_tools: int = 0
    n_candidates_screened: int = 0
    n_candidates_viable: int = 0
    results_dir: str = ""
    description: str = ""


# ---------------------------------------------------------------------------
# Lab definitions — one per computational lab
# ---------------------------------------------------------------------------

LAB_DEFINITIONS = {
    "drug_therapy": {
        "title": "Drug Therapy Screening",
        "description": "PK/PD screening pipeline: half-life, therapeutic index, CYP interactions, DDI risk",
        "n_tools": 12,
    },
    "genetic_therapeutics": {
        "title": "Genetic Therapeutics Pipeline",
        "description": "CRISPR guide scoring, mRNA optimization, neoantigen evaluation, antibody developability",
        "n_tools": 18,
    },
    "catalyst_discovery": {
        "title": "Catalyst Discovery (HER)",
        "description": "DFT-informed catalyst prescreening: d-band center, volcano position, BEP activation energy",
        "n_tools": 8,
    },
    "climate_analysis": {
        "title": "Climate Sensitivity Analysis",
        "description": "CO2 forcing, feedback profiles, equilibrium climate sensitivity across emission scenarios",
        "n_tools": 6,
    },
    "epidemiology": {
        "title": "Epidemic Dynamics",
        "description": "SIR modeling, R0 estimation, herd immunity thresholds, doubling times, attack rates",
        "n_tools": 8,
    },
    "topological_materials": {
        "title": "Topological Materials Classification",
        "description": "Berry phase, Chern number, Z2 invariant, AZ class, bulk-boundary correspondence",
        "n_tools": 6,
    },
    "conservation_mining": {
        "title": "Conservation Law Mining",
        "description": "Automated discovery of approximate conservation laws in dynamical systems",
        "n_tools": 5,
    },
    "bio_ai": {
        "title": "Bio-AI Convergence",
        "description": "Systematic verification of computational parallels between biological and artificial systems",
        "n_tools": 8,
    },
    "origin_of_life": {
        "title": "Abiogenesis & Prebiotic Chemistry",
        "description": "Miller-Urey yield, autocatalytic set verification, prebiotic plausibility scoring",
        "n_tools": 5,
    },
    "battery_materials": {
        "title": "Battery Materials Screening",
        "description": "LFP vs NCA vs NMC: cycle aging, calendar aging, total degradation comparison",
        "n_tools": 4,
    },
    "quantum_mechanics": {
        "title": "Quantum Mechanics Verification",
        "description": "Particle-in-box, hydrogen energy levels, tunneling, uncertainty, harmonic oscillator",
        "n_tools": 7,
    },
    "behavioral_economics": {
        "title": "Behavioral Economics & Decision Theory",
        "description": "Prospect theory, loss aversion, Allais paradox, framing effects, temporal discounting",
        "n_tools": 7,
    },
    "ai_safety": {
        "title": "AI Safety Quantitative Evaluation",
        "description": "Adversarial robustness, reward hacking risk, oversight coverage, corrigibility checks",
        "n_tools": 7,
    },
    "supply_chain": {
        "title": "Supply Chain Optimization",
        "description": "EOQ, safety stock, newsvendor, vehicle routing, bin packing",
        "n_tools": 5,
    },
}


def _count_results(lab_dir: Path) -> tuple[int, int]:
    """Count total screened candidates and viable ones from JSON results."""
    if not lab_dir.exists():
        return 0, 0
    total = 0
    viable = 0
    for jf in lab_dir.glob("*.json"):
        try:
            with open(jf) as f:
                data = json.load(f)
        except Exception:
            continue

        # Collect all result items from various formats
        items = []
        for key in ("results", "scenarios", "top_candidates",
                     "crispr", "mrna", "neoantigens", "molecules",
                     "inventory", "perishable"):
            v = data.get(key, [])
            if isinstance(v, list):
                items.extend(v)

        total += len(items)
        for r in items:
            v = r.get("verdict", r.get("phase", ""))
            if v in ("PASS", "TOP", "VIABLE", "CONVERGENT", "topological",
                     "controlled"):
                viable += 1
            elif r.get("pipeline_pass") or r.get("herd_achieved"):
                viable += 1
            elif r.get("novel") or r.get("classification") == "novel_approximate":
                viable += 1
    return total, viable


def grade_all_labs() -> list[LabProject]:
    """Grade all labs and return LabProject list."""
    labs = []
    for lab_id, defn in LAB_DEFINITIONS.items():
        lab_dir = LABS_DIR / lab_id
        results_dir = f"results/labs/{lab_id}"

        n_screened, n_viable = _count_results(lab_dir)

        # Determine status from results
        if n_screened == 0:
            status = "prototyping"
        elif n_viable == 0:
            status = "running"
        else:
            status = "producing"

        labs.append(LabProject(
            lab_id=lab_id,
            title=defn["title"],
            status=status,
            n_tools=defn.get("n_tools", 0),
            n_candidates_screened=n_screened,
            n_candidates_viable=n_viable,
            results_dir=results_dir,
            description=defn.get("description", ""),
        ))

    # Sort: producing first, then running, then prototyping
    status_order = {"producing": 0, "running": 1, "prototyping": 2,
                    "design": 3, "idea": 4}
    labs.sort(key=lambda l: (status_order.get(l.status, 5), -l.n_candidates_screened))
    return labs


def save_lab_registry(labs: list[LabProject]):
    """Save graded labs to JSON."""
    REGISTRY_FILE.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "timestamp": datetime.now().isoformat(),
        "n_labs": len(labs),
        "labs": [asdict(l) for l in labs],
    }
    with open(REGISTRY_FILE, "w") as f:
        json.dump(data, f, indent=2)


if __name__ == "__main__":
    labs = grade_all_labs()
    save_lab_registry(labs)
    print(f"Graded {len(labs)} labs:")
    for l in labs:
        print(f"  [{l.status:10s}] {l.title:40s} "
              f"{l.n_candidates_viable}/{l.n_candidates_screened} viable, "
              f"{l.n_tools} tools")
