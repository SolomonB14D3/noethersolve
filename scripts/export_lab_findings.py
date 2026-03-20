#!/usr/bin/env python3
"""
export_lab_findings.py — Convert lab result JSONs to novel_findings .md files.

The paper_watchdog routes findings from results/discoveries/novel_findings/*.md.
Labs save JSON to results/labs/*/. This script bridges that gap.

Run after any lab completes, or run --all to back-fill all labs.

Usage:
    python scripts/export_lab_findings.py --all          # all labs
    python scripts/export_lab_findings.py --lab bio_ai   # specific lab
    python scripts/export_lab_findings.py --list         # show status
"""
import argparse
import json
from datetime import datetime
from pathlib import Path

PROJECT = Path(__file__).parent.parent
LABS_DIR = PROJECT / "results" / "labs"
FINDINGS_DIR = PROJECT / "results" / "discoveries" / "novel_findings"
FINDINGS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Per-lab converters — each returns (filename_stem, md_content)
# ---------------------------------------------------------------------------

def convert_bio_ai(data: dict) -> tuple[str, str]:
    scenarios = data.get("scenarios", [])
    n = len(scenarios)
    mean_score = data.get("mean_convergence_score", 0)
    convergent = sum(1 for s in scenarios if s.get("verdict") == "CONVERGENT")
    known = data.get("known_convergent_solutions", [])

    lines = [
        f"# Bio-AI Computational Convergence: {convergent}/{n} Scenarios Verified",
        "",
        "## Discovery Summary",
        "",
        f"Systematic verification of computational parallels between biological neural circuits "
        f"and artificial architectures. Mean convergence score: **{mean_score:.3f}** across {n} scenarios.",
        "",
        "## Key Findings",
        "",
    ]
    for s in scenarios:
        score = s.get("convergence_score", 0)
        verdict = s.get("verdict", "?")
        name = s.get("name", "?").replace("_", " ").title()
        lines.append(f"### {name} (Score: {score:.3f} — {verdict})")
        lines.append("")
        for f in s.get("findings", []):
            lines.append(f"- {f}")
        lines.append("")

    if known:
        lines += ["## Known Convergent Solutions (Historical)", ""]
        lines.append("| Domain | Biological | Algorithm | Conservation Score |")
        lines.append("|--------|-----------|-----------|-------------------|")
        for k in known:
            lines.append(f"| {k.get('domain','')} | {k.get('biological','')} | "
                         f"{k.get('algorithmic','')} | {k.get('conservation_score',0):.2f} |")
        lines.append("")

    lines += [
        "## Implications",
        "",
        "Convergence between evolution and gradient descent on the same computational solution "
        "indicates fundamental constraints — when the optimization landscape has a unique efficient "
        "solution under shared constraints (local info, energy budget, sparse rewards), both searches "
        "find it independently.",
        "",
        f"## Date Discovered",
        data.get("timestamp", "")[:10],
        "",
        "## Tools Used",
        "NoetherSolve Bio-AI Bridge module",
    ]
    return "bio_ai_computational_convergence", "\n".join(lines)


def convert_conservation_mining(data: dict) -> tuple[str, str]:
    n_novel = data.get("n_novel_approximate", 0)
    n_exact = data.get("n_exact_known", 0)
    n_total = data.get("n_candidates_total", 0)
    tops = data.get("top_candidates", [])[:8]

    lines = [
        f"# Conservation Law Mining: {n_novel} Novel Approximate Invariants Discovered",
        "",
        "## Discovery Summary",
        "",
        f"Automated sweep of {n_total} candidate expressions across {data.get('n_systems', 0)} "
        f"dynamical systems. Found **{n_novel} novel approximate** and {n_exact} exact known invariants.",
        "",
        "## Top Candidates",
        "",
        "| System | Expression | frac_var | Class | Novel |",
        "|--------|-----------|---------|-------|-------|",
    ]
    for t in tops:
        novel = "YES" if not t.get("known") else "no"
        lines.append(f"| {t.get('system','')} | {t.get('name','')} | "
                     f"{t.get('frac_var',0):.2e} | {t.get('classification','')} | {novel} |")
    lines += [
        "",
        f"## Date Discovered",
        data.get("timestamp", "")[:10],
        "",
        "## Tools Used",
        "NoetherSolve conservation checker, vortex_checker, RK45 integrator",
    ]
    return "conservation_law_mining_results", "\n".join(lines)


def convert_ai_safety(data: dict) -> tuple[str, str]:
    sections = ["reward_hacking", "calibration", "corrigibility", "oversight", "robustness", "alignment"]
    lines = [
        "# AI Safety Evaluation: Quantitative Metrics Across Six Dimensions",
        "",
        "## Discovery Summary",
        "",
        "Numerical evaluation of AI safety properties using NoetherSolve tools.",
        "",
        "## Results by Dimension",
        "",
    ]
    for sec in sections:
        items = data.get(sec, [])
        if not items:
            continue
        # Handle both list and dict
        if isinstance(items, dict):
            items = [items]
        lines.append(f"### {sec.replace('_', ' ').title()}")
        lines.append("")
        for item in items[:4]:
            name = item.get("name", "?")
            # Pick the most informative numeric field
            score_key = next((k for k in ["risk_score", "corrigibility_score", "coverage",
                              "ece", "clean_accuracy", "agreement_rate"] if k in item), None)
            score_str = f" — {item[score_key]:.3f}" if score_key else ""
            verdict = item.get("verdict", item.get("is_safe", ""))
            lines.append(f"- **{name}**{score_str} {verdict}")
        lines.append("")
    lines += [
        f"## Date Discovered",
        data.get("timestamp", "")[:10],
        "",
        "## Tools Used",
        "NoetherSolve autonomy/safety tools",
    ]
    return "ai_safety_quantitative_evaluation", "\n".join(lines)


def convert_battery_materials(data: dict) -> tuple[str, str]:
    results = data.get("results", [])
    n_pass = sum(1 for r in results if r.get("verdict") in ("EXCELLENT", "GOOD"))
    lines = [
        f"# Battery Materials Analysis: {n_pass}/{len(results)} Candidates Viable",
        "",
        "## Discovery Summary",
        "",
        f"Screening {len(results)} battery material candidates across aging, capacity, and safety metrics.",
        "",
        "## Top Candidates",
        "",
    ]
    for r in sorted(results, key=lambda x: x.get("total_score", 0), reverse=True)[:6]:
        lines.append(f"- **{r.get('name','?')}** ({r.get('chemistry','?')}): "
                     f"score={r.get('total_score',0):.1f}, verdict={r.get('verdict','?')}")
    lines += [
        "",
        f"## Date Discovered",
        data.get("timestamp", "")[:10],
        "",
        "## Tools Used",
        "NoetherSolve battery aging/capacity tools",
    ]
    return "battery_materials_screening", "\n".join(lines)


def convert_behavioral_economics(data: dict) -> tuple[str, str]:
    results = data.get("results", [])
    lines = [
        "# Behavioral Economics: Quantified Cognitive Bias Magnitudes",
        "",
        "## Discovery Summary",
        "",
        "Numerical measurement of cognitive biases using NoetherSolve decision theory tools.",
        "",
        "## Key Findings",
        "",
    ]
    for r in results[:8]:
        name = r.get("bias_name", r.get("name", "?"))
        magnitude = r.get("magnitude", r.get("effect_size", None))
        finding = r.get("finding", r.get("description", ""))
        if magnitude is not None:
            lines.append(f"- **{name}**: magnitude={magnitude:.3f} — {finding}")
        else:
            lines.append(f"- **{name}**: {finding}")
    lines += [
        "",
        f"## Date Discovered",
        data.get("timestamp", "")[:10],
        "",
        "## Tools Used",
        "NoetherSolve prospect theory, loss aversion, framing effect tools",
    ]
    return "behavioral_economics_bias_magnitudes", "\n".join(lines)


def convert_catalyst(data: dict) -> tuple[str, str]:
    results = data.get("results", [])
    reaction = data.get("reaction", "unknown reaction")
    n_top = sum(1 for r in results if r.get("verdict") == "TOP")
    lines = [
        f"# Catalyst Discovery: {n_top} Top Candidates for {reaction}",
        "",
        "## Discovery Summary",
        "",
        f"Screening {len(results)} metal catalysts for {reaction} at {data.get('temperature_K','?')}K "
        f"using d-band theory and volcano plot analysis.",
        "",
        "## Top Candidates",
        "",
        "| Metal | Score | d-band (eV) | ΔG (eV) | Verdict |",
        "|-------|-------|------------|---------|---------|",
    ]
    for r in sorted(results, key=lambda x: x.get("total_score", 0), reverse=True)[:6]:
        lines.append(f"| {r.get('symbol','?')} ({r.get('name','')}) | "
                     f"{r.get('total_score',0):.1f} | {r.get('d_band_center_eV',0):.2f} | "
                     f"{r.get('estimated_dG_eV',0):.3f} | {r.get('verdict','?')} |")
    lines += [
        "",
        f"## Date Discovered",
        data.get("timestamp", "")[:10],
        "",
        "## Tools Used",
        "NoetherSolve d-band center, volcano position, catalytic efficiency tools",
    ]
    return f"catalyst_discovery_{reaction.lower().replace(' ','_')[:30]}", "\n".join(lines)


def convert_climate(data: dict) -> tuple[str, str]:
    results = data.get("results", [])
    scenarios = sorted(set(r.get("scenario", "") for r in results))
    lines = [
        f"# Climate Sensitivity: {len(scenarios)} Scenarios × {data.get('n_profiles','?')} Feedback Profiles",
        "",
        "## Discovery Summary",
        "",
        "Quantitative climate sensitivity analysis across emission scenarios and feedback profiles.",
        "",
        "## Key Results",
        "",
        "| Scenario | CO₂ (ppm) | Feedback | ECS (K) | ΔT (K) |",
        "|----------|-----------|---------|---------|--------|",
    ]
    for r in results[:10]:
        baseline_t = 288.15
        delta_t = r.get("new_surface_temp_K", baseline_t) - baseline_t
        lines.append(f"| {r.get('scenario','')} | {r.get('co2_ppm',0):.0f} | "
                     f"{r.get('feedback_profile','')} | {r.get('ecs_K',0):.2f} | "
                     f"{delta_t:+.2f} |")
    lines += [
        "",
        f"## Date Discovered",
        data.get("timestamp", "")[:10],
        "",
        "## Tools Used",
        "NoetherSolve climate sensitivity, CO2 forcing, Stefan-Boltzmann tools",
    ]
    return "climate_sensitivity_analysis", "\n".join(lines)


def convert_drug_therapy(data: dict) -> tuple[str, str]:
    results = data.get("results", [])
    n_pass = sum(1 for r in results if r.get("verdict") == "PASS")
    lines = [
        f"# Drug Therapy Screening: {n_pass}/{len(results)} Candidates Pass Safety",
        "",
        "## Discovery Summary",
        "",
        f"Pharmacokinetic + safety screening of {len(results)} drug candidates.",
        "",
        "## Results",
        "",
        "| Drug | Verdict | Half-life (h) | TI | Interactions |",
        "|------|---------|--------------|-----|------------|",
    ]
    for r in results:
        lines.append(f"| {r.get('name','?')} | {r.get('verdict','?')} | "
                     f"{r.get('half_life_h',0):.1f} | {r.get('therapeutic_index',0):.1f} | "
                     f"{r.get('n_interactions',0)} |")
    lines += [
        "",
        f"## Date Discovered",
        data.get("timestamp", "")[:10],
        "",
        "## Tools Used",
        "NoetherSolve pharmacokinetics, drug interaction, dose adjustment tools",
    ]
    return "drug_therapy_screening", "\n".join(lines)


def convert_epidemiology(data: dict) -> tuple[str, str]:
    results = data.get("results", [])
    n_controlled = data.get("n_controlled", 0)
    lines = [
        f"# Epidemiology: {n_controlled}/{len(results)} Diseases Reach Herd Immunity",
        "",
        "## Discovery Summary",
        "",
        "Quantitative epidemic dynamics across diseases using SIR model and vaccine impact analysis.",
        "",
        "## Disease Table",
        "",
        "| Disease | R0 | Herd Imm % | Doubling (d) | Herd Achieved |",
        "|---------|-----|-----------|-------------|--------------|",
    ]
    for r in sorted(results, key=lambda x: x.get("R0", 0), reverse=True):
        lines.append(f"| {r.get('display_name', r.get('name','?'))} | "
                     f"{r.get('R0',0):.1f} | {r.get('herd_immunity_pct',0):.1f} | "
                     f"{r.get('doubling_time_days',0):.2f} | "
                     f"{'YES' if r.get('herd_achieved') else 'NO'} |")
    lines += [
        "",
        f"## Date Discovered",
        data.get("timestamp", "")[:10],
        "",
        "## Tools Used",
        "NoetherSolve SIR model, reproduction number, vaccine impact, herd immunity tools",
    ]
    return "epidemiology_dynamics_quantified", "\n".join(lines)


def convert_genetic_therapeutics(data: dict) -> tuple[str, str]:
    crispr = data.get("crispr", [])
    mrna = data.get("mrna", [])
    neoantigens = data.get("neoantigens", [])
    lines = [
        "# Genetic Therapeutics Pipeline: CRISPR + mRNA + Neoantigen Screening",
        "",
        "## Discovery Summary",
        "",
        f"End-to-end therapeutic pipeline evaluation: {len(crispr)} CRISPR guides, "
        f"{len(mrna)} mRNA therapeutics, {len(neoantigens)} neoantigen candidates.",
        "",
    ]
    if crispr:
        n_pass = sum(1 for c in crispr if c.get("verdict") == "PASS")
        lines += [f"## CRISPR Guides ({n_pass}/{len(crispr)} pass)", ""]
        for c in crispr[:5]:
            lines.append(f"- **{c.get('name','?')}**: on-target={c.get('on_target_score',0):.2f}, "
                         f"off-target risk={c.get('offtarget_risk','?')}, verdict={c.get('verdict','?')}")
        lines.append("")
    if mrna:
        n_pass = sum(1 for m in mrna if m.get("verdict") == "PASS")
        lines += [f"## mRNA Therapeutics ({n_pass}/{len(mrna)} pass)", ""]
        for m in mrna[:5]:
            lines.append(f"- **{m.get('name','?')}**: CAI={m.get('optimized_cai',0):.2f}, "
                         f"TLR risk={m.get('tlr7_8_risk','?')}, verdict={m.get('verdict','?')}")
        lines.append("")
    lines += [
        f"## Date Discovered",
        data.get("timestamp", "")[:10],
        "",
        "## Tools Used",
        "NoetherSolve CRISPR scorer, mRNA optimization, neoantigen pipeline tools",
    ]
    return "genetic_therapeutics_pipeline", "\n".join(lines)


def convert_origin_of_life(data: dict) -> tuple[str, str]:
    summary = data.get("summary", {})
    lines = [
        "# Origin of Life: Abiogenesis Pathway Plausibility Scores",
        "",
        "## Discovery Summary",
        "",
        "Quantitative prebiotic chemistry analysis: autocatalysis, RNA world, Miller-Urey synthesis.",
        "",
        "## Results",
        "",
    ]
    summary = data.get("summary", {})
    if summary:
        lines += ["### Summary", ""]
        for k, v in summary.items():
            lines.append(f"- **{k}**: {v}")
        lines.append("")

    # Autocatalytic sets (may be list or dict)
    ac_raw = data.get("autocatalytic", [])
    ac_list = ac_raw if isinstance(ac_raw, list) else [ac_raw]
    if ac_list:
        lines += ["### Autocatalytic Sets", ""]
        for ac in ac_list[:3]:
            lines.append(f"- **{ac.get('name','?')}**: RAF={ac.get('raf_size','?')}, "
                         f"autocatalytic={ac.get('is_autocatalytic','?')}")
        lines.append("")

    # Miller-Urey (may be list)
    mu_raw = data.get("miller_urey", [])
    mu_list = mu_raw if isinstance(mu_raw, list) else [mu_raw]
    if mu_list:
        lines += ["### Miller-Urey Synthesis", ""]
        for mu in mu_list[:3]:
            lines.append(f"- **{mu.get('name','?')}** ({mu.get('atmosphere','?')}): "
                         f"total yield={mu.get('total_yield',0):.4f}, "
                         f"amino acids={mu.get('amino_acids',0):.4f}")
        lines.append("")

    # RNA folding
    rna = data.get("rna_folding", [])
    if rna:
        lines += ["### RNA Folding Stability", ""]
        for r in rna[:4]:
            seq = r.get('sequence', r.get('name', '?'))[:20]
            dg = r.get('delta_g', r.get('folding_energy', 0))
            lines.append(f"- **{seq}**: ΔG={dg:.2f} kcal/mol")
        lines.append("")
    lines += [
        f"## Date Discovered",
        data.get("timestamp", "")[:10],
        "",
        "## Tools Used",
        "NoetherSolve autocatalytic set checker, Miller-Urey yield, RNA folding tools",
    ]
    return "origin_of_life_abiogenesis", "\n".join(lines)


def convert_quantum_mechanics(data: dict) -> tuple[str, str]:
    results = data.get("results", [])
    n_pass = data.get("n_pass", 0)
    lines = [
        f"# Quantum Mechanics: {n_pass}/{len(results)} Systems Pass Verification",
        "",
        "## Discovery Summary",
        "",
        "Exact quantum mechanical calculations for benchmark systems using NoetherSolve QM tools.",
        "",
        "## Results",
        "",
        "| System | Quantity | Value | Verdict |",
        "|--------|---------|-------|---------|",
    ]
    for r in results:
        lines.append(f"| {r.get('system','?')} | {r.get('quantity','?')} | "
                     f"{r.get('value','?')} | {r.get('verdict','?')} |")
    lines += [
        "",
        f"## Date Discovered",
        data.get("timestamp", "")[:10],
        "",
        "## Tools Used",
        "NoetherSolve QM calculator: particle-in-box, hydrogen, tunneling, uncertainty tools",
    ]
    return "quantum_mechanics_verification", "\n".join(lines)


def convert_supply_chain(data: dict) -> tuple[str, str]:
    results = data.get("results", [])
    lines = [
        "# Supply Chain Optimization: Quantified Cost vs. Service Tradeoffs",
        "",
        "## Discovery Summary",
        "",
        "Optimal inventory and routing solutions using NoetherSolve operations research tools.",
        "",
        "## Results",
        "",
    ]
    for r in results[:6]:
        name = r.get("scenario", r.get("name", "?"))
        lines.append(f"- **{name}**: {r.get('finding', r.get('description', ''))}")
    lines += [
        "",
        f"## Date Discovered",
        data.get("timestamp", "")[:10],
        "",
        "## Tools Used",
        "NoetherSolve EOQ, newsvendor, safety stock, vehicle routing, scheduling tools",
    ]
    return "supply_chain_optimization", "\n".join(lines)


def convert_topological_materials(data: dict) -> tuple[str, str]:
    results = data.get("results", [])
    n_topo = data.get("n_topological", 0)
    lines = [
        f"# Topological Materials: {n_topo}/{len(results)} Systems Classified as Topological",
        "",
        "## Discovery Summary",
        "",
        "Classification of quantum materials by topological phase using Chern numbers and Z₂ invariants.",
        "",
        "## Classification Table",
        "",
        "| System | Chern | Z₂ | AZ Class | Phase |",
        "|--------|-------|-----|---------|-------|",
    ]
    for r in results:
        lines.append(f"| {r.get('name','?')} | {r.get('chern_number','?')} | "
                     f"{r.get('z2_classification','—') or '—'} | "
                     f"{r.get('az_class','?')} | {r.get('phase','?')} |")
    lines += [
        "",
        f"## Date Discovered",
        data.get("timestamp", "")[:10],
        "",
        "## Tools Used",
        "NoetherSolve Chern number, Z₂ invariant, bulk-boundary correspondence tools",
    ]
    return "topological_materials_classification", "\n".join(lines)


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

CONVERTERS = {
    "bio_ai":               ("convergence_results.json",     convert_bio_ai),
    "conservation_mining":  ("discovery_results.json",       convert_conservation_mining),
    "ai_safety":            ("safety_evaluation.json",       convert_ai_safety),
    "battery_materials":    ("analysis_results.json",        convert_battery_materials),
    "behavioral_economics": ("analysis_results.json",        convert_behavioral_economics),
    "catalyst_discovery":   ("screening_results.json",       convert_catalyst),
    "climate_analysis":     ("scenario_results.json",        convert_climate),
    "drug_therapy":         ("screening_results.json",       convert_drug_therapy),
    "epidemiology":         ("scenario_results.json",        convert_epidemiology),
    "genetic_therapeutics": ("screening_results.json",       convert_genetic_therapeutics),
    "origin_of_life":       ("abiogenesis_results.json",     convert_origin_of_life),
    "quantum_mechanics":    ("calculation_results.json",     convert_quantum_mechanics),
    "supply_chain":         ("optimization_results.json",    convert_supply_chain),
    "topological_materials":("classification_results.json",  convert_topological_materials),
}


def export_lab(lab_name: str, force: bool = False) -> bool:
    """Convert a lab's JSON result to a .md finding. Returns True if written."""
    if lab_name not in CONVERTERS:
        print(f"  [SKIP] Unknown lab: {lab_name}")
        return False

    json_file, converter_fn = CONVERTERS[lab_name]
    json_path = LABS_DIR / lab_name / json_file

    if not json_path.exists():
        print(f"  [SKIP] {lab_name}: no results file ({json_file})")
        return False

    try:
        with open(json_path) as f:
            data = json.load(f)
    except Exception as e:
        print(f"  [ERROR] {lab_name}: failed to load JSON — {e}")
        return False

    try:
        stem, md_content = converter_fn(data)
    except Exception as e:
        print(f"  [ERROR] {lab_name}: converter failed — {e}")
        return False

    out_path = FINDINGS_DIR / f"{stem}.md"
    if out_path.exists() and not force:
        # Update only if JSON is newer
        if json_path.stat().st_mtime <= out_path.stat().st_mtime:
            print(f"  [SKIP] {lab_name}: finding up to date ({out_path.name})")
            return False

    out_path.write_text(md_content)
    ts = data.get("timestamp", "")[:10]
    print(f"  [WROTE] {lab_name} -> {out_path.name}  (source: {ts})")
    return True


def list_status():
    print(f"\n{'Lab':<25} {'JSON exists':<14} {'MD exists':<14} {'MD up to date'}")
    print("-" * 70)
    for lab_name, (json_file, _) in sorted(CONVERTERS.items()):
        json_path = LABS_DIR / lab_name / json_file
        json_exists = json_path.exists()

        # Find the md file by running the converter to get the stem
        md_exists = False
        md_fresh = False
        if json_exists:
            try:
                with open(json_path) as f:
                    data = json.load(f)
                _, converter_fn = CONVERTERS[lab_name]
                stem, _ = converter_fn(data)
                md_path = FINDINGS_DIR / f"{stem}.md"
                md_exists = md_path.exists()
                if md_exists:
                    md_fresh = json_path.stat().st_mtime <= md_path.stat().st_mtime
            except Exception:
                pass

        j = "✓" if json_exists else "✗"
        m = "✓" if md_exists else "✗"
        fresh = "✓ fresh" if md_fresh else ("stale" if md_exists else "—")
        print(f"  {lab_name:<23} {j:<14} {m:<14} {fresh}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Export lab results to finding .md files")
    parser.add_argument("--all", action="store_true", help="Export all labs")
    parser.add_argument("--lab", help="Export a specific lab")
    parser.add_argument("--force", action="store_true", help="Overwrite even if up to date")
    parser.add_argument("--list", action="store_true", help="Show status of all labs")
    args = parser.parse_args()

    if args.list:
        list_status()
        return

    labs_to_export = list(CONVERTERS.keys()) if args.all else ([args.lab] if args.lab else [])

    if not labs_to_export:
        parser.print_help()
        return

    print(f"\nExporting {len(labs_to_export)} lab(s) to {FINDINGS_DIR.relative_to(PROJECT)}/\n")
    written = 0
    for lab in labs_to_export:
        if export_lab(lab, force=args.force):
            written += 1

    print(f"\nDone: {written}/{len(labs_to_export)} finding files written.")
    if written:
        print(f"Run paper_watchdog.py --once to route new findings to papers.")


if __name__ == "__main__":
    main()
