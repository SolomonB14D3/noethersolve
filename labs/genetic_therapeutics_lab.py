#!/usr/bin/env python3
"""
Genetic Therapeutics Lab — Screen CRISPR, mRNA, and neoantigen candidates.

Chains NoetherSolve tools to evaluate therapeutic candidates across three
modalities: CRISPR guide RNAs, mRNA therapeutics, and neoantigen peptides.

Usage:
    python labs/genetic_therapeutics_lab.py
    python labs/genetic_therapeutics_lab.py --verbose

Data sources:
    - Built-in scoring matrices from literature (Rule Set 2, Doench 2016)
    - Future: Ensembl REST API for gene sequences

⚠️  DISCLAIMER: FOR RESEARCH AND EDUCATIONAL USE ONLY
    This tool provides COMPUTATIONAL PREDICTIONS that require experimental
    validation. CRISPR off-target analysis must be confirmed with empirical
    methods (GUIDE-seq, CIRCLE-seq, DISCOVER-seq). mRNA and neoantigen
    candidates require in vitro and in vivo testing before clinical use.
    Not validated for clinical decision-making or regulatory submissions.
"""

import json
import sys
from datetime import datetime
from pathlib import Path

# Ensure noethersolve is importable
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from noethersolve.crispr import score_guide
from noethersolve.audit_sequence import audit_sequence
from noethersolve.mrna_design import analyze_mrna_design, optimize_codons, calculate_duplex_stability
from noethersolve.neoantigen_pipeline import (
    score_cleavage, score_tap, score_mhc_binding, evaluate_neoantigen,
)
from noethersolve.antibody_developability import (
    analyze_charge, analyze_aggregation, analyze_polyreactivity,
    analyze_liabilities, assess_developability, RiskLevel,
)

RESULTS_DIR = _ROOT / "results" / "labs" / "genetic_therapeutics"

# ── Candidate Definitions ─────────────────────────────────────────────

CRISPR_CANDIDATES = [
    {
        "name": "BRCA1-guide-1",
        "target_gene": "BRCA1",
        "spacer": "GTGGATCCAGACTGCCTTCC",
        "pam": "NGG",
        "note": "Targets exon 11 BRCT domain region",
    },
    {
        "name": "TP53-guide-1",
        "target_gene": "TP53",
        "spacer": "CCATTGTTCAATATCGTCCG",
        "pam": "NGG",
        "note": "Targets DNA-binding domain, codon 248 region",
    },
    {
        "name": "VEGFA-guide-1",
        "target_gene": "VEGFA",
        "spacer": "GGGTGGGTGTGTCTACAGGA",
        "pam": "NGG",
        "note": "Targets promoter proximal region for anti-angiogenic editing",
    },
]

MRNA_CANDIDATES = [
    {
        "name": "mRNA-VEGF-trap",
        "target": "VEGF decoy receptor",
        "coding_sequence": "AUGGCUAAAGCUGCUGGACUGGCUCCUGGUUUUACUGGUACCUGCCAUGGCAGAAGGCAGUGAUUUCCAUCUGCUGUUCCUGAACAAAGCUAG",
        "five_prime_utr": "GGGAAAUAAGAGAGAAAAGAAGAGUAAGAAGAAAUAUAAGAGCCACC",
        "three_prime_utr": "UGAUAAUAGGCUGGAGCCUCGGUGGCCUAGCUUCUUGCCCCUUGGGCCUCCCCCCAG",
        "note": "Encodes soluble VEGF-binding fragment for anti-angiogenic therapy",
    },
    {
        "name": "mRNA-IL12-immunostim",
        "target": "IL-12 p35 subunit",
        "coding_sequence": "AUGUCCCCUGAUGCCGCUGUUGCUGCUGCCGCUGCCGCUGUCUGCUACCCCUGCUGCUGCCUGCCGCCAGCGCUGAAACCAGCGUAG",
        "five_prime_utr": "",
        "three_prime_utr": "",
        "note": "Intratumoral IL-12 expression for immune activation",
    },
    {
        "name": "mRNA-EPO-anemia",
        "target": "Erythropoietin",
        "coding_sequence": "AUGGGGGUAGCGACAGCUUCCCAGCCCCUGAAAGCUGGCAGCUACGCUUCUGCUGCUGCUGCCUCUGCUGUCUUCCGCUCCUGCUGCUAG",
        "five_prime_utr": "GGGAAAUAAGAGAGAAAAGAAGAGUAAGAAGAAAUAUAAGAGCCACC",
        "three_prime_utr": "",
        "note": "EPO mRNA for chronic kidney disease anemia",
    },
]

NEOANTIGEN_CANDIDATES = [
    {
        "name": "KRAS-G12D-neo",
        "peptide": "VVVGADGVGK",
        "allele": "HLA-A*02:01",
        "wildtype": "VVVGAGGVGK",
        "note": "KRAS G12D driver mutation — pancreatic cancer",
    },
    {
        "name": "TP53-R175H-neo",
        "peptide": "HMTEVVRRC",
        "allele": "HLA-A*02:01",
        "wildtype": "RMTEVVRRC",
        "note": "TP53 R175H hotspot — multiple cancers",
    },
    {
        "name": "BRAF-V600E-neo",
        "peptide": "LATEKSRWSG",
        "allele": "HLA-A*02:01",
        "wildtype": "LATEKSRWSV",
        "note": "BRAF V600E — melanoma driver (10-mer)",
    },
]

# Antibody variable region candidates (VH domains)
ANTIBODY_CANDIDATES = [
    {
        "name": "Trastuzumab-VH",
        "target": "HER2/neu",
        "sequence": "EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIYPTNG"
                    "YTRYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYCSRWGGDGFYAMDYWGQ"
                    "GTLVTVSS",
        "note": "Anti-HER2 — breast cancer, FDA approved",
    },
    {
        "name": "Adalimumab-VH",
        "target": "TNF-alpha",
        "sequence": "EVQLVESGGGLVQPGRSLRLSCAASGFTFDDYAMHWVRQAPGKGLEWVSAITWNSGH"
                    "IDYADSVEGRFTISRDNAKNSLYLQMNSLRAEDTAVYYCAKVSYLSTASSLDYWGQ"
                    "GTLVTVSS",
        "note": "Anti-TNFα — rheumatoid arthritis, FDA approved",
    },
    {
        "name": "Pembrolizumab-VH",
        "target": "PD-1",
        "sequence": "QVQLVQSGVEVKKPGASVKVSCKASGYTFTNYYMYWVRQAPGQGLEWMGGINPSNGG"
                    "TNFNEKFKNRVTLTTDSSTTTAYMELKSLQFDDTAVYYCARRDYRFDMGFDYWGQG"
                    "TTVTVSS",
        "note": "Anti-PD-1 checkpoint inhibitor — multiple cancers",
    },
    {
        "name": "Test-HighCharge-VH",
        "target": "Test",
        "sequence": "KKKKKKKKKKVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKKKKKK"
                    "KKKKGLEWVARIYPTNGYTRYRRRRRRRRRRFTISADTSKNTAYLQMNSLRAEDTA"
                    "VYYCSRWGGDGFYAMDYWGQGTLVTVSS",
        "note": "High positive charge test — expect low viscosity, high polyreactivity",
    },
]


# ── Screening Functions ───────────────────────────────────────────────

def screen_crispr(candidates):
    """Screen CRISPR guide RNA candidates."""
    results = []
    print("\n" + "=" * 70)
    print("  CRISPR Guide RNA Screening")
    print("=" * 70)

    for cand in candidates:
        print(f"\n--- {cand['name']} ({cand['target_gene']}) ---")
        print(f"  Spacer: {cand['spacer']}")
        print(f"  Note:   {cand['note']}")

        # Score guide RNA
        guide_report = score_guide(cand["spacer"], pam=cand["pam"])
        print(f"  Activity score: {guide_report.activity_score:.1f}/100")
        print(f"  Off-target risk: {guide_report.offtarget_risk}")
        print(f"  GC content: {guide_report.gc_content:.1%}")
        print(f"  Verdict: {guide_report.verdict}")
        if guide_report.issues:
            for issue in guide_report.issues:
                print(f"    [{issue.severity}] {issue.message}")

        # Audit DNA sequence
        seq_report = audit_sequence(cand["spacer"])
        n_issues = len(seq_report.issues) if hasattr(seq_report, "issues") else 0
        print(f"  Sequence audit: {n_issues} issue(s)")

        result = {
            "name": cand["name"],
            "target_gene": cand["target_gene"],
            "spacer": cand["spacer"],
            "activity_score": guide_report.activity_score,
            "offtarget_risk": guide_report.offtarget_risk,
            "gc_content": guide_report.gc_content,
            "verdict": guide_report.verdict,
            "n_guide_issues": len(guide_report.issues),
            "n_sequence_issues": n_issues,
            "composite_score": guide_report.activity_score
                * (1.0 if guide_report.offtarget_risk == "LOW" else
                   0.7 if guide_report.offtarget_risk == "MODERATE" else 0.4),
        }
        results.append(result)

    # Rank
    results.sort(key=lambda r: r["composite_score"], reverse=True)
    print("\n  Ranked CRISPR candidates:")
    for i, r in enumerate(results, 1):
        print(f"    {i}. {r['name']}  score={r['composite_score']:.1f}  "
              f"verdict={r['verdict']}  offtarget={r['offtarget_risk']}")
    return results


def screen_mrna(candidates):
    """Screen mRNA therapeutic candidates."""
    results = []
    print("\n" + "=" * 70)
    print("  mRNA Therapeutic Screening")
    print("=" * 70)

    for cand in candidates:
        print(f"\n--- {cand['name']} ({cand['target']}) ---")
        print(f"  Note: {cand['note']}")

        coding = cand["coding_sequence"]

        # Full mRNA design analysis
        kwargs = {"coding_sequence": coding, "use_pseudouridine": True}
        if cand["five_prime_utr"]:
            kwargs["five_prime_utr"] = cand["five_prime_utr"]
        if cand["three_prime_utr"]:
            kwargs["three_prime_utr"] = cand["three_prime_utr"]
        design_report = analyze_mrna_design(**kwargs)
        print(f"  Design quality: {design_report.overall_quality}")
        print(f"    Thermodynamic dG: {design_report.thermodynamics.delta_G:.2f} kcal/mol")
        print(f"    TLR7/8 risk: {design_report.immunogenicity.tlr7_8_risk}")

        # Codon optimization
        codon_report = optimize_codons(coding, strategy="balanced")
        print(f"  Codon optimization:")
        print(f"    Original CAI: {codon_report.original_cai:.3f}")
        print(f"    Optimized CAI: {codon_report.optimized_cai:.3f}")
        print(f"    CAI improvement: {codon_report.optimized_cai - codon_report.original_cai:+.3f}")

        # Duplex stability for a representative region (first 8 nt)
        region = coding[:8]
        complement = region.replace("A", "x").replace("U", "A").replace("x", "U") \
                          .replace("G", "x").replace("C", "G").replace("x", "C")
        duplex_report = calculate_duplex_stability(region, complement, modified=True)
        print(f"  Duplex stability (first 8nt): dG={duplex_report.delta_G:.2f} kcal/mol")

        quality_score = {"excellent": 4, "good": 3, "fair": 2, "poor": 1}.get(
            design_report.overall_quality, 1)
        risk_score = {"low": 3, "medium": 2, "high": 1}.get(
            design_report.immunogenicity.tlr7_8_risk.lower(), 1)

        result = {
            "name": cand["name"],
            "target": cand["target"],
            "coding_length": len(coding),
            "overall_quality": design_report.overall_quality,
            "delta_G": design_report.thermodynamics.delta_G,
            "tlr7_8_risk": design_report.immunogenicity.tlr7_8_risk,
            "original_cai": codon_report.original_cai,
            "optimized_cai": codon_report.optimized_cai,
            "duplex_dG": duplex_report.delta_G,
            "composite_score": quality_score * 10 + risk_score * 5 + codon_report.optimized_cai * 10,
        }
        results.append(result)

    results.sort(key=lambda r: r["composite_score"], reverse=True)
    print("\n  Ranked mRNA candidates:")
    for i, r in enumerate(results, 1):
        print(f"    {i}. {r['name']}  score={r['composite_score']:.1f}  "
              f"quality={r['overall_quality']}  tlr7_8={r['tlr7_8_risk']}")
    return results


def screen_neoantigens(candidates):
    """Screen neoantigen peptide candidates."""
    results = []
    print("\n" + "=" * 70)
    print("  Neoantigen Peptide Screening")
    print("=" * 70)

    for cand in candidates:
        print(f"\n--- {cand['name']} ---")
        print(f"  Peptide:  {cand['peptide']}")
        print(f"  Wildtype: {cand.get('wildtype', 'N/A')}")
        print(f"  Allele:   {cand['allele']}")
        print(f"  Note:     {cand['note']}")

        peptide = cand["peptide"]
        allele = cand["allele"]

        # Step 1: Proteasomal cleavage
        cleavage = score_cleavage(peptide)
        print(f"  1. Cleavage: score={cleavage.cleavage_probability:.3f}  "
              f"C-term={peptide[-1]}")

        # Step 2: TAP transport
        tap = score_tap(peptide)
        print(f"  2. TAP transport: score={tap.tap_score:.3f}  "
              f"probability={tap.transport_probability:.1%}")

        # Step 3: MHC binding
        mhc = score_mhc_binding(peptide, allele=allele)
        print(f"  3. MHC binding: score={mhc.binding_score:.3f}  "
              f"level={mhc.binding_level}")

        # Step 4: Full pipeline evaluation
        pipeline = evaluate_neoantigen(
            peptide, allele=allele,
            wildtype=cand.get("wildtype"),
        )
        pipeline_pass = "PASS" if pipeline.pipeline_pass else "FAIL"
        print(f"  4. Pipeline: {pipeline_pass}  limiting_step={pipeline.limiting_step}")
        print(f"     Combined score: {pipeline.combined_score:.3f}")
        print(f"     Recommendation: {pipeline.recommendation}")

        result = {
            "name": cand["name"],
            "peptide": peptide,
            "allele": allele,
            "wildtype": cand.get("wildtype", ""),
            "cleavage_score": cleavage.cleavage_probability,
            "tap_score": tap.tap_score,
            "tap_probability": tap.transport_probability,
            "mhc_score": mhc.binding_score,
            "mhc_level": mhc.binding_level,
            "pipeline_pass": pipeline.pipeline_pass,
            "combined_score": pipeline.combined_score,
            "limiting_step": pipeline.limiting_step,
            "recommendation": pipeline.recommendation,
            "composite_score": pipeline.combined_score,
        }
        results.append(result)

    results.sort(key=lambda r: r["composite_score"], reverse=True)
    print("\n  Ranked neoantigen candidates:")
    for i, r in enumerate(results, 1):
        status = "PASS" if r["pipeline_pass"] else "FAIL"
        print(f"    {i}. {r['name']}  score={r['composite_score']:.3f}  "
              f"pipeline={status}  limiting={r['limiting_step']}")
    return results


def screen_antibodies(candidates):
    """Screen antibody VH domain candidates for developability."""
    results = []
    print("\n" + "=" * 70)
    print("  Antibody Developability Screening")
    print("=" * 70)

    risk_order = [RiskLevel.LOW, RiskLevel.MODERATE, RiskLevel.HIGH, RiskLevel.VERY_HIGH]

    for cand in candidates:
        print(f"\n--- {cand['name']} ({cand['target']}) ---")
        print(f"  Sequence length: {len(cand['sequence'])} aa")
        print(f"  Note: {cand['note']}")

        seq = cand["sequence"]

        # Full developability assessment
        dev = assess_developability(seq)

        # Charge analysis (viscosity predictor)
        charge = analyze_charge(seq)
        print(f"  Charge analysis:")
        print(f"    Net charge (pH 7): {charge.net_charge_pH7:+.1f}")
        print(f"    pI estimate: {charge.isoelectric_point:.1f}")
        print(f"    Viscosity risk: {charge.viscosity_risk.value}")

        # Aggregation
        agg = analyze_aggregation(seq)
        print(f"  Aggregation:")
        print(f"    Aggregation score: {agg.aggregation_score:.2f}")
        print(f"    Risk: {agg.aggregation_risk.value}")
        print(f"    Hotspot regions: {len(agg.hotspot_regions)}")

        # Polyreactivity
        poly = analyze_polyreactivity(seq)
        print(f"  Polyreactivity:")
        print(f"    Positive charge density: {poly.positive_charge_density:.1f}%")
        print(f"    Aromatic density: {poly.aromatic_density:.1f}%")
        print(f"    Risk: {poly.polyreactivity_risk.value}")

        # Liabilities
        liab = analyze_liabilities(seq)
        print(f"  Chemical liabilities: {liab.total_liabilities} total")
        print(f"    Deamidation sites: {len(liab.deamidation_sites)}")
        print(f"    Oxidation sites: {len(liab.oxidation_sites)}")
        print(f"    Glycosylation sites: {len(liab.glycosylation_sites)}")

        # Overall
        print(f"  Overall risk: {dev.overall_risk.value}")
        print(f"  Recommendation: {dev.recommendation}")

        # Composite score: lower risk = higher score
        def risk_to_score(risk):
            return (4 - risk_order.index(risk)) * 25  # 100, 75, 50, 25

        viscosity_score = risk_to_score(charge.viscosity_risk)
        agg_score = risk_to_score(agg.aggregation_risk)
        poly_score = risk_to_score(poly.polyreactivity_risk)
        liab_score = max(0, 100 - liab.total_liabilities * 5)  # -5 per liability

        composite = (viscosity_score + agg_score + poly_score + liab_score) / 4.0

        verdict = "PASS" if dev.overall_risk in (RiskLevel.LOW, RiskLevel.MODERATE) else "CAUTION"

        result = {
            "name": cand["name"],
            "target": cand["target"],
            "sequence_length": len(seq),
            "net_charge": charge.net_charge_pH7,
            "pI": charge.isoelectric_point,
            "viscosity_risk": charge.viscosity_risk.value,
            "aggregation_score": agg.aggregation_score,
            "aggregation_risk": agg.aggregation_risk.value,
            "n_hotspots": len(agg.hotspot_regions),
            "polyreactivity_risk": poly.polyreactivity_risk.value,
            "positive_charge_density": poly.positive_charge_density,
            "aromatic_density": poly.aromatic_density,
            "total_liabilities": liab.total_liabilities,
            "overall_risk": dev.overall_risk.value,
            "recommendation": dev.recommendation,
            "composite_score": composite,
            "verdict": verdict,
        }
        results.append(result)

    results.sort(key=lambda r: r["composite_score"], reverse=True)
    print("\n  Ranked antibody candidates:")
    for i, r in enumerate(results, 1):
        print(f"    {i}. {r['name']}  score={r['composite_score']:.1f}  "
              f"verdict={r['verdict']}  overall_risk={r['overall_risk']}")
    return results


# ── Main ──────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  Genetic Therapeutics Lab — Candidate Screening")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    crispr_results = screen_crispr(CRISPR_CANDIDATES)
    mrna_results = screen_mrna(MRNA_CANDIDATES)
    neoantigen_results = screen_neoantigens(NEOANTIGEN_CANDIDATES)
    antibody_results = screen_antibodies(ANTIBODY_CANDIDATES)

    # Summary
    print("\n" + "=" * 70)
    print("  SCREENING SUMMARY")
    print("=" * 70)
    print(f"\n  CRISPR guides screened:  {len(crispr_results)}")
    crispr_pass = sum(1 for r in crispr_results if r["verdict"] == "PASS")
    print(f"    Passed: {crispr_pass}/{len(crispr_results)}")

    print(f"\n  mRNA therapeutics screened: {len(mrna_results)}")
    mrna_good = sum(1 for r in mrna_results
                    if r["overall_quality"] in ("excellent", "good"))
    print(f"    Good/Excellent: {mrna_good}/{len(mrna_results)}")

    print(f"\n  Neoantigens screened: {len(neoantigen_results)}")
    neo_pass = sum(1 for r in neoantigen_results if r["pipeline_pass"])
    print(f"    Promising/Passed: {neo_pass}/{len(neoantigen_results)}")

    print(f"\n  Antibodies screened: {len(antibody_results)}")
    ab_pass = sum(1 for r in antibody_results if r["verdict"] == "PASS")
    print(f"    Developable (PASS): {ab_pass}/{len(antibody_results)}")

    # Top pick per modality
    print("\n  Top candidates:")
    if crispr_results:
        top = crispr_results[0]
        print(f"    CRISPR:     {top['name']} (score {top['composite_score']:.1f})")
    if mrna_results:
        top = mrna_results[0]
        print(f"    mRNA:       {top['name']} (score {top['composite_score']:.1f})")
    if neoantigen_results:
        top = neoantigen_results[0]
        print(f"    Neoantigen: {top['name']} (score {top['composite_score']:.3f})")
    if antibody_results:
        top = antibody_results[0]
        print(f"    Antibody:   {top['name']} (score {top['composite_score']:.1f})")

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "timestamp": datetime.now().isoformat(),
        "lab": "genetic_therapeutics",
        "crispr": crispr_results,
        "mrna": mrna_results,
        "neoantigens": neoantigen_results,
        "antibodies": antibody_results,
        "summary": {
            "crispr_screened": len(crispr_results),
            "crispr_passed": crispr_pass,
            "mrna_screened": len(mrna_results),
            "mrna_good": mrna_good,
            "neoantigen_screened": len(neoantigen_results),
            "neoantigen_passed": neo_pass,
            "antibody_screened": len(antibody_results),
            "antibody_passed": ab_pass,
        },
    }
    out_path = RESULTS_DIR / "screening_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
