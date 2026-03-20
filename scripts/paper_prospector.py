#!/usr/bin/env python3
"""
paper_prospector.py — Scans for paper-ready clusters and collects evidence.

This agent does three things:
1. SCAN — reads all findings, candidates, and open questions to identify
   which discovery clusters have enough material for a paper
2. COLLECT — gathers evidence into a structured "paper brief" per cluster:
   facts flipped, novel findings, domain results, open questions
3. GRADE — updates the discovery grader with fresh metrics

Runs as a lightweight background process (no GPU needed).
The orchestrator can run this periodically to keep paper readiness current.

Usage:
    python scripts/paper_prospector.py              # Full scan + report
    python scripts/paper_prospector.py --brief <id> # Show brief for one cluster
    python scripts/paper_prospector.py --clues      # Show unclustered clues
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

_HERE = Path(__file__).parent.parent
sys.path.insert(0, str(_HERE))

CANDIDATES_FILE = _HERE / "results" / "candidates.tsv"
FINDINGS_DIR = _HERE / "results" / "discoveries" / "novel_findings"
OPEN_QUESTIONS = _HERE / "results" / "open_questions.jsonl"
BRIEFS_DIR = _HERE / "results" / "paper_briefs"
PROSPECTOR_LOG = _HERE / "results" / "prospector_log.txt"


def log(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(PROSPECTOR_LOG, "a") as f:
        f.write(line + "\n")


def load_candidates() -> list[dict]:
    """Parse candidates.tsv into structured records."""
    rows = []
    if not CANDIDATES_FILE.exists():
        return rows
    with open(CANDIDATES_FILE) as f:
        header = None
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if header is None:
                header = parts
                continue
            row = {}
            for i, h in enumerate(header):
                row[h] = parts[i] if i < len(parts) else ""
            rows.append(row)
    return rows


def load_findings() -> list[dict]:
    """Load all novel findings markdown files."""
    findings = []
    if not FINDINGS_DIR.exists():
        return findings
    for f in sorted(FINDINGS_DIR.glob("*.md")):
        content = f.read_text()
        # Extract title from first heading
        title = f.stem.replace("_", " ").title()
        for line in content.split("\n"):
            if line.startswith("# "):
                title = line[2:].strip()
                break
        findings.append({
            "file": f.name,
            "title": title,
            "path": str(f),
            "size": len(content),
            "content_preview": content[:300],
        })
    return findings


def load_open_questions() -> list[dict]:
    """Load open questions queue."""
    questions = []
    if not OPEN_QUESTIONS.exists():
        return questions
    with open(OPEN_QUESTIONS) as f:
        for line in f:
            try:
                q = json.loads(line)
                if q.get("status") != "done":
                    questions.append(q)
            except json.JSONDecodeError:
                pass
    return questions


def load_clusters() -> dict:
    """Load discovery clusters from grader."""
    try:
        from scripts.discovery_grader import DISCOVERY_CLUSTERS
        return dict(DISCOVERY_CLUSTERS)
    except ImportError:
        return {}


def match_finding_to_cluster(finding: dict, clusters: dict) -> str | None:
    """Try to match a finding to an existing cluster by keyword overlap."""
    fname = finding["file"].lower()
    title = finding["title"].lower()
    text = (fname + " " + title).lower()

    # Direct keyword matches
    matches = {
        "vortex": "d1_vortex_conservation",
        "q_f": "d1_vortex_conservation",
        "qf": "d1_vortex_conservation",
        "choreograph": "d2_z3_cancellation",
        "z3": "d2_z3_cancellation",
        "z₃": "d2_z3_cancellation",
        "knowledge_gap": "d3_llm_knowledge_gaps",
        "confidently_wrong": "d3_llm_knowledge_gaps",
        "orthogonal": "d4_orthogonal_routing",
        "routing": "d4_orthogonal_routing",
        "certainty": "d5_certainty_contamination",
        "contamination": "d5_certainty_contamination",
        "resolvent": "d6_resolvent_unification",
        "spectral": "d6_resolvent_unification",
        "length_ratio": "d3_llm_knowledge_gaps",
        "length_bias": "d3_llm_knowledge_gaps",
        "distractor": "d3_llm_knowledge_gaps",
        "anti_fluency": "d5_certainty_contamination",
        "conservation": "d1_vortex_conservation",
        "stretch_resist": "d1_vortex_conservation",
        "curvature": "d1_vortex_conservation",
        "clinical": "clinical_translation",
        "bio_ai": "bio_ai_parallels",
    }

    for keyword, cluster_id in matches.items():
        if keyword in text:
            return cluster_id

    return None


def build_paper_brief(cluster_id: str, clusters: dict, candidates: list,
                      findings: list, questions: list) -> dict:
    """Build a structured brief for a paper cluster."""
    cluster = clusters.get(cluster_id, {})
    domains = cluster.get("domains", [])

    # Gather domain-specific candidates
    cluster_candidates = []
    for c in candidates:
        domain = c.get("domain", "")
        if domain in domains or cluster_id in domain:
            cluster_candidates.append(c)

    # Gather matched findings
    cluster_findings = []
    for f in findings:
        match = match_finding_to_cluster(f, clusters)
        if match == cluster_id:
            cluster_findings.append(f)

    # Gather related open questions
    cluster_questions = []
    for q in questions:
        q_domain = q.get("domain", "")
        q_text = q.get("text", "").lower()
        if q_domain in domains or any(d.lower() in q_text for d in domains):
            cluster_questions.append(q)

    # Compute evidence strength
    n_flipped = sum(1 for c in cluster_candidates
                    if "FLIPPED" in c.get("verdict", ""))
    n_pass = sum(1 for c in cluster_candidates
                 if c.get("verdict", "").startswith("PASS")
                 or "DUAL" in c.get("verdict", ""))
    n_fail = sum(1 for c in cluster_candidates
                 if c.get("verdict", "").startswith("FAIL"))

    brief = {
        "cluster_id": cluster_id,
        "title": cluster.get("title", cluster_id),
        "doi": cluster.get("doi"),
        "date_written": cluster.get("date_written"),
        "stages_complete": cluster.get("stages_complete", []),
        "domains": domains,
        "evidence": {
            "candidates_total": len(cluster_candidates),
            "flipped": n_flipped,
            "passing": n_pass,
            "failing": n_fail,
        },
        "findings": [f["title"] for f in cluster_findings],
        "findings_files": [f["file"] for f in cluster_findings],
        "open_questions": len(cluster_questions),
        "clues_summary": (
            f"{len(cluster_findings)} findings, "
            f"{n_flipped} flipped, "
            f"{n_pass} passing, "
            f"{len(cluster_questions)} open questions"
        ),
        "updated": datetime.now().isoformat(),
    }
    return brief


def find_unclustered_clues(findings: list, clusters: dict) -> list[dict]:
    """Find findings that don't match any existing cluster."""
    unclustered = []
    for f in findings:
        match = match_finding_to_cluster(f, clusters)
        if match is None:
            unclustered.append(f)
    return unclustered


def scan_lab_opportunities():
    """Check if tool inventory suggests new lab projects."""
    try:
        from scripts.lab_registry import (
            scan_for_lab_opportunities, grade_all_labs, save_lab_registry,
            LAB_PROJECTS
        )

        # Get tool names from MCP server
        server_path = _HERE / "noethersolve" / "mcp_server" / "server.py"
        tool_names = []
        if server_path.exists():
            in_tool = False
            with open(server_path) as f:
                for line in f:
                    if "@mcp.tool()" in line:
                        in_tool = True
                    elif in_tool and line.strip().startswith("def "):
                        name = line.strip().split("def ")[1].split("(")[0]
                        tool_names.append(name)
                        in_tool = False

        suggestions = scan_for_lab_opportunities(tool_names)

        # Also grade existing labs
        graded = grade_all_labs()
        save_lab_registry(graded)

        return suggestions, graded
    except Exception as e:
        log(f"  Lab scan error: {e}")
        return [], []


def full_scan():
    """Run a full scan and generate all paper briefs."""
    log("=== Paper Prospector Scan ===")

    candidates = load_candidates()
    findings = load_findings()
    questions = load_open_questions()
    clusters = load_clusters()

    log(f"  Loaded: {len(candidates)} candidates, {len(findings)} findings, "
        f"{len(questions)} open questions, {len(clusters)} clusters")

    # Build briefs for all clusters
    BRIEFS_DIR.mkdir(parents=True, exist_ok=True)
    briefs = []
    for cluster_id in clusters:
        brief = build_paper_brief(cluster_id, clusters, candidates,
                                  findings, questions)
        briefs.append(brief)

        # Save individual brief
        brief_file = BRIEFS_DIR / f"{cluster_id}.json"
        with open(brief_file, "w") as f:
            json.dump(brief, f, indent=2)

    # Find unclustered clues
    unclustered = find_unclustered_clues(findings, clusters)

    # Summary
    log(f"  Generated {len(briefs)} paper briefs")
    if unclustered:
        log(f"  {len(unclustered)} unclustered findings (potential new papers):")
        for u in unclustered:
            log(f"    - {u['title']} ({u['file']})")

    # Print report
    print(f"\n{'='*70}")
    print(f"  Paper Prospector Report")
    print(f"{'='*70}")

    # Sort by evidence strength
    briefs.sort(key=lambda b: b["evidence"]["flipped"] + b["evidence"]["passing"],
                reverse=True)

    for b in briefs:
        ev = b["evidence"]
        if ev["candidates_total"] == 0 and not b["findings"]:
            continue  # Skip empty clusters
        stage = b["stages_complete"][-1] if b["stages_complete"] else "raw"
        doi = "DOI" if b["doi"] else "   "
        print(f"\n  [{doi}] {b['title']}")
        print(f"       Stage: {stage} | {b['clues_summary']}")
        if b["findings"]:
            for f in b["findings"][:3]:
                print(f"       Finding: {f}")

    if unclustered:
        print(f"\n  --- Unclustered Clues ({len(unclustered)}) ---")
        print(f"  These findings don't belong to any paper yet.")
        print(f"  Consider creating new clusters for them:\n")
        for u in unclustered:
            print(f"    {u['title']}")
            print(f"      File: {u['file']}")
            print(f"      Preview: {u['content_preview'][:100]}...")
            print()

    # Scan for lab opportunities
    lab_suggestions, lab_grades = scan_lab_opportunities()
    if lab_suggestions:
        log(f"  {len(lab_suggestions)} new lab opportunities detected:")
        for s in lab_suggestions:
            log(f"    - {s['cluster_id']}: {s['n_tools']} tools ({s['description']})")

    if lab_grades:
        running = sum(1 for g in lab_grades if g.status in ("running", "producing"))
        log(f"  {len(lab_grades)} labs tracked, {running} running")

    # Save summary
    summary = {
        "scan_time": datetime.now().isoformat(),
        "n_clusters": len(clusters),
        "n_briefs_with_evidence": sum(1 for b in briefs
                                       if b["evidence"]["candidates_total"] > 0),
        "n_unclustered": len(unclustered),
        "unclustered_titles": [u["title"] for u in unclustered],
        "total_findings": len(findings),
        "total_candidates": len(candidates),
        "n_lab_suggestions": len(lab_suggestions),
        "n_labs_running": sum(1 for g in lab_grades
                              if g.status in ("running", "producing")),
    }
    with open(BRIEFS_DIR / "scan_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return briefs, unclustered


def show_brief(cluster_id: str):
    """Show detailed brief for one cluster."""
    brief_file = BRIEFS_DIR / f"{cluster_id}.json"
    if brief_file.exists():
        with open(brief_file) as f:
            brief = json.load(f)
    else:
        # Generate fresh
        clusters = load_clusters()
        if cluster_id not in clusters:
            print(f"Unknown cluster: {cluster_id}")
            return
        brief = build_paper_brief(
            cluster_id, clusters,
            load_candidates(), load_findings(), load_open_questions()
        )

    print(json.dumps(brief, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Paper prospector — find paper-ready clusters")
    parser.add_argument("--brief", default=None, help="Show brief for a cluster")
    parser.add_argument("--clues", action="store_true", help="Show unclustered clues only")
    args = parser.parse_args()

    if args.brief:
        show_brief(args.brief)
        return

    briefs, unclustered = full_scan()

    if args.clues:
        if unclustered:
            for u in unclustered:
                print(f"{u['title']}")
                print(f"  {u['file']}")
                print()
        else:
            print("No unclustered clues — all findings belong to a paper.")


if __name__ == "__main__":
    main()
