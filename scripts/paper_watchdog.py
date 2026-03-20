#!/usr/bin/env python3
"""Paper Watchdog — runs periodic checks for paper-ready clusters.

Enhanced pipeline:
1. Check for new findings
2. Route each finding: NEW paper or EXISTING paper
3. If existing: integrate, regenerate PDF, update Zenodo
4. If new: queue for paper creation when threshold met

Usage:
    python scripts/paper_watchdog.py --hours 12 --interval 2
    python scripts/paper_watchdog.py --once --publish  # Single cycle with publishing
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from noethersolve.paper_agent import PaperAgent, ACKNOWLEDGMENT_TEMPLATE


# Published paper registry - maps cluster_id to DOI and covered topics
PUBLISHED_PAPERS = {
    "d1_vortex_conservation": {
        "doi": "10.5281/zenodo.19055338",
        "title": "Approximate Conservation Laws in Point Vortex Dynamics",
        "keywords": ["vortex", "q_f", "qf", "kirchhoff", "conservation", "stretch", "curvature", "dipole", "viscous"],
        "paper_dir": "d1_qf_conservation",
    },
    "d2_z3_cancellation": {
        "doi": "10.5281/zenodo.19055580",
        "title": "Z₃ Phase Cancellation in Choreographic Orbits",
        "keywords": ["choreograph", "z3", "figure-8", "figure8", "3-body", "3body"],
        "paper_dir": "d2_z3_cancellation",
    },
    "d3_llm_knowledge_gaps": {
        "doi": "10.5281/zenodo.19055582",
        "title": "Where LLMs Are Confidently Wrong",
        "keywords": ["knowledge_gap", "confidently_wrong", "llm", "blind_spot"],
        "paper_dir": "d3_llm_knowledge_gaps",
    },
    "d4_orthogonal_routing": {
        "doi": "10.5281/zenodo.19055588",
        "title": "Orthogonal Adapter Routing",
        "keywords": ["orthogonal", "routing", "adapter", "stacking", "interference"],
        "paper_dir": "d4_orthogonal_routing",
    },
    "d5_certainty_contamination": {
        "doi": "10.5281/zenodo.19068373",
        "title": "Certainty Contamination",
        "keywords": ["certainty", "contamination", "hedged", "definitive", "language_bias"],
        "paper_dir": "d5_certainty_contamination",
    },
    "d6_resolvent_unification": {
        "doi": "10.5281/zenodo.19071198",
        "title": "Resolvent-Conservation Unification",
        "keywords": ["resolvent", "spectral", "green", "unification", "laplacian"],
        "paper_dir": "d6_resolvent_unification",
    },
    "d7_oracle_biases": {
        "doi": "10.5281/zenodo.19124851",
        "title": "Nine Systematic Biases in Log-Probability LLM Evaluation",
        "keywords": ["oracle", "bias", "length_ratio", "distractor", "phrasing"],
        "paper_dir": "d7_oracle_biases",
    },
}


@dataclass
class FindingRouting:
    """Routing decision for a finding."""
    finding_file: str
    finding_title: str
    decision: str  # "new_paper", "extend_existing", "already_covered"
    target_paper: Optional[str] = None
    confidence: float = 0.0
    reason: str = ""


@dataclass
class PaperUpdate:
    """Result of updating an existing paper."""
    paper_id: str
    findings_added: list = field(default_factory=list)
    pdf_path: Optional[Path] = None
    zenodo_updated: bool = False
    new_version: Optional[str] = None
    errors: list = field(default_factory=list)


def log(msg: str, log_file: Path):
    """Append timestamped message to log file and print."""
    timestamp = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    line = f"{timestamp} | {msg}"
    print(line)
    with open(log_file, "a") as f:
        f.write(line + "\n")


def get_finding_keywords(finding_path: Path) -> list[str]:
    """Extract keywords from a finding file."""
    content = finding_path.read_text().lower()

    # Extract from filename
    keywords = finding_path.stem.replace("_", " ").split()

    # Extract from content - look for key terms
    key_patterns = [
        r"q_f", r"vortex", r"conservation", r"invariant", r"euler",
        r"llm", r"oracle", r"bias", r"adapter", r"routing",
        r"resolvent", r"spectral", r"certainty", r"contamination",
        r"choreograph", r"z3", r"3-body", r"figure-8",
        r"bio[-_]?ai", r"convergent", r"chemotaxis", r"dopamine",
        r"27b", r"knowledge", r"blind.?spot",
    ]

    for pattern in key_patterns:
        if re.search(pattern, content):
            keywords.append(pattern.replace(r"[-_]?", "").replace(".", ""))

    return list(set(keywords))


def route_finding(finding_path: Path, processed_findings: set) -> FindingRouting:
    """Decide if a finding should go to new or existing paper."""
    finding_file = finding_path.name

    # Skip if already processed
    if finding_file in processed_findings:
        return FindingRouting(
            finding_file=finding_file,
            finding_title=finding_path.stem,
            decision="already_processed",
            confidence=1.0,
            reason="Previously processed"
        )

    # Read finding content
    try:
        content = finding_path.read_text()
        # Extract title from first # heading
        title_match = re.search(r"^#\s+(.+)", content, re.MULTILINE)
        title = title_match.group(1) if title_match else finding_path.stem
    except Exception:
        title = finding_path.stem

    keywords = get_finding_keywords(finding_path)

    # Check against published papers
    best_match = None
    best_score = 0

    for paper_id, paper_info in PUBLISHED_PAPERS.items():
        paper_keywords = paper_info["keywords"]
        # Count keyword matches
        matches = sum(1 for kw in keywords if any(pk in kw or kw in pk for pk in paper_keywords))
        score = matches / max(len(paper_keywords), 1)

        if score > best_score:
            best_score = score
            best_match = paper_id

    # Decision thresholds
    if best_score >= 0.4:
        return FindingRouting(
            finding_file=finding_file,
            finding_title=title,
            decision="extend_existing",
            target_paper=best_match,
            confidence=best_score,
            reason=f"Matches {best_match} with {best_score:.0%} keyword overlap"
        )
    elif best_score >= 0.2:
        return FindingRouting(
            finding_file=finding_file,
            finding_title=title,
            decision="extend_existing",
            target_paper=best_match,
            confidence=best_score,
            reason=f"Weak match to {best_match} ({best_score:.0%}) - review recommended"
        )
    else:
        return FindingRouting(
            finding_file=finding_file,
            finding_title=title,
            decision="new_paper",
            confidence=1 - best_score,
            reason="No strong match to existing papers"
        )


def integrate_finding_into_paper(
    finding_path: Path,
    paper_id: str,
    paper_dir: Path,
    log_file: Path
) -> bool:
    """Integrate a finding into an existing paper's markdown."""
    paper_info = PUBLISHED_PAPERS.get(paper_id)
    if not paper_info:
        return False

    paper_subdir = paper_dir / paper_info.get("paper_dir", paper_id)
    draft_path = paper_subdir / "paper.md"

    if not draft_path.exists():
        # Try alternative names
        for alt in ["draft.md", "paper.tex", "main.md"]:
            alt_path = paper_subdir / alt
            if alt_path.exists():
                draft_path = alt_path
                break

    if not draft_path.exists():
        log(f"  Paper draft not found: {draft_path}", log_file)
        return False

    # Read finding content
    finding_content = finding_path.read_text()

    # Extract the main content (skip title)
    lines = finding_content.split("\n")
    finding_body = "\n".join(l for l in lines if not l.startswith("# "))

    # Read current paper
    paper_content = draft_path.read_text()

    # Find insertion point - before "## Discussion" or "## Conclusion" or at end
    insertion_patterns = [
        r"(##\s+Discussion)",
        r"(##\s+Conclusion)",
        r"(##\s+Future Work)",
        r"(##\s+Acknowledgment)",
    ]

    insert_pos = len(paper_content)
    for pattern in insertion_patterns:
        match = re.search(pattern, paper_content, re.IGNORECASE)
        if match:
            insert_pos = match.start()
            break

    # Create new finding section
    finding_title = lines[0].replace("# ", "### ") if lines[0].startswith("# ") else f"### {finding_path.stem}"
    new_section = f"\n\n{finding_title}\n\n{finding_body.strip()}\n\n"

    # Insert
    updated_content = paper_content[:insert_pos] + new_section + paper_content[insert_pos:]

    # Add update note
    update_note = f"\n<!-- Updated {datetime.utcnow().strftime('%Y-%m-%d')}: Added {finding_path.stem} -->\n"
    updated_content = update_note + updated_content

    # Write back
    draft_path.write_text(updated_content)
    log(f"  Integrated {finding_path.name} into {draft_path.name}", log_file)

    return True


def regenerate_pdf(paper_id: str, paper_dir: Path, log_file: Path) -> Optional[Path]:
    """Regenerate PDF for a paper."""
    paper_info = PUBLISHED_PAPERS.get(paper_id)
    if not paper_info:
        return None

    paper_subdir = paper_dir / paper_info.get("paper_dir", paper_id)

    # Find markdown source
    md_path = None
    for name in ["paper.md", "draft.md", "main.md"]:
        p = paper_subdir / name
        if p.exists():
            md_path = p
            break

    if not md_path:
        log(f"  No markdown source found for {paper_id}", log_file)
        return None

    pdf_path = paper_subdir / f"{md_path.stem}.pdf"

    # Run pandoc
    cmd = [
        "pandoc", str(md_path),
        "-o", str(pdf_path),
        "--pdf-engine=xelatex",
        "-V", "geometry:margin=1in",
        "-V", "fontsize=11pt",
    ]

    # Add bibliography if exists
    bib_path = paper_subdir / "references.bib"
    if bib_path.exists():
        cmd.extend(["--bibliography", str(bib_path), "--citeproc"])

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode == 0:
            log(f"  PDF regenerated: {pdf_path.name}", log_file)
            return pdf_path
        else:
            log(f"  PDF generation failed: {result.stderr[:200]}", log_file)
            return None
    except Exception as e:
        log(f"  PDF generation error: {e}", log_file)
        return None


def update_zenodo(paper_id: str, pdf_path: Path, log_file: Path) -> Optional[str]:
    """Update Zenodo with new version of paper."""
    paper_info = PUBLISHED_PAPERS.get(paper_id)
    if not paper_info:
        return None

    zenodo_token = os.environ.get("ZENODO_TOKEN")
    if not zenodo_token:
        log(f"  ZENODO_TOKEN not set, skipping upload", log_file)
        return None

    try:
        import requests
    except ImportError:
        log(f"  requests package not available", log_file)
        return None

    doi = paper_info["doi"]
    # Extract record ID from DOI
    record_id = doi.split(".")[-1] if "zenodo" in doi else None
    if not record_id:
        log(f"  Could not extract record ID from DOI: {doi}", log_file)
        return None

    headers = {"Authorization": f"Bearer {zenodo_token}"}
    base_url = "https://zenodo.org/api/deposit/depositions"

    try:
        # 1. Create new version
        resp = requests.post(
            f"{base_url}/{record_id}/actions/newversion",
            headers=headers
        )
        if resp.status_code != 201:
            log(f"  Zenodo new version failed: {resp.text[:200]}", log_file)
            return None

        new_deposit = resp.json()
        new_id = new_deposit["links"]["latest_draft"].split("/")[-1]

        # 2. Get the new draft
        resp = requests.get(f"{base_url}/{new_id}", headers=headers)
        draft = resp.json()
        bucket_url = draft["links"]["bucket"]

        # 3. Delete old files
        for f in draft.get("files", []):
            requests.delete(f"{base_url}/{new_id}/files/{f['id']}", headers=headers)

        # 4. Upload new PDF
        with open(pdf_path, "rb") as f:
            resp = requests.put(
                f"{bucket_url}/{pdf_path.name}",
                headers=headers,
                data=f,
            )

        # 5. Update metadata
        metadata = draft.get("metadata", {})
        metadata["description"] = metadata.get("description", "") + f"\n\nUpdated {datetime.utcnow().strftime('%Y-%m-%d')}"

        resp = requests.put(
            f"{base_url}/{new_id}",
            headers=headers,
            json={"metadata": metadata}
        )

        # 6. Publish
        resp = requests.post(
            f"{base_url}/{new_id}/actions/publish",
            headers=headers
        )

        if resp.status_code == 202:
            published = resp.json()
            new_doi = published.get("doi", doi)
            log(f"  Zenodo updated: {new_doi}", log_file)
            return new_doi
        else:
            log(f"  Zenodo publish failed: {resp.text[:200]}", log_file)
            return None

    except Exception as e:
        log(f"  Zenodo error: {e}", log_file)
        return None


def process_finding_routing(
    results_dir: Path,
    paper_dir: Path,
    log_file: Path,
    publish: bool = False
) -> dict:
    """Process all findings and route them appropriately."""
    findings_dir = results_dir / "discoveries" / "novel_findings"
    if not findings_dir.exists():
        return {"routed": 0, "extended": 0, "new": 0}

    # Load processed findings tracker
    tracker_path = results_dir / "paper_routing_tracker.json"
    if tracker_path.exists():
        with open(tracker_path) as f:
            tracker = json.load(f)
    else:
        tracker = {"processed": [], "extended_papers": {}, "new_paper_queue": []}

    processed = set(tracker["processed"])

    stats = {"routed": 0, "extended": 0, "new": 0, "updates": []}

    # Get all finding files
    finding_files = sorted(findings_dir.glob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True)

    for finding_path in finding_files:
        routing = route_finding(finding_path, processed)

        if routing.decision == "already_processed":
            continue

        stats["routed"] += 1
        log(f"Routing: {routing.finding_file} -> {routing.decision}", log_file)

        if routing.decision == "extend_existing" and routing.target_paper:
            log(f"  Target: {routing.target_paper} (confidence: {routing.confidence:.0%})", log_file)

            if publish and routing.confidence >= 0.3:
                # Integrate into paper
                success = integrate_finding_into_paper(
                    finding_path, routing.target_paper, paper_dir, log_file
                )

                if success:
                    stats["extended"] += 1

                    # Regenerate PDF
                    pdf_path = regenerate_pdf(routing.target_paper, paper_dir, log_file)

                    # Update Zenodo if PDF generated
                    if pdf_path:
                        new_doi = update_zenodo(routing.target_paper, pdf_path, log_file)
                        if new_doi:
                            stats["updates"].append({
                                "paper": routing.target_paper,
                                "finding": routing.finding_file,
                                "doi": new_doi
                            })

                    # Track
                    tracker["processed"].append(routing.finding_file)
                    if routing.target_paper not in tracker["extended_papers"]:
                        tracker["extended_papers"][routing.target_paper] = []
                    tracker["extended_papers"][routing.target_paper].append({
                        "finding": routing.finding_file,
                        "timestamp": datetime.utcnow().isoformat()
                    })
            else:
                # Queue for review
                tracker["processed"].append(routing.finding_file)

        elif routing.decision == "new_paper":
            stats["new"] += 1
            # Add to new paper queue
            if routing.finding_file not in [q["file"] for q in tracker["new_paper_queue"]]:
                tracker["new_paper_queue"].append({
                    "file": routing.finding_file,
                    "title": routing.finding_title,
                    "timestamp": datetime.utcnow().isoformat()
                })
            tracker["processed"].append(routing.finding_file)

    # Save tracker
    with open(tracker_path, "w") as f:
        json.dump(tracker, f, indent=2)

    return stats


def check_escalations(results_dir: Path) -> tuple[int, int, list]:
    """Check and resolve open escalations with fact-quality issues."""
    escalations_path = results_dir / "escalations.jsonl"
    if not escalations_path.exists():
        return 0, 0, []

    lines = []
    open_count = 0
    resolved_this_run = 0
    issues = []

    # Known domain -> v2 file mappings
    DOMAIN_V2_FILES = {
        "Continuous Q_f": "continuous_qf_facts_v2.json",
        "continuous_qf": "continuous_qf_facts_v2.json",
        "kinetic_k": "kinetic_k_facts_v2.json",
        "intersection_theory": "intersection_theory_facts_v2.json",
        "information_theory": "information_theory_facts_v2.json",
        "computational_conjectures": "computational_conjectures_facts_v2.json",
        "llm_alignment": "llm_alignment_facts_v2.json",
    }

    with open(escalations_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            entry = json.loads(line)
            if entry.get("status") == "open":
                open_count += 1
                domain = entry.get("domain", "unknown")

                # Check if v2 facts file exists
                problems_dir = results_dir.parent / "problems"
                found_v2 = False

                # Try direct mapping first
                for key, v2_file in DOMAIN_V2_FILES.items():
                    if key.lower() in domain.lower():
                        if (problems_dir / v2_file).exists():
                            entry["status"] = "resolved"
                            entry["resolution"] = f"V2 facts file exists: {v2_file}"
                            resolved_this_run += 1
                            found_v2 = True
                            break

                # Fall back to glob search
                if not found_v2:
                    domain_slug = domain.lower().replace(" ", "_").replace("-", "_")
                    keywords = [w for w in domain_slug.split("_") if len(w) > 2][:2]
                    for kw in keywords:
                        v2_facts = list(problems_dir.glob(f"*{kw}*_v2*.json"))
                        v2_facts += list(problems_dir.glob(f"*{kw}*v2*.json"))
                        if v2_facts:
                            entry["status"] = "resolved"
                            entry["resolution"] = f"V2 facts file exists: {v2_facts[0].name}"
                            resolved_this_run += 1
                            found_v2 = True
                            break

                if not found_v2:
                    issues.append(f"{domain}: needs v2 facts (length-matched distractors)")

            lines.append(json.dumps(entry))

    with open(escalations_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    return open_count, resolved_this_run, issues


def check_clusters(agent: PaperAgent, cluster_ids: list, threshold: float = 0.82) -> dict:
    """Check paper readiness for clusters."""
    results = {
        "ready": [],
        "not_ready": [],
        "already_published": [],
        "insufficient_novelty": [],
    }

    for cluster_id in cluster_ids:
        metrics = agent.get_cluster_metrics(cluster_id)
        if metrics is None:
            continue

        novelty = agent.check_novelty(cluster_id)

        if novelty["recommendation"] == "already_published":
            results["already_published"].append(cluster_id)
        elif novelty["recommendation"] == "insufficient":
            results["insufficient_novelty"].append(cluster_id)
        elif metrics.maturity_score >= threshold:
            results["ready"].append({
                "cluster_id": cluster_id,
                "maturity": metrics.maturity_score,
                "novel_findings": len(novelty["novel_findings"]),
            })
        else:
            results["not_ready"].append({
                "cluster_id": cluster_id,
                "maturity": metrics.maturity_score,
            })

    return results


def check_labs(results_dir: Path) -> list:
    """Check labs for publishable findings."""
    registry_path = results_dir / "lab_registry.json"
    if not registry_path.exists():
        return []

    with open(registry_path) as f:
        registry = json.load(f)

    publishable = []
    for lab in registry.get("labs", []):
        if lab.get("status") != "running":
            continue

        lab_id = lab.get("lab_id", "")
        findings = lab.get("findings", [])
        n_findings = lab.get("n_findings", 0)

        if n_findings >= 5:
            publishable.append({
                "lab_id": lab_id,
                "n_findings": n_findings,
                "sample_findings": findings[:2] if findings else [],
            })

    return publishable


def check_research_status(results_dir: Path) -> dict:
    """Get current research status."""
    status_path = results_dir / "research_status.json"
    if not status_path.exists():
        return {}

    with open(status_path) as f:
        return json.load(f)


def run_watchdog_cycle(results_dir: Path, log_file: Path, publish: bool = False):
    """Run one watchdog cycle."""
    agent = PaperAgent(results_dir=results_dir)
    paper_dir = results_dir.parent / "paper"

    # 1. Check research status
    status = check_research_status(results_dir)
    current_domain = status.get("current_domain", "unknown")
    n_completed = len(status.get("completed_domains", []))
    domain_results = status.get("domain_results", {})
    n_passing = sum(1 for r in domain_results.values() if r.get("verdict") == "PASS")
    n_failing = sum(1 for r in domain_results.values() if r.get("verdict") == "FAIL")

    log(f"Research: {n_completed} domains ({n_passing} pass, {n_failing} fail), current: {current_domain}", log_file)

    # 2. Check escalations
    open_esc, resolved_esc, issues = check_escalations(results_dir)
    if resolved_esc > 0:
        log(f"Escalations: resolved {resolved_esc} (had v2 facts)", log_file)
    if issues:
        for issue in issues[:3]:
            log(f"Escalation open: {issue}", log_file)

    # 3. FINDING ROUTING - New/Existing paper decision
    routing_stats = process_finding_routing(results_dir, paper_dir, log_file, publish=publish)
    if routing_stats["routed"] > 0:
        log(f"Finding routing: {routing_stats['routed']} processed, {routing_stats['extended']} extended, {routing_stats['new']} new", log_file)
        for update in routing_stats.get("updates", []):
            log(f"  PAPER UPDATED: {update['paper']} <- {update['finding']} (DOI: {update['doi']})", log_file)

    # 4. Check paper clusters for new papers
    clusters = [
        "d1_vortex_conservation", "d2_z3_cancellation", "d3_llm_knowledge_gaps",
        "d4_orthogonal_routing", "d5_certainty_contamination", "d6_resolvent_unification",
    ]
    cluster_results = check_clusters(agent, clusters)

    if cluster_results["ready"]:
        for item in cluster_results["ready"]:
            log(f"PAPER READY: {item['cluster_id']} (maturity={item['maturity']:.2f}, novel={item['novel_findings']})", log_file)
            if publish:
                try:
                    result = agent.write_and_publish(item["cluster_id"])
                    if result.success:
                        log(f"PUBLISHED: {item['cluster_id']} -> DOI: {result.doi}", log_file)
                    else:
                        log(f"Publish failed: {'; '.join(result.errors)}", log_file)
                except Exception as e:
                    log(f"Publish error: {e}", log_file)
    else:
        n_published = len(cluster_results["already_published"])
        n_insufficient = len(cluster_results["insufficient_novelty"])
        log(f"Clusters: {n_published} published, {n_insufficient} insufficient novelty, 0 ready", log_file)

    # 5. Check labs
    publishable_labs = check_labs(results_dir)
    if publishable_labs:
        for lab in publishable_labs:
            log(f"Lab potential: {lab['lab_id']} ({lab['n_findings']} findings)", log_file)

    # 6. Check discovery grades
    grades_path = results_dir / "discovery_grades.json"
    if grades_path.exists():
        with open(grades_path) as f:
            grades = json.load(f)
        summary = grades.get("summary", {})
        eureka = summary.get("by_grade", {}).get("EUREKA", 0)
        gem = summary.get("by_grade", {}).get("GEM", 0)
        paper_ready = summary.get("paper_ready", 0)
        if paper_ready > 0:
            log(f"Discovery grades: {paper_ready} paper-ready ({eureka} EUREKA, {gem} GEM)", log_file)

    # 7. Report new paper queue
    tracker_path = results_dir / "paper_routing_tracker.json"
    if tracker_path.exists():
        with open(tracker_path) as f:
            tracker = json.load(f)
        queue = tracker.get("new_paper_queue", [])
        if len(queue) >= 3:
            log(f"New paper queue: {len(queue)} findings ready for new paper", log_file)

    log("Cycle complete", log_file)


def main():
    parser = argparse.ArgumentParser(description="Paper Watchdog with Finding Routing")
    parser.add_argument("--hours", type=float, default=12, help="Total runtime in hours")
    parser.add_argument("--interval", type=float, default=2, help="Check interval in hours")
    parser.add_argument("--publish", action="store_true", help="Actually publish/update papers")
    parser.add_argument("--once", action="store_true", help="Run once and exit")
    args = parser.parse_args()

    results_dir = PROJECT_ROOT / "results"
    log_file = results_dir / "paper_watchdog.log"

    log("=" * 60, log_file)
    log(f"Watchdog started: {args.hours}h runtime, {args.interval}h interval, publish={args.publish}", log_file)

    if args.once:
        run_watchdog_cycle(results_dir, log_file, publish=args.publish)
        return

    n_cycles = int(args.hours / args.interval)
    interval_seconds = args.interval * 3600

    for i in range(n_cycles):
        log(f"Cycle {i+1}/{n_cycles}", log_file)
        try:
            run_watchdog_cycle(results_dir, log_file, publish=args.publish)
        except Exception as e:
            log(f"Cycle error: {e}", log_file)

        if i < n_cycles - 1:
            log(f"Sleeping {args.interval}h until next cycle...", log_file)
            time.sleep(interval_seconds)

    log("Watchdog complete", log_file)


if __name__ == "__main__":
    main()
