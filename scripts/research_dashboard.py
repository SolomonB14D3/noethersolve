#!/usr/bin/env python3
"""
research_dashboard.py — Live HTML research dashboard for NoetherSolve.

Shows:
  - Resource orchestrator status and GPU allocation
  - Agent status (running/idle, PID)
  - Escalations needing attention
  - Domain-by-domain results (pass rate, margins)
  - Open questions queue
  - Recent log entries

Usage:
    python scripts/research_dashboard.py              # Serve on http://localhost:8050
    python scripts/research_dashboard.py --port 9000  # Custom port
    python scripts/research_dashboard.py --once        # Write single HTML file and exit
    python scripts/research_dashboard.py --json        # Machine-readable output
    python scripts/research_dashboard.py --terminal    # Old terminal mode
"""

import argparse
import http.server
import json
import os
import re
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

_HERE = Path(__file__).parent.parent
RESULTS = _HERE / "results"
STATUS_FILE = RESULTS / "research_status.json"
LOG_FILE = RESULTS / "research_log.txt"
ORCHESTRATOR_STATE = RESULTS / "orchestrator_state.json"
PROBLEMS_DIR = _HERE / "problems"
CANDIDATES_TSV = RESULTS / "candidates.tsv"
OPEN_QUESTIONS = RESULTS / "open_questions.jsonl"
ESCALATION_FILE = RESULTS / "escalations.jsonl"
HTML_OUTPUT = RESULTS / "dashboard.html"
GRADES_FILE = RESULTS / "discovery_grades.json"


def is_pid_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def load_status() -> dict:
    if STATUS_FILE.exists():
        with open(STATUS_FILE) as f:
            return json.load(f)
    return {}


def load_orchestrator() -> dict:
    if ORCHESTRATOR_STATE.exists():
        with open(ORCHESTRATOR_STATE) as f:
            return json.load(f)
    return {}


def count_candidates() -> dict:
    counts = {"total": 0, "pass": 0, "fail": 0, "flipped": 0}
    if not CANDIDATES_TSV.exists():
        return counts
    with open(CANDIDATES_TSV) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            counts["total"] += 1
            upper = line.upper()
            if "DUAL-PASS" in upper or "FLIPPED" in upper:
                counts["flipped"] += 1
            elif "PASS" in upper:
                counts["pass"] += 1
            elif "FAIL" in upper:
                counts["fail"] += 1
    return counts


def count_open_questions() -> int:
    if not OPEN_QUESTIONS.exists():
        return 0
    count = 0
    with open(OPEN_QUESTIONS) as f:
        for line in f:
            try:
                q = json.loads(line)
                if q.get("status") == "open":
                    count += 1
            except json.JSONDecodeError:
                pass
    return count


def count_domains() -> dict:
    import yaml
    domains = {}
    for yaml_path in sorted(PROBLEMS_DIR.glob("*.yaml")):
        try:
            with open(yaml_path) as f:
                prob = yaml.safe_load(f)
            vs = prob.get("verification_set")
            if not vs:
                continue
            vs_path = yaml_path.parent / vs
            if not vs_path.exists():
                continue
            with open(vs_path) as f:
                data = json.load(f)
            n = len(data) if isinstance(data, list) else len(data.get("facts", []))
            domains[prob.get("name", yaml_path.stem)] = {
                "facts": n,
                "model": prob.get("model", "Qwen/Qwen3-4B-Base"),
                "threshold": prob.get("pass_threshold", 0.5),
            }
        except Exception:
            continue
    return domains


def count_tools() -> int:
    server_path = _HERE / "noethersolve" / "mcp_server" / "server.py"
    if not server_path.exists():
        return 0
    count = 0
    with open(server_path) as f:
        for line in f:
            if "@mcp.tool()" in line:
                count += 1
    return count


def count_tests() -> int:
    test_dir = _HERE / "tests"
    if not test_dir.exists():
        return 0
    count = 0
    for tf in test_dir.glob("test_*.py"):
        with open(tf) as f:
            for line in f:
                if line.strip().startswith("def test_"):
                    count += 1
    return count


def check_agents() -> list[dict]:
    agents = []
    agent_defs = [
        ("27B Oracle Runner", "research_run.pid", "research_run.log"),
        ("Resource Orchestrator", "orchestrator.pid", "orchestrator_log.txt"),
        ("Autonomy Loop", "autonomy_run.pid", "autonomy_run.log"),
        ("Teacher-Student", "teacher_student.pid", "teacher_student_run.log"),
        ("Benchmark Runner", "benchmark_run.pid", "benchmark_run.log"),
        ("Adapter Trainer", "adapter_train.pid", "adapter_train.log"),
        ("27B Download", "27b_download.pid", "27b_download.log"),
    ]
    for name, pid_fname, log_fname in agent_defs:
        pid_file = RESULTS / pid_fname
        if pid_file.exists():
            try:
                pid = int(pid_file.read_text().strip())
                agents.append({
                    "name": name,
                    "pid": pid,
                    "alive": is_pid_running(pid),
                    "log": str(RESULTS / log_fname),
                })
            except (ValueError, FileNotFoundError):
                pass

    paper_dir = _HERE / "paper"
    if paper_dir.exists():
        for draft in paper_dir.glob("*/draft.md"):
            mtime = datetime.fromtimestamp(draft.stat().st_mtime)
            age_hours = (datetime.now() - mtime).total_seconds() / 3600
            if age_hours < 24:
                agents.append({
                    "name": f"Paper: {draft.parent.name}",
                    "pid": None,
                    "alive": None,
                    "log": str(draft),
                    "note": f"last modified {age_hours:.1f}h ago",
                })
    return agents


def get_escalations() -> list[dict]:
    if not ESCALATION_FILE.exists():
        return []
    escalations = []
    with open(ESCALATION_FILE) as f:
        for line in f:
            try:
                e = json.loads(line)
                if e.get("status") == "open":
                    escalations.append(e)
            except json.JSONDecodeError:
                pass
    return escalations


def load_grades() -> list[dict]:
    """Load discovery grades, regenerating if stale."""
    # Regenerate grades each time for freshness
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from discovery_grader import grade_all, save_grades, GRADES
        graded = grade_all()
        save_grades(graded)
        from dataclasses import asdict
        return [asdict(g) for g in graded], GRADES
    except Exception:
        # Fallback to cached file
        if GRADES_FILE.exists():
            with open(GRADES_FILE) as f:
                data = json.load(f)
            return data.get("grades", []), {}
        return [], {}


LAB_REGISTRY_FILE = RESULTS / "lab_registry.json"

def load_labs() -> list[dict]:
    """Load lab project data."""
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from lab_registry import grade_all_labs, save_lab_registry
        from dataclasses import asdict
        graded = grade_all_labs()
        save_lab_registry(graded)
        return [asdict(g) for g in graded]
    except Exception:
        if LAB_REGISTRY_FILE.exists():
            with open(LAB_REGISTRY_FILE) as f:
                data = json.load(f)
            return data.get("labs", [])
        return []


def recent_log(n: int = 15) -> list[str]:
    if not LOG_FILE.exists():
        return []
    with open(LOG_FILE) as f:
        lines = f.readlines()
    return [l.rstrip() for l in lines[-n:]]


# ---------------------------------------------------------------------------
# HTML generation
# ---------------------------------------------------------------------------

GRADE_COLORS = {
    "NOBEL": ("#FFD700", "👑"),
    "EUREKA": ("#FF6B35", "⚡"),
    "GEM": ("#00D4FF", "💎"),
    "SOLID": ("#4CAF50", "🪨"),
    "NUGGET": ("#9C27B0", "🔮"),
    "ORE": ("#6E7681", "⛏️"),
}


def _build_paper_rows(graded_list: list, grade_defs: dict) -> list[str]:
    rows = []
    for g in graded_list:
        cluster_id = g.get("cluster_id", "")

        # Find DOI from cluster data
        try:
            from discovery_grader import DISCOVERY_CLUSTERS
            doi_str = DISCOVERY_CLUSTERS.get(cluster_id, {}).get("doi", "")
            venue = DISCOVERY_CLUSTERS.get(cluster_id, {}).get("venue", "")
        except Exception:
            doi_str = ""
            venue = ""

        # Only show entries that have a DOI (actual papers on Zenodo)
        if not doi_str:
            continue

        grade = g.get("grade", "ORE")
        color, emoji = GRADE_COLORS.get(grade, ("#6E7681", "?"))
        title = g.get("title", "?")
        current_stage = g.get("current_stage", "")
        stages_complete = g.get("stages_complete", [])
        readiness = g.get("paper_readiness", 0)
        next_action = g.get("next_action", "")

        date_written = DISCOVERY_CLUSTERS.get(cluster_id, {}).get("date_written", "")

        doi_link = (f'<a href="https://doi.org/{doi_str}" target="_blank" '
                    f'style="color:#58a6ff;text-decoration:none">{doi_str}</a>'
                    if doi_str else '<span style="color:#6e7681">—</span>')

        # Build stage pipeline visualization
        all_stages = ["discovery", "evidence", "writing", "zenodo",
                      "submitted", "in_review", "revised", "accepted", "published"]
        stage_labels = ["Disc", "Evid", "Write", "Zen",
                        "Sub", "Rev", "Revis", "Acc", "Pub"]
        stage_dots = []
        for s, label in zip(all_stages, stage_labels):
            if s in stages_complete:
                stage_dots.append(
                    f'<span title="{s}" style="color:#3fb950;font-size:10px">'
                    f'&#9679; {label}</span>')
            elif s == current_stage:
                stage_dots.append(
                    f'<span title="{s}" style="color:#d29922;font-size:10px">'
                    f'&#9679; {label}</span>')
            else:
                stage_dots.append(
                    f'<span title="{s}" style="color:#30363d;font-size:10px">'
                    f'&#9675; {label}</span>')
        pipeline = ' '.join(stage_dots)

        rows.append(f"""<tr>
            <td><span style="color:{color};font-weight:700">{emoji} {grade}</span></td>
            <td>{title}</td>
            <td style="font-size:11px;color:#8b949e">{date_written or '—'}</td>
            <td style="font-size:10px">{pipeline}</td>
            <td style="font-size:11px;color:#8b949e">{venue or '—'}</td>
            <td style="font-size:11px;color:#8b949e">{readiness:.0f}%</td>
            <td style="font-size:11px">{doi_link}</td>
        </tr>""")
    return rows


def _load_lab_results(lab: dict) -> str:
    """Load and format results from a lab's results directory."""
    results_dir_rel = lab.get("results_dir", "")
    if not results_dir_rel:
        return '<span style="color:#8b949e">No results directory configured</span>'

    results_dir = _HERE / results_dir_rel
    if not results_dir.exists():
        return '<span style="color:#8b949e">No results yet — run lab to generate</span>'

    json_files = sorted(results_dir.glob("*.json"))
    if not json_files:
        return '<span style="color:#8b949e">No results yet — run lab to generate</span>'

    parts = []
    for jf in json_files:
        try:
            with open(jf) as f:
                data = json.load(f)
        except Exception:
            parts.append(f'<div style="color:#f85149">Error reading {jf.name}</div>')
            continue

        ts = data.get("timestamp", "unknown")
        if isinstance(ts, str) and "T" in ts:
            ts = ts.split("T")[0] + " " + ts.split("T")[1][:5]

        lab_id = lab.get("lab_id", "")

        # Drug therapy results
        if "drug_therapy" in lab_id or "drug_therapy" in str(jf):
            results = data.get("results", [])
            if results:
                header = (f'<div style="margin-bottom:4px;color:#58a6ff">'
                          f'Drug Screening — {len(results)} candidates '
                          f'({data.get("n_pass", 0)} pass, '
                          f'{data.get("n_caution", 0)} caution, '
                          f'{data.get("n_fail", 0)} fail) '
                          f'<span style="color:#8b949e;font-size:11px">{ts}</span></div>')
                tbl = ('<table style="width:100%;font-size:11px;border-collapse:collapse">'
                       '<tr style="color:#8b949e"><th style="text-align:left;padding:2px 6px">Drug</th>'
                       '<th>Score</th><th>Verdict</th><th>T&frac12; (h)</th>'
                       '<th>TI</th><th>Interactions</th></tr>')
                for r in results:
                    vc = "#3fb950" if r.get("verdict") == "PASS" else (
                        "#d29922" if r.get("verdict") == "CAUTION" else "#f85149")
                    tbl += (f'<tr><td style="padding:2px 6px">{r.get("name", "?")}</td>'
                            f'<td style="text-align:center">{r.get("score", 0):.1f}</td>'
                            f'<td style="text-align:center;color:{vc}">{r.get("verdict", "?")}</td>'
                            f'<td style="text-align:center">{r.get("half_life_h", 0):.1f}</td>'
                            f'<td style="text-align:center">{r.get("therapeutic_index", 0):.1f}</td>'
                            f'<td style="text-align:center">{r.get("n_interactions", 0)}</td></tr>')
                tbl += '</table>'
                parts.append(header + tbl)
            continue

        # Genetic therapeutics results
        if "genetic" in lab_id or "genetic" in str(jf):
            sections = []
            for section_key, section_name in [("crispr", "CRISPR Guides"),
                                               ("mrna", "mRNA Therapeutics"),
                                               ("neoantigens", "Neoantigens")]:
                items = data.get(section_key, [])
                if not items:
                    continue
                n_pass = sum(1 for i in items if i.get("verdict") == "PASS" or i.get("pipeline_pass"))
                sec = (f'<div style="margin:4px 0 2px;color:#58a6ff">{section_name} — '
                       f'{n_pass}/{len(items)} pass</div>')
                sec += ('<table style="width:100%;font-size:11px;border-collapse:collapse">'
                        '<tr style="color:#8b949e"><th style="text-align:left;padding:2px 6px">Name</th>'
                        '<th>Score</th><th>Verdict</th><th>Details</th></tr>')
                for item in items:
                    v = item.get("verdict", "FAIL" if not item.get("pipeline_pass") else "PASS")
                    vc = "#3fb950" if v == "PASS" else "#f85149"
                    score = item.get("composite_score", item.get("combined_score", 0))
                    score_str = f"{score:.1f}" if isinstance(score, float) else str(score)
                    detail = ""
                    if section_key == "crispr":
                        detail = f'GC={item.get("gc_content", 0):.0%} risk={item.get("offtarget_risk", "?")}'
                    elif section_key == "mrna":
                        detail = f'CAI={item.get("optimized_cai", 0):.2f} TLR={item.get("tlr7_8_risk", "?")}'
                    elif section_key == "neoantigens":
                        detail = f'limit={item.get("limiting_step", "?")}'
                    sec += (f'<tr><td style="padding:2px 6px">{item.get("name", "?")}</td>'
                            f'<td style="text-align:center">{score_str}</td>'
                            f'<td style="text-align:center;color:{vc}">{v}</td>'
                            f'<td style="text-align:center;color:#8b949e">{detail}</td></tr>')
                sec += '</table>'
                sections.append(sec)
            if sections:
                parts.append(f'<div style="color:#8b949e;font-size:11px;margin-bottom:2px">{ts}</div>'
                             + ''.join(sections))
            continue

        # Catalyst discovery results
        if "catalyst" in lab_id or "catalyst" in str(jf):
            results = data.get("results", [])
            if results:
                header = (f'<div style="margin-bottom:4px;color:#58a6ff">'
                          f'{data.get("reaction", "?")} Catalyst Screening — '
                          f'{len(results)} candidates at {data.get("temperature_K", "?")}K '
                          f'<span style="color:#8b949e;font-size:11px">{ts}</span></div>')
                tbl = ('<table style="width:100%;font-size:11px;border-collapse:collapse">'
                       '<tr style="color:#8b949e"><th style="text-align:left;padding:2px 6px">Metal</th>'
                       '<th>Total Score</th><th>Verdict</th><th>d-band (eV)</th>'
                       '<th>dG (eV)</th><th>Limiting</th></tr>')
                for r in results:
                    vc = "#3fb950" if r.get("verdict") == "TOP" else (
                        "#d29922" if r.get("verdict") == "VIABLE" else "#8b949e")
                    tbl += (f'<tr><td style="padding:2px 6px">{r.get("symbol", "?")} '
                            f'({r.get("name", "")})</td>'
                            f'<td style="text-align:center">{r.get("total_score", 0):.1f}</td>'
                            f'<td style="text-align:center;color:{vc}">{r.get("verdict", "?")}</td>'
                            f'<td style="text-align:center">{r.get("d_band_center_eV", 0):.1f}</td>'
                            f'<td style="text-align:center">{r.get("estimated_dG_eV", 0):.3f}</td>'
                            f'<td style="text-align:center;color:#8b949e">'
                            f'{r.get("limiting_side", "?")[:20]}</td></tr>')
                tbl += '</table>'
                parts.append(header + tbl)
            continue

        # Climate results
        if "climate" in lab_id or ("climate" in str(jf) and "epidem" not in lab_id):
            results = data.get("results", [])
            if results and results[0].get("co2_ppm") is not None:
                scenarios = sorted(set(r.get("scenario", "") for r in results))
                header = (f'<div style="margin-bottom:4px;color:#58a6ff">'
                          f'Climate Scenarios — {len(scenarios)} scenarios x '
                          f'{data.get("n_profiles", "?")} feedback profiles '
                          f'<span style="color:#8b949e;font-size:11px">{ts}</span></div>')
                tbl = ('<table style="width:100%;font-size:11px;border-collapse:collapse">'
                       '<tr style="color:#8b949e"><th style="text-align:left;padding:2px 6px">Scenario</th>'
                       '<th>CO2 (ppm)</th><th>Feedback</th><th>Forcing (W/m2)</th>'
                       '<th>ECS (K)</th><th>Temp (K)</th></tr>')
                for r in results:
                    tbl += (f'<tr><td style="padding:2px 6px">{r.get("scenario", "?")}</td>'
                            f'<td style="text-align:center">{r.get("co2_ppm", 0):.0f}</td>'
                            f'<td style="text-align:center">{r.get("feedback_profile", "?")}</td>'
                            f'<td style="text-align:center">{r.get("forcing_Wm2", 0):.2f}</td>'
                            f'<td style="text-align:center">{r.get("ecs_K", 0):.2f}</td>'
                            f'<td style="text-align:center">{r.get("new_surface_temp_K", 0):.1f}</td></tr>')
                tbl += '</table>'
                parts.append(header + tbl)
            continue

        # Conservation mining results
        if "conservation" in lab_id or "discovery" in jf.name:
            header = (f'<div style="margin-bottom:4px;color:#58a6ff">'
                      f'Conservation Law Mining — {data.get("n_systems", 0)} systems, '
                      f'{data.get("n_candidates_total", 0)} candidates '
                      f'<span style="color:#8b949e;font-size:11px">{ts}</span></div>')
            summary = (f'<div style="margin-bottom:4px">'
                       f'<span style="color:#3fb950">{data.get("n_novel_approximate", 0)} novel approximate</span> | '
                       f'<span style="color:#58a6ff">{data.get("n_exact_known", 0)} exact known</span> | '
                       f'<span style="color:#8b949e">{data.get("n_artifacts", 0)} artifacts</span></div>')
            tops = data.get("top_candidates", [])[:6]
            if tops:
                tbl = ('<table style="width:100%;font-size:11px;border-collapse:collapse">'
                       '<tr style="color:#8b949e"><th style="text-align:left;padding:2px 6px">System</th>'
                       '<th>Name</th><th>frac_var</th><th>Class</th><th>Novel</th></tr>')
                for t in tops:
                    nc = "#3fb950" if not t.get("known") else "#8b949e"
                    tbl += (f'<tr><td style="padding:2px 6px">{t.get("system", "?")}</td>'
                            f'<td style="text-align:center">{t.get("name", "?")}</td>'
                            f'<td style="text-align:center">{t.get("frac_var", 0):.1e}</td>'
                            f'<td style="text-align:center">{t.get("classification", "?")}</td>'
                            f'<td style="text-align:center;color:{nc}">'
                            f'{"YES" if not t.get("known") else "no"}</td></tr>')
                tbl += '</table>'
                parts.append(header + summary + tbl)
            else:
                parts.append(header + summary)
            continue

        # Epidemiology results
        if "epidem" in lab_id or "epidemic" in str(data.get("pipeline", "")):
            results = data.get("results", [])
            if results:
                header = (f'<div style="margin-bottom:4px;color:#58a6ff">'
                          f'Epidemic Scenarios — {len(results)} diseases '
                          f'({data.get("n_controlled", 0)} controlled, '
                          f'{data.get("n_uncontrolled", 0)} uncontrolled) '
                          f'<span style="color:#8b949e;font-size:11px">{ts}</span></div>')
                tbl = ('<table style="width:100%;font-size:11px;border-collapse:collapse">'
                       '<tr style="color:#8b949e"><th style="text-align:left;padding:2px 6px">Disease</th>'
                       '<th>R0</th><th>Herd Imm %</th><th>Doubling (d)</th>'
                       '<th>Attack %</th><th>Herd?</th></tr>')
                for r in results:
                    hc = "#3fb950" if r.get("herd_achieved") else "#f85149"
                    tbl += (f'<tr><td style="padding:2px 6px">{r.get("display_name", r.get("name", "?"))}</td>'
                            f'<td style="text-align:center">{r.get("R0", 0):.1f}</td>'
                            f'<td style="text-align:center">{r.get("herd_immunity_pct", 0):.1f}</td>'
                            f'<td style="text-align:center">{r.get("doubling_time_days", 0):.2f}</td>'
                            f'<td style="text-align:center">{r.get("attack_rate_pct", 0):.0f}</td>'
                            f'<td style="text-align:center;color:{hc}">{"YES" if r.get("herd_achieved") else "NO"}</td></tr>')
                tbl += '</table>'
                parts.append(header + tbl)
            continue

        # Topological materials results
        if "topolog" in lab_id or "topolog" in str(data.get("pipeline", "")):
            results = data.get("results", [])
            if results:
                header = (f'<div style="margin-bottom:4px;color:#58a6ff">'
                          f'Topological Classification — {data.get("n_topological", 0)} topological, '
                          f'{data.get("n_trivial", 0)} trivial '
                          f'<span style="color:#8b949e;font-size:11px">{ts}</span></div>')
                tbl = ('<table style="width:100%;font-size:11px;border-collapse:collapse">'
                       '<tr style="color:#8b949e"><th style="text-align:left;padding:2px 6px">System</th>'
                       '<th>Chern</th><th>Z2</th><th>AZ Class</th>'
                       '<th>Phase</th><th>Bulk-Boundary</th></tr>')
                for r in results:
                    pc = "#3fb950" if r.get("phase") == "topological" else "#8b949e"
                    z2_str = r.get("z2_classification", "—") or "—"
                    tbl += (f'<tr><td style="padding:2px 6px">{r.get("name", "?")}</td>'
                            f'<td style="text-align:center">{r.get("chern_number", "?")}</td>'
                            f'<td style="text-align:center">{z2_str}</td>'
                            f'<td style="text-align:center">{r.get("az_class", "?")}</td>'
                            f'<td style="text-align:center;color:{pc}">{r.get("phase", "?")}</td>'
                            f'<td style="text-align:center">{"OK" if r.get("bulk_boundary_satisfied") else "FAIL"}</td></tr>')
                tbl += '</table>'
                parts.append(header + tbl)
            continue

        # Bio-AI convergence results
        if "bio" in lab_id or "convergence" in jf.name:
            scenarios = data.get("scenarios", [])
            if scenarios:
                mean_conv = data.get("mean_convergence_score", 0)
                header = (f'<div style="margin-bottom:4px;color:#58a6ff">'
                          f'Bio-AI Convergence — {len(scenarios)} scenarios, '
                          f'mean convergence {mean_conv:.3f} '
                          f'<span style="color:#8b949e;font-size:11px">{ts}</span></div>')
                tbl = ('<table style="width:100%;font-size:11px;border-collapse:collapse">'
                       '<tr style="color:#8b949e"><th style="text-align:left;padding:2px 6px">Scenario</th>'
                       '<th>Convergence</th><th>Verdict</th><th>Tools Used</th></tr>')
                for sc in scenarios:
                    vc = "#3fb950" if sc.get("verdict") == "CONVERGENT" else "#d29922"
                    tools = ", ".join(sc.get("tools_used", [])[:3])
                    if len(sc.get("tools_used", [])) > 3:
                        tools += f" +{len(sc['tools_used']) - 3}"
                    tbl += (f'<tr><td style="padding:2px 6px">{sc.get("name", "?")}</td>'
                            f'<td style="text-align:center">{sc.get("convergence_score", 0):.3f}</td>'
                            f'<td style="text-align:center;color:{vc}">{sc.get("verdict", "?")}</td>'
                            f'<td style="text-align:center;color:#8b949e;font-size:10px">{tools}</td></tr>')
                tbl += '</table>'
                parts.append(header + tbl)
            continue

        # Generic fallback for unknown lab types
        n_results = len(data.get("results", data.get("scenarios", [])))
        parts.append(f'<div style="color:#8b949e">{jf.name}: {n_results} results '
                     f'<span style="font-size:11px">{ts}</span></div>')

    return ''.join(parts) if parts else '<span style="color:#8b949e">No results yet — run lab to generate</span>'


def _build_lab_rows(labs: list) -> list[str]:
    """Build HTML blocks for lab projects (expandable details per lab)."""
    blocks = []
    status_colors = {
        "idea": ("#d29922", "💡"),
        "design": ("#58a6ff", "📐"),
        "prototyping": ("#f0883e", "🔧"),
        "running": ("#3fb950", "🔬"),
        "producing": ("#a371f7", "📊"),
        "paper_feed": ("#f778ba", "📝"),
    }

    for lab in labs:
        status = lab.get("status", "idea")
        color, emoji = status_colors.get(status, ("#8b949e", "?"))

        # Build stage pipeline
        all_stages = ["idea", "design", "prototyping", "running", "producing", "paper_feed"]
        stage_labels = ["Idea", "Design", "Proto", "Run", "Prod", "Paper"]
        stages_complete = lab.get("stages_complete", [])
        current = lab.get("current_stage", "")

        stage_dots = []
        for s, label in zip(all_stages, stage_labels):
            if s in stages_complete:
                stage_dots.append(
                    f'<span title="{s}" style="color:#3fb950;font-size:10px">'
                    f'&#9679; {label}</span>')
            elif s == current:
                stage_dots.append(
                    f'<span title="{s}" style="color:#d29922;font-size:10px">'
                    f'&#9679; {label}</span>')
            else:
                stage_dots.append(
                    f'<span title="{s}" style="color:#30363d;font-size:10px">'
                    f'&#9675; {label}</span>')
        pipeline = ' '.join(stage_dots)

        n_screened = lab.get("n_candidates_screened", 0)
        n_viable = lab.get("n_candidates_viable", 0)
        results_str = f"{n_viable}/{n_screened}" if n_screened else "—"

        n_tools = lab.get("n_tools", 0)
        title = lab.get("title", "?")

        results_html = _load_lab_results(lab)

        blocks.append(
            f'<details style="margin-bottom:8px">'
            f'<summary style="cursor:pointer">'
            f'<span style="color:{color};font-weight:600">{emoji} {status.upper()}</span> '
            f'{title} — {n_tools} tools — {pipeline} — {results_str}'
            f'</summary>'
            f'<div style="padding:8px 16px;font-size:12px;color:#c9d1d9">'
            f'{results_html}'
            f'</div>'
            f'</details>'
        )
    return blocks


def generate_html() -> str:
    status = load_status()
    orch = load_orchestrator()
    candidates = count_candidates()
    open_qs = count_open_questions()
    all_domains = count_domains()
    n_tools = count_tools()
    n_tests = count_tests()
    agents = check_agents()
    escalations = get_escalations()
    results = status.get("domain_results", {})
    logs = recent_log(15)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    graded_list, grade_defs = load_grades()
    labs = load_labs()

    running_count = sum(1 for a in agents if a.get("alive"))
    passing = sum(1 for r in results.values() if r.get("verdict") == "PASS")
    failing = len(results) - passing
    n_published = sum(1 for g in graded_list if g.get("status") == "on_zenodo")

    # 27B model status
    model_running = any(a.get("alive") and "27B" in a.get("name", "") for a in agents)
    # Also check by process name
    if not model_running:
        import subprocess
        try:
            r = subprocess.run(["pgrep", "-f", "Qwen3.5-27B"], capture_output=True)
            model_running = r.returncode == 0
        except Exception:
            pass
    current_domain = status.get("current_domain", "idle")
    current_phase = status.get("current_phase", "idle")
    last_update = status.get("last_update", "")
    if last_update:
        try:
            lu = datetime.fromisoformat(last_update)
            age_min = (datetime.now() - lu).total_seconds() / 60
            age_str = f"{age_min:.0f}m ago" if age_min < 60 else f"{age_min/60:.1f}h ago"
        except Exception:
            age_str = last_update[:16]
    else:
        age_str = "unknown"

    # Orchestrator GPU allocation
    gpu_minutes = orch.get("total_gpu_minutes", {})
    gpu_total = sum(gpu_minutes.values()) or 1

    # Build domain results rows
    pass_rows = []
    fail_rows = []
    for name, r in sorted(results.items()):
        v = r.get("verdict", "?")
        pr = r.get("pass_rate", 0)
        mm = r.get("mean_margin", 0)
        n_pass = r.get("n_pass", 0)
        n_total = r.get("n_total", 0)
        phase = r.get("phase", "")
        phase_badge = f'<span class="badge badge-phase">{phase}</span>' if phase not in ("baseline",) else ""

        row = f"""<tr>
            <td>{name}</td>
            <td><span class="badge {"badge-pass" if v == "PASS" else "badge-fail"}">{v}</span></td>
            <td>{n_pass}/{n_total} ({pr:.0%})</td>
            <td class="{"positive" if mm > 0 else "negative"}">{mm:+.1f}</td>
            <td>{phase_badge}</td>
        </tr>"""
        if v == "PASS":
            pass_rows.append(row)
        else:
            fail_rows.append(row)

    # Build agent rows
    agent_rows = []
    for a in agents:
        if a["alive"] is True:
            status_badge = '<span class="badge badge-running">RUNNING</span>'
        elif a["alive"] is False:
            status_badge = '<span class="badge badge-stopped">STOPPED</span>'
        else:
            status_badge = '<span class="badge badge-info">INFO</span>'
        pid_str = f'PID {a["pid"]}' if a.get("pid") else a.get("note", "")
        agent_rows.append(f"""<tr>
            <td>{status_badge}</td>
            <td>{a["name"]}</td>
            <td>{pid_str}</td>
        </tr>""")

    # Build escalation rows
    esc_rows = []
    for e in escalations:
        reason = e.get("reason", "")
        reason_class = "badge-deep" if "deep" in reason else "badge-total" if "total" in reason else "badge-regression"
        suggestion = e.get("details", {}).get("suggestion", "")
        margin = e.get("details", {}).get("mean_margin", "?")
        pr = e.get("details", {}).get("pass_rate", "?")
        if isinstance(pr, float):
            pr = f"{pr:.0%}"
        if isinstance(margin, float):
            margin = f"{margin:+.1f}"
        ts = e.get("timestamp", "")[:16]
        esc_rows.append(f"""<tr>
            <td>{e.get("domain", "?")}</td>
            <td><span class="badge {reason_class}">{reason}</span></td>
            <td>{margin}</td>
            <td>{pr}</td>
            <td class="suggestion">{suggestion}</td>
        </tr>""")

    # GPU allocation bars
    gpu_bars = []
    agent_configs = {
        "research_runner": ("27B Research", "#4CAF50", 0.5),
        "adapter_trainer": ("4B Adapters", "#2196F3", 0.3),
        "teacher_student": ("Teacher-Student", "#FF9800", 0.2),
        "escalation_handler": ("Claude Code", "#9C27B0", 0.0),
    }
    for key, (label, color, target) in agent_configs.items():
        actual = gpu_minutes.get(key, 0) / gpu_total * 100 if gpu_total > 1 else 0
        target_pct = target * 100
        mins = gpu_minutes.get(key, 0)
        gpu_bars.append(f"""
        <div class="gpu-row">
            <span class="gpu-label">{label}</span>
            <div class="gpu-bar-container">
                <div class="gpu-bar-actual" style="width: {min(actual, 100):.1f}%; background: {color};"></div>
                <div class="gpu-bar-target" style="left: {target_pct:.0f}%;"></div>
            </div>
            <span class="gpu-stats">{actual:.0f}% / {target_pct:.0f}% ({mins:.0f}m)</span>
        </div>""")

    # Log entries
    log_lines = "\n".join(
        f'<div class="log-line">{l[:120]}</div>' for l in logs
    )

    # Untested domains
    tested = set(results.keys())
    untested = [n for n in all_domains if n not in tested]

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>NoetherSolve Research Dashboard</title>
<style>
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{
        font-family: -apple-system, BlinkMacSystemFont, 'SF Mono', 'Menlo', monospace;
        background: #0d1117;
        color: #c9d1d9;
        padding: 20px;
        font-size: 13px;
    }}
    .header {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 16px 20px;
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        margin-bottom: 16px;
    }}
    .header h1 {{ font-size: 18px; color: #58a6ff; font-weight: 600; }}
    .header .time {{ color: #8b949e; font-size: 12px; }}
    .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 16px; }}
    .grid-full {{ grid-column: 1 / -1; }}
    .card {{
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 16px;
    }}
    .card h2 {{
        font-size: 13px;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #8b949e;
        margin-bottom: 12px;
        padding-bottom: 8px;
        border-bottom: 1px solid #21262d;
    }}
    .card.escalation {{ border-color: #f85149; }}
    .card.escalation h2 {{ color: #f85149; }}
    .stats-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; }}
    .stat {{ text-align: center; }}
    .stat .value {{ font-size: 24px; font-weight: 700; color: #58a6ff; }}
    .stat .label {{ font-size: 11px; color: #8b949e; margin-top: 2px; }}
    .stat .value.green {{ color: #3fb950; }}
    .stat .value.red {{ color: #f85149; }}
    .stat .value.yellow {{ color: #d29922; }}
    table {{ width: 100%; border-collapse: collapse; }}
    th {{ text-align: left; padding: 6px 8px; color: #8b949e; font-size: 11px;
         text-transform: uppercase; letter-spacing: 0.5px; border-bottom: 1px solid #21262d; }}
    td {{ padding: 5px 8px; border-bottom: 1px solid #21262d; font-size: 12px; }}
    tr:hover {{ background: #1c2128; }}
    .badge {{
        display: inline-block; padding: 2px 8px; border-radius: 12px;
        font-size: 11px; font-weight: 600;
    }}
    .badge-pass {{ background: #238636; color: #fff; }}
    .badge-fail {{ background: #da3633; color: #fff; }}
    .badge-running {{ background: #238636; color: #fff; }}
    .badge-stopped {{ background: #6e7681; color: #fff; }}
    .badge-info {{ background: #1f6feb; color: #fff; }}
    .badge-phase {{ background: #1f6feb22; color: #58a6ff; border: 1px solid #1f6feb44; }}
    .badge-deep {{ background: #da363333; color: #f85149; border: 1px solid #f8514944; }}
    .badge-total {{ background: #d2992233; color: #d29922; border: 1px solid #d2992244; }}
    .badge-regression {{ background: #da363333; color: #f85149; border: 1px solid #f8514944; }}
    .positive {{ color: #3fb950; }}
    .negative {{ color: #f85149; }}
    .suggestion {{ font-size: 11px; color: #8b949e; max-width: 300px; }}
    .gpu-row {{ display: flex; align-items: center; margin-bottom: 8px; }}
    .gpu-label {{ width: 120px; font-size: 12px; }}
    .gpu-bar-container {{
        flex: 1; height: 16px; background: #21262d; border-radius: 4px;
        position: relative; overflow: visible; margin: 0 12px;
    }}
    .gpu-bar-actual {{
        height: 100%; border-radius: 4px; transition: width 0.5s;
        min-width: 2px;
    }}
    .gpu-bar-target {{
        position: absolute; top: -2px; bottom: -2px; width: 2px;
        background: #c9d1d966;
    }}
    .gpu-stats {{ font-size: 11px; color: #8b949e; width: 140px; text-align: right; }}
    .log-line {{
        font-family: 'SF Mono', 'Menlo', 'Courier New', monospace;
        font-size: 11px; padding: 2px 0; color: #8b949e;
        white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
    }}
    .pass-rate-bar {{
        display: inline-block; height: 10px; border-radius: 2px; margin-right: 6px;
        vertical-align: middle;
    }}
    .summary-bar {{
        display: flex; height: 24px; border-radius: 4px; overflow: hidden;
        margin-top: 8px;
    }}
    .summary-bar .pass {{ background: #238636; }}
    .summary-bar .fail {{ background: #da3633; }}
    .summary-text {{ margin-top: 6px; font-size: 12px; color: #8b949e; }}
    details {{ border-radius: 8px; }}
    details summary {{
        cursor: pointer; font-size: 16px; font-weight: 600;
        padding: 12px 0 8px 0; color: #e6edf3; list-style: none;
        display: flex; align-items: center; gap: 8px;
    }}
    details summary::-webkit-details-marker {{ display: none; }}
    details summary::before {{
        content: "▶"; font-size: 11px; color: #8b949e;
        transition: transform 0.15s ease;
    }}
    details[open] summary::before {{ transform: rotate(90deg); }}
    @keyframes pulse {{
        0%, 100% {{ opacity: 1; }}
        50% {{ opacity: 0.4; }}
    }}
</style>
</head>
<body>

<div class="header">
    <h1>NoetherSolve Research Dashboard</h1>
    <div class="time">Last refresh: {now} &nbsp; (auto-refresh 10s)</div>
</div>

<!-- 27B Model Status Banner -->
<div style="background:{'#0f291a' if model_running else '#2a1215'};border:1px solid {'#238636' if model_running else '#da3633'};border-radius:8px;padding:14px 20px;margin-bottom:16px;display:flex;align-items:center;gap:20px;flex-wrap:wrap">
    <div style="display:flex;align-items:center;gap:10px">
        <div style="width:12px;height:12px;border-radius:50%;background:{'#3fb950' if model_running else '#f85149'};{'animation:pulse 2s infinite' if model_running else ''}"></div>
        <span style="font-size:15px;font-weight:700;color:{'#3fb950' if model_running else '#f85149'}">
            Qwen3.5-27B-4bit {'RUNNING' if model_running else 'STOPPED'}
        </span>
    </div>
    <div style="display:flex;gap:24px;flex-wrap:wrap">
        <div><span style="color:#8b949e;font-size:11px">DOMAIN</span><br><span style="color:#e6edf3;font-size:14px;font-weight:600">{current_domain}</span></div>
        <div><span style="color:#8b949e;font-size:11px">PHASE</span><br><span style="color:#e6edf3;font-size:14px;font-weight:600">{current_phase}</span></div>
        <div><span style="color:#8b949e;font-size:11px">PROGRESS</span><br><span style="color:#e6edf3;font-size:14px;font-weight:600">{passing + failing} domains ({passing} pass, {failing} fail)</span></div>
        <div><span style="color:#8b949e;font-size:11px">UPDATED</span><br><span style="color:#8b949e;font-size:14px">{age_str}</span></div>
    </div>
</div>

<!-- Stats overview -->
<div class="grid">
    <div class="card grid-full">
        <div class="stats-grid">
            <div class="stat">
                <div class="value green">{passing}</div>
                <div class="label">Domains Passing</div>
            </div>
            <div class="stat">
                <div class="value red">{failing}</div>
                <div class="label">Domains Failing</div>
            </div>
            <div class="stat">
                <div class="value">{n_tools}</div>
                <div class="label">MCP Tools</div>
            </div>
            <div class="stat">
                <div class="value">{n_tests}</div>
                <div class="label">Tests</div>
            </div>
            <div class="stat">
                <div class="value yellow">{len(escalations)}</div>
                <div class="label">Escalations</div>
            </div>
            <div class="stat">
                <div class="value">{running_count}</div>
                <div class="label">Agents Running</div>
            </div>
            <div class="stat">
                <div class="value" style="color:#FFD700">{n_published}</div>
                <div class="label">Papers on Zenodo</div>
            </div>
            <div class="stat">
                <div class="value" style="color:#a371f7">{len(labs)}</div>
                <div class="label">Lab Projects</div>
            </div>
        </div>
        <div class="summary-bar">
            <div class="pass" style="width: {passing / max(len(results), 1) * 100:.1f}%"></div>
            <div class="fail" style="width: {failing / max(len(results), 1) * 100:.1f}%"></div>
        </div>
        <div class="summary-text">
            {passing}/{len(results)} domains pass ({passing / max(len(results), 1) * 100:.0f}%)
        </div>
    </div>
</div>

<!-- Agents & GPU -->
<div class="grid">
    <div class="card">
        <details open>
            <summary>Agents</summary>
            <table>
                <tr><th>Status</th><th>Agent</th><th>Info</th></tr>
                {"".join(agent_rows) if agent_rows else '<tr><td colspan="3" style="color:#8b949e">No agents registered</td></tr>'}
            </table>
        </details>
    </div>
    <div class="card">
        <details open>
            <summary>GPU Time Allocation</summary>
            {"".join(gpu_bars)}
            <div style="margin-top: 8px; font-size: 11px; color: #8b949e;">
                Rotations: {orch.get("rotation_count", 0)}
                &nbsp;|&nbsp; Last rebalance: {(orch.get("last_rebalance") or "never")[:16]}
            </div>
        </details>
    </div>
</div>

<!-- Escalations (default open — urgent) -->
{f"""<div class="grid">
    <div class="card escalation grid-full">
        <details open>
            <summary>Escalations ({len(escalations)} need attention)</summary>
            <table>
                <tr><th>Domain</th><th>Reason</th><th>Margin</th><th>Pass Rate</th><th>Suggestion</th></tr>
                {"".join(esc_rows)}
            </table>
        </details>
    </div>
</div>""" if escalations else ''}

<!-- Papers & Discoveries -->
<div class="grid">
    <div class="card grid-full">
        <details open>
            <summary>Papers &amp; Discoveries ({n_published} on Zenodo, {len(graded_list)} total)</summary>
            <table>
                <tr><th>Grade</th><th>Title</th><th>Written</th><th>Pipeline</th><th>Target Venue</th><th>Ready</th><th>DOI</th></tr>
                {"".join(_build_paper_rows(graded_list, grade_defs))}
            </table>
        </details>
    </div>
</div>

<!-- Lab Projects -->
<div class="grid">
    <div class="card grid-full">
        <details open>
            <summary>Lab Projects ({len(labs)} labs, {sum(1 for l in labs if l.get('status') in ('running', 'producing'))} active)</summary>
            <div style="padding:8px 0">
                {"".join(_build_lab_rows(labs)) if labs else '<div style="color:#8b949e;padding:8px">No labs configured</div>'}
            </div>
        </details>
    </div>
</div>

<!-- Domain Results -->
<div class="grid">
    <div class="card grid-full">
        <details>
            <summary>Domain Results &mdash; 27B Oracle ({len(results)} domains)</summary>
            <table>
                <tr><th>Domain</th><th>Verdict</th><th>Pass Rate</th><th>Margin</th><th>Phase</th></tr>
                {"".join(pass_rows)}
                {"".join(fail_rows)}
            </table>
        </details>
    </div>
</div>

<!-- Log -->
<div class="grid">
    <div class="card grid-full">
        <details>
            <summary>Recent Log</summary>
            {log_lines if log_lines else '<div class="log-line">No log entries</div>'}
        </details>
    </div>
</div>

<script>
// Auto-refresh content without resetting dropdown state
setInterval(async () => {{
    try {{
        const resp = await fetch('/');
        const html = await resp.text();
        const parser = new DOMParser();
        const newDoc = parser.parseFromString(html, 'text/html');

        // Save which <details> are open
        const openState = {{}};
        document.querySelectorAll('details').forEach((d, i) => {{
            openState[i] = d.open;
        }});

        // Replace body content
        document.body.innerHTML = newDoc.body.innerHTML;

        // Restore open/closed state
        document.querySelectorAll('details').forEach((d, i) => {{
            if (i in openState) d.open = openState[i];
        }});

        // Update timestamp
        const timeEl = document.querySelector('.time');
        if (timeEl) timeEl.textContent = 'Last refresh: ' + new Date().toLocaleTimeString() + '  (live)';
    }} catch (e) {{}}
}}, 10000);
</script>

</body>
</html>"""
    return html


# ---------------------------------------------------------------------------
# HTTP server
# ---------------------------------------------------------------------------

class DashboardHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            html = generate_html()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(html.encode())
        elif self.path == "/api/status":
            status = load_status()
            status["candidates"] = count_candidates()
            status["escalations"] = get_escalations()
            status["agents"] = check_agents()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(status, indent=2, default=str).encode())
        else:
            self.send_error(404)

    def log_message(self, format, *args):
        pass  # Suppress request logging


# ---------------------------------------------------------------------------
# Terminal mode (legacy)
# ---------------------------------------------------------------------------

def render_terminal():
    status = load_status()
    candidates = count_candidates()
    agents = check_agents()
    escalations = get_escalations()
    results = status.get("domain_results", {})
    passing = sum(1 for r in results.values() if r.get("verdict") == "PASS")
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print(f"\033[1m{'='*70}\033[0m")
    print(f"\033[1m  NoetherSolve Research Dashboard\033[0m          {now}")
    print(f"\033[1m{'='*70}\033[0m")

    print(f"\n\033[1m  Agents\033[0m")
    for a in agents:
        if a["alive"] is True:
            icon = "\033[32mRUNNING\033[0m"
        elif a["alive"] is False:
            icon = "\033[31mSTOPPED\033[0m"
        else:
            icon = "\033[36mINFO\033[0m"
        pid_str = f"PID {a['pid']}" if a.get("pid") else a.get("note", "")
        print(f"    {icon}  {a['name']:<25} {pid_str}")

    print(f"\n\033[1m  Summary\033[0m")
    print(f"    Domains: {passing}/{len(results)} passing | Escalations: {len(escalations)}")
    print(f"    Current: {status.get('current_domain', 'idle')} [{status.get('current_phase', 'idle')}]")

    if escalations:
        print(f"\n\033[1;31m  Escalations ({len(escalations)})\033[0m")
        for e in escalations:
            print(f"    \033[31m{e['domain']:<35}\033[0m {e['reason']}")

    print(f"\n  Dashboard: http://localhost:8050")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="NoetherSolve research dashboard")
    parser.add_argument("--port", type=int, default=8050, help="HTTP port (default 8050)")
    parser.add_argument("--once", action="store_true", help="Write HTML file and exit")
    parser.add_argument("--json", action="store_true", help="JSON output")
    parser.add_argument("--terminal", action="store_true", help="Terminal mode (legacy)")
    args = parser.parse_args()

    if args.json:
        status = load_status()
        status["candidates"] = count_candidates()
        status["open_questions"] = count_open_questions()
        status["n_tools"] = count_tools()
        status["n_tests"] = count_tests()
        status["escalations"] = get_escalations()
        status["runner_alive"] = status.get("pid") and is_pid_running(status["pid"])
        print(json.dumps(status, indent=2, default=str))
        return

    if args.terminal:
        render_terminal()
        return

    if args.once:
        html = generate_html()
        with open(HTML_OUTPUT, "w") as f:
            f.write(html)
        print(f"Dashboard written to {HTML_OUTPUT}")
        return

    # Serve HTTP
    print(f"NoetherSolve Dashboard: http://localhost:{args.port}")
    print(f"Press Ctrl+C to stop\n")
    server = http.server.HTTPServer(("", args.port), DashboardHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nDashboard stopped.")
        server.server_close()


if __name__ == "__main__":
    main()
