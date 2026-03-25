#!/usr/bin/env python3
"""
NoetherSolve Results Dashboard
───────────────────────────────
Reads results/candidates.tsv and generates a self-contained HTML report
at results/dashboard.html with embedded charts.

Usage:
    python dashboard.py                  # write results/dashboard.html
    python dashboard.py --open           # write + open in default browser
    python dashboard.py --out my.html    # custom output path
    python dashboard.py --png            # also save individual PNG charts
"""

import argparse
import base64
import io
import json
import os
import sys
import csv
import textwrap
from datetime import datetime

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("WARNING: matplotlib/numpy not found — charts will be text-only.\n"
          "  pip install matplotlib numpy", file=sys.stderr)

HERE = os.path.dirname(os.path.abspath(__file__))
CANDIDATES_TSV = os.path.join(HERE, "results", "candidates.tsv")
MCP_SERVER_PY = os.path.join(HERE, "noethersolve", "mcp_server", "server.py")
RESEARCH_STATUS = os.path.join(HERE, "results", "research_status.json")

# ── Colour palette ────────────────────────────────────────────────────────────
COLOURS = {
    "DUAL-PASS":              "#2ecc71",   # green
    "QUADRANT3→FLIPPED":      "#27ae60",   # dark green (also a win)
    "ORACLE-FAIL+CHECKER-PASS": "#e67e22", # orange  (open gap)
    "CHECKER-FAIL":           "#e74c3c",   # red
    "ORACLE-FAIL":            "#c0392b",   # dark red
    "pending":                "#95a5a6",   # grey
}
QUADRANT_LABEL = {
    "DUAL-PASS":              "Q1 Dual-Pass",
    "QUADRANT3→FLIPPED":      "Q3→Flipped",
    "ORACLE-FAIL+CHECKER-PASS": "Q3 Open Gap",
    "CHECKER-FAIL":           "Q4 Checker-Fail",
    "ORACLE-FAIL":            "Oracle-Fail",
    "pending":                "Pending",
}

# ─────────────────────────────────────────────────────────────────────────────
# TSV parser
# ─────────────────────────────────────────────────────────────────────────────

def load_candidates(path: str) -> list[dict]:
    """Parse candidates.tsv into a list of row dicts."""
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            # Parse margin: may be "−29.959→+3.987" or "+4.50" or "pending"
            raw_margin = row.get("margin_mean", "").strip()
            margin_start = None
            margin_end   = None
            if "→" in raw_margin:
                parts = raw_margin.split("→")
                try:
                    margin_start = float(parts[0].replace("−", "-").replace("–", "-"))
                    margin_end   = float(parts[1].replace("−", "-").replace("–", "-"))
                except ValueError:
                    pass
            elif raw_margin not in ("pending", "baseline", ""):
                try:
                    margin_start = float(raw_margin.replace("−", "-").replace("–", "-"))
                    margin_end   = margin_start
                except ValueError:
                    pass

            # Parse frac_var from classification column
            frac_var = None
            classification = row.get("classification") or ""
            for token in classification.split(";"):
                token = token.strip()
                if "frac_var=" in token:
                    try:
                        frac_var = float(token.split("frac_var=")[1].split()[0].rstrip(","))
                    except (ValueError, IndexError):
                        pass

            rows.append({
                "timestamp":      row.get("timestamp", ""),
                "hypothesis":     row.get("hypothesis", ""),
                "verdict":        row.get("verdict", "pending").strip(),
                "classification": classification,
                "margin_raw":     raw_margin,
                "margin_start":   margin_start,
                "margin_end":     margin_end,
                "frac_var":       frac_var,
            })
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Chart generators (return PNG bytes encoded as base64 string)
# ─────────────────────────────────────────────────────────────────────────────

def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=130)
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return encoded


def chart_quadrant_donut(rows: list[dict]) -> str:
    """Donut chart of verdict distribution."""
    from collections import Counter
    counts = Counter(r["verdict"] for r in rows)
    labels = list(counts.keys())
    values = list(counts.values())
    colors = [COLOURS.get(l, "#bdc3c7") for l in labels]
    display_labels = [QUADRANT_LABEL.get(l, l) for l in labels]

    fig, ax = plt.subplots(figsize=(7, 5), facecolor="#0d1117")
    ax.set_facecolor("#0d1117")

    wedges, _ = ax.pie(
        values, colors=colors, startangle=90,
        wedgeprops=dict(width=0.55, edgecolor="#0d1117", linewidth=2),
    )
    # Centre label
    total = sum(values)
    ax.text(0, 0, f"{total}\ncandidates", ha="center", va="center",
            fontsize=12, color="white", fontweight="bold")

    ax.legend(
        wedges, [f"{l}  ({v})" for l, v in zip(display_labels, values)],
        loc="lower center", bbox_to_anchor=(0.5, -0.15),
        frameon=False, fontsize=9, labelcolor="white",
        ncol=2,
    )
    ax.set_title("Verdict Distribution", color="white", fontsize=13, pad=12)
    fig.tight_layout()
    return _fig_to_b64(fig)


def chart_margin_bar(rows: list[dict]) -> str:
    """Horizontal bar chart — top failures and top wins (not all candidates)."""
    valid = [r for r in rows if r["margin_start"] is not None]
    if not valid:
        return ""

    # Sort by final margin; show top 15 worst + top 10 best to keep chart readable
    sorted_rows = sorted(valid, key=lambda r: r["margin_end"] if r["margin_end"] is not None else r["margin_start"])
    worst = sorted_rows[:15]
    best = sorted_rows[-10:]
    # Deduplicate in case overlap
    seen = set()
    display = []
    for r in worst + best:
        key = r["hypothesis"]
        if key not in seen:
            seen.add(key)
            display.append(r)
    # Re-sort for display
    display.sort(key=lambda r: r["margin_end"] if r["margin_end"] is not None else r["margin_start"])

    labels   = [textwrap.shorten(r["hypothesis"], 50) for r in display]
    m_start  = [r["margin_start"] for r in display]
    m_end    = [r["margin_end"]   if r["margin_end"] is not None else r["margin_start"]
                for r in display]
    verdicts = [r["verdict"] for r in display]
    colors   = [COLOURS.get(v, "#95a5a6") for v in verdicts]

    n = len(display)
    omitted = len(valid) - n
    fig, ax = plt.subplots(figsize=(9, 5), facecolor="#0d1117")
    ax.set_facecolor("#161b22")
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#30363d")

    y = np.arange(n)
    bar_h = 0.35

    ax.barh(y + bar_h/2, m_start, height=bar_h,
            color=[c + "66" for c in colors], label="baseline")

    has_repair = any(r["margin_end"] != r["margin_start"] for r in display
                     if r["margin_end"] is not None)
    if has_repair:
        ax.barh(y - bar_h/2, m_end, height=bar_h,
                color=colors, label="after adapter")

    ax.axvline(0, color="#586069", linewidth=1, linestyle="--")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7.5, color="#c9d1d9")
    ax.set_xlabel("Oracle margin  (log P(truth) − log P(best distractor))",
                  color="#8b949e", fontsize=9)
    title = "Oracle Margins — Top Failures & Wins"
    if omitted > 0:
        title += f"  ({omitted} middle candidates omitted)"
    ax.set_title(title, color="white", fontsize=12)
    fig.tight_layout()
    if has_repair:
        baseline_patch = mpatches.Patch(color="#2ecc7166", label="baseline")
        adapted_patch  = mpatches.Patch(color="#2ecc71",   label="after adapter")
        fig.legend(handles=[baseline_patch, adapted_patch],
                   loc="upper right", bbox_to_anchor=(0.98, 0.98),
                   bbox_transform=fig.transFigure,
                   frameon=False, fontsize=8, labelcolor="white", ncol=1)
    return _fig_to_b64(fig)


def chart_frac_var_scatter(rows: list[dict]) -> str:
    """frac_var vs oracle margin scatter — the 'quality space'."""
    valid = [r for r in rows
             if r["frac_var"] is not None and r["margin_end"] is not None]
    if not valid:
        return ""

    fv      = np.array([r["frac_var"]   for r in valid])
    mg      = np.array([r["margin_end"] for r in valid])
    colors  = [COLOURS.get(r["verdict"], "#95a5a6") for r in valid]
    labels  = [textwrap.shorten(r["hypothesis"], 40) for r in valid]

    fig, ax = plt.subplots(figsize=(7, 5), facecolor="#0d1117")
    ax.set_facecolor("#161b22")
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#30363d")

    ax.scatter(fv, mg, c=colors, s=60, zorder=3, edgecolors="#30363d", linewidths=0.5, alpha=0.8)

    # Only annotate outliers (extreme margins) to avoid clutter
    if len(valid) > 20:
        mg_sorted = sorted(mg)
        threshold_lo = mg_sorted[min(5, len(mg_sorted)-1)]
        threshold_hi = mg_sorted[max(-5, -len(mg_sorted))]
        for i, (x, y_val, lbl) in enumerate(zip(fv, mg, labels)):
            if y_val <= threshold_lo or y_val >= threshold_hi:
                ax.annotate(lbl, (x, y_val), textcoords="offset points",
                            xytext=(5, 2), fontsize=6, color="#8b949e")
    else:
        for i, (x, y_val, lbl) in enumerate(zip(fv, mg, labels)):
            ax.annotate(lbl, (x, y_val), textcoords="offset points",
                        xytext=(5, 2), fontsize=6.5, color="#8b949e")

    # Target zone shading: frac_var < 5e-3 AND margin > 0
    ax.axhline(0,    color="#586069", linewidth=1, linestyle="--")
    ax.axvline(5e-3, color="#586069", linewidth=1, linestyle="--")
    ax.fill_betweenx([-999, 0], 0, 5e-3, alpha=0.06, color="red")
    ax.fill_betweenx([0, 999],  0, 5e-3, alpha=0.10, color="green",
                     label="target zone (checker PASS + oracle PASS)")

    ax.set_xscale("log")
    ax.set_xlabel("frac_var  σ/|mean|  (lower = better conserved)", color="#8b949e", fontsize=9)
    ax.set_ylabel("Oracle margin (positive = model knows it)", color="#8b949e", fontsize=9)
    ax.set_title("Quality Space: frac_var vs Oracle Margin", color="white", fontsize=13)

    # Legend patches
    patches = [mpatches.Patch(color=c, label=QUADRANT_LABEL.get(v, v))
               for v, c in COLOURS.items() if v in {r["verdict"] for r in valid}]
    patches.append(mpatches.Patch(color="green", alpha=0.3, label="target zone"))
    ax.legend(handles=patches, loc="upper right", frameon=False,
              fontsize=7.5, labelcolor="white")

    ax.set_xlim(max(1e-8, fv.min() * 0.1), max(fv) * 10)
    fig.tight_layout()
    return _fig_to_b64(fig)


def chart_discovery_timeline(rows: list[dict]) -> str:
    """Timeline of discoveries (DUAL-PASS + FLIPPED) — last 30 max."""
    wins = [r for r in rows if r["verdict"] in ("DUAL-PASS", "QUADRANT3→FLIPPED")]
    if not wins:
        return ""

    # Show last 30 to avoid crowding
    if len(wins) > 30:
        wins = wins[-30:]

    fig, ax = plt.subplots(figsize=(7, 5), facecolor="#0d1117")
    ax.set_facecolor("#161b22")
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#30363d")

    margins = [r["margin_end"] or 0 for r in wins]
    bar_colors = [COLOURS.get(r["verdict"], "#2ecc71") for r in wins]

    ax.bar(range(len(wins)), margins, color=bar_colors, edgecolor="#0d1117", width=0.7)
    ax.axhline(0, color="#586069", linewidth=1, linestyle="--")

    # Only show x-tick labels for every Nth to avoid overlap
    step = max(1, len(wins) // 10)
    ax.set_xticks(range(0, len(wins), step))
    ax.set_xticklabels([wins[i]["timestamp"] for i in range(0, len(wins), step)],
                       rotation=30, ha="right", fontsize=8, color="#8b949e")
    ax.set_ylabel("Final oracle margin", color="#8b949e", fontsize=9)
    ax.set_title(f"Discovery Timeline — {len(wins)} Discoveries", color="white", fontsize=13)
    fig.tight_layout()
    return _fig_to_b64(fig)


# ─────────────────────────────────────────────────────────────────────────────
# HTML assembly
# ─────────────────────────────────────────────────────────────────────────────

HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>NoetherSolve Dashboard</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    background: #0d1117;
    color: #c9d1d9;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
    font-size: 14px;
    padding: 24px;
  }}
  h1 {{ color: #e6edf3; font-size: 1.6em; margin-bottom: 4px; }}
  .subtitle {{ color: #8b949e; margin-bottom: 32px; font-size: 0.9em; }}
  .grid {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 24px;
    margin-bottom: 32px;
  }}
  @media (max-width: 900px) {{
    .grid {{ grid-template-columns: 1fr; }}
  }}
  .card {{
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 16px;
  }}
  .card h2 {{ color: #e6edf3; font-size: 1em; margin-bottom: 12px; }}
  .card img {{ width: 100%; height: auto; border-radius: 4px; }}
  .wide {{ grid-column: 1 / -1; }}
  table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 0.82em;
  }}
  th {{
    background: #21262d;
    color: #8b949e;
    padding: 8px 10px;
    text-align: left;
    font-weight: 600;
    border-bottom: 1px solid #30363d;
  }}
  td {{
    padding: 7px 10px;
    border-bottom: 1px solid #21262d;
    vertical-align: top;
    max-width: 340px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }}
  tr:last-child td {{ border-bottom: none; }}
  .badge {{
    display: inline-block;
    padding: 2px 8px;
    border-radius: 20px;
    font-size: 0.78em;
    font-weight: 600;
    color: #0d1117;
  }}
  .pos {{ color: #2ecc71; }}
  .neg {{ color: #e74c3c; }}
  .footer {{
    color: #484f58;
    font-size: 0.78em;
    margin-top: 16px;
    text-align: center;
  }}
  .stat-row {{
    display: flex;
    gap: 16px;
    flex-wrap: wrap;
    margin-bottom: 24px;
  }}
  .stat {{
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 14px 20px;
    min-width: 120px;
  }}
  .stat .val {{ font-size: 1.9em; font-weight: 700; color: #e6edf3; }}
  .stat .lbl {{ color: #8b949e; font-size: 0.82em; margin-top: 2px; }}
</style>
</head>
<body>

<h1>🔬 NoetherSolve Dashboard</h1>
<p class="subtitle">Find what LLMs don't know about what the universe conserves. Then fix it. &nbsp;·&nbsp; Generated {generated_at}</p>

<div class="stat-row">
  <div class="stat"><div class="val">{n_total}</div><div class="lbl">Candidates tested</div></div>
  <div class="stat"><div class="val" style="color:#2ecc71">{n_wins}</div><div class="lbl">Discoveries (PASS or FLIPPED)</div></div>
  <div class="stat"><div class="val" style="color:#e67e22">{n_open}</div><div class="lbl">Open gaps (checker PASS)</div></div>
  <div class="stat"><div class="val" style="color:#e74c3c">{n_fail}</div><div class="lbl">Checker failures</div></div>
  <div class="stat"><div class="val" style="color:#58a6ff">{n_tools}</div><div class="lbl">MCP Tools</div></div>
  <div class="stat"><div class="val">{n_domains}</div><div class="lbl">Domains active</div></div>
</div>

{research_status}

<div class="grid">
  {chart_donut}
  {chart_scatter}
  {chart_timeline}
  {chart_tools}
</div>

<div class="card" style="margin-bottom:24px">
  <h2>14B Oracle Margins — Top Failures &amp; Wins (Qwen3-14B-Base + adapters)</h2>
  {chart_margin}
</div>

<div class="card" style="margin-bottom:24px">
  <h2>MCP Tools ({n_tools} total)</h2>
  <div style="max-height:400px;overflow-y:auto">
  <table>
    <thead>
      <tr>
        <th>Tool Name</th>
        <th>Description</th>
        <th>Date Built</th>
      </tr>
    </thead>
    <tbody>
      {tool_rows}
    </tbody>
  </table>
  </div>
</div>

<div class="card">
  <h2>All Candidates ({n_total})</h2>
  <div style="max-height:500px;overflow-y:auto">
  <table>
    <thead>
      <tr>
        <th>Date</th>
        <th>Hypothesis</th>
        <th>frac_var</th>
        <th>Margin</th>
        <th>Verdict</th>
        <th>Notes</th>
      </tr>
    </thead>
    <tbody>
      {table_rows}
    </tbody>
  </table>
  </div>
</div>

<p class="footer">NoetherSolve · Built on the STEM Truth Oracle (Paper 9, DOI: 10.5281/zenodo.19005729) and Snap-On Communication Modules (Paper 8, DOI: 10.5281/zenodo.18902616)</p>
</body>
</html>
"""


VERDICT_COLOURS_CSS = {
    "DUAL-PASS":                "#2ecc71",
    "QUADRANT3→FLIPPED":        "#27ae60",
    "ORACLE-FAIL+CHECKER-PASS": "#e67e22",
    "CHECKER-FAIL":             "#e74c3c",
    "ORACLE-FAIL":              "#c0392b",
    "pending":                  "#586069",
}


def _img_card(title: str, b64: str, wide: bool = False) -> str:
    if not b64:
        return ""
    cls = "card wide" if wide else "card"
    return f'<div class="{cls}"><h2>{title}</h2><img src="data:image/png;base64,{b64}" alt="{title}"></div>'


def _verdict_badge(verdict: str) -> str:
    colour = VERDICT_COLOURS_CSS.get(verdict, "#586069")
    label  = QUADRANT_LABEL.get(verdict, verdict)
    return f'<span class="badge" style="background:{colour}">{label}</span>'


def _margin_cell(row: dict) -> str:
    raw = row["margin_raw"]
    if not raw or raw in ("pending", "baseline", ""):
        return f'<td style="color:#586069">{raw or "—"}</td>'
    if row["margin_end"] is not None and row["margin_end"] > 0:
        cls = "pos"
    elif row["margin_end"] is not None:
        cls = "neg"
    else:
        cls = ""
    return f'<td class="{cls}">{raw}</td>'


def build_table_rows(rows: list[dict]) -> str:
    parts = []
    for r in rows:
        fv_str = f"{r['frac_var']:.2e}" if r["frac_var"] is not None else "—"
        notes  = r["classification"]
        # Truncate notes for display
        if len(notes) > 90:
            notes = notes[:87] + "…"
        parts.append(
            f"<tr>"
            f"<td>{r['timestamp']}</td>"
            f'<td style="max-width:260px;white-space:normal">{r["hypothesis"]}</td>'
            f"<td>{fv_str}</td>"
            f"{_margin_cell(r)}"
            f"<td>{_verdict_badge(r['verdict'])}</td>"
            f'<td style="color:#8b949e;white-space:normal;max-width:280px">{notes}</td>'
            f"</tr>"
        )
    return "\n      ".join(parts)


def build_stats(rows: list[dict]) -> dict:
    n_total  = len(rows)
    n_wins   = sum(1 for r in rows if r["verdict"] in ("DUAL-PASS", "QUADRANT3→FLIPPED"))
    n_open   = sum(1 for r in rows if r["verdict"] == "ORACLE-FAIL+CHECKER-PASS")
    n_fail   = sum(1 for r in rows if r["verdict"] == "CHECKER-FAIL")
    # Infer domains from hypothesis text
    domains  = set()
    for r in rows:
        h = r["hypothesis"].lower()
        if "vortex" in h or "vp" in h.split()[0]:
            domains.add("point-vortex")
        elif "figure" in h or "3-body" in h or "r12" in h or "choreograph" in h:
            domains.add("figure-8 3-body")
        elif "kinetic" in h or "pilot" in h:
            domains.add("kinetic-energy pilot")
        else:
            domains.add("unknown")
    return dict(n_total=n_total, n_wins=n_wins, n_open=n_open,
                n_fail=n_fail, n_domains=len(domains))


def load_research_status() -> dict | None:
    """Load the 27B research runner's current status."""
    if not os.path.exists(RESEARCH_STATUS):
        return None
    try:
        with open(RESEARCH_STATUS, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def build_research_status_html(status: dict | None) -> str:
    """Build HTML for the 27B research runner status section."""
    if not status:
        return '<div class="card"><h2>27B Research Runner</h2><p style="color:#586069">Not running — no research_status.json</p></div>'

    domain = status.get("current_domain", "idle")
    phase = status.get("current_phase", "—")
    completed = status.get("completed_domains", [])
    results = status.get("domain_results", {})
    last_update = status.get("last_update", "unknown")

    # Count verdicts
    pass_count = sum(1 for v in results.values() if v.get("verdict") == "PASS")
    fail_count = sum(1 for v in results.values() if v.get("verdict") == "FAIL")
    total = len(results)

    # Build domain results table (compact)
    domain_rows = []
    for dom_name, info in sorted(results.items()):
        verdict = info.get("verdict", "?")
        margin = info.get("margin_avg")
        margin_str = f"{margin:.1f}" if margin is not None else "—"
        color = "#2ecc71" if verdict == "PASS" else "#e74c3c"
        domain_rows.append(
            f'<tr>'
            f'<td>{dom_name}</td>'
            f'<td style="color:{color};font-weight:600">{verdict}</td>'
            f'<td>{margin_str}</td>'
            f'</tr>'
        )

    domain_table = "\n".join(domain_rows) if domain_rows else (
        '<tr><td colspan="3" style="color:#586069">No domain results yet</td></tr>'
    )

    return f'''<div class="card">
  <h2>27B Research Runner (Qwen3.5-27B-4bit)</h2>
  <div style="display:flex;gap:16px;flex-wrap:wrap;margin-bottom:12px">
    <div class="stat" style="padding:10px 14px"><div class="val" style="font-size:1.2em">{domain}</div><div class="lbl">Current Domain</div></div>
    <div class="stat" style="padding:10px 14px"><div class="val" style="font-size:1.2em">{phase}</div><div class="lbl">Phase</div></div>
    <div class="stat" style="padding:10px 14px"><div class="val" style="font-size:1.2em">{total}</div><div class="lbl">Domains Scanned</div></div>
    <div class="stat" style="padding:10px 14px"><div class="val" style="font-size:1.2em;color:#2ecc71">{pass_count}</div><div class="lbl">Passing</div></div>
    <div class="stat" style="padding:10px 14px"><div class="val" style="font-size:1.2em;color:#e74c3c">{fail_count}</div><div class="lbl">Failing</div></div>
  </div>
  <p style="color:#8b949e;font-size:0.82em;margin-bottom:8px">Last update: {last_update}</p>
  <div style="max-height:300px;overflow-y:auto">
  <table>
    <thead><tr><th>Domain</th><th>Verdict</th><th>Margin</th></tr></thead>
    <tbody>{domain_table}</tbody>
  </table>
  </div>
</div>'''


def load_mcp_tools() -> list[dict]:
    """Extract MCP tool names, descriptions, and creation dates from server.py + git."""
    import re
    import subprocess

    if not os.path.exists(MCP_SERVER_PY):
        return []

    with open(MCP_SERVER_PY, encoding="utf-8") as f:
        content = f.read()

    # Extract tool name and first line of docstring
    pattern = r'@mcp\.tool\(\)\ndef (\w+)\([^)]*\)[^:]*:\s*"""([^\n]*)'
    matches = list(re.finditer(pattern, content))

    tools = []
    for m in matches:
        name = m.group(1)
        doc = m.group(2).strip()
        line_num = content[:m.start()].count('\n') + 2
        tools.append({"name": name, "doc": doc, "line": line_num, "date": ""})

    # Try to get dates from git blame
    try:
        result = subprocess.run(
            ["git", "blame", "--line-porcelain", "noethersolve/mcp_server/server.py"],
            capture_output=True, text=True, cwd=HERE, timeout=30,
        )
        if result.returncode == 0:
            line_dates = {}
            current_line = None
            for bl in result.stdout.split('\n'):
                if bl.startswith('author-time '):
                    ts = int(bl.split()[1])
                    dt = datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                    if current_line:
                        line_dates[current_line] = dt
                parts = bl.split()
                if len(parts) >= 3 and len(parts[0]) == 40:
                    try:
                        current_line = int(parts[2])
                    except ValueError:
                        pass

            for t in tools:
                t["date"] = line_dates.get(t["line"], "")
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    return tools


def build_tool_table(tools: list[dict]) -> str:
    """Build HTML table rows for MCP tools."""
    if not tools:
        return '<tr><td colspan="3" style="color:#586069">No tools found</td></tr>'

    # Group by date
    parts = []
    for t in tools:
        name = t["name"]
        doc = t["doc"][:80] + ("..." if len(t["doc"]) > 80 else "")
        date = t["date"] or "unknown"
        parts.append(
            f'<tr>'
            f'<td style="font-family:monospace;font-size:0.82em;color:#58a6ff">{name}</td>'
            f'<td>{doc}</td>'
            f'<td style="color:#8b949e;white-space:nowrap">{date}</td>'
            f'</tr>'
        )
    return "\n      ".join(parts)


def chart_tools_by_date(tools: list[dict]) -> str:
    """Bar chart of tools added per date."""
    if not HAS_MPL or not tools:
        return ""

    from collections import Counter
    dates = Counter(t["date"] for t in tools if t["date"])
    if not dates:
        return ""

    sorted_dates = sorted(dates.keys())
    counts = [dates[d] for d in sorted_dates]

    fig, ax = plt.subplots(figsize=(7, 5), facecolor="#0d1117")
    ax.set_facecolor("#161b22")
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#30363d")

    bars = ax.bar(range(len(sorted_dates)), counts, color="#58a6ff", edgecolor="#0d1117")

    # Add count labels on bars
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                str(count), ha="center", va="bottom", color="white", fontsize=9)

    ax.set_xticks(range(len(sorted_dates)))
    ax.set_xticklabels(sorted_dates, rotation=30, ha="right", fontsize=8, color="#8b949e")
    ax.set_ylabel("Tools added", color="#8b949e", fontsize=9)
    ax.set_title("MCP Tools Built by Date", color="white", fontsize=13)
    fig.tight_layout()
    return _fig_to_b64(fig)


def generate_dashboard(out_path: str, png_dir: str | None = None) -> None:
    rows = load_candidates(CANDIDATES_TSV)
    if not rows:
        print(f"  No candidates found in {CANDIDATES_TSV}", file=sys.stderr)
        return

    stats = build_stats(rows)
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")

    # Load MCP tools
    tools = load_mcp_tools()
    stats["n_tools"] = len(tools)

    # Load 27B research runner status
    research = load_research_status()

    if HAS_MPL:
        b64_donut    = chart_quadrant_donut(rows)
        b64_margin   = chart_margin_bar(rows)
        b64_scatter  = chart_frac_var_scatter(rows)
        b64_timeline = chart_discovery_timeline(rows)
        b64_tools    = chart_tools_by_date(tools)

        if png_dir:
            os.makedirs(png_dir, exist_ok=True)
            for name, b64 in [("quadrant_donut.png", b64_donut),
                               ("margin_bar.png",     b64_margin),
                               ("frac_var_scatter.png", b64_scatter),
                               ("discovery_timeline.png", b64_timeline),
                               ("tools_by_date.png", b64_tools)]:
                if b64:
                    with open(os.path.join(png_dir, name), "wb") as f:
                        f.write(base64.b64decode(b64))
            print(f"  PNGs saved to: {png_dir}")

        chart_donut    = _img_card("Verdict Distribution", b64_donut)
        chart_scatter  = _img_card("Quality Space", b64_scatter)
        chart_margin   = f'<img src="data:image/png;base64,{b64_margin}" alt="margins" style="width:100%;height:auto;border-radius:4px">' if b64_margin else "<p style='color:#586069'>No margin data</p>"
        chart_timeline = _img_card("Discovery Timeline", b64_timeline)
        chart_tools    = _img_card("Tools Built by Date", b64_tools) if b64_tools else ""
    else:
        chart_donut    = '<div class="card"><p style="color:#586069">Install matplotlib for charts.</p></div>'
        chart_scatter  = ""
        chart_margin   = "<p style='color:#586069'>Install matplotlib for charts.</p>"
        chart_timeline = ""
        chart_tools    = ""

    html = HTML_TEMPLATE.format(
        generated_at    = generated_at,
        research_status = build_research_status_html(research),
        chart_donut     = chart_donut,
        chart_scatter   = chart_scatter,
        chart_margin    = chart_margin,
        chart_timeline  = chart_timeline,
        chart_tools     = chart_tools,
        table_rows      = build_table_rows(rows),
        tool_rows       = build_tool_table(tools),
        **stats,
    )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  Dashboard written: {out_path}  ({len(rows)} candidates)")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate NoetherSolve results dashboard",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--out",  default=os.path.join(HERE, "results", "dashboard.html"),
                        help="Output HTML path")
    parser.add_argument("--open", action="store_true", help="Open dashboard in browser after writing")
    parser.add_argument("--png",  action="store_true", help="Also save individual PNG charts")
    args = parser.parse_args()

    png_dir = os.path.join(HERE, "results", "charts") if args.png else None
    generate_dashboard(args.out, png_dir)

    if args.open:
        import webbrowser
        webbrowser.open(f"file://{os.path.abspath(args.out)}")
        print("  Opened in browser.")


if __name__ == "__main__":
    main()
