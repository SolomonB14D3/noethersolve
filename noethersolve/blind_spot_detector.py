"""Blind spot detection and MCP tool routing.

Detects cross-domain and single-domain blind spots from query text,
recommends appropriate MCP tools.
"""

import json
import os
from dataclasses import dataclass
from typing import List, Optional

BLIND_SPOTS_PATH = os.path.join(os.path.dirname(__file__), "blind_spots.json")


@dataclass
class BlindSpotMatch:
    """A matched blind spot with recommended tools."""
    id: str
    domains: List[str]
    insight: str
    tools: List[str]
    interpretation: str
    match_score: float  # 0-1, based on keyword matches


def load_blind_spots() -> dict:
    """Load blind spots registry."""
    with open(BLIND_SPOTS_PATH) as f:
        return json.load(f)


def detect_blind_spots(query: str) -> List[BlindSpotMatch]:
    """Detect blind spots triggered by query keywords.

    Args:
        query: User question or context

    Returns:
        List of matching blind spots, sorted by match score
    """
    data = load_blind_spots()
    query_lower = query.lower()
    matches = []

    # Check cross-domain connections
    for conn in data.get("cross_domain_connections", []):
        keywords = conn.get("trigger_keywords", [])
        matched = sum(1 for k in keywords if k.lower() in query_lower)
        if matched >= 2:  # Need at least 2 keyword matches
            score = matched / len(keywords)
            matches.append(BlindSpotMatch(
                id=conn["id"],
                domains=conn["domains"],
                insight=conn["insight"],
                tools=conn["tools"],
                interpretation=conn["interpretation"],
                match_score=score
            ))

    # Check single-domain blind spots
    for spot in data.get("single_domain_blind_spots", []):
        keywords = spot.get("trigger_keywords", [])
        matched = sum(1 for k in keywords if k.lower() in query_lower)
        if matched >= 1:
            score = matched / len(keywords)
            matches.append(BlindSpotMatch(
                id=spot["id"],
                domains=[spot["domain"]],
                insight=spot.get("reason", ""),
                tools=spot["tools"],
                interpretation="",
                match_score=score
            ))

    # Sort by score descending
    matches.sort(key=lambda m: m.match_score, reverse=True)
    return matches


def get_recommended_tools(query: str) -> List[str]:
    """Get list of MCP tools recommended for this query.

    Args:
        query: User question

    Returns:
        List of tool names to call
    """
    matches = detect_blind_spots(query)
    tools = []
    for m in matches:
        tools.extend(m.tools)
    # Deduplicate while preserving order
    seen = set()
    unique = []
    for t in tools:
        if t not in seen:
            seen.add(t)
            unique.append(t)
    return unique


def format_blind_spot_warning(matches: List[BlindSpotMatch]) -> str:
    """Format a warning message about detected blind spots.

    Args:
        matches: List of matched blind spots

    Returns:
        Human-readable warning with tool recommendations
    """
    if not matches:
        return ""

    lines = ["⚠️ BLIND SPOT DETECTED - Use MCP tools, not memory:\n"]

    for m in matches:
        lines.append(f"  [{m.id}] {' ↔ '.join(m.domains)}")
        if m.insight:
            lines.append(f"    Insight: {m.insight}")
        if m.tools:
            lines.append(f"    Tools: {', '.join(m.tools)}")
        else:
            lines.append(f"    ⚠️ NO TOOLS AVAILABLE - use fallback protocol")
        if m.interpretation:
            lines.append(f"    Interpretation: {m.interpretation}")
        lines.append("")

    # Add fallback instructions if any match has no tools
    no_tool_matches = [m for m in matches if not m.tools]
    if no_tool_matches:
        lines.append("FALLBACK PROTOCOL (for blind spots without tools):")
        lines.append("  1. State the known insight explicitly (from above)")
        lines.append("  2. Acknowledge this is a model blind spot")
        lines.append("  3. Do NOT fabricate details beyond the insight")
        lines.append("  4. Recommend user verify with domain experts")
        lines.append("")

    return "\n".join(lines)


def list_all_blind_spots(needs_tool_only: bool = False) -> str:
    """List all known blind spots as research opportunities.

    Args:
        needs_tool_only: If True, only show blind spots that need tools built

    Returns:
        Formatted list of blind spots with status and tool ideas
    """
    data = load_blind_spots()
    lines = ["# Known Model Blind Spots\n"]

    # Cross-domain
    lines.append("## Cross-Domain Connections")
    lines.append("(Model fails to connect these separate domains)\n")

    for conn in data.get("cross_domain_connections", []):
        needs_tool = conn.get("needs_tool", False)
        if needs_tool_only and not needs_tool:
            continue

        domains = " ↔ ".join(conn["domains"])
        status = "🔧 NEEDS TOOL" if needs_tool else "✓ Has tools"
        lines.append(f"### {conn['id']}")
        lines.append(f"**Domains:** {domains}")
        lines.append(f"**Status:** {status}")
        lines.append(f"**Insight:** {conn['insight']}")
        if conn.get("tools"):
            lines.append(f"**Tools:** {', '.join(conn['tools'])}")
        if conn.get("tool_idea"):
            lines.append(f"**Tool idea:** {conn['tool_idea']}")
        lines.append(f"**Interpretation:** {conn.get('interpretation', 'N/A')}")
        lines.append("")

    # Single-domain
    if not needs_tool_only:
        lines.append("## Single-Domain Blind Spots")
        lines.append("(Model is miscalibrated within these domains)\n")

        for spot in data.get("single_domain_blind_spots", []):
            lines.append(f"### {spot['id']}")
            lines.append(f"**Domain:** {spot['domain']}")
            lines.append(f"**Reason:** {spot.get('reason', 'N/A')}")
            lines.append(f"**Tools:** {', '.join(spot.get('tools', []))}")
            lines.append("")

    # Summary
    cross = data.get("cross_domain_connections", [])
    single = data.get("single_domain_blind_spots", [])
    needs_tools = sum(1 for c in cross if c.get("needs_tool", False))

    lines.append("## Summary")
    lines.append(f"- Cross-domain blind spots: {len(cross)}")
    lines.append(f"- Single-domain blind spots: {len(single)}")
    lines.append(f"- **Needing tools (research opportunities): {needs_tools}**")

    return "\n".join(lines)


# Quick test
if __name__ == "__main__":
    test_queries = [
        "What's the relationship between deadlock and detailed balance?",
        "Is P = NP proven?",
        "How does PageRank relate to thermodynamic equilibrium?",
        "What's the connection between database isolation and quantum decoherence?",
        "Does RLHF eliminate hallucination?",
    ]

    for q in test_queries:
        print(f"Query: {q}")
        matches = detect_blind_spots(q)
        if matches:
            print(format_blind_spot_warning(matches))
        else:
            print("  No blind spots detected\n")
        print("-" * 60)
