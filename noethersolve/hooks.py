"""
NoetherSolve Hooks — automatic resource monitoring and tool integration.

These hooks are called by Claude Code at various points:
- PreToolUse: Before any tool is called
- PostToolUse: After any tool completes
- Stop: When session ends

The hooks:
1. Monitor resource usage (local vs API)
2. Track tool availability and usage patterns
3. Enforce loving service principles automatically
4. Generate session reports
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict

# Session state file
STATE_DIR = Path.home() / ".noethersolve"
STATE_FILE = STATE_DIR / "session_state.json"
TOOL_REGISTRY = STATE_DIR / "tool_registry.json"
USAGE_LOG = STATE_DIR / "usage_log.jsonl"


def ensure_state_dir():
    """Ensure state directory exists."""
    STATE_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class SessionState:
    """Track state across a Claude Code session."""
    session_id: str = ""
    start_time: str = ""
    tool_calls: int = 0
    local_tool_calls: int = 0
    api_calls: int = 0
    verifications: int = 0
    blind_spot_checks: int = 0
    resources_used: Dict[str, float] = field(default_factory=lambda: {
        "local_compute": 0.0,
        "api_tokens": 0.0,
    })
    tools_used: List[str] = field(default_factory=list)
    domains_touched: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "SessionState":
        return cls(**data)

    def save(self):
        ensure_state_dir()
        with open(STATE_FILE, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls) -> "SessionState":
        ensure_state_dir()
        if STATE_FILE.exists():
            try:
                with open(STATE_FILE) as f:
                    return cls.from_dict(json.load(f))
            except:
                pass
        # New session
        return cls(
            session_id=datetime.now().strftime("%Y%m%d_%H%M%S"),
            start_time=datetime.now().isoformat(),
        )


# Tool classification for resource tracking
LOCAL_TOOLS = {
    # All NoetherSolve MCP tools are local
    "check_vortex_conservation", "check_hamiltonian_system", "check_conjecture",
    "calc_iv_bolus", "calc_oral_dose", "calc_half_life", "calc_steady_state",
    "calc_michaelis_menten", "calc_enzyme_inhibition", "check_complexity_inclusion",
    "check_dimension_physics", "check_llm_claim", "verify_goldbach", "verify_collatz",
    "calc_herd_immunity", "calc_reproduction_number", "calc_co2_forcing",
    "check_drug_interaction", "score_crispr_guide", "audit_dna_sequence",
    "calc_black_scholes", "calc_quorum", "calc_byzantine", "calc_subnet",
    "calc_page_table", "detect_deadlock", "simulate_pid", "analyze_stability",
    "should_check_tool", "decide_with_love", "get_loving_service_principles",
    "get_service_checklist", "get_resource_aware_strategy", "list_free_verification_tools",
    "assess_autonomy_requirements", "get_autonomy_roadmap", "get_llm_metacognition_assessment",
    # Add more as they're created
}

BLIND_SPOT_DOMAINS = {
    "dimension_physics", "intersection_theory", "mathematical_status",
    "recent_discoveries", "llm_benchmarks",
}

# Tools that indicate verification happened
VERIFICATION_TOOLS = {
    "check_", "verify_", "calc_", "audit_", "validate_", "analyze_",
}


def is_local_tool(tool_name: str) -> bool:
    """Check if a tool runs locally (free) vs API (expensive)."""
    # NoetherSolve tools are local
    if tool_name in LOCAL_TOOLS:
        return True
    # MCP tools from noethersolve are local
    if "noethersolve" in tool_name.lower():
        return True
    # Standard Claude Code tools
    if tool_name in {"Read", "Write", "Edit", "Glob", "Grep", "Bash"}:
        return True
    return False


def is_verification_tool(tool_name: str) -> bool:
    """Check if this tool verifies a claim."""
    return any(tool_name.startswith(prefix) for prefix in VERIFICATION_TOOLS)


def get_tool_domain(tool_name: str) -> Optional[str]:
    """Infer domain from tool name."""
    domain_map = {
        "calc_iv": "pharmacokinetics",
        "calc_oral": "pharmacokinetics",
        "calc_half": "pharmacokinetics",
        "calc_michaelis": "enzyme_kinetics",
        "calc_enzyme": "enzyme_kinetics",
        "check_conjecture": "mathematics",
        "check_complexity": "complexity_theory",
        "check_dimension": "dimension_physics",
        "verify_goldbach": "number_theory",
        "calc_herd": "epidemiology",
        "calc_co2": "climate_science",
        "check_drug": "drug_interactions",
        "score_crispr": "genetics",
    }
    for prefix, domain in domain_map.items():
        if tool_name.startswith(prefix):
            return domain
    return None


def pre_tool_use(tool_name: str):
    """
    Called before any tool is used.

    Checks:
    1. Is this a blind spot domain? Log warning if not verifying.
    2. Track resource allocation decision.
    """
    state = SessionState.load()

    # Log the call
    state.tool_calls += 1
    if tool_name not in state.tools_used:
        state.tools_used.append(tool_name)

    # Track domain
    domain = get_tool_domain(tool_name)
    if domain and domain not in state.domains_touched:
        state.domains_touched.append(domain)

    # Check if this is a blind spot domain without verification
    if domain in BLIND_SPOT_DOMAINS and not is_verification_tool(tool_name):
        warning = f"Blind spot domain '{domain}' accessed without verification tool"
        if warning not in state.warnings:
            state.warnings.append(warning)

    state.save()


def post_tool_use(tool_name: str, result: str = ""):
    """
    Called after any tool completes.

    Tracks:
    1. Local vs API resource usage
    2. Verification count
    3. Usage patterns for optimization
    """
    state = SessionState.load()

    # Track resource type
    if is_local_tool(tool_name):
        state.local_tool_calls += 1
        state.resources_used["local_compute"] += 0.01
    else:
        state.api_calls += 1
        state.resources_used["api_tokens"] += 0.02

    # Track verifications
    if is_verification_tool(tool_name):
        state.verifications += 1

    # Track blind spot checks
    domain = get_tool_domain(tool_name)
    if domain in BLIND_SPOT_DOMAINS:
        state.blind_spot_checks += 1

    # Log to usage file
    log_usage(tool_name, domain, is_local_tool(tool_name))

    state.save()


def session_end():
    """
    Called when session ends.

    Generates:
    1. Session report with loving service metrics
    2. Resource usage summary
    3. Recommendations for next session
    """
    state = SessionState.load()

    # Calculate metrics
    total_calls = state.tool_calls or 1
    local_ratio = state.local_tool_calls / total_calls
    verification_ratio = state.verifications / total_calls

    # Generate report
    report = {
        "session_id": state.session_id,
        "duration": f"Started {state.start_time}",
        "metrics": {
            "total_tool_calls": state.tool_calls,
            "local_tool_calls": state.local_tool_calls,
            "api_calls": state.api_calls,
            "verifications": state.verifications,
            "blind_spot_checks": state.blind_spot_checks,
        },
        "ratios": {
            "local_usage": f"{local_ratio:.0%}",
            "verification_rate": f"{verification_ratio:.0%}",
        },
        "resources": state.resources_used,
        "domains_touched": state.domains_touched,
        "warnings": state.warnings,
        "loving_service_alignment": {
            "truth_verified": verification_ratio > 0.3,
            "local_preferred": local_ratio > 0.8,
            "blind_spots_checked": state.blind_spot_checks > 0 if any(d in BLIND_SPOT_DOMAINS for d in state.domains_touched) else True,
        },
    }

    # Save report
    ensure_state_dir()
    report_file = STATE_DIR / f"session_{state.session_id}.json"
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)

    # Print summary
    print("\n" + "=" * 50)
    print("NOETHERSOLVE SESSION REPORT")
    print("=" * 50)
    print(f"Tool calls: {state.tool_calls} ({state.local_tool_calls} local, {state.api_calls} API)")
    print(f"Verification rate: {verification_ratio:.0%}")
    print(f"Local usage: {local_ratio:.0%}")
    if state.warnings:
        print(f"Warnings: {len(state.warnings)}")
        for w in state.warnings[:3]:
            print(f"  - {w}")
    print("=" * 50)

    # Clear state for next session
    if STATE_FILE.exists():
        STATE_FILE.unlink()


def log_usage(tool_name: str, domain: Optional[str], is_local: bool):
    """Append to usage log for long-term analysis."""
    ensure_state_dir()
    entry = {
        "timestamp": datetime.now().isoformat(),
        "tool": tool_name,
        "domain": domain,
        "is_local": is_local,
    }
    with open(USAGE_LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")


def get_tool_registry() -> Dict[str, Any]:
    """Get the current tool registry."""
    ensure_state_dir()
    if TOOL_REGISTRY.exists():
        with open(TOOL_REGISTRY) as f:
            return json.load(f)
    return {"tools": [], "last_updated": None}


def update_tool_registry():
    """
    Update the tool registry with current MCP tools.
    Called by git hooks when new tools are committed.
    """
    try:
        from noethersolve.mcp_server.server import mcp
        tools = list(mcp._tool_manager._tools.keys())
    except:
        tools = list(LOCAL_TOOLS)

    registry = {
        "tools": tools,
        "count": len(tools),
        "last_updated": datetime.now().isoformat(),
        "categories": categorize_tools(tools),
    }

    ensure_state_dir()
    with open(TOOL_REGISTRY, "w") as f:
        json.dump(registry, f, indent=2)

    return registry


def categorize_tools(tools: List[str]) -> Dict[str, List[str]]:
    """Categorize tools by domain."""
    categories = {
        "pharmacokinetics": [],
        "enzyme_kinetics": [],
        "mathematics": [],
        "physics": [],
        "chemistry": [],
        "genetics": [],
        "computing": [],
        "metacognition": [],
        "other": [],
    }

    for tool in tools:
        if "calc_iv" in tool or "calc_oral" in tool or "half_life" in tool:
            categories["pharmacokinetics"].append(tool)
        elif "michaelis" in tool or "enzyme" in tool:
            categories["enzyme_kinetics"].append(tool)
        elif "conjecture" in tool or "verify_" in tool:
            categories["mathematics"].append(tool)
        elif "dimension" in tool or "conservation" in tool:
            categories["physics"].append(tool)
        elif "drug" in tool or "chem" in tool:
            categories["chemistry"].append(tool)
        elif "crispr" in tool or "dna" in tool or "protein" in tool:
            categories["genetics"].append(tool)
        elif "loving" in tool or "metacog" in tool or "autonomy" in tool:
            categories["metacognition"].append(tool)
        elif any(x in tool for x in ["calc_", "check_", "analyze_"]):
            categories["computing"].append(tool)
        else:
            categories["other"].append(tool)

    return {k: v for k, v in categories.items() if v}


def get_session_stats() -> Dict:
    """Get statistics from current session."""
    state = SessionState.load()
    return state.to_dict()


def get_usage_stats() -> Dict:
    """Get long-term usage statistics."""
    if not USAGE_LOG.exists():
        return {"total_calls": 0}

    tool_counts = {}
    domain_counts = {}
    local_count = 0
    total = 0

    with open(USAGE_LOG) as f:
        for line in f:
            try:
                entry = json.loads(line)
                total += 1
                tool = entry.get("tool", "unknown")
                domain = entry.get("domain", "unknown")
                tool_counts[tool] = tool_counts.get(tool, 0) + 1
                if domain:
                    domain_counts[domain] = domain_counts.get(domain, 0) + 1
                if entry.get("is_local", False):
                    local_count += 1
            except:
                continue

    return {
        "total_calls": total,
        "local_calls": local_count,
        "local_ratio": local_count / max(1, total),
        "top_tools": sorted(tool_counts.items(), key=lambda x: -x[1])[:10],
        "top_domains": sorted(domain_counts.items(), key=lambda x: -x[1])[:5],
    }
