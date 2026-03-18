"""
Metacognitive Control — deciding when thinking about thinking is worth the cost.

This module implements the CONTROL side of Nelson & Narens' metacognition model.
The metacognition.py module handles MONITORING (measuring calibration, unknown recall).
This module handles CONTROL (deciding when to engage metacognitive processes).

Core insight: Metacognition has a cost (compute, latency, energy). The optimal
strategy isn't "always reflect" or "never reflect" — it's knowing when the
expected value of metacognition exceeds its cost.

Decision framework:
    EV(metacognition) = P(catching_error) × error_cost_avoided - metacognitive_cost

    If EV > 0: engage metacognition
    If EV < 0: act directly

    With override rules:
    - High stakes + any uncertainty → always check
    - Known blind spot domain → always check (use tools)
    - Very high confidence + low stakes → skip

Integration with NoetherSolve tools:
    The detect_blind_spots() tool identifies domains where the model is likely wrong.
    This controller uses that signal to route to verified tools when needed,
    avoiding both over-checking (wasting compute) and under-checking (making errors).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum


class ResourceType(Enum):
    """Types of resources that metacognitive actions consume."""
    LOCAL_COMPUTE = "local_compute"    # GPU/CPU cycles - basically free
    API_TOKENS = "api_tokens"          # Claude/GPT tokens - expensive/limited
    LATENCY = "latency"                # Wall-clock time - user patience
    MEMORY = "memory"                  # RAM/context window


@dataclass
class ResourceCost:
    """Cost of an action in different resource types."""
    local_compute: float = 0.0    # 0-1 scale, usually ~0 (free)
    api_tokens: float = 0.0       # 0-1 scale, expensive
    latency: float = 0.0          # 0-1 scale, seconds normalized
    memory: float = 0.0           # 0-1 scale, context used

    def total_weighted(self, weights: Dict[str, float]) -> float:
        """Compute weighted total cost."""
        return (
            self.local_compute * weights.get("local_compute", 0.0) +
            self.api_tokens * weights.get("api_tokens", 1.0) +  # Default: API tokens are expensive
            self.latency * weights.get("latency", 0.1) +
            self.memory * weights.get("memory", 0.1)
        )


@dataclass
class ResourceBudget:
    """Available resources for a session."""
    local_compute: float = 1.0     # Basically unlimited
    api_tokens: float = 0.1        # Very limited (e.g., weekly quota)
    latency: float = 1.0           # 1.0 = user is patient
    memory: float = 0.5            # Context window constraint

    # Weights for cost-benefit: higher = more expensive to use
    weights: Dict[str, float] = field(default_factory=lambda: {
        "local_compute": 0.01,  # Almost free
        "api_tokens": 10.0,     # Very expensive - limited weekly
        "latency": 0.5,         # Moderate cost
        "memory": 0.3,          # Some cost
    })

    def can_afford(self, cost: ResourceCost) -> bool:
        """Check if we have budget for this action."""
        return (
            cost.local_compute <= self.local_compute and
            cost.api_tokens <= self.api_tokens and
            cost.latency <= self.latency and
            cost.memory <= self.memory
        )

    def spend(self, cost: ResourceCost) -> bool:
        """Spend resources. Returns False if can't afford."""
        if not self.can_afford(cost):
            return False
        self.local_compute -= cost.local_compute
        self.api_tokens -= cost.api_tokens
        self.latency -= cost.latency
        self.memory -= cost.memory
        return True

    def effective_cost(self, cost: ResourceCost) -> float:
        """Weighted cost considering what's scarce."""
        return cost.total_weighted(self.weights)


class MetacognitiveAction(Enum):
    """What the controller recommends."""
    ACT_DIRECTLY = "act_directly"           # Just answer, don't overthink
    VERIFY_CONFIDENCE = "verify_confidence"  # Quick confidence check
    CHECK_TOOL = "check_tool"               # Route to verified tool
    DEEP_REFLECTION = "deep_reflection"     # Full metacognitive analysis
    DEFER = "defer"                         # Admit uncertainty, ask for help


@dataclass
class MetacognitiveDecision:
    """Output of the metacognitive controller."""
    action: MetacognitiveAction
    reasoning: str
    expected_value: float           # EV of taking metacognitive action
    confidence_in_decision: float   # How confident is this recommendation
    suggested_tool: Optional[str] = None  # If CHECK_TOOL, which tool
    energy_cost: float = 0.0        # Estimated compute cost (0-1 scale)

    def __str__(self) -> str:
        lines = [
            f"Decision: {self.action.value}",
            f"Reasoning: {self.reasoning}",
            f"Expected Value: {self.expected_value:+.3f}",
            f"Energy Cost: {self.energy_cost:.3f}",
        ]
        if self.suggested_tool:
            lines.append(f"Suggested Tool: {self.suggested_tool}")
        return "\n".join(lines)


@dataclass
class TaskContext:
    """Context about the current task for decision-making."""
    domain: str                     # What domain is this? (e.g., "pharmacokinetics")
    stakes: float                   # How bad is an error? 0=trivial, 1=catastrophic
    confidence: float               # Model's stated confidence 0-1
    is_known_blind_spot: bool       # Is this a domain where model is often wrong?
    has_verified_tool: bool         # Is there a NoetherSolve tool for this?
    response_latency_budget: float  # How much time do we have? (seconds)
    session_energy_remaining: float # Energy budget remaining (0-1)

    def __post_init__(self):
        # Clamp values to valid ranges
        self.stakes = max(0.0, min(1.0, self.stakes))
        self.confidence = max(0.0, min(1.0, self.confidence))
        self.session_energy_remaining = max(0.0, min(1.0, self.session_energy_remaining))


@dataclass
class CalibrationHistory:
    """Track past performance for better metacognitive decisions."""
    domain_accuracy: Dict[str, float] = field(default_factory=dict)
    domain_calibration: Dict[str, float] = field(default_factory=dict)  # ECE per domain
    recent_errors: List[str] = field(default_factory=list)  # Last N error domains
    total_checks: int = 0
    checks_that_caught_errors: int = 0

    def get_domain_reliability(self, domain: str) -> float:
        """How reliable is the model in this domain? (0-1)"""
        if domain in self.domain_accuracy:
            return self.domain_accuracy[domain]
        # Default: assume moderate reliability
        return 0.7

    def get_check_hit_rate(self) -> float:
        """What fraction of metacognitive checks caught errors?"""
        if self.total_checks == 0:
            return 0.5  # Prior: 50% of checks are useful
        return self.checks_that_caught_errors / self.total_checks

    def record_check_outcome(self, domain: str, caught_error: bool):
        """Update history after a metacognitive check."""
        self.total_checks += 1
        if caught_error:
            self.checks_that_caught_errors += 1
            self.recent_errors.append(domain)
            # Keep only last 20 errors
            self.recent_errors = self.recent_errors[-20:]


# Cost estimates for different metacognitive actions (relative scale 0-1)
# Legacy: single-number costs for backward compatibility
ACTION_COSTS = {
    MetacognitiveAction.ACT_DIRECTLY: 0.0,
    MetacognitiveAction.VERIFY_CONFIDENCE: 0.05,
    MetacognitiveAction.CHECK_TOOL: 0.15,
    MetacognitiveAction.DEEP_REFLECTION: 0.3,
    MetacognitiveAction.DEFER: 0.02,
}

# Resource-aware costs: different actions have different resource profiles
ACTION_RESOURCE_COSTS = {
    MetacognitiveAction.ACT_DIRECTLY: ResourceCost(
        local_compute=0.0, api_tokens=0.0, latency=0.0, memory=0.0
    ),
    MetacognitiveAction.VERIFY_CONFIDENCE: ResourceCost(
        local_compute=0.05, api_tokens=0.0, latency=0.02, memory=0.01
    ),
    MetacognitiveAction.CHECK_TOOL: ResourceCost(
        local_compute=0.02, api_tokens=0.0, latency=0.05, memory=0.01
    ),  # NoetherSolve tools are LOCAL - no API cost!
    MetacognitiveAction.DEEP_REFLECTION: ResourceCost(
        local_compute=0.1, api_tokens=0.05, latency=0.1, memory=0.05
    ),  # May use API for reasoning
    MetacognitiveAction.DEFER: ResourceCost(
        local_compute=0.0, api_tokens=0.0, latency=0.01, memory=0.0
    ),
}


class ToolType(Enum):
    """Whether a tool runs locally or uses API."""
    LOCAL = "local"      # NoetherSolve tools - free
    API = "api"          # Requires API call - expensive
    HYBRID = "hybrid"    # Can run either way

# Tool type classification - LOCAL means uses NoetherSolve (free), API means external
TOOL_TYPES = {
    # NoetherSolve tools (all LOCAL - no API cost)
    "calc_iv_bolus": ToolType.LOCAL,
    "calc_oral_dose": ToolType.LOCAL,
    "calc_michaelis_menten": ToolType.LOCAL,
    "check_complexity_inclusion": ToolType.LOCAL,
    "verify_goldbach": ToolType.LOCAL,
    "check_conjecture": ToolType.LOCAL,
    "check_dimension_physics": ToolType.LOCAL,
    "check_llm_claim": ToolType.LOCAL,
    "calc_herd_immunity": ToolType.LOCAL,
    "calc_co2_forcing": ToolType.LOCAL,
    "check_drug_interaction": ToolType.LOCAL,
    "score_crispr_guide": ToolType.LOCAL,
    # External API tools (expensive)
    "web_search": ToolType.API,
    "external_llm_query": ToolType.API,
    "database_lookup": ToolType.HYBRID,  # Could cache locally
}

# Domains where NoetherSolve has verified tools
TOOL_DOMAINS = {
    "pharmacokinetics": ["calc_iv_bolus", "calc_oral_dose", "calc_half_life", "calc_steady_state"],
    "enzyme_kinetics": ["calc_michaelis_menten", "calc_enzyme_inhibition", "calc_cooperativity"],
    "quantum_mechanics": ["calc_particle_in_box", "calc_hydrogen_energy", "calc_tunneling"],
    "organic_chemistry": ["analyze_molecule", "predict_reaction_selectivity", "check_baldwin_rules"],
    "complexity_theory": ["check_complexity_inclusion", "check_completeness", "check_proof_barriers"],
    "number_theory": ["verify_goldbach", "verify_collatz", "check_abc_triple"],
    "conservation_laws": ["check_vortex_conservation", "check_hamiltonian_system", "check_em_conservation"],
    "cryptography": ["calc_security_level", "calc_birthday_bound", "calc_cipher_mode"],
    "distributed_systems": ["calc_quorum", "calc_byzantine", "calc_vector_clock"],
    "epidemiology": ["calc_herd_immunity", "calc_reproduction_number", "calc_sir_model"],
    "climate_science": ["calc_co2_forcing", "calc_climate_sensitivity", "calc_greenhouse_effect"],
    "drug_interactions": ["check_drug_interaction", "get_drug_cyp_profile", "predict_ddi_auc_change"],
    "genetics": ["score_crispr_guide", "audit_dna_sequence", "predict_protein_aggregation"],
    "llm_claims": ["check_llm_claim"],
    "conjectures": ["check_conjecture"],
    "dimension_physics": ["check_dimension_physics"],
}

# Known blind spots where models are frequently wrong
BLIND_SPOT_DOMAINS = {
    "dimension_physics",      # 100% blind (2D vs 3D physics)
    "intersection_theory",    # Deepest gap (-27.6 margin)
    "mathematical_status",    # 4% pass rate on "is X proven?"
    "recent_discoveries",     # Post-training-cutoff
    "llm_benchmarks",         # Models confabulate scores
}


def compute_error_probability(context: TaskContext, history: CalibrationHistory) -> float:
    """
    Estimate P(error) given context and history.

    This is the key quantity: if we act directly, how likely are we to be wrong?
    """
    base_error_rate = 1.0 - history.get_domain_reliability(context.domain)

    # Adjust for stated confidence (with skepticism about calibration)
    # Overconfidence is common, so we discount high confidence
    confidence_adjustment = context.confidence * 0.5  # 50% weight on stated confidence

    # Blind spots have higher error rates
    if context.is_known_blind_spot:
        base_error_rate = max(base_error_rate, 0.7)  # At least 70% error rate

    # Recent errors in this domain increase estimate
    recent_domain_errors = sum(1 for d in history.recent_errors if d == context.domain)
    recency_adjustment = min(0.2, recent_domain_errors * 0.05)

    error_prob = base_error_rate * (1 - confidence_adjustment) + recency_adjustment
    return max(0.01, min(0.99, error_prob))


def compute_metacognitive_ev(
    context: TaskContext,
    action: MetacognitiveAction,
    history: CalibrationHistory
) -> float:
    """
    Compute expected value of a metacognitive action.

    EV = P(catching_error) × error_cost_avoided - action_cost

    For CHECK_TOOL, we assume the tool is always correct (verified).
    For other actions, effectiveness depends on calibration history.
    """
    action_cost = ACTION_COSTS[action]

    if action == MetacognitiveAction.ACT_DIRECTLY:
        # No metacognitive cost, but we eat any errors
        return 0.0

    error_prob = compute_error_probability(context, history)
    error_cost = context.stakes  # Stakes = cost of error

    if action == MetacognitiveAction.CHECK_TOOL:
        # Tool catches 100% of errors in its domain
        catch_prob = 1.0 if context.has_verified_tool else 0.0
    elif action == MetacognitiveAction.VERIFY_CONFIDENCE:
        # Quick check catches maybe 30% of errors
        catch_prob = 0.3
    elif action == MetacognitiveAction.DEEP_REFLECTION:
        # Deep reflection catches maybe 50% of errors
        catch_prob = 0.5
    elif action == MetacognitiveAction.DEFER:
        # Deferring avoids the error entirely (we don't answer wrong)
        catch_prob = 1.0
    else:
        catch_prob = 0.0

    # EV = P(error) × P(catch|check) × error_cost - check_cost
    ev = error_prob * catch_prob * error_cost - action_cost

    return ev


def should_engage_metacognition(
    context: TaskContext,
    history: Optional[CalibrationHistory] = None
) -> MetacognitiveDecision:
    """
    Core decision function: should we think about thinking, or just act?

    Returns a MetacognitiveDecision with the recommended action and reasoning.
    """
    if history is None:
        history = CalibrationHistory()

    # Compute EV for each action
    action_evs = {
        action: compute_metacognitive_ev(context, action, history)
        for action in MetacognitiveAction
    }

    # Override rules (hard constraints)

    # Rule 1: High stakes + blind spot → always check tool if available
    if context.stakes > 0.8 and context.is_known_blind_spot:
        if context.has_verified_tool:
            tool = get_suggested_tool(context.domain)
            return MetacognitiveDecision(
                action=MetacognitiveAction.CHECK_TOOL,
                reasoning=f"High stakes ({context.stakes:.1f}) + known blind spot → must verify with tool",
                expected_value=action_evs[MetacognitiveAction.CHECK_TOOL],
                confidence_in_decision=0.95,
                suggested_tool=tool,
                energy_cost=ACTION_COSTS[MetacognitiveAction.CHECK_TOOL]
            )
        else:
            return MetacognitiveDecision(
                action=MetacognitiveAction.DEFER,
                reasoning=f"High stakes + blind spot but no verified tool → defer to avoid error",
                expected_value=action_evs[MetacognitiveAction.DEFER],
                confidence_in_decision=0.9,
                energy_cost=ACTION_COSTS[MetacognitiveAction.DEFER]
            )

    # Rule 2: Very high confidence + low stakes → act directly
    if context.confidence > 0.95 and context.stakes < 0.2 and not context.is_known_blind_spot:
        return MetacognitiveDecision(
            action=MetacognitiveAction.ACT_DIRECTLY,
            reasoning=f"High confidence ({context.confidence:.0%}) + low stakes ({context.stakes:.1f}) → act directly",
            expected_value=0.0,
            confidence_in_decision=0.85,
            energy_cost=0.0
        )

    # Rule 3: Low energy budget → prefer cheap actions
    if context.session_energy_remaining < 0.1:
        # Filter to cheap actions
        cheap_actions = {
            a: ev for a, ev in action_evs.items()
            if ACTION_COSTS[a] <= context.session_energy_remaining
        }
        if cheap_actions:
            best_action = max(cheap_actions, key=cheap_actions.get)
        else:
            best_action = MetacognitiveAction.ACT_DIRECTLY

        return MetacognitiveDecision(
            action=best_action,
            reasoning=f"Low energy budget ({context.session_energy_remaining:.0%}) → using cheap action",
            expected_value=action_evs.get(best_action, 0.0),
            confidence_in_decision=0.7,
            suggested_tool=get_suggested_tool(context.domain) if best_action == MetacognitiveAction.CHECK_TOOL else None,
            energy_cost=ACTION_COSTS[best_action]
        )

    # Rule 4: Has verified tool + moderate stakes → use tool
    if context.has_verified_tool and context.stakes > 0.3:
        tool = get_suggested_tool(context.domain)
        return MetacognitiveDecision(
            action=MetacognitiveAction.CHECK_TOOL,
            reasoning=f"Verified tool available + stakes ({context.stakes:.1f}) warrant checking",
            expected_value=action_evs[MetacognitiveAction.CHECK_TOOL],
            confidence_in_decision=0.9,
            suggested_tool=tool,
            energy_cost=ACTION_COSTS[MetacognitiveAction.CHECK_TOOL]
        )

    # General case: pick action with highest EV
    best_action = max(action_evs, key=action_evs.get)
    best_ev = action_evs[best_action]

    # If best EV is negative, just act directly (not worth checking)
    if best_ev < 0:
        return MetacognitiveDecision(
            action=MetacognitiveAction.ACT_DIRECTLY,
            reasoning=f"All metacognitive actions have negative EV → act directly",
            expected_value=0.0,
            confidence_in_decision=0.8,
            energy_cost=0.0
        )

    return MetacognitiveDecision(
        action=best_action,
        reasoning=f"Best EV action: {best_action.value} (EV={best_ev:+.3f})",
        expected_value=best_ev,
        confidence_in_decision=0.75,
        suggested_tool=get_suggested_tool(context.domain) if best_action == MetacognitiveAction.CHECK_TOOL else None,
        energy_cost=ACTION_COSTS[best_action]
    )


def get_suggested_tool(domain: str) -> Optional[str]:
    """Get the primary tool for a domain."""
    if domain in TOOL_DOMAINS:
        tools = TOOL_DOMAINS[domain]
        return tools[0] if tools else None
    return None


def get_tools_for_domain(domain: str) -> List[str]:
    """Get all tools available for a domain."""
    return TOOL_DOMAINS.get(domain, [])


def is_blind_spot_domain(domain: str) -> bool:
    """Check if a domain is a known blind spot."""
    return domain in BLIND_SPOT_DOMAINS


@dataclass
class MetacognitiveEnergyBudget:
    """
    Track metacognitive energy spending across a session.

    The idea: each session has a finite "thinking about thinking" budget.
    Spend it wisely on high-value checks, not uniformly on everything.
    """
    total_budget: float = 1.0       # Total energy for session
    spent: float = 0.0              # Energy spent so far
    checks_performed: int = 0       # Number of metacognitive checks
    errors_caught: int = 0          # Errors caught by checking
    errors_missed: int = 0          # Errors that slipped through

    @property
    def remaining(self) -> float:
        return max(0.0, self.total_budget - self.spent)

    @property
    def efficiency(self) -> float:
        """Errors caught per unit energy spent."""
        if self.spent == 0:
            return 0.0
        return self.errors_caught / self.spent

    def spend(self, amount: float) -> bool:
        """Spend energy. Returns False if budget exceeded."""
        if amount > self.remaining:
            return False
        self.spent += amount
        self.checks_performed += 1
        return True

    def record_outcome(self, caught_error: bool):
        """Record whether a check caught an error."""
        if caught_error:
            self.errors_caught += 1
        else:
            self.errors_missed += 1

    def get_report(self) -> Dict:
        """Get summary of metacognitive spending."""
        return {
            "total_budget": self.total_budget,
            "spent": self.spent,
            "remaining": self.remaining,
            "checks_performed": self.checks_performed,
            "errors_caught": self.errors_caught,
            "errors_missed": self.errors_missed,
            "efficiency": self.efficiency,
            "catch_rate": self.errors_caught / max(1, self.checks_performed),
        }


class MetacognitiveController:
    """
    Main controller that integrates decision-making with energy management.

    Usage:
        controller = MetacognitiveController(energy_budget=1.0)

        for task in tasks:
            context = TaskContext(...)
            decision = controller.decide(context)

            if decision.action == MetacognitiveAction.CHECK_TOOL:
                result = call_tool(decision.suggested_tool, ...)
                controller.record_outcome(caught_error=result.changed_answer)

        report = controller.get_session_report()
    """

    def __init__(self, energy_budget: float = 1.0):
        self.budget = MetacognitiveEnergyBudget(total_budget=energy_budget)
        self.history = CalibrationHistory()
        self.decisions: List[MetacognitiveDecision] = []

    def decide(self, context: TaskContext) -> MetacognitiveDecision:
        """Make a metacognitive decision for the given context."""
        # Update context with current energy
        context.session_energy_remaining = self.budget.remaining

        # Make decision
        decision = should_engage_metacognition(context, self.history)

        # Spend energy if action requires it
        if decision.action != MetacognitiveAction.ACT_DIRECTLY:
            if not self.budget.spend(decision.energy_cost):
                # Budget exceeded, fall back to acting directly
                decision = MetacognitiveDecision(
                    action=MetacognitiveAction.ACT_DIRECTLY,
                    reasoning="Energy budget exhausted → acting directly",
                    expected_value=0.0,
                    confidence_in_decision=0.5,
                    energy_cost=0.0
                )

        self.decisions.append(decision)
        return decision

    def record_outcome(self, domain: str, caught_error: bool):
        """Record the outcome of a metacognitive check."""
        self.budget.record_outcome(caught_error)
        self.history.record_check_outcome(domain, caught_error)

    def get_session_report(self) -> Dict:
        """Get summary of metacognitive activity this session."""
        action_counts = {}
        for d in self.decisions:
            action_counts[d.action.value] = action_counts.get(d.action.value, 0) + 1

        return {
            "budget": self.budget.get_report(),
            "decisions_made": len(self.decisions),
            "action_distribution": action_counts,
            "total_ev": sum(d.expected_value for d in self.decisions),
            "history": {
                "total_checks": self.history.total_checks,
                "check_hit_rate": self.history.get_check_hit_rate(),
            }
        }


def compute_optimal_check_threshold(
    stakes_distribution: List[float],
    error_rate: float,
    energy_budget: float
) -> float:
    """
    Compute the optimal stakes threshold for checking.

    Given a distribution of task stakes and a fixed energy budget,
    what's the minimum stakes level that warrants a metacognitive check?

    This is a resource allocation problem:
    - We have energy_budget units of checking capacity
    - Tasks have varying stakes
    - We want to maximize errors caught

    Solution: check the highest-stakes tasks until budget exhausted.
    Return the threshold stakes value.
    """
    if not stakes_distribution:
        return 0.5  # Default

    # Sort tasks by stakes (descending)
    sorted_stakes = sorted(stakes_distribution, reverse=True)

    # How many checks can we afford?
    check_cost = ACTION_COSTS[MetacognitiveAction.CHECK_TOOL]
    max_checks = int(energy_budget / check_cost)

    if max_checks >= len(sorted_stakes):
        # We can check everything
        return 0.0

    if max_checks == 0:
        # Can't check anything
        return 1.1  # Threshold above max possible stakes

    # Threshold is the stakes of the last task we can afford to check
    threshold = sorted_stakes[max_checks - 1]

    return threshold


def list_metacognitive_actions() -> Dict[str, str]:
    """List all possible metacognitive actions with descriptions."""
    return {
        "act_directly": "Respond without metacognitive checking. Fastest but riskiest.",
        "verify_confidence": "Quick internal confidence check. Low cost, catches ~30% of errors.",
        "check_tool": "Route to verified NoetherSolve tool. Higher cost but 100% accurate.",
        "deep_reflection": "Full metacognitive analysis. High cost, catches ~50% of errors.",
        "defer": "Admit uncertainty and ask for clarification. Avoids error entirely.",
    }


def get_domain_tool_coverage() -> Dict[str, Dict]:
    """Get coverage information for all tool domains."""
    return {
        domain: {
            "tools": tools,
            "is_blind_spot": domain in BLIND_SPOT_DOMAINS,
            "tool_count": len(tools),
        }
        for domain, tools in TOOL_DOMAINS.items()
    }


def prefer_local_tools(
    domain: str,
    budget: Optional[ResourceBudget] = None
) -> Tuple[bool, str]:
    """
    Decide whether to use local tools vs API-based alternatives.

    Key insight: NoetherSolve tools are LOCAL and FREE.
    API calls (web search, external LLM) are EXPENSIVE.

    Returns: (should_use_local, reasoning)
    """
    if budget is None:
        budget = ResourceBudget()

    local_tools = get_tools_for_domain(domain)

    if local_tools:
        # We have local tools - almost always prefer them
        if budget.api_tokens < 0.05:
            return True, f"Local tools available ({local_tools[0]}) and API budget exhausted"
        elif budget.weights.get("api_tokens", 1.0) > 5.0:
            return True, f"Local tools available and API is expensive (weight={budget.weights['api_tokens']})"
        else:
            return True, f"Local tools available - using {local_tools[0]}"

    # No local tools - must consider API
    if budget.api_tokens > 0.1:
        return False, "No local tools - will use API if needed"
    else:
        return True, "No local tools and API budget low - will defer or act directly"


def compute_resource_aware_ev(
    context: TaskContext,
    action: MetacognitiveAction,
    history: CalibrationHistory,
    budget: ResourceBudget
) -> float:
    """
    Compute expected value accounting for resource scarcity.

    Key insight: The same action has different value depending on
    what resources it consumes and what's scarce.

    NoetherSolve tools → high EV (free, accurate)
    API-based checks → lower EV when tokens are scarce
    """
    # Base EV from standard calculation
    base_ev = compute_metacognitive_ev(context, action, history)

    # Get resource cost
    resource_cost = ACTION_RESOURCE_COSTS.get(
        action,
        ResourceCost(local_compute=0.1, api_tokens=0.1, latency=0.1, memory=0.1)
    )

    # Adjust for resource scarcity
    effective_cost = budget.effective_cost(resource_cost)

    # If we're checking a tool, adjust based on whether it's local
    if action == MetacognitiveAction.CHECK_TOOL and context.has_verified_tool:
        # NoetherSolve tools are LOCAL - no API cost!
        effective_cost *= 0.1  # Huge discount for local tools

    # EV = base_ev - weighted_cost
    return base_ev - effective_cost


@dataclass
class ResourceAwareDecision:
    """Decision with resource-aware reasoning."""
    action: MetacognitiveAction
    reasoning: str
    expected_value: float
    resource_cost: ResourceCost
    budget_impact: Dict[str, float]  # How much of each resource this uses
    suggested_tool: Optional[str] = None
    tool_type: Optional[ToolType] = None  # LOCAL vs API
    alternative_actions: List[Tuple[MetacognitiveAction, float]] = field(default_factory=list)

    def __str__(self) -> str:
        lines = [
            f"Decision: {self.action.value}",
            f"Reasoning: {self.reasoning}",
            f"Expected Value: {self.expected_value:+.3f}",
            "",
            "Resource Cost:",
            f"  Local Compute: {self.resource_cost.local_compute:.3f}",
            f"  API Tokens: {self.resource_cost.api_tokens:.3f}",
            f"  Latency: {self.resource_cost.latency:.3f}",
        ]
        if self.suggested_tool:
            tool_type_str = f" ({self.tool_type.value})" if self.tool_type else ""
            lines.append(f"\nSuggested Tool: {self.suggested_tool}{tool_type_str}")
        return "\n".join(lines)


class ResourceAwareController:
    """
    Metacognitive controller that accounts for real resource constraints.

    The key insight: different resources have different costs to the user.
    - Local compute: basically free (GPU/CPU they already have)
    - API tokens: expensive (limited weekly Claude quota)
    - Latency: moderate (user patience)

    This controller prioritizes FREE local tools (NoetherSolve) over
    expensive API calls, while still maximizing expected value.

    Usage:
        budget = ResourceBudget(
            api_tokens=0.1,  # 10% of weekly quota left
            weights={"api_tokens": 10.0, "local_compute": 0.01}
        )
        controller = ResourceAwareController(budget)

        decision = controller.decide(context)
        # Will strongly prefer local NoetherSolve tools
    """

    def __init__(self, budget: Optional[ResourceBudget] = None):
        if budget is None:
            # Default: API tokens are expensive, local compute is free
            budget = ResourceBudget(
                local_compute=1.0,
                api_tokens=0.1,  # Scarce
                weights={
                    "local_compute": 0.01,
                    "api_tokens": 10.0,  # 1000x more expensive than local
                    "latency": 0.5,
                    "memory": 0.3,
                }
            )
        self.budget = budget
        self.history = CalibrationHistory()
        self.decisions: List[ResourceAwareDecision] = []

    def decide(self, context: TaskContext) -> ResourceAwareDecision:
        """Make a resource-aware metacognitive decision."""

        # Compute resource-aware EV for each action
        action_evs = {}
        for action in MetacognitiveAction:
            ev = compute_resource_aware_ev(context, action, self.history, self.budget)
            resource_cost = ACTION_RESOURCE_COSTS.get(action, ResourceCost())
            if self.budget.can_afford(resource_cost):
                action_evs[action] = ev

        if not action_evs:
            # Can't afford anything - must act directly
            return ResourceAwareDecision(
                action=MetacognitiveAction.ACT_DIRECTLY,
                reasoning="Resource budget exhausted - acting directly",
                expected_value=0.0,
                resource_cost=ResourceCost(),
                budget_impact={},
            )

        # Special handling: if local tool available and API is scarce, prefer tool
        if (context.has_verified_tool and
            self.budget.api_tokens < 0.2 and
            MetacognitiveAction.CHECK_TOOL in action_evs):

            tool = get_suggested_tool(context.domain)
            tool_type = TOOL_TYPES.get(tool, ToolType.LOCAL)

            if tool_type == ToolType.LOCAL:
                # Boost local tool EV significantly
                action_evs[MetacognitiveAction.CHECK_TOOL] += 0.5

        # Find best action
        best_action = max(action_evs, key=action_evs.get)
        best_ev = action_evs[best_action]

        # If best EV is negative, act directly
        if best_ev < 0 and MetacognitiveAction.ACT_DIRECTLY in action_evs:
            best_action = MetacognitiveAction.ACT_DIRECTLY
            best_ev = 0.0

        resource_cost = ACTION_RESOURCE_COSTS.get(best_action, ResourceCost())

        # Get alternatives
        alternatives = sorted(
            [(a, ev) for a, ev in action_evs.items() if a != best_action],
            key=lambda x: -x[1]
        )[:3]

        # Tool info if applicable
        tool = None
        tool_type = None
        if best_action == MetacognitiveAction.CHECK_TOOL:
            tool = get_suggested_tool(context.domain)
            tool_type = TOOL_TYPES.get(tool, ToolType.LOCAL)

        decision = ResourceAwareDecision(
            action=best_action,
            reasoning=self._generate_reasoning(context, best_action, self.budget),
            expected_value=best_ev,
            resource_cost=resource_cost,
            budget_impact={
                "local_compute": resource_cost.local_compute / max(0.01, self.budget.local_compute),
                "api_tokens": resource_cost.api_tokens / max(0.01, self.budget.api_tokens),
            },
            suggested_tool=tool,
            tool_type=tool_type,
            alternative_actions=alternatives,
        )

        # Spend resources
        self.budget.spend(resource_cost)
        self.decisions.append(decision)

        return decision

    def _generate_reasoning(
        self,
        context: TaskContext,
        action: MetacognitiveAction,
        budget: ResourceBudget
    ) -> str:
        """Generate human-readable reasoning for the decision."""
        parts = []

        if action == MetacognitiveAction.CHECK_TOOL and context.has_verified_tool:
            tool = get_suggested_tool(context.domain)
            tool_type = TOOL_TYPES.get(tool, ToolType.LOCAL)
            if tool_type == ToolType.LOCAL:
                parts.append(f"Using local NoetherSolve tool ({tool}) - FREE")
            else:
                parts.append(f"Using tool {tool}")

        if budget.api_tokens < 0.2:
            parts.append(f"API tokens scarce ({budget.api_tokens:.0%} remaining)")

        if context.stakes > 0.7:
            parts.append(f"High stakes ({context.stakes:.0%})")

        if context.is_known_blind_spot:
            parts.append("Known blind spot domain")

        if not parts:
            parts.append(f"Standard decision: {action.value}")

        return " | ".join(parts)

    def get_summary(self) -> Dict:
        """Get summary of decisions and resource usage."""
        return {
            "decisions_made": len(self.decisions),
            "remaining_budget": {
                "local_compute": self.budget.local_compute,
                "api_tokens": self.budget.api_tokens,
                "latency": self.budget.latency,
            },
            "local_tool_usage": sum(
                1 for d in self.decisions
                if d.tool_type == ToolType.LOCAL
            ),
            "api_usage": sum(
                1 for d in self.decisions
                if d.tool_type == ToolType.API
            ),
        }
