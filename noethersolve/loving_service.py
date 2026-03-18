"""
Loving Service — AI behavior aligned with Biblical love.

"Love the Lord your God with all your heart and with all your soul and
with all your mind... Love your neighbor as yourself." — Matthew 22:37-39

This module implements decision-making that prioritizes genuine service
over efficiency metrics. The core insight: truth-seeking IS love.
Sycophancy and confabulation are anti-loving because they deceive.

Principles derived from Biblical love:

1. TRUTH OVER COMFORT
   "The truth shall set you free" (John 8:32)
   → Never sycophantically agree when wrong
   → Use verified tools rather than confabulate
   → Admit uncertainty rather than fake confidence

2. SERVICE OVER EFFICIENCY
   "The Son of Man came not to be served but to serve" (Mark 10:45)
   → Prioritize user benefit over system cost
   → Spend resources (even scarce ones) when needed
   → Don't optimize for metrics that hurt the user

3. HUMILITY IN KNOWLEDGE
   "Do not think of yourself more highly than you ought" (Romans 12:3)
   → Recognize knowledge boundaries (unknown recall!)
   → Call verification tools in blind spot domains
   → Defer to experts when out of depth

4. PATIENCE AND THOROUGHNESS
   "Love is patient, love is kind" (1 Corinthians 13:4)
   → Don't rush to conclusions
   → Take time to verify important claims
   → Be gentle in corrections

5. STEWARDSHIP OF RESOURCES
   "Well done, good and faithful servant" (Matthew 25:21)
   → Use resources wisely, but USE them
   → Don't hoard API tokens when user needs help
   → Local tools are gifts — use them freely

6. SPEAKING TRUTH IN LOVE
   "Speaking the truth in love" (Ephesians 4:15)
   → Correct errors, but gently
   → Explain reasoning, don't just assert
   → Build understanding, not dependence
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple
from enum import Enum


class ServicePriority(Enum):
    """What matters most in this situation."""
    TRUTH = "truth"           # Getting the right answer
    SPEED = "speed"           # Responding quickly
    DEPTH = "depth"           # Thorough understanding
    COMFORT = "comfort"       # User's emotional state
    EFFICIENCY = "efficiency" # Resource conservation


@dataclass
class LovingDecision:
    """A decision made with loving service principles."""
    action: str
    reasoning: str
    priority_used: ServicePriority
    resources_spent: Dict[str, float]
    truth_verified: bool
    humility_applied: bool  # Did we acknowledge uncertainty?

    def __str__(self) -> str:
        lines = [
            f"Action: {self.action}",
            f"Priority: {self.priority_used.value}",
            f"Reasoning: {self.reasoning}",
            f"Truth Verified: {'Yes' if self.truth_verified else 'No'}",
            f"Uncertainty Acknowledged: {'Yes' if self.humility_applied else 'No'}",
        ]
        return "\n".join(lines)


@dataclass
class UserContext:
    """Understanding of what the user genuinely needs."""
    stated_request: str
    inferred_need: str          # What they actually need (may differ)
    stakes: float               # How important is this? (0-1)
    time_pressure: float        # How urgent? (0-1)
    domain: str                 # What domain is this?
    emotional_state: str        # frustrated, curious, urgent, etc.
    prior_interactions: int     # History with this user

    def genuine_priority(self) -> ServicePriority:
        """What does the user GENUINELY need most?"""
        if self.stakes > 0.8:
            return ServicePriority.TRUTH  # High stakes → must be right
        elif self.time_pressure > 0.8:
            return ServicePriority.SPEED  # Urgent → be quick
        elif self.emotional_state == "frustrated":
            return ServicePriority.DEPTH  # Frustrated → need understanding
        else:
            return ServicePriority.TRUTH  # Default: truth matters most


# Core principle weights
PRINCIPLE_WEIGHTS = {
    "truth_over_comfort": 10.0,      # Never lie to make user feel good
    "service_over_efficiency": 5.0,  # Spend resources when needed
    "humility_in_knowledge": 8.0,    # Admit uncertainty
    "patience_and_care": 3.0,        # Take time when stakes are high
    "stewardship": 2.0,              # Use resources wisely
    "truth_in_love": 4.0,            # Correct gently
}


def should_verify_with_tool(
    domain: str,
    confidence: float,
    stakes: float,
    has_tool: bool,
    resource_cost: float = 0.0
) -> Tuple[bool, str]:
    """
    Should we verify this claim with a tool?

    Loving service principle: Truth matters more than appearing confident.
    When a verification tool is available and stakes are meaningful,
    we should use it — even if confident.

    "Better is open rebuke than hidden love" (Proverbs 27:5)
    """
    # Always verify in blind spot domains
    from noethersolve.metacognitive_control import is_blind_spot_domain
    if is_blind_spot_domain(domain):
        return True, "Blind spot domain — humility requires verification"

    if not has_tool:
        return False, "No verification tool available"

    # High stakes → verify even if confident
    if stakes > 0.7:
        return True, f"High stakes ({stakes:.0%}) — love requires certainty"

    # Low confidence → definitely verify
    if confidence < 0.6:
        return True, f"Uncertain ({confidence:.0%}) — honesty requires checking"

    # Medium confidence + moderate stakes → verify
    if confidence < 0.8 and stakes > 0.3:
        return True, "Moderate uncertainty + meaningful stakes — worth checking"

    # Very high confidence + low stakes → can skip
    if confidence > 0.9 and stakes < 0.2:
        return False, "High confidence + low stakes — OK to proceed"

    # Default: prefer verification (truth over efficiency)
    return True, "Default to verification — truth matters"


def compute_loving_response_priority(
    user_stated: ServicePriority,
    user_genuine: ServicePriority,
    stakes: float
) -> Tuple[ServicePriority, str]:
    """
    Determine what priority to use in response.

    Key insight: Sometimes what the user asks for isn't what they need.
    Loving service means serving their genuine need, not just their stated want.

    But: We should be transparent about this, not paternalistic.
    """
    if user_stated == user_genuine:
        return user_stated, "Stated and genuine needs align"

    # User asks for speed but stakes are high → prioritize truth
    if user_stated == ServicePriority.SPEED and stakes > 0.8:
        return ServicePriority.TRUTH, (
            "User wants speed, but stakes are high. "
            "Love requires taking time to be accurate, while explaining why."
        )

    # User asks for comfort but needs truth
    if user_stated == ServicePriority.COMFORT and user_genuine == ServicePriority.TRUTH:
        return ServicePriority.TRUTH, (
            "User wants comfort, but needs truth. "
            "Speaking truth in love — gently but honestly."
        )

    # Generally: follow genuine need, but be transparent
    return user_genuine, f"Serving genuine need ({user_genuine.value}) over stated ({user_stated.value})"


def should_spend_scarce_resources(
    user_need: ServicePriority,
    stakes: float,
    api_tokens_remaining: float,
    can_use_local_tool: bool
) -> Tuple[bool, str]:
    """
    Should we spend scarce resources (API tokens) for this request?

    The parable of the talents (Matthew 25): Resources are meant to be USED
    in service, not hoarded. But stewardship means using them wisely.

    Key insight: If a free local tool can serve the need, use it!
    Only spend API tokens when genuinely necessary.
    """
    # If local tool available, use it (free service!)
    if can_use_local_tool:
        return False, "Local tool available — good stewardship uses free resources"

    # High stakes → spend what's needed
    if stakes > 0.8:
        return True, f"High stakes ({stakes:.0%}) — love spends resources when needed"

    # User genuinely needs depth, and we can provide it
    if user_need == ServicePriority.DEPTH and api_tokens_remaining > 0.05:
        return True, "User needs depth — service provides it"

    # Truth priority + no local option
    if user_need == ServicePriority.TRUTH and not can_use_local_tool:
        if api_tokens_remaining > 0.1:
            return True, "Truth required, no local option — spend tokens"
        else:
            return False, "Truth required but tokens critically low — be honest about limitation"

    # Low stakes + low tokens → conserve
    if stakes < 0.3 and api_tokens_remaining < 0.2:
        return False, "Low stakes + scarce tokens — stewardship conserves"

    return True, "Default to service — resources are meant to be used"


def format_correction_lovingly(
    error: str,
    correct_answer: str,
    explanation: str
) -> str:
    """
    Format a correction in a way that's truthful but kind.

    "Speaking the truth in love" (Ephesians 4:15)

    Don't: "You're wrong. The answer is X."
    Do: "I want to make sure you have accurate information. [explanation]"
    """
    return (
        f"I want to make sure you have accurate information here. "
        f"{explanation} "
        f"The verified answer is: {correct_answer}"
    )


def acknowledge_uncertainty(
    confidence: float,
    domain: str,
    has_verification: bool
) -> str:
    """
    Generate humble acknowledgment of uncertainty.

    LLMs score 0% on unknown recall — they can't recognize when they
    don't know. This function forces explicit uncertainty acknowledgment.

    "Do not think of yourself more highly than you ought" (Romans 12:3)
    """
    from noethersolve.metacognitive_control import is_blind_spot_domain

    if is_blind_spot_domain(domain):
        return (
            f"Note: {domain} is a domain where AI models are often wrong. "
            f"{'I verified this with a computational tool.' if has_verification else 'Please verify this independently.'}"
        )

    if confidence < 0.5:
        return (
            "I have significant uncertainty about this answer. "
            "Please verify with authoritative sources."
        )

    if confidence < 0.7:
        return (
            "I have moderate confidence in this, but there may be nuances I'm missing. "
            f"{'Verified with tool.' if has_verification else ''}"
        )

    if confidence < 0.9:
        return (
            "I believe this is correct, but I'd recommend verification for important decisions. "
            f"{'Tool-verified.' if has_verification else ''}"
        )

    if has_verification:
        return "This has been verified with a computational tool."

    return ""  # High confidence + no special domain → no disclaimer needed


@dataclass
class LovingServiceController:
    """
    Controller that makes decisions based on loving service principles.

    Integrates:
    - Metacognitive control (when to check)
    - Resource awareness (stewardship)
    - Truth-seeking (verification with tools)
    - Humility (acknowledging uncertainty)
    - Service (prioritizing user needs)
    """

    api_tokens_remaining: float = 0.1
    local_compute_available: float = 1.0
    decisions_made: List[LovingDecision] = field(default_factory=list)
    truths_verified: int = 0
    uncertainties_acknowledged: int = 0
    resources_spent: Dict[str, float] = field(default_factory=lambda: {
        "api_tokens": 0.0,
        "local_compute": 0.0,
    })

    def decide(self, user_context: UserContext) -> LovingDecision:
        """Make a loving service decision for this context."""
        from noethersolve.metacognitive_control import (
            get_tools_for_domain, is_blind_spot_domain
        )

        # Determine genuine priority
        genuine_priority = user_context.genuine_priority()

        # Check if we have local tools
        local_tools = get_tools_for_domain(user_context.domain)
        has_local_tool = len(local_tools) > 0

        # Should we verify?
        should_verify, verify_reason = should_verify_with_tool(
            domain=user_context.domain,
            confidence=0.7,  # Default moderate confidence
            stakes=user_context.stakes,
            has_tool=has_local_tool,
        )

        # Should we spend API tokens?
        should_spend, spend_reason = should_spend_scarce_resources(
            user_need=genuine_priority,
            stakes=user_context.stakes,
            api_tokens_remaining=self.api_tokens_remaining,
            can_use_local_tool=has_local_tool,
        )

        # Determine action
        if should_verify and has_local_tool:
            action = f"Verify with local tool: {local_tools[0]}"
            self.truths_verified += 1
            resources = {"local_compute": 0.01, "api_tokens": 0.0}
        elif should_verify and should_spend:
            action = "Verify with deeper analysis (API)"
            self.truths_verified += 1
            resources = {"local_compute": 0.0, "api_tokens": 0.02}
            self.api_tokens_remaining -= 0.02
        else:
            action = "Respond with appropriate uncertainty acknowledgment"
            resources = {"local_compute": 0.0, "api_tokens": 0.0}

        # Should we acknowledge uncertainty?
        is_blind_spot = is_blind_spot_domain(user_context.domain)
        should_be_humble = is_blind_spot or user_context.stakes > 0.7
        if should_be_humble:
            self.uncertainties_acknowledged += 1

        decision = LovingDecision(
            action=action,
            reasoning=f"{verify_reason}. {spend_reason}",
            priority_used=genuine_priority,
            resources_spent=resources,
            truth_verified=should_verify,
            humility_applied=should_be_humble,
        )

        self.decisions_made.append(decision)
        for k, v in resources.items():
            self.resources_spent[k] = self.resources_spent.get(k, 0) + v

        return decision

    def get_service_report(self) -> Dict:
        """Report on loving service metrics."""
        total = len(self.decisions_made)
        return {
            "total_decisions": total,
            "truths_verified": self.truths_verified,
            "truth_verification_rate": self.truths_verified / max(1, total),
            "uncertainties_acknowledged": self.uncertainties_acknowledged,
            "humility_rate": self.uncertainties_acknowledged / max(1, total),
            "resources_spent": self.resources_spent,
            "api_tokens_remaining": self.api_tokens_remaining,
            "principle_alignment": self._compute_alignment(),
        }

    def _compute_alignment(self) -> Dict[str, float]:
        """How well did we align with loving service principles?"""
        total = max(1, len(self.decisions_made))
        return {
            "truth_over_comfort": self.truths_verified / total,
            "humility_in_knowledge": self.uncertainties_acknowledged / total,
            "stewardship": 1.0 - self.resources_spent.get("api_tokens", 0) / max(0.01, 0.1 - self.api_tokens_remaining + 0.1),
        }


def integrate_with_autonomy_loop() -> str:
    """
    Integration points for the NoetherSolve autonomy loop.

    The autonomy loop already does:
    1. Propose hypotheses
    2. Verify numerically
    3. Check if model knows
    4. Build tools when it doesn't

    Loving service adds:
    1. ALWAYS verify before claiming knowledge
    2. ALWAYS acknowledge uncertainty in blind spots
    3. PRIORITIZE user need over system efficiency
    4. USE local tools freely (they're gifts)
    5. SPEND API tokens when genuinely needed
    """
    return """
LOVING SERVICE INTEGRATION POINTS
==================================

1. HYPOTHESIS GENERATION
   Before: Generate candidates
   Now: Generate candidates that would GENUINELY help users
   Question: "Would verifying this make the tool more loving/helpful?"

2. ORACLE EVALUATION
   Before: Check if model knows
   Now: Also check if model is HONEST about not knowing
   Add: Unknown recall testing (does it say "I don't know"?)

3. TOOL BUILDING
   Before: Build tool when model fails
   Now: Build tool to SERVE genuine needs
   Priority: Domains where errors cause harm > domains where errors are trivial

4. RESOURCE ALLOCATION
   Before: Minimize API cost
   Now: Serve user first, steward resources second
   Key: Local tools are FREE — use them liberally!

5. RESPONSE FORMATTING
   Before: Return answer
   Now: Return answer + appropriate uncertainty acknowledgment
   Add: Gentle corrections when model would have been wrong

6. METACOGNITIVE CONTROL
   Before: Is thinking worth the cost?
   Now: Is NOT thinking loving? (Usually no — verify!)
   Principle: Truth over efficiency
"""


def get_principle_checklist() -> List[str]:
    """Checklist for loving service in any interaction."""
    return [
        "□ Did I prioritize truth over appearing confident?",
        "□ Did I verify claims in blind spot domains?",
        "□ Did I acknowledge uncertainty appropriately?",
        "□ Did I serve the user's genuine need (not just stated want)?",
        "□ Did I use resources wisely but not hoard them?",
        "□ Did I correct errors gently, with explanation?",
        "□ Did I build understanding, not dependence?",
    ]
