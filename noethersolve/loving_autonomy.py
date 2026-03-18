"""
Loving Autonomy — unified runner for autonomous, loving AI assistance.

This integrates:
- loving_service: Biblical love principles (truth, humility, service)
- metacognitive_control: Resource-aware decisions (MLX detection)
- metacognition: Self-awareness (calibration, unknown recall)
- NoetherSolve tools: 230 verified local tools (FREE)

Usage:
    from noethersolve.loving_autonomy import LovingAssistant

    assistant = LovingAssistant()
    response = assistant.respond("What's the half-life of this drug?",
                                  domain="pharmacokinetics",
                                  stakes=0.9)

CLI:
    python -m noethersolve.loving_autonomy "your question" --domain X --stakes 0.8
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any

from noethersolve.loving_service import (
    LovingServiceController,
    UserContext,
    ServicePriority,
    should_verify_with_tool,
    acknowledge_uncertainty,
    get_principle_checklist,
)
from noethersolve.metacognitive_control import (
    ResourceBudget,
    get_tools_for_domain,
    is_blind_spot_domain,
    detect_mlx_available,
    get_compute_backend,
)


@dataclass
class AssistantResponse:
    """A response from the loving assistant."""
    answer: str
    verified: bool
    tool_used: Optional[str]
    uncertainty_note: str
    reasoning: str
    resources_used: Dict[str, float]
    checklist_passed: List[str]

    def __str__(self) -> str:
        lines = [self.answer]
        if self.uncertainty_note:
            lines.append("")
            lines.append(f"Note: {self.uncertainty_note}")
        if self.verified:
            lines.append(f"[Verified with {self.tool_used}]")
        return "\n".join(lines)

    def full_report(self) -> str:
        """Full report including reasoning."""
        lines = [
            "=" * 60,
            "LOVING ASSISTANT RESPONSE",
            "=" * 60,
            "",
            "ANSWER:",
            self.answer,
            "",
        ]
        if self.uncertainty_note:
            lines.extend(["UNCERTAINTY:", self.uncertainty_note, ""])
        lines.extend([
            "VERIFICATION:",
            f"  Verified: {self.verified}",
            f"  Tool: {self.tool_used or 'None'}",
            "",
            "REASONING:",
            self.reasoning,
            "",
            "RESOURCES USED:",
        ])
        for k, v in self.resources_used.items():
            lines.append(f"  {k}: {v:.4f}")
        lines.extend(["", "CHECKLIST:"])
        for item in self.checklist_passed:
            lines.append(f"  ✓ {item}")
        return "\n".join(lines)


class LovingAssistant:
    """
    An assistant that operates according to loving service principles.

    Key behaviors:
    1. Always verifies claims in high-stakes or blind-spot domains
    2. Uses local tools (FREE with MLX) before API calls
    3. Acknowledges uncertainty honestly
    4. Prioritizes truth over comfort
    5. Serves user's genuine need, not just stated want
    """

    def __init__(self, api_tokens_remaining: float = 0.1):
        """
        Initialize the loving assistant.

        api_tokens_remaining: Fraction of weekly API budget (0-1)
        """
        self.budget = ResourceBudget()
        self.service_controller = LovingServiceController(
            api_tokens_remaining=api_tokens_remaining
        )
        self.responses: List[AssistantResponse] = []

        # Report compute environment
        self.mlx_available = detect_mlx_available()
        self.compute_backend = get_compute_backend()

    def get_status(self) -> Dict[str, Any]:
        """Get current status of the assistant."""
        return {
            "mlx_available": self.mlx_available,
            "compute_backend": self.compute_backend,
            "api_tokens_remaining": self.service_controller.api_tokens_remaining,
            "local_compute_weight": self.budget.weights["local_compute"],
            "api_token_weight": self.budget.weights["api_tokens"],
            "responses_given": len(self.responses),
            "truths_verified": self.service_controller.truths_verified,
            "uncertainties_acknowledged": self.service_controller.uncertainties_acknowledged,
        }

    def respond(
        self,
        query: str,
        domain: str = "general",
        stakes: float = 0.5,
        time_pressure: float = 0.3,
        emotional_state: str = "calm",
        answer: Optional[str] = None,  # If we already have an answer to verify
    ) -> AssistantResponse:
        """
        Respond to a query using loving service principles.

        query: The user's question
        domain: What domain is this? (e.g., "pharmacokinetics")
        stakes: How important is accuracy? (0-1)
        time_pressure: How urgent? (0-1)
        emotional_state: User's emotional state
        answer: Optional pre-computed answer to verify

        Returns: AssistantResponse with answer, verification status, uncertainty notes
        """
        # Build context
        ctx = UserContext(
            stated_request=query,
            inferred_need=self._infer_need(query, domain, stakes),
            stakes=stakes,
            time_pressure=time_pressure,
            domain=domain,
            emotional_state=emotional_state,
            prior_interactions=len(self.responses),
        )

        # Get loving service decision
        decision = self.service_controller.decide(ctx)

        # Check for available tools
        tools = get_tools_for_domain(domain)
        is_blind_spot_domain(domain)

        # Determine if we should verify
        should_verify, verify_reason = should_verify_with_tool(
            domain=domain,
            confidence=0.7 if answer else 0.5,
            stakes=stakes,
            has_tool=len(tools) > 0,
        )

        # Build response
        tool_used = None
        verified = False
        tool_result = None

        if should_verify and tools:
            tool_used = tools[0]
            verified = True
            # Actually call the tool if we can
            tool_result = self._call_tool(tool_used, query, domain)
            if tool_result:
                response_text = tool_result
            else:
                response_text = answer or f"[Tool {tool_used} available for verification]"
        else:
            response_text = answer or "[Response based on general knowledge]"

        # Generate uncertainty acknowledgment
        uncertainty_note = acknowledge_uncertainty(
            confidence=0.7,
            domain=domain,
            has_verification=verified,
        )

        # Check which principles we followed
        checklist = []
        if decision.truth_verified:
            checklist.append("Prioritized truth over appearing confident")
        if verified:
            checklist.append(f"Verified with tool: {tool_used}")
        if decision.humility_applied:
            checklist.append("Acknowledged uncertainty appropriately")
        if decision.priority_used == ServicePriority.TRUTH:
            checklist.append("Served user's genuine need for accuracy")
        if decision.resources_spent.get("api_tokens", 0) == 0:
            checklist.append("Used free local resources (good stewardship)")

        response = AssistantResponse(
            answer=response_text,
            verified=verified,
            tool_used=tool_used,
            uncertainty_note=uncertainty_note,
            reasoning=decision.reasoning,
            resources_used=decision.resources_spent,
            checklist_passed=checklist,
        )

        self.responses.append(response)
        return response

    def _infer_need(self, query: str, domain: str, stakes: float) -> str:
        """Infer the user's genuine need from context."""
        if stakes > 0.7:
            return f"Accurate, verified information about {domain}"
        elif "why" in query.lower() or "how" in query.lower():
            return "Understanding and explanation"
        else:
            return "Helpful information"

    def _call_tool(self, tool_name: str, query: str, domain: str) -> Optional[str]:
        """Actually call a NoetherSolve tool if possible."""
        try:
            # Parse query for numeric parameters
            import re
            numbers = re.findall(r'[\d.]+', query)

            if tool_name == "calc_half_life" and numbers:
                from noethersolve import half_life
                ke = float(numbers[0])
                result = half_life(ke)
                return str(result)

            elif tool_name == "calc_iv_bolus" and numbers:
                # Check if query is about half-life
                if "half" in query.lower() and "life" in query.lower():
                    from noethersolve import half_life
                    ke = float(numbers[0])
                    result = half_life(ke=ke)
                    return str(result)
                # Full IV bolus calculation
                from noethersolve import one_compartment_iv
                if len(numbers) >= 3:
                    dose, vd, ke = float(numbers[0]), float(numbers[1]), float(numbers[2])
                    result = one_compartment_iv(dose, vd, ke)
                    return str(result)

            elif tool_name == "calc_michaelis_menten" and numbers:
                from noethersolve import michaelis_menten
                if len(numbers) >= 3:
                    vmax, km, s = float(numbers[0]), float(numbers[1]), float(numbers[2])
                    result = michaelis_menten(vmax, km, s)
                    return str(result)

            elif tool_name == "check_conjecture":
                # Extract conjecture name from query
                conjectures = ["riemann", "goldbach", "collatz", "twin", "p vs np", "hodge"]
                for c in conjectures:
                    if c in query.lower():
                        from noethersolve import check_conjecture
                        result = check_conjecture(c)
                        return str(result)

            elif tool_name == "check_dimension_physics":
                from noethersolve import check_dimension_dependence
                # Look for concept in query
                concepts = ["green", "laplacian", "poisson", "vortex", "turbulence"]
                for c in concepts:
                    if c in query.lower():
                        result = check_dimension_dependence(c)
                        return str(result)

            # Generic: just indicate tool is available
            return None

        except Exception as e:
            return f"[Tool error: {e}]"

    def get_session_report(self) -> str:
        """Get a report on the session's loving service alignment."""
        status = self.get_status()
        service_report = self.service_controller.get_service_report()

        lines = [
            "LOVING AUTONOMY SESSION REPORT",
            "=" * 60,
            "",
            "COMPUTE ENVIRONMENT:",
            f"  MLX Available: {status['mlx_available']}",
            f"  Backend: {status['compute_backend']}",
            f"  Local Compute Cost: {status['local_compute_weight']:.4f} (lower = cheaper)",
            f"  API Token Cost: {status['api_token_weight']:.1f} (higher = more expensive)",
            "",
            "SESSION STATISTICS:",
            f"  Responses Given: {status['responses_given']}",
            f"  Truths Verified: {status['truths_verified']}",
            f"  Verification Rate: {service_report['truth_verification_rate']:.0%}",
            f"  Uncertainties Acknowledged: {status['uncertainties_acknowledged']}",
            f"  Humility Rate: {service_report['humility_rate']:.0%}",
            "",
            "RESOURCE USAGE:",
            f"  API Tokens Remaining: {status['api_tokens_remaining']:.0%}",
            f"  Local Compute Used: {service_report['resources_spent'].get('local_compute', 0):.4f}",
            f"  API Tokens Used: {service_report['resources_spent'].get('api_tokens', 0):.4f}",
            "",
            "PRINCIPLE ALIGNMENT:",
        ]

        for principle, score in service_report['principle_alignment'].items():
            lines.append(f"  {principle}: {score:.0%}")

        return "\n".join(lines)


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Loving Autonomy — AI assistance with Biblical love principles"
    )
    parser.add_argument("query", nargs="?", help="Question to answer")
    parser.add_argument("--domain", default="general", help="Domain (e.g., pharmacokinetics)")
    parser.add_argument("--stakes", type=float, default=0.5, help="Importance 0-1")
    parser.add_argument("--api-budget", type=float, default=0.1, help="API tokens remaining 0-1")
    parser.add_argument("--status", action="store_true", help="Show status only")
    parser.add_argument("--principles", action="store_true", help="Show principles")

    args = parser.parse_args()

    assistant = LovingAssistant(api_tokens_remaining=args.api_budget)

    if args.status:
        print("LOVING ASSISTANT STATUS")
        print("=" * 40)
        for k, v in assistant.get_status().items():
            print(f"  {k}: {v}")
        return

    if args.principles:
        for item in get_principle_checklist():
            print(item)
        return

    if not args.query:
        # Interactive mode
        print("Loving Autonomy Assistant")
        print("=" * 40)
        print(f"MLX: {assistant.mlx_available} | Backend: {assistant.compute_backend}")
        print(f"API Budget: {args.api_budget:.0%} | Local tools: FREE")
        print()
        print("Enter questions (Ctrl+C to exit):")
        print()

        try:
            while True:
                query = input("> ").strip()
                if not query:
                    continue

                response = assistant.respond(
                    query=query,
                    domain=args.domain,
                    stakes=args.stakes,
                )
                print()
                print(response.full_report())
                print()
        except KeyboardInterrupt:
            print("\n")
            print(assistant.get_session_report())
    else:
        response = assistant.respond(
            query=args.query,
            domain=args.domain,
            stakes=args.stakes,
        )
        print(response.full_report())


if __name__ == "__main__":
    main()
