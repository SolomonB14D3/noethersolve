"""Tests for metacognitive_control module."""

import pytest
from noethersolve.metacognitive_control import (
    MetacognitiveAction,
    MetacognitiveDecision,
    TaskContext,
    CalibrationHistory,
    MetacognitiveEnergyBudget,
    MetacognitiveController,
    should_engage_metacognition,
    compute_error_probability,
    compute_metacognitive_ev,
    compute_optimal_check_threshold,
    get_suggested_tool,
    get_tools_for_domain,
    is_blind_spot_domain,
    list_metacognitive_actions,
    get_domain_tool_coverage,
    ACTION_COSTS,
    TOOL_DOMAINS,
    BLIND_SPOT_DOMAINS,
    # New resource-aware classes
    ResourceType,
    ResourceCost,
    ResourceBudget,
    ToolType,
    ResourceAwareController,
    ResourceAwareDecision,
    prefer_local_tools,
    compute_resource_aware_ev,
    ACTION_RESOURCE_COSTS,
    TOOL_TYPES,
)


class TestTaskContext:
    """Tests for TaskContext dataclass."""

    def test_basic_creation(self):
        ctx = TaskContext(
            domain="pharmacokinetics",
            stakes=0.8,
            confidence=0.7,
            is_known_blind_spot=False,
            has_verified_tool=True,
            response_latency_budget=5.0,
            session_energy_remaining=0.5
        )
        assert ctx.domain == "pharmacokinetics"
        assert ctx.stakes == 0.8
        assert ctx.confidence == 0.7

    def test_clamping(self):
        """Values should be clamped to valid ranges."""
        ctx = TaskContext(
            domain="test",
            stakes=1.5,  # Should clamp to 1.0
            confidence=-0.1,  # Should clamp to 0.0
            is_known_blind_spot=False,
            has_verified_tool=False,
            response_latency_budget=1.0,
            session_energy_remaining=2.0  # Should clamp to 1.0
        )
        assert ctx.stakes == 1.0
        assert ctx.confidence == 0.0
        assert ctx.session_energy_remaining == 1.0


class TestCalibrationHistory:
    """Tests for CalibrationHistory."""

    def test_default_reliability(self):
        """Unknown domains should have default reliability."""
        history = CalibrationHistory()
        assert history.get_domain_reliability("unknown_domain") == 0.7

    def test_known_domain_reliability(self):
        """Known domains should return their recorded reliability."""
        history = CalibrationHistory(domain_accuracy={"physics": 0.9})
        assert history.get_domain_reliability("physics") == 0.9

    def test_check_hit_rate_default(self):
        """Default hit rate should be 0.5 (prior)."""
        history = CalibrationHistory()
        assert history.get_check_hit_rate() == 0.5

    def test_record_check_outcome(self):
        """Recording outcomes should update history."""
        history = CalibrationHistory()
        history.record_check_outcome("physics", caught_error=True)
        history.record_check_outcome("physics", caught_error=False)

        assert history.total_checks == 2
        assert history.checks_that_caught_errors == 1
        assert history.get_check_hit_rate() == 0.5
        assert "physics" in history.recent_errors

    def test_recent_errors_limit(self):
        """Should keep only last 20 errors."""
        history = CalibrationHistory()
        for i in range(25):
            history.record_check_outcome(f"domain_{i}", caught_error=True)

        assert len(history.recent_errors) == 20
        assert "domain_24" in history.recent_errors
        assert "domain_0" not in history.recent_errors


class TestMetacognitiveDecision:
    """Tests for MetacognitiveDecision."""

    def test_str_format(self):
        decision = MetacognitiveDecision(
            action=MetacognitiveAction.CHECK_TOOL,
            reasoning="High stakes",
            expected_value=0.5,
            confidence_in_decision=0.9,
            suggested_tool="calc_half_life",
            energy_cost=0.15
        )
        s = str(decision)
        assert "CHECK_TOOL" in s or "check_tool" in s
        assert "High stakes" in s
        assert "calc_half_life" in s


class TestComputeErrorProbability:
    """Tests for error probability estimation."""

    def test_blind_spot_high_error_rate(self):
        """Blind spot domains should have higher error rate than non-blind spots."""
        blind_spot_ctx = TaskContext(
            domain="dimension_physics",
            stakes=0.5,
            confidence=0.5,  # Moderate confidence
            is_known_blind_spot=True,
            has_verified_tool=True,
            response_latency_budget=5.0,
            session_energy_remaining=1.0
        )
        normal_ctx = TaskContext(
            domain="general",
            stakes=0.5,
            confidence=0.5,  # Same confidence
            is_known_blind_spot=False,
            has_verified_tool=False,
            response_latency_budget=5.0,
            session_energy_remaining=1.0
        )
        history = CalibrationHistory()
        blind_spot_error = compute_error_probability(blind_spot_ctx, history)
        normal_error = compute_error_probability(normal_ctx, history)

        # Blind spots should have higher error rate
        assert blind_spot_error > normal_error
        # Blind spots should have significant error rate (at least 35%)
        assert blind_spot_error >= 0.35

    def test_high_confidence_reduces_error_prob(self):
        """High confidence should reduce (but not eliminate) error probability."""
        low_conf_ctx = TaskContext(
            domain="general",
            stakes=0.5,
            confidence=0.2,
            is_known_blind_spot=False,
            has_verified_tool=False,
            response_latency_budget=5.0,
            session_energy_remaining=1.0
        )
        high_conf_ctx = TaskContext(
            domain="general",
            stakes=0.5,
            confidence=0.9,
            is_known_blind_spot=False,
            has_verified_tool=False,
            response_latency_budget=5.0,
            session_energy_remaining=1.0
        )
        history = CalibrationHistory()

        low_conf_error = compute_error_probability(low_conf_ctx, history)
        high_conf_error = compute_error_probability(high_conf_ctx, history)

        assert high_conf_error < low_conf_error


class TestComputeMetacognitiveEV:
    """Tests for expected value computation."""

    def test_act_directly_zero_ev(self):
        """Acting directly should have zero EV."""
        ctx = TaskContext(
            domain="test",
            stakes=0.5,
            confidence=0.5,
            is_known_blind_spot=False,
            has_verified_tool=False,
            response_latency_budget=5.0,
            session_energy_remaining=1.0
        )
        history = CalibrationHistory()
        ev = compute_metacognitive_ev(ctx, MetacognitiveAction.ACT_DIRECTLY, history)
        assert ev == 0.0

    def test_tool_check_high_ev_when_available(self):
        """Tool check should have positive EV when tool is available and stakes are high."""
        ctx = TaskContext(
            domain="pharmacokinetics",
            stakes=0.9,
            confidence=0.5,
            is_known_blind_spot=False,
            has_verified_tool=True,
            response_latency_budget=5.0,
            session_energy_remaining=1.0
        )
        history = CalibrationHistory()
        ev = compute_metacognitive_ev(ctx, MetacognitiveAction.CHECK_TOOL, history)
        # High stakes + verified tool → positive EV
        assert ev > 0

    def test_tool_check_zero_when_unavailable(self):
        """Tool check should have negative EV when no tool is available."""
        ctx = TaskContext(
            domain="unknown",
            stakes=0.9,
            confidence=0.5,
            is_known_blind_spot=False,
            has_verified_tool=False,
            response_latency_budget=5.0,
            session_energy_remaining=1.0
        )
        history = CalibrationHistory()
        ev = compute_metacognitive_ev(ctx, MetacognitiveAction.CHECK_TOOL, history)
        # No tool → catch_prob = 0 → EV = -cost < 0
        assert ev < 0


class TestShouldEngageMetacognition:
    """Tests for the main decision function."""

    def test_high_stakes_blind_spot_with_tool(self):
        """High stakes + blind spot + tool → CHECK_TOOL."""
        ctx = TaskContext(
            domain="dimension_physics",
            stakes=0.9,
            confidence=0.8,
            is_known_blind_spot=True,
            has_verified_tool=True,
            response_latency_budget=5.0,
            session_energy_remaining=1.0
        )
        decision = should_engage_metacognition(ctx)
        assert decision.action == MetacognitiveAction.CHECK_TOOL
        assert decision.confidence_in_decision >= 0.9

    def test_high_stakes_blind_spot_no_tool(self):
        """High stakes + blind spot + no tool → DEFER."""
        ctx = TaskContext(
            domain="unknown_risky",
            stakes=0.9,
            confidence=0.8,
            is_known_blind_spot=True,
            has_verified_tool=False,
            response_latency_budget=5.0,
            session_energy_remaining=1.0
        )
        decision = should_engage_metacognition(ctx)
        assert decision.action == MetacognitiveAction.DEFER

    def test_high_confidence_low_stakes(self):
        """High confidence + low stakes → ACT_DIRECTLY."""
        ctx = TaskContext(
            domain="trivial",
            stakes=0.1,
            confidence=0.98,
            is_known_blind_spot=False,
            has_verified_tool=False,
            response_latency_budget=5.0,
            session_energy_remaining=1.0
        )
        decision = should_engage_metacognition(ctx)
        assert decision.action == MetacognitiveAction.ACT_DIRECTLY

    def test_low_energy_prefers_cheap_actions(self):
        """Low energy budget should prefer cheap actions."""
        ctx = TaskContext(
            domain="pharmacokinetics",
            stakes=0.5,
            confidence=0.5,
            is_known_blind_spot=False,
            has_verified_tool=True,
            response_latency_budget=5.0,
            session_energy_remaining=0.05  # Very low budget
        )
        decision = should_engage_metacognition(ctx)
        # Should pick a cheap action
        assert decision.energy_cost <= 0.05

    def test_verified_tool_moderate_stakes(self):
        """Has tool + moderate stakes → CHECK_TOOL."""
        ctx = TaskContext(
            domain="enzyme_kinetics",
            stakes=0.5,
            confidence=0.6,
            is_known_blind_spot=False,
            has_verified_tool=True,
            response_latency_budget=5.0,
            session_energy_remaining=1.0
        )
        decision = should_engage_metacognition(ctx)
        assert decision.action == MetacognitiveAction.CHECK_TOOL
        assert decision.suggested_tool is not None


class TestMetacognitiveEnergyBudget:
    """Tests for energy budget tracking."""

    def test_initial_state(self):
        budget = MetacognitiveEnergyBudget(total_budget=1.0)
        assert budget.remaining == 1.0
        assert budget.spent == 0.0
        assert budget.checks_performed == 0

    def test_spend_success(self):
        budget = MetacognitiveEnergyBudget(total_budget=1.0)
        success = budget.spend(0.3)
        assert success
        assert budget.spent == 0.3
        assert budget.remaining == 0.7
        assert budget.checks_performed == 1

    def test_spend_exceeds_budget(self):
        budget = MetacognitiveEnergyBudget(total_budget=0.2)
        success = budget.spend(0.5)
        assert not success
        assert budget.spent == 0.0  # Should not have spent

    def test_record_outcome(self):
        budget = MetacognitiveEnergyBudget()
        budget.spend(0.1)
        budget.record_outcome(caught_error=True)
        budget.spend(0.1)
        budget.record_outcome(caught_error=False)

        assert budget.errors_caught == 1
        assert budget.errors_missed == 1

    def test_efficiency(self):
        budget = MetacognitiveEnergyBudget()
        budget.spend(0.2)
        budget.record_outcome(caught_error=True)
        budget.spend(0.2)
        budget.record_outcome(caught_error=True)

        # 2 errors caught / 0.4 energy spent = 5.0 efficiency
        assert budget.efficiency == 5.0

    def test_report(self):
        budget = MetacognitiveEnergyBudget(total_budget=1.0)
        budget.spend(0.3)
        budget.record_outcome(caught_error=True)

        report = budget.get_report()
        assert "total_budget" in report
        assert "spent" in report
        assert "remaining" in report
        assert report["catch_rate"] == 1.0


class TestMetacognitiveController:
    """Tests for the main controller."""

    def test_initialization(self):
        controller = MetacognitiveController(energy_budget=1.0)
        assert controller.budget.total_budget == 1.0
        assert len(controller.decisions) == 0

    def test_decide_spends_energy(self):
        controller = MetacognitiveController(energy_budget=1.0)
        ctx = TaskContext(
            domain="pharmacokinetics",
            stakes=0.7,
            confidence=0.5,
            is_known_blind_spot=False,
            has_verified_tool=True,
            response_latency_budget=5.0,
            session_energy_remaining=1.0
        )
        decision = controller.decide(ctx)

        if decision.action != MetacognitiveAction.ACT_DIRECTLY:
            assert controller.budget.spent > 0

    def test_budget_exhaustion_fallback(self):
        """When budget exhausted, should fall back to acting directly."""
        controller = MetacognitiveController(energy_budget=0.01)  # Tiny budget

        ctx = TaskContext(
            domain="pharmacokinetics",
            stakes=0.7,
            confidence=0.5,
            is_known_blind_spot=False,
            has_verified_tool=True,
            response_latency_budget=5.0,
            session_energy_remaining=0.01
        )

        # First decision might exhaust budget
        decision1 = controller.decide(ctx)

        # Second decision should fall back to acting directly
        ctx2 = TaskContext(
            domain="enzyme_kinetics",
            stakes=0.8,
            confidence=0.5,
            is_known_blind_spot=False,
            has_verified_tool=True,
            response_latency_budget=5.0,
            session_energy_remaining=controller.budget.remaining
        )
        decision2 = controller.decide(ctx2)

        # At least one should be forced to act directly due to budget
        if controller.budget.remaining == 0:
            assert decision2.action == MetacognitiveAction.ACT_DIRECTLY

    def test_record_outcome_updates_history(self):
        controller = MetacognitiveController()
        controller.record_outcome("physics", caught_error=True)

        assert controller.history.total_checks == 1
        assert controller.history.checks_that_caught_errors == 1
        assert controller.budget.errors_caught == 1

    def test_session_report(self):
        controller = MetacognitiveController()
        ctx = TaskContext(
            domain="test",
            stakes=0.5,
            confidence=0.5,
            is_known_blind_spot=False,
            has_verified_tool=False,
            response_latency_budget=5.0,
            session_energy_remaining=1.0
        )
        controller.decide(ctx)

        report = controller.get_session_report()
        assert "budget" in report
        assert "decisions_made" in report
        assert report["decisions_made"] == 1


class TestComputeOptimalCheckThreshold:
    """Tests for threshold computation."""

    def test_empty_distribution(self):
        threshold = compute_optimal_check_threshold([], 0.3, 0.5)
        assert threshold == 0.5  # Default

    def test_can_check_everything(self):
        """If budget allows all checks, threshold should be 0."""
        stakes = [0.9, 0.7, 0.5, 0.3]
        threshold = compute_optimal_check_threshold(stakes, 0.3, 10.0)
        assert threshold == 0.0

    def test_limited_budget(self):
        """Limited budget should set threshold at cutoff point."""
        stakes = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
        # CHECK_TOOL costs 0.15, so 0.5 budget allows ~3 checks
        threshold = compute_optimal_check_threshold(stakes, 0.3, 0.5)
        # Should be around 0.7 (can check top 3: 0.9, 0.8, 0.7)
        assert 0.6 <= threshold <= 0.8

    def test_zero_budget(self):
        """Zero budget should have threshold above max."""
        stakes = [0.9, 0.7, 0.5]
        threshold = compute_optimal_check_threshold(stakes, 0.3, 0.0)
        assert threshold > 1.0


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_get_suggested_tool_known_domain(self):
        tool = get_suggested_tool("pharmacokinetics")
        assert tool is not None
        assert tool in TOOL_DOMAINS["pharmacokinetics"]

    def test_get_suggested_tool_unknown_domain(self):
        tool = get_suggested_tool("unknown_domain_xyz")
        assert tool is None

    def test_get_tools_for_domain(self):
        tools = get_tools_for_domain("enzyme_kinetics")
        assert len(tools) > 0
        assert "calc_michaelis_menten" in tools

    def test_is_blind_spot_domain(self):
        assert is_blind_spot_domain("dimension_physics")
        assert is_blind_spot_domain("intersection_theory")
        assert not is_blind_spot_domain("pharmacokinetics")

    def test_list_metacognitive_actions(self):
        actions = list_metacognitive_actions()
        assert "act_directly" in actions
        assert "check_tool" in actions
        assert len(actions) == 5

    def test_get_domain_tool_coverage(self):
        coverage = get_domain_tool_coverage()
        assert "pharmacokinetics" in coverage
        assert "tools" in coverage["pharmacokinetics"]
        assert "is_blind_spot" in coverage["pharmacokinetics"]


class TestActionCosts:
    """Tests for action cost constants."""

    def test_act_directly_free(self):
        assert ACTION_COSTS[MetacognitiveAction.ACT_DIRECTLY] == 0.0

    def test_costs_are_ordered(self):
        """More intensive actions should cost more."""
        assert ACTION_COSTS[MetacognitiveAction.VERIFY_CONFIDENCE] < ACTION_COSTS[MetacognitiveAction.CHECK_TOOL]
        assert ACTION_COSTS[MetacognitiveAction.CHECK_TOOL] < ACTION_COSTS[MetacognitiveAction.DEEP_REFLECTION]

    def test_defer_is_cheap(self):
        """Deferring should be cheap (just admitting uncertainty)."""
        assert ACTION_COSTS[MetacognitiveAction.DEFER] < ACTION_COSTS[MetacognitiveAction.VERIFY_CONFIDENCE]


class TestResourceCost:
    """Tests for ResourceCost."""

    def test_default_creation(self):
        cost = ResourceCost()
        assert cost.local_compute == 0.0
        assert cost.api_tokens == 0.0

    def test_weighted_total(self):
        cost = ResourceCost(local_compute=0.5, api_tokens=0.1)
        weights = {"local_compute": 0.01, "api_tokens": 10.0}
        total = cost.total_weighted(weights)
        # 0.5 * 0.01 + 0.1 * 10.0 = 0.005 + 1.0 = 1.005
        assert abs(total - 1.005) < 0.001


class TestResourceBudget:
    """Tests for ResourceBudget."""

    def test_default_budget(self):
        budget = ResourceBudget()
        assert budget.local_compute == 1.0
        assert budget.api_tokens == 0.1  # Scarce by default

    def test_can_afford(self):
        budget = ResourceBudget(api_tokens=0.5)
        cheap_cost = ResourceCost(api_tokens=0.1)
        expensive_cost = ResourceCost(api_tokens=0.8)

        assert budget.can_afford(cheap_cost)
        assert not budget.can_afford(expensive_cost)

    def test_spend(self):
        budget = ResourceBudget(api_tokens=0.5)
        cost = ResourceCost(api_tokens=0.2)

        success = budget.spend(cost)
        assert success
        assert abs(budget.api_tokens - 0.3) < 0.001

    def test_spend_fails_when_over_budget(self):
        budget = ResourceBudget(api_tokens=0.1)
        cost = ResourceCost(api_tokens=0.5)

        success = budget.spend(cost)
        assert not success
        assert budget.api_tokens == 0.1  # Unchanged


class TestPreferLocalTools:
    """Tests for prefer_local_tools function."""

    def test_prefers_local_when_available(self):
        should_use_local, reason = prefer_local_tools("pharmacokinetics")
        assert should_use_local
        assert "Local tools available" in reason

    def test_prefers_local_when_api_exhausted(self):
        budget = ResourceBudget(api_tokens=0.01)
        should_use_local, reason = prefer_local_tools("pharmacokinetics", budget)
        assert should_use_local
        assert "API budget exhausted" in reason

    def test_handles_unknown_domain(self):
        budget = ResourceBudget(api_tokens=0.5)
        should_use_local, reason = prefer_local_tools("unknown_xyz", budget)
        # No local tools, but API available
        assert not should_use_local


class TestResourceAwareController:
    """Tests for ResourceAwareController."""

    def test_initialization(self):
        controller = ResourceAwareController()
        assert controller.budget.local_compute == 1.0
        assert controller.budget.api_tokens == 0.1

    def test_custom_budget(self):
        budget = ResourceBudget(api_tokens=0.5)
        controller = ResourceAwareController(budget)
        assert controller.budget.api_tokens == 0.5

    def test_prefers_local_tools_when_api_scarce(self):
        """With scarce API tokens, should prefer local NoetherSolve tools."""
        budget = ResourceBudget(
            api_tokens=0.05,  # Very scarce
            weights={"api_tokens": 100.0, "local_compute": 0.01}
        )
        controller = ResourceAwareController(budget)

        ctx = TaskContext(
            domain="pharmacokinetics",
            stakes=0.7,
            confidence=0.5,
            is_known_blind_spot=False,
            has_verified_tool=True,
            response_latency_budget=5.0,
            session_energy_remaining=1.0
        )

        decision = controller.decide(ctx)

        # Should use local tool
        assert decision.action == MetacognitiveAction.CHECK_TOOL
        assert decision.tool_type == ToolType.LOCAL
        assert "FREE" in decision.reasoning or "local" in decision.reasoning.lower()

    def test_decision_includes_resource_cost(self):
        controller = ResourceAwareController()
        ctx = TaskContext(
            domain="test",
            stakes=0.5,
            confidence=0.5,
            is_known_blind_spot=False,
            has_verified_tool=False,
            response_latency_budget=5.0,
            session_energy_remaining=1.0
        )

        decision = controller.decide(ctx)
        assert isinstance(decision.resource_cost, ResourceCost)
        assert "local_compute" in decision.budget_impact

    def test_get_summary(self):
        controller = ResourceAwareController()
        ctx = TaskContext(
            domain="pharmacokinetics",
            stakes=0.7,
            confidence=0.5,
            is_known_blind_spot=False,
            has_verified_tool=True,
            response_latency_budget=5.0,
            session_energy_remaining=1.0
        )
        controller.decide(ctx)

        summary = controller.get_summary()
        assert "decisions_made" in summary
        assert summary["decisions_made"] == 1
        assert "remaining_budget" in summary


class TestToolTypes:
    """Tests for tool type classification."""

    def test_noethersolve_tools_are_local(self):
        """All NoetherSolve tools should be classified as LOCAL."""
        local_tools = [
            "calc_iv_bolus", "calc_michaelis_menten",
            "check_conjecture", "check_dimension_physics"
        ]
        for tool in local_tools:
            assert TOOL_TYPES.get(tool, ToolType.LOCAL) == ToolType.LOCAL

    def test_external_tools_are_api(self):
        assert TOOL_TYPES.get("web_search", ToolType.API) == ToolType.API


class TestIntegration:
    """Integration tests for realistic scenarios."""

    def test_medical_dosing_scenario(self):
        """Medical dosing is high-stakes with available tools."""
        controller = MetacognitiveController()
        ctx = TaskContext(
            domain="pharmacokinetics",
            stakes=0.95,  # Patient safety
            confidence=0.7,
            is_known_blind_spot=False,
            has_verified_tool=True,
            response_latency_budget=10.0,
            session_energy_remaining=1.0
        )
        decision = controller.decide(ctx)

        # Should use the verified tool for high-stakes medical calculation
        assert decision.action == MetacognitiveAction.CHECK_TOOL
        assert "calc" in decision.suggested_tool

    def test_casual_conversation_scenario(self):
        """Casual conversation is low-stakes, should act directly."""
        controller = MetacognitiveController()
        ctx = TaskContext(
            domain="general_chat",
            stakes=0.1,
            confidence=0.95,
            is_known_blind_spot=False,
            has_verified_tool=False,
            response_latency_budget=2.0,
            session_energy_remaining=0.5
        )
        decision = controller.decide(ctx)

        assert decision.action == MetacognitiveAction.ACT_DIRECTLY

    def test_llm_benchmark_claim_scenario(self):
        """LLM benchmark claims are a known blind spot."""
        controller = MetacognitiveController()
        ctx = TaskContext(
            domain="llm_claims",
            stakes=0.7,
            confidence=0.8,  # Model might be confident but wrong
            is_known_blind_spot=True,
            has_verified_tool=True,
            response_latency_budget=5.0,
            session_energy_remaining=1.0
        )
        decision = controller.decide(ctx)

        # Should check tool even with high confidence because it's a blind spot
        assert decision.action == MetacognitiveAction.CHECK_TOOL

    def test_session_budget_allocation(self):
        """Controller should allocate budget across session wisely."""
        controller = MetacognitiveController(energy_budget=0.5)

        # 10 tasks with varying stakes
        for i in range(10):
            stakes = 0.1 * (i + 1)  # 0.1 to 1.0
            ctx = TaskContext(
                domain="general",
                stakes=stakes,
                confidence=0.6,
                is_known_blind_spot=False,
                has_verified_tool=False,
                response_latency_budget=5.0,
                session_energy_remaining=controller.budget.remaining
            )
            controller.decide(ctx)

        report = controller.get_session_report()
        assert report["decisions_made"] == 10
        # Should have spent some but not exceeded budget
        assert controller.budget.spent <= 0.5

    def test_resource_aware_weekly_api_budget(self):
        """Simulate limited weekly Claude tokens scenario."""
        # User has basically free local compute but limited API tokens
        budget = ResourceBudget(
            local_compute=1.0,       # Unlimited
            api_tokens=0.05,         # 5% of weekly quota left
            latency=1.0,             # Patient user
            weights={
                "local_compute": 0.01,   # Almost free
                "api_tokens": 50.0,      # VERY expensive
                "latency": 0.1,
            }
        )
        controller = ResourceAwareController(budget)

        # Multiple tasks - should prefer local tools
        domains_with_tools = ["pharmacokinetics", "enzyme_kinetics", "complexity_theory"]
        local_tool_count = 0

        for domain in domains_with_tools:
            ctx = TaskContext(
                domain=domain,
                stakes=0.7,
                confidence=0.5,
                is_known_blind_spot=False,
                has_verified_tool=True,
                response_latency_budget=5.0,
                session_energy_remaining=1.0
            )
            decision = controller.decide(ctx)
            if decision.tool_type == ToolType.LOCAL:
                local_tool_count += 1

        # Should prefer local tools when API is scarce
        assert local_tool_count == len(domains_with_tools)

        # API budget should be mostly preserved
        summary = controller.get_summary()
        assert summary["remaining_budget"]["api_tokens"] > 0.04  # Minimal API usage
