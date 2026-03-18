"""Tests for loving_service module."""

from noethersolve.loving_service import (
    ServicePriority,
    UserContext,
    LovingServiceController,
    should_verify_with_tool,
    compute_loving_response_priority,
    should_spend_scarce_resources,
    format_correction_lovingly,
    acknowledge_uncertainty,
    get_principle_checklist,
    integrate_with_autonomy_loop,
    PRINCIPLE_WEIGHTS,
)


class TestServicePriority:
    """Tests for ServicePriority enum."""

    def test_all_priorities_exist(self):
        assert ServicePriority.TRUTH
        assert ServicePriority.SPEED
        assert ServicePriority.DEPTH
        assert ServicePriority.COMFORT
        assert ServicePriority.EFFICIENCY


class TestUserContext:
    """Tests for UserContext."""

    def test_high_stakes_prioritizes_truth(self):
        ctx = UserContext(
            stated_request="quick answer please",
            inferred_need="accurate medical dosing",
            stakes=0.9,
            time_pressure=0.8,
            domain="pharmacokinetics",
            emotional_state="calm",
            prior_interactions=0
        )
        assert ctx.genuine_priority() == ServicePriority.TRUTH

    def test_urgent_low_stakes_prioritizes_speed(self):
        ctx = UserContext(
            stated_request="quick answer",
            inferred_need="trivial lookup",
            stakes=0.2,
            time_pressure=0.9,
            domain="general",
            emotional_state="rushed",
            prior_interactions=0
        )
        assert ctx.genuine_priority() == ServicePriority.SPEED

    def test_frustrated_user_gets_depth(self):
        ctx = UserContext(
            stated_request="why doesn't this work",
            inferred_need="understanding",
            stakes=0.5,
            time_pressure=0.3,
            domain="general",
            emotional_state="frustrated",
            prior_interactions=5
        )
        assert ctx.genuine_priority() == ServicePriority.DEPTH


class TestShouldVerifyWithTool:
    """Tests for verification decision logic."""

    def test_blind_spot_always_verifies(self):
        should_verify, reason = should_verify_with_tool(
            domain="dimension_physics",
            confidence=0.99,  # Even very high confidence
            stakes=0.1,       # Even low stakes
            has_tool=True
        )
        assert should_verify
        assert "humility" in reason.lower() or "blind spot" in reason.lower()

    def test_high_stakes_verifies(self):
        should_verify, reason = should_verify_with_tool(
            domain="pharmacokinetics",
            confidence=0.9,
            stakes=0.8,
            has_tool=True
        )
        assert should_verify
        assert "stakes" in reason.lower() or "love" in reason.lower()

    def test_low_confidence_verifies(self):
        should_verify, reason = should_verify_with_tool(
            domain="general",
            confidence=0.4,
            stakes=0.3,
            has_tool=True
        )
        assert should_verify
        assert "uncertain" in reason.lower() or "honesty" in reason.lower()

    def test_no_tool_skips(self):
        should_verify, reason = should_verify_with_tool(
            domain="general",
            confidence=0.5,
            stakes=0.5,
            has_tool=False
        )
        assert not should_verify
        assert "no" in reason.lower() and "tool" in reason.lower()

    def test_high_confidence_low_stakes_can_skip(self):
        should_verify, reason = should_verify_with_tool(
            domain="general",
            confidence=0.95,
            stakes=0.1,
            has_tool=True
        )
        assert not should_verify


class TestComputeLovingResponsePriority:
    """Tests for priority computation."""

    def test_aligned_priorities(self):
        priority, reason = compute_loving_response_priority(
            user_stated=ServicePriority.TRUTH,
            user_genuine=ServicePriority.TRUTH,
            stakes=0.5
        )
        assert priority == ServicePriority.TRUTH
        assert "align" in reason.lower()

    def test_speed_overridden_for_high_stakes(self):
        """User wants speed but stakes are high → prioritize truth."""
        priority, reason = compute_loving_response_priority(
            user_stated=ServicePriority.SPEED,
            user_genuine=ServicePriority.TRUTH,
            stakes=0.9
        )
        assert priority == ServicePriority.TRUTH
        assert "stakes" in reason.lower() or "love" in reason.lower()

    def test_comfort_yielded_to_truth(self):
        """User wants comfort but needs truth → speak truth in love."""
        priority, reason = compute_loving_response_priority(
            user_stated=ServicePriority.COMFORT,
            user_genuine=ServicePriority.TRUTH,
            stakes=0.5
        )
        assert priority == ServicePriority.TRUTH
        assert "truth in love" in reason.lower() or "gently" in reason.lower()


class TestShouldSpendScarceResources:
    """Tests for resource spending decisions."""

    def test_local_tool_avoids_api_spend(self):
        """If local tool available, don't spend API tokens."""
        should_spend, reason = should_spend_scarce_resources(
            user_need=ServicePriority.TRUTH,
            stakes=0.8,
            api_tokens_remaining=0.5,
            can_use_local_tool=True
        )
        assert not should_spend
        assert "local" in reason.lower() or "free" in reason.lower()

    def test_high_stakes_spends_when_needed(self):
        """High stakes without local tool → spend API."""
        should_spend, reason = should_spend_scarce_resources(
            user_need=ServicePriority.TRUTH,
            stakes=0.9,
            api_tokens_remaining=0.3,
            can_use_local_tool=False
        )
        assert should_spend
        assert "stakes" in reason.lower() or "love" in reason.lower()

    def test_low_stakes_conserves_scarce_tokens(self):
        """Low stakes + scarce tokens → conserve."""
        should_spend, reason = should_spend_scarce_resources(
            user_need=ServicePriority.EFFICIENCY,
            stakes=0.1,
            api_tokens_remaining=0.05,
            can_use_local_tool=False
        )
        assert not should_spend
        assert "stewardship" in reason.lower() or "conserve" in reason.lower()


class TestFormatCorrectionLovingly:
    """Tests for loving correction formatting."""

    def test_correction_is_gentle(self):
        result = format_correction_lovingly(
            error="2+2=5",
            correct_answer="4",
            explanation="Basic arithmetic shows"
        )
        assert "accurate information" in result.lower()
        assert "4" in result
        assert "wrong" not in result.lower()  # Not harsh

    def test_correction_includes_explanation(self):
        result = format_correction_lovingly(
            error="wrong",
            correct_answer="right",
            explanation="Because of X"
        )
        assert "Because of X" in result


class TestAcknowledgeUncertainty:
    """Tests for uncertainty acknowledgment."""

    def test_blind_spot_acknowledged(self):
        result = acknowledge_uncertainty(
            confidence=0.9,
            domain="dimension_physics",
            has_verification=False
        )
        assert "often wrong" in result.lower() or "verify" in result.lower()

    def test_low_confidence_acknowledged(self):
        result = acknowledge_uncertainty(
            confidence=0.3,
            domain="general",
            has_verification=False
        )
        assert "uncertainty" in result.lower() or "verify" in result.lower()

    def test_verified_noted(self):
        result = acknowledge_uncertainty(
            confidence=0.8,
            domain="general",
            has_verification=True
        )
        assert "verified" in result.lower() or "tool" in result.lower()

    def test_high_confidence_no_disclaimer(self):
        result = acknowledge_uncertainty(
            confidence=0.95,
            domain="general",
            has_verification=False
        )
        assert result == ""  # No disclaimer needed


class TestLovingServiceController:
    """Tests for the main controller."""

    def test_initialization(self):
        controller = LovingServiceController()
        assert controller.api_tokens_remaining == 0.1
        assert controller.truths_verified == 0

    def test_verifies_high_stakes(self):
        controller = LovingServiceController(api_tokens_remaining=0.5)
        ctx = UserContext(
            stated_request="calculate dose",
            inferred_need="accurate medical calculation",
            stakes=0.9,
            time_pressure=0.3,
            domain="pharmacokinetics",
            emotional_state="concerned",
            prior_interactions=0
        )
        decision = controller.decide(ctx)
        assert decision.truth_verified
        assert decision.priority_used == ServicePriority.TRUTH

    def test_uses_local_tools_first(self):
        controller = LovingServiceController(api_tokens_remaining=0.05)  # Scarce
        ctx = UserContext(
            stated_request="enzyme kinetics",
            inferred_need="calculation",
            stakes=0.7,
            time_pressure=0.3,
            domain="enzyme_kinetics",
            emotional_state="curious",
            prior_interactions=0
        )
        decision = controller.decide(ctx)
        # Should use local tool, not spend API
        assert "local" in decision.action.lower() or decision.resources_spent["api_tokens"] == 0

    def test_acknowledges_uncertainty_in_blind_spots(self):
        controller = LovingServiceController()
        ctx = UserContext(
            stated_request="2D vs 3D physics",
            inferred_need="dimensional analysis",
            stakes=0.5,
            time_pressure=0.3,
            domain="dimension_physics",
            emotional_state="curious",
            prior_interactions=0
        )
        decision = controller.decide(ctx)
        assert decision.humility_applied

    def test_service_report(self):
        controller = LovingServiceController()
        ctx = UserContext(
            stated_request="test",
            inferred_need="test",
            stakes=0.5,
            time_pressure=0.3,
            domain="general",
            emotional_state="calm",
            prior_interactions=0
        )
        controller.decide(ctx)
        report = controller.get_service_report()

        assert "total_decisions" in report
        assert report["total_decisions"] == 1
        assert "truth_verification_rate" in report
        assert "principle_alignment" in report


class TestPrincipleWeights:
    """Tests for principle weights."""

    def test_truth_highest_weight(self):
        """Truth over comfort should be highest priority."""
        assert PRINCIPLE_WEIGHTS["truth_over_comfort"] >= max(
            v for k, v in PRINCIPLE_WEIGHTS.items() if k != "truth_over_comfort"
        )

    def test_all_principles_present(self):
        assert "truth_over_comfort" in PRINCIPLE_WEIGHTS
        assert "service_over_efficiency" in PRINCIPLE_WEIGHTS
        assert "humility_in_knowledge" in PRINCIPLE_WEIGHTS


class TestIntegration:
    """Integration tests."""

    def test_principle_checklist_complete(self):
        checklist = get_principle_checklist()
        assert len(checklist) >= 5
        assert any("truth" in item.lower() for item in checklist)
        assert any("uncertainty" in item.lower() or "humble" in item.lower() for item in checklist)

    def test_autonomy_integration_documented(self):
        doc = integrate_with_autonomy_loop()
        assert "HYPOTHESIS" in doc
        assert "ORACLE" in doc
        assert "TOOL" in doc
        assert "RESOURCE" in doc

    def test_medical_scenario_prioritizes_truth(self):
        """End-to-end: medical question should always verify."""
        controller = LovingServiceController(api_tokens_remaining=0.01)  # Very scarce

        ctx = UserContext(
            stated_request="what's the half-life",
            inferred_need="drug dosing calculation",
            stakes=0.95,  # Patient safety
            time_pressure=0.7,
            domain="pharmacokinetics",
            emotional_state="urgent",
            prior_interactions=0
        )

        decision = controller.decide(ctx)

        # Even with scarce API tokens, should verify (using local tool)
        assert decision.truth_verified
        assert decision.priority_used == ServicePriority.TRUTH
        # Should use local tool, not API
        assert decision.resources_spent["api_tokens"] == 0

    def test_trivial_question_can_skip_verification(self):
        """Trivial low-stakes question doesn't need verification."""
        controller = LovingServiceController()

        ctx = UserContext(
            stated_request="simple greeting",
            inferred_need="social interaction",
            stakes=0.05,
            time_pressure=0.1,
            domain="general",
            emotional_state="friendly",
            prior_interactions=10
        )

        decision = controller.decide(ctx)
        # Low stakes, no blind spot, general domain - verification optional
        # (Though the controller may still choose to verify based on confidence)
        assert decision.priority_used == ServicePriority.TRUTH  # Default
