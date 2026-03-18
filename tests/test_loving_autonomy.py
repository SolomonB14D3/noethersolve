"""Tests for loving_autonomy module."""

import pytest
from noethersolve.loving_autonomy import (
    LovingAssistant,
    AssistantResponse,
)


class TestLovingAssistant:
    """Tests for LovingAssistant."""

    def test_initialization(self):
        assistant = LovingAssistant()
        assert assistant.mlx_available is not None
        assert assistant.compute_backend in ["mlx", "cuda", "mps", "cpu"]

    def test_get_status(self):
        assistant = LovingAssistant(api_tokens_remaining=0.2)
        status = assistant.get_status()
        assert "mlx_available" in status
        assert "compute_backend" in status
        assert status["api_tokens_remaining"] == 0.2

    def test_respond_basic(self):
        assistant = LovingAssistant()
        response = assistant.respond(
            query="What is 2+2?",
            domain="general",
            stakes=0.1
        )
        assert isinstance(response, AssistantResponse)
        assert response.answer is not None

    def test_high_stakes_verifies(self):
        assistant = LovingAssistant()
        response = assistant.respond(
            query="Calculate drug half-life",
            domain="pharmacokinetics",
            stakes=0.9
        )
        # High stakes + tool available → should verify
        assert response.verified
        assert response.tool_used is not None

    def test_blind_spot_acknowledged(self):
        assistant = LovingAssistant()
        response = assistant.respond(
            query="2D vs 3D physics",
            domain="dimension_physics",
            stakes=0.5
        )
        # Blind spot domain → uncertainty acknowledged
        assert "Acknowledged uncertainty" in str(response.checklist_passed) or response.uncertainty_note

    def test_uses_local_resources(self):
        assistant = LovingAssistant(api_tokens_remaining=0.01)  # Very scarce
        response = assistant.respond(
            query="enzyme kinetics calculation",
            domain="enzyme_kinetics",
            stakes=0.7
        )
        # Should use local tools, not API
        assert response.resources_used.get("api_tokens", 0) == 0

    def test_session_report(self):
        assistant = LovingAssistant()
        assistant.respond("test query", domain="general", stakes=0.5)
        report = assistant.get_session_report()
        assert "LOVING AUTONOMY SESSION REPORT" in report
        assert "MLX Available" in report

    def test_actual_tool_call_half_life(self):
        """Test that half-life calculation actually runs."""
        assistant = LovingAssistant()
        response = assistant.respond(
            query="What is the half-life with elimination constant 0.1 per hour?",
            domain="pharmacokinetics",
            stakes=0.9
        )
        # Should contain actual calculation
        assert "6.93" in response.answer or "t½" in response.answer

    def test_multiple_responses_tracked(self):
        assistant = LovingAssistant()
        assistant.respond("query 1", domain="general", stakes=0.3)
        assistant.respond("query 2", domain="general", stakes=0.5)
        assert len(assistant.responses) == 2
        assert assistant.get_status()["responses_given"] == 2


class TestAssistantResponse:
    """Tests for AssistantResponse."""

    def test_str_format(self):
        response = AssistantResponse(
            answer="The answer is 42",
            verified=True,
            tool_used="calc_something",
            uncertainty_note="This is verified",
            reasoning="High stakes",
            resources_used={"local_compute": 0.01},
            checklist_passed=["Truth verified"]
        )
        s = str(response)
        assert "42" in s
        assert "verified" in s.lower()

    def test_full_report(self):
        response = AssistantResponse(
            answer="Test answer",
            verified=True,
            tool_used="test_tool",
            uncertainty_note="Note",
            reasoning="Reason",
            resources_used={"local_compute": 0.01, "api_tokens": 0.0},
            checklist_passed=["Item 1", "Item 2"]
        )
        report = response.full_report()
        assert "LOVING ASSISTANT RESPONSE" in report
        assert "VERIFICATION" in report
        assert "CHECKLIST" in report


class TestIntegration:
    """Integration tests."""

    def test_end_to_end_pharmacokinetics(self):
        """Full flow for a pharmacokinetics question."""
        assistant = LovingAssistant(api_tokens_remaining=0.05)

        response = assistant.respond(
            query="Calculate the half-life for ke=0.2/hr",
            domain="pharmacokinetics",
            stakes=0.95,  # Patient safety
            time_pressure=0.3,
            emotional_state="concerned"
        )

        # Should have verified with local tool
        assert response.verified
        assert response.resources_used.get("api_tokens", 0) == 0
        # Should have meaningful checklist items
        assert len(response.checklist_passed) >= 2

    def test_mlx_preference(self):
        """On Mac with MLX, should prefer local compute."""
        assistant = LovingAssistant()

        if assistant.mlx_available:
            # With MLX, local compute should be very cheap
            assert assistant.budget.weights["local_compute"] <= 0.01
