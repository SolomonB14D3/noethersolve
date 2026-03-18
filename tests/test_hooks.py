"""Tests for hooks module."""

import tempfile
from pathlib import Path
from unittest.mock import patch


from noethersolve.hooks import (
    SessionState,
    is_local_tool,
    is_verification_tool,
    get_tool_domain,
    pre_tool_use,
    post_tool_use,
    get_session_stats,
    categorize_tools,
)


class TestIsLocalTool:
    """Tests for is_local_tool function."""

    def test_noethersolve_tools_are_local(self):
        assert is_local_tool("check_vortex_conservation") is True
        assert is_local_tool("calc_iv_bolus") is True
        assert is_local_tool("calc_michaelis_menten") is True

    def test_mcp_noethersolve_tools_are_local(self):
        assert is_local_tool("mcp__noethersolve__calc_half_life") is True
        assert is_local_tool("noethersolve_check_conjecture") is True

    def test_claude_code_tools_are_local(self):
        assert is_local_tool("Read") is True
        assert is_local_tool("Write") is True
        assert is_local_tool("Edit") is True
        assert is_local_tool("Glob") is True
        assert is_local_tool("Grep") is True
        assert is_local_tool("Bash") is True

    def test_unknown_tools_are_not_local(self):
        assert is_local_tool("some_random_api_tool") is False
        assert is_local_tool("external_service_call") is False


class TestIsVerificationTool:
    """Tests for is_verification_tool function."""

    def test_check_tools_are_verification(self):
        assert is_verification_tool("check_conjecture") is True
        assert is_verification_tool("check_vortex_conservation") is True

    def test_verify_tools_are_verification(self):
        assert is_verification_tool("verify_goldbach") is True
        assert is_verification_tool("verify_collatz") is True

    def test_calc_tools_are_verification(self):
        assert is_verification_tool("calc_iv_bolus") is True
        assert is_verification_tool("calc_half_life") is True

    def test_audit_tools_are_verification(self):
        assert is_verification_tool("audit_dna_sequence") is True
        assert is_verification_tool("audit_network") is True

    def test_non_verification_tools(self):
        assert is_verification_tool("Read") is False
        assert is_verification_tool("Write") is False
        assert is_verification_tool("some_random_tool") is False


class TestGetToolDomain:
    """Tests for get_tool_domain function."""

    def test_pharmacokinetics_domain(self):
        assert get_tool_domain("calc_iv_bolus") == "pharmacokinetics"
        assert get_tool_domain("calc_oral_dose") == "pharmacokinetics"
        assert get_tool_domain("calc_half_life") == "pharmacokinetics"

    def test_enzyme_kinetics_domain(self):
        assert get_tool_domain("calc_michaelis_menten") == "enzyme_kinetics"
        assert get_tool_domain("calc_enzyme_inhibition") == "enzyme_kinetics"

    def test_mathematics_domain(self):
        assert get_tool_domain("check_conjecture") == "mathematics"

    def test_unknown_domain(self):
        assert get_tool_domain("some_random_tool") is None
        assert get_tool_domain("Read") is None


class TestSessionState:
    """Tests for SessionState class."""

    def test_initialization(self):
        state = SessionState()
        assert state.tool_calls == 0
        assert state.local_tool_calls == 0
        assert state.api_calls == 0
        assert state.verifications == 0
        assert state.tools_used == []
        assert state.domains_touched == []
        assert state.warnings == []

    def test_to_dict(self):
        state = SessionState(tool_calls=5, local_tool_calls=3)
        d = state.to_dict()
        assert d["tool_calls"] == 5
        assert d["local_tool_calls"] == 3
        assert isinstance(d, dict)

    def test_from_dict(self):
        d = {
            "session_id": "test123",
            "start_time": "2026-03-17T10:00:00",
            "tool_calls": 10,
            "local_tool_calls": 8,
            "api_calls": 2,
            "verifications": 5,
            "blind_spot_checks": 1,
            "resources_used": {"local_compute": 0.1, "api_tokens": 0.02},
            "tools_used": ["calc_half_life"],
            "domains_touched": ["pharmacokinetics"],
            "warnings": [],
        }
        state = SessionState.from_dict(d)
        assert state.session_id == "test123"
        assert state.tool_calls == 10
        assert state.local_tool_calls == 8

    def test_save_and_load(self):
        # Use temp directory
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("noethersolve.hooks.STATE_DIR", Path(tmpdir)):
                with patch("noethersolve.hooks.STATE_FILE", Path(tmpdir) / "state.json"):
                    state = SessionState(
                        session_id="test_save",
                        tool_calls=7,
                        local_tool_calls=5,
                    )
                    state.save()

                    loaded = SessionState.load()
                    assert loaded.session_id == "test_save"
                    assert loaded.tool_calls == 7


class TestPreToolUse:
    """Tests for pre_tool_use hook."""

    def test_increments_tool_calls(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("noethersolve.hooks.STATE_DIR", Path(tmpdir)):
                with patch("noethersolve.hooks.STATE_FILE", Path(tmpdir) / "state.json"):
                    # Clear state
                    state = SessionState()
                    state.save()

                    pre_tool_use("calc_half_life")

                    state = SessionState.load()
                    assert state.tool_calls == 1
                    assert "calc_half_life" in state.tools_used

    def test_tracks_domain(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("noethersolve.hooks.STATE_DIR", Path(tmpdir)):
                with patch("noethersolve.hooks.STATE_FILE", Path(tmpdir) / "state.json"):
                    state = SessionState()
                    state.save()

                    pre_tool_use("calc_iv_bolus")

                    state = SessionState.load()
                    assert "pharmacokinetics" in state.domains_touched


class TestPostToolUse:
    """Tests for post_tool_use hook."""

    def test_tracks_local_tool(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("noethersolve.hooks.STATE_DIR", Path(tmpdir)):
                with patch("noethersolve.hooks.STATE_FILE", Path(tmpdir) / "state.json"):
                    with patch("noethersolve.hooks.USAGE_LOG", Path(tmpdir) / "usage.jsonl"):
                        state = SessionState()
                        state.save()

                        post_tool_use("calc_half_life", "result")

                        state = SessionState.load()
                        assert state.local_tool_calls == 1
                        assert state.api_calls == 0

    def test_tracks_verification(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("noethersolve.hooks.STATE_DIR", Path(tmpdir)):
                with patch("noethersolve.hooks.STATE_FILE", Path(tmpdir) / "state.json"):
                    with patch("noethersolve.hooks.USAGE_LOG", Path(tmpdir) / "usage.jsonl"):
                        state = SessionState()
                        state.save()

                        post_tool_use("verify_goldbach", "result")

                        state = SessionState.load()
                        assert state.verifications == 1


class TestCategorizeTool:
    """Tests for categorize_tools function."""

    def test_categorizes_pharmacokinetics(self):
        tools = ["calc_iv_bolus", "calc_oral_dose", "calc_half_life"]
        categories = categorize_tools(tools)
        assert "pharmacokinetics" in categories
        assert len(categories["pharmacokinetics"]) == 3

    def test_categorizes_enzyme_kinetics(self):
        tools = ["calc_michaelis_menten", "calc_enzyme_inhibition"]
        categories = categorize_tools(tools)
        assert "enzyme_kinetics" in categories
        assert len(categories["enzyme_kinetics"]) == 2

    def test_categorizes_mathematics(self):
        tools = ["check_conjecture", "verify_goldbach"]
        categories = categorize_tools(tools)
        assert "mathematics" in categories
        assert len(categories["mathematics"]) == 2

    def test_mixed_tools(self):
        tools = [
            "calc_iv_bolus",
            "calc_michaelis_menten",
            "check_conjecture",
            "some_unknown",
        ]
        categories = categorize_tools(tools)
        assert "pharmacokinetics" in categories
        assert "enzyme_kinetics" in categories
        assert "mathematics" in categories
        assert "other" in categories


class TestGetSessionStats:
    """Tests for get_session_stats function."""

    def test_returns_dict(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("noethersolve.hooks.STATE_DIR", Path(tmpdir)):
                with patch("noethersolve.hooks.STATE_FILE", Path(tmpdir) / "state.json"):
                    state = SessionState(tool_calls=5, verifications=2)
                    state.save()

                    stats = get_session_stats()
                    assert stats["tool_calls"] == 5
                    assert stats["verifications"] == 2
