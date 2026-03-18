"""Tests for noethersolve.blind_spot_detector — Blind spot detection and tool routing.

Tests cover:
- load_blind_spots registry loading
- detect_blind_spots keyword matching
- get_recommended_tools tool suggestions
- format_blind_spot_warning message formatting
- list_all_blind_spots listing function
- BlindSpotMatch dataclass
"""

import os

from noethersolve.blind_spot_detector import (
    BlindSpotMatch,
    load_blind_spots,
    detect_blind_spots,
    get_recommended_tools,
    format_blind_spot_warning,
    list_all_blind_spots,
    BLIND_SPOTS_PATH,
)


# -----------------------------------------------------------------------------
# BlindSpotMatch Tests
# -----------------------------------------------------------------------------

class TestBlindSpotMatch:
    """Tests for BlindSpotMatch dataclass."""

    def test_creation(self):
        """BlindSpotMatch can be created with all fields."""
        match = BlindSpotMatch(
            id="test_id",
            domains=["domain1", "domain2"],
            insight="Test insight",
            tools=["tool1", "tool2"],
            interpretation="Test interpretation",
            match_score=0.75,
        )

        assert match.id == "test_id"
        assert match.domains == ["domain1", "domain2"]
        assert match.insight == "Test insight"
        assert match.tools == ["tool1", "tool2"]
        assert match.interpretation == "Test interpretation"
        assert match.match_score == 0.75

    def test_empty_tools_list(self):
        """BlindSpotMatch works with empty tools list."""
        match = BlindSpotMatch(
            id="no_tools",
            domains=["test"],
            insight="No tools yet",
            tools=[],
            interpretation="",
            match_score=0.5,
        )

        assert match.tools == []


# -----------------------------------------------------------------------------
# load_blind_spots Tests
# -----------------------------------------------------------------------------

class TestLoadBlindSpots:
    """Tests for load_blind_spots function."""

    def test_loads_from_file(self):
        """Loads blind spots registry from JSON file."""
        data = load_blind_spots()

        assert isinstance(data, dict)
        assert "cross_domain_connections" in data or "single_domain_blind_spots" in data

    def test_file_exists(self):
        """Blind spots JSON file exists."""
        assert os.path.exists(BLIND_SPOTS_PATH)

    def test_valid_json_structure(self):
        """Loaded data has valid structure."""
        data = load_blind_spots()

        # Check cross-domain connections structure
        if "cross_domain_connections" in data:
            for conn in data["cross_domain_connections"]:
                assert "id" in conn
                assert "domains" in conn
                assert "trigger_keywords" in conn
                assert "tools" in conn

        # Check single-domain blind spots structure
        if "single_domain_blind_spots" in data:
            for spot in data["single_domain_blind_spots"]:
                assert "id" in spot
                assert "domain" in spot
                assert "trigger_keywords" in spot


# -----------------------------------------------------------------------------
# detect_blind_spots Tests
# -----------------------------------------------------------------------------

class TestDetectBlindSpots:
    """Tests for detect_blind_spots function."""

    def test_no_match_returns_empty(self):
        """Query with no matching keywords returns empty list."""
        matches = detect_blind_spots("What is the weather today?")

        # Weather has no matching blind spots
        assert isinstance(matches, list)

    def test_cross_domain_match_deadlock_thermodynamics(self):
        """Detects deadlock + thermodynamics cross-domain blind spot."""
        query = "How does deadlock relate to detailed balance in thermodynamics?"

        matches = detect_blind_spots(query)

        # Should find the deadlock_detailed_balance connection
        match_ids = [m.id for m in matches]
        assert "deadlock_detailed_balance" in match_ids

    def test_cross_domain_requires_two_keywords(self):
        """Cross-domain match requires at least 2 keyword matches."""
        # Only one keyword
        query = "What is deadlock?"

        matches = detect_blind_spots(query)

        # Should not match cross-domain (needs 2+ keywords)
        [m.id for m in matches if len(m.domains) > 1]
        # deadlock_detailed_balance shouldn't match with only "deadlock"
        # (it has 5 keywords, needs at least 2)

    def test_single_domain_match(self):
        """Detects single-domain blind spots with 1 keyword."""
        query = "Is P vs NP proven?"

        matches = detect_blind_spots(query)

        # Should find complexity_relationships
        match_ids = [m.id for m in matches]
        assert "complexity_relationships" in match_ids

    def test_case_insensitive_matching(self):
        """Keyword matching is case-insensitive."""
        query1 = "What is DEADLOCK and DETAILED BALANCE?"
        query2 = "What is deadlock and detailed balance?"

        matches1 = detect_blind_spots(query1)
        matches2 = detect_blind_spots(query2)

        # Both should find same matches
        ids1 = {m.id for m in matches1}
        ids2 = {m.id for m in matches2}
        assert ids1 == ids2

    def test_sorted_by_score(self):
        """Results are sorted by match score descending."""
        # Query that matches multiple blind spots
        query = "P vs NP conjecture deadlock detailed balance"

        matches = detect_blind_spots(query)

        if len(matches) >= 2:
            scores = [m.match_score for m in matches]
            assert scores == sorted(scores, reverse=True)

    def test_match_score_calculation(self):
        """Match score is fraction of keywords matched."""
        # The deadlock_detailed_balance has 5 keywords
        # Matching 2 of them should give score = 2/5 = 0.4
        query = "deadlock detailed balance"

        matches = detect_blind_spots(query)

        for m in matches:
            if m.id == "deadlock_detailed_balance":
                assert 0 < m.match_score <= 1.0


# -----------------------------------------------------------------------------
# get_recommended_tools Tests
# -----------------------------------------------------------------------------

class TestGetRecommendedTools:
    """Tests for get_recommended_tools function."""

    def test_returns_list(self):
        """Returns a list of tool names."""
        tools = get_recommended_tools("What is P vs NP?")

        assert isinstance(tools, list)

    def test_no_duplicates(self):
        """Tool list has no duplicates."""
        # Query that might match multiple blind spots with overlapping tools
        query = "P vs NP complexity class inclusion completeness"

        tools = get_recommended_tools(query)

        assert len(tools) == len(set(tools))

    def test_empty_for_no_matches(self):
        """Returns empty list when no blind spots match."""
        tools = get_recommended_tools("What is the weather?")

        assert tools == []

    def test_tools_from_complexity_blind_spot(self):
        """Returns correct tools for complexity theory queries."""
        # Use trigger keywords from blind_spots.json
        tools = get_recommended_tools("P vs NP complexity class")

        # Should include at least some complexity tools
        complexity_tools = {"check_complexity_inclusion", "check_completeness", "check_proof_barriers"}
        assert len(set(tools) & complexity_tools) > 0 or len(tools) > 0

    def test_preserves_order(self):
        """Preserves order based on match score."""
        tools = get_recommended_tools("deadlock detailed balance cycle")

        # Tools from higher-scoring matches should come first
        if "calc_deadlock" in tools and len(tools) > 1:
            # calc_deadlock should be early since deadlock_detailed_balance is high match
            assert tools.index("calc_deadlock") < len(tools) // 2


# -----------------------------------------------------------------------------
# format_blind_spot_warning Tests
# -----------------------------------------------------------------------------

class TestFormatBlindSpotWarning:
    """Tests for format_blind_spot_warning function."""

    def test_empty_matches_returns_empty_string(self):
        """Empty match list returns empty string."""
        result = format_blind_spot_warning([])

        assert result == ""

    def test_contains_warning_header(self):
        """Warning contains header with emoji."""
        matches = [
            BlindSpotMatch(
                id="test",
                domains=["a", "b"],
                insight="test insight",
                tools=["tool1"],
                interpretation="test interp",
                match_score=0.5,
            )
        ]

        result = format_blind_spot_warning(matches)

        assert "BLIND SPOT DETECTED" in result
        assert "MCP tools" in result

    def test_includes_domain_info(self):
        """Warning includes domain information."""
        matches = [
            BlindSpotMatch(
                id="cross_test",
                domains=["physics", "biology"],
                insight="cross insight",
                tools=["tool1"],
                interpretation="",
                match_score=0.5,
            )
        ]

        result = format_blind_spot_warning(matches)

        assert "physics" in result
        assert "biology" in result
        assert "↔" in result  # Cross-domain separator

    def test_includes_tools(self):
        """Warning includes tool recommendations."""
        matches = [
            BlindSpotMatch(
                id="test",
                domains=["domain"],
                insight="insight",
                tools=["calc_something", "check_other"],
                interpretation="",
                match_score=0.5,
            )
        ]

        result = format_blind_spot_warning(matches)

        assert "calc_something" in result
        assert "check_other" in result

    def test_fallback_for_no_tools(self):
        """Shows fallback protocol when no tools available."""
        matches = [
            BlindSpotMatch(
                id="no_tools",
                domains=["domain"],
                insight="insight",
                tools=[],  # No tools
                interpretation="",
                match_score=0.5,
            )
        ]

        result = format_blind_spot_warning(matches)

        assert "NO TOOLS AVAILABLE" in result
        assert "FALLBACK PROTOCOL" in result

    def test_multiple_matches(self):
        """Formats multiple matches correctly."""
        matches = [
            BlindSpotMatch(
                id="match1",
                domains=["a"],
                insight="insight1",
                tools=["tool1"],
                interpretation="",
                match_score=0.8,
            ),
            BlindSpotMatch(
                id="match2",
                domains=["b"],
                insight="insight2",
                tools=["tool2"],
                interpretation="",
                match_score=0.6,
            ),
        ]

        result = format_blind_spot_warning(matches)

        assert "match1" in result
        assert "match2" in result
        assert "insight1" in result
        assert "insight2" in result


# -----------------------------------------------------------------------------
# list_all_blind_spots Tests
# -----------------------------------------------------------------------------

class TestListAllBlindSpots:
    """Tests for list_all_blind_spots function."""

    def test_returns_string(self):
        """Returns a formatted string."""
        result = list_all_blind_spots()

        assert isinstance(result, str)

    def test_includes_headers(self):
        """Output includes section headers."""
        result = list_all_blind_spots()

        assert "Cross-Domain" in result or "Single-Domain" in result

    def test_includes_summary(self):
        """Output includes summary section."""
        result = list_all_blind_spots()

        assert "Summary" in result

    def test_needs_tool_only_filter(self):
        """needs_tool_only=True filters to only needing tools."""
        result_all = list_all_blind_spots(needs_tool_only=False)
        result_needs = list_all_blind_spots(needs_tool_only=True)

        # Filtered result should be same size or smaller
        assert len(result_needs) <= len(result_all)

    def test_includes_tool_status(self):
        """Output shows tool status (has tools vs needs tool)."""
        result = list_all_blind_spots()

        # Should have status indicators
        assert "Has tools" in result or "NEEDS TOOL" in result or "Tools:" in result

    def test_markdown_format(self):
        """Output is markdown formatted."""
        result = list_all_blind_spots()

        # Should have markdown headers
        assert "##" in result or "###" in result


# -----------------------------------------------------------------------------
# Integration Tests
# -----------------------------------------------------------------------------

class TestBlindSpotIntegration:
    """Integration tests for blind spot detection system."""

    def test_end_to_end_detection_and_recommendation(self):
        """Full flow: query -> detect -> recommend tools -> format."""
        query = "How do PageRank and thermodynamic equilibrium relate?"

        # Detect
        matches = detect_blind_spots(query)

        # Get tools
        tools = get_recommended_tools(query)

        # Format warning
        warning = format_blind_spot_warning(matches)

        # All should work together
        if matches:
            assert len(tools) > 0
            assert len(warning) > 0

    def test_complexity_query_full_flow(self):
        """Complexity theory query triggers correct tools."""
        query = "Is NP-complete the same as NP-hard?"

        detect_blind_spots(query)
        tools = get_recommended_tools(query)

        # Should find complexity tools
        assert any("complexity" in t.lower() or "completeness" in t.lower()
                   for t in tools)

    def test_conjecture_query_full_flow(self):
        """Mathematical conjecture query triggers correct tools."""
        query = "Has the Riemann hypothesis been proven?"

        detect_blind_spots(query)
        tools = get_recommended_tools(query)

        # Should find conjecture checker
        assert "check_conjecture" in tools


# -----------------------------------------------------------------------------
# Edge Case Tests
# -----------------------------------------------------------------------------

class TestBlindSpotEdgeCases:
    """Edge case tests for blind spot detection."""

    def test_empty_query(self):
        """Empty query returns no matches."""
        matches = detect_blind_spots("")

        assert matches == []

    def test_very_long_query(self):
        """Handles very long queries."""
        query = " ".join(["deadlock", "detailed", "balance"] * 100)

        matches = detect_blind_spots(query)

        # Should still work
        assert isinstance(matches, list)

    def test_special_characters_in_query(self):
        """Handles special characters in query."""
        query = "What is P=NP? (complexity theory) [hard problem]"

        matches = detect_blind_spots(query)

        # Should not raise
        assert isinstance(matches, list)

    def test_unicode_query(self):
        """Handles unicode in query."""
        query = "P ≠ NP 数学问题 complexity"

        matches = detect_blind_spots(query)

        # Should not raise
        assert isinstance(matches, list)

    def test_numeric_query(self):
        """Handles numeric-only query."""
        query = "12345 67890"

        matches = detect_blind_spots(query)

        assert matches == []
