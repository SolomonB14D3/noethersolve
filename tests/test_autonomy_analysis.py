"""Tests for autonomy_analysis module."""

import pytest
from noethersolve.autonomy_analysis import (
    AutonomyFramework,
    ComponentStatus,
    AutonomyComponent,
    ComponentAssessment,
    AutonomyReport,
    AUTONOMY_COMPONENTS,
    SYSTEM_PROFILES,
    IMPLEMENTATION_APPROACHES,
    get_all_components,
    assess_system,
    assess_predefined_system,
    compare_systems,
    identify_autonomy_gaps,
    check_autonomy_requirements,
    analyze_transformer_autonomy,
    list_frameworks,
    list_predefined_systems,
    get_implementation_roadmap,
    design_autonomous_system,
    get_minimum_viable_autonomy,
    get_full_autonomy_blueprint,
)


class TestAutonomyFramework:
    """Tests for the AutonomyFramework enum."""

    def test_all_frameworks_present(self):
        """All five theoretical frameworks should be present."""
        frameworks = list(AutonomyFramework)
        assert len(frameworks) == 5
        assert AutonomyFramework.CONTROL_THEORY in frameworks
        assert AutonomyFramework.AUTOPOIESIS in frameworks
        assert AutonomyFramework.PHILOSOPHY in frameworks
        assert AutonomyFramework.ROBOTICS in frameworks
        assert AutonomyFramework.COGNITIVE_SCIENCE in frameworks


class TestComponentStatus:
    """Tests for ComponentStatus enum."""

    def test_all_statuses(self):
        """All status levels should exist."""
        statuses = list(ComponentStatus)
        assert len(statuses) == 4
        assert ComponentStatus.PRESENT in statuses
        assert ComponentStatus.PARTIAL in statuses
        assert ComponentStatus.ABSENT in statuses
        assert ComponentStatus.UNKNOWN in statuses


class TestAutonomyComponents:
    """Tests for the component definitions."""

    def test_all_frameworks_have_components(self):
        """Each framework should have defined components."""
        for framework in AutonomyFramework:
            assert framework in AUTONOMY_COMPONENTS
            assert len(AUTONOMY_COMPONENTS[framework]) >= 4

    def test_component_structure(self):
        """Components should have required fields."""
        for framework, components in AUTONOMY_COMPONENTS.items():
            for component in components:
                assert isinstance(component, AutonomyComponent)
                assert component.name
                assert component.framework == framework
                assert component.description
                assert isinstance(component.is_necessary, bool)

    def test_get_all_components(self):
        """get_all_components should return all components."""
        all_components = get_all_components()
        total = sum(len(comps) for comps in AUTONOMY_COMPONENTS.values())
        assert len(all_components) == total
        # Should be 25 total (5 per framework)
        assert len(all_components) == 25

    def test_control_theory_components(self):
        """Control theory should have specific components."""
        components = AUTONOMY_COMPONENTS[AutonomyFramework.CONTROL_THEORY]
        names = [c.name for c in components]
        assert "feedback_loop" in names
        assert "setpoint_generation" in names
        assert "error_correction" in names

    def test_autopoiesis_components(self):
        """Autopoiesis should have specific components."""
        components = AUTONOMY_COMPONENTS[AutonomyFramework.AUTOPOIESIS]
        names = [c.name for c in components]
        assert "operational_closure" in names
        assert "self_production" in names
        assert "boundary_maintenance" in names


class TestSystemProfiles:
    """Tests for predefined system profiles."""

    def test_all_profiles_exist(self):
        """All canonical systems should be defined."""
        assert "llm_transformer" in SYSTEM_PROFILES
        assert "thermostat" in SYSTEM_PROFILES
        assert "living_cell" in SYSTEM_PROFILES
        assert "autonomous_robot" in SYSTEM_PROFILES
        assert "human" in SYSTEM_PROFILES

    def test_profile_structure(self):
        """Each profile should have description and assessments."""
        for key, profile in SYSTEM_PROFILES.items():
            assert "description" in profile
            assert "assessments" in profile
            assert len(profile["assessments"]) >= 20  # At least 20 components assessed

    def test_assessment_structure(self):
        """Assessments should have (status, evidence, confidence) tuples."""
        for key, profile in SYSTEM_PROFILES.items():
            for component, assessment in profile["assessments"].items():
                assert len(assessment) == 3
                status, evidence, confidence = assessment
                assert isinstance(status, ComponentStatus)
                assert isinstance(evidence, str)
                assert 0.0 <= confidence <= 1.0


class TestAssessSystem:
    """Tests for assess_system function."""

    def test_basic_assessment(self):
        """Basic assessment should work."""
        statuses = {
            "feedback_loop": (ComponentStatus.PRESENT, "Has feedback", 0.9),
            "setpoint_generation": (ComponentStatus.ABSENT, "No goals", 0.95),
        }
        report = assess_system("test_system", "Test System", statuses)

        assert report.system_name == "test_system"
        assert report.system_type == "Test System"
        assert len(report.assessments) == 25  # All components assessed
        assert 0.0 <= report.overall_autonomy_score <= 1.0

    def test_critical_gaps_identified(self):
        """Critical gaps should be identified."""
        statuses = {
            "feedback_loop": (ComponentStatus.ABSENT, "No feedback", 0.95),
            "operational_closure": (ComponentStatus.ABSENT, "No closure", 0.95),
        }
        report = assess_system("gap_system", "Gap System", statuses)

        assert len(report.critical_gaps) >= 2
        assert any("feedback_loop" in gap for gap in report.critical_gaps)

    def test_report_string(self):
        """Report should have readable string output."""
        statuses = {
            "feedback_loop": (ComponentStatus.PRESENT, "Has feedback", 0.9),
        }
        report = assess_system("test", "Test", statuses)
        s = str(report)

        assert "AUTONOMY ANALYSIS" in s
        assert "test" in s
        assert "Overall Autonomy Score" in s


class TestAssessPredefinedSystem:
    """Tests for assess_predefined_system function."""

    def test_llm_transformer(self):
        """LLM transformer should have low autonomy score."""
        report = assess_predefined_system("llm_transformer")

        assert report.system_name == "llm_transformer"
        assert report.overall_autonomy_score < 0.5  # Non-autonomous
        assert len(report.critical_gaps) > 5  # Many gaps

    def test_human(self):
        """Human should have high autonomy score."""
        report = assess_predefined_system("human")

        assert report.system_name == "human"
        assert report.overall_autonomy_score > 0.8  # Highly autonomous
        assert len(report.critical_gaps) == 0  # No critical gaps

    def test_living_cell(self):
        """Living cell should score well on autopoiesis."""
        report = assess_predefined_system("living_cell")

        autopoiesis_score = report.framework_scores["autopoiesis"]
        assert autopoiesis_score > 0.8  # Canonical autopoietic system

    def test_thermostat(self):
        """Thermostat should score well on control theory only."""
        report = assess_predefined_system("thermostat")

        control_score = report.framework_scores["control_theory"]
        cognitive_score = report.framework_scores["cognitive_science"]

        assert control_score > cognitive_score  # Better at control than cognition

    def test_unknown_system_raises(self):
        """Unknown system should raise ValueError."""
        with pytest.raises(ValueError):
            assess_predefined_system("nonexistent_system")


class TestCompareSystems:
    """Tests for compare_systems function."""

    def test_compare_two_systems(self):
        """Comparing two systems should work."""
        result = compare_systems(["llm_transformer", "human"])

        assert "systems" in result
        assert "key_differences" in result
        assert "ranking" in result
        assert len(result["systems"]) == 2

    def test_ranking_order(self):
        """Systems should be ranked by autonomy score."""
        result = compare_systems(["llm_transformer", "human", "thermostat"])

        ranking = result["ranking"]
        assert ranking[0] == "human"  # Human should be most autonomous
        assert "llm_transformer" in ranking

    def test_key_differences(self):
        """Key differences should be identified."""
        result = compare_systems(["llm_transformer", "living_cell"])

        differences = result["key_differences"]
        assert len(differences) > 0

        # Should find differences in autopoiesis components
        component_names = [d["component"] for d in differences]
        assert any(c in component_names for c in ["operational_closure", "self_production"])


class TestIdentifyAutonomyGaps:
    """Tests for identify_autonomy_gaps function."""

    def test_llm_gaps(self):
        """LLM should have identifiable gaps."""
        gaps = identify_autonomy_gaps("llm_transformer")

        assert gaps["system"] == "llm_transformer"
        assert gaps["absent_count"] > 5
        assert len(gaps["architectural_limitations"]) > 0
        assert "conclusion" in gaps

    def test_human_no_gaps(self):
        """Human should have minimal gaps."""
        gaps = identify_autonomy_gaps("human")

        assert gaps["absent_count"] == 0
        assert len(gaps["architectural_limitations"]) == 0

    def test_fixable_vs_unfixable(self):
        """Gaps should be categorized as fixable or unfixable."""
        gaps = identify_autonomy_gaps("llm_transformer")

        # LLM has unfixable gaps
        assert len(gaps["architectural_limitations"]) > 0

        # Check specific unfixable gaps
        unfixable_names = [g["component"] for g in gaps["architectural_limitations"]]
        assert "embodiment" in unfixable_names
        assert "persistent_identity" in unfixable_names

    def test_gaps_by_framework(self):
        """Gaps should be organized by framework."""
        gaps = identify_autonomy_gaps("llm_transformer")

        assert "gaps_by_framework" in gaps
        for framework in AutonomyFramework:
            assert framework.value in gaps["gaps_by_framework"]


class TestCheckAutonomyRequirements:
    """Tests for check_autonomy_requirements function."""

    def test_no_requirements_met(self):
        """System with no requirements should be non-autonomous."""
        result = check_autonomy_requirements(
            has_feedback=False,
            has_goals=False,
            has_embodiment=False,
            has_memory=False,
            has_self_production=False,
            maintains_boundary=False,
            has_metacognition=False
        )

        assert result["autonomy_level"] == "NON_AUTONOMOUS"
        assert result["score"] == 0.0
        assert len(result["requirements_missing"]) == 7

    def test_all_requirements_met(self):
        """System with all requirements should be fully autonomous."""
        result = check_autonomy_requirements(
            has_feedback=True,
            has_goals=True,
            has_embodiment=True,
            has_memory=True,
            has_self_production=True,
            maintains_boundary=True,
            has_metacognition=True
        )

        assert result["autonomy_level"] == "FULLY_AUTONOMOUS"
        assert result["score"] == 1.0
        assert len(result["requirements_met"]) == 7

    def test_partial_requirements(self):
        """System with some requirements should be partially autonomous."""
        result = check_autonomy_requirements(
            has_feedback=True,
            has_goals=True,
            has_embodiment=True,  # 3/7 = 43% -> MINIMALLY_AUTONOMOUS
            has_memory=False,
            has_self_production=False,
            maintains_boundary=False,
            has_metacognition=False
        )

        assert result["autonomy_level"] in ["MINIMALLY_AUTONOMOUS", "PARTIALLY_AUTONOMOUS"]
        assert 0.3 <= result["score"] < 0.6

    def test_llm_like_profile(self):
        """LLM-like profile should be non-autonomous."""
        result = check_autonomy_requirements(
            has_feedback=False,  # No environment feedback
            has_goals=False,     # Goals from prompts
            has_embodiment=False,
            has_memory=False,    # No persistent memory
            has_self_production=False,
            maintains_boundary=False,
            has_metacognition=False
        )

        assert result["autonomy_level"] == "NON_AUTONOMOUS"


class TestAnalyzeTransformerAutonomy:
    """Tests for analyze_transformer_autonomy function."""

    def test_returns_analysis(self):
        """Should return comprehensive analysis."""
        analysis = analyze_transformer_autonomy()

        assert "summary" in analysis
        assert "key_insight" in analysis
        assert "framework_analysis" in analysis
        assert "what_transformers_have" in analysis
        assert "what_transformers_lack" in analysis
        assert "architectural_fixes_needed" in analysis

    def test_summary_is_non_autonomous(self):
        """Summary should indicate non-autonomous."""
        analysis = analyze_transformer_autonomy()

        assert "non-autonomous" in analysis["summary"].lower()

    def test_key_insight_mentions_architecture(self):
        """Key insight should mention architectural limitations."""
        analysis = analyze_transformer_autonomy()

        assert "architectural" in analysis["key_insight"].lower()

    def test_what_transformers_lack(self):
        """Should list key missing components."""
        analysis = analyze_transformer_autonomy()

        lacks = analysis["what_transformers_lack"]
        lack_text = " ".join(lacks).lower()

        assert "operational closure" in lack_text
        assert "persistent identity" in lack_text or "identity" in lack_text
        assert "embodiment" in lack_text

    def test_framework_analysis_complete(self):
        """All frameworks should be analyzed."""
        analysis = analyze_transformer_autonomy()

        for framework in AutonomyFramework:
            assert framework.value in analysis["framework_analysis"]


class TestListFrameworks:
    """Tests for list_frameworks function."""

    def test_lists_all_frameworks(self):
        """Should list all five frameworks."""
        frameworks = list_frameworks()

        assert len(frameworks) == 5
        for framework in AutonomyFramework:
            assert framework.value in frameworks

    def test_framework_has_components(self):
        """Each framework should list its components."""
        frameworks = list_frameworks()

        for name, data in frameworks.items():
            assert "components" in data
            assert "count" in data
            assert data["count"] == len(data["components"])
            assert data["count"] >= 4


class TestListPredefinedSystems:
    """Tests for list_predefined_systems function."""

    def test_lists_all_systems(self):
        """Should list all predefined systems."""
        systems = list_predefined_systems()

        assert "llm_transformer" in systems
        assert "human" in systems
        assert "living_cell" in systems
        assert "thermostat" in systems
        assert "autonomous_robot" in systems

    def test_returns_descriptions(self):
        """Should return descriptions for each system."""
        systems = list_predefined_systems()

        for name, description in systems.items():
            assert isinstance(description, str)
            assert len(description) > 10


class TestAutonomyHierarchy:
    """Tests for autonomy score hierarchy across systems."""

    def test_human_most_autonomous(self):
        """Human should be the most autonomous."""
        human = assess_predefined_system("human")
        llm = assess_predefined_system("llm_transformer")
        thermostat = assess_predefined_system("thermostat")
        robot = assess_predefined_system("autonomous_robot")

        assert human.overall_autonomy_score > llm.overall_autonomy_score
        assert human.overall_autonomy_score > thermostat.overall_autonomy_score
        assert human.overall_autonomy_score > robot.overall_autonomy_score

    def test_living_cell_more_autonomous_than_thermostat(self):
        """Living cell should be more autonomous than thermostat."""
        cell = assess_predefined_system("living_cell")
        thermostat = assess_predefined_system("thermostat")

        assert cell.overall_autonomy_score > thermostat.overall_autonomy_score

    def test_robot_more_autonomous_than_llm(self):
        """Autonomous robot should be more autonomous than LLM."""
        robot = assess_predefined_system("autonomous_robot")
        llm = assess_predefined_system("llm_transformer")

        assert robot.overall_autonomy_score > llm.overall_autonomy_score


class TestSpecificFindings:
    """Tests for specific research findings about LLM autonomy."""

    def test_llm_lacks_operational_closure(self):
        """LLM should lack operational closure."""
        report = assess_predefined_system("llm_transformer")

        closure_assessment = None
        for a in report.assessments:
            if a.component.name == "operational_closure":
                closure_assessment = a
                break

        assert closure_assessment is not None
        assert closure_assessment.status == ComponentStatus.ABSENT

    def test_llm_lacks_persistent_identity(self):
        """LLM should lack persistent identity."""
        report = assess_predefined_system("llm_transformer")

        identity_assessment = None
        for a in report.assessments:
            if a.component.name == "persistent_identity":
                identity_assessment = a
                break

        assert identity_assessment is not None
        assert identity_assessment.status == ComponentStatus.ABSENT

    def test_llm_has_partial_attention(self):
        """LLM should have partial attention control."""
        report = assess_predefined_system("llm_transformer")

        attention_assessment = None
        for a in report.assessments:
            if a.component.name == "attention_control":
                attention_assessment = a
                break

        assert attention_assessment is not None
        assert attention_assessment.status == ComponentStatus.PARTIAL

    def test_cell_has_operational_closure(self):
        """Living cell should have operational closure."""
        report = assess_predefined_system("living_cell")

        closure_assessment = None
        for a in report.assessments:
            if a.component.name == "operational_closure":
                closure_assessment = a
                break

        assert closure_assessment is not None
        assert closure_assessment.status == ComponentStatus.PRESENT


class TestReportFormatting:
    """Tests for report formatting and display."""

    def test_report_str_has_sections(self):
        """Report string should have all sections."""
        report = assess_predefined_system("llm_transformer")
        s = str(report)

        assert "AUTONOMY ANALYSIS" in s
        assert "Overall Autonomy Score" in s
        assert "Framework Scores" in s
        assert "Component Assessments" in s
        assert "Critical Gaps" in s

    def test_assessment_icons(self):
        """Assessments should use status icons."""
        report = assess_predefined_system("llm_transformer")
        s = str(report)

        # Check for status icons
        assert "✓" in s or "◐" in s or "✗" in s

    def test_component_str(self):
        """Component string should include framework."""
        component = AutonomyComponent(
            name="test_component",
            framework=AutonomyFramework.CONTROL_THEORY,
            description="Test"
        )
        s = str(component)

        assert "test_component" in s
        assert "control_theory" in s


# ── Design Guidance Tests ────────────────────────────────────────────

class TestImplementationApproaches:
    """Tests for implementation approach definitions."""

    def test_all_components_have_approaches(self):
        """Most components should have implementation approaches."""
        # At least the major ones
        critical_components = [
            "feedback_loop",
            "setpoint_generation",
            "operational_closure",
            "persistent_identity",
            "embodiment",
            "metacognition"
        ]
        for comp in critical_components:
            assert comp in IMPLEMENTATION_APPROACHES
            assert len(IMPLEMENTATION_APPROACHES[comp]) >= 1

    def test_approach_structure(self):
        """Each approach should have required fields."""
        for component, approaches in IMPLEMENTATION_APPROACHES.items():
            for approach in approaches:
                assert approach.name
                assert approach.description
                assert approach.difficulty in ["low", "medium", "high", "research"]
                assert isinstance(approach.existing_examples, list)
                assert approach.integration_notes

    def test_difficulty_distribution(self):
        """Should have approaches at different difficulty levels."""
        difficulties = {"low": 0, "medium": 0, "high": 0, "research": 0}
        for approaches in IMPLEMENTATION_APPROACHES.values():
            for approach in approaches:
                difficulties[approach.difficulty] += 1

        assert difficulties["low"] >= 3  # Some quick wins
        assert difficulties["medium"] >= 5  # Medium-term goals
        assert difficulties["high"] + difficulties["research"] >= 3  # Frontier


class TestImplementationRoadmap:
    """Tests for get_implementation_roadmap function."""

    def test_roadmap_structure(self):
        """Roadmap should have required sections."""
        roadmap = get_implementation_roadmap("llm_transformer")

        assert "system" in roadmap
        assert "current_score" in roadmap
        assert "quick_wins" in roadmap
        assert "medium_term" in roadmap
        assert "research_frontier" in roadmap
        assert "implementation_details" in roadmap

    def test_roadmap_has_quick_wins(self):
        """LLM transformer should have quick wins available."""
        roadmap = get_implementation_roadmap("llm_transformer")

        assert len(roadmap["quick_wins"]) >= 2
        # Quick wins should be low difficulty
        for win in roadmap["quick_wins"]:
            assert win["difficulty"] == "low"

    def test_roadmap_has_research_items(self):
        """LLM transformer should have research frontier items."""
        roadmap = get_implementation_roadmap("llm_transformer")

        assert len(roadmap["research_frontier"]) >= 1
        # Research items should be high or research difficulty
        for item in roadmap["research_frontier"]:
            assert item["difficulty"] in ["high", "research"]

    def test_roadmap_includes_examples(self):
        """Approaches should include existing examples."""
        roadmap = get_implementation_roadmap("llm_transformer")

        # At least some approaches should have examples
        has_examples = False
        for items in [roadmap["quick_wins"], roadmap["medium_term"]]:
            for item in items:
                if item.get("examples"):
                    has_examples = True
                    break
        assert has_examples


class TestDesignAutonomousSystem:
    """Tests for design_autonomous_system function."""

    def test_basic_design(self):
        """Basic design should work."""
        design = design_autonomous_system(
            base_system="llm_transformer",
            max_difficulty="medium"
        )

        assert "base_system" in design
        assert "current_autonomy_score" in design
        assert "estimated_final_score" in design
        assert "selected_approaches" in design
        assert "integration_checklist" in design

    def test_design_improves_score(self):
        """Design should estimate score improvement."""
        design = design_autonomous_system(
            base_system="llm_transformer",
            max_difficulty="medium"
        )

        assert design["estimated_final_score"] >= design["current_autonomy_score"]

    def test_design_respects_difficulty(self):
        """Design should respect max difficulty constraint."""
        design = design_autonomous_system(
            base_system="llm_transformer",
            max_difficulty="low"
        )

        for approach in design["selected_approaches"]:
            assert approach["difficulty"] == "low"

    def test_design_with_target_capabilities(self):
        """Design should filter by target capabilities."""
        design = design_autonomous_system(
            base_system="llm_transformer",
            target_capabilities=["feedback_loop", "persistent_identity"],
            max_difficulty="high"
        )

        components = [a["component"] for a in design["selected_approaches"]]
        for comp in components:
            assert comp in ["feedback_loop", "persistent_identity"]

    def test_integration_checklist_generated(self):
        """Integration checklist should be generated."""
        design = design_autonomous_system(
            base_system="llm_transformer",
            max_difficulty="medium"
        )

        checklist = design["integration_checklist"]
        assert len(checklist) >= 3
        checklist_text = "\n".join(checklist)
        assert "VERIFICATION" in checklist_text


class TestMinimumViableAutonomy:
    """Tests for get_minimum_viable_autonomy function."""

    def test_mva_returns_design(self):
        """MVA should return a complete design."""
        mva = get_minimum_viable_autonomy()

        assert "base_system" in mva
        assert "selected_approaches" in mva
        assert "note" in mva

    def test_mva_includes_core_components(self):
        """MVA should include core autonomy components."""
        mva = get_minimum_viable_autonomy()

        components = [a["component"] for a in mva["selected_approaches"]]
        # Should include at least some core components
        core_found = 0
        for core in ["feedback_loop", "setpoint_generation", "persistent_identity"]:
            if core in components:
                core_found += 1
        assert core_found >= 2

    def test_mva_is_achievable(self):
        """MVA should be achievable with medium difficulty max."""
        mva = get_minimum_viable_autonomy()

        for approach in mva["selected_approaches"]:
            assert approach["difficulty"] in ["low", "medium"]


class TestFullAutonomyBlueprint:
    """Tests for get_full_autonomy_blueprint function."""

    def test_blueprint_returns_design(self):
        """Blueprint should return a complete design."""
        blueprint = get_full_autonomy_blueprint()

        assert "base_system" in blueprint
        assert "selected_approaches" in blueprint
        assert "note" in blueprint

    def test_blueprint_is_comprehensive(self):
        """Blueprint should cover many components."""
        blueprint = get_full_autonomy_blueprint()

        # Should have more components than MVA
        mva = get_minimum_viable_autonomy()
        assert len(blueprint["selected_approaches"]) >= len(mva["selected_approaches"])

    def test_blueprint_includes_research(self):
        """Blueprint should include research-level items."""
        blueprint = get_full_autonomy_blueprint()

        difficulties = [a["difficulty"] for a in blueprint["selected_approaches"]]
        assert "research" in difficulties or "high" in difficulties

    def test_blueprint_acknowledges_challenges(self):
        """Blueprint note should mention open problems."""
        blueprint = get_full_autonomy_blueprint()

        note = blueprint["note"].lower()
        assert "research" in note or "challenge" in note or "open" in note


class TestDesignGuidancePractical:
    """Practical tests for design guidance usability."""

    def test_human_has_no_gaps(self):
        """Human should have minimal implementation needs."""
        roadmap = get_implementation_roadmap("human")

        # Human is reference - shouldn't need many fixes
        assert len(roadmap["quick_wins"]) < 3
        assert len(roadmap["research_frontier"]) < 3

    def test_thermostat_needs_cognition(self):
        """Thermostat should need cognitive components."""
        roadmap = get_implementation_roadmap("thermostat")

        all_components = []
        for category in ["quick_wins", "medium_term", "research_frontier"]:
            for item in roadmap[category]:
                all_components.append(item["component"])

        # Thermostat should need cognitive components
        cognitive_needed = [c for c in all_components if c in [
            "metacognition", "persistent_identity", "episodic_memory", "prospection"
        ]]
        assert len(cognitive_needed) >= 2

    def test_design_for_robot_vs_llm(self):
        """Robot should need different components than LLM."""
        robot_roadmap = get_implementation_roadmap("autonomous_robot")
        llm_roadmap = get_implementation_roadmap("llm_transformer")

        robot_gaps = set(c["component"] for cat in ["quick_wins", "medium_term", "research_frontier"]
                         for c in robot_roadmap[cat])
        llm_gaps = set(c["component"] for cat in ["quick_wins", "medium_term", "research_frontier"]
                       for c in llm_roadmap[cat])

        # LLM needs embodiment, robot already has it
        assert "embodiment" in llm_gaps
        # There should be some difference
        assert robot_gaps != llm_gaps
