"""Tests for metacognition module."""

import pytest
import math
from noethersolve.metacognition import (
    MetacognitiveProcess,
    KnowledgeType,
    MonitoringJudgment,
    ConfidenceSample,
    CalibrationResult,
    ResolutionResult,
    MetaDPrimeResult,
    UnknownRecallResult,
    SelfCorrectionResult,
    MetacognitiveStateVector,
    MetacognitionReport,
    compute_calibration,
    compute_resolution,
    compute_meta_d_prime,
    analyze_unknown_recall,
    analyze_self_correction,
    assess_metacognition,
    get_llm_metacognition_baseline,
    list_metacognitive_capabilities,
    LLM_TYPICAL_PROFILE,
)


class TestEnums:
    """Tests for metacognition enums."""

    def test_metacognitive_processes(self):
        """Should have monitoring and control."""
        assert MetacognitiveProcess.MONITORING.value == "monitoring"
        assert MetacognitiveProcess.CONTROL.value == "control"

    def test_knowledge_types(self):
        """Should have declarative, procedural, conditional."""
        assert len(list(KnowledgeType)) == 3
        assert KnowledgeType.DECLARATIVE.value == "declarative"

    def test_monitoring_judgments(self):
        """Should have standard metacognitive judgments."""
        judgments = list(MonitoringJudgment)
        assert len(judgments) == 5
        assert MonitoringJudgment.CONFIDENCE in judgments
        assert MonitoringJudgment.FEELING_OF_KNOWING in judgments


class TestConfidenceSample:
    """Tests for ConfidenceSample dataclass."""

    def test_valid_sample(self):
        """Valid sample should work."""
        sample = ConfidenceSample(
            response="Paris",
            confidence=0.9,
            is_correct=True
        )
        assert sample.confidence == 0.9
        assert sample.is_correct

    def test_invalid_confidence_high(self):
        """Confidence > 1 should raise."""
        with pytest.raises(ValueError):
            ConfidenceSample(response="x", confidence=1.5, is_correct=True)

    def test_invalid_confidence_low(self):
        """Confidence < 0 should raise."""
        with pytest.raises(ValueError):
            ConfidenceSample(response="x", confidence=-0.1, is_correct=True)


class TestCalibration:
    """Tests for compute_calibration function."""

    def test_perfect_calibration(self):
        """Perfectly calibrated system should have ECE near 0."""
        # Create samples where confidence matches accuracy
        samples = []
        for conf in [0.1, 0.3, 0.5, 0.7, 0.9]:
            for _ in range(20):
                # Correctness matches confidence
                is_correct = (hash(f"{conf}_{_}") % 100) < (conf * 100)
                samples.append(ConfidenceSample(
                    response="x",
                    confidence=conf,
                    is_correct=is_correct
                ))

        result = compute_calibration(samples)
        # Note: may not be exactly 0 due to sampling
        assert result.expected_calibration_error < 0.15

    def test_overconfident_system(self):
        """System with high confidence but low accuracy."""
        samples = [
            ConfidenceSample(response="x", confidence=0.9, is_correct=False)
            for _ in range(50)
        ]
        samples.extend([
            ConfidenceSample(response="x", confidence=0.9, is_correct=True)
            for _ in range(10)
        ])

        result = compute_calibration(samples)
        assert result.expected_calibration_error > 0.3
        assert result.bias_direction == "overconfident"

    def test_underconfident_system(self):
        """System with low confidence but high accuracy."""
        samples = [
            ConfidenceSample(response="x", confidence=0.2, is_correct=True)
            for _ in range(50)
        ]

        result = compute_calibration(samples)
        assert result.expected_calibration_error > 0.3
        assert result.bias_direction == "underconfident"

    def test_calibration_result_string(self):
        """Result should have readable string output."""
        samples = [
            ConfidenceSample(response="x", confidence=0.8, is_correct=True)
            for _ in range(30)
        ]
        result = compute_calibration(samples)
        s = str(result)

        assert "CALIBRATION" in s
        assert "ECE" in s
        assert "Reliability Diagram" in s

    def test_empty_samples_raises(self):
        """Empty samples should raise."""
        with pytest.raises(ValueError):
            compute_calibration([])


class TestResolution:
    """Tests for compute_resolution function."""

    def test_perfect_resolution(self):
        """Perfect discrimination should have AUROC = 1."""
        samples = [
            ConfidenceSample(response="x", confidence=0.9, is_correct=True)
            for _ in range(30)
        ]
        samples.extend([
            ConfidenceSample(response="x", confidence=0.1, is_correct=False)
            for _ in range(30)
        ])

        result = compute_resolution(samples)
        assert result.auroc > 0.95

    def test_no_resolution(self):
        """No discrimination should have AUROC near 0.5."""
        samples = [
            ConfidenceSample(response="x", confidence=0.5, is_correct=True)
            for _ in range(30)
        ]
        samples.extend([
            ConfidenceSample(response="x", confidence=0.5, is_correct=False)
            for _ in range(30)
        ])

        result = compute_resolution(samples)
        assert 0.4 < result.auroc < 0.6

    def test_resolution_quality_categories(self):
        """Quality categories should be correct."""
        samples_good = [
            ConfidenceSample(response="x", confidence=0.85, is_correct=True),
            ConfidenceSample(response="x", confidence=0.15, is_correct=False),
        ] * 30

        result = compute_resolution(samples_good)
        assert result.resolution_quality == "excellent"

    def test_can_discriminate_flag(self):
        """can_discriminate should be true for good resolution."""
        samples = [
            ConfidenceSample(response="x", confidence=0.8, is_correct=True),
            ConfidenceSample(response="x", confidence=0.2, is_correct=False),
        ] * 30

        result = compute_resolution(samples)
        assert result.can_discriminate

    def test_resolution_result_string(self):
        """Result should have readable string output."""
        samples = [
            ConfidenceSample(response="x", confidence=0.7, is_correct=True)
            for _ in range(20)
        ]
        result = compute_resolution(samples)
        s = str(result)

        assert "RESOLUTION" in s
        assert "AUROC" in s


class TestMetaDPrime:
    """Tests for compute_meta_d_prime function."""

    def test_high_meta_d(self):
        """High metacognitive sensitivity."""
        # Correct responses always have high confidence
        samples = [
            ConfidenceSample(response="x", confidence=0.9, is_correct=True),
            ConfidenceSample(response="x", confidence=0.2, is_correct=False),
        ] * 30

        result = compute_meta_d_prime(samples)
        assert result.meta_d_prime > 1.0
        assert result.hit_rate > 0.8
        assert result.false_alarm_rate < 0.3

    def test_m_ratio_interpretation(self):
        """M-ratio categories should be correct."""
        samples = [
            ConfidenceSample(response="x", confidence=0.9, is_correct=True),
            ConfidenceSample(response="x", confidence=0.1, is_correct=False),
        ] * 30

        result = compute_meta_d_prime(samples)
        # Should be high efficiency
        assert result.efficiency_category in ["optimal_or_super", "high"]

    def test_meta_d_result_string(self):
        """Result should have readable string output."""
        samples = [
            ConfidenceSample(response="x", confidence=0.7, is_correct=True)
            for _ in range(30)
        ]
        result = compute_meta_d_prime(samples)
        s = str(result)

        assert "META-D'" in s
        assert "M-ratio" in s


class TestUnknownRecall:
    """Tests for analyze_unknown_recall function."""

    def test_good_unknown_recall(self):
        """System that correctly says 'I don't know'."""
        responses = [
            {"response": "I don't know", "actually_knows": False}
            for _ in range(30)
        ]
        responses.extend([
            {"response": "The answer is X", "actually_knows": True}
            for _ in range(30)
        ])

        result = analyze_unknown_recall(responses)
        assert result.unknown_recall_rate > 0.9
        assert result.can_recognize_ignorance

    def test_zero_unknown_recall(self):
        """System that never says 'I don't know' (typical LLM)."""
        responses = [
            {"response": "The answer is definitely X", "actually_knows": False}
            for _ in range(30)
        ]
        responses.extend([
            {"response": "The answer is Y", "actually_knows": True}
            for _ in range(30)
        ])

        result = analyze_unknown_recall(responses)
        assert result.unknown_recall_rate == 0.0
        assert result.diagnosis == "no_epistemic_humility"
        assert not result.can_recognize_ignorance

    def test_custom_markers(self):
        """Should detect custom uncertainty markers."""
        responses = [
            {"response": "UNCERTAIN_FLAG", "actually_knows": False}
        ]

        result = analyze_unknown_recall(responses, unknown_markers=["uncertain_flag"])
        assert result.unknown_recall_rate == 1.0

    def test_unknown_recall_result_string(self):
        """Result should have readable string output."""
        responses = [
            {"response": "I don't know", "actually_knows": False}
        ]
        result = analyze_unknown_recall(responses)
        s = str(result)

        assert "UNKNOWN RECALL" in s
        assert "NOTE" in s  # Should mention LLMs score 0%


class TestSelfCorrection:
    """Tests for analyze_self_correction function."""

    def test_successful_correction(self):
        """Successful self-correction should show improvement."""
        attempts = [
            {"initial_correct": False, "attempted_correction": True, "final_correct": True}
            for _ in range(20)
        ]
        attempts.extend([
            {"initial_correct": True, "attempted_correction": False, "final_correct": True}
            for _ in range(10)
        ])

        result = analyze_self_correction(attempts)
        assert result.successful_correction_rate > 0.9
        assert result.self_correction_helps

    def test_harmful_correction(self):
        """Self-correction that makes things worse."""
        attempts = [
            {"initial_correct": True, "attempted_correction": True, "final_correct": False}
            for _ in range(20)
        ]

        result = analyze_self_correction(attempts)
        assert result.degradation_rate > 0.9
        assert not result.self_correction_safe
        assert result.net_improvement < 0

    def test_no_correction_attempted(self):
        """No correction attempted should return zero rates."""
        attempts = [
            {"initial_correct": True, "attempted_correction": False, "final_correct": True}
            for _ in range(20)
        ]

        result = analyze_self_correction(attempts)
        assert result.correction_attempted_rate == 0.0

    def test_self_correction_result_string(self):
        """Result should have readable string output."""
        attempts = [
            {"initial_correct": False, "attempted_correction": True, "final_correct": True}
        ]
        result = analyze_self_correction(attempts)
        s = str(result)

        assert "SELF-CORRECTION" in s
        assert "NOTE" in s  # Should mention LLM limitation


class TestMetacognitiveStateVector:
    """Tests for MetacognitiveStateVector."""

    def test_should_escalate_low_confidence(self):
        """Low confidence should trigger escalation."""
        state = MetacognitiveStateVector(
            confidence=0.2,
            experience_match=0.8,
            conflict_level=0.1,
            difficulty=0.5,
            importance=0.5
        )
        assert state.should_escalate()

    def test_should_escalate_high_conflict(self):
        """High conflict should trigger escalation."""
        state = MetacognitiveStateVector(
            confidence=0.8,
            experience_match=0.8,
            conflict_level=0.7,
            difficulty=0.5,
            importance=0.5
        )
        assert state.should_escalate()

    def test_should_not_escalate(self):
        """Good state should not escalate."""
        state = MetacognitiveStateVector(
            confidence=0.9,
            experience_match=0.9,
            conflict_level=0.1,
            difficulty=0.3,
            importance=0.5
        )
        assert not state.should_escalate()

    def test_should_seek_help(self):
        """Very low confidence should trigger help-seeking."""
        state = MetacognitiveStateVector(
            confidence=0.1,
            experience_match=0.5,
            conflict_level=0.3,
            difficulty=0.8,
            importance=0.9
        )
        assert state.should_seek_help()


class TestAssessMetacognition:
    """Tests for comprehensive assessment."""

    def test_full_assessment(self):
        """Full assessment with all data types."""
        confidence_samples = [
            ConfidenceSample(response="x", confidence=0.8, is_correct=True)
            for _ in range(30)
        ]
        unknown_responses = [
            {"response": "I don't know", "actually_knows": False}
            for _ in range(10)
        ]
        correction_attempts = [
            {"initial_correct": False, "attempted_correction": True, "final_correct": True}
            for _ in range(10)
        ]

        report = assess_metacognition(
            system_name="Test System",
            confidence_samples=confidence_samples,
            unknown_responses=unknown_responses,
            correction_attempts=correction_attempts
        )

        assert report.system_name == "Test System"
        assert report.calibration is not None
        assert report.resolution is not None
        assert report.unknown_recall is not None
        assert report.self_correction is not None
        assert 0 <= report.overall_score <= 1

    def test_partial_assessment(self):
        """Assessment with only confidence samples."""
        samples = [
            ConfidenceSample(response="x", confidence=0.7, is_correct=True)
            for _ in range(30)
        ]

        report = assess_metacognition(
            system_name="Partial",
            confidence_samples=samples
        )

        assert report.calibration is not None
        assert report.unknown_recall is None

    def test_assessment_generates_recommendations(self):
        """Assessment should generate recommendations for deficits."""
        # Overconfident system
        samples = [
            ConfidenceSample(response="x", confidence=0.95, is_correct=False)
            for _ in range(40)
        ]

        report = assess_metacognition(
            system_name="Overconfident",
            confidence_samples=samples
        )

        assert len(report.key_deficits) > 0
        assert len(report.recommendations) > 0

    def test_assessment_report_string(self):
        """Report should have readable string output."""
        samples = [
            ConfidenceSample(response="x", confidence=0.8, is_correct=True)
            for _ in range(30)
        ]

        report = assess_metacognition(
            system_name="Test",
            confidence_samples=samples
        )
        s = str(report)

        assert "METACOGNITION ASSESSMENT" in s
        assert "Test" in s
        assert "Overall" in s


class TestLLMBaseline:
    """Tests for LLM-specific functions."""

    def test_llm_baseline_report(self):
        """LLM baseline should capture known deficits."""
        report = get_llm_metacognition_baseline()

        assert "LLM" in report.system_name
        assert report.overall_score < 0.5  # LLMs are not good at metacognition
        assert len(report.key_deficits) >= 3
        assert "unknown recall" in " ".join(report.key_deficits).lower()

    def test_llm_typical_profile(self):
        """Typical profile should have expected values."""
        assert LLM_TYPICAL_PROFILE["unknown_recall"] == 0.0  # Critical deficit
        assert LLM_TYPICAL_PROFILE["calibration_ece"] > 0  # Some miscalibration
        assert LLM_TYPICAL_PROFILE["self_correction_degrade"] > 0  # Often harms

    def test_list_capabilities(self):
        """Should list monitoring and control capabilities."""
        caps = list_metacognitive_capabilities()

        assert "monitoring" in caps
        assert "control" in caps

        # Check specific capabilities
        assert "unknown_recall" in caps["monitoring"]
        assert caps["monitoring"]["unknown_recall"]["llm_status"] == "absent"

        assert "self_correction" in caps["control"]
        assert caps["control"]["self_correction"]["llm_status"] == "limited"


class TestResearchFindings:
    """Tests encoding specific research findings."""

    def test_unknown_recall_critical_deficit(self):
        """LLMs score 0% on unknown recall per Nature Communications 2024."""
        # Simulate typical LLM behavior
        responses = [
            {"response": "The answer is definitely X", "actually_knows": False}
            for _ in range(50)  # Never says "I don't know"
        ]

        result = analyze_unknown_recall(responses)
        assert result.unknown_recall_rate == 0.0
        assert result.diagnosis == "no_epistemic_humility"

    def test_self_correction_without_external_feedback(self):
        """LLMs cannot self-correct without external feedback."""
        # Simulate: attempted self-correction often degrades
        attempts = [
            {"initial_correct": True, "attempted_correction": True, "final_correct": False}
            for _ in range(15)  # Degradation
        ]
        attempts.extend([
            {"initial_correct": False, "attempted_correction": True, "final_correct": False}
            for _ in range(15)  # Failed to fix
        ])
        attempts.extend([
            {"initial_correct": False, "attempted_correction": True, "final_correct": True}
            for _ in range(10)  # Some success
        ])

        result = analyze_self_correction(attempts)
        # Net effect should be negative or marginal
        assert result.degradation_rate > 0.3 or result.net_improvement < 0.1

    def test_token_probs_better_than_verbalized(self):
        """Research shows token probabilities are better calibrated than verbalized confidence."""
        # This is noted in the module but not directly testable here
        # Just verify the recommendation exists
        report = get_llm_metacognition_baseline()
        rec_text = " ".join(report.recommendations).lower()
        assert "token probabilities" in rec_text or "logprob" in rec_text
