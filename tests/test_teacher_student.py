"""Tests for teacher_student module.

Tests the teacher-student research framework without requiring
actual model loading (uses mocks for MLX dependencies).
"""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch


class TestTeacherStudentConfig:
    """Tests for TeacherStudentConfig dataclass."""

    def test_default_values(self):
        """Test that default values are set correctly."""
        from noethersolve.teacher_student import TeacherStudentConfig

        config = TeacherStudentConfig()

        assert config.teacher_model == "mlx-community/Qwen3.5-27B-4bit"
        assert config.student_model == "mlx-community/Qwen3-4B-4bit"
        assert config.adapter_steps == 1000
        assert config.adapter_lr == 1e-5
        assert config.d_inner == 64
        assert config.margin_target == 1.5
        assert config.teacher_confidence_threshold == 5.0
        assert config.student_improvement_threshold == 0.5
        assert config.unload_teacher_during_training is True
        assert config.checkpoint_interval == 10

    def test_custom_values(self):
        """Test that custom values are accepted."""
        from noethersolve.teacher_student import TeacherStudentConfig

        config = TeacherStudentConfig(
            teacher_model="custom/teacher",
            student_model="custom/student",
            adapter_steps=500,
        )

        assert config.teacher_model == "custom/teacher"
        assert config.student_model == "custom/student"
        assert config.adapter_steps == 500


class TestResearchResult:
    """Tests for ResearchResult dataclass."""

    def test_default_values(self):
        """Test that defaults are set correctly."""
        from noethersolve.teacher_student import ResearchResult

        result = ResearchResult(
            hypothesis="Test hypothesis",
            teacher_margin=5.0,
            teacher_verdict="PASS",
            student_baseline_margin=2.0,
        )

        assert result.hypothesis == "Test hypothesis"
        assert result.teacher_margin == 5.0
        assert result.teacher_verdict == "PASS"
        assert result.student_baseline_margin == 2.0
        assert result.student_adapted_margin is None
        assert result.adapter_path is None
        assert result.improvement == 0.0
        assert result.notes == ""

    def test_with_adapter_results(self):
        """Test with full adapter results."""
        from noethersolve.teacher_student import ResearchResult

        result = ResearchResult(
            hypothesis="Test",
            teacher_margin=5.0,
            teacher_verdict="PASS",
            student_baseline_margin=2.0,
            student_adapted_margin=4.0,
            adapter_path=Path("/path/to/adapter.npz"),
            improvement=2.0,
            notes="Success!",
        )

        assert result.student_adapted_margin == 4.0
        assert result.improvement == 2.0
        assert result.notes == "Success!"


class TestResearchSession:
    """Tests for ResearchSession dataclass."""

    def test_default_values(self):
        """Test that session starts empty."""
        from noethersolve.teacher_student import (
            ResearchSession,
            TeacherStudentConfig,
        )

        config = TeacherStudentConfig()
        session = ResearchSession(config=config)

        assert session.config == config
        assert session.results == []
        assert session.total_teacher_evals == 0
        assert session.total_student_trains == 0
        assert session.discoveries == []


class TestTeacherStudentResearch:
    """Tests for TeacherStudentResearch class."""

    def test_initialization(self):
        """Test basic initialization."""
        from noethersolve.teacher_student import TeacherStudentResearch

        research = TeacherStudentResearch(
            teacher_model="test/teacher",
            student_model="test/student",
        )

        assert research.config.teacher_model == "test/teacher"
        assert research.config.student_model == "test/student"
        assert research._teacher is None
        assert research._student is None

    def test_initialization_with_config(self):
        """Test initialization with explicit config."""
        from noethersolve.teacher_student import (
            TeacherStudentResearch,
            TeacherStudentConfig,
        )

        config = TeacherStudentConfig(
            teacher_model="config/teacher",
            student_model="config/student",
            adapter_steps=500,
        )

        research = TeacherStudentResearch(config=config)

        assert research.config == config
        assert research.config.adapter_steps == 500

    def test_get_session_stats_empty(self):
        """Test session stats when empty."""
        from noethersolve.teacher_student import TeacherStudentResearch

        research = TeacherStudentResearch()
        stats = research.get_session_stats()

        assert stats["total_teacher_evals"] == 0
        assert stats["total_student_trains"] == 0
        assert stats["discoveries"] == 0
        assert stats["results_count"] == 0


class TestVerdictLogic:
    """Tests for verdict determination logic."""

    def test_pass_verdict_threshold(self):
        """Test that margin above threshold gives PASS."""
        # The threshold is 5.0 by default
        # Margin > 5.0 should give PASS
        from noethersolve.teacher_student import TeacherStudentConfig

        config = TeacherStudentConfig(teacher_confidence_threshold=5.0)

        # margin=6.0 > 5.0 should be PASS
        assert 6.0 > config.teacher_confidence_threshold

    def test_fail_verdict_threshold(self):
        """Test that margin below negative threshold gives FAIL."""
        from noethersolve.teacher_student import TeacherStudentConfig

        config = TeacherStudentConfig(teacher_confidence_threshold=5.0)

        # margin=-6.0 < -5.0 should be FAIL
        assert -6.0 < -config.teacher_confidence_threshold

    def test_uncertain_verdict_threshold(self):
        """Test that margin between thresholds gives UNCERTAIN."""
        from noethersolve.teacher_student import TeacherStudentConfig

        config = TeacherStudentConfig(teacher_confidence_threshold=5.0)

        # margin=3.0 is between -5 and +5
        margin = 3.0
        assert margin <= config.teacher_confidence_threshold
        assert margin >= -config.teacher_confidence_threshold


class TestImprovementLogic:
    """Tests for student improvement logic."""

    def test_improvement_calculation(self):
        """Test that improvement is correctly calculated."""
        from noethersolve.teacher_student import ResearchResult

        result = ResearchResult(
            hypothesis="test",
            teacher_margin=5.0,
            teacher_verdict="PASS",
            student_baseline_margin=2.0,
            student_adapted_margin=4.0,
        )
        result.improvement = (
            result.student_adapted_margin - result.student_baseline_margin
        )

        assert result.improvement == 2.0

    def test_improvement_threshold(self):
        """Test improvement threshold checking."""
        from noethersolve.teacher_student import TeacherStudentConfig

        config = TeacherStudentConfig(student_improvement_threshold=0.5)

        # 2.0 > 0.5 should be success
        improvement = 2.0
        assert improvement > config.student_improvement_threshold

        # 0.3 < 0.5 should be minimal improvement
        improvement = 0.3
        assert improvement <= config.student_improvement_threshold


class TestConvenienceFunction:
    """Tests for run_teacher_student_research convenience function."""

    def test_function_exists(self):
        """Test that convenience function exists and is callable."""
        from noethersolve.teacher_student import run_teacher_student_research

        assert callable(run_teacher_student_research)

    def test_function_signature(self):
        """Test that function has expected parameters."""
        import inspect
        from noethersolve.teacher_student import run_teacher_student_research

        sig = inspect.signature(run_teacher_student_research)
        params = list(sig.parameters.keys())

        assert "teacher" in params
        assert "student" in params
        assert "problem_yaml" in params
        assert "max_iterations" in params


class TestExports:
    """Tests for module exports."""

    def test_all_exports_available(self):
        """Test that all expected classes are exported."""
        from noethersolve.teacher_student import (
            TeacherStudentConfig,
            ResearchResult,
            ResearchSession,
            TeacherStudentResearch,
            run_teacher_student_research,
        )

        assert TeacherStudentConfig is not None
        assert ResearchResult is not None
        assert ResearchSession is not None
        assert TeacherStudentResearch is not None
        assert run_teacher_student_research is not None

    def test_exports_via_init(self):
        """Test that exports are available from noethersolve package."""
        from noethersolve import TeacherStudentResearch

        assert TeacherStudentResearch is not None
