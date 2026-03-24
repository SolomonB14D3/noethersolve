"""Tests for the derivation verifier tool."""

import pytest

# Skip entire module if sympy is not available
pytest.importorskip("sympy", reason="sympy required for derivation verification")

from noethersolve.derivation_verifier import (
    verify_derivation,
    verify_step,
    quick_verify,
    DerivationReport,
    StepResult,
)


class TestVerifyStep:
    """Test single-step verification."""

    def test_simple_algebraic_equality(self):
        """Test basic algebraic identity."""
        result = verify_step("x**2 - 1 = (x-1)*(x+1)")
        assert result.valid is True

    def test_quadratic_expansion(self):
        """Test quadratic expansion."""
        result = verify_step("(x+1)**2 = x**2 + 2*x + 1")
        assert result.valid is True

    def test_wrong_algebraic(self):
        """Test incorrect algebraic claim."""
        result = verify_step("x**2 + 1 = (x+1)**2")
        assert result.valid is False

    def test_derivative_polynomial(self):
        """Test polynomial derivative."""
        result = verify_step("d/dx(x**3) = 3*x**2")
        assert result.valid is True

    def test_derivative_wrong(self):
        """Test incorrect derivative."""
        result = verify_step("d/dx(x**3) = 2*x**2")
        assert result.valid is False
        assert "3*x**2" in result.reason

    def test_derivative_sin(self):
        """Test trig derivative."""
        result = verify_step("d/dx(sin(x)) = cos(x)")
        assert result.valid is True

    def test_derivative_cos(self):
        """Test cos derivative."""
        result = verify_step("d/dx(cos(x)) = -sin(x)")
        assert result.valid is True

    def test_derivative_cos_wrong(self):
        """Test wrong cos derivative."""
        result = verify_step("d/dx(cos(x)) = sin(x)")
        assert result.valid is False

    def test_derivative_chain_rule(self):
        """Test chain rule."""
        result = verify_step("d/dx(sin(x**2)) = 2*x*cos(x**2)")
        assert result.valid is True

    def test_derivative_product_rule(self):
        """Test product rule."""
        result = verify_step("d/dx(x*sin(x)) = sin(x) + x*cos(x)")
        assert result.valid is True

    def test_derivative_exp(self):
        """Test exponential derivative."""
        result = verify_step("d/dx(exp(x)) = exp(x)")
        assert result.valid is True

    def test_derivative_log(self):
        """Test log derivative."""
        result = verify_step("d/dx(log(x)) = 1/x")
        assert result.valid is True

    def test_integral_polynomial(self):
        """Test polynomial integral."""
        result = verify_step("∫x**2 dx = x**3/3 + C")
        assert result.valid is True

    def test_integral_wrong(self):
        """Test incorrect integral."""
        result = verify_step("∫x**2 dx = x**3/2 + C")
        assert result.valid is False

    def test_integral_exp(self):
        """Test exponential integral."""
        result = verify_step("∫exp(x) dx = exp(x) + C")
        assert result.valid is True

    def test_integral_trig(self):
        """Test trig integral."""
        result = verify_step("∫cos(x) dx = sin(x) + C")
        assert result.valid is True


class TestVerifyDerivation:
    """Test multi-step derivation verification."""

    def test_all_correct(self):
        """Test derivation with all correct steps."""
        steps = [
            "d/dx(x**3) = 3*x**2",
            "d/dx(x**2) = 2*x",
            "∫2*x dx = x**2 + C",
        ]
        report = verify_derivation(steps)
        assert report.all_valid is True
        assert report.first_error is None
        assert len(report.steps) == 3

    def test_with_error(self):
        """Test derivation with an error."""
        steps = [
            "d/dx(sin(x)) = cos(x)",  # correct
            "d/dx(cos(x)) = sin(x)",  # WRONG
            "d/dx(tan(x)) = 1/cos(x)**2",  # correct
        ]
        report = verify_derivation(steps)
        assert report.all_valid is False
        assert report.first_error == 2  # Second step is wrong

    def test_empty_derivation(self):
        """Test empty derivation."""
        report = verify_derivation([])
        assert report.all_valid is True
        assert len(report.steps) == 0

    def test_single_step(self):
        """Test single-step derivation."""
        report = verify_derivation(["x**2 = x*x"])
        assert report.all_valid is True


class TestQuickVerify:
    """Test the quick_verify convenience function."""

    def test_true(self):
        """Test valid equation."""
        assert quick_verify("x**2 - 1 = (x-1)*(x+1)") is True

    def test_false(self):
        """Test invalid equation."""
        assert quick_verify("x**2 + 1 = (x-1)*(x+1)") is False


class TestDataclasses:
    """Test dataclass functionality."""

    def test_step_result_str(self):
        """Test StepResult string representation."""
        result = StepResult(
            step_num=1,
            original="x = x",
            valid=True,
            reason="Equal",
        )
        assert result.step_num == 1
        assert result.valid is True

    def test_derivation_report_str(self):
        """Test DerivationReport string representation."""
        steps = [
            StepResult(1, "a = a", True, "Equal"),
            StepResult(2, "b = c", False, "Not equal"),
        ]
        report = DerivationReport(
            steps=steps,
            all_valid=False,
            first_error=2,
            summary="Test summary",
        )
        s = str(report)
        assert "Test summary" in s
        assert "Step 1" in s
        assert "Step 2" in s
