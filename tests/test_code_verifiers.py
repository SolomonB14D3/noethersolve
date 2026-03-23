"""Tests for code verification tools."""

import pytest
from noethersolve.code_verifiers import (
    verify_complexity,
    generate_edge_cases,
    check_numerical_stability,
    verify_regex,
    check_sql_injection,
    verify_sort,
    check_physics_simulation,
    ComplexityResult,
    EdgeCases,
    StabilityResult,
    RegexTestResult,
    SQLInjectionResult,
    SortVerificationResult,
    PhysicsCodeResult,
)


class TestVerifyComplexity:
    """Tests for complexity verification."""

    def test_linear_function(self):
        """Test detection of O(n) complexity."""
        def linear_sum(arr):
            return sum(arr)

        result = verify_complexity(
            linear_sum,
            "O(n)",
            lambda n: list(range(n)),
            sizes=[100, 500, 1000],
        )
        assert result.measured == "O(n)"
        assert result.match is True

    def test_quadratic_function(self):
        """Test detection of O(n²) complexity."""
        def quadratic(arr):
            total = 0
            for i in arr:
                for j in arr:
                    total += i * j
            return total

        result = verify_complexity(
            quadratic,
            "O(n²)",
            lambda n: list(range(n)),
            sizes=[10, 20, 40, 80],
        )
        assert result.measured == "O(n²)"

    def test_wrong_claim(self):
        """Test detection of wrong complexity claim."""
        def quadratic(arr):
            return sum(i * j for i in arr for j in arr)

        result = verify_complexity(
            quadratic,
            "O(n)",  # Wrong claim
            lambda n: list(range(n)),
            sizes=[10, 20, 40],
        )
        # Should detect it's not O(n)
        assert result.measured in ["O(n²)", "O(n log n)"]


class TestGenerateEdgeCases:
    """Tests for edge case generation."""

    def test_string_edge_cases(self):
        """Test string edge cases include expected values."""
        cases = generate_edge_cases("str")
        descriptions = [desc for desc, _ in cases.cases]
        assert "empty" in descriptions
        assert "SQL injection" in descriptions
        assert "unicode" in descriptions

    def test_int_edge_cases(self):
        """Test integer edge cases."""
        cases = generate_edge_cases("int")
        values = [val for _, val in cases.cases]
        assert 0 in values
        assert -1 in values

    def test_list_edge_cases(self):
        """Test list edge cases."""
        cases = generate_edge_cases("list")
        descriptions = [desc for desc, _ in cases.cases]
        assert "empty" in descriptions
        assert "nested" in descriptions

    def test_unknown_type(self):
        """Test handling of unknown types."""
        cases = generate_edge_cases("unknown_type_xyz")
        assert len(cases.cases) == 1
        assert "unknown" in cases.cases[0][0].lower()


class TestNumericalStability:
    """Tests for numerical stability checking."""

    def test_stable_function(self):
        """Test detection of stable function."""
        def stable(arr):
            return sum(arr) / len(arr)

        result = check_numerical_stability(stable, [1.0, 2.0, 3.0, 4.0, 5.0])
        assert result.stable is True

    def test_function_crash(self):
        """Test handling of crashing function."""
        def crasher(arr):
            raise ValueError("Boom!")

        result = check_numerical_stability(crasher, [1, 2, 3])
        assert result.stable is False
        assert "failed" in result.details.lower() or "crash" in result.details.lower()


class TestVerifyRegex:
    """Tests for regex verification."""

    def test_valid_regex(self):
        """Test valid regex with passing tests."""
        result = verify_regex(
            r"^\d+$",
            [("123", True), ("abc", False), ("", False)],
        )
        assert result.valid_syntax is True
        assert len(result.failures) == 0

    def test_failing_regex(self):
        """Test regex with failing tests."""
        result = verify_regex(
            r"^\d+$",
            [("abc", True)],  # This should fail
        )
        assert len(result.failures) > 0

    def test_invalid_syntax(self):
        """Test invalid regex syntax."""
        result = verify_regex(r"[invalid")
        assert result.valid_syntax is False

    def test_email_pattern_type(self):
        """Test auto-generated email test cases."""
        result = verify_regex(r".+@.+\..+", pattern_type="email")
        assert result.valid_syntax is True


class TestSQLInjection:
    """Tests for SQL injection detection."""

    def test_vulnerable_fstring(self):
        """Test detection of f-string SQL injection."""
        code = '''
user_id = input()
query = f"SELECT * FROM users WHERE id = {user_id}"
cursor.execute(query)
'''
        result = check_sql_injection(code)
        assert result.vulnerable is True
        assert len(result.vulnerable_lines) > 0

    def test_safe_parameterized(self):
        """Test safe parameterized query is not flagged."""
        code = '''
user_id = input()
cursor.execute("SELECT * FROM users WHERE id = ?", [user_id])
'''
        result = check_sql_injection(code)
        # Should detect safe pattern
        assert result.safe_patterns > 0

    def test_no_sql(self):
        """Test non-SQL code."""
        code = '''
x = 1 + 2
print(x)
'''
        result = check_sql_injection(code)
        assert result.vulnerable is False


class TestVerifySort:
    """Tests for sort verification."""

    def test_correct_sort(self):
        """Test correct sorting function."""
        def good_sort(arr):
            return sorted(arr)

        result = verify_sort(good_sort)
        assert result.correct is True
        assert result.sorted_correctly is True
        assert result.handles_empty is True

    def test_buggy_sort(self):
        """Test buggy sorting function."""
        def bad_sort(arr):
            return arr  # Doesn't actually sort!

        result = verify_sort(bad_sort)
        assert result.sorted_correctly is False

    def test_crashing_sort(self):
        """Test sort that crashes on empty."""
        def crash_on_empty(arr):
            if len(arr) == 0:
                raise IndexError("Cannot sort empty list")
            return sorted(arr)

        result = verify_sort(crash_on_empty)
        assert result.handles_empty is False


class TestPhysicsSimulation:
    """Tests for physics simulation verification."""

    def test_energy_conserving(self):
        """Test energy-conserving simulation."""
        def euler_step(pos, vel, masses, dt):
            # Simple free motion (no forces) - should conserve energy
            return pos + vel * dt, vel

        result = check_physics_simulation(
            positions=[[0, 0, 0], [1, 0, 0]],
            velocities=[[1, 0, 0], [-1, 0, 0]],
            masses=[1.0, 1.0],
            step_func=euler_step,
            n_steps=100,
        )
        assert result.conserves_energy == True

    def test_crashing_simulation(self):
        """Test simulation that crashes."""
        def crasher(pos, vel, masses, dt):
            raise RuntimeError("Crash!")

        result = check_physics_simulation(
            positions=[[0, 0, 0]],
            velocities=[[1, 0, 0]],
            masses=[1.0],
            step_func=crasher,
            n_steps=10,
        )
        assert len(result.issues) > 0


class TestDataclasses:
    """Test dataclass string representations."""

    def test_complexity_result_str(self):
        """Test ComplexityResult string output."""
        result = ComplexityResult(
            claimed="O(n)",
            measured="O(n)",
            match=True,
            timings={100: 0.001},
            fit_quality=0.99,
            details="Test",
        )
        s = str(result)
        assert "O(n)" in s
        assert "✓" in s

    def test_edge_cases_str(self):
        """Test EdgeCases string output."""
        cases = EdgeCases("int", [("zero", 0), ("one", 1)])
        s = str(cases)
        assert "int" in s
        assert "zero" in s

    def test_sql_injection_result_str(self):
        """Test SQLInjectionResult string output."""
        result = SQLInjectionResult(
            vulnerable=True,
            vulnerable_lines=[(1, "bad code", "reason")],
            safe_patterns=0,
            recommendations=["fix it"],
        )
        s = str(result)
        assert "✗" in s
        assert "fix it" in s
