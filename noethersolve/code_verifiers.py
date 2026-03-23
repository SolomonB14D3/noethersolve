"""
Code Verification Tools — capability-expanding tools for AI coding assistants.

These tools let AI agents verify their own code, catching errors that
models systematically miss.
"""

from dataclasses import dataclass, field
from typing import Callable, Any, Optional
import re
import time
import math


# ═══════════════════════════════════════════════════════════════════════════
# 1. VERIFY_COMPLEXITY - Check if claimed O(n) is actually O(n)
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class ComplexityResult:
    """Result of complexity verification."""
    claimed: str
    measured: str
    match: bool
    timings: dict[int, float]
    fit_quality: float  # R² of best fit
    details: str

    def __str__(self) -> str:
        status = "✓" if self.match else "✗"
        return f"{status} Claimed: {self.claimed}, Measured: {self.measured}\n{self.details}"


def _fit_complexity(sizes: list[int], times: list[float]) -> tuple[str, float]:
    """Fit timing data to common complexity classes.

    Returns (complexity_class, r_squared).
    """
    import numpy as np

    n = np.array(sizes, dtype=float)
    t = np.array(times, dtype=float)

    # Normalize times to avoid numerical issues
    t = t / t[-1] if t[-1] > 0 else t

    # Complexity models: O(1), O(log n), O(n), O(n log n), O(n²), O(n³), O(2^n)
    models = {
        "O(1)": np.ones_like(n),
        "O(log n)": np.log(n + 1),
        "O(n)": n,
        "O(n log n)": n * np.log(n + 1),
        "O(n²)": n ** 2,
        "O(n³)": n ** 3,
    }

    best_fit = None
    best_r2 = -float('inf')

    for name, model in models.items():
        # Normalize model
        model = model / model[-1] if model[-1] > 0 else model

        # Calculate R² (coefficient of determination)
        ss_res = np.sum((t - model) ** 2)
        ss_tot = np.sum((t - np.mean(t)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        if r2 > best_r2:
            best_r2 = r2
            best_fit = name

    return best_fit, best_r2


def verify_complexity(
    func: Callable,
    claimed: str,
    input_generator: Callable[[int], Any],
    sizes: list[int] = None,
    repeats: int = 3,
) -> ComplexityResult:
    """Verify if a function matches its claimed complexity.

    Args:
        func: The function to test
        claimed: Claimed complexity like "O(n)", "O(n log n)", "O(n²)"
        input_generator: Function that takes size n and returns input for func
        sizes: List of input sizes to test (default: [10, 50, 100, 500, 1000])
        repeats: Number of times to repeat each measurement

    Returns:
        ComplexityResult with analysis

    Example:
        >>> def linear_search(arr):
        ...     return sum(arr)
        >>> verify_complexity(linear_search, "O(n)", lambda n: list(range(n)))
    """
    if sizes is None:
        sizes = [10, 50, 100, 500, 1000, 2000]

    timings = {}

    for size in sizes:
        inp = input_generator(size)

        # Warm up
        func(inp)

        # Measure
        times = []
        for _ in range(repeats):
            start = time.perf_counter()
            func(inp)
            end = time.perf_counter()
            times.append(end - start)

        timings[size] = sum(times) / len(times)

    # Fit to complexity classes
    measured, r2 = _fit_complexity(list(timings.keys()), list(timings.values()))

    # Check if claimed matches measured
    claimed_normalized = claimed.upper().replace(" ", "")
    measured_normalized = measured.upper().replace(" ", "")
    match = claimed_normalized == measured_normalized

    # Build details
    details_lines = ["Timing data:"]
    for size, t in timings.items():
        details_lines.append(f"  n={size}: {t*1000:.3f}ms")
    details_lines.append(f"Best fit: {measured} (R²={r2:.3f})")

    if not match:
        details_lines.append(f"⚠️  Claimed {claimed} but measured {measured}")

    return ComplexityResult(
        claimed=claimed,
        measured=measured,
        match=match,
        timings=timings,
        fit_quality=r2,
        details="\n".join(details_lines),
    )


# ═══════════════════════════════════════════════════════════════════════════
# 2. GENERATE_EDGE_CASES - Find inputs that might break code
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class EdgeCases:
    """Collection of edge cases for a type."""
    type_name: str
    cases: list[tuple[str, Any]]  # (description, value)

    def __str__(self) -> str:
        lines = [f"Edge cases for {self.type_name}:"]
        for desc, val in self.cases:
            val_repr = repr(val) if len(repr(val)) < 50 else repr(val)[:47] + "..."
            lines.append(f"  • {desc}: {val_repr}")
        return "\n".join(lines)


# Edge case database by type
_EDGE_CASES: dict[str, list[tuple[str, Any]]] = {
    "int": [
        ("zero", 0),
        ("one", 1),
        ("negative one", -1),
        ("large positive", 2**31 - 1),
        ("large negative", -(2**31)),
        ("power of two", 1024),
        ("prime", 17),
    ],
    "float": [
        ("zero", 0.0),
        ("negative zero", -0.0),
        ("one", 1.0),
        ("small positive", 1e-10),
        ("small negative", -1e-10),
        ("large", 1e308),
        ("infinity", float('inf')),
        ("negative infinity", float('-inf')),
        ("NaN", float('nan')),
        ("pi", 3.14159265359),
    ],
    "str": [
        ("empty", ""),
        ("single char", "a"),
        ("whitespace only", "   "),
        ("newline", "\n"),
        ("tab", "\t"),
        ("unicode", "héllo wörld 🌍"),
        ("emoji", "👨‍👩‍👧‍👦"),
        ("null char", "\x00"),
        ("very long", "a" * 10000),
        ("SQL injection", "'; DROP TABLE users; --"),
        ("XSS", "<script>alert('xss')</script>"),
        ("path traversal", "../../../etc/passwd"),
        ("numbers as string", "12345"),
        ("whitespace around", "  hello  "),
    ],
    "list": [
        ("empty", []),
        ("single element", [1]),
        ("two elements", [1, 2]),
        ("duplicates", [1, 1, 1]),
        ("sorted", [1, 2, 3, 4, 5]),
        ("reverse sorted", [5, 4, 3, 2, 1]),
        ("with None", [1, None, 3]),
        ("nested", [[1, 2], [3, 4]]),
        ("large", list(range(10000))),
        ("mixed types", [1, "two", 3.0, None]),
    ],
    "dict": [
        ("empty", {}),
        ("single key", {"a": 1}),
        ("numeric keys", {1: "one", 2: "two"}),
        ("nested", {"a": {"b": {"c": 1}}}),
        ("None value", {"key": None}),
        ("empty string key", {"": "value"}),
    ],
    "bool": [
        ("true", True),
        ("false", False),
    ],
    "None": [
        ("None", None),
    ],
    "bytes": [
        ("empty", b""),
        ("single byte", b"\x00"),
        ("ascii", b"hello"),
        ("binary", bytes(range(256))),
    ],
}


def generate_edge_cases(type_name: str) -> EdgeCases:
    """Generate edge cases for a given type.

    Args:
        type_name: Type to generate edge cases for.
                   Supported: int, float, str, list, dict, bool, None, bytes

    Returns:
        EdgeCases with test inputs

    Example:
        >>> cases = generate_edge_cases("str")
        >>> for desc, val in cases.cases:
        ...     test_my_function(val)
    """
    type_lower = type_name.lower().strip()

    # Handle common aliases
    aliases = {
        "string": "str",
        "integer": "int",
        "number": "float",
        "array": "list",
        "object": "dict",
        "boolean": "bool",
        "none": "None",
        "null": "None",
    }
    type_lower = aliases.get(type_lower, type_lower)

    if type_lower not in _EDGE_CASES:
        return EdgeCases(
            type_name=type_name,
            cases=[("unknown type", f"No edge cases for type: {type_name}")],
        )

    return EdgeCases(
        type_name=type_name,
        cases=_EDGE_CASES[type_lower].copy(),
    )


def generate_all_edge_cases(type_names: list[str]) -> dict[str, EdgeCases]:
    """Generate edge cases for multiple types.

    Args:
        type_names: List of types

    Returns:
        Dictionary mapping type name to EdgeCases
    """
    return {t: generate_edge_cases(t) for t in type_names}


# ═══════════════════════════════════════════════════════════════════════════
# 3. CHECK_NUMERICAL_STABILITY - Verify numerical algorithms
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class StabilityResult:
    """Result of numerical stability analysis."""
    algorithm: str
    stable: bool
    condition_number: Optional[float]
    error_amplification: float
    details: str
    recommendations: list[str]

    def __str__(self) -> str:
        status = "✓ Stable" if self.stable else "✗ Unstable"
        lines = [f"{status}: {self.algorithm}"]
        if self.condition_number is not None:
            lines.append(f"Condition number: {self.condition_number:.2e}")
        lines.append(f"Error amplification: {self.error_amplification:.2e}x")
        lines.append(self.details)
        if self.recommendations:
            lines.append("Recommendations:")
            for rec in self.recommendations:
                lines.append(f"  • {rec}")
        return "\n".join(lines)


def check_numerical_stability(
    func: Callable[[Any], float],
    test_input: Any,
    perturbation: float = 1e-10,
) -> StabilityResult:
    """Check numerical stability of a function by perturbation analysis.

    Args:
        func: Function to analyze (should return a float)
        test_input: Test input value
        perturbation: Size of perturbation to apply

    Returns:
        StabilityResult with stability analysis

    Example:
        >>> def naive_variance(arr):
        ...     return sum(x**2 for x in arr)/len(arr) - (sum(arr)/len(arr))**2
        >>> check_numerical_stability(naive_variance, [1e8, 1e8+1, 1e8+2])
    """
    import numpy as np

    # Get baseline result
    try:
        baseline = func(test_input)
    except Exception as e:
        return StabilityResult(
            algorithm="unknown",
            stable=False,
            condition_number=None,
            error_amplification=float('inf'),
            details=f"Function failed on input: {e}",
            recommendations=["Fix function to handle input"],
        )

    # Perturb input and measure output change
    perturbations = []

    if isinstance(test_input, (list, tuple)):
        test_input = list(test_input)
        for i in range(min(len(test_input), 10)):
            if isinstance(test_input[i], (int, float)):
                perturbed = test_input.copy()
                perturbed[i] = test_input[i] * (1 + perturbation)
                try:
                    result = func(perturbed)
                    if baseline != 0:
                        rel_change = abs(result - baseline) / abs(baseline)
                        perturbations.append(rel_change / perturbation)
                except:
                    pass
    elif isinstance(test_input, (int, float)):
        perturbed = test_input * (1 + perturbation)
        try:
            result = func(perturbed)
            if baseline != 0:
                rel_change = abs(result - baseline) / abs(baseline)
                perturbations.append(rel_change / perturbation)
        except:
            pass

    if not perturbations:
        return StabilityResult(
            algorithm="unknown",
            stable=True,
            condition_number=None,
            error_amplification=1.0,
            details="Could not perform perturbation analysis",
            recommendations=[],
        )

    error_amp = max(perturbations)
    condition = error_amp

    # Determine stability
    stable = error_amp < 1e6  # Amplification < 1 million is considered stable

    recommendations = []
    if not stable:
        recommendations.append("Consider using a more stable algorithm")
        recommendations.append("Check for catastrophic cancellation")
        recommendations.append("Consider using higher precision (e.g., Decimal)")

    if error_amp > 100:
        recommendations.append("Large condition number - results sensitive to input errors")

    return StabilityResult(
        algorithm="analyzed function",
        stable=stable,
        condition_number=condition,
        error_amplification=error_amp,
        details=f"Tested {len(perturbations)} perturbations, max amplification: {error_amp:.2e}x",
        recommendations=recommendations,
    )


# ═══════════════════════════════════════════════════════════════════════════
# 4. VERIFY_REGEX - Check if regex matches what's claimed
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class RegexTestResult:
    """Result of regex verification."""
    pattern: str
    valid_syntax: bool
    test_results: list[tuple[str, bool, bool]]  # (test_string, should_match, did_match)
    failures: list[str]
    edge_case_issues: list[str]

    def __str__(self) -> str:
        if not self.valid_syntax:
            return f"✗ Invalid regex syntax: {self.pattern}"

        lines = [f"Regex: {self.pattern}"]

        if self.failures:
            lines.append(f"✗ {len(self.failures)} test failures:")
            for f in self.failures[:5]:
                lines.append(f"  • {f}")
            if len(self.failures) > 5:
                lines.append(f"  ... and {len(self.failures) - 5} more")
        else:
            lines.append("✓ All tests passed")

        if self.edge_case_issues:
            lines.append("Edge case warnings:")
            for issue in self.edge_case_issues:
                lines.append(f"  ⚠️  {issue}")

        return "\n".join(lines)


# Common regex edge cases
_REGEX_EDGE_CASES = {
    "email": [
        ("simple@example.com", True),
        ("user.name+tag@example.co.uk", True),
        ("user@subdomain.example.com", True),
        ("", False),
        ("@example.com", False),
        ("user@", False),
        ("user@.com", False),
        ("user name@example.com", False),
        ("user@example", True),  # Technically valid
    ],
    "url": [
        ("https://example.com", True),
        ("http://example.com/path", True),
        ("https://sub.example.com:8080/path?q=1", True),
        ("", False),
        ("not a url", False),
        ("ftp://files.example.com", True),
        ("//example.com", False),
    ],
    "phone": [
        ("555-1234", True),
        ("(555) 123-4567", True),
        ("+1-555-123-4567", True),
        ("", False),
        ("abc-defg", False),
        ("123", False),
    ],
    "ip": [
        ("192.168.1.1", True),
        ("0.0.0.0", True),
        ("255.255.255.255", True),
        ("", False),
        ("256.1.1.1", False),
        ("1.2.3", False),
        ("1.2.3.4.5", False),
    ],
}


def verify_regex(
    pattern: str,
    test_cases: list[tuple[str, bool]] = None,
    pattern_type: str = None,
) -> RegexTestResult:
    """Verify a regex pattern against test cases.

    Args:
        pattern: The regex pattern to test
        test_cases: List of (test_string, should_match) tuples
        pattern_type: Optional type hint for auto-generated test cases
                      ("email", "url", "phone", "ip")

    Returns:
        RegexTestResult with analysis

    Example:
        >>> verify_regex(r"\\d+", [("123", True), ("abc", False)])
    """
    # Check syntax
    try:
        compiled = re.compile(pattern)
    except re.error as e:
        return RegexTestResult(
            pattern=pattern,
            valid_syntax=False,
            test_results=[],
            failures=[str(e)],
            edge_case_issues=[],
        )

    # Use provided test cases or generate from type
    if test_cases is None:
        if pattern_type and pattern_type.lower() in _REGEX_EDGE_CASES:
            test_cases = _REGEX_EDGE_CASES[pattern_type.lower()]
        else:
            # Generate generic test cases
            test_cases = [
                ("", False),  # Empty string
                ("a", None),  # Single char (unknown)
                ("test", None),
                ("123", None),
                ("!@#$%", None),
            ]

    results = []
    failures = []

    for test_str, should_match in test_cases:
        if should_match is None:
            continue  # Skip unknown cases

        did_match = bool(compiled.fullmatch(test_str))
        results.append((test_str, should_match, did_match))

        if should_match != did_match:
            action = "match" if should_match else "not match"
            actual = "matched" if did_match else "didn't match"
            failures.append(f"'{test_str}' should {action} but {actual}")

    # Check for common issues
    edge_issues = []

    if ".*" in pattern:
        edge_issues.append("Contains .* which may match too greedily")
    if pattern.startswith("^") and not pattern.endswith("$"):
        edge_issues.append("Anchored at start but not end - may match partial strings")
    if not pattern.startswith("^") and pattern.endswith("$"):
        edge_issues.append("Anchored at end but not start - may match partial strings")
    if "\\s" not in pattern and " " not in pattern:
        if any(" " in tc[0] for tc in test_cases if tc[1]):
            edge_issues.append("Pattern doesn't handle spaces but test cases contain them")

    return RegexTestResult(
        pattern=pattern,
        valid_syntax=True,
        test_results=results,
        failures=failures,
        edge_case_issues=edge_issues,
    )


# ═══════════════════════════════════════════════════════════════════════════
# 5. CHECK_SQL_INJECTION - Find SQL injection vulnerabilities
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class SQLInjectionResult:
    """Result of SQL injection analysis."""
    vulnerable: bool
    vulnerable_lines: list[tuple[int, str, str]]  # (line_num, code, reason)
    safe_patterns: int
    recommendations: list[str]

    def __str__(self) -> str:
        if not self.vulnerable:
            return f"✓ No SQL injection vulnerabilities found ({self.safe_patterns} safe patterns)"

        lines = [f"✗ Found {len(self.vulnerable_lines)} potential SQL injection vulnerabilities:"]
        for line_num, code, reason in self.vulnerable_lines[:5]:
            lines.append(f"  Line {line_num}: {reason}")
            lines.append(f"    {code.strip()}")

        if len(self.vulnerable_lines) > 5:
            lines.append(f"  ... and {len(self.vulnerable_lines) - 5} more")

        if self.recommendations:
            lines.append("Recommendations:")
            for rec in self.recommendations:
                lines.append(f"  • {rec}")

        return "\n".join(lines)


# Patterns that indicate potential SQL injection
_SQL_INJECTION_PATTERNS = [
    (r'["\'].*%s.*["\']', "String formatting in SQL query"),
    (r'\.format\s*\(', "String .format() in SQL context"),
    (r'f["\'].*\{.*\}.*["\']', "f-string in SQL context"),
    (r'\+\s*["\']', "String concatenation in SQL context"),
    (r'execute\s*\(\s*["\'].*\+', "Concatenation in execute()"),
    (r'execute\s*\(\s*f["\']', "f-string in execute()"),
    (r'cursor\.execute\s*\(\s*[^,\)]+\s*%', "% formatting in cursor.execute()"),
]

# Safe patterns (parameterized queries)
_SAFE_SQL_PATTERNS = [
    r'execute\s*\([^,]+,\s*[\[\(]',  # execute(query, [params])
    r'execute\s*\([^,]+,\s*\{',  # execute(query, {params})
    r'\?\s*[,\)]',  # ? placeholders
    r'%\([a-zA-Z_]+\)s',  # %(name)s placeholders
]


def check_sql_injection(code: str) -> SQLInjectionResult:
    """Check code for potential SQL injection vulnerabilities.

    Args:
        code: Source code to analyze

    Returns:
        SQLInjectionResult with analysis

    Example:
        >>> code = '''
        ... query = f"SELECT * FROM users WHERE id = {user_id}"
        ... cursor.execute(query)
        ... '''
        >>> check_sql_injection(code)
    """
    lines = code.split('\n')
    vulnerabilities = []
    safe_count = 0

    # Check if this looks like SQL-related code
    sql_keywords = ['SELECT', 'INSERT', 'UPDATE', 'DELETE', 'execute', 'cursor', 'query']
    has_sql = any(kw.lower() in code.lower() for kw in sql_keywords)

    if not has_sql:
        return SQLInjectionResult(
            vulnerable=False,
            vulnerable_lines=[],
            safe_patterns=0,
            recommendations=["No SQL-related code detected"],
        )

    for i, line in enumerate(lines, 1):
        # Check for safe patterns first
        for pattern in _SAFE_SQL_PATTERNS:
            if re.search(pattern, line):
                safe_count += 1
                break

        # Check for vulnerable patterns
        for pattern, reason in _SQL_INJECTION_PATTERNS:
            if re.search(pattern, line, re.IGNORECASE):
                # Skip if it's a safe parameterized query
                is_safe = any(re.search(sp, line) for sp in _SAFE_SQL_PATTERNS)
                if not is_safe:
                    vulnerabilities.append((i, line, reason))
                    break

    recommendations = []
    if vulnerabilities:
        recommendations = [
            "Use parameterized queries: cursor.execute('SELECT * FROM t WHERE id = ?', [user_id])",
            "Never concatenate user input into SQL strings",
            "Use an ORM (SQLAlchemy, Django ORM) for automatic parameterization",
            "Validate and sanitize all user inputs",
        ]

    return SQLInjectionResult(
        vulnerable=len(vulnerabilities) > 0,
        vulnerable_lines=vulnerabilities,
        safe_patterns=safe_count,
        recommendations=recommendations,
    )


# ═══════════════════════════════════════════════════════════════════════════
# 6. VERIFY_SORT_CORRECTNESS - Prove a sorting algorithm is correct
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class SortVerificationResult:
    """Result of sort verification."""
    correct: bool
    sorted_correctly: bool
    preserves_elements: bool
    stable: Optional[bool]
    handles_empty: bool
    handles_single: bool
    handles_duplicates: bool
    failures: list[str]

    def __str__(self) -> str:
        status = "✓" if self.correct else "✗"
        lines = [f"{status} Sort Verification Results:"]
        lines.append(f"  Sorted correctly: {'✓' if self.sorted_correctly else '✗'}")
        lines.append(f"  Preserves elements: {'✓' if self.preserves_elements else '✗'}")
        lines.append(f"  Handles empty: {'✓' if self.handles_empty else '✗'}")
        lines.append(f"  Handles single: {'✓' if self.handles_single else '✗'}")
        lines.append(f"  Handles duplicates: {'✓' if self.handles_duplicates else '✗'}")
        if self.stable is not None:
            lines.append(f"  Stable sort: {'✓' if self.stable else '✗'}")

        if self.failures:
            lines.append("Failures:")
            for f in self.failures:
                lines.append(f"  • {f}")

        return "\n".join(lines)


def verify_sort(
    sort_func: Callable[[list], list],
    in_place: bool = False,
) -> SortVerificationResult:
    """Verify correctness of a sorting function.

    Args:
        sort_func: Function that sorts a list (returns sorted list or sorts in place)
        in_place: If True, the function sorts in place and returns None

    Returns:
        SortVerificationResult with analysis

    Example:
        >>> def my_sort(arr):
        ...     return sorted(arr)
        >>> verify_sort(my_sort)
    """
    import random

    failures = []

    def run_sort(arr):
        arr_copy = arr.copy()
        if in_place:
            sort_func(arr_copy)
            return arr_copy
        else:
            return sort_func(arr_copy)

    # Test 1: Empty list
    try:
        result = run_sort([])
        handles_empty = result == []
    except Exception as e:
        handles_empty = False
        failures.append(f"Crashed on empty list: {e}")

    # Test 2: Single element
    try:
        result = run_sort([42])
        handles_single = result == [42]
    except Exception as e:
        handles_single = False
        failures.append(f"Crashed on single element: {e}")

    # Test 3: Basic sorting
    test_cases = [
        [3, 1, 4, 1, 5, 9, 2, 6],
        [5, 4, 3, 2, 1],
        [1, 2, 3, 4, 5],
        list(range(100, 0, -1)),
    ]

    sorted_correctly = True
    for tc in test_cases:
        try:
            result = run_sort(tc)
            if result != sorted(tc):
                sorted_correctly = False
                failures.append(f"Failed to sort: {tc[:5]}... -> {result[:5]}...")
        except Exception as e:
            sorted_correctly = False
            failures.append(f"Crashed on {tc[:5]}...: {e}")

    # Test 4: Preserves elements
    test = [random.randint(0, 100) for _ in range(50)]
    try:
        result = run_sort(test)
        preserves = sorted(result) == sorted(test)
    except:
        preserves = False

    # Test 5: Duplicates
    try:
        result = run_sort([1, 1, 1, 2, 2, 3])
        handles_dups = result == [1, 1, 1, 2, 2, 3]
    except:
        handles_dups = False
        failures.append("Failed on duplicates")

    # Test 6: Stability (for sorts that claim to be stable)
    try:
        # Use tuples where first element is the key
        stable_test = [(1, 'a'), (2, 'b'), (1, 'c'), (2, 'd')]
        sorted_test = run_sort([x[0] for x in stable_test])
        # Can't easily test stability without access to original indices
        stable = None  # Unknown
    except:
        stable = None

    return SortVerificationResult(
        correct=sorted_correctly and preserves and handles_empty and handles_single,
        sorted_correctly=sorted_correctly,
        preserves_elements=preserves,
        stable=stable,
        handles_empty=handles_empty,
        handles_single=handles_single,
        handles_duplicates=handles_dups,
        failures=failures,
    )


# ═══════════════════════════════════════════════════════════════════════════
# 7. CHECK_PHYSICS_CODE - Verify physics simulation conserves correctly
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class PhysicsCodeResult:
    """Result of physics code verification."""
    conserves_energy: Optional[bool]
    conserves_momentum: Optional[bool]
    conserves_angular_momentum: Optional[bool]
    energy_drift: Optional[float]
    momentum_drift: Optional[float]
    issues: list[str]
    recommendations: list[str]

    def __str__(self) -> str:
        lines = ["Physics Code Verification:"]

        if self.conserves_energy is not None:
            status = "✓" if self.conserves_energy else "✗"
            drift = f" (drift: {self.energy_drift:.2e})" if self.energy_drift else ""
            lines.append(f"  Energy conservation: {status}{drift}")

        if self.conserves_momentum is not None:
            status = "✓" if self.conserves_momentum else "✗"
            drift = f" (drift: {self.momentum_drift:.2e})" if self.momentum_drift else ""
            lines.append(f"  Momentum conservation: {status}{drift}")

        if self.conserves_angular_momentum is not None:
            status = "✓" if self.conserves_angular_momentum else "✗"
            lines.append(f"  Angular momentum: {status}")

        if self.issues:
            lines.append("Issues:")
            for issue in self.issues:
                lines.append(f"  ⚠️  {issue}")

        if self.recommendations:
            lines.append("Recommendations:")
            for rec in self.recommendations:
                lines.append(f"  • {rec}")

        return "\n".join(lines)


def check_physics_simulation(
    positions: list[list[float]],
    velocities: list[list[float]],
    masses: list[float],
    step_func: Callable,
    n_steps: int = 100,
    dt: float = 0.01,
) -> PhysicsCodeResult:
    """Verify conservation laws in a physics simulation.

    Args:
        positions: Initial positions [[x1,y1,z1], [x2,y2,z2], ...]
        velocities: Initial velocities
        masses: Particle masses
        step_func: Function(positions, velocities, masses, dt) -> (new_pos, new_vel)
        n_steps: Number of steps to simulate
        dt: Time step

    Returns:
        PhysicsCodeResult with conservation analysis
    """
    import numpy as np

    pos = np.array(positions, dtype=float)
    vel = np.array(velocities, dtype=float)
    m = np.array(masses, dtype=float)

    def kinetic_energy():
        return 0.5 * np.sum(m[:, np.newaxis] * vel**2)

    def momentum():
        return np.sum(m[:, np.newaxis] * vel, axis=0)

    def angular_momentum():
        return np.sum(np.cross(pos, m[:, np.newaxis] * vel), axis=0)

    # Initial values
    E0 = kinetic_energy()
    P0 = momentum()
    L0 = angular_momentum()

    # Run simulation
    energies = [E0]
    momenta = [np.linalg.norm(P0)]

    issues = []

    try:
        for _ in range(n_steps):
            pos, vel = step_func(pos, vel, m, dt)
            energies.append(kinetic_energy())
            momenta.append(np.linalg.norm(momentum()))
    except Exception as e:
        issues.append(f"Simulation crashed: {e}")
        return PhysicsCodeResult(
            conserves_energy=None,
            conserves_momentum=None,
            conserves_angular_momentum=None,
            energy_drift=None,
            momentum_drift=None,
            issues=issues,
            recommendations=["Fix simulation crash before checking conservation"],
        )

    # Analyze conservation
    energy_drift = abs(energies[-1] - E0) / (abs(E0) + 1e-10)
    momentum_drift = abs(momenta[-1] - momenta[0]) / (abs(momenta[0]) + 1e-10)

    conserves_E = energy_drift < 1e-3
    conserves_P = momentum_drift < 1e-3

    recommendations = []

    if not conserves_E:
        issues.append(f"Energy drift: {energy_drift:.2%} over {n_steps} steps")
        recommendations.append("Consider using a symplectic integrator (Verlet, leapfrog)")
        recommendations.append("Check force calculation for errors")

    if not conserves_P:
        issues.append(f"Momentum drift: {momentum_drift:.2%}")
        recommendations.append("Ensure forces are applied in equal-opposite pairs (Newton's 3rd law)")

    return PhysicsCodeResult(
        conserves_energy=conserves_E,
        conserves_momentum=conserves_P,
        conserves_angular_momentum=None,  # Would need potential energy for full check
        energy_drift=energy_drift,
        momentum_drift=momentum_drift,
        issues=issues,
        recommendations=recommendations,
    )
