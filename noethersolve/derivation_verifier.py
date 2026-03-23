"""
Derivation Verifier - Check multi-step mathematical reasoning.

Parses mathematical steps and verifies each transformation is valid.
"""

from dataclasses import dataclass
from typing import Optional
import re


@dataclass
class StepResult:
    """Result of verifying a single step."""
    step_num: int
    original: str
    valid: bool
    reason: str
    parsed_lhs: Optional[str] = None
    parsed_rhs: Optional[str] = None


@dataclass
class DerivationReport:
    """Full report on a derivation."""
    steps: list[StepResult]
    all_valid: bool
    first_error: Optional[int]
    summary: str

    def __str__(self) -> str:
        lines = [self.summary, ""]
        for step in self.steps:
            status = "✓" if step.valid else "✗"
            lines.append(f"Step {step.step_num}: {status} {step.original}")
            if not step.valid:
                lines.append(f"         Error: {step.reason}")
        return "\n".join(lines)


def _parse_equation(eq_str: str) -> tuple[str, str, str]:
    """Parse an equation string into (lhs, relation, rhs).

    Handles: =, ==, →, ⇒, :=
    """
    # Normalize different equality symbols
    eq_str = eq_str.replace("==", "=").replace("→", "=").replace("⇒", "=").replace(":=", "=")

    if "=" not in eq_str:
        return eq_str.strip(), "statement", ""

    parts = eq_str.split("=", 1)
    return parts[0].strip(), "=", parts[1].strip()


def _normalize_for_sympy(expr: str) -> str:
    """Convert common math notation to SymPy-parseable form."""
    s = expr

    # Handle d/dx notation for derivatives - need to match balanced parens
    # First try: d/dx(something with possible nested parens)
    def replace_derivative(match):
        var = match.group(1)
        rest = match.group(2)
        # Find the matching closing paren
        depth = 1
        i = 0
        while i < len(rest) and depth > 0:
            if rest[i] == '(':
                depth += 1
            elif rest[i] == ')':
                depth -= 1
            i += 1
        if depth == 0:
            inner = rest[:i-1]
            return f'diff({inner}, {var})' + rest[i:]
        return match.group(0)

    # Simple pattern for d/dx(expr) without nested parens
    s = re.sub(r'd/d([a-z])\s*\(([^()]+)\)', r'diff(\2, \1)', s)
    # Pattern for simple variable
    s = re.sub(r'd/d([a-z])\s+([a-zA-Z0-9_]+)(?!\()', r'diff(\2, \1)', s)

    # Handle integral notation: ∫f dx or ∫f(x)dx
    s = re.sub(r'∫\s*([^d]+)\s*d([a-z])', r'integrate(\1, \2)', s)
    s = re.sub(r'\\int\s*([^d]+)\s*d([a-z])', r'integrate(\1, \2)', s)

    # Handle common functions
    s = s.replace("√", "sqrt")
    s = s.replace("^", "**")
    s = s.replace("ln", "log")  # SymPy uses log for natural log

    # Handle π and e
    s = s.replace("π", "pi")

    # Handle absolute value
    s = re.sub(r'\|([^|]+)\|', r'Abs(\1)', s)

    return s


def _sympy_equal(expr1: str, expr2: str, variables: list[str] = None, allow_constant_diff: bool = False) -> tuple[bool, str]:
    """Check if two expressions are mathematically equal using SymPy.

    Args:
        expr1: First expression
        expr2: Second expression
        variables: List of variable names to use
        allow_constant_diff: If True, expressions differing by a constant are considered equal
                            (used for indefinite integrals)

    Returns (is_equal, reason).
    """
    try:
        import sympy as sp
        from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application

        transformations = standard_transformations + (implicit_multiplication_application,)

        # Define common symbols
        if variables is None:
            variables = ['x', 'y', 'z', 't', 'n', 'a', 'b', 'c', 'C']

        local_dict = {v: sp.Symbol(v) for v in variables}
        local_dict['C'] = sp.Symbol('C')  # Integration constant
        local_dict['e'] = sp.E
        local_dict['pi'] = sp.pi
        local_dict['i'] = sp.I

        # Parse expressions
        e1 = parse_expr(_normalize_for_sympy(expr1), local_dict=local_dict, transformations=transformations)
        e2 = parse_expr(_normalize_for_sympy(expr2), local_dict=local_dict, transformations=transformations)

        # Try simplification
        diff = sp.simplify(e1 - e2)
        if diff == 0:
            return True, "Expressions are equal (simplification)"

        # Try expansion
        diff_expanded = sp.expand(e1 - e2)
        if diff_expanded == 0:
            return True, "Expressions are equal (expansion)"

        # Try trigsimp for trig identities
        diff_trig = sp.trigsimp(e1 - e2)
        if diff_trig == 0:
            return True, "Expressions are equal (trig simplification)"

        # Check if difference is just a constant (only for integrals)
        if allow_constant_diff:
            free_symbols = diff.free_symbols
            if len(free_symbols) == 0 or (len(free_symbols) == 1 and sp.Symbol('C') in free_symbols):
                # Difference is a constant - valid for indefinite integrals
                return True, "Expressions differ by a constant (valid for integrals)"

        return False, f"Expressions differ: {expr1} ≠ {expr2}, difference = {diff}"

    except Exception as e:
        return False, f"Parse error: {str(e)}"


def _verify_derivative(lhs: str, rhs: str) -> tuple[bool, str]:
    """Verify a derivative claim like d/dx(x²) = 2x."""
    try:
        import sympy as sp
        from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application

        transformations = standard_transformations + (implicit_multiplication_application,)

        # Parse d/dx(...) notation directly
        # Match d/d followed by variable, then extract content in parens
        match = re.match(r'd/d([a-z])\s*\((.+)\)\s*$', lhs.strip())
        if not match:
            # Try already-normalized form
            lhs_norm = _normalize_for_sympy(lhs)
            match = re.match(r'diff\((.+),\s*([a-z])\)\s*$', lhs_norm)
            if match:
                expr_str, var_str = match.groups()
            else:
                return False, "Could not parse derivative notation"
        else:
            var_str = match.group(1)
            expr_str = match.group(2)

        local_dict = {v: sp.Symbol(v) for v in ['x', 'y', 'z', 't', 'n', 'a', 'b', 'c']}
        local_dict['e'] = sp.E
        local_dict['pi'] = sp.pi

        expr = parse_expr(expr_str, local_dict=local_dict, transformations=transformations)
        var = sp.Symbol(var_str)

        # Compute derivative
        computed = sp.diff(expr, var)

        # Parse claimed result
        rhs_parsed = parse_expr(_normalize_for_sympy(rhs), local_dict=local_dict, transformations=transformations)

        # Compare
        if sp.simplify(computed - rhs_parsed) == 0:
            return True, f"Derivative verified: d/d{var_str}({expr}) = {computed}"
        else:
            return False, f"Derivative incorrect: d/d{var_str}({expr}) = {computed}, not {rhs}"

    except Exception as e:
        return False, f"Error verifying derivative: {str(e)}"


def _verify_integral(lhs: str, rhs: str) -> tuple[bool, str]:
    """Verify an integral claim like ∫x² dx = x³/3 + C."""
    try:
        import sympy as sp
        from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application

        transformations = standard_transformations + (implicit_multiplication_application,)

        lhs_norm = _normalize_for_sympy(lhs)

        # Check if it's an integrate() expression
        match = re.match(r'integrate\(([^,]+),\s*([a-z])\)', lhs_norm)
        if not match:
            return False, "Could not parse integral notation"

        expr_str, var_str = match.groups()

        local_dict = {v: sp.Symbol(v) for v in ['x', 'y', 'z', 't', 'n', 'a', 'b', 'c', 'C']}
        local_dict['e'] = sp.E
        local_dict['pi'] = sp.pi

        expr = parse_expr(expr_str.strip(), local_dict=local_dict, transformations=transformations)
        var = sp.Symbol(var_str)

        # For indefinite integrals, verify by differentiating the result
        rhs_norm = _normalize_for_sympy(rhs)
        # Remove + C for differentiation check
        rhs_no_const = re.sub(r'\+\s*C\s*$', '', rhs_norm).strip()

        rhs_parsed = parse_expr(rhs_no_const, local_dict=local_dict, transformations=transformations)

        # Differentiate RHS and compare to integrand
        derivative = sp.diff(rhs_parsed, var)

        if sp.simplify(derivative - expr) == 0:
            return True, f"Integral verified: ∫{expr} d{var_str} = {rhs_parsed} + C"
        else:
            return False, f"Integral incorrect: d/d{var_str}({rhs_parsed}) = {derivative}, not {expr}"

    except Exception as e:
        return False, f"Error verifying integral: {str(e)}"


def _detect_step_type(lhs: str, rhs: str) -> str:
    """Detect what type of mathematical step this is."""
    lhs_lower = lhs.lower()

    if 'd/d' in lhs or 'diff(' in lhs.lower():
        return "derivative"
    if '∫' in lhs or '\\int' in lhs or 'integrate(' in lhs.lower():
        return "integral"
    if 'lim' in lhs_lower:
        return "limit"
    if 'sum' in lhs_lower or 'Σ' in lhs:
        return "sum"

    return "equality"


def verify_step(step: str, previous_rhs: Optional[str] = None) -> StepResult:
    """Verify a single mathematical step.

    Args:
        step: The mathematical statement (e.g., "x² + 2x + 1 = (x+1)²")
        previous_rhs: The RHS of the previous step, for chain verification

    Returns:
        StepResult with validation details
    """
    lhs, relation, rhs = _parse_equation(step)

    if relation == "statement":
        return StepResult(
            step_num=0,
            original=step,
            valid=True,
            reason="Statement (no verification needed)",
            parsed_lhs=lhs
        )

    step_type = _detect_step_type(lhs, rhs)

    if step_type == "derivative":
        valid, reason = _verify_derivative(lhs, rhs)
    elif step_type == "integral":
        valid, reason = _verify_integral(lhs, rhs)
    else:
        # General equality check
        valid, reason = _sympy_equal(lhs, rhs)

    return StepResult(
        step_num=0,
        original=step,
        valid=valid,
        reason=reason,
        parsed_lhs=lhs,
        parsed_rhs=rhs
    )


def verify_derivation(steps: list[str]) -> DerivationReport:
    """Verify a complete mathematical derivation.

    Args:
        steps: List of mathematical steps/equations

    Returns:
        DerivationReport with full analysis
    """
    results = []
    first_error = None

    for i, step in enumerate(steps):
        result = verify_step(step)
        result.step_num = i + 1
        results.append(result)

        if not result.valid and first_error is None:
            first_error = i + 1

    all_valid = all(r.valid for r in results)

    if all_valid:
        summary = f"✓ All {len(steps)} steps verified correct"
    else:
        n_errors = sum(1 for r in results if not r.valid)
        summary = f"✗ Found {n_errors} error(s) in {len(steps)} steps (first error: step {first_error})"

    return DerivationReport(
        steps=results,
        all_valid=all_valid,
        first_error=first_error,
        summary=summary
    )


# Convenience function for quick checks
def quick_verify(equation: str) -> bool:
    """Quick check if an equation is valid.

    Args:
        equation: A mathematical equation like "x² - 1 = (x-1)(x+1)"

    Returns:
        True if the equation is mathematically valid
    """
    result = verify_step(equation)
    return result.valid
