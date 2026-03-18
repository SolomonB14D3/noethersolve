"""Tests for dimension-dependent physics formulas."""

from noethersolve.dimension_physics import (
    check_dimension_dependence,
    get_formula,
    list_dimension_dependent_concepts,
    DIMENSIONAL_PHYSICS,
)


class TestCheckDimensionDependence:
    """Test the main dimension checking function."""

    def test_greens_function_is_dimension_dependent(self):
        result = check_dimension_dependence("greens_function")
        assert result.dimension_dependent is True
        assert 1 in result.formulas
        assert 2 in result.formulas
        assert 3 in result.formulas

    def test_greens_function_2d_is_logarithmic(self):
        result = check_dimension_dependence("laplacian_greens_function")
        formula_2d = result.formulas[2]
        assert "-ln(r)" in formula_2d.formula
        assert "logarithmic" in formula_2d.notes.lower()

    def test_greens_function_3d_is_1_over_r(self):
        result = check_dimension_dependence("greens_function")
        formula_3d = result.formulas[3]
        assert "1/(4πr)" in formula_3d.formula

    def test_vortex_topology_differs_2d_3d(self):
        result = check_dimension_dependence("vortex")
        assert result.dimension_dependent is True
        assert "point" in result.formulas[2].formula.lower()
        assert "line" in result.formulas[3].formula.lower()

    def test_turbulence_cascade_differs_2d_3d(self):
        result = check_dimension_dependence("turbulence")
        f2 = result.formulas[2]
        f3 = result.formulas[3]
        assert "inverse" in f2.formula.lower() or "large" in f2.formula.lower()
        assert "forward" in f3.formula.lower() or "small" in f3.formula.lower()

    def test_ns_regularity_2d_is_solved(self):
        result = check_dimension_dependence("ns")
        f2 = result.formulas[2]
        assert "proven" in f2.formula.lower() or "cannot" in f2.formula.lower()

    def test_ns_regularity_3d_is_open(self):
        result = check_dimension_dependence("navier_stokes")
        f3 = result.formulas[3]
        assert "open" in f3.formula.lower() or "millennium" in f3.formula.lower()

    def test_unknown_concept_returns_empty_formulas(self):
        result = check_dimension_dependence("definitely_not_a_physics_concept")
        assert len(result.formulas) == 0
        assert "Unknown" in result.model_common_error

    def test_partial_name_matching(self):
        # Should match "coulomb_potential" from "coulomb"
        result = check_dimension_dependence("coulomb")
        assert len(result.formulas) > 0

    def test_name_normalization(self):
        # Spaces and hyphens should be normalized
        r1 = check_dimension_dependence("greens function")
        r2 = check_dimension_dependence("greens-function")
        r3 = check_dimension_dependence("greens_function")
        assert r1.concept == r2.concept == r3.concept


class TestGetFormula:
    """Test getting specific formulas."""

    def test_get_2d_greens(self):
        formula = get_formula("greens_function", 2)
        assert formula is not None
        assert "-ln(r)" in formula.formula

    def test_get_3d_greens(self):
        formula = get_formula("greens_function", 3)
        assert formula is not None
        assert "1/(4πr)" in formula.formula

    def test_get_invalid_dimension(self):
        formula = get_formula("greens_function", 4)  # 4D not in database
        assert formula is None

    def test_get_invalid_concept(self):
        formula = get_formula("not_a_concept", 3)
        assert formula is None


class TestDimensionalFormula:
    """Test DimensionalFormula dataclass properties."""

    def test_all_formulas_have_required_fields(self):
        for concept, dims in DIMENSIONAL_PHYSICS.items():
            for dim, formula in dims.items():
                assert formula.concept, f"{concept} {dim}D missing concept"
                assert formula.dimension == dim, f"{concept} dimension mismatch"
                assert formula.formula, f"{concept} {dim}D missing formula"
                assert formula.latex, f"{concept} {dim}D missing latex"
                assert formula.notes, f"{concept} {dim}D missing notes"
                assert formula.common_error, f"{concept} {dim}D missing common_error"


class TestListConcepts:
    """Test listing available concepts."""

    def test_list_returns_multiple_concepts(self):
        concepts = list_dimension_dependent_concepts()
        assert len(concepts) >= 7  # We have at least 7 concepts

    def test_list_includes_expected_concepts(self):
        concepts = list_dimension_dependent_concepts()
        assert "laplacian_greens_function" in concepts
        assert "vortex_topology" in concepts
        assert "turbulence_energy_cascade" in concepts


class TestPhysicsCorrectness:
    """Verify the physics is actually correct."""

    def test_2d_laplacian_greens_is_logarithmic(self):
        """The 2D Laplacian Green's function satisfies ∇²G = δ with G = -ln(r)/(2π)."""
        formula = get_formula("laplacian", 2)
        # The formula should contain ln(r)
        assert "ln" in formula.formula.lower()

    def test_3d_laplacian_greens_is_inverse_r(self):
        """The 3D Laplacian Green's function is 1/(4πr)."""
        formula = get_formula("laplacian", 3)
        assert "1/" in formula.formula and "r" in formula.formula

    def test_2d_turbulence_has_inverse_cascade(self):
        """2D turbulence has an inverse energy cascade (Kraichnan 1967)."""
        formula = get_formula("turbulence", 2)
        assert "inverse" in formula.formula.lower() or "large" in formula.formula.lower()

    def test_2d_ns_is_globally_regular(self):
        """2D NS has global regularity (Ladyzhenskaya 1969)."""
        formula = get_formula("ns", 2)
        assert "proven" in formula.formula.lower() or "no" in formula.formula.lower()


class TestStringOutput:
    """Test string representations."""

    def test_result_str_includes_formulas(self):
        result = check_dimension_dependence("greens_function")
        s = str(result)
        assert "2D" in s or "2d" in s.lower()
        assert "3D" in s or "3d" in s.lower()
        assert "-ln(r)" in s
        assert "1/(4πr)" in s

    def test_result_str_includes_common_error(self):
        result = check_dimension_dependence("greens_function")
        s = str(result)
        assert "Common model error" in s or "common_error" in s.lower()
