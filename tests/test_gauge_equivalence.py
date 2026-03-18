"""Tests for gauge_equivalence module — type inference / gauge theory parallel."""

from noethersolve.gauge_equivalence import (
    check_gauge_equivalence,
    explain_parallel,
    list_parallels,
    simple_unify,
    Domain,
    KNOWN_PARALLELS,
)


class TestCheckGaugeEquivalence:
    """Tests for check_gauge_equivalence function."""

    def test_u1_gauge_theory(self):
        """U(1) gauge theory should identify gauge freedom."""
        report = check_gauge_equivalence("U(1) electromagnetism")

        assert report.has_redundancy
        assert report.domain == Domain.GAUGE_THEORY
        assert len(report.redundant_dofs) > 0
        assert "U(1)" in report.redundant_dofs[0].name
        assert len(report.fixing_conditions) >= 2  # Coulomb, Lorenz, etc.
        assert "Coulomb" in str(report)

    def test_yang_mills(self):
        """Yang-Mills should identify Gribov ambiguity."""
        report = check_gauge_equivalence("Yang-Mills SU(2)")

        assert report.has_redundancy
        assert report.domain == Domain.GAUGE_THEORY
        assert "Gribov" in (report.residual_freedom or "")

    def test_maxwell(self):
        """Maxwell equations should be recognized as gauge theory."""
        report = check_gauge_equivalence("Maxwell equations")

        assert report.domain == Domain.GAUGE_THEORY
        assert report.has_redundancy

    def test_hindley_milner(self):
        """Hindley-Milner should identify type variable freedom."""
        report = check_gauge_equivalence("Hindley-Milner type inference")

        assert report.has_redundancy
        assert report.domain == Domain.TYPE_SYSTEM
        assert any("type variable" in dof.name.lower() for dof in report.redundant_dofs)
        assert "principal" in report.residual_freedom.lower()

    def test_polymorphism(self):
        """Polymorphism should be recognized as type system."""
        report = check_gauge_equivalence("parametric polymorphism")

        assert report.domain == Domain.TYPE_SYSTEM
        assert report.has_redundancy

    def test_cross_domain_analogy_present(self):
        """Should always include cross-domain analogy."""
        report = check_gauge_equivalence("U(1) gauge theory")
        assert report.cross_domain_analogy is not None

        report = check_gauge_equivalence("type inference")
        assert report.cross_domain_analogy is not None

    def test_report_str(self):
        """Report should have readable string representation."""
        report = check_gauge_equivalence("U(1) electromagnetism")
        s = str(report)

        assert "GAUGE EQUIVALENCE" in s
        assert "redundancy" in s.lower()
        assert report.input_system in s


class TestExplainParallel:
    """Tests for explain_parallel function."""

    def test_known_concept(self):
        """Known concepts should return explanations."""
        result = explain_parallel("most_general_unifier")

        assert result is not None
        assert "type_concept" in result
        assert "gauge_concept" in result
        assert "shared_structure" in result

    def test_partial_match(self):
        """Should work with partial matches."""
        result = explain_parallel("unifier")
        assert result is not None
        assert "MGU" in result.get("type_concept", "")

    def test_unknown_concept(self):
        """Unknown concept should return None."""
        result = explain_parallel("completely_unknown_concept_xyz")
        assert result is None


class TestListParallels:
    """Tests for list_parallels function."""

    def test_returns_list(self):
        """Should return list of known parallels."""
        parallels = list_parallels()

        assert isinstance(parallels, list)
        assert len(parallels) >= 5  # At least 5 known parallels

    def test_contains_key_concepts(self):
        """Should contain key concepts."""
        parallels = list_parallels()

        assert "most_general_unifier" in parallels
        assert "type_variable" in parallels
        assert "occurs_check" in parallels


class TestSimpleUnify:
    """Tests for simple_unify function."""

    def test_identical_types(self):
        """Identical types should unify with empty substitution."""
        result = simple_unify("Int", "Int")

        assert result.success
        assert len(result.substitution) == 0

    def test_variable_to_concrete(self):
        """Type variable should unify with concrete type."""
        result = simple_unify("α", "Int")

        assert result.success
        assert result.substitution == {"α": "Int"}

    def test_concrete_to_variable(self):
        """Concrete type should unify with type variable."""
        result = simple_unify("Int", "β")

        assert result.success
        assert result.substitution == {"β": "Int"}

    def test_constructor_unification(self):
        """List[α] should unify with List[Int]."""
        result = simple_unify("List[α]", "List[Int]")

        assert result.success
        assert result.substitution == {"α": "Int"}

    def test_occurs_check_failure(self):
        """α = List[α] should fail with occurs check."""
        result = simple_unify("α", "List[α]")

        assert not result.success
        assert result.occurs_check_failure
        assert "occurs" in str(result).lower()

    def test_constructor_mismatch(self):
        """List[α] should not unify with Set[α]."""
        result = simple_unify("List[α]", "Set[Int]")

        assert not result.success
        assert not result.occurs_check_failure
        assert "mismatch" in str(result).lower()

    def test_result_str(self):
        """Result should have readable string."""
        result = simple_unify("α", "Int")
        s = str(result)

        assert "MGU" in s
        assert "α" in s


class TestKnownParallels:
    """Tests for the KNOWN_PARALLELS database."""

    def test_all_parallels_have_required_fields(self):
        """All parallels should have required fields."""
        required = ["type_concept", "gauge_concept", "shared_structure"]

        for name, parallel in KNOWN_PARALLELS.items():
            for field in required:
                assert field in parallel, f"{name} missing {field}"

    def test_parallels_have_examples(self):
        """Most parallels should have examples."""
        with_examples = sum(1 for p in KNOWN_PARALLELS.values()
                          if "example_type" in p or "example_gauge" in p)

        # At least half should have examples
        assert with_examples >= len(KNOWN_PARALLELS) // 2


class TestDomainDetection:
    """Tests for automatic domain detection."""

    def test_detects_gauge_keywords(self):
        """Should detect gauge theory from keywords."""
        report = check_gauge_equivalence("QED field theory")
        assert report.domain == Domain.GAUGE_THEORY

    def test_detects_type_keywords(self):
        """Should detect type system from keywords."""
        report = check_gauge_equivalence("ML type inference")
        assert report.domain == Domain.TYPE_SYSTEM

    def test_explicit_domain_override(self):
        """Should respect explicit domain specification."""
        report = check_gauge_equivalence(
            "some ambiguous system",
            domain="type_system"
        )
        assert report.domain == Domain.TYPE_SYSTEM


class TestCrossValidation:
    """Tests validating the core insight: type/gauge parallel."""

    def test_mgu_gauge_orbit_parallel(self):
        """MGU finding and gauge orbit selection should be parallel."""
        # Type side
        type_result = simple_unify("List[α]", "List[Int]")
        assert type_result.success

        # Gauge side
        gauge_report = check_gauge_equivalence("U(1) gauge theory")

        # Both should involve:
        # 1. Redundant DOF (type variable / gauge parameter)
        # 2. Fixing condition (substitution / gauge condition)
        # 3. Canonical result (unified type / gauge-fixed A)

        assert type_result.substitution  # non-empty substitution
        assert gauge_report.fixing_conditions  # has gauge-fixing options

    def test_occurs_check_gribov_parallel(self):
        """Occurs check failure parallels Gribov ambiguity."""
        # Type side: infinite type
        type_result = simple_unify("α", "List[α]")
        assert not type_result.success
        assert type_result.occurs_check_failure

        # Gauge side: multiple solutions
        gauge_report = check_gauge_equivalence("Yang-Mills SU(2)")
        assert "Gribov" in (gauge_report.residual_freedom or "")

        # Both indicate: constraints don't uniquely determine solution

    def test_principal_type_residual_gauge(self):
        """Principal type parallels residual gauge freedom."""
        parallel = explain_parallel("principal_type")

        assert parallel is not None
        assert "principal" in parallel["type_concept"].lower()
        assert "residual" in parallel["gauge_concept"].lower()
