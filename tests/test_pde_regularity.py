"""Tests for noethersolve.pde_regularity — PDE regularity and Sobolev embedding checker."""

import math

import pytest

from noethersolve.pde_regularity import (
    check_sobolev_embedding,
    check_pde_regularity,
    critical_exponent,
    check_blowup,
    sobolev_conjugate,
    EmbeddingReport,
    RegularityReport,
    CriticalExponentReport,
    BlowupReport,
    EmbeddingIssue,
)


# ─── sobolev_conjugate utility ───────────────────────────────────────────────

class TestSobolevConjugate:
    def test_w12_r3(self):
        # W^{1,2}(R^3): p* = 3*2/(3-2) = 6
        assert sobolev_conjugate(1, 2.0, 3) == pytest.approx(6.0)

    def test_w11_r2(self):
        # W^{1,1}(R^2): p* = 2*1/(2-1) = 2
        assert sobolev_conjugate(1, 1.0, 2) == pytest.approx(2.0)

    def test_critical_case_raises(self):
        # kp = n: W^{1,3}(R^3), kp = 3 = n => not subcritical
        with pytest.raises(ValueError, match="critical"):
            sobolev_conjugate(1, 3.0, 3)

    def test_supercritical_raises(self):
        # kp > n: W^{2,2}(R^3), kp = 4 > 3
        with pytest.raises(ValueError):
            sobolev_conjugate(2, 2.0, 3)

    def test_invalid_k_raises(self):
        with pytest.raises(ValueError):
            sobolev_conjugate(-1, 2.0, 3)

    def test_invalid_p_raises(self):
        with pytest.raises(ValueError):
            sobolev_conjugate(1, 0.5, 3)

    def test_invalid_n_raises(self):
        with pytest.raises(ValueError):
            sobolev_conjugate(1, 2.0, 0)


# ─── check_sobolev_embedding ─────────────────────────────────────────────────

class TestCheckSobolevEmbedding:
    def test_subcritical_w12_r3(self):
        """W^{1,2}(R^3) -> L^6: classic subcritical case."""
        report = check_sobolev_embedding(k=1, p=2.0, n=3)
        assert isinstance(report, EmbeddingReport)
        assert report.case == "subcritical"
        assert report.sobolev_conjugate == pytest.approx(6.0)
        assert "L^{6}" in report.target_space
        assert report.verdict == "PASS"

    def test_critical_case_kp_equals_n(self):
        """W^{1,3}(R^3): kp = 3 = n, critical case."""
        report = check_sobolev_embedding(k=1, p=3.0, n=3)
        assert report.case == "critical"
        assert report.sobolev_conjugate is None
        assert "L^q" in report.target_space
        assert report.verdict == "PASS"

    def test_supercritical_holder(self):
        """W^{2,2}(R^3): kp=4 > n=3, supercritical => Holder embedding."""
        report = check_sobolev_embedding(k=2, p=2.0, n=3)
        assert report.case == "supercritical"
        assert "C^" in report.target_space
        assert report.verdict == "PASS"

    def test_invalid_k_negative(self):
        report = check_sobolev_embedding(k=-1, p=2.0, n=3)
        assert report.verdict == "FAIL"
        assert report.case == "invalid"

    def test_invalid_p_below_1(self):
        report = check_sobolev_embedding(k=1, p=0.5, n=3)
        assert report.verdict == "FAIL"

    def test_invalid_n_below_1(self):
        report = check_sobolev_embedding(k=1, p=2.0, n=0)
        assert report.verdict == "FAIL"

    def test_kp_value(self):
        report = check_sobolev_embedding(k=2, p=3.0, n=5)
        assert report.kp == 6.0

    def test_critical_w12_r2(self):
        """W^{1,2}(R^2): kp=2=n, critical case with Moser-Trudinger."""
        report = check_sobolev_embedding(k=1, p=2.0, n=2)
        assert report.case == "critical"
        # Should mention BMO or Moser-Trudinger in issues
        descs = " ".join(i.description for i in report.issues)
        assert "Moser-Trudinger" in descs or "BMO" in descs

    def test_passed_property(self):
        report = check_sobolev_embedding(k=1, p=2.0, n=3)
        assert report.passed is True

    def test_supercritical_integer_n_over_p(self):
        """W^{3,2}(R^2): kp=6 > n=2, n/p = 1 (integer)."""
        report = check_sobolev_embedding(k=3, p=2.0, n=2)
        assert report.case == "supercritical"
        assert report.verdict == "PASS"


# ─── check_pde_regularity ───────────────────────────────────────────────────

class TestCheckPdeRegularity:
    def test_laplace_smooth(self):
        report = check_pde_regularity("laplace", dimension=3)
        assert isinstance(report, RegularityReport)
        assert report.status == "proved"
        assert "C^inf" in report.known_regularity

    def test_ns_3d_open(self):
        report = check_pde_regularity("navier-stokes", dimension=3)
        assert report.status == "open"
        assert "OPEN" in report.known_regularity or "open" in report.known_regularity.lower()

    def test_ns_2d_proved(self):
        report = check_pde_regularity("navier-stokes", dimension=2)
        assert report.status == "proved"

    def test_ns_3d_claiming_global_smooth_fails(self):
        report = check_pde_regularity(
            "navier-stokes", dimension=3, claimed_regularity="global smooth"
        )
        assert report.verdict == "FAIL"
        high_issues = [i for i in report.issues if i.severity == "HIGH"]
        assert len(high_issues) > 0

    def test_ns_3d_claiming_blowup_fails(self):
        """Claiming blowup for NS 3D is also open."""
        report = check_pde_regularity(
            "navier-stokes", dimension=3, claimed_regularity="finite-time blow-up"
        )
        assert report.verdict == "FAIL"

    def test_heat_equation(self):
        report = check_pde_regularity("heat", dimension=3)
        assert report.status == "proved"
        assert "C^inf" in report.known_regularity

    def test_wave_depends_on_data(self):
        report = check_pde_regularity("wave", dimension=3)
        assert report.status == "depends_on_data"

    def test_unknown_equation(self):
        report = check_pde_regularity("fake-equation", dimension=3)
        assert report.verdict == "WARN"
        assert report.status == "unknown"

    def test_euler_3d_blowup(self):
        report = check_pde_regularity("euler", dimension=3)
        assert report.status == "proved_blowup"

    def test_euler_2d_proved(self):
        report = check_pde_regularity("euler", dimension=2)
        assert report.status == "proved"

    def test_nls_conditional(self):
        report = check_pde_regularity("nls", dimension=3)
        assert report.status == "conditional"

    def test_alias_ns(self):
        report = check_pde_regularity("ns", dimension=3)
        assert report.status == "open"

    def test_ns_2d_dimension_check_info(self):
        report = check_pde_regularity("navier-stokes", dimension=2)
        dim_issues = [i for i in report.issues if i.check_type == "DIMENSION_CHECK"]
        assert len(dim_issues) > 0
        assert any("Ladyzhenskaya" in i.description for i in dim_issues)

    def test_passed_property(self):
        report = check_pde_regularity("laplace", dimension=3)
        assert report.passed is True


# ─── critical_exponent ───────────────────────────────────────────────────────

class TestCriticalExponent:
    def test_nls_3d_l2_critical(self):
        report = critical_exponent("nls", dimension=3)
        assert isinstance(report, CriticalExponentReport)
        # L2 critical = 1 + 4/3 = 7/3
        assert report.exponents["L2_critical"] == pytest.approx(7.0 / 3.0)

    def test_nls_3d_h1_critical(self):
        report = critical_exponent("nls", dimension=3)
        # H1 critical = 1 + 4/(3-2) = 5
        assert report.exponents["H1_critical"] == pytest.approx(5.0)

    def test_nls_2d_h1_inf(self):
        report = critical_exponent("nls", dimension=2)
        assert report.exponents["H1_critical"] == math.inf

    def test_nls_1d_h1_inf(self):
        report = critical_exponent("nls", dimension=1)
        assert report.exponents["H1_critical"] == math.inf

    def test_fujita_exponent(self):
        report = critical_exponent("semilinear_heat", dimension=3)
        # Fujita = 1 + 2/3 = 5/3
        assert report.exponents["Fujita"] == pytest.approx(5.0 / 3.0)

    def test_fujita_alias(self):
        report = critical_exponent("fujita", dimension=3)
        assert "Fujita" in report.exponents

    def test_ns_scaling_critical(self):
        report = critical_exponent("navier-stokes", dimension=3)
        assert report.exponents["critical_Lebesgue"] == 3.0
        assert report.exponents["critical_Sobolev_regularity"] == pytest.approx(0.5)

    def test_wave_strauss(self):
        report = critical_exponent("wave", dimension=3)
        assert "Strauss" in report.exponents
        # Strauss(3): (n-1)p^2-(n+1)p-2=0 => 2p^2-4p-2=0 => p=(4+sqrt(16+16))/4
        # = (4+sqrt(32))/4 = 1 + sqrt(2) ~ 2.414
        assert report.exponents["Strauss"] == pytest.approx(1 + math.sqrt(2))

    def test_unknown_equation(self):
        report = critical_exponent("fake-pde", dimension=3)
        assert report.verdict == "WARN"
        assert len(report.exponents) == 0

    def test_claimed_exponent_matches(self):
        report = critical_exponent("nls", dimension=3, claimed_exponent=7.0 / 3.0)
        assert report.verdict == "PASS"
        info_issues = [i for i in report.issues if "matches" in i.description.lower()]
        assert len(info_issues) > 0

    def test_claimed_exponent_wrong(self):
        report = critical_exponent("nls", dimension=3, claimed_exponent=3.0)
        assert report.verdict == "FAIL"
        high_issues = [i for i in report.issues if i.severity == "HIGH"]
        assert len(high_issues) > 0

    def test_passed_property(self):
        report = critical_exponent("nls", dimension=3)
        assert report.passed is True


# ─── check_blowup ───────────────────────────────────────────────────────────

class TestCheckBlowup:
    def test_euler_3d_claiming_global_contradicted(self):
        """3D Euler has known blow-up for C^{1,alpha}."""
        report = check_blowup("euler", dimension=3, claimed_global=True)
        assert isinstance(report, BlowupReport)
        # smooth_data_open is True, so not "contradicted" but "open"
        assert report.consistency == "open"
        assert report.verdict == "WARN"

    def test_burgers_inviscid_blowup(self):
        """Inviscid Burgers has shocks."""
        report = check_blowup("burgers", dimension=1, claimed_global=False)
        assert report.consistency == "consistent"
        assert report.known_blowup is not None

    def test_burgers_claiming_global_contradicted(self):
        """Claiming global for inviscid Burgers contradicts known shocks."""
        report = check_blowup("burgers", dimension=1, claimed_global=True)
        assert report.consistency == "contradicted"
        assert report.verdict == "FAIL"

    def test_heat_global_consistent(self):
        report = check_blowup("heat", dimension=3, claimed_global=True)
        assert report.consistency == "consistent"
        assert report.verdict == "PASS"

    def test_heat_blowup_contradicted(self):
        """Claiming blowup for heat equation contradicts known global regularity."""
        report = check_blowup("heat", dimension=3, claimed_global=False)
        assert report.consistency == "contradicted"
        assert report.verdict == "FAIL"

    def test_ns_3d_global_claim_open(self):
        """NS 3D: smooth data blow-up is open."""
        report = check_blowup("navier-stokes", dimension=3, claimed_global=True)
        assert report.consistency == "open"

    def test_ns_2d_global_consistent(self):
        report = check_blowup("navier-stokes", dimension=2, claimed_global=True)
        assert report.consistency == "consistent"
        assert report.verdict == "PASS"

    def test_unknown_equation_open(self):
        report = check_blowup("fake-equation", dimension=3, claimed_global=True)
        assert report.consistency == "open"
        assert report.verdict == "WARN"

    def test_passed_property(self):
        report = check_blowup("heat", dimension=3, claimed_global=True)
        assert report.passed is True

    def test_nls_blowup_consistent(self):
        report = check_blowup("nls", dimension=3, claimed_global=False)
        assert report.consistency == "consistent"


# ─── Report formatting ──────────────────────────────────────────────────────

class TestReportFormatting:
    def test_embedding_report_str(self):
        report = check_sobolev_embedding(k=1, p=2.0, n=3)
        text = str(report)
        assert "Sobolev" in text
        assert "W^{1,2.0}" in text
        assert "PASS" in text

    def test_regularity_report_str(self):
        report = check_pde_regularity("navier-stokes", dimension=3)
        text = str(report)
        assert "PDE Regularity" in text
        assert "navier-stokes" in text

    def test_critical_exponent_report_str(self):
        report = critical_exponent("nls", dimension=3)
        text = str(report)
        assert "Critical Exponent" in text
        assert "nls" in text

    def test_blowup_report_str(self):
        report = check_blowup("euler", dimension=3, claimed_global=True)
        text = str(report)
        assert "Blow-up" in text
        assert "global regularity" in text

    def test_blowup_report_str_claiming_blowup(self):
        report = check_blowup("burgers", dimension=1, claimed_global=False)
        text = str(report)
        assert "finite-time blow-up" in text

    def test_embedding_issue_str(self):
        issue = EmbeddingIssue(
            check_type="TEST", severity="HIGH", description="test description"
        )
        text = str(issue)
        assert "HIGH" in text
        assert "TEST" in text


# ─── Edge cases ──────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_high_dimension_embedding(self):
        """W^{1,2}(R^10): kp=2 < n=10, subcritical."""
        report = check_sobolev_embedding(k=1, p=2.0, n=10)
        assert report.case == "subcritical"
        expected_pstar = 10 * 2.0 / (10 - 2)
        assert report.sobolev_conjugate == pytest.approx(expected_pstar)

    def test_k_zero_embedding(self):
        """W^{0,2}(R^3) = L^2(R^3): kp=0 < n, subcritical."""
        report = check_sobolev_embedding(k=0, p=2.0, n=3)
        assert report.case == "subcritical"
        assert report.sobolev_conjugate == pytest.approx(2.0)  # 3*2/(3-0) = 2

    def test_euler_2d_global_claim(self):
        report = check_blowup("euler", dimension=2, claimed_global=True)
        assert report.consistency == "consistent"

    def test_kdv_regularity(self):
        report = check_pde_regularity("kdv", dimension=1)
        assert report.status == "proved"

    def test_wave_1d_strauss_not_defined(self):
        report = critical_exponent("wave", dimension=1)
        assert "Strauss" not in report.exponents

    def test_regularity_claim_consistent_with_proved(self):
        """Claiming C^inf for Laplace is consistent."""
        report = check_pde_regularity(
            "laplace", dimension=3, claimed_regularity="C^inf"
        )
        assert report.verdict == "PASS"

    def test_wave_blanket_claim_warned(self):
        """Wave equation regularity depends on data; blanket claim is warned."""
        report = check_pde_regularity(
            "wave", dimension=3, claimed_regularity="global smooth"
        )
        assert report.verdict == "WARN"
