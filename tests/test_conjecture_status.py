"""Tests for noethersolve.conjecture_status — Mathematical conjecture status checker."""


from noethersolve.conjecture_status import (
    check_conjecture,
    check_claim,
    list_conjectures,
    get_conjecture,
    ConjectureIssue,
)


# ─── check_conjecture — correct status claims ────────────────────────────────

class TestCheckConjectureCorrect:
    def test_rh_open_passes(self):
        report = check_conjecture("Riemann Hypothesis", claimed_status="OPEN")
        assert report.passed
        assert report.verdict == "PASS"

    def test_poincare_solved_passes(self):
        report = check_conjecture("Poincare conjecture", claimed_status="SOLVED")
        assert report.passed

    def test_flt_solved_passes(self):
        report = check_conjecture("Fermat's Last Theorem", claimed_status="SOLVED")
        assert report.passed

    def test_continuum_hypothesis_independent(self):
        report = check_conjecture("Continuum hypothesis", claimed_status="INDEPENDENT")
        assert report.passed

    def test_p_vs_np_open_passes(self):
        report = check_conjecture("P vs NP", claimed_status="OPEN")
        assert report.passed

    def test_abc_disputed(self):
        report = check_conjecture("ABC conjecture", claimed_status="DISPUTED")
        assert report.verdict in ("PASS", "WARN")  # WARN because disputed gets controversy note

    def test_kervaire_mostly_solved(self):
        report = check_conjecture("Kervaire invariant one problem", claimed_status="MOSTLY_SOLVED")
        assert report.verdict in ("PASS", "WARN")

    def test_p_equals_bpp_conjectured_true(self):
        report = check_conjecture("P = BPP?", claimed_status="CONJECTURED_TRUE")
        assert report.verdict in ("PASS", "WARN")

    def test_no_claimed_status_info_only(self):
        """Looking up without claiming a status should pass."""
        report = check_conjecture("Riemann Hypothesis")
        assert report.passed
        assert report.actual_status == "OPEN"


# ─── check_conjecture — wrong status claims ──────────────────────────────────

class TestCheckConjectureWrong:
    def test_rh_claimed_solved_fails(self):
        report = check_conjecture("Riemann Hypothesis", claimed_status="SOLVED")
        assert report.verdict == "FAIL"
        assert any(i.severity == "HIGH" for i in report.issues)

    def test_poincare_claimed_open_fails(self):
        report = check_conjecture("Poincare conjecture", claimed_status="OPEN")
        assert report.verdict == "FAIL"

    def test_goldbach_claimed_solved_fails(self):
        report = check_conjecture("Goldbach conjecture", claimed_status="SOLVED")
        assert report.verdict == "FAIL"

    def test_continuum_claimed_solved_fails(self):
        """CH is INDEPENDENT, not SOLVED."""
        report = check_conjecture("Continuum hypothesis", claimed_status="SOLVED")
        assert report.verdict == "FAIL"

    def test_abc_claimed_solved_warns(self):
        """ABC is DISPUTED, claiming SOLVED is a MODERATE issue."""
        report = check_conjecture("ABC conjecture", claimed_status="SOLVED")
        assert report.verdict == "WARN"
        assert any(i.severity == "MODERATE" for i in report.issues)

    def test_kervaire_claimed_fully_solved_warns(self):
        """Kervaire is MOSTLY_SOLVED; claiming SOLVED is moderate."""
        report = check_conjecture("Kervaire invariant one problem", claimed_status="SOLVED")
        assert report.verdict == "WARN"

    def test_p_bpp_claimed_solved_fails(self):
        """P = BPP is CONJECTURED_TRUE, not SOLVED."""
        report = check_conjecture("P = BPP?", claimed_status="SOLVED")
        assert report.verdict == "FAIL"

    def test_unrecognized_status_warns(self):
        report = check_conjecture("Riemann Hypothesis", claimed_status="BANANA")
        assert report.verdict == "WARN"
        assert any("Unrecognized" in i.description for i in report.issues)


# ─── check_conjecture — unknown conjecture ───────────────────────────────────

class TestCheckConjectureUnknown:
    def test_nonexistent_conjecture_fails(self):
        report = check_conjecture("Spaghetti theorem")
        assert report.verdict == "FAIL"
        assert report.conjecture is None

    def test_nonexistent_with_claimed_status(self):
        report = check_conjecture("Fizzbuzz conjecture", claimed_status="OPEN")
        assert report.verdict == "FAIL"


# ─── check_claim — natural language ──────────────────────────────────────────

class TestCheckClaim:
    def test_goldbach_proved_fails(self):
        """Strong Goldbach is OPEN."""
        report = check_claim("Goldbach conjecture was proved")
        assert report.verdict == "FAIL"

    def test_rh_proved_fails(self):
        report = check_claim("the Riemann Hypothesis was proved in 2018")
        assert report.verdict == "FAIL"

    def test_poincare_solved_passes(self):
        report = check_claim("the Poincare conjecture was solved by Perelman")
        assert report.passed

    def test_flt_proved_passes(self):
        report = check_claim("Fermat's Last Theorem was proved by Wiles")
        assert report.passed

    def test_collatz_solved_fails(self):
        report = check_claim("the Collatz conjecture has been solved")
        assert report.verdict == "FAIL"

    def test_twin_primes_proved_fails(self):
        report = check_claim("the twin prime conjecture was proven")
        assert report.verdict == "FAIL"

    def test_abc_proved_warns(self):
        """ABC is disputed, not cleanly proved."""
        report = check_claim("the ABC conjecture was proved by Mochizuki")
        assert report.verdict in ("WARN", "FAIL")

    def test_ch_is_true_fails(self):
        """CH is independent of ZFC, not true."""
        report = check_claim("the continuum hypothesis is true")
        assert report.verdict in ("WARN", "FAIL")

    def test_unrecognized_conjecture_in_claim(self):
        report = check_claim("the spaghetti theorem was proved")
        assert report.verdict == "FAIL"
        assert report.conjecture is None

    def test_claim_with_no_status_keyword(self):
        """A claim that mentions a conjecture but no status verb."""
        report = check_claim("Riemann Hypothesis")
        # Should find the conjecture but no claimed status
        if report.conjecture is not None:
            assert report.conjecture.name == "Riemann Hypothesis"


# ─── check_claim — partial results confusion ─────────────────────────────────

class TestPartialResults:
    def test_weak_goldbach_solved_but_strong_open(self):
        """Weak/ternary Goldbach is SOLVED, strong is OPEN."""
        info_weak = get_conjecture("Goldbach weak conjecture")
        assert info_weak is not None
        assert info_weak.status == "SOLVED"
        assert info_weak.solver == "Harald Helfgott"

        info_strong = get_conjecture("Goldbach conjecture")
        assert info_strong is not None
        assert info_strong.status == "OPEN"

    def test_bsd_partial_rank0_rank1(self):
        info = get_conjecture("Birch and Swinnerton-Dyer conjecture")
        assert info is not None
        assert info.status == "OPEN"
        # Key facts mention rank 0 and 1
        assert any("rank 0" in f or "rank 1" in f for f in info.key_facts)


# ─── list_conjectures ────────────────────────────────────────────────────────

class TestListConjectures:
    def test_list_all_nonempty(self):
        all_names = list_conjectures()
        assert len(all_names) > 40  # ~70 conjectures in the DB

    def test_list_open_subset(self):
        open_names = list_conjectures(status="OPEN")
        all_names = list_conjectures()
        assert len(open_names) > 0
        assert len(open_names) < len(all_names)

    def test_list_solved(self):
        solved = list_conjectures(status="SOLVED")
        assert "Poincare conjecture" in solved
        assert "Fermat's Last Theorem" in solved

    def test_list_disputed(self):
        disputed = list_conjectures(status="DISPUTED")
        assert "ABC conjecture" in disputed

    def test_list_independent(self):
        independent = list_conjectures(status="INDEPENDENT")
        assert "Continuum hypothesis" in independent

    def test_list_sorted(self):
        names = list_conjectures()
        assert names == sorted(names)

    def test_rh_in_open(self):
        open_names = list_conjectures(status="OPEN")
        assert "Riemann Hypothesis" in open_names

    def test_poincare_not_in_open(self):
        open_names = list_conjectures(status="OPEN")
        assert "Poincare conjecture" not in open_names


# ─── get_conjecture ──────────────────────────────────────────────────────────

class TestGetConjecture:
    def test_get_rh(self):
        info = get_conjecture("Riemann Hypothesis")
        assert info is not None
        assert info.status == "OPEN"
        assert info.domain == "Millennium"
        assert info.prize == "$1M"

    def test_get_poincare(self):
        info = get_conjecture("Poincare conjecture")
        assert info is not None
        assert info.status == "SOLVED"
        assert info.solver == "Grigori Perelman"
        assert info.solver_year == 2003

    def test_get_flt(self):
        info = get_conjecture("Fermat's Last Theorem")
        assert info is not None
        assert info.solver == "Andrew Wiles"
        assert info.solver_year == 1995

    def test_alias_rh(self):
        info = get_conjecture("RH")
        assert info is not None
        assert info.name == "Riemann Hypothesis"

    def test_alias_flt(self):
        info = get_conjecture("FLT")
        assert info is not None
        assert info.name == "Fermat's Last Theorem"

    def test_alias_poincare(self):
        info = get_conjecture("poincare")
        assert info is not None
        assert info.name == "Poincare conjecture"

    def test_none_for_unknown(self):
        assert get_conjecture("Nonexistent theorem") is None

    def test_conjecture_info_str(self):
        info = get_conjecture("Riemann Hypothesis")
        text = str(info)
        assert "OPEN" in text
        assert "Riemann" in text


# ─── Disputed conjectures ───────────────────────────────────────────────────

class TestDisputedConjectures:
    def test_abc_is_disputed(self):
        info = get_conjecture("ABC conjecture")
        assert info.status == "DISPUTED"

    def test_abc_controversy_note(self):
        """Looking up ABC without claiming status should mention controversy."""
        report = check_conjecture("ABC conjecture")
        assert any(i.check_type == "CONTROVERSY_CHECK" for i in report.issues)

    def test_abc_claimed_disputed_severity(self):
        report = check_conjecture("ABC conjecture", claimed_status="DISPUTED")
        # Even with correct status, DISPUTED gets a controversy flag
        controversy = [i for i in report.issues if i.check_type == "CONTROVERSY_CHECK"]
        assert len(controversy) > 0
        # Should be MODERATE or INFO, not HIGH
        assert all(i.severity in ("MODERATE", "INFO") for i in controversy)


# ─── Report formatting ──────────────────────────────────────────────────────

class TestReportFormatting:
    def test_report_str_contains_verdict(self):
        report = check_conjecture("Riemann Hypothesis", claimed_status="SOLVED")
        text = str(report)
        assert "FAIL" in text

    def test_report_str_contains_conjecture_name(self):
        report = check_conjecture("Riemann Hypothesis")
        text = str(report)
        assert "Riemann Hypothesis" in text

    def test_report_str_contains_actual_status(self):
        report = check_conjecture("Riemann Hypothesis")
        text = str(report)
        assert "OPEN" in text

    def test_report_passed_property(self):
        report_pass = check_conjecture("Riemann Hypothesis", claimed_status="OPEN")
        assert report_pass.passed is True
        report_fail = check_conjecture("Riemann Hypothesis", claimed_status="SOLVED")
        assert report_fail.passed is False

    def test_issue_str(self):
        issue = ConjectureIssue(
            check_type="STATUS_CHECK",
            severity="HIGH",
            description="test description",
        )
        text = str(issue)
        assert "HIGH" in text
        assert "STATUS_CHECK" in text
