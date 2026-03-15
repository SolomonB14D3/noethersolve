"""Tests for noethersolve.number_theory — Number theory conjecture verifier."""

import math

import pytest

from noethersolve.number_theory import (
    verify_goldbach,
    verify_collatz,
    verify_twin_primes,
    check_abc_triple,
    verify_legendre,
    prime_gap_analysis,
    is_prime,
    prime_sieve,
    radical,
    GoldbachReport,
    CollatzReport,
    TwinPrimeReport,
    ABCReport,
    LegendreReport,
    PrimeGapReport,
)


# ─── is_prime ────────────────────────────────────────────────────────────────

class TestIsPrime:
    def test_small_primes(self):
        known_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
        for p in known_primes:
            assert is_prime(p), f"{p} should be prime"

    def test_small_composites(self):
        composites = [4, 6, 8, 9, 10, 12, 14, 15, 16, 18, 20, 21, 25]
        for c in composites:
            assert not is_prime(c), f"{c} should not be prime"

    def test_edge_cases_0_and_1(self):
        assert not is_prime(0)
        assert not is_prime(1)

    def test_two_is_prime(self):
        assert is_prime(2)

    def test_negative_numbers(self):
        assert not is_prime(-1)
        assert not is_prime(-7)

    def test_large_prime(self):
        # 104729 is the 10000th prime
        assert is_prime(104729)

    def test_large_composite(self):
        # 104729 * 3 is composite
        assert not is_prime(104729 * 3)

    def test_miller_rabin_range(self):
        """Test primes above the trial division threshold (>=1000)."""
        assert is_prime(1009)
        assert not is_prime(1001)  # 7 * 11 * 13


# ─── prime_sieve ─────────────────────────────────────────────────────────────

class TestPrimeSieve:
    def test_sieve_30(self):
        expected = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
        assert prime_sieve(30) == expected

    def test_sieve_below_2(self):
        assert prime_sieve(1) == []
        assert prime_sieve(0) == []

    def test_sieve_exactly_2(self):
        assert prime_sieve(2) == [2]

    def test_sieve_100_count(self):
        # There are 25 primes below 100
        assert len(prime_sieve(100)) == 25

    def test_sieve_inclusive(self):
        """Limit is inclusive — if limit is prime, it should be included."""
        assert 29 in prime_sieve(29)
        assert 31 not in prime_sieve(30)


# ─── radical ─────────────────────────────────────────────────────────────────

class TestRadical:
    def test_radical_12(self):
        # 12 = 2^2 * 3, so rad(12) = 2 * 3 = 6
        assert radical(12) == 6

    def test_radical_1(self):
        # rad(1) = 1 (empty product)
        assert radical(1) == 1

    def test_radical_prime(self):
        # rad(p) = p for prime p
        assert radical(7) == 7

    def test_radical_prime_power(self):
        # rad(2^10) = 2
        assert radical(1024) == 2

    def test_radical_squarefree(self):
        # rad(30) = 30 since 30 = 2*3*5 already squarefree
        assert radical(30) == 30

    def test_radical_raises_for_nonpositive(self):
        with pytest.raises(ValueError):
            radical(0)
        with pytest.raises(ValueError):
            radical(-5)


# ─── verify_goldbach ─────────────────────────────────────────────────────────

class TestVerifyGoldbach:
    def test_goldbach_100_has_6_decompositions(self):
        report = verify_goldbach(100)
        assert isinstance(report, GoldbachReport)
        assert report.is_verified
        assert report.decomposition_count == 6

    def test_goldbach_4_equals_2_plus_2(self):
        report = verify_goldbach(4)
        assert report.is_verified
        assert report.decomposition_count == 1
        assert report.smallest_pair == (2, 2)
        assert report.largest_pair == (2, 2)

    def test_goldbach_odd_raises(self):
        with pytest.raises(ValueError, match="odd"):
            verify_goldbach(7)

    def test_goldbach_2_raises(self):
        with pytest.raises(ValueError):
            verify_goldbach(2)

    def test_goldbach_negative_raises(self):
        with pytest.raises(ValueError):
            verify_goldbach(-4)

    def test_goldbach_passed_property(self):
        report = verify_goldbach(20)
        assert report.passed is True

    def test_goldbach_severity_info(self):
        report = verify_goldbach(10)
        assert report.severity == "INFO"

    def test_goldbach_small_even_numbers(self):
        """All even numbers from 4 to 100 should be verified."""
        for n in range(4, 102, 2):
            report = verify_goldbach(n)
            assert report.is_verified, f"Goldbach failed for {n}"


# ─── verify_collatz ──────────────────────────────────────────────────────────

class TestVerifyCollatz:
    def test_collatz_27_takes_111_steps(self):
        report = verify_collatz(27)
        assert isinstance(report, CollatzReport)
        assert report.reached_one
        assert report.steps == 111

    def test_collatz_1_takes_0_steps(self):
        report = verify_collatz(1)
        assert report.reached_one
        assert report.steps == 0
        assert report.max_value == 1

    def test_collatz_trajectory_stored_for_small_n(self):
        report = verify_collatz(5)
        assert report.trajectory is not None
        assert report.trajectory[0] == 5
        assert report.trajectory[-1] == 1

    def test_collatz_trajectory_not_stored_for_large_n(self):
        report = verify_collatz(1001)
        assert report.trajectory is None

    def test_collatz_max_value_27(self):
        report = verify_collatz(27)
        assert report.max_value == 9232

    def test_collatz_raises_for_zero(self):
        with pytest.raises(ValueError):
            verify_collatz(0)

    def test_collatz_raises_for_negative(self):
        with pytest.raises(ValueError):
            verify_collatz(-3)

    def test_collatz_passed_property(self):
        report = verify_collatz(7)
        assert report.passed is True

    def test_collatz_severity_info(self):
        report = verify_collatz(10)
        assert report.severity == "INFO"

    def test_collatz_trajectory_length(self):
        report = verify_collatz(27)
        # trajectory_length = steps + 1 (includes starting value)
        assert report.trajectory_length == 112


# ─── verify_twin_primes ─────────────────────────────────────────────────────

class TestVerifyTwinPrimes:
    def test_twin_primes_100_known_pairs(self):
        report = verify_twin_primes(100)
        assert isinstance(report, TwinPrimeReport)
        known_pairs = [(3, 5), (5, 7), (11, 13), (17, 19), (29, 31),
                       (41, 43), (59, 61), (71, 73)]
        for pair in known_pairs:
            assert pair in report.pairs, f"Missing twin prime pair {pair}"

    def test_twin_primes_100_count(self):
        report = verify_twin_primes(100)
        assert report.count == 8

    def test_twin_primes_density_positive(self):
        report = verify_twin_primes(100)
        assert report.density > 0

    def test_twin_primes_limit_2(self):
        report = verify_twin_primes(2)
        assert report.count == 0

    def test_twin_primes_limit_below_2_raises(self):
        with pytest.raises(ValueError):
            verify_twin_primes(1)

    def test_twin_primes_passed_property(self):
        report = verify_twin_primes(100)
        assert report.passed is True

    def test_twin_primes_largest_pair(self):
        report = verify_twin_primes(100)
        assert report.largest_pair == (71, 73)

    def test_twin_primes_1000_count(self):
        report = verify_twin_primes(1000)
        assert report.count == 35


# ─── check_abc_triple ────────────────────────────────────────────────────────

class TestCheckABCTriple:
    def test_exceptional_triple_1_8_9(self):
        report = check_abc_triple(1, 8, 9)
        assert isinstance(report, ABCReport)
        assert report.is_valid_triple
        assert report.is_exceptional
        assert report.quality > 1.0

    def test_ordinary_triple(self):
        # 1 + 2 = 3, gcd(1,2) = 1, rad(1*2*3) = 6, q = log(3)/log(6) ~ 0.613
        report = check_abc_triple(1, 2, 3)
        assert report.is_valid_triple
        assert not report.is_exceptional
        assert report.quality < 1.0

    def test_invalid_sum(self):
        report = check_abc_triple(1, 2, 4)
        assert not report.is_valid_triple

    def test_invalid_gcd(self):
        # 2 + 4 = 6, gcd(2,4) = 2 != 1
        report = check_abc_triple(2, 4, 6)
        assert not report.is_valid_triple

    def test_nonpositive_raises_invalid(self):
        report = check_abc_triple(0, 1, 1)
        assert not report.is_valid_triple

    def test_abc_radical_value(self):
        # (1, 8, 9): abc = 1*8*9 = 72 = 2^3 * 3^2, rad = 2*3 = 6
        report = check_abc_triple(1, 8, 9)
        assert report.radical == 6

    def test_abc_quality_1_8_9(self):
        report = check_abc_triple(1, 8, 9)
        expected_q = math.log(9) / math.log(6)
        assert abs(report.quality - expected_q) < 1e-10

    def test_abc_severity_exceptional(self):
        report = check_abc_triple(1, 8, 9)
        assert report.severity == "MODERATE"  # q > 1 but < 1.4

    def test_abc_severity_ordinary(self):
        report = check_abc_triple(1, 2, 3)
        assert report.severity == "LOW"

    def test_abc_passed_property(self):
        report = check_abc_triple(1, 8, 9)
        assert report.passed is True
        report2 = check_abc_triple(1, 2, 4)
        assert report2.passed is False

    def test_high_quality_triple(self):
        # (2, 3^10-2, 3^10) = (2, 59047, 59049)
        # gcd(2, 59047) = 1, 2+59047=59049, rad = 2*59047*3... need to check
        # Use known: (1, 2^6*3, 5^3) is not valid. Use (5, 27, 32) instead.
        report = check_abc_triple(5, 27, 32)
        assert report.is_valid_triple
        assert report.quality > 1.0  # This is a known exceptional triple


# ─── verify_legendre ─────────────────────────────────────────────────────────

class TestVerifyLegendre:
    def test_legendre_n1(self):
        # Interval (1, 4): primes 2, 3
        report = verify_legendre(1)
        assert isinstance(report, LegendreReport)
        assert report.is_verified
        assert 2 in report.primes_found
        assert 3 in report.primes_found

    def test_legendre_n0(self):
        # Interval (0, 1): no primes
        report = verify_legendre(0)
        assert not report.is_verified

    def test_legendre_n10(self):
        # Interval (100, 121): should have primes 101, 103, 107, 109, 113
        report = verify_legendre(10)
        assert report.is_verified
        assert report.count == 5
        assert report.n_squared == 100
        assert report.n_plus_1_squared == 121

    def test_legendre_negative_raises(self):
        with pytest.raises(ValueError):
            verify_legendre(-1)

    def test_legendre_small_values_verified(self):
        """Legendre verified for n = 1 through 50."""
        for n in range(1, 51):
            report = verify_legendre(n)
            assert report.is_verified, f"Legendre failed for n={n}"

    def test_legendre_severity(self):
        report = verify_legendre(5)
        assert report.severity == "INFO"

    def test_legendre_passed_property(self):
        report = verify_legendre(5)
        assert report.passed is True


# ─── prime_gap_analysis ──────────────────────────────────────────────────────

class TestPrimeGapAnalysis:
    def test_gap_analysis_10000(self):
        report = prime_gap_analysis(10000)
        assert isinstance(report, PrimeGapReport)
        assert report.prime_count > 0
        assert report.max_gap > 0
        assert report.avg_gap > 0

    def test_gap_analysis_cramer_ratio_under_1(self):
        report = prime_gap_analysis(10000)
        assert report.cramer_ratio < 1.0

    def test_gap_analysis_limit_too_small(self):
        with pytest.raises(ValueError):
            prime_gap_analysis(2)

    def test_gap_analysis_max_gap_location(self):
        report = prime_gap_analysis(100)
        start, end = report.max_gap_location
        assert end - start == report.max_gap
        assert is_prime(start)
        assert is_prime(end)

    def test_gap_analysis_passed_property(self):
        report = prime_gap_analysis(10000)
        assert report.passed is True  # cramer_ratio < 1

    def test_gap_analysis_severity(self):
        report = prime_gap_analysis(10000)
        assert report.severity in ("INFO", "MODERATE")

    def test_gap_analysis_prime_count(self):
        report = prime_gap_analysis(100)
        assert report.prime_count == 25


# ─── Report formatting ──────────────────────────────────────────────────────

class TestReportFormatting:
    def test_goldbach_str(self):
        report = verify_goldbach(100)
        text = str(report)
        assert "Goldbach" in text
        assert "100" in text

    def test_collatz_str(self):
        report = verify_collatz(27)
        text = str(report)
        assert "Collatz" in text
        assert "111" in text

    def test_twin_prime_str(self):
        report = verify_twin_primes(100)
        text = str(report)
        assert "Twin Prime" in text

    def test_abc_str(self):
        report = check_abc_triple(1, 8, 9)
        text = str(report)
        assert "ABC" in text

    def test_legendre_str(self):
        report = verify_legendre(10)
        text = str(report)
        assert "Legendre" in text

    def test_prime_gap_str(self):
        report = prime_gap_analysis(10000)
        text = str(report)
        assert "Prime Gap" in text
        assert "Cramer" in text

    def test_collatz_short_trajectory_in_str(self):
        """Short trajectory should be printed in full."""
        report = verify_collatz(5)
        text = str(report)
        assert "Trajectory" in text

    def test_abc_invalid_str_shows_reason(self):
        report = check_abc_triple(1, 2, 4)
        text = str(report)
        assert "INVALID" in text
