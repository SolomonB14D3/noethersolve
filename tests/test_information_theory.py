"""Tests for information theory module."""

import math
from noethersolve.information_theory import (
    binary_entropy,
    entropy,
    relative_entropy,
    mutual_information,
    capacity_bsc,
    capacity_bec,
    capacity_awgn,
    capacity_z_channel,
    rate_distortion_binary,
    rate_distortion_gaussian,
    source_coding_bound,
    mac_capacity_region_2user,
    check_data_processing,
    fano_inequality,
    typical_set_bounds,
)


class TestEntropy:
    """Test entropy functions."""

    def test_binary_entropy_half(self):
        """H(0.5) = 1 bit."""
        assert abs(binary_entropy(0.5) - 1.0) < 1e-10

    def test_binary_entropy_zero(self):
        """H(0) = 0."""
        assert binary_entropy(0) == 0

    def test_binary_entropy_one(self):
        """H(1) = 0."""
        assert binary_entropy(1) == 0

    def test_binary_entropy_symmetric(self):
        """H(p) = H(1-p)."""
        assert abs(binary_entropy(0.3) - binary_entropy(0.7)) < 1e-10

    def test_entropy_uniform(self):
        """Uniform distribution maximizes entropy."""
        probs = [0.25, 0.25, 0.25, 0.25]
        assert abs(entropy(probs) - 2.0) < 1e-10  # log2(4) = 2

    def test_entropy_deterministic(self):
        """Deterministic distribution has zero entropy."""
        probs = [1.0, 0.0, 0.0]
        assert entropy(probs) == 0

    def test_relative_entropy_same(self):
        """D(P||P) = 0."""
        p = [0.5, 0.5]
        assert relative_entropy(p, p) == 0

    def test_relative_entropy_divergence(self):
        """D(P||Q) > 0 for P ≠ Q."""
        p = [0.5, 0.5]
        q = [0.9, 0.1]
        assert relative_entropy(p, q) > 0

    def test_mutual_information_independent(self):
        """I(X;Y) = 0 for independent X, Y."""
        # Independent: P(x,y) = P(x)P(y)
        joint = [[0.25, 0.25], [0.25, 0.25]]
        assert abs(mutual_information(joint)) < 1e-10


class TestChannelCapacity:
    """Test channel capacity calculations."""

    def test_bsc_noiseless(self):
        """BSC with p=0 has capacity 1."""
        report = capacity_bsc(0)
        assert abs(report.capacity - 1.0) < 1e-10

    def test_bsc_useless(self):
        """BSC with p=0.5 has capacity 0."""
        report = capacity_bsc(0.5)
        assert abs(report.capacity) < 1e-10

    def test_bsc_typical(self):
        """BSC with p=0.1 has C = 1 - H(0.1)."""
        report = capacity_bsc(0.1)
        expected = 1 - binary_entropy(0.1)
        assert abs(report.capacity - expected) < 1e-10

    def test_bec_noiseless(self):
        """BEC with ε=0 has capacity 1."""
        report = capacity_bec(0)
        assert abs(report.capacity - 1.0) < 1e-10

    def test_bec_useless(self):
        """BEC with ε=1 has capacity 0."""
        report = capacity_bec(1)
        assert abs(report.capacity) < 1e-10

    def test_bec_linear(self):
        """BEC capacity is linear: C = 1 - ε."""
        report = capacity_bec(0.3)
        assert abs(report.capacity - 0.7) < 1e-10

    def test_bsc_vs_bec(self):
        """BEC has higher capacity than BSC at same parameter."""
        # At p = ε = 0.1:
        bsc = capacity_bsc(0.1)
        bec = capacity_bec(0.1)
        assert bec.capacity > bsc.capacity  # 0.9 > 1 - H(0.1) ≈ 0.531

    def test_awgn_zero_snr(self):
        """AWGN with SNR=0 has capacity 0."""
        report = capacity_awgn(0)
        assert abs(report.capacity) < 1e-10

    def test_awgn_high_snr(self):
        """AWGN capacity grows logarithmically."""
        c1 = capacity_awgn(10).capacity
        c2 = capacity_awgn(100).capacity
        # C(100) ≈ log2(101) ≈ 6.66, C(10) ≈ log2(11) ≈ 3.46
        assert c2 > c1
        assert c2 < 2 * c1  # Logarithmic, not linear

    def test_awgn_bandwidth_scaling(self):
        """AWGN capacity scales linearly with bandwidth."""
        c1 = capacity_awgn(10, bandwidth=1).capacity
        c2 = capacity_awgn(10, bandwidth=2).capacity
        assert abs(c2 - 2 * c1) < 1e-10

    def test_z_channel_noiseless(self):
        """Z-channel with p=0 has capacity 1."""
        report = capacity_z_channel(0)
        assert abs(report.capacity - 1.0) < 1e-10

    def test_z_channel_useless(self):
        """Z-channel with p=1 has capacity 0."""
        report = capacity_z_channel(1)
        assert abs(report.capacity) < 1e-10

    def test_z_channel_asymmetric_input(self):
        """Z-channel optimal input is not uniform."""
        report = capacity_z_channel(0.2)
        # Optimal P(X=1) should be less than 0.5
        assert report.parameters["optimal_P1"] < 0.5


class TestRateDistortion:
    """Test rate-distortion calculations."""

    def test_binary_zero_distortion(self):
        """R(0) = H(X) for binary source."""
        report = rate_distortion_binary(0)
        assert abs(report.rate - 1.0) < 1e-10  # H(0.5) = 1

    def test_binary_max_distortion(self):
        """R(0.5) = 0 for binary source."""
        report = rate_distortion_binary(0.5)
        assert abs(report.rate) < 1e-10

    def test_gaussian_zero_distortion(self):
        """R(0) = ∞ for Gaussian source."""
        report = rate_distortion_gaussian(0.001, variance=1.0)
        assert report.rate > 4  # Very high rate needed

    def test_gaussian_high_distortion(self):
        """R(D) = 0 for D ≥ σ²."""
        report = rate_distortion_gaussian(2.0, variance=1.0)
        assert abs(report.rate) < 1e-10

    def test_gaussian_formula(self):
        """R(D) = (1/2) log₂(σ²/D)."""
        D = 0.25
        var = 1.0
        report = rate_distortion_gaussian(D, var)
        expected = 0.5 * math.log2(var / D)  # = 0.5 * 2 = 1
        assert abs(report.rate - expected) < 1e-10


class TestSourceCoding:
    """Test source coding bounds."""

    def test_entropy_bound(self):
        """Minimum rate equals entropy."""
        probs = [0.5, 0.25, 0.125, 0.125]
        report = source_coding_bound(probs)
        expected_H = entropy(probs)  # 1.75 bits
        assert abs(report.entropy - expected_H) < 1e-10

    def test_valid_code(self):
        """Valid prefix code satisfies Kraft."""
        probs = [0.5, 0.25, 0.125, 0.125]
        lengths = [1, 2, 3, 3]  # Kraft sum = 0.5 + 0.25 + 0.125 + 0.125 = 1
        report = source_coding_bound(probs, lengths)
        assert report.passed

    def test_invalid_code(self):
        """Invalid code violates Kraft."""
        probs = [0.5, 0.5]
        # Actually [1,1] is valid (sum = 1). Let's use impossible lengths
        report = source_coding_bound(probs, [0, 0])  # 2^0 + 2^0 = 2 > 1
        assert not report.passed


class TestMAC:
    """Test multiple access channel calculations."""

    def test_mac_pentagon(self):
        """MAC region has 5 corners."""
        report = mac_capacity_region_2user(1.0, 1.0, 1.5)
        assert len(report.corner_points) == 5

    def test_mac_sum_rate(self):
        """Sum rate is I(X1,X2;Y)."""
        report = mac_capacity_region_2user(1.0, 1.0, 1.5)
        assert abs(report.sum_rate - 1.5) < 1e-10

    def test_mac_invalid(self):
        """Invalid if I(X1,X2;Y) > sum of individual."""
        report = mac_capacity_region_2user(0.5, 0.5, 2.0)
        assert not report.passed


class TestDataProcessing:
    """Test data processing inequality."""

    def test_dpi_satisfied(self):
        """Valid Markov chain satisfies DPI."""
        report = check_data_processing(1.0, 0.5)
        assert report.satisfies_dpi

    def test_dpi_violated(self):
        """Invalid if I(X;Z) > I(X;Y)."""
        report = check_data_processing(0.5, 1.0)
        assert not report.satisfies_dpi

    def test_dpi_equality(self):
        """Equality when Z sufficient statistic."""
        report = check_data_processing(1.0, 1.0)
        assert report.satisfies_dpi


class TestFano:
    """Test Fano's inequality."""

    def test_fano_zero_error(self):
        """Zero error when H(X|Y) = 0."""
        report = fano_inequality(1.0, 0.0, 2)
        assert report.P_error == 0

    def test_fano_high_error(self):
        """High error when H(X|Y) high."""
        report = fano_inequality(1.0, 0.9, 2)
        assert report.P_error > 0


class TestTypicalSet:
    """Test typical set calculations."""

    def test_typical_set_size(self):
        """Typical set has ~2^(nH) elements."""
        probs = [0.5, 0.5]  # H = 1
        report = typical_set_bounds(probs, n=100)
        # Size ≈ 2^100
        assert abs(math.log2(report.typical_set_size) - 100) < 1

    def test_typical_set_smaller_than_full(self):
        """Typical set smaller than full space when H < log|X|."""
        probs = [0.9, 0.1]  # H ≈ 0.47
        report = typical_set_bounds(probs, n=100)
        full_space = 2 ** 100
        assert report.typical_set_size < full_space
