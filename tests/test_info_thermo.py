"""Tests for noethersolve.info_thermo — information-thermodynamics tools."""

import math
import pytest

from noethersolve.info_thermo import (
    calc_landauer_bound,
    calc_shannon_entropy,
    calc_info_thermo_bridge,
    calc_huffman_landauer_parallel,
    K_B,
    LN_2,
)


class TestLandauerBound:
    def test_one_bit_room_temp(self):
        """Erasing 1 bit at 300K costs kT×ln(2)."""
        report = calc_landauer_bound(1.0, 300.0)
        expected = K_B * 300.0 * LN_2
        assert abs(report.min_energy_joules - expected) < 1e-25
        assert abs(report.min_energy_kT - LN_2) < 1e-10

    def test_zero_bits(self):
        """Erasing 0 bits costs 0 energy."""
        report = calc_landauer_bound(0.0, 300.0)
        assert report.min_energy_joules == 0.0

    def test_fractional_bits(self):
        """Fractional bits scale linearly."""
        report_full = calc_landauer_bound(1.0, 300.0)
        report_half = calc_landauer_bound(0.5, 300.0)
        assert abs(report_half.min_energy_joules - 0.5 * report_full.min_energy_joules) < 1e-30

    def test_temperature_scaling(self):
        """Energy scales linearly with temperature."""
        report_300 = calc_landauer_bound(1.0, 300.0)
        report_600 = calc_landauer_bound(1.0, 600.0)
        assert abs(report_600.min_energy_joules - 2 * report_300.min_energy_joules) < 1e-25

    def test_entropy_increase(self):
        """Environment entropy increases by k×ln(2) per bit."""
        report = calc_landauer_bound(1.0, 300.0)
        expected_entropy = K_B * LN_2
        assert abs(report.entropy_increase - expected_entropy) < 1e-30

    def test_invalid_temperature(self):
        """Negative temperature raises error."""
        with pytest.raises(ValueError):
            calc_landauer_bound(1.0, -100.0)

    def test_invalid_bits(self):
        """Negative bits raises error."""
        with pytest.raises(ValueError):
            calc_landauer_bound(-1.0, 300.0)

    def test_report_str(self):
        """Report has readable string representation."""
        report = calc_landauer_bound(1.0, 300.0)
        s = str(report)
        assert "Landauer" in s
        assert "kT" in s


class TestShannonEntropy:
    def test_fair_coin(self):
        """Fair coin has entropy 1 bit."""
        report = calc_shannon_entropy([0.5, 0.5])
        assert abs(report.entropy_bits - 1.0) < 1e-10

    def test_certain_outcome(self):
        """Certain outcome has entropy 0."""
        report = calc_shannon_entropy([1.0, 0.0, 0.0])
        assert abs(report.entropy_bits) < 1e-10

    def test_uniform_distribution(self):
        """Uniform over n symbols has entropy log2(n)."""
        n = 8
        probs = [1/n] * n
        report = calc_shannon_entropy(probs)
        assert abs(report.entropy_bits - math.log2(n)) < 1e-10

    def test_biased_coin(self):
        """Biased coin has entropy < 1 bit."""
        report = calc_shannon_entropy([0.9, 0.1])
        assert 0 < report.entropy_bits < 1.0
        # H = -0.9*log2(0.9) - 0.1*log2(0.1) ≈ 0.469
        assert abs(report.entropy_bits - 0.4689955935892812) < 1e-6

    def test_nats_vs_bits(self):
        """Entropy in nats = bits × ln(2)."""
        report = calc_shannon_entropy([0.5, 0.5])
        assert abs(report.entropy_nats - report.entropy_bits * LN_2) < 1e-10

    def test_efficiency(self):
        """Efficiency is H / H_max."""
        report = calc_shannon_entropy([0.5, 0.5])
        assert abs(report.efficiency - 1.0) < 1e-10  # Maximum entropy

        report2 = calc_shannon_entropy([0.9, 0.1])
        assert 0 < report2.efficiency < 1.0

    def test_invalid_probabilities(self):
        """Probabilities not summing to 1 raise error."""
        with pytest.raises(ValueError):
            calc_shannon_entropy([0.5, 0.3])  # Sum = 0.8

    def test_report_str(self):
        """Report has readable string representation."""
        report = calc_shannon_entropy([0.5, 0.5])
        s = str(report)
        assert "Shannon" in s
        assert "bits" in s


class TestInfoThermoBridge:
    def test_bridge_consistency(self):
        """Bridge connects Shannon and Gibbs correctly."""
        probs = [0.5, 0.5]
        report = calc_info_thermo_bridge(probs, 300.0)

        # Gibbs entropy = k × H_nats
        expected_gibbs = K_B * report.shannon_entropy_bits * LN_2
        assert abs(report.gibbs_entropy_JK - expected_gibbs) < 1e-30

        # Landauer energy = kT × ln(2) × H_bits
        expected_landauer = K_B * 300.0 * LN_2 * report.shannon_entropy_bits
        assert abs(report.landauer_energy_joules - expected_landauer) < 1e-25

    def test_report_str(self):
        """Report has readable string representation."""
        report = calc_info_thermo_bridge([0.5, 0.5], 300.0)
        s = str(report)
        assert "Bridge" in s
        assert "Gibbs" in s


class TestHuffmanLandauerParallel:
    def test_parallel_structure(self):
        """Both optimize same objective."""
        probs = [0.5, 0.25, 0.125, 0.125]
        report = calc_huffman_landauer_parallel(probs, 300.0)

        # Shannon entropy is the bound for both
        expected_H = -sum(p * math.log2(p) for p in probs if p > 0)
        assert abs(report.shannon_entropy - expected_H) < 1e-10
        assert abs(report.huffman_bound - expected_H) < 1e-10

        # Energy per bit is kT×ln(2)
        assert abs(report.energy_per_bit - K_B * 300.0 * LN_2) < 1e-30

    def test_report_str(self):
        """Report shows the parallel clearly."""
        report = calc_huffman_landauer_parallel([0.5, 0.5], 300.0)
        s = str(report)
        assert "Huffman" in s
        assert "Landauer" in s
        assert "p_i" in s  # The objective function
