"""Tests for the cryptography calculator module."""

import math
import pytest
from noethersolve.crypto_calc import (
    security_level,
    birthday_bound,
    rsa_key_analysis,
    cipher_mode_analysis,
)


class TestSecurityLevel:
    def test_aes256_classical(self):
        """AES-256 has 256-bit classical security."""
        r = security_level("AES", 256)
        assert r.security_bits == 256

    def test_aes256_quantum(self):
        """AES-256 has 128-bit post-quantum security (Grover halves)."""
        r = security_level("AES", 256)
        assert r.quantum_security_bits == 128

    def test_aes128(self):
        r = security_level("AES", 128)
        assert r.security_bits == 128
        assert r.quantum_security_bits == 64

    def test_3des_meet_in_middle(self):
        """3DES effective security = 112 bits due to meet-in-the-middle."""
        r = security_level("3DES", 168)
        assert r.security_bits == 112
        assert r.quantum_security_bits == 56

    def test_rsa2048_equivalent(self):
        """RSA-2048 has ~112-bit symmetric equivalent."""
        r = security_level("RSA", 2048)
        assert r.security_bits == 112

    def test_rsa_quantum_zero(self):
        """RSA has 0-bit quantum security (Shor's algorithm)."""
        r = security_level("RSA", 2048)
        assert r.quantum_security_bits == 0

    def test_ecc256(self):
        """ECC-256 has ~128-bit classical security."""
        r = security_level("ECC", 256)
        assert r.security_bits == 128

    def test_ecc_quantum_zero(self):
        """ECC has 0-bit quantum security (Shor's algorithm breaks ECDLP)."""
        r = security_level("ECC", 256)
        assert r.quantum_security_bits == 0

    def test_sha256_preimage(self):
        """SHA-256 has 256-bit preimage resistance."""
        r = security_level("SHA256", 256)
        assert r.security_bits == 256

    def test_brute_force_ops(self):
        """Brute force operations = 2^security_bits."""
        r = security_level("AES", 128)
        assert abs(r.brute_force_ops - 2**128) < 1e20

    def test_time_at_rate(self):
        """Time = 2^security_bits / ops_per_second."""
        r = security_level("AES", 128, ops_per_second=1e12)
        expected = 2**128 / 1e12
        assert abs(r.time_at_rate - expected) / expected < 1e-6

    def test_des_weak(self):
        r = security_level("DES", 56)
        assert r.security_bits == 56

    def test_report_string(self):
        r = security_level("AES", 256)
        s = str(r)
        assert "Security Level" in s
        assert "AES" in s
        assert "256" in s
        assert "Post-quantum" in s


class TestBirthdayBound:
    def test_sha256_trials_for_50pct(self):
        """SHA-256: ~2^128 trials for 50% collision probability."""
        r = birthday_bound(256)
        log2_trials = math.log2(r.trials_for_50pct)
        assert abs(log2_trials - 128) < 1.0  # within 1 bit

    def test_small_output_space(self):
        """16-bit output: trials_for_50pct ~ 2^8 = 256."""
        r = birthday_bound(16)
        assert 200 < r.trials_for_50pct < 400

    def test_probability_at_trials(self):
        """Given specific trial count, probability should be computed."""
        r = birthday_bound(32, trials=2**16)
        assert r.probability_at_trials is not None
        assert 0 < r.probability_at_trials < 1

    def test_many_trials_high_probability(self):
        """Many trials relative to output space gives probability near 1."""
        r = birthday_bound(16, trials=2**15)
        assert r.probability_at_trials > 0.99

    def test_few_trials_low_probability(self):
        """Very few trials gives low probability."""
        r = birthday_bound(128, trials=100)
        assert r.probability_at_trials < 0.01

    def test_error_zero_bits(self):
        with pytest.raises(ValueError, match="positive"):
            birthday_bound(0)

    def test_error_negative_trials(self):
        with pytest.raises(ValueError, match="positive"):
            birthday_bound(128, trials=-1)

    def test_report_string(self):
        r = birthday_bound(256)
        s = str(r)
        assert "Birthday Bound" in s
        assert "collision" in s


class TestRSAKeyAnalysis:
    def test_rsa2048_security(self):
        """RSA-2048 has ~112-bit equivalent security."""
        r = rsa_key_analysis(2048)
        assert r.security_bits == 112
        assert r.recommended is True

    def test_rsa1024_not_recommended(self):
        """RSA-1024 is below NIST minimum."""
        r = rsa_key_analysis(1024)
        assert r.security_bits == 80
        assert r.recommended is False
        assert "below NIST" in r.recommendation.lower() or "upgrade" in r.recommendation.lower()

    def test_rsa4096(self):
        r = rsa_key_analysis(4096)
        assert r.security_bits == 128
        assert r.recommended is True

    def test_always_quantum_vulnerable(self):
        """All RSA keys are vulnerable to Shor's algorithm."""
        r = rsa_key_analysis(4096)
        assert r.quantum_vulnerable is True

    def test_gnfs_ops_positive(self):
        r = rsa_key_analysis(2048)
        assert r.gnfs_ops > 0

    def test_error_too_small(self):
        with pytest.raises(ValueError, match="512"):
            rsa_key_analysis(256)

    def test_report_string(self):
        r = rsa_key_analysis(2048)
        s = str(r)
        assert "RSA Key" in s
        assert "2048" in s
        assert "GNFS" in s


class TestCipherModeAnalysis:
    def test_cbc_requires_random_iv(self):
        """CBC requires a random (unpredictable) IV."""
        r = cipher_mode_analysis("CBC")
        assert r.requires_iv is True
        assert r.iv_must_be_random is True

    def test_gcm_is_authenticated(self):
        """GCM is an AEAD mode."""
        r = cipher_mode_analysis("GCM")
        assert r.authenticated is True

    def test_ecb_insecure(self):
        """ECB does not require an IV and is not authenticated."""
        r = cipher_mode_analysis("ECB")
        assert r.requires_iv is False
        assert r.authenticated is False

    def test_ctr_nonce_not_random(self):
        """CTR nonce must be unique but need not be random."""
        r = cipher_mode_analysis("CTR")
        assert r.requires_iv is True
        assert r.iv_must_be_random is False

    def test_cbc_not_parallel_encrypt(self):
        """CBC encryption is not parallelizable."""
        r = cipher_mode_analysis("CBC")
        assert r.parallelizable_encrypt is False
        assert r.parallelizable_decrypt is True

    def test_gcm_fully_parallel(self):
        r = cipher_mode_analysis("GCM")
        assert r.parallelizable_encrypt is True
        assert r.parallelizable_decrypt is True

    def test_ccm_authenticated(self):
        r = cipher_mode_analysis("CCM")
        assert r.authenticated is True

    def test_birthday_bound_128bit(self):
        """128-bit block: max safe blocks ~ 2^64."""
        r = cipher_mode_analysis("CBC", block_bits=128)
        assert r.max_safe_blocks == 2**64

    def test_error_unknown_mode(self):
        with pytest.raises(ValueError, match="Unknown mode"):
            cipher_mode_analysis("XYZ")

    def test_report_string(self):
        r = cipher_mode_analysis("GCM")
        s = str(r)
        assert "Cipher Mode" in s
        assert "GCM" in s
        assert "Authenticated" in s
