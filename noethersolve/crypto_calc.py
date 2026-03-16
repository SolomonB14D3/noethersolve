"""Cryptography calculator — derives security properties from first principles.

Covers key security levels, birthday bounds, brute force estimates,
RSA key sizing, and cipher parameter analysis.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class SecurityLevelReport:
    """Result of security level analysis."""
    algorithm: str
    key_bits: int
    security_bits: int  # effective security level
    brute_force_ops: float  # operations to brute force
    time_at_rate: float  # seconds at given ops/sec
    quantum_security_bits: int  # post-Grover security
    explanation: str

    def __str__(self) -> str:
        lines = [
            f"Security Level Analysis — {self.algorithm}",
            f"  Key size: {self.key_bits} bits",
            f"  Classical security: {self.security_bits} bits",
            f"  Brute force ops: 2^{self.security_bits} = {self.brute_force_ops:.2e}",
            f"  Time at given rate: {self.time_at_rate:.2e} seconds",
            f"  Post-quantum (Grover): {self.quantum_security_bits} bits",
            f"  {self.explanation}",
        ]
        return "\n".join(lines)


@dataclass
class BirthdayBoundReport:
    """Result of birthday bound calculation."""
    output_bits: int
    collision_probability: float
    trials_for_50pct: float
    trials_given: Optional[int]
    probability_at_trials: Optional[float]

    def __str__(self) -> str:
        lines = [
            f"Birthday Bound Analysis:",
            f"  Output space: 2^{self.output_bits} = {2**self.output_bits:.2e}",
            f"  Trials for 50% collision: ~2^{math.log2(self.trials_for_50pct):.1f} = {self.trials_for_50pct:.2e}",
        ]
        if self.trials_given is not None:
            lines.append(f"  At {self.trials_given} trials: P(collision) ≈ {self.probability_at_trials:.6e}")
        return "\n".join(lines)


@dataclass
class RSAKeyReport:
    """Result of RSA key sizing analysis."""
    modulus_bits: int
    security_bits: int  # approximate NIST equivalent
    recommended: bool
    gnfs_ops: float  # General Number Field Sieve complexity
    quantum_vulnerable: bool
    recommendation: str

    def __str__(self) -> str:
        lines = [
            f"RSA Key Analysis:",
            f"  Modulus size: {self.modulus_bits} bits",
            f"  Security level: ~{self.security_bits} bits (symmetric equivalent)",
            f"  GNFS complexity: ~2^{math.log2(self.gnfs_ops):.1f} operations",
            f"  Recommended: {self.recommended}",
            f"  Quantum vulnerable: {self.quantum_vulnerable} (Shor's algorithm)",
            f"  {self.recommendation}",
        ]
        return "\n".join(lines)


@dataclass
class CipherModeReport:
    """Analysis of block cipher mode properties."""
    mode: str
    requires_iv: bool
    iv_must_be_random: bool
    parallelizable_encrypt: bool
    parallelizable_decrypt: bool
    authenticated: bool
    ciphertext_expansion: str
    max_safe_blocks: Optional[float]  # before birthday bound risk
    properties: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        lines = [
            f"Cipher Mode Analysis — {self.mode}:",
            f"  Requires IV: {self.requires_iv}",
            f"  IV must be random: {self.iv_must_be_random}",
            f"  Parallel encrypt: {self.parallelizable_encrypt}",
            f"  Parallel decrypt: {self.parallelizable_decrypt}",
            f"  Authenticated: {self.authenticated}",
            f"  Ciphertext expansion: {self.ciphertext_expansion}",
        ]
        if self.max_safe_blocks is not None:
            lines.append(f"  Max safe blocks: 2^{math.log2(self.max_safe_blocks):.0f}")
        for prop in self.properties:
            lines.append(f"  • {prop}")
        return "\n".join(lines)


def security_level(
    algorithm: str,
    key_bits: int,
    ops_per_second: float = 1e12,
) -> SecurityLevelReport:
    """Calculate effective security level for a cryptographic algorithm.

    Derives security bits from key size and algorithm type.
    Accounts for known attacks (e.g., meet-in-the-middle for 3DES,
    Grover's algorithm for quantum).

    Args:
        algorithm: Algorithm name (aes, 3des, chacha20, rsa, ecc, sha)
        key_bits: Key or output size in bits
        ops_per_second: Attacker's throughput (default 10^12)

    Returns:
        SecurityLevelReport with classical and quantum security levels.
    """
    algo = algorithm.lower().replace("-", "").replace("_", "")

    if algo in ("aes", "chacha20", "chacha20poly1305"):
        security_bits = key_bits
        quantum_bits = key_bits // 2  # Grover
        explanation = f"Symmetric cipher: security = key size. Grover halves to {quantum_bits}-bit."

    elif algo in ("3des", "tripledes", "tdes"):
        security_bits = 112  # meet-in-the-middle reduces 168 to 112
        quantum_bits = 56
        explanation = "3DES: 168-bit key but meet-in-the-middle reduces to 112-bit effective."

    elif algo in ("des",):
        security_bits = 56
        quantum_bits = 28
        explanation = "DES: 56-bit key, trivially brute-forceable."

    elif algo in ("rsa",):
        # NIST SP 800-57 approximation
        security_bits = _rsa_security_bits(key_bits)
        quantum_bits = 0  # Shor's algorithm breaks RSA
        explanation = f"RSA-{key_bits}: ~{security_bits}-bit symmetric equivalent. Broken by quantum (Shor)."

    elif algo in ("ecc", "ecdsa", "ecdh", "ed25519", "x25519"):
        security_bits = key_bits // 2  # ECDLP
        quantum_bits = 0  # Shor's algorithm breaks ECDLP (not just Grover)
        explanation = f"ECC-{key_bits}: ~{security_bits}-bit symmetric equivalent. Broken by quantum (Shor)."

    elif algo in ("sha", "sha2", "sha256", "sha384", "sha512", "sha3"):
        # Preimage: full bits. Collision: half bits.
        security_bits = key_bits  # preimage resistance
        quantum_bits = key_bits * 2 // 3  # Grover on preimage
        explanation = f"Hash: {key_bits}-bit preimage, {key_bits//2}-bit collision resistance."

    else:
        # Generic symmetric assumption
        security_bits = key_bits
        quantum_bits = key_bits // 2
        explanation = f"Unknown algorithm, assuming symmetric: {key_bits}-bit security."

    brute_force_ops = 2.0 ** security_bits
    time_seconds = brute_force_ops / ops_per_second

    return SecurityLevelReport(
        algorithm=algorithm,
        key_bits=key_bits,
        security_bits=security_bits,
        brute_force_ops=brute_force_ops,
        time_at_rate=time_seconds,
        quantum_security_bits=quantum_bits,
        explanation=explanation,
    )


def birthday_bound(
    output_bits: int,
    trials: Optional[int] = None,
) -> BirthdayBoundReport:
    """Calculate birthday bound collision probability.

    For a hash/random function with output_bits output space,
    calculates the number of trials needed for 50% collision
    probability, and optionally the probability at a given number of trials.

    Args:
        output_bits: Size of output space in bits
        trials: Optional number of trials to evaluate

    Returns:
        BirthdayBoundReport with collision probabilities.
    """
    if output_bits <= 0:
        raise ValueError("output_bits must be positive")

    # Trials for 50% collision: sqrt(2 * N * ln(2)) ≈ 1.177 * sqrt(N)
    # where N = 2^output_bits
    N = 2.0 ** output_bits
    trials_50 = math.sqrt(2 * N * math.log(2))

    prob_at_trials = None
    if trials is not None:
        if trials <= 0:
            raise ValueError("trials must be positive")
        # Approximation: P ≈ 1 - exp(-trials^2 / (2*N))
        exponent = -(trials ** 2) / (2 * N)
        if exponent < -700:  # prevent underflow
            prob_at_trials = 1.0
        else:
            prob_at_trials = 1 - math.exp(exponent)

    return BirthdayBoundReport(
        output_bits=output_bits,
        collision_probability=0.5,
        trials_for_50pct=trials_50,
        trials_given=trials,
        probability_at_trials=prob_at_trials,
    )


def rsa_key_analysis(modulus_bits: int) -> RSAKeyReport:
    """Analyze RSA key strength using GNFS complexity estimate.

    Uses the General Number Field Sieve complexity:
    L_n = exp(1.923 * (ln N)^(1/3) * (ln ln N)^(2/3))

    Args:
        modulus_bits: RSA modulus size in bits

    Returns:
        RSAKeyReport with security analysis.
    """
    if modulus_bits < 512:
        raise ValueError("RSA modulus must be at least 512 bits")

    security_bits = _rsa_security_bits(modulus_bits)

    # GNFS complexity (number of operations)
    ln_N = modulus_bits * math.log(2)
    ln_ln_N = math.log(ln_N)
    gnfs_exponent = 1.923 * (ln_N ** (1 / 3)) * (ln_ln_N ** (2 / 3))
    gnfs_ops = math.exp(gnfs_exponent)

    recommended = modulus_bits >= 2048
    if modulus_bits >= 4096:
        rec = f"RSA-{modulus_bits}: strong for classical, but migrate to post-quantum."
    elif modulus_bits >= 2048:
        rec = f"RSA-{modulus_bits}: acceptable through ~2030. Plan PQ migration."
    elif modulus_bits >= 1024:
        rec = f"RSA-{modulus_bits}: below NIST minimum. Upgrade to 2048+ immediately."
    else:
        rec = f"RSA-{modulus_bits}: critically weak. Factored in practice."

    return RSAKeyReport(
        modulus_bits=modulus_bits,
        security_bits=security_bits,
        recommended=recommended,
        gnfs_ops=gnfs_ops,
        quantum_vulnerable=True,
        recommendation=rec,
    )


def cipher_mode_analysis(mode: str, block_bits: int = 128) -> CipherModeReport:
    """Analyze properties of a block cipher mode of operation.

    Derives security properties, parallelizability, and birthday bound
    limits from the mode specification.

    Args:
        mode: Mode name (ECB, CBC, CTR, GCM, CCM, OFB, CFB)
        block_bits: Block size in bits (default 128 for AES)

    Returns:
        CipherModeReport with derived properties.
    """
    mode = mode.upper()

    # Birthday bound: trouble after 2^(block_bits/2) blocks
    max_safe = 2.0 ** (block_bits / 2)

    modes = {
        "ECB": CipherModeReport(
            mode="ECB",
            requires_iv=False,
            iv_must_be_random=False,
            parallelizable_encrypt=True,
            parallelizable_decrypt=True,
            authenticated=False,
            ciphertext_expansion="none",
            max_safe_blocks=None,
            properties=[
                "INSECURE: identical plaintext blocks produce identical ciphertext",
                "Leaks plaintext patterns (e.g., ECB penguin)",
                "Never use for data longer than one block",
            ],
        ),
        "CBC": CipherModeReport(
            mode="CBC",
            requires_iv=True,
            iv_must_be_random=True,
            parallelizable_encrypt=False,
            parallelizable_decrypt=True,
            authenticated=False,
            ciphertext_expansion="+1 block (IV)",
            max_safe_blocks=max_safe,
            properties=[
                "IV must be unpredictable (random), not just unique",
                "Predictable IV enables BEAST-style attacks",
                "Padding oracle attacks possible without MAC",
                "Decrypt is parallelizable, encrypt is not",
            ],
        ),
        "CTR": CipherModeReport(
            mode="CTR",
            requires_iv=True,
            iv_must_be_random=False,  # nonce, must be unique but not random
            parallelizable_encrypt=True,
            parallelizable_decrypt=True,
            authenticated=False,
            ciphertext_expansion="+nonce",
            max_safe_blocks=max_safe,
            properties=[
                "Nonce must be UNIQUE per key, but need not be random",
                "Nonce reuse is catastrophic (XOR of plaintexts leaked)",
                "Fully parallelizable in both directions",
                "Acts as a stream cipher",
            ],
        ),
        "GCM": CipherModeReport(
            mode="GCM",
            requires_iv=True,
            iv_must_be_random=False,
            parallelizable_encrypt=True,
            parallelizable_decrypt=True,
            authenticated=True,
            ciphertext_expansion="+nonce +tag (16 bytes)",
            max_safe_blocks=max_safe,
            properties=[
                "Authenticated encryption with associated data (AEAD)",
                "Nonce MUST be unique per key; reuse breaks authentication AND confidentiality",
                "96-bit nonce recommended; 12-byte nonce is most efficient",
                "Auth tag forgery probability: 2^(-tag_bits)",
                "Limit: ~2^32 blocks per nonce (64 GB) before birthday risk",
            ],
        ),
        "CCM": CipherModeReport(
            mode="CCM",
            requires_iv=True,
            iv_must_be_random=False,
            parallelizable_encrypt=False,
            parallelizable_decrypt=False,
            authenticated=True,
            ciphertext_expansion="+nonce +tag",
            max_safe_blocks=max_safe,
            properties=[
                "Authenticated encryption (AEAD)",
                "Two-pass: CBC-MAC then CTR encryption (not parallelizable)",
                "Nonce must be unique per key",
                "Used in IEEE 802.11i (WiFi WPA2)",
            ],
        ),
        "OFB": CipherModeReport(
            mode="OFB",
            requires_iv=True,
            iv_must_be_random=True,
            parallelizable_encrypt=False,
            parallelizable_decrypt=False,
            authenticated=False,
            ciphertext_expansion="+IV",
            max_safe_blocks=max_safe,
            properties=[
                "Output feedback: keystream independent of plaintext",
                "Bit-flipping attacks possible (malleable)",
                "IV must be unique; reuse leaks XOR of plaintexts",
                "Neither encrypt nor decrypt is parallelizable",
            ],
        ),
        "CFB": CipherModeReport(
            mode="CFB",
            requires_iv=True,
            iv_must_be_random=True,
            parallelizable_encrypt=False,
            parallelizable_decrypt=True,
            authenticated=False,
            ciphertext_expansion="+IV",
            max_safe_blocks=max_safe,
            properties=[
                "Cipher feedback: self-synchronizing stream mode",
                "Decrypt is parallelizable, encrypt is not",
                "Error propagation: one corrupted block affects next block too",
            ],
        ),
    }

    if mode not in modes:
        raise ValueError(f"Unknown mode: {mode}. Known: {', '.join(sorted(modes))}")

    return modes[mode]


def _rsa_security_bits(modulus_bits: int) -> int:
    """Approximate RSA security level in symmetric-equivalent bits (NIST SP 800-57)."""
    if modulus_bits >= 15360:
        return 256
    elif modulus_bits >= 7680:
        return 192
    elif modulus_bits >= 3072:
        return 128
    elif modulus_bits >= 2048:
        return 112
    elif modulus_bits >= 1024:
        return 80
    else:
        return max(40, modulus_bits // 20)
