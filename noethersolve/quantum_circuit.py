"""Quantum circuit simulator — a CALCULATOR for small quantum systems.

Simulates quantum circuits by matrix multiplication on state vectors.
Computes measurement probabilities, entanglement, and expectation values.
Pure Python — no external quantum libraries required.

Usage:
    from noethersolve.quantum_circuit import simulate_circuit, measure_state

    # Bell state: H on qubit 0, CNOT(0,1)
    report = simulate_circuit(
        n_qubits=2,
        gates=[("H", [0]), ("CNOT", [0, 1])],
    )
    print(report)  # state vector, measurement probabilities, entanglement

    # GHZ state
    report = simulate_circuit(
        n_qubits=3,
        gates=[("H", [0]), ("CNOT", [0, 1]), ("CNOT", [0, 2])],
    )
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


# ── Gate definitions (2x2 and controlled) ────────────────────────────

# Each gate is a 2x2 complex matrix [[a,b],[c,d]]
# stored as list of list of complex

_I = [[1+0j, 0+0j], [0+0j, 1+0j]]
_X = [[0+0j, 1+0j], [1+0j, 0+0j]]
_Y = [[0+0j, -1j], [1j, 0+0j]]
_Z = [[1+0j, 0+0j], [0+0j, -1+0j]]
_H = [[1/math.sqrt(2)+0j, 1/math.sqrt(2)+0j],
      [1/math.sqrt(2)+0j, -1/math.sqrt(2)+0j]]
_S = [[1+0j, 0+0j], [0+0j, 1j]]
_T = [[1+0j, 0+0j], [0+0j, complex(math.cos(math.pi/4), math.sin(math.pi/4))]]

SINGLE_GATES: Dict[str, List[List[complex]]] = {
    "I": _I, "X": _X, "Y": _Y, "Z": _Z,
    "H": _H, "S": _S, "T": _T,
    "NOT": _X,  # alias
}


def _rx(theta: float) -> List[List[complex]]:
    """Rotation around X axis by angle theta."""
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    return [[complex(c, 0), complex(0, -s)],
            [complex(0, -s), complex(c, 0)]]


def _ry(theta: float) -> List[List[complex]]:
    """Rotation around Y axis by angle theta."""
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    return [[complex(c, 0), complex(-s, 0)],
            [complex(s, 0), complex(c, 0)]]


def _rz(theta: float) -> List[List[complex]]:
    """Rotation around Z axis by angle theta."""
    return [[complex(math.cos(theta/2), -math.sin(theta/2)), 0+0j],
            [0+0j, complex(math.cos(theta/2), math.sin(theta/2))]]


@dataclass
class CircuitReport:
    """Report from quantum circuit simulation."""

    n_qubits: int
    n_gates: int
    state_vector: List[complex]
    probabilities: Dict[str, float]
    entangled: bool
    entropy: float  # von Neumann entropy of qubit 0
    gates_applied: List[str]

    def __str__(self) -> str:
        len(self.state_vector)
        lines = [
            "=" * 60,
            f"  Quantum Circuit Simulation: {self.n_qubits} qubits, {self.n_gates} gates",
            "=" * 60,
            f"  Gates: {' → '.join(self.gates_applied)}",
            "",
            "  State vector (non-zero amplitudes):",
        ]

        for i, amp in enumerate(self.state_vector):
            if abs(amp) > 1e-10:
                basis = format(i, f'0{self.n_qubits}b')
                prob = abs(amp) ** 2
                phase = math.atan2(amp.imag, amp.real)
                lines.append(
                    f"    |{basis}⟩: {amp.real:+.4f}{amp.imag:+.4f}j "
                    f"(prob={prob:.4f}, phase={math.degrees(phase):.1f}°)"
                )

        lines.append("\n  Measurement probabilities:")
        # Sort by probability descending
        sorted_probs = sorted(self.probabilities.items(),
                              key=lambda x: x[1], reverse=True)
        for basis, prob in sorted_probs:
            if prob > 1e-10:
                bar = "█" * int(prob * 30)
                lines.append(f"    |{basis}⟩: {prob:.4f} {bar}")

        lines.append(f"\n  Entangled: {'YES' if self.entangled else 'NO'}")
        lines.append(f"  Von Neumann entropy (qubit 0): {self.entropy:.4f}")
        if self.entangled:
            lines.append("  (Entropy > 0 indicates entanglement between qubit 0 and the rest)")

        lines.append("=" * 60)
        return "\n".join(lines)


# ── Core simulation ──────────────────────────────────────────────────

def simulate_circuit(
    n_qubits: int,
    gates: List[Tuple],
    initial_state: Optional[List[complex]] = None,
) -> CircuitReport:
    """Simulate a quantum circuit.

    Parameters
    ----------
    n_qubits : int
        Number of qubits (max 10 for memory reasons).
    gates : list of tuples
        Each gate is (name, qubits) or (name, qubits, param).
        name: "H", "X", "Y", "Z", "S", "T", "CNOT", "CZ", "SWAP",
              "RX", "RY", "RZ", "TOFFOLI"
        qubits: list of qubit indices (0-indexed)
        param: optional float parameter for rotation gates
    initial_state : list of complex, optional
        Initial state vector. Default: |00...0⟩.
    """
    if n_qubits > 10:
        raise ValueError("Max 10 qubits (2^10 = 1024 amplitudes)")
    if n_qubits < 1:
        raise ValueError("Need at least 1 qubit")

    dim = 2 ** n_qubits

    # Initialize state
    if initial_state is not None:
        if len(initial_state) != dim:
            raise ValueError(f"Initial state must have {dim} amplitudes")
        state = [complex(a) for a in initial_state]
    else:
        state = [0+0j] * dim
        state[0] = 1+0j  # |00...0⟩

    gate_names = []

    for gate_spec in gates:
        if len(gate_spec) == 2:
            name, qubits = gate_spec
            param = None
        elif len(gate_spec) == 3:
            name, qubits, param = gate_spec
        else:
            raise ValueError("Gate spec must be (name, qubits) or (name, qubits, param)")

        name = name.upper()
        qubits = list(qubits)

        if name in ("CNOT", "CX"):
            if len(qubits) != 2:
                raise ValueError("CNOT requires exactly 2 qubits [control, target]")
            state = _apply_controlled(state, n_qubits, qubits[0], _X, qubits[1])
            gate_names.append(f"CNOT({qubits[0]},{qubits[1]})")

        elif name == "CZ":
            if len(qubits) != 2:
                raise ValueError("CZ requires exactly 2 qubits")
            state = _apply_controlled(state, n_qubits, qubits[0], _Z, qubits[1])
            gate_names.append(f"CZ({qubits[0]},{qubits[1]})")

        elif name == "SWAP":
            if len(qubits) != 2:
                raise ValueError("SWAP requires exactly 2 qubits")
            # SWAP = CNOT(a,b) CNOT(b,a) CNOT(a,b)
            state = _apply_controlled(state, n_qubits, qubits[0], _X, qubits[1])
            state = _apply_controlled(state, n_qubits, qubits[1], _X, qubits[0])
            state = _apply_controlled(state, n_qubits, qubits[0], _X, qubits[1])
            gate_names.append(f"SWAP({qubits[0]},{qubits[1]})")

        elif name in ("TOFFOLI", "CCX"):
            if len(qubits) != 3:
                raise ValueError("Toffoli requires exactly 3 qubits [c1, c2, target]")
            state = _apply_toffoli(state, n_qubits, qubits[0], qubits[1], qubits[2])
            gate_names.append(f"CCX({qubits[0]},{qubits[1]},{qubits[2]})")

        elif name in ("RX", "RY", "RZ"):
            if param is None:
                raise ValueError(f"{name} requires a parameter (angle)")
            if len(qubits) != 1:
                raise ValueError(f"{name} requires exactly 1 qubit")
            rot = {"RX": _rx, "RY": _ry, "RZ": _rz}[name]
            mat = rot(param)
            state = _apply_single(state, n_qubits, mat, qubits[0])
            gate_names.append(f"{name}({qubits[0]},{param:.2f})")

        elif name in SINGLE_GATES:
            if len(qubits) != 1:
                raise ValueError(f"{name} requires exactly 1 qubit")
            state = _apply_single(state, n_qubits, SINGLE_GATES[name], qubits[0])
            gate_names.append(f"{name}({qubits[0]})")

        else:
            raise ValueError(f"Unknown gate '{name}'")

    # Compute probabilities
    probs = {}
    for i in range(dim):
        basis = format(i, f'0{n_qubits}b')
        probs[basis] = abs(state[i]) ** 2

    # Check entanglement via von Neumann entropy of qubit 0
    entropy = _von_neumann_entropy_qubit0(state, n_qubits)
    entangled = entropy > 0.01

    return CircuitReport(
        n_qubits=n_qubits,
        n_gates=len(gates),
        state_vector=state,
        probabilities=probs,
        entangled=entangled,
        entropy=entropy,
        gates_applied=gate_names,
    )


def measure_state(
    state: List[complex],
    n_qubits: int,
    qubit: int,
) -> Dict[str, float]:
    """Compute measurement probabilities for a specific qubit.

    Returns dict with keys "0" and "1" mapping to probabilities.
    """
    dim = 2 ** n_qubits
    p0 = 0.0
    p1 = 0.0
    for i in range(dim):
        bit = (i >> (n_qubits - 1 - qubit)) & 1
        p = abs(state[i]) ** 2
        if bit == 0:
            p0 += p
        else:
            p1 += p
    return {"0": p0, "1": p1}


# ── Internal helpers ─────────────────────────────────────────────────

def _apply_single(
    state: List[complex],
    n_qubits: int,
    gate: List[List[complex]],
    target: int,
) -> List[complex]:
    """Apply a single-qubit gate to the state vector."""
    dim = 2 ** n_qubits
    new_state = list(state)  # copy
    bit = n_qubits - 1 - target  # bit position (MSB = qubit 0)

    # Process each pair (i0, i1) exactly once: iterate only over i where bit=0
    for i in range(dim):
        if i & (1 << bit):
            continue  # skip — this is the i1 of a pair, handled when i was i0

        i0 = i
        i1 = i | (1 << bit)

        a0 = state[i0]
        a1 = state[i1]

        new_state[i0] = gate[0][0] * a0 + gate[0][1] * a1
        new_state[i1] = gate[1][0] * a0 + gate[1][1] * a1

    return new_state


def _apply_controlled(
    state: List[complex],
    n_qubits: int,
    control: int,
    gate: List[List[complex]],
    target: int,
) -> List[complex]:
    """Apply a controlled single-qubit gate."""
    dim = 2 ** n_qubits
    new_state = list(state)  # copy
    ctrl_bit = n_qubits - 1 - control
    tgt_bit = n_qubits - 1 - target

    processed = set()
    for i in range(dim):
        if i in processed:
            continue
        # Only apply when control bit is 1
        if not (i & (1 << ctrl_bit)):
            continue

        i0 = i & ~(1 << tgt_bit)
        i1 = i | (1 << tgt_bit)

        if i0 in processed or i1 in processed:
            continue

        # Only process pairs where control is 1
        if not (i0 & (1 << ctrl_bit)) or not (i1 & (1 << ctrl_bit)):
            continue

        a0 = state[i0]
        a1 = state[i1]

        new_state[i0] = gate[0][0] * a0 + gate[0][1] * a1
        new_state[i1] = gate[1][0] * a0 + gate[1][1] * a1

        processed.add(i0)
        processed.add(i1)

    return new_state


def _apply_toffoli(
    state: List[complex],
    n_qubits: int,
    ctrl1: int,
    ctrl2: int,
    target: int,
) -> List[complex]:
    """Apply Toffoli (CCX) gate."""
    dim = 2 ** n_qubits
    new_state = list(state)
    c1_bit = n_qubits - 1 - ctrl1
    c2_bit = n_qubits - 1 - ctrl2
    tgt_bit = n_qubits - 1 - target

    for i in range(dim):
        # Only flip target when both controls are 1
        if (i & (1 << c1_bit)) and (i & (1 << c2_bit)):
            j = i ^ (1 << tgt_bit)  # flip target bit
            if i < j:  # only process each pair once
                new_state[i], new_state[j] = state[j], state[i]

    return new_state


def _von_neumann_entropy_qubit0(state: List[complex], n_qubits: int) -> float:
    """Compute von Neumann entropy of the reduced density matrix of qubit 0.

    Traces out all qubits except qubit 0 to get a 2x2 reduced density matrix,
    then computes S = -Tr(ρ log₂ ρ).
    """
    if n_qubits == 1:
        return 0.0

    dim = 2 ** n_qubits
    half = dim // 2

    # Reduced density matrix for qubit 0 (2x2)
    # ρ_00 = sum of |amp|^2 where qubit 0 = 0
    # ρ_11 = sum of |amp|^2 where qubit 0 = 1
    # ρ_01 = sum of amp_0k * conj(amp_1k) for matching k
    rho_00 = 0.0
    rho_11 = 0.0
    rho_01 = 0+0j

    q0_bit = n_qubits - 1  # MSB is qubit 0

    for k in range(half):
        # indices where qubit 0 = 0 and qubit 0 = 1, same other bits
        # Construct index with qubit 0 = 0
        # Actually, simpler: iterate over all states with q0=0
        pass

    # Cleaner approach: iterate all basis states
    for i in range(dim):
        q0_val = (i >> q0_bit) & 1
        if q0_val == 0:
            rho_00 += abs(state[i]) ** 2
            # Find matching state with q0=1
            j = i | (1 << q0_bit)
            rho_01 += state[i] * state[j].conjugate()
        else:
            rho_11 += abs(state[i]) ** 2

    # Eigenvalues of 2x2 density matrix [[rho_00, rho_01], [rho_01*, rho_11]]
    trace = rho_00 + rho_11
    det = rho_00 * rho_11 - abs(rho_01) ** 2
    disc = max(0.0, (trace / 2) ** 2 - det)
    sq = math.sqrt(disc)

    lambda1 = trace / 2 + sq
    lambda2 = trace / 2 - sq

    # Von Neumann entropy
    entropy = 0.0
    for lam in [lambda1, lambda2]:
        if lam > 1e-15:
            entropy -= lam * math.log2(lam)

    return entropy
