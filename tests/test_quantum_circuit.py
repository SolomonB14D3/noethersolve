"""Tests for the quantum circuit simulator."""

import math
import pytest
from noethersolve.quantum_circuit import (
    simulate_circuit, measure_state,
)


class TestBasicGates:
    def test_identity(self):
        """I gate doesn't change state."""
        r = simulate_circuit(1, [("I", [0])])
        assert abs(r.state_vector[0] - 1.0) < 1e-10
        assert abs(r.state_vector[1]) < 1e-10

    def test_x_gate_flips(self):
        """|0⟩ → |1⟩."""
        r = simulate_circuit(1, [("X", [0])])
        assert abs(r.state_vector[0]) < 1e-10
        assert abs(r.state_vector[1] - 1.0) < 1e-10

    def test_x_gate_round_trip(self):
        """X twice = identity."""
        r = simulate_circuit(1, [("X", [0]), ("X", [0])])
        assert abs(r.state_vector[0] - 1.0) < 1e-10

    def test_hadamard_superposition(self):
        """H|0⟩ = (|0⟩ + |1⟩)/√2."""
        r = simulate_circuit(1, [("H", [0])])
        assert abs(r.probabilities["0"] - 0.5) < 1e-10
        assert abs(r.probabilities["1"] - 0.5) < 1e-10

    def test_hadamard_round_trip(self):
        """H twice = identity."""
        r = simulate_circuit(1, [("H", [0]), ("H", [0])])
        assert abs(r.state_vector[0] - 1.0) < 1e-10

    def test_z_gate_phase(self):
        """Z|1⟩ = -|1⟩."""
        r = simulate_circuit(1, [("X", [0]), ("Z", [0])])
        assert abs(r.state_vector[1] + 1.0) < 1e-10

    def test_y_gate(self):
        """Y|0⟩ = i|1⟩."""
        r = simulate_circuit(1, [("Y", [0])])
        assert abs(r.state_vector[0]) < 1e-10
        assert abs(r.state_vector[1] - 1j) < 1e-10


class TestBellState:
    def test_bell_state_probabilities(self):
        """H on q0, CNOT(0,1) → 50% |00⟩ + 50% |11⟩."""
        r = simulate_circuit(2, [("H", [0]), ("CNOT", [0, 1])])
        assert abs(r.probabilities["00"] - 0.5) < 1e-10
        assert abs(r.probabilities["11"] - 0.5) < 1e-10
        assert abs(r.probabilities["01"]) < 1e-10
        assert abs(r.probabilities["10"]) < 1e-10

    def test_bell_state_entangled(self):
        r = simulate_circuit(2, [("H", [0]), ("CNOT", [0, 1])])
        assert r.entangled is True
        assert abs(r.entropy - 1.0) < 0.01  # max entropy for 2 qubits

    def test_product_state_not_entangled(self):
        """H on q0 only — product state, not entangled."""
        r = simulate_circuit(2, [("H", [0])])
        assert r.entangled is False
        assert r.entropy < 0.01


class TestGHZState:
    def test_ghz_3_qubit(self):
        """GHZ: (|000⟩ + |111⟩)/√2."""
        r = simulate_circuit(3, [
            ("H", [0]), ("CNOT", [0, 1]), ("CNOT", [0, 2]),
        ])
        assert abs(r.probabilities["000"] - 0.5) < 1e-10
        assert abs(r.probabilities["111"] - 0.5) < 1e-10
        assert r.entangled

    def test_ghz_4_qubit(self):
        r = simulate_circuit(4, [
            ("H", [0]), ("CNOT", [0, 1]), ("CNOT", [0, 2]), ("CNOT", [0, 3]),
        ])
        assert abs(r.probabilities["0000"] - 0.5) < 1e-10
        assert abs(r.probabilities["1111"] - 0.5) < 1e-10


class TestMultiQubitGates:
    def test_cnot_control_0_target_1(self):
        """CNOT flips target when control is |1⟩."""
        # |10⟩ → |11⟩
        r = simulate_circuit(2, [("X", [0]), ("CNOT", [0, 1])])
        assert abs(r.probabilities["11"] - 1.0) < 1e-10

    def test_cnot_no_flip_when_control_0(self):
        """CNOT doesn't flip target when control is |0⟩."""
        # |00⟩ → |00⟩
        r = simulate_circuit(2, [("CNOT", [0, 1])])
        assert abs(r.probabilities["00"] - 1.0) < 1e-10

    def test_swap_gate(self):
        """SWAP exchanges two qubits."""
        # |10⟩ → |01⟩
        r = simulate_circuit(2, [("X", [0]), ("SWAP", [0, 1])])
        assert abs(r.probabilities["01"] - 1.0) < 1e-10

    def test_toffoli_gate(self):
        """Toffoli flips target only when both controls are |1⟩."""
        # |110⟩ → |111⟩
        r = simulate_circuit(3, [("X", [0]), ("X", [1]), ("TOFFOLI", [0, 1, 2])])
        assert abs(r.probabilities["111"] - 1.0) < 1e-10

    def test_toffoli_no_flip_one_control(self):
        """Toffoli doesn't flip when only one control is |1⟩."""
        # |100⟩ → |100⟩
        r = simulate_circuit(3, [("X", [0]), ("TOFFOLI", [0, 1, 2])])
        assert abs(r.probabilities["100"] - 1.0) < 1e-10

    def test_cz_gate(self):
        """CZ adds phase when both qubits are |1⟩."""
        # |11⟩ → -|11⟩
        r = simulate_circuit(2, [("X", [0]), ("X", [1]), ("CZ", [0, 1])])
        assert abs(r.state_vector[3] + 1.0) < 1e-10


class TestRotationGates:
    def test_rx_pi_equals_x(self):
        """RX(π) ≈ -iX (up to global phase, same probabilities as X)."""
        r = simulate_circuit(1, [("RX", [0], math.pi)])
        assert abs(r.probabilities["1"] - 1.0) < 1e-10

    def test_ry_pi_equals_y_probs(self):
        """RY(π)|0⟩ = |1⟩ (up to phase)."""
        r = simulate_circuit(1, [("RY", [0], math.pi)])
        assert abs(r.probabilities["1"] - 1.0) < 1e-10

    def test_rz_preserves_z_basis(self):
        """RZ on |0⟩ only changes phase, prob stays 1.0."""
        r = simulate_circuit(1, [("RZ", [0], 1.5)])
        assert abs(r.probabilities["0"] - 1.0) < 1e-10

    def test_rotation_requires_param(self):
        with pytest.raises(ValueError):
            simulate_circuit(1, [("RX", [0])])


class TestNormalization:
    def test_probabilities_sum_to_one(self):
        """Probabilities should sum to 1 for any circuit."""
        circuits = [
            (2, [("H", [0]), ("CNOT", [0, 1])]),
            (3, [("H", [0]), ("H", [1]), ("CNOT", [0, 2])]),
            (1, [("H", [0]), ("T", [0]), ("H", [0])]),
        ]
        for n, gates in circuits:
            r = simulate_circuit(n, gates)
            total = sum(r.probabilities.values())
            assert abs(total - 1.0) < 1e-8, f"Probs sum to {total}"


class TestMeasureState:
    def test_measure_bell_state(self):
        r = simulate_circuit(2, [("H", [0]), ("CNOT", [0, 1])])
        m = measure_state(r.state_vector, 2, 0)
        assert abs(m["0"] - 0.5) < 1e-10
        assert abs(m["1"] - 0.5) < 1e-10


class TestEdgeCases:
    def test_single_qubit(self):
        r = simulate_circuit(1, [("H", [0])])
        assert r.n_qubits == 1

    def test_too_many_qubits(self):
        with pytest.raises(ValueError):
            simulate_circuit(11, [])

    def test_unknown_gate(self):
        with pytest.raises(ValueError):
            simulate_circuit(1, [("FAKE_GATE", [0])])

    def test_report_string(self):
        r = simulate_circuit(2, [("H", [0]), ("CNOT", [0, 1])])
        s = str(r)
        assert "Quantum Circuit Simulation" in s
        assert "Entangled: YES" in s
        assert "prob=" in s

    def test_empty_circuit(self):
        r = simulate_circuit(2, [])
        assert abs(r.probabilities["00"] - 1.0) < 1e-10
