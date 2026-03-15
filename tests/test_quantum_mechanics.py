"""Tests for the quantum mechanics lookup-table module."""

import pytest

from noethersolve.quantum_mechanics import (
    check_quantum_mechanics,
    get_quantum_mechanics_topic,
    list_quantum_mechanics_topics,
    QuantumMechanicsReport,
    QuantumMechanicsIssue,
    QMTopicInfo,
)


# ── list_quantum_mechanics_topics ──────────────────────────────────────


class TestListTopics:
    """Tests for list_quantum_mechanics_topics()."""

    def test_all_topics_returns_12(self):
        topics = list_quantum_mechanics_topics()
        assert len(topics) == 12

    def test_returns_strings(self):
        topics = list_quantum_mechanics_topics()
        assert all(isinstance(t, str) for t in topics)

    def test_filter_by_foundations_cluster(self):
        topics = list_quantum_mechanics_topics(cluster="foundations")
        assert len(topics) == 5

    def test_filter_by_phenomena_cluster(self):
        topics = list_quantum_mechanics_topics(cluster="phenomena")
        assert len(topics) == 4

    def test_filter_by_systems_cluster(self):
        topics = list_quantum_mechanics_topics(cluster="systems")
        assert len(topics) == 3

    def test_unknown_cluster_raises_or_empty(self):
        try:
            topics = list_quantum_mechanics_topics(cluster="nonexistent")
            assert len(topics) == 0
        except ValueError:
            pass  # Also acceptable to raise ValueError


# ── get_quantum_mechanics_topic ────────────────────────────────────────


class TestGetTopic:
    """Tests for get_quantum_mechanics_topic()."""

    def test_exact_id_lookup(self):
        topic = get_quantum_mechanics_topic("qm01_uncertainty")
        assert topic is not None
        assert topic.id == "qm01_uncertainty"
        assert topic.cluster == "foundations"

    def test_name_lookup(self):
        topic = get_quantum_mechanics_topic("uncertainty")
        assert topic is not None
        assert topic.id == "qm01_uncertainty"

    def test_case_insensitive(self):
        topic = get_quantum_mechanics_topic("PAULI")
        assert topic is not None
        assert topic.id == "qm03_pauli"

    def test_partial_match(self):
        topic = get_quantum_mechanics_topic("tunneling")
        assert topic is not None
        assert topic.id == "qm04_tunneling"

    def test_unknown_returns_none(self):
        topic = get_quantum_mechanics_topic("nonexistent_topic_xyz_123")
        assert topic is None

    def test_topic_has_key_facts(self):
        topic = get_quantum_mechanics_topic("qm08_entanglement")
        assert topic is not None
        assert len(topic.key_facts) >= 2

    def test_topic_has_common_errors(self):
        topic = get_quantum_mechanics_topic("qm05_spin")
        assert topic is not None
        assert len(topic.common_errors) >= 2

    def test_topic_str_output(self):
        topic = get_quantum_mechanics_topic("qm01_uncertainty")
        s = str(topic)
        assert "uncertainty" in s.lower() or "heisenberg" in s.lower()


# ── check_quantum_mechanics ────────────────────────────────────────────


class TestCheckQuantumMechanics:
    """Tests for check_quantum_mechanics()."""

    def test_correct_claim_passes(self):
        report = check_quantum_mechanics(
            "uncertainty principle",
            claim="position and momentum cannot both be precisely determined",
        )
        assert isinstance(report, QuantumMechanicsReport)
        assert report.verdict == "PASS"

    def test_wrong_claim_fails(self):
        report = check_quantum_mechanics(
            "pauli exclusion",
            claim="applies to bosons",
        )
        assert report.verdict in ("WARN", "FAIL")
        assert len(report.issues) > 0

    def test_no_claim_returns_info(self):
        report = check_quantum_mechanics("entanglement")
        assert isinstance(report, QuantumMechanicsReport)
        assert report.verdict in ("PASS", "INFO")

    def test_unknown_topic_fails_or_warns(self):
        report = check_quantum_mechanics("nonexistent_xyz_123")
        assert report.verdict in ("FAIL", "WARN")

    def test_report_str_output(self):
        report = check_quantum_mechanics("tunneling")
        s = str(report)
        assert len(s) > 0

    def test_report_has_issues_list(self):
        report = check_quantum_mechanics("born rule")
        assert hasattr(report, "issues")
        assert isinstance(report.issues, list)

    # ── Content correctness tests ────────────────────────────────────

    def test_uncertainty_hbar(self):
        topic = get_quantum_mechanics_topic("qm01_uncertainty")
        facts_text = " ".join(topic.key_facts).lower()
        assert "position" in facts_text and "momentum" in facts_text

    def test_collapse_measurement(self):
        topic = get_quantum_mechanics_topic("qm02_collapse")
        facts_text = " ".join(topic.key_facts).lower()
        assert "measurement" in facts_text or "eigenstate" in facts_text

    def test_pauli_fermions(self):
        topic = get_quantum_mechanics_topic("qm03_pauli")
        facts_text = " ".join(topic.key_facts).lower()
        assert "fermion" in facts_text

    def test_tunneling_barrier(self):
        topic = get_quantum_mechanics_topic("qm04_tunneling")
        facts_text = " ".join(topic.key_facts).lower()
        assert "barrier" in facts_text

    def test_spin_720(self):
        topic = get_quantum_mechanics_topic("qm05_spin")
        facts_text = " ".join(topic.key_facts).lower()
        assert "720" in facts_text or "4π" in facts_text or "4pi" in facts_text

    def test_born_amplitude_squared(self):
        topic = get_quantum_mechanics_topic("qm06_born")
        facts_text = " ".join(topic.key_facts).lower()
        assert "squared" in facts_text or "|ψ|²" in facts_text or "amplitude" in facts_text

    def test_degeneracy_symmetry(self):
        topic = get_quantum_mechanics_topic("qm07_degeneracy")
        facts_text = " ".join(topic.key_facts).lower()
        assert "symmetry" in facts_text or "perturbation" in facts_text

    def test_entanglement_bell(self):
        topic = get_quantum_mechanics_topic("qm08_entanglement")
        facts_text = " ".join(topic.key_facts).lower()
        assert "bell" in facts_text or "local" in facts_text

    def test_superposition_simultaneous(self):
        topic = get_quantum_mechanics_topic("qm09_superposition")
        facts_text = " ".join(topic.key_facts).lower()
        assert "simultaneous" in facts_text or "linear" in facts_text or "superposition" in facts_text

    def test_harmonic_zero_point(self):
        topic = get_quantum_mechanics_topic("qm10_harmonic")
        facts_text = " ".join(topic.key_facts).lower()
        assert "zero-point" in facts_text or "nonzero" in facts_text or "1/2" in facts_text

    def test_hydrogen_n_squared(self):
        topic = get_quantum_mechanics_topic("qm11_hydrogen")
        facts_text = " ".join(topic.key_facts).lower()
        assert "13.6" in facts_text or "principal" in facts_text

    def test_commutator_incompatible(self):
        topic = get_quantum_mechanics_topic("qm12_commutator")
        facts_text = " ".join(topic.key_facts).lower()
        assert "incompatible" in facts_text or "uncertainty" in facts_text or "commut" in facts_text


# ── Edge cases ─────────────────────────────────────────────────────────


class TestEdgeCases:
    """Edge case and robustness tests."""

    def test_all_topics_have_required_fields(self):
        for tid in list_quantum_mechanics_topics():
            topic = get_quantum_mechanics_topic(tid)
            assert topic is not None, f"Could not look up {tid}"
            assert topic.id
            assert topic.name
            assert topic.cluster
            assert topic.description
            assert len(topic.key_facts) >= 2
            assert len(topic.common_errors) >= 2

    def test_clusters_are_valid(self):
        valid = {"foundations", "phenomena", "systems"}
        for tid in list_quantum_mechanics_topics():
            topic = get_quantum_mechanics_topic(tid)
            assert topic.cluster in valid, f"{topic.id} has invalid cluster {topic.cluster}"

    def test_ids_are_unique(self):
        topics = list_quantum_mechanics_topics()
        assert len(topics) == len(set(topics))
