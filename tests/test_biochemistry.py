"""Tests for the biochemistry lookup-table module."""

import pytest

from noethersolve.biochemistry import (
    check_biochemistry,
    get_biochemistry_topic,
    list_biochemistry_topics,
    BiochemistryReport,
    BiochemistryIssue,
    BiochemistryInfo,
)


# ── list_biochemistry_topics ───────────────────────────────────────────


class TestListTopics:
    """Tests for list_biochemistry_topics()."""

    def test_all_topics_returns_12(self):
        topics = list_biochemistry_topics()
        assert len(topics) == 12

    def test_returns_strings(self):
        topics = list_biochemistry_topics()
        assert all(isinstance(t, str) for t in topics)

    def test_filter_by_enzymes_cluster(self):
        topics = list_biochemistry_topics(cluster="enzymes")
        assert len(topics) == 3

    def test_filter_by_metabolism_cluster(self):
        topics = list_biochemistry_topics(cluster="metabolism")
        assert len(topics) == 3

    def test_filter_by_molbio_cluster(self):
        topics = list_biochemistry_topics(cluster="molbio")
        assert len(topics) == 2

    def test_filter_by_proteins_cluster(self):
        topics = list_biochemistry_topics(cluster="proteins")
        assert len(topics) == 2

    def test_filter_by_signaling_cluster(self):
        topics = list_biochemistry_topics(cluster="signaling")
        assert len(topics) == 2

    def test_unknown_cluster_returns_empty(self):
        topics = list_biochemistry_topics(cluster="nonexistent")
        assert len(topics) == 0


# ── get_biochemistry_topic ─────────────────────────────────────────────


class TestGetTopic:
    """Tests for get_biochemistry_topic()."""

    def test_exact_id_lookup(self):
        topic = get_biochemistry_topic("bc01_michaelis")
        assert topic is not None
        assert topic.id == "bc01_michaelis"
        assert topic.cluster == "enzymes"

    def test_name_lookup(self):
        topic = get_biochemistry_topic("michaelis")
        assert topic is not None
        assert topic.id == "bc01_michaelis"

    def test_case_insensitive(self):
        topic = get_biochemistry_topic("HEMOGLOBIN")
        assert topic is not None
        assert topic.id == "bc09_hemoglobin"

    def test_partial_match(self):
        topic = get_biochemistry_topic("krebs")
        assert topic is not None
        assert topic.id == "bc04_krebs"

    def test_unknown_returns_none(self):
        topic = get_biochemistry_topic("nonexistent_topic_xyz_123")
        assert topic is None

    def test_topic_has_key_facts(self):
        topic = get_biochemistry_topic("bc05_oxphos")
        assert topic is not None
        assert len(topic.key_facts) >= 2

    def test_topic_has_common_errors(self):
        topic = get_biochemistry_topic("bc03_competitive")
        assert topic is not None
        assert len(topic.common_errors) >= 2

    def test_topic_str_output(self):
        topic = get_biochemistry_topic("bc01_michaelis")
        s = str(topic)
        assert "michaelis" in s.lower() or "Michaelis" in s


# ── check_biochemistry ─────────────────────────────────────────────────


class TestCheckBiochemistry:
    """Tests for check_biochemistry()."""

    def test_no_claim_passes(self):
        """Without a claim, should return a report about the topic."""
        report = check_biochemistry("glycolysis")
        assert isinstance(report, BiochemistryReport)
        assert report.verdict in ("PASS", "INFO", "WARN")

    def test_unknown_topic_fails(self):
        report = check_biochemistry("nonexistent_xyz_123")
        assert report.verdict == "FAIL"

    def test_report_str_output(self):
        report = check_biochemistry("hemoglobin")
        s = str(report)
        assert len(s) > 0

    def test_report_has_issues_list(self):
        report = check_biochemistry("krebs")
        assert hasattr(report, "issues")
        assert isinstance(report.issues, list)

    # ── Content correctness tests ────────────────────────────────────

    def test_michaelis_km_definition(self):
        topic = get_biochemistry_topic("bc01_michaelis")
        facts_text = " ".join(topic.key_facts).lower()
        assert "km" in facts_text or "substrate" in facts_text

    def test_allosteric_distinct_site(self):
        topic = get_biochemistry_topic("bc02_allosteric")
        facts_text = " ".join(topic.key_facts).lower()
        assert "distinct" in facts_text or "different" in facts_text or "separate" in facts_text or "not the active site" in facts_text

    def test_competitive_vmax_unchanged(self):
        topic = get_biochemistry_topic("bc03_competitive")
        facts_text = " ".join(topic.key_facts).lower()
        assert "vmax" in facts_text

    def test_krebs_yields(self):
        topic = get_biochemistry_topic("bc04_krebs")
        facts_text = " ".join(topic.key_facts).lower()
        assert "nadh" in facts_text

    def test_oxphos_proton_gradient(self):
        topic = get_biochemistry_topic("bc05_oxphos")
        facts_text = " ".join(topic.key_facts).lower()
        assert "proton" in facts_text or "chemiosmotic" in facts_text

    def test_glycolysis_net_2_atp(self):
        topic = get_biochemistry_topic("bc06_glycolysis")
        facts_text = " ".join(topic.key_facts).lower()
        assert "2" in facts_text and "atp" in facts_text

    def test_dna_pol_primer(self):
        topic = get_biochemistry_topic("bc07_dna_pol")
        facts_text = " ".join(topic.key_facts).lower()
        assert "primer" in facts_text

    def test_mrna_processing_cap_and_tail(self):
        topic = get_biochemistry_topic("bc08_transcription")
        facts_text = " ".join(topic.key_facts).lower()
        assert "cap" in facts_text or "poly" in facts_text

    def test_hemoglobin_sigmoidal(self):
        topic = get_biochemistry_topic("bc09_hemoglobin")
        facts_text = " ".join(topic.key_facts).lower()
        assert "sigmoidal" in facts_text or "cooperative" in facts_text

    def test_gpcr_g_proteins(self):
        topic = get_biochemistry_topic("bc10_gpcr")
        facts_text = " ".join(topic.key_facts).lower()
        assert "g protein" in facts_text or "heterotrimeric" in facts_text or "g-protein" in facts_text

    def test_kinase_phosphate_transfer(self):
        topic = get_biochemistry_topic("bc11_kinase")
        facts_text = " ".join(topic.key_facts).lower()
        assert "phosphate" in facts_text or "phosphorylat" in facts_text

    def test_chaperone_aggregation(self):
        topic = get_biochemistry_topic("bc12_chaperone")
        facts_text = " ".join(topic.key_facts).lower()
        assert "aggregation" in facts_text or "folding" in facts_text


# ── Edge cases ─────────────────────────────────────────────────────────


class TestEdgeCases:
    """Edge case and robustness tests."""

    def test_empty_string_topic(self):
        report = check_biochemistry("")
        assert report.verdict == "FAIL"

    def test_all_topics_have_required_fields(self):
        for tid in list_biochemistry_topics():
            topic = get_biochemistry_topic(tid)
            assert topic is not None, f"Could not look up {tid}"
            assert topic.id
            assert topic.name
            assert topic.cluster
            assert topic.description
            assert len(topic.key_facts) >= 2
            assert len(topic.common_errors) >= 2

    def test_clusters_are_valid(self):
        valid = {"enzymes", "metabolism", "molbio", "proteins", "signaling"}
        for tid in list_biochemistry_topics():
            topic = get_biochemistry_topic(tid)
            assert topic.cluster in valid, f"{topic.id} has invalid cluster {topic.cluster}"

    def test_ids_are_unique(self):
        topics = list_biochemistry_topics()
        assert len(topics) == len(set(topics))
