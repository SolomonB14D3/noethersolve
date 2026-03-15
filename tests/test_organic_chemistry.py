"""Tests for the organic chemistry lookup-table module."""

import pytest

from noethersolve.organic_chemistry import (
    check_organic_chemistry,
    get_organic_chemistry_topic,
    list_organic_chemistry_topics,
    OrganicChemistryReport,
    OrganicChemistryIssue,
    OrganicChemistryTopic,
)


# ── list_organic_chemistry_topics ──────────────────────────────────────


class TestListTopics:
    """Tests for list_organic_chemistry_topics()."""

    def test_all_topics_returns_12(self):
        topics = list_organic_chemistry_topics()
        assert len(topics) == 12

    def test_returns_strings(self):
        topics = list_organic_chemistry_topics()
        assert all(isinstance(t, str) for t in topics)

    def test_filter_by_mechanisms_cluster(self):
        topics = list_organic_chemistry_topics(cluster="mechanisms")
        assert len(topics) == 3

    def test_filter_by_reagents_cluster(self):
        topics = list_organic_chemistry_topics(cluster="reagents")
        assert len(topics) == 2

    def test_filter_by_reactions_cluster(self):
        topics = list_organic_chemistry_topics(cluster="reactions")
        assert len(topics) == 3

    def test_filter_by_stereochem_cluster(self):
        topics = list_organic_chemistry_topics(cluster="stereochem")
        assert len(topics) == 2

    def test_filter_by_synthesis_cluster(self):
        topics = list_organic_chemistry_topics(cluster="synthesis")
        assert len(topics) == 2

    def test_unknown_cluster_returns_empty(self):
        topics = list_organic_chemistry_topics(cluster="nonexistent")
        assert len(topics) == 0


# ── get_organic_chemistry_topic ────────────────────────────────────────


class TestGetTopic:
    """Tests for get_organic_chemistry_topic()."""

    def test_exact_id_lookup(self):
        topic = get_organic_chemistry_topic("oc01_e1")
        assert topic is not None
        assert topic.id == "oc01_e1"
        assert topic.cluster == "mechanisms"

    def test_name_lookup(self):
        topic = get_organic_chemistry_topic("grignard")
        assert topic is not None
        assert topic.id == "oc03_grignard"

    def test_case_insensitive(self):
        topic = get_organic_chemistry_topic("diels-alder")
        assert topic is not None
        assert topic.id == "oc06_diels_alder"

    def test_partial_match(self):
        topic = get_organic_chemistry_topic("friedel")
        assert topic is not None
        assert topic.id == "oc04_friedel"

    def test_unknown_returns_none(self):
        topic = get_organic_chemistry_topic("nonexistent_topic_xyz_123")
        assert topic is None

    def test_topic_has_key_facts(self):
        topic = get_organic_chemistry_topic("oc05_aldol")
        assert topic is not None
        assert len(topic.key_facts) >= 2

    def test_topic_has_common_errors(self):
        topic = get_organic_chemistry_topic("oc07_chirality")
        assert topic is not None
        assert len(topic.common_errors) >= 2

    def test_topic_str_output(self):
        topic = get_organic_chemistry_topic("oc01_e1")
        s = str(topic)
        assert "e1" in s.lower() or "elimination" in s.lower()


# ── check_organic_chemistry ────────────────────────────────────────────


class TestCheckOrganicChemistry:
    """Tests for check_organic_chemistry()."""

    def test_correct_claim_passes(self):
        report = check_organic_chemistry(
            "e2 elimination",
            claim="requires anti-periplanar geometry",
        )
        assert isinstance(report, OrganicChemistryReport)
        assert report.verdict == "PASS"

    def test_no_claim_returns_info(self):
        report = check_organic_chemistry("chirality")
        assert isinstance(report, OrganicChemistryReport)
        assert report.verdict in ("PASS", "INFO")

    def test_unknown_topic_fails(self):
        report = check_organic_chemistry("nonexistent_xyz_123")
        assert report.verdict == "FAIL"

    def test_report_str_output(self):
        report = check_organic_chemistry("grignard")
        s = str(report)
        assert len(s) > 0

    def test_report_has_issues_list(self):
        report = check_organic_chemistry("aldol")
        assert hasattr(report, "issues")
        assert isinstance(report.issues, list)

    # ── Content correctness tests ────────────────────────────────────

    def test_e1_carbocation(self):
        topic = get_organic_chemistry_topic("oc01_e1")
        facts_text = " ".join(topic.key_facts).lower()
        assert "carbocation" in facts_text

    def test_e2_anti_periplanar(self):
        topic = get_organic_chemistry_topic("oc02_e2")
        facts_text = " ".join(topic.key_facts).lower()
        assert "anti-periplanar" in facts_text or "anti periplanar" in facts_text or "180" in facts_text

    def test_grignard_water_reaction(self):
        topic = get_organic_chemistry_topic("oc03_grignard")
        facts_text = " ".join(topic.key_facts).lower()
        assert "water" in facts_text or "protic" in facts_text

    def test_friedel_crafts_rearrangement(self):
        topic = get_organic_chemistry_topic("oc04_friedel")
        facts_text = " ".join(topic.key_facts).lower()
        assert "rearrangement" in facts_text or "rearrange" in facts_text

    def test_aldol_enone(self):
        topic = get_organic_chemistry_topic("oc05_aldol")
        facts_text = " ".join(topic.key_facts).lower()
        assert "unsaturated" in facts_text or "enone" in facts_text or "dehydration" in facts_text

    def test_diels_alder_concerted(self):
        topic = get_organic_chemistry_topic("oc06_diels_alder")
        facts_text = " ".join(topic.key_facts).lower()
        assert "concerted" in facts_text or "[4+2]" in facts_text

    def test_chirality_mirror_image(self):
        topic = get_organic_chemistry_topic("oc07_chirality")
        facts_text = " ".join(topic.key_facts).lower()
        assert "mirror" in facts_text or "non-superimposable" in facts_text or "superimposable" in facts_text

    def test_r_s_cip(self):
        topic = get_organic_chemistry_topic("oc08_r_s")
        facts_text = " ".join(topic.key_facts).lower()
        assert "cahn" in facts_text or "priority" in facts_text or "cip" in facts_text

    def test_protecting_groups_temporary(self):
        topic = get_organic_chemistry_topic("oc09_protecting")
        facts_text = " ".join(topic.key_facts).lower()
        assert "temporary" in facts_text or "temporarily" in facts_text or "mask" in facts_text or "selectively" in facts_text

    def test_retrosynthesis_backward(self):
        topic = get_organic_chemistry_topic("oc10_retro")
        facts_text = " ".join(topic.key_facts).lower()
        assert "backward" in facts_text or "disconnect" in facts_text or "target" in facts_text

    def test_nucleophilicity_charge(self):
        topic = get_organic_chemistry_topic("oc11_nucleophile")
        facts_text = " ".join(topic.key_facts).lower()
        assert "charge" in facts_text or "polariz" in facts_text

    def test_aromaticity_huckel(self):
        topic = get_organic_chemistry_topic("oc12_aromatic")
        facts_text = " ".join(topic.key_facts).lower()
        assert "4n+2" in facts_text or "huckel" in facts_text or "hückel" in facts_text


# ── Edge cases ─────────────────────────────────────────────────────────


class TestEdgeCases:
    """Edge case and robustness tests."""

    def test_empty_string_topic(self):
        report = check_organic_chemistry("")
        assert report.verdict == "FAIL"

    def test_all_topics_have_required_fields(self):
        for name in list_organic_chemistry_topics():
            topic = get_organic_chemistry_topic(name)
            assert topic is not None, f"Could not look up {name}"
            assert topic.id
            assert topic.name
            assert topic.cluster
            assert topic.description
            assert len(topic.key_facts) >= 2
            assert len(topic.common_errors) >= 2

    def test_clusters_are_valid(self):
        valid = {"mechanisms", "reagents", "reactions", "stereochem", "synthesis"}
        for name in list_organic_chemistry_topics():
            topic = get_organic_chemistry_topic(name)
            assert topic.cluster in valid, f"{topic.id} has invalid cluster {topic.cluster}"

    def test_ids_are_unique(self):
        topics = list_organic_chemistry_topics()
        # Even if the list returns names, they should be unique
        assert len(topics) == len(set(topics))
