"""Tests for noethersolve.pipeline — therapeutic pipeline consistency validator."""

import pytest
from noethersolve.pipeline import (
    validate_pipeline,
    validate_pipeline_dict,
    TherapyDesign,
    PipelineReport,
    PipelineIssue,
)


# ─── Well-designed pipelines ──────────────────────────────────────────────────

class TestGoodDesigns:
    def test_liver_aav8_passes(self):
        design = TherapyDesign(
            modality="aav",
            target_tissue="liver",
            transgene_size_kb=4.0,
            vector_serotype="AAV8",
            promoter="TBG",
            route="iv",
            payload_type="gene_replacement",
        )
        report = validate_pipeline(design)
        # No HIGH issues expected (serotype, promoter, route all match liver)
        high = [i for i in report.issues if i.severity == "HIGH"]
        assert len(high) == 0

    def test_cns_aav9_passes(self):
        design = TherapyDesign(
            modality="aav",
            target_tissue="cns",
            transgene_size_kb=3.5,
            vector_serotype="AAV9",
            route="intrathecal",
            payload_type="gene_replacement",
        )
        report = validate_pipeline(design)
        high = [i for i in report.issues if i.severity == "HIGH"]
        assert len(high) == 0

    def test_eye_aav2_passes(self):
        design = TherapyDesign(
            modality="aav",
            target_tissue="eye",
            transgene_size_kb=2.8,
            vector_serotype="AAV2",
            route="subretinal",
            payload_type="gene_replacement",
        )
        report = validate_pipeline(design)
        high = [i for i in report.issues if i.severity == "HIGH"]
        assert len(high) == 0

    def test_lnp_mrna_liver_passes(self):
        design = TherapyDesign(
            modality="lnp_mrna",
            target_tissue="liver",
            route="iv",
            payload_type="gene_editing",
        )
        report = validate_pipeline(design)
        high = [i for i in report.issues if i.severity == "HIGH"]
        assert len(high) == 0


# ─── Vector capacity ─────────────────────────────────────────────────────────

class TestVectorCapacity:
    def test_oversized_transgene_high(self):
        design = TherapyDesign(
            modality="aav",
            target_tissue="liver",
            transgene_size_kb=5.2,
        )
        report = validate_pipeline(design)
        cap = [i for i in report.issues if i.rule_name == "VECTOR_CAPACITY"]
        assert any(i.severity == "HIGH" for i in cap)

    def test_near_limit_moderate(self):
        design = TherapyDesign(
            modality="aav",
            target_tissue="liver",
            transgene_size_kb=4.5,
        )
        report = validate_pipeline(design)
        cap = [i for i in report.issues if i.rule_name == "VECTOR_CAPACITY"]
        assert any(i.severity == "MODERATE" for i in cap)

    def test_small_transgene_clean(self):
        design = TherapyDesign(
            modality="aav",
            target_tissue="liver",
            transgene_size_kb=3.0,
        )
        report = validate_pipeline(design)
        cap = [i for i in report.issues if i.rule_name == "VECTOR_CAPACITY"]
        assert len(cap) == 0

    def test_lnp_no_capacity_check(self):
        design = TherapyDesign(
            modality="lnp_mrna",
            target_tissue="liver",
            transgene_size_kb=10.0,  # doesn't matter for LNP
        )
        report = validate_pipeline(design)
        cap = [i for i in report.issues if i.rule_name == "VECTOR_CAPACITY"]
        assert len(cap) == 0


# ─── Serotype-tissue match ────────────────────────────────────────────────────

class TestSerotypeTissue:
    def test_aav8_liver_clean(self):
        design = TherapyDesign(
            modality="aav",
            target_tissue="liver",
            vector_serotype="AAV8",
        )
        report = validate_pipeline(design)
        sero = [i for i in report.issues if i.rule_name == "SEROTYPE_TISSUE"]
        assert not any(i.severity == "HIGH" for i in sero)

    def test_aav2_cns_mismatch(self):
        design = TherapyDesign(
            modality="aav",
            target_tissue="cns",
            vector_serotype="AAV2",
        )
        report = validate_pipeline(design)
        sero = [i for i in report.issues if i.rule_name == "SEROTYPE_TISSUE"]
        assert any(i.severity == "HIGH" for i in sero)

    def test_unknown_serotype_low(self):
        design = TherapyDesign(
            modality="aav",
            target_tissue="liver",
            vector_serotype="AAV42",
        )
        report = validate_pipeline(design)
        sero = [i for i in report.issues if i.rule_name == "SEROTYPE_TISSUE"]
        assert any(i.severity == "LOW" for i in sero)

    def test_non_aav_skipped(self):
        design = TherapyDesign(
            modality="lnp_mrna",
            target_tissue="liver",
            vector_serotype="AAV8",  # irrelevant
        )
        report = validate_pipeline(design)
        sero = [i for i in report.issues if i.rule_name == "SEROTYPE_TISSUE"]
        assert len(sero) == 0


# ─── Promoter-tissue match ────────────────────────────────────────────────────

class TestPromoterTissue:
    def test_tbg_liver_clean(self):
        design = TherapyDesign(
            modality="aav",
            target_tissue="liver",
            promoter="TBG",
        )
        report = validate_pipeline(design)
        promo = [i for i in report.issues if i.rule_name == "PROMOTER_TISSUE"]
        assert not any(i.severity == "HIGH" for i in promo)

    def test_tbg_muscle_mismatch(self):
        design = TherapyDesign(
            modality="aav",
            target_tissue="muscle",
            promoter="TBG",
        )
        report = validate_pipeline(design)
        promo = [i for i in report.issues if i.rule_name == "PROMOTER_TISSUE"]
        assert any(i.severity == "HIGH" for i in promo)

    def test_cmv_moderate(self):
        design = TherapyDesign(
            modality="aav",
            target_tissue="liver",
            promoter="CMV",
        )
        report = validate_pipeline(design)
        promo = [i for i in report.issues if i.rule_name == "PROMOTER_TISSUE"]
        assert any(i.severity == "MODERATE" for i in promo)

    def test_cag_ubiquitous_moderate(self):
        design = TherapyDesign(
            modality="aav",
            target_tissue="cns",
            promoter="CAG",
        )
        report = validate_pipeline(design)
        promo = [i for i in report.issues if i.rule_name == "PROMOTER_TISSUE"]
        assert any(i.severity == "MODERATE" for i in promo)


# ─── Route-tissue match ──────────────────────────────────────────────────────

class TestRouteTissue:
    def test_iv_liver_clean(self):
        design = TherapyDesign(
            modality="aav",
            target_tissue="liver",
            route="iv",
        )
        report = validate_pipeline(design)
        route = [i for i in report.issues if i.rule_name == "ROUTE_TISSUE"]
        assert not any(i.severity == "HIGH" for i in route)

    def test_inhaled_cns_mismatch(self):
        design = TherapyDesign(
            modality="aav",
            target_tissue="cns",
            route="inhaled",
        )
        report = validate_pipeline(design)
        route = [i for i in report.issues if i.rule_name == "ROUTE_TISSUE"]
        assert any(i.severity == "HIGH" for i in route)

    def test_intrathecal_cns_clean(self):
        design = TherapyDesign(
            modality="aav",
            target_tissue="cns",
            route="intrathecal",
        )
        report = validate_pipeline(design)
        route = [i for i in report.issues if i.rule_name == "ROUTE_TISSUE"]
        assert not any(i.severity == "HIGH" for i in route)


# ─── Modality-payload compatibility ──────────────────────────────────────────

class TestModalityPayload:
    def test_aav_gene_replacement_clean(self):
        design = TherapyDesign(
            modality="aav",
            target_tissue="liver",
            payload_type="gene_replacement",
        )
        report = validate_pipeline(design)
        mp = [i for i in report.issues if i.rule_name == "MODALITY_PAYLOAD"]
        assert not any(i.severity == "HIGH" for i in mp)

    def test_sirna_gene_replacement_mismatch(self):
        design = TherapyDesign(
            modality="lnp_sirna",
            target_tissue="liver",
            payload_type="gene_replacement",
        )
        report = validate_pipeline(design)
        mp = [i for i in report.issues if i.rule_name == "MODALITY_PAYLOAD"]
        assert any(i.severity == "HIGH" for i in mp)

    def test_aso_silencing_clean(self):
        design = TherapyDesign(
            modality="aso",
            target_tissue="liver",
            payload_type="gene_silencing",
        )
        report = validate_pipeline(design)
        mp = [i for i in report.issues if i.rule_name == "MODALITY_PAYLOAD"]
        assert not any(i.severity == "HIGH" for i in mp)


# ─── Redosing immunity ───────────────────────────────────────────────────────

class TestRedosingImmunity:
    def test_aav_redosing_high(self):
        design = TherapyDesign(
            modality="aav",
            target_tissue="liver",
            redosing_planned=True,
        )
        report = validate_pipeline(design)
        redose = [i for i in report.issues if i.rule_name == "REDOSING_IMMUNITY"]
        assert any(i.severity == "HIGH" for i in redose)

    def test_lnp_redosing_clean(self):
        design = TherapyDesign(
            modality="lnp_mrna",
            target_tissue="liver",
            redosing_planned=True,
        )
        report = validate_pipeline(design)
        redose = [i for i in report.issues if i.rule_name == "REDOSING_IMMUNITY"]
        assert len(redose) == 0

    def test_no_redosing_clean(self):
        design = TherapyDesign(
            modality="aav",
            target_tissue="liver",
            redosing_planned=False,
        )
        report = validate_pipeline(design)
        redose = [i for i in report.issues if i.rule_name == "REDOSING_IMMUNITY"]
        assert len(redose) == 0


# ─── Safety monitoring ────────────────────────────────────────────────────────

class TestSafetyMonitoring:
    def test_aav_monitoring_includes_alt(self):
        design = TherapyDesign(modality="aav", target_tissue="liver")
        report = validate_pipeline(design)
        assert any("ALT" in m for m in report.required_monitoring)

    def test_lnp_monitoring_includes_cytokine(self):
        design = TherapyDesign(modality="lnp_mrna", target_tissue="liver")
        report = validate_pipeline(design)
        assert any("Cytokine" in m or "CRS" in m for m in report.required_monitoring)

    def test_gene_editing_requires_offtarget(self):
        design = TherapyDesign(
            modality="aav",
            target_tissue="liver",
            payload_type="gene_editing",
        )
        report = validate_pipeline(design)
        assert any("off-target" in m.lower() or "Off-target" in m for m in report.required_monitoring)

    def test_all_have_biodistribution(self):
        for modality in ["aav", "lnp_mrna", "lnp_sirna", "aso"]:
            design = TherapyDesign(modality=modality, target_tissue="liver")
            report = validate_pipeline(design)
            assert any("Biodistribution" in m for m in report.required_monitoring)

    def test_monitoring_is_info_only(self):
        design = TherapyDesign(modality="aav", target_tissue="liver")
        report = validate_pipeline(design)
        safety = [i for i in report.issues if i.rule_name == "SAFETY_MONITORING"]
        assert all(i.severity == "INFO" for i in safety)


# ─── Verdict logic ────────────────────────────────────────────────────────────

class TestVerdict:
    def test_fail_on_high(self):
        design = TherapyDesign(
            modality="aav",
            target_tissue="liver",
            transgene_size_kb=5.5,  # HIGH
        )
        report = validate_pipeline(design)
        assert report.verdict == "FAIL"
        assert not report.passed

    def test_warn_on_moderate(self):
        design = TherapyDesign(
            modality="aav",
            target_tissue="liver",
            transgene_size_kb=4.5,  # MODERATE
            vector_serotype="AAV8",
            promoter="TBG",
            route="iv",
            payload_type="gene_replacement",
        )
        report = validate_pipeline(design)
        assert report.verdict == "WARN"

    def test_pass_when_clean(self):
        design = TherapyDesign(
            modality="aav",
            target_tissue="liver",
            transgene_size_kb=3.0,
            vector_serotype="AAV8",
            promoter="TBG",
            route="iv",
            payload_type="gene_replacement",
        )
        report = validate_pipeline(design)
        assert report.verdict == "PASS"
        assert report.passed


# ─── Dict convenience ────────────────────────────────────────────────────────

class TestDictInterface:
    def test_basic_dict(self):
        report = validate_pipeline_dict({
            "modality": "aav",
            "target_tissue": "liver",
            "transgene_size_kb": 3.0,
        })
        assert isinstance(report, PipelineReport)

    def test_unknown_keys_ignored(self):
        report = validate_pipeline_dict({
            "modality": "aav",
            "target_tissue": "liver",
            "unknown_key": "should be ignored",
        })
        assert isinstance(report, PipelineReport)

    def test_missing_optional_uses_defaults(self):
        report = validate_pipeline_dict({
            "modality": "aav",
            "target_tissue": "liver",
        })
        assert report.design.transgene_size_kb == 0.0
        assert report.design.redosing_planned is False


# ─── Report formatting ───────────────────────────────────────────────────────

class TestReportFormat:
    def test_str_contains_verdict(self):
        design = TherapyDesign(modality="aav", target_tissue="liver")
        report = validate_pipeline(design)
        s = str(report)
        assert "PASS" in s or "WARN" in s or "FAIL" in s

    def test_str_contains_modality(self):
        design = TherapyDesign(modality="aav", target_tissue="liver")
        report = validate_pipeline(design)
        assert "aav" in str(report)

    def test_str_contains_monitoring(self):
        design = TherapyDesign(modality="aav", target_tissue="liver")
        report = validate_pipeline(design)
        assert "Required monitoring" in str(report)

    def test_suggestion_displayed(self):
        design = TherapyDesign(
            modality="aav",
            target_tissue="liver",
            transgene_size_kb=5.5,
        )
        report = validate_pipeline(design)
        s = str(report)
        assert "->" in s  # suggestion arrow


# ─── Edge cases ──────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_minimal_design(self):
        design = TherapyDesign(modality="aav", target_tissue="liver")
        report = validate_pipeline(design)
        assert isinstance(report, PipelineReport)

    def test_all_fields_populated(self):
        design = TherapyDesign(
            modality="aav",
            target_tissue="liver",
            transgene_size_kb=4.0,
            vector_serotype="AAV8",
            promoter="TBG",
            route="iv",
            payload_type="gene_replacement",
            redosing_planned=False,
        )
        report = validate_pipeline(design)
        assert isinstance(report, PipelineReport)

    def test_empty_strings_skip_checks(self):
        design = TherapyDesign(
            modality="aav",
            target_tissue="liver",
            vector_serotype="",
            promoter="",
            route="",
            payload_type="",
        )
        report = validate_pipeline(design)
        # Only safety monitoring issues (INFO)
        non_info = [i for i in report.issues if i.severity != "INFO"]
        assert len(non_info) == 0
