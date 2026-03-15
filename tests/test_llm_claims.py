"""Tests for the LLM Claims Auditor (noethersolve.llm_claims)."""

import pytest

from noethersolve.llm_claims import (
    LLMClaimIssue,
    LLMClaimReport,
    LLMClaimResult,
    LLMTopicInfo,
    audit_llm_claims,
    check_benchmark_score,
    check_llm_claim,
    chinchilla_optimal,
    get_llm_topic,
    list_domains,
    list_llm_topics,
)


# ── Chinchilla Scaling ───────────────────────────────────────────────────


class TestChinchillaOptimal:
    """Tests for the chinchilla_optimal() scaling calculator."""

    def test_params_only(self):
        result = chinchilla_optimal(params_B=7.0)
        assert result["params_B"] == 7.0
        assert result["tokens_B"] == 140.0
        assert result["is_chinchilla_optimal"] is True

    def test_tokens_only(self):
        result = chinchilla_optimal(tokens_B=140.0)
        assert result["params_B"] == 7.0
        assert result["tokens_B"] == 140.0
        assert result["is_chinchilla_optimal"] is True

    def test_compute_only(self):
        # 7B params, 140B tokens → C = 6 * 7e9 * 140e9 = 5.88e21
        result = chinchilla_optimal(compute_flops=5.88e21)
        assert abs(result["params_B"] - 7.0) < 0.5
        assert result["is_chinchilla_optimal"] is True

    def test_undertrained_detection(self):
        # 70B params with only 100B tokens → ratio 1.4× (should be 20×)
        result = chinchilla_optimal(params_B=70.0, tokens_B=100.0)
        assert result["is_chinchilla_optimal"] is False
        assert "Undertrained" in result["notes"]

    def test_overtrained_detection(self):
        # 1B params with 200B tokens → ratio 200× (way over 20×)
        result = chinchilla_optimal(params_B=1.0, tokens_B=200.0)
        assert result["is_chinchilla_optimal"] is False
        assert "Overtrained" in result["notes"]

    def test_approximately_optimal(self):
        # 7B with 140B tokens → ratio 20×
        result = chinchilla_optimal(params_B=7.0, tokens_B=140.0)
        assert result["is_chinchilla_optimal"] is True
        assert "optimal" in result["notes"].lower()

    def test_no_args_raises(self):
        with pytest.raises(ValueError):
            chinchilla_optimal()


# ── Benchmark Score Checking ─────────────────────────────────────────────


class TestBenchmarkScores:
    """Tests for check_benchmark_score()."""

    def test_valid_score_passes(self):
        result = check_benchmark_score("gpt-4", "mmlu", 87.0)
        assert result.verdict == "TRUE"

    def test_inflated_score_fails(self):
        result = check_benchmark_score("gpt-4", "mmlu", 99.0)
        assert result.verdict == "FALSE"
        assert "Above" in result.evidence

    def test_deflated_score_fails(self):
        result = check_benchmark_score("gpt-4", "mmlu", 50.0)
        assert result.verdict == "FALSE"
        assert "Below" in result.evidence

    def test_unknown_benchmark(self):
        result = check_benchmark_score("gpt-4", "nonexistent_bench", 80.0)
        assert result.verdict == "UNKNOWN"

    def test_unknown_model(self):
        result = check_benchmark_score("mystery-model", "mmlu", 80.0)
        assert result.verdict == "UNKNOWN"

    def test_case_insensitive_benchmark(self):
        result = check_benchmark_score("gpt-4", "MMLU", 87.0)
        assert result.verdict == "TRUE"

    def test_model_alias_gpt4(self):
        result = check_benchmark_score("gpt4", "mmlu", 87.0)
        assert result.verdict == "TRUE"

    def test_random_baseline(self):
        result = check_benchmark_score("random-baseline", "mmlu", 25.0)
        assert result.verdict == "TRUE"


# ── Single Claim Checking ────────────────────────────────────────────────


class TestCheckLLMClaim:
    """Tests for check_llm_claim() against individual claims."""

    # Known FALSE claims (misconceptions)
    def test_rlhf_eliminates_sycophancy_false(self):
        result = check_llm_claim("RLHF eliminates sycophancy")
        assert result.verdict == "FALSE"
        assert result.domain == "alignment"

    def test_scaling_eliminates_hallucination_false(self):
        result = check_llm_claim("scaling eliminates hallucination")
        assert result.verdict == "FALSE"

    def test_rag_guarantees_accuracy_false(self):
        result = check_llm_claim("RAG guarantees factual accuracy")
        assert result.verdict == "FALSE"

    def test_models_are_well_calibrated_false(self):
        result = check_llm_claim("models are well-calibrated by default")
        assert result.verdict == "FALSE"

    def test_cot_guarantees_correctness_false(self):
        result = check_llm_claim("chain-of-thought guarantees logical correctness")
        assert result.verdict == "FALSE"

    def test_jailbreaks_impossible_false(self):
        result = check_llm_claim("jailbreaks are impossible with RLHF")
        assert result.verdict == "FALSE"

    def test_sycophancy_solved_false(self):
        result = check_llm_claim("sycophancy is a solved problem")
        assert result.verdict == "FALSE"

    def test_knowledge_symmetric_false(self):
        result = check_llm_claim(
            "LLM knowledge is symmetric by default — learning A is B means B is A"
        )
        assert result.verdict == "FALSE"

    def test_catastrophic_forgetting_solved_false(self):
        result = check_llm_claim(
            "catastrophic forgetting is a solved problem with modern techniques"
        )
        assert result.verdict == "FALSE"

    def test_contamination_impossible_false(self):
        result = check_llm_claim("benchmark contamination is impossible to occur")
        assert result.verdict == "FALSE"

    # Known TRUE claims
    def test_scaling_laws_power_law_true(self):
        result = check_llm_claim(
            "scaling laws follow power-law relationships with compute and data"
        )
        assert result.verdict == "TRUE"

    def test_sycophancy_approval_true(self):
        result = check_llm_claim(
            "sycophancy means agreeing with users even when wrong, to maximize approval"
        )
        assert result.verdict == "TRUE"

    def test_lora_freezes_weights_true(self):
        result = check_llm_claim(
            "LoRA trains small additional parameters while freezing pretrained weights"
        )
        assert result.verdict == "TRUE"

    def test_kv_cache_linear_true(self):
        result = check_llm_claim(
            "KV cache memory grows linearly with sequence length"
        )
        assert result.verdict == "TRUE"

    def test_stateless_api_true(self):
        result = check_llm_claim(
            "LLMs are stateless between API calls with no persistent memory"
        )
        assert result.verdict == "TRUE"

    # DEBATED topics
    def test_emergence_is_nuanced(self):
        result = check_llm_claim(
            "emergent abilities appear suddenly at certain scales"
        )
        # Should be NUANCED since emergence is DEBATED
        assert result.verdict in ("TRUE", "NUANCED")

    # UNKNOWN claim
    def test_totally_unrelated_claim(self):
        result = check_llm_claim("bananas are yellow")
        assert result.verdict == "UNKNOWN"

    def test_result_has_correct_property(self):
        true_r = check_llm_claim(
            "scaling laws follow power-law relationships with compute"
        )
        false_r = check_llm_claim("RLHF eliminates sycophancy")
        assert true_r.correct is True
        assert false_r.correct is False


# ── Batch Audit ──────────────────────────────────────────────────────────


class TestAuditLLMClaims:
    """Tests for audit_llm_claims() batch auditing."""

    def test_all_true_passes(self):
        report = audit_llm_claims([
            "scaling laws follow power-law relationships with compute",
            "KV cache memory grows linearly with sequence length",
        ])
        assert report.verdict == "PASS"
        assert report.passed is True
        assert report.n_high == 0

    def test_false_claim_fails(self):
        report = audit_llm_claims([
            "RLHF eliminates sycophancy",
            "scaling eliminates hallucination",
        ])
        assert report.verdict == "FAIL"
        assert report.passed is False
        assert report.n_high >= 1

    def test_mixed_claims(self):
        report = audit_llm_claims([
            "scaling laws follow power-law relationships",  # TRUE
            "RLHF eliminates sycophancy",  # FALSE
        ])
        assert report.verdict == "FAIL"
        assert report.n_claims == 2

    def test_empty_claims(self):
        report = audit_llm_claims([])
        assert report.verdict == "PASS"
        assert report.n_claims == 0

    def test_unknown_claim_becomes_warning(self):
        report = audit_llm_claims(["bananas are yellow"])
        assert len(report.warnings) == 1
        assert report.verdict == "PASS"  # Unknown != fail

    def test_severity_counts_consistent(self):
        report = audit_llm_claims([
            "RLHF eliminates sycophancy",
            "scaling laws follow power-law relationships",
            "bananas are yellow",
        ])
        total = report.n_high + report.n_moderate + report.n_low + report.n_info
        assert total == report.n_issues

    def test_n_issues_matches_len(self):
        report = audit_llm_claims([
            "RLHF eliminates sycophancy",
            "RAG guarantees factual accuracy",
        ])
        assert report.n_issues == len(report.issues)


# ── Report Formatting ────────────────────────────────────────────────────


class TestReportFormatting:
    """Tests for __str__ formatting of reports and issues."""

    def test_report_str_contains_verdict(self):
        report = audit_llm_claims(["RLHF eliminates sycophancy"])
        s = str(report)
        assert "FAIL" in s
        assert "LLM Claims Audit" in s

    def test_report_str_contains_summary(self):
        report = audit_llm_claims([
            "RLHF eliminates sycophancy",
            "scaling laws follow power-law relationships",
        ])
        s = str(report)
        assert "2 checked" in s

    def test_issue_str_contains_severity(self):
        issue = LLMClaimIssue(
            severity="HIGH",
            claim="test",
            description="test description",
        )
        assert "[HIGH]" in str(issue)

    def test_issue_str_contains_references(self):
        issue = LLMClaimIssue(
            severity="HIGH",
            claim="test",
            description="test description",
            references=["Smith 2023"],
        )
        assert "Smith 2023" in str(issue)

    def test_pass_report_str(self):
        report = audit_llm_claims([
            "scaling laws follow power-law relationships with compute",
        ])
        s = str(report)
        assert "PASS" in s

    def test_empty_report_str(self):
        report = audit_llm_claims([])
        s = str(report)
        assert "PASS" in s
        assert "0 checked" in s


# ── Topic Lookup ─────────────────────────────────────────────────────────


class TestTopicLookup:
    """Tests for get_llm_topic() and list_llm_topics()."""

    def test_get_by_exact_id(self):
        info = get_llm_topic("sycophancy")
        assert info is not None
        assert info.domain == "alignment"

    def test_get_by_keyword(self):
        info = get_llm_topic("hallucination")
        assert info is not None
        assert info.domain == "hallucination"

    def test_get_nonexistent_returns_none(self):
        info = get_llm_topic("zxcvbnm_nonexistent")
        assert info is None

    def test_list_all_topics(self):
        topics = list_llm_topics()
        assert len(topics) >= 30  # We have ~35 topics

    def test_list_by_domain(self):
        topics = list_llm_topics(domain="hallucination")
        assert len(topics) >= 4
        # All should be hallucination domain
        for tid in topics:
            info = get_llm_topic(tid)
            assert info.domain == "hallucination"

    def test_list_domains(self):
        domains = list_domains()
        assert "hallucination" in domains
        assert "reasoning" in domains
        assert "alignment" in domains
        assert "training" in domains
        assert "evaluation" in domains
        assert "context_memory" in domains

    def test_topic_has_required_fields(self):
        info = get_llm_topic("sycophancy")
        assert info.topic_id != ""
        assert info.domain != ""
        assert info.status in ("ESTABLISHED", "OPEN", "DEBATED", "PARTIAL")
        assert info.description != ""
        assert info.truth != ""
        assert len(info.misconceptions) > 0
        assert len(info.keywords) > 0


# ── Domain Coverage ──────────────────────────────────────────────────────


class TestDomainCoverage:
    """Tests that all 6 LLM domains have adequate coverage."""

    def test_hallucination_coverage(self):
        topics = list_llm_topics(domain="hallucination")
        assert len(topics) >= 5

    def test_reasoning_coverage(self):
        topics = list_llm_topics(domain="reasoning")
        assert len(topics) >= 5

    def test_alignment_coverage(self):
        topics = list_llm_topics(domain="alignment")
        assert len(topics) >= 5

    def test_training_coverage(self):
        topics = list_llm_topics(domain="training")
        assert len(topics) >= 5

    def test_evaluation_coverage(self):
        topics = list_llm_topics(domain="evaluation")
        assert len(topics) >= 3

    def test_context_memory_coverage(self):
        topics = list_llm_topics(domain="context_memory")
        assert len(topics) >= 4


# ── Edge Cases ───────────────────────────────────────────────────────────


class TestEdgeCases:
    """Edge cases and error handling."""

    def test_empty_string_claim(self):
        result = check_llm_claim("")
        assert result.verdict == "UNKNOWN"

    def test_very_long_claim(self):
        result = check_llm_claim("hallucination " * 100)
        assert result.verdict != "UNKNOWN"

    def test_special_characters(self):
        result = check_llm_claim("RLHF < 50% effective!!!???")
        # Should still match RLHF
        assert result.verdict != "UNKNOWN"

    def test_case_insensitive_claim(self):
        r1 = check_llm_claim("RLHF eliminates sycophancy")
        r2 = check_llm_claim("rlhf eliminates sycophancy")
        assert r1.verdict == r2.verdict

    def test_chinchilla_zero_params(self):
        result = chinchilla_optimal(params_B=0.0, tokens_B=100.0)
        # Should handle gracefully
        assert "ratio" in result

    def test_audit_single_claim(self):
        report = audit_llm_claims(["RLHF eliminates sycophancy"])
        assert report.n_claims == 1
        assert report.n_high >= 1
