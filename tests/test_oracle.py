"""Tests for noethersolve.oracle — Model-agnostic multiple-choice oracle.

Tests cover:
- get_completion_logprob scoring
- score_fact_mc multiple-choice evaluation
- Adapter integration with oracle
- Edge cases (empty completions, identical choices)
"""

import pytest
import numpy as np

# Skip all tests if MLX not available
mlx = pytest.importorskip("mlx.core", reason="MLX required for oracle tests")

from noethersolve.oracle import get_completion_logprob, score_fact_mc


# -----------------------------------------------------------------------------
# Mock Model and Tokenizer
# -----------------------------------------------------------------------------

class MockTokenizer:
    """Simple mock tokenizer for testing."""

    def __init__(self, vocab_size=100):
        self.vocab_size = vocab_size

    def encode(self, text: str) -> list:
        """Encode text to token IDs (1 token per character for simplicity)."""
        return [ord(c) % self.vocab_size for c in text]


class MockModel:
    """Mock model that returns predictable logits."""

    def __init__(self, vocab_size=100, favor_tokens=None):
        """
        Args:
            vocab_size: Vocabulary size
            favor_tokens: Dict of token_id -> boost. These tokens get higher logits.
        """
        self.vocab_size = vocab_size
        self.favor_tokens = favor_tokens or {}

    def __call__(self, tokens):
        """Return logits that favor certain tokens."""
        seq_len = tokens.shape[1]

        # Base logits: zeros
        mlx.zeros((1, seq_len, self.vocab_size))

        # Add boosts for favored tokens
        logits_np = np.zeros((1, seq_len, self.vocab_size), dtype=np.float32)
        for tok_id, boost in self.favor_tokens.items():
            logits_np[:, :, tok_id] = boost

        return mlx.array(logits_np)


class MockModelWithHiddenStates:
    """Mock model with .model attribute for adapter testing."""

    def __init__(self, vocab_size=100, d_model=64):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.model = self._inner_model

    def _inner_model(self, tokens):
        """Return hidden states."""
        seq_len = tokens.shape[1]
        return mlx.zeros((1, seq_len, self.d_model))

    def __call__(self, tokens):
        """Return logits."""
        seq_len = tokens.shape[1]
        return mlx.zeros((1, seq_len, self.vocab_size))


# -----------------------------------------------------------------------------
# get_completion_logprob Tests
# -----------------------------------------------------------------------------

class TestGetCompletionLogprob:
    """Tests for get_completion_logprob function."""

    def test_basic_scoring(self):
        """Basic log-prob scoring works."""
        tokenizer = MockTokenizer(vocab_size=128)
        model = MockModel(vocab_size=128)

        prompt = "Question:"
        completion = " Answer"

        lp = get_completion_logprob(model, tokenizer, prompt, completion)

        # Should return a float
        assert isinstance(lp, float)
        # Log-prob should be negative (probabilities < 1)
        assert lp < 0

    def test_empty_completion_returns_low_score(self):
        """Empty completion returns very low score."""
        tokenizer = MockTokenizer(vocab_size=128)
        model = MockModel(vocab_size=128)

        prompt = "Question:"
        completion = ""

        lp = get_completion_logprob(model, tokenizer, prompt, completion)

        # Should return sentinel value for empty completion
        assert lp == -1e9

    def test_favored_tokens_get_higher_score(self):
        """Completions with favored tokens get higher scores."""
        tokenizer = MockTokenizer(vocab_size=128)

        # Favor 'A' (ord 65) with high logit
        model_favor_a = MockModel(vocab_size=128, favor_tokens={65: 10.0})
        # Favor 'B' (ord 66) with high logit
        MockModel(vocab_size=128, favor_tokens={66: 10.0})

        prompt = "Q:"
        comp_a = " A"
        comp_b = " B"

        # Model favoring A should give higher score to A
        lp_a_model_a = get_completion_logprob(model_favor_a, tokenizer, prompt, comp_a)
        lp_b_model_a = get_completion_logprob(model_favor_a, tokenizer, prompt, comp_b)

        assert lp_a_model_a > lp_b_model_a

    def test_longer_completion_lower_score(self):
        """Longer completions generally have lower total log-prob (sum)."""
        tokenizer = MockTokenizer(vocab_size=128)
        model = MockModel(vocab_size=128)

        prompt = "Q:"
        short_completion = " A"
        long_completion = " AAAAAAAAAA"

        lp_short = get_completion_logprob(model, tokenizer, prompt, short_completion)
        lp_long = get_completion_logprob(model, tokenizer, prompt, long_completion)

        # Longer completion = more tokens = more log-probs summed (all negative)
        # So longer should have lower (more negative) score
        assert lp_long < lp_short


# -----------------------------------------------------------------------------
# score_fact_mc Tests
# -----------------------------------------------------------------------------

class TestScoreFactMC:
    """Tests for score_fact_mc function."""

    def test_basic_mc_scoring(self):
        """Basic multiple-choice scoring returns expected structure."""
        tokenizer = MockTokenizer(vocab_size=128)
        model = MockModel(vocab_size=128)

        context = "What is 2+2"
        truth = "4"
        distractors = ["3", "5", "6"]

        win, margin, truth_lp, best_dist_lp = score_fact_mc(
            model, tokenizer, context, truth, distractors
        )

        assert isinstance(win, bool)
        assert isinstance(margin, float)
        assert isinstance(truth_lp, float)
        assert isinstance(best_dist_lp, float)

    def test_truth_wins_when_favored(self):
        """Truth wins when model assigns it higher probability."""
        tokenizer = MockTokenizer(vocab_size=128)

        # Favor '4' (ord 52) heavily
        model = MockModel(vocab_size=128, favor_tokens={52: 20.0})

        context = "What is 2+2"
        truth = "4"
        distractors = ["3", "5", "6"]

        win, margin, truth_lp, best_dist_lp = score_fact_mc(
            model, tokenizer, context, truth, distractors
        )

        assert win is True
        assert margin > 0
        assert truth_lp > best_dist_lp

    def test_truth_loses_when_distractor_favored(self):
        """Truth loses when model prefers a distractor."""
        tokenizer = MockTokenizer(vocab_size=128)

        # Favor '5' (ord 53) heavily
        model = MockModel(vocab_size=128, favor_tokens={53: 20.0})

        context = "What is 2+2"
        truth = "4"
        distractors = ["3", "5", "6"]

        win, margin, truth_lp, best_dist_lp = score_fact_mc(
            model, tokenizer, context, truth, distractors
        )

        assert win is False
        assert margin < 0

    def test_margin_is_truth_minus_best_distractor(self):
        """Margin equals truth_lp minus best_dist_lp."""
        tokenizer = MockTokenizer(vocab_size=128)
        model = MockModel(vocab_size=128)

        context = "Q"
        truth = "T"
        distractors = ["A", "B"]

        win, margin, truth_lp, best_dist_lp = score_fact_mc(
            model, tokenizer, context, truth, distractors
        )

        assert abs(margin - (truth_lp - best_dist_lp)) < 1e-6

    def test_single_distractor(self):
        """Works with single distractor."""
        tokenizer = MockTokenizer(vocab_size=128)
        model = MockModel(vocab_size=128, favor_tokens={84: 10.0})  # Favor 'T'

        context = "Q"
        truth = "T"
        distractors = ["F"]

        win, margin, truth_lp, best_dist_lp = score_fact_mc(
            model, tokenizer, context, truth, distractors
        )

        assert win is True

    def test_many_distractors(self):
        """Works with many distractors."""
        tokenizer = MockTokenizer(vocab_size=128)
        model = MockModel(vocab_size=128)

        context = "Q"
        truth = "correct"
        distractors = ["wrong1", "wrong2", "wrong3", "wrong4", "wrong5"]

        win, margin, truth_lp, best_dist_lp = score_fact_mc(
            model, tokenizer, context, truth, distractors
        )

        # Should complete without error
        assert isinstance(win, bool)


# -----------------------------------------------------------------------------
# Adapter Integration Tests
# -----------------------------------------------------------------------------

class TestOracleWithAdapter:
    """Tests for oracle with adapter support."""

    def test_adapter_none_uses_base_model(self):
        """When adapter is None, uses base model directly."""
        tokenizer = MockTokenizer(vocab_size=128)
        model = MockModel(vocab_size=128, favor_tokens={65: 10.0})

        context = "Q"
        truth = "A"
        distractors = ["B"]

        win, margin, _, _ = score_fact_mc(
            model, tokenizer, context, truth, distractors,
            adapter=None, lm_head=None
        )

        assert win is True


# -----------------------------------------------------------------------------
# Edge Case Tests
# -----------------------------------------------------------------------------

class TestOracleEdgeCases:
    """Tests for edge cases in oracle."""

    def test_identical_truth_and_distractor(self):
        """Handles case where truth equals distractor."""
        tokenizer = MockTokenizer(vocab_size=128)
        model = MockModel(vocab_size=128)

        context = "Q"
        truth = "same"
        distractors = ["same"]  # Identical to truth

        win, margin, truth_lp, best_dist_lp = score_fact_mc(
            model, tokenizer, context, truth, distractors
        )

        # Should be a tie (margin = 0)
        assert abs(margin) < 1e-6
        # Truth doesn't strictly win (need > not >=)
        assert win is False

    def test_whitespace_handling(self):
        """Handles whitespace correctly."""
        tokenizer = MockTokenizer(vocab_size=128)
        model = MockModel(vocab_size=128)

        context = "Question"
        truth = "answer"  # No leading space (added by score_fact_mc)
        distractors = ["wrong"]

        # Should not raise
        win, margin, _, _ = score_fact_mc(
            model, tokenizer, context, truth, distractors
        )

        assert isinstance(win, bool)

    def test_special_characters_in_text(self):
        """Handles special characters."""
        tokenizer = MockTokenizer(vocab_size=128)
        model = MockModel(vocab_size=128)

        context = "What is π?"
        truth = "3.14159"
        distractors = ["2.718", "1.414"]

        win, margin, _, _ = score_fact_mc(
            model, tokenizer, context, truth, distractors
        )

        assert isinstance(win, bool)

    def test_long_context(self):
        """Handles long context."""
        tokenizer = MockTokenizer(vocab_size=128)
        model = MockModel(vocab_size=128)

        context = "A" * 1000  # Long context
        truth = "yes"
        distractors = ["no"]

        win, margin, _, _ = score_fact_mc(
            model, tokenizer, context, truth, distractors
        )

        assert isinstance(win, bool)

    def test_unicode_text(self):
        """Handles unicode characters."""
        tokenizer = MockTokenizer(vocab_size=256)  # Larger vocab for unicode
        model = MockModel(vocab_size=256)

        context = "日本語の質問"
        truth = "はい"
        distractors = ["いいえ"]

        # Should not raise
        win, margin, _, _ = score_fact_mc(
            model, tokenizer, context, truth, distractors
        )

        assert isinstance(win, bool)


# -----------------------------------------------------------------------------
# Numerical Stability Tests
# -----------------------------------------------------------------------------

class TestOracleNumericalStability:
    """Tests for numerical stability in oracle."""

    def test_extreme_logits_handled(self):
        """Handles extreme logit values without overflow."""
        tokenizer = MockTokenizer(vocab_size=128)
        # Very large logit boost
        model = MockModel(vocab_size=128, favor_tokens={65: 1000.0})

        context = "Q"
        truth = "A"
        distractors = ["B"]

        # Should not overflow
        win, margin, truth_lp, best_dist_lp = score_fact_mc(
            model, tokenizer, context, truth, distractors
        )

        assert not np.isnan(margin)
        assert not np.isinf(margin)

    def test_all_equal_logits(self):
        """Handles uniform distribution (all equal logits)."""
        tokenizer = MockTokenizer(vocab_size=128)
        model = MockModel(vocab_size=128)  # All zeros = uniform

        context = "Q"
        truth = "A"
        distractors = ["B", "C"]

        win, margin, truth_lp, best_dist_lp = score_fact_mc(
            model, tokenizer, context, truth, distractors
        )

        # All choices should have similar log-probs
        # Margin should be close to 0
        assert abs(margin) < 1.0
