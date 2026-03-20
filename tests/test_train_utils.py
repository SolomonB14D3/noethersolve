"""Tests for train_utils module.

Tests logit adapter application with zero-mean centering and softcap.
Requires MLX for actual computation.
"""

import pytest

# Skip entire module if MLX not available
try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False

pytestmark = pytest.mark.skipif(not HAS_MLX, reason="MLX not available")


class TestLogitSoftcap:
    """Tests for LOGIT_SOFTCAP constant."""

    def test_softcap_is_float(self):
        """Test that LOGIT_SOFTCAP is a float."""
        from noethersolve.train_utils import LOGIT_SOFTCAP

        assert isinstance(LOGIT_SOFTCAP, float)

    def test_softcap_value(self):
        """Test that LOGIT_SOFTCAP has expected Gemma 2 value."""
        from noethersolve.train_utils import LOGIT_SOFTCAP

        assert LOGIT_SOFTCAP == 30.0


class TestApplyAdapter:
    """Tests for apply_adapter function."""

    def test_output_shape_preserved(self):
        """Test that output has same shape as input."""
        from noethersolve.train_utils import apply_adapter

        # Create mock adapter that returns same shape
        class MockAdapter:
            def __call__(self, x):
                return mx.zeros_like(x)

        adapter = MockAdapter()
        base_logits = mx.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        result = apply_adapter(adapter, base_logits)

        assert result.shape == base_logits.shape

    def test_softcap_bounds_output(self):
        """Test that softcap bounds output to [-30, 30]."""
        from noethersolve.train_utils import apply_adapter, LOGIT_SOFTCAP

        class MockAdapter:
            def __call__(self, x):
                return x * 100  # Large shifts

        adapter = MockAdapter()
        base_logits = mx.array([1000.0, -1000.0, 0.0])

        result = apply_adapter(adapter, base_logits)

        # All values should be within softcap bounds
        assert mx.all(result >= -LOGIT_SOFTCAP).item()
        assert mx.all(result <= LOGIT_SOFTCAP).item()

    def test_zero_mean_centering(self):
        """Test that adapter shifts are zero-mean centered."""
        from noethersolve.train_utils import apply_adapter

        shift_values = None

        class MockAdapter:
            def __call__(self, x):
                nonlocal shift_values
                shift_values = mx.array([10.0, 20.0, 30.0])  # Mean = 20
                return shift_values

        adapter = MockAdapter()
        base_logits = mx.array([0.0, 0.0, 0.0])

        result = apply_adapter(adapter, base_logits)

        # The centered shifts would be [-10, 0, 10]
        # So result ≈ softcap([-10, 0, 10])
        # Just verify the middle value is near zero (it was at mean)
        assert abs(result[1].item()) < abs(result[0].item())
        assert abs(result[1].item()) < abs(result[2].item())

    def test_identity_adapter_preserves_logits(self):
        """Test that zero-shift adapter approximately preserves base logits."""
        from noethersolve.train_utils import apply_adapter, LOGIT_SOFTCAP

        class ZeroAdapter:
            def __call__(self, x):
                return mx.zeros_like(x)

        adapter = ZeroAdapter()
        base_logits = mx.array([1.0, 2.0, 3.0])

        result = apply_adapter(adapter, base_logits)

        # Result should be softcap(base_logits), which for small values ≈ base
        # For values << 30, tanh(x/30)*30 ≈ x
        for i in range(3):
            expected = LOGIT_SOFTCAP * float(mx.tanh(base_logits[i] / LOGIT_SOFTCAP).item())
            assert abs(result[i].item() - expected) < 1e-5


class TestApplyAdapterStack:
    """Tests for apply_adapter_stack function."""

    def test_empty_stack(self):
        """Test that empty adapter stack returns softcapped base logits."""
        from noethersolve.train_utils import apply_adapter_stack, LOGIT_SOFTCAP

        base_logits = mx.array([1.0, 2.0, 3.0])

        result = apply_adapter_stack([], base_logits)

        # Should be equivalent to just applying softcap to base
        for i in range(3):
            expected = LOGIT_SOFTCAP * float(mx.tanh(base_logits[i] / LOGIT_SOFTCAP).item())
            assert abs(result[i].item() - expected) < 1e-5

    def test_single_adapter_matches_apply_adapter(self):
        """Test that single adapter in stack matches apply_adapter."""
        from noethersolve.train_utils import apply_adapter, apply_adapter_stack

        class MockAdapter:
            def __call__(self, x):
                return x * 0.5

        adapter = MockAdapter()
        base_logits = mx.array([1.0, 2.0, 3.0])

        single_result = apply_adapter(adapter, base_logits)
        stack_result = apply_adapter_stack([adapter], base_logits)

        for i in range(3):
            assert abs(single_result[i].item() - stack_result[i].item()) < 1e-5

    def test_multiple_adapters_accumulate(self):
        """Test that multiple adapters accumulate their shifts."""
        from noethersolve.train_utils import apply_adapter_stack

        class ShiftAdapter:
            def __init__(self, shift):
                self.shift = shift

            def __call__(self, x):
                return mx.ones_like(x) * self.shift

        adapter1 = ShiftAdapter(10.0)
        adapter2 = ShiftAdapter(5.0)

        base_logits = mx.array([0.0, 0.0, 0.0])

        # With two uniform adapters, shifts are zero after centering
        # so result should be softcap(base_logits) = softcap(0) ≈ 0
        result = apply_adapter_stack([adapter1, adapter2], base_logits)

        # All values should be near zero (uniform shifts cancel after centering)
        for i in range(3):
            assert abs(result[i].item()) < 1e-5

    def test_output_shape_preserved(self):
        """Test that output shape is preserved through stack."""
        from noethersolve.train_utils import apply_adapter_stack

        class MockAdapter:
            def __call__(self, x):
                return mx.zeros_like(x)

        adapters = [MockAdapter() for _ in range(3)]
        base_logits = mx.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        result = apply_adapter_stack(adapters, base_logits)

        assert result.shape == base_logits.shape


class TestGetLmHeadFn:
    """Tests for get_lm_head_fn function.

    These tests are more limited since they require actual model objects.
    """

    def test_raises_on_missing_head(self):
        """Test that missing lm_head raises RuntimeError."""
        from noethersolve.train_utils import get_lm_head_fn

        class EmptyModel:
            pass

        model = EmptyModel()

        with pytest.raises(RuntimeError, match="Cannot find lm_head"):
            get_lm_head_fn(model)

    def test_returns_lm_head_if_present(self):
        """Test that explicit lm_head is returned."""
        from noethersolve.train_utils import get_lm_head_fn

        class MockLmHead:
            def __init__(self):
                self.weight = mx.zeros((1000, 768))

        class ModelWithHead:
            def __init__(self):
                self.lm_head = MockLmHead()

        model = ModelWithHead()
        result = get_lm_head_fn(model)

        assert result is model.lm_head


class TestMonotonicity:
    """Tests verifying softcap preserves ranking (argmax)."""

    def test_softcap_preserves_argmax(self):
        """Test that softcap preserves argmax rankings."""
        from noethersolve.train_utils import apply_adapter

        class ZeroAdapter:
            def __call__(self, x):
                return mx.zeros_like(x)

        adapter = ZeroAdapter()

        # Various input distributions
        test_cases = [
            mx.array([1.0, 5.0, 3.0]),  # argmax = 1
            mx.array([100.0, -50.0, 25.0]),  # argmax = 0
            mx.array([-10.0, -5.0, -1.0]),  # argmax = 2
        ]

        for base_logits in test_cases:
            result = apply_adapter(adapter, base_logits)

            # argmax should be preserved
            assert mx.argmax(result).item() == mx.argmax(base_logits).item()
