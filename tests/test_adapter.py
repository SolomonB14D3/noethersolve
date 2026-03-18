"""Tests for noethersolve.adapter — Snap-On logit-space adapter architectures.

Tests cover:
- SnapOnConfig serialization/deserialization
- SnapOnMLP initialization and forward pass
- SnapOnLogitMLP initialization and forward pass
- SnapOnTransformer initialization and forward pass
- create_adapter factory function
- Zero initialization property (adapters start as identity)
"""

import json
import os
import tempfile
import pytest

# Skip all tests if MLX not available (non-Apple Silicon)
mlx = pytest.importorskip("mlx.core", reason="MLX required for adapter tests")
nn = pytest.importorskip("mlx.nn", reason="MLX required for adapter tests")

from noethersolve.adapter import (
    SnapOnConfig,
    SnapOnMLP,
    SnapOnLogitMLP,
    SnapOnTransformer,
    TransformerBlock,
    create_adapter,
)


# -----------------------------------------------------------------------------
# SnapOnConfig Tests
# -----------------------------------------------------------------------------

class TestSnapOnConfig:
    """Tests for SnapOnConfig dataclass."""

    def test_default_values(self):
        """Config has sensible defaults."""
        cfg = SnapOnConfig()
        assert cfg.d_model == 3584
        assert cfg.d_inner == 1024
        assert cfg.n_layers == 0
        assert cfg.n_heads == 8
        assert cfg.mode == "hidden"
        assert cfg.vocab_size == 0

    def test_custom_values(self):
        """Config accepts custom values."""
        cfg = SnapOnConfig(
            d_model=512,
            d_inner=128,
            n_layers=2,
            n_heads=4,
            mode="logit",
            vocab_size=32000,
        )
        assert cfg.d_model == 512
        assert cfg.d_inner == 128
        assert cfg.n_layers == 2
        assert cfg.n_heads == 4
        assert cfg.mode == "logit"
        assert cfg.vocab_size == 32000

    def test_save_load_roundtrip(self):
        """Config can be saved and loaded."""
        cfg = SnapOnConfig(d_model=256, d_inner=64, mode="logit", vocab_size=1000)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            cfg.save(f.name)
            loaded = SnapOnConfig.load(f.name)

        os.unlink(f.name)

        assert loaded.d_model == cfg.d_model
        assert loaded.d_inner == cfg.d_inner
        assert loaded.mode == cfg.mode
        assert loaded.vocab_size == cfg.vocab_size

    def test_load_ignores_unknown_fields(self):
        """Loading ignores unknown JSON fields gracefully."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({
                "d_model": 256,
                "d_inner": 64,
                "unknown_field": "should_be_ignored",
                "another_unknown": 42,
            }, f)
            f.flush()
            loaded = SnapOnConfig.load(f.name)

        os.unlink(f.name)

        assert loaded.d_model == 256
        assert loaded.d_inner == 64
        assert not hasattr(loaded, "unknown_field")


# -----------------------------------------------------------------------------
# SnapOnMLP Tests
# -----------------------------------------------------------------------------

class TestSnapOnMLP:
    """Tests for SnapOnMLP (hidden-space adapter)."""

    def test_initialization(self):
        """MLP adapter initializes with correct shapes."""
        cfg = SnapOnConfig(d_model=64, d_inner=16)
        adapter = SnapOnMLP(cfg)

        assert adapter.gate_proj.weight.shape == (16, 64)
        assert adapter.up_proj.weight.shape == (16, 64)
        assert adapter.down_proj.weight.shape == (64, 16)

    def test_zero_init_output(self):
        """down_proj is zero-initialized, so output starts as zeros."""
        cfg = SnapOnConfig(d_model=64, d_inner=16)
        adapter = SnapOnMLP(cfg)

        # down_proj should be zeros
        assert mlx.allclose(adapter.down_proj.weight, mlx.zeros_like(adapter.down_proj.weight))

    def test_forward_zero_output(self):
        """Fresh adapter produces zero adjustment (identity behavior)."""
        cfg = SnapOnConfig(d_model=64, d_inner=16)
        adapter = SnapOnMLP(cfg)

        # Random input
        h = mlx.random.normal((2, 10, 64))  # [batch, seq, d_model]
        out = adapter(h)
        mlx.eval(out)

        # Output should be all zeros due to zero-init down_proj
        assert mlx.allclose(out, mlx.zeros_like(out), atol=1e-6)

    def test_forward_shape(self):
        """Forward pass preserves shape."""
        cfg = SnapOnConfig(d_model=64, d_inner=16)
        adapter = SnapOnMLP(cfg)

        h = mlx.random.normal((2, 10, 64))
        out = adapter(h)
        mlx.eval(out)

        assert out.shape == h.shape

    def test_non_zero_after_modification(self):
        """After modifying down_proj, output is non-zero."""
        cfg = SnapOnConfig(d_model=64, d_inner=16)
        adapter = SnapOnMLP(cfg)

        # Set down_proj to non-zero
        adapter.down_proj.weight = mlx.ones((64, 16))

        h = mlx.random.normal((2, 10, 64))
        out = adapter(h)
        mlx.eval(out)

        # Output should now be non-zero
        assert not mlx.allclose(out, mlx.zeros_like(out))


# -----------------------------------------------------------------------------
# SnapOnLogitMLP Tests
# -----------------------------------------------------------------------------

class TestSnapOnLogitMLP:
    """Tests for SnapOnLogitMLP (logit-space adapter)."""

    def test_initialization(self):
        """Logit adapter initializes with vocab-sized layers."""
        cfg = SnapOnConfig(d_model=64, d_inner=16, mode="logit", vocab_size=1000)
        adapter = SnapOnLogitMLP(cfg)

        assert adapter.gate_proj.weight.shape == (16, 1000)
        assert adapter.up_proj.weight.shape == (16, 1000)
        assert adapter.down_proj.weight.shape == (1000, 16)

    def test_requires_vocab_size(self):
        """Logit adapter raises error if vocab_size not set."""
        cfg = SnapOnConfig(d_model=64, d_inner=16, mode="logit", vocab_size=0)

        with pytest.raises(AssertionError, match="vocab_size must be set"):
            SnapOnLogitMLP(cfg)

    def test_zero_init_output(self):
        """down_proj is zero-initialized."""
        cfg = SnapOnConfig(d_model=64, d_inner=16, mode="logit", vocab_size=1000)
        adapter = SnapOnLogitMLP(cfg)

        assert mlx.allclose(adapter.down_proj.weight, mlx.zeros_like(adapter.down_proj.weight))

    def test_forward_zero_output(self):
        """Fresh adapter produces zero logit adjustment."""
        cfg = SnapOnConfig(d_model=64, d_inner=16, mode="logit", vocab_size=1000)
        adapter = SnapOnLogitMLP(cfg)

        logits = mlx.random.normal((2, 10, 1000))  # [batch, seq, vocab]
        out = adapter(logits)
        mlx.eval(out)

        assert mlx.allclose(out, mlx.zeros_like(out), atol=1e-6)

    def test_forward_shape(self):
        """Forward pass preserves shape."""
        cfg = SnapOnConfig(d_model=64, d_inner=16, mode="logit", vocab_size=1000)
        adapter = SnapOnLogitMLP(cfg)

        logits = mlx.random.normal((2, 10, 1000))
        out = adapter(logits)
        mlx.eval(out)

        assert out.shape == logits.shape


# -----------------------------------------------------------------------------
# TransformerBlock Tests
# -----------------------------------------------------------------------------

class TestTransformerBlock:
    """Tests for TransformerBlock (used in SnapOnTransformer)."""

    def test_initialization(self):
        """Block initializes with correct components."""
        block = TransformerBlock(d_model=64, n_heads=4)

        assert hasattr(block, "norm1")
        assert hasattr(block, "attn")
        assert hasattr(block, "norm2")
        assert hasattr(block, "ffn_gate")
        assert hasattr(block, "ffn_up")
        assert hasattr(block, "ffn_down")

    def test_forward_shape(self):
        """Forward pass preserves shape."""
        block = TransformerBlock(d_model=64, n_heads=4)

        x = mlx.random.normal((2, 10, 64))
        out = block(x)
        mlx.eval(out)

        assert out.shape == x.shape

    def test_forward_with_mask(self):
        """Forward pass works with explicit mask."""
        block = TransformerBlock(d_model=64, n_heads=4)

        x = mlx.random.normal((2, 10, 64))
        # Causal mask
        L = 10
        mask = nn.MultiHeadAttention.create_additive_causal_mask(L)
        mask = mask.astype(mlx.float32)

        out = block(x, mask=mask)
        mlx.eval(out)

        assert out.shape == x.shape


# -----------------------------------------------------------------------------
# SnapOnTransformer Tests
# -----------------------------------------------------------------------------

class TestSnapOnTransformer:
    """Tests for SnapOnTransformer (multi-layer transformer adapter)."""

    def test_initialization(self):
        """Transformer adapter initializes with correct structure."""
        cfg = SnapOnConfig(d_model=64, d_inner=32, n_layers=2, n_heads=4)
        adapter = SnapOnTransformer(cfg)

        assert adapter.proj_in.weight.shape == (32, 64)
        assert adapter.proj_out.weight.shape == (64, 32)
        assert len(adapter.layers) == 2

    def test_zero_init_output(self):
        """proj_out is zero-initialized."""
        cfg = SnapOnConfig(d_model=64, d_inner=32, n_layers=2, n_heads=4)
        adapter = SnapOnTransformer(cfg)

        assert mlx.allclose(adapter.proj_out.weight, mlx.zeros_like(adapter.proj_out.weight))

    def test_forward_zero_output(self):
        """Fresh adapter produces zero adjustment."""
        cfg = SnapOnConfig(d_model=64, d_inner=32, n_layers=2, n_heads=4)
        adapter = SnapOnTransformer(cfg)

        h = mlx.random.normal((2, 10, 64))
        out = adapter(h)
        mlx.eval(out)

        assert mlx.allclose(out, mlx.zeros_like(out), atol=1e-6)

    def test_forward_shape(self):
        """Forward pass preserves shape."""
        cfg = SnapOnConfig(d_model=64, d_inner=32, n_layers=2, n_heads=4)
        adapter = SnapOnTransformer(cfg)

        h = mlx.random.normal((2, 10, 64))
        out = adapter(h)
        mlx.eval(out)

        assert out.shape == h.shape

    def test_single_token_no_mask_needed(self):
        """Single-token input doesn't need mask."""
        cfg = SnapOnConfig(d_model=64, d_inner=32, n_layers=2, n_heads=4)
        adapter = SnapOnTransformer(cfg)

        h = mlx.random.normal((2, 1, 64))  # single token
        out = adapter(h)
        mlx.eval(out)

        assert out.shape == h.shape


# -----------------------------------------------------------------------------
# create_adapter Factory Tests
# -----------------------------------------------------------------------------

class TestCreateAdapter:
    """Tests for create_adapter factory function."""

    def test_creates_logit_mlp(self):
        """Factory creates SnapOnLogitMLP for logit mode."""
        cfg = SnapOnConfig(mode="logit", vocab_size=1000)
        adapter = create_adapter(cfg)

        assert isinstance(adapter, SnapOnLogitMLP)

    def test_creates_mlp_for_hidden_zero_layers(self):
        """Factory creates SnapOnMLP for hidden mode with n_layers=0."""
        cfg = SnapOnConfig(mode="hidden", n_layers=0)
        adapter = create_adapter(cfg)

        assert isinstance(adapter, SnapOnMLP)

    def test_creates_transformer_for_hidden_with_layers(self):
        """Factory creates SnapOnTransformer for hidden mode with n_layers>0."""
        cfg = SnapOnConfig(mode="hidden", n_layers=2, n_heads=4)
        adapter = create_adapter(cfg)

        assert isinstance(adapter, SnapOnTransformer)


# -----------------------------------------------------------------------------
# Integration Tests
# -----------------------------------------------------------------------------

class TestAdapterIntegration:
    """Integration tests for adapter components."""

    def test_adapter_composability(self):
        """Multiple adapters can be composed (summed outputs)."""
        cfg = SnapOnConfig(d_model=64, d_inner=16)
        adapter1 = SnapOnMLP(cfg)
        adapter2 = SnapOnMLP(cfg)

        # Set different weights
        adapter1.down_proj.weight = mlx.ones((64, 16)) * 0.1
        adapter2.down_proj.weight = mlx.ones((64, 16)) * 0.2

        h = mlx.random.normal((2, 10, 64))

        out1 = adapter1(h)
        out2 = adapter2(h)
        combined = out1 + out2

        mlx.eval(combined)

        # Combined should be non-zero and sum of both
        assert not mlx.allclose(combined, mlx.zeros_like(combined))

    def test_param_count_logit_adapter(self):
        """Logit adapter has expected parameter count."""
        # d_inner=64, vocab=1000 → 3 * 64 * 1000 = 192,000 params
        cfg = SnapOnConfig(d_inner=64, mode="logit", vocab_size=1000)
        adapter = SnapOnLogitMLP(cfg)

        # Count params by summing weight sizes
        total_params = (
            adapter.gate_proj.weight.size +
            adapter.up_proj.weight.size +
            adapter.down_proj.weight.size
        )
        expected = 3 * 64 * 1000  # gate + up + down

        assert total_params == expected

    def test_config_roundtrip_with_adapter(self):
        """Config saved from adapter can recreate same architecture."""
        cfg = SnapOnConfig(d_model=128, d_inner=32, n_layers=1, n_heads=4)
        adapter = create_adapter(cfg)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            adapter.config.save(f.name)
            loaded_cfg = SnapOnConfig.load(f.name)

        os.unlink(f.name)

        adapter2 = create_adapter(loaded_cfg)

        # Both should be same type
        assert type(adapter) is type(adapter2)
