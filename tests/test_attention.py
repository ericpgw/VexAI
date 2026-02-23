"""Tests for the attention utilities."""

import numpy as np
import pytest

from vexai.utils.attention import scaled_dot_product_attention, multi_head_attention


RNG = np.random.default_rng(0)


class TestScaledDotProductAttention:
    def test_output_shape(self):
        q = RNG.standard_normal((3, 8))
        k = RNG.standard_normal((5, 8))
        v = RNG.standard_normal((5, 16))
        out, weights = scaled_dot_product_attention(q, k, v)
        assert out.shape == (3, 16)
        assert weights.shape == (3, 5)

    def test_weights_sum_to_one(self):
        q = RNG.standard_normal((4, 8))
        k = RNG.standard_normal((6, 8))
        v = RNG.standard_normal((6, 8))
        _, weights = scaled_dot_product_attention(q, k, v)
        row_sums = weights.sum(axis=-1)
        np.testing.assert_allclose(row_sums, np.ones(4), atol=1e-6)

    def test_mask_suppresses_positions(self):
        q = RNG.standard_normal((2, 4))
        k = RNG.standard_normal((3, 4))
        v = RNG.standard_normal((3, 4))
        # Mask all except the last key.
        mask = np.array([[True, True, False], [True, True, False]])
        _, weights = scaled_dot_product_attention(q, k, v, mask=mask)
        np.testing.assert_allclose(weights[:, :2], 0.0, atol=1e-6)

    def test_batched_input(self):
        q = RNG.standard_normal((2, 3, 8))
        k = RNG.standard_normal((2, 5, 8))
        v = RNG.standard_normal((2, 5, 8))
        out, _ = scaled_dot_product_attention(q, k, v)
        assert out.shape == (2, 3, 8)


class TestMultiHeadAttention:
    def test_output_shape_unbatched(self):
        q = RNG.standard_normal((4, 16))
        out, heads = multi_head_attention(q, q, q, num_heads=4, rng=RNG)
        assert out.shape == (4, 16)
        assert len(heads) == 4

    def test_output_shape_batched(self):
        q = RNG.standard_normal((2, 4, 16))
        out, _ = multi_head_attention(q, q, q, num_heads=4, rng=RNG)
        assert out.shape == (2, 4, 16)

    def test_invalid_num_heads(self):
        q = RNG.standard_normal((4, 15))
        with pytest.raises(ValueError):
            multi_head_attention(q, q, q, num_heads=4, rng=RNG)
