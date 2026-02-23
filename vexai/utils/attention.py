"""
Attention utilities.

Implements scaled dot-product attention and multi-head attention, the
core mechanism used throughout the brain-inspired pipeline (analogous to
the brain's selective-attention networks).

Reference: "Attention Is All You Need" – Vaswani et al. (2017).
"""

from __future__ import annotations

import numpy as np


def scaled_dot_product_attention(
    queries: np.ndarray,
    keys: np.ndarray,
    values: np.ndarray,
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute scaled dot-product attention.

    Parameters
    ----------
    queries:
        Query matrix of shape ``(..., seq_q, d_k)``.
    keys:
        Key matrix of shape ``(..., seq_k, d_k)``.
    values:
        Value matrix of shape ``(..., seq_k, d_v)``.
    mask:
        Optional boolean mask of shape ``(..., seq_q, seq_k)``.  Positions
        where the mask is ``True`` are suppressed (set to ``-inf`` before
        softmax).

    Returns
    -------
    output:
        Attended values of shape ``(..., seq_q, d_v)``.
    attention_weights:
        Softmax attention weights of shape ``(..., seq_q, seq_k)``.
    """
    d_k = queries.shape[-1]
    scores = np.matmul(queries, keys.swapaxes(-2, -1)) / np.sqrt(d_k)

    if mask is not None:
        scores = np.where(mask, -1e9, scores)

    # Softmax over the last axis (key dimension).
    scores_max = scores.max(axis=-1, keepdims=True)
    exp_scores = np.exp(scores - scores_max)
    attention_weights = exp_scores / (exp_scores.sum(axis=-1, keepdims=True) + 1e-12)

    output = np.matmul(attention_weights, values)
    return output, attention_weights


def multi_head_attention(
    queries: np.ndarray,
    keys: np.ndarray,
    values: np.ndarray,
    num_heads: int = 4,
    mask: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Multi-head attention over arbitrary embedding sequences.

    Each head projects queries, keys and values to a lower-dimensional
    subspace, computes scaled dot-product attention, and the head outputs
    are concatenated.

    Parameters
    ----------
    queries:
        Shape ``(batch, seq_q, d_model)`` or ``(seq_q, d_model)``.
    keys:
        Shape ``(batch, seq_k, d_model)`` or ``(seq_k, d_model)``.
    values:
        Shape ``(batch, seq_k, d_model)`` or ``(seq_k, d_model)``.
    num_heads:
        Number of attention heads.  ``d_model`` must be divisible by this.
    mask:
        Optional mask forwarded to :func:`scaled_dot_product_attention`.
    rng:
        Numpy random generator used for initialising projection weights.
        Passing the same generator yields deterministic results.

    Returns
    -------
    output:
        Shape matching the *queries* leading dimensions with ``d_model``
        features.
    head_weights:
        List of per-head attention weight matrices.
    """
    rng = rng if rng is not None else np.random.default_rng(0)

    # Accept unbatched inputs by adding a batch dimension temporarily.
    unbatched = queries.ndim == 2
    if unbatched:
        queries = queries[np.newaxis]
        keys = keys[np.newaxis]
        values = values[np.newaxis]

    batch, seq_q, d_model = queries.shape
    d_head = d_model // num_heads

    if d_model % num_heads != 0:
        raise ValueError(
            f"d_model ({d_model}) must be divisible by num_heads ({num_heads})."
        )

    # Initialise learnable projections (scaled Xavier initialisation).
    scale = np.sqrt(2.0 / (d_model + d_head))
    Wq = [rng.normal(0, scale, (d_model, d_head)) for _ in range(num_heads)]
    Wk = [rng.normal(0, scale, (d_model, d_head)) for _ in range(num_heads)]
    Wv = [rng.normal(0, scale, (d_model, d_head)) for _ in range(num_heads)]

    head_outputs = []
    head_weights = []
    for h in range(num_heads):
        q_h = queries @ Wq[h]  # (batch, seq_q, d_head)
        k_h = keys @ Wk[h]     # (batch, seq_k, d_head)
        v_h = values @ Wv[h]   # (batch, seq_k, d_head)
        out_h, w_h = scaled_dot_product_attention(q_h, k_h, v_h, mask)
        head_outputs.append(out_h)
        head_weights.append(w_h)

    # Concatenate heads: (batch, seq_q, d_model)
    output = np.concatenate(head_outputs, axis=-1)

    if unbatched:
        output = output[0]

    return output, head_weights
