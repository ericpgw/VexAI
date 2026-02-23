"""
Neocortex module.

The neocortex is the brain's **highest-order integration layer**.  It
receives processed signals from all other regions and produces a unified
output representation.

In VexAI the Neocortex:

1. **Integrates** feature vectors from the sensory cortex, the
   Hippocampus context, the Amygdala gate, the Cerebellum refinement,
   and the PrefrontalCortex reasoning output via cross-attention.
2. Applies a **feed-forward network** (two-layer MLP with GELU) similar
   to the feed-forward sublayer in a Transformer.
3. Produces a final **output embedding** of configurable dimension.
"""

from __future__ import annotations

import numpy as np

from vexai.utils.attention import multi_head_attention


class Neocortex:
    """Higher-order integration and final output generation.

    Parameters
    ----------
    feature_dim:
        Shared dimensionality of all incoming feature vectors.
    output_dim:
        Dimensionality of the final output embedding.
    num_heads:
        Number of cross-attention heads.
    ffn_hidden_dim:
        Hidden dimension of the internal feed-forward network.
    rng:
        Numpy random generator.
    """

    def __init__(
        self,
        feature_dim: int = 64,
        output_dim: int = 64,
        num_heads: int = 4,
        ffn_hidden_dim: int = 128,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.feature_dim = feature_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self._rng = rng if rng is not None else np.random.default_rng(0)

        # Feed-forward network weights.
        scale1 = np.sqrt(2.0 / (feature_dim + ffn_hidden_dim))
        scale2 = np.sqrt(2.0 / (ffn_hidden_dim + output_dim))
        self._W1 = self._rng.normal(0, scale1, (feature_dim, ffn_hidden_dim))
        self._b1 = np.zeros(ffn_hidden_dim)
        self._W2 = self._rng.normal(0, scale2, (ffn_hidden_dim, output_dim))
        self._b2 = np.zeros(output_dim)

        # Output projection layer norm parameters.
        self._gamma = np.ones(output_dim)
        self._beta = np.zeros(output_dim)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def integrate(
        self,
        sensory: np.ndarray,
        memory_context: np.ndarray,
        reasoning_context: np.ndarray,
        cerebellum_output: np.ndarray,
        amygdala_gate: float = 1.0,
    ) -> np.ndarray:
        """Integrate signals from all brain regions into a final output.

        Parameters
        ----------
        sensory:
            Processed sensory features ``(feature_dim,)``.
        memory_context:
            Aggregated memory context ``(feature_dim,)``.
        reasoning_context:
            PFC reasoning context ``(feature_dim,)``.
        cerebellum_output:
            Error-corrected features from Cerebellum ``(feature_dim,)``.
        amygdala_gate:
            Salience gate ``[0, 1]`` from Amygdala – scales the memory
            contribution.

        Returns
        -------
        output:
            Final integrated embedding ``(output_dim,)``.
        """
        # Stack all inputs as a sequence: (4, feature_dim).
        tokens = np.stack([
            sensory,
            memory_context * amygdala_gate,
            reasoning_context,
            cerebellum_output,
        ])  # (4, d)

        # Cross-attend: use sensory as the query, rest as context.
        query = sensory[np.newaxis]   # (1, d)
        attended, _ = multi_head_attention(
            query, tokens, tokens, num_heads=self.num_heads, rng=self._rng
        )
        attended = attended[0]  # (d,)

        # Residual add.
        fused = sensory + attended  # (d,)

        # Feed-forward network with GELU activation.
        hidden = self._gelu(fused @ self._W1 + self._b1)  # (ffn_hidden_dim,)
        out = hidden @ self._W2 + self._b2                 # (output_dim,)

        # Layer norm on output.
        out = self._layer_norm(out)
        return out

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _gelu(x: np.ndarray) -> np.ndarray:
        """Gaussian Error Linear Unit approximation."""
        return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))

    def _layer_norm(self, x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        mean = x.mean()
        std = x.std()
        return self._gamma * (x - mean) / (std + eps) + self._beta
