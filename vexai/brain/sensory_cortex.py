"""
Sensory Cortex module.

Analogous to the primary sensory cortices of the brain (visual, auditory,
somatosensory, …).  It acts as the **first processing stage**: it receives
raw inputs, normalises them, applies a learned feature projection, and
returns a structured feature vector ready for downstream brain regions.

Processing steps
----------------
1. **Layer normalisation** – stabilises the input distribution.
2. **Linear projection** – maps raw features to the internal dimension.
3. **ReLU non-linearity** – introduces sparse activation similar to
   cortical tuning curves.
4. **Multi-head self-attention** – lets the cortex attend to different
   aspects of the input simultaneously.
"""

from __future__ import annotations

import numpy as np

from vexai.utils.attention import multi_head_attention


class SensoryCortex:
    """Primary sensory processing region.

    Parameters
    ----------
    input_dim:
        Dimensionality of raw input vectors.
    output_dim:
        Dimensionality of the processed feature representation.
    num_heads:
        Number of attention heads for intra-input self-attention.
    rng:
        Numpy random generator (for reproducibility).
    """

    def __init__(
        self,
        input_dim: int = 64,
        output_dim: int = 64,
        num_heads: int = 4,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self._rng = rng if rng is not None else np.random.default_rng(42)

        # Projection weights (Xavier initialisation).
        scale = np.sqrt(2.0 / (input_dim + output_dim))
        self._W_proj = self._rng.normal(0, scale, (input_dim, output_dim))
        self._b_proj = np.zeros(output_dim)

        # Layer-norm parameters.
        self._gamma = np.ones(input_dim)
        self._beta = np.zeros(input_dim)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def process(self, x: np.ndarray) -> np.ndarray:
        """Process raw sensory input.

        Parameters
        ----------
        x:
            Raw input of shape ``(input_dim,)`` or ``(seq, input_dim)``.

        Returns
        -------
        features:
            Processed features of shape ``(output_dim,)`` or
            ``(seq, output_dim)``.
        """
        single = x.ndim == 1
        if single:
            x = x[np.newaxis, :]  # (1, input_dim)

        # 1. Layer normalisation.
        x_norm = self._layer_norm(x)

        # 2. Linear projection + bias.
        projected = x_norm @ self._W_proj + self._b_proj  # (seq, output_dim)

        # 3. ReLU activation (sparse cortical coding).
        activated = np.maximum(0, projected)

        # 4. Self-attention across sequence positions (if seq > 1).
        if activated.shape[0] > 1:
            attended, _ = multi_head_attention(
                activated, activated, activated,
                num_heads=self.num_heads,
                rng=self._rng,
            )
        else:
            attended = activated

        return attended[0] if single else attended

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _layer_norm(self, x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        mean = x.mean(axis=-1, keepdims=True)
        std = x.std(axis=-1, keepdims=True)
        return self._gamma * (x - mean) / (std + eps) + self._beta
