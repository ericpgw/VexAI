"""
Cerebellum module.

The cerebellum is the brain's **pattern-coordination and error-correction**
centre.  It does not initiate actions but *fine-tunes* them by learning
the residual error between expected and actual outputs.

In VexAI the Cerebellum:

1. Maintains an **internal forward model** – a lightweight linear map
   that predicts the expected output from the current features.
2. Computes the **prediction error** (residual).
3. Applies an **error-corrected refinement** to the feature vector by
   blending the original with the correction signal.
4. Updates the forward model weights with a simple delta (Widrow–Hoff)
   learning rule so predictions improve over time.
"""

from __future__ import annotations

import numpy as np


class Cerebellum:
    """Pattern coordination and residual error correction.

    Parameters
    ----------
    feature_dim:
        Dimensionality of the feature vectors processed.
    learning_rate:
        Step size used for the online weight update.
    correction_strength:
        Weight (0–1) blended into the output refinement.
        ``0`` = no correction; ``1`` = replace output with prediction.
    """

    def __init__(
        self,
        feature_dim: int = 64,
        learning_rate: float = 0.01,
        correction_strength: float = 0.3,
    ) -> None:
        self.feature_dim = feature_dim
        self.learning_rate = learning_rate
        self.correction_strength = correction_strength

        # Forward model: a linear map from features to predicted output.
        self._W = np.eye(feature_dim)   # start as identity (no correction)
        self._b = np.zeros(feature_dim)

        self._cumulative_error: float = 0.0
        self._update_count: int = 0

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def coordinate(self, features: np.ndarray) -> np.ndarray:
        """Apply error-corrected coordination to *features*.

        Parameters
        ----------
        features:
            Input feature vector ``(feature_dim,)`` or sequence
            ``(seq, feature_dim)``.

        Returns
        -------
        refined:
            Coordinated feature vector of the same shape.
        """
        single = features.ndim == 1
        x = features[np.newaxis] if single else features  # (seq, d)

        prediction = x @ self._W.T + self._b  # (seq, d)
        error = x - prediction                  # residual

        # Blend: output = x + strength * correction
        refined = x + self.correction_strength * error

        # Track cumulative error magnitude.
        self._cumulative_error += float(np.mean(np.abs(error)))

        return refined[0] if single else refined

    def update(self, features: np.ndarray, target: np.ndarray) -> float:
        """Update the forward model to better predict *target* from *features*.

        Parameters
        ----------
        features:
            Input features of shape ``(feature_dim,)`` or
            ``(seq, feature_dim)``.
        target:
            Desired output of same shape as *features*.

        Returns
        -------
        float
            Mean absolute prediction error before the update.
        """
        x = features.reshape(-1, self.feature_dim)
        t = target.reshape(-1, self.feature_dim)

        prediction = x @ self._W.T + self._b
        error = t - prediction  # (seq, d)
        mae = float(np.mean(np.abs(error)))

        # Delta rule: ΔW = lr * (error)^T * x  (averaged over seq)
        self._W += self.learning_rate * (error.T @ x) / max(len(x), 1)
        self._b += self.learning_rate * error.mean(axis=0)
        self._update_count += 1

        return mae

    @property
    def mean_error(self) -> float:
        """Average error magnitude observed during :meth:`coordinate` calls."""
        if self._update_count == 0:
            return 0.0
        return self._cumulative_error / max(self._update_count, 1)

    def reset_error_tracking(self) -> None:
        """Reset cumulative error statistics."""
        self._cumulative_error = 0.0
        self._update_count = 0
