"""
Amygdala module.

The amygdala is the brain's **salience and priority detector**.  It
assigns an importance score to incoming signals – analogous to how the
biological amygdala tags emotionally significant events for enhanced
memory encoding.

In VexAI the Amygdala:

1. Computes a **salience score** for each feature vector based on its
   L2 norm and deviation from the running baseline.
2. Provides a **gating signal** (0–1) that downstream regions use to
   modulate how much weight they give to an input.
3. Maintains an **exponential moving average** baseline to distinguish
   novel/important signals from routine ones.
"""

from __future__ import annotations

import numpy as np


class Amygdala:
    """Salience detection and priority scoring.

    Parameters
    ----------
    feature_dim:
        Dimensionality of the feature vectors it receives.
    ema_alpha:
        Smoothing factor for the exponential moving average baseline.
        Smaller values make the baseline change more slowly.
    sensitivity:
        Scaling factor that controls how sharply the sigmoid gating
        function responds to deviations from the baseline.
    """

    def __init__(
        self,
        feature_dim: int = 64,
        ema_alpha: float = 0.1,
        sensitivity: float = 3.0,
    ) -> None:
        self.feature_dim = feature_dim
        self.ema_alpha = ema_alpha
        self.sensitivity = sensitivity

        self._baseline_norm: float | None = None   # running average signal norm
        self._step: int = 0

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def evaluate(self, features: np.ndarray) -> tuple[float, float]:
        """Compute salience score and gate value for *features*.

        Parameters
        ----------
        features:
            Feature vector of shape ``(feature_dim,)`` or a 2-D matrix
            ``(seq, feature_dim)`` – in the latter case the mean vector is
            used.

        Returns
        -------
        salience:
            Raw importance score (non-negative float).
        gate:
            Sigmoid-compressed gate in ``[0, 1]``.  Values close to 1
            indicate high priority.
        """
        vec = features.mean(axis=0) if features.ndim == 2 else features
        norm = float(np.linalg.norm(vec))

        # Initialise baseline on first call.
        if self._baseline_norm is None:
            self._baseline_norm = norm

        deviation = abs(norm - self._baseline_norm)
        salience = deviation / (self._baseline_norm + 1e-12)

        # Update running baseline.
        self._baseline_norm = (
            (1 - self.ema_alpha) * self._baseline_norm + self.ema_alpha * norm
        )
        self._step += 1

        gate = self._sigmoid(self.sensitivity * salience)
        return salience, gate

    def reset(self) -> None:
        """Reset the running baseline (e.g. at the start of a new task)."""
        self._baseline_norm = None
        self._step = 0

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _sigmoid(x: float) -> float:
        return 1.0 / (1.0 + np.exp(-x))
