"""
Prefrontal Cortex module.

The prefrontal cortex (PFC) is the seat of **executive function**:
working memory, planning, reasoning, and decision-making.

In VexAI the PrefrontalCortex:

1. Maintains a **working-memory buffer** – a fixed-size queue of the
   most recent feature vectors, analogous to the PFC's role in
   maintaining active representations.
2. Performs **multi-step reasoning** by iteratively applying self-
   attention over the working-memory buffer.
3. Produces a **context vector** that summarises the current working
   state for use by downstream regions.
4. Supports **goal-directed attention** – a goal embedding can bias
   attention towards relevant memories.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from vexai.utils.attention import multi_head_attention


class PrefrontalCortex:
    """Executive function: working memory, planning, and reasoning.

    Parameters
    ----------
    feature_dim:
        Dimensionality of feature vectors.
    working_memory_size:
        Maximum number of recent feature vectors held in the working-
        memory buffer.
    reasoning_steps:
        Number of iterative self-attention rounds applied during
        :meth:`reason`.
    num_heads:
        Number of attention heads per reasoning step.
    rng:
        Numpy random generator.
    """

    def __init__(
        self,
        feature_dim: int = 64,
        working_memory_size: int = 8,
        reasoning_steps: int = 2,
        num_heads: int = 4,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.feature_dim = feature_dim
        self.working_memory_size = working_memory_size
        self.reasoning_steps = reasoning_steps
        self.num_heads = num_heads
        self._rng = rng if rng is not None else np.random.default_rng(0)

        self._buffer: deque[np.ndarray] = deque(maxlen=working_memory_size)
        self._goal: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def update_working_memory(self, features: np.ndarray) -> None:
        """Push new features into the working-memory buffer.

        Parameters
        ----------
        features:
            Shape ``(feature_dim,)`` or ``(seq, feature_dim)``.
        """
        if features.ndim == 1:
            self._buffer.append(features.copy())
        else:
            for vec in features:
                self._buffer.append(vec.copy())

    def set_goal(self, goal_embedding: np.ndarray) -> None:
        """Set a goal embedding that biases attention during reasoning.

        Parameters
        ----------
        goal_embedding:
            Shape ``(feature_dim,)``.
        """
        self._goal = goal_embedding.copy()

    def reason(self, query: np.ndarray | None = None) -> np.ndarray:
        """Run multi-step reasoning over working memory.

        Parameters
        ----------
        query:
            Optional query vector ``(feature_dim,)`` used as the initial
            query in the first attention step.  If ``None``, the most
            recent working-memory entry is used.

        Returns
        -------
        context:
            Summarised context vector ``(feature_dim,)``.
        """
        if not self._buffer:
            dim = self.feature_dim
            return np.zeros(dim)

        # Build the sequence from working memory.
        seq = np.stack(list(self._buffer))  # (T, d)

        # Optionally prepend the goal as an extra token.
        if self._goal is not None:
            seq = np.vstack([self._goal[np.newaxis], seq])  # (T+1, d)

        # Starting query.
        q = query if query is not None else seq[-1:]  # (1, d)
        if q.ndim == 1:
            q = q[np.newaxis]

        # Multi-step reasoning via iterative self-attention.
        state = seq.copy()
        for _ in range(self.reasoning_steps):
            attended, _ = multi_head_attention(
                q, state, state, num_heads=self.num_heads, rng=self._rng
            )
            # Residual connection: update query with attended output.
            q = q + attended

        context = q[0]  # (d,)
        return context

    def clear_working_memory(self) -> None:
        """Flush the working-memory buffer."""
        self._buffer.clear()

    @property
    def working_memory_length(self) -> int:
        return len(self._buffer)
