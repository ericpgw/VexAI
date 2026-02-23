"""
Memory store utility.

Provides a simple key-value store with cosine-similarity retrieval used
by the Hippocampus and PrefrontalCortex modules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class MemoryEntry:
    """A single stored memory."""

    key: np.ndarray      # embedding used for retrieval
    value: Any           # arbitrary payload
    importance: float = 1.0
    access_count: int = 0


class MemoryStore:
    """Fixed-capacity associative memory with cosine-similarity retrieval.

    Parameters
    ----------
    capacity:
        Maximum number of entries.  When the store is full the entry with
        the lowest ``importance`` score is evicted.
    """

    def __init__(self, capacity: int = 256) -> None:
        self.capacity = capacity
        self._entries: list[MemoryEntry] = []

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def store(self, key: np.ndarray, value: Any, importance: float = 1.0) -> None:
        """Store a new memory.

        Parameters
        ----------
        key:
            1-D embedding vector.
        value:
            Arbitrary payload to associate with the key.
        importance:
            Priority score; lower-importance entries are evicted first.
        """
        key = _normalise(key)
        if len(self._entries) >= self.capacity:
            self._evict_least_important()
        self._entries.append(MemoryEntry(key=key, value=value, importance=importance))

    def retrieve(
        self,
        query: np.ndarray,
        top_k: int = 5,
    ) -> list[tuple[float, Any]]:
        """Retrieve the *top_k* most similar memories.

        Parameters
        ----------
        query:
            1-D query embedding.
        top_k:
            Number of entries to return.

        Returns
        -------
        List of ``(similarity, value)`` tuples sorted by descending similarity.
        """
        if not self._entries:
            return []

        q = _normalise(query)
        keys_matrix = np.stack([e.key for e in self._entries])  # (N, d)
        similarities = keys_matrix @ q  # cosine similarity (keys are normalised)

        k = min(top_k, len(self._entries))
        top_indices = np.argpartition(similarities, -k)[-k:]
        top_indices = top_indices[np.argsort(similarities[top_indices])[::-1]]

        results = []
        for idx in top_indices:
            self._entries[idx].access_count += 1
            results.append((float(similarities[idx]), self._entries[idx].value))
        return results

    def update_importance(self, query: np.ndarray, delta: float) -> None:
        """Increase importance of the memory closest to *query* by *delta*."""
        if not self._entries:
            return
        q = _normalise(query)
        keys_matrix = np.stack([e.key for e in self._entries])
        idx = int(np.argmax(keys_matrix @ q))
        self._entries[idx].importance = max(0.0, self._entries[idx].importance + delta)

    def clear(self) -> None:
        """Remove all stored memories."""
        self._entries.clear()

    def __len__(self) -> int:
        return len(self._entries)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _evict_least_important(self) -> None:
        if not self._entries:
            return
        importances = [e.importance for e in self._entries]
        idx = int(np.argmin(importances))
        self._entries.pop(idx)


def _normalise(v: np.ndarray) -> np.ndarray:
    """Return L2-normalised copy of *v*."""
    norm = np.linalg.norm(v)
    if norm < 1e-12:
        return v
    return v / norm
