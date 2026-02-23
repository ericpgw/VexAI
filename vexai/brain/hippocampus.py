"""
Hippocampus module.

The hippocampus is responsible for **memory consolidation and retrieval**.
It maintains both:

* **Episodic memory** – stores specific experiences/inputs with their
  contextual embeddings.
* **Semantic memory** – stores consolidated abstract patterns derived
  from many episodes.

Analogies to the biological hippocampus
-----------------------------------------
* Pattern separation (encoding distinct memories for similar inputs).
* Pattern completion (retrieving a full memory from a partial cue).
* Memory consolidation (promoting important episodic memories to semantic
  long-term storage).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from vexai.utils.memory import MemoryStore


class Hippocampus:
    """Memory consolidation and retrieval region.

    Parameters
    ----------
    episodic_capacity:
        Maximum number of episodic (short-term) memories.
    semantic_capacity:
        Maximum number of semantic (long-term) memories.
    consolidation_threshold:
        Importance score at which an episodic memory is promoted to
        semantic long-term storage.
    """

    def __init__(
        self,
        episodic_capacity: int = 128,
        semantic_capacity: int = 512,
        consolidation_threshold: float = 2.0,
    ) -> None:
        self.consolidation_threshold = consolidation_threshold
        self._episodic = MemoryStore(episodic_capacity)
        self._semantic = MemoryStore(semantic_capacity)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def encode(
        self,
        embedding: np.ndarray,
        content: Any,
        importance: float = 1.0,
    ) -> None:
        """Encode a new experience into episodic memory.

        If the importance exceeds :attr:`consolidation_threshold` the
        experience is immediately written to semantic memory as well
        (fast-track consolidation).

        Parameters
        ----------
        embedding:
            Feature vector representing the experience.
        content:
            Arbitrary payload (e.g. original text, model output).
        importance:
            Initial importance score.
        """
        self._episodic.store(embedding, content, importance)
        if importance >= self.consolidation_threshold:
            self._semantic.store(embedding, content, importance)

    def retrieve(
        self,
        query: np.ndarray,
        top_k: int = 5,
        memory_type: str = "both",
    ) -> list[tuple[float, Any]]:
        """Retrieve the most relevant memories.

        Parameters
        ----------
        query:
            Query embedding.
        top_k:
            Number of results per memory store.
        memory_type:
            ``"episodic"``, ``"semantic"``, or ``"both"``.

        Returns
        -------
        List of ``(similarity, content)`` tuples, sorted by descending
        similarity.
        """
        results: list[tuple[float, Any]] = []
        if memory_type in ("episodic", "both"):
            results += self._episodic.retrieve(query, top_k)
        if memory_type in ("semantic", "both"):
            results += self._semantic.retrieve(query, top_k)

        # Deduplicate by content identity and sort.
        seen: set[int] = set()
        unique: list[tuple[float, Any]] = []
        for sim, val in sorted(results, key=lambda t: t[0], reverse=True):
            vid = id(val)
            if vid not in seen:
                seen.add(vid)
                unique.append((sim, val))
        return unique[:top_k]

    def consolidate(self) -> int:
        """Promote important episodic memories to semantic storage.

        Returns
        -------
        int
            Number of memories consolidated in this call.
        """
        count = 0
        for entry in list(self._episodic._entries):
            if entry.importance >= self.consolidation_threshold:
                self._semantic.store(entry.key, entry.value, entry.importance)
                count += 1
        return count

    def reinforce(self, query: np.ndarray, delta: float = 0.5) -> None:
        """Strengthen the memory closest to *query* in both stores."""
        self._episodic.update_importance(query, delta)
        self._semantic.update_importance(query, delta)

    def clear_episodic(self) -> None:
        """Flush short-term (episodic) memory."""
        self._episodic.clear()

    @property
    def episodic_size(self) -> int:
        return len(self._episodic)

    @property
    def semantic_size(self) -> int:
        return len(self._semantic)
