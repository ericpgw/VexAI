"""Tests for the MemoryStore utility."""

import numpy as np
import pytest

from vexai.utils.memory import MemoryStore


RNG = np.random.default_rng(1)


class TestMemoryStore:
    def test_store_and_len(self):
        store = MemoryStore(capacity=10)
        for i in range(5):
            store.store(RNG.standard_normal(8), f"item_{i}")
        assert len(store) == 5

    def test_capacity_eviction(self):
        store = MemoryStore(capacity=3)
        for i in range(5):
            store.store(RNG.standard_normal(8), f"item_{i}", importance=float(i))
        # Capacity is 3; oldest least-important entries should be evicted.
        assert len(store) == 3

    def test_retrieve_returns_sorted_by_similarity(self):
        store = MemoryStore(capacity=10)
        target = np.array([1.0, 0.0, 0.0, 0.0])
        store.store(target.copy(), "exact_match", importance=1.0)
        store.store(np.array([0.0, 1.0, 0.0, 0.0]), "orthogonal", importance=1.0)
        store.store(np.array([0.9, 0.1, 0.0, 0.0]), "close", importance=1.0)

        results = store.retrieve(target, top_k=3)
        assert len(results) == 3
        # Best match should be first.
        assert results[0][0] >= results[1][0] >= results[2][0]
        assert results[0][1] == "exact_match"

    def test_retrieve_empty_store(self):
        store = MemoryStore()
        result = store.retrieve(RNG.standard_normal(8))
        assert result == []

    def test_update_importance(self):
        store = MemoryStore()
        key = np.array([1.0, 0.0])
        store.store(key.copy(), "item", importance=1.0)
        store.update_importance(key, delta=2.0)
        assert store._entries[0].importance == pytest.approx(3.0)

    def test_clear(self):
        store = MemoryStore()
        store.store(RNG.standard_normal(8), "x")
        store.clear()
        assert len(store) == 0
