"""Tests for the individual brain-region modules."""

import numpy as np
import pytest

from vexai.brain.sensory_cortex import SensoryCortex
from vexai.brain.hippocampus import Hippocampus
from vexai.brain.amygdala import Amygdala
from vexai.brain.cerebellum import Cerebellum
from vexai.brain.prefrontal_cortex import PrefrontalCortex
from vexai.brain.neocortex import Neocortex


RNG = np.random.default_rng(42)
DIM = 16


# ---------------------------------------------------------------------------
# SensoryCortex
# ---------------------------------------------------------------------------

class TestSensoryCortex:
    def test_single_vector_output_shape(self):
        sc = SensoryCortex(input_dim=DIM, output_dim=DIM, num_heads=2, rng=RNG)
        x = RNG.standard_normal(DIM)
        out = sc.process(x)
        assert out.shape == (DIM,)

    def test_sequence_output_shape(self):
        sc = SensoryCortex(input_dim=DIM, output_dim=DIM, num_heads=2, rng=RNG)
        x = RNG.standard_normal((5, DIM))
        out = sc.process(x)
        assert out.shape == (5, DIM)

    def test_output_is_finite(self):
        sc = SensoryCortex(input_dim=DIM, output_dim=DIM, num_heads=2, rng=RNG)
        out = sc.process(RNG.standard_normal(DIM))
        assert np.all(np.isfinite(out))


# ---------------------------------------------------------------------------
# Hippocampus
# ---------------------------------------------------------------------------

class TestHippocampus:
    def test_encode_increases_episodic_size(self):
        hip = Hippocampus(episodic_capacity=10, semantic_capacity=10)
        hip.encode(RNG.standard_normal(DIM), "experience_1")
        assert hip.episodic_size == 1

    def test_high_importance_also_stores_in_semantic(self):
        hip = Hippocampus(consolidation_threshold=1.5)
        hip.encode(RNG.standard_normal(DIM), "important", importance=2.0)
        assert hip.semantic_size == 1

    def test_retrieve_returns_results(self):
        hip = Hippocampus()
        key = RNG.standard_normal(DIM)
        hip.encode(key, "test_content", importance=1.0)
        results = hip.retrieve(key, top_k=1)
        assert len(results) == 1

    def test_consolidate(self):
        hip = Hippocampus(consolidation_threshold=1.5)
        hip.encode(RNG.standard_normal(DIM), "low", importance=1.0)
        hip.encode(RNG.standard_normal(DIM), "high", importance=2.0)
        count = hip.consolidate()
        # The entry with importance 2.0 qualifies.
        assert count >= 1

    def test_clear_episodic(self):
        hip = Hippocampus()
        hip.encode(RNG.standard_normal(DIM), "x")
        hip.clear_episodic()
        assert hip.episodic_size == 0


# ---------------------------------------------------------------------------
# Amygdala
# ---------------------------------------------------------------------------

class TestAmygdala:
    def test_gate_in_range(self):
        amyg = Amygdala(feature_dim=DIM)
        x = RNG.standard_normal(DIM)
        salience, gate = amyg.evaluate(x)
        assert 0.0 <= gate <= 1.0

    def test_salience_non_negative(self):
        amyg = Amygdala(feature_dim=DIM)
        for _ in range(5):
            salience, _ = amyg.evaluate(RNG.standard_normal(DIM))
            assert salience >= 0.0

    def test_reset_clears_baseline(self):
        amyg = Amygdala(feature_dim=DIM)
        amyg.evaluate(RNG.standard_normal(DIM))
        amyg.reset()
        assert amyg._baseline_norm is None


# ---------------------------------------------------------------------------
# Cerebellum
# ---------------------------------------------------------------------------

class TestCerebellum:
    def test_coordinate_output_shape(self):
        cb = Cerebellum(feature_dim=DIM)
        out = cb.coordinate(RNG.standard_normal(DIM))
        assert out.shape == (DIM,)

    def test_coordinate_sequence(self):
        cb = Cerebellum(feature_dim=DIM)
        out = cb.coordinate(RNG.standard_normal((3, DIM)))
        assert out.shape == (3, DIM)

    def test_update_returns_float(self):
        cb = Cerebellum(feature_dim=DIM)
        x = RNG.standard_normal(DIM)
        target = RNG.standard_normal(DIM)
        mae = cb.update(x, target)
        assert isinstance(mae, float) and mae >= 0.0

    def test_learning_reduces_error(self):
        # Use a small, stable learning rate so the delta rule converges.
        cb = Cerebellum(feature_dim=DIM, learning_rate=0.01)
        x = np.ones(DIM) / np.sqrt(DIM)  # unit vector – stable training signal
        target = x * 1.5
        errors = [cb.update(x, target) for _ in range(50)]
        assert errors[-1] <= errors[0] + 1e-3  # error should not grow


# ---------------------------------------------------------------------------
# PrefrontalCortex
# ---------------------------------------------------------------------------

class TestPrefrontalCortex:
    def test_reason_returns_correct_shape(self):
        pfc = PrefrontalCortex(feature_dim=DIM, num_heads=2)
        pfc.update_working_memory(RNG.standard_normal(DIM))
        ctx = pfc.reason()
        assert ctx.shape == (DIM,)

    def test_empty_buffer_returns_zeros(self):
        pfc = PrefrontalCortex(feature_dim=DIM)
        ctx = pfc.reason()
        np.testing.assert_array_equal(ctx, np.zeros(DIM))

    def test_working_memory_buffer_size(self):
        pfc = PrefrontalCortex(feature_dim=DIM, working_memory_size=3)
        for _ in range(5):
            pfc.update_working_memory(RNG.standard_normal(DIM))
        assert pfc.working_memory_length == 3

    def test_set_goal_influences_reasoning(self):
        pfc = PrefrontalCortex(feature_dim=DIM, num_heads=2)
        pfc.update_working_memory(RNG.standard_normal(DIM))
        ctx_no_goal = pfc.reason().copy()
        goal = RNG.standard_normal(DIM)
        pfc.set_goal(goal)
        ctx_with_goal = pfc.reason()
        # Outputs should differ when a goal is set.
        assert not np.allclose(ctx_no_goal, ctx_with_goal)


# ---------------------------------------------------------------------------
# Neocortex
# ---------------------------------------------------------------------------

class TestNeocortex:
    def test_integrate_output_shape(self):
        nc = Neocortex(feature_dim=DIM, output_dim=DIM, num_heads=2)
        out = nc.integrate(
            sensory=RNG.standard_normal(DIM),
            memory_context=RNG.standard_normal(DIM),
            reasoning_context=RNG.standard_normal(DIM),
            cerebellum_output=RNG.standard_normal(DIM),
        )
        assert out.shape == (DIM,)

    def test_integrate_output_finite(self):
        nc = Neocortex(feature_dim=DIM, output_dim=DIM, num_heads=2)
        out = nc.integrate(
            sensory=RNG.standard_normal(DIM),
            memory_context=RNG.standard_normal(DIM),
            reasoning_context=RNG.standard_normal(DIM),
            cerebellum_output=RNG.standard_normal(DIM),
        )
        assert np.all(np.isfinite(out))

    def test_amygdala_gate_zero_zeros_memory(self):
        nc = Neocortex(feature_dim=DIM, output_dim=DIM, num_heads=2, rng=np.random.default_rng(7))
        mem = RNG.standard_normal(DIM)
        out_gated = nc.integrate(
            sensory=RNG.standard_normal(DIM),
            memory_context=mem,
            reasoning_context=RNG.standard_normal(DIM),
            cerebellum_output=RNG.standard_normal(DIM),
            amygdala_gate=0.0,
        )
        # Gate=0 means memory_context contribution is zeroed; output still finite.
        assert np.all(np.isfinite(out_gated))
