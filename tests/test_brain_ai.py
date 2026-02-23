"""Integration tests for BrainAI – the full processing pipeline."""

import numpy as np
import pytest

from vexai import BrainAI


RNG = np.random.default_rng(99)
DIM = 16


class TestBrainAIInit:
    def test_default_init(self):
        brain = BrainAI(feature_dim=DIM)
        assert brain.feature_dim == DIM
        assert brain.output_dim == DIM

    def test_custom_output_dim(self):
        brain = BrainAI(feature_dim=DIM, output_dim=8)
        assert brain.output_dim == 8


class TestBrainAIProcess:
    def setup_method(self):
        self.brain = BrainAI(feature_dim=DIM, num_heads=2, seed=0)

    def test_output_shape(self):
        x = RNG.standard_normal(DIM)
        out = self.brain.process(x)
        assert out.shape == (DIM,)

    def test_output_is_finite(self):
        x = RNG.standard_normal(DIM)
        out = self.brain.process(x)
        assert np.all(np.isfinite(out))

    def test_repeated_calls_work(self):
        for _ in range(10):
            x = RNG.standard_normal(DIM)
            out = self.brain.process(x)
            assert out.shape == (DIM,)

    def test_sequence_input(self):
        x = RNG.standard_normal((4, DIM))
        out = self.brain.process(x)
        assert out.shape == (DIM,)

    def test_memory_grows_with_calls(self):
        brain = BrainAI(feature_dim=DIM, num_heads=2, seed=1)
        for _ in range(3):
            brain.process(RNG.standard_normal(DIM))
        assert brain.hippocampus.episodic_size == 3

    def test_content_stored_in_memory(self):
        brain = BrainAI(feature_dim=DIM, num_heads=2, seed=2)
        x = RNG.standard_normal(DIM)
        brain.process(x, content="hello_world")
        results = brain.hippocampus.retrieve(x, top_k=1)
        assert len(results) == 1
        assert results[0][1] == "hello_world"


class TestBrainAIGoalSetting:
    def test_set_goal_does_not_crash(self):
        brain = BrainAI(feature_dim=DIM, num_heads=2, seed=3)
        goal = RNG.standard_normal(DIM)
        brain.set_goal(goal)
        out = brain.process(RNG.standard_normal(DIM))
        assert out.shape == (DIM,)


class TestBrainAILearning:
    def test_learn_returns_float(self):
        brain = BrainAI(feature_dim=DIM, num_heads=2, seed=4)
        x = RNG.standard_normal(DIM)
        target = RNG.standard_normal(DIM)
        mae = brain.learn(x, target)
        assert isinstance(mae, float) and mae >= 0.0

    def test_learning_reduces_error_over_iterations(self):
        brain = BrainAI(feature_dim=DIM, num_heads=2, seed=5)
        x = RNG.standard_normal(DIM)
        target = x * 1.5
        errors = [brain.learn(x, target) for _ in range(30)]
        # Error should trend downward.
        assert errors[-1] <= errors[0] + 1e-3


class TestBrainAIConsolidation:
    def test_consolidate_returns_int(self):
        brain = BrainAI(feature_dim=DIM, num_heads=2, seed=6)
        brain.process(RNG.standard_normal(DIM), importance=3.0)
        count = brain.consolidate_memory()
        assert isinstance(count, int) and count >= 0


class TestBrainAIReset:
    def test_reset_clears_working_memory(self):
        brain = BrainAI(feature_dim=DIM, num_heads=2, seed=7)
        brain.process(RNG.standard_normal(DIM))
        assert brain.prefrontal_cortex.working_memory_length > 0
        brain.reset()
        assert brain.prefrontal_cortex.working_memory_length == 0
