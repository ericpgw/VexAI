"""
BrainAI – the central orchestrator of the brain-inspired AI system.

Architecture overview
---------------------

::

    Raw Input
        │
        ▼
   SensoryCortex          ← primary feature extraction (layer-norm +
        │                    projection + self-attention)
        ├──────────────────────────────────────────┐
        ▼                                          │
    Amygdala               ← salience / priority   │
    (salience, gate)                               │
        │                                          │
        ▼                                          │
    Hippocampus            ← memory encode/recall  │
    (memory_context)                               │
        │                                          │
        ▼                                          │
    Cerebellum             ← error correction      │
    (refined_features)                             │
        │                                          │
        ▼                                          │
 PrefrontalCortex          ← planning / reasoning  │
   (reasoning_ctx)                                 │
        │                                          │
        └──────────────────┐                       │
                           ▼                       ▼
                        Neocortex          ← integration + final output

Usage
-----
::

    from vexai import BrainAI
    import numpy as np

    brain = BrainAI(feature_dim=64)

    raw = np.random.randn(64)
    output = brain.process(raw)          # returns enhanced embedding
    print(output.shape)                  # (64,)

    # Store a goal embedding to guide reasoning.
    goal = np.random.randn(64)
    brain.set_goal(goal)

    # After receiving feedback, update Cerebellum's forward model.
    brain.learn(raw, target=output)
"""

from __future__ import annotations

from typing import Any

import numpy as np

from vexai.brain.sensory_cortex import SensoryCortex
from vexai.brain.hippocampus import Hippocampus
from vexai.brain.amygdala import Amygdala
from vexai.brain.cerebellum import Cerebellum
from vexai.brain.prefrontal_cortex import PrefrontalCortex
from vexai.brain.neocortex import Neocortex


class BrainAI:
    """Brain-inspired AI enhancement system.

    This class orchestrates all brain-region modules and exposes a simple
    :meth:`process` API that transforms raw input features into an
    enhanced output embedding.

    Parameters
    ----------
    feature_dim:
        Internal dimensionality shared by all brain regions.
    output_dim:
        Dimensionality of the final output.  Defaults to ``feature_dim``.
    num_heads:
        Number of attention heads used across all regions.
    episodic_capacity:
        Maximum episodic-memory entries in the Hippocampus.
    semantic_capacity:
        Maximum semantic-memory entries in the Hippocampus.
    working_memory_size:
        Size of the PFC working-memory buffer.
    reasoning_steps:
        Number of iterative reasoning passes in the PFC.
    seed:
        Random seed for reproducibility.
    """

    def __init__(
        self,
        feature_dim: int = 64,
        output_dim: int | None = None,
        num_heads: int = 4,
        episodic_capacity: int = 128,
        semantic_capacity: int = 512,
        working_memory_size: int = 8,
        reasoning_steps: int = 2,
        seed: int = 42,
    ) -> None:
        self.feature_dim = feature_dim
        self.output_dim = output_dim if output_dim is not None else feature_dim

        rng = np.random.default_rng(seed)

        self.sensory_cortex = SensoryCortex(
            input_dim=feature_dim,
            output_dim=feature_dim,
            num_heads=num_heads,
            rng=rng,
        )
        self.hippocampus = Hippocampus(
            episodic_capacity=episodic_capacity,
            semantic_capacity=semantic_capacity,
        )
        self.amygdala = Amygdala(feature_dim=feature_dim)
        self.cerebellum = Cerebellum(feature_dim=feature_dim)
        self.prefrontal_cortex = PrefrontalCortex(
            feature_dim=feature_dim,
            working_memory_size=working_memory_size,
            reasoning_steps=reasoning_steps,
            num_heads=num_heads,
            rng=rng,
        )
        self.neocortex = Neocortex(
            feature_dim=feature_dim,
            output_dim=self.output_dim,
            num_heads=num_heads,
            rng=rng,
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def process(
        self,
        raw_input: np.ndarray,
        content: Any = None,
        importance: float = 1.0,
    ) -> np.ndarray:
        """Run a full forward pass through all brain regions.

        Parameters
        ----------
        raw_input:
            Raw feature vector of shape ``(feature_dim,)`` or a sequence
            ``(seq, feature_dim)``.
        content:
            Optional metadata/payload to store alongside the memory
            embedding (e.g. original text, class label).
        importance:
            Initial importance score for the Hippocampus encoder.

        Returns
        -------
        output:
            Enhanced output embedding of shape ``(output_dim,)``.
        """
        # 1. Sensory Cortex – normalise & project.
        sensory_features = self.sensory_cortex.process(raw_input)
        mean_sensory = (
            sensory_features.mean(axis=0)
            if sensory_features.ndim == 2
            else sensory_features
        )

        # 2. Amygdala – compute salience gate.
        salience, gate = self.amygdala.evaluate(mean_sensory)
        effective_importance = importance * (1.0 + salience)

        # 3. Hippocampus – encode & retrieve relevant memories.
        self.hippocampus.encode(mean_sensory, content, effective_importance)
        memories = self.hippocampus.retrieve(mean_sensory, top_k=3)
        if memories:
            # Aggregate retrieved memory embeddings into a single context.
            mem_vecs = []
            for sim, _ in memories:
                # Use the similarity as a weight; retrieve matching entries.
                mem_vecs.append(mean_sensory * sim)
            memory_context = np.mean(mem_vecs, axis=0)
        else:
            memory_context = np.zeros_like(mean_sensory)

        # 4. Cerebellum – apply error-corrected refinement.
        cerebellum_output = self.cerebellum.coordinate(mean_sensory)

        # 5. PrefrontalCortex – update working memory and reason.
        self.prefrontal_cortex.update_working_memory(mean_sensory)
        reasoning_context = self.prefrontal_cortex.reason(query=mean_sensory)

        # 6. Neocortex – integrate all signals.
        output = self.neocortex.integrate(
            sensory=mean_sensory,
            memory_context=memory_context,
            reasoning_context=reasoning_context,
            cerebellum_output=cerebellum_output,
            amygdala_gate=gate,
        )

        return output

    def learn(self, raw_input: np.ndarray, target: np.ndarray) -> float:
        """Provide supervised feedback to the Cerebellum's forward model.

        Parameters
        ----------
        raw_input:
            The raw input that was previously processed.
        target:
            Desired output features of shape ``(feature_dim,)``.

        Returns
        -------
        float
            Mean absolute prediction error before the update.
        """
        sensory_features = self.sensory_cortex.process(raw_input)
        mean_sensory = (
            sensory_features.mean(axis=0)
            if sensory_features.ndim == 2
            else sensory_features
        )
        return self.cerebellum.update(mean_sensory, target)

    def set_goal(self, goal_embedding: np.ndarray) -> None:
        """Set a goal embedding to guide PFC attention during reasoning.

        Parameters
        ----------
        goal_embedding:
            Shape ``(feature_dim,)``.
        """
        self.prefrontal_cortex.set_goal(goal_embedding)

    def consolidate_memory(self) -> int:
        """Promote important episodic memories to long-term semantic storage.

        Returns
        -------
        int
            Number of memories consolidated.
        """
        return self.hippocampus.consolidate()

    def reset(self) -> None:
        """Reset transient state (working memory, amygdala baseline)."""
        self.prefrontal_cortex.clear_working_memory()
        self.amygdala.reset()
