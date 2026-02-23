# VexAI

**VexAI** is a brain-inspired AI enhancement system built with NumPy.  It
wraps any feature-embedding pipeline in a modular architecture that mirrors
the structure and function of the biological brain, delivering richer
representations through hierarchical memory, salience gating, error
correction, and multi-step reasoning.

---

## Architecture

```
Raw Input
    │
    ▼
SensoryCortex          ← layer-norm → linear projection → ReLU → self-attention
    │
    ├── Amygdala       ← salience scoring & sigmoid gate
    │
    ├── Hippocampus    ← episodic + semantic memory (encode / retrieve)
    │
    ├── Cerebellum     ← forward-model error correction (Widrow-Hoff learning)
    │
    ├── PrefrontalCortex  ← working-memory buffer + iterative self-attention reasoning
    │
    └── Neocortex      ← cross-attention integration + FFN (GELU) + layer-norm
                              │
                              ▼
                         Output Embedding
```

### Brain-region modules

| Module | Brain analogy | Role |
|---|---|---|
| `SensoryCortex` | Primary sensory cortices | Normalise, project and encode raw inputs |
| `Hippocampus` | Hippocampus | Episodic + semantic memory; cosine-similarity retrieval |
| `Amygdala` | Amygdala | Salience / importance gating via EMA deviation |
| `Cerebellum` | Cerebellum | Residual error correction; online linear forward-model |
| `PrefrontalCortex` | Prefrontal cortex | Working-memory buffer + multi-step reasoning |
| `Neocortex` | Neocortex | Cross-attention integration of all region outputs |

### Utility modules

| Module | Description |
|---|---|
| `vexai.utils.attention` | Scaled dot-product attention & multi-head attention (NumPy) |
| `vexai.utils.memory` | Fixed-capacity associative `MemoryStore` with cosine retrieval |

---

## Installation

```bash
pip install -e .
```

Python ≥ 3.9 and NumPy ≥ 1.24 are required.

---

## Quick start

```python
import numpy as np
from vexai import BrainAI

# Create a brain with 64-dimensional internal representations.
brain = BrainAI(feature_dim=64, num_heads=4, seed=42)

# Process a raw feature vector.
raw = np.random.randn(64)
output = brain.process(raw, content="my first experience")
print(output.shape)  # (64,)

# Attach a goal embedding to guide PFC attention.
goal = np.random.randn(64)
brain.set_goal(goal)

# Provide feedback to improve the Cerebellum's forward model.
target = np.random.randn(64)
mae = brain.learn(raw, target)
print(f"MAE before update: {mae:.4f}")

# Promote important memories from episodic to semantic storage.
n = brain.consolidate_memory()
print(f"Consolidated {n} memories")

# Reset transient state (working memory + amygdala baseline).
brain.reset()
```

### Sequence inputs

```python
# Pass a sequence of feature vectors (e.g. token embeddings).
seq = np.random.randn(10, 64)
output = brain.process(seq)  # still returns shape (64,)
```

### Using individual regions

```python
from vexai.brain import Hippocampus, Amygdala

hip = Hippocampus(episodic_capacity=256, semantic_capacity=1024)
hip.encode(np.random.randn(64), content="fact about cats", importance=1.5)

results = hip.retrieve(np.random.randn(64), top_k=3)
for similarity, content in results:
    print(f"{similarity:.3f}  {content}")

amyg = Amygdala(feature_dim=64, sensitivity=3.0)
salience, gate = amyg.evaluate(np.random.randn(64))
print(f"salience={salience:.3f}  gate={gate:.3f}")
```

---

## Running tests

```bash
pip install pytest
pytest tests/ -v
```

---

## Design principles

* **Pure NumPy** – no deep-learning framework required; easy to integrate
  into any Python project.
* **Modular** – each brain region is a self-contained class that can be
  used independently or composed via `BrainAI`.
* **Online learning** – the Cerebellum updates its forward model
  incrementally with each `brain.learn()` call.
* **Hierarchical memory** – the Hippocampus separates short-term episodic
  and long-term semantic memory with automatic consolidation.
* **Attention throughout** – multi-head self-attention and cross-attention
  are used at every stage, mirroring the brain's selective-attention
  networks described in the Transformer literature.
