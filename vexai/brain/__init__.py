"""Brain sub-package – individual brain-region modules."""

from vexai.brain.sensory_cortex import SensoryCortex
from vexai.brain.hippocampus import Hippocampus
from vexai.brain.amygdala import Amygdala
from vexai.brain.cerebellum import Cerebellum
from vexai.brain.prefrontal_cortex import PrefrontalCortex
from vexai.brain.neocortex import Neocortex
from vexai.brain.brain_ai import BrainAI

__all__ = [
    "SensoryCortex",
    "Hippocampus",
    "Amygdala",
    "Cerebellum",
    "PrefrontalCortex",
    "Neocortex",
    "BrainAI",
]
