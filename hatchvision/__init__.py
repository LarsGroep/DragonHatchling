"""hatchvision — a reusable PyTorch image-classification framework.

Design goals:

* **Dataset-agnostic** — swapping datasets only requires changing (or adding)
  a dataset loader; models, training, and explainability are untouched.
* **Modular backbones** — encoders are registered in a registry and expose a
  common interface, so anything from a small CNN to an experimental
  Baby Dragon Hatchling (BDH) encoder can be plugged in.
* **Explainability-first** — an optional Hebbian feature memory records
  neuron co-activation during training (observation only, it never touches
  gradients), and Grad-CAM / SHAP compute per-pixel feature importance.
* **Exportable** — Hebbian concept graphs serialize to IVGraph JSON and can
  be explored in the bundled Vercel web app (``webapp/``).
"""

__version__ = "0.1.0"

from hatchvision.data import DatasetSpec, build_loader, register_loader
from hatchvision.models import build_backbone, create_model, register_backbone
from hatchvision.hebbian import HebbianFeatureMemory
from hatchvision.engine import Trainer, TrainConfig

__all__ = [
    "DatasetSpec",
    "build_loader",
    "register_loader",
    "build_backbone",
    "create_model",
    "register_backbone",
    "HebbianFeatureMemory",
    "Trainer",
    "TrainConfig",
    "__version__",
]
