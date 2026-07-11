"""umtvit.engine — training loop and ablation runner (ARCHITECTURE §5, §9).

Ships the U4 :class:`~umtvit.engine.trainer.Trainer`: a resumable, callback-driven
self-supervised trainer with AMP, optional encoder gradient checkpointing, cosine
LR + warmup, and the σ neighbourhood anneal. U5 adds
:class:`~umtvit.engine.ablation.AblationRunner`, which toggles loss terms /
architecture variants (the canonical :data:`~umtvit.engine.ablation.ABLATIONS`
axes) into a §6 comparison table.
"""

from __future__ import annotations

from umtvit.engine.ablation import ABLATIONS, AblationRunner
from umtvit.engine.trainer import Trainer

__all__ = ["Trainer", "AblationRunner", "ABLATIONS"]
