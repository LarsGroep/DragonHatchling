"""umtvit.engine — training loop and ablation runner (ARCHITECTURE §5, §9).

Ships the U4 :class:`~umtvit.engine.trainer.Trainer`: a resumable, callback-driven
self-supervised trainer with AMP, optional encoder gradient checkpointing, cosine
LR + warmup, and the σ neighbourhood anneal. The ablation runner (toggling loss
terms / architecture variants into comparison tables) lands with U5.
"""

from __future__ import annotations

from umtvit.engine.trainer import Trainer

__all__ = ["Trainer"]
