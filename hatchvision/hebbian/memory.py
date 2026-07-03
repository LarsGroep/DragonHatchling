"""Hebbian feature memory.

Records neuron co-activation statistics during training via forward hooks.
"Neurons that fire together, wire together": for every observed layer we keep
an exponential moving average of the outer product of (spatially pooled,
rectified, L2-normalized) activations.  The memory is **pure observation** —
hooks detach everything and never modify activations or gradients, so
attaching it has zero effect on optimization.

Alongside co-activation it tracks per-class conditional firing rates, which
later lets concept clusters be mapped to the classes they respond to.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from torch import nn


@dataclass
class _LayerStats:
    dim: int
    coact: torch.Tensor          # [dim, dim] EMA of a a^T
    mean_act: torch.Tensor       # [dim]      EMA of a
    class_act: torch.Tensor      # [num_classes, dim] sum of a per label
    class_count: torch.Tensor    # [num_classes]
    updates: int = 0
    unit_index: Optional[torch.Tensor] = None  # subsampled channel ids


class HebbianFeatureMemory:
    """Observe co-activation of hidden units while a model trains.

    Parameters
    ----------
    model:
        Any module exposing ``hebbian_layers() -> Dict[str, nn.Module]``
        (every hatchvision backbone does), or pass ``layers`` explicitly.
    num_classes:
        Enables class-conditional firing statistics when labels are supplied
        through :meth:`observe_labels` (the Trainer does this automatically).
    max_units:
        Layers wider than this are subsampled to a fixed random set of
        channels so the co-activation matrix stays tractable.
    momentum:
        EMA momentum; higher forgets old batches faster.
    """

    def __init__(
        self,
        model: nn.Module,
        num_classes: int,
        layers: Optional[Dict[str, nn.Module]] = None,
        max_units: int = 256,
        momentum: float = 0.05,
        seed: int = 0,
    ) -> None:
        if layers is None:
            if not hasattr(model, "hebbian_layers"):
                raise ValueError(
                    "model has no hebbian_layers(); pass layers= explicitly"
                )
            layers = model.hebbian_layers()
        if not layers:
            raise ValueError("no layers to observe")
        self.num_classes = num_classes
        self.max_units = max_units
        self.momentum = momentum
        self.enabled = True
        self._labels: Optional[torch.Tensor] = None
        self._handles: List[torch.utils.hooks.RemovableHandle] = []
        self.stats: Dict[str, _LayerStats] = {}
        self._gen = torch.Generator().manual_seed(seed)
        for name, module in layers.items():
            self._handles.append(
                module.register_forward_hook(self._make_hook(name))
            )

    # ------------------------------------------------------------------ hooks

    @staticmethod
    def _pool(out: torch.Tensor) -> torch.Tensor:
        """Reduce any activation tensor to [batch, units]."""
        if out.dim() == 4:                      # conv maps [B, C, H, W]
            return out.mean(dim=(2, 3))
        if out.dim() == 3:                      # token seqs [B, T, D]
            return out.mean(dim=1)
        return out                              # already [B, D]

    def _make_hook(self, name: str):
        def hook(_module, _inputs, output):
            if not self.enabled or not isinstance(output, torch.Tensor):
                return
            with torch.no_grad():
                a = self._pool(output.detach().float())
                a = torch.relu(a)               # firing rates are positive
                self._update(name, a)
        return hook

    def _update(self, name: str, a: torch.Tensor) -> None:
        if name not in self.stats:
            dim = a.shape[1]
            unit_index = None
            if dim > self.max_units:
                unit_index = torch.randperm(dim, generator=self._gen)[: self.max_units].sort().values
                dim = self.max_units
            self.stats[name] = _LayerStats(
                dim=dim,
                coact=torch.zeros(dim, dim),
                mean_act=torch.zeros(dim),
                class_act=torch.zeros(self.num_classes, dim),
                class_count=torch.zeros(self.num_classes),
                unit_index=unit_index,
            )
        st = self.stats[name]
        if st.unit_index is not None:
            a = a[:, st.unit_index]
        a = a.to(st.coact.device)
        # Normalize each sample so co-activation reflects firing *pattern*,
        # not overall magnitude (which drifts during training).
        a_hat = a / (a.norm(dim=1, keepdim=True) + 1e-8)
        batch_co = a_hat.t() @ a_hat / a.shape[0]
        m = self.momentum
        st.coact.mul_(1 - m).add_(batch_co, alpha=m)
        st.mean_act.mul_(1 - m).add_(a_hat.mean(dim=0), alpha=m)
        if self._labels is not None and self._labels.shape[0] == a.shape[0]:
            labels = self._labels.to(a_hat.device)
            st.class_act.index_add_(0, labels, a_hat)
            st.class_count.index_add_(
                0, labels, torch.ones(labels.shape[0])
            )
        st.updates += 1

    # ------------------------------------------------------------- public API

    def observe_labels(self, labels: torch.Tensor) -> None:
        """Provide labels for the *next* forward pass (class statistics)."""
        self._labels = labels.detach().cpu()

    @contextmanager
    def paused(self):
        prev, self.enabled = self.enabled, False
        try:
            yield self
        finally:
            self.enabled = prev

    def detach(self) -> None:
        """Remove all hooks from the model."""
        for h in self._handles:
            h.remove()
        self._handles.clear()

    @property
    def layer_names(self) -> List[str]:
        return list(self.stats)

    def coactivation(self, layer: str) -> torch.Tensor:
        return self.stats[layer].coact.clone()

    def correlation(self, layer: str) -> torch.Tensor:
        """Co-activation normalized to [0, 1] cosine-style similarity."""
        c = self.stats[layer].coact
        d = torch.sqrt(torch.clamp(c.diagonal(), min=1e-12))
        return c / (d[:, None] * d[None, :])

    def class_affinity(self, layer: str) -> torch.Tensor:
        """[num_classes, units] mean firing rate per class (0 if unseen)."""
        st = self.stats[layer]
        denom = torch.clamp(st.class_count[:, None], min=1.0)
        return st.class_act / denom

    def top_edges(self, layer: str, k: int = 200) -> List[Tuple[int, int, float]]:
        """Strongest off-diagonal co-activation pairs, as (i, j, weight)."""
        corr = self.correlation(layer)
        n = corr.shape[0]
        triu = torch.triu(corr, diagonal=1)
        flat = triu.flatten()
        k = min(k, int((flat > 0).sum().item()))
        if k == 0:
            return []
        vals, idx = flat.topk(k)
        return [
            (int(i // n), int(i % n), float(v))
            for i, v in zip(idx.tolist(), vals.tolist())
        ]

    def unit_ids(self, layer: str) -> List[int]:
        """Original channel indices of the tracked units."""
        st = self.stats[layer]
        if st.unit_index is None:
            return list(range(st.dim))
        return st.unit_index.tolist()

    # -------------------------------------------------------- serialization

    def state_dict(self) -> dict:
        return {
            "num_classes": self.num_classes,
            "momentum": self.momentum,
            "layers": {
                name: {
                    "coact": st.coact,
                    "mean_act": st.mean_act,
                    "class_act": st.class_act,
                    "class_count": st.class_count,
                    "updates": st.updates,
                    "unit_index": st.unit_index,
                }
                for name, st in self.stats.items()
            },
        }

    @classmethod
    def from_state(cls, state: dict) -> "HebbianFeatureMemory":
        """Rehydrate a memory from ``state_dict()`` without a model.

        The result observes nothing (no hooks) but supports all analysis:
        correlation, class affinity, concept clustering, graph export.
        Lets saved training statistics be re-analyzed post hoc — e.g. by
        ``scripts/rebuild_graph.py`` — without retraining.
        """
        mem = cls.__new__(cls)
        mem.enabled = False
        mem._labels = None
        mem._handles = []
        mem.stats = {}
        mem.load_state_dict(state)
        mem.max_units = max((st.dim for st in mem.stats.values()), default=256)
        return mem

    def load_state_dict(self, state: dict) -> None:
        self.num_classes = state["num_classes"]
        self.momentum = state["momentum"]
        for name, s in state["layers"].items():
            self.stats[name] = _LayerStats(
                dim=s["coact"].shape[0],
                coact=s["coact"],
                mean_act=s["mean_act"],
                class_act=s["class_act"],
                class_count=s["class_count"],
                updates=s["updates"],
                unit_index=s["unit_index"],
            )
