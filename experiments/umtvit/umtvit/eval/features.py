"""Frozen-feature extraction for the evaluation suite (ARCHITECTURE §6.1).

Labels enter the experiment only in :mod:`umtvit.eval`. This module runs the
trained encoder over an eval-mode dataset/split in ``eval()`` + ``no_grad`` and
returns the pooled features, the class labels (when present), and the flattened
input pixels — the three arrays every downstream probe reads:

- ``pooled``  — the label-free representation the linear probe / k-NN read out.
- ``labels``  — integer class indices, or ``None`` in fully-unlabeled mode (a
  dataset with no ``label_column`` yields ``-1`` for every item; detecting that
  here lets the probes skip gracefully instead of crashing).
- ``pixels``  — the flattened resize-only input, the *high-dimensional* space
  the manifold (trustworthiness / continuity) metrics compare against.

The dataset must be in ``"eval"`` mode (deterministic resize-only view,
``(image, label)`` per item — see :class:`umtvit.data.dataset.UniversalDataset`);
no augmentation touches the features. Extraction is batched and capped by an
optional ``max_n`` for the metrics that only need a subsample.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import Tensor
from torch.utils.data import DataLoader

__all__ = ["FrozenFeatures", "extract_features", "standardize"]


@dataclass
class FrozenFeatures:
    """Frozen-feature bundle over one dataset split.

    Attributes:
        pooled: ``[N, D]`` pooled encoder features (``D = depth*volume_channels``).
        labels: ``[N]`` long class indices, or ``None`` in fully-unlabeled mode.
        pixels: ``[N, channels*H*W]`` flattened resize-only input pixels.
        num_classes: number of distinct classes (0 when unlabeled).
    """

    pooled: Tensor
    labels: Optional[Tensor]
    pixels: Tensor
    num_classes: int

    @property
    def labeled(self) -> bool:
        """Whether class labels are available (probe/k-NN can run)."""
        return self.labels is not None

    def __len__(self) -> int:
        return int(self.pooled.shape[0])


def _model_device(model: torch.nn.Module) -> torch.device:
    """Best-effort device of ``model`` (falls back to CPU for a param-less module)."""
    try:
        return next(model.parameters()).device
    except StopIteration:  # pragma: no cover - models here always have params
        return torch.device("cpu")


@torch.no_grad()
def extract_features(
    model: torch.nn.Module,
    dataset,
    *,
    device: Optional[torch.device] = None,
    batch_size: int = 64,
    max_n: Optional[int] = None,
) -> FrozenFeatures:
    """Extract pooled features, labels, and flattened pixels over a split.

    Args:
        model: A UMT-ViT encoder whose ``forward(x)`` returns a dict with a
            ``"pooled"`` ``[B, D]`` entry.
        dataset: An eval-mode dataset yielding ``(image [C,H,W], label)``.
        device: Where to run the forward pass; defaults to the model's device.
        batch_size: Extraction batch size (no effect on the result values).
        max_n: Optional cap on the number of items returned (first ``max_n``).

    Returns:
        A :class:`FrozenFeatures`. ``labels`` is ``None`` iff the dataset is in
        unlabeled mode (every label ``< 0``).
    """
    device = device if device is not None else _model_device(model)
    was_training = model.training
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    feats, labels, pixels, seen = [], [], [], 0
    for x, y in loader:
        out = model(x.to(device))
        feats.append(out["pooled"].detach().to("cpu").float())
        labels.append(torch.as_tensor(y).reshape(-1).long())
        pixels.append(x.reshape(x.shape[0], -1).cpu().float())
        seen += x.shape[0]
        if max_n is not None and seen >= max_n:
            break

    if was_training:
        model.train()

    if not feats:
        return FrozenFeatures(
            pooled=torch.zeros(0, 0),
            labels=None,
            pixels=torch.zeros(0, 0),
            num_classes=0,
        )

    pooled = torch.cat(feats)
    label_vec = torch.cat(labels)
    pixel_mat = torch.cat(pixels)
    if max_n is not None:
        pooled, label_vec, pixel_mat = pooled[:max_n], label_vec[:max_n], pixel_mat[:max_n]

    # Fully-unlabeled mode: the dataset carries no label_column and yields -1
    # for every item. Any negative label ⇒ treat the whole split as unlabeled.
    if label_vec.numel() == 0 or bool((label_vec < 0).any()):
        return FrozenFeatures(pooled, None, pixel_mat, 0)

    num_classes = int(label_vec.max().item()) + 1
    return FrozenFeatures(pooled, label_vec, pixel_mat, num_classes)


def standardize(train: Tensor, test: Tensor) -> Tuple[Tensor, Tensor]:
    """Z-score ``train``/``test`` by the *train* per-feature mean and std.

    Matches the notebook probe/k-NN pre-processing: the test split is scaled by
    train statistics only (no leakage). A small epsilon guards zero-variance
    features.
    """
    mu = train.mean(0, keepdim=True)
    sd = train.std(0, keepdim=True) + 1e-6
    return (train - mu) / sd, (test - mu) / sd
