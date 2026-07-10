"""Stable, leakage-free train/val/test assignment (ARCHITECTURE §4).

A split is decided by hashing a *key* — a group id when one is configured,
otherwise the item's index — together with the split seed, and mapping the
resulting fraction onto the configured ``train``/``val``/``test`` bands. Two
consequences follow directly and are what the milestone requires:

- **Determinism.** The same ``(key, seed)`` always lands in the same split,
  on any machine and in any process — the hash is content-based, not tied to
  iteration order or RNG state.
- **Grouped leakage freedom.** When items carry a group id (e.g. a HAM10000
  ``lesion_id``), every item with that id hashes to the same key and therefore
  the same split, so a group can never straddle two splits.

This module is deliberately torch-free (stdlib ``hashlib`` only); it depends
on nothing but :class:`~umtvit.config.SplitConfig`.
"""

from __future__ import annotations

import hashlib
from typing import Union

from ..config import SplitConfig

__all__ = ["SPLIT_NAMES", "split_of"]

SPLIT_NAMES = ("train", "val", "test")

# Resolution of the hash → [0, 1) mapping. A million buckets is far finer than
# any split fraction we care about, and keeps the arithmetic exact in ints.
_BUCKETS = 10 ** 6


def _unit_hash(key: str) -> float:
    """Map an arbitrary string to a stable fraction in ``[0, 1)`` via md5."""
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()
    return (int(digest, 16) % _BUCKETS) / _BUCKETS


def split_of(key: Union[str, int], splits: SplitConfig) -> str:
    """Return the split (``"train"``/``"val"``/``"test"``) for ``key``.

    Args:
        key: the grouping key. Pass the configured group id when present so a
            group stays whole; otherwise pass the item's integer index.
        splits: the fractions and seed to honour.

    The seed is folded into the hashed string, so changing ``splits.seed``
    reshuffles the whole assignment deterministically.
    """
    fraction = _unit_hash(f"{key}-{splits.seed}")
    if fraction < splits.train:
        return "train"
    if fraction < splits.train + splits.val:
        return "val"
    return "test"
