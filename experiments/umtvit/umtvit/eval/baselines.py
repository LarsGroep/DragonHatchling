"""Read-out baselines — the floor a probe actually has to clear.

Every probe/k-NN number in this repo used to be reported against **uniform
chance** (``1/num_classes``) and nothing else. On a balanced dataset that is
fine. On an imbalanced one it is badly misleading: HAM10000 is ~67 % ``nv``
(nv 6705 · mel 1113 · bkl 1099 · bcc 514 · akiec 327 · vasc 142 · df 115 =
10 015), so a model that has learned nothing except the class prior — always
answer "nevus" — scores **0.669**, while uniform chance sits at 0.143. Reading
a 0.79 probe against 0.143 turns a ~12 pp improvement into an apparent ~5.5×
one, and a *degenerate* model still looks like a triumph.

So this module emits both floors plus the metrics that survive imbalance:

- ``chance`` — uniform ``1/K``, retained for continuity with recorded runs.
- ``majority_baseline`` — the majority-class rate on the evaluated split. This
  is the meaningful floor and the one to quote.
- ``balanced_accuracy`` — mean per-class recall, which a majority-class model
  cannot game (it scores ``1/K`` there, by construction).

The definitions match :mod:`vitreous.clinical` in ``packages/core``, which is
the canonical implementation (numpy-only, Wilson intervals, melanoma-specific
recall). It is not imported here because ``experiments/umtvit`` is a standalone
package that does not depend on ``vitreous``; if that ever changes, delete this
module and call ``vitreous.clinical.baselines`` directly.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Optional, Sequence

__all__ = ["balanced_accuracy", "baseline_block", "majority_baseline"]


def majority_baseline(labels: Sequence[int], num_classes: int) -> Optional[float]:
    """Accuracy of always predicting the most common class in ``labels``.

    Returns ``None`` for an empty split (no floor is defined). Computed on the
    split being evaluated, so it is the honest floor *for this measurement* —
    not the dataset-wide prior, which can differ after a grouped split.
    """
    counts = [0] * num_classes
    n = 0
    for y in labels:
        y = int(y)
        if 0 <= y < num_classes:
            counts[y] += 1
            n += 1
    if n == 0:
        return None
    return max(counts) / n


def balanced_accuracy(per_class_recall: Mapping[int, float]) -> Optional[float]:
    """Mean per-class recall, ignoring classes with no support.

    A majority-class predictor scores ``1/K`` here however skewed the data is,
    which is exactly why this is the imbalance-proof companion to accuracy.
    Classes absent from the split contribute ``nan`` upstream and are skipped
    rather than poisoning the mean; ``None`` if no class had any support.
    """
    vals = [v for v in per_class_recall.values() if v is not None and not math.isnan(v)]
    if not vals:
        return None
    return sum(vals) / len(vals)


def baseline_block(
    labels: Sequence[int],
    num_classes: int,
    *,
    per_class_recall: Optional[Mapping[int, float]] = None,
) -> Dict[str, Any]:
    """The baseline fields every read-out dict should carry.

    ``meaningful_floor`` names which baseline a reader should compare against,
    so a downstream renderer never has to guess (``apps/web`` uses it to decide
    whether a metric may be coloured as evidence).
    """
    majority = majority_baseline(labels, num_classes)
    imbalanced = majority is not None and majority > 1.5 / num_classes
    block: Dict[str, Any] = {
        "chance": 1.0 / num_classes,
        "majority_baseline": majority,
        "num_classes": num_classes,
        "imbalanced": imbalanced,
        "meaningful_floor": "majority_baseline" if imbalanced else "chance",
    }
    if per_class_recall is not None:
        block["balanced_accuracy"] = balanced_accuracy(per_class_recall)
    return block
