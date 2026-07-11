"""umtvit.losses — self-supervised objectives (ARCHITECTURE §3.6-§3.8).

The U4 loss suite. Every term is a plain function of already-computed tensors so
the trainer (:mod:`umtvit.engine.trainer`) can weight, switch, and compose them:

- :func:`nt_xent`                 NT-Xent contrastive loss over pooled projections
- :func:`smoothness_loss`         total-variation smoothness (axes-configurable)
- :func:`ordering_loss`           layer-scale ordering regulariser ``L_order`` (§3.7)
- :func:`monotone_centroid_loss`  per-slice spectral-centroid monotonicity (§3.7)
- :func:`geodesic_loss`           ablation-gated geodesic loss (weight 0 default; §3.6)
- :func:`total_loss`              weighted composition → ``(total, detached floats)``

The SOM quantization loss ``L_som`` lives in
:meth:`umtvit.models.som3d.Soft3DSOM.loss` (reused, not duplicated here).
"""

from __future__ import annotations

from umtvit.losses.compose import total_loss
from umtvit.losses.geodesic import geodesic_loss
from umtvit.losses.ntxent import nt_xent
from umtvit.losses.ordering import monotone_centroid_loss, ordering_loss
from umtvit.losses.smoothness import smoothness_loss

__all__ = [
    "nt_xent",
    "smoothness_loss",
    "ordering_loss",
    "monotone_centroid_loss",
    "geodesic_loss",
    "total_loss",
]
