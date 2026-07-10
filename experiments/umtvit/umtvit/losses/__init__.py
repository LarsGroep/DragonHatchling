"""umtvit.losses — self-supervised objectives (ARCHITECTURE §3.6-§3.8).

Planned modules (land in U4, per ARCHITECTURE §5/§9):

- ``ntxent``      NT-Xent contrastive loss over pooled projections
- ``som``         soft-SOM quantization loss ``L_som``
- ``smoothness``  total-variation smoothness over the volume neighbour graph
- ``ordering``    layer-scale ordering regulariser ``L_order`` (§3.7)
- ``geodesic``    ablation-gated geodesic loss (weight 0 by default; §3.6)

Stub package (U0). No loss code is defined yet.
"""

from __future__ import annotations

__all__: list[str] = []
