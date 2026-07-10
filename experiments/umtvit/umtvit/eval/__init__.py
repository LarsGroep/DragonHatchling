"""umtvit.eval — evaluation suite (ARCHITECTURE §6, §9).

Labels enter the experiment only here. Planned (lands in U5):

- ``linear_probe`` / ``knn`` on frozen pooled features
- ``som_metrics``  quantization error, topographic error, dead-neuron rate
- ``manifold``     trustworthiness and continuity
- ``zaxis_probe``  per-slice frequency / receptive-field analysis — the
                   measured answer to "did scale ordering emerge?"

Stub package (U0). No eval code is defined yet.
"""

from __future__ import annotations

__all__: list[str] = []
