"""umtvit.engine — training loop and ablation runner (ARCHITECTURE §5, §9).

Planned (lands in U4): a resumable trainer with AMP, gradient checkpointing,
cosine LR + warmup, and the σ/τ_som anneal schedules, plus an ablation runner
that toggles loss terms and architecture variants to generate comparison
tables (not hand-assembled).

Stub package (U0). No engine code is defined yet.
"""

from __future__ import annotations

__all__: list[str] = []
