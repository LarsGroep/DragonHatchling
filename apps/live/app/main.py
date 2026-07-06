"""ViTreous live service (§14) — FastAPI on Hugging Face Spaces free CPU.

M0 ships only ``GET /health``. The analyze/jobs/pack routes (§14) land at M8;
they will import the same ``vitreous.packs.build_pack()`` the Kaggle notebooks
use — one code path, two venues.
"""

from __future__ import annotations

from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(
    title="vitreous-live",
    version="0.1.0",
    description="ViTreous live analysis service (CPU) — M0 health stub.",
)


class Health(BaseModel):
    status: str
    model: Optional[str]
    dataset: Optional[str]
    warm: bool


@app.get("/health", response_model=Health)
def health() -> Health:
    """Liveness probe (§14).

    ``model``/``dataset`` are null until a model is loaded (M8); ``warm`` is
    True once the process is ready to serve. The UI uses this to detect
    503-warming honestly.
    """
    return Health(status="ok", model=None, dataset=None, warm=True)
