"""ViTreous live service (§14) — FastAPI on Hugging Face Spaces free CPU.

One code path, two venues: the analyzer is ``vitreous.packs.build_pack`` — the
same function the Kaggle precompute notebook runs. Uploads are ephemeral
(tmpdir per job, never persisted), 10 MB cap, EXIF stripped by re-encoding.
Single worker thread + tiny in-memory job queue (portfolio traffic); staged
progress streams over SSE so the UI shows predict → attention → attributions
→ … instead of a blank spinner.

The heavy analyzer is injected (``set_analyzer``) so tests run without torch;
the default analyzer lazy-imports vitreous+timm on first use.
"""

from __future__ import annotations

import asyncio
import io
import json
import os
import queue as queue_mod
import tempfile
import threading
import uuid
from pathlib import Path
from typing import Callable, Optional

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

MAX_UPLOAD = 10 * 1024 * 1024

app = FastAPI(
    title="vitreous-live",
    version="1.0.0",
    description="ViTreous live analysis service (CPU): §14 analyze/jobs/pack API.",
)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"], allow_headers=["*"]
)


class Health(BaseModel):
    status: str
    model: Optional[str]
    dataset: Optional[str]
    warm: bool


class Job:
    def __init__(self, job_id: str, workdir: Path) -> None:
        self.id = job_id
        self.workdir = workdir
        self.stage = "queued"
        self.error: Optional[str] = None
        self.events: "queue_mod.Queue[str]" = queue_mod.Queue()

    def emit(self, stage: str, **extra: object) -> None:
        self.stage = stage
        self.events.put(json.dumps({"stage": stage, **extra}))


JOBS: dict[str, Job] = {}
WORK: "queue_mod.Queue[tuple[Job, bytes, str]]" = queue_mod.Queue()

# Analyzer contract: (image_bytes, dataset, out_dir, progress_cb) -> None.
# progress_cb(stage) is called as each §14 stage completes; the pack directory
# must contain manifest.json when the analyzer returns.
Analyzer = Callable[[bytes, str, Path, Callable[[str], None]], None]
_analyzer: Optional[Analyzer] = None
_loaded: dict[str, Optional[str]] = {"model": None, "dataset": None}


def set_analyzer(fn: Optional[Analyzer]) -> None:
    """Inject the analyzer (tests use a fake; None restores the default)."""
    global _analyzer
    _analyzer = fn


def _default_analyzer(data: bytes, dataset: str, out: Path, progress: Callable[[str], None]) -> None:
    """The real path: vitreous build_pack on CPU (lazy heavy imports)."""
    import torch
    from PIL import Image

    from vitreous.data import get_dataset
    from vitreous.models import load_model
    from vitreous.packs import build_pack

    adapter = get_dataset(dataset)()
    loaded = load_model(os.environ.get("VITREOUS_MODEL", "vit_s16"), adapter.spec)
    weights = os.environ.get("VITREOUS_WEIGHTS")  # optional fine-tuned checkpoint
    if weights and Path(weights).exists():
        loaded.model.load_state_dict(torch.load(weights, map_location="cpu"))
    _loaded["model"], _loaded["dataset"] = loaded.spec.arch, dataset

    img = Image.open(io.BytesIO(data)).convert("RGB")  # re-encode: EXIF dropped
    tensor = adapter.preprocess()(img).unsqueeze(0)
    progress("predict")
    build_pack(
        loaded,
        tensor,
        image_meta={"id": out.name, "width": img.width, "height": img.height, "source": "upload"},
        dataset_spec=adapter.spec,
        out_dir=out,
    )


def _worker() -> None:
    while True:
        job, data, dataset = WORK.get()
        try:
            fn = _analyzer or _default_analyzer
            fn(data, dataset, job.workdir, lambda s, j=job: j.emit(s))
            job.emit("done", ok=True)
        except Exception as exc:  # surfaced to the client via SSE
            job.error = str(exc)
            job.emit("error", message=str(exc))


threading.Thread(target=_worker, daemon=True).start()


@app.get("/health", response_model=Health)
def health() -> Health:
    return Health(status="ok", model=_loaded["model"], dataset=_loaded["dataset"], warm=True)


@app.get("/models")
def models() -> list[dict[str, str]]:
    from vitreous.data import list_datasets

    model = os.environ.get("VITREOUS_MODEL", "vit_s16")
    return [{"dataset": name, "model": model} for name in list_datasets()]


@app.post("/analyze")
async def analyze(image: UploadFile, dataset: str = "eurosat") -> dict[str, str]:
    data = await image.read()
    if len(data) > MAX_UPLOAD:
        raise HTTPException(413, "image exceeds 10 MB")
    if not data:
        raise HTTPException(400, "empty upload")
    job_id = uuid.uuid4().hex[:12]
    job = Job(job_id, Path(tempfile.mkdtemp(prefix=f"vitreous-{job_id}-")))
    JOBS[job_id] = job
    job.emit("queued")
    WORK.put((job, data, dataset))
    return {"job_id": job_id}


@app.get("/jobs/{job_id}/events")
async def events(job_id: str) -> StreamingResponse:
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "no such job")

    async def stream():
        while True:
            try:
                msg = job.events.get_nowait()
            except queue_mod.Empty:
                await asyncio.sleep(0.15)
                continue
            yield f"data: {msg}\n\n"
            if json.loads(msg)["stage"] in ("done", "error"):
                return

    return StreamingResponse(stream(), media_type="text/event-stream")


@app.get("/jobs/{job_id}/pack/{asset}")
def pack_asset(job_id: str, asset: str) -> FileResponse:
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "no such job")
    root = job.workdir.resolve()
    path = (root / asset).resolve()
    if root not in path.parents or not path.is_file():
        raise HTTPException(404, "no such asset")
    return FileResponse(path)
