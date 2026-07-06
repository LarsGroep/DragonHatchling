# vitreous-live

FastAPI service (§14) that runs the same `vitreous` code as the Kaggle batch
notebooks, deployed on a Hugging Face Spaces free CPU instance. M0 ships only
`GET /health`; the analyze/jobs/pack routes land at M8.

## Run locally

```bash
pip install -e ../../packages/core
pip install -r requirements-dev.txt
uvicorn app.main:app --reload --port 7860
pytest
```

## Docker (HF Spaces)

Build from the **repo root** so `packages/core` is in context:

```bash
docker build -f apps/live/Dockerfile -t vitreous-live .
docker run -p 7860:7860 vitreous-live
```

The container listens on port 7860 (HF Spaces requirement).
