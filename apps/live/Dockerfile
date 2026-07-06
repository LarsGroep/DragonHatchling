# ViTreous live service — Hugging Face Spaces (free CPU, Docker SDK).
#
# Build context is the REPO ROOT (so packages/core is visible):
#   docker build -f apps/live/Dockerfile -t vitreous-live .
#
# HF Spaces requires the app to listen on port 7860.

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=7860

WORKDIR /app

# Install the core package first (its own dep set) for better layer caching.
COPY packages/core /app/packages/core
RUN pip install --upgrade pip && pip install /app/packages/core

# Live service deps + source.
COPY apps/live/requirements.txt /app/apps/live/requirements.txt
RUN pip install -r /app/apps/live/requirements.txt

COPY apps/live/app /app/apps/live/app

WORKDIR /app/apps/live

EXPOSE 7860

# Single worker: portfolio traffic, small in-memory job queue (§14).
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
