# ViTreous — quick start & deployment

The multi-view explainable ViT workbench. Design: [`ARCHITECTURE.md`](./ARCHITECTURE.md) ·
research: [`RESEARCH.md`](./RESEARCH.md) · decisions: [`DECISION-LOG.md`](./DECISION-LOG.md).

## Run the workbench locally (zero backend)

```bash
cd apps/web && npm install
NEXT_PUBLIC_VITREOUS_MOCK=1 npm run dev   # bundled fixture packs (EuroSAT + Pet)
```

Four synchronized views (image · Gaussian field · graph · embeddings), replay
transport (space / arrows / scrub), hover/click anywhere to sync everywhere.

## Full pipeline ($0 topology)

1. **Kaggle** (free GPU): run `kaggle/train.ipynb` (set `DATASET = "eurosat"` or
   `"oxford_pet"`), then `kaggle/precompute.ipynb` (gallery → Explanation Packs →
   Supabase upload), then `kaggle/sae.ipynb` (concept dictionary; k-means
   fallback auto-applies via the quality gate).
2. **Supabase**: apply `supabase/migrations/0001_init.sql` (see `supabase/README.md`);
   set service-key env vars only inside Kaggle.
3. **Vercel**: deploy `apps/web` with `NEXT_PUBLIC_SUPABASE_URL` +
   `NEXT_PUBLIC_SUPABASE_ANON_KEY` (omit both → mock mode).
4. **HF Spaces** (free CPU, live uploads): Docker Space from `apps/live/Dockerfile`;
   optional `VITREOUS_WEIGHTS` for a fine-tuned checkpoint. API: `POST /analyze` →
   `GET /jobs/{id}/events` (SSE stages) → `GET /jobs/{id}/pack/{asset}`.

## Dataset swapping

One string. `DATASET` in the notebooks; the web app lists whatever the
`datasets` table (or mock fixture) contains. New datasets = one adapter class
(`packages/core`, `@register_dataset`) — see `docs/ARCHITECTURE.md` §4.

## Test suites

```bash
.venv-m0/bin/python -m pytest packages/core/tests -q   # 157 tests
cd apps/live && ../../.venv-m0/bin/python -m pytest -q # service flow (no torch)
cd apps/web && npx vitest run && npx tsc --noEmit      # 42+ tests
```
