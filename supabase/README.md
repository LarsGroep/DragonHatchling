# ViTreous — Supabase (Postgres + Storage)

Metadata store and artifact host for the workbench (ARCHITECTURE.md §15). The
web app (M5) reads with the **anon key**; Kaggle notebooks write with the
**service-role key**.

## What lives here

- `migrations/0001_init.sql` — the schema: `datasets`, `models`,
  `gallery_images`, `projections`, `concept_dictionaries` (§15), plus
  public-read Row-Level Security policies.

Heavy artifacts do **not** live in Postgres — they go in public-read Storage
buckets; the tables hold metadata and the storage URLs/prefixes.

## Storage buckets (create these, all public-read)

| Bucket        | Contents                                      | Key layout |
|---------------|-----------------------------------------------|------------|
| `packs`       | Explanation Packs (one dir per image)         | `packs/{dataset}/{image_id}/…` |
| `projections` | dataset-level coord buffers + reducers (§10)  | `projections/{dataset}/{model}/…` |
| `concepts`    | concept dictionaries + exemplar assets (§9)   | `concepts/{model}/L{layer}/…` |
| `thumbs`      | gallery thumbnails                            | `thumbs/{dataset}/…` |

Create each in the dashboard (**Storage → New bucket → Public**) or via SQL:

```sql
insert into storage.buckets (id, name, public)
values ('packs','packs',true), ('projections','projections',true),
       ('concepts','concepts',true), ('thumbs','thumbs',true)
on conflict (id) do update set public = true;
```

Public buckets are GET-able (and **HTTP range-request** capable) without auth,
which is exactly what the frontend's streaming pack loader needs.

## Applying the migration

**Option A — Supabase CLI (recommended):**

```bash
supabase link --project-ref <your-project-ref>
supabase db push          # applies everything under supabase/migrations/
```

**Option B — SQL editor:** paste `migrations/0001_init.sql` into the dashboard
SQL editor and run it. It is idempotent (safe to re-run).

## Credentials (never commit these)

Read from the environment; the code never hardcodes them
(`vitreous.storage`):

| Variable               | Used by                | Notes |
|------------------------|------------------------|-------|
| `SUPABASE_URL`         | web app + notebooks    | `https://<ref>.supabase.co` |
| `SUPABASE_ANON_KEY`    | web app (public reads) | safe to ship to the browser |
| `SUPABASE_SERVICE_KEY` | Kaggle notebooks (writes) | **secret** — Kaggle Secrets only |

## RLS model

Every table has RLS **enabled** with a single `SELECT` policy for `anon` +
`authenticated`. There is intentionally **no** anon write policy: notebooks
write with the service-role key, which bypasses RLS. This makes the whole
gallery experience readable by a static frontend while keeping writes locked to
the batch pipeline.

## Emitting rows from the precompute notebook

`kaggle/precompute.ipynb` uploads packs via a `StorageAdapter` and then inserts
the corresponding `gallery_images` / `projections` / `concept_dictionaries`
rows (service key). See that notebook's final cells for the exact SQL / client
calls.

## Go-live runbook — HAM10000 (`kaggle/ham10000_live.ipynb`)

One notebook does the whole live run. It uses a **single public `packs`
bucket** (auto-created) and namespaces everything under it — packs at
`{dataset}/{id}/…`, plus `thumbs/…`, `projections/…`, `concepts/…`; non-pack
assets are referenced by full URL from the DB rows, so one bucket is enough.

1. **Create the tables once** — paste `migrations/0001_init.sql` into the
   Supabase SQL editor and run it (idempotent). DDL can't be done from the
   notebook, so this is the only manual DB step.
2. **Kaggle**: attach `kmader/skin-cancer-mnist-ham10000`, enable Internet +
   GPU (T4), add a `SUPABASE_SERVICE_KEY` secret (service-role key), *Run All*.
   The notebook trains ViT-S/16, publishes the gallery packs + projections
   (+ concept dictionary), and inserts every row.
3. **Vercel**: set `NEXT_PUBLIC_SUPABASE_URL` and `NEXT_PUBLIC_SUPABASE_ANON_KEY`
   (printed by the notebook's last cell) on the `apps/web` project and redeploy.
   The dataset switcher then lists **HAM10000** live.

Project URL and anon key are public; the service-role key stays in Kaggle
Secrets and is never committed.
