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
