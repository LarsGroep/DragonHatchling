-- ViTreous — initial Postgres schema (ARCHITECTURE.md §15).
--
-- Metadata for the workbench: datasets, fine-tuned models, curated gallery
-- images, dataset-level latent projections, and per-model concept dictionaries.
-- Heavy artifacts (Explanation Packs, projection coordinate buffers, concept
-- exemplar assets) live in public-read Supabase Storage buckets; these tables
-- hold the metadata + the storage URLs/prefixes the web app reads.
--
-- Access model (§15): the web app reads with the anon key; Kaggle notebooks
-- write with the service-role key. Row-Level Security is ON for every table
-- with a public SELECT policy; writes are performed by the service role, which
-- bypasses RLS, so no INSERT/UPDATE/DELETE policy is granted to anon.
--
-- Idempotent: safe to re-run (IF NOT EXISTS / DROP POLICY IF EXISTS).

create extension if not exists "pgcrypto";  -- gen_random_uuid()

-- --------------------------------------------------------------------------- --
-- datasets
-- --------------------------------------------------------------------------- --
create table if not exists public.datasets (
  id            uuid primary key default gen_random_uuid(),
  name          text not null unique,          -- registry key, e.g. 'eurosat'
  display_name  text not null,
  spec          jsonb not null default '{}'::jsonb,  -- serialized DatasetSpec (§4)
  created_at    timestamptz not null default now()
);

-- --------------------------------------------------------------------------- --
-- models
-- --------------------------------------------------------------------------- --
create table if not exists public.models (
  id          uuid primary key default gen_random_uuid(),
  dataset_id  uuid not null references public.datasets (id) on delete cascade,
  arch        text not null,                   -- e.g. 'deit_small_patch16_224'
  hf_repo     text,                            -- weights location (may be null)
  metrics     jsonb not null default '{}'::jsonb,  -- {top1, top5, loss, ...}
  created_at  timestamptz not null default now()
);
create index if not exists models_dataset_id_idx on public.models (dataset_id);

-- --------------------------------------------------------------------------- --
-- gallery_images
-- --------------------------------------------------------------------------- --
create table if not exists public.gallery_images (
  id           uuid primary key default gen_random_uuid(),
  dataset_id   uuid not null references public.datasets (id) on delete cascade,
  model_id     uuid not null references public.models (id) on delete cascade,
  class_label  text,                           -- ground-truth class name
  pred_label   text,                           -- model prediction
  confidence   real,                           -- top-1 probability
  pack_prefix  text not null,                  -- storage prefix: packs/{dataset}/{image_id}/
  thumb_url    text,                           -- public thumbnail URL
  tags         text[] not null default '{}',
  created_at   timestamptz not null default now()
);
create index if not exists gallery_images_dataset_id_idx on public.gallery_images (dataset_id);
create index if not exists gallery_images_model_id_idx   on public.gallery_images (model_id);

-- --------------------------------------------------------------------------- --
-- projections  (dataset-level latent embeddings, §10)
-- --------------------------------------------------------------------------- --
create table if not exists public.projections (
  id          uuid primary key default gen_random_uuid(),
  dataset_id  uuid not null references public.datasets (id) on delete cascade,
  model_id    uuid not null references public.models (id) on delete cascade,
  layer       int  not null,                   -- probe layer ∈ {0,3,6,9,12}
  method      text not null,                   -- 'umap' | 'pca' | 'tsne'
  url         text not null,                   -- fp16 [N,2] coord buffer URL
  reducer_url text,                             -- persisted reducer (.joblib); null for t-SNE
  created_at  timestamptz not null default now()
);
create index if not exists projections_dataset_id_idx on public.projections (dataset_id);
create unique index if not exists projections_unique_idx
  on public.projections (dataset_id, model_id, layer, method);

-- --------------------------------------------------------------------------- --
-- concept_dictionaries  (per model+layer SAE / k-means dictionary, §9)
-- --------------------------------------------------------------------------- --
create table if not exists public.concept_dictionaries (
  id         uuid primary key default gen_random_uuid(),
  model_id   uuid not null references public.models (id) on delete cascade,
  layer      int  not null,                    -- SAE probe layer (default 9)
  url        text not null,                    -- dictionary artifact URL (JSON)
  quality    jsonb not null default '{}'::jsonb,  -- QualityReport (§9): dead rate,
                                                --   coherence, dup rate, use_sae
  created_at timestamptz not null default now()
);
create index if not exists concept_dictionaries_model_id_idx on public.concept_dictionaries (model_id);

-- --------------------------------------------------------------------------- --
-- Row-Level Security: public read, service-role write (§15)
-- --------------------------------------------------------------------------- --
alter table public.datasets             enable row level security;
alter table public.models               enable row level security;
alter table public.gallery_images       enable row level security;
alter table public.projections          enable row level security;
alter table public.concept_dictionaries enable row level security;

-- Public SELECT for the anon (and authenticated) roles. Writes require the
-- service-role key, which bypasses RLS entirely — so no write policy is needed.
drop policy if exists "public read datasets"             on public.datasets;
drop policy if exists "public read models"               on public.models;
drop policy if exists "public read gallery_images"       on public.gallery_images;
drop policy if exists "public read projections"          on public.projections;
drop policy if exists "public read concept_dictionaries" on public.concept_dictionaries;

create policy "public read datasets"
  on public.datasets for select to anon, authenticated using (true);
create policy "public read models"
  on public.models for select to anon, authenticated using (true);
create policy "public read gallery_images"
  on public.gallery_images for select to anon, authenticated using (true);
create policy "public read projections"
  on public.projections for select to anon, authenticated using (true);
create policy "public read concept_dictionaries"
  on public.concept_dictionaries for select to anon, authenticated using (true);
