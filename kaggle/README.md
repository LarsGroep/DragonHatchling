# kaggle/

Batch GPU notebooks (§2, §3). Thin shells around `packages/core` (`vitreous`)
— one code path, two venues (Kaggle + the HF Space).

Notebooks (landed at **M4**, §16):

- `train.ipynb` — fine-tune ViT-S/16 per dataset. Weights default to the Kaggle
  output dir; set `PUSH_TO_HF = True` (+ `HF_TOKEN`) to also push to the Hub.
- `precompute.ipynb` — curated gallery → Explanation Packs → dataset-level
  projections → upload via a `StorageAdapter` → emit the Postgres rows (§15).
- `sae.ipynb` — layer-9 token activations → k-sparse autoencoder → quality gate
  → concept dictionary (k-means fallback behind the same interface) → upload.

## Dataset swap = one string

Each notebook's first code cell is a **knobs** block whose leading line is
`DATASET = "eurosat"`. Change it to `"oxford_pet"` (or any registered adapter)
and re-run top-to-bottom — model head sizing, transforms, splits, gallery
selection, class names and colors all derive from the dataset adapter (§4).
`MODEL`, `SEED`, and storage/venue toggles sit in the same block.

## How the package is installed

The `pip install` cell installs `packages/core` (the `vitreous` package) from
this repo/branch — the same code path the live CPU service runs. On Kaggle,
enable Internet, or attach the repo as a dataset and `pip install -e` it.

## Credentials

Never hardcoded. `SUPABASE_URL` / `SUPABASE_SERVICE_KEY` (writes) and `HF_TOKEN`
come from Kaggle Secrets / env vars, read by `vitreous.storage`. Use
`STORAGE = "local"` for a fully offline dry-run (`file://` URLs).
