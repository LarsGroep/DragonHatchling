# Project review — 2026-07-12

Extensive review of the DragonHatchling repository at `c3d2397` (current
`main`). Scope: all four active components (`packages/core`, `packages/schema`,
`apps/web`, `apps/live`), the UMT-ViT experiment, the legacy hatchvision code,
CI, Supabase, Kaggle notebooks, and docs. Every test suite was executed in a
fresh environment mirroring CI.

## Verdict

The codebase is unusually well-crafted for a research project — disciplined
module boundaries, a frozen, schema-validated pack format mirrored across
JSON Schema / Pydantic / TypeScript, deterministic seeding everywhere, and
docstrings that tie code back to architecture sections. The web app is fully
green (typecheck, lint, 110 tests, production build).

**However, CI on `main` is red and has been for at least the last 8 pushes.**
The Python job fails in exactly the environment CI installs. Two small,
fixable causes (below). Beyond that, the biggest structural risk is that the
entire torch-dependent surface (core ML paths, UMT-ViT's ~139 tests, legacy
tests) is never exercised by CI.

## Test matrix (executed 2026-07-12, clean venv, CI-identical installs)

| Suite | Command | Result |
|---|---|---|
| packages/core (M0) | `pip install -e packages/core[dev]` + `pytest packages/core -q` | ❌ **6 failed**, 96 passed, 9 skipped |
| apps/live | `pip install -r apps/live/requirements-dev.txt` + `pytest -q` | ✅ 2 passed |
| packages/schema | `npm ci && npm run typecheck` | ✅ |
| apps/web typecheck | `npm run typecheck` | ✅ |
| apps/web tests | `npx vitest run` | ✅ 110 passed (13 files) |
| apps/web lint | `npm run lint` | ✅ |
| apps/web build | `npm run build` | ✅ (static, 3 routes) |
| experiments/umtvit | — | ⚠️ not runnable without torch; **not in CI** |
| legacy `tests/` (hatchvision) | — | ⚠️ needs torch; **not in CI** |

GitHub Actions confirms: run #79 on `main` (`c3d2397`) failed ~45 s in, as did
every sampled prior run — the failure signature matches the local repro.

## Critical findings

### 1. CI is red on `main` — two causes (P0)

**(a) `networkx` is used but not declared.**
`vitreous/graph.py:220` imports `networkx` for Louvain community detection,
but `packages/core/pyproject.toml` declares only `pydantic`, `numpy`,
`jsonschema` (+ `pytest` in `[dev]`). Five tests in
`packages/core/tests/test_graph.py` fail with `ModuleNotFoundError`.
Fix: add `networkx>=3` to the core dependencies (it is needed at runtime by
`build_graph_asset`, so it belongs in base deps, not an extra).

**(b) A torch-requiring assertion sits in the torch-free M0 suite.**
`packages/core/tests/test_imports.py:121` (`test_concepts_public_surface`)
touches `concepts.KSparseAutoencoder`, whose lazy module `__getattr__`
imports torch (`concepts.py:236`) — so the *import-purity* test file itself
violates the M0 guarantee it exists to enforce. Fix: guard that one
assertion with `pytest.importorskip("torch")` (or move it to an ML-gated
test).

Both fixes are a few lines; after them the CI-identical run should be green.

### 2. `apps/live`: fine-tuned-checkpoint path crashes (P1)

`apps/live/app/main.py:94`:

```python
loaded.model.load_state_dict(torch.load(weights, map_location="cpu"))
```

`LoadedModel` (`packages/core/src/vitreous/models.py`) has no `.model`
attribute — the module lives in `.module`. Any deployment that sets
`VITREOUS_WEIGHTS` raises `AttributeError` on the first analyze job. The
tests never catch it because they inject a fake analyzer. Fix:
`loaded.module.load_state_dict(...)`, plus one test that exercises the real
analyzer wiring with a monkeypatched loader.

### 3. `_register_builtins()` forgot `ham10000` (P2)

`packages/core/src/vitreous/data.py:391` restores only `eurosat`,
`oxford_pet`, `imagefolder` after a test clears the registry — the
`ham10000` adapter added later was not appended, reintroducing exactly the
test-order dependence this helper exists to prevent.

## Structural risks

### 4. The torch surface has zero CI coverage

CI installs `packages/core` *without* ML extras, so the 9 torch-gated core
tests always skip; `experiments/umtvit/tests` (~139 tests per the U7/N4
commit messages) and the legacy root `tests/` never run at all. Consequences
compound with finding 5. Recommendation: add a second (possibly
`schedule:`-triggered or manually-dispatched) CI job that installs
`packages/core[ml]` + `experiments/umtvit` on CPU torch and runs the full
matrix — CPU-only DeiT-S forward passes are perfectly feasible on a runner.

### 5. `Instrumenter` is coupled to timm's private internals

`instrument.py` recomputes attention from `module.qkv / q_norm / k_norm /
scale / head_dim`, and the grad path additionally relies on `attn_mod.norm`
and `attn_mod.attn_dim`. With `timm>=1.0` (no upper bound) a timm minor
release can silently change these attributes; because of finding 4, nothing
would catch it until a Kaggle run breaks. Recommendation: pin timm to a
tested minor (`timm>=1.0,<1.1`-style) or add an import-time capability check
with a clear error.

### 6. `apps/live` job lifecycle leaks

- `JOBS` and each job's `tempfile.mkdtemp` directory are never cleaned up —
  unbounded memory + disk on a long-lived HF Space.
- The SSE queue is single-consumer: a client that reconnects misses all
  prior events.
- The module docstring promises staged progress (`predict → attention →
  attributions → …`) but only `queued`/`predict`/`done` are ever emitted —
  `build_pack` receives no progress callback.

None of these block a portfolio demo; all three deserve a TODO or a small
eviction pass (e.g. drop jobs + workdirs N minutes after terminal state).

## Minor findings

- **`store.ts` race guard** (`apps/web/src/lib/state/store.ts:96`): the
  in-flight selection token is smuggled onto the store object via
  `(get() as unknown as { _sel?: string })` — works, but bypasses the typed
  state; a module-level variable would be cleaner and honest to the types.
- **Docs drift**: `docs/VITREOUS.md` claims 157 core tests (111 collect in
  the M0 environment) and "42+" web tests (there are 110); it also
  references a `.venv-m0` that is not part of the repo. Small, but this is
  the quick-start page.
- **CI duplication**: `on: push: branches: ["**"]` **plus** `pull_request`
  double-runs every PR commit; scoping push to `main` is the usual fix.
- **License inconsistency**: `packages/core` is MIT, `experiments/umtvit`
  says `Proprietary`, the root hatchvision `pyproject.toml` declares none.
  Worth a deliberate decision.
- **Repo weight**: an executed notebook with base64-embedded outputs is
  8.4 MB (`experiments/umtvit/notebooks/kaggle_umtvit_ham10000_run3.ipynb`),
  `webapp/vendor` carries ~12 MB of onnxruntime for the *legacy* app; pack
  size is ~22 MiB and growing. Consider stripping outputs from committed
  notebooks (keep results as small JSON/MD artifacts) and pruning or
  LFS-ing the legacy vendor blobs.
- **Stale remote branches**: ~10 `claude/*` work branches plus
  `backup/main-pre-vitreous` linger on the remote.

## Per-component notes

### packages/core — excellent

The strongest code in the repo. Highlights: torch-free import discipline
enforced by subprocess tests; the frozen pack format with per-row uint8
attention quantization and exact offsets recorded in the manifest; the
observation-only Instrumenter with the bit-identical-logits guarantee (and
the honest, clearly-separated unfused monkeypatch for Chefer gradients);
deterministic, stratified, group-aware splits (HAM10000's `lesion_id`
grouping prevents leakage); the Gaussian field's "honesty rule"
(every channel derived from a measured quantity, normalization constants
recorded). Faithfulness eval implements Spearman locally to avoid a scipy
dependency. Docstrings consistently reference architecture sections.

### packages/schema — solid

The TS mirror is small, well-commented, and drift is caught by a fixture
round-trip typecheck in CI. `AssetEncoding` admits `zstd`/`gzip` that
nothing writes yet — fine as reserved values since the schema is frozen.

### apps/web — solid

Clean layering (db → pack → state → views), injectable fetch, Range requests
with full-GET fallback, fp16/dequant decoding tested against fixtures
generated by the same rules as the Python writer. The three.js renderer is
careful (non-instanced by documented SwiftShader rationale, disposal
implemented, camera math isolated). The mock mode makes the whole app run
with zero backend — great for CI and demos.

### apps/live — adequate, with the bugs above

Simple and appropriately sized for its traffic model; the injected-analyzer
test seam is good. Fix the `.module` bug and add lifecycle cleanup.

### experiments/umtvit — good research code, unguarded by CI

Config-driven (validated schema), resumable trainer with full RNG-state
checkpoints, DESOM/Kohonen dual-mode SOM with revive + metrics, gated loss
terms. The engineering is above typical experiment quality; its test suite
just needs a CI venue (finding 4).

### Legacy (hatchvision, webapp/, notebooks/, scripts/, root tests/)

Clearly marked legacy in the README, which is the right call. It still owns
the repo root (`pyproject.toml`, `requirements.txt`, `tests/`), which makes
the first `pip install -e .` a legacy install — consider moving it under
`legacy/` or at least renaming the root project metadata, and slimming the
vendored onnxruntime.

### Supabase & Kaggle

The migration is idempotent, RLS-on with public SELECT policies and
service-role writes — correct for the stated access model. Kaggle notebooks
follow the "one string dataset swap" contract and keep credentials in
secrets/env.

## Prioritized recommendations

1. **P0 — make CI green**: add `networkx>=3` to `packages/core`
   dependencies; torch-gate the `KSparseAutoencoder` assertion in
   `test_imports.py`.
2. **P1 — fix `apps/live` weights path** (`loaded.module`), add a covering
   test; add `ham10000` to `_register_builtins`.
3. **P2 — CI job for the torch surface**: core `[ml]` tests + umtvit suite
   on CPU torch (scheduled or manual dispatch if runtime is a concern); pin
   or guard the timm version the Instrumenter depends on.
4. **P3 — hygiene**: live-service job eviction; refresh VITREOUS.md test
   counts; strip notebook outputs / prune legacy vendor blobs; delete stale
   remote branches; align licenses; scope CI push triggers to `main`.
