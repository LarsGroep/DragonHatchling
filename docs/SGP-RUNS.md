# SGP / UMT-ViT — HAM10000 run log

Owner-executed Kaggle GPU runs, consolidated from the owner's run summaries
(2026-07-12). Dataset: HAM10000 (7 classes, ~10k images, 128 px). All runs are
self-supervised — labels enter only the evaluation probes. This file is the
running record the SGP notebook's knob comments cite; earlier UMT-ViT run
narratives (Runs 1–3 of the original notebook line) live in
[`UMT-VIT-REPORT.md`](./UMT-VIT-REPORT.md) §5.

## Head-to-head

| metric | v14 | v16 (best model) | SGP run 2 |
|---|---|---|---|
| notebook | inline `new.ipynb` | inline, path fixes | `kaggle_umtvit_sgp.ipynb` (packaged `umtvit` + `vitreous`) |
| SOM grid | 6×6×6 (216) | 6×6×6 (216) | 8×8×8 (512) |
| linear probe (chance 0.143) | 0.7700 | **0.7816** | — (eval step omitted) |
| k-NN (k=5, cosine) | 0.7384 | **0.7384** | — |
| trustworthiness (k=7) | 0.7424 | **0.7706** | — |
| SOM quantization error | 0.2329 | 0.2387 | **0.1994** (larger map — not directly comparable) |
| SOM topographic error | **0.0845** | 0.1147 | 0.1089 |
| dead fraction (training) | 0.1944 | **0.1852** | 0.4746 |
| SGP graph export | — | — | ✅ `sgp_ham10000.json` (0.29 MB) + 8 probe packs |
| wall time (30 epochs, T4) | ~1 h | ~1 h | ~3.1 h (larger SOM + per-epoch checkpoints) |

**Recommendation (owner + adopted as notebook defaults):** v16 is the best
model for downstream use. The SGP pipeline should train a **6×6×6** SOM at 30
epochs (or 8×8×8 at ~60+) to match v16's map health while keeping the graph
export.

## SGP run 2 — findings folded back into the repo

1. **`torch.fft.rfft2` crashed under mixed precision** ("not implemented for
   BFloat16"); the owner hot-patched `losses/_common.py` at runtime. Fixed at
   the source: `slice_power_spectrum` now upcasts to float32 before the FFT
   (autograd-transparent), with a bf16/fp16 regression test in
   `tests/test_losses.py`.
2. **512 neurons don't fill in 30 epochs** (dead 0.47 at the end; the epoch
   trace was still improving: 0.69 → 0.45 by epoch 20, TE 0.93 → 0.11). The
   notebook now defaults `SOM_GRID = (6, 6, 6)` and keys the checkpoint dir by
   grid so a grid change can't resume an incompatible checkpoint.
3. **No probe/k-NN numbers** — the SGP notebook didn't call the eval suite, so
   the run can't be ranked against v14/v16 on representation quality. The
   notebook now runs `umtvit.eval.run_evaluation` (`RUN_EVAL = True`) and
   stamps the metrics into the bundle's provenance.
4. **Probe-only hit counts overstated deadness** (0.686 "dead" from 8 probe
   images vs 0.475 measured in training). The notebook now measures
   `som.json` hits over a broad eval pass (`HITS_IMAGES = 256` images) so node
   sizing and dead flags reflect the dataset's use of the map.

Positives worth keeping: the packaged pipeline ran end-to-end from a fresh
clone, auto-resume checkpoints worked, QE was the best of the three runs, all
12 k-means communities were populated, and the exported bundle loaded in the
`/sgp` viewer as designed.
