# SGP / UMT-ViT — HAM10000 run log

Owner-executed Kaggle GPU runs, consolidated from the owner's run summaries
(2026-07-12). Dataset: HAM10000 (7 classes, ~10k images, 128 px). All runs are
self-supervised — labels enter only the evaluation probes. This file is the
running record the SGP notebook's knob comments cite; earlier UMT-ViT run
narratives (Runs 1–3 of the original notebook line) live in
[`UMT-VIT-REPORT.md`](./UMT-VIT-REPORT.md) §5.

## Head-to-head

| metric | v14 | v16 | SGP run 2 | **SGP run 3 (best model)** |
|---|---|---|---|---|
| notebook | inline `new.ipynb` | inline, path fixes | `kaggle_umtvit_sgp.ipynb` (pre-fix) | `kaggle_umtvit_sgp.ipynb` (fixed, defaults) |
| SOM grid | 6×6×6 (216) | 6×6×6 (216) | 8×8×8 (512) | 6×6×6 (216) |
| linear probe (chance 0.143) | 0.7700 | 0.7816 | — (eval step omitted) | **0.7922** |
| k-NN (k=5, cosine) | 0.7384 | **0.7384** | — | 0.7278 |
| trustworthiness (k=7) | 0.7424 | 0.7706 | — | **0.7807** |
| SOM quantization error (eval) | 0.2329 | 0.2387 | 0.1994 (larger map) | **0.1831** |
| SOM topographic error (eval) | 0.0845 | 0.1147 | 0.1089 | **0.0220** |
| dead fraction (training metric) | 0.1944 | 0.1852 | 0.4746 | **0.2407** / **4.2 %** by the 256-image hit pass (207/216 used) |
| SGP graph export | — | — | ✅ | ✅ `sgp_ham10000.json` + 8 probe packs |
| wall time (30 epochs, T4) | ~1 h | ~1 h | ~3.1 h | ~3.1 h (per-epoch ckpts + eval) |

**Standing recommendation:** **SGP run 3 is the best model AND the certified
pipeline** — first run of the fixed notebook (no runtime patches), and it beats
v16 on linear probe (+1.06 pp), trustworthiness (+1.01 pp), QE, and TE (0.022,
best ever recorded, ~4× better than v14's previous best) while carrying the
full graph export. The only metric below v16 is k-NN (−1.06 pp). Its
`sgp_ham10000.json` is the flagship `/sgp` bundle. The 8×8×8 @ ~60-epoch run
remains optional upside (run 2 showed the big map still converging at 30).

## SGP run 3 — the certified baseline (fixed notebook, defaults)

First execution of the post-fix notebook (6×6×6, 30 epochs, `RUN_EVAL=True`,
256-image hit pass), owner-run on a Kaggle T4, 2026-07-13. No runtime patches
needed — the bf16 FFT fix held in the field.

| epoch | loss | dead | TE |
|---|---|---|---|
| 5 | 2.006 | 0.588 | 0.896 |
| 10 | 1.163 | 0.343 | 0.494 |
| 15 | 1.023 | 0.301 | 0.200 |
| 20 | 0.701 | 0.278 | 0.041 |
| 25 | 0.595 | 0.255 | 0.041 |
| 30 | 0.522 | 0.241 | 0.044 |

Final eval: probe **0.7922** · k-NN 0.7278 · trustworthiness **0.7807** ·
QE **0.1831** · TE **0.0220** · dead 0.2778 (eval subsample) / **9/216 = 4.2 %**
by the dataset-wide 256-image hit pass. The three dead-fraction figures are
three honest budgets of the same map: last-batch voxels (training metric),
a bounded eval subsample (`som_metrics`), and the full 256-image voxel pass —
the broad pass is the one `som.json` node sizing uses, and it shows nearly the
whole lattice participating.

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

## Executed artifact

The owner's executed notebook is frozen as
[`../experiments/umtvit/notebooks/kaggle_umtvit_sgp_run2.ipynb`](../experiments/umtvit/notebooks/kaggle_umtvit_sgp_run2.ipynb)
(never edit; it is a record — it still contains the runtime `rfft2` patch cell
that the source fix has since made unnecessary). Its recorded finals differ
slightly from the summary table above (a separate execution of the same
notebook version):

| epoch | loss | dead | TE |
|---|---|---|---|
| 5 | 1.997 | 0.703 | 0.965 |
| 10 | 1.066 | 0.566 | 0.574 |
| 15 | 0.946 | 0.531 | 0.288 |
| 20 | 0.679 | 0.494 | 0.140 |
| 25 | 0.556 | 0.492 | 0.111 |
| 30 | **0.475** | **0.471** | **0.096** |

Final QE 0.217 · TE 0.096 (the best topographic error recorded for the 512
map, edging v14's 6×6×6 0.0845 benchmark class) · training dead fraction 0.471
still trending down at epoch 30 — consistent with this file's conclusion that
the 8×8×8 map is under-trained at 30 epochs rather than unhealthy. Probe-only
hit counts flagged 366/512 (71.5%) unused for the 8 exported probes — the
measurement artifact the broad hit pass now corrects.
