# UMT-ViT — Notebook Run Feedback

Consolidated feedback on owner-executed runs of
`experiments/umtvit/notebooks/kaggle_umtvit.ipynb`. Companion to
[`UMT-VIT-ARCHITECTURE.md`](./UMT-VIT-ARCHITECTURE.md) (design contract) and
[`UMT-VIT-DECISION-LOG.md`](./UMT-VIT-DECISION-LOG.md) (milestones). Add a
section per notable run; keep tuning guidance and its outcome together so
the experiment record stays in one place.

---

## Run 1 — HAM10000, Kaggle GPU, 2026-07-11

First full-scale run (preset `ham10000`: 128 px, dim 256, L = 8 depth,
volume 16×16×8×64, SOM 8×8×8, 30 epochs; wall time ≈ 75 min).

### Results

| Metric | Value | Reference |
|---|---|---|
| Linear probe accuracy | **0.768** | chance 0.143 (7 classes) |
| k-NN (k=5, cosine) | **0.730** | chance 0.143 |
| SOM quantization error | 0.243 | lower = better |
| SOM topographic error | **0.008** | lower = better |
| SOM dead-neuron fraction | **0.977** | ⚠ see below |
| Trustworthiness (k=7) | 0.759 | 1.0 = perfect |
| Spectral centroids (z = 0…7) | 0.125, 0.194, 0.118, 0.102, 0.126, 0.162, 0.155, 0.107 | ⚠ non-monotone |

### What is working

- **Strong label-free signal.** Probe at 0.768 and k-NN at 0.730 against
  0.143 chance means the SSL objectives learned genuinely discriminative
  dermoscopy features — no labels touched training. (These are frozen-feature
  SSL yardsticks; do not compare directly against supervised end-to-end
  results such as DSCATNet's 97.8%.)
- **Topology preservation is near-perfect** (TE 0.008): the voxels' nearest
  and second-nearest SOM neurons are almost always grid neighbors — the map
  that *is* used is genuinely topographic.

### Issue 1 — SOM under-utilization (dead-neuron fraction 0.977)

Only ~12 of 512 neurons ever win a best-matching-unit assignment: the latent
volume's voxels collapse onto a tiny corner of the SOM. QE and TE look good
partly *because* so few neurons compete.

Likely causes: the σ anneal ends too tight (`sigma_end 0.5` on an 8×8×8
grid), 512 neurons vs. limited voxel diversity, and the SOM loss weight
(0.5) being dominated once NT-Xent shapes the space.

**Tuning guidance for the next run, in priority order:**
1. Slower/looser neighborhood anneal: `loss.sigma_end: 1.0` (first choice).
2. Smaller grid: `model.som_grid: [6, 6, 6]` (216 neurons).
3. Raise `loss.som` toward 1.0 if 1–2 don't spread usage.

Watch dead-fraction per epoch in the training log — it should fall through
training, not rise. Revisit systematically in the U5 ablation matrix.

### Issue 2 — spectral centroids non-monotone (the Z-axis question)

The centroid sequence rises at z=1 (0.194), dips through z=3 (0.102), rises
again mid-depth (0.162), and only the deepest slice (0.107) lands where the
hierarchy predicts. This is the outcome the research record flagged as the
experiment's central open question (RESEARCH §3): depth-ordering was never
going to emerge for free. Three compounding reasons:

1. **The ordering regularizer is one-sided.** It penalizes power *above* a
   depth-decreasing cutoff — it forbids deep slices from being sharp but
   never requires shallow slices to *be* sharp; low-frequency content is
   free at every depth. The deep end behaving (0.107) while z=1 spikes to
   0.194 is fully consistent with the loss doing exactly — and only — what
   it says. A monotone *upper envelope* was enforced, a monotone *centroid*
   never was.
2. **The smoothness loss is an antagonist.** `L_smooth` takes total
   variation along H, W **and Z**, so it actively pulls adjacent depth
   slices toward each other — directly opposing depth differentiation. At
   λ_smooth = λ_order = 0.1 the two roughly cancel mid-volume. The residual
   stream compounds this (Raghu et al.: ViT representations are uniform
   across depth).
3. **Measurement caveat.** The centroid probe runs on the *channel-mean* of
   each slice, which can cancel per-channel high-frequency structure; a
   mean of per-channel centroids is the fairer measurement.

**Options for the next iteration, cheapest first:**
1. Exclude the Z-axis from `L_smooth` (keep H/W terms) — removes the direct
   antagonist; one-line change.
2. Make the ordering constraint two-sided: add a low-frequency penalty on
   shallow slices (per-depth band-pass target) so early layers must carry
   detail, not merely be allowed to.
3. Raise `loss.order` to 0.3–0.5 and/or use a convex cutoff schedule
   `f(l) = f_max · (1 − l/L)^γ`, γ ≈ 2, so the constraint bites earlier.
4. Most direct: a differentiable monotonicity penalty on per-slice spectral
   centroids, `Σ_l relu(c_{l+1} − c_l)`.
5. Fix the probe: per-channel centroids, then average.

**Honest-result framing:** the run is also reportable exactly as measured —
*"one-sided frequency regularization at λ = 0.1 does not induce monotone
scale ordering in a residual ViT"* is a legitimate finding of the experiment,
not a failure. Whatever tuning is tried next, both outcomes stay in this
file.

### Web explorer note

This run predates the notebook's "Export web bundle" cell (added in V1). To
explore a run at `/umtvit` on the Vercel app, re-run the updated notebook
and drag the produced `umtvit_web.json` onto the page.

---

## Run 2 — HAM10000, Kaggle GPU, 2026-07-11 (owner notebook upload)

Config delta from Run 1: `som_grid [6,6,6]` (216 neurons), slower/wider σ
anneal (≈5.7 → 1.5), 30 epochs, manual checkpoint save added by owner.
Applied guidance: Run-1 items 1–2 (σ + grid). Wall time ≈ 73 min.

### Results

| Metric | Run 2 | Run 1 | Reading |
|---|---|---|---|
| Linear probe | **0.774** | 0.768 | slightly better |
| k-NN (k=5, cosine) | **0.743** | 0.730 | slightly better |
| SOM quantization error | 0.205 | 0.243 | better (but see below) |
| SOM topographic error | **0.973** | 0.008 | **collapsed** |
| SOM dead-neuron fraction | **0.991** | 0.977 | **collapsed** (~2/216 alive) |
| Trustworthiness (k=7) | 0.766 | 0.759 | flat |
| Spectral centroids | 0.161, 0.135, 0.131, 0.130, 0.136, 0.136, 0.131, 0.135 | non-monotone | z=0 sharpest, then flat |

### Diagnosis

- **SOM collapse, root cause σ_start ≈ 5.7 > grid radius.** With the
  neighborhood wider than the entire 6³ grid, every neuron was pulled toward
  the global voxel mean from step one (epoch log: TE ≈ 1.0 from epoch 2);
  the map never differentiated and QE looks good only because ~2 neurons
  quantize everything. Schedule tweaks alone cannot fix this —
  **structural fixes needed**: data-driven weight init, dead-neuron
  revival, and σ defaults derived from the grid (σ_start = max(grid)/2).
- **Centroids**: z=0 now correctly sharpest; the flat tail (≈0.13) is the
  smoothness Z-term homogenizing depth, exactly as predicted in Run 1
  option 1 (exclude Z from `L_smooth`).
- **Viz bug found in the run**: matplotlib warning `handles=1 vs labels=7` —
  the embedding subset takes the *first* 400 eval-train items and HAM10000's
  CSV ordering makes that nearly single-class (biased animation + broken
  legend). Fix: seeded random subset + explicit per-class legend handles.
- **Cosmetic**: the eval banner hardcodes "short CPU smoke run" on GPU runs.

### Actions (work order N2, implemented in the notebook)

Config-flagged: `som_init: data` + `som_revival: true` + grid-derived σ
defaults (`sigma_start/end: null`); `smooth_axes: [h, w]` (Z excluded);
monotone-centroid penalty `order_monotone: 0.05`; per-channel centroid
probe (old metric printed once alongside for comparability); seeded random
embedding subset + fixed legend; formalized checkpoint save/load;
conditional eval banner. Web-bundle schema v1 unchanged.

#### Post-run fix (U2b): cls_bridged cross-attention was inert

**What was wrong.** Each cross-scale round ran `per-stream self-attn →
cross-attention`. In `cls_bridged` mode the cross step writes *only* to each
stream's CLS token, and fusion then drops the CLS tokens. With the default
`cross_rounds: 1` (used by every preset, including ham10000/eurosat), the
CLS-bridge information therefore never reached the patch tokens: the
cross-scale exchange was functionally inert and the cross-attention
parameters received a strictly zero gradient. The model degenerated to two
independent streams plus sum fusion. (CrossViT avoids this because a
*subsequent* self-attention block redistributes the re-injected CLS into the
patch tokens.)

**Implication for runs 1–2.** Both runs effectively trained *without any
cross-scale exchange*. Their metrics remain valid — but specifically as a
**no-cross-attention baseline**: useful ablation data isolating the
dual-stream-plus-fusion contribution from the cross-scale bridge.

**The fix.** Reorder each round to `cross-attention → per-stream self-attn`,
so the self-attn spreads the updated CLS into the patch tokens *before*
fusion drops the CLS. This makes every round live for any `cross_rounds`
(purely an ordering change — no new blocks or parameters). `full_pair` is
unaffected by the ordering but uses the same order for consistency. Applied
to both the notebook `UMTViT.forward` and the package backbone.

**From run 3 onward** the cross-scale bridge is live and contributes to the
representation.

---

## Run 3 pre-flight (N3) — preset route reproduced on CPU + GPU/AMP hardening

Before the owner's next Kaggle GPU run we reproduced the `ham10000` preset
route locally and hardened the paths a CPU cannot exercise. The owner hit
errors on Kaggle (T4 GPU, AMP) but did not capture the message, so this pass
is split: **reproduce what CPU can run**, **audit what it cannot**.

### Preset route reproduced locally (CPU) — clean

Synthetic HAM10000-layout dataset (160 128 px JPEGs across
`HAM10000_images_part_1/2`, a 7-class `dx` column, grouped `lesion_id`), config
edited exactly as the owner does — `CONFIG = apply_preset("ham10000")` plus a
dataset-path override — with **only run-budget values shrunk** (epochs 2,
batch 16, `embed_subset` 120, `som_sample_voxels` 1024). **Model scale stayed
at the preset** (dim 256, depth 8, image 128 px, volume 16²×8×64, SOM 6×6×6,
6.55 M params) so the preset's real code paths executed. Result: the full
preset route ran **end-to-end clean on CPU** (~55 s). No CPU-reproducible
crash exists — which localizes the owner's failure to a **GPU/AMP-only path**
(the T4 has no bf16, so AMP selects fp16).

### GPU/AMP hardening applied (the likely Kaggle killers)

- **`torch.fft.rfft2` under fp16 autocast** — `ordering_loss` and
  `monotone_centroid_loss` FFT the latent volume, which is fp16 inside a T4
  autocast context; cuFFT rejects half precision (and imposes power-of-two size
  constraints), throwing at runtime. Both now cast `V = V.float()` before the
  transform. *(Most probable cause of the owner's crash — this is the only path
  that raises rather than merely degrading.)*
- **SOM math under autocast** — `Soft3DSOM.loss / bmu / metrics / data_init /
  revive` now run inside `torch.autocast(enabled=False)` with explicit `.float()`
  casts. Under autocast `torch.cdist` and the Hebbian `lerp_` would otherwise
  downcast the fp32 weight buffer to fp16, corrupting BMU assignment and the EMA
  write. Inputs were already `.float()`ed in the loop; this closes the weight-op
  gap.
- **GradScaler skipped-first-step** — AMP may skip the first optimizer step
  while it calibrates the loss scale (inf/NaN → scale halved, step skipped)
  while `sched.step()` still runs, emitting PyTorch's "scheduler before
  optimizer" warning. The loop now advances the LR schedule only when the step
  actually landed (`scaler.get_scale() >= prev_scale`); constant on CPU, so
  behaviour there is unchanged.
- **`torch.load` on resume** — the commented resume snippet now passes
  `weights_only=False`. torch 2.6+ flipped that default to `True`, whose
  restricted unpickler rejects this checkpoint (it stores the `CONFIG` dict +
  history, not a bare tensor state-dict); Kaggle images ship recent torch.
- **numpy 2.x vs Kaggle 1.26** — audited; the only at-risk call is `np.ptp`,
  used in function form (`np.ptp(s)`), which is valid on both. No 2.x-only API
  is generated.

Both routes (default `shapes` and `ham10000` preset) re-execute clean after
the hardening; the shapes run regenerated `apps/web/public/umtvit/demo.json`
(schema v1, validates against `apps/web/src/lib/umtvit.ts`).

**If it still fails on Kaggle:** export the failing cell's error text
(`Cell → Run`, copy the traceback tail) and attach it — the remaining
GPU-only causes are indistinguishable without the actual message.

---

*Add subsequent runs above this line as new sections (most recent last),
each with: config delta from the previous run, results table, which
guidance items were applied, and what changed.*
