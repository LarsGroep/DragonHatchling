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

*Add subsequent runs above this line as new sections (most recent first
below Run 1), each with: config delta from the previous run, results table,
which guidance items were applied, and what changed.*
