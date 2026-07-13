# Malignancy lens — a toggle-able clinical reading for ViTreous

**Design contract · v0.1 (proposal) · 2026-07-13**

A new **lens toggle** in the ViTreous workbench that reads a HAM10000 pack three
honest ways beyond the raw 7-way diagnosis:

1. **Malignant vs benign** — a deterministic grouping of the diagnosis softmax.
2. **A benign → in-situ → invasive *category* axis** — the honest stand-in for
   "progression" (a coarse biological ordering of the diagnosis, **not** clinical
   AJCC/TNM staging).
3. **An unsupervised *manifold-position* scalar** — where the lesion sits on the
   learned benign↔malignant axis in SSL feature space, with an **off-manifold**
   gate that doubles as the out-of-distribution refusal for phone uploads.

The lens is **additive and non-breaking**: `pack_version` stays `1.0.0`, the
frozen `prediction` field is untouched, the selection-sync model (§11) is
untouched. It follows the exact pattern of the concepts tier and SGP —
derived readouts + one small optional asset.

## 1. The honesty contract (non-negotiable, medical context)

| We compute | We render it as | We NEVER call it |
|---|---|---|
| `Σ P(class)` over the malignant group | "model's malignancy probability" | "diagnosis" / "result" |
| argmax/expected level over {benign, in-situ, invasive} | "lesion category axis" | "cancer stage" / "Stage I–IV" |
| projection onto the SSL benign↔malignant axis | "position on the learned manifold" | "how advanced the cancer is" |
| residual distance off that axis | "outside training distribution" | anything reassuring |

Clinical stage (Breslow depth, ulceration, nodal/metastatic status) is **not
present in a dermatoscopic surface image** and is never emitted. Every lens
readout traces to a measurement (a softmax sum, a class grouping, or a
feature-space projection) — the ViTreous honesty rule. The workbench keeps its
"explainability demo, not a diagnostic tool" framing; the lens makes that
framing *more* honest by adding an explicit OOD refusal.

## 2. The taxonomy (zero manual labels — a grouping of the existing `dx`)

One source of truth, declared on the dataset. HAM10000's 7 diagnoses group as:

| dx | class | malignant | category axis (0→2) |
|---|---|---|---|
| nv | melanocytic nevi | benign | 0 benign |
| bkl | benign keratosis | benign | 0 benign |
| df | dermatofibroma | benign | 0 benign |
| vasc | vascular | benign | 0 benign |
| akiec | actinic keratosis / in-situ SCC | malignant | 1 in-situ / premalignant |
| bcc | basal cell carcinoma | malignant | 2 invasive |
| mel | melanoma | malignant | 2 invasive |

This grouping is **confirmed against the dermatology literature** (§9), not a
guess, and it is the *medically correct* one — with two cautions the research
surfaced:

- **Do NOT copy the popular Kaggle split.** A widely-reproduced HAM10000 binary
  split reports "benign 6705 / malignant 3295", which is effectively *nevi vs.
  everything else* and, in some copies, literally files **BCC (a carcinoma) as
  benign**. That is clinically wrong; we use the correct grouping above
  (malignant = mel + bcc + akiec = 1954 images; benign = 8061; ≈19.5%
  malignant).
- **The in-situ rung is real but keratinocyte-specific.** `akiec` (actinic
  keratosis / Bowen's disease) is genuinely *pre-invasive*: AK is the most
  common precursor of cutaneous SCC (progression ≈10% over 10 years; Bowen's ≈5%
  to invasive), so `benign → akiec (in-situ) → invasive SCC` is a textbook
  pathway (§9). `mel` and `bcc` have no separate in-situ class in HAM10000, so
  they sit at level 2 as "invasive malignancy"; only `akiec` truly occupies the
  in-situ rung. The UI copy should say exactly that rather than implying every
  malignancy passed through a labelled in-situ stage.

## 3. Where it lives (core = one source of truth)

`vitreous.data.DatasetSpec` gains an **optional** `taxonomy` field
(default `None`, so every other dataset and all existing tests are unaffected):

```python
@dataclass(frozen=True)
class Taxonomy:
    """Honest, label-free groupings of a dataset's classes (§ malignancy lens)."""
    # class_name -> True/False; malignant group for the binary readout.
    malignant: Dict[str, bool]
    # class_name -> ordinal category level (0..K-1) + the level labels.
    category_level: Dict[str, int]
    category_labels: List[str]      # e.g. ["benign", "in-situ", "invasive"]
    axis_pair: Tuple[str, str] = ("benign", "malignant")  # manifold endpoints
```

`HAM10000Adapter.spec` sets it; the CSV/label paths don't change. Because the
taxonomy is a pure function of `class_names`, the **web can derive axes 1 & 2
with no new pack asset** — it just needs the grouping, delivered either in the
mock `datasets.json` / Supabase `datasets.spec` (already a jsonb blob) or copied
into the pack via a tiny additive `taxonomy` key on an asset's `meta`.

## 4. The three readouts

### 4.1 Malignancy probability (axis 1) — derived, web-side, free
`P(malignant) = Σ_{c ∈ malignant} prediction.probabilities[c]`. No asset, no
retraining. (A dedicated trained binary head is a possible v2 for better
calibration — see §8 — but the derived readout is honest and immediate.)

### 4.2 Category axis (axis 2) — derived, web-side, free
Two honest presentations of the same grouping:
- **hard**: `level(argmax class)` → one of {benign, in-situ, invasive}.
- **expected** (smoother, preferred): `Σ_c P(c) · level(c)` → a continuous
  `[0, 2]` "category coordinate" that moves with the softmax mass. Shown as a
  3-stop track, not a number pretending to be a stage.

### 4.3 Manifold position (axis 3) — one small dataset-level asset
The only new artifact: **`malignancy_axis.json`** (per dataset+model, ~a few KB),
built at precompute time from label-free features:

```jsonc
{
  "provider": "malignancy_axis",
  "space": "cls_final",           // which feature: final-step CLS of tokens.bin
  "dim": 384,
  "u": [ ... ],                   // unit axis c_M - c_B in feature space
  "anchor_lo": 0.14,             // benign projection anchor (5th pct of benign)
  "anchor_hi": 0.86,             // malignant projection anchor (95th pct of malignant)
  "centroid_benign": [ ... ],    // for the off-manifold residual
  "residual_p95": 0.42,          // off-manifold threshold (95th pct train residual)
  "provenance": { "dataset": "ham10000", "n_benign": 0, "n_malignant": 0 }
}
```

Per image, entirely from data already in the pack:
```
f  = tokens[final_step][CLS]                 # from tokens.bin (already shipped)
s  = clamp((dot(f, u) - anchor_lo) / (anchor_hi - anchor_lo), 0, 1)   # position
r  = ||f - proj_onto_axis(f)||               # off-manifold residual
ood = r > residual_p95                        # honest refusal flag
```
`s` is the unsupervised "how far along the learned benign→malignant manifold"
coordinate; `ood` is the phone-photo / OOD gate. Both trace to a projection — a
measurement — so the honesty rule holds. Builder + math live in
`vitreous/malignancy.py` (numpy-only, unit-tested like `vitreous/som.py`);
precompute writes the asset next to the pack (dataset-level, shared).

## 5. The toggle (presentation only — no sync change)

`useWorkbench` gains `lens: "diagnosis" | "malignancy" | "category" | "manifold"`
(default `"diagnosis"` — nothing changes until you switch), alongside the
existing `mode: plain | expert`. A segmented control in `WorkbenchHeader` flips
it. What each lens changes is **only colour + the verdict readout**:

- **diagnosis** (today): class colours, top-class verdict.
- **malignancy**: two-tone (benign teal / malignant amber-red), verdict =
  `P(malignant)` with the honest confidence caveat; OOD → "outside distribution".
- **category**: 3-stop benign→in-situ→invasive track along the verdict; brain
  nodes tinted by the winning level.
- **manifold**: the embedding view / SGP lattice recoloured by `s` (a benign→
  malignant ramp), the off-manifold region greyed — the "progression" picture.

The BrainView verdict blend (`verdictProgress(t)`) already fades to green as
evidence forms; under a lens it fades to that lens's colour and resolves to that
lens's label. No view learns a new data source it didn't already read.

## 6. Data flow

```
precompute (Kaggle, has features + labels for the grouping)
  ├─ per gallery image: pack as today (prediction.probabilities unchanged)
  └─ once per dataset+model: extract CLS features → benign/malignant centroids
       → vitreous.malignancy.build_axis(...) → malignancy_axis.json  (upload)
apps/web
  ├─ taxonomy from datasets.spec (mock/Supabase) — axes 1 & 2 (free, web-side)
  └─ malignancy_axis.json (range-fetched once) + tokens.bin CLS — axis 3 + OOD
      ⇄ lens toggle recolours verdict / brain / embedding / SGP
```

## 7. Milestones

| M | Deliverable | Acceptance |
|---|---|---|
| L0 | This contract reviewed; §2 cutpoints + §8 decisions confirmed | owner sign-off |
| L1 | `Taxonomy` on `DatasetSpec` + HAM10000 grouping; `vitreous/malignancy.py` (axis build, projection, OOD) + tests | numpy-only; grouping is a pure fn of class_names; projection recovers planted axis on synthetic features |
| L2 | Web: `taxonomy` in db types + `lib/malignancy.ts` (derive axes 1&2 from probabilities; decode axis 3 from tokens CLS + axis asset) + tests | derived readouts match a fixture; OOD flag fires on a far feature |
| L3 | `lens` store field + `WorkbenchHeader` toggle + recolour in Brain/Image/Embedding verdict; honesty microcopy | diagnosis lens pixel-identical to today; lenses recolour; OOD refuses |
| L4 | precompute cell emits `malignancy_axis.json`; mock fixture gains a taxonomy + axis so `/` demos the lens with no backend | fixture round-trips; lens works in mock mode |

L1–L3 need no GPU and no new training run — they read models you already have.

## 8. Decisions — resolved by research (§9)

The owner is not a dermatology expert and asked these to be settled from the
literature. Resolved recommendations (all reversible — each is a lookup table or
a config field):

1. **Category cutpoints — RESOLVED.** Use the medically-correct grouping in §2:
   malignant `{mel, bcc, akiec}`, benign `{nv, bkl, df, vasc}`; ordinal
   `benign(0)` → `akiec` in-situ `(1)` → `{mel, bcc}` invasive `(2)`. Reject the
   common Kaggle "bcc-as-benign" split. **Optional secondary "urgency" toggle:**
   because melanoma is by far the deadliest, a variant that isolates `mel` at the
   top is worth offering as a second colouring — cheap (another lookup) and
   clinically meaningful.
2. **Malignancy readout — RESOLVED for v1: derived, high-sensitivity.** Ship the
   softmax-sum readout (free, honest). Given the ~19.5% malignant imbalance and
   that a missed melanoma is the worst possible error, expose an **adjustable
   decision threshold defaulting to a high-sensitivity operating point** (flag
   "possible malignancy" well below 0.5) rather than a plain argmax — this
   matches how the field treats the asymmetric cost (§9). A trained binary head
   stays a v2 calibration option.
3. **Manifold feature space — RESOLVED for v1: supervised CLS, SSL as a study.**
   Default to the final-step CLS from the supervised pack's `tokens.bin` (ships
   today; the axis is meaningful because the model was trained to separate these
   classes). Keep the UMT-ViT SSL pooled feature selectable via `space` — and
   note it as a genuinely interesting `/sgp` experiment: *does the label-free
   manifold recover the same benign→malignant axis the supervised model learned?*
   That question is on-brand and doesn't block v1.

## 9. Evidence base

- **Malignant grouping & the Kaggle-split caution** — the HAM10000 source paper
  and multiple classification studies; malignant = melanoma + BCC + AK/intra-
  epithelial carcinoma is the clinically standard split (benign = nv, bkl, df,
  vasc). Tschandl et al., *Sci Data* 2018; SkinNet-16, *Front. Oncol.* 2022.
- **`akiec` as pre-invasive / the in-situ rung** — actinic keratosis is the
  commonest precursor of cutaneous SCC (≈10%/10 yr progression); Bowen's disease
  is SCC in-situ (≈5% → invasive). dermoscopedia "AK/Bowen's/SCC"; RCPA
  *Pathology* 2024; ScienceDirect S0190962200254601.
- **Honesty / disclaimer framing & the OOD gate** — no skin-cancer app has FDA
  approval; professional guidance is that such tools be labelled *educational,
  not diagnostic, not clinically validated*, and that they under-perform on real
  populations vs. reported. BMJ systematic review (PMC7190019); *AI smartphone
  apps need better regulation* (PMC8144419). This directly motivates the §1
  honesty table and the manifold OOD refusal.
