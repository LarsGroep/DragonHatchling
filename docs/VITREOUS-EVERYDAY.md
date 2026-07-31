# ViTreous in everyday life — extensions for ordinary phone photos

Research memo · 2026-07-31 · scope: what it would take for ViTreous to work on
pictures people actually take, and what must never be claimed if it does.

---

## 0. The one thing to internalise first

**A model trained on HAM10000 will not work on a phone photo, and the failure
is quiet.**

HAM10000 is *dermatoscopic*: a contact lens pressed to the skin, polarised
light, immersion fluid, fixed magnification, the lesion centred and filling the
frame. A phone photo has none of that — variable distance, uncontrolled white
balance, motion blur, shadows, hair, background skin, sometimes the wrong body
part entirely. These are different imaging modalities that happen to both be
"pictures of skin."

The dangerous part is that a classifier does not know this. Softmax on
out-of-distribution input is confidently wrong, not uncertain. Point a
dermoscopy-trained model at a phone photo and it returns a crisp 7-way
probability vector that means nothing. Everything in §3 exists to make that
failure loud.

This is why the highest-value extensions are *not* accuracy work. They are
**refusal, calibration, and honest scope**.

---

## 1. What the evidence says

| Finding | Number | Source |
|---|---|---|
| Pooled AI sensitivity / specificity, skin lesion, >70 000 test images | **0.91 / 0.64** (AUROC 0.88) | [Equity & Generalizability meta-analysis, *Medicina* 2025](https://doi.org/10.3390/medicina61122186) |
| Standalone AI vs dermoscopy, same review | AI **0.855 / 0.731** · dermoscopy **0.879 / 0.779** | same |
| Skin-cancer apps with regulatory approval (US) | **none** | [BMJ systematic review](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7190019/), [Physics World summary](https://physicsworld.com/a/smartphone-apps-fall-short-on-skin-cancer-diagnosis/) |
| Fitzpatrick IV–VI representation | severely underrepresented; measurably worse sensitivity **and** specificity for melanoma on darker skin | [meta-analysis](https://doi.org/10.3390/medicina61122186), [narrative review 2025](https://www.cureus.com/articles/415719) |
| Fitzpatrick17k label noise / tone imbalance | **>30 %** label noise, **3.6:1** light:dark | [SkinCon](https://arxiv.org/abs/2302.00785) |

Read the first row carefully: **specificity 0.64** is the number that decides
whether a consumer tool is usable. At 0.64, roughly a third of benign lesions
get flagged. On a population where malignancy prevalence is low — which is
every consumer population — that produces overwhelmingly more false alarms than
true findings. The clinical harm of a consumer skin app is rarely the missed
melanoma; it is the anxiety, the unnecessary biopsies, and the eventual
learned dismissal of *all* its warnings.

### 1.1 What that specificity actually costs — computed, not asserted

Running the pooled operating point (sens 0.91, spec 0.64) through
`vitreous.clinical.binary_metrics_at_threshold` over a 1 000 000-person cohort,
varying only prevalence:

| Setting | Prevalence | **PPV** | False alarms per true find |
|---|---:|---:|---:|
| HAM10000 as-is | 19.5 % | 37.9 % | 1.6 : 1 |
| Dermatology clinic referral | 10 % | 21.9 % | 3.6 : 1 |
| GP / primary care | 3 % | 7.2 % | 12.8 : 1 |
| **Consumer app, general public** | **1 %** | **2.5 %** | **39 : 1** |
| Consumer app, worried-well | 0.3 % | 0.8 % | 131 : 1 |

Reproduce with `packages/core` installed — it is ten lines against the public
API, and worth re-running against your own model's numbers.

**This is the central finding of this memo.** Nothing about the model changes
across those rows; only who points the camera. The same classifier that looks
respectable on a curated benchmark produces ~39 false alarms per genuine
finding in consumer use, and the published operating point is *already* tilted
toward sensitivity.

Three consequences follow directly:

1. **Prevalence is a design parameter, not a footnote.** A tool aimed at
   already-worried users with a suspicious lesion sits at a very different row
   from one offering whole-body screening. Screening is the row you cannot win.
2. **Chasing sensitivity makes this worse.** The instinct — "a missed melanoma
   is the worst error, so lower the threshold" — costs specificity, which at
   these prevalences is what the whole readout rests on. The high-sensitivity
   default of 0.2 in `/lens` is defensible *clinically* and expensive
   *statistically*; that trade-off should be a deliberate, documented decision.
3. **Refusal is the only lever that helps both.** Declining to answer on
   unusable input removes cases from the denominator without trading sensitivity
   for specificity — which is why §3.1 leads the roadmap.

---

## 2. Datasets that are actually phone photos

HAM10000 cannot support this direction. These can:

| Dataset | What it is | Why it matters here |
|---|---|---|
| **PAD-UFES-20** | 2 298 smartphone images, 1 373 patients, 6 classes; 58 % biopsy-proven, **100 % of cancers biopsy-proven**; ships patient metadata | The single best fit. Real phone cameras, real clinical setting, hard labels where it counts. [Paper](https://www.sciencedirect.com/science/article/pii/S235234092031115X) · [PMC](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7479321/) |
| **SCIN** (Google) | Crowdsourced consumer smartphone images of skin concerns | Closest to true "everyday life" capture conditions — self-taken, unposed |
| **DDI** | 656 clinical images, dermatologist-curated, Fitzpatrick-labelled, malignant/benign | The honest fairness benchmark. Small, but every image is verified |
| **Fitzpatrick17k** | 16 577 clinical images, tone-annotated | Large but noisy (>30 % label noise); use for pretraining, never for reported metrics |
| **SkinCon** | Dense expert annotation of 48 clinical concepts over Fitzpatrick17k/DDI | **Directly relevant** — see §3.3 |

Note the scale drop: HAM10000 gives you 10 015 dermoscopic images; PAD-UFES-20
gives 2 298 phone images. Any phone-photo model will be data-poor, which points
straight at foundation-model transfer (§3.5) rather than training from scratch.

---

## 3. Extensions, in priority order

### 3.1 Turn the OOD gate into the primary feature — not a footnote

You already have the right primitive. `vitreous.malignancy.build_malignancy_axis`
computes an off-axis residual and `project_feature` returns an `ood` flag. Today
it's a caveat inside the lens; for phone photos it should be **the first thing
that runs and the most common outcome**.

Concretely:
- Calibrate `residual_threshold` on real phone photos, not on a percentile of
  the dermoscopic training set. A threshold fit to HAM10000 will pass almost
  nothing *or* almost everything on phone input — it has never seen the domain.
- Add a **capture-quality gate before the model**: blur (variance of Laplacian),
  exposure clipping, lesion-fills-frame ratio, colour-cast estimate. Cheap,
  runs client-side, and rejects the majority of unusable uploads without
  invoking a model at all. Medical-imaging QC and OOD detection are increasingly
  treated as one problem ([OOD survey](https://arxiv.org/abs/2404.18279)).
- Make refusal informative: *"too blurry"*, *"lesion too small in frame"*,
  *"not a skin photo"* are actionable; *"out of distribution"* is not.

**A tool that refuses 70 % of inputs and is honest about the other 30 % is a
real product. One that answers everything is a liability.**

### 3.2 Selective prediction with a coverage guarantee

Beyond a hand-tuned threshold, **conformal prediction** gives distribution-free
finite-sample guarantees: emit a *prediction set* with a calibrated error rate,
and abstain when that set is uninformative. This is the principled version of
what §3.1 does heuristically, and it composes with abstention directly
(see [Selective Conformal Risk Control](https://arxiv.org/html/2512.12844v1)
and the [class-wise coverage work](https://arxiv.org/pdf/2406.06818)).

Two cautions that matter here:
- Conformal guarantees hold **only if the calibration set matches the
  operational distribution**. Calibrate on phone photos from the population you
  will serve, or the guarantee is decorative.
- Marginal coverage can hide catastrophic per-class failure. Use **class-wise**
  coverage so melanoma is not the class that absorbs the error budget.

Fits cleanly as a new numpy-only module beside `vitreous.clinical`.

### 3.3 Ground concepts in a vocabulary that survives the domain change

`vitreous.dermoscopy` grounds concepts in the five ISIC 2018 Task 2 criteria —
pigment network, negative network, streaks, milia-like cysts, globules. **These
are dermoscopic structures. Most are not visible without a dermatoscope.** Ship
that vocabulary against phone photos and you will name concepts after features
the camera physically cannot resolve.

**SkinCon** is the replacement: 48 clinical concepts (papule, plaque, scale,
crust, erosion, telangiectasia, …) densely annotated by dermatologists over
Fitzpatrick17k and DDI — chosen precisely because they are describable from
clinical photographs. It plugs into the existing
`ground_concepts(activations, attribute_matrix, names)` signature with no
mechanism change; only the vocabulary constant and loader differ.

The existing minimum-support and effect-size bars carry over unchanged, as does
the metadata confound probe — which becomes *more* important on phone photos,
where background, lighting and body site vary far more than in dermoscopy and
offer far more for a model to cheat on.

### 3.4 Make fairness a first-class measured output

The evidence is unambiguous that performance drops on Fitzpatrick IV–VI. With
`vitreous.clinical` this is nearly free: it already returns per-class recall
with Wilson intervals and support counts. Extend it to **stratified reporting**
— sensitivity/specificity per Fitzpatrick group, each with its interval and n.

Two design commitments worth making explicitly:
- If a stratum's support is too small to estimate, say *"not enough data to
  report performance for this skin tone"*. That sentence is more valuable than
  any number, and the Wilson interval already tells you when to print it.
- Estimate tone from the image only to *route the measurement*, never to alter
  the prediction. And validate the estimator — [tone-scale labelling is itself
  contested](https://www.nature.com/articles/s41746-025-02245-2), and
  [label granularity changes measured fairness](https://arxiv.org/pdf/2509.11184).

This is also the strongest honest claim the project could make: most tools
report one aggregate number; reporting *where it fails* is differentiating.

### 3.5 Swap the backbone for a dermatology foundation model

Training a phone-photo model from 2 298 PAD-UFES-20 images will not work well.
The current answer is transfer from a foundation model:

**PanDerm** ([Nature Medicine 2025](https://www.nature.com/articles/s41591-025-03747-y),
[arXiv](https://arxiv.org/html/2410.15038v2)) — self-supervised on >2 M skin
images across 4 modalities and 11 institutions. Reported: +11 % clinician
accuracy on dermoscopy, **+16.5 % differential-diagnosis improvement for
non-dermatologists across 128 conditions on clinical photographs**, and
state-of-the-art with **10 % of labelled data**. It explicitly retains
performance on clinical photographs it was not trained on. **DermINO**
([arXiv](https://arxiv.org/pdf/2508.12190)) is a comparable hybrid-pretrained
alternative.

Architecturally this is cheap for you: `vitreous.models` already abstracts the
backbone, and `packages/core`'s instrumentation hooks are backbone-agnostic.
The Hebbian recorder observes *any* layer's activations, so
`vitreous.hebbian` works unchanged on a foundation-model encoder — the
co-activation graph is a property of the units you hook, not of how they were
trained.

### 3.6 Run it on-device

Everything about consumer skin photos argues for local inference: the images
are sensitive, intermittent connectivity is normal, and not transmitting is a
stronger privacy claim than any policy page. The legacy `webapp/` already
proved the pattern — ONNX + `onnxruntime-web`, vendored, no CDN, fp16 export at
half the download with identical top-5. That capability regressed when the
project moved to `apps/web`; restoring it is well-trodden ground rather than
research.

The capture-quality gate in §3.1 is pure client-side image processing and needs
no model at all.

---

## 4. "Universal" beyond skin

The adapter registry (`@register_dataset`) and the `GraphProvider` Protocol
already make ViTreous domain-agnostic — the workbench does not know what a
lesion is. The honest framing of "universal" is therefore:

> ViTreous is an **interpretability workbench for image classifiers**, which
> ships a skin-lesion instance.

That is a defensible and genuinely useful claim. The version to avoid is "it
classifies any phone photo" — every §1 number would have to be re-established
per domain, and the medical framing does not transfer to, say, plant disease or
food logging.

If you want a second domain to prove universality, pick one where **being wrong
is cheap** (plant leaves, recycling sorting, bird ID). It exercises the same
pipeline, demonstrates the swap, and carries none of the clinical risk. This is
probably the fastest way to show the architecture's value without inheriting
medical-device obligations.

---

## 5. What must not be claimed

Non-negotiable, and load-bearing given §1:

1. **No diagnosis.** No skin-cancer app has FDA approval. In the EU such apps
   are class I self-certified — a status
   [criticised as not fit for the risk](https://physicsworld.com/a/smartphone-apps-fall-short-on-skin-cancer-diagnosis/).
   Recent approvals (e.g. DermaSensor, FDA-cleared Jan 2025) are **hardware
   devices used by clinicians**, not consumer apps, and are not precedent.
2. **No triage advice.** "Probably fine, no need to see anyone" is the sentence
   that causes harm. The tool may say *what it sees*; it may not say what to do.
3. **No aggregate accuracy without the baseline and the stratification.**
   Already enforced by `format_honest_summary`; keep it that way.
4. **No silent domain transfer.** If the model was trained on dermoscopy and
   the input is a phone photo, that must be stated on screen, not buried.
5. **Refusal is a feature, and must never be tuned down for demo appeal.**

The framing that survives all of this: **an explainability instrument that
shows you what a model looks at and how sure it is — including when it should
not answer at all.** That is honest, useful, and a genuinely under-served niche.

---

## 6. Suggested order of work

| # | Step | Why first |
|---|---|---|
| 0 | **Decide the target prevalence row in §1.1** | Free, and it determines whether anything below is worth doing. Screening the general public is not a winnable configuration at spec 0.64 |
| 1 | Compute PPV for *your* model, not the pooled figure | §1.1 uses published numbers; your model's specificity is the one that counts |
| 2 | Client-side capture-quality gate | No model, no data, no training — immediate, and rejects most bad input |
| 3 | Add PAD-UFES-20 as an adapter | Small, phone-native, biopsy-proven cancers; makes the domain gap measurable |
| 4 | Recalibrate the OOD residual on real phone photos | Turns the existing gate from decorative to functional |
| 5 | Stratified fairness reporting | Cheap given `vitreous.clinical`; the strongest honest claim available |
| 6 | Foundation-model backbone | Highest cost, but the only route to workable accuracy on 2 k images |
| 7 | Conformal selective prediction | Principled abstention once (3)–(4) give a real calibration set |

Steps 1–3 are each a day or less and together settle whether this direction is
viable. Do them before step 6.

---

## Sources

- [PAD-UFES-20 (ScienceDirect)](https://www.sciencedirect.com/science/article/pii/S235234092031115X) · [PMC](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7479321/) · [arXiv](https://arxiv.org/pdf/2007.00478)
- [Equity and Generalizability of AI for Skin-Lesion Diagnosis (*Medicina*, 2025)](https://doi.org/10.3390/medicina61122186)
- [PanDerm — A multimodal vision foundation model for clinical dermatology (*Nature Medicine*, 2025)](https://www.nature.com/articles/s41591-025-03747-y) · [arXiv](https://arxiv.org/html/2410.15038v2)
- [DermINO: Hybrid Pretraining for a Versatile Dermatology Foundation Model](https://arxiv.org/pdf/2508.12190)
- [SkinCon: a densely annotated clinical-concept dermatology dataset](https://arxiv.org/abs/2302.00785) · [site](https://skincon-dataset.github.io/)
- [Out-of-distribution Detection in Medical Image Analysis: A survey](https://arxiv.org/abs/2404.18279)
- [Selective Conformal Risk Control](https://arxiv.org/html/2512.12844v1) · [Class-wise conformal coverage](https://arxiv.org/pdf/2406.06818)
- [Evaluating skin tone scales for dermatologic dataset labeling (*npj Digital Medicine*, 2025)](https://www.nature.com/articles/s41746-025-02245-2)
- [Impact of Skin Tone Label Granularity on Performance and Fairness](https://arxiv.org/pdf/2509.11184)
- [Diagnostic capability of AI in dermatology for darker skin tones (narrative review, 2025)](https://www.cureus.com/articles/415719)
- [Smartphone apps fall short on skin cancer diagnosis (Physics World)](https://physicsworld.com/a/smartphone-apps-fall-short-on-skin-cancer-diagnosis/) · [BMJ systematic review](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7190019/)
