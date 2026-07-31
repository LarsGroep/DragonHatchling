# Can ViTreous be built as a trial?

Design memo · 2026-07-31 · in response to: *"if we keep the structure of defining
cancer areas of interest for diagnosis with certainty we can use that for
accuracy and reliability."*

**Short answer: yes — and this reframe is the strongest move available. It
changes the regulatory class, replaces the metric that was fatal, and makes the
first study runnable with data that already exists and no patients. One phrase
in it has to change.**

---

## 1. Why the reframe works

### 1.1 It moves the product from CADx to CADe

These are formally distinct device categories, not marketing terms:

| | **CADe** — detection | **CADx** — diagnosis |
|---|---|---|
| What it does | marks / localizes regions that may contain an abnormality | characterizes disease, type, severity, stage |
| **How it is evaluated** | **localization accuracy** | accuracy of the likelihood-of-disease output |
| FDA class | II, special controls, 510(k) | II, special controls, 510(k) |

Both were [reclassified from Class III to Class II](https://www.federalregister.gov/documents/2020/01/22/2020-00497/medical-devices-radiology-devices-classification-of-the-radiological-computer-assisted-diagnostic).
"Defining areas of interest" is textbook CADe, and CADe is graded on
**localization accuracy** — exactly the substitution you proposed.

### 1.2 It deletes the prevalence term

This is the real win. From [`VITREOUS-EVERYDAY.md` §1.1](./VITREOUS-EVERYDAY.md),
at the published operating point (sens 0.91 / spec 0.64) positive predictive
value collapses with prevalence:

| Setting | Prevalence | PPV | False alarms per true find |
|---|---:|---:|---:|
| HAM10000 as-is | 19.5 % | 37.9 % | 1.6 : 1 |
| Consumer app | 1 % | 2.5 % | 39 : 1 |

PPV is `f(sensitivity, specificity, prevalence)`. **Localization metrics
computed conditional on a lesion being present have no prevalence term at
all.** IoU, Dice, pointing-game hit rate and attribution-mass-within-mask are
properties of a single image with a lesion in it. The 39:1 problem is not
improved — it is *not applicable*, because you are no longer asking "does this
person have cancer".

That is a genuine escape, not a relabelling. But it holds only under the
condition in §2.

### 1.3 The metrics are better, and one already exists

You are not starting from zero. `vitreous/xai/eval.py` already implements
**deletion/insertion AUC** — mask the highest-attributed patches first and watch
the class probability collapse. That is a *faithfulness* measure: does the
explanation reflect what the model actually used? Localization overlap is the
complementary *correctness* measure: does it land where a dermatologist would
point? Reliability needs both, and they can disagree — an explanation can be
perfectly faithful to a model that is looking at a ruler.

---

## 2. The one thing that has to change

> "cancer areas of interest"

Drop the word **cancer**. Use **"areas of diagnostic interest"** or **"regions
the model attends to"**.

This is not pedantry, and it is not a legal footnote. It decides three things at
once:

1. **Statistics.** "Cancer area" is a malignancy claim about a region, which
   re-introduces a detection task with false-marks-per-image and brings the
   prevalence burden back in FROC form. "Area of diagnostic interest" is
   conditional on a lesion already being examined, and keeps §1.2's escape.
2. **Regulation.** "This region is cancer" is a diagnostic claim — CADx, device,
   510(k). "This is where the model looked" is not a clinical claim at all.
3. **Truth.** The model cannot localize cancer. It localizes *pixels that drove
   its own output*. Those coincide with pathology only to the extent the model
   is right, which is the thing under test. Naming the output after the
   conclusion rather than the measurement is the same error as naming a neuron
   cluster "location: foot".

Same measurements, same product, radically different exposure. Take the
cheaper phrase.

### On "with certainty"

Achievable as **calibrated** certainty, not actual certainty. The honest
construction is a coverage guarantee: *"across cases like these, the marked
region contains the lesion at least 90 % of the time"*. That is
[conformal prediction](https://arxiv.org/html/2512.12844v1) with abstention —
emit a region only when the calibrated confidence clears a bar, decline
otherwise. Two constraints carry over from `VITREOUS-EVERYDAY.md` §3.2: the
calibration set must match the deployment distribution, and coverage must be
computed **per class** so melanoma does not absorb the error budget.

---

## 3. The evidence ladder

Four tiers. Each is a prerequisite for the next, and the word "trial" only
properly applies from Tier 3.

### Tier 0 — Retrospective localization validation · **runnable now, no patients, no ethics approval**

The ground truth already exists and is public:

| Source | What it gives | Use |
|---|---|---|
| **ISIC 2018 Task 1** | binary **lesion segmentation** masks | "did the model look at the lesion at all" |
| **ISIC 2018 Task 2** | pixel-level masks for 5 dermoscopic attributes over 2 594 HAM10000 images | "did it look at the *diagnostic structures*" |

Task 2 is the stronger endpoint and you are already wired for it —
`vitreous.dermoscopy` loads exactly this. The new `vitreous.localization`
module supplies IoU/Dice, pointing-game hit rate with Wilson intervals,
attribution-mass-within-mask, and a FROC summary.

**Primary endpoint:** attribution mass falling inside the annotated lesion,
with a confidence interval. **Secondary:** pointing-game hit rate; deletion AUC
(faithfulness); mass falling in the image border (the shortcut detector).

This is a complete, publishable study. It requires no new data collection.

### Tier 1 — Reliability

Accuracy is not reliability. Distinct, cheap, and rarely reported:

- **Test–retest** — same lesion, re-photographed (different angle, lighting,
  device). Does the marked region move? Instability here is disqualifying and
  invisible to Tier 0.
- **Inter-method agreement** — Grad-CAM vs Chefer vs IG on the same image.
  `method_agreement` in `xai/eval.py` already computes this. Treat disagreement
  as signal, per the existing honesty rule.
- **Inter-rater ceiling** — dermatologists disagree with each other. Measure
  that first; a model matching human-to-human agreement has hit the ceiling,
  and reporting model-vs-consensus without it overstates the gap.
- **Stratified by Fitzpatrick type**, per `VITREOUS-EVERYDAY.md` §3.4.

### Tier 2 — Reader study (MRMC) · *the real trial*

The FDA-accepted design for CADe: multiple readers each interpret multiple
cases **with and without** the tool, fully crossed, analysed with
Obuchowski–Rockette. The endpoint is not the model's accuracy — it is
**whether clinicians do better with it**, which is the only question that
justifies deployment.

This is where an interpretability tool should outperform a classifier. A
black-box probability gives a clinician nothing to check; a marked region with
a faithfulness score is auditable, and the literature on foundation models
already shows the largest gains for *non-specialists*
([PanDerm: +16.5 % differential-diagnosis improvement for non-dermatologists](https://www.nature.com/articles/s41591-025-03747-y)).

Be realistic: MRMC studies are [expensive and statistically non-trivial](https://www.medrxiv.org/content/10.1101/2023.09.25.23296069v1.full),
needing several readers × a few hundred cases, each read twice with washout.
Power it properly ([`MRMCsamplesize`](https://www.medrxiv.org/content/10.1101/2023.09.25.23296069v1.full)) before committing.

### Tier 3 — Prospective clinical trial

Only after Tiers 0–2. Follow the standards from the start, not at write-up:

- **[SPIRIT-AI](https://www.nature.com/articles/s41591-020-1037-7)** — protocol
- **[CONSORT-AI](https://www.nature.com/articles/s41591-020-1034-x)** — reporting
- **[DECIDE-AI](https://www.nature.com/articles/s41591-022-01772-9)** — early-stage live clinical evaluation. The most relevant of the four for you: it exists precisely because many AI systems perform well preclinically and few show real patient benefit, and it covers small-scale safety and human-factors evaluation *before* a large trial
- **STARD-AI** — diagnostic accuracy studies

Register the protocol before collecting data. Both AI extensions specifically
require reporting the algorithm version and the input-acquisition procedure —
which for phone photos means the capture conditions are part of the protocol,
not an afterthought.

---

## 4. The option worth considering first

There is a path with **zero regulatory burden** that you are already standing on.

If the user is a **researcher or developer** rather than a patient or clinician,
and the claim is *"this is what the model looked at and here is how reliable
that explanation is"*, then there is no device claim, no clinical endpoint, and
no ethics approval — and the study is still real, quantitative and publishable.

That is Tier 0 + Tier 1, and it is what ViTreous already is: an
**interpretability instrument for image classifiers**, shipping a skin-lesion
instance. The deliverable is a validated statement of the form:

> On 2 594 ISIC-annotated dermoscopy images, X % of this model's attribution
> mass falls inside the dermatologist-annotated lesion (95 % CI a–b), the
> pointing-game hit rate is Y % (95 % CI c–d), and deletion AUC is Z. On the
> N % of cases where mass falls predominantly on the image border, the
> explanation is flagged rather than shown.

Nobody in this space publishes that last clause. It is a differentiator, it is
honest, and it needs no patients.

**Recommendation:** do Tier 0 now. It is a week of work against data you can
already load, it produces the numbers every later tier depends on, and if the
attribution mass turns out to land mostly outside the lesion, you have learned
that for the cost of a week instead of the cost of a reader study.

---

## 5. What this does *not* fix

State plainly, because the reframe is genuinely strong and that makes it easy
to over-claim:

1. **Domain shift is untouched.** A dermoscopy-trained model still cannot be
   pointed at a phone photo. Localization on phone images needs phone-image
   ground truth, and ISIC Task 1/2 are dermoscopic.
2. **Whole-body screening brings prevalence back.** §1.2's escape holds only
   *conditional on a lesion being examined*. "Find the concerning spots on this
   back" is detection over an image, with false marks per image, and the burden
   returns — FROC is then the right metric precisely because it counts them.
3. **A perfectly localized explanation of a wrong model is still wrong.**
   Localization measures where it looked, not whether the answer is right.
   Report it alongside classification metrics, never instead of them.
4. **Fitzpatrick coverage is still the limiting factor** for any claim of
   general use, and no reframe changes that.

---

## Sources

- [FDA classification: radiological computer-assisted diagnostic software (Federal Register)](https://www.federalregister.gov/documents/2020/01/22/2020-00497/medical-devices-radiology-devices-classification-of-the-radiological-computer-assisted-diagnostic) · [reclassification summary](https://www.auntminnie.com/imaging-informatics/advanced-visualization/image-processing/article/15748477/fda-issues-final-order-reclassifying-radiological-cad-software)
- [Performance evaluation for CADe/CADx/triage, post-market (PMC)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9994700/)
- [MRMCsamplesize: sample sizes for multi-reader multi-case studies](https://www.medrxiv.org/content/10.1101/2023.09.25.23296069v1.full)
- [SPIRIT-AI protocol guideline (*Nature Medicine*)](https://www.nature.com/articles/s41591-020-1037-7) · [CONSORT-AI reporting guideline (*Nature Medicine*)](https://www.nature.com/articles/s41591-020-1034-x) · [both, in *Trials*](https://link.springer.com/article/10.1186/s13063-020-04951-6)
- [DECIDE-AI (*Nature Medicine* 2022)](https://www.nature.com/articles/s41591-022-01772-9) · [PubMed](https://pubmed.ncbi.nlm.nih.gov/35585198/)
- [PanDerm (*Nature Medicine* 2025)](https://www.nature.com/articles/s41591-025-03747-y)
- [Selective Conformal Risk Control](https://arxiv.org/html/2512.12844v1) · [class-wise conformal coverage](https://arxiv.org/pdf/2406.06818)
- [SkinCon / ISIC annotation context](https://arxiv.org/abs/2302.00785)
