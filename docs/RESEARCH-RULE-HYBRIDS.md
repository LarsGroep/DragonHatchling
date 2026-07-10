# Hybrid Rule-Based Transparency for Vision Classifiers — Research Record (Phase 2)

Companion to [`RESEARCH.md`](./RESEARCH.md) (Phase 1: ViT attribution/workbench
review) and [`ARCHITECTURE.md`](./ARCHITECTURE.md). This document is the
literature review and solution proposal for the next research question:

> **How do we combine state-of-the-art vision backbones with rule-based
> systems whose rulesets are learned and changed dynamically by ML — so that
> the influence of concepts on class probabilities becomes explicit,
> auditable threshold rules rather than post-hoc attributions?**

Prepared 2026-07-10 by Claude Fable 5 (research/orchestration model);
implementation is to be delegated to Claude Opus agents per the work packages
in §9. No code was written in this phase.

Owner decisions that scope this review (from the 2026-07-10 interview):

| Question | Decision |
|---|---|
| Meaning of "dynamically changing rulesets" | **Threshold/split rules à la decision trees** — defined values on concept/feature activations that determine a classification — with the *ruleset itself learned and updated by ML*, not hand-authored |
| First comparison target | **Generic demo / CIFAR** (the existing `notebooks/explainability_demo.ipynb` CIFAR-10 path) |
| Role of the Hebbian memory | **Hybrid** — it stays as a first-class concept source, but the research must treat it as *one contender* and benchmark it against alternatives |
| Accuracy ↔ transparency trade | Open — **each proposed solution must state its expected cost**, and the side-by-side notebook must measure it |

---

## 1. What the back-end does today, and where it is weakest

The current pipeline (`hatchvision/`) is:

1. `HebbianFeatureMemory` (`hatchvision/hebbian/memory.py`) — observation-only
   forward hooks keep an EMA of the outer product of pooled, rectified,
   L2-normalized activations per observed layer, plus class-conditional
   firing rates. Subsampled to ≤ `max_units` channels.
2. `cluster_concepts` (`hatchvision/explain/concepts.py`) — Ward
   agglomeration on correlation-row "fingerprints" → concept clusters, each
   with coherence, importance, class affinity, and probe-set exemplars.
3. `unit_class_influence` (`hatchvision/explain/influence.py`) —
   expected-gradients (exact-linear when the readout is linear) attribution
   of tracked units → class logits, aggregated to per-concept SHAP-style
   contributions.

This is a genuinely honest *descriptive* pipeline (nothing modifies the
model; the exact-linear detection is a nice touch). Its transparency
limits, measured against the goal above:

- **No decision semantics.** Concepts *correlate* with classes
  (`class_affinity`) and *push* logits (`unit_class_influence`), but nothing
  in the pipeline says "*this image is class k because concept A > θ₁ and
  concept B < θ₂*". The explanation is a weighting, not a rule; a user
  cannot audit, verify, or intervene on it.
- **Correlational concept discovery.** Hebbian co-activation is symmetric
  second-order statistics. Two units firing together does not imply they
  encode one human concept; polysemantic units land in one cluster and
  a concept's "meaning" rests entirely on exemplar grids. (This is the same
  criticism Phase 1 recorded against Hebbian edges as graph semantics.)
- **Spatial pooling.** `_pool` averages over space/tokens, so a concept is
  a *global image property*; part-level concepts (a wheel, a muzzle) blur
  into whole-image statistics. Patch-level competitors (ProtoPNet-line,
  SAEs over tokens) do not have this handicap.
- **Cluster granularity is a hyperparameter** (`n_concepts=8`), not learned,
  and clusters can be unstable across seeds/training runs — a known failure
  mode we must measure (§7, rule/concept stability).
- **The model itself is untouched** — this is post-hoc; nothing constrains
  the network to *use* the concepts. All faithfulness rests on the
  attribution method.

The literature below addresses exactly these four gaps.

---

## 2. A taxonomy of "dynamically changing rulesets" in the literature

The owner's definition — *threshold rules learned by ML* — maps to five
distinct mechanism families. They are not competitors; they are choices on
two axes: **when rules change** (training-time / per-input / online) and
**what selects the active rule** (routing, generation, retrieval, querying).

| # | Family | Rules change… | Mechanism | Canonical works |
|---|---|---|---|---|
| D1 | **Differentiable threshold/rule learning** | during training (gradient descent moves thresholds & rule structure) | soft or straight-through relaxations of discrete splits/logic gates | Soft Decision Trees (Frosst & Hinton 2017); [GradTree](https://www.researchgate.net/publication/379281664_GradTree_Learning_Axis-Aligned_Decision_Trees_with_Gradient_Descent) / [GRANDE (ICLR 2024)](https://arxiv.org/abs/2309.17130); [RRL (NeurIPS 2021 / TPAMI 2023)](https://arxiv.org/abs/2109.15103); DNDF (Kontschieder ICCV 2015); [DCNFIS](https://arxiv.org/html/2308.06378v3) |
| D2 | **Per-instance rule generation** | per input at inference | a neural module *emits* a symbolic rule for each sample, then the rule (not the network) is evaluated on concept truth values | [Deep Concept Reasoner, ICML 2023](https://arxiv.org/abs/2304.14068) |
| D3 | **Learned rule memory + neural selection** | memory learned at training; *which* rule fires is dynamic per input | a bank of learnable logic rules; a selector picks one; prediction = symbolic evaluation of the selected rule | [Concept-Based Memory Reasoner, NeurIPS 2024](https://arxiv.org/abs/2407.15527) |
| D4 | **Sequential/adaptive query policies** | the *chain* of concept tests is assembled per input | ask the most informative concept question next, stop when confident — an input-dependent decision list | [V-IP, ICLR 2023](https://arxiv.org/abs/2302.02876); [learned queries follow-up](https://arxiv.org/html/2312.11548v1); [uncertainty-aware V-IP 2025](https://arxiv.org/abs/2506.16742) |
| D5 | **Evolving / online rule bases** | after deployment, from data streams | rules are added, merged, pruned as distributions drift | evolving fuzzy systems ([2025 survey](https://www.sciencedirect.com/science/article/pii/S1568494625013717)); [MFRBCS mixture of fuzzy rule systems, 2025](https://link.springer.com/article/10.1007/s44196-025-01112-y) |

**Reading of the owner's intent:** the tree-based example ("split on
threshold rules, defined values that determine a classification") is D1 at
its core, but the phrase *dynamically changing* pushes beyond a single
static learned tree — D2/D3/D4 are the modern neuro-symbolic answers to
"the ruleset is itself a learned, input-adaptive object". D5 (stream
adaptation) is out of scope for a CIFAR demo but worth one paragraph in the
notebook's discussion section as future work. **Recommendation: build the
comparison around D1 (learned threshold rules over concepts) as the
baseline family, with D3 (CMR) and D2 (DCR) as the "dynamic" contenders,
and D4 (V-IP) as an optional stretch entry.**

---

## 3. Concept sources — the bottleneck layer (where the Hebbian memory competes)

Every hybrid below factors into **concept extraction → rule-based decision
head**. The concept layer is where the Hebbian memory either earns its
place or gets outcompeted. Contenders:

### 3.1 Supervised Concept Bottleneck Models (CBM) — the reference frame

Koh et al. (ICML 2020) train `x → concepts → y` with human concept labels.
Gold standard for concept *semantics* and **intervention** (flip a wrong
concept, watch the prediction correct itself), but needs annotations CIFAR
does not have. Not directly buildable for the demo; matters because the
whole evaluation methodology (concept accuracy, intervenability, leakage)
comes from this line. A [comprehensive CBM survey/collection](https://github.com/kkzhang95/Awesome_Concept_Bottleneck_Models)
tracks the explosion of variants (2023–2026).

### 3.2 Language-guided, label-free CBMs — the strongest practical contender on CIFAR

- **Label-free CBM** (Oikarinen et al., ICLR 2023) — GPT-generated concept
  candidates per class, CLIP-scored concept activations, then a **sparse
  linear head** (GLM-SAGA). Runs on CIFAR-10/100 out of the box with ~1–2 %
  accuracy drop vs. the backbone.
- **[LaBo (CVPR 2023)](https://lilywenglab.github.io/VLG-CBM/)** — GPT-3
  concept generation + submodular selection for discriminative,
  non-overlapping concept sets.
- **[VLG-CBM (NeurIPS 2024)](https://arxiv.org/abs/2408.01432)** — grounds
  concepts with an open-vocabulary detector (vision-language guidance),
  trains a sparse final layer, and — important for us — introduces
  **NEC (Number of Effective Concepts)** as a *controlled interpretability
  budget*: compare methods at equal NEC or the comparison is meaningless.
- **[Discover-then-Name / CLIP-free & label-free variants (2025–26)](https://arxiv.org/abs/2503.10981)**
  — SAE-discovered concepts that are *then* named, removing both the CLIP
  dependency and the concept-label dependency.

Why this matters for the demo: it gives **named** concepts ("has wings",
"metallic surface") on CIFAR for free, which is a dramatically better UX
than "concept 3: cat / dog 0.71" from cluster affinity. Rules over named
concepts read as sentences.

### 3.3 Sparse autoencoders (SAE) — the mechanistic contender

Phase 1 already selected k-sparse SAEs as ViTreous's concept tier. Since
then the vision-SAE literature matured:
[Interpretable & Testable Vision Features via SAEs (2025)](https://arxiv.org/abs/2502.06755),
[hierarchical/Matryoshka SAEs for CLIP (ICML 2025)](https://arxiv.org/abs/2502.20578),
[steering CLIP's ViT with SAEs (2025)](https://arxiv.org/abs/2504.08729),
and — directly relevant — **[Concept Bottleneck Sparse Autoencoders
(2025)](https://arxiv.org/pdf/2512.10805)**, which fuse the SAE and CBM
views. SAE features are sparse, positive, patch-localizable, and
causally testable by ablation/steering — everything the pooled Hebbian
units are not. Cost: a separate SAE training pass and a naming step
(exemplars, or CLIP-based auto-naming).

### 3.4 Hebbian co-activation clusters — the incumbent

Strengths worth defending in the comparison: **zero extra training** (piggy-
backs on the classifier's own training pass), **temporal dimension** (the
EMA evolves during training — none of the competitors can show concepts
*forming*, which is the signature asset of the Brain view in
[`UX-VISION-2.md`](./UX-VISION-2.md)), and architecture-agnostic hooks.
Weaknesses (§1): pooled/global, correlational, granularity by hand.

Two literature-grounded upgrades keep it competitive:

- **H1 — Hebbian units → concept scores as a bottleneck.** `concept_scores`
  (mean member-unit activation) already gives an `[n_images, n_concepts]`
  matrix. Treat *that* as the CBM bottleneck and train rule heads on it.
  This is precisely the [Post-hoc CBM (ICLR 2023)](https://openreview.net/pdf?id=nA5AZ8CEyow)
  recipe with Hebbian clusters instead of CAV concept banks — and PCBM's
  **hybrid residual trick** (fit a small black-box residual on top of the
  interpretable path, recovering original accuracy; PCBM-h) transfers
  directly, giving the demo an explicit, *tunable* transparency-accuracy
  dial (see §6, design principle P3).
- **H2 — patch-level Hebbian statistics.** Keep per-token/per-position
  activations before pooling for the observed layer (memory cost is why it
  pools; a top-k sparsification like the ViTreous packs makes it viable).
  This closes the part-vs-whole gap against SAEs and is a genuinely novel
  little contribution (Hebbian co-activation over *spatial* units).

### 3.5 Self-supervised concept learners (BotCL / E-BotCL)

[BotCL and its 2025 successor E-BotCL](https://www.mdpi.com/1424-8220/25/8/2398)
learn task-relevant concept slots via contrastive self-supervision inside
the classifier. Interesting middle ground (no labels, no CLIP, learned
end-to-end) but adds training complexity; note as an alternative, don't
build first.

### 3.6 Concept Whitening (Chen, Bei & Rudin, Nature MI 2020)

Replaces a BatchNorm with a whitening + rotation layer that axis-aligns
latent dimensions with chosen concepts. Cheap, drop-in, and makes the
*backbone itself* concept-aligned rather than adding a probe. A good
optional module if we want the "concepts inside the network" story without
full CBM training.

**Concept-source verdict for the notebook:** benchmark **Hebbian clusters
(H1) vs. k-sparse SAE vs. CLIP/label-free named concepts** as three
interchangeable `ConceptProvider`s feeding identical rule heads. That
isolates *concept quality* from *rule-head quality* — the single most
important design decision in this proposal.

---

## 4. Rule heads — turning concepts into dynamic threshold rulesets

Given per-image concept activations `c ∈ R^m`, the decision head makes the
classification *and* is the artifact the user audits. Ranked by fit:

### 4.1 Sparse linear head (baseline, must-have)

Label-free CBM / PCBM standard: elastic-net logistic regression on
concepts. Not a threshold ruleset, but the indispensable control — if
trees/logic don't beat sparse-linear on the transparency metrics *or*
accuracy, complexity isn't paying. Also the fairest accuracy ceiling for
any concept source. ([Sparse linear concept discovery models](https://arxiv.org/pdf/2308.10782).)

### 4.2 Classical decision tree / rule list on concepts (baseline #2)

CART / optimal shallow trees (or CORELS-style rule lists) fit on
`concept_scores`. *Static* ruleset, fully faithful-by-construction (the
tree IS the classifier), instantly visualizable: `IF sky-like > 0.42 AND
wheels < 0.11 THEN airplane`. This is the owner's example rendered
literally, and it's a one-`sklearn`-call addition to the notebook. Expected
accuracy: noticeably below the backbone on CIFAR unless depth grows past
readability — measuring exactly *how far* below is a headline result of the
comparison.

### 4.3 Differentiable rule/tree learners (D1 — the "learned, moving thresholds" family)

- **[RRL — Rule-based Representation Learner (NeurIPS 2021, TPAMI 2023)](https://arxiv.org/abs/2310.14336)**
  ([code](https://github.com/12wang3/rrl)) — layers of learnable
  conjunctions/disjunctions over *learned feature discretizations* (the
  thresholds are trainable bins), optimized discretely via **gradient
  grafting**. The final model is an exactly-discrete rule set + linear
  vote — i.e., the ruleset that changed dynamically during training is,
  at the end, a fully auditable object. Designed for tabular inputs, and a
  concept-activation matrix *is* tabular — this is the cleanest "SOTA
  backbone + ML-learned threshold rules" marriage available.
- **[GradTree / GRANDE (ICLR 2024)](https://arxiv.org/abs/2309.17130)** —
  hard, axis-aligned trees learned end-to-end with straight-through
  estimators; GRANDE ensembles them. Same tabular-on-concepts fit as RRL;
  a single GradTree stays interpretable, a GRANDE ensemble does not (flag
  in the comparison as "accuracy upper bound of the tree family").
- **Soft/neural decision trees** (Frosst & Hinton 2017 distillation; DNDF
  ICCV 2015): probabilistic routing — every leaf contributes to every
  prediction, so the "rule" story is soft. Prefer hard variants above;
  include SDT only if we want the distillation-fidelity storyline.
- **[DCNFIS — Deep Convolutional Neuro-Fuzzy Inference System](https://arxiv.org/html/2308.06378v3)**
  and the [fusion of soft decision trees with concept models](https://www.sciencedirect.com/science/article/pii/S156849462400406X):
  fuzzy rulebases end-to-end with a CNN. Fuzzy membership functions are
  *graded* thresholds — arguably closest to how concept activations
  actually behave (nothing is binarily "present"). Higher implementation
  cost; keep as an alternative in the same family slot as RRL.

### 4.4 Neuro-symbolic concept reasoners (D2/D3 — the genuinely *dynamic* rulesets)

- **[Deep Concept Reasoner (ICML 2023)](https://arxiv.org/abs/2304.14068)** —
  from concept *embeddings* it generates, per sample, a fuzzy/boolean DNF
  rule, then evaluates that rule on the concept *truth values*. The
  prediction is provably the rule's output — semantics stay clean — while
  the rule itself adapts per image ("this cat because whiskers ∧ fur ∧
  ¬wings", a different clause for the next image). Requires concept
  embeddings, which [Concept Embedding Models (NeurIPS 2022)](https://www.semanticscholar.org/paper/ce3036fadfa9692867532fe472ea40b4b81a6dc3)
  provide from any concept source.
- **[Concept-Based Memory Reasoner (NeurIPS 2024)](https://arxiv.org/abs/2407.15527)**
  ([code](https://github.com/daviddebot/CMR)) — a learned **memory of logic
  rules** + a neural selector that picks *one* rule per input; prediction =
  symbolic evaluation of the selected rule. This is the purest embodiment
  of "dynamically changing ruleset": the ruleset is explicit, enumerable,
  formally **verifiable pre-deployment** ("no rule ever concludes *truck*
  without *wheels*"), yet which rule applies is decided neurally per image.
  The authors report accuracy-interpretability trade-offs at or above
  state-of-the-art CBMs, and near-black-box accuracy when concepts are
  informative. **This is the flagship experimental architecture this
  research recommends.**
- **Logic Explained Networks / entropy-based LENs** (Ciravegna et al.,
  AIJ 2023) — extract global class-level FOL rules from an entropy-
  regularized network over concepts. Older and weaker than DCR/CMR but
  extremely cheap; candidate for a "global ruleset summary" cell.
- Newer descendants worth tracking, not building:
  [hierarchical concept reasoning via attention-guided graph learning
  (2025)](https://arxiv.org/html/2506.21102), [concept-driven logical rules
  for medical imaging (2025)](https://arxiv.org/html/2505.14049),
  [local+global integrated DCR (2025)](https://link.springer.com/chapter/10.1007/978-3-031-92648-8_14).

### 4.5 Sequential query policies (D4)

**[V-IP (ICLR 2023)](https://arxiv.org/abs/2302.02876)**
([code](https://github.com/ryanchankh/VariationalInformationPursuit)):
learn a querier that asks concept questions in information-gain order and
stops when confident. The explanation is a *short, input-specific decision
list* — "asked: wings? no. fur? yes. whiskers? yes → cat (3 queries)".
Beautiful demo material (an animated interrogation pairs naturally with the
existing Brain view), moderate implementation cost, and CIFAR-10 results
exist in the paper. Stretch goal.

### 4.6 Structural baselines that are *not* concept-based

- **[NBDT (ICLR 2021)](https://arxiv.org/abs/2004.00221)** — induce a class
  hierarchy from the final-layer weights, fine-tune with a tree-supervision
  loss; inference walks the tree. Stays within ~1 % of backbone accuracy on
  CIFAR-10/100. The "rules" are class-hierarchy routing decisions, not
  concept thresholds — include as the "high-accuracy, shallow-transparency"
  corner of the comparison space, or skip if scope is tight.
- **[ProtoTree (CVPR 2021)](https://arxiv.org/abs/2012.02046)** /
  **[PIP-Net (CVPR 2023)](https://github.com/M-Nauta/PIPNet)** — prototype
  *is* the rule antecedent: a ProtoTree node tests "does a patch match this
  learned prototype ≥ τ?" and routes accordingly — a decision tree whose
  thresholds are visual similarities, learned end-to-end. This is the
  strongest *integrated* (non-two-stage) contender and the best-published
  accuracy/interpretability combination in the part-prototype line
  ([overview](https://arxiv.org/pdf/2410.19856)). Cost: its own training
  regime; prototypes replace, not reuse, the Hebbian concepts. Include one
  of the two (PIP-Net is the more robust/recent) as the "integrated"
  entrant if budget allows.

---

## 5. Cross-cutting pitfalls the comparison must be honest about

1. **Concept leakage.** Soft concept activations smuggle non-concept
   information into the head, inflating accuracy while gutting
   interpretability and breaking interventions —
   [Havasi et al., NeurIPS 2022](https://papers.neurips.cc/paper_files/paper/2022/file/944ecf65a46feb578a43abfd5cddd960-Paper-Conference.pdf);
   [Margeloiu et al. 2021; hard-CBM fixes](https://arxiv.org/abs/2402.05945);
   [even a 2026 counterpoint defending leakage](https://arxiv.org/html/2606.10669)
   — the debate is live, so *measure*, don't assume. Mitigations to carry
   into the design: evaluate heads on **binarized** concepts as well as soft
   ones; report both; use VLG-CBM's **NEC** to hold the interpretability
   budget fixed across methods.
2. **Faithfulness ≠ plausibility.** A rule that *reads* well isn't the
   model's computation unless the architecture forces it (DCR/CMR/trees do;
   post-hoc rule extraction from a black box does not). For any two-stage
   design, the head is faithful *to the concept layer*; the concept layer's
   fidelity to the backbone still needs deletion/insertion-style checks
   (Phase 1 already ships these in `vitreous.xai.eval` — reuse).
3. **Intervenability as a metric, not a vibe.** The CBM literature's
   sharpest test: set a concept to its true value (or ablate it), measure
   task-accuracy change. Works for all two-stage contenders including the
   Hebbian source (clamp a cluster's activation). See
   [intervenable black boxes](https://arxiv.org/pdf/2401.13544).
4. **Rule instability.** Retrain with a new seed — do the rules survive?
   Report rule-set Jaccard overlap / concept-cluster ARI across ≥ 3 seeds.
   Hebbian clusters are especially at risk (§1); publishing this honestly
   is more valuable than hiding it.
5. **CIFAR-specific caveat.** 32×32 images make *part-level* concepts
   marginal (a 3-pixel wheel); named-concept CBMs and prototypes shine at
   224px. Standard practice (Label-free CBM, NBDT) still works on CIFAR,
   but the notebook should upsample inputs for CLIP scoring and the doc
   should note CUB/HAM10000 (already in `notebooks/`) as the natural
   second target where concepts get sharper.

---

## 6. Proposed architecture: one comparison harness, five contenders

Design principles (these bind the Opus implementation):

- **P1 — Factorize.** Everything is `Backbone (frozen) → ConceptProvider →
  RuleHead`. Concept sources and rule heads are registries, exactly like
  `available_loaders()`/`available_backbones()` already are. The Hebbian
  memory becomes `HebbianConceptProvider` — *one contender, still first-class*
  (owner's "hybrid" decision).
- **P2 — Same backbone for everyone.** Train the existing classifier once
  (any registered backbone; `simple_cnn`/`resnet18` for CPU, `bdh` for the
  sparse-space story), freeze it, and give every contender the same
  features. Integrated contenders (PIP-Net/ProtoTree, NBDT) that must
  fine-tune are labeled as such in the results table.
- **P3 — Transparency dial, not transparency dogma.** Every two-stage
  contender ships with an optional **PCBM-h residual** (small black-box fit
  on the interpretable path's residuals,
  [Yuksekgonul et al., ICLR 2023](https://openreview.net/pdf?id=nA5AZ8CEyow)).
  The notebook plots each contender as a *curve* (rule-only → rule+residual)
  in accuracy-vs-transparency space rather than a single point — this is
  how we answer the owner's "depends on the proposed solutions".
- **P4 — Rules are pack assets.** Winning contenders export their ruleset
  (global rules, per-image fired rule, thresholds, concept references) into
  the Explanation Pack / IVGraph so the web Brain view can render "which
  rule fired" — the concept graph gains rule nodes. (Schema work deferred
  to a later milestone; the notebook JSON just needs to be forward-compatible.)

The five contenders for the side-by-side:

| # | Contender | Family | Concepts from | Ruleset dynamics | Expected CIFAR-10 accuracy vs backbone | Effort |
|---|---|---|---|---|---|---|
| C1 | **Hebbian-CBM** — `concept_scores` → {sparse linear, shallow CART} | D1 (static→learned thresholds) | Hebbian clusters (existing code + H1) | thresholds learned per training run | −3…−10 pts (concepts are coarse/global) | **Low** — mostly wiring |
| C2 | **SAE-CBM** — k-sparse SAE features → same heads | D1 | SAE (ViTreous `kaggle/sae.ipynb` already exists) | same | −1…−5 pts | Low-Med |
| C3 | **Named-concept CBM** — CLIP-scored GPT/LLM concept bank → sparse linear + CART + RRL | D1 | Label-free CBM / VLG-CBM style | RRL: discrete ruleset evolves under gradient grafting | −1…−3 pts (literature-backed) | Med |
| C4 | **CMR head** (flagship) — rule memory + neural selector over the best concept source from C1–C3 | **D3 (dynamic selection)** | pluggable | learned rule memory; per-image rule choice; formally verifiable | ≈ sparse-linear CBM or better (paper claims) | Med — official code exists |
| C5 | **DCR head** — per-sample DNF generation over concept embeddings | **D2 (dynamic generation)** | pluggable (needs CEM-style embeddings) | a fresh rule per image | ≈ CBM baselines | Med |
| S1 | *(stretch)* V-IP querier | D4 | named concepts | per-image adaptive query chain | competitive per paper | Med-High |
| S2 | *(stretch)* PIP-Net or ProtoTree | integrated | own prototypes | similarity thresholds in a tree | near-backbone (paper) | High — own training loop |

The head-to-head that answers the owner's core question is **C1 vs C2 vs
C3 with identical heads** (does Hebbian survive as a concept source?) and
**CART vs RRL vs CMR vs DCR on the winning concept source** (which ruleset
dynamics buy what, at what accuracy cost?).

---

## 7. Evaluation protocol for the side-by-side notebook

Fixed setup: CIFAR-10, the notebook's existing capped splits for CPU-speed
demo plus a full-data path for Kaggle; ≥ 3 seeds for stability metrics;
NEC (effective concepts) matched across contenders at, e.g., {5, 15, 30}.

| Axis | Metric | Applies to |
|---|---|---|
| Task performance | top-1 accuracy (rule-only and +residual); Δ vs frozen backbone probe | all |
| Rule complexity | # rules; mean antecedent length; NEC; per-image explanation size (concepts actually consulted) | all rule heads |
| Faithfulness | head-is-the-classifier (by construction: tree/RRL/CMR/DCR = ✓); concept-layer fidelity via deletion/insertion AUC of concept ablations | all |
| Intervenability | accuracy after oracle-correcting k concepts on misclassified images; effect of ablating top-rule concepts | two-stage contenders |
| Leakage probe | accuracy of a strong MLP on binarized vs soft concepts (gap ≈ leakage); hard-vs-soft head accuracy gap | C1–C5 |
| Stability | concept-cluster ARI across seeds (sources); ruleset Jaccard overlap across seeds (heads) | all |
| Dynamics diagnostics | rule-churn curve during training (D1: threshold drift; D3: memory edit distance per epoch); per-image rule diversity at test time (D2/D3: how many distinct rules actually fire) | C3-RRL, C4, C5 |
| Qualitative | per-contender "explanation card" for the same 8 fixed images: fired rule rendered as text + exemplars/heatmaps | all |

The **dynamics diagnostics** row is the novel reporting element — nobody's
notebook shows "watch the ruleset change as the model learns", and the
Hebbian memory's EMA gives the concept-side twin of that plot for free.
This is the demo's signature move: *concept formation (Hebbian EMA) and
rule formation (threshold drift / memory churn) on one timeline.*

---

## 8. Notebook expansion design (`notebooks/explainability_demo.ipynb` → comparison sections)

Design only — cells to be implemented by Opus agents:

1. **§7 (new) — From influence to rules.** Motivation cell: show the
   current `unit_class_influence` output next to a hand-rendered threshold
   rule for one image; state the gap this section closes.
2. **§8 — Concept providers.** One cell per source (Hebbian [existing
   memory], SAE, CLIP-named), each ending in the same
   `[n_images, m]` concept matrix + `concept_names` + exemplar grid.
   Provider registry mirrors `build_loader`.
3. **§9 — Rule heads.** Sparse-linear, CART (rendered tree with thresholds
   over named concepts), RRL, CMR, DCR — each a registered head trained on
   any provider's matrix; each prints its ruleset (global) and its fired
   rule (for the 8 fixed demo images).
4. **§10 — The scoreboard.** One DataFrame per §7's metric table; the
   accuracy-vs-NEC curve plot; the rule-churn-over-training plot aligned
   with the Hebbian co-activation heatmap timeline.
5. **§11 — Interventions.** Interactive-ish cell: clamp a concept, rerun
   heads, show which contenders respond correctly.
6. **§12 — Export.** Ruleset + fired-rules into the IVGraph JSON
   (forward-compatible extension: `rules: [{id, class, clauses:
   [{concept_id, op, threshold}], support, per_image_fired}]`).

Runtime budget: §8–§10 must run in ≤ ~15 min CPU on the capped CIFAR split
(providers cached to disk after first run); the full-data variant goes into
`kaggle/` like the existing train/precompute/sae notebooks.

---

## 9. Opus work packages

| WP | Deliverable | Depends | Acceptance |
|---|---|---|---|
| W0 | `ConceptProvider` + `RuleHead` protocols + registries in `hatchvision/` (or `packages/core` if the owner prefers the ViTreous side); Hebbian provider wraps existing `concept_scores` | — | provider/head swap = one string; existing demo unaffected; unit tests for both registries |
| W1 | CART + sparse-linear heads; §7–§9 notebook cells for the Hebbian provider (C1 complete) | W0 | notebook runs top-to-bottom CPU-only; rendered tree uses concept labels; accuracy Δ reported |
| W2 | SAE provider (reuse `kaggle/sae.ipynb` machinery, CIFAR-scaled) + CLIP-named provider (LLM concept bank checked into repo as JSON; CLIP scoring cell) | W0 | all three providers emit identical-shape matrices; exemplar grids per provider |
| W3 | RRL head (vendor [12wang3/rrl](https://github.com/12wang3/rrl) or reimplement the binarization+logic layers minimally) + rule-churn logging | W1 | discrete ruleset extractable & printed; churn plot over epochs |
| W4 | CMR head (adapt [daviddebot/CMR](https://github.com/daviddebot/CMR)) + DCR head (from [pyc / torch-concepts](https://arxiv.org/abs/2304.14068) line) | W2 | per-image fired rule printed; memory contents enumerable; verification demo ("no class-k rule without concept-j") |
| W5 | Scoreboard + leakage/intervention/stability cells (§10–§11); 3-seed run script | W1–W4 | metrics table reproduces within tolerance across reruns; leakage probe implemented as specified |
| W6 | IVGraph/pack rule export + minimal Brain-view rendering of fired rules | W5 | round-trip test: notebook JSON → webapp renders rule nodes |
| S | Stretch: V-IP querier; PIP-Net entrant; HAM10000/CUB rerun of the whole scoreboard | W5 | — |

Sequencing: W0→W1 serial; W2‖W3 after; W4 after W2; W5 gates on all; W6
last. W0–W1 alone already deliver a working "Hebbian concepts → threshold
rules" demo — the minimum viable answer to the original question.

---

## 10. Honest assessment & recommendation

- The literature is unambiguous that the **two-stage concept→rule
  factorization** is the right chassis: it is where all 2023–2026 progress
  (label-free CBMs, DCR, CMR, VLG-CBM, CB-SAEs) concentrates, it slots the
  Hebbian memory in as a swappable provider instead of a load-bearing
  assumption, and it reuses both existing codebases (`hatchvision`
  memory/influence; ViTreous SAE/faithfulness).
- The **Hebbian memory will likely lose the pure-accuracy contest** against
  SAE and CLIP-named concepts on CIFAR — pooled correlational clusters are
  simply coarser. It should still be in the comparison because (a) it is
  free at train time, (b) its temporal EMA enables the concept-formation
  timeline nothing else offers, and (c) losing honestly, measurably, and
  publicly is itself the transparency story this project sells. The H2
  patch-level upgrade (§3.4) is the one investment that could change the
  outcome and is genuinely novel.
- For "dynamically changing rulesets", **CMR is the recommended flagship**
  (explicit, enumerable, verifiable rule memory with per-input neural
  selection — the strongest match to the owner's phrase with published
  NeurIPS 2024 code), **RRL the recommended workhorse** (thresholds
  learned by gradient descent, exactly the tree-style rules requested), and
  **CART the indispensable baseline** that costs one afternoon.
- Expected accuracy cost on CIFAR-10, based on the cited results: ~0–3 pts
  for named-concept CBM + sparse linear; ~1–5 pts more for hard threshold
  rules at readable complexity; recoverable to ≈ backbone with the PCBM-h
  residual at the price of a labeled "black-box remainder" — which is why
  every contender is reported as a curve, not a point.

## 11. Reference list (Phase 2 additions)

- Koh et al., *Concept Bottleneck Models*, ICML 2020
- Espinosa Zarlenga, Barbiero et al., *Concept Embedding Models*, NeurIPS 2022 — [semanticscholar](https://www.semanticscholar.org/paper/ce3036fadfa9692867532fe472ea40b4b81a6dc3)
- Barbiero et al., *Interpretable Neural-Symbolic Concept Reasoning* (DCR), ICML 2023 — [arXiv:2304.14068](https://arxiv.org/abs/2304.14068)
- Debot et al., *Interpretable Concept-Based Memory Reasoning* (CMR), NeurIPS 2024 — [arXiv:2407.15527](https://arxiv.org/abs/2407.15527) · [code](https://github.com/daviddebot/CMR) · [poster](https://neurips.cc/virtual/2024/poster/94840)
- Oikarinen et al., *Label-Free Concept Bottleneck Models*, ICLR 2023
- Yang et al., *LaBo: Language in a Bottle*, CVPR 2023
- Srivastava et al., *VLG-CBM*, NeurIPS 2024 — [project](https://lilywenglab.github.io/VLG-CBM/) · [code](https://github.com/Trustworthy-ML-Lab/VLG-CBM)
- Yuksekgonul et al., *Post-hoc Concept Bottleneck Models*, ICLR 2023 — [openreview](https://openreview.net/pdf?id=nA5AZ8CEyow)
- Shang et al., *Incremental Residual CBMs*, 2024 — [arXiv:2404.08978](https://arxiv.org/html/2404.08978v2)
- Sammani et al., *CLIP-Free, Label-Free, Unsupervised CBMs*, CVPR 2026 — [arXiv:2503.10981](https://arxiv.org/abs/2503.10981)
- Wang et al., *Scalable Rule-Based Representation Learning* (RRL), NeurIPS 2021 / TPAMI 2023 — [arXiv:2109.15103](https://arxiv.org/abs/2109.15103) · [arXiv:2310.14336](https://arxiv.org/abs/2310.14336) · [code](https://github.com/12wang3/rrl)
- Marton et al., *GradTree*, AAAI 2024 · *GRANDE*, ICLR 2024 — [arXiv:2309.17130](https://arxiv.org/abs/2309.17130)
- Frosst & Hinton, *Distilling a Neural Network into a Soft Decision Tree*, 2017 · Kontschieder et al., *Deep Neural Decision Forests*, ICCV 2015
- Wan et al., *NBDT: Neural-Backed Decision Trees*, ICLR 2021 — [arXiv:2004.00221](https://arxiv.org/abs/2004.00221)
- Nauta et al., *ProtoTree*, CVPR 2021 — [arXiv:2012.02046](https://arxiv.org/abs/2012.02046) · *PIP-Net*, CVPR 2023 — [code](https://github.com/M-Nauta/PIPNet)
- Chan et al., *Variational Information Pursuit*, ICLR 2023 — [arXiv:2302.02876](https://arxiv.org/abs/2302.02876) · [code](https://github.com/ryanchankh/VariationalInformationPursuit) · [learned queries](https://arxiv.org/html/2312.11548v1) · [uncertainty-aware V-IP 2025](https://arxiv.org/abs/2506.16742)
- Ciravegna et al., *Logic Explained Networks*, AIJ 2023
- Chen, Bei, Rudin, *Concept Whitening*, Nature Machine Intelligence 2020
- Havasi et al., *Addressing Leakage in CBMs*, NeurIPS 2022 — [paper](https://papers.neurips.cc/paper_files/paper/2022/file/944ecf65a46feb578a43abfd5cddd960-Paper-Conference.pdf) · hard-CBM leakage fixes — [arXiv:2402.05945](https://arxiv.org/abs/2402.05945) · counterpoint — [arXiv:2606.10669](https://arxiv.org/html/2606.10669)
- Laguna et al., *Beyond CBMs: Intervenable Black Boxes*, 2024 — [arXiv:2401.13544](https://arxiv.org/pdf/2401.13544)
- Vision SAEs: *Interpretable & Testable Vision Features via SAEs* — [arXiv:2502.06755](https://arxiv.org/abs/2502.06755) · *Hierarchical SAEs for CLIP*, ICML 2025 — [arXiv:2502.20578](https://arxiv.org/abs/2502.20578) · *Steering CLIP's ViT with SAEs* — [arXiv:2504.08729](https://arxiv.org/abs/2504.08729) · *Concept Bottleneck SAEs* — [arXiv:2512.10805](https://arxiv.org/pdf/2512.10805) · [saev toolkit](https://github.com/OSU-NLP-Group/saev)
- Neuro-fuzzy: *DCNFIS* — [arXiv:2308.06378](https://arxiv.org/html/2308.06378v3) · *MFRBCS*, 2025 — [springer](https://link.springer.com/article/10.1007/s44196-025-01112-y) · evolving fuzzy control survey, 2025 — [sciencedirect](https://www.sciencedirect.com/science/article/pii/S1568494625013717) · SDT+concept-model fusion — [sciencedirect](https://www.sciencedirect.com/science/article/pii/S156849462400406X)
- E-BotCL, 2025 — [mdpi](https://www.mdpi.com/1424-8220/25/8/2398) · CBM survey collection — [github](https://github.com/kkzhang95/Awesome_Concept_Bottleneck_Models) · hierarchical DCR follow-ups — [arXiv:2506.21102](https://arxiv.org/html/2506.21102) · [medical logical rules](https://arxiv.org/html/2505.14049)
