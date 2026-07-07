# ViTreous — "Watch It Think" UX Vision

Design contract for the visualization feedback loop (2026-07-07). Fable 5
orchestrates; Opus agents implement one sprint per work order; every sprint
ends with browser screenshots for the owner's judgment before the next begins.

## North star

A nontechnical person — a doctor, a friend, a recruiter — opens the page,
picks a skin-lesion photo, presses **▶ Watch it think**, and *sees* the model
reason: the image dissolves into glowing particles, attention pulses ripple
through a living network, evidence regions ignite on the photo, concepts
light up with example patches, and a verdict forms with an honest
trust-o-meter. Within two minutes they can answer, correctly: **"should I
believe this prediction?"**

Two audiences, one UI: a **Plain** mode (default: lay language, guided,
cinematic) and an **Expert** mode (current workbench: methods, AUCs, layers).
One toggle; nothing hidden, everything translated.

## The four lay questions the UI must answer (judgment aids)

Nontechnical users can correctly judge a model only if the UI answers these:

1. **"Where is it looking?"** — Evidence view: Grad-CAM/Chefer regions as
   soft outlined glows on the photo with captions ("strongest evidence
   here"). NOT a raw heatmap dump: 2–3 discrete outlined regions, ranked.
2. **"Is that the right place to look?"** — Shortcut detector: for
   dermatoscopy, compute the fraction of attribution mass inside the central
   lesion region vs the border/corners (rulers, markers, vignettes are
   classic shortcuts). Show plainly: "✓ focused on the lesion" or
   "⚠ distracted by the image border — treat with caution".
3. **"How sure is it, and does the explanation hold up?"** — Trust gauge
   (0–100) fusing prediction confidence, deletion-AUC faithfulness, and
   method agreement, with the three ingredients shown as plain sentences
   ("when we hide what it says matters, its confidence collapses — good").
4. **"What does it think this looks like?"** — Concept cards: the firing
   SAE concepts rendered as exemplar-patch strips ("this pattern also
   appears in these 8 training images, mostly Melanoma").

Honesty rule (inherited from §7): every lay visual is a direct rendering of
measured quantities; captions state uncertainty; the trust gauge can and
must read LOW when the numbers are bad. A tool that always reassures is
worse than none.

## Experiential set pieces

- **S1. Inference Theater** — a cinematic auto-replay driven by the existing
  timeline clock: staged captions ("1/5 · the photo becomes 196 patches…"),
  camera-choreographed hand-offs between views, particle dissolve of the
  image into the Gaussian field, attention pulses traveling graph edges as
  moving sparks (the "network firing" feel), confidence bar growing as
  layers deepen. Skippable, scrubbable, loopable.
- **S2. 3D Gaussian relief** — the flagship field gains a 2.5D/3D mode:
  splat height (z) = attribution, orbitable camera, importance as terrain —
  "mountains are what the model cares about". three.js is already in place;
  this is a vertex-shader + camera change, with a 2D/3D toggle.
- **S3. Living network** — graph view upgrades: nodes pulse with activation,
  edges carry animated flow particles weighted by attention, communities
  breathe (hue-grouped), concept nodes join the layer view with thumbnail
  badges. Plain-mode label: "the model's neurons — watch ideas connect".
- **S4. Verdict panel** — prediction + trust gauge + the four lay answers
  composed into one right-rail summary, updating live during replay.

## Sprint plan (one Opus work order each)

| Sprint | Contents | Judged by |
|---|---|---|
| V1 | S1 Inference Theater (staged captions, auto-replay choreography, Plain/Expert toggle scaffold) + S2 3D Gaussian relief | screenshots + feel of the replay |
| V2 | S4 Verdict panel: trust gauge, shortcut detector, evidence outlines with lay captions | can a layperson answer the four questions? |
| V3 | S3 Living network: pulse/flow animations, concept nodes w/ exemplar thumbnails, compare-two-images mode | does it read as "thinking"? |
| V4+ | Owner-feedback-driven polish rounds | owner |

Loop mechanics: Opus implements on the feature branch → orchestrator verifies
(tsc/lint/build/tests + Playwright screenshots in mock mode) → screenshots to
owner → feedback becomes the next work order. Deploy to Vercel when the owner
says a round is worth shipping.

## Constraints

- All computed from existing pack assets — no backend changes needed (the
  shortcut detector derives from the attribution grids; concept thumbnails
  come from the dictionary exemplars + pack images).
- 60 fps target on integrated GPUs; graceful degradation (3D mode optional).
- Mock mode must remain fully demoable; HAM10000 via Supabase env vars.
- Dark-instrument aesthetic stays; Plain mode may soften labels, not looks.
