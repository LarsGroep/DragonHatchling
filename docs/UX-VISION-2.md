# ViTreous — Hebbian Brain First (v2 design brief, supersedes UX-VISION.md)

Owner-authored brief, 2026-07-07. The Hebbian graph IS the identity of
ViTreous; every other visual explains why regions of it activate.

## Vision
A window into the model's brain. The graph occupies 60–70% of the screen;
a first-time user thinks "I am watching the AI reason", never "a dashboard".
Scientific instrument, not debugging tool.

## Design language
Clean/professional: Apple, Notion, Linear, Figma, Obsidian graph view.
AVOID: cyberpunk, neon glow, HUDs, gaming UI, heavy gradients.
Palette: white background, light-gray panels, dark type, soft blue =
activity, green = confirmed evidence, orange = intermediate activation,
purple = latent space only. Animations subtle, smooth, purposeful.

## Layout
Top ~65%: the Brain (Obsidian-style force-directed graph; smooth pan/zoom,
soft curved edges, small circular nodes, subtle gray edges, clusters emerge
naturally, NO rigid layer grid, alive even before inference).
Bottom strip: [Input image + GradCAM] [Gaussian field] [Concepts + Prediction].

## Inference = activation flowing through memory (continuous loop)
image → evidence extracted → signals enter graph → local regions activate →
propagation → clusters reinforce → strong pathways → one region dominates →
prediction. NEVER flash the whole graph; only small portions at a time.
Nodes: gradually brighten, slightly grow, slowly fade. Edges: soft
illumination, traveling pulses, fade after passage. Active clusters brighten
subtly while the rest stays quiet.

## Labels
Most nodes unlabeled. Community labels fade in only while their cluster is
active, then fade out (e.g. "Border Irregularity", "Pigment Variation").
Labels MUST derive from real concept-dictionary data (exemplar dominant
classes / stats) — never fabricated semantics (honesty rule §7 holds).

## Gaussian field = the bridge
Image features become particles that drift toward and merge into the
activated graph communities; denser flow = stronger evidence. No random
motion — every particle movement represents measured evidence.

## Concepts panel
Only active concepts appear, as activation reaches their community;
confidence bars grow gradually; prediction appears after concepts stabilize.

## Interaction
Hover node → connected nodes + originating image region + particle path +
concept label. Hover image region → particle stream + community + concepts.
All synchronized via the existing selection store (§11).

## Expert mode
Default = clarity. Expert reveals neuron metadata, heads, embeddings,
layer activations, covariance, rollout, graph stats — as overlays, never
replacing the brain.
