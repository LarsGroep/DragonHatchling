/**
 * Ambient inference-loop schedule (S1) — the pure, WebGL/React-free logic that
 * turns the timeline clock `t ∈ [0, 12]` into (a) a narrative stage with lay /
 * expert captions and (b) an ease-in-out sweep velocity. Kept import-free of any
 * view so vitest exercises it headless (§11: no view-to-view coupling — the loop
 * publishes stage state through the store and every pane subscribes).
 *
 * The workbench is ALWAYS ALIVE: `t` sweeps 0→12 continuously (this module gives
 * the per-frame velocity, slowing to a soft ease at every stage boundary), holds
 * briefly at the verdict, fades, and restarts. Nothing here mutates state; the
 * LoopController integrates `loopVelocity` and drives the phase machine.
 */

/** The four synchronized panes a stage can spotlight (mirrors PanelAccent). */
export type FeaturePane = "image" | "gauss" | "graph" | "latent";

export interface StageCopy {
  caption: string;
  sub: string;
}

export interface LoopStage {
  id: string;
  /** Inclusive timeline start (in layers). */
  tStart: number;
  /** Exclusive timeline end (inclusive for the final stage). */
  tEnd: number;
  /** Which pane this stage is "about" (gets the gentle vignette pulse). */
  feature: FeaturePane;
  plain: StageCopy;
  expert: StageCopy;
}

/** Full timeline span in layers; the store clamps `t` to [0, TIMELINE_MAX]. */
export const TIMELINE_MAX = 12;

/**
 * The narrative. Captions differ by mode (plain = lay language; expert = layer
 * numbers + method names). Boundaries are tuned so each stage lands on a real
 * phase of the forward pass.
 */
export const LOOP_STAGES: LoopStage[] = [
  {
    id: "patchify",
    tStart: 0,
    tEnd: 1,
    feature: "image",
    plain: {
      caption: "The photo becomes 196 patches",
      sub: "Each square is one piece the model reads.",
    },
    expert: {
      caption: "Patch embedding · 14×14 grid → 196 tokens + CLS",
      sub: "conv stem, patch_size=16, embed_dim=384",
    },
  },
  {
    id: "attention",
    tStart: 1,
    tEnd: 5,
    feature: "graph",
    plain: {
      caption: "Each patch looks at the others — attention forms",
      sub: "Links show which patches compare notes.",
    },
    expert: {
      caption: "Self-attention · layers 1–4",
      sub: "multi-head softmax(QKᵀ/√d); top-k edges shown",
    },
  },
  {
    id: "strengthen",
    tStart: 5,
    tEnd: 8,
    feature: "gauss",
    plain: {
      caption: "Patterns strengthen; some patches matter more",
      sub: "Brighter spots carry more of the decision.",
    },
    expert: {
      caption: "Layers 5–8 · attribution (Chefer) accumulates",
      sub: "opacity ∝ ‖token‖, glow ∝ attribution",
    },
  },
  {
    id: "concentrate",
    tStart: 8,
    tEnd: 11,
    feature: "gauss",
    plain: {
      caption: "Evidence concentrates",
      sub: "The model settles on where to look.",
    },
    expert: {
      caption: "Layers 9–11 · importance diffusion",
      sub: "attribution mass localizes toward the class evidence",
    },
  },
  {
    id: "verdict",
    tStart: 11,
    tEnd: 12,
    feature: "image",
    plain: {
      caption: "A prediction forms",
      sub: "The model commits to an answer.",
    },
    expert: {
      caption: "Layer 12 · CLS → linear head → softmax",
      sub: "argmax over class logits",
    },
  },
];

/** Clamp to [0, 1]. */
export function clamp01(x: number): number {
  return x < 0 ? 0 : x > 1 ? 1 : x;
}

/** Cubic smoothstep 3x²−2x³ on a clamped input — our canonical ease-in-out. */
export function easeInOut(x: number): number {
  const c = clamp01(x);
  return c * c * (3 - 2 * c);
}

/** Index of the stage containing `t` (final stage is inclusive at TIMELINE_MAX). */
export function stageForT(t: number, stages: LoopStage[] = LOOP_STAGES): number {
  for (let i = 0; i < stages.length; i++) {
    if (t < stages[i].tEnd) return i;
  }
  return stages.length - 1;
}

/** The mode-appropriate copy for a stage. */
export function stageCopy(stage: LoopStage, mode: "plain" | "expert"): StageCopy {
  return mode === "expert" ? stage.expert : stage.plain;
}

/** Base sweep rate (layers/sec) at a stage's midpoint. ~1 layer/s on average. */
export const LOOP_BASE_LPS = 1.25;
/** Velocity floor as a fraction of base — the ease-in-out slow-down at boundaries. */
export const LOOP_EASE_MIN = 0.35;

/**
 * Sweep velocity (layers/sec) at timeline position `t`. A pure function of `t`
 * (so the loop resumes seamlessly from any scrub position): within each stage
 * the rate follows a half-sine, dipping to `min·base` at both boundaries and
 * peaking at the stage midpoint — giving a gentle ease-in-out at every hand-off
 * without any elapsed-time bookkeeping.
 */
export function loopVelocity(
  t: number,
  stages: LoopStage[] = LOOP_STAGES,
  base: number = LOOP_BASE_LPS,
  min: number = LOOP_EASE_MIN,
): number {
  const i = stageForT(t, stages);
  const s = stages[i];
  const span = s.tEnd - s.tStart || 1;
  const frac = clamp01((t - s.tStart) / span);
  const factor = min + (1 - min) * Math.sin(Math.PI * frac);
  return base * factor;
}

/**
 * Verdict reveal progress (0→1) as `t` crosses the final stage — drives the
 * confidence count-up. 0 before the verdict stage, 1 at t = TIMELINE_MAX.
 */
export function verdictProgress(t: number, stages: LoopStage[] = LOOP_STAGES): number {
  const s = stages[stages.length - 1];
  return easeInOut((t - s.tStart) / (s.tEnd - s.tStart || 1));
}
