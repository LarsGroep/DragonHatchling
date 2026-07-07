"use client";

/**
 * LoopCaption (S1) — the ambient narrative strip. A subtle, non-blocking line
 * near the top of the workbench that names the current stage in Plain or Expert
 * language, synchronized to the ambient loop via the store's `loopStage` (which
 * the LoopController publishes). On the verdict stage it reveals the predicted
 * label with a confidence that counts up as `t` crosses the final layer — the
 * count-up is read imperatively from the timeline clock so it stays smooth
 * without per-frame React re-renders.
 */
import { useEffect, useRef } from "react";
import { useWorkbench } from "@/src/lib/state/store";
import {
  LOOP_STAGES,
  TIMELINE_MAX,
  stageCopy,
  verdictProgress,
} from "@/src/lib/loop/schedule";

const VERDICT_INDEX = LOOP_STAGES.length - 1;

export function LoopCaption() {
  const mode = useWorkbench((s) => s.mode);
  const loopStage = useWorkbench((s) => s.loopStage);
  const manifest = useWorkbench((s) => s.manifest);
  const confRef = useRef<HTMLSpanElement>(null);

  const stage = LOOP_STAGES[Math.min(loopStage, LOOP_STAGES.length - 1)];
  const copy = stageCopy(stage, mode);
  const isVerdict = loopStage === VERDICT_INDEX;
  const pred = manifest?.prediction;

  // Smooth confidence count-up during the verdict stage (imperative, no re-render).
  useEffect(() => {
    if (!isVerdict || !pred) return;
    let raf = 0;
    const tick = () => {
      raf = requestAnimationFrame(tick);
      const t = useWorkbench.getState().t;
      const p = verdictProgress(t, LOOP_STAGES);
      if (confRef.current) {
        confRef.current.textContent = `${(pred.confidence * p * 100).toFixed(0)}%`;
      }
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [isVerdict, pred]);

  if (!manifest) return null;

  return (
    <div className="pointer-events-none absolute inset-x-0 top-2 z-30 flex justify-center px-4">
      <div className="flex max-w-[42rem] flex-col items-center gap-0.5 rounded-2xl border border-edge/60 bg-void/80 px-6 py-2 text-center shadow-glow shadow-black/60 backdrop-blur-md">
        <div className="flex items-center gap-2 text-[10px] uppercase tracking-[0.25em] text-muted">
          <span className="tabular-nums">
            {loopStage + 1} / {LOOP_STAGES.length}
          </span>
          <span className="text-muted/50">·</span>
          <span className="inline-block h-1.5 w-1.5 animate-pulse rounded-full bg-signal/70" />
          <span>{mode === "expert" ? "forward pass" : "watching it think"}</span>
        </div>

        <div
          key={`${loopStage}-${mode}`}
          className="animate-[fadeIn_360ms_ease-out] text-balance text-lg font-light tracking-wide text-signal drop-shadow-[0_0_18px_rgba(232,238,252,0.25)] sm:text-xl"
        >
          {copy.caption}
        </div>

        <div className="text-[11px] leading-snug text-readout/70">
          {copy.sub}
        </div>

        {isVerdict && pred ? (
          <div className="mt-1 flex items-baseline gap-2 rounded-full border border-image/40 bg-image/10 px-3 py-1">
            <span className="text-[10px] uppercase tracking-widest text-image/80">
              {mode === "expert" ? "argmax" : "answer"}
            </span>
            <span className="text-sm font-medium text-signal">{pred.label}</span>
            <span
              ref={confRef}
              className="tabular-nums text-sm text-image"
              // seed value in case the rAF hasn't ticked yet (e.g. static capture)
            >
              {`${(pred.confidence * verdictProgress(TIMELINE_MAX) * 100).toFixed(0)}%`}
            </span>
          </div>
        ) : null}
      </div>
    </div>
  );
}
