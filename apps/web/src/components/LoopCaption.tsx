"use client";

/**
 * LoopCaption (B1) — the ambient narrative strip, restyled to a clean light band
 * that lives in its OWN reserved row above the brain (fixing the V1 caption /
 * header collision). It names the current loop stage in Plain or Expert language
 * via the store's `loopStage`, and on the verdict stage reveals the predicted
 * label with a confidence that counts up as `t` crosses the final layer — read
 * imperatively so it stays smooth without per-frame React re-renders.
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

  useEffect(() => {
    if (!isVerdict || !pred) return;
    let raf = 0;
    const tick = () => {
      raf = requestAnimationFrame(tick);
      const p = verdictProgress(useWorkbench.getState().t, LOOP_STAGES);
      if (confRef.current) {
        confRef.current.textContent = `${(pred.confidence * p * 100).toFixed(0)}%`;
      }
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [isVerdict, pred]);

  if (!manifest) {
    return <div className="h-9" aria-hidden />;
  }

  return (
    <div className="flex h-9 items-center justify-center gap-3 px-4">
      <div className="flex items-center gap-2 text-[10px] uppercase tracking-[0.18em] text-muted">
        <span className="inline-block h-1.5 w-1.5 animate-pulse rounded-full bg-image" />
        <span className="hidden sm:inline">
          {mode === "expert" ? "forward pass" : "watching it think"}
        </span>
        <span className="font-mono tabular-nums">
          {loopStage + 1}/{LOOP_STAGES.length}
        </span>
      </div>

      <div
        key={`${loopStage}-${mode}`}
        className="animate-[fadeIn_320ms_ease-out] truncate text-sm font-medium text-signal"
        title={copy.sub}
      >
        {copy.caption}
      </div>

      {isVerdict && pred ? (
        <div className="flex items-baseline gap-2 rounded-full border border-evidence/30 bg-evidence/10 px-2.5 py-0.5">
          <span className="text-[9px] uppercase tracking-widest text-evidence/80">
            {mode === "expert" ? "argmax" : "answer"}
          </span>
          <span className="text-xs font-semibold text-signal">{pred.label}</span>
          <span ref={confRef} className="font-mono text-xs tabular-nums text-evidence">
            {`${(pred.confidence * verdictProgress(TIMELINE_MAX) * 100).toFixed(0)}%`}
          </span>
        </div>
      ) : null}
    </div>
  );
}
