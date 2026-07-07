"use client";

/**
 * LoopController (S1) — the ambient inference-replay clock. The workbench is
 * ALWAYS ALIVE: on image select the store sets `playing`, and this controller
 * sweeps the timeline `t` 0→12 forever with an ease-in-out velocity
 * (loop/schedule.ts), holds briefly at the verdict, fades, and restarts — a
 * seamless fade masks the 12→0 reset (no hard jump).
 *
 * It publishes only the narrative STAGE index to the store (§11: no view-to-view
 * coupling — panes subscribe to `loopStage`); everything else is read
 * imperatively via getState() so the per-frame clock triggers zero React
 * re-renders. The fade veil opacity is written straight to a ref.
 *
 * Interaction model: scrubbing/stepping pauses the loop (store.scrub/stepLayer);
 * hovering a pane does NOT (hover never touches `t`). After ~8s idle the loop
 * resumes on its own. While paused the workbench keeps its idle micro-motion
 * (splat shimmer + 3D auto-orbit in the renderer; vignette pulse in CSS).
 */
import { useEffect, useRef } from "react";
import { useWorkbench } from "@/src/lib/state/store";
import { LOOP_STAGES, TIMELINE_MAX, loopVelocity, stageForT } from "@/src/lib/loop/schedule";

const HOLD_MS = 1900; // dwell at the verdict (t=12) with the prediction visible
const FADE_OUT_MS = 420; // veil fades in while we reset t 12→0
const FADE_IN_MS = 560; // veil fades back out as the next sweep begins
const IDLE_MS = 8000; // untouched-for-this-long → resume the loop
const VEIL_MAX = 0.9;

type Phase = "sweep" | "hold" | "fadeout" | "fadein";

export function LoopController() {
  const veilRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    let raf = 0;
    let last = 0;
    let phase: Phase = "sweep";
    let phaseElapsed = 0;
    let prevPlaying = useWorkbench.getState().playing;

    const setVeil = (o: number) => {
      if (veilRef.current) veilRef.current.style.opacity = o.toFixed(3);
    };

    function resetToSweep() {
      phase = "sweep";
      phaseElapsed = 0;
      setVeil(0);
    }

    function tick(now: number) {
      raf = requestAnimationFrame(tick);
      const st = useWorkbench.getState();
      const dt = last ? Math.min((now - last) / 1000, 0.05) : 0;
      last = now;

      // idle-resume: a paused loop wakes itself after IDLE_MS untouched.
      if (!st.playing) {
        if (st.lastInteraction > 0 && Date.now() - st.lastInteraction > IDLE_MS) {
          st.play(); // clears lastInteraction; loop resumes next frame
          resetToSweep();
        }
      }

      // A fresh resume (paused → playing) always restarts the phase machine so
      // the sweep continues cleanly from wherever `t` currently sits.
      if (st.playing && !prevPlaying) resetToSweep();
      prevPlaying = st.playing;

      // publish the current stage (cheap: only writes on change).
      st.setLoopStage(stageForT(st.t, LOOP_STAGES));

      if (!st.playing) return; // frozen; renderer/CSS keep the micro-motion

      switch (phase) {
        case "sweep": {
          let nt = st.t + loopVelocity(st.t) * dt;
          if (nt >= TIMELINE_MAX) {
            nt = TIMELINE_MAX;
            phase = "hold";
            phaseElapsed = 0;
          }
          st.setT(nt);
          break;
        }
        case "hold": {
          phaseElapsed += dt * 1000;
          if (phaseElapsed >= HOLD_MS) {
            phase = "fadeout";
            phaseElapsed = 0;
          }
          break;
        }
        case "fadeout": {
          phaseElapsed += dt * 1000;
          const p = Math.min(1, phaseElapsed / FADE_OUT_MS);
          setVeil(p * VEIL_MAX);
          if (p >= 1) {
            st.setT(0); // reset hidden behind the veil — seamless
            st.setLoopStage(0);
            phase = "fadein";
            phaseElapsed = 0;
          }
          break;
        }
        case "fadein": {
          phaseElapsed += dt * 1000;
          const p = Math.min(1, phaseElapsed / FADE_IN_MS);
          setVeil((1 - p) * VEIL_MAX);
          if (p < 1) st.setT(loopVelocity(0) * (phaseElapsed / 1000)); // ease the sweep back in
          if (p >= 1) resetToSweep();
          break;
        }
      }
    }

    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, []);

  return (
    <div
      ref={veilRef}
      aria-hidden
      className="pointer-events-none absolute inset-0 z-20 bg-void transition-none"
      style={{ opacity: 0 }}
    />
  );
}
