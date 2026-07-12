"use client";

/**
 * SomLatticeView — React shell around `SomLatticeRenderer`. Owns the canvas,
 * the rAF loop, pointer handling (drag = orbit, hover = pick, click = pin),
 * and the per-frame activation EMA. Fully controlled: the parent owns the
 * fractional depth `t`, the probe's per-depth activation histograms, and the
 * hover/pin selection — this component renders and emits, exactly like the
 * workbench views (§11 one-selection-model discipline, without the store since
 * /sgp is a standalone surface).
 */
import { useEffect, useRef } from "react";
import type { SgpSom } from "@/src/lib/sgp";
import { activationAt, emaToward } from "./lattice";
import { SomLatticeRenderer } from "./latticeRenderer";

export interface SomLatticeViewProps {
  som: SgpSom;
  /** Per-depth activation histograms for the active probe ([Z][K], each sums 1). */
  activations: Float32Array[] | null;
  /** Fractional depth (0..Z-1) — the transport clock. */
  t: number;
  hoverNeuron: number | null;
  pinnedNeuron: number | null;
  onHover: (k: number | null) => void;
  onPick: (k: number | null) => void;
}

export function SomLatticeView({
  som,
  activations,
  t,
  hoverNeuron,
  pinnedNeuron,
  onHover,
  onPick,
}: SomLatticeViewProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const rendererRef = useRef<SomLatticeRenderer | null>(null);
  // Live values read inside the rAF loop without re-subscribing it.
  const tRef = useRef(t);
  tRef.current = t;
  const actsRef = useRef(activations);
  actsRef.current = activations;

  // Renderer lifecycle.
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const renderer = new SomLatticeRenderer(canvas);
    rendererRef.current = renderer;

    const ro = new ResizeObserver(() => {
      renderer.resize(canvas.clientWidth, canvas.clientHeight);
    });
    ro.observe(canvas);
    renderer.resize(canvas.clientWidth, canvas.clientHeight);

    const K = 4096; // upper bound before setSom; replaced on first bind below
    let current = new Float32Array(K);
    let target = new Float32Array(K);
    let prev = performance.now();
    let raf = 0;
    const tick = (now: number) => {
      raf = requestAnimationFrame(tick);
      const dt = Math.min((now - prev) / 1000, 0.05);
      prev = now;
      const acts = actsRef.current;
      if (acts && acts.length > 0) {
        const k = acts[0].length;
        if (current.length !== k) {
          current = new Float32Array(k);
          target = new Float32Array(k);
        }
        activationAt(acts, tRef.current, target);
        emaToward(current, target, dt);
        renderer.setActivation(current);
      }
      renderer.render();
    };
    raf = requestAnimationFrame(tick);

    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
      renderer.dispose();
      rendererRef.current = null;
    };
  }, []);

  // (Re)bind the SOM.
  useEffect(() => {
    rendererRef.current?.setSom(som);
  }, [som]);

  // Selection highlight.
  useEffect(() => {
    const lit = new Set<number>();
    if (hoverNeuron !== null) lit.add(hoverNeuron);
    if (pinnedNeuron !== null) lit.add(pinnedNeuron);
    rendererRef.current?.setHighlight(lit);
  }, [hoverNeuron, pinnedNeuron]);

  // Pointer: drag orbits; still-hover picks; click pins.
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    let down = false;
    let moved = false;
    let lastX = 0;
    let lastY = 0;

    const onPointerDown = (e: PointerEvent) => {
      down = true;
      moved = false;
      lastX = e.clientX;
      lastY = e.clientY;
      canvas.setPointerCapture(e.pointerId);
      rendererRef.current?.setDragging(true);
    };
    const onPointerMove = (e: PointerEvent) => {
      const rect = canvas.getBoundingClientRect();
      if (down) {
        const dx = e.clientX - lastX;
        const dy = e.clientY - lastY;
        if (Math.abs(dx) + Math.abs(dy) > 2) moved = true;
        rendererRef.current?.dragOrbit(dx, dy);
        lastX = e.clientX;
        lastY = e.clientY;
      } else {
        const k = rendererRef.current?.pick(
          e.clientX - rect.left,
          e.clientY - rect.top,
          rect.width,
          rect.height,
        );
        onHover(k !== undefined && k >= 0 ? k : null);
      }
    };
    const onPointerUp = (e: PointerEvent) => {
      rendererRef.current?.setDragging(false);
      if (down && !moved) {
        const rect = canvas.getBoundingClientRect();
        const k = rendererRef.current?.pick(
          e.clientX - rect.left,
          e.clientY - rect.top,
          rect.width,
          rect.height,
        );
        onPick(k !== undefined && k >= 0 ? k : null);
      }
      down = false;
    };
    const onLeave = () => {
      onHover(null);
    };

    canvas.addEventListener("pointerdown", onPointerDown);
    canvas.addEventListener("pointermove", onPointerMove);
    canvas.addEventListener("pointerup", onPointerUp);
    canvas.addEventListener("pointerleave", onLeave);
    return () => {
      canvas.removeEventListener("pointerdown", onPointerDown);
      canvas.removeEventListener("pointermove", onPointerMove);
      canvas.removeEventListener("pointerup", onPointerUp);
      canvas.removeEventListener("pointerleave", onLeave);
    };
  }, [onHover, onPick]);

  return (
    <div className="relative h-full w-full overflow-hidden rounded-lg">
      <canvas ref={canvasRef} className="h-full w-full cursor-crosshair touch-none" />
      {/* Axis honesty label (SGP §1 / UMT-ViT convention). */}
      <span className="pointer-events-none absolute bottom-2 left-2 rounded bg-black/35 px-1.5 py-0.5 font-mono text-[9px] tracking-wide text-white/70">
        ↑ z = learned hierarchy (encoder depth) · positions = real neuron lattice
      </span>
    </div>
  );
}
