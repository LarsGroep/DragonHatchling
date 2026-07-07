"use client";

/**
 * Gaussian Feature Field (§7) — the flagship view. 197 anisotropic Gaussians
 * rendered with three.js instanced quads on the near-black instrument canvas;
 * every parameter is a deterministic function of a model-measured quantity, so
 * the view is a *lens, not evidence* (labeled per the §7 honesty rule).
 *
 * PLAYBACK SUBSCRIPTION (perf, §12)
 * ---------------------------------
 * The component deliberately does NOT subscribe to the timeline clock `t`. A
 * single requestAnimationFrame loop reads `useWorkbench.getState().t`
 * imperatively and pushes it to the renderer as a scalar uniform — so smooth
 * playback triggers ZERO React re-renders per frame and ZERO buffer uploads
 * (interpolation happens in the vertex shader). React re-renders only on
 * user-paced changes: pack load, hover, pinned. The highlight set is derived
 * from hover/pinned idxs (layer-independent), so it too needs no `t` dependency.
 *
 * ROUND-TRIP SYNC (§11)
 * ---------------------
 * Emits `{kind:"gaussian", layer: layerForT(t), idx}` on hover (analytic ellipse
 * hit-test); subscribes to the store's hover/pinned, resolves them to token idxs
 * and brightens + outlines the matching Gaussians. Hovering a patch in Image
 * Space lights the Gaussian here; hovering here lights the patch there.
 */
import { useEffect, useMemo, useRef, useState } from "react";
import { useWorkbench, layerForT } from "@/src/lib/state/store";
import { resolve } from "@/src/lib/state/resolver";
import { GaussianFieldRenderer } from "./renderer";
import { useGaussianFieldData } from "./useGaussianFieldData";
import {
  FIELD_MARGIN,
  hitTest,
  interpAll,
  squareToWorld,
  type GaussianInstance,
} from "./interp";

export function GaussianFieldView() {
  const packIndex = useWorkbench((s) => s.packIndex);
  const hover = useWorkbench((s) => s.hover);
  const pinned = useWorkbench((s) => s.pinned);
  const setHover = useWorkbench((s) => s.setHover);
  const togglePin = useWorkbench((s) => s.togglePin);
  const mode = useWorkbench((s) => s.mode);

  const { gaussians, absent, error } = useGaussianFieldData();

  const [is3D, setIs3D] = useState(false);

  const canvasRef = useRef<HTMLCanvasElement>(null);
  const wrapRef = useRef<HTMLDivElement>(null);
  const rendererRef = useRef<GaussianFieldRenderer | null>(null);
  const glFailed = useRef(false);
  const dragRef = useRef({ x: 0, y: 0, moved: 0, down: false });

  // Token idxs lit by the current hover/pinned selection. Layer-independent
  // (we only collect idxs), so this needs no `t` subscription.
  const litIdx = useMemo(() => {
    const set = new Set<number>();
    if (!packIndex) return set;
    const add = (ref: (typeof pinned)[number] | null) => {
      if (!ref) return;
      for (const e of resolve(ref, packIndex, 0).refs) {
        if (e.kind === "gaussian" || e.kind === "token") set.add(e.idx);
      }
    };
    for (const p of pinned) add(p);
    add(hover);
    return set;
  }, [hover, pinned, packIndex]);

  // -- renderer lifecycle: (re)build when the gaussian data changes ---------- //
  useEffect(() => {
    if (!gaussians || !canvasRef.current) return;
    let renderer: GaussianFieldRenderer;
    try {
      renderer = new GaussianFieldRenderer(canvasRef.current);
      renderer.setData(gaussians);
    } catch (e) {
      glFailed.current = true;
      // eslint-disable-next-line no-console
      console.warn("GaussianField: WebGL init failed", e);
      return;
    }
    rendererRef.current = renderer;

    const resize = () => {
      const el = wrapRef.current;
      if (!el) return;
      renderer.resize(el.clientWidth, el.clientHeight);
    };
    resize();
    const ro = new ResizeObserver(resize);
    if (wrapRef.current) ro.observe(wrapRef.current);

    let raf = 0;
    const tick = () => {
      renderer.setT(useWorkbench.getState().t);
      renderer.render();
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);

    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
      renderer.dispose();
      rendererRef.current = null;
    };
  }, [gaussians]);

  // Push highlight changes to the GPU (only when the selection changes).
  useEffect(() => {
    rendererRef.current?.setHighlight(litIdx);
  }, [litIdx]);

  // Apply the 2D/3D mode to the renderer (also re-applied when it rebuilds).
  useEffect(() => {
    rendererRef.current?.setMode3D(is3D);
  }, [is3D, gaussians]);

  // -- pointer → analytic ellipse hit-test → gaussian EntityRef -------------- //
  function pointerToWorld(e: React.PointerEvent): { x: number; y: number } | null {
    const canvas = canvasRef.current;
    if (!canvas) return null;
    const rect = canvas.getBoundingClientRect();
    const side = Math.min(rect.width, rect.height);
    const ox = (rect.width - side) / 2;
    const oy = (rect.height - side) / 2;
    const u = (e.clientX - rect.left - ox) / side;
    const v = (e.clientY - rect.top - oy) / side;
    if (u < 0 || u > 1 || v < 0 || v > 1) return null;
    return squareToWorld(u, v, FIELD_MARGIN);
  }

  function instancesAtNow(): GaussianInstance[] | null {
    if (!gaussians) return null;
    const t = useWorkbench.getState().t;
    return interpAll(gaussians.data, gaussians.steps, gaussians.tokens, t);
  }

  const curLayer = () => layerForT(useWorkbench.getState().t, packIndex?.numLayers ?? 12);

  function handleMove(e: React.PointerEvent) {
    const w = pointerToWorld(e);
    const inst = instancesAtNow();
    if (!w || !inst) {
      if (hover?.kind === "gaussian") setHover(null);
      return;
    }
    const idx = hitTest(w.x, w.y, inst);
    if (idx < 0) {
      if (hover?.kind === "gaussian") setHover(null);
      return;
    }
    setHover({ kind: "gaussian", layer: curLayer(), idx });
  }

  function handleClick(e: React.PointerEvent) {
    const w = pointerToWorld(e);
    const inst = instancesAtNow();
    if (!w || !inst) return;
    const idx = hitTest(w.x, w.y, inst);
    if (idx < 0) return;
    togglePin({ kind: "token", layer: curLayer(), idx });
  }

  // -- 3D relief: drag-to-orbit + screen-space nearest pick ------------------ //
  function pick3DAt(e: React.PointerEvent): number {
    const canvas = canvasRef.current;
    const r = rendererRef.current;
    if (!canvas || !r) return -1;
    const rect = canvas.getBoundingClientRect();
    return r.pick3D(e.clientX - rect.left, e.clientY - rect.top, rect.width, rect.height);
  }

  function onPointerDown(e: React.PointerEvent) {
    if (!is3D) {
      handleClick(e);
      return;
    }
    dragRef.current = { x: e.clientX, y: e.clientY, moved: 0, down: true };
    rendererRef.current?.setDragging(true);
    (e.target as Element).setPointerCapture?.(e.pointerId);
  }

  function onPointerMove(e: React.PointerEvent) {
    if (!is3D) {
      handleMove(e);
      return;
    }
    const d = dragRef.current;
    if (d.down) {
      const dx = e.clientX - d.x;
      const dy = e.clientY - d.y;
      d.moved += Math.abs(dx) + Math.abs(dy);
      d.x = e.clientX;
      d.y = e.clientY;
      rendererRef.current?.dragOrbit(dx, dy);
      return;
    }
    const idx = pick3DAt(e);
    if (idx < 0) {
      if (hover?.kind === "gaussian") setHover(null);
      return;
    }
    setHover({ kind: "gaussian", layer: curLayer(), idx });
  }

  function onPointerUp(e: React.PointerEvent) {
    if (!is3D) return;
    const d = dragRef.current;
    d.down = false;
    rendererRef.current?.setDragging(false);
    if (d.moved < 5) {
      const idx = pick3DAt(e);
      if (idx >= 0) togglePin({ kind: "token", layer: curLayer(), idx });
    }
  }

  const ready = !!gaussians && !glFailed.current;

  return (
    <div
      ref={wrapRef}
      data-testid="gaussian-field"
      data-gaussian-ready={ready ? "1" : "0"}
      data-gaussian-highlights={litIdx.size}
      data-gaussian-absent={absent ? "1" : "0"}
      className="relative h-full w-full overflow-hidden rounded bg-black"
    >
      <canvas
        ref={canvasRef}
        onPointerMove={onPointerMove}
        onPointerLeave={() => !dragRef.current.down && hover?.kind === "gaussian" && setHover(null)}
        onPointerDown={onPointerDown}
        onPointerUp={onPointerUp}
        className={`absolute inset-0 h-full w-full ${is3D ? "cursor-grab active:cursor-grabbing" : ""}`}
      />

      {/* 2D / 3D relief toggle (S2) */}
      {ready ? (
        <div className="absolute right-2 top-2 flex gap-1">
          {(["2D", "3D"] as const).map((m) => {
            const active = (m === "3D") === is3D;
            return (
              <button
                key={m}
                type="button"
                onClick={() => setIs3D(m === "3D")}
                title={m === "3D" ? "3D relief — height = attribution" : "2D field"}
                className={`rounded border px-1.5 py-0.5 text-[10px] uppercase tracking-widest ${
                  active
                    ? "border-gauss/60 bg-gauss/10 text-gauss"
                    : "border-edge text-muted hover:text-readout"
                }`}
              >
                {m}
              </button>
            );
          })}
        </div>
      ) : null}

      {/* honesty label + visual-encoding legend (§7) */}
      {ready ? (
        <div className="pointer-events-none absolute left-2 top-2 flex flex-col gap-1 rounded border border-edge/70 bg-panel/70 px-2 py-1.5 text-[9px] leading-tight backdrop-blur-sm">
          <div className="flex items-center gap-1 text-gauss">
            <span className="inline-block h-2 w-2 rounded-full bg-gauss" />
            <span className="uppercase tracking-widest">lens, not evidence</span>
          </div>
          <LegendRow color="#c8d3e6" label="opacity = activation" />
          <LegendRow color="#b5179e" label={is3D ? "glow + height = attribution" : "glow = attribution"} />
          <LegendRow color="#4cc9f0" label="halo = attention-in" />
          {is3D && mode === "plain" ? (
            <div className="mt-0.5 max-w-[19ch] text-[9px] leading-tight text-readout/70">
              height = how much this spot influenced the answer
            </div>
          ) : null}
        </div>
      ) : null}

      {/* CLS gutter caption */}
      {ready ? (
        <div className="pointer-events-none absolute bottom-1 left-2 text-[8px] uppercase tracking-widest text-muted/70">
          CLS
        </div>
      ) : null}

      {/* status overlays */}
      {!gaussians ? (
        <div className="pointer-events-none absolute inset-0 flex items-center justify-center text-center text-[11px] tracking-widest text-muted">
          {absent
            ? "no gaussian asset"
            : error
              ? `gaussian load failed: ${error}`
              : glFailed.current
                ? "WebGL unavailable"
                : "loading field…"}
        </div>
      ) : null}
    </div>
  );
}

function LegendRow({ color, label }: { color: string; label: string }) {
  return (
    <div className="flex items-center gap-1 text-muted">
      <span className="inline-block h-2 w-2 rounded-full" style={{ backgroundColor: color }} />
      <span>{label}</span>
    </div>
  );
}
