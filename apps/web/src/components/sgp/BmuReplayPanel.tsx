"use client";

/**
 * BmuReplayPanel — the image side of the SGP sync. Shows the active probe
 * image with its BMU map at the scrubbed depth painted over it as a soft
 * community-hued grid, plus the migration curve (fraction of voxels
 * re-assigned between consecutive depths).
 *
 * Bidirectional sync, mirroring the workbench's §11 discipline:
 *   • hovering/pinning a NEURON (in the lattice) outlines the voxel cells it
 *     owns at the current depth here;
 *   • hovering a CELL here emits that cell's BMU neuron (and the lattice
 *     highlights it).
 * Every painted value is measured: cell hue = the BMU neuron's community; cell
 * opacity is a fixed display exposure.
 */
import { useEffect, useMemo, useRef } from "react";
import type { SgpProbe, SgpSom } from "@/src/lib/sgp";
import { communityRgb } from "./lattice";

const OVERLAY_ALPHA = 0.42;

export interface BmuReplayPanelProps {
  som: SgpSom;
  probe: SgpProbe;
  /** Integer depth to display (the parent rounds the fractional clock). */
  depth: number;
  hoverNeuron: number | null;
  pinnedNeuron: number | null;
  onHoverNeuron: (k: number | null) => void;
  migration: number[];
}

export function BmuReplayPanel({
  som,
  probe,
  depth,
  hoverNeuron,
  pinnedNeuron,
  onHoverNeuron,
  migration,
}: BmuReplayPanelProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const imgRef = useRef<HTMLImageElement | null>(null);

  const communityOf = useMemo(() => {
    const m = new Map<number, number>();
    for (const n of som.nodes) m.set(n.idx, n.community);
    return m;
  }, [som.nodes]);

  // Decode the thumbnail once per probe.
  useEffect(() => {
    imgRef.current = null;
    if (!probe.thumb_png_b64) return;
    const img = new Image();
    img.onload = () => {
      imgRef.current = img;
      draw();
    };
    img.src = `data:image/png;base64,${probe.thumb_png_b64}`;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [probe]);

  function draw() {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const slice = probe.bmu[Math.max(0, Math.min(probe.bmu.length - 1, depth))];
    const rows = slice.length;
    const cols = slice[0]?.length ?? 0;
    if (!rows || !cols) return;

    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    canvas.width = w * dpr;
    canvas.height = h * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);

    // the probe image beneath.
    if (imgRef.current) {
      ctx.imageSmoothingEnabled = true;
      ctx.drawImage(imgRef.current, 0, 0, w, h);
    } else {
      ctx.fillStyle = "#0e1626";
      ctx.fillRect(0, 0, w, h);
    }

    const cw = w / cols;
    const ch = h / rows;
    const active = hoverNeuron ?? pinnedNeuron;

    for (let y = 0; y < rows; y++) {
      for (let x = 0; x < cols; x++) {
        const k = slice[y][x];
        const [r, g, b] = communityRgb(communityOf.get(k) ?? 0);
        ctx.fillStyle = `rgba(${Math.round(r * 255)},${Math.round(g * 255)},${Math.round(b * 255)},${OVERLAY_ALPHA})`;
        ctx.fillRect(x * cw, y * ch, cw + 0.5, ch + 0.5);
      }
    }
    // outline the active neuron's cells on top (the sync highlight).
    if (active !== null) {
      ctx.strokeStyle = "rgba(255,255,255,0.95)";
      ctx.lineWidth = 1.5;
      for (let y = 0; y < rows; y++)
        for (let x = 0; x < cols; x++)
          if (slice[y][x] === active) ctx.strokeRect(x * cw + 1, y * ch + 1, cw - 2, ch - 2);
    }
  }

  // Redraw on any input change (cheap: ≤ 16×16 cells).
  useEffect(() => {
    draw();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [depth, hoverNeuron, pinnedNeuron, probe, som]);

  // Cell hover → neuron.
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const onMove = (e: PointerEvent) => {
      const slice = probe.bmu[Math.max(0, Math.min(probe.bmu.length - 1, depth))];
      const rows = slice.length;
      const cols = slice[0]?.length ?? 0;
      const rect = canvas.getBoundingClientRect();
      const x = Math.floor(((e.clientX - rect.left) / rect.width) * cols);
      const y = Math.floor(((e.clientY - rect.top) / rect.height) * rows);
      if (y >= 0 && y < rows && x >= 0 && x < cols) onHoverNeuron(slice[y][x]);
    };
    const onLeave = () => onHoverNeuron(null);
    canvas.addEventListener("pointermove", onMove);
    canvas.addEventListener("pointerleave", onLeave);
    return () => {
      canvas.removeEventListener("pointermove", onMove);
      canvas.removeEventListener("pointerleave", onLeave);
    };
  }, [probe, depth, onHoverNeuron]);

  const maxMig = Math.max(0.001, ...migration);

  return (
    <div className="flex flex-col gap-2">
      <div className="relative mx-auto aspect-square w-full max-w-[240px] overflow-hidden rounded-lg border border-edge">
        <canvas ref={canvasRef} className="h-full w-full cursor-crosshair" />
      </div>
      {/* migration sparkline — measured re-assignment between depths */}
      <div className="flex flex-col gap-1">
        <span className="text-[10px] uppercase tracking-wide text-muted">
          BMU migration between depths
        </span>
        <div className="flex h-9 items-end gap-[3px]">
          {migration.map((m, i) => (
            <div
              key={i}
              title={`z=${i}→${i + 1}: ${(m * 100).toFixed(0)}% re-assigned`}
              className="flex-1 rounded-t bg-warm/70"
              style={{ height: `${Math.max(6, (m / maxMig) * 100)}%` }}
            />
          ))}
        </div>
        <div className="flex justify-between font-mono text-[9px] text-muted">
          <span>z=0→1</span>
          <span>
            z={migration.length - 1}→{migration.length}
          </span>
        </div>
      </div>
    </div>
  );
}
