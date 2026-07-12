"use client";

/**
 * UMatrixPanel — the map's cluster structure as per-depth-layer small
 * multiples: the U-matrix (viridis; bright walls = cluster boundaries) with
 * BMU hit counts available behind a toggle (hot). Clicking a cell pins that
 * neuron everywhere (lattice + image overlay); the active neuron's cell is
 * outlined. Same GridCanvas approach as the UMT-ViT SomPanel, driven by the
 * flat per-neuron arrays of `som.json` reshaped by the real lattice grid.
 */
import { useMemo, useState } from "react";
import type { SgpSom } from "@/src/lib/sgp";
import { hot, viridis, rgbCss } from "../umtvit/colormaps";

const CELL = 20;

export interface UMatrixPanelProps {
  som: SgpSom;
  hoverNeuron: number | null;
  pinnedNeuron: number | null;
  onHoverNeuron: (k: number | null) => void;
  onPinNeuron: (k: number | null) => void;
}

export function UMatrixPanel({
  som,
  hoverNeuron,
  pinnedNeuron,
  onHoverNeuron,
  onPinNeuron,
}: UMatrixPanelProps) {
  const [mode, setMode] = useState<"umatrix" | "hits">("umatrix");
  const [gz, gy, gx] = som.grid;

  const { values, vmax } = useMemo(() => {
    const vals = new Float64Array(som.num_neurons);
    let max = 0;
    for (const n of som.nodes) {
      const v = mode === "umatrix" ? n.umatrix : n.hits;
      vals[n.idx] = v;
      if (v > max) max = v;
    }
    return { values: vals, vmax: max || 1 };
  }, [som, mode]);

  const cmap = mode === "umatrix" ? viridis : hot;
  const active = hoverNeuron ?? pinnedNeuron;

  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-center gap-2">
        <span className="text-[10px] uppercase tracking-wide text-muted">
          {mode === "umatrix" ? "U-matrix (bright = cluster wall)" : "BMU hits (bright = busy)"}
        </span>
        <button
          type="button"
          onClick={() => setMode((m) => (m === "umatrix" ? "hits" : "umatrix"))}
          className="ml-auto rounded-md border border-edge bg-void px-2 py-0.5 text-[10px] text-readout transition-colors hover:text-image"
        >
          show {mode === "umatrix" ? "hits" : "U-matrix"}
        </button>
      </div>
      <div className="overflow-x-auto">
        <div className="flex gap-2">
          {Array.from({ length: gz }, (_, z) => (
            <div key={z} className="flex flex-col items-center gap-1">
              <div
                className="grid overflow-hidden rounded border border-edge"
                style={{
                  gridTemplateColumns: `repeat(${gx}, ${CELL}px)`,
                  gridTemplateRows: `repeat(${gy}, ${CELL}px)`,
                }}
              >
                {Array.from({ length: gy * gx }, (_, i) => {
                  const y = Math.floor(i / gx);
                  const x = i % gx;
                  const k = z * gy * gx + y * gx + x;
                  const isActive = active === k;
                  const isPinned = pinnedNeuron === k;
                  return (
                    <button
                      key={i}
                      type="button"
                      aria-label={`neuron ${k}`}
                      onPointerEnter={() => onHoverNeuron(k)}
                      onPointerLeave={() => onHoverNeuron(null)}
                      onClick={() => onPinNeuron(isPinned ? null : k)}
                      className="relative block h-full w-full"
                      style={{
                        backgroundColor: rgbCss(cmap(values[k] / vmax)),
                        outline: isActive ? "1.5px solid #fff" : undefined,
                        outlineOffset: "-1.5px",
                        zIndex: isActive ? 1 : 0,
                      }}
                    />
                  );
                })}
              </div>
              <span className="font-mono text-[9px] text-muted">z={z}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
