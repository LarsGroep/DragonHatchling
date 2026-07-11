"use client";

/**
 * SOM panel — the 3-D Self-Organizing Map rendered as per-grid-z-layer maps.
 * Top row: the U-matrix (viridis; dark valleys = coherent regions, bright walls
 * = cluster boundaries) at the scrubbed epoch, replaying self-organization.
 * Bottom row: the final voxel hit maps (hot). Epoch scrubber + play drive the
 * U-matrix animation; hit maps show the converged state.
 */
import { useEffect, useMemo, useRef, useState } from "react";
import type { UmtvitBundle } from "@/src/lib/umtvit";
import { hot, viridis, type RGB } from "./colormaps";
import { Panel, PlayScrubber } from "./controls";

const CELL = 26;

function GridCanvas({
  layer,
  vmin,
  vmax,
  cmap,
}: {
  layer: number[][];
  vmin: number;
  vmax: number;
  cmap: (t: number) => RGB;
}) {
  const ref = useRef<HTMLCanvasElement | null>(null);
  const rows = layer.length;
  const cols = layer[0]?.length ?? 0;
  useEffect(() => {
    const canvas = ref.current;
    if (!canvas || !rows || !cols) return;
    const off = document.createElement("canvas");
    off.width = cols;
    off.height = rows;
    const octx = off.getContext("2d");
    if (!octx) return;
    const img = octx.createImageData(cols, rows);
    const span = vmax - vmin || 1;
    for (let y = 0; y < rows; y++) {
      for (let x = 0; x < cols; x++) {
        const [r, g, b] = cmap((layer[y][x] - vmin) / span);
        const k = (y * cols + x) * 4;
        img.data[k] = r;
        img.data[k + 1] = g;
        img.data[k + 2] = b;
        img.data[k + 3] = 255;
      }
    }
    octx.putImageData(img, 0, 0);
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.imageSmoothingEnabled = false;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(off, 0, 0, cols, rows, 0, 0, canvas.width, canvas.height);
  }, [layer, vmin, vmax, cmap, rows, cols]);
  return (
    <canvas
      ref={ref}
      width={cols * CELL}
      height={rows * CELL}
      className="rounded border border-edge"
      style={{ width: cols * CELL, height: rows * CELL, imageRendering: "pixelated" }}
    />
  );
}

export function SomPanel({ bundle }: { bundle: UmtvitBundle }) {
  const { som } = bundle;
  const [gz] = som.grid;
  const [epochIdx, setEpochIdx] = useState(som.umatrix.length - 1);

  const uMax = useMemo(() => {
    let m = 0;
    for (const frame of som.umatrix)
      for (const layer of frame) for (const row of layer) for (const v of row) if (v > m) m = v;
    return m;
  }, [som.umatrix]);

  const hitMax = useMemo(() => {
    let m = 0;
    for (const layer of som.hits_final) for (const row of layer) for (const v of row) if (v > m) m = v;
    return m;
  }, [som.hits_final]);

  const idx = Math.min(Math.round(epochIdx), som.umatrix.length - 1);
  const frame = som.umatrix[idx];
  const epochLabel = som.epochs[idx] ?? idx;

  return (
    <Panel
      title="Self-Organizing Map"
      accent="gauss"
      subtitle={`${som.grid.join("×")} neuron grid`}
    >
      <div className="flex flex-col gap-3">
        <div className="overflow-x-auto">
          <div className="flex flex-col gap-2">
            <div className="flex items-center gap-3">
              <span className="w-14 shrink-0 text-[10px] uppercase tracking-wide text-muted">
                U-matrix
              </span>
              <div className="flex gap-2">
                {Array.from({ length: gz }, (_, z) => (
                  <div key={z} className="flex flex-col items-center gap-1">
                    <GridCanvas layer={frame[z]} vmin={0} vmax={uMax} cmap={viridis} />
                    <span className="font-mono text-[9px] text-muted">z{z}</span>
                  </div>
                ))}
              </div>
            </div>
            <div className="flex items-center gap-3">
              <span className="w-14 shrink-0 text-[10px] uppercase tracking-wide text-muted">
                Hits (final)
              </span>
              <div className="flex gap-2">
                {Array.from({ length: gz }, (_, z) => (
                  <div key={z} className="flex flex-col items-center gap-1">
                    <GridCanvas layer={som.hits_final[z]} vmin={0} vmax={hitMax} cmap={hot} />
                    <span className="font-mono text-[9px] text-muted">z{z}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>

        <PlayScrubber
          value={epochIdx}
          min={0}
          max={Math.max(0, som.umatrix.length - 1)}
          step={1}
          speed={2}
          accent="gauss"
          label="epoch"
          onChange={setEpochIdx}
          format={() => `ep ${epochLabel}`}
        />
        <p className="text-[11px] text-muted">
          The U-matrix replays the map organizing itself as the neighborhood radius σ anneals; the
          hit maps show where the eval images&rsquo; voxels land at convergence.
        </p>
      </div>
    </Panel>
  );
}
