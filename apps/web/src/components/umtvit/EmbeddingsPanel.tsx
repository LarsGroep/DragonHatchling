"use client";

/**
 * Embeddings panel — the pooled-feature scatter in a fixed PCA basis (computed
 * in Python on the final epoch), replaying how the embedding space takes shape
 * over training. Colored by class when labels are present, single hue when
 * unlabeled. Rendered as an SVG; the epoch scrubber + play drive the animation.
 */
import { useMemo, useState } from "react";
import type { UmtvitBundle } from "@/src/lib/umtvit";
import { classColor } from "./colormaps";
import { Panel, PlayScrubber } from "./controls";

const SIZE = 300;
const PAD = 10;

export function EmbeddingsPanel({ bundle }: { bundle: UmtvitBundle }) {
  const { embeddings, classes } = bundle;
  const [epochIdx, setEpochIdx] = useState(embeddings.coords.length - 1);

  // Fixed axis limits across all epochs (basis is already fixed in Python).
  const lim = useMemo(() => {
    let m = 1e-6;
    for (const frame of embeddings.coords)
      for (const [x, y] of frame) {
        m = Math.max(m, Math.abs(x), Math.abs(y));
      }
    return m * 1.1;
  }, [embeddings.coords]);

  if (!embeddings.coords.length) {
    return (
      <Panel title="Embedding space" accent="image">
        <p className="text-[12px] text-muted">No embedding snapshots in this run.</p>
      </Panel>
    );
  }

  const idx = Math.min(Math.round(epochIdx), embeddings.coords.length - 1);
  const frame = embeddings.coords[idx];
  const labels = embeddings.labels;
  const epochLabel = embeddings.epochs[idx] ?? idx;

  const toX = (x: number) => PAD + ((x + lim) / (2 * lim)) * (SIZE - 2 * PAD);
  const toY = (y: number) => PAD + ((lim - y) / (2 * lim)) * (SIZE - 2 * PAD);

  return (
    <Panel title="Embedding space" accent="image" subtitle="pooled features · fixed PCA basis">
      <div className="flex flex-col gap-3">
        <div className="flex flex-col gap-3 sm:flex-row sm:items-start">
          <svg
            viewBox={`0 0 ${SIZE} ${SIZE}`}
            className="w-full max-w-[300px] rounded-lg border border-edge bg-void"
            role="img"
            aria-label={`embedding scatter at epoch ${epochLabel}`}
          >
            <line x1={SIZE / 2} y1={PAD} x2={SIZE / 2} y2={SIZE - PAD} stroke="#e3e6ec" strokeWidth={1} />
            <line x1={PAD} y1={SIZE / 2} x2={SIZE - PAD} y2={SIZE / 2} stroke="#e3e6ec" strokeWidth={1} />
            {frame.map(([x, y], i) => (
              <circle
                key={i}
                cx={toX(x)}
                cy={toY(y)}
                r={2.6}
                fill={labels ? classColor(labels[i] ?? 0) : "#8b5cf6"}
                fillOpacity={0.8}
              />
            ))}
          </svg>

          {classes && labels ? (
            <ul className="flex flex-wrap gap-x-3 gap-y-1 sm:flex-col">
              {classes.map((c, i) => (
                <li key={c} className="flex items-center gap-1.5 text-[11px] text-readout">
                  <span
                    className="h-2.5 w-2.5 rounded-full"
                    style={{ backgroundColor: classColor(i) }}
                  />
                  {c}
                </li>
              ))}
            </ul>
          ) : (
            <p className="text-[11px] text-muted sm:max-w-[10rem]">
              Unlabeled run — a single hue; structure here is self-supervised, not class-driven.
            </p>
          )}
        </div>

        <PlayScrubber
          value={epochIdx}
          min={0}
          max={Math.max(0, embeddings.coords.length - 1)}
          step={1}
          speed={2}
          accent="image"
          label="epoch"
          onChange={setEpochIdx}
          format={() => `ep ${epochLabel}`}
        />
      </div>
    </Panel>
  );
}
