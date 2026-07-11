"use client";

/**
 * Training curves — the self-supervised loss history (total + component terms)
 * as SVG polylines with a hover crosshair that reads out every series at the
 * hovered step. Each series is min-max normalized to its own range so terms of
 * very different magnitude stay legible together; raw values show in the
 * tooltip.
 */
import { useMemo, useRef, useState } from "react";
import type { UmtvitBundle, UmtvitSeriesKey } from "@/src/lib/umtvit";

const W = 640;
const H = 220;
const PAD = { top: 12, right: 12, bottom: 24, left: 12 };

const SERIES: { key: UmtvitSeriesKey; color: string; label: string }[] = [
  { key: "total", color: "#28313f", label: "total" },
  { key: "ntxent", color: "#3b82f6", label: "NT-Xent" },
  { key: "som", color: "#0d9488", label: "SOM" },
  { key: "smooth", color: "#f59e0b", label: "smooth" },
  { key: "order", color: "#8b5cf6", label: "order" },
];

export function TrainingCurves({ bundle }: { bundle: UmtvitBundle }) {
  const { history } = bundle;
  const svgRef = useRef<SVGSVGElement | null>(null);
  const [hover, setHover] = useState<number | null>(null);

  const n = history.steps.length;
  const plotW = W - PAD.left - PAD.right;
  const plotH = H - PAD.top - PAD.bottom;

  const ranges = useMemo(() => {
    const r: Record<string, { min: number; max: number }> = {};
    for (const s of SERIES) {
      const seq = history.series[s.key];
      if (seq.length) {
        r[s.key] = { min: Math.min(...seq), max: Math.max(...seq) };
      }
    }
    return r;
  }, [history.series]);

  if (n < 2) {
    return (
      <div className="rounded-xl border border-edge bg-panel p-4 text-[12px] text-muted shadow-soft">
        Not enough training steps recorded to plot curves.
      </div>
    );
  }

  const xAt = (i: number) => PAD.left + (i / (n - 1)) * plotW;
  const yAt = (key: UmtvitSeriesKey, v: number) => {
    const rg = ranges[key];
    const span = rg ? rg.max - rg.min || 1 : 1;
    const t = rg ? (v - rg.min) / span : 0.5;
    return PAD.top + (1 - t) * plotH;
  };

  const paths = SERIES.map((s) => {
    const seq = history.series[s.key];
    if (!seq.length) return { ...s, d: "" };
    const d = seq.map((v, i) => `${i === 0 ? "M" : "L"}${xAt(i).toFixed(1)},${yAt(s.key, v).toFixed(1)}`).join(" ");
    return { ...s, d };
  });

  const onMove = (e: React.MouseEvent<SVGSVGElement>) => {
    const rect = svgRef.current?.getBoundingClientRect();
    if (!rect) return;
    const px = ((e.clientX - rect.left) / rect.width) * W;
    const i = Math.round(((px - PAD.left) / plotW) * (n - 1));
    setHover(Math.max(0, Math.min(n - 1, i)));
  };

  return (
    <div className="flex flex-col gap-2 rounded-xl border border-edge bg-panel p-4 shadow-soft">
      <div className="flex items-center gap-2">
        <span className="h-1.5 w-1.5 rounded-full bg-image" />
        <h2 className="text-[11px] font-semibold uppercase tracking-wide text-readout">
          Training curves
        </h2>
        <span className="text-[11px] text-muted">· self-supervised losses (per step)</span>
        <div className="ml-auto flex flex-wrap gap-x-3 gap-y-0.5">
          {SERIES.map((s) => (
            <span key={s.key} className="flex items-center gap-1 text-[10px] text-muted">
              <span className="h-1.5 w-3 rounded-full" style={{ backgroundColor: s.color }} />
              {s.label}
            </span>
          ))}
        </div>
      </div>

      <svg
        ref={svgRef}
        viewBox={`0 0 ${W} ${H}`}
        className="w-full"
        style={{ aspectRatio: `${W} / ${H}` }}
        onMouseMove={onMove}
        onMouseLeave={() => setHover(null)}
        role="img"
        aria-label="training loss curves"
      >
        {paths.map((p) =>
          p.d ? <path key={p.key} d={p.d} fill="none" stroke={p.color} strokeWidth={1.4} /> : null,
        )}
        {hover !== null ? (
          <>
            <line
              x1={xAt(hover)}
              y1={PAD.top}
              x2={xAt(hover)}
              y2={PAD.top + plotH}
              stroke="#8b94a4"
              strokeDasharray="3 3"
              strokeWidth={1}
            />
            {SERIES.map((s) => {
              const seq = history.series[s.key];
              if (!seq.length) return null;
              return (
                <circle key={s.key} cx={xAt(hover)} cy={yAt(s.key, seq[hover])} r={3} fill={s.color} />
              );
            })}
          </>
        ) : null}
      </svg>

      <div className="min-h-[1.25rem] font-mono text-[11px] tabular-nums text-muted">
        {hover !== null ? (
          <span>
            <span className="text-readout">step {history.steps[hover]}</span>
            {"  "}
            {SERIES.filter((s) => history.series[s.key].length).map((s) => (
              <span key={s.key} className="ml-2">
                <span style={{ color: s.color }}>{s.label}</span> {history.series[s.key][hover].toFixed(3)}
              </span>
            ))}
          </span>
        ) : (
          <span>hover to read values</span>
        )}
      </div>
    </div>
  );
}
