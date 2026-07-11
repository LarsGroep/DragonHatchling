"use client";

/**
 * Z-axis honesty panel — the notebook's core caveat made visible. The Z axis is
 * transformer depth, a *learned* hierarchy imposed by the ordering regularizer,
 * not physical depth. Beneath the note is the per-slice spectral-centroid bar
 * chart with a measured verdict: a centroid that falls monotonically with depth
 * means the ordering bias took hold; anything else is a reported negative
 * result, not a bug.
 */
import { useMemo } from "react";
import type { UmtvitBundle } from "@/src/lib/umtvit";
import { Panel } from "./controls";

export function HonestyNote({ bundle }: { bundle: UmtvitBundle }) {
  const centroids = bundle.spectral_centroids;

  const { monotone, max, values } = useMemo(() => {
    const vals = centroids.map((c) => (c ?? Number.NaN));
    const finite = vals.filter((v) => Number.isFinite(v));
    let mono = finite.length > 1;
    for (let i = 1; i < vals.length; i++) {
      if (Number.isFinite(vals[i]) && Number.isFinite(vals[i - 1]) && vals[i] > vals[i - 1] + 1e-4) {
        mono = false;
      }
    }
    return { monotone: mono, max: Math.max(1e-9, ...finite), values: vals };
  }, [centroids]);

  return (
    <Panel title="Z-axis honesty" accent="warm" subtitle="did scale ordering emerge?">
      <div className="flex flex-col gap-3">
        <p className="rounded-lg border border-warm/30 bg-warm/5 px-3 py-2 text-[12px] leading-relaxed text-readout">
          <span className="font-semibold">Z is transformer depth — a learned hierarchy, not
          physical depth.</span>{" "}
          ViT layers do not order themselves by spatial scale; the ordering regularizer{" "}
          <em>imposes</em> that bias, and the chart below <em>measures</em> whether it took hold.
        </p>

        <div className="flex flex-col gap-1">
          <span className="text-[10px] uppercase tracking-wide text-muted">
            Spectral centroid by depth slice (cycles/px)
          </span>
          <div className="flex items-end gap-1.5" style={{ height: 90 }}>
            {values.map((v, i) => {
              const h = Number.isFinite(v) ? Math.max(2, (v / max) * 84) : 2;
              return (
                <div key={i} className="flex flex-1 flex-col items-center justify-end gap-1">
                  <span className="font-mono text-[9px] tabular-nums text-muted">
                    {Number.isFinite(v) ? v.toFixed(2) : "—"}
                  </span>
                  <div
                    className="w-full rounded-t"
                    style={{
                      height: h,
                      backgroundColor: monotone ? "#22c55e" : "#f59e0b",
                      opacity: 0.35 + 0.6 * (i / Math.max(1, values.length - 1)),
                    }}
                    title={`z=${i}: ${Number.isFinite(v) ? v.toFixed(4) : "n/a"}`}
                  />
                  <span className="font-mono text-[9px] text-muted">z{i}</span>
                </div>
              );
            })}
          </div>
        </div>

        <p className={`text-[12px] font-medium ${monotone ? "text-evidence" : "text-warm"}`}>
          {monotone
            ? "Centroid falls with depth ✓ — the imposed ordering emerged on this run."
            : "Centroid is NOT monotone — ordering only partially emerged (an honest negative result, not a bug)."}
        </p>
      </div>
    </Panel>
  );
}
