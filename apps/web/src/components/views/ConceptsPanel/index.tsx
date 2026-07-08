"use client";

/**
 * Concepts + Prediction (Sprint B1 stub). The prediction label with a
 * confidence bar that fills as the loop crosses the final layer, plus a
 * placeholder "active concepts" list fed from concepts.json's top firing SAE
 * features (when the pack carries a concept tier). Sprint B2 makes concepts
 * appear per-community as activation reaches them and wires the confidence bars
 * to real stabilization — here they simply reveal with the verdict.
 */
import { useEffect, useMemo, useRef, useState } from "react";
import type { ConceptsJson } from "@/src/lib/pack/types";
import { useWorkbench } from "@/src/lib/state/store";
import { verdictProgress } from "@/src/lib/loop/schedule";
import { topFiringFeatures } from "./concepts";

export function ConceptsPanel({ classNames }: { classNames?: string[] }) {
  const client = useWorkbench((s) => s.client);
  const manifest = useWorkbench((s) => s.manifest);
  const [concepts, setConcepts] = useState<ConceptsJson | null>(null);
  const barRef = useRef<HTMLDivElement>(null);
  const pctRef = useRef<HTMLSpanElement>(null);

  useEffect(() => {
    let alive = true;
    setConcepts(null);
    if (!client || !manifest) return;
    client
      .loadConcepts(manifest)
      .then((c) => alive && setConcepts(c))
      .catch(() => {});
    return () => {
      alive = false;
    };
  }, [client, manifest]);

  const pred = manifest?.prediction;
  const features = useMemo(() => (concepts ? topFiringFeatures(concepts, 6) : []), [concepts]);
  const maxStrength = features.length ? features[0].strength : 1;

  // Confidence bar fills with the loop's verdict progress (imperative, smooth).
  useEffect(() => {
    if (!pred) return;
    let raf = 0;
    const tick = () => {
      raf = requestAnimationFrame(tick);
      const p = verdictProgress(useWorkbench.getState().t);
      const val = pred.confidence * p;
      if (barRef.current) barRef.current.style.width = `${val * 100}%`;
      if (pctRef.current) pctRef.current.textContent = `${(val * 100).toFixed(0)}%`;
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [pred]);

  if (!manifest || !pred) {
    return (
      <div className="flex h-full items-center justify-center text-[11px] tracking-wide text-muted">
        awaiting pack
      </div>
    );
  }

  const className = classNames?.[pred.class_index] ?? pred.label;

  return (
    <div className="flex h-full min-h-0 flex-col gap-3 overflow-y-auto p-1">
      {/* prediction */}
      <div>
        <div className="text-[10px] uppercase tracking-widest text-muted">prediction</div>
        <div className="mt-0.5 flex items-baseline justify-between gap-2">
          <span className="truncate text-lg font-semibold text-signal" title={className}>
            {className}
          </span>
          <span ref={pctRef} className="font-mono text-sm tabular-nums text-evidence">
            {(pred.confidence * 100).toFixed(0)}%
          </span>
        </div>
        <div className="mt-1.5 h-2 overflow-hidden rounded-full bg-panel-hi">
          <div
            ref={barRef}
            className="h-full rounded-full bg-evidence transition-[width] duration-150"
            style={{ width: `${pred.confidence * 100}%` }}
          />
        </div>
      </div>

      {/* active concepts */}
      <div className="flex min-h-0 flex-1 flex-col">
        <div className="flex items-center justify-between">
          <span className="text-[10px] uppercase tracking-widest text-muted">active concepts</span>
          {concepts ? (
            <span className="font-mono text-[9px] text-muted">SAE · L{concepts.layer}</span>
          ) : null}
        </div>
        {features.length ? (
          <div className="mt-1.5 flex flex-col gap-1.5">
            {features.map((f) => (
              <div key={f.id} className="flex items-center gap-2">
                <span className="w-16 shrink-0 font-mono text-[10px] tabular-nums text-latent">
                  #{f.id}
                </span>
                <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-panel-hi">
                  <div
                    className="h-full rounded-full bg-latent/70"
                    style={{ width: `${Math.max(6, (f.strength / maxStrength) * 100)}%` }}
                  />
                </div>
                <span className="w-6 shrink-0 text-right font-mono text-[9px] text-muted">
                  {f.count}
                </span>
              </div>
            ))}
          </div>
        ) : (
          <div className="mt-2 text-[11px] leading-relaxed text-muted">
            {concepts === null
              ? "no concept dictionary in this pack"
              : "no firing features"}
          </div>
        )}
        <div className="mt-auto pt-2 text-[9px] leading-snug text-muted/80">
          Concepts are SAE features; semantic names are withheld until validated
          (honesty rule). B2 links these to the brain&rsquo;s communities.
        </div>
      </div>
    </div>
  );
}
